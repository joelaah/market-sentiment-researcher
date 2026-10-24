"""
Historical Backtesting Engine for Financial Market Sentiment & Multi-Bot Arena.
Provides chronological bar-by-bar simulation without lookahead bias across US and Indian markets.
"""

from typing import List, Dict, Any, Optional, Tuple, Union, Set
from datetime import datetime, timezone
from pathlib import Path
import os
import sys
import json
import re
import logging
import warnings
import uuid
import pandas as pd
import numpy as np
import yfinance as yf

from src.paper_trader import (
    BotArchetype,
    BorderlistGuardrail,
    is_benchmark_index,
    is_inr_symbol,
    INDEX_TRADEABLE_MAP,
    calculate_probability_weighted_upside,
    get_min_price_for_symbol,
    safe_print,
    get_default_tournament_bots,
    TRUSTED_SYMBOLS,
    MIN_SHARE_PRICE,
    MIN_SHARE_PRICE_INR,
    MIN_DAILY_VOLUME,
    MIN_MARKET_CAP
)
from src.news_fetcher import HistoricalNewsFetcher


# ============================================================================
# HISTORICAL DATA INGESTION & CACHING MODULE
# ============================================================================

class HistoricalDataFetcher:
    """
    Ingests and caches historical daily OHLCV bars across US and Indian instruments.
    Supports offline execution, corrupt cache recovery, and strict column sanitization.
    """

    DEFAULT_UNIVERSE: List[str] = ["^NSEI", "NIFTYBEES.NS", "NVDA", "AAPL", "MSFT", "SPY"]
    REQUIRED_COLUMNS: List[str] = ["open", "high", "low", "close", "volume"]

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        offline: bool = False,
        cache_format: str = "csv",
        offline_only: Optional[bool] = None,
    ) -> None:
        """
        Initializes the fetcher with a persistent cache directory and offline mode flag.

        Args:
            cache_dir: Directory where bar files are stored. Defaults to 'data/historical_bars/'.
            offline: If True, suppresses all network calls and only reads from local cache.
            cache_format: 'csv' (default).
            offline_only: Alias for offline.
        """
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(__file__).resolve().parent.parent / "data" / "historical_bars"

        # Check offline flag from constructor or environment variable
        effective_offline = offline if offline_only is None else offline_only
        env_offline = os.environ.get("BACKTEST_OFFLINE", "").strip().lower() in ("1", "true", "yes")
        self.offline = bool(effective_offline or env_offline)
        self.cache_format = str(cache_format).lower().strip()

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            safe_print(f"[HistoricalDataFetcher] Warning: Could not create cache dir {self.cache_dir}: {e}")

    @staticmethod
    def get_safe_symbol(symbol: str) -> str:
        """
        Normalizes symbols containing special filesystem characters (e.g. '^' in '^NSEI')
        for operating system path safety across Windows and POSIX systems.
        """
        safe = symbol.replace("^", "_").replace(":", "_").replace("/", "_").replace("\\", "_")
        return safe

    def get_cache_path(self, symbol: str) -> Path:
        """Returns the primary deterministic cache file path for a symbol."""
        safe_sym = self.get_safe_symbol(symbol)
        return self.cache_dir / f"{safe_sym}_1d.{self.cache_format}"

    def find_existing_cache_file(self, symbol: str) -> Optional[Path]:
        """
        Searches candidate filenames for an existing cached file:
        1. {safe_symbol}_1d.csv (e.g. _NSEI_1d.csv)
        2. {symbol}_1d.csv (e.g. ^NSEI_1d.csv if created on POSIX)
        3. {symbol.lstrip('^')}_1d.csv (e.g. NSEI_1d.csv)
        """
        safe_sym = self.get_safe_symbol(symbol)
        stripped_sym = symbol.lstrip("^")

        candidates = [
            self.cache_dir / f"{safe_sym}_1d.{self.cache_format}",
            self.cache_dir / f"{symbol}_1d.{self.cache_format}",
            self.cache_dir / f"{stripped_sym}_1d.{self.cache_format}",
        ]

        for p in candidates:
            if p.exists() and p.is_file():
                return p
        return None

    def validate_cache_file(self, filepath: Path) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
        """
        Validates cache file integrity:
        - Non-zero file size
        - Parseable CSV structure
        - Presence of required OHLCV columns
        - Presence of valid rows

        Returns:
            (is_valid, parsed_df_or_None, error_message_or_None)
        """
        if not filepath.exists():
            return False, None, f"File {filepath} does not exist"

        try:
            file_size = filepath.stat().st_size
            if file_size == 0:
                return False, None, f"File {filepath} is 0 bytes (corrupt/empty)"
        except Exception as e:
            return False, None, f"Could not inspect file {filepath}: {e}"

        try:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, Exception) as e:
            return False, None, f"Corrupt cache format or unparseable CSV: {e}"

        if df.empty:
            return False, None, "Cache file parsed to empty DataFrame"

        # Check required columns
        cols = [str(c).lower().strip() for c in df.columns]
        missing = [c for c in self.REQUIRED_COLUMNS if c not in cols]
        if missing:
            return False, None, f"Cache file missing required columns: {missing}"

        return True, df, None

    @classmethod
    def sanitize_bars(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sanitizes raw provider or cached DataFrames:
        1. Flattens MultiIndex columns if present (taking level 0).
        2. Lowercases and strips all column names.
        3. Validates that required columns ['open', 'high', 'low', 'close', 'volume'] exist.
        4. Filters strictly to the 5 required columns.
        5. Normalizes index: converts to tz-naive DatetimeIndex using calendar dates.
        6. Drops rows with missing or non-positive OHLC values.
        7. Fills missing volume values with 0.0.
        8. Sorts ascending chronologically and drops duplicate dates.
        """
        if df is None or df.empty:
            raise ValueError("Cannot sanitize None or empty DataFrame")

        clean = df.copy()

        # 1. Flatten MultiIndex if present
        if isinstance(clean.columns, pd.MultiIndex):
            clean.columns = [
                c[0].lower().strip() if isinstance(c, tuple) else str(c).lower().strip()
                for c in clean.columns
            ]
        else:
            clean.columns = [str(c).lower().strip() for c in clean.columns]

        # 2. Check required columns
        missing = [c for c in cls.REQUIRED_COLUMNS if c not in clean.columns]
        if missing:
            raise ValueError(f"Input DataFrame is missing required OHLCV columns: {missing}")

        # 3. Filter strictly to required columns
        clean = clean[cls.REQUIRED_COLUMNS].copy()

        # 4. Normalize DatetimeIndex (calendar date, tz-naive)
        if not isinstance(clean.index, pd.DatetimeIndex):
            clean.index = pd.to_datetime(clean.index)

        # Convert to tz-naive calendar date (preventing EDT/IST UTC offset date shifts)
        clean.index = pd.to_datetime(clean.index.date)

        # 5. Enforce numeric types
        for col in cls.REQUIRED_COLUMNS:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")

        # 6. Fill volume NaNs with 0.0
        clean["volume"] = clean["volume"].fillna(0.0)

        # 7. Drop rows with invalid or non-positive prices
        clean = clean.dropna(subset=["open", "high", "low", "close"])
        clean = clean[
            (clean["open"] > 0) &
            (clean["high"] > 0) &
            (clean["low"] > 0) &
            (clean["close"] > 0)
        ]

        # 8. Sort chronologically ascending and drop duplicate dates
        clean = clean.sort_index()
        clean = clean[~clean.index.duplicated(keep="last")]

        if clean.empty:
            raise ValueError("Sanitized DataFrame contains 0 valid positive price bars")

        return clean

    def fetch_symbol_bars(
        self,
        symbol: str,
        years: int = 3,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetches historical daily bars for a single symbol, checking local cache first.
        """
        # Step 1: Probe local cache
        candidate = self.find_existing_cache_file(symbol)
        if candidate and not force_refresh:
            is_valid, cached_df, err = self.validate_cache_file(candidate)
            if is_valid and cached_df is not None:
                try:
                    return self.sanitize_bars(cached_df)
                except Exception as e:
                    err = f"Sanitization of cached data failed: {e}"

            # If cache file is invalid
            if self.offline:
                raise ValueError(f"Corrupt cache file for {symbol} at {candidate}: {err}")
            else:
                safe_print(f"[HistoricalDataFetcher] Warning: Corrupt cache {candidate} ({err}). Purging and refetching...")
                try:
                    candidate.unlink()
                except Exception:
                    pass

        # Step 2: Handle missing cache in offline mode
        if self.offline:
            expected_path = self.get_cache_path(symbol)
            raise FileNotFoundError(
                f"No cached data found for {symbol} at {expected_path} and offline mode is active. "
                f"Zero network requests attempted."
            )

        # Step 3: Online download via yfinance
        try:
            if start_date and end_date:
                raw_df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            else:
                raw_df = yf.Ticker(symbol).history(period=f"{years}y", interval="1d")
        except Exception as e:
            raise ValueError(f"Network error querying market data for {symbol}: {e}")

        if raw_df is None or raw_df.empty:
            raise ValueError(f"No historical data returned for symbol: {symbol}")

        # Step 4: Sanitize and cache
        sanitized = self.sanitize_bars(raw_df)
        cache_path = self.get_cache_path(symbol)
        try:
            sanitized.to_csv(cache_path, index_label="date")
        except Exception as e:
            safe_print(f"[HistoricalDataFetcher] Warning: Failed to write cache for {symbol} at {cache_path}: {e}")

        return sanitized

    def fetch_universe_bars(
        self,
        symbols: Optional[List[str]] = None,
        years: int = 3,
        force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetches historical bars for all symbols in the target universe.
        """
        universe = list(symbols or self.DEFAULT_UNIVERSE)
        results: Dict[str, pd.DataFrame] = {}
        for s in universe:
            results[s] = self.fetch_symbol_bars(s, years=years, force_refresh=force_refresh)
        return results


# ============================================================================
# TECHNICAL INDICATORS & SYNTHETIC SIMULATION HELPERS
# ============================================================================

def compute_rolling_indicators(
    df: pd.DataFrame,
    window_sma: int = 20,
    lookback_mom: int = 5
) -> Dict[str, Any]:
    """
    Computes backward-looking rolling technical indicators strictly from data up to the latest row.
    Guarantees zero forward lookahead leakage.
    """
    if df is None or df.empty:
        return {"valid": False, "price": 0.0, "volume": 0.0, "sma20": 0.0, "above_sma20": False, "momentum_5d": 0.0}

    close = df["close"]
    n = len(close)
    valid = bool(n >= window_sma)

    cur_close = round(float(close.iloc[-1]), 2)
    cur_vol = float(df["volume"].iloc[-1]) if "volume" in df else 0.0

    if n >= window_sma:
        sma20 = round(float(close.iloc[-window_sma:].mean()), 2)
    else:
        sma20 = round(float(close.mean()), 2)

    above_sma20 = bool(cur_close >= sma20)

    if n > lookback_mom:
        past_close = float(close.iloc[-lookback_mom - 1])
        mom_5d = round(((cur_close - past_close) / past_close * 100.0), 2) if past_close > 0 else 0.0
    else:
        mom_5d = 0.0

    return {
        "valid": valid,
        "price": cur_close,
        "volume": cur_vol,
        "sma20": sma20,
        "above_sma20": above_sma20,
        "momentum_5d": mom_5d
    }


def simulate_single_position(
    df: pd.DataFrame,
    entry_bar: int = 0,
    stop_loss_pct: float = 5.0,
    take_profit_pct: float = 10.0,
    slippage_pct: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Deterministic single-position simulator for unit test fixtures.
    Simulates entering at Close of entry_bar, and evaluating subsequent bars for stop-loss / take-profit.
    Applies pessimistic tie-breaking if both SL and TP boundaries are crossed in the same bar.
    """
    if df is None or entry_bar >= len(df) - 1:
        return []

    entry_price = float(df["close"].iloc[entry_bar])
    # Note: stop_loss_pct is passed as positive magnitude (e.g. 5.0 -> -5%)
    sl_target = entry_price * (1.0 - abs(stop_loss_pct) / 100.0)
    tp_target = entry_price * (1.0 + abs(take_profit_pct) / 100.0)

    trades = []
    for bar_idx in range(entry_bar + 1, len(df)):
        low_p = float(df["low"].iloc[bar_idx])
        high_p = float(df["high"].iloc[bar_idx])
        open_p = float(df["open"].iloc[bar_idx])
        close_p = float(df["close"].iloc[bar_idx])

        # Pessimistic tie-breaking: Check stop-loss first
        if low_p <= sl_target and high_p >= tp_target:
            exit_price = round(min(open_p, sl_target) * (1.0 - slippage_pct), 2)
            trades.append({
                "entry_bar": entry_bar,
                "exit_bar": bar_idx,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_reason": "STOP_LOSS",
                "pnl_pct": round(((exit_price - entry_price) / entry_price * 100.0), 2)
            })
            break
        elif low_p <= sl_target:
            exit_price = round(min(open_p, sl_target) * (1.0 - slippage_pct), 2)
            trades.append({
                "entry_bar": entry_bar,
                "exit_bar": bar_idx,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_reason": "STOP_LOSS",
                "pnl_pct": round(((exit_price - entry_price) / entry_price * 100.0), 2)
            })
            break
        elif high_p >= tp_target:
            exit_price = round(max(open_p, tp_target) * (1.0 - slippage_pct), 2)
            trades.append({
                "entry_bar": entry_bar,
                "exit_bar": bar_idx,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_reason": "TAKE_PROFIT",
                "pnl_pct": round(((exit_price - entry_price) / entry_price * 100.0), 2)
            })
            break

    return trades


# ============================================================================
# PERFORMANCE METRICS CALCULATION
# ============================================================================

def calculate_performance_metrics(
    trades: Optional[List[Dict[str, Any]]] = None,
    equity_curve: Optional[List[Union[float, Tuple[Any, float]]]] = None,
    initial_balance: Optional[float] = 5000.0,
    years: Optional[float] = 3.0
) -> Dict[str, float]:
    """
    Computes quantitative performance metrics from backtest trade logs and equity curve:
    - Win Rate %: Wins / (Wins + Losses) * 100
    - Profit Factor: Gross Profit / Gross Loss (capped at 999.0 for lossless strategies)
    - Max Drawdown (MDD) %: Peak-to-trough drop on marked-to-market daily total equity
    - Net Return %: ((Final Equity - Initial Balance) / Initial Balance) * 100
    - CAGR %: Compound Annual Growth Rate over duration in years
    """
    try:
        init_bal = float(initial_balance) if initial_balance is not None else 5000.0
    except (TypeError, ValueError):
        init_bal = 5000.0

    try:
        yr = float(years) if years is not None else 3.0
    except (TypeError, ValueError):
        yr = 3.0

    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0

    for t in (trades or []):
        if not isinstance(t, dict):
            continue

        raw_pnl = t.get("realized_pnl")
        if raw_pnl is None:
            raw_pnl = t.get("pnl")
        if raw_pnl is None:
            raw_pnl = 0.0

        try:
            pnl = float(raw_pnl)
            if np.isnan(pnl) or np.isinf(pnl):
                pnl = 0.0
        except (TypeError, ValueError):
            pnl = 0.0

        if pnl > 1e-6:
            wins += 1
            gross_profit += pnl
        elif pnl < -1e-6:
            losses += 1
            gross_loss += abs(pnl)

    total_closed = wins + losses
    win_rate_pct = round((wins / total_closed * 100.0), 2) if total_closed > 0 else 0.0

    # Profit Factor with JSON-safe numerical ceiling (prevents inf serialization error)
    if gross_loss > 1e-6:
        profit_factor = round(gross_profit / gross_loss, 2)
    elif gross_profit > 1e-6:
        profit_factor = 999.0  # JSON-compliant ceiling for 0 gross loss
    else:
        profit_factor = 0.0

    # Marked-to-market Equity & Max Drawdown %
    cleaned_equity: List[float] = []
    for eq in (equity_curve or []):
        if isinstance(eq, (tuple, list)) and len(eq) >= 2:
            candidate = eq[1]
        else:
            candidate = eq
        try:
            val = float(candidate)
            if not (np.isnan(val) or np.isinf(val)):
                cleaned_equity.append(val)
        except (TypeError, ValueError):
            continue

    if not cleaned_equity:
        final_equity = init_bal
        max_drawdown_pct = 0.0
    else:
        final_equity = cleaned_equity[-1]
        running_peak = cleaned_equity[0]
        max_dd = 0.0
        for eq_val in cleaned_equity:
            if eq_val > running_peak:
                running_peak = eq_val
            elif running_peak > 0:
                dd = (running_peak - eq_val) / running_peak * 100.0
                if dd > max_dd:
                    max_dd = dd
        max_drawdown_pct = round(max_dd, 2)

    # Net Return %
    if init_bal > 0:
        net_return_pct = round(((final_equity - init_bal) / init_bal * 100.0), 2)
    else:
        net_return_pct = 0.0

    # CAGR % with complex root / bankruptcy guard
    if init_bal <= 0 or yr <= 0:
        cagr_pct = 0.0
    elif final_equity <= 0:
        cagr_pct = -100.0
    else:
        growth_factor = final_equity / init_bal
        cagr = (growth_factor ** (1.0 / yr)) - 1.0
        cagr_pct = round(cagr * 100.0, 2)

    return {
        "win_rate_pct": win_rate_pct,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown_pct,
        "net_return_pct": net_return_pct,
        "cagr_pct": cagr_pct,
        "total_trades": total_closed,
        "wins": wins,
        "losses": losses,
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2)
    }


# ============================================================================
# SIMULATED PORTFOLIO
# ============================================================================

class SimulatedPortfolio:
    """Tracks isolated cash, holdings, and trade history for a single bot during backtesting."""

    def __init__(self, bot: BotArchetype, initial_balance: float = 5000.0):
        self.bot = bot
        self.bot_id = bot.bot_id
        self.cash = float(initial_balance)
        self.initial_balance = float(initial_balance)
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trades: List[Dict[str, Any]] = []
        self.closed_trades: List[Dict[str, Any]] = []
        self.equity_curve: List[Tuple[str, float]] = []

    def get_total_equity(self, current_prices: Dict[str, float]) -> float:
        """Calculates mark-to-market total portfolio equity."""
        pos_val = 0.0
        for sym, pos in self.positions.items():
            shares = float(pos.get("shares") or 0.0)
            avg_p = float(pos.get("avg_buy_price") or 0.0)
            raw_p = current_prices.get(sym) if isinstance(current_prices, dict) else None
            price = float(raw_p) if raw_p is not None else avg_p
            pos_val += shares * price
        return round(self.cash + pos_val, 2)


# ============================================================================
# HISTORICAL BACKTEST ENGINE
# ============================================================================

class HistoricalBacktestEngine:
    """
    Chronological bar-by-bar historical replay engine.
    Steps across unified trading dates without lookahead bias.
    """

    def __init__(
        self,
        initial_balance: float = 5000.0,
        slippage_bps: float = 5.0,  # 0.05% slippage
        guardrail: Optional[BorderlistGuardrail] = None
    ) -> None:
        self.initial_balance = initial_balance
        self.slippage_pct = slippage_bps / 10000.0
        self.guardrail = guardrail or BorderlistGuardrail()

    def _prepare_data_and_technicals(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, pd.Series]]]:
        """Precomputes backward-looking rolling indicators without lookahead leakage."""
        sanitized = {}
        tech_series = {}
        for sym, df in data.items():
            df_clean = df.copy()
            df_clean.columns = [c.lower() for c in df_clean.columns]
            df_clean.sort_index(inplace=True)
            sanitized[sym] = df_clean

            # Causal backward-looking indicators (center=False by default in pandas)
            sma20 = df_clean["close"].rolling(window=20, min_periods=1).mean()
            mom5d = (df_clean["close"] - df_clean["close"].shift(5)) / df_clean["close"].shift(5) * 100.0
            adv20 = df_clean["volume"].rolling(window=20, min_periods=1).mean() if "volume" in df_clean else None

            tech_series[sym] = {
                "sma20": sma20,
                "mom5d": mom5d,
                "adv20": adv20
            }
        return sanitized, tech_series

    def _build_master_calendar(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Any]:
        """Constructs unified chronological trading calendar."""
        all_dates: Set[Any] = set()
        for df in data.values():
            all_dates.update(df.index)
        sorted_dates = sorted(list(all_dates))
        if start_date:
            sorted_dates = [d for d in sorted_dates if str(d)[:10] >= start_date[:10]]
        if end_date:
            sorted_dates = [d for d in sorted_dates if str(d)[:10] <= end_date[:10]]
        return sorted_dates

    def run_backtest(
        self,
        bots: Optional[List[BotArchetype]] = None,
        data: Optional[Dict[str, pd.DataFrame]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        sentiment_feed: Optional[Union[Dict[str, Any], HistoricalNewsFetcher]] = None
    ) -> Dict[str, Any]:
        """
        Executes a chronological bar-by-bar backtest across all supplied bots.
        """
        bot_list = bots or get_default_tournament_bots()
        if not data:
            raise ValueError("No historical data provided for backtest")

        # 1. Sanitize Data & Compute Vectorized Rolling Indicators (Strictly Backward-Looking)
        sanitized_data, tech_series = self._prepare_data_and_technicals(data)

        # 2. Build Master Chronological Union Calendar
        master_dates = self._build_master_calendar(sanitized_data, start_date, end_date)
        if not master_dates:
            raise ValueError("No trading dates found in the specified backtest range")

        # 3. Initialize Isolated Portfolios
        portfolios = {b.bot_id: SimulatedPortfolio(b, self.initial_balance) for b in bot_list}

        # Keep track of last known prices across holidays
        latest_known_prices: Dict[str, float] = {}

        # Track whether real sentiment feed is active and whether synthetic sentiment was used
        is_real_feed_mode = sentiment_feed is not None
        _backtest_logger = logging.getLogger("BacktestEngine")
        used_synthetic_sentiment = False
        _synthetic_warned = False

        # 4. Chronological Bar-by-Bar Stepping
        for bar_idx, current_date in enumerate(master_dates):
            date_str = str(current_date)[:10]

            # A. Build Current Market Snapshot at bar t
            bar_prices: Dict[str, float] = {}
            bar_technicals: Dict[str, Dict[str, Any]] = {}
            bar_finbert: Dict[str, Dict[str, Any]] = {}
            bar_articles: Dict[str, List[Dict[str, Any]]] = {}
            bar_volumes: Dict[str, float] = {}
            bar_caps: Dict[str, Optional[float]] = {}
            upside_metrics: Dict[str, Dict[str, Any]] = {}

            # Populate snapshot data for active symbols on this date
            for sym, df in sanitized_data.items():
                if current_date in df.index:
                    row = df.loc[current_date]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[-1]

                    close_p = round(float(row["close"]), 2)
                    open_p = round(float(row["open"]), 2)
                    high_p = round(float(row["high"]), 2)
                    low_p = round(float(row["low"]), 2)
                    vol_p = float(row.get("volume", 0.0))

                    bar_prices[sym] = close_p
                    latest_known_prices[sym] = close_p
                    bar_volumes[sym] = vol_p
                    bar_caps[sym] = None  # Market cap can be static or None

                    sma20_val = tech_series[sym]["sma20"].loc[current_date]
                    if isinstance(sma20_val, pd.Series):
                        sma20_val = sma20_val.iloc[-1]
                    sma20_val = round(float(sma20_val), 2) if not pd.isna(sma20_val) else close_p

                    mom5d_val = tech_series[sym]["mom5d"].loc[current_date]
                    if isinstance(mom5d_val, pd.Series):
                        mom5d_val = mom5d_val.iloc[-1]
                    mom5d_val = round(float(mom5d_val), 2) if not pd.isna(mom5d_val) else 0.0

                    bar_technicals[sym] = {
                        "valid": bool(bar_idx >= 20),
                        "price": close_p,
                        "open": open_p,
                        "high": high_p,
                        "low": low_p,
                        "volume": vol_p,
                        "market_cap": None,
                        "sma20": sma20_val,
                        "above_sma20": bool(close_p >= sma20_val),
                        "momentum_5d": mom5d_val,
                        "currency": "INR" if is_inr_symbol(sym) else "USD"
                    }

                    # Sentiment construction
                    # If external sentiment feed has an entry for (date_str, sym)
                    feed_entry = None
                    if sentiment_feed is not None:
                        if isinstance(sentiment_feed, HistoricalNewsFetcher):
                            feed_entry = sentiment_feed.get_news_for_bar(sym, date_str)
                        elif hasattr(sentiment_feed, "get_news_for_bar"):
                            feed_entry = sentiment_feed.get_news_for_bar(sym, date_str)
                        elif isinstance(sentiment_feed, dict):
                            feed_key = f"{date_str}_{sym}"
                            if feed_key in sentiment_feed:
                                feed_entry = sentiment_feed[feed_key]
                            elif (date_str, sym) in sentiment_feed:
                                feed_entry = sentiment_feed[(date_str, sym)]
                            elif date_str in sentiment_feed and isinstance(sentiment_feed[date_str], dict) and sym in sentiment_feed[date_str]:
                                feed_entry = sentiment_feed[date_str][sym]
                            elif sym in sentiment_feed and isinstance(sentiment_feed[sym], dict) and date_str in sentiment_feed[sym]:
                                feed_entry = sentiment_feed[sym][date_str]
                        elif (date_str, sym) in sentiment_feed:
                            feed_entry = sentiment_feed[(date_str, sym)]
                        elif f"{date_str}_{sym}" in sentiment_feed:
                            feed_entry = sentiment_feed[f"{date_str}_{sym}"]

                    if is_real_feed_mode:
                        # REAL HISTORICAL NEWS FEED MODE
                        # When real news feeds are supplied, synthetic sentiment is NEVER generated.
                        if feed_entry is not None:
                            # Real news article(s) found on this trading bar
                            if isinstance(feed_entry, dict) and "bert" in feed_entry:
                                b_dict = dict(feed_entry.get("bert", {}))
                                bar_articles[sym] = list(feed_entry.get("articles", []))
                            elif isinstance(feed_entry, dict):
                                b_dict = dict(feed_entry)
                                bar_articles[sym] = list(feed_entry.get("articles", []))
                            else:
                                b_dict = {}
                                bar_articles[sym] = []
                            b_dict["synthetic"] = False
                            b_dict["has_news"] = True
                            bar_finbert[sym] = b_dict
                        else:
                            # Quiet trading day (no news published on date t)
                            # Gracefully disable sentiment weighting without synthesizing fake sentiment from price action
                            bar_finbert[sym] = {
                                "news_score": 0.0,
                                "net_score": 0.0,
                                "positive": 0.0,
                                "negative": 0.0,
                                "neutral": 1.0,
                                "polarity": "neutral",
                                "confidence": 1.0,
                                "info": "No News (Neutral)",
                                "headline": "",
                                "synthetic": False,
                                "has_news": False,
                                "sentiment_logits": [0.0, 0.0, 1.0],
                                "dense_text_embedding": None,
                                "embedding": None,
                                "calibrated_probs": {
                                    "positive": 0.0,
                                    "negative": 0.0,
                                    "neutral": 1.0
                                }
                            }
                            bar_articles[sym] = []
                    else:
                        # LEGACY SYNTHETIC MODE (only active when sentiment_feed is None)
                        if feed_entry is not None:
                            if isinstance(feed_entry, dict) and "bert" in feed_entry:
                                bar_finbert[sym] = feed_entry.get("bert", {})
                                bar_articles[sym] = feed_entry.get("articles", [])
                            elif isinstance(feed_entry, dict):
                                bar_finbert[sym] = feed_entry
                                bar_articles[sym] = feed_entry.get("articles", [])
                            else:
                                bar_finbert[sym] = {}
                                bar_articles[sym] = []
                        else:
                            # WARNING: Regime-Conditioned Catalyst Proxy (SYNTHETIC).
                            # This generates mock sentiment directly from price moves and
                            # volume surges, creating CIRCULAR LOGIC — sentiment is derived
                            # from the very price data the strategy trades on. Results
                            # using synthetic sentiment are UNRELIABLE for live deployment
                            # validation. Supply a real sentiment_feed dict to get
                            # meaningful backtest results.
                            if not _synthetic_warned:
                                warnings.warn(
                                    "Backtest is using SYNTHETIC sentiment derived from price data. "
                                    "This creates circular logic and results are unreliable. "
                                    "Provide a real sentiment_feed with historical news/FinBERT "
                                    "scores for trustworthy backtest results.",
                                    UserWarning,
                                    stacklevel=2
                                )
                                _backtest_logger.warning(
                                    "Synthetic price-derived sentiment active — backtest results "
                                    "should NOT be used for live trading validation."
                                )
                                _synthetic_warned = True
                            used_synthetic_sentiment = True

                            # Check price delta and volume surge relative to adv20
                            adv = tech_series[sym]["adv20"].loc[current_date] if tech_series[sym]["adv20"] is not None else None
                            if isinstance(adv, pd.Series):
                                adv = adv.iloc[-1]
                            adv_val = float(adv) if adv is not None and not pd.isna(adv) else 100000.0

                            vol_surge = bool(vol_p >= 1.5 * adv_val) and (adv_val > 0)
                            # Estimate daily return from open to close or 5d momentum
                            daily_ret = (close_p - open_p) / open_p if open_p > 0 else 0.0

                            if (vol_surge and daily_ret >= 0.02) or mom5d_val >= 3.0:
                                pos_prob = 0.82
                                neg_prob = 0.08
                                net_sc = 0.65
                                news_sc = 2.6
                                headline = f"{sym} Volume Expansion Breakout"
                                articles_list = [{
                                    "title": headline,
                                    "summary": f"{sym} demonstrates strong volume and upward momentum.",
                                    "source": "Synthetic-Market-Replay"
                                }]
                            elif daily_ret <= -0.025 or mom5d_val <= -4.0:
                                pos_prob = 0.08
                                neg_prob = 0.78
                                net_sc = -0.55
                                news_sc = -2.2
                                headline = f"{sym} Price Retracement"
                                articles_list = [{
                                    "title": headline,
                                    "summary": f"{sym} experiences sharp selling pressure.",
                                    "source": "Synthetic-Market-Replay"
                                }]
                            else:
                                pos_prob = 0.33
                                neg_prob = 0.33
                                net_sc = 0.0
                                news_sc = 0.0
                                headline = f"{sym} Trading Range"
                                articles_list = [{
                                    "title": headline,
                                    "summary": f"{sym} trades in range.",
                                    "source": "Synthetic-Market-Replay"
                                }]

                            bar_finbert[sym] = {
                                "news_score": news_sc,
                                "net_score": net_sc,
                                "positive": pos_prob,
                                "negative": neg_prob,
                                "neutral": 1.0 - (pos_prob + neg_prob),
                                "info": f"FinBERT[Pos={pos_prob:.2f}, Neg={neg_prob:.2f}, Net={net_sc:.2f}]",
                                "headline": headline,
                                "synthetic": True
                            }
                            bar_articles[sym] = articles_list

            # B. Candidate Ranking
            # Qualified candidates: active tradeable symbols or benchmark indices mapping to valid tradeable targets
            qualified_candidates = []
            for sym, price in bar_prices.items():
                tech = bar_technicals.get(sym, {})
                vol = bar_volumes.get(sym)
                mcap = bar_caps.get(sym)

                if is_benchmark_index(sym):
                    tradeable_target = INDEX_TRADEABLE_MAP.get(sym)
                    if not tradeable_target or tradeable_target not in bar_prices:
                        continue
                    target_price = bar_prices.get(tradeable_target)
                    if not target_price or target_price <= 0:
                        continue
                    target_tech = bar_technicals.get(tradeable_target, {})
                    target_vol = bar_volumes.get(tradeable_target)
                    target_mcap = bar_caps.get(tradeable_target)
                    is_safe, _ = self.guardrail.evaluate(
                        tradeable_target,
                        target_price,
                        volume=target_vol,
                        market_cap=target_mcap,
                        currency=target_tech.get("currency")
                    )
                    if not is_safe:
                        continue
                    eval_price = target_price
                    eval_mcap = target_mcap
                    eval_vol = target_vol
                else:
                    is_safe, _ = self.guardrail.evaluate(
                        sym, price, volume=vol, market_cap=mcap, currency=tech.get("currency")
                    )
                    if not is_safe:
                        continue
                    eval_price = price
                    eval_mcap = mcap
                    eval_vol = vol

                # Calculate probability-weighted upside using the candidate's signals
                upside = calculate_probability_weighted_upside(
                    sym,
                    eval_price,
                    tech,
                    bar_finbert.get(sym, {}),
                    bar_articles.get(sym, []),
                    market_cap=eval_mcap,
                    volume=eval_vol
                )
                upside_metrics[sym] = upside
                qualified_candidates.append(sym)

            def _candidate_rank_key(s: str) -> Tuple[float, float]:
                m = upside_metrics.get(s) or {}
                ret = m.get("expected_return_pct")
                prob = m.get("upside_probability_pct")
                f_ret = float(ret) if ret is not None else 0.0
                f_prob = float(prob) if prob is not None else 0.0
                return (f_ret, f_prob)

            qualified_candidates.sort(key=_candidate_rank_key, reverse=True)

            # C. Evaluate Bots
            for b in bot_list:
                port = portfolios[b.bot_id]

                # --- STEP 1: EVALUATE EXITS FIRST ---
                positions_to_remove = []
                for sym, pos in list(port.positions.items()):
                    if sym not in bar_technicals:
                        # Exchange closed for this symbol on current date
                        continue

                    tech = bar_technicals[sym]
                    bert = bar_finbert.get(sym, {})
                    cur_p = bar_prices.get(sym, pos["avg_buy_price"])
                    high_p = tech.get("high", cur_p)
                    low_p = tech.get("low", cur_p)
                    open_p = tech.get("open", cur_p)

                    avg_buy = float(pos["avg_buy_price"])
                    shares_held = float(pos["shares"])
                    pnl_pct = round(((cur_p - avg_buy) / avg_buy * 100.0), 2) if avg_buy > 0 else 0.0

                    # Intrabar Stop Loss and Take Profit
                    # Note: b.stop_loss_pct is negative (e.g. -7.5%)
                    sl_target = avg_buy * (1.0 + b.stop_loss_pct / 100.0)
                    tp_target = avg_buy * (1.0 + b.take_profit_pct / 100.0)

                    triggered_exit = False
                    exit_reason = ""
                    exec_exit_price = cur_p

                    # Pessimistic tie-breaking if both hit
                    if low_p <= sl_target and high_p >= tp_target:
                        triggered_exit = True
                        exec_exit_price = min(open_p, sl_target) * (1.0 - self.slippage_pct)
                        exit_reason = f"Stop Loss triggered: Cut loss at {pnl_pct:.2f}% (limit {b.stop_loss_pct:.1f}%)."
                    elif low_p <= sl_target:
                        triggered_exit = True
                        exec_exit_price = min(open_p, sl_target) * (1.0 - self.slippage_pct)
                        exit_reason = f"Stop Loss triggered: Cut loss at {pnl_pct:.2f}% (limit {b.stop_loss_pct:.1f}%)."
                    elif high_p >= tp_target:
                        triggered_exit = True
                        exec_exit_price = max(open_p, tp_target) * (1.0 - self.slippage_pct)
                        exit_reason = f"Take Profit triggered: Position reached +{pnl_pct:.2f}% (target +{b.take_profit_pct:.1f}%)."
                    else:
                        # Evaluate strategy dynamic exit
                        pos_dict = {
                            "symbol": sym,
                            "shares": shares_held,
                            "average_buy_price": avg_buy,
                            "cur_price": cur_p,
                            "pnl_pct": pnl_pct
                        }
                        should_dyn_exit, dyn_reason = b.evaluate_exit(pos_dict, tech, bert)
                        if should_dyn_exit:
                            triggered_exit = True
                            exec_exit_price = cur_p * (1.0 - self.slippage_pct)
                            exit_reason = dyn_reason

                    if triggered_exit:
                        exec_exit_price = round(exec_exit_price, 2)
                        gross_proceeds = round(shares_held * exec_exit_price, 2)
                        cost_basis = round(shares_held * avg_buy, 2)
                        realized_pnl = round(gross_proceeds - cost_basis, 2)
                        realized_pnl_pct = round((realized_pnl / cost_basis * 100.0), 2) if cost_basis > 0 else 0.0
                        bars_held = bar_idx - pos.get("entry_bar", bar_idx)

                        port.cash = round(port.cash + gross_proceeds, 2)
                        positions_to_remove.append(sym)

                        trade_record = {
                            "trade_id": str(uuid.uuid4()),
                            "bot_id": b.bot_id,
                            "symbol": sym,
                            "action": "SELL",
                            "shares": shares_held,
                            "price": exec_exit_price,
                            "total_value": gross_proceeds,
                            "cost_basis": cost_basis,
                            "pnl": realized_pnl,
                            "pnl_pct": realized_pnl_pct,
                            "holding_period_bars": bars_held,
                            "timestamp": date_str,
                            "rationale": f"[{b.name}] {exit_reason}"
                        }
                        port.trades.append(trade_record)

                        closed_record = {
                            "bot_id": b.bot_id,
                            "symbol": sym,
                            "entry_date": pos.get("entry_date", date_str),
                            "exit_date": date_str,
                            "entry_price": avg_buy,
                            "exit_price": exec_exit_price,
                            "shares": shares_held,
                            "cost_basis": cost_basis,
                            "proceeds": gross_proceeds,
                            "realized_pnl": realized_pnl,
                            "pnl": realized_pnl,
                            "pnl_pct": realized_pnl_pct,
                            "exit_reason": exit_reason,
                            "bars_held": bars_held
                        }
                        port.closed_trades.append(closed_record)

                for sym in positions_to_remove:
                    del port.positions[sym]

                # --- STEP 2: EVALUATE ENTRIES ON RANKED CANDIDATES ---
                open_symbols = list(port.positions.keys())
                for candidate in qualified_candidates:
                    if len(port.positions) >= b.max_positions:
                        break

                    tradeable_target = INDEX_TRADEABLE_MAP.get(candidate, candidate)
                    if tradeable_target in port.positions or candidate in port.positions:
                        continue

                    tech = bar_technicals.get(candidate, {"valid": False})
                    bert = bar_finbert.get(candidate, {})
                    articles = bar_articles.get(candidate, [])

                    should_enter, conviction, rationale = b.evaluate_entry(
                        candidate, tech, bert, articles, open_symbols
                    )

                    if should_enter:
                        # Determine execution price from tradeable target's bar
                        target_price = bar_prices.get(tradeable_target)
                        if not target_price or target_price <= 0:
                            continue

                        # Execute entry with slippage
                        exec_buy_price = round(target_price * (1.0 + self.slippage_pct), 2)
                        current_equity = port.get_total_equity(latest_known_prices)
                        allocation = round(min(current_equity * b.allocation_pct, b.max_trade_dollars), 2)

                        if port.cash >= allocation and exec_buy_price > 0:
                            shares_to_buy = round(allocation / exec_buy_price, 4)
                            actual_cost = round(shares_to_buy * exec_buy_price, 2)

                            if shares_to_buy > 0 and port.cash >= actual_cost:
                                port.cash = round(port.cash - actual_cost, 2)
                                port.positions[tradeable_target] = {
                                    "symbol": tradeable_target,
                                    "shares": shares_to_buy,
                                    "avg_buy_price": exec_buy_price,
                                    "entry_date": date_str,
                                    "entry_bar": bar_idx
                                }

                                up_m = upside_metrics.get(candidate, {})
                                full_rationale = (
                                    f"[{b.name}] {rationale} "
                                    f"[Upside Prob: {up_m.get('upside_probability_pct', 0)}%, "
                                    f"Exp Return: +{up_m.get('expected_return_pct', 0)}%]"
                                )

                                buy_trade = {
                                    "trade_id": str(uuid.uuid4()),
                                    "bot_id": b.bot_id,
                                    "symbol": tradeable_target,
                                    "action": "BUY",
                                    "shares": shares_to_buy,
                                    "price": exec_buy_price,
                                    "total_value": actual_cost,
                                    "timestamp": date_str,
                                    "rationale": full_rationale
                                }
                                port.trades.append(buy_trade)
                                open_symbols.append(tradeable_target)

                # --- STEP 3: MARK TO MARKET & RECORD DAILY EQUITY CURVE ---
                current_total_equity = port.get_total_equity(latest_known_prices)
                port.equity_curve.append((date_str, current_total_equity))

        # 5. Compile Results & Quantitative Metrics
        duration_years = max(0.1, len(master_dates) / 252.0)
        results: Dict[str, Any] = {
            "calendar_dates": [str(d)[:10] for d in master_dates],
            "total_bars": len(master_dates),
            "used_synthetic_sentiment": used_synthetic_sentiment,
            "synthetic_sentiment_used": used_synthetic_sentiment,
            "sentiment_source": "synthetic-price-derived" if used_synthetic_sentiment else "real-news-feed",
            "bots": {}
        }

        if used_synthetic_sentiment:
            results["sentiment_warning"] = (
                "CAUTION: This backtest used synthetic sentiment derived from price data. "
                "Sentiment signals are circularly constructed from the same price/volume "
                "data the strategies trade on. These results should NOT be used for live "
                "deployment validation. Re-run with a real sentiment_feed for reliable results."
            )

        for b in bot_list:
            port = portfolios[b.bot_id]
            equity_values = [eq for _, eq in port.equity_curve]
            final_equity = equity_values[-1] if equity_values else port.cash

            metrics = calculate_performance_metrics(
                port.closed_trades,
                equity_values,
                initial_balance=self.initial_balance,
                years=duration_years
            )
            metrics["synthetic_sentiment_used"] = used_synthetic_sentiment
            metrics["used_synthetic_sentiment"] = used_synthetic_sentiment

            bot_summary = {
                "bot_id": b.bot_id,
                "name": b.name,
                "persona": b.persona,
                "initial_balance": self.initial_balance,
                "final_equity": final_equity,
                "final_cash": round(port.cash, 2),
                "metrics": metrics,
                "trades": port.trades,
                "closed_trades": port.closed_trades,
                "equity_curve": port.equity_curve,
                "open_positions": port.positions
            }
            results["bots"][b.bot_id] = bot_summary

        return results
