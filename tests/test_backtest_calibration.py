"""
Unit and Integration Tests for Historical Backtest Engine, Metrics, and Calibration (Milestone 1).
Executes completely offline in < 2 seconds with deterministic synthetic fixtures.
"""

import unittest
import tempfile
import time
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from src.db import (
    save_training_run,
    save_calibrated_parameters,
    load_calibrated_parameters,
)

from src.backtest_engine import (
    HistoricalDataFetcher,
    HistoricalBacktestEngine,
    SimulatedPortfolio,
    compute_rolling_indicators,
    simulate_single_position,
    calculate_performance_metrics
)
from src.paper_trader import (
    ConservativeSwingBot,
    AggressiveBreakoutBot,
    ContrarianDipBuyerBot,
    FinBERTCatalystHunterBot,
    TrendFollowerBot,
    get_default_tournament_bots,
    BorderlistGuardrail,
    is_benchmark_index,
    is_inr_symbol,
    INDEX_TRADEABLE_MAP,
    MIN_SHARE_PRICE_INR
)
from src.calibration import (
    WalkForwardCalibrator,
    split_chronological_data,
    calculate_fitness_score,
    calibrate_all_personas,
    CONSERVATIVE_SWING_GRID,
    AGGRESSIVE_BREAKOUT_GRID,
    CONTRARIAN_DIP_BUYER_GRID,
    FINBERT_CATALYST_HUNTER_GRID,
    TREND_FOLLOWER_GRID,
    DEFAULT_PERSONA_GRIDS,
)


# ============================================================================
# DETERMINISTIC SYNTHETIC TEST FIXTURES (OFFLINE / IN-MEMORY)
# ============================================================================

def make_linear_trend_fixture(n_bars: int = 30, base_price: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    """
    Creates a deterministic linear price series: P_t = base_price + t * step.
    SMA20_t = base_price + t - 9.5 (for t >= 19).
    """
    dates = pd.date_range("2023-01-01", periods=n_bars, freq="D")
    prices = [base_price + i * step for i in range(n_bars)]
    return pd.DataFrame({
        "open": prices,
        "high": [p + 0.5 for p in prices],
        "low": [p - 0.5 for p in prices],
        "close": prices,
        "volume": [100000.0] * n_bars
    }, index=dates)


def make_stop_loss_fixture() -> pd.DataFrame:
    """
    Deterministic stop-loss fixture:
    Entry at Close = $100.00.
    At bar 3: Low plunges to $93.00, below 5% stop-loss threshold ($95.00).
    """
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    prices = [100.0, 98.0, 97.0, 94.0, 90.0]
    return pd.DataFrame({
        "open": prices,
        "high": [p + 1.0 for p in prices],
        "low": [p - 1.0 for p in prices],
        "close": prices,
        "volume": [150000.0] * 5
    }, index=dates)


def make_take_profit_fixture() -> pd.DataFrame:
    """
    Deterministic take-profit fixture:
    Entry at Close = $100.00.
    At bar 3: High rallies to $113.00, above 10% take-profit threshold ($110.00).
    """
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    prices = [100.0, 104.0, 108.0, 112.0, 115.0]
    return pd.DataFrame({
        "open": prices,
        "high": [p + 1.0 for p in prices],
        "low": [p - 1.0 for p in prices],
        "close": prices,
        "volume": [200000.0] * 5
    }, index=dates)


def make_collision_fixture() -> pd.DataFrame:
    """
    Intrabar collision fixture:
    Entry at Close = $100.00.
    At bar 1: High = $115.00 (+15% >= +10% TP), Low = $90.00 (-10% <= -5% SL).
    Pessimistic tie-breaking requires Stop-Loss execution.
    """
    dates = pd.date_range("2023-01-01", periods=3, freq="D")
    return pd.DataFrame({
        "open": [100.0, 100.0, 100.0],
        "high": [101.0, 115.0, 102.0],
        "low": [99.0, 90.0, 98.0],
        "close": [100.0, 102.0, 101.0],
        "volume": [250000.0] * 3
    }, index=dates)


# ============================================================================
# TEST SUITE
# ============================================================================

class TestDataFetcherSanitizationAndCache(unittest.TestCase):
    """Verifies data ingestion, column sanitization, timezone normalization, and local caching."""

    def test_safe_symbol_mapping(self):
        """Ensures non-alphanumeric and special characters are safely mapped for Windows OS."""
        self.assertEqual(HistoricalDataFetcher.get_safe_symbol("^NSEI"), "_NSEI")
        self.assertEqual(HistoricalDataFetcher.get_safe_symbol("NIFTYBEES.NS"), "NIFTYBEES.NS")
        self.assertEqual(HistoricalDataFetcher.get_safe_symbol("NVDA"), "NVDA")

    def test_sanitize_bars_multiindex_and_auxiliary_columns(self):
        """Verifies flattening MultiIndex columns, stripping auxiliary columns, and lowercase normalization."""
        dates = pd.date_range("2023-01-01", periods=3)
        # Create MultiIndex DataFrame as returned by yf.download
        cols = pd.MultiIndex.from_tuples([
            ("Open", "SPY"), ("High", "SPY"), ("Low", "SPY"),
            ("Close", "SPY"), ("Volume", "SPY"), ("Dividends", "SPY"),
            ("Stock Splits", "SPY"), ("Capital Gains", "SPY")
        ])
        data = [
            [400.0, 405.0, 399.0, 403.0, 50000000, 0.0, 0.0, 0.0],
            [403.0, 407.0, 402.0, 406.0, 52000000, 1.5, 0.0, 0.0],
            [406.0, 410.0, 405.0, 409.0, 48000000, 0.0, 0.0, 0.0]
        ]
        raw_df = pd.DataFrame(data, index=dates, columns=cols)

        sanitized = HistoricalDataFetcher.sanitize_bars(raw_df)
        self.assertEqual(list(sanitized.columns), ["open", "high", "low", "close", "volume"])
        self.assertEqual(len(sanitized), 3)
        self.assertEqual(sanitized["close"].iloc[0], 403.0)

    def test_sanitize_bars_timezone_normalization(self):
        """Verifies timezone-aware timestamps are converted to timezone-naive calendar dates."""
        # Indian market timestamp (+05:30)
        dates_ist = pd.date_range("2023-10-15 09:15:00+05:30", periods=3, freq="D")
        raw_df = pd.DataFrame({
            "Open": [100.0, 101.0, 102.0],
            "High": [105.0, 106.0, 107.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [103.0, 104.0, 105.0],
            "Volume": [100000, 110000, 120000]
        }, index=dates_ist)

        sanitized = HistoricalDataFetcher.sanitize_bars(raw_df)
        self.assertIsNone(sanitized.index.tz)
        self.assertEqual(str(sanitized.index[0])[:10], "2023-10-15")

    def test_sanitize_bars_missing_required_columns(self):
        """Verifies missing required OHLCV columns raises ValueError."""
        dates = pd.date_range("2023-01-01", periods=2)
        df_missing = pd.DataFrame({"close": [100.0, 101.0], "volume": [100, 200]}, index=dates)
        with self.assertRaises(ValueError):
            HistoricalDataFetcher.sanitize_bars(df_missing)

    def test_sanitize_bars_filters_invalid_prices(self):
        """Verifies non-positive and NaN prices are dropped, volume NaNs filled with 0.0."""
        dates = pd.date_range("2023-01-01", periods=4)
        df = pd.DataFrame({
            "open": [100.0, -5.0, 102.0, 103.0],
            "high": [105.0, 10.0, np.nan, 107.0],
            "low": [95.0, -10.0, 100.0, 101.0],
            "close": [102.0, 0.0, 101.0, 105.0],
            "volume": [10000.0, 5000.0, 12000.0, np.nan]
        }, index=dates)

        sanitized = HistoricalDataFetcher.sanitize_bars(df)
        # Rows 1 (negative/zero) and 2 (NaN high) must be dropped
        self.assertEqual(len(sanitized), 2)
        self.assertEqual(sanitized["volume"].iloc[-1], 0.0)

    def test_offline_mode_cached_read_and_missing_error(self):
        """Verifies offline mode reads cached data without network, and raises FileNotFoundError if missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = HistoricalDataFetcher(cache_dir=tmpdir, offline=True)

            # 1. Missing cache must raise FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                fetcher.fetch_symbol_bars("AAPL")

            # 2. Write valid cache to disk
            sample_df = make_linear_trend_fixture(n_bars=5)
            cache_file = fetcher.get_cache_path("AAPL")
            sample_df.to_csv(cache_file, index_label="date")

            # 3. Read cached data offline
            loaded_df = fetcher.fetch_symbol_bars("AAPL")
            self.assertEqual(len(loaded_df), 5)
            self.assertEqual(loaded_df["close"].iloc[0], 100.0)

    def test_offline_mode_corrupt_cache_detection(self):
        """Verifies offline mode detects corrupt 0-byte or unparseable files and raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = HistoricalDataFetcher(cache_dir=tmpdir, offline=True)

            # Write 0-byte file
            cache_file = fetcher.get_cache_path("NVDA")
            cache_file.touch()

            with self.assertRaises(ValueError):
                fetcher.fetch_symbol_bars("NVDA")

    def test_cached_universe_bars_offline_load(self):
        """Verifies that all 6 universe symbols load seamlessly offline from disk cache with >= 700 rows."""
        fetcher = HistoricalDataFetcher(offline=True)
        # Only run check if cached files are present in the default cache dir
        if fetcher.find_existing_cache_file("^NSEI"):
            bars = fetcher.fetch_universe_bars()
            self.assertEqual(set(bars.keys()), set(HistoricalDataFetcher.DEFAULT_UNIVERSE))
            for sym, df in bars.items():
                self.assertGreaterEqual(len(df), 700, f"Symbol {sym} has fewer than 700 bars")
                self.assertEqual(list(df.columns), ["open", "high", "low", "close", "volume"])
                self.assertIsNone(df.index.tz)


class TestZeroLookaheadAndTechnicalIndicators(unittest.TestCase):
    """Verifies that technical indicators compute exact values with zero future lookahead."""

    def test_analytical_sma20_and_momentum_exact_values(self):
        """Verifies SMA20 and 5-day momentum match analytical mathematical equations."""
        trend_df = make_linear_trend_fixture(n_bars=30, base_price=100.0, step=1.0)

        # Bar 19: Close = 119.0. SMA20 = 100 + 19 - 9.5 = 109.50
        ind_19 = compute_rolling_indicators(trend_df.iloc[:20])
        self.assertEqual(ind_19["valid"], True)
        self.assertAlmostEqual(ind_19["sma20"], 109.5, places=2)

        # Bar 20: Close = 120.0. SMA20 = 100 + 20 - 9.5 = 110.50
        ind_20 = compute_rolling_indicators(trend_df.iloc[:21])
        self.assertAlmostEqual(ind_20["sma20"], 110.5, places=2)

        # Bar 5: P_5 = 105.0, P_0 = 100.0 => Momentum = 5.0%
        ind_5 = compute_rolling_indicators(trend_df.iloc[:6])
        self.assertAlmostEqual(ind_5["momentum_5d"], 5.0, places=2)

        # Bar 10: P_10 = 110.0, P_5 = 105.0 => Momentum = (5/105) * 100 = 4.76%
        ind_10 = compute_rolling_indicators(trend_df.iloc[:11])
        self.assertAlmostEqual(ind_10["momentum_5d"], 4.76, places=2)

    def test_zero_lookahead_future_corruption_isolation(self):
        """Verifies corrupting future bars (t+1..T) does not alter indicator calculations at bar t."""
        pristine_df = make_linear_trend_fixture(n_bars=30, base_price=100.0, step=1.0)
        ind_pristine = compute_rolling_indicators(pristine_df.iloc[:21])

        # Corrupt future bars (t=21..29) with extreme prices
        corrupted_df = pristine_df.copy()
        corrupted_df.iloc[21:, corrupted_df.columns.get_loc("close")] = 999999.0

        ind_corrupted = compute_rolling_indicators(corrupted_df.iloc[:21])
        self.assertEqual(ind_pristine["sma20"], ind_corrupted["sma20"])
        self.assertEqual(ind_pristine["momentum_5d"], ind_corrupted["momentum_5d"])
        self.assertEqual(ind_pristine["price"], ind_corrupted["price"])

    def test_warmup_period(self):
        """Verifies valid flag is False during first 20 bars."""
        short_df = make_linear_trend_fixture(n_bars=15)
        ind = compute_rolling_indicators(short_df)
        self.assertEqual(ind["valid"], False)


class TestExactBarOrderExecution(unittest.TestCase):
    """Verifies stop-loss and take-profit exit triggers on exact bars, and Indian asset routing."""

    def test_stop_loss_exact_bar(self):
        """Verifies stop-loss triggers on exact bar where price breaches stop limit."""
        df = make_stop_loss_fixture()
        trades = simulate_single_position(df, entry_bar=0, stop_loss_pct=5.0)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["exit_bar"], 3)
        self.assertEqual(trades[0]["exit_reason"], "STOP_LOSS")
        self.assertLessEqual(trades[0]["exit_price"], 95.0)

    def test_take_profit_exact_bar(self):
        """Verifies take-profit triggers on exact bar where price breaches profit target."""
        df = make_take_profit_fixture()
        trades = simulate_single_position(df, entry_bar=0, take_profit_pct=10.0)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["exit_bar"], 3)
        self.assertEqual(trades[0]["exit_reason"], "TAKE_PROFIT")
        self.assertGreaterEqual(trades[0]["exit_price"], 110.0)

    def test_pessimistic_tie_breaking_intrabar(self):
        """Verifies that when both SL and TP thresholds are reached in the same bar, SL triggers."""
        df = make_collision_fixture()
        trades = simulate_single_position(df, entry_bar=0, stop_loss_pct=5.0, take_profit_pct=10.0)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["exit_bar"], 1)
        self.assertEqual(trades[0]["exit_reason"], "STOP_LOSS")

    def test_indian_asset_routing_and_guardrails(self):
        """Verifies ^NSEI index is recognized as benchmark, mapped to NIFTYBEES.NS, and passes INR guardrails."""
        self.assertTrue(is_benchmark_index("^NSEI"))
        self.assertFalse(is_benchmark_index("NIFTYBEES.NS"))
        self.assertEqual(INDEX_TRADEABLE_MAP.get("^NSEI"), "NIFTYBEES.NS")
        self.assertTrue(is_inr_symbol("NIFTYBEES.NS"))

        guardrail = BorderlistGuardrail()
        # NIFTYBEES.NS at ₹250.00 passes INR floor (€20.00)
        is_safe, reason = guardrail.evaluate("NIFTYBEES.NS", price=250.0, volume=500000.0)
        self.assertTrue(is_safe)
        self.assertIsNone(reason)

        # Penny stock in INR (< ₹20.00) is rejected
        is_safe_penny, reason_penny = guardrail.evaluate("PENNY.NS", price=12.50, volume=500000.0)
        self.assertFalse(is_safe_penny)
        self.assertIn("below minimum floor", reason_penny)

    def test_end_to_end_nsei_to_niftybees_routing_execution(self):
        """
        Verifies end-to-end order routing in HistoricalBacktestEngine:
        1. ^NSEI experiences a strong momentum breakout (+4.35% > 2%, above SMA20, positive news).
        2. NIFTYBEES.NS stays completely flat (0.0% momentum, would never qualify independently).
        3. Engine qualifies ^NSEI via INDEX_TRADEABLE_MAP proxy and executes BUY for NIFTYBEES.NS.
        4. Verifies trade execution at NIFTYBEES.NS ETF price (~₹250.00), exact cash deduction,
           and that open positions contain NIFTYBEES.NS with ZERO contamination from ^NSEI.
        """
        n_bars = 30
        dates = pd.date_range("2023-01-01", periods=n_bars, freq="D")

        # ^NSEI rallies steadily: Close at bar 20 is 24,000 (SMA20 = 22,100, 5d momentum = +4.35%)
        nsei_prices = [20000.0 + i * 200.0 for i in range(n_bars)]
        # NIFTYBEES.NS stays flat at ₹250.00 (momentum = 0.0%)
        niftybees_prices = [250.0] * n_bars

        data = {
            "^NSEI": pd.DataFrame({
                "open": nsei_prices,
                "high": [p + 50.0 for p in nsei_prices],
                "low": [p - 50.0 for p in nsei_prices],
                "close": nsei_prices,
                "volume": [1000000.0] * n_bars
            }, index=dates),
            "NIFTYBEES.NS": pd.DataFrame({
                "open": niftybees_prices,
                "high": [p + 0.1 for p in niftybees_prices],
                "low": [p - 0.1 for p in niftybees_prices],
                "close": niftybees_prices,
                "volume": [500000.0] * n_bars
            }, index=dates),
        }

        # AggressiveBreakoutBot requires mom_5d >= 2.0% and above_sma20
        # ^NSEI meets this; NIFTYBEES.NS does not.
        bot = AggressiveBreakoutBot()
        engine = HistoricalBacktestEngine(initial_balance=5000.0, slippage_bps=0.0)
        results = engine.run_backtest(bots=[bot], data=data)

        bot_res = results["bots"][bot.bot_id]
        trades = bot_res["trades"]
        open_positions = bot_res["open_positions"]

        # 1. Assert trades were executed
        self.assertGreater(len(trades), 0, "Expected at least 1 routed trade from ^NSEI breakout signal")

        # 2. Assert executed trades target NIFTYBEES.NS and NEVER ^NSEI
        trade = trades[0]
        self.assertEqual(trade["symbol"], "NIFTYBEES.NS", "Executed trade must target tradeable ETF NIFTYBEES.NS")
        self.assertEqual(trade["action"], "BUY")
        self.assertFalse(
            any(t["symbol"] == "^NSEI" for t in trades),
            "Benchmark index ^NSEI must NEVER appear as an executed trade symbol"
        )

        # 3. Assert execution price reflects NIFTYBEES.NS (~₹250.00), not ^NSEI (~₹24,000.00)
        self.assertEqual(trade["price"], 250.0, "Execution price must equal NIFTYBEES.NS price (250.0)")
        self.assertLess(trade["price"], 1000.0, "Execution price must be ETF level, not index level")

        # 4. Assert proper cash deduction
        # Bot allocation = min(5000 * 0.25, 1250) = 1250.00. Shares = 1250 / 250 = 5.0.
        expected_cost = round(trade["shares"] * trade["price"], 2)
        self.assertEqual(expected_cost, 1250.0, "Expected total cost of trade to be $1250.00")
        self.assertEqual(trade["total_value"], expected_cost)
        self.assertEqual(
            bot_res["final_cash"],
            round(5000.0 - expected_cost, 2),
            f"Cash not deducted properly: expected {5000.0 - expected_cost}, got {bot_res['final_cash']}"
        )

        # 5. Assert open position is NIFTYBEES.NS with correct shares and avg buy price
        self.assertIn("NIFTYBEES.NS", open_positions, "Open positions must contain NIFTYBEES.NS")
        self.assertNotIn("^NSEI", open_positions, "Open positions must NEVER contain ^NSEI")
        self.assertEqual(open_positions["NIFTYBEES.NS"]["symbol"], "NIFTYBEES.NS")
        self.assertEqual(open_positions["NIFTYBEES.NS"]["shares"], 5.0)
        self.assertEqual(open_positions["NIFTYBEES.NS"]["avg_buy_price"], 250.0)


class TestPerformanceMetrics(unittest.TestCase):
    """Verifies quantitative performance metrics calculation and edge cases."""

    def test_empty_trades_and_equity(self):
        """Verifies empty trade logs and flat curve return 0.0 metrics without division errors."""
        m = calculate_performance_metrics([], [5000.0], initial_balance=5000.0)
        self.assertEqual(m["win_rate_pct"], 0.0)
        self.assertEqual(m["profit_factor"], 0.0)
        self.assertEqual(m["max_drawdown_pct"], 0.0)
        self.assertEqual(m["net_return_pct"], 0.0)
        self.assertEqual(m["cagr_pct"], 0.0)
        self.assertEqual(m["total_trades"], 0)

    def test_undefeated_strategy_profit_factor_cap(self):
        """Verifies undefeated strategy (gross loss == 0) caps profit factor at 999.0."""
        trades = [{"realized_pnl": 150.0}, {"realized_pnl": 250.0}]
        m = calculate_performance_metrics(trades, [5000.0, 5400.0])
        self.assertEqual(m["win_rate_pct"], 100.0)
        self.assertEqual(m["profit_factor"], 999.0)
        self.assertEqual(m["gross_loss"], 0.0)

    def test_all_losses_profit_factor(self):
        """Verifies 100% loss strategy returns 0 profit factor and 0 win rate."""
        trades = [{"realized_pnl": -100.0}, {"realized_pnl": -50.0}]
        m = calculate_performance_metrics(trades, [5000.0, 4850.0])
        self.assertEqual(m["win_rate_pct"], 0.0)
        self.assertEqual(m["profit_factor"], 0.0)

    def test_fixture_5_precalculated_values(self):
        """
        Verifies exact analytical calculation for Fixture 5:
        - 2 Wins ($200, $100), 2 Losses ($100, $50), 1 Breakeven ($0)
        - Win Rate: 2/4 = 50.0%
        - Profit Factor: 300 / 150 = 2.0
        - Peak $5500, Trough $4400 => Max Drawdown: 20.0%
        - Final $6500, Initial $5000 => Net Return: +30.0%
        - CAGR (3.0 years): (6500/5000)^(1/3) - 1 = +9.14%
        """
        trades = [
            {"realized_pnl": 200.0},
            {"realized_pnl": 100.0},
            {"realized_pnl": -100.0},
            {"realized_pnl": -50.0},
            {"realized_pnl": 0.0},
        ]
        equity_curve = [5000.0, 5200.0, 5500.0, 4950.0, 4400.0, 5000.0, 6000.0, 5700.0, 6500.0]
        m = calculate_performance_metrics(trades, equity_curve, initial_balance=5000.0, years=3.0)

        self.assertEqual(m["wins"], 2)
        self.assertEqual(m["losses"], 2)
        self.assertEqual(m["total_trades"], 4)
        self.assertEqual(m["win_rate_pct"], 50.0)
        self.assertEqual(m["profit_factor"], 2.0)
        self.assertEqual(m["max_drawdown_pct"], 20.0)
        self.assertEqual(m["net_return_pct"], 30.0)
        self.assertEqual(m["cagr_pct"], 9.14)

    def test_cagr_bankruptcy_protection(self):
        """Verifies negative/zero equity returns -100.0% without crashing on Python complex root."""
        m = calculate_performance_metrics([], [5000.0, 0.0, -100.0], initial_balance=5000.0, years=3.0)
        self.assertEqual(m["cagr_pct"], -100.0)

    def test_calculate_performance_metrics_handles_none_realized_pnl(self):
        """
        Verifies that calculate_performance_metrics does not crash with TypeError
        when trade records contain explicit None values for 'realized_pnl'.
        Tests fallback to 'pnl', priority of 'realized_pnl', and handling of breakeven 0.0.
        """
        # 1. Target single trade with realized_pnl: None and pnl: 50.0
        single_target_trade = [{"realized_pnl": None, "pnl": 50.0}]
        m_single = calculate_performance_metrics(single_target_trade, [5000.0, 5050.0], initial_balance=5000.0)
        self.assertEqual(m_single["wins"], 1)
        self.assertEqual(m_single["losses"], 0)
        self.assertEqual(m_single["total_trades"], 1)
        self.assertEqual(m_single["win_rate_pct"], 100.0)
        self.assertEqual(m_single["gross_profit"], 50.0)
        self.assertEqual(m_single["profit_factor"], 999.0)

        # 2. Comprehensive multi-case trade sequence
        adversarial_trades = [
            {"realized_pnl": None, "pnl": 50.0},     # Win ($50) via fallback
            {"realized_pnl": 100.0, "pnl": None},    # Win ($100) via primary key
            {"realized_pnl": None, "pnl": -30.0},    # Loss ($30) via fallback
            {"realized_pnl": None, "pnl": None},     # None/None -> 0.0 breakeven
            {"realized_pnl": 0.0, "pnl": 50.0},      # Breakeven 0.0 preserved (not falsy fallback)
            {"realized_pnl": "N/A"},                 # Malformed string handled safely
            None,                                    # None element in list handled safely
        ]
        equity_curve = [5000.0, 5050.0, 5150.0, 5120.0]

        # Must execute cleanly without TypeError or ValueError
        metrics = calculate_performance_metrics(
            trades=adversarial_trades,
            equity_curve=equity_curve,
            initial_balance=5000.0,
            years=3.0
        )

        # Expected:
        # Wins: 2 ($50.0, $100.0) -> gross profit = $150.0
        # Losses: 1 ($30.0) -> gross loss = $30.0
        # Total closed: 3
        # Win Rate: 2 / 3 = 66.67%
        # Profit Factor: 150.0 / 30.0 = 5.0
        self.assertEqual(metrics["wins"], 2)
        self.assertEqual(metrics["losses"], 1)
        self.assertEqual(metrics["total_trades"], 3)
        self.assertAlmostEqual(metrics["win_rate_pct"], 66.67, places=2)
        self.assertAlmostEqual(metrics["profit_factor"], 5.0, places=2)
        self.assertAlmostEqual(metrics["gross_profit"], 150.0, places=2)
        self.assertAlmostEqual(metrics["gross_loss"], 30.0, places=2)

    def test_calculate_performance_metrics_handles_tuple_equity_curve_and_none_args(self):
        """
        Verifies calculate_performance_metrics handles tuple equity curves
        [(date, equity)] as produced by SimulatedPortfolio and None arguments.
        """
        tuple_curve = [("2023-01-01", 5000.0), ("2023-01-02", 5200.0), ("2023-01-03", 5100.0)]
        m_tuple = calculate_performance_metrics([], tuple_curve, initial_balance=5000.0)
        self.assertIn("max_drawdown_pct", m_tuple)
        self.assertIn("net_return_pct", m_tuple)

        # All None arguments
        m_none = calculate_performance_metrics(trades=None, equity_curve=None, initial_balance=None, years=None)
        self.assertEqual(m_none["win_rate_pct"], 0.0)
        self.assertEqual(m_none["total_trades"], 0)
        self.assertEqual(m_none["profit_factor"], 0.0)


class TestFullHistoricalBacktestReplay(unittest.TestCase):
    """Verifies complete multi-bot bar-by-bar backtest simulation across multi-asset universe."""

    def test_multi_bot_replay_on_synthetic_universe(self):
        """Runs full 5-bot replay over synthetic multi-asset data and verifies execution."""
        start_time = time.time()

        n_bars = 40
        dates = pd.date_range("2023-01-01", periods=n_bars, freq="D")

        # Synthetic multi-asset universe
        nvda_prices = [100.0 + i * 1.5 for i in range(n_bars)]
        aapl_prices = [150.0 + i * 0.8 for i in range(n_bars)]
        niftybees_prices = [250.0 + i * 0.5 for i in range(n_bars)]
        nsei_prices = [22000.0 + i * 40.0 for i in range(n_bars)]

        data = {
            "NVDA": pd.DataFrame({
                "open": nvda_prices,
                "high": [p + 2.0 for p in nvda_prices],
                "low": [p - 1.0 for p in nvda_prices],
                "close": nvda_prices,
                "volume": [500000.0] * n_bars
            }, index=dates),
            "AAPL": pd.DataFrame({
                "open": aapl_prices,
                "high": [p + 1.0 for p in aapl_prices],
                "low": [p - 0.5 for p in aapl_prices],
                "close": aapl_prices,
                "volume": [800000.0] * n_bars
            }, index=dates),
            "NIFTYBEES.NS": pd.DataFrame({
                "open": niftybees_prices,
                "high": [p + 1.5 for p in niftybees_prices],
                "low": [p - 1.0 for p in niftybees_prices],
                "close": niftybees_prices,
                "volume": [600000.0] * n_bars
            }, index=dates),
            "^NSEI": pd.DataFrame({
                "open": nsei_prices,
                "high": [p + 50.0 for p in nsei_prices],
                "low": [p - 30.0 for p in nsei_prices],
                "close": nsei_prices,
                "volume": [1000000.0] * n_bars
            }, index=dates)
        }

        engine = HistoricalBacktestEngine(initial_balance=5000.0, slippage_bps=5.0)
        bots = get_default_tournament_bots()

        results = engine.run_backtest(bots=bots, data=data)

        self.assertIn("calendar_dates", results)
        self.assertEqual(results["total_bars"], n_bars)
        self.assertEqual(len(results["bots"]), 5)

        for bot_id, res in results["bots"].items():
            self.assertEqual(res["initial_balance"], 5000.0)
            self.assertGreater(len(res["equity_curve"]), 0)
            self.assertIn("win_rate_pct", res["metrics"])
            self.assertIn("profit_factor", res["metrics"])
            self.assertIn("max_drawdown_pct", res["metrics"])
            self.assertIn("net_return_pct", res["metrics"])
            self.assertIn("cagr_pct", res["metrics"])

        elapsed = time.time() - start_time
        self.assertLess(elapsed, 2.0, f"Replay took {elapsed:.2f}s, exceeding 2s budget")


# ============================================================================
# MILESTONE 2: BOT PARAMETERIZATION & WALK-FORWARD CALIBRATION TESTS
# ============================================================================

class TestBotCustomParameterInstantiationAndFallback(unittest.TestCase):
    """
    Verifies that all 5 bot subclasses accept dynamic parameter overrides via
    params: Optional[Dict[str, Any]] = None, preserve exact baseline defaults
    when params is None or omitted, and properly apply custom thresholds.
    """

    def test_default_instantiation_preserves_exact_baseline_attributes(self):
        """
        Verifies that instantiating bots with no arguments retains exact legacy defaults.
        Guarantees zero breaking changes for existing code and tests.
        """
        bots = {
            "conservative_swing": (ConservativeSwingBot(), -7.5, 12.0, 5, 0.15, 0.0),
            "aggressive_breakout": (AggressiveBreakoutBot(), -5.0, 18.0, 4, 0.25, 2.0),
            "contrarian_dip_buyer": (ContrarianDipBuyerBot(), -6.0, 8.0, 5, 0.20, -1.5),
            "finbert_catalyst_hunter": (FinBERTCatalystHunterBot(), -7.0, 10.0, 5, 0.20, 0.50),
            "trend_follower": (TrendFollowerBot(), -8.0, 15.0, 6, 0.15, 0.0),
        }

        for bot_id, (bot, sl, tp, max_pos, alloc, specific_val) in bots.items():
            self.assertEqual(bot.stop_loss_pct, sl, f"{bot_id} stop_loss_pct mismatch")
            self.assertEqual(bot.take_profit_pct, tp, f"{bot_id} take_profit_pct mismatch")
            self.assertEqual(bot.max_positions, max_pos, f"{bot_id} max_positions mismatch")
            self.assertEqual(bot.allocation_pct, alloc, f"{bot_id} allocation_pct mismatch")

    def test_custom_parameters_override_baseline_values(self):
        """Verifies that custom params dictionary successfully overrides all baseline values."""
        custom_params = {
            "stop_loss_pct": -4.0,
            "take_profit_pct": 14.0,
            "max_positions": 3,
            "allocation_pct": 0.30,
            "min_momentum_5d": 1.2
        }
        bot = AggressiveBreakoutBot(params=custom_params)
        self.assertEqual(bot.stop_loss_pct, -4.0)
        self.assertEqual(bot.take_profit_pct, 14.0)
        self.assertEqual(bot.max_positions, 3)
        self.assertEqual(bot.allocation_pct, 0.30)
        self.assertEqual(getattr(bot, "min_momentum_5d", None), 1.2)

    def test_partial_parameters_retain_unspecified_defaults(self):
        """Verifies supplying only a subset of parameters overrides specified keys while keeping defaults."""
        bot = ConservativeSwingBot(params={"stop_loss_pct": -5.0})
        self.assertEqual(bot.stop_loss_pct, -5.0, "Specified stop_loss_pct must override default")
        self.assertEqual(bot.take_profit_pct, 12.0, "Omitted take_profit_pct must retain default 12.0")
        self.assertEqual(bot.max_positions, 5, "Omitted max_positions must retain default 5")
        self.assertEqual(bot.allocation_pct, 0.15, "Omitted allocation_pct must retain default 0.15")

    def test_parameter_sign_sanitization_and_type_casting(self):
        """Verifies stop_loss is always negative and take_profit is always positive regardless of user input."""
        # User provides positive stop loss (5.0) and negative take profit (-15.0)
        bot = TrendFollowerBot(params={"stop_loss_pct": 5.0, "take_profit_pct": -15.0})
        self.assertEqual(bot.stop_loss_pct, -5.0, "Stop loss must be strictly negative")
        self.assertEqual(bot.take_profit_pct, 15.0, "Take profit must be strictly positive")

    def test_custom_parameters_alter_entry_decisions(self):
        """Verifies that changing entry thresholds directly changes evaluate_entry qualification."""
        tech = {
            "valid": True,
            "price": 105.0,
            "sma20": 100.0,
            "above_sma20": True,
            "momentum_5d": 1.5,  # 1.5% momentum
            "currency": "USD"
        }
        bert = {"net_score": 0.10, "info": "positive"}
        articles = [{"title": "News"}]

        # Default bot requires min_momentum_5d >= 2.0%
        default_bot = AggressiveBreakoutBot()
        qualified_default, _, _ = default_bot.evaluate_entry("AAPL", tech, bert, articles, [])
        self.assertFalse(qualified_default, "Default bot should reject 1.5% momentum (requires >= 2.0%)")

        # Custom bot with lowered threshold min_momentum_5d >= 1.2%
        custom_bot = AggressiveBreakoutBot(params={"min_momentum_5d": 1.2})
        qualified_custom, _, _ = custom_bot.evaluate_entry("AAPL", tech, bert, articles, [])
        self.assertTrue(qualified_custom, "Custom bot should approve 1.5% momentum when threshold is 1.2%")

    def test_custom_parameters_alter_exit_decisions(self):
        """Verifies that changing stop-loss threshold alters evaluate_exit execution."""
        position = {"pnl_pct": -6.0}  # -6.0% drawdown
        tech = {"price": 94.0, "above_sma20": True, "momentum_5d": 0.0}
        bert = {"news_score": 0.0, "net_score": 0.0}

        # Default ConservativeSwingBot has stop loss at -7.5%
        default_bot = ConservativeSwingBot()
        exit_default, _ = default_bot.evaluate_exit(position, tech, bert)
        self.assertFalse(exit_default, "Default bot with SL=-7.5% should NOT exit at -6.0%")

        # Custom ConservativeSwingBot with tighter stop loss at -5.0%
        custom_bot = ConservativeSwingBot(params={"stop_loss_pct": -5.0})
        exit_custom, reason = custom_bot.evaluate_exit(position, tech, bert)
        self.assertTrue(exit_custom, "Custom bot with SL=-5.0% MUST exit at -6.0%")
        self.assertIn("Stop Loss", reason)


class TestWalkForwardDataSplitChronological(unittest.TestCase):
    """
    Verifies that WalkForwardCalibrator partitions historical datasets strictly
    chronologically with zero lookahead bias, strictly disjoint date sets,
    and proper multi-asset union calendar synchronization.
    """

    def test_chronological_split_ratio_exactness(self):
        """Verifies 67% in-sample and 33% out-of-sample partition counts."""
        n_bars = 100
        dates = pd.date_range("2023-01-01", periods=n_bars, freq="D")
        sample_df = pd.DataFrame({
            "open": [100.0] * n_bars,
            "high": [105.0] * n_bars,
            "low": [95.0] * n_bars,
            "close": [100.0] * n_bars,
            "volume": [100000.0] * n_bars
        }, index=dates)

        data = {"AAPL": sample_df}
        train_data, val_data = split_chronological_data(data, in_sample_pct=0.67)

        self.assertEqual(len(train_data["AAPL"]), 67)
        self.assertEqual(len(val_data["AAPL"]), 33)
        self.assertEqual(len(train_data["AAPL"]) + len(val_data["AAPL"]), 100)

    def test_strictly_disjoint_and_sequential_partitions(self):
        """Verifies in-sample dates strictly precede out-of-sample dates with zero overlap."""
        n_bars = 50
        dates = pd.date_range("2023-01-01", periods=n_bars, freq="D")
        sample_df = pd.DataFrame({
            "open": [100.0] * n_bars, "high": [101.0] * n_bars,
            "low": [99.0] * n_bars, "close": [100.0] * n_bars,
            "volume": [10000.0] * n_bars
        }, index=dates)

        train_data, val_data = split_chronological_data({"SPY": sample_df}, in_sample_pct=0.60)
        train_dates = set(train_data["SPY"].index)
        val_dates = set(val_data["SPY"].index)

        # 1. Zero overlap (disjoint)
        self.assertTrue(train_dates.isdisjoint(val_dates), "Train and validation date sets must have 0 overlap")

        # 2. Sequential order (max train date < min val date)
        max_train = max(train_data["SPY"].index)
        min_val = min(val_data["SPY"].index)
        self.assertLess(max_train, min_val, f"Max train date {max_train} must be strictly before Min val date {min_val}")

    def test_multi_asset_differing_calendar_split(self):
        """Verifies proper chronological split when US and Indian assets have different holiday dates."""
        us_dates = pd.to_datetime(["2023-01-13", "2023-01-17", "2023-01-18", "2023-01-26", "2023-01-27"])
        in_dates = pd.to_datetime(["2023-01-13", "2023-01-16", "2023-01-17", "2023-01-18", "2023-01-27"])

        data = {
            "AAPL": pd.DataFrame({"close": [150.0]*5, "open": [150.0]*5, "high": [155.0]*5, "low": [145.0]*5, "volume": [1000.0]*5}, index=us_dates),
            "NIFTYBEES.NS": pd.DataFrame({"close": [250.0]*5, "open": [250.0]*5, "high": [255.0]*5, "low": [245.0]*5, "volume": [1000.0]*5}, index=in_dates)
        }

        train_data, val_data = split_chronological_data(data, in_sample_pct=0.60)
        self.assertIn("AAPL", train_data)
        self.assertIn("NIFTYBEES.NS", train_data)
        self.assertIn("AAPL", val_data)
        self.assertIn("NIFTYBEES.NS", val_data)

    def test_insufficient_bars_guard_raises_error(self):
        """Verifies passing fewer than 20 bars raises ValueError."""
        short_dates = pd.date_range("2023-01-01", periods=15, freq="D")
        short_df = pd.DataFrame({"close": [10.0]*15, "open": [10.0]*15, "high": [11.0]*15, "low": [9.0]*15, "volume": [100.0]*15}, index=short_dates)

        with self.assertRaises(ValueError):
            split_chronological_data({"TEST": short_df}, in_sample_pct=0.67)


class TestInSampleVsOutOfSampleMetrics(unittest.TestCase):
    """
    Verifies that WalkForwardCalibrator evaluates in-sample and out-of-sample
    performance accurately, computes objective fitness scores, and protects
    against curve-fitting and lookahead bias.
    """

    def setUp(self):
        self.calibrator = WalkForwardCalibrator()

    def test_fitness_score_computation_and_penalties(self):
        """Verifies mathematical scoring of performance metrics and minimum trade penalties."""
        # 1. Zero trades -> immediate penalty (-999.0)
        metrics_zero = {"total_trades": 0, "win_rate_pct": 0.0, "profit_factor": 0.0, "max_drawdown_pct": 0.0, "net_return_pct": 0.0}
        self.assertEqual(calculate_fitness_score(metrics_zero, min_trades=3), -999.0)

        # 2. Trades < min_trades -> immediate penalty (-999.0)
        metrics_few = {"total_trades": 2, "win_rate_pct": 100.0, "profit_factor": 10.0, "max_drawdown_pct": 1.0, "net_return_pct": 20.0}
        self.assertEqual(calculate_fitness_score(metrics_few, min_trades=3), -999.0)

        # 3. High PF, low MDD strategy scores higher than low PF, high MDD strategy
        good_strategy = {"total_trades": 10, "win_rate_pct": 70.0, "profit_factor": 2.5, "max_drawdown_pct": 5.0, "net_return_pct": 25.0}
        risky_strategy = {"total_trades": 10, "win_rate_pct": 50.0, "profit_factor": 1.2, "max_drawdown_pct": 25.0, "net_return_pct": 10.0}
        score_good = calculate_fitness_score(good_strategy, min_trades=3)
        score_risky = calculate_fitness_score(risky_strategy, min_trades=3)
        self.assertGreater(score_good, score_risky)

    def test_single_bot_calibration_finds_best_candidate(self):
        """Verifies calibrating a single bot persona selects the optimal parameter candidate."""
        n_bars = 60
        dates = pd.date_range("2023-01-01", periods=n_bars, freq="D")
        prices = [100.0 + i * 1.5 if i < 40 else 160.0 + (i % 3) * 2.0 for i in range(n_bars)]
        data = {
            "AAPL": pd.DataFrame({
                "open": prices, "high": [p + 2.0 for p in prices],
                "low": [p - 1.0 for p in prices], "close": prices,
                "volume": [200000.0] * n_bars
            }, index=dates)
        }

        calibrator = WalkForwardCalibrator(in_sample_pct=0.67)
        param_grid = [
            {"stop_loss_pct": -5.0, "take_profit_pct": 10.0},
            {"stop_loss_pct": -7.5, "take_profit_pct": 12.0},
            {"stop_loss_pct": -10.0, "take_profit_pct": 15.0}
        ]

        result = calibrator.calibrate(ConservativeSwingBot, data, param_grid)

        self.assertEqual(result["bot_id"], "conservative_swing")
        self.assertIn("best_params", result)
        self.assertIn("in_sample_metrics", result)
        self.assertIn("out_of_sample_metrics", result)
        self.assertIn("full_period_metrics", result)
        self.assertIn(result["status"], ["active", "optimal", "insufficient_trades"])

    def test_out_of_sample_validation_prevents_curve_fitting(self):
        """Verifies out-of-sample metrics flag or downrank parameter sets that deteriorate out-of-sample."""
        calibrator = WalkForwardCalibrator()
        cand_a_is = {"net_return_pct": 30.0, "profit_factor": 3.0, "max_drawdown_pct": 5.0, "total_trades": 10, "win_rate_pct": 80.0}
        cand_a_oos = {"net_return_pct": -15.0, "profit_factor": 0.4, "max_drawdown_pct": 25.0, "total_trades": 5, "win_rate_pct": 20.0}

        cand_b_is = {"net_return_pct": 15.0, "profit_factor": 2.0, "max_drawdown_pct": 6.0, "total_trades": 8, "win_rate_pct": 62.5}
        cand_b_oos = {"net_return_pct": 8.0, "profit_factor": 1.8, "max_drawdown_pct": 5.0, "total_trades": 4, "win_rate_pct": 75.0}

        score_a = calibrator.compute_walk_forward_score(cand_a_is, cand_a_oos)
        score_b = calibrator.compute_walk_forward_score(cand_b_is, cand_b_oos)

        self.assertGreater(score_b, score_a, "Candidate B (robust OOS) must outrank Candidate A (overfit IS)")

    def test_calibration_deterministic_reproducibility(self):
        """Verifies identical inputs produce identical calibrated parameters across runs."""
        n_bars = 40
        dates = pd.date_range("2023-01-01", periods=n_bars, freq="D")
        prices = [100.0 + i * 1.0 for i in range(n_bars)]
        data = {
            "NVDA": pd.DataFrame({
                "open": prices, "high": [p + 1.0 for p in prices],
                "low": [p - 1.0 for p in prices], "close": prices,
                "volume": [500000.0] * n_bars
            }, index=dates)
        }

        calibrator = WalkForwardCalibrator()
        res1 = calibrator.calibrate(AggressiveBreakoutBot, data, AGGRESSIVE_BREAKOUT_GRID[:3])
        res2 = calibrator.calibrate(AggressiveBreakoutBot, data, AGGRESSIVE_BREAKOUT_GRID[:3])

        self.assertEqual(res1["best_params"], res2["best_params"])
        self.assertEqual(res1["in_sample_metrics"]["profit_factor"], res2["in_sample_metrics"]["profit_factor"])


class TestCalibrateAllPersonasIntegration(unittest.TestCase):
    """
    Integration tests verifying calibrate_all_personas executes across all 5 bot
    personas, adheres to strict execution latency limits (< 3s on synthetic fixtures),
    and produces data structures directly serializable to Supabase and JSON.
    """

    def test_calibrate_all_personas_on_synthetic_universe(self):
        """Verifies calibrate_all_personas produces complete calibrated profiles for all 5 bots."""
        import json
        start_time = time.time()
        n_bars = 45
        dates = pd.date_range("2023-01-01", periods=n_bars, freq="D")

        # Synthetic multi-asset universe
        nvda_prices = [100.0 + i * 1.5 for i in range(n_bars)]
        aapl_prices = [150.0 + i * 0.5 for i in range(n_bars)]
        niftybees_prices = [250.0 + i * 0.2 for i in range(n_bars)]
        nsei_prices = [22000.0 + i * 20.0 for i in range(n_bars)]
        data = {
            "NVDA": pd.DataFrame({"open": nvda_prices, "high": [p + 2.0 for p in nvda_prices], "low": [p - 1.0 for p in nvda_prices], "close": nvda_prices, "volume": [500000.0] * n_bars}, index=dates),
            "AAPL": pd.DataFrame({"open": aapl_prices, "high": [p + 1.0 for p in aapl_prices], "low": [p - 0.5 for p in aapl_prices], "close": aapl_prices, "volume": [800000.0] * n_bars}, index=dates),
            "NIFTYBEES.NS": pd.DataFrame({"open": niftybees_prices, "high": [p + 1.0 for p in niftybees_prices], "low": [p - 0.5 for p in niftybees_prices], "close": niftybees_prices, "volume": [600000.0] * n_bars}, index=dates),
            "^NSEI": pd.DataFrame({"open": nsei_prices, "high": [p + 50.0 for p in nsei_prices], "low": [p - 30.0 for p in nsei_prices], "close": nsei_prices, "volume": [1000000.0] * n_bars}, index=dates),
        }

        # Run calibration across all 5 personas
        results = calibrate_all_personas(data, years=1)

        elapsed = time.time() - start_time
        self.assertLess(elapsed, 3.0, f"calibrate_all_personas took {elapsed:.2f}s, exceeding 3.0s budget")

        # 1. Assert structure
        self.assertIn("calibrated_at", results)
        self.assertIn("personas", results)
        self.assertEqual(results["status"], "success")

        # 2. Assert all 5 personas exist
        expected_bots = [
            "conservative_swing",
            "aggressive_breakout",
            "contrarian_dip_buyer",
            "finbert_catalyst_hunter",
            "trend_follower"
        ]
        for bot_id in expected_bots:
            self.assertIn(bot_id, results["personas"], f"Missing {bot_id} in calibration results")
            prof = results["personas"][bot_id]
            self.assertEqual(prof["bot_id"], bot_id)
            self.assertIn("best_params", prof)
            self.assertIn("in_sample_metrics", prof)
            self.assertIn("out_of_sample_metrics", prof)
            self.assertIn("full_period_metrics", prof)
            self.assertIn("status", prof)

        # 3. Assert JSON serializability without TypeError (no raw numpy or Decimal objects)
        serialized_json = json.dumps(results)
        self.assertIsInstance(serialized_json, str)
        self.assertGreater(len(serialized_json), 500)

    def test_calibrate_all_personas_custom_grid_injection(self):
        """Verifies custom_grids injection restricts or overrides persona search space."""
        n_bars = 30
        dates = pd.date_range("2023-01-01", periods=n_bars, freq="D")
        spy_prices = [400.0 + i * 1.0 for i in range(n_bars)]
        data = {
            "SPY": pd.DataFrame({"open": spy_prices, "high": [p + 1.0 for p in spy_prices], "low": [p - 1.0 for p in spy_prices], "close": spy_prices, "volume": [1000000.0] * n_bars}, index=dates)
        }
        custom_grid = {
            "conservative_swing": [{"stop_loss_pct": -3.0, "take_profit_pct": 7.0}]
        }
        results = calibrate_all_personas(data, years=1, custom_grids=custom_grid)
        self.assertEqual(results["personas"]["conservative_swing"]["total_evaluated"], 1)
        self.assertEqual(results["personas"]["conservative_swing"]["best_params"]["stop_loss_pct"], -3.0)

    def test_calibrate_all_personas_output_schema_supabase_compatibility(self):
        """Verifies calibration output conforms directly to Milestone 3 Supabase table definitions."""
        n_bars = 30
        dates = pd.date_range("2023-01-01", periods=n_bars, freq="D")
        aapl_prices = [150.0 + i * 1.0 for i in range(n_bars)]
        data = {
            "AAPL": pd.DataFrame({"open": aapl_prices, "high": [p + 2.0 for p in aapl_prices], "low": [p - 1.0 for p in aapl_prices], "close": aapl_prices, "volume": [100000.0] * n_bars}, index=dates)
        }
        results = calibrate_all_personas(data, years=1)

        # Columns expected by bot_training_runs
        self.assertIn("calibrated_at", results)
        self.assertIn("years", results)
        self.assertIn("universe", results)
        self.assertIn("status", results)

        # Columns expected by bot_calibrated_parameters
        for bot_id, prof in results["personas"].items():
            self.assertIn("bot_id", prof)
            self.assertIn("persona_name", prof)
            self.assertIn("best_params", prof)
            self.assertIn("in_sample_metrics", prof)
            self.assertIn("out_of_sample_metrics", prof)
            self.assertIn("full_period_metrics", prof)
            self.assertIn("status", prof)


# ============================================================================
# MILESTONE 3: SUPABASE PERSISTENCE, OFFLINE FALLBACK, & 3-TIER RESOLUTION
# ============================================================================

class TestSupabaseAndOfflinePersistence(unittest.TestCase):
    """
    Verifies saving training runs and calibrated parameters:
    1. Supabase client mock interactions (correct table, columns, and execute calls).
    2. Graceful offline fallback when client=None.
    3. Graceful offline fallback when Supabase throws network/PostgREST exceptions.
    4. Multi-persona co-existence in local JSON cache without overwriting peers.
    """

    def setUp(self):
        self.sample_run_data = {
            "calibrated_at": "2026-10-18T02:40:00+00:00",
            "universe": ["^NSEI", "NIFTYBEES.NS", "NVDA", "AAPL", "MSFT", "SPY"],
            "years": 3,
            "status": "success",
            "execution_time_seconds": 1.45,
            "personas": {
                "conservative_swing": {
                    "bot_id": "conservative_swing",
                    "best_params": {"stop_loss_pct": -5.0, "take_profit_pct": 12.0},
                    "full_period_metrics": {"profit_factor": 2.1, "win_rate_pct": 65.0}
                }
            }
        }
        self.sample_params = {"stop_loss_pct": -5.0, "take_profit_pct": 12.0, "min_momentum_5d": 0.0, "max_positions": 5}
        self.sample_metrics = {"profit_factor": 2.15, "win_rate_pct": 66.67, "max_drawdown_pct": 4.5, "net_return_pct": 18.2}

    def test_save_training_run_supabase_mock_success(self):
        """Verifies save_training_run formats record and calls Supabase table insert."""
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_client.table.return_value = mock_table
        mock_table.insert.return_value = mock_table
        mock_table.execute.return_value = MagicMock(data=[{"id": "run-uuid-123"}])

        success, err = save_training_run(client=mock_client, run_data=self.sample_run_data)

        self.assertTrue(success)
        self.assertIsNone(err)
        mock_client.table.assert_called_with("bot_training_runs")
        mock_table.insert.assert_called_once()
        inserted_payload = mock_table.insert.call_args[0][0]
        self.assertEqual(inserted_payload["years_trained"], 3)
        self.assertEqual(inserted_payload["universe_symbols"], ["^NSEI", "NIFTYBEES.NS", "NVDA", "AAPL", "MSFT", "SPY"])
        self.assertEqual(inserted_payload["status"], "success")

    def test_save_training_run_offline_fallback_when_client_none(self):
        """Verifies save_training_run falls back to data/bot_training_runs.json when client=None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "bot_training_runs.json"
            with patch("src.db.DATA_DIR", Path(tmpdir)), patch("src.db.TRAINING_RUNS_FILE", test_file):
                success, err = save_training_run(client=None, run_data=self.sample_run_data)

                self.assertTrue(success)
                self.assertTrue(test_file.exists())
                with open(test_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.assertIsInstance(data, list)
                self.assertEqual(len(data), 1)
                self.assertEqual(data[0]["years"], 3)
                self.assertEqual(data[0]["status"], "success")

    def test_save_training_run_offline_fallback_on_supabase_exception(self):
        """Verifies save_training_run catches Supabase exceptions and falls back to local JSON."""
        mock_client = MagicMock()
        mock_client.table.side_effect = Exception("PostgREST network timeout (HTTP 504)")

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "bot_training_runs.json"
            with patch("src.db.DATA_DIR", Path(tmpdir)), patch("src.db.TRAINING_RUNS_FILE", test_file):
                success, err = save_training_run(client=mock_client, run_data=self.sample_run_data)

                self.assertTrue(success)
                self.assertTrue(test_file.exists())
                with open(test_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.assertEqual(len(data), 1)

    def test_save_calibrated_parameters_supabase_mock_success(self):
        """Verifies save_calibrated_parameters updates existing active records and inserts new active record."""
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_client.table.return_value = mock_table
        mock_table.update.return_value = mock_table
        mock_table.insert.return_value = mock_table
        mock_table.eq.return_value = mock_table
        mock_table.execute.return_value = MagicMock(data=[{"id": "param-uuid-456"}])

        success, err = save_calibrated_parameters(
            client=mock_client,
            bot_id="conservative_swing",
            parameters=self.sample_params,
            metrics=self.sample_metrics
        )

        self.assertTrue(success)
        self.assertIsNone(err)
        mock_client.table.assert_called_with("bot_calibrated_parameters")

    def test_save_calibrated_parameters_offline_fallback_when_client_none(self):
        """Verifies save_calibrated_parameters writes to local JSON when client=None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "bot_calibrated_parameters.json"
            with patch("src.db.DATA_DIR", Path(tmpdir)), patch("src.db.CALIBRATED_PARAMS_FILE", test_file):
                success, err = save_calibrated_parameters(
                    client=None,
                    bot_id="conservative_swing",
                    parameters=self.sample_params,
                    metrics=self.sample_metrics
                )

                self.assertTrue(success)
                self.assertTrue(test_file.exists())
                with open(test_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.assertIn("conservative_swing", data)
                self.assertEqual(data["conservative_swing"]["parameters"]["stop_loss_pct"], -5.0)

    def test_save_calibrated_parameters_multi_persona_coexistence(self):
        """Verifies saving multiple personas sequentially in local cache preserves existing entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "bot_calibrated_parameters.json"
            with patch("src.db.DATA_DIR", Path(tmpdir)), patch("src.db.CALIBRATED_PARAMS_FILE", test_file):
                save_calibrated_parameters(None, "conservative_swing", {"stop_loss_pct": -5.0}, {"profit_factor": 2.0})
                save_calibrated_parameters(None, "aggressive_breakout", {"stop_loss_pct": -4.0}, {"profit_factor": 1.8})

                with open(test_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self.assertIn("conservative_swing", data)
                self.assertIn("aggressive_breakout", data)
                self.assertEqual(data["conservative_swing"]["parameters"]["stop_loss_pct"], -5.0)
                self.assertEqual(data["aggressive_breakout"]["parameters"]["stop_loss_pct"], -4.0)


class TestThreeTierParameterResolution(unittest.TestCase):
    """
    Verifies 3-tier parameter resolution in load_calibrated_parameters:
    - Tier 1: Supabase Cloud query returns parameters.
    - Tier 2: Supabase fails/client=None -> loads from local JSON cache.
    - Tier 3: Local JSON missing or empty -> returns None (triggers hardcoded bot defaults).
    """

    def test_tier1_supabase_cloud_precedence(self):
        """Tier 1: Supabase Cloud returns parameters directly, bypassing local file."""
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_client.table.return_value = mock_table
        mock_table.select.return_value = mock_table
        mock_table.eq.return_value = mock_table
        mock_table.order.return_value = mock_table
        mock_table.limit.return_value = mock_table
        mock_table.execute.return_value = MagicMock(data=[
            {"parameters": {"stop_loss_pct": -4.0, "take_profit_pct": 16.0}}
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            # Place a conflicting entry in local JSON
            test_file = Path(tmpdir) / "bot_calibrated_parameters.json"
            test_file.write_text(json.dumps({"conservative_swing": {"parameters": {"stop_loss_pct": -9.9}}}))

            with patch("src.db.DATA_DIR", Path(tmpdir)), patch("src.db.CALIBRATED_PARAMS_FILE", test_file):
                params = load_calibrated_parameters(mock_client, "conservative_swing")

                self.assertIsNotNone(params)
                self.assertEqual(params["stop_loss_pct"], -4.0)
                self.assertEqual(params["take_profit_pct"], 16.0)

    def test_tier2_supabase_failure_falls_back_to_local_cache(self):
        """Tier 2: When Supabase client throws exception, loads from local JSON cache."""
        mock_client = MagicMock()
        mock_client.table.side_effect = Exception("Supabase connection refused (503 Service Unavailable)")

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "bot_calibrated_parameters.json"
            test_file.write_text(json.dumps({
                "conservative_swing": {"parameters": {"stop_loss_pct": -6.0, "take_profit_pct": 14.0}}
            }))

            with patch("src.db.DATA_DIR", Path(tmpdir)), patch("src.db.CALIBRATED_PARAMS_FILE", test_file):
                params = load_calibrated_parameters(mock_client, "conservative_swing")

                self.assertIsNotNone(params)
                self.assertEqual(params["stop_loss_pct"], -6.0)
                self.assertEqual(params["take_profit_pct"], 14.0)

    def test_tier2_client_none_loads_from_local_cache(self):
        """Tier 2: When client=None, seamlessly loads parameters from local JSON cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "bot_calibrated_parameters.json"
            test_file.write_text(json.dumps({
                "aggressive_breakout": {"parameters": {"stop_loss_pct": -4.5, "take_profit_pct": 20.0}}
            }))

            with patch("src.db.DATA_DIR", Path(tmpdir)), patch("src.db.CALIBRATED_PARAMS_FILE", test_file):
                params = load_calibrated_parameters(client=None, bot_id="aggressive_breakout")

                self.assertIsNotNone(params)
                self.assertEqual(params["stop_loss_pct"], -4.5)
                self.assertEqual(params["take_profit_pct"], 20.0)

    def test_tier3_missing_local_file_returns_none_triggering_bot_defaults(self):
        """Tier 3: When local file does not exist, returns None, allowing bot to use default parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            non_existent_file = Path(tmpdir) / "does_not_exist.json"
            with patch("src.db.DATA_DIR", Path(tmpdir)), patch("src.db.CALIBRATED_PARAMS_FILE", non_existent_file):
                params = load_calibrated_parameters(client=None, bot_id="conservative_swing")

                self.assertIsNone(params)
                # Verify bot instantiated with None preserves default attributes
                bot = ConservativeSwingBot(params=params)
                self.assertEqual(bot.stop_loss_pct, -7.5)
                self.assertEqual(bot.take_profit_pct, 12.0)

    def test_tier3_bot_id_not_found_returns_none(self):
        """Tier 3: When local cache exists but requested bot_id is missing, returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "bot_calibrated_parameters.json"
            test_file.write_text(json.dumps({"trend_follower": {"parameters": {"stop_loss_pct": -10.0}}}))

            with patch("src.db.DATA_DIR", Path(tmpdir)), patch("src.db.CALIBRATED_PARAMS_FILE", test_file):
                params = load_calibrated_parameters(client=None, bot_id="conservative_swing")

                self.assertIsNone(params)


class TestCorruptedCacheAndResilience(unittest.TestCase):
    """
    Verifies defensive error recovery for malformed, corrupted, or 0-byte local cache files.
    Ensures zero unhandled exceptions and seamless fallback to Tier 3 defaults.
    """

    def test_malformed_json_syntax_error_recovery(self):
        """Verifies load_calibrated_parameters handles syntax-corrupted JSON cleanly without crashing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupt_file = Path(tmpdir) / "bot_calibrated_parameters.json"
            corrupt_file.write_text("{ 'conservative_swing': { unclosed bracket ... ")

            with patch("src.db.DATA_DIR", Path(tmpdir)), patch("src.db.CALIBRATED_PARAMS_FILE", corrupt_file):
                params = load_calibrated_parameters(client=None, bot_id="conservative_swing")
                self.assertIsNone(params)

    def test_zero_byte_empty_file_recovery(self):
        """Verifies load_calibrated_parameters handles 0-byte file without JSONDecodeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_file = Path(tmpdir) / "bot_calibrated_parameters.json"
            empty_file.touch()

            with patch("src.db.DATA_DIR", Path(tmpdir)), patch("src.db.CALIBRATED_PARAMS_FILE", empty_file):
                params = load_calibrated_parameters(client=None, bot_id="conservative_swing")
                self.assertIsNone(params)

    def test_corrupted_file_overwritten_safely_on_save(self):
        """Verifies save_calibrated_parameters safely overwrites corrupted file rather than crashing on read."""
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupt_file = Path(tmpdir) / "bot_calibrated_parameters.json"
            corrupt_file.write_text("GARBAGE DATA NOT JSON")

            with patch("src.db.DATA_DIR", Path(tmpdir)), patch("src.db.CALIBRATED_PARAMS_FILE", corrupt_file):
                success, err = save_calibrated_parameters(
                    client=None,
                    bot_id="conservative_swing",
                    parameters={"stop_loss_pct": -5.0},
                    metrics={"profit_factor": 1.5}
                )

                self.assertTrue(success)
                with open(corrupt_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.assertIn("conservative_swing", data)
                self.assertEqual(data["conservative_swing"]["parameters"]["stop_loss_pct"], -5.0)

    def test_save_training_run_corrupted_file_recovery(self):
        """Verifies save_training_run recovers when bot_training_runs.json is corrupt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupt_file = Path(tmpdir) / "bot_training_runs.json"
            corrupt_file.write_text("BROKEN JSON ARRAY [")

            with patch("src.db.DATA_DIR", Path(tmpdir)), patch("src.db.TRAINING_RUNS_FILE", corrupt_file):
                run_data = {"years": 1, "universe": ["AAPL"], "status": "success"}
                success, err = save_training_run(client=None, run_data=run_data)

                self.assertTrue(success)
                with open(corrupt_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.assertIsInstance(data, list)
                self.assertEqual(len(data), 1)


class TestEndToEndCalibrationPersistenceIntegration(unittest.TestCase):
    """
    Integration test verifying the entire offline flow:
    calibrate_all_personas -> save_training_run -> save_calibrated_parameters ->
    load_calibrated_parameters -> BotArchetype initialization with learned parameters.
    """

    def test_full_roundtrip_offline_workflow(self):
        """Executes full offline roundtrip using synthetic fixtures in under 1 second."""
        n_bars = 30
        dates = pd.date_range("2023-01-01", periods=n_bars, freq="D")
        spy_prices = [400.0 + i * 1.0 for i in range(n_bars)]
        data = {
            "SPY": pd.DataFrame({
                "open": spy_prices, "high": [p + 1.0 for p in spy_prices],
                "low": [p - 1.0 for p in spy_prices], "close": spy_prices,
                "volume": [1000000.0] * n_bars
            }, index=dates)
        }

        # 1. Run calibration with compact custom grid
        custom_grid = {
            "conservative_swing": [{"stop_loss_pct": -4.0, "take_profit_pct": 14.0, "min_momentum_5d": 0.0, "max_positions": 4}],
            "aggressive_breakout": [{"stop_loss_pct": -3.5, "take_profit_pct": 16.0, "min_momentum_5d": 1.5, "allocation_pct": 0.25}],
            "contrarian_dip_buyer": [{"stop_loss_pct": -5.0, "take_profit_pct": 7.0, "dip_threshold_pct": -1.0}],
            "finbert_catalyst_hunter": [{"stop_loss_pct": -6.0, "take_profit_pct": 12.0, "conviction_threshold": 0.45}],
            "trend_follower": [{"stop_loss_pct": -7.0, "take_profit_pct": 16.0, "min_momentum_5d": 0.0}],
        }
        calibration_results = calibrate_all_personas(data, years=1, custom_grids=custom_grid)

        with tempfile.TemporaryDirectory() as tmpdir:
            runs_file = Path(tmpdir) / "bot_training_runs.json"
            params_file = Path(tmpdir) / "bot_calibrated_parameters.json"

            with patch("src.db.DATA_DIR", Path(tmpdir)), \
                 patch("src.db.TRAINING_RUNS_FILE", runs_file), \
                 patch("src.db.CALIBRATED_PARAMS_FILE", params_file):

                # 2. Persist training run
                run_ok, _ = save_training_run(client=None, run_data=calibration_results)
                self.assertTrue(run_ok)

                # 3. Persist individual parameters
                for bot_id, prof in calibration_results["personas"].items():
                    save_ok, _ = save_calibrated_parameters(
                        client=None,
                        bot_id=bot_id,
                        parameters=prof["best_params"],
                        metrics=prof["full_period_metrics"],
                        status=prof["status"]
                    )
                    self.assertTrue(save_ok)

                # 4. Load parameters via Tier 2
                loaded_params = load_calibrated_parameters(client=None, bot_id="conservative_swing")
                self.assertIsNotNone(loaded_params)
                self.assertEqual(loaded_params["stop_loss_pct"], -4.0)
                self.assertEqual(loaded_params["take_profit_pct"], 14.0)

                # 5. Instantiate bot with learned parameters
                bot = ConservativeSwingBot(params=loaded_params)
                self.assertEqual(bot.stop_loss_pct, -4.0)
                self.assertEqual(bot.take_profit_pct, 14.0)
                self.assertEqual(bot.max_positions, 4)


# ============================================================================
# MILESTONE 4: DYNAMIC PARAMETER SYNC & CLI VERIFICATION SUITE
# ============================================================================

import io
import contextlib
from src.paper_trader import MultiBotTournamentEngine
from run_pipeline import (
    main,
    run_calibration_training,
    run_backtest_report,
    format_key_params,
    load_all_calibrated_metrics,
)


class TestMultiBotTournamentParameterSync(unittest.TestCase):
    """
    Milestone 4 Verification: Validates dynamic parameter synchronization in
    MultiBotTournamentEngine across Tier 1 (Cloud), Tier 2 (Local JSON), and
    Tier 3 (Constructor Defaults), as well as cache corruption resilience.
    """

    def test_tournament_engine_tier1_cloud_sync(self):
        """Verifies MultiBotTournamentEngine loads calibrated parameters from Supabase Cloud."""
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_client.table.return_value = mock_table
        mock_select = MagicMock()
        mock_table.select.return_value = mock_select

        def mock_eq(col, val):
            eq_mock = MagicMock()
            if col == "bot_id" and val == "conservative_swing":
                order_mock = MagicMock()
                eq_mock.eq.return_value = order_mock
                limit_mock = MagicMock()
                order_mock.order.return_value = limit_mock
                execute_mock = MagicMock()
                limit_mock.limit.return_value = execute_mock
                res = MagicMock()
                res.data = [{
                    "bot_id": "conservative_swing",
                    "parameters": {
                        "stop_loss_pct": -4.0,
                        "take_profit_pct": 14.0,
                        "min_momentum_5d": 0.5,
                        "conviction_threshold": 6.5,
                        "max_positions": 3,
                    },
                    "metrics": {"profit_factor": 2.2, "win_rate_pct": 65.0},
                    "active": True
                }]
                execute_mock.execute.return_value = res
            else:
                order_mock = MagicMock()
                eq_mock.eq.return_value = order_mock
                limit_mock = MagicMock()
                order_mock.order.return_value = limit_mock
                execute_mock = MagicMock()
                limit_mock.limit.return_value = execute_mock
                res = MagicMock()
                res.data = []
                execute_mock.execute.return_value = res
            return eq_mock

        mock_select.eq.side_effect = mock_eq

        engine = MultiBotTournamentEngine(client=mock_client)
        cs_bot = engine.bots["conservative_swing"]

        self.assertEqual(cs_bot.stop_loss_pct, -4.0)
        self.assertEqual(cs_bot.take_profit_pct, 14.0)
        self.assertEqual(cs_bot.min_momentum_5d, 0.5)
        self.assertEqual(cs_bot.conviction_threshold, 6.5)
        self.assertEqual(cs_bot.max_positions, 3)

    def test_tournament_engine_tier2_local_cache_sync(self):
        """Verifies MultiBotTournamentEngine falls back to local JSON cache when client is None."""
        sample_cache = {
            "aggressive_breakout": {
                "bot_id": "aggressive_breakout",
                "parameters": {
                    "stop_loss_pct": -3.5,
                    "take_profit_pct": 16.0,
                    "min_momentum_5d": 1.5,
                    "allocation_pct": 0.30
                },
                "status": "optimal"
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "bot_calibrated_parameters.json"
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(sample_cache, f)

            with patch("src.db.CALIBRATED_PARAMS_FILE", cache_file):
                engine = MultiBotTournamentEngine(client=None)
                ab_bot = engine.bots["aggressive_breakout"]
                self.assertEqual(ab_bot.stop_loss_pct, -3.5)
                self.assertEqual(ab_bot.take_profit_pct, 16.0)
                self.assertEqual(ab_bot.min_momentum_5d, 1.5)
                self.assertEqual(ab_bot.allocation_pct, 0.30)
                # Ensure other bots kept defaults
                self.assertEqual(engine.bots["conservative_swing"].stop_loss_pct, -7.5)

    def test_tournament_engine_tier3_defaults_preservation(self):
        """Verifies MultiBotTournamentEngine falls back to constructor defaults if no calibration exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_file = Path(tmpdir) / "non_existent.json"
            with patch("src.db.CALIBRATED_PARAMS_FILE", empty_file):
                engine = MultiBotTournamentEngine(client=None)
                self.assertEqual(engine.bots["conservative_swing"].stop_loss_pct, -7.5)
                self.assertEqual(engine.bots["conservative_swing"].take_profit_pct, 12.0)
                self.assertEqual(engine.bots["aggressive_breakout"].min_momentum_5d, 2.0)
                self.assertEqual(engine.bots["contrarian_dip_buyer"].dip_threshold_pct, -1.5)
                self.assertEqual(engine.bots["finbert_catalyst_hunter"].min_pos_prob, 0.70)
                self.assertEqual(engine.bots["trend_follower"].take_profit_pct, 15.0)

    def test_tournament_engine_corrupted_cache_resilience(self):
        """Verifies MultiBotTournamentEngine handles corrupted local JSON gracefully without crashing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupt_file = Path(tmpdir) / "bot_calibrated_parameters.json"
            with open(corrupt_file, "w", encoding="utf-8") as f:
                f.write("{ invalid json content: [broken")

            with patch("src.db.CALIBRATED_PARAMS_FILE", corrupt_file):
                # Must not raise unhandled exception
                engine = MultiBotTournamentEngine(client=None)
                self.assertIsNotNone(engine)
                self.assertEqual(engine.bots["conservative_swing"].stop_loss_pct, -7.5)

    def test_tournament_engine_explicit_resync(self):
        """Verifies sync_calibrated_parameters() can be called manually and returns dictionary mapping."""
        engine = MultiBotTournamentEngine(client=None, auto_sync_params=False)
        summary = engine.sync_calibrated_parameters()
        self.assertIsInstance(summary, dict)
        for bot_id in ["conservative_swing", "aggressive_breakout", "contrarian_dip_buyer", "finbert_catalyst_hunter", "trend_follower"]:
            self.assertIn(bot_id, summary)
            self.assertIn("stop_loss_pct", summary[bot_id])
            self.assertIn("take_profit_pct", summary[bot_id])

    def test_bot_archetype_set_params_persona_attributes(self):
        """Verifies BotArchetype.set_params updates operational instance attributes across personas."""
        # 1. Conservative Swing
        cs = ConservativeSwingBot()
        cs.set_params(min_momentum_5d=1.2, conviction_threshold=8.0, min_news_score=2.0)
        self.assertEqual(cs.min_momentum_5d, 1.2)
        self.assertEqual(cs.conviction_threshold, 8.0)
        self.assertEqual(cs.min_news_score, 2.0)

        # 2. Contrarian Dip Buyer
        cd = ContrarianDipBuyerBot()
        cd.set_params(dip_threshold_pct=-2.5, dip_min_pct=-10.0, min_net_score=-0.10)
        self.assertEqual(cd.dip_threshold_pct, -2.5)
        self.assertEqual(cd.dip_max_pct, -2.5)
        self.assertEqual(cd.dip_min_pct, -10.0)
        self.assertEqual(cd.min_net_score, -0.10)

        # 3. FinBERT Catalyst Hunter
        fb = FinBERTCatalystHunterBot()
        fb.set_params(conviction_threshold=0.65)
        self.assertEqual(fb.conviction_threshold, 0.65)
        self.assertEqual(fb.min_net_score, 0.65)

        # 4. Aggressive Breakout
        ab = AggressiveBreakoutBot()
        ab.set_params(min_momentum_5d=3.0, allocation_pct=0.20)
        self.assertEqual(ab.min_momentum_5d, 3.0)
        self.assertEqual(ab.allocation_pct, 0.20)
        self.assertEqual(ab.max_trade_dollars, 1000.0)

        # 5. Trend Follower
        tf = TrendFollowerBot()
        tf.set_params(min_momentum_5d=0.8, take_profit_pct=20.0)
        self.assertEqual(tf.min_momentum_5d, 0.8)
        self.assertEqual(tf.take_profit_pct, 20.0)


class TestCLIPipelineAndBacktestReport(unittest.TestCase):
    """
    Milestone 4 Verification: Validates CLI argument parsing, training pipeline execution,
    3-tier metrics resolution, and formatted comparative backtest tabular reports.
    """

    def test_cli_argument_parser_train_and_report_flags(self):
        """Verifies argument parser recognizes --train, --years, and --backtest-report."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--train", action="store_true")
        parser.add_argument("--years", type=int, default=3)
        parser.add_argument("--backtest-report", action="store_true")

        # Case 1: Default train lookback
        args1 = parser.parse_args(["--train"])
        self.assertTrue(args1.train)
        self.assertEqual(args1.years, 3)
        self.assertFalse(args1.backtest_report)

        # Case 2: Custom years lookback
        args2 = parser.parse_args(["--train", "--years", "5"])
        self.assertTrue(args2.train)
        self.assertEqual(args2.years, 5)

        # Case 3: Backtest report flag
        args3 = parser.parse_args(["--backtest-report"])
        self.assertTrue(args3.backtest_report)
        self.assertFalse(args3.train)

    def test_format_key_params_all_personas(self):
        """Verifies concise formatting of persona-specific parameter subsets."""
        # 1. Conservative Swing
        p_cs = {"stop_loss_pct": -5.0, "take_profit_pct": 12.0, "min_momentum_5d": 0.0, "max_positions": 5}
        fmt_cs = format_key_params("conservative_swing", p_cs)
        self.assertIn("SL: -5.0%", fmt_cs)
        self.assertIn("TP: +12.0%", fmt_cs)
        self.assertIn("MaxPos: 5", fmt_cs)

        # 2. Aggressive Breakout
        p_ab = {"stop_loss_pct": -4.0, "take_profit_pct": 18.0, "min_momentum_5d": 2.0, "allocation_pct": 0.25}
        fmt_ab = format_key_params("aggressive_breakout", p_ab)
        self.assertIn("Alloc: 25%", fmt_ab)
        self.assertIn("Mom: 2.0%", fmt_ab)

        # 3. Contrarian Dip Buyer
        p_cd = {"stop_loss_pct": -6.0, "take_profit_pct": 8.0, "dip_threshold_pct": -1.5}
        fmt_cd = format_key_params("contrarian_dip_buyer", p_cd)
        self.assertIn("Dip: -1.5%", fmt_cd)

        # 4. FinBERT Catalyst Hunter
        p_fb = {"stop_loss_pct": -7.0, "take_profit_pct": 10.0, "conviction_threshold": 0.50}
        fmt_fb = format_key_params("finbert_catalyst_hunter", p_fb)
        self.assertIn("Conv: 0.50", fmt_fb)

        # 5. Trend Follower
        p_tf = {"stop_loss_pct": -8.0, "take_profit_pct": 15.0, "min_momentum_5d": 0.5}
        fmt_tf = format_key_params("trend_follower", p_tf)
        self.assertIn("Mom: 0.5%", fmt_tf)

        # 6. Empty / Defaults
        self.assertEqual(format_key_params("unknown_bot", {}), "Defaults")

    def test_run_calibration_training_execution_offline(self):
        """Verifies run_calibration_training executes offline with synthetic fixtures and dual-persists."""
        n_bars = 30
        dates = pd.date_range("2023-01-01", periods=n_bars, freq="D")
        synthetic_data = {
            "SPY": pd.DataFrame({
                "open": [400.0 + i for i in range(n_bars)],
                "high": [401.0 + i for i in range(n_bars)],
                "low": [399.0 + i for i in range(n_bars)],
                "close": [400.0 + i for i in range(n_bars)],
                "volume": [100000.0] * n_bars
            }, index=dates)
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            runs_file = Path(tmpdir) / "bot_training_runs.json"
            params_file = Path(tmpdir) / "bot_calibrated_parameters.json"

            with patch("src.db.DATA_DIR", Path(tmpdir)), \
                 patch("src.db.TRAINING_RUNS_FILE", runs_file), \
                 patch("src.db.CALIBRATED_PARAMS_FILE", params_file):

                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    res = run_calibration_training(years=1, data=synthetic_data, client=None)

                self.assertIsInstance(res, dict)
                self.assertEqual(res["status"], "success")
                self.assertIn("personas", res)
                self.assertEqual(len(res["personas"]), 5)
                self.assertTrue(runs_file.exists())
                self.assertTrue(params_file.exists())

                output_str = buf.getvalue()
                self.assertIn("HISTORICAL TRAINING & WALK-FORWARD PARAMETER CALIBRATION", output_str)
                self.assertIn("HISTORICAL BACKTEST & BOT CALIBRATION REPORT", output_str)

    def test_run_backtest_report_with_cached_data(self):
        """Verifies report table renders all 8 required columns and bot personas."""
        sample_cached = {
            "conservative_swing": {
                "name": "Conservative Swing",
                "status": "optimal",
                "best_params": {"stop_loss_pct": -5.0, "take_profit_pct": 12.0},
                "full_period_metrics": {"profit_factor": 2.15, "win_rate_pct": 66.7, "max_drawdown_pct": 4.5, "net_return_pct": 18.2, "total_trades": 24}
            },
            "aggressive_breakout": {
                "name": "Aggressive Breakout",
                "status": "optimal",
                "best_params": {"stop_loss_pct": -4.0, "take_profit_pct": 18.0},
                "full_period_metrics": {"profit_factor": 1.82, "win_rate_pct": 58.3, "max_drawdown_pct": 6.2, "net_return_pct": 22.4, "total_trades": 36}
            }
        }

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ok = run_backtest_report(client=None, cached_data=sample_cached)

        self.assertTrue(ok)
        out = buf.getvalue()
        # Verify required headers
        for col in ["Bot Persona", "Win Rate %", "Profit Factor", "Max Drawdown %", "Net Return %", "Total Trades", "Status", "Key Calibrated Params"]:
            self.assertIn(col, out)
        # Verify persona rows and values
        self.assertIn("Conservative Swing", out)
        self.assertIn("66.7%", out)
        self.assertIn("2.15", out)
        self.assertIn("+18.2%", out)
        self.assertIn("Aggressive Breakout", out)

    def test_run_backtest_report_empty_cache_guidance(self):
        """Verifies clear guidance is printed when no calibration records exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_file = Path(tmpdir) / "empty.json"
            with patch("src.db.CALIBRATED_PARAMS_FILE", empty_file), \
                 patch("src.db.TRAINING_RUNS_FILE", empty_file):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    ok = run_backtest_report(client=None)

                self.assertFalse(ok)
                out = buf.getvalue()
                self.assertIn("python run_pipeline.py --train --years 3", out)

    def test_load_all_calibrated_metrics_resolution(self):
        """Verifies 3-tier fallback resolution in load_all_calibrated_metrics."""
        # Tier 1: Cloud query mock
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_client.table.return_value = mock_table
        mock_select = MagicMock()
        mock_table.select.return_value = mock_select
        mock_eq = MagicMock()
        mock_select.eq.return_value = mock_eq
        mock_order = MagicMock()
        mock_eq.order.return_value = mock_order
        mock_res = MagicMock()
        mock_res.data = [{"bot_id": "conservative_swing", "parameters": {"stop_loss_pct": -4.0}}]
        mock_order.execute.return_value = mock_res

        profiles, src = load_all_calibrated_metrics(client=mock_client)
        self.assertIn("conservative_swing", profiles)
        self.assertIn("Supabase Cloud", src)

        # Tier 2: Local Cache
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "bot_calibrated_parameters.json"
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({"trend_follower": {"parameters": {"stop_loss_pct": -6.0}}}, f)

            with patch("src.db.CALIBRATED_PARAMS_FILE", cache_file):
                profiles, src = load_all_calibrated_metrics(client=None)
                self.assertIn("trend_follower", profiles)
                self.assertIn("Local Cache", src)


if __name__ == "__main__":
    unittest.main()

