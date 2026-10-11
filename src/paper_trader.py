"""
Multi-Bot Tournament Architecture for Financial Market Sentiment & Paper Trading.

Features:
1. Multi-Bot Personas:
   - Conservative Swing: High-conviction AI sentiment + confirmed technical trend.
   - Aggressive Breakout: High-momentum breakout chaser with tight risk stops.
   - Dip Buyer / Contrarian: Mean-reversion seeker buying oversold pullbacks on solid stocks.
   - FinBERT Catalyst Hunter: News-first trader reacting aggressively to AI sentiment surges.
   - Trend Follower: Systematic trend-following bot riding moving averages.

2. Isolated $5,000 Paper Funds:
   Each bot operates its own cash balance, positions, and trade history.

3. Concurrent Shared Market Evaluation:
   All bots evaluate against the exact same snapshot of live news, real-time market prices,
   technical indicators, and FinBERT scores.

4. Backward Compatibility:
   Gracefully supports both legacy single-portfolio Supabase tables and modern multi-bot
   schemas with `bot_id`, falling back to local tournament persistence if needed.

5. Live Multi-Bot Tournament Leaderboard:
   Renders a ranked leaderboard with medals, equity, net P&L, win rate, open positions,
   and best trades for CLI and GitHub Actions logs.
"""

from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timezone
import sys
import os
import json
import re
from pathlib import Path
from tabulate import tabulate
import yfinance as yf

# Ensure UTF-8 / safe character encoding for Windows consoles
if sys.platform.startswith("win"):
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def safe_print(*args, **kwargs):
    """Safely prints strings without failing on limited console encodings (e.g. cp1252)."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        try:
            encoding = getattr(sys.stdout, "encoding", None) or "ascii"
            clean_args = [
                str(arg).encode(encoding, errors="replace").decode(encoding)
                for arg in args
            ]
            print(*clean_args, **kwargs)
        except Exception:
            clean_args = [
                str(arg).encode("ascii", errors="replace").decode("ascii")
                for arg in args
            ]
            print(*clean_args, **kwargs)

from src.db import get_supabase_client
from src.finbert_analyzer import FinBERTAnalyzer

# Sentinel for default client initialization
_DEFAULT_CLIENT = object()

# Tickers monitored across all tournament bots (Mega-Caps + High-Potential Small/Mid-Caps)
DEFAULT_WATCHLIST = [
    "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AMD", "SPY", "QQQ",
    "PLTR", "SOFI", "IONQ", "ASTS", "RKLB", "SMCI", "COIN", "HOOD"
]
WATCHLIST = DEFAULT_WATCHLIST

# Sentiment keyword fallback signals
BULLISH_SIGNALS = [
    "surge", "record high", "beat", "rally", "growth", "jump",
    "bull", "upgrade", "comeback", "outperform", "rebound", "breakout"
]
BEARISH_SIGNALS = [
    "sink", "slump", "plunge", "rout", "miss", "crash",
    "bear", "downgrade", "warning", "investigation", "lawsuit", "layoff"
]

# Verified catalyst keywords indicating actionable news events
CATALYST_KEYWORDS = [
    "earnings", "beat", "record", "contract", "surge", "breakout",
    "deal", "fda", "approval", "patent", "growth", "revenue",
    "guidance", "upgrade", "partnership"
]

# Safety Border / Guardrail Filter Constants
MIN_SHARE_PRICE = 5.00
MIN_DAILY_VOLUME = 200_000.0
MIN_MARKET_CAP = 250_000_000.0  # $250M

# Local fallback storage directory
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOCAL_STATE_FILE = DATA_DIR / "bot_tournament_state.json"


# ============================================================================
# SAFETY BORDER / GUARDRAIL FILTER ("BORDERLIST")
# ============================================================================

class BorderlistGuardrail:
    """
    Safety Border / Guardrail Filter ("Borderlist").
    Protects bots from penny stock manipulation, reverse splits, and illiquid traps.

    Rules:
    1. Minimum share price floor: do NOT buy stocks trading below $5.00.
    2. Minimum liquidity / market cap: require legitimate trading volume
       (minimum 200,000 daily volume OR market cap >= $250M) so bots don't trade illiquid micro-caps.
    3. Automatically logs a skipped notice if a hot-trending ticker is disqualified.
    """
    def __init__(
        self,
        min_price: float = MIN_SHARE_PRICE,
        min_volume: float = MIN_DAILY_VOLUME,
        min_market_cap: float = MIN_MARKET_CAP
    ):
        self.min_price = min_price
        self.min_volume = min_volume
        self.min_market_cap = min_market_cap
        self.disqualified: Dict[str, Dict[str, Any]] = {}

    def evaluate(
        self,
        symbol: str,
        price: Optional[float],
        volume: Optional[float] = None,
        market_cap: Optional[float] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluates whether a ticker passes safety guardrails.
        Returns (is_qualified, disqualify_reason).
        """
        if price is None or price <= 0:
            return False, f"Invalid or unavailable price ({price})"

        # Sanitize NaN or negative values
        if volume is not None and (volume != volume or volume < 0):
            volume = None
        if market_cap is not None and (market_cap != market_cap or market_cap < 0):
            market_cap = None

        # 1. Share price floor ($5.00)
        if price < self.min_price:
            return False, f"Share price ${price:.2f} is below minimum floor of ${self.min_price:.2f} (penny stock risk)"

        # 2. Minimum liquidity / market cap requirement:
        # Require legitimate trading volume (>= 200,000) OR market cap >= $250M
        has_volume = volume is not None and volume >= self.min_volume
        has_market_cap = market_cap is not None and market_cap >= self.min_market_cap

        # If neither volume nor market cap data is provided or verifiable:
        if volume is None and market_cap is None:
            # Trusted mega-caps/index ETFs known to possess billions in liquidity
            if symbol.upper() in {"NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AMD", "SPY", "QQQ"}:
                return True, None
            return False, "Unverified liquidity and market cap (both volume and market cap are missing/unavailable)"

        # If liquidity data is provided, require volume >= 200,000 OR market cap >= $250M
        if not (has_volume or has_market_cap):
            vol_str = f"{int(volume):,}" if volume is not None else "Unknown"
            cap_str = f"${market_cap/1e6:.1f}M" if market_cap is not None else "Unknown"
            return False, (
                f"Illiquid micro-cap risk (Volume: {vol_str} < {int(self.min_volume):,} "
                f"AND Market Cap: {cap_str} < ${self.min_market_cap/1e6:.0f}M)"
            )

        return True, None

    def disqualify(
        self,
        symbol: str,
        price: Optional[float],
        volume: Optional[float],
        market_cap: Optional[float],
        reason: str,
        is_hot: bool = False
    ):
        """Records a disqualified ticker in borderlist and logs skipped notice."""
        now_iso = datetime.now(timezone.utc).isoformat()
        self.disqualified[symbol] = {
            "symbol": symbol,
            "price": price,
            "volume": volume,
            "market_cap": market_cap,
            "reason": reason,
            "is_hot": is_hot,
            "disqualified_at": now_iso
        }
        hot_label = " (🔥 Hot Trending)" if is_hot else ""
        safe_print(f"⚠️ [BORDERLIST GUARDRAIL] Skipped {symbol}{hot_label}: Disqualified by safety filter. Reason: {reason}")


# ============================================================================
# PROBABILITY-WEIGHTED UPSIDE & EXPECTED RETURN SCORING
# ============================================================================

def calculate_probability_weighted_upside(
    symbol: str,
    price: float,
    tech: Dict[str, Any],
    bert: Dict[str, Any],
    articles: List[Dict[str, Any]],
    market_cap: Optional[float] = None,
    volume: Optional[float] = None
) -> Dict[str, Any]:
    """
    Ranks candidates by probability-weighted upside and expected return.
    Combines:
    1. FinBERT net positive sentiment (institutional NLP probability)
    2. Upward technical breakout momentum (SMA20 position and 5-day velocity)
    3. Catalyst clarity (article count, sentiment polarity, and verified news catalysts)
    Allows small/mid-cap high-potential stocks to be picked over sluggish mega-caps when real catalysts exist.
    """
    # 1. FinBERT Sentiment Probability Component [0.05, 0.98]
    pos = float(bert.get("positive", 0.0))
    neg = float(bert.get("negative", 0.0))
    net = float(bert.get("net_score", 0.0))
    news_score = float(bert.get("news_score", 0.0))

    if pos > 0.0 or net != 0.0:
        p_sentiment = 0.50 + (net * 0.35) + (pos * 0.15)
    else:
        p_sentiment = 0.50 + (news_score / 4.0) * 0.35
    p_sentiment = max(0.05, min(0.98, p_sentiment))

    # 2. Upward Technical Breakout Momentum Component [0.05, 0.95]
    above_sma = tech.get("above_sma20", False)
    mom_5d = float(tech.get("momentum_5d", 0.0))
    p_tech = 0.50 + (0.15 if above_sma else -0.15)
    if mom_5d >= 3.0:
        p_tech += 0.18
    elif mom_5d >= 1.0:
        p_tech += 0.10
    elif mom_5d >= 0.0:
        p_tech += 0.04
    elif mom_5d >= -3.0:
        p_tech -= 0.06
    else:
        p_tech -= 0.18
    p_tech = max(0.05, min(0.95, p_tech))

    # 3. Catalyst Clarity Component [0.05, 1.00]
    c_clarity = 0.10
    has_catalyst_kw = False
    all_headlines = " ".join(art.get("title", "").lower() for art in (articles or []))
    if bert.get("headline"):
        all_headlines = f"{all_headlines} {bert.get('headline', '').lower()}".strip()

    if articles or bert.get("headline"):
        c_clarity = 0.40
        c_clarity += min(0.25, abs(net) * 0.25)
        has_catalyst_kw = any(kw in all_headlines for kw in CATALYST_KEYWORDS)
        if has_catalyst_kw:
            c_clarity += 0.30
    c_clarity = max(0.05, min(1.00, c_clarity))

    # Combined Upside Probability P_upside [0.05, 0.98]
    p_upside = round(0.40 * p_sentiment + 0.35 * p_tech + 0.25 * c_clarity, 4)
    p_upside = max(0.05, min(0.98, p_upside))
    upside_prob_pct = round(p_upside * 100, 1)

    # Market Cap Categorization and Projected Upside Target
    # Small/mid-cap high potential applies when genuine catalysts exist (or strong catalyst clarity)
    has_real_catalyst = has_catalyst_kw or (c_clarity >= 0.65 and net > 0.10)

    if market_cap and market_cap < 2_000_000_000:
        cap_tier = "Small-Cap (<$2B)"
        if has_real_catalyst:
            potential_upside_pct = 22.0
            downside_risk_pct = 6.0
        else:
            potential_upside_pct = 10.0
            downside_risk_pct = 8.0
    elif market_cap and market_cap < 10_000_000_000:
        cap_tier = "Mid-Cap ($2B-$10B)"
        if has_real_catalyst:
            potential_upside_pct = 16.0
            downside_risk_pct = 6.0
        else:
            potential_upside_pct = 9.0
            downside_risk_pct = 6.5
    else:
        cap_tier = "Large/Mega-Cap"
        if has_real_catalyst:
            potential_upside_pct = 11.0
            downside_risk_pct = 5.0
        else:
            potential_upside_pct = 9.0
            downside_risk_pct = 6.0

    expected_return_pct = round((p_upside * potential_upside_pct) - ((1.0 - p_upside) * downside_risk_pct), 2)
    prob_weighted_upside = round(p_upside * potential_upside_pct, 2)

    catalyst_label = "Strong Catalyst" if has_catalyst_kw else ("News Active" if articles else "Technical Only")

    return {
        "symbol": symbol,
        "price": price,
        "sentiment_prob": round(p_sentiment, 4),
        "technical_prob": round(p_tech, 4),
        "catalyst_clarity": round(c_clarity, 4),
        "catalyst_label": catalyst_label,
        "upside_probability": p_upside,
        "upside_probability_pct": upside_prob_pct,
        "cap_tier": cap_tier,
        "potential_upside_pct": potential_upside_pct,
        "expected_return_pct": expected_return_pct,
        "prob_weighted_upside": prob_weighted_upside
    }


def extract_candidate_tickers_from_articles(articles: List[Dict[str, Any]]) -> List[str]:
    """Extracts ticker symbols referenced in live news articles."""
    KNOWN_MAP = {
        "nvidia": "NVDA", "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL",
        "alphabet": "GOOGL", "amazon": "AMZN", "meta": "META", "tesla": "TSLA",
        "amd": "AMD", "palantir": "PLTR", "sofi": "SOFI", "ionq": "IONQ",
        "ast spacemobile": "ASTS", "rocket lab": "RKLB", "super micro": "SMCI",
        "coinbase": "COIN", "robinhood": "HOOD", "crowdstrike": "CRWD", "marathon digital": "MARA"
    }
    IGNORED_WORDS = {
        "CEO", "CFO", "CTO", "USA", "USD", "GDP", "FED", "SEC", "IPO", "ETF",
        "CPI", "AI", "API", "WSJ", "CNBC", "DJIA", "NYSE", "NASDAQ", "SPY", "QQQ",
        "US", "UK", "EU", "EST", "EDT", "PST", "UTC", "GMT", "AM", "PM", "NEW",
        "OLD", "NOW", "ALL", "TOP", "BIG", "OUT", "FOR", "AND", "THE", "NOT",
        "YES", "NO", "MORE", "FREE", "POST", "PRE", "EDIT", "LIVE", "UPDATE"
    }
    found = set()
    for art in articles:
        text = f"{art.get('title', '')} {art.get('summary', '')}"
        # Dollar tags like $PLTR, $SOFI
        for m in re.finditer(r'\$([A-Za-z]{1,5})\b', text):
            sym = m.group(1).upper()
            if sym not in IGNORED_WORDS:
                found.add(sym)
        # Parenthetical tickers like (PLTR)
        for m in re.finditer(r'\(([A-Za-z]{2,5})\)', text):
            sym = m.group(1).upper()
            if sym not in IGNORED_WORDS:
                found.add(sym)
        # Company name mentions
        text_lower = text.lower()
        for company, sym in KNOWN_MAP.items():
            if re.search(r'\b' + re.escape(company) + r'\b', text_lower):
                found.add(sym)
    return list(found)


# ============================================================================
# LIVE MARKET DATA & TECHNICAL ANALYSIS UTILITIES
# ============================================================================

def get_current_price(symbol: str) -> Optional[float]:
    """Fetches real-time / latest market price for a given ticker with fallback."""
    try:
        ticker = yf.Ticker(symbol)
        fast_info = getattr(ticker, "fast_info", None)
        if fast_info:
            last_price = getattr(fast_info, "last_price", None)
            if last_price and not (isinstance(last_price, float) and (last_price != last_price)):
                return round(float(last_price), 2)

        # Fallback to recent history (period 5d handles weekends/holidays)
        hist = ticker.history(period="5d")
        if not hist.empty and "Close" in hist:
            valid_closes = hist["Close"].dropna()
            if not valid_closes.empty:
                return round(float(valid_closes.iloc[-1]), 2)
    except Exception as e:
        safe_print(f"[MarketData] Error fetching price for {symbol}: {e}")
    return None


def get_stock_market_metrics(symbol: str) -> Dict[str, Any]:
    """
    Fetches real-time price, trading volume, market cap, and technical indicators (SMA20, 5d momentum).
    """
    res = {
        "valid": False,
        "price": 0.0,
        "volume": 0.0,
        "market_cap": None,
        "sma20": 0.0,
        "above_sma20": False,
        "momentum_5d": 0.0
    }
    try:
        t = yf.Ticker(symbol)
        fast_info = getattr(t, "fast_info", None)
        if fast_info:
            p = getattr(fast_info, "last_price", None)
            if p and not (isinstance(p, float) and (p != p)):
                res["price"] = round(float(p), 2)
            v = getattr(fast_info, "last_volume", None)
            if v and not (isinstance(v, float) and (v != v)):
                res["volume"] = float(v)
            mc = getattr(fast_info, "market_cap", None)
            if mc and not (isinstance(mc, float) and (mc != mc)):
                res["market_cap"] = float(mc)

        hist = t.history(period="1mo")
        if not hist.empty and "Close" in hist:
            valid_closes = hist["Close"].dropna()
            if not valid_closes.empty:
                cur_price = float(valid_closes.iloc[-1])
                if res["price"] <= 0:
                    res["price"] = round(cur_price, 2)
                lookback_sma = min(20, len(valid_closes))
                sma20 = float(valid_closes.tail(lookback_sma).mean())
                lookback_5d = min(5, len(valid_closes))
                price_5d_ago = float(valid_closes.iloc[-lookback_5d])
                mom_5d = round(((cur_price - price_5d_ago) / price_5d_ago) * 100, 2) if price_5d_ago > 0 else 0.0

                res["sma20"] = round(sma20, 2)
                res["above_sma20"] = cur_price >= sma20
                res["momentum_5d"] = mom_5d
                res["valid"] = True

        if not hist.empty and "Volume" in hist and res["volume"] <= 0:
            valid_vols = hist["Volume"].dropna()
            if not valid_vols.empty:
                res["volume"] = float(valid_vols.iloc[-1])
    except Exception as e:
        safe_print(f"[MarketData] Error fetching metrics for {symbol}: {e}")
    return res


def get_stock_technical_signal(symbol: str) -> Dict[str, Any]:
    """
    Analyzes technical momentum (20-day Simple Moving Average & 5-day return)
    to confirm whether price trend confirms positive news.
    """
    return get_stock_market_metrics(symbol)


# ============================================================================
# SHARED MARKET SNAPSHOT (FAIR CONCURRENT EVALUATION CONTEXT)
# ============================================================================

class MarketSnapshot:
    """
    Caches market prices, volumes, market caps, technical indicators, FinBERT sentiment,
    and probability-weighted upside metrics for all candidate tickers during an evaluation run.
    Guarantees all bots evaluate against the exact same market reality.
    """
    def __init__(
        self,
        prices: Dict[str, float],
        technicals: Dict[str, Dict[str, Any]],
        finbert_scores: Dict[str, Dict[str, Any]],
        ticker_articles: Dict[str, List[Dict[str, Any]]],
        volumes: Optional[Dict[str, float]] = None,
        market_caps: Optional[Dict[str, Optional[float]]] = None,
        upside_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
        borderlist: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        self.prices = prices
        self.technicals = technicals
        self.finbert_scores = finbert_scores
        self.ticker_articles = ticker_articles
        self.volumes = volumes or {}
        self.market_caps = market_caps or {}
        self.upside_metrics = upside_metrics or {}
        self.borderlist = borderlist or {}

    def get_ranked_candidates(self) -> List[str]:
        """
        Returns candidate tickers that passed border guardrails,
        ranked by probability-weighted expected return descending.
        """
        qualified = [
            t for t in self.prices
            if t in self.upside_metrics
            and t not in self.borderlist
            and self.prices.get(t, 0.0) >= MIN_SHARE_PRICE
        ]
        qualified.sort(
            key=lambda t: (
                self.upside_metrics[t].get("expected_return_pct", 0.0),
                self.upside_metrics[t].get("upside_probability_pct", 0.0)
            ),
            reverse=True
        )
        return qualified

    @classmethod
    def capture(
        cls,
        client: Optional[Any],
        watchlist: Optional[List[str]] = None,
        guardrail: Optional[BorderlistGuardrail] = None
    ) -> "MarketSnapshot":
        """Builds a market snapshot across candidate tickers with guardrails and upside ranking."""
        safe_print("\n--- 📡 Capturing Live Market Snapshot for Tournament Arena ---")
        articles = []
        if client:
            try:
                news_res = client.table("market_news").select("title, summary, link").order("created_at", desc=True).limit(30).execute()
                articles = news_res.data or []
            except Exception as e:
                safe_print(f"[MarketData] Note: Could not fetch market_news from Supabase ({e}).")

        # Discover dynamic candidate universe (base watchlist + tickers in news)
        base_candidates = list(watchlist or WATCHLIST)
        news_discovered = extract_candidate_tickers_from_articles(articles)
        for nt in news_discovered:
            if nt not in base_candidates:
                base_candidates.append(nt)

        ticker_articles: Dict[str, List[Dict[str, Any]]] = {t: [] for t in base_candidates}
        for art in articles:
            combined = f"{art.get('title', '')} {art.get('summary', '')}".lower()
            for ticker in base_candidates:
                if re.search(r'\b' + re.escape(ticker.lower()) + r'\b', combined):
                    ticker_articles[ticker].append(art)

        prices: Dict[str, float] = {}
        volumes: Dict[str, float] = {}
        market_caps: Dict[str, Optional[float]] = {}
        technicals: Dict[str, Dict[str, Any]] = {}
        finbert_scores: Dict[str, Dict[str, Any]] = {}
        upside_metrics: Dict[str, Dict[str, Any]] = {}
        border_filter = guardrail or BorderlistGuardrail()

        finbert = None
        try:
            finbert = FinBERTAnalyzer()
        except Exception as e:
            safe_print(f"[FinBERT] Note: Initialized in keyword fallback mode ({e})")

        for ticker in base_candidates:
            # 1. Fetch Market Metrics (Price, Volume, Market Cap, Technicals)
            metrics = get_stock_market_metrics(ticker)
            price = metrics.get("price") or get_current_price(ticker)
            if price:
                prices[ticker] = price
            vol = metrics.get("volume", 0.0)
            volumes[ticker] = vol
            mcap = metrics.get("market_cap")
            market_caps[ticker] = mcap
            technicals[ticker] = metrics

            # 2. Sentiment Analysis (FinBERT + keyword fallback)
            t_arts = ticker_articles.get(ticker, [])
            bullish_count = 0
            bearish_count = 0
            for art in t_arts:
                combined_text = f"{art.get('title', '')} {art.get('summary', '')}".lower()
                bullish_count += sum(1 for kw in BULLISH_SIGNALS if kw in combined_text)
                bearish_count += sum(1 for kw in BEARISH_SIGNALS if kw in combined_text)

            bert_res = None
            if finbert and t_arts:
                try:
                    sample = f"{t_arts[0].get('title', '')}. {t_arts[0].get('summary', '')}"
                    bert_res = finbert.analyze(sample)
                except Exception:
                    bert_res = None

            if bert_res:
                net_score = bert_res["net_score"]
                news_score = round(net_score * 4.0, 2)
                bert_info = f"FinBERT[Pos={bert_res['positive']:.2f}, Neg={bert_res['negative']:.2f}, Net={bert_res['net_score']:.2f}]"
                finbert_scores[ticker] = {
                    "news_score": news_score,
                    "net_score": net_score,
                    "positive": bert_res["positive"],
                    "negative": bert_res["negative"],
                    "neutral": bert_res["neutral"],
                    "info": bert_info,
                    "headline": t_arts[0].get("title", "") if t_arts else ""
                }
            else:
                news_score = 0.0
                if bullish_count > bearish_count:
                    news_score = min(4.0, bullish_count * 1.5)
                elif bearish_count > bullish_count:
                    news_score = -min(4.0, bearish_count * 1.5)
                finbert_scores[ticker] = {
                    "news_score": news_score,
                    "net_score": round(news_score / 4.0, 2),
                    "positive": 1.0 if bullish_count > bearish_count else 0.0,
                    "negative": 1.0 if bearish_count > bullish_count else 0.0,
                    "neutral": 1.0 if bullish_count == bearish_count else 0.0,
                    "info": f"Keywords(+{bullish_count}/-{bearish_count})",
                    "headline": t_arts[0].get("title", "") if t_arts else ""
                }

            # 3. Safety Border Guardrail Evaluation
            net_sentiment = finbert_scores[ticker].get("net_score", 0.0)
            is_hot = (net_sentiment > 0.20 or bullish_count > 0 or metrics.get("momentum_5d", 0.0) > 1.5)
            is_safe, reject_reason = border_filter.evaluate(ticker, price, vol, mcap)

            if not is_safe:
                border_filter.disqualify(ticker, price, vol, mcap, reject_reason, is_hot=is_hot)
            else:
                # 4. Probability-Weighted Upside & Expected Return
                upside = calculate_probability_weighted_upside(
                    ticker, price, metrics, finbert_scores[ticker], t_arts,
                    market_cap=mcap, volume=vol
                )
                upside_metrics[ticker] = upside

        safe_print(
            f"[MarketData] Snapshot ready: {len(prices)} ticker prices, "
            f"{len(border_filter.disqualified)} borderlist disqualified, "
            f"{len(upside_metrics)} qualified ranked candidates.\n"
        )
        return cls(
            prices=prices,
            technicals=technicals,
            finbert_scores=finbert_scores,
            ticker_articles=ticker_articles,
            volumes=volumes,
            market_caps=market_caps,
            upside_metrics=upside_metrics,
            borderlist=border_filter.disqualified
        )


# ============================================================================
# BOT ARCHETYPES & STRATEGY ENGINE
# ============================================================================

class BotArchetype:
    """Base class for autonomous tournament bot archetypes."""
    def __init__(
        self,
        bot_id: str,
        name: str,
        persona: str,
        description: str,
        starting_balance: float = 5000.00,
        max_positions: int = 5,
        allocation_pct: float = 0.15,
        max_trade_dollars: float = 750.00,
        stop_loss_pct: float = -7.5,
        take_profit_pct: float = 12.0
    ):
        self.bot_id = bot_id
        self.name = name
        self.persona = persona
        self.description = description
        self.starting_balance = starting_balance
        self.max_positions = max_positions
        self.allocation_pct = allocation_pct
        self.max_trade_dollars = max_trade_dollars
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def compute_conviction(self, news_score: float, tech: Dict[str, Any]) -> float:
        """Standard multi-factor conviction score (0 to 10 points)."""
        trend_score = 3.0 if tech.get("above_sma20", False) else 0.0
        mom_5d = tech.get("momentum_5d", 0.0)
        mom_score = 2.0 if mom_5d > 0.5 else (1.0 if mom_5d >= 0.0 else 0.0)
        rr_score = 1.0
        return round(max(0.0, news_score) + trend_score + mom_score + rr_score, 1)

    def evaluate_exit(self, position: Dict[str, Any], tech: Dict[str, Any], bert: Dict[str, Any]) -> Tuple[bool, str]:
        """Evaluates whether an existing holding should be sold."""
        pnl_pct = position.get("pnl_pct", 0.0)

        # Baseline Stop Loss & Take Profit
        if pnl_pct >= self.take_profit_pct:
            return True, f"Take Profit triggered: Position reached +{pnl_pct:.2f}% (target +{self.take_profit_pct:.1f}%)."
        if pnl_pct <= self.stop_loss_pct:
            return True, f"Stop Loss triggered: Cut loss at {pnl_pct:.2f}% (limit {self.stop_loss_pct:.1f}%)."

        return False, ""

    def evaluate_entry(
        self,
        ticker: str,
        tech: Dict[str, Any],
        bert: Dict[str, Any],
        articles: List[Dict[str, Any]],
        current_positions: List[str]
    ) -> Tuple[bool, float, str]:
        """Evaluates whether to enter a new buy trade."""
        raise NotImplementedError


class ConservativeSwingBot(BotArchetype):
    """
    Conservative Swing: Risk-averse swing trader.
    Requires high AI conviction (>= 7.0/10), positive FinBERT (news_score >= 1.5),
    and price strictly above 20-day SMA to confirm upward trend.
    """
    def __init__(self):
        super().__init__(
            bot_id="conservative_swing",
            name="Conservative Swing",
            persona="Risk-Managed Swing Trader",
            description="High-conviction swing strategy requiring AI sentiment + confirmed technical trend alignment.",
            starting_balance=5000.00,
            max_positions=5,
            allocation_pct=0.15,
            max_trade_dollars=750.00,
            stop_loss_pct=-7.5,
            take_profit_pct=12.0
        )

    def evaluate_entry(self, ticker: str, tech: Dict[str, Any], bert: Dict[str, Any], articles: List[Dict[str, Any]], current_positions: List[str]) -> Tuple[bool, float, str]:
        if not tech.get("valid", False) or not articles:
            return False, 0.0, ""

        news_score = bert.get("news_score", 0.0)
        conviction = self.compute_conviction(news_score, tech)

        if conviction >= 7.0 and news_score >= 1.5 and tech.get("above_sma20", False):
            rationale = (
                f"Conservative Swing BUY (Score {conviction}/10). "
                f"Sentiment: {bert.get('info')}. "
                f"Trend: ${tech.get('price')} > 20d SMA ${tech.get('sma20')} (5d: +{tech.get('momentum_5d')}%, SL -7.5%, TP +12.0%)."
            )
            return True, conviction, rationale
        return False, conviction, ""

    def evaluate_exit(self, position: Dict[str, Any], tech: Dict[str, Any], bert: Dict[str, Any]) -> Tuple[bool, str]:
        sold, reason = super().evaluate_exit(position, tech, bert)
        if sold:
            return True, reason

        news_score = bert.get("news_score", 0.0)
        # Severe sentiment breakdown or breakdown below SMA with sharp momentum fall
        if news_score <= -2.0 or (not tech.get("above_sma20", True) and tech.get("momentum_5d", 0.0) < -3.0):
            return True, f"Risk Exit: Bearish breakdown ({bert.get('info')}, 5d: {tech.get('momentum_5d', 0.0)}%)."
        return False, ""


class AggressiveBreakoutBot(BotArchetype):
    """
    Aggressive Breakout: Momentum breakout chaser.
    Looks for strong 5-day velocity (>= +2.0%), price pressing above SMA20,
    and non-negative news catalyst. Employs tighter stop losses and ambitious profit targets.
    """
    def __init__(self):
        super().__init__(
            bot_id="aggressive_breakout",
            name="Aggressive Breakout",
            persona="High-Momentum Breakout Chaser",
            description="Aggressive velocity strategy seeking explosive breakouts with tight 5% stops and 18% upside.",
            starting_balance=5000.00,
            max_positions=4,
            allocation_pct=0.25,
            max_trade_dollars=1250.00,
            stop_loss_pct=-5.0,
            take_profit_pct=18.0
        )

    def evaluate_entry(self, ticker: str, tech: Dict[str, Any], bert: Dict[str, Any], articles: List[Dict[str, Any]], current_positions: List[str]) -> Tuple[bool, float, str]:
        if not tech.get("valid", False):
            return False, 0.0, ""

        mom_5d = tech.get("momentum_5d", 0.0)
        net_score = bert.get("net_score", 0.0)
        above_sma = tech.get("above_sma20", False)

        # Triggers on strong positive velocity + above SMA + non-negative news
        if mom_5d >= 2.0 and above_sma and net_score >= 0.05:
            conviction = round(min(10.0, 5.0 + (mom_5d * 0.8) + max(0.0, net_score * 3.0)), 1)
            rationale = (
                f"Aggressive Breakout BUY. Strong 5d velocity +{mom_5d}% > SMA20 (${tech.get('price')} > ${tech.get('sma20')}). "
                f"Sentiment: {bert.get('info')}. Target: +18%, Stop: -5%."
            )
            return True, conviction, rationale
        return False, 0.0, ""

    def evaluate_exit(self, position: Dict[str, Any], tech: Dict[str, Any], bert: Dict[str, Any]) -> Tuple[bool, str]:
        sold, reason = super().evaluate_exit(position, tech, bert)
        if sold:
            return True, reason

        mom_5d = tech.get("momentum_5d", 0.0)
        # Momentum stall or price falling back below SMA
        if mom_5d < -1.5 or (not tech.get("above_sma20", True) and mom_5d < 0.0):
            return True, f"Breakout Stall: Velocity collapsed ({mom_5d}%), exiting to protect capital."
        return False, ""


class ContrarianDipBuyerBot(BotArchetype):
    """
    Dip Buyer / Contrarian: Mean-reversion value seeker.
    Buys short-term pullbacks (-1.5% to -8.0% 5-day return) on solid stocks
    where news is stabilizing/not disastrous, anticipating a sharp rebound.
    """
    def __init__(self):
        super().__init__(
            bot_id="contrarian_dip_buyer",
            name="Dip Buyer / Contrarian",
            persona="Mean-Reversion Value Seeker",
            description="Buys oversold short-term dips on quality tickers with stabilizing sentiment for quick rebound gains.",
            starting_balance=5000.00,
            max_positions=5,
            allocation_pct=0.20,
            max_trade_dollars=1000.00,
            stop_loss_pct=-6.0,
            take_profit_pct=8.0
        )

    def evaluate_entry(self, ticker: str, tech: Dict[str, Any], bert: Dict[str, Any], articles: List[Dict[str, Any]], current_positions: List[str]) -> Tuple[bool, float, str]:
        if not tech.get("valid", False):
            return False, 0.0, ""

        mom_5d = tech.get("momentum_5d", 0.0)
        net_score = bert.get("net_score", 0.0)
        price = tech.get("price", 0.0)
        sma20 = tech.get("sma20", 0.0)

        # Looking for dip of -1.5% to -8.0%, price near SMA20 (within 5%), sentiment not catastrophic
        if -8.0 <= mom_5d <= -1.5 and net_score >= -0.20 and (price >= sma20 * 0.95):
            conviction = round(min(10.0, 6.0 + abs(mom_5d) * 0.4 + max(0.0, net_score * 2.0)), 1)
            rationale = (
                f"Contrarian Dip BUY. Oversold pullback ({mom_5d}% 5d drop), price ${price} within 5% of SMA20 (${sma20}). "
                f"News sentiment stable ({bert.get('info')}). Target: +8.0% bounce."
            )
            return True, conviction, rationale
        return False, 0.0, ""

    def evaluate_exit(self, position: Dict[str, Any], tech: Dict[str, Any], bert: Dict[str, Any]) -> Tuple[bool, str]:
        sold, reason = super().evaluate_exit(position, tech, bert)
        if sold:
            return True, reason

        net_score = bert.get("net_score", 0.0)
        news_score = bert.get("news_score", 0.0)
        mom_5d = tech.get("momentum_5d", 0.0)
        # If dip turns into structural collapse (technicals broke or news sentiment turned sharply negative)
        if mom_5d < -9.0 or net_score <= -0.35 or news_score <= -1.5:
            return True, f"Dip Invalidation: Severe deterioration (5d: {mom_5d}%, news net: {net_score:.2f})."
        return False, ""


class FinBERTCatalystHunterBot(BotArchetype):
    """
    FinBERT Catalyst Hunter: News-first sentiment trader.
    Pounces aggressively on high-conviction positive NLP classifications
    (net score >= +0.55 or positive probability >= 0.70).
    """
    def __init__(self):
        super().__init__(
            bot_id="finbert_catalyst_hunter",
            name="FinBERT Catalyst Hunter",
            persona="AI NLP Catalyst Sniper",
            description="Pure sentiment intelligence strategy exploiting institutional NLP conviction surges and breaking catalysts.",
            starting_balance=5000.00,
            max_positions=5,
            allocation_pct=0.20,
            max_trade_dollars=1000.00,
            stop_loss_pct=-7.0,
            take_profit_pct=10.0
        )

    def evaluate_entry(self, ticker: str, tech: Dict[str, Any], bert: Dict[str, Any], articles: List[Dict[str, Any]], current_positions: List[str]) -> Tuple[bool, float, str]:
        if not articles:
            return False, 0.0, ""

        pos_prob = bert.get("positive", 0.0)
        net_score = bert.get("net_score", 0.0)
        headline = bert.get("headline", "")

        # Triggers on strong positive FinBERT catalyst
        if pos_prob >= 0.70 or net_score >= 0.50:
            conviction = round(min(10.0, 5.0 + (pos_prob * 3.5) + (net_score * 1.5)), 1)
            rationale = (
                f"FinBERT Catalyst BUY (Conviction {conviction}/10). "
                f"Headline: '{headline[:55]}...'. "
                f"Classifier: {bert.get('info')}."
            )
            return True, conviction, rationale
        return False, 0.0, ""

    def evaluate_exit(self, position: Dict[str, Any], tech: Dict[str, Any], bert: Dict[str, Any]) -> Tuple[bool, str]:
        sold, reason = super().evaluate_exit(position, tech, bert)
        if sold:
            return True, reason

        net_score = bert.get("net_score", 0.0)
        # If catalyst flips negative
        if net_score <= -0.40:
            return True, f"Catalyst Invalidation: FinBERT sentiment turned bearish ({bert.get('info')})."
        return False, ""


class TrendFollowerBot(BotArchetype):
    """
    Trend Follower: Classic systematic trend following.
    Enters on confirmed price support above 20-day SMA, non-negative momentum,
    and neutral-to-positive news flow, letting winners run up to +15%.
    """
    def __init__(self):
        super().__init__(
            bot_id="trend_follower",
            name="Trend Follower",
            persona="Systematic Trend Follower",
            description="Systematic trend-following bot riding consistent moving average channels with patient 15% targets.",
            starting_balance=5000.00,
            max_positions=6,
            allocation_pct=0.15,
            max_trade_dollars=750.00,
            stop_loss_pct=-8.0,
            take_profit_pct=15.0
        )

    def evaluate_entry(self, ticker: str, tech: Dict[str, Any], bert: Dict[str, Any], articles: List[Dict[str, Any]], current_positions: List[str]) -> Tuple[bool, float, str]:
        if not tech.get("valid", False):
            return False, 0.0, ""

        above_sma = tech.get("above_sma20", False)
        mom_5d = tech.get("momentum_5d", 0.0)
        net_score = bert.get("net_score", 0.0)

        if above_sma and mom_5d >= 0.0 and net_score >= 0.0:
            conviction = round(min(10.0, 6.0 + (1.5 if above_sma else 0.0) + min(2.0, mom_5d * 0.5)), 1)
            rationale = (
                f"Trend Follower BUY. Sustained uptrend: ${tech.get('price')} > 20d SMA ${tech.get('sma20')} (5d: +{mom_5d}%). "
                f"Sentiment non-negative: {bert.get('info')}."
            )
            return True, conviction, rationale
        return False, 0.0, ""

    def evaluate_exit(self, position: Dict[str, Any], tech: Dict[str, Any], bert: Dict[str, Any]) -> Tuple[bool, str]:
        sold, reason = super().evaluate_exit(position, tech, bert)
        if sold:
            return True, reason

        price = tech.get("price", 0.0)
        sma20 = tech.get("sma20", 0.0)
        mom_5d = tech.get("momentum_5d", 0.0)

        # Trend break: price drops > 1.5% below 20-day SMA or momentum turns sharply negative
        if (sma20 > 0 and price < sma20 * 0.985) or mom_5d < -2.5:
            return True, f"Trend Breakdown: Price lost 20d SMA support (${price} vs SMA ${sma20}, 5d: {mom_5d}%)."
        return False, ""


def get_default_tournament_bots() -> List[BotArchetype]:
    """Returns the official roster of 5 competing tournament bots."""
    return [
        ConservativeSwingBot(),
        AggressiveBreakoutBot(),
        ContrarianDipBuyerBot(),
        FinBERTCatalystHunterBot(),
        TrendFollowerBot()
    ]


# ============================================================================
# TOURNAMENT PERSISTENCE & DATA MANAGER
# ============================================================================

class TournamentDataManager:
    """
    Manages portfolio cash, open positions, and audit trades for all tournament bots.
    - Seamlessly checks if remote Supabase supports `bot_id`.
    - If `bot_id` is supported, reads & writes per bot directly in Supabase.
    - If legacy single-bot schema is active, keeps primary bot in Supabase and
      persists other bots in a local JSON tournament cache (`data/bot_tournament_state.json`).
    - Ensures 100% crash-proof operation and backward compatibility.
    """
    def __init__(self, client: Optional[Any] = None):
        self.client = client
        self.has_supabase_bot_id = False
        self._local_cache: Dict[str, Any] = {}
        self._init_storage()

    def _init_storage(self):
        """Initializes and detects database schema capabilities."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._load_local_cache()

        if self.client:
            try:
                # Test if bot_id column exists on paper_portfolio
                self.client.table("paper_portfolio").select("bot_id").limit(1).execute()
                self.has_supabase_bot_id = True
                safe_print("✅ [Storage Engine] Supabase Multi-Bot schema detected (`bot_id` active).")
            except Exception:
                self.has_supabase_bot_id = False
                safe_print(
                    "ℹ️  [Storage Engine] Supabase running in legacy single-portfolio schema.\n"
                    "    Primary bot ('conservative_swing') mapped to cloud database.\n"
                    "    Tournament arena bots persisting in local cache (`data/bot_tournament_state.json`).\n"
                    "    (To enable full cloud multi-bot storage, run updated schema.sql in Supabase SQL editor)."
                )

    def _load_local_cache(self):
        if LOCAL_STATE_FILE.exists():
            try:
                with open(LOCAL_STATE_FILE, "r", encoding="utf-8") as f:
                    self._local_cache = json.load(f)
            except Exception as e:
                safe_print(f"[Storage] Note: Error reading {LOCAL_STATE_FILE} ({e}), initializing fresh state.")
                self._local_cache = {}
        else:
            self._local_cache = {}

    def _save_local_cache(self):
        try:
            with open(LOCAL_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(self._local_cache, f, indent=2)
        except Exception as e:
            safe_print(f"[Storage] Error writing local state: {e}")

    def ensure_bot_initialized(self, bot: BotArchetype):
        """Ensures the bot has an initialized starting cash record ($5,000.00)."""
        bot_id = bot.bot_id
        if self.has_supabase_bot_id and self.client:
            res = self.client.table("paper_portfolio").select("*").eq("bot_id", bot_id).limit(1).execute()
            if not res.data:
                self.client.table("paper_portfolio").insert({
                    "bot_id": bot_id,
                    "cash_balance": bot.starting_balance,
                    "initial_balance": bot.starting_balance
                }).execute()
        elif not self.has_supabase_bot_id and self.client and bot_id == "conservative_swing":
            # Legacy single row
            res = self.client.table("paper_portfolio").select("*").limit(1).execute()
            if not res.data:
                self.client.table("paper_portfolio").insert({
                    "cash_balance": bot.starting_balance,
                    "initial_balance": bot.starting_balance
                }).execute()
        else:
            # Local store
            if bot_id not in self._local_cache:
                self._local_cache[bot_id] = {
                    "cash_balance": bot.starting_balance,
                    "initial_balance": bot.starting_balance,
                    "positions": {},
                    "trades": []
                }
                self._save_local_cache()

    def get_portfolio_balances(self, bot_id: str, default_balance: float = 5000.00) -> Tuple[float, float]:
        """Returns (cash_balance, initial_balance) for a specific bot."""
        if self.has_supabase_bot_id and self.client:
            res = self.client.table("paper_portfolio").select("*").eq("bot_id", bot_id).limit(1).execute()
            if res.data:
                return float(res.data[0]["cash_balance"]), float(res.data[0]["initial_balance"])
        elif not self.has_supabase_bot_id and self.client and bot_id == "conservative_swing":
            res = self.client.table("paper_portfolio").select("*").limit(1).execute()
            if res.data:
                return float(res.data[0]["cash_balance"]), float(res.data[0]["initial_balance"])

        # Fallback local
        bot_data = self._local_cache.get(bot_id, {})
        cash = float(bot_data.get("cash_balance", default_balance))
        init = float(bot_data.get("initial_balance", default_balance))
        return cash, init

    def update_cash_balance(self, bot_id: str, new_cash: float):
        """Updates available cash balance for a specific bot."""
        now_iso = datetime.now(timezone.utc).isoformat()
        if self.has_supabase_bot_id and self.client:
            self.client.table("paper_portfolio").update({
                "cash_balance": new_cash,
                "updated_at": now_iso
            }).eq("bot_id", bot_id).execute()
        elif not self.has_supabase_bot_id and self.client and bot_id == "conservative_swing":
            res = self.client.table("paper_portfolio").select("id").limit(1).execute()
            if res.data:
                self.client.table("paper_portfolio").update({
                    "cash_balance": new_cash,
                    "updated_at": now_iso
                }).eq("id", res.data[0]["id"]).execute()
        else:
            if bot_id not in self._local_cache:
                self._local_cache[bot_id] = {"cash_balance": new_cash, "initial_balance": 5000.00, "positions": {}, "trades": []}
            self._local_cache[bot_id]["cash_balance"] = new_cash
            self._save_local_cache()

    def get_positions(self, bot_id: str) -> List[Dict[str, Any]]:
        """Returns open positions for a specific bot."""
        if self.has_supabase_bot_id and self.client:
            res = self.client.table("paper_positions").select("*").eq("bot_id", bot_id).execute()
            return res.data or []
        elif not self.has_supabase_bot_id and self.client and bot_id == "conservative_swing":
            res = self.client.table("paper_positions").select("*").execute()
            return res.data or []

        # Local
        bot_data = self._local_cache.get(bot_id, {})
        pos_dict = bot_data.get("positions", {})
        return list(pos_dict.values())

    def upsert_position(self, bot_id: str, symbol: str, shares: float, avg_price: float):
        """Updates or adds an open position for a specific bot."""
        now_iso = datetime.now(timezone.utc).isoformat()
        if self.has_supabase_bot_id and self.client:
            pos_res = self.client.table("paper_positions").select("*").eq("bot_id", bot_id).eq("symbol", symbol).execute()
            if pos_res.data:
                self.client.table("paper_positions").update({
                    "shares": shares,
                    "average_buy_price": avg_price,
                    "updated_at": now_iso
                }).eq("bot_id", bot_id).eq("symbol", symbol).execute()
            else:
                self.client.table("paper_positions").insert({
                    "bot_id": bot_id,
                    "symbol": symbol,
                    "shares": shares,
                    "average_buy_price": avg_price
                }).execute()
        elif not self.has_supabase_bot_id and self.client and bot_id == "conservative_swing":
            pos_res = self.client.table("paper_positions").select("*").eq("symbol", symbol).execute()
            if pos_res.data:
                self.client.table("paper_positions").update({
                    "shares": shares,
                    "average_buy_price": avg_price,
                    "updated_at": now_iso
                }).eq("symbol", symbol).execute()
            else:
                self.client.table("paper_positions").insert({
                    "symbol": symbol,
                    "shares": shares,
                    "average_buy_price": avg_price
                }).execute()
        else:
            if bot_id not in self._local_cache:
                self._local_cache[bot_id] = {"cash_balance": 5000.00, "initial_balance": 5000.00, "positions": {}, "trades": []}
            self._local_cache[bot_id]["positions"][symbol] = {
                "symbol": symbol,
                "shares": shares,
                "average_buy_price": avg_price,
                "updated_at": now_iso
            }
            self._save_local_cache()

    def remove_position(self, bot_id: str, symbol: str):
        """Removes an open position upon full exit."""
        if self.has_supabase_bot_id and self.client:
            self.client.table("paper_positions").delete().eq("bot_id", bot_id).eq("symbol", symbol).execute()
        elif not self.has_supabase_bot_id and self.client and bot_id == "conservative_swing":
            self.client.table("paper_positions").delete().eq("symbol", symbol).execute()
        else:
            if bot_id in self._local_cache and symbol in self._local_cache[bot_id].get("positions", {}):
                del self._local_cache[bot_id]["positions"][symbol]
                self._save_local_cache()

    def record_trade(self, bot_id: str, trade_data: Dict[str, Any]):
        """Records a simulated trade in audit history."""
        now_iso = datetime.now(timezone.utc).isoformat()
        record = {
            "symbol": trade_data["symbol"],
            "action": trade_data["action"],
            "shares": trade_data["shares"],
            "price": trade_data["price"],
            "total_value": trade_data["total_value"],
            "rationale": trade_data.get("rationale", "")
        }
        if self.has_supabase_bot_id and self.client:
            record["bot_id"] = bot_id
            self.client.table("paper_trades").insert(record).execute()
        elif not self.has_supabase_bot_id and self.client and bot_id == "conservative_swing":
            self.client.table("paper_trades").insert(record).execute()
        else:
            if bot_id not in self._local_cache:
                self._local_cache[bot_id] = {"cash_balance": 5000.00, "initial_balance": 5000.00, "positions": {}, "trades": []}
            record["created_at"] = now_iso
            record["bot_id"] = bot_id
            if "pnl" in trade_data:
                record["pnl"] = trade_data["pnl"]
            if "pnl_pct" in trade_data:
                record["pnl_pct"] = trade_data["pnl_pct"]
            self._local_cache[bot_id]["trades"].insert(0, record)
            self._save_local_cache()

    def get_trades(self, bot_id: str, limit: int = 15) -> List[Dict[str, Any]]:
        """Returns executed trades for a specific bot."""
        if self.has_supabase_bot_id and self.client:
            res = self.client.table("paper_trades").select("*").eq("bot_id", bot_id).order("created_at", desc=True).limit(limit).execute()
            return res.data or []
        elif not self.has_supabase_bot_id and self.client and bot_id == "conservative_swing":
            res = self.client.table("paper_trades").select("*").order("created_at", desc=True).limit(limit).execute()
            return res.data or []

        # Local
        bot_data = self._local_cache.get(bot_id, {})
        trades = bot_data.get("trades", [])
        return trades[:limit]

    def record_borderlist_item(self, symbol: str, data: Dict[str, Any]):
        """Persists a disqualified ticker in borderlist storage."""
        if "borderlist" not in self._local_cache:
            self._local_cache["borderlist"] = {}
        self._local_cache["borderlist"][symbol] = data
        self._save_local_cache()

    def remove_borderlist_item(self, symbol: str):
        """Removes a ticker from borderlist storage once it meets safety requirements."""
        if "borderlist" in self._local_cache and symbol in self._local_cache["borderlist"]:
            del self._local_cache["borderlist"][symbol]
            self._save_local_cache()

    def record_borderlist_batch(self, items: Dict[str, Dict[str, Any]]):
        """Persists multiple disqualified tickers in borderlist storage."""
        if not items:
            return
        if "borderlist" not in self._local_cache:
            self._local_cache["borderlist"] = {}
        self._local_cache["borderlist"].update(items)
        self._save_local_cache()

    def get_borderlist(self) -> Dict[str, Dict[str, Any]]:
        """Returns disqualified borderlist records."""
        return self._local_cache.get("borderlist", {})


# ============================================================================
# MULTI-BOT TOURNAMENT ENGINE
# ============================================================================

class MultiBotTournamentEngine:
    """
    Main tournament arena engine:
    1. Manages competing bot personas.
    2. Enforces Safety Border / Guardrail Filter ("Borderlist"):
       - $5.00 share price floor
       - Legitimate volume (>= 200,000) OR Market Cap (>= $250M)
       - Automated skipped logging for disqualified tickers
    3. Ranks candidates by probability-weighted upside and expected return:
       - FinBERT net positive sentiment + upward breakout momentum + catalyst clarity
       - Small/mid-cap high potential candidates prioritized when real catalysts exist
    4. Evaluates all bots concurrently and independently.
    5. Computes performance metrics (PnL, Win Rate %, Best Trade).
    6. Displays Live Leaderboard, Borderlist status, and audit records.
    """
    def __init__(self, bots: Optional[List[BotArchetype]] = None, client: Any = _DEFAULT_CLIENT):
        if client is _DEFAULT_CLIENT:
            try:
                self.client = get_supabase_client()
            except Exception:
                self.client = None
        else:
            self.client = client

        self.storage = TournamentDataManager(self.client)
        self.guardrail = BorderlistGuardrail()
        self.watchlist = list(WATCHLIST)
        self.bots: Dict[str, BotArchetype] = {
            b.bot_id: b for b in (bots or get_default_tournament_bots())
        }

        # Initialize funds for all registered bots
        for bot in self.bots.values():
            self.storage.ensure_bot_initialized(bot)

    def get_bot_status(self, bot_id: str, snapshot: Optional[MarketSnapshot] = None) -> Dict[str, Any]:
        """Calculates current portfolio equity, positions, P&L, win rate, and best trade for a bot."""
        bot = self.bots[bot_id]
        cash, initial = self.storage.get_portfolio_balances(bot_id, bot.starting_balance)
        raw_positions = self.storage.get_positions(bot_id)

        holdings_value = 0.0
        enriched_positions = []

        for pos in raw_positions:
            symbol = pos["symbol"]
            shares = float(pos["shares"])
            avg_price = float(pos["average_buy_price"])

            cur_price = avg_price
            if snapshot and symbol in snapshot.prices:
                cur_price = snapshot.prices[symbol]
            else:
                fetched_price = get_current_price(symbol)
                if fetched_price:
                    cur_price = fetched_price

            market_val = round(shares * cur_price, 2)
            cost_basis = round(shares * avg_price, 2)
            pnl = round(market_val - cost_basis, 2)
            pnl_pct = round((pnl / cost_basis * 100), 2) if cost_basis > 0 else 0.0

            holdings_value += market_val
            enriched_positions.append({
                "symbol": symbol,
                "shares": shares,
                "buy_price": avg_price,
                "cur_price": cur_price,
                "market_value": market_val,
                "pnl": pnl,
                "pnl_pct": pnl_pct
            })

        total_equity = round(cash + holdings_value, 2)
        total_pnl = round(total_equity - initial, 2)
        total_pnl_pct = round((total_pnl / initial * 100), 2) if initial > 0 else 0.0

        # Calculate Win Rate & Best Trade from trade logs
        trades = self.storage.get_trades(bot_id, limit=50)
        wins = 0
        losses = 0
        best_trade_val = None
        best_trade_sym = None

        # Check completed round-trip trades from rationale metadata
        for t in trades:
            if t.get("action") == "SELL":
                rationale = t.get("rationale", "")
                pnl_match = re.search(r'PnL:\s*([+-]?\$?[0-9,.]+)', rationale)
                if pnl_match:
                    try:
                        clean_val = pnl_match.group(1).replace("$", "").replace(",", "")
                        val = float(clean_val)
                        if val > 0:
                            wins += 1
                        else:
                            losses += 1
                        if best_trade_val is None or val > best_trade_val:
                            best_trade_val = val
                            best_trade_sym = t.get("symbol", "")
                    except Exception:
                        pass
                else:
                    pass

        # If no closed trades, look at open positions for best unrealized trade
        if best_trade_val is None and enriched_positions:
            top_pos = max(enriched_positions, key=lambda x: x["pnl"])
            best_trade_val = top_pos["pnl"]
            best_trade_sym = f"{top_pos['symbol']} (Open)"

        total_closed = wins + losses
        win_rate_pct = round((wins / total_closed * 100), 1) if total_closed > 0 else None

        return {
            "bot_id": bot_id,
            "name": bot.name,
            "persona": bot.persona,
            "description": bot.description,
            "cash": cash,
            "initial": initial,
            "holdings_value": round(holdings_value, 2),
            "total_equity": total_equity,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "positions": enriched_positions,
            "wins": wins,
            "losses": losses,
            "total_closed": total_closed,
            "win_rate_pct": win_rate_pct,
            "best_trade_val": best_trade_val,
            "best_trade_sym": best_trade_sym
        }

    def execute_buy(
        self,
        bot_id: str,
        symbol: str,
        amount_dollars: float,
        rationale: str,
        current_price: Optional[float] = None,
        volume: Optional[float] = None,
        market_cap: Optional[float] = None
    ) -> bool:
        """Simulates buying stock for a specific bot's independent fund with border guardrail verification."""
        bot = self.bots[bot_id]
        price = current_price or get_current_price(symbol)
        if not price or price <= 0:
            safe_print(f"[Tournament | {bot.name}] Cannot buy {symbol}: invalid price {price}")
            return False

        # Safety Border Guardrail Verification
        # If liquidity / market cap are not supplied, attempt to query live market metrics
        if volume is None or market_cap is None:
            metrics = get_stock_market_metrics(symbol)
            if metrics.get("valid"):
                if volume is None and metrics.get("volume", 0.0) > 0:
                    volume = metrics.get("volume")
                if market_cap is None and metrics.get("market_cap"):
                    market_cap = metrics.get("market_cap")

        is_safe, reject_reason = self.guardrail.evaluate(symbol, price, volume, market_cap)
        if not is_safe:
            is_hot = any(term in rationale.lower() for term in ["catalyst", "surge", "breakout", "momentum", "sentiment", "hot", "bull"])
            self.guardrail.disqualify(symbol, price, volume, market_cap, reject_reason, is_hot=is_hot)
            self.storage.record_borderlist_item(symbol, self.guardrail.disqualified[symbol])
            return False

        # If security passed guardrails, clear any stale borderlist status
        self.storage.remove_borderlist_item(symbol)

        shares = round(amount_dollars / price, 4)
        if shares <= 0:
            return False

        actual_cost = round(shares * price, 2)
        cash, _ = self.storage.get_portfolio_balances(bot_id, bot.starting_balance)

        if cash < actual_cost:
            safe_print(f"[Tournament | {bot.name}] Insufficient cash for ${actual_cost:.2f} of {symbol}. Available: ${cash:.2f}")
            return False

        # 1. Update cash
        new_cash = round(cash - actual_cost, 2)
        self.storage.update_cash_balance(bot_id, new_cash)

        # 2. Update position
        existing_positions = {p["symbol"]: p for p in self.storage.get_positions(bot_id)}
        if symbol in existing_positions:
            old_p = existing_positions[symbol]
            old_shares = float(old_p["shares"])
            old_price = float(old_p["average_buy_price"])
            new_shares = round(old_shares + shares, 4)
            blended_price = round(((old_shares * old_price) + (shares * price)) / new_shares, 2)
            self.storage.upsert_position(bot_id, symbol, new_shares, blended_price)
        else:
            self.storage.upsert_position(bot_id, symbol, shares, price)

        # 3. Log Trade Audit
        self.storage.record_trade(bot_id, {
            "symbol": symbol,
            "action": "BUY",
            "shares": shares,
            "price": price,
            "total_value": actual_cost,
            "rationale": f"[{bot.name}] {rationale}"
        })

        safe_print(f"✅ [PAPER BUY] [{bot.name}] {shares:.4f} shares of {symbol} @ ${price:.2f} (${actual_cost:.2f})")
        safe_print(f"   ↳ Rationale: {rationale}")
        return True

    def execute_sell(self, bot_id: str, symbol: str, shares: float, rationale: str, current_price: Optional[float] = None) -> bool:
        """Simulates selling a stock holding back to a bot's cash fund."""
        bot = self.bots[bot_id]
        price = current_price or get_current_price(symbol)
        if not price or price <= 0:
            return False

        existing_positions = {p["symbol"]: p for p in self.storage.get_positions(bot_id)}
        if symbol not in existing_positions:
            return False

        pos = existing_positions[symbol]
        held_shares = float(pos["shares"])
        avg_buy_price = float(pos["average_buy_price"])
        shares_to_sell = min(shares, held_shares)
        gross_proceeds = round(shares_to_sell * price, 2)
        cost_basis = round(shares_to_sell * avg_buy_price, 2)
        pnl = round(gross_proceeds - cost_basis, 2)
        pnl_pct = round((pnl / cost_basis * 100), 2) if cost_basis > 0 else 0.0

        cash, _ = self.storage.get_portfolio_balances(bot_id, bot.starting_balance)
        new_cash = round(cash + gross_proceeds, 2)

        # 1. Update cash
        self.storage.update_cash_balance(bot_id, new_cash)

        # 2. Update position
        rem_shares = round(held_shares - shares_to_sell, 4)
        if rem_shares <= 0.0001:
            self.storage.remove_position(bot_id, symbol)
        else:
            self.storage.upsert_position(bot_id, symbol, rem_shares, avg_buy_price)

        # 3. Log Trade Audit
        enhanced_rationale = f"[{bot.name}] {rationale} [PnL: {'+' if pnl >= 0 else ''}${pnl:.2f} ({'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%)]"
        self.storage.record_trade(bot_id, {
            "symbol": symbol,
            "action": "SELL",
            "shares": shares_to_sell,
            "price": price,
            "total_value": gross_proceeds,
            "rationale": enhanced_rationale
        })

        pnl_str = f"{'+' if pnl >= 0 else ''}${pnl:.2f} ({'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%)"
        safe_print(f"🔴 [PAPER SELL] [{bot.name}] {shares_to_sell:.4f} shares of {symbol} @ ${price:.2f} (${gross_proceeds:.2f}) | PnL: {pnl_str}")
        safe_print(f"   ↳ Rationale: {rationale}")
        return True

    def get_borderlist(self) -> Dict[str, Dict[str, Any]]:
        """Returns consolidated borderlist of disqualified tickers."""
        disq = dict(self.storage.get_borderlist())
        disq.update(self.guardrail.disqualified)
        return disq

    def print_borderlist(self):
        """Displays formatted table of all tickers disqualified by the safety border guardrails."""
        safe_print("\n" + "=" * 105)
        safe_print("           🛡️  SAFETY BORDERLIST / GUARDRAILS AUDIT (DISQUALIFIED TICKERS) 🛡️           ")
        safe_print("=" * 105)
        borderlist = self.get_borderlist()
        if borderlist:
            border_rows = []
            for sym, item in borderlist.items():
                p_val = f"${item.get('price', 0.0):.2f}" if item.get('price') else "N/A"
                v_val = f"{int(item.get('volume', 0)):,}" if item.get('volume') else "N/A"
                mc_val = f"${item.get('market_cap', 0)/1e6:.1f}M" if item.get('market_cap') else "N/A"
                hot_str = "🔥 Hot Trending" if item.get("is_hot") else "Normal"
                border_rows.append([
                    sym,
                    p_val,
                    v_val,
                    mc_val,
                    hot_str,
                    item.get("reason", "Disqualified by guardrail")
                ])
            safe_print(tabulate(border_rows, headers=["Ticker", "Price", "Daily Volume", "Market Cap", "Trend Status", "Disqualification Reason"], tablefmt="grid"))
        else:
            safe_print("   ✅ No tickers currently disqualified. All candidate securities meet safety border guardrails.")
        safe_print("=" * 105 + "\n")

    def evaluate_tournament(self) -> Dict[str, Any]:
        """
        Executes a synchronized evaluation cycle for all registered bots:
        1. Captures a single shared market snapshot (news, prices, FinBERT, indicators).
        2. Filters universe via BorderlistGuardrail (blocks penny stocks <$5, illiquid micro-caps).
        3. Ranks qualified candidates by probability-weighted upside and expected return.
        4. Evaluates exit triggers on open positions per bot.
        5. Evaluates entry triggers on ranked candidate tickers per bot rules.
        """
        safe_print("\n" + "=" * 90)
        safe_print("          ⚔️  STARTING MULTI-BOT ARENA TOURNAMENT EVALUATION ⚔️          ")
        safe_print("=" * 90)

        snapshot = MarketSnapshot.capture(self.client, self.watchlist, guardrail=self.guardrail)

        # Persist borderlist disqualified items
        if snapshot.borderlist:
            self.storage.record_borderlist_batch(snapshot.borderlist)

        # Clear any qualified candidate tickers from persisted borderlist
        for cand_sym in snapshot.upside_metrics:
            if cand_sym not in snapshot.borderlist:
                self.storage.remove_borderlist_item(cand_sym)

        ranked_candidates = snapshot.get_ranked_candidates()

        if ranked_candidates:
            safe_print("\n📈 Candidates Ranked by Probability-Weighted Upside / Expected Return:")
            rank_rows = []
            for rank_i, cand_sym in enumerate(ranked_candidates[:10], 1):
                up = snapshot.upside_metrics.get(cand_sym, {})
                rank_rows.append([
                    rank_i,
                    cand_sym,
                    up.get("cap_tier", "N/A"),
                    f"${up.get('price', 0.0):.2f}",
                    f"{up.get('upside_probability_pct', 0.0)}%",
                    f"+{up.get('expected_return_pct', 0.0)}%",
                    f"{up.get('technical_prob', 0.0):.2f}",
                    up.get("catalyst_label", "N/A")
                ])
            safe_print(tabulate(rank_rows, headers=["Rank", "Ticker", "Cap Tier", "Price", "Upside Prob", "Exp Return", "Tech Score", "Catalyst"], tablefmt="grid"))

        # Iterate over all registered bots concurrently
        for bot_id, bot in self.bots.items():
            safe_print(f"\n--- 🤖 Evaluating Bot: [{bot.name}] ({bot.persona}) ---")
            status = self.get_bot_status(bot_id, snapshot)
            safe_print(f"Current Fund: Cash ${status['cash']:.2f} | Total Equity: ${status['total_equity']:.2f} | Open Positions: {len(status['positions'])}")

            # 1. Evaluate Exit Triggers on Open Positions
            for pos in status["positions"]:
                symbol = pos["symbol"]
                tech = snapshot.technicals.get(symbol, {"valid": False})
                bert = snapshot.finbert_scores.get(symbol, {})
                cur_price = snapshot.prices.get(symbol, pos["cur_price"])

                should_exit, exit_reason = bot.evaluate_exit(pos, tech, bert)
                if should_exit:
                    self.execute_sell(bot_id, symbol, pos["shares"], exit_reason, current_price=cur_price)

            # Refresh status after exits
            refreshed_status = self.get_bot_status(bot_id, snapshot)
            open_symbols = [p["symbol"] for p in refreshed_status["positions"]]

            # 2. Evaluate Entry Candidates in ranked order of probability-weighted upside
            for ticker in ranked_candidates:
                if len(refreshed_status["positions"]) >= bot.max_positions and ticker not in open_symbols:
                    continue  # Position limit reached for this bot

                tech = snapshot.technicals.get(ticker, {"valid": False})
                bert = snapshot.finbert_scores.get(ticker, {})
                articles = snapshot.ticker_articles.get(ticker, [])
                price = snapshot.prices.get(ticker)

                if not price or price <= 0:
                    continue

                should_enter, conviction, rationale = bot.evaluate_entry(
                    ticker, tech, bert, articles, open_symbols
                )

                if should_enter and ticker not in open_symbols:
                    # Enrich rationale with probability-weighted upside metrics
                    metrics = snapshot.upside_metrics.get(ticker, {})
                    upside_str = f" [Upside Prob: {metrics.get('upside_probability_pct', 0)}%, Exp Return: +{metrics.get('expected_return_pct', 0)}%, Tier: {metrics.get('cap_tier', 'N/A')}]"
                    full_rationale = f"{rationale}{upside_str}"

                    trade_allocation = round(min(refreshed_status["total_equity"] * bot.allocation_pct, bot.max_trade_dollars), 2)
                    if refreshed_status["cash"] >= trade_allocation:
                        self.execute_buy(
                            bot_id,
                            ticker,
                            trade_allocation,
                            full_rationale,
                            current_price=price,
                            volume=snapshot.volumes.get(ticker),
                            market_cap=snapshot.market_caps.get(ticker)
                        )
                        # Re-update status so we don't exceed cash on subsequent candidate
                        refreshed_status = self.get_bot_status(bot_id, snapshot)
                        open_symbols.append(ticker)

        safe_print("\n" + "=" * 90)
        safe_print("          ✅ MULTI-BOT TOURNAMENT EVALUATION COMPLETE           ")
        safe_print("=" * 90)
        return {"status": "success", "bots_evaluated": len(self.bots)}

    def get_leaderboard_data(self) -> List[Dict[str, Any]]:
        """Returns sorted list of bots ranked by total equity descending."""
        standings = [self.get_bot_status(bid) for bid in self.bots]
        standings.sort(key=lambda x: (x["total_equity"], x["total_pnl"]), reverse=True)

        for i, s in enumerate(standings, 1):
            s["rank"] = i
            if i == 1:
                s["medal"] = "🥇"
            elif i == 2:
                s["medal"] = "🥈"
            elif i == 3:
                s["medal"] = "🥉"
            else:
                s["medal"] = "  "
        return standings

    def print_leaderboard(self):
        """Renders the formatted Multi-Bot Leaderboard and Borderlist Status."""
        standings = self.get_leaderboard_data()

        safe_print("\n" + "=" * 105)
        safe_print("                             🏆 AI MULTI-BOT TOURNAMENT LIVE LEADERBOARD 🏆                             ")
        safe_print("=" * 105)

        table_rows = []
        for b in standings:
            rank_str = f"{b['rank']} {b['medal']}".strip()
            equity_str = f"${b['total_equity']:,.2f}"
            pnl_sign = "+" if b["total_pnl"] >= 0 else ""
            pnl_str = f"{pnl_sign}${b['total_pnl']:,.2f} ({pnl_sign}{b['total_pnl_pct']:.2f}%)"

            if b["win_rate_pct"] is not None:
                win_str = f"{b['win_rate_pct']:.1f}% ({b['wins']}/{b['total_closed']})"
            else:
                win_str = f"0.0% (0/{b['total_closed']})"

            open_count = len(b["positions"])
            if open_count > 0:
                symbols = ", ".join(p["symbol"] for p in b["positions"])
                pos_str = f"{open_count} ({symbols})"
            else:
                pos_str = "0 (Cash)"

            if b["best_trade_val"] is not None and b["best_trade_sym"]:
                best_sign = "+" if b["best_trade_val"] >= 0 else ""
                best_str = f"{best_sign}${b['best_trade_val']:.2f} ({b['best_trade_sym']})"
            else:
                best_str = "N/A"

            table_rows.append([
                rank_str,
                b["name"],
                equity_str,
                pnl_str,
                win_str,
                pos_str,
                best_str
            ])

        headers = ["Rank", "Bot Persona", "Total Equity", "Net P&L", "Win Rate %", "Open Positions", "Best Trade"]
        safe_print(tabulate(table_rows, headers=headers, tablefmt="grid"))

        # Arena Summary Statistics
        total_starting = sum(b["initial"] for b in standings)
        total_current = sum(b["total_equity"] for b in standings)
        arena_pnl = round(total_current - total_starting, 2)
        arena_pnl_sign = "+" if arena_pnl >= 0 else ""
        arena_pnl_pct = round((arena_pnl / total_starting * 100), 2) if total_starting > 0 else 0.0

        leader = standings[0] if standings else None
        leader_str = f"{leader['name']} ({leader['medal']} ${leader['total_equity']:,.2f})" if leader else "None"

        safe_print("-" * 105)
        safe_print(f"🏟️  Arena Pool Capital: ${total_starting:,.2f}  |  Current Pool Equity: ${total_current:,.2f} ({arena_pnl_sign}${arena_pnl:,.2f} / {arena_pnl_sign}{arena_pnl_pct:.2f}%)")
        safe_print(f"👑  Current Tournament Leader: {leader_str}")
        safe_print("=" * 105)

        # Open Positions Detailed Breakdown
        has_positions = any(len(b["positions"]) > 0 for b in standings)
        if has_positions:
            safe_print("\n📊 Active Holdings Across All Bots:")
            pos_table = []
            for b in standings:
                for p in b["positions"]:
                    pnl_sign = "+" if p["pnl"] >= 0 else ""
                    pos_table.append([
                        b["name"],
                        p["symbol"],
                        f"{p['shares']:.4f}",
                        f"${p['buy_price']:.2f}",
                        f"${p['cur_price']:.2f}",
                        f"${p['market_value']:.2f}",
                        f"{pnl_sign}${p['pnl']:.2f} ({pnl_sign}{p['pnl_pct']:.2f}%)"
                    ])
            safe_print(tabulate(pos_table, headers=["Bot", "Ticker", "Shares", "Buy Price", "Market Price", "Value", "Unrealized P&L"], tablefmt="grid"))
        else:
            safe_print("\n📊 Active Holdings: All bots are currently 100% in cash waiting for high-conviction catalysts.")

        # Recent Trades Audit
        all_recent_trades = []
        for b in standings:
            trades = self.storage.get_trades(b["bot_id"], limit=3)
            for t in trades:
                t["bot_name"] = b["name"]
                all_recent_trades.append(t)

        if all_recent_trades:
            all_recent_trades.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
            safe_print("\n📝 Recent Executed Paper Trades (Latest 5):")
            recent_table = []
            for t in all_recent_trades[:5]:
                date_str = str(t.get("created_at", ""))[:16].replace("T", " ")
                recent_table.append([
                    date_str,
                    t.get("bot_name", ""),
                    t.get("action", ""),
                    t.get("symbol", ""),
                    f"{float(t.get('shares', 0)):.4f}",
                    f"${float(t.get('price', 0)):.2f}",
                    f"${float(t.get('total_value', 0)):.2f}",
                    (t.get("rationale", "")[:45] + "...") if len(t.get("rationale", "")) > 45 else t.get("rationale", "")
                ])
            safe_print(tabulate(recent_table, headers=["Date", "Bot", "Action", "Symbol", "Shares", "Price", "Total", "Rationale"], tablefmt="grid"))

        # Safety Border / Guardrail Filter Status
        safe_print("\n🛡️  SAFETY BORDERLIST & GUARDRAIL STATUS (Min Price: $5.00 | Min Vol: 200,000 | Min MCap: $250M):")
        borderlist = self.get_borderlist()
        if borderlist:
            border_rows = []
            for sym, item in borderlist.items():
                p_val = f"${item.get('price', 0.0):.2f}" if item.get('price') else "N/A"
                v_val = f"{int(item.get('volume', 0)):,}" if item.get('volume') else "N/A"
                mc_val = f"${item.get('market_cap', 0)/1e6:.1f}M" if item.get('market_cap') else "N/A"
                hot_str = "🔥 Hot Trending" if item.get("is_hot") else "Normal"
                border_rows.append([
                    sym,
                    p_val,
                    v_val,
                    mc_val,
                    hot_str,
                    item.get("reason", "Disqualified by guardrail")
                ])
            safe_print(tabulate(border_rows, headers=["Ticker", "Price", "Daily Volume", "Market Cap", "Trend Status", "Disqualification Reason"], tablefmt="grid"))
        else:
            safe_print("   ✅ No tickers currently disqualified. All monitored candidate securities meet safety border guardrails.")

        safe_print("=" * 105 + "\n")


# ============================================================================
# BACKWARD COMPATIBILITY ADAPTER
# ============================================================================

class PaperTradingEngine:
    """
    Backward-compatibility wrapper for existing single-bot code or imports.
    Delegates to MultiBotTournamentEngine while maintaining single-portfolio APIs.
    """
    def __init__(self, bot_id: str = "conservative_swing"):
        self.bot_id = bot_id
        self.tournament = MultiBotTournamentEngine()

    def get_portfolio_status(self) -> Dict[str, Any]:
        """Returns portfolio status for the primary bot."""
        return self.tournament.get_bot_status(self.bot_id)

    def execute_buy(self, symbol: str, amount_dollars: float, rationale: str):
        """Simulates buying stock for the primary bot."""
        self.tournament.execute_buy(self.bot_id, symbol, amount_dollars, rationale)

    def execute_sell(self, symbol: str, shares: float, rationale: str):
        """Simulates selling stock for the primary bot."""
        self.tournament.execute_sell(self.bot_id, symbol, shares, rationale)

    def evaluate_news_and_trade(self):
        """Runs the multi-bot tournament evaluation cycle."""
        self.tournament.evaluate_tournament()

    def print_dashboard(self):
        """Prints the full tournament leaderboard."""
        self.tournament.print_leaderboard()
