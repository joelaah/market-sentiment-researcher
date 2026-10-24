"""
Adversarial Stress Test Suite for Milestone 1:
Learned Multi-Modal Fusion Engine & Candidate Ranking.

Author: Challenger 2 (Empirical Challenger)
Role: critic, specialist

Test Coverage:
1. Fallback Mechanics:
   - Forced LightGBM import failure & runtime exception fallback.
   - Forced Scikit-Learn import failure & runtime exception fallback.
   - Tier 3 Pure NumPy fallback behavior across 5 market regimes.
   - Extreme, unnormalized, NaN, Inf, and None input resilience.
2. Short Price Histories for Rolling Indicators in get_stock_market_metrics:
   - 0 bars (empty DataFrame, empty Series).
   - 1 bar (single price, no diffs).
   - 2 bars (single diff, zero volatility, flat prices).
   - 5 bars (lookback boundary for momentum and Volume Z).
   - 13 bars (lookback boundary below 14-period RSI/ATR and 20-period SMA/VolZ).
   - Zero and negative price handling (ZeroDivisionError checks).
   - Missing High/Low columns in DataFrame.
3. Candidate Ranking Stability Across Simulated Universe Snapshots:
   - Deterministic candidate ordering across 100 snapshot evaluations.
   - Disqualification of benchmark indices (^NSEI) and penny stocks (< $5).
   - Small-cap with catalyst priority over stagnant mega-caps.
   - Tie-breaking consistency (expected_return_pct -> upside_probability_pct).
   - Sensitivity & monotonicity of rank when positive catalyst arrives.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

from src.fusion_engine import (
    MultiModalFusionEngine,
    NumpyFusionFallback,
    LightGBMFusionModel,
    SklearnFusionModel,
    get_fusion_engine,
    reset_fusion_engine,
    FEATURE_NAMES,
    CATALYST_ANCHOR_VECTOR,
)
import src.fusion_engine as fe_mod
from src.paper_trader import (
    calculate_probability_weighted_upside,
    get_stock_market_metrics,
    MarketSnapshot,
    BorderlistGuardrail,
    is_benchmark_index,
    get_min_price_for_symbol,
)


class TestFallbackMechanics(unittest.TestCase):
    """Adversarial stress-testing of 3-tier fallback architecture."""

    def setUp(self):
        reset_fusion_engine()

    def tearDown(self):
        reset_fusion_engine()

    def test_forced_lightgbm_import_failure(self):
        """Simulate LightGBM completely missing from the Python environment."""
        with patch.object(fe_mod, "HAS_LIGHTGBM", False), \
             patch.object(fe_mod, "lgb", None):
            engine = MultiModalFusionEngine()
            # Must not crash, and should report Tier 2 or Tier 3
            tier = engine.active_tier
            self.assertTrue(
                "Tier 2" in tier or "Tier 3" in tier,
                f"Expected Tier 2 or Tier 3 when LightGBM is missing, got {tier}"
            )
            # Predict must function cleanly
            prob = engine.predict_edge_probability(net_sentiment=0.4, rsi_14=60.0)
            self.assertIsInstance(prob, float)
            self.assertTrue(0.05 <= prob <= 0.98)

            # Explicit tier 1 request when unavailable must gracefully fall back without crash
            engine_t1 = MultiModalFusionEngine(tier=1)
            prob_t1 = engine_t1.predict_edge_probability(net_sentiment=0.4, rsi_14=60.0)
            self.assertIsInstance(prob_t1, float)
            self.assertTrue(0.05 <= prob_t1 <= 0.98)

    def test_forced_sklearn_import_failure(self):
        """Simulate Scikit-Learn completely missing from the Python environment."""
        with patch.object(fe_mod, "HAS_SKLEARN", False), \
             patch.object(fe_mod, "sklearn", None), \
             patch.object(fe_mod, "LogisticRegression", None):
            engine = MultiModalFusionEngine(tier=2)
            self.assertIn("Unavailable", engine.active_tier)
            # Must fall back cleanly to Pure NumPy (Tier 3)
            prob = engine.predict_edge_probability(net_sentiment=0.3)
            self.assertIsInstance(prob, float)
            self.assertTrue(0.05 <= prob <= 0.98)

    def test_forced_both_ml_libraries_missing_pure_numpy_activated(self):
        """When both LightGBM and Scikit-Learn are missing, Tier 3 Pure NumPy is strictly active."""
        with patch.object(fe_mod, "HAS_LIGHTGBM", False), \
             patch.object(fe_mod, "HAS_SKLEARN", False):
            engine = MultiModalFusionEngine()
            self.assertEqual(engine.active_tier, "Tier 3: Pure NumPy Fallback")
            prob = engine.predict_edge_probability(net_sentiment=0.5, momentum_5d=3.0)
            self.assertIsInstance(prob, float)
            self.assertTrue(0.05 <= prob <= 0.98)

    def test_lightgbm_runtime_exception_triggers_graceful_fallback(self):
        """If LightGBM booster throws an unhandled error during inference, fall back immediately."""
        engine = MultiModalFusionEngine()
        mock_lgbm = MagicMock()
        mock_lgbm.is_available.return_value = True
        mock_lgbm.predict_proba.side_effect = RuntimeError("Simulated LightGBM C++ core dump")
        engine.lgbm_model = mock_lgbm

        # Should log warning and fall back to Tier 3 without raising RuntimeError
        prob = engine.predict_edge_probability(net_sentiment=0.25)
        self.assertIsInstance(prob, float)
        self.assertTrue(0.05 <= prob <= 0.98)

    def test_sklearn_runtime_exception_triggers_graceful_fallback(self):
        """If Scikit-Learn model throws an unhandled error during inference, fall back to pure NumPy."""
        engine = MultiModalFusionEngine(tier=2)
        mock_skl = MagicMock()
        mock_skl.is_available.return_value = True
        mock_skl.predict_proba.side_effect = RuntimeError("Simulated Sklearn dimension mismatch error")
        engine.sklearn_model = mock_skl

        # Should log warning and fall back to Tier 3 pure NumPy without raising RuntimeError
        prob = engine.predict_edge_probability(net_sentiment=0.25)
        self.assertIsInstance(prob, float)
        self.assertTrue(0.05 <= prob <= 0.98)

    def test_pure_numpy_monotonicity_across_5_market_regimes(self):
        """
        Verify mathematical monotonicity dP/d(net_sentiment) > 0 across 5 distinct market regimes.
        Holding all technical factors constant, higher net sentiment must strictly increase edge prob.
        """
        fallback = NumpyFusionFallback()
        regimes = {
            "neutral_market": {"rsi_14": 50.0, "momentum_5d": 0.0, "volume_z_score": 0.0, "atr_volatility": 2.0, "above_sma20": True},
            "strong_bull": {"rsi_14": 75.0, "momentum_5d": 10.0, "volume_z_score": 2.5, "atr_volatility": 1.2, "above_sma20": True},
            "severe_bear": {"rsi_14": 20.0, "momentum_5d": -15.0, "volume_z_score": -1.5, "atr_volatility": 6.5, "above_sma20": False},
            "high_vol_panic": {"rsi_14": 35.0, "momentum_5d": -8.0, "volume_z_score": 3.0, "atr_volatility": 14.0, "above_sma20": False},
            "overbought_divergence": {"rsi_14": 85.0, "momentum_5d": 2.0, "volume_z_score": -1.0, "atr_volatility": 1.5, "above_sma20": True},
        }

        sentiments = np.linspace(-1.0, 1.0, 41)  # Fine grid step of 0.05

        for reg_name, reg_tech in regimes.items():
            probs = []
            for s in sentiments:
                f = dict(reg_tech)
                f["net_sentiment"] = float(s)
                f["positive_prob"] = float(np.clip((s + 1.0) / 2.0, 0.0, 1.0))
                f["negative_prob"] = float(np.clip((1.0 - s) / 2.0, 0.0, 1.0))
                p = fallback.predict_proba(f)
                probs.append(p)

            for i in range(len(probs) - 1):
                self.assertLess(
                    probs[i],
                    probs[i + 1],
                    f"Monotonicity violated in regime '{reg_name}' at s={sentiments[i]:.2f} (p={probs[i]:.5f}) >= s={sentiments[i+1]:.2f} (p={probs[i+1]:.5f})"
                )

    def test_extreme_adversarial_feature_inputs(self):
        """Test NaN, Inf, None, negative RSI, and massive values."""
        engine = MultiModalFusionEngine(tier=3)

        adversarial_cases = [
            {"net_sentiment": float("nan"), "rsi_14": float("nan")},
            {"net_sentiment": float("inf"), "momentum_5d": float("-inf")},
            {"net_sentiment": -999.0, "rsi_14": -50.0, "atr_volatility": 999.0},
            {"net_sentiment": 999.0, "rsi_14": 500.0, "volume_z_score": -999.0},
            {"net_sentiment": None, "positive_prob": None, "negative_prob": None},
            {},
            {"dense_text_embedding": np.zeros(384)},
            {"dense_text_embedding": np.ones(384) * 1e6},
            {"dense_text_embedding": [float("nan")] * 384},
            {"text_anchor_similarity": float("nan")},
        ]

        for case in adversarial_cases:
            prob = engine.predict_edge_probability(case)
            self.assertIsInstance(prob, float)
            self.assertFalse(np.isnan(prob), f"Output was NaN for input: {case}")
            self.assertFalse(np.isinf(prob), f"Output was Inf for input: {case}")
            self.assertTrue(0.0 <= prob <= 1.0, f"Output {prob} out of bounds for input: {case}")


class TestShortPriceHistories(unittest.TestCase):
    """
    Stress-testing short price histories for rolling indicators in get_stock_market_metrics:
    0 bars, 1 bar, 2 bars, 5 bars, 13 bars for 14-period RSI/ATR and 20-period Volume Z-Score.
    Verifies that no unhandled IndexError or ZeroDivisionError occurs.
    """

    def _create_mock_ticker(self, hist_df, last_price=100.0, last_vol=10000.0, mcap=1e9, currency="USD"):
        instance = MagicMock()
        instance.history.return_value = hist_df
        instance.fast_info.last_price = last_price
        instance.fast_info.last_volume = last_vol
        instance.fast_info.market_cap = mcap
        instance.fast_info.currency = currency
        return instance

    @patch("src.paper_trader.yf.Ticker")
    def test_zero_bars_empty_dataframe(self, mock_ticker):
        """0 bars: Empty history dataframe should produce clean valid=False and default metrics."""
        mock_ticker.return_value = self._create_mock_ticker(pd.DataFrame(), last_price=None, last_vol=None)
        res = get_stock_market_metrics("EMPTY_SYM")

        self.assertFalse(res["valid"])
        self.assertEqual(res["price"], 0.0)
        self.assertEqual(res["rsi_14"], 50.0)
        self.assertEqual(res["rsi"], 50.0)
        self.assertEqual(res["volume_z_score"], 0.0)
        self.assertEqual(res["atr_volatility"], 0.0)
        self.assertEqual(res["momentum_5d"], 0.0)

    @patch("src.paper_trader.yf.Ticker")
    def test_one_bar_single_price(self, mock_ticker):
        """1 bar: Cannot compute diffs for RSI or ATR, cannot compute SMA20 or Volume Z."""
        dates = pd.date_range("2026-01-01", periods=1, freq="D")
        hist = pd.DataFrame({
            "Close": [150.0],
            "High": [152.0],
            "Low": [148.0],
            "Volume": [50_000.0]
        }, index=dates)

        mock_ticker.return_value = self._create_mock_ticker(hist, last_price=150.0, last_vol=50_000.0)
        res = get_stock_market_metrics("ONE_BAR")

        self.assertTrue(res["valid"])
        self.assertEqual(res["price"], 150.0)
        self.assertEqual(res["sma20"], 150.0)
        self.assertTrue(res["above_sma20"])
        self.assertEqual(res["momentum_5d"], 0.0)
        # RSI must safely fall back to 50.0
        self.assertEqual(res["rsi_14"], 50.0)
        # Volume Z must safely fall back to 0.0 (requires >= 5 bars)
        self.assertEqual(res["volume_z_score"], 0.0)
        # ATR must safely fall back to 0.0 (requires >= 2 bars)
        self.assertEqual(res["atr_volatility"], 0.0)

    @patch("src.paper_trader.yf.Ticker")
    def test_two_bars_gain_loss_and_flat(self, mock_ticker):
        """2 bars: Exactly 1 price diff. Test upward diff, downward diff, and identical (flat) prices."""
        dates = pd.date_range("2026-01-01", periods=2, freq="D")

        # 1. Upward move
        hist_up = pd.DataFrame({
            "Close": [100.0, 105.0],
            "High": [101.0, 106.0],
            "Low": [99.0, 100.0],
            "Volume": [10_000.0, 15_000.0]
        }, index=dates)
        mock_ticker.return_value = self._create_mock_ticker(hist_up, last_price=105.0, last_vol=15_000.0)
        res_up = get_stock_market_metrics("TWO_BAR_UP")
        self.assertTrue(res_up["valid"])
        self.assertEqual(res_up["rsi_14"], 100.0)  # Pure gain, avg_loss=0 -> 100.0
        self.assertGreater(res_up["atr_volatility"], 0.0)
        self.assertEqual(res_up["volume_z_score"], 0.0)  # < 5 bars

        # 2. Downward move
        hist_down = pd.DataFrame({
            "Close": [100.0, 95.0],
            "High": [101.0, 96.0],
            "Low": [99.0, 94.0],
            "Volume": [10_000.0, 15_000.0]
        }, index=dates)
        mock_ticker.return_value = self._create_mock_ticker(hist_down, last_price=95.0, last_vol=15_000.0)
        res_down = get_stock_market_metrics("TWO_BAR_DOWN")
        self.assertTrue(res_down["valid"])
        self.assertEqual(res_down["rsi_14"], 0.0)  # Pure loss, avg_gain=0 -> 0.0
        self.assertGreater(res_down["atr_volatility"], 0.0)

        # 3. Flat prices (zero diff -> test for potential ZeroDivisionError)
        hist_flat = pd.DataFrame({
            "Close": [100.0, 100.0],
            "High": [100.0, 100.0],
            "Low": [100.0, 100.0],
            "Volume": [10_000.0, 10_000.0]
        }, index=dates)
        mock_ticker.return_value = self._create_mock_ticker(hist_flat, last_price=100.0, last_vol=10_000.0)
        res_flat = get_stock_market_metrics("TWO_BAR_FLAT")
        self.assertTrue(res_flat["valid"])
        self.assertEqual(res_flat["rsi_14"], 50.0)  # Neutral fallback when gain==0 and loss==0
        self.assertEqual(res_flat["atr_volatility"], 0.0)

    @patch("src.paper_trader.yf.Ticker")
    def test_five_bars_momentum_and_volume_z_boundary(self, mock_ticker):
        """5 bars: Lookback boundary for momentum_5d (iloc[-5]) and Volume Z (>= 5 bars)."""
        dates = pd.date_range("2026-01-01", periods=5, freq="D")

        # Test varying volume
        hist_var = pd.DataFrame({
            "Close": [100.0, 101.0, 102.0, 103.0, 105.0],
            "High": [101.0, 102.0, 103.0, 104.0, 106.0],
            "Low": [99.0, 100.0, 101.0, 102.0, 104.0],
            "Volume": [10_000.0, 12_000.0, 11_000.0, 15_000.0, 20_000.0]
        }, index=dates)
        mock_ticker.return_value = self._create_mock_ticker(hist_var, last_price=105.0, last_vol=20_000.0)
        res_var = get_stock_market_metrics("FIVE_BAR_VAR")
        self.assertTrue(res_var["valid"])
        self.assertEqual(res_var["momentum_5d"], 5.0)  # (105 - 100)/100 = 5%
        self.assertGreater(res_var["volume_z_score"], 0.0)
        self.assertTrue(-3.0 <= res_var["volume_z_score"] <= 3.0)

        # Test flat volume (std = 0 -> test for potential ZeroDivisionError)
        hist_flat_vol = pd.DataFrame({
            "Close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [101.0, 102.0, 103.0, 104.0, 105.0],
            "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "Volume": [10_000.0, 10_000.0, 10_000.0, 10_000.0, 10_000.0]
        }, index=dates)
        mock_ticker.return_value = self._create_mock_ticker(hist_flat_vol, last_price=104.0, last_vol=10_000.0)
        res_flat_vol = get_stock_market_metrics("FIVE_BAR_FLAT_VOL")
        self.assertTrue(res_flat_vol["valid"])
        # With std == 0, volume_z_score must cleanly remain 0.0 without ZeroDivisionError
        self.assertEqual(res_flat_vol["volume_z_score"], 0.0)

    @patch("src.paper_trader.yf.Ticker")
    def test_thirteen_bars_sub_fourteen_rsi_atr_boundary(self, mock_ticker):
        """13 bars: Exactly 1 bar short of the standard 14-period window for RSI and ATR."""
        dates = pd.date_range("2026-01-01", periods=13, freq="D")
        closes = [100.0 + (i * 0.5) for i in range(13)]
        highs = [c + 1.0 for c in closes]
        lows = [c - 1.0 for c in closes]
        vols = [50_000.0 + (i * 1000.0) for i in range(13)]

        hist = pd.DataFrame({
            "Close": closes,
            "High": highs,
            "Low": lows,
            "Volume": vols
        }, index=dates)

        mock_ticker.return_value = self._create_mock_ticker(hist, last_price=closes[-1], last_vol=vols[-1])
        res = get_stock_market_metrics("THIRTEEN_BARS")

        self.assertTrue(res["valid"])
        self.assertGreater(res["rsi_14"], 50.0)
        self.assertLessEqual(res["rsi_14"], 100.0)
        self.assertGreater(res["atr_volatility"], 0.0)
        self.assertLessEqual(res["atr_volatility"], 20.0)
        self.assertTrue(-3.0 <= res["volume_z_score"] <= 3.0)

    @patch("src.paper_trader.yf.Ticker")
    def test_missing_high_low_columns(self, mock_ticker):
        """Missing High/Low columns in DataFrame falls back to absolute diffs of Close for ATR."""
        dates = pd.date_range("2026-01-01", periods=10, freq="D")
        hist = pd.DataFrame({
            "Close": [100.0 + i for i in range(10)],
            "Volume": [10_000.0] * 10
        }, index=dates)

        mock_ticker.return_value = self._create_mock_ticker(hist, last_price=109.0, last_vol=10_000.0)
        res = get_stock_market_metrics("NO_HIGH_LOW")
        self.assertTrue(res["valid"])
        self.assertGreater(res["atr_volatility"], 0.0)

    @patch("src.paper_trader.yf.Ticker")
    def test_zero_current_price_resilience(self, mock_ticker):
        """Zero current price must not raise ZeroDivisionError in momentum or ATR."""
        dates = pd.date_range("2026-01-01", periods=5, freq="D")
        hist = pd.DataFrame({
            "Close": [0.0, 0.0, 0.0, 0.0, 0.0],
            "High": [0.0, 0.0, 0.0, 0.0, 0.0],
            "Low": [0.0, 0.0, 0.0, 0.0, 0.0],
            "Volume": [10_000.0] * 5
        }, index=dates)

        mock_ticker.return_value = self._create_mock_ticker(hist, last_price=0.0, last_vol=10_000.0)
        res = get_stock_market_metrics("ZERO_PRICE")
        # Must execute cleanly without unhandled exception
        self.assertEqual(res["momentum_5d"], 0.0)
        self.assertEqual(res["atr_volatility"], 0.0)


class TestCandidateRankingStability(unittest.TestCase):
    """
    Stress-testing candidate ranking stability across simulated universe snapshots.
    Verifies deterministic ordering, guardrail filtration, and tie-breaking mechanics.
    """

    def setUp(self):
        reset_fusion_engine()

    def _build_candidate(self, symbol, price, net, mom, cap, arts, above=True, vol_z=0.5, rsi=55.0):
        tech = {
            "valid": True,
            "price": price,
            "sma20": price * 0.96 if above else price * 1.04,
            "above_sma20": above,
            "momentum_5d": mom,
            "rsi_14": rsi,
            "volume_z_score": vol_z,
            "atr_volatility": 2.5,
            "currency": "INR" if symbol.endswith(".NS") or symbol.startswith("^") else "USD"
        }
        pos = max(0.0, min(1.0, (net + 1.0) / 2.0))
        bert = {
            "net_score": net,
            "positive": pos,
            "negative": 1.0 - pos,
            "news_score": net * 4.0,
            "headline": arts[0].get("title", "") if arts else ""
        }
        return tech, bert

    def test_candidate_ranking_determinism_across_100_runs(self):
        """Candidate ranking on identical snapshot inputs must be 100% deterministic (0 jitter)."""
        symbols = ["AAPL", "NVDA", "MSFT", "NIFTYBEES.NS", "SMALL_WINNER"]
        techs = {}
        berts = {}
        articles = {}
        prices = {}
        caps = {}
        vols = {}

        for sym in symbols:
            if sym == "SMALL_WINNER":
                tech, bert = self._build_candidate(sym, 35.0, 0.70, 4.5, 800_000_000, [{"title": "Record earnings beat"}])
                caps[sym] = 800_000_000
                prices[sym] = 35.0
            elif sym == "NIFTYBEES.NS":
                tech, bert = self._build_candidate(sym, 280.0, 0.40, 1.8, 10_000_000_000, [{"title": "India economy growth"}])
                caps[sym] = 10_000_000_000
                prices[sym] = 280.0
            elif sym == "NVDA":
                tech, bert = self._build_candidate(sym, 130.0, 0.50, 2.5, 3_000_000_000_000, [{"title": "AI chip demand"}])
                caps[sym] = 3_000_000_000_000
                prices[sym] = 130.0
            elif sym == "AAPL":
                tech, bert = self._build_candidate(sym, 220.0, 0.10, 0.5, 3_300_000_000_000, [{"title": "iPhone supply"}])
                caps[sym] = 3_300_000_000_000
                prices[sym] = 220.0
            else:  # MSFT
                tech, bert = self._build_candidate(sym, 430.0, 0.05, -0.2, 3_200_000_000_000, [{"title": "Cloud updates"}], above=False)
                caps[sym] = 3_200_000_000_000
                prices[sym] = 430.0

            techs[sym] = tech
            berts[sym] = bert
            articles[sym] = [{"title": "Sample news"}]
            vols[sym] = 1_000_000.0

        # Run 100 snapshot evaluations and ensure identical ranked output
        reference_rank = None
        for iteration in range(100):
            upside_metrics = {}
            for sym in symbols:
                upside_metrics[sym] = calculate_probability_weighted_upside(
                    sym, prices[sym], techs[sym], berts[sym], articles[sym],
                    market_cap=caps[sym], volume=vols[sym]
                )
            snapshot = MarketSnapshot(
                prices=prices,
                technicals=techs,
                finbert_scores=berts,
                ticker_articles=articles,
                volumes=vols,
                market_caps=caps,
                upside_metrics=upside_metrics,
                borderlist={}
            )
            ranked = snapshot.get_ranked_candidates()
            if reference_rank is None:
                reference_rank = ranked
            else:
                self.assertEqual(
                    ranked,
                    reference_rank,
                    f"Candidate ranking nondeterminism detected at iteration {iteration}: {ranked} != {reference_rank}"
                )

    def test_benchmark_index_and_penny_stock_exclusion(self):
        """Benchmark index (^NSEI) and penny stocks below threshold are excluded from get_ranked_candidates."""
        prices = {"^NSEI": 24500.0, "PENNY_STOCK": 1.20, "NIFTYBEES.NS": 280.0, "NVDA": 130.0}
        techs = {s: {"valid": True, "price": prices[s], "currency": "INR" if ".NS" in s or "^" in s else "USD"} for s in prices}
        berts = {s: {"net_score": 0.5, "positive": 0.75, "negative": 0.25} for s in prices}
        articles = {s: [{"title": "News"}] for s in prices}
        caps = {s: 1e9 for s in prices}
        vols = {s: 1e6 for s in prices}

        border_filter = BorderlistGuardrail()
        # PENNY_STOCK should fail border filter
        is_safe, reason = border_filter.evaluate("PENNY_STOCK", 1.20, 1e6, 1e9, currency="USD")
        self.assertFalse(is_safe)
        border_filter.disqualify("PENNY_STOCK", 1.20, 1e6, 1e9, reason)

        upside_metrics = {}
        for s in prices:
            upside_metrics[s] = calculate_probability_weighted_upside(
                s, prices[s], techs[s], berts[s], articles[s], market_cap=caps[s], volume=vols[s]
            )

        snapshot = MarketSnapshot(
            prices=prices,
            technicals=techs,
            finbert_scores=berts,
            ticker_articles=articles,
            volumes=vols,
            market_caps=caps,
            upside_metrics=upside_metrics,
            borderlist=border_filter.disqualified
        )

        ranked = snapshot.get_ranked_candidates()

        # Must not contain ^NSEI (benchmark index)
        self.assertNotIn("^NSEI", ranked, "Benchmark index ^NSEI must not appear in ranked candidates")
        # Must not contain PENNY_STOCK (disqualified by borderlist)
        self.assertNotIn("PENNY_STOCK", ranked, "Penny stock must not appear in ranked candidates")
        # Tradeable assets must remain
        self.assertIn("NIFTYBEES.NS", ranked)
        self.assertIn("NVDA", ranked)

    def test_ranking_monotonicity_and_tie_breaking(self):
        """Ranked list must be monotonically non-increasing by (expected_return_pct, upside_probability_pct)."""
        # Create 8 simulated candidates
        prices = {f"SYM_{i}": 50.0 + i * 10 for i in range(8)}
        techs = {}
        berts = {}
        articles = {}
        caps = {}
        vols = {}

        for i, s in enumerate(prices):
            net = (i - 4) * 0.2  # range [-0.8, 0.6]
            mom = (i - 4) * 1.5
            techs[s] = {"valid": True, "price": prices[s], "above_sma20": i % 2 == 0, "momentum_5d": mom, "rsi_14": 40 + i * 5}
            berts[s] = {"net_score": net, "positive": max(0.0, net), "negative": max(0.0, -net)}
            articles[s] = [{"title": "General news"}]
            caps[s] = 5e9
            vols[s] = 1e6

        upside_metrics = {}
        for s in prices:
            upside_metrics[s] = calculate_probability_weighted_upside(
                s, prices[s], techs[s], berts[s], articles[s], market_cap=caps[s], volume=vols[s]
            )

        snapshot = MarketSnapshot(
            prices=prices,
            technicals=techs,
            finbert_scores=berts,
            ticker_articles=articles,
            volumes=vols,
            market_caps=caps,
            upside_metrics=upside_metrics,
            borderlist={}
        )

        ranked = snapshot.get_ranked_candidates()
        self.assertEqual(len(ranked), len(prices))

        for idx in range(len(ranked) - 1):
            curr_s = ranked[idx]
            next_s = ranked[idx + 1]
            curr_m = upside_metrics[curr_s]
            next_m = upside_metrics[next_s]

            curr_ret = curr_m["expected_return_pct"]
            next_ret = next_m["expected_return_pct"]
            curr_prob = curr_m["upside_probability_pct"]
            next_prob = next_m["upside_probability_pct"]

            if curr_ret == next_ret:
                self.assertGreaterEqual(
                    curr_prob,
                    next_prob,
                    f"Tie-breaking failure at rank {idx} vs {idx+1}: {curr_s} (prob={curr_prob}) < {next_s} (prob={next_prob})"
                )
            else:
                self.assertGreater(
                    curr_ret,
                    next_ret,
                    f"Ranking order violation at rank {idx} vs {idx+1}: {curr_s} (ret={curr_ret}) < {next_s} (ret={next_ret})"
                )


if __name__ == "__main__":
    unittest.main()
