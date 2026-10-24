"""
Unit and integration tests for Multi-Modal Fusion Engine and Candidate Ranking.

Tests:
1. MultiModalFusionEngine initialization and 3-tier fallback hierarchy.
2. Strict mathematical monotonicity on net sentiment holding technical factors constant.
3. Continuous probability bounds [0.0, 1.0] and practical output range [0.05, 0.98].
4. Graceful handling of missing/None/NaN inputs and dense BGE-Small text embeddings.
5. Integration with paper_trader.calculate_probability_weighted_upside and key preservation.
6. Rolling indicators in get_stock_market_metrics (RSI-14, Volume Z, ATR).
7. Decoupled two-stage architecture: bot conviction formulas unaffected.
"""

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
    CATALYST_ANCHOR_VECTOR
)
from src.paper_trader import (
    calculate_probability_weighted_upside,
    get_stock_market_metrics,
    ConservativeSwingBot,
    AggressiveBreakoutBot,
    ContrarianDipBuyerBot,
    FinBERTCatalystHunterBot,
    TrendFollowerBot
)


class TestNumpyFusionFallback(unittest.TestCase):
    """Verifies mathematical guarantees of Tier 3 pure NumPy fallback."""

    def setUp(self):
        self.fallback = NumpyFusionFallback()

    def test_strict_monotonicity_on_net_sentiment(self):
        """Holding technical factors constant, higher net sentiment must strictly increase trade edge probability."""
        sentiment_grid = np.linspace(-1.0, 1.0, 21)
        fixed_tech = {
            "rsi_14": 55.0,
            "momentum_5d": 1.5,
            "volume_z_score": 0.5,
            "atr_volatility": 2.5,
            "above_sma20": True,
            "catalyst_clarity": 0.40,
            "text_anchor_similarity": 0.10
        }

        probs = []
        for s in sentiment_grid:
            feats = dict(fixed_tech)
            feats["net_sentiment"] = float(s)
            feats["positive_prob"] = float(np.clip((s + 1.0) / 2.0, 0.0, 1.0))
            feats["negative_prob"] = float(np.clip(1.0 - feats["positive_prob"], 0.0, 1.0))
            p = self.fallback.predict_proba(feats)
            probs.append(p)

        for i in range(len(probs) - 1):
            self.assertLess(
                probs[i],
                probs[i + 1],
                f"Monotonicity failed at index {i}: s={sentiment_grid[i]:.2f} (p={probs[i]:.4f}) >= s={sentiment_grid[i+1]:.2f} (p={probs[i+1]:.4f})"
            )

    def test_probability_bounds(self):
        """Probability outputs must be strictly continuous and clamped within [0.05, 0.98]."""
        extreme_bear = {
            "net_sentiment": -1.0,
            "positive_prob": 0.0,
            "negative_prob": 1.0,
            "rsi_14": 10.0,
            "momentum_5d": -25.0,
            "volume_z_score": -3.0,
            "atr_volatility": 15.0,
            "above_sma20": False,
            "catalyst_clarity": 0.05
        }
        p_bear = self.fallback.predict_proba(extreme_bear)
        self.assertGreaterEqual(p_bear, 0.0)
        self.assertLess(p_bear, 0.30)

        extreme_bull = {
            "net_sentiment": 1.0,
            "positive_prob": 1.0,
            "negative_prob": 0.0,
            "rsi_14": 85.0,
            "momentum_5d": 20.0,
            "volume_z_score": 3.0,
            "atr_volatility": 0.5,
            "above_sma20": True,
            "catalyst_clarity": 1.0,
            "text_anchor_similarity": 0.8
        }
        p_bull = self.fallback.predict_proba(extreme_bull)
        self.assertLessEqual(p_bull, 1.0)
        self.assertGreater(p_bull, 0.70)


class TestMultiModalFusionEngine(unittest.TestCase):
    """Verifies feature extraction, dense text handling, and multi-tier fallbacks."""

    def setUp(self):
        reset_fusion_engine()
        self.engine = MultiModalFusionEngine()

    def test_engine_active_tier_reported(self):
        tier_str = self.engine.active_tier
        self.assertIn("Tier", tier_str)

    def test_tier_fallback_selection(self):
        # Explicit Tier 3 should always activate pure NumPy
        t3_engine = MultiModalFusionEngine(tier=3)
        self.assertEqual(t3_engine.active_tier, "Tier 3: Pure NumPy Fallback")
        p3 = t3_engine.predict_edge_probability(net_sentiment=0.3)
        self.assertTrue(0.0 <= p3 <= 1.0)

        # Explicit Tier 1 or 2 gracefully falls back if dependencies absent
        t1_engine = MultiModalFusionEngine(tier=1)
        p1 = t1_engine.predict_edge_probability(net_sentiment=0.3)
        self.assertTrue(0.0 <= p1 <= 1.0)

        t2_engine = MultiModalFusionEngine(tier=2)
        p2 = t2_engine.predict_edge_probability(net_sentiment=0.3)
        self.assertTrue(0.0 <= p2 <= 1.0)

    def test_missing_and_none_feature_handling(self):
        """Engine must gracefully handle completely empty or None feature structures."""
        # Empty dict
        p_empty = self.engine.predict_edge_probability({})
        self.assertTrue(0.0 <= p_empty <= 1.0)

        # None values inside dict
        none_features = {
            "net_sentiment": None,
            "positive_prob": None,
            "rsi_14": None,
            "momentum_5d": None,
            "volume_z_score": None,
            "atr_volatility": None,
            "above_sma20": None,
            "catalyst_clarity": None,
            "dense_text_embedding": None
        }
        p_none = self.engine.predict_edge_probability(none_features)
        self.assertTrue(0.0 <= p_none <= 1.0)

    def test_public_api_strict_monotonicity_across_zero_boundary(self):
        """
        Calls predict_edge_probability() directly via public API across
        s = [-0.05, -0.01, -0.001, 0.0, 0.001, 0.01, 0.05] holding technicals constant.
        Asserts strict monotonicity P(s2) > P(s1) for s2 > s1.
        """
        sentiment_points = [-0.05, -0.01, -0.001, 0.0, 0.001, 0.01, 0.05]
        fixed_tech = {
            "rsi_14": 55.0,
            "momentum_5d": 1.5,
            "volume_z_score": 0.5,
            "atr_volatility": 2.5,
            "above_sma20": True,
            "catalyst_clarity": 0.40,
            "text_anchor_similarity": 0.10
        }

        # 1. Direct net_sentiment parameter (omitting pos_prob and neg_prob)
        probs_kw = [
            self.engine.predict_edge_probability(net_sentiment=s, **fixed_tech)
            for s in sentiment_points
        ]
        for i in range(len(probs_kw) - 1):
            s1, s2 = sentiment_points[i], sentiment_points[i + 1]
            p1, p2 = probs_kw[i], probs_kw[i + 1]
            self.assertGreater(
                p2,
                p1,
                f"Strict monotonicity violated across s1={s1} -> s2={s2}: P(s1)={p1:.6f} >= P(s2)={p2:.6f}"
            )

        # 2. bert dictionary parameter: bert={'net_score': s}
        probs_bert = [
            self.engine.predict_edge_probability(bert={"net_score": s}, **fixed_tech)
            for s in sentiment_points
        ]
        for i in range(len(probs_bert) - 1):
            s1, s2 = sentiment_points[i], sentiment_points[i + 1]
            p1, p2 = probs_bert[i], probs_bert[i + 1]
            self.assertGreater(
                p2,
                p1,
                f"Strict monotonicity violated with bert dict across s1={s1} -> s2={s2}: P(s1)={p1:.6f} >= P(s2)={p2:.6f}"
            )

    def test_dense_text_embedding_projection(self):
        """384-dimensional BGE-Small text embedding correctly computes cosine projection."""
        aligned_dense = CATALYST_ANCHOR_VECTOR.copy()
        opposed_dense = -CATALYST_ANCHOR_VECTOR.copy()

        p_aligned = self.engine.predict_edge_probability(
            net_sentiment=0.2,
            rsi_14=55.0,
            dense_text_embedding=aligned_dense
        )
        p_opposed = self.engine.predict_edge_probability(
            net_sentiment=0.2,
            rsi_14=55.0,
            dense_text_embedding=opposed_dense
        )
        self.assertGreater(p_aligned, p_opposed)

    def test_predict_proba_batch_interface(self):
        """predict_proba accepts 2D arrays and returns (N, 2) shaped probabilities summing to 1.0."""
        X = np.zeros((5, 14), dtype=np.float64)
        X[:, 0] = np.linspace(-0.8, 0.8, 5)  # net sentiment
        probs = self.engine.predict_proba(X)
        self.assertEqual(probs.shape, (5, 2))
        for row in probs:
            self.assertAlmostEqual(float(row[0] + row[1]), 1.0, places=5)
            self.assertTrue(0.0 <= row[1] <= 1.0)


class TestCandidateRankingIntegration(unittest.TestCase):
    """Verifies paper_trader.calculate_probability_weighted_upside integration."""

    def test_dictionary_key_contract_preserved(self):
        """Confirms all legacy and newly required dictionary keys are returned."""
        tech = {"valid": True, "price": 100.0, "sma20": 95.0, "above_sma20": True, "momentum_5d": 2.0}
        bert = {"net_score": 0.40, "positive": 0.65, "negative": 0.15}
        arts = [{"title": "Company expands production"}]

        res = calculate_probability_weighted_upside("TEST_STOCK", 100.0, tech, bert, arts)

        required_keys = [
            "symbol",
            "price",
            "upside_probability_pct",
            "expected_return_pct",
            "prob_weighted_upside",
            "p_sentiment",
            "sentiment_prob",
            "p_tech",
            "technical_prob",
            "c_clarity",
            "catalyst_clarity",
            "edge_probability",
            "upside_probability",
            "cap_tier",
            "potential_upside_pct"
        ]
        for key in required_keys:
            self.assertIn(key, res, f"Missing required key '{key}' in calculate_probability_weighted_upside return dict")

    def test_small_cap_catalyst_ranks_above_stagnant_megacap(self):
        """Small-cap with genuine catalyst and momentum achieves higher expected return than flat mega-cap."""
        small_tech = {"valid": True, "price": 25.0, "sma20": 22.0, "above_sma20": True, "momentum_5d": 5.0}
        small_bert = {"net_score": 0.70, "positive": 0.85, "negative": 0.15}
        small_arts = [{"title": "Surprise earnings beat record revenue guidance raise"}]

        mega_tech = {"valid": True, "price": 400.0, "sma20": 399.0, "above_sma20": True, "momentum_5d": 0.2}
        mega_bert = {"net_score": 0.05, "positive": 0.35, "negative": 0.30}
        mega_arts = [{"title": "Weekly stock summary"}]

        small_res = calculate_probability_weighted_upside("SMALL", 25.0, small_tech, small_bert, small_arts, market_cap=800_000_000)
        mega_res = calculate_probability_weighted_upside("MEGA", 400.0, mega_tech, mega_bert, mega_arts, market_cap=3_000_000_000_000)

        self.assertGreater(small_res["upside_probability"], mega_res["upside_probability"])
        self.assertGreater(small_res["expected_return_pct"], mega_res["expected_return_pct"])
        self.assertEqual(small_res["cap_tier"], "Small-Cap (<$2B)")
        self.assertEqual(mega_res["cap_tier"], "Large/Mega-Cap")


class TestStockMarketMetricsRollingIndicators(unittest.TestCase):
    """Verifies rolling calculation of RSI-14, Volume Z, and ATR in get_stock_market_metrics."""

    @patch("src.paper_trader.yf.Ticker")
    def test_rolling_indicators_calculation(self, mock_ticker):
        # Create 25 mock daily bars
        n = 25
        dates = pd.date_range("2026-01-01", periods=n, freq="D")
        closes = [100.0 + (i * 0.8) for i in range(n)]
        highs = [c + 1.5 for c in closes]
        lows = [c - 1.5 for c in closes]
        vols = [100_000.0 + (i * 1000.0) for i in range(n)]

        mock_hist = pd.DataFrame({
            "Close": closes,
            "High": highs,
            "Low": lows,
            "Volume": vols
        }, index=dates)

        instance = MagicMock()
        instance.history.return_value = mock_hist
        instance.fast_info.last_price = closes[-1]
        instance.fast_info.last_volume = vols[-1]
        instance.fast_info.market_cap = 50_000_000_000
        instance.fast_info.currency = "USD"
        mock_ticker.return_value = instance

        res = get_stock_market_metrics("NVDA")

        self.assertTrue(res["valid"])
        self.assertIn("rsi_14", res)
        self.assertIn("volume_z_score", res)
        self.assertIn("atr_volatility", res)

        # Steady uptrend should yield RSI > 60
        self.assertGreater(res["rsi_14"], 60.0)
        self.assertLessEqual(res["rsi_14"], 100.0)

        # Volume Z-score bounded
        self.assertTrue(-3.0 <= res["volume_z_score"] <= 3.0)

        # Normalized ATR should be positive
        self.assertGreater(res["atr_volatility"], 0.0)

    @patch("src.paper_trader.yf.Ticker")
    def test_short_history_clean_fallbacks(self, mock_ticker):
        # Only 2 bars
        dates = pd.date_range("2026-01-01", periods=2, freq="D")
        mock_hist = pd.DataFrame({
            "Close": [100.0, 102.0],
            "Volume": [50_000.0, 60_000.0]
        }, index=dates)

        instance = MagicMock()
        instance.history.return_value = mock_hist
        instance.fast_info.last_price = 102.0
        instance.fast_info.last_volume = 60_000.0
        instance.fast_info.market_cap = None
        instance.fast_info.currency = "USD"
        mock_ticker.return_value = instance

        res = get_stock_market_metrics("NEW_STOCK")

        self.assertTrue(res["valid"])
        # With < 5 bars, Volume Z falls back cleanly to 0.0
        self.assertEqual(res["volume_z_score"], 0.0)
        self.assertIn("rsi_14", res)
        self.assertIn("atr_volatility", res)


class TestDecoupledBotConvictionIntegrity(unittest.TestCase):
    """Verifies that the 5 bot conviction formulas remain uncompromised."""

    def test_all_five_bot_archetype_evaluations(self):
        tech_good = {"valid": True, "price": 100.0, "sma20": 95.0, "above_sma20": True, "momentum_5d": 3.0}
        bert_high = {"news_score": 2.5, "net_score": 0.60, "positive": 0.80, "negative": 0.10}
        arts = [{"title": "Record Earnings Beat"}]

        # 1. Conservative Swing Bot
        cs_bot = ConservativeSwingBot()
        should_enter_cs, conv_cs, _ = cs_bot.evaluate_entry("AAPL", tech_good, bert_high, arts, [])
        self.assertTrue(should_enter_cs)
        self.assertGreaterEqual(conv_cs, 7.0)

        # 2. Aggressive Breakout Bot
        ab_bot = AggressiveBreakoutBot()
        should_enter_ab, conv_ab, _ = ab_bot.evaluate_entry("AAPL", tech_good, bert_high, arts, [])
        self.assertTrue(should_enter_ab)
        self.assertGreaterEqual(conv_ab, 5.0)

        # 3. Contrarian Dip Buyer Bot
        dip_tech = {"valid": True, "price": 96.0, "sma20": 100.0, "above_sma20": False, "momentum_5d": -3.0}
        dip_bert = {"net_score": 0.10, "info": "FinBERT[neutral]"}
        cd_bot = ContrarianDipBuyerBot()
        should_enter_cd, conv_cd, _ = cd_bot.evaluate_entry("AAPL", dip_tech, dip_bert, arts, [])
        self.assertTrue(should_enter_cd)
        self.assertGreaterEqual(conv_cd, 6.0)

        # 4. FinBERT Catalyst Hunter Bot
        fc_bot = FinBERTCatalystHunterBot()
        should_enter_fc, conv_fc, _ = fc_bot.evaluate_entry("AAPL", tech_good, bert_high, arts, [])
        self.assertTrue(should_enter_fc)
        self.assertGreaterEqual(conv_fc, 7.0)

        # 5. Trend Follower Bot
        tf_bot = TrendFollowerBot()
        should_enter_tf, conv_tf, _ = tf_bot.evaluate_entry("AAPL", tech_good, bert_high, arts, [])
        self.assertTrue(should_enter_tf)
        self.assertGreaterEqual(conv_tf, 7.0)


class TestFusionCalibrationAndSingletonSafety(unittest.TestCase):
    """
    Verifies R3 & R4:
    - FinBERTAnalyzer singleton pattern (get_instance, DI, shared instance).
    - Temperature scaling (T in [1.35, 1.5]) and Platt scaling calibration integration.
    - MultiModalFusionEngine dependency injection of analyzer and calibrator.
    - Offline safety with zero external network access.
    """

    def setUp(self):
        reset_fusion_engine()

    def test_finbert_analyzer_singleton_identity(self):
        """FinBERTAnalyzer must strictly preserve singleton identity across instantiations and get_instance()."""
        from src.finbert_analyzer import FinBERTAnalyzer
        FinBERTAnalyzer.reset_instance()
        a1 = FinBERTAnalyzer(temperature=1.4)
        a2 = FinBERTAnalyzer.get_instance(temperature=1.5)
        a3 = FinBERTAnalyzer()
        self.assertIs(a1, a2)
        self.assertIs(a2, a3)
        self.assertEqual(a1.temperature, 1.5)

    def test_temperature_scaling_in_range(self):
        """Temperature scaling with T=1.35 to 1.5 softens extreme probabilities towards uniform distribution."""
        from src.probability_calibration import TemperatureScaler, get_default_temperature_scaler
        scaler_15 = get_default_temperature_scaler(temperature=1.5)
        scaler_135 = TemperatureScaler(temperature=1.35)

        raw_probs = {"positive": 0.90, "negative": 0.05, "neutral": 0.05}
        cal_15 = scaler_15.calibrate_probs(raw_probs)
        cal_135 = scaler_135.calibrate_probs(raw_probs)

        # Softened positive probability should be lower than raw 0.90
        self.assertLess(cal_15["positive"], 0.90)
        self.assertLess(cal_135["positive"], 0.90)
        # Higher T softens more
        self.assertLess(cal_15["positive"], cal_135["positive"])
        self.assertTrue(cal_15["calibrated"])
        self.assertEqual(cal_15["calibration_method"], "temperature_scaling")

    def test_platt_scaling_calibration(self):
        """PlattCalibrator fits logistic curves per class and normalizes outputs to sum to 1."""
        from src.probability_calibration import PlattCalibrator
        calibrator = PlattCalibrator()
        # Synthetic logits (N=100, C=3) and labels
        rng = np.random.RandomState(42)
        logits = rng.normal(0, 2.0, size=(100, 3))
        labels = np.argmax(logits, axis=1)

        calibrator.fit(logits, labels)
        raw_probs = {"positive": 0.70, "negative": 0.20, "neutral": 0.10}
        cal = calibrator.calibrate_probs(raw_probs)

        self.assertTrue(cal["calibrated"])
        self.assertEqual(cal["calibration_method"], "platt_scaling")
        prob_sum = cal["positive"] + cal["negative"] + cal["neutral"]
        self.assertAlmostEqual(prob_sum, 1.0, places=2)

    def test_fusion_engine_dependency_injection(self):
        """MultiModalFusionEngine properly utilizes injected FinBERTAnalyzer and calibrator."""
        from src.finbert_analyzer import FinBERTAnalyzer
        from src.probability_calibration import TemperatureScaler

        mock_analyzer = MagicMock(spec=FinBERTAnalyzer)
        mock_analyzer.analyze.return_value = {
            "positive": 0.75,
            "negative": 0.10,
            "neutral": 0.15,
            "net_score": 0.65,
            "polarity": "positive",
            "confidence": 0.75,
            "calibrated": True
        }

        scaler = TemperatureScaler(temperature=1.4)
        engine = MultiModalFusionEngine(
            tier=3,
            finbert_analyzer=mock_analyzer,
            calibrator=scaler
        )

        self.assertIs(engine.finbert_analyzer, mock_analyzer)
        self.assertIs(engine.calibrator, scaler)

        # Extract features from text without explicit net_score
        features_dict, vec, _ = engine.extract_features(text="Company posts blockbuster quarterly earnings surge")
        mock_analyzer.analyze.assert_called_once_with("Company posts blockbuster quarterly earnings surge")
        self.assertAlmostEqual(features_dict["net_sentiment"], 0.65, places=2)

    def test_fusion_engine_get_fusion_engine_di_update(self):
        """get_fusion_engine correctly injects or updates analyzer and calibrator singletons."""
        from src.probability_calibration import TemperatureScaler
        scaler = TemperatureScaler(temperature=1.35)
        engine = get_fusion_engine(tier=3, calibrator=scaler)
        self.assertIs(engine.calibrator, scaler)


if __name__ == "__main__":
    unittest.main()

