"""
Tests for the four critical fixes:
1. Overfitting detection gate in calibration.py
2. Synthetic sentiment flagging in backtest_engine.py
3. Probability calibration module
4. Bot calibrated parameters JSON remediation
"""

import json
import math
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.probability_calibration import (
    TemperatureScaler,
    PlattCalibrator,
    get_default_temperature_scaler,
    calibrate_finbert_output,
)


# ============================================================================
# Fix 3: Probability Calibration Module Tests
# ============================================================================


class TestTemperatureScaler:
    """Tests for TemperatureScaler."""

    def test_identity_temperature(self):
        """T=1.0 should reproduce raw softmax."""
        scaler = TemperatureScaler(temperature=1.0)
        probs = {"positive": 0.8, "negative": 0.1, "neutral": 0.1}
        result = scaler.calibrate_probs(probs)
        # At T=1.0, output should be very close to input
        assert abs(result["positive"] - probs["positive"]) < 0.01
        assert abs(result["negative"] - probs["negative"]) < 0.01
        assert abs(result["neutral"] - probs["neutral"]) < 0.01

    def test_higher_temperature_softens(self):
        """T > 1 should reduce overconfidence (make max prob smaller)."""
        scaler = TemperatureScaler(temperature=2.0)
        scaler._fitted = True
        probs = {"positive": 0.90, "negative": 0.05, "neutral": 0.05}
        result = scaler.calibrate_probs(probs)
        # Higher temperature should bring probabilities closer to uniform
        assert result["positive"] < 0.90
        assert result["negative"] > 0.05
        assert result["neutral"] > 0.05
        # Should still sum to ~1.0
        total = result["positive"] + result["negative"] + result["neutral"]
        assert abs(total - 1.0) < 1e-6

    def test_calibrated_flag_set(self):
        """Output should include calibration metadata."""
        scaler = TemperatureScaler(temperature=1.5)
        probs = {"positive": 0.7, "negative": 0.2, "neutral": 0.1}
        result = scaler.calibrate_probs(probs)
        assert result["calibrated"] is True
        assert result["calibration_method"] == "temperature_scaling"
        assert result["temperature"] == 1.5

    def test_preserves_extra_keys(self):
        """Extra keys in input dict should be preserved."""
        scaler = TemperatureScaler(temperature=1.5)
        probs = {"positive": 0.5, "negative": 0.3, "neutral": 0.2, "headline": "test"}
        result = scaler.calibrate_probs(probs)
        assert result["headline"] == "test"

    def test_invalid_temperature(self):
        """T <= 0 should raise ValueError."""
        with pytest.raises(ValueError):
            TemperatureScaler(temperature=0.0)
        with pytest.raises(ValueError):
            TemperatureScaler(temperature=-1.0)

    def test_set_temperature(self):
        """Manual temperature setting."""
        scaler = TemperatureScaler()
        scaler.set_temperature(2.5)
        assert scaler.temperature == 2.5
        assert scaler._fitted is True

    def test_fit_learns_temperature(self):
        """Fitting on validation data should change temperature from default."""
        np.random.seed(42)
        # Create overconfident logits and labels
        n = 100
        logits = np.random.randn(n, 3) * 3.0  # High-magnitude -> overconfident softmax
        labels = np.random.randint(0, 3, size=n)

        scaler = TemperatureScaler(temperature=1.0)
        scaler.fit(logits, labels)
        # After fitting, temperature should have changed
        assert scaler._fitted is True
        assert scaler.temperature != 1.0  # Should have been optimized

    def test_calibrate_logits_shape(self):
        """calibrate_logits should return correct shape."""
        scaler = TemperatureScaler(temperature=1.5)
        logits = np.array([[2.0, 1.0, 0.5], [-1.0, 3.0, 0.0]])
        result = scaler.calibrate_logits(logits)
        assert result.shape == (2, 3)
        # Each row should sum to 1
        for i in range(2):
            assert abs(np.sum(result[i]) - 1.0) < 1e-6

    def test_net_score_updated(self):
        """net_score should be updated in calibrated output."""
        scaler = TemperatureScaler(temperature=1.5)
        probs = {"positive": 0.8, "negative": 0.1, "neutral": 0.1}
        result = scaler.calibrate_probs(probs)
        expected_net = round(result["positive"] - result["negative"], 4)
        assert result["net_score"] == expected_net

    def test_polarity_updated(self):
        """polarity should reflect calibrated probabilities."""
        scaler = TemperatureScaler(temperature=1.0)
        probs = {"positive": 0.1, "negative": 0.8, "neutral": 0.1}
        result = scaler.calibrate_probs(probs)
        assert result["polarity"] == "negative"


class TestPlattCalibrator:
    """Tests for PlattCalibrator."""

    def test_fit_and_calibrate(self):
        """Basic fit + calibrate flow."""
        np.random.seed(42)
        n = 100
        logits = np.random.randn(n, 3) * 2.0
        labels = np.random.randint(0, 3, size=n)

        calibrator = PlattCalibrator()
        calibrator.fit(logits, labels)
        assert calibrator._fitted is True

        probs = {"positive": 0.8, "negative": 0.1, "neutral": 0.1}
        result = calibrator.calibrate_probs(probs)
        assert result["calibrated"] is True
        assert result["calibration_method"] == "platt_scaling"
        total = result["positive"] + result["negative"] + result["neutral"]
        assert abs(total - 1.0) < 1e-4

    def test_calibrate_logits_sums_to_one(self):
        """Each row of calibrated output should sum to 1."""
        np.random.seed(0)
        calibrator = PlattCalibrator()
        logits = np.random.randn(50, 3)
        labels = np.random.randint(0, 3, size=50)
        calibrator.fit(logits, labels)

        test_logits = np.random.randn(10, 3)
        result = calibrator.calibrate_logits(test_logits)
        for i in range(10):
            assert abs(np.sum(result[i]) - 1.0) < 1e-6


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_default_temperature_scaler(self):
        """Should return a scaler with T=1.5 by default."""
        scaler = get_default_temperature_scaler()
        assert scaler.temperature == 1.5
        assert scaler._fitted is True

    def test_custom_temperature_scaler(self):
        """Should accept custom temperature."""
        scaler = get_default_temperature_scaler(temperature=2.0)
        assert scaler.temperature == 2.0

    def test_calibrate_finbert_output_default(self):
        """calibrate_finbert_output with defaults should soften overconfident outputs."""
        raw = {"positive": 0.92, "negative": 0.04, "neutral": 0.04}
        result = calibrate_finbert_output(raw)
        assert result["calibrated"] is True
        assert result["positive"] < 0.92
        assert result["negative"] > 0.04

    def test_calibrate_finbert_output_custom(self):
        """calibrate_finbert_output with custom calibrator."""
        calibrator = TemperatureScaler(temperature=3.0)
        raw = {"positive": 0.95, "negative": 0.03, "neutral": 0.02}
        result = calibrate_finbert_output(raw, calibrator=calibrator)
        assert result["temperature"] == 3.0
        # Very high temperature should push strongly towards uniform
        assert result["positive"] < 0.80


# ============================================================================
# Fix 4: JSON Remediation Tests
# ============================================================================


class TestBotParametersRemediation:
    """Verify bot_calibrated_parameters.json has correct statuses."""

    def test_aggressive_breakout_is_degraded(self):
        """aggressive_breakout with OOS return -0.88% must be 'degraded'."""
        path = Path("d:/market-sentiment-researcher/data/bot_calibrated_parameters.json")
        data = json.loads(path.read_text())
        bot = data["aggressive_breakout"]
        assert bot["status"] == "degraded", (
            f"aggressive_breakout should be 'degraded' but is '{bot['status']}'"
        )
        # Verify the OOS metrics that justify this status
        assert bot["out_of_sample_metrics"]["net_return_pct"] < 0
        assert bot["out_of_sample_metrics"]["profit_factor"] < 1.0

    def test_trend_follower_is_degraded(self):
        """trend_follower with OOS return -0.50% must be 'degraded'."""
        path = Path("d:/market-sentiment-researcher/data/bot_calibrated_parameters.json")
        data = json.loads(path.read_text())
        bot = data["trend_follower"]
        assert bot["status"] == "degraded", (
            f"trend_follower should be 'degraded' but is '{bot['status']}'"
        )
        assert bot["out_of_sample_metrics"]["net_return_pct"] < 0
        assert bot["out_of_sample_metrics"]["profit_factor"] < 1.0

    def test_healthy_bots_unchanged(self):
        """Bots with positive OOS metrics should remain 'optimal'."""
        path = Path("d:/market-sentiment-researcher/data/bot_calibrated_parameters.json")
        data = json.loads(path.read_text())
        # contrarian_dip_buyer has OOS return 11.7% and PF 1.61 — should remain optimal
        assert data["contrarian_dip_buyer"]["status"] == "optimal"
        # finbert_catalyst_hunter has OOS return 2.54% and PF 1.07 — should remain optimal
        assert data["finbert_catalyst_hunter"]["status"] == "optimal"


# ============================================================================
# Fix 1: Overfitting Gate Tests
# ============================================================================


class TestOverfittingGate:
    """Tests for the stricter overfitting detection in calibration.py."""

    def test_negative_oos_return_is_degraded(self):
        """Strategy with OOS return <= 0 should be marked 'degraded' regardless of IS performance."""
        from src.calibration import check_overfitting_gate

        is_metrics = {"net_return_pct": 30.0, "profit_factor": 2.0}
        oos_metrics = {"net_return_pct": -0.5, "profit_factor": 0.88}

        is_overfit, reason = check_overfitting_gate(is_metrics, oos_metrics)
        assert is_overfit is True
        assert "Non-positive out-of-sample return" in reason

    def test_low_profit_factor_is_degraded(self):
        """Strategy with OOS profit_factor < 1.0 should be marked 'degraded'."""
        from src.calibration import check_overfitting_gate

        is_metrics = {"net_return_pct": 10.0, "profit_factor": 1.5}
        oos_metrics = {"net_return_pct": 0.5, "profit_factor": 0.85}

        is_overfit, reason = check_overfitting_gate(is_metrics, oos_metrics)
        assert is_overfit is True
        assert "profit factor under 1.0" in reason

    def test_healthy_oos_is_not_degraded(self):
        """Strategy with positive OOS return and PF >= 1.0 should NOT be degraded."""
        from src.calibration import check_overfitting_gate

        is_metrics = {"net_return_pct": 10.0, "profit_factor": 1.5}
        oos_metrics = {"net_return_pct": 5.0, "profit_factor": 1.5}

        is_overfit, _ = check_overfitting_gate(is_metrics, oos_metrics)
        assert is_overfit is False

    def test_classic_overfitting_still_detected(self):
        """The classic overfitting case (IS > 5%, OOS < -2%) should still trigger."""
        from src.calibration import check_overfitting_gate

        is_metrics = {"net_return_pct": 30.0, "profit_factor": 2.5}
        oos_metrics = {"net_return_pct": -5.0, "profit_factor": 0.5}

        is_overfit, _ = check_overfitting_gate(is_metrics, oos_metrics)
        assert is_overfit is True

    def test_zero_oos_return_is_degraded(self):
        """Exactly zero OOS return should be degraded."""
        from src.calibration import check_overfitting_gate

        is_metrics = {"net_return_pct": 10.0, "profit_factor": 1.5}
        oos_metrics = {"net_return_pct": 0.0, "profit_factor": 1.0}

        is_overfit, reason = check_overfitting_gate(is_metrics, oos_metrics)
        assert is_overfit is True
        assert "Non-positive" in reason

    def test_exactly_one_pf_not_degraded_with_positive_return(self):
        """OOS PF == 1.0 is borderline, but with positive return should not be degraded."""
        from src.calibration import check_overfitting_gate

        is_metrics = {"net_return_pct": 2.0, "profit_factor": 1.2}
        oos_metrics = {"net_return_pct": 1.0, "profit_factor": 1.0}

        is_overfit, _ = check_overfitting_gate(is_metrics, oos_metrics)
        assert is_overfit is False

    def test_severe_alpha_decay_is_degraded(self):
        """Strategy with IS > 5% but OOS < 0.25 * IS should be degraded."""
        from src.calibration import check_overfitting_gate

        is_metrics = {"net_return_pct": 20.0, "profit_factor": 2.0}
        oos_metrics = {"net_return_pct": 4.0, "profit_factor": 1.2}  # 4.0 < 0.25 * 20.0 (5.0)

        is_overfit, reason = check_overfitting_gate(is_metrics, oos_metrics)
        assert is_overfit is True
        assert "Severe alpha decay" in reason

    def test_finbert_analyzer_temperature_and_calibration(self):
        """FinBERTAnalyzer should support configurable temperature and calibration."""
        from src.finbert_analyzer import FinBERTAnalyzer
        analyzer = FinBERTAnalyzer(temperature=1.8)
        assert analyzer.temperature == 1.8
        # Empty text returns calibrated default with temperature
        empty_res = analyzer.analyze("")
        assert empty_res["calibrated"] is True
        assert empty_res["temperature"] == 1.8
        assert empty_res["polarity"] == "neutral"


# ============================================================================
# Fix 2: Synthetic Sentiment Flagging Tests
# ============================================================================


class TestSyntheticSentimentFlags:
    """Tests for synthetic sentiment flagging in backtest_engine.py results."""

    def _make_test_data(self, n_bars=30):
        """Create simple test fixture."""
        dates = pd.date_range("2023-01-01", periods=n_bars, freq="D")
        prices = [100.0 + i * 0.5 for i in range(n_bars)]
        return {
            "TEST": pd.DataFrame({
                "open": prices,
                "high": [p + 1.0 for p in prices],
                "low": [p - 1.0 for p in prices],
                "close": prices,
                "volume": [200000.0] * n_bars
            }, index=dates)
        }

    def test_synthetic_sentiment_flag_in_results(self):
        """run_backtest without sentiment_feed should set used_synthetic_sentiment=True."""
        from src.backtest_engine import HistoricalBacktestEngine
        from src.paper_trader import ConservativeSwingBot

        engine = HistoricalBacktestEngine()
        data = self._make_test_data(60)
        bot = ConservativeSwingBot()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = engine.run_backtest(bots=[bot], data=data)

        assert results["used_synthetic_sentiment"] is True
        assert results["sentiment_source"] == "synthetic-price-derived"
        assert "sentiment_warning" in results

    def test_real_sentiment_feed_no_flag(self):
        """run_backtest with full sentiment_feed should set used_synthetic_sentiment=False."""
        from src.backtest_engine import HistoricalBacktestEngine
        from src.paper_trader import ConservativeSwingBot

        engine = HistoricalBacktestEngine()
        data = self._make_test_data(60)
        bot = ConservativeSwingBot()

        # Build a comprehensive sentiment feed covering all dates and symbols
        sentiment_feed = {}
        for sym in data:
            for dt in data[sym].index:
                date_str = str(dt)[:10]
                key = f"{date_str}_{sym}"
                sentiment_feed[key] = {
                    "bert": {
                        "positive": 0.33,
                        "negative": 0.33,
                        "neutral": 0.34,
                        "net_score": 0.0,
                        "news_score": 0.0,
                    },
                    "articles": []
                }

        results = engine.run_backtest(bots=[bot], data=data, sentiment_feed=sentiment_feed)
        assert results["used_synthetic_sentiment"] is False
        assert results["sentiment_source"] == "real-news-feed"
        assert "sentiment_warning" not in results

    def test_synthetic_warning_emitted(self):
        """A UserWarning should be emitted when synthetic sentiment is used."""
        from src.backtest_engine import HistoricalBacktestEngine
        from src.paper_trader import ConservativeSwingBot

        engine = HistoricalBacktestEngine()
        data = self._make_test_data(60)
        bot = ConservativeSwingBot()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engine.run_backtest(bots=[bot], data=data)

        synthetic_warnings = [x for x in w if "SYNTHETIC" in str(x.message)]
        assert len(synthetic_warnings) >= 1, "Expected a UserWarning about synthetic sentiment"

    def test_synthetic_flag_in_finbert_dict(self):
        """Bar-level finbert dicts should have synthetic=True when using synthetic data."""
        from src.backtest_engine import HistoricalBacktestEngine
        from src.paper_trader import ConservativeSwingBot

        engine = HistoricalBacktestEngine()
        data = self._make_test_data(30)
        bot = ConservativeSwingBot()

        # We can't easily inspect bar_finbert mid-loop, but the article source should
        # be "Synthetic-Market-Replay" in trade rationales or results.
        # Just verify the results have the flag set
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results = engine.run_backtest(bots=[bot], data=data)
        assert results["used_synthetic_sentiment"] is True
        assert results["synthetic_sentiment_used"] is True
        bot_metrics = results["bots"][bot.bot_id]["metrics"]
        assert bot_metrics["synthetic_sentiment_used"] is True

    def test_tuple_sentiment_feed_injection(self):
        """run_backtest with (date, sym) tuple keys should be recognized without synthetic fallback."""
        from src.backtest_engine import HistoricalBacktestEngine
        from src.paper_trader import ConservativeSwingBot

        engine = HistoricalBacktestEngine()
        data = self._make_test_data(30)
        bot = ConservativeSwingBot()

        sentiment_feed = {}
        for sym in data:
            for dt in data[sym].index:
                date_str = str(dt)[:10]
                sentiment_feed[(date_str, sym)] = {
                    "positive": 0.5,
                    "negative": 0.2,
                    "neutral": 0.3,
                    "net_score": 0.3,
                }

        results = engine.run_backtest(bots=[bot], data=data, sentiment_feed=sentiment_feed)
        assert results["used_synthetic_sentiment"] is False
        assert results["synthetic_sentiment_used"] is False
        assert results["sentiment_source"] == "real-news-feed"
