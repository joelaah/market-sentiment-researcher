"""
Comprehensive Unit Tests for Continuous Learning Upgrades:
- Upgrade A: Automated Weekly Rolling Retraining Engine & Strict Overfitting Gate
- Upgrade B: Trade Feedback Loop (Experience Replay / Mistake Learning)
- Upgrade C: LoRA Domain Adaptation Pipeline for FinBERT
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.fusion_engine import MultiModalFusionEngine, NumpyFusionFallback, get_fusion_engine, reset_fusion_engine
from src.paper_trader import (
    BorderlistGuardrail,
    ConservativeSwingBot,
    ContrarianDipBuyerBot,
    MultiBotTournamentEngine,
    TradeFeedbackStore,
    TournamentDataManager,
)
from src.rolling_retrainer import RollingRetrainingEngine, run_rolling_retraining
from src.finbert_lora_trainer import (
    FinancialDomainDataset,
    FinBERTLoraConfig,
    FinBERTLoraTrainer,
    MockLoRAAdapter,
    adapt_finbert_lora,
)
from src.finbert_analyzer import FinBERTAnalyzer


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp(prefix="test_learning_upgrades_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_bars_data():
    """Generates synthetic 120-day OHLCV bar history across 2 symbols."""
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    rng = np.random.RandomState(42)

    data = {}
    for sym in ["TEST_US", "TEST_IN.NS"]:
        base_p = 100.0
        prices = [base_p]
        for _ in range(1, 120):
            prices.append(prices[-1] * (1.0 + rng.normal(0.001, 0.015)))

        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": prices,
                "volume": rng.uniform(250000, 500000, size=120),
            },
            index=dates,
        )
        data[sym] = df
    return data


# ============================================================================
# UPGRADE A: AUTOMATED WEEKLY ROLLING RETRAINING ENGINE TESTS
# ============================================================================

class TestUpgradeARollingRetrainer:
    """Tests for Automated Weekly Rolling Retraining Engine & Overfitting Gate."""

    def test_rolling_bars_slicing(self, temp_dir, sample_bars_data):
        """Trailing rolling window of 90 days should be sliced accurately."""
        engine = RollingRetrainingEngine(
            rolling_window_days=90,
            universe=["TEST_US", "TEST_IN.NS"],
            offline=True,
            data_dir=temp_dir,
        )
        rolling = engine.fetch_rolling_bars(full_data=sample_bars_data)
        assert "TEST_US" in rolling
        assert "TEST_IN.NS" in rolling
        assert len(rolling["TEST_US"]) == 90
        assert len(rolling["TEST_IN.NS"]) == 90
        # Verify chronological tail: last date should match full data last date
        assert rolling["TEST_US"].index[-1] == sample_bars_data["TEST_US"].index[-1]

    def test_overfitting_gate_rejects_degraded_candidate(self, temp_dir, sample_bars_data):
        """
        When a calibrated candidate is degraded (e.g. negative OOS return or PF < 1.0),
        the candidate must be REJECTED and previous optimal parameters preserved.
        """
        params_file = temp_dir / "bot_calibrated_parameters.json"
        initial_params = {
            "conservative_swing": {
                "bot_id": "conservative_swing",
                "status": "optimal",
                "parameters": {"stop_loss_pct": -5.0, "take_profit_pct": 10.0, "max_positions": 3},
                "out_of_sample_metrics": {"net_return_pct": 8.5, "profit_factor": 1.7},
            }
        }
        params_file.write_text(json.dumps(initial_params), encoding="utf-8")

        engine = RollingRetrainingEngine(
            rolling_window_days=60,
            params_file=params_file,
            data_dir=temp_dir,
        )

        # Mock WalkForwardCalibrator.calibrate to return an overfitted candidate
        with patch("src.rolling_retrainer.WalkForwardCalibrator.calibrate") as mock_cal:
            mock_cal.return_value = {
                "bot_id": "conservative_swing",
                "best_params": {"stop_loss_pct": -2.0, "take_profit_pct": 25.0, "max_positions": 10},
                "in_sample_metrics": {"net_return_pct": 45.0, "profit_factor": 3.0, "total_trades": 20},
                "out_of_sample_metrics": {"net_return_pct": -3.5, "profit_factor": 0.72, "total_trades": 8},
                "full_period_metrics": {"net_return_pct": 22.0, "profit_factor": 1.4},
                "walk_forward_score": -0.5,
            }

            res = engine.recalibrate_bots(sample_bars_data)

            # Candidate must be rejected
            bot_eval = res["bot_evaluations"]["conservative_swing"]
            assert bot_eval["action"] == "rejected"
            assert bot_eval["is_overfit"] is True
            assert "Non-positive out-of-sample return" in bot_eval["reason"] or "profit factor" in bot_eval["reason"]

            # Previous optimal parameters should be preserved in updated_params
            preserved = res["updated_params"]["conservative_swing"]
            assert preserved["parameters"]["stop_loss_pct"] == -5.0
            assert preserved["parameters"]["max_positions"] == 3
            assert preserved["status"] == "optimal"

    def test_overfitting_gate_accepts_healthy_candidate(self, temp_dir, sample_bars_data):
        """
        When a calibrated candidate has positive OOS return and PF >= 1.0,
        the candidate must be ACCEPTED and parameters updated.
        """
        params_file = temp_dir / "bot_calibrated_parameters.json"
        initial_params = {
            "conservative_swing": {
                "bot_id": "conservative_swing",
                "status": "optimal",
                "parameters": {"stop_loss_pct": -5.0, "take_profit_pct": 10.0},
            }
        }
        params_file.write_text(json.dumps(initial_params), encoding="utf-8")

        engine = RollingRetrainingEngine(
            rolling_window_days=60,
            params_file=params_file,
            data_dir=temp_dir,
        )

        with patch("src.rolling_retrainer.WalkForwardCalibrator.calibrate") as mock_cal:
            mock_cal.return_value = {
                "bot_id": "conservative_swing",
                "best_params": {"stop_loss_pct": -4.5, "take_profit_pct": 12.0},
                "in_sample_metrics": {"net_return_pct": 15.0, "profit_factor": 1.8, "total_trades": 18},
                "out_of_sample_metrics": {"net_return_pct": 6.2, "profit_factor": 1.6, "total_trades": 10},
                "full_period_metrics": {"net_return_pct": 21.2, "profit_factor": 1.7},
                "walk_forward_score": 1.25,
            }

            res = engine.recalibrate_bots(sample_bars_data)
            bot_eval = res["bot_evaluations"]["conservative_swing"]
            assert bot_eval["action"] == "accepted"
            assert bot_eval["is_overfit"] is False
            assert res["updated_params"]["conservative_swing"]["parameters"]["stop_loss_pct"] == -4.5

    def test_run_retraining_end_to_end(self, temp_dir, sample_bars_data):
        """End-to-end retraining run produces complete audit summary."""
        params_file = temp_dir / "bot_calibrated_parameters.json"
        engine = RollingRetrainingEngine(
            rolling_window_days=40,
            params_file=params_file,
            data_dir=temp_dir,
            offline=True,
        )

        # Mock calibrate to speed up test execution
        with patch("src.rolling_retrainer.WalkForwardCalibrator.calibrate") as mock_cal:
            mock_cal.return_value = {
                "bot_id": "mock_bot",
                "best_params": {"stop_loss_pct": -6.0},
                "in_sample_metrics": {"net_return_pct": 10.0, "profit_factor": 1.5, "total_trades": 10},
                "out_of_sample_metrics": {"net_return_pct": 4.0, "profit_factor": 1.3, "total_trades": 5},
                "full_period_metrics": {},
                "walk_forward_score": 0.8,
            }

            result = engine.run_retraining(full_data=sample_bars_data, save_parameters=True)
            assert result["status"] == "success"
            assert result["overfitting_gate_enforced"] is True
            assert result["rolling_window_days"] == 40
            assert "fusion_retraining" in result
            assert params_file.exists()


# ============================================================================
# UPGRADE B: TRADE FEEDBACK LOOP (EXPERIENCE REPLAY) TESTS
# ============================================================================

class TestUpgradeBTradeFeedback:
    """Tests for Trade Feedback Loop & Mistake Learning."""

    def test_trade_feedback_store_persistence(self, temp_dir):
        """TradeFeedbackStore should record feedback to JSON and Parquet."""
        store = TradeFeedbackStore(data_dir=temp_dir)
        record = {
            "trade_id": "test-uuid-1",
            "bot_id": "conservative_swing",
            "symbol": "AAPL",
            "pnl": -120.50,
            "label": 0,
            "exit_reason": "Stop Loss triggered: Cut loss at -5.0%",
            "net_sentiment": 0.35,
            "rsi_14": 62.0,
            "momentum_5d": 3.4,
            "fusion_probability": 0.74,
        }
        store.record_feedback(record)

        # JSON file must exist
        assert store.json_path.exists()
        loaded = store.load_feedback()
        assert len(loaded) == 1
        assert loaded[0]["trade_id"] == "test-uuid-1"
        assert loaded[0]["label"] == 0
        assert loaded[0]["pnl"] == -120.50

        # Parquet file should also exist if pyarrow is present
        try:
            import pyarrow
            assert store.parquet_path.exists()
        except ImportError:
            pass

    def test_paper_trader_captures_feedback_on_sell(self, temp_dir):
        """
        When MultiBotTournamentEngine sells a holding, a feedback record
        with entry snapshot features and win/loss label must be persisted.
        """
        store = TradeFeedbackStore(data_dir=temp_dir)
        engine = MultiBotTournamentEngine(
            bots=[ConservativeSwingBot()],
            client=None,
            auto_sync_params=False,
            feedback_store=store,
            data_dir=temp_dir,
        )

        bot_id = "conservative_swing"
        symbol = "NVDA"

        # Feature snapshot from entry time
        entry_snapshot = {
            "net_sentiment": 0.45,
            "positive_prob": 0.65,
            "negative_prob": 0.15,
            "neutral_prob": 0.20,
            "sentiment_logit_pos": 0.8,
            "sentiment_logit_neg": -1.2,
            "sentiment_logit_neu": -0.5,
            "rsi_14": 58.0,
            "momentum_5d": 4.2,
            "volume_z_score": 1.2,
            "atr_volatility": 2.1,
            "above_sma20": 1.0,
            "catalyst_clarity": 0.75,
            "text_anchor_similarity": 0.30,
            "fusion_probability": 0.78,
        }

        # 1. Execute Buy
        engine.execute_buy(
            bot_id=bot_id,
            symbol=symbol,
            amount_dollars=1000.0,
            rationale="Test AI Entry",
            current_price=100.0,
            volume=500000.0,
            market_cap=1e9,
            features_snapshot=entry_snapshot,
        )

        # Verify entry features are stored in open position
        positions = engine.storage.get_positions(bot_id)
        assert len(positions) == 1
        assert "entry_features" in positions[0]
        assert positions[0]["entry_features"]["net_sentiment"] == 0.45

        # 2. Execute Sell at a Loss (Stop Loss triggered) -> pnl < 0 => label=0
        engine.execute_sell(
            bot_id=bot_id,
            symbol=symbol,
            shares=positions[0]["shares"],
            rationale="Stop Loss triggered: Cut loss at -7.0%",
            current_price=93.0,
        )

        feedback = store.load_feedback()
        assert len(feedback) == 1
        fb = feedback[0]
        assert fb["bot_id"] == bot_id
        assert fb["symbol"] == symbol
        assert fb["label"] == 0  # Loss label
        assert fb["pnl"] < 0
        assert fb["net_sentiment"] == 0.45
        assert fb["fusion_probability"] == 0.78
        assert "Stop Loss" in fb["exit_reason"]

    def test_incorporate_feedback_penalizes_stopped_out_pattern(self):
        """
        The fusion engine must penalize feature patterns that led to stopped-out trades.
        After incorporating a stopped-out trade (y=0), the predicted edge probability
        for that feature pattern must STRICTLY DECREASE.
        """
        engine = MultiModalFusionEngine(tier=3)
        pattern = {
            "net_sentiment": 0.50,
            "positive_prob": 0.70,
            "negative_prob": 0.10,
            "neutral_prob": 0.20,
            "rsi_14": 65.0,
            "momentum_5d": 6.0,
            "volume_z_score": 1.5,
            "atr_volatility": 3.0,
            "above_sma20": 1.0,
            "catalyst_clarity": 0.60,
            "text_anchor_similarity": 0.20,
        }

        # Baseline edge probability before feedback
        prob_before = engine.predict_edge_probability(features=pattern)
        assert prob_before > 0.50

        # Labeled feedback from a trade that entered on this pattern and got stopped out
        stopped_out_feedback = [
            {
                **pattern,
                "label": 0,  # STOPPED OUT LOSS
                "pnl": -150.0,
                "exit_reason": "Stop Loss triggered",
            }
        ]

        res = engine.incorporate_trade_feedback(
            feedback_data=stopped_out_feedback,
            learning_rate=0.08,
            loss_penalty_multiplier=2.0,
            epochs=5,
        )
        assert res["status"] == "success"
        assert res["losses"] == 1

        prob_after = engine.predict_edge_probability(features=pattern)

        # STRICT MISTAKE PENALTY VERIFICATION:
        assert prob_after < prob_before, (
            f"Expected prob_after ({prob_after}) < prob_before ({prob_before}) after mistake penalty"
        )

    def test_monotonicity_strictly_preserved_after_feedback(self):
        """
        Monotonic sentiment contract (w_sent >= 0.10) must remain strictly intact
        even after multiple negative feedback updates.
        """
        engine = MultiModalFusionEngine(tier=3)

        # Feed 10 consecutive stopped-out trades
        bad_trades = [
            {
                "net_sentiment": 0.40,
                "positive_prob": 0.65,
                "negative_prob": 0.15,
                "neutral_prob": 0.20,
                "rsi_14": 60.0,
                "momentum_5d": 3.0,
                "label": 0,
                "pnl": -100.0,
            }
            for _ in range(10)
        ]

        engine.incorporate_trade_feedback(
            feedback_data=bad_trades,
            learning_rate=0.15,
            loss_penalty_multiplier=3.0,
            epochs=5,
        )

        # Monotonic weight must never drop below 0.10
        assert engine.numpy_fallback.w_sent >= 0.10

        # Verify monotonicity: higher sentiment still strictly yields higher edge prob
        base_features = {"rsi_14": 50.0, "momentum_5d": 0.0, "volume_z_score": 0.0}
        p_high = engine.predict_edge_probability(features={**base_features, "net_sentiment": 0.80})
        p_low = engine.predict_edge_probability(features={**base_features, "net_sentiment": -0.20})
        assert p_high > p_low


# ============================================================================
# UPGRADE C: LORA DOMAIN ADAPTATION PIPELINE TESTS
# ============================================================================

class TestUpgradeCFinBERTLoraTrainer:
    """Tests for Parameter-Efficient Fine-Tuning (LoRA) Pipeline for FinBERT."""

    def test_financial_domain_dataset_normalization(self):
        """FinancialDomainDataset extracts text from various financial fields and normalizes labels."""
        records = [
            {"statement": "Revenue increased 28% year-over-year beating guidance.", "polarity": "bullish"},
            {"transcript": "Operating margins compressed due to inflationary headwinds.", "sentiment": "negative"},
            {"headline": "Federal Reserve maintains benchmark interest rate unchanged.", "label": 2},
            {"content": "Strong quarterly dividend declared with record free cash flow.", "score": 0.65},
        ]
        ds = FinancialDomainDataset.from_records(records)
        assert len(ds) == 4

        # Verify normalized labels: 0=positive, 1=negative, 2=neutral
        assert ds.items[0]["label"] == 0  # bullish -> positive
        assert ds.items[1]["label"] == 1  # negative
        assert ds.items[2]["label"] == 2  # neutral
        assert ds.items[3]["label"] == 0  # positive from score 0.65

        # Verify text normalization
        assert "Revenue increased" in ds.items[0]["text"]
        assert "Operating margins" in ds.items[1]["text"]

    def test_financial_domain_dataset_json_loader(self, temp_dir):
        """Loads dataset from JSON file."""
        sample_json = temp_dir / "financial_news.json"
        data = [
            {"title": "Tata Motors commercial vehicle sales surge 18%", "sentiment": "positive"},
            {"title": "Tech sector valuations under pressure amid bond yield surge", "sentiment": "negative"}
        ]
        sample_json.write_text(json.dumps(data), encoding="utf-8")

        ds = FinancialDomainDataset.from_json(sample_json)
        assert len(ds) == 2
        assert ds.items[0]["label"] == 0
        assert ds.items[1]["label"] == 1

    def test_lora_config_parameter_efficiency(self):
        """FinBERTLoraConfig defines target modules and verifies <1% trainable parameters."""
        cfg = FinBERTLoraConfig(r=8, lora_alpha=16, target_modules=["query", "value"])
        assert cfg.r == 8
        assert cfg.lora_alpha == 16
        assert "query" in cfg.target_modules
        assert "value" in cfg.target_modules

        trainer = FinBERTLoraTrainer(lora_config=cfg)
        summary = trainer.get_trainable_parameters_summary()

        # Parameter efficiency invariant: < 1.0% trainable
        assert summary["trainable_percent"] < 1.0
        assert summary["is_parameter_efficient"] is True
        assert summary["trainable_parameters"] > 0

    def test_mock_lora_adapter_serialization(self, temp_dir):
        """Trainer serialization writes adapter_config.json and metadata."""
        trainer = FinBERTLoraTrainer(output_dir=temp_dir)
        records = [
            {"text": "Nifty surges to all-time high following robust GDP data.", "sentiment": "positive"},
            {"text": "Manufacturing PMI contracts to 48.2 amid slowdown.", "sentiment": "negative"},
        ]
        train_res = trainer.train(dataset=records, epochs=2)
        assert train_res["status"] == "completed"
        assert len(train_res["history"]) == 2
        # Loss must show decreasing progression
        assert train_res["history"][1]["loss"] < train_res["history"][0]["loss"]

        save_path = trainer.save_adapter()
        assert Path(save_path).exists()
        assert (Path(save_path) / "adapter_config.json").exists()
        assert (Path(save_path) / "adapter_metadata.json").exists()

        # Load adapter back
        loaded_model = trainer.load_adapter(save_path)
        assert loaded_model is not None

    def test_finbert_analyzer_supports_adapter_path(self, temp_dir):
        """FinBERTAnalyzer accepts adapter_path parameter without failure."""
        adapter_dir = temp_dir / "test_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}), encoding="utf-8")

        analyzer = FinBERTAnalyzer(adapter_path=str(adapter_dir))
        assert analyzer.adapter_path == str(adapter_dir)

        # Empty analysis works with adapter configured
        empty_res = analyzer.analyze("")
        assert empty_res["calibrated"] is True
        assert empty_res["polarity"] == "neutral"

    def test_adapt_finbert_lora_convenience_function(self, temp_dir):
        """Convenience functional interface executes end-to-end adaptation."""
        records = [
            {"statement": "Quarterly net profit rose 34% beating analyst consensus.", "polarity": "positive"},
            {"statement": "Subdued consumer demand impacts seasonal inventory clearance.", "polarity": "negative"},
        ]
        res = adapt_finbert_lora(records=records, output_dir=temp_dir, epochs=2)
        assert res["status"] == "completed"
        assert "saved_adapter_path" in res
        assert Path(res["saved_adapter_path"]).exists()

    def test_financial_domain_dataset_composite_title_snippet(self):
        """FinancialDomainDataset combines headline and snippet into full context."""
        records = [
            {
                "title": "Fed announces 25 bps rate hike",
                "snippet": "Policymakers cite stubborn inflation as primary rationale for tighter monetary conditions.",
                "sentiment": "negative"
            }
        ]
        ds = FinancialDomainDataset.from_records(records)
        assert len(ds) == 1
        assert "Fed announces 25 bps rate hike" in ds.items[0]["text"]
        assert "Policymakers cite stubborn inflation" in ds.items[0]["text"]
        assert ds.items[0]["label"] == 1

    def test_financial_domain_dataset_nested_json_keys(self, temp_dir):
        """FinancialDomainDataset correctly parses JSON with 'articles' and 'news' keys."""
        sample_json = temp_dir / "nested_news.json"
        data = {
            "articles": [
                {"headline": "Infosys signs $1.5B cloud deal", "sentiment": "bullish"},
                {"headline": "Crude oil declines 3% amid inventory build", "sentiment": "bearish"}
            ]
        }
        sample_json.write_text(json.dumps(data), encoding="utf-8")
        ds = FinancialDomainDataset.from_json(sample_json)
        assert len(ds) == 2
        assert ds.items[0]["label"] == 0
        assert ds.items[1]["label"] == 1


# ============================================================================
# COMPREHENSIVE EDGE CASE & RESILIENCE TESTS
# ============================================================================

class TestEdgeCasesAndResilience:
    """Rigorous tests probing boundaries, error paths, and robustness across all 3 upgrades."""

    def test_empty_rolling_bars_graceful_handling(self, temp_dir):
        """RollingRetrainingEngine must handle empty or missing bar data without raising unhandled exceptions."""
        engine = RollingRetrainingEngine(
            rolling_window_days=60,
            data_dir=temp_dir,
            offline=True,
        )
        res = engine.recalibrate_bots({})
        assert res["accepted_count"] == 0
        assert res["rejected_count"] == 0
        assert isinstance(res["updated_params"], dict)

    def test_calibration_error_resilience(self, temp_dir, sample_bars_data):
        """If calibration for a bot throws an exception, candidate is rejected and defaults safely retained."""
        engine = RollingRetrainingEngine(
            rolling_window_days=60,
            data_dir=temp_dir,
            offline=True,
        )
        with patch("src.rolling_retrainer.WalkForwardCalibrator.calibrate", side_effect=RuntimeError("Simulated backtest fault")):
            res = engine.recalibrate_bots(sample_bars_data)
            assert res["rejected_count"] == 5
            assert res["accepted_count"] == 0
            for bot_id, b_eval in res["bot_evaluations"].items():
                assert b_eval["action"] == "rejected"
                assert "Calibration error" in b_eval["reason"]
                assert b_eval["is_overfit"] is True

    def test_fusion_engine_retrain_formats(self):
        """MultiModalFusionEngine.retrain must handle list of tuples, list of dicts, and arrays seamlessly."""
        engine = MultiModalFusionEngine(tier=3)
        sample_dict = {
            "net_sentiment": 0.40,
            "positive_prob": 0.60,
            "negative_prob": 0.20,
            "rsi_14": 55.0,
            "momentum_5d": 2.0,
            "volume_z_score": 0.5,
            "atr_volatility": 1.5,
            "above_sma20": 1.0,
            "catalyst_clarity": 0.5,
            "text_anchor_similarity": 0.1,
        }

        # 1. List of tuples (dict, label)
        samples_tuples = [(dict(sample_dict), 1), (dict(sample_dict), 0)]
        res_tuples = engine.retrain(samples_tuples)
        assert res_tuples["status"] == "success"
        assert res_tuples["samples"] == 2

        # 2. List of dicts with label
        samples_dicts = [{**sample_dict, "label": 1}, {**sample_dict, "label": 0}]
        res_dicts = engine.retrain(samples_dicts)
        assert res_dicts["status"] == "success"
        assert res_dicts["samples"] == 2

        # 3. Features array with separate labels array
        X_arr = np.tile(np.array([0.4, 0.6, 0.2, 0.2, 0.0, 0.0, 0.0, 55.0, 2.0, 0.5, 1.5, 1.0, 0.5, 0.1]), (4, 1))
        y_arr = np.array([1, 0, 1, 0])
        res_arr = engine.retrain(X_arr, y=y_arr)
        assert res_arr["status"] == "success"
        assert res_arr["samples"] == 4

    def test_supabase_entry_features_preservation(self, temp_dir):
        """
        When TournamentDataManager runs with Supabase client (which has no entry_features column),
        entry features must be cached in local memory and re-attached on get_positions.
        """
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
            {"bot_id": "conservative_swing", "symbol": "AAPL", "shares": 10.0, "average_buy_price": 150.0}
        ]

        mgr = TournamentDataManager(client=mock_client, data_dir=temp_dir)
        mgr.has_supabase_bot_id = True

        entry_feat = {"net_sentiment": 0.85, "rsi_14": 58.0, "fusion_probability": 0.82}
        mgr.upsert_position("conservative_swing", "AAPL", 10.0, 150.0, entry_features=entry_feat)

        # Retrieve positions - entry features must be enriched from local cache
        positions = mgr.get_positions("conservative_swing")
        assert len(positions) == 1
        assert "entry_features" in positions[0]
        assert positions[0]["entry_features"]["net_sentiment"] == 0.85
        assert positions[0]["entry_features"]["fusion_probability"] == 0.82

    def test_negative_sentiment_weight_increase_on_loss(self):
        """
        When a trade with high negative sentiment results in a loss (y=0),
        the negative sentiment penalty weight w_neg must increase (or stay non-negative).
        """
        engine = MultiModalFusionEngine(tier=3)
        initial_w_neg = engine.numpy_fallback.w_neg

        bad_feedback = [{
            "net_sentiment": -0.40,
            "positive_prob": 0.15,
            "negative_prob": 0.75,
            "neutral_prob": 0.10,
            "rsi_14": 42.0,
            "momentum_5d": -3.0,
            "label": 0,
            "pnl": -80.0,
        }]

        engine.incorporate_trade_feedback(
            feedback_data=bad_feedback,
            learning_rate=0.10,
            loss_penalty_multiplier=2.0,
            epochs=3,
        )

        assert engine.numpy_fallback.w_neg >= initial_w_neg

