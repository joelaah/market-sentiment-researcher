"""
Adversarial Stress Test Suite for Milestone 5 Challenger 2
Focus: Tournament Parameter Sync, 3-Tier Resolution Faults, Cache Corruption, Persona Attribute Propagation, CLI Stress.

Author: m5_challenger_2 (Tournament Sync & CLI Stress Tester)
"""

import os
import sys
import json
import tempfile
import unittest
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paper_trader import (
    MultiBotTournamentEngine,
    ConservativeSwingBot,
    AggressiveBreakoutBot,
    ContrarianDipBuyerBot,
    FinBERTCatalystHunterBot,
    TrendFollowerBot,
    get_default_tournament_bots,
)
from src.db import (
    load_calibrated_parameters,
    save_calibrated_parameters,
    save_training_run,
    _safe_read_json,
    _atomic_write_json,
)
import src.db as db_module
import run_pipeline


class TestSupabaseNetworkFaults(unittest.TestCase):
    """
    Challenge 1: Supabase client unhandled network errors.
    Tests timeouts, 500 errors, socket disconnects, and malformed responses.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.original_data_dir = getattr(db_module, "DATA_DIR", None)
        self.original_params_file = getattr(db_module, "CALIBRATED_PARAMS_FILE", None)
        self.original_runs_file = getattr(db_module, "TRAINING_RUNS_FILE", None)

        db_module.DATA_DIR = Path(self.temp_dir)
        db_module.CALIBRATED_PARAMS_FILE = Path(self.temp_dir) / "bot_calibrated_parameters.json"
        db_module.TRAINING_RUNS_FILE = Path(self.temp_dir) / "bot_training_runs.json"

    def tearDown(self):
        if self.original_data_dir:
            db_module.DATA_DIR = self.original_data_dir
        if self.original_params_file:
            db_module.CALIBRATED_PARAMS_FILE = self.original_params_file
        if self.original_runs_file:
            db_module.TRAINING_RUNS_FILE = self.original_runs_file
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_timeout_network_fault_falls_back_cleanly(self):
        """Simulates network timeout on Supabase execute call."""
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_client.table.return_value = mock_table
        mock_table.select.return_value = mock_table
        mock_table.eq.return_value = mock_table
        mock_table.order.return_value = mock_table
        mock_table.limit.return_value = mock_table
        mock_table.execute.side_effect = TimeoutError("Connection timed out after 30000ms")

        # Save parameters locally first so Tier 2 has valid data
        cached_params = {
            "conservative_swing": {
                "bot_id": "conservative_swing",
                "parameters": {"stop_loss_pct": -4.0, "take_profit_pct": 16.0}
            }
        }
        _atomic_write_json(db_module.CALIBRATED_PARAMS_FILE, cached_params)

        # Call load_calibrated_parameters with fault-injecting client
        result = load_calibrated_parameters(mock_client, "conservative_swing")

        # Must NOT raise exception; must successfully fall back to Tier 2 local JSON
        self.assertIsNotNone(result)
        self.assertEqual(result["stop_loss_pct"], -4.0)
        self.assertEqual(result["take_profit_pct"], 16.0)

    def test_http_500_api_error_falls_back_cleanly(self):
        """Simulates Supabase HTTP 500 internal server error."""
        mock_client = MagicMock()
        mock_client.table.side_effect = RuntimeError("HTTP 500 Internal Server Error: Database unavailable")

        # No local cache either -> should gracefully fall back to Tier 3 (None)
        result = load_calibrated_parameters(mock_client, "aggressive_breakout")
        self.assertIsNone(result)

    def test_socket_disconnect_during_tournament_sync(self):
        """Simulates socket disconnect specifically on bot_calibrated_parameters query during sync."""
        mock_client = MagicMock()
        mock_param_table = MagicMock()
        mock_portfolio_table = MagicMock()

        def table_router(table_name):
            if table_name == "bot_calibrated_parameters":
                return mock_param_table
            return mock_portfolio_table

        mock_client.table.side_effect = table_router

        # paper_portfolio queries succeed with empty data
        mock_portfolio_table.select.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
        mock_portfolio_table.select.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
        mock_portfolio_table.insert.return_value.execute.return_value = MagicMock(data=[])

        # bot_calibrated_parameters queries throw socket ConnectionResetError
        mock_param_table.select.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.side_effect = ConnectionResetError("Connection reset by peer")

        engine = MultiBotTournamentEngine(client=mock_client, auto_sync_params=False)
        # Should not raise exception; must gracefully catch and retain defaults
        summary = engine.sync_calibrated_parameters()

        self.assertIsInstance(summary, dict)
        self.assertEqual(len(summary), 5)
        # Bots should retain default parameters
        self.assertEqual(engine.bots["conservative_swing"].stop_loss_pct, -7.5)
        self.assertEqual(engine.bots["aggressive_breakout"].min_momentum_5d, 2.0)

    def test_tournament_data_manager_vulnerability_on_socket_disconnect(self):
        """
        Adversarial Probe: Checks TournamentDataManager.ensure_bot_initialized behavior
        when paper_portfolio query throws socket ConnectionResetError.
        """
        mock_client = MagicMock()
        mock_portfolio_table = MagicMock()
        mock_client.table.return_value = mock_portfolio_table
        mock_portfolio_table.select.side_effect = ConnectionResetError("Connection reset by peer")

        # When client is active but throws socket disconnect during portfolio init,
        # ensure_bot_initialized raises ConnectionResetError because lines 1364/1373 lack try/except.
        with self.assertRaises(ConnectionResetError):
            engine = MultiBotTournamentEngine(client=mock_client, auto_sync_params=False)


class TestLocalCacheCorruption(unittest.TestCase):
    """
    Challenge 2: Local cache corruption.
    Tests syntax errors, truncated JSON, empty file, non-dict JSON, and write recovery.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.original_data_dir = getattr(db_module, "DATA_DIR", None)
        self.original_params_file = getattr(db_module, "CALIBRATED_PARAMS_FILE", None)

        db_module.DATA_DIR = Path(self.temp_dir)
        self.cache_file = Path(self.temp_dir) / "bot_calibrated_parameters.json"
        db_module.CALIBRATED_PARAMS_FILE = self.cache_file

    def tearDown(self):
        if self.original_data_dir:
            db_module.DATA_DIR = self.original_data_dir
        if self.original_params_file:
            db_module.CALIBRATED_PARAMS_FILE = self.original_params_file
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_empty_cache_file_0_bytes(self):
        """Empty 0-byte cache file must be caught gracefully and return None (defaults)."""
        self.cache_file.write_text("", encoding="utf-8")

        result = load_calibrated_parameters(None, "conservative_swing")
        self.assertIsNone(result)

    def test_truncated_json_file(self):
        """Truncated JSON syntax error must be caught gracefully."""
        self.cache_file.write_text('{"conservative_swing": {"parameters": {"stop_loss_pct": -5.0', encoding="utf-8")

        result = load_calibrated_parameters(None, "conservative_swing")
        self.assertIsNone(result)

    def test_malformed_syntax_error(self):
        """Malformed garbage content must return default without unhandled exception."""
        self.cache_file.write_text("<<<CORRUPT_NOT_JSON>>>", encoding="utf-8")

        result = load_calibrated_parameters(None, "conservative_swing")
        self.assertIsNone(result)

    def test_non_dict_json_root(self):
        """JSON root is a list or scalar instead of a dictionary."""
        self.cache_file.write_text('[1, 2, "unexpected", false]', encoding="utf-8")

        result = load_calibrated_parameters(None, "conservative_swing")
        self.assertIsNone(result)

    def test_missing_parameters_key_in_entry(self):
        """Entry exists but has no valid parameters dictionary or recognizable keys."""
        self.cache_file.write_text(json.dumps({
            "conservative_swing": {"invalid_data": 123}
        }), encoding="utf-8")

        result = load_calibrated_parameters(None, "conservative_swing")
        self.assertIsNone(result)

    def test_tournament_engine_sync_with_corrupt_cache(self):
        """MultiBotTournamentEngine must initialize and sync cleanly even when cache is corrupt."""
        self.cache_file.write_text("!!!CORRUPT_BYTES!!!", encoding="utf-8")

        engine = MultiBotTournamentEngine(client=None, auto_sync_params=True)
        self.assertEqual(len(engine.bots), 5)
        # All bots must retain their default parameters safely
        self.assertEqual(engine.bots["conservative_swing"].stop_loss_pct, -7.5)
        self.assertEqual(engine.bots["aggressive_breakout"].min_momentum_5d, 2.0)
        self.assertEqual(engine.bots["contrarian_dip_buyer"].dip_threshold_pct, -1.5)
        self.assertEqual(engine.bots["finbert_catalyst_hunter"].min_pos_prob, 0.70)
        self.assertEqual(engine.bots["trend_follower"].take_profit_pct, 15.0)


class TestPersonaAttributePropagationAndExecution(unittest.TestCase):
    """
    Challenge 3: Persona attribute propagation to live bot instances
    and empirical verification that calibrated parameters alter trade execution.
    """

    def test_all_5_bots_receive_instance_attributes(self):
        """Verify that set_params directly sets persona-specific attributes on all 5 bot instances."""
        # 1. ConservativeSwingBot
        cs = ConservativeSwingBot()
        self.assertEqual(cs.min_momentum_5d, 0.0)
        self.assertEqual(cs.conviction_threshold, 7.0)
        self.assertEqual(cs.stop_loss_pct, -7.5)
        self.assertEqual(cs.take_profit_pct, 12.0)

        cs.set_params(min_momentum_5d=1.2, conviction_threshold=8.5, stop_loss_pct=-4.0, take_profit_pct=15.0)
        self.assertEqual(cs.min_momentum_5d, 1.2)
        self.assertEqual(cs.conviction_threshold, 8.5)
        self.assertEqual(cs.stop_loss_pct, -4.0)
        self.assertEqual(cs.take_profit_pct, 15.0)

        # 2. AggressiveBreakoutBot
        ab = AggressiveBreakoutBot()
        self.assertEqual(ab.min_momentum_5d, 2.0)
        self.assertEqual(ab.min_net_score, 0.05)
        self.assertEqual(ab.allocation_pct, 0.25)

        ab.set_params(min_momentum_5d=3.5, min_net_score=0.20, allocation_pct=0.30)
        self.assertEqual(ab.min_momentum_5d, 3.5)
        self.assertEqual(ab.min_net_score, 0.20)
        self.assertEqual(ab.allocation_pct, 0.30)
        self.assertEqual(ab.max_trade_dollars, 1500.0)  # Starting balance 5000 * 0.30 = 1500

        # 3. ContrarianDipBuyerBot
        cdb = ContrarianDipBuyerBot()
        self.assertEqual(cdb.dip_threshold_pct, -1.5)
        self.assertEqual(cdb.dip_max_pct, -1.5)
        self.assertEqual(cdb.dip_min_pct, -8.0)

        cdb.set_params(dip_threshold_pct=-2.5, dip_min_pct=-10.0)
        self.assertEqual(cdb.dip_threshold_pct, -2.5)
        self.assertEqual(cdb.dip_max_pct, -2.5)
        self.assertEqual(cdb.dip_min_pct, -10.0)

        # 4. FinBERTCatalystHunterBot
        fb = FinBERTCatalystHunterBot()
        self.assertEqual(fb.conviction_threshold, 0.50)
        self.assertEqual(fb.min_net_score, 0.50)

        fb.set_params(conviction_threshold=0.65)
        self.assertEqual(fb.conviction_threshold, 0.65)
        self.assertEqual(fb.min_net_score, 0.65)

        # 5. TrendFollowerBot
        tf = TrendFollowerBot()
        self.assertEqual(tf.min_momentum_5d, 0.0)
        self.assertEqual(tf.take_profit_pct, 15.0)

        tf.set_params(min_momentum_5d=1.5, take_profit_pct=22.0)
        self.assertEqual(tf.min_momentum_5d, 1.5)
        self.assertEqual(tf.take_profit_pct, 22.0)

    def test_trade_execution_altered_by_calibrated_parameters(self):
        """Empirical verification: Calibrated parameters change buy and sell decisions."""
        # Scenario A: AggressiveBreakoutBot momentum filtering
        bot_ab = AggressiveBreakoutBot()
        tech = {"valid": True, "price": 105.0, "sma20": 100.0, "above_sma20": True, "momentum_5d": 2.2}
        bert = {"net_score": 0.10, "info": "Bullish"}
        articles = [{"title": "Strong earnings"}]

        # Default: min_momentum_5d is 2.0 -> 2.2 >= 2.0 -> BUY triggered
        buy_default, _, _ = bot_ab.evaluate_entry("NVDA", tech, bert, articles, [])
        self.assertTrue(buy_default)

        # Calibrate: min_momentum_5d raised to 2.5% -> 2.2 < 2.5 -> BUY REJECTED
        bot_ab.set_params(min_momentum_5d=2.5)
        buy_calibrated, _, _ = bot_ab.evaluate_entry("NVDA", tech, bert, articles, [])
        self.assertFalse(buy_calibrated)

        # Scenario B: ContrarianDipBuyerBot dip threshold
        bot_cdb = ContrarianDipBuyerBot()
        tech_dip = {"valid": True, "price": 98.0, "sma20": 100.0, "momentum_5d": -1.8}
        bert_dip = {"net_score": 0.0, "info": "Neutral"}

        # Default: dip_max_pct is -1.5 -> -1.8 is in [-8.0, -1.5] -> BUY triggered
        buy_dip_def, _, _ = bot_cdb.evaluate_entry("AAPL", tech_dip, bert_dip, articles, [])
        self.assertTrue(buy_dip_def)

        # Calibrate: dip_threshold_pct tightened to -2.0% (must drop at least -2.0%)
        # Now -1.8 > -2.0 -> BUY REJECTED
        bot_cdb.set_params(dip_threshold_pct=-2.0)
        buy_dip_cal, _, _ = bot_cdb.evaluate_entry("AAPL", tech_dip, bert_dip, articles, [])
        self.assertFalse(buy_dip_cal)

        # Scenario C: FinBERTCatalystHunterBot conviction cutoff
        bot_fb = FinBERTCatalystHunterBot()
        bert_cat = {"net_score": 0.55, "positive": 0.60, "headline": "New product launch", "info": "Positive"}

        # Default: conviction_threshold / min_net_score is 0.50 -> 0.55 >= 0.50 -> BUY triggered
        buy_fb_def, _, _ = bot_fb.evaluate_entry("MSFT", {}, bert_cat, articles, [])
        self.assertTrue(buy_fb_def)

        # Calibrate: conviction_threshold raised to 0.60 -> 0.55 < 0.60 -> BUY REJECTED
        bot_fb.set_params(conviction_threshold=0.60)
        buy_fb_cal, _, _ = bot_fb.evaluate_entry("MSFT", {}, bert_cat, articles, [])
        self.assertFalse(buy_fb_cal)

        # Scenario D: Stop Loss execution alteration
        bot_cs = ConservativeSwingBot()
        pos = {"symbol": "SPY", "pnl_pct": -6.0}

        # Default SL: -7.5% -> -6.0% does NOT trigger SL
        sold_def, _ = bot_cs.evaluate_exit(pos, {"above_sma20": True, "momentum_5d": 1.0}, {"news_score": 2.0})
        self.assertFalse(sold_def)

        # Calibrate SL: -5.0% -> -6.0% <= -5.0% -> TRIGGERS STOP LOSS
        bot_cs.set_params(stop_loss_pct=-5.0)
        sold_cal, reason = bot_cs.evaluate_exit(pos, {"above_sma20": True, "momentum_5d": 1.0}, {"news_score": 2.0})
        self.assertTrue(sold_cal)
        self.assertIn("Stop Loss triggered", reason)


class TestCLIExecutionAndErrorHandling(unittest.TestCase):
    """
    Challenge 4: CLI error handling in run_pipeline.py.
    Tests invalid --years, missing cache files, and offline training.
    """

    def test_cli_invalid_years_zero(self):
        """CLI with --years 0 must cleanly exit with error code 1."""
        with self.assertRaises(SystemExit) as cm:
            run_pipeline.run_calibration_training(years=0)
        self.assertEqual(cm.exception.code, 1)

    def test_cli_invalid_years_negative(self):
        """CLI with --years -5 must cleanly exit with error code 1."""
        with self.assertRaises(SystemExit) as cm:
            run_pipeline.run_calibration_training(years=-5)
        self.assertEqual(cm.exception.code, 1)

    def test_cli_backtest_report_missing_cache_files(self):
        """--backtest-report when no Supabase and no cache files exists must return False without crashing."""
        temp_dir = tempfile.mkdtemp()
        orig_data = getattr(db_module, "DATA_DIR", None)
        orig_params = getattr(db_module, "CALIBRATED_PARAMS_FILE", None)
        orig_runs = getattr(db_module, "TRAINING_RUNS_FILE", None)

        db_module.DATA_DIR = Path(temp_dir)
        db_module.CALIBRATED_PARAMS_FILE = Path(temp_dir) / "bot_calibrated_parameters.json"
        db_module.TRAINING_RUNS_FILE = Path(temp_dir) / "bot_training_runs.json"

        try:
            # Both files are absent
            ok = run_pipeline.run_backtest_report(client=None)
            self.assertFalse(ok)
        finally:
            if orig_data:
                db_module.DATA_DIR = orig_data
            if orig_params:
                db_module.CALIBRATED_PARAMS_FILE = orig_params
            if orig_runs:
                db_module.TRAINING_RUNS_FILE = orig_runs
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_cli_offline_training_with_cached_data(self):
        """--train execution completely offline using cached data must succeed and persist locally."""
        temp_dir = tempfile.mkdtemp()
        orig_data = getattr(db_module, "DATA_DIR", None)
        orig_params = getattr(db_module, "CALIBRATED_PARAMS_FILE", None)
        orig_runs = getattr(db_module, "TRAINING_RUNS_FILE", None)

        db_module.DATA_DIR = Path(temp_dir)
        db_module.CALIBRATED_PARAMS_FILE = Path(temp_dir) / "bot_calibrated_parameters.json"
        db_module.TRAINING_RUNS_FILE = Path(temp_dir) / "bot_training_runs.json"

        # Generate synthetic cached bars for fast testing
        dates = pd.date_range("2023-01-01", periods=100, freq="B")
        synthetic_data = {}
        for sym in ["^NSEI", "NIFTYBEES.NS", "NVDA", "AAPL", "MSFT", "SPY"]:
            np.random.seed(42)
            prices = 100.0 + np.cumsum(np.random.randn(100) * 0.5)
            df = pd.DataFrame({
                "open": prices,
                "high": prices + 1.0,
                "low": prices - 1.0,
                "close": prices + 0.2,
                "volume": 100000.0
            }, index=dates)
            synthetic_data[sym] = df

        try:
            # Run calibration training offline with synthetic bars and client=None (offline)
            output = run_pipeline.run_calibration_training(years=1, data=synthetic_data, client=None, offline=True)
            self.assertIn("personas", output)
            self.assertEqual(len(output["personas"]), 5)

            # Verify local files were created
            self.assertTrue(db_module.CALIBRATED_PARAMS_FILE.exists())
            self.assertTrue(db_module.TRAINING_RUNS_FILE.exists())

            # Verify backtest report renders from the local cache
            report_ok = run_pipeline.run_backtest_report(client=None)
            self.assertTrue(report_ok)
        finally:
            if orig_data:
                db_module.DATA_DIR = orig_data
            if orig_params:
                db_module.CALIBRATED_PARAMS_FILE = orig_params
            if orig_runs:
                db_module.TRAINING_RUNS_FILE = orig_runs
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_cli_subprocess_years_invalid_string(self):
        """CLI invocation with non-integer --years should fail with argparse error and code 2."""
        import subprocess
        proc = subprocess.run(
            [sys.executable, "run_pipeline.py", "--train", "--years", "invalid_number"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("invalid int value", proc.stderr)

    def test_cli_subprocess_years_zero(self):
        """CLI invocation with --years 0 should cleanly exit with code 1 and clear error message."""
        import subprocess
        proc = subprocess.run(
            [sys.executable, "run_pipeline.py", "--train", "--years", "0"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("Invalid years parameter: 0. Must be >= 1.", proc.stdout)

    def test_cli_subprocess_years_negative(self):
        """CLI invocation with --years -5 should cleanly exit with code 1 and clear error message."""
        import subprocess
        proc = subprocess.run(
            [sys.executable, "run_pipeline.py", "--train", "--years", "-5"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("Invalid years parameter: -5. Must be >= 1.", proc.stdout)

    def test_cli_subprocess_backtest_report_runs_cleanly(self):
        """CLI invocation with --backtest-report executes cleanly with code 0."""
        import subprocess
        proc = subprocess.run(
            [sys.executable, "run_pipeline.py", "--backtest-report"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIsNotNone(proc.stdout)
        self.assertIn("HISTORICAL BACKTEST & BOT CALIBRATION REPORT", proc.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
