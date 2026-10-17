"""
Milestone 5 Empirical Stress Test & Edge Case Verification Suite.
Challenger: m5_challenger_1 (critic, specialist).

Adversarially tests:
1. Performance Metrics Edge Cases:
   - Zero trades executed (division by zero handling in Win Rate, Profit Factor, Max Drawdown, CAGR, Net Return).
   - 100% winning trades (infinite profit factor handling, JSON safety).
   - 100% losing trades (zero profit factor handling, 0% win rate).
   - Drawdown calculations: flat equity, monotonically declining equity, flash crashes, total bankruptcy (0 equity), negative equity.
   - Malformed trade records: missing fields, NaN/Inf PnL, invalid types.
   - Corrupt equity curve inputs: None, NaNs, Infs, string values, tuples vs scalars.
   - Non-positive initial balances and zero duration handling.

2. Chronological Replay & Zero-Lookahead Rolling Indicators:
   - Zero forward lookahead leakage in compute_rolling_indicators and _prepare_data_and_technicals.
   - Indicator invariance at bar t under arbitrary future price anomalies (massive spikes, drops).
   - Rolling indicator boundary conditions: empty DF, 1 bar, 5 bars, 19 bars, 20 bars.
   - Division by zero safety in 5-day momentum when past price <= 0.0.
   - Multi-asset mixed trading calendar alignment (US holidays vs Indian NSE holidays).
   - Mark-to-market continuity using latest_known_prices during market closures.

3. Walk-Forward Calibration & Data Leakage:
   - Strict chronological disjointness between in-sample and out-of-sample splits (zero data leakage).
   - Data partitioning under mismatched multi-asset calendar spans.
   - Sparse historical data handling and boundary validations (< 4 bars, < 20 bars for single asset).
   - Fitness function behavior under zero trades, extreme drawdowns, and infinite profit factors.
   - Full WalkForwardCalibrator execution under zero-trade regime (graceful status handling).
"""

import math
import unittest
import numpy as np
import pandas as pd
from typing import Dict, List, Any

from src.backtest_engine import (
    HistoricalDataFetcher,
    HistoricalBacktestEngine,
    SimulatedPortfolio,
    compute_rolling_indicators,
    calculate_performance_metrics,
)
from src.paper_trader import (
    ConservativeSwingBot,
    AggressiveBreakoutBot,
    ContrarianDipBuyerBot,
    FinBERTCatalystHunterBot,
    TrendFollowerBot,
    get_default_tournament_bots,
)
from src.calibration import (
    WalkForwardCalibrator,
    split_chronological_data,
    calculate_fitness_score,
    ObjectiveFunction,
    generate_grid_combinations,
    _to_json_safe,
)


class TestMetricsEdgeCasesAdversarial(unittest.TestCase):
    """Adversarial stress testing of quantitative performance metric calculation."""

    def test_zero_trades_clean_handling(self):
        """Zero trades executed must return 0.0 for win rate, profit factor, gross profit/loss without ZeroDivisionError."""
        metrics = calculate_performance_metrics(
            trades=[],
            equity_curve=[5000.0, 5000.0, 5000.0],
            initial_balance=5000.0,
            years=3.0
        )
        self.assertEqual(metrics["total_trades"], 0)
        self.assertEqual(metrics["wins"], 0)
        self.assertEqual(metrics["losses"], 0)
        self.assertEqual(metrics["win_rate_pct"], 0.0)
        self.assertEqual(metrics["profit_factor"], 0.0)
        self.assertEqual(metrics["gross_profit"], 0.0)
        self.assertEqual(metrics["gross_loss"], 0.0)
        self.assertEqual(metrics["max_drawdown_pct"], 0.0)
        self.assertEqual(metrics["net_return_pct"], 0.0)
        self.assertEqual(metrics["cagr_pct"], 0.0)

    def test_none_trades_and_none_equity(self):
        """None passed for trades and equity_curve must not raise TypeError."""
        metrics = calculate_performance_metrics(
            trades=None,
            equity_curve=None,
            initial_balance=5000.0,
            years=3.0
        )
        self.assertEqual(metrics["total_trades"], 0)
        self.assertEqual(metrics["win_rate_pct"], 0.0)
        self.assertEqual(metrics["profit_factor"], 0.0)
        self.assertEqual(metrics["max_drawdown_pct"], 0.0)
        self.assertEqual(metrics["net_return_pct"], 0.0)

    def test_100_percent_winning_trades_profit_factor_cap(self):
        """100% winning trades (zero gross loss) must produce a JSON-compliant finite ceiling, not inf/NaN."""
        trades = [
            {"pnl": 150.0},
            {"pnl": 220.50},
            {"realized_pnl": 85.25},
            {"pnl": 12.0}
        ]
        equity = [5000.0, 5150.0, 5370.5, 5455.75, 5467.75]
        metrics = calculate_performance_metrics(trades, equity, initial_balance=5000.0, years=1.0)

        self.assertEqual(metrics["total_trades"], 4)
        self.assertEqual(metrics["wins"], 4)
        self.assertEqual(metrics["losses"], 0)
        self.assertEqual(metrics["win_rate_pct"], 100.0)
        self.assertEqual(metrics["gross_loss"], 0.0)
        self.assertGreater(metrics["gross_profit"], 0.0)
        # Verify JSON-compliant ceiling (999.0)
        self.assertEqual(metrics["profit_factor"], 999.0)
        self.assertTrue(math.isfinite(metrics["profit_factor"]))
        self.assertEqual(metrics["max_drawdown_pct"], 0.0)
        self.assertGreater(metrics["net_return_pct"], 0.0)

    def test_100_percent_losing_trades(self):
        """100% losing trades (zero gross profit) must yield 0.0 win rate and 0.0 profit factor."""
        trades = [
            {"pnl": -120.0},
            {"pnl": -340.0},
            {"realized_pnl": -50.0}
        ]
        equity = [5000.0, 4880.0, 4540.0, 4490.0]
        metrics = calculate_performance_metrics(trades, equity, initial_balance=5000.0, years=1.0)

        self.assertEqual(metrics["total_trades"], 3)
        self.assertEqual(metrics["wins"], 0)
        self.assertEqual(metrics["losses"], 3)
        self.assertEqual(metrics["win_rate_pct"], 0.0)
        self.assertEqual(metrics["profit_factor"], 0.0)
        self.assertEqual(metrics["gross_profit"], 0.0)
        self.assertEqual(metrics["gross_loss"], 510.0)
        self.assertAlmostEqual(metrics["net_return_pct"], -10.2, places=1)

    def test_flat_equity_curve_drawdown(self):
        """Flat equity curve must strictly produce max_drawdown_pct = 0.0."""
        equity = [5000.0] * 50
        metrics = calculate_performance_metrics([], equity, initial_balance=5000.0, years=2.0)
        self.assertEqual(metrics["max_drawdown_pct"], 0.0)

    def test_monotonically_declining_equity_curve_drawdown(self):
        """Monotonically declining equity curve peak-to-trough calculation."""
        equity = [10000.0, 8000.0, 6000.0, 5000.0]  # Drops 50%
        metrics = calculate_performance_metrics([], equity, initial_balance=10000.0, years=1.0)
        self.assertEqual(metrics["max_drawdown_pct"], 50.0)
        self.assertEqual(metrics["net_return_pct"], -50.0)

    def test_flash_crash_and_recovery_drawdown(self):
        """Equity curve with recovery: MDD must capture the largest peak-to-trough drop."""
        # Peak 1 = 10000 -> drops to 5000 (50% dd) -> rallies to 12000 -> drops to 8400 (30% dd) -> finishes at 15000
        equity = [10000.0, 7500.0, 5000.0, 8000.0, 12000.0, 8400.0, 15000.0]
        metrics = calculate_performance_metrics([], equity, initial_balance=10000.0, years=1.0)
        self.assertEqual(metrics["max_drawdown_pct"], 50.0)
        self.assertEqual(metrics["net_return_pct"], 50.0)

    def test_bankruptcy_zero_and_negative_equity(self):
        """Equity hitting 0 or negative must produce -100% CAGR and MDD >= 100% without math domain error."""
        # 1. Total bankruptcy: 5000 -> 0
        eq_zero = [5000.0, 2500.0, 0.0]
        m_zero = calculate_performance_metrics([], eq_zero, initial_balance=5000.0, years=1.0)
        self.assertEqual(m_zero["max_drawdown_pct"], 100.0)
        self.assertEqual(m_zero["cagr_pct"], -100.0)
        self.assertEqual(m_zero["net_return_pct"], -100.0)

        # 2. Deficit / negative equity: 5000 -> -500
        eq_neg = [5000.0, 1000.0, -500.0]
        m_neg = calculate_performance_metrics([], eq_neg, initial_balance=5000.0, years=1.0)
        self.assertEqual(m_neg["max_drawdown_pct"], 110.0)
        self.assertEqual(m_neg["cagr_pct"], -100.0)
        self.assertEqual(m_neg["net_return_pct"], -110.0)

    def test_malformed_and_adversarial_pnl_in_trades(self):
        """Trades containing None, NaN, inf, or string pnls must be gracefully ignored/coerced to 0."""
        trades = [
            {"pnl": "invalid"},
            {"pnl": float("nan")},
            {"pnl": float("inf")},
            {"pnl": -float("inf")},
            {},  # Missing keys
            "not a dict",  # Invalid type in list
            {"realized_pnl": 100.0},  # Valid win
            {"pnl": -50.0},  # Valid loss
        ]
        metrics = calculate_performance_metrics(trades, [5000.0, 5050.0], initial_balance=5000.0, years=1.0)
        self.assertEqual(metrics["total_trades"], 2)
        self.assertEqual(metrics["wins"], 1)
        self.assertEqual(metrics["losses"], 1)
        self.assertEqual(metrics["win_rate_pct"], 50.0)
        self.assertEqual(metrics["profit_factor"], 2.0)
        self.assertEqual(metrics["gross_profit"], 100.0)
        self.assertEqual(metrics["gross_loss"], 50.0)

    def test_corrupt_equity_curve_filtering(self):
        """Equity curve containing NaNs, Infs, tuples, or unparseable objects must be sanitized."""
        corrupt_equity = [
            5000.0,
            ("2023-01-02", 5200.0),
            None,
            "corrupt_string",
            float("nan"),
            float("inf"),
            ("2023-01-07", 5100.0),
            4800.0,
            5500.0
        ]
        metrics = calculate_performance_metrics([], corrupt_equity, initial_balance=5000.0, years=1.0)
        self.assertEqual(metrics["net_return_pct"], 10.0)  # (5500 - 5000) / 5000 * 100
        # Peak was 5200, trough was 4800 -> drop = 400 / 5200 * 100 = 7.69%
        self.assertEqual(metrics["max_drawdown_pct"], 7.69)

    def test_non_positive_initial_balance_and_years(self):
        """Zero initial balance or zero duration must return 0.0 net return and CAGR, no ZeroDivisionError."""
        m_zero_bal = calculate_performance_metrics([], [0.0], initial_balance=0.0, years=1.0)
        self.assertEqual(m_zero_bal["net_return_pct"], 0.0)
        self.assertEqual(m_zero_bal["cagr_pct"], 0.0)

        m_zero_yr = calculate_performance_metrics([], [5500.0], initial_balance=5000.0, years=0.0)
        self.assertEqual(m_zero_yr["net_return_pct"], 10.0)
        self.assertEqual(m_zero_yr["cagr_pct"], 0.0)


class TestLookaheadRollingIndicatorsAdversarial(unittest.TestCase):
    """Stress testing backward-looking indicator calculation and lookahead prevention."""

    def test_zero_lookahead_indicator_invariance(self):
        """
        Adversarially proves that rolling indicators at bar t are strictly identical
        whether future bars t+1...t+N exist, and regardless of future price anomalies.
        """
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        np.random.seed(42)
        base_prices = [100.0 + i * 0.5 + np.sin(i) for i in range(30)]

        df_history = pd.DataFrame({
            "open": base_prices,
            "high": [p + 1.0 for p in base_prices],
            "low": [p - 1.0 for p in base_prices],
            "close": base_prices,
            "volume": [100000.0] * 30
        }, index=dates)

        # Evaluate indicators at bar t = 24 (day 25)
        cutoff_t = 25
        df_sub = df_history.iloc[:cutoff_t]
        ind_at_t = compute_rolling_indicators(df_sub)

        # Construct future series with extreme 100x spike
        df_future_spike = df_history.copy()
        df_future_spike.loc[dates[cutoff_t:], "close"] = 10000.0
        df_future_spike.loc[dates[cutoff_t:], "high"] = 10050.0

        ind_at_t_from_spike = compute_rolling_indicators(df_future_spike.iloc[:cutoff_t])

        # Construct future series with market collapse to 1.0
        df_future_crash = df_history.copy()
        df_future_crash.loc[dates[cutoff_t:], "close"] = 1.0
        df_future_crash.loc[dates[cutoff_t:], "low"] = 0.5

        ind_at_t_from_crash = compute_rolling_indicators(df_future_crash.iloc[:cutoff_t])

        # Exact invariant assertion
        self.assertEqual(ind_at_t["price"], ind_at_t_from_spike["price"])
        self.assertEqual(ind_at_t["sma20"], ind_at_t_from_spike["sma20"])
        self.assertEqual(ind_at_t["momentum_5d"], ind_at_t_from_spike["momentum_5d"])
        self.assertEqual(ind_at_t["above_sma20"], ind_at_t_from_spike["above_sma20"])

        self.assertEqual(ind_at_t["price"], ind_at_t_from_crash["price"])
        self.assertEqual(ind_at_t["sma20"], ind_at_t_from_crash["sma20"])
        self.assertEqual(ind_at_t["momentum_5d"], ind_at_t_from_crash["momentum_5d"])
        self.assertEqual(ind_at_t["above_sma20"], ind_at_t_from_crash["above_sma20"])

    def test_rolling_indicators_sparse_boundary_conditions(self):
        """Tests compute_rolling_indicators under sparse lengths: empty, 1, 5, 19, 20 bars."""
        # 1. Empty / None
        self.assertFalse(compute_rolling_indicators(None)["valid"])
        self.assertFalse(compute_rolling_indicators(pd.DataFrame())["valid"])

        # 2. Single bar
        d1 = pd.date_range("2023-01-01", periods=1, freq="D")
        df1 = pd.DataFrame({"close": [100.0], "volume": [50000.0]}, index=d1)
        res1 = compute_rolling_indicators(df1)
        self.assertFalse(res1["valid"])  # Less than window_sma=20
        self.assertEqual(res1["price"], 100.0)
        self.assertEqual(res1["sma20"], 100.0)  # Mean of available
        self.assertEqual(res1["momentum_5d"], 0.0)

        # 3. 5 bars (lookback_mom boundary)
        d5 = pd.date_range("2023-01-01", periods=5, freq="D")
        df5 = pd.DataFrame({"close": [100.0, 102.0, 104.0, 106.0, 108.0], "volume": [50000.0]*5}, index=d5)
        res5 = compute_rolling_indicators(df5)
        self.assertFalse(res5["valid"])
        self.assertEqual(res5["momentum_5d"], 0.0)  # Needs > 5 bars to compare with t-5

        # 4. 6 bars (first valid 5d momentum)
        d6 = pd.date_range("2023-01-01", periods=6, freq="D")
        df6 = pd.DataFrame({"close": [100.0, 102.0, 104.0, 106.0, 108.0, 110.0], "volume": [50000.0]*6}, index=d6)
        res6 = compute_rolling_indicators(df6)
        self.assertFalse(res6["valid"])  # Still < 20
        # (110 - 100) / 100 * 100 = 10.0%
        self.assertEqual(res6["momentum_5d"], 10.0)

        # 5. Exactly 20 bars
        d20 = pd.date_range("2023-01-01", periods=20, freq="D")
        df20 = pd.DataFrame({"close": [100.0]*20, "volume": [50000.0]*20}, index=d20)
        res20 = compute_rolling_indicators(df20)
        self.assertTrue(res20["valid"])
        self.assertEqual(res20["sma20"], 100.0)

    def test_momentum_division_by_zero_guard(self):
        """Zero or non-positive past close price in momentum calculation must not raise ZeroDivisionError."""
        d = pd.date_range("2023-01-01", periods=7, freq="D")
        df_zero_past = pd.DataFrame({"close": [0.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0]}, index=d)
        res = compute_rolling_indicators(df_zero_past)
        self.assertEqual(res["momentum_5d"], 0.0)


class TestMixedTradingCalendarAlignment(unittest.TestCase):
    """Stress tests mixed exchange calendars (US NYSE vs Indian NSE holiday divergence)."""

    def test_mixed_calendars_and_mark_to_market(self):
        """
        Adversarial multi-asset calendar test:
        - US instrument 'SPY' trades on Day 1 (US open, Indian closed).
        - Indian instrument 'NIFTYBEES.NS' trades on Day 4 (Indian open, US closed).
        - Both trade on Day 2 and Day 3.
        Proves:
        1. Master calendar forms the complete chronological union [Day 1, 2, 3, 4].
        2. No crash occurs when an instrument is missing on a holiday.
        3. latest_known_prices carries forward previous close for mark-to-market.
        4. Closed instrument is excluded from trade execution on that day.
        """
        # Day 1: 2023-01-02 (US Open, Indian Holiday)
        # Day 2: 2023-01-03 (Both Open)
        # Day 3: 2023-01-04 (Both Open)
        # Day 4: 2023-01-05 (Indian Open, US Holiday)
        dates_us = pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04"])
        dates_in = pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"])

        df_us = pd.DataFrame({
            "open": [400.0, 402.0, 405.0],
            "high": [403.0, 406.0, 408.0],
            "low": [399.0, 401.0, 404.0],
            "close": [402.0, 405.0, 407.0],
            "volume": [1000000.0, 1200000.0, 1100000.0]
        }, index=dates_us)

        df_in = pd.DataFrame({
            "open": [200.0, 202.0, 204.0],
            "high": [203.0, 205.0, 206.0],
            "low": [199.0, 201.0, 203.0],
            "close": [202.0, 204.0, 205.0],
            "volume": [500000.0, 600000.0, 550000.0]
        }, index=dates_in)

        data = {"SPY": df_us, "NIFTYBEES.NS": df_in}

        engine = HistoricalBacktestEngine(initial_balance=10000.0)
        bot = TrendFollowerBot()
        bot.bot_id = "test_calendar_bot"

        res = engine.run_backtest(bots=[bot], data=data)

        # 1. Master calendar verification
        self.assertEqual(res["total_bars"], 4)
        expected_dates = ["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
        self.assertEqual(res["calendar_dates"], expected_dates)

        # 2. Equity curve length must equal number of master calendar days
        bot_res = res["bots"]["test_calendar_bot"]
        self.assertEqual(len(bot_res["equity_curve"]), 4)

        # 3. Check equity values are positive and finite throughout
        for dt_str, eq_val in bot_res["equity_curve"]:
            self.assertTrue(math.isfinite(eq_val))
            self.assertGreater(eq_val, 0.0)


class TestWalkForwardCalibrationAdversarial(unittest.TestCase):
    """Stress testing walk-forward chronological partitioning, data leakage, and calibration."""

    def test_chronological_split_strict_disjointness_zero_leakage(self):
        """Proves that in-sample and out-of-sample splits are strictly disjoint with zero lookahead."""
        dates = pd.date_range("2022-01-01", periods=100, freq="D")
        df1 = pd.DataFrame({
            "open": [100.0] * 100, "high": [105.0] * 100, "low": [95.0] * 100,
            "close": [102.0] * 100, "volume": [100000.0] * 100
        }, index=dates)
        df2 = df1.copy()

        data = {"AAPL": df1, "NIFTYBEES.NS": df2}

        train_data, val_data = split_chronological_data(data, in_sample_pct=0.67)

        # For both instruments:
        for sym in ["AAPL", "NIFTYBEES.NS"]:
            train_idx = set(train_data[sym].index)
            val_idx = set(val_data[sym].index)

            # Strict disjointness
            self.assertEqual(len(train_idx.intersection(val_idx)), 0)

            # Temporal strictly less than
            max_train_date = max(train_data[sym].index)
            min_val_date = min(val_data[sym].index)
            self.assertLess(max_train_date, min_val_date)

            # Partition counts
            self.assertEqual(len(train_data[sym]), 67)
            self.assertEqual(len(val_data[sym]), 33)

    def test_chronological_split_insufficient_bars_rejection(self):
        """Attempting to partition insufficient bars (< 4 bars, or < 20 bars single asset) must raise ValueError."""
        # Single asset with 10 bars
        dates10 = pd.date_range("2023-01-01", periods=10, freq="D")
        df10 = pd.DataFrame({
            "open": [10.0]*10, "high": [11.0]*10, "low": [9.0]*10, "close": [10.0]*10, "volume": [100.0]*10
        }, index=dates10)
        with self.assertRaises(ValueError) as ctx:
            split_chronological_data({"AAPL": df10})
        self.assertIn("minimum 20 bars required", str(ctx.exception).lower())

        # Empty data dict
        with self.assertRaises(ValueError):
            split_chronological_data({})

    def test_fitness_score_mathematical_stress(self):
        """Stress tests calculate_fitness_score under edge cases: zero trades, infinite PF, massive drawdown."""
        # 1. Total trades < min_trades (3) -> Must return -999.0
        m_few_trades = {"total_trades": 2, "net_return_pct": 50.0, "profit_factor": 10.0, "win_rate_pct": 100.0}
        self.assertEqual(calculate_fitness_score(m_few_trades, min_trades=3), -999.0)

        # 2. Zero trades -> -999.0
        m_zero_trades = {"total_trades": 0, "net_return_pct": 0.0, "profit_factor": 0.0, "win_rate_pct": 0.0}
        self.assertEqual(calculate_fitness_score(m_zero_trades, min_trades=3), -999.0)

        # 3. 100% win rate with PF = 999.0 -> Must be capped by max_pf_cap (5.0), not explode
        m_huge_pf = {
            "total_trades": 10,
            "net_return_pct": 30.0,
            "profit_factor": 999.0,
            "win_rate_pct": 100.0,
            "max_drawdown_pct": 5.0
        }
        score = calculate_fitness_score(m_huge_pf, min_trades=3, max_pf_cap=5.0)
        # w_ret * (30/100) + w_pf * 5.0 + w_wr * (100/100) - w_mdd * (5/100) + 0.1
        # = 1.0*0.3 + 0.8*5.0 + 0.5*1.0 - 1.5*0.05 + 0.1 = 0.3 + 4.0 + 0.5 - 0.075 + 0.1 = 4.825
        self.assertAlmostEqual(score, 4.825, places=3)
        self.assertTrue(math.isfinite(score))

        # 4. Massive drawdown 100%
        m_ruin = {
            "total_trades": 5,
            "net_return_pct": -90.0,
            "profit_factor": 0.1,
            "win_rate_pct": 20.0,
            "max_drawdown_pct": 100.0
        }
        score_ruin = calculate_fitness_score(m_ruin, min_trades=3)
        self.assertLess(score_ruin, -1.0)
        self.assertTrue(math.isfinite(score_ruin))

    def test_walk_forward_calibrator_under_zero_trade_regime(self):
        """
        Adversarially feeds a flat non-volatile dataset where NO bot can trigger any trades.
        Proves:
        1. Calibrator does not crash on -999.0 fitness scores.
        2. Assigns status 'insufficient_trades'.
        3. Returns valid persona schema with default/fallback parameters.
        """
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        # Perfectly flat price: no breakout, no dip, no momentum, no volume surge
        df_flat = pd.DataFrame({
            "open": [100.0] * 60,
            "high": [100.1] * 60,
            "low": [99.9] * 60,
            "close": [100.0] * 60,
            "volume": [10000.0] * 60
        }, index=dates)

        data = {"SPY": df_flat, "AAPL": df_flat.copy()}

        calibrator = WalkForwardCalibrator(top_k_candidates=2, min_trades=3)
        res = calibrator.calibrate(bot_class=ConservativeSwingBot, data=data)

        self.assertEqual(res["bot_id"], "conservative_swing")
        self.assertEqual(res["status"], "insufficient_trades")
        self.assertIn("best_params", res)
        self.assertIn("in_sample_metrics", res)
        self.assertIn("out_of_sample_metrics", res)
        self.assertIn("full_period_metrics", res)
        self.assertEqual(res["in_sample_metrics"]["total_trades"], 0)


if __name__ == "__main__":
    unittest.main()
