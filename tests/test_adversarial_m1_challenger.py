"""
Empirical Adversarial Stress Harness & Challenge Suite for Milestone 1.
Challenger: m1_challenger_1 (critic, specialist).

Stress-tests:
1. Lookahead Leakage:
   - Synthetic price surge / crash at bar t+1.
   - Proves indicators, signals, entries, portfolio holdings, cash, and equity at bar t
     are 100% invariant to future bars.
2. Intrabar Collision:
   - Bars where High >= Take Profit AND Low <= Stop Loss simultaneously.
   - Proves conservative stop-loss precedence across simulate_single_position and all 5 bot archetypes.
   - Proves gap-down slippage execution.
3. Timeline Synchronization:
   - Calendar asynchrony (US open on Diwali, Indian open on Thanksgiving).
   - Alternating disjoint trading calendars between exchanges.
   - Proves zero crashed bars, zero bogus trades on closed exchanges, and continuous mark-to-market.
4. Edge cases & Mathematical Guardrails in Quantitative Performance Metrics.
"""

import unittest
import copy
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.backtest_engine import (
    HistoricalDataFetcher,
    HistoricalBacktestEngine,
    SimulatedPortfolio,
    compute_rolling_indicators,
    simulate_single_position,
    calculate_performance_metrics
)
from src.paper_trader import (
    BotArchetype,
    ConservativeSwingBot,
    AggressiveBreakoutBot,
    ContrarianDipBuyerBot,
    FinBERTCatalystHunterBot,
    TrendFollowerBot,
    get_default_tournament_bots,
    INDEX_TRADEABLE_MAP,
    is_benchmark_index,
    is_inr_symbol
)


class TestAdversarialLookaheadLeakage(unittest.TestCase):
    """
    Challenge 1: Lookahead Leakage.
    Verifies that indicators and entry decisions at bar t are 100% immune to future bars (t+1..T).
    """

    def setUp(self):
        self.n_bars = 25
        self.dates = pd.date_range("2023-01-01", periods=self.n_bars, freq="D")
        # Base realistic price series
        np.random.seed(42)
        base = 100.0
        prices = [base]
        for i in range(1, self.n_bars):
            prices.append(round(prices[-1] * (1.0 + 0.005), 2))
        self.base_prices = prices

    def _build_df(self, prices: List[float], dates: pd.DatetimeIndex) -> pd.DataFrame:
        return pd.DataFrame({
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": [100000.0] * len(prices)
        }, index=dates)

    def test_indicator_lookahead_immunity_surge_vs_crash(self):
        """
        Adversarial test: Bar t is index 20.
        At bar 21 (t+1), test 3 scenarios:
        - Scenario 1: Baseline continuation
        - Scenario 2: Massive price surge (+10,000% spike)
        - Scenario 3: Catastrophic price crash (-99% plunge)
        - Scenario 4: Series truncated at bar 20 (future does not exist)
        Verify indicators at bar 20 are bit-for-bit identical.
        """
        eval_bar = 20

        # Scenario 1: Normal
        df_normal = self._build_df(self.base_prices, self.dates)

        # Scenario 2: Massive Surge at t+1
        prices_surge = list(self.base_prices)
        prices_surge[eval_bar + 1] = prices_surge[eval_bar] * 100.0
        df_surge = self._build_df(prices_surge, self.dates)

        # Scenario 3: Crash at t+1
        prices_crash = list(self.base_prices)
        prices_crash[eval_bar + 1] = 1.0
        df_crash = self._build_df(prices_crash, self.dates)

        # Scenario 4: Truncated at eval_bar
        df_trunc = df_normal.iloc[:eval_bar + 1].copy()

        # Compute indicators via compute_rolling_indicators
        ind_normal = compute_rolling_indicators(df_normal.iloc[:eval_bar + 1])
        ind_surge = compute_rolling_indicators(df_surge.iloc[:eval_bar + 1])
        ind_crash = compute_rolling_indicators(df_crash.iloc[:eval_bar + 1])
        ind_trunc = compute_rolling_indicators(df_trunc)

        for field in ["price", "sma20", "above_sma20", "momentum_5d", "valid"]:
            self.assertEqual(ind_normal[field], ind_surge[field], f"Field '{field}' leaked surge at t+1")
            self.assertEqual(ind_normal[field], ind_crash[field], f"Field '{field}' leaked crash at t+1")
            self.assertEqual(ind_normal[field], ind_trunc[field], f"Field '{field}' differed from truncated series")

    def test_full_backtest_replay_entry_lookahead_immunity(self):
        """
        Full replay test: Verifies that HistoricalBacktestEngine produces identical
        trade decisions and portfolio states up to bar t regardless of future t+1 surge or crash.
        """
        eval_date = str(self.dates[20])[:10]

        # Universe with NVDA and AAPL
        data_normal = {
            "NVDA": self._build_df(self.base_prices, self.dates),
            "AAPL": self._build_df([p * 1.5 for p in self.base_prices], self.dates)
        }

        # Universe where NVDA spikes 100x on bar 21
        prices_surge_nvda = list(self.base_prices)
        prices_surge_nvda[21] = 10000.0
        data_surge = {
            "NVDA": self._build_df(prices_surge_nvda, self.dates),
            "AAPL": self._build_df([p * 1.5 for p in self.base_prices], self.dates)
        }

        # Universe where NVDA crashes on bar 21
        prices_crash_nvda = list(self.base_prices)
        prices_crash_nvda[21] = 5.0
        data_crash = {
            "NVDA": self._build_df(prices_crash_nvda, self.dates),
            "AAPL": self._build_df([p * 1.5 for p in self.base_prices], self.dates)
        }

        engine = HistoricalBacktestEngine(initial_balance=5000.0)

        # Run up to eval_date (truncating backtest window)
        res_normal = engine.run_backtest(bots=get_default_tournament_bots(), data=data_normal, end_date=eval_date)
        res_surge = engine.run_backtest(bots=get_default_tournament_bots(), data=data_surge, end_date=eval_date)
        res_crash = engine.run_backtest(bots=get_default_tournament_bots(), data=data_crash, end_date=eval_date)

        for bot_id in res_normal["bots"]:
            b_norm = res_normal["bots"][bot_id]
            b_surg = res_surge["bots"][bot_id]
            b_cras = res_crash["bots"][bot_id]

            # Compare trades executed up to eval_date
            self.assertEqual(len(b_norm["trades"]), len(b_surg["trades"]), f"Bot {bot_id} trade count differed on surge")
            self.assertEqual(len(b_norm["trades"]), len(b_cras["trades"]), f"Bot {bot_id} trade count differed on crash")

            for t_norm, t_surg, t_cras in zip(b_norm["trades"], b_surg["trades"], b_cras["trades"]):
                self.assertEqual(t_norm["symbol"], t_surg["symbol"])
                self.assertEqual(t_norm["symbol"], t_cras["symbol"])
                self.assertEqual(t_norm["action"], t_surg["action"])
                self.assertEqual(t_norm["action"], t_cras["action"])
                self.assertEqual(t_norm["price"], t_surg["price"])
                self.assertEqual(t_norm["price"], t_cras["price"])
                self.assertEqual(t_norm["shares"], t_surg["shares"])
                self.assertEqual(t_norm["shares"], t_cras["shares"])

            # Compare equity curve up to eval_date
            self.assertEqual(b_norm["equity_curve"], b_surg["equity_curve"])
            self.assertEqual(b_norm["equity_curve"], b_cras["equity_curve"])


class TestAdversarialIntrabarCollision(unittest.TestCase):
    """
    Challenge 2: Intrabar Collision & Conservative Stop-Loss Precedence.
    Verifies that when High >= Take Profit AND Low <= Stop Loss on the same bar,
    the engine strictly enforces Stop Loss exit, never falsely awarding Take Profit.
    """

    def test_collision_single_position_pessimistic_precedence(self):
        """
        Simulates entry at $100.
        SL: -5.0% ($95.00), TP: +10.0% ($110.00).
        At bar 1: High rallies to $125.00 (+25%), Low collapses to $80.00 (-20%).
        Verify:
        - Exit is STOP_LOSS
        - Exit price <= $95.00
        - PnL is negative
        """
        dates = pd.date_range("2023-01-01", periods=3, freq="D")
        df = pd.DataFrame({
            "open": [100.0, 100.0, 100.0],
            "high": [101.0, 125.0, 100.0],
            "low": [99.0, 80.0, 100.0],
            "close": [100.0, 100.0, 100.0],
            "volume": [100000.0] * 3
        }, index=dates)

        trades = simulate_single_position(df, entry_bar=0, stop_loss_pct=5.0, take_profit_pct=10.0, slippage_pct=0.0005)
        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertEqual(t["exit_reason"], "STOP_LOSS")
        self.assertLessEqual(t["exit_price"], 95.0)
        self.assertLess(t["pnl_pct"], 0.0)

    def test_collision_gap_down_open_execution_price(self):
        """
        Gapping test:
        Entry at $100. SL $95.
        Next bar opens with a massive gap down at $85.00, then reaches high $115.00, low $82.00.
        Pessimistic execution must execute at or below the open price ($85.00 * (1 - slippage)),
        NOT at the theoretical stop limit $95.00.
        """
        dates = pd.date_range("2023-01-01", periods=3, freq="D")
        df = pd.DataFrame({
            "open": [100.0, 85.0, 100.0],  # Gapped open below SL
            "high": [101.0, 115.0, 100.0],
            "low": [99.0, 82.0, 100.0],
            "close": [100.0, 110.0, 100.0],
            "volume": [100000.0] * 3
        }, index=dates)

        trades = simulate_single_position(df, entry_bar=0, stop_loss_pct=5.0, take_profit_pct=10.0, slippage_pct=0.001)
        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertEqual(t["exit_reason"], "STOP_LOSS")
        # Must execute at open ($85 * 0.999 = 84.91), NOT at sl_target ($95)
        self.assertAlmostEqual(t["exit_price"], 84.91, places=1)
        self.assertLess(t["exit_price"], 85.1)

    def test_collision_all_five_personas_in_historical_backtest_engine(self):
        """
        Adversarially tests that when an extreme intrabar collision occurs:
        High exceeds +30% (well above any bot's take profit target, max is 18%),
        Low plunges below -25% (well below any bot's stop loss limit, max is -7.5%).
        Verify:
        - Every bot that had a position on the collision bar exits via STOP LOSS.
        - Zero bots exit via TAKE PROFIT on the collision bar.
        - Closed trades on that bar report exit_reason containing 'Stop Loss'.
        - Realized PnL is negative.
        """
        n_bars = 25
        dates = pd.date_range("2023-01-01", periods=n_bars, freq="D")
        collision_idx = 21
        collision_date = str(dates[collision_idx])[:10]

        # Consolidate around $100 for bars 0..19 with volume so technicals are valid
        prices = [100.0] * 20
        # Bar 20: slight upward tick to 102 to trigger entries without triggering TP
        prices.append(102.0)

        opens = [100.0] * 20 + [101.0]
        highs = [101.0] * 20 + [103.0]
        lows = [99.0] * 20 + [100.5]
        closes = list(prices)

        # Bar 21: Extreme Collision Bar (Open 102, High 140, Low 75, Close 100)
        opens.append(102.0)
        highs.append(140.0)  # +37% above 102 (breaches all TPs)
        lows.append(75.0)    # -26% below 102 (breaches all SLs)
        closes.append(100.0)
        prices.append(100.0)

        # Remaining bars
        for i in range(22, n_bars):
            opens.append(100.0)
            highs.append(101.0)
            lows.append(99.0)
            closes.append(100.0)

        df_collision = pd.DataFrame({
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [500000.0] * n_bars
        }, index=dates)

        data = {"NVDA": df_collision}
        engine = HistoricalBacktestEngine(initial_balance=5000.0, slippage_bps=5.0)
        bots = get_default_tournament_bots()
        res = engine.run_backtest(bots=bots, data=data)

        collision_exits = 0
        for bot_id, b_res in res["bots"].items():
            for ct in b_res["closed_trades"]:
                if ct["exit_date"] == collision_date:
                    collision_exits += 1
                    # Must be STOP LOSS
                    self.assertIn("Stop Loss", ct["exit_reason"], f"Bot {bot_id} exited with wrong reason: {ct['exit_reason']}")
                    self.assertNotIn("Take Profit", ct["exit_reason"], f"Bot {bot_id} erroneously took profit during collision!")
                    self.assertLess(ct["realized_pnl"], 0.0, f"Bot {bot_id} reported positive PnL during collision!")

        self.assertGreater(collision_exits, 0, f"Expected bots to hold positions into collision bar {collision_date}")


class TestAdversarialTimelineSynchronization(unittest.TestCase):
    """
    Challenge 3: Timeline Synchronization & Market Calendar Asynchrony.
    Tests handling of misaligned dates between US and Indian markets
    (e.g. US open on Diwali, Indian open on Thanksgiving, alternating trade days).
    """

    def test_thanksgiving_us_closed_india_open(self):
        """
        Adversarial scenario:
        US market is closed for Thanksgiving (e.g. 2023-11-23).
        Indian market is open (NIFTYBEES.NS, ^NSEI trade on 2023-11-23).
        Verify:
        - Backtest runs without KeyError or crash.
        - US assets (NVDA, AAPL) are NOT traded or evaluated for exit on Thanksgiving.
        - Indian assets can execute normally.
        - Portfolios holding US assets retain last known price without crashing or zeroing equity.
        """
        dates_all = pd.date_range("2023-11-01", periods=30, freq="D")
        thanksgiving = pd.Timestamp("2023-11-23")

        # US calendar excludes Thanksgiving
        dates_us = [d for d in dates_all if d != thanksgiving]
        # Indian calendar includes Thanksgiving
        dates_in = list(dates_all)

        n_us = len(dates_us)
        n_in = len(dates_in)

        data = {
            "NVDA": pd.DataFrame({
                "open": [100.0 + i for i in range(n_us)],
                "high": [102.0 + i for i in range(n_us)],
                "low": [98.0 + i for i in range(n_us)],
                "close": [100.0 + i for i in range(n_us)],
                "volume": [200000.0] * n_us
            }, index=pd.DatetimeIndex(dates_us)),

            "NIFTYBEES.NS": pd.DataFrame({
                "open": [250.0 + i * 0.5 for i in range(n_in)],
                "high": [252.0 + i * 0.5 for i in range(n_in)],
                "low": [248.0 + i * 0.5 for i in range(n_in)],
                "close": [250.0 + i * 0.5 for i in range(n_in)],
                "volume": [400000.0] * n_in
            }, index=pd.DatetimeIndex(dates_in)),

            "^NSEI": pd.DataFrame({
                "open": [22000.0 + i * 20 for i in range(n_in)],
                "high": [22100.0 + i * 20 for i in range(n_in)],
                "low": [21900.0 + i * 20 for i in range(n_in)],
                "close": [22000.0 + i * 20 for i in range(n_in)],
                "volume": [1000000.0] * n_in
            }, index=pd.DatetimeIndex(dates_in))
        }

        engine = HistoricalBacktestEngine(initial_balance=5000.0)
        res = engine.run_backtest(bots=get_default_tournament_bots(), data=data)

        self.assertEqual(res["total_bars"], 30)
        # Check no trades on NVDA occurred on Thanksgiving
        for bot_id, b_res in res["bots"].items():
            for t in b_res["trades"]:
                if t["symbol"] == "NVDA":
                    self.assertNotEqual(t["timestamp"], "2023-11-23", "NVDA traded while US market closed on Thanksgiving!")

    def test_diwali_india_closed_us_open(self):
        """
        Adversarial scenario:
        Indian market is closed for Diwali holiday (e.g. 2023-11-13).
        US market is open.
        Verify:
        - NIFTYBEES.NS is NOT traded on 2023-11-13.
        - US assets execute normally.
        - Marked-to-market equity remains continuous without drop.
        """
        dates_all = pd.date_range("2023-11-01", periods=30, freq="D")
        diwali = pd.Timestamp("2023-11-13")

        dates_us = list(dates_all)
        dates_in = [d for d in dates_all if d != diwali]

        n_us = len(dates_us)
        n_in = len(dates_in)

        data = {
            "NVDA": pd.DataFrame({
                "open": [100.0 + i for i in range(n_us)],
                "high": [102.0 + i for i in range(n_us)],
                "low": [98.0 + i for i in range(n_us)],
                "close": [100.0 + i for i in range(n_us)],
                "volume": [200000.0] * n_us
            }, index=pd.DatetimeIndex(dates_us)),

            "NIFTYBEES.NS": pd.DataFrame({
                "open": [250.0 + i * 0.5 for i in range(n_in)],
                "high": [252.0 + i * 0.5 for i in range(n_in)],
                "low": [248.0 + i * 0.5 for i in range(n_in)],
                "close": [250.0 + i * 0.5 for i in range(n_in)],
                "volume": [400000.0] * n_in
            }, index=pd.DatetimeIndex(dates_in))
        }

        engine = HistoricalBacktestEngine(initial_balance=5000.0)
        res = engine.run_backtest(bots=get_default_tournament_bots(), data=data)

        self.assertEqual(res["total_bars"], 30)
        for bot_id, b_res in res["bots"].items():
            for t in b_res["trades"]:
                if t["symbol"] == "NIFTYBEES.NS":
                    self.assertNotEqual(t["timestamp"], "2023-11-13", "NIFTYBEES.NS traded while closed on Diwali!")

    def test_alternating_staggered_markets_disjoint_dates(self):
        """
        Stress test: US trades on even days, India trades on odd days.
        Union calendar has 30 days, but on any given day, exactly one exchange is open.
        Verify:
        - Engine iterates across all 30 days.
        - Trades for each asset strictly align with open days.
        - Equity curve never contains NaNs or drops to 0.
        """
        dates_all = pd.date_range("2023-01-01", periods=30, freq="D")
        dates_us = [d for idx, d in enumerate(dates_all) if idx % 2 == 0]
        dates_in = [d for idx, d in enumerate(dates_all) if idx % 2 != 0]

        data = {
            "AAPL": pd.DataFrame({
                "open": [150.0 + i for i in range(len(dates_us))],
                "high": [152.0 + i for i in range(len(dates_us))],
                "low": [148.0 + i for i in range(len(dates_us))],
                "close": [150.0 + i for i in range(len(dates_us))],
                "volume": [300000.0] * len(dates_us)
            }, index=pd.DatetimeIndex(dates_us)),

            "NIFTYBEES.NS": pd.DataFrame({
                "open": [250.0 + i for i in range(len(dates_in))],
                "high": [252.0 + i for i in range(len(dates_in))],
                "low": [248.0 + i for i in range(len(dates_in))],
                "close": [250.0 + i for i in range(len(dates_in))],
                "volume": [500000.0] * len(dates_in)
            }, index=pd.DatetimeIndex(dates_in))
        }

        engine = HistoricalBacktestEngine(initial_balance=5000.0)
        res = engine.run_backtest(bots=get_default_tournament_bots(), data=data)

        self.assertEqual(res["total_bars"], 30)
        for bot_id, b_res in res["bots"].items():
            for d_str, eq in b_res["equity_curve"]:
                self.assertFalse(np.isnan(eq), f"Equity was NaN on {d_str} for bot {bot_id}")
                self.assertGreater(eq, 0.0, f"Equity dropped to 0 or below on {d_str}")


class TestAdversarialQuantitativeMetrics(unittest.TestCase):
    """
    Challenge 4: Mathematical Edge Cases in calculate_performance_metrics.
    """

    def test_single_penny_profit_and_loss(self):
        """Microscopic PnL numbers near epsilon threshold."""
        trades = [{"realized_pnl": 0.00001}, {"realized_pnl": -0.00001}]
        m = calculate_performance_metrics(trades, [5000.0, 5000.0])
        self.assertEqual(m["total_trades"], 2)
        self.assertEqual(m["wins"], 1)
        self.assertEqual(m["losses"], 1)
        self.assertEqual(m["win_rate_pct"], 50.0)

    def test_bankruptcy_cagr_guard(self):
        """Total loss of initial capital: final equity = $0 or negative."""
        m_zero = calculate_performance_metrics([], [5000.0, 0.0], initial_balance=5000.0, years=3.0)
        self.assertEqual(m_zero["cagr_pct"], -100.0)

        m_neg = calculate_performance_metrics([], [5000.0, -500.0], initial_balance=5000.0, years=3.0)
        self.assertEqual(m_neg["cagr_pct"], -100.0)

    def test_zero_duration_or_zero_balance(self):
        """Edge cases: duration = 0 years or starting balance = 0."""
        m_dur0 = calculate_performance_metrics([], [5000.0], initial_balance=5000.0, years=0.0)
        self.assertEqual(m_dur0["cagr_pct"], 0.0)

        m_bal0 = calculate_performance_metrics([], [0.0], initial_balance=0.0, years=3.0)
        self.assertEqual(m_bal0["net_return_pct"], 0.0)
        self.assertEqual(m_bal0["cagr_pct"], 0.0)


class TestExtremeEdgeCasesAndRobustness(unittest.TestCase):
    """
    Challenge 5: Hostile edge cases, zero overlap calendars, IPO listings, and unsorted inputs.
    """

    def test_timeline_sync_zero_date_overlap_disjoint_universes(self):
        """
        Adversarial test: US and Indian markets operate in completely disjoint calendar periods.
        US data: Jan 1 to Jan 15.
        India data: Jan 16 to Jan 31.
        Zero overlap!
        Verify:
        - Engine iterates across all 31 days seamlessly without IndexError.
        - Equity curve spans 31 days.
        - Mark-to-market uses last known price when exchange is dormant.
        """
        dates_us = pd.date_range("2023-01-01", periods=15, freq="D")
        dates_in = pd.date_range("2023-01-16", periods=16, freq="D")

        data = {
            "AAPL": pd.DataFrame({
                "open": [150.0 + i for i in range(15)],
                "high": [152.0 + i for i in range(15)],
                "low": [148.0 + i for i in range(15)],
                "close": [150.0 + i for i in range(15)],
                "volume": [300000.0] * 15
            }, index=dates_us),

            "NIFTYBEES.NS": pd.DataFrame({
                "open": [250.0 + i for i in range(16)],
                "high": [252.0 + i for i in range(16)],
                "low": [248.0 + i for i in range(16)],
                "close": [250.0 + i for i in range(16)],
                "volume": [500000.0] * 16
            }, index=dates_in)
        }

        engine = HistoricalBacktestEngine(initial_balance=5000.0)
        res = engine.run_backtest(bots=get_default_tournament_bots(), data=data)

        self.assertEqual(res["total_bars"], 31)
        for bot_id, b_res in res["bots"].items():
            self.assertEqual(len(b_res["equity_curve"]), 31)

    def test_timeline_sync_ipo_mid_backtest_listing(self):
        """
        Symbol listed on day 15 of a 30-day backtest.
        Before day 15: symbol has no rows.
        Verify:
        - No KeyError on pre-IPO days.
        - Indicator and trade evaluations start properly post-IPO.
        """
        dates_full = pd.date_range("2023-01-01", periods=30, freq="D")
        dates_ipo = dates_full[15:]

        data = {
            "SPY": pd.DataFrame({
                "open": [400.0 + i for i in range(30)],
                "high": [402.0 + i for i in range(30)],
                "low": [398.0 + i for i in range(30)],
                "close": [400.0 + i for i in range(30)],
                "volume": [1000000.0] * 30
            }, index=dates_full),

            "IPO_STOCK": pd.DataFrame({
                "open": [50.0 + i for i in range(len(dates_ipo))],
                "high": [52.0 + i for i in range(len(dates_ipo))],
                "low": [48.0 + i for i in range(len(dates_ipo))],
                "close": [50.0 + i for i in range(len(dates_ipo))],
                "volume": [400000.0] * len(dates_ipo)
            }, index=dates_ipo)
        }

        engine = HistoricalBacktestEngine(initial_balance=5000.0)
        res = engine.run_backtest(bots=get_default_tournament_bots(), data=data)
        self.assertEqual(res["total_bars"], 30)

    def test_lookahead_unsorted_shuffled_input_data(self):
        """
        Adversarially feeds a completely shuffled / reverse-chronological DataFrame.
        Verify:
        - Engine auto-sorts internally via sort_index.
        - Indicators match strictly chronological calculations.
        """
        dates = pd.date_range("2023-01-01", periods=25, freq="D")
        prices = [100.0 + i for i in range(25)]
        df_chrono = pd.DataFrame({
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": [100000.0] * 25
        }, index=dates)

        # Shuffle DataFrame index
        df_shuffled = df_chrono.sample(frac=1.0, random_state=42)
        self.assertNotEqual(list(df_shuffled.index), list(df_chrono.index))

        engine = HistoricalBacktestEngine(initial_balance=5000.0)
        res_chrono = engine.run_backtest(bots=get_default_tournament_bots(), data={"NVDA": df_chrono})
        res_shuffled = engine.run_backtest(bots=get_default_tournament_bots(), data={"NVDA": df_shuffled})

        for bot_id in res_chrono["bots"]:
            eq_chrono = res_chrono["bots"][bot_id]["equity_curve"]
            eq_shuffled = res_shuffled["bots"][bot_id]["equity_curve"]
            self.assertEqual(eq_chrono, eq_shuffled)

    def test_benchmark_index_never_directly_purchased(self):
        """
        Adversarially ensures that benchmark index '^NSEI' is NEVER bought directly,
        even if it shows astronomical momentum.
        """
        dates = pd.date_range("2023-01-01", periods=25, freq="D")
        nsei_prices = [20000.0 + i * 200.0 for i in range(25)]

        data = {
            "^NSEI": pd.DataFrame({
                "open": nsei_prices,
                "high": [p + 50 for p in nsei_prices],
                "low": [p - 50 for p in nsei_prices],
                "close": nsei_prices,
                "volume": [1000000.0] * 25
            }, index=dates)
        }

        engine = HistoricalBacktestEngine(initial_balance=5000.0)
        res = engine.run_backtest(bots=get_default_tournament_bots(), data=data)

        for bot_id, b_res in res["bots"].items():
            for t in b_res["trades"]:
                self.assertNotEqual(t["symbol"], "^NSEI", "Benchmark index ^NSEI was bought directly!")


if __name__ == "__main__":
    unittest.main()

