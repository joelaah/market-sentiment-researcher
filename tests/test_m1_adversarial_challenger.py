"""
Adversarial Stress Test Suite for Milestone 1: Historical Backtest Engine (src/backtest_engine.py).
Written by m1_challenger_2 (EMPIRICAL CHALLENGER).

Tests:
1. Metric extreme boundaries:
   - Zero trades (empty trade log) -> Win Rate 0.0, Profit Factor 0.0, Net Return 0.0, CAGR 0.0, no NaN or ZeroDivisionError
   - 100% winning trades (0 losses) -> Profit Factor finite (999.0) and JSON serializable
   - 100% losing trades (0 wins) -> Profit Factor 0.0, MDD accurate
   - Negative terminal equity (equity <= 0) -> CAGR doesn't throw complex numbers or mathematical errors
2. Indian asset routing:
   - Verify ^NSEI trades execute at NIFTYBEES.NS ETF price (~264) rather than index points (~25,000)
   - Verify cash deductions match ETF share cost rather than index points
   - Verify portfolio valuation and drawdown are not poisoned by index price
"""

import unittest
import json
import math
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest_engine import (
    calculate_performance_metrics,
    HistoricalBacktestEngine,
    HistoricalDataFetcher,
    SimulatedPortfolio,
    compute_rolling_indicators
)
from src.paper_trader import (
    ConservativeSwingBot,
    AggressiveBreakoutBot,
    ContrarianDipBuyerBot,
    FinBERTCatalystHunterBot,
    TrendFollowerBot,
    get_default_tournament_bots,
    is_benchmark_index,
    is_inr_symbol,
    INDEX_TRADEABLE_MAP,
    BorderlistGuardrail
)


class TestMetricExtremeBoundaries(unittest.TestCase):
    """Stress tests boundary conditions in calculate_performance_metrics."""

    def test_zero_trades_empty_equity_curve(self):
        """Zero trades and empty equity curve must return 0.0 metrics without NaN/ZeroDivisionError."""
        m = calculate_performance_metrics(trades=[], equity_curve=[], initial_balance=5000.0, years=3.0)
        self.assertEqual(m["win_rate_pct"], 0.0)
        self.assertEqual(m["profit_factor"], 0.0)
        self.assertEqual(m["net_return_pct"], 0.0)
        self.assertEqual(m["cagr_pct"], 0.0)
        self.assertEqual(m["max_drawdown_pct"], 0.0)
        self.assertEqual(m["total_trades"], 0)
        self.assertEqual(m["wins"], 0)
        self.assertEqual(m["losses"], 0)
        self.assertEqual(m["gross_profit"], 0.0)
        self.assertEqual(m["gross_loss"], 0.0)

        # JSON Serialization Verification
        serialized = json.dumps(m)
        self.assertNotIn("NaN", serialized)
        self.assertNotIn("Infinity", serialized)

    def test_zero_trades_flat_equity_curve(self):
        """Zero trades with flat equity curve must yield exact 0.0 return and CAGR."""
        m = calculate_performance_metrics(trades=[], equity_curve=[5000.0, 5000.0, 5000.0], initial_balance=5000.0, years=3.0)
        self.assertEqual(m["win_rate_pct"], 0.0)
        self.assertEqual(m["profit_factor"], 0.0)
        self.assertEqual(m["net_return_pct"], 0.0)
        self.assertEqual(m["cagr_pct"], 0.0)
        self.assertEqual(m["max_drawdown_pct"], 0.0)

    def test_breakeven_trades_only(self):
        """Trades with realized_pnl == 0.0 must not count as wins or losses and not divide by zero."""
        trades = [{"realized_pnl": 0.0}, {"realized_pnl": 0.0}]
        m = calculate_performance_metrics(trades=trades, equity_curve=[5000.0, 5000.0], initial_balance=5000.0, years=1.0)
        self.assertEqual(m["win_rate_pct"], 0.0)
        self.assertEqual(m["profit_factor"], 0.0)
        self.assertEqual(m["total_trades"], 0)

    def test_100_percent_winning_trades_profit_factor_finite_and_json(self):
        """100% win rate (0 losses) must have finite profit_factor (999.0) and be strictly JSON-serializable."""
        trades = [
            {"realized_pnl": 120.50},
            {"realized_pnl": 340.00},
            {"realized_pnl": 85.25}
        ]
        equity_curve = [5000.0, 5120.50, 5460.50, 5545.75]
        m = calculate_performance_metrics(trades=trades, equity_curve=equity_curve, initial_balance=5000.0, years=1.0)

        self.assertEqual(m["win_rate_pct"], 100.0)
        self.assertEqual(m["wins"], 3)
        self.assertEqual(m["losses"], 0)
        self.assertEqual(m["gross_loss"], 0.0)
        self.assertEqual(m["gross_profit"], 545.75)
        # Must be capped at 999.0 and MUST NOT be float('inf')
        self.assertEqual(m["profit_factor"], 999.0)
        self.assertFalse(math.isinf(m["profit_factor"]))
        self.assertFalse(math.isnan(m["profit_factor"]))

        # Strictly verify json.dumps does not produce Infinity or fail
        serialized = json.dumps(m)
        self.assertIn('"profit_factor": 999.0', serialized)
        deserialized = json.loads(serialized)
        self.assertEqual(deserialized["profit_factor"], 999.0)

    def test_100_percent_losing_trades_profit_factor_and_mdd(self):
        """100% losing trades (0 wins) must return profit_factor 0.0, win_rate 0.0, and accurate MDD."""
        trades = [
            {"realized_pnl": -200.0},
            {"realized_pnl": -300.0},
            {"realized_pnl": -500.0}
        ]
        # Peak was 5000, trough is 4000
        equity_curve = [5000.0, 4800.0, 4500.0, 4000.0]
        m = calculate_performance_metrics(trades=trades, equity_curve=equity_curve, initial_balance=5000.0, years=2.0)

        self.assertEqual(m["win_rate_pct"], 0.0)
        self.assertEqual(m["wins"], 0)
        self.assertEqual(m["losses"], 3)
        self.assertEqual(m["profit_factor"], 0.0)
        self.assertEqual(m["gross_profit"], 0.0)
        self.assertEqual(m["gross_loss"], 1000.0)
        # MDD = (5000 - 4000) / 5000 * 100 = 20.0%
        self.assertEqual(m["max_drawdown_pct"], 20.0)
        self.assertEqual(m["net_return_pct"], -20.0)

    def test_negative_terminal_equity_no_complex_number_crash(self):
        """
        Adversarial probe:
        When terminal equity is <= 0 (e.g. bankruptcy or account wipeout),
        standard CAGR formula (final / initial) ** (1/years) - 1 raises complex numbers in Python:
        e.g. (-100 / 5000) ** (1/3) yields complex (0.1357 + 0.2351j),
        which fails round() with TypeError: type complex doesn't define __round__ method.
        Must return -100.0% cleanly as a float.
        """
        # Test terminal equity == 0.0
        m_zero = calculate_performance_metrics(trades=[], equity_curve=[5000.0, 2000.0, 0.0], initial_balance=5000.0, years=3.0)
        self.assertEqual(m_zero["cagr_pct"], -100.0)
        self.assertIsInstance(m_zero["cagr_pct"], float)

        # Test terminal equity < 0 (negative)
        m_neg = calculate_performance_metrics(trades=[], equity_curve=[5000.0, 1000.0, -250.0], initial_balance=5000.0, years=3.0)
        self.assertEqual(m_neg["cagr_pct"], -100.0)
        self.assertIsInstance(m_neg["cagr_pct"], float)
        self.assertFalse(isinstance(m_neg["cagr_pct"], complex))

        # Test fractional years (e.g. 0.333 years)
        m_frac = calculate_performance_metrics(trades=[], equity_curve=[5000.0, -100.0], initial_balance=5000.0, years=0.333)
        self.assertEqual(m_frac["cagr_pct"], -100.0)
        self.assertIsInstance(m_frac["cagr_pct"], float)

        # Ensure JSON-compliant
        serialized = json.dumps(m_neg)
        self.assertIn('"cagr_pct": -100.0', serialized)

    def test_initial_balance_zero_or_negative(self):
        """Initial balance <= 0 must not cause ZeroDivisionError in CAGR or Net Return."""
        m_zero_init = calculate_performance_metrics(trades=[], equity_curve=[0.0], initial_balance=0.0, years=3.0)
        self.assertEqual(m_zero_init["net_return_pct"], 0.0)
        self.assertEqual(m_zero_init["cagr_pct"], 0.0)


class TestIndianAssetRoutingAndExecution(unittest.TestCase):
    """
    Empirical stress tests for Indian asset routing:
    ^NSEI trades must execute at NIFTYBEES.NS ETF price (~264)
    and cash deductions must match ETF share cost rather than index points (~25,000).
    """

    def setUp(self):
        self.engine = HistoricalBacktestEngine(initial_balance=5000.0, slippage_bps=5.0)

    def test_benchmark_mapping_definitions(self):
        """Verifies symbol classification and mapping contracts."""
        self.assertTrue(is_benchmark_index("^NSEI"))
        self.assertFalse(is_benchmark_index("NIFTYBEES.NS"))
        self.assertEqual(INDEX_TRADEABLE_MAP.get("^NSEI"), "NIFTYBEES.NS")
        self.assertTrue(is_inr_symbol("^NSEI"))
        self.assertTrue(is_inr_symbol("NIFTYBEES.NS"))

    def test_candidate_ranking_with_index_and_etf(self):
        """
        Adversarial Probe:
        Examines how HistoricalBacktestEngine handles ^NSEI and NIFTYBEES.NS during candidate ranking.
        Does ^NSEI get evaluated as a tradeable target, or does NIFTYBEES.NS trade at ETF price?
        """
        dates = pd.date_range("2023-01-01", periods=25, freq="D")

        # ^NSEI prices: Index levels ~23,000 - 25,000
        nsei_prices = [23000.0 + i * 80.0 for i in range(25)]
        # NIFTYBEES.NS prices: ETF levels ~260 - 280
        niftybees_prices = [260.0 + i * 0.8 for i in range(25)]

        data = {
            "^NSEI": pd.DataFrame({
                "open": nsei_prices,
                "high": [p + 50.0 for p in nsei_prices],
                "low": [p - 30.0 for p in nsei_prices],
                "close": nsei_prices,
                "volume": [2000000.0] * 25
            }, index=dates),
            "NIFTYBEES.NS": pd.DataFrame({
                "open": niftybees_prices,
                "high": [p + 1.0 for p in niftybees_prices],
                "low": [p - 0.5 for p in niftybees_prices],
                "close": niftybees_prices,
                "volume": [500000.0] * 25
            }, index=dates)
        }

        # Run single bot backtest (TrendFollowerBot)
        bot = TrendFollowerBot()
        results = self.engine.run_backtest(bots=[bot], data=data)
        bot_res = results["bots"][bot.bot_id]

        trades = bot_res["trades"]
        positions = bot_res["open_positions"]

        print(f"\n[Audit] Bot Trades Count: {len(trades)}")
        for t in trades:
            print(f"  Trade: symbol={t['symbol']}, action={t['action']}, price={t['price']}, total_value={t['total_value']}, shares={t['shares']}")

        print(f"[Audit] Bot Open Positions: {list(positions.keys())}")
        for sym, pos in positions.items():
            print(f"  Position: symbol={pos['symbol']}, shares={pos['shares']}, avg_buy_price={pos['avg_buy_price']}")

        # VERIFICATIONS:
        # 1. No trade or position should ever have symbol '^NSEI' with execution price ~23,000+
        for t in trades:
            self.assertNotEqual(t["price"], 23000.0, "FATAL: Trade price recorded at index level!")
            if t["symbol"] == "NIFTYBEES.NS":
                self.assertLess(t["price"], 1000.0, f"ETF price should be around ₹260-280, got {t['price']}")
                self.assertGreater(t["price"], 200.0)

        for sym, pos in positions.items():
            self.assertNotEqual(sym, "^NSEI", "Position should never hold index directly")
            if sym == "NIFTYBEES.NS":
                self.assertLess(pos["avg_buy_price"], 1000.0)
                self.assertGreater(pos["avg_buy_price"], 200.0)

        # 2. Portfolio Cash Accounting Verification:
        # If any buy happened, cash deduction must match ETF share cost (allocation ~$500 - $1000), NOT $23,000+
        if trades:
            buy_trade = trades[0]
            expected_cost = round(buy_trade["shares"] * buy_trade["price"], 2)
            self.assertAlmostEqual(buy_trade["total_value"], expected_cost, places=2)
            self.assertLess(buy_trade["total_value"], 5000.0)
            self.assertGreaterEqual(bot_res["final_cash"], 0.0, "Cash should never become negative due to index price error")

    def test_routing_when_candidate_is_index(self):
        """
        Adversarial Probe for Line 878-879:
        Simulate an engine pass where qualified_candidates has '^NSEI'.
        Verify that INDEX_TRADEABLE_MAP.get(candidate, candidate) resolves to 'NIFTYBEES.NS',
        fetches 'NIFTYBEES.NS' bar price (not 25,000), executes at ETF price (~264),
        and records ETF shares and cash deduction correctly.
        """
        port = SimulatedPortfolio(ConservativeSwingBot(), initial_balance=5000.0)
        bar_prices = {"^NSEI": 24500.0, "NIFTYBEES.NS": 265.0}
        latest_known_prices = dict(bar_prices)
        b = port.bot

        # Simulate the execution loop block from HistoricalBacktestEngine lines 878-935
        candidate = "^NSEI"
        tradeable_target = INDEX_TRADEABLE_MAP.get(candidate, candidate)
        self.assertEqual(tradeable_target, "NIFTYBEES.NS")

        target_price = bar_prices.get(tradeable_target)
        self.assertEqual(target_price, 265.0)

        slippage_pct = 0.0005  # 5 bps
        exec_buy_price = round(target_price * (1.0 + slippage_pct), 2)
        self.assertEqual(exec_buy_price, 265.13)

        current_equity = port.get_total_equity(latest_known_prices)
        allocation = round(min(current_equity * b.allocation_pct, b.max_trade_dollars), 2)
        # b.allocation_pct is 0.15 (15% of $5000 = $750)
        self.assertEqual(allocation, 750.0)

        shares_to_buy = round(allocation / exec_buy_price, 4)
        actual_cost = round(shares_to_buy * exec_buy_price, 2)

        self.assertAlmostEqual(shares_to_buy, round(750.0 / 265.13, 4))
        self.assertAlmostEqual(actual_cost, 750.00, delta=0.50)

        port.cash = round(port.cash - actual_cost, 2)
        port.positions[tradeable_target] = {
            "symbol": tradeable_target,
            "shares": shares_to_buy,
            "avg_buy_price": exec_buy_price,
            "entry_date": "2023-01-15",
            "entry_bar": 10
        }

        # Assertions on portfolio state
        self.assertIn("NIFTYBEES.NS", port.positions)
        self.assertNotIn("^NSEI", port.positions)
        self.assertGreater(port.cash, 4200.0)
        self.assertEqual(port.positions["NIFTYBEES.NS"]["avg_buy_price"], 265.13)
        self.assertNotEqual(port.positions["NIFTYBEES.NS"]["avg_buy_price"], 24500.0)

    def test_indian_asset_roundtrip_exit_and_valuation(self):
        """
        Adversarial Probe:
        Simulate full roundtrip:
        1. Buy NIFTYBEES.NS at ₹260.0
        2. Exit at ₹290.0 (+11.5% take profit)
        3. Verify proceeds = shares * 290.0, realized PnL > 0,
           and no artificial 99% drawdown occurs (which would happen if bought at 25,000 and sold at 290).
        """
        n_bars = 30
        dates = pd.date_range("2023-01-01", periods=n_bars, freq="D")
        # Bars 0-20: steady uptrend from 250 to 290. Bar 20 triggers entry at ~290.
        # Bars 21-29: surge to 350 (+20% > +15% take profit target), triggering take-profit exit.
        etf_prices = [250.0 + i * 2.0 for i in range(21)] + [292.0 + (i - 20) * 6.0 for i in range(21, n_bars)]
        index_prices = [22000.0 + i * 150.0 for i in range(n_bars)]

        data = {
            "NIFTYBEES.NS": pd.DataFrame({
                "open": etf_prices,
                "high": [p + 4.0 for p in etf_prices],
                "low": [p - 1.0 for p in etf_prices],
                "close": etf_prices,
                "volume": [500000.0] * n_bars
            }, index=dates),
            "^NSEI": pd.DataFrame({
                "open": index_prices,
                "high": [p + 100.0 for p in index_prices],
                "low": [p - 50.0 for p in index_prices],
                "close": index_prices,
                "volume": [2000000.0] * n_bars
            }, index=dates)
        }

        engine = HistoricalBacktestEngine(initial_balance=5000.0, slippage_bps=5.0)
        bot = TrendFollowerBot()
        results = engine.run_backtest(bots=[bot], data=data)
        bot_res = results["bots"][bot.bot_id]

        trades = bot_res["trades"]
        closed_trades = bot_res["closed_trades"]

        print(f"\n[Roundtrip Audit] Closed Trades: {len(closed_trades)}")
        for ct in closed_trades:
            print(f"  Closed Trade: sym={ct['symbol']}, entry_price={ct['entry_price']}, exit_price={ct['exit_price']}, realized_pnl={ct['realized_pnl']}, pnl_pct={ct['pnl_pct']}%")

        # Must have completed a profitable trade
        if closed_trades:
            for ct in closed_trades:
                self.assertEqual(ct["symbol"], "NIFTYBEES.NS")
                self.assertLess(ct["entry_price"], 500.0)
                self.assertLess(ct["exit_price"], 500.0)
                self.assertGreater(ct["realized_pnl"], 0.0)
                self.assertGreater(ct["pnl_pct"], 0.0)

        # Max drawdown must NOT be ~99%
        metrics = bot_res["metrics"]
        self.assertLess(metrics["max_drawdown_pct"], 20.0, f"Max drawdown anomaly: {metrics['max_drawdown_pct']}%")
        self.assertGreater(bot_res["final_equity"], 5000.0, "Should have positive net profit")

    def test_audit_line_737_benchmark_skip_in_backtest_engine(self):
        """
        Adversarial Challenge on Line 737:
        In HistoricalBacktestEngine.run_backtest:
            for sym, price in bar_prices.items():
                if is_benchmark_index(sym):
                    continue
        This skips ^NSEI from qualified_candidates completely.
        Verify what happens when data contains ONLY ^NSEI vs ^NSEI + NIFTYBEES.NS.
        """
        dates = pd.date_range("2023-01-01", periods=25, freq="D")
        nsei_prices = [23000.0 + i * 80.0 for i in range(25)]

        # Dataset with ONLY ^NSEI
        data_only_index = {
            "^NSEI": pd.DataFrame({
                "open": nsei_prices,
                "high": [p + 50.0 for p in nsei_prices],
                "low": [p - 30.0 for p in nsei_prices],
                "close": nsei_prices,
                "volume": [2000000.0] * 25
            }, index=dates)
        }

        # Because ^NSEI is skipped at line 737, 0 trades can execute
        res_only_index = self.engine.run_backtest(bots=[TrendFollowerBot()], data=data_only_index)
        bot_res = res_only_index["bots"]["trend_follower"]
        self.assertEqual(len(bot_res["trades"]), 0)
        self.assertEqual(bot_res["final_cash"], 5000.0)


if __name__ == "__main__":
    unittest.main()

