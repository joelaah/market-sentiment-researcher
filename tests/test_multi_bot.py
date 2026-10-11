"""
Unit and integration tests for Multi-Bot Tournament Architecture.
Tests archetype rules, portfolio isolation, shared evaluation, win rate calculations,
and backward compatibility.
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import shutil
from pathlib import Path

from src.paper_trader import (
    BotArchetype,
    ConservativeSwingBot,
    AggressiveBreakoutBot,
    ContrarianDipBuyerBot,
    FinBERTCatalystHunterBot,
    TrendFollowerBot,
    get_default_tournament_bots,
    MarketSnapshot,
    TournamentDataManager,
    MultiBotTournamentEngine,
    PaperTradingEngine,
    BorderlistGuardrail,
    calculate_probability_weighted_upside,
    extract_candidate_tickers_from_articles,
    MIN_SHARE_PRICE,
    MIN_DAILY_VOLUME,
    MIN_MARKET_CAP,
    LOCAL_STATE_FILE
)


class TestBotArchetypes(unittest.TestCase):
    def test_default_roster(self):
        bots = get_default_tournament_bots()
        self.assertEqual(len(bots), 5)
        bot_ids = [b.bot_id for b in bots]
        self.assertEqual(len(set(bot_ids)), 5, "All bots must have unique bot_ids")
        for b in bots:
            self.assertEqual(b.starting_balance, 5000.00)

    def test_conservative_swing_rules(self):
        bot = ConservativeSwingBot()
        tech_good = {"valid": True, "price": 100.0, "sma20": 95.0, "above_sma20": True, "momentum_5d": 1.5}
        tech_bad = {"valid": True, "price": 90.0, "sma20": 95.0, "above_sma20": False, "momentum_5d": -2.0}
        bert_high = {"news_score": 2.5, "net_score": 0.62, "positive": 0.8, "negative": 0.1, "info": "FinBERT[+]"}
        bert_low = {"news_score": 0.5, "net_score": 0.12, "positive": 0.4, "negative": 0.2, "info": "FinBERT[neutral]"}
        arts = [{"title": "Strong Earnings"}]

        # Good technicals and high FinBERT should trigger
        should_enter, conv, rationale = bot.evaluate_entry("NVDA", tech_good, bert_high, arts, [])
        self.assertTrue(should_enter)
        self.assertGreaterEqual(conv, 7.0)

        # Bad technicals (below SMA20) should NOT trigger
        should_enter2, _, _ = bot.evaluate_entry("NVDA", tech_bad, bert_high, arts, [])
        self.assertFalse(should_enter2)

        # Low FinBERT score should NOT trigger
        should_enter3, _, _ = bot.evaluate_entry("NVDA", tech_good, bert_low, arts, [])
        self.assertFalse(should_enter3)

        # Exits
        sold_tp, _ = bot.evaluate_exit({"pnl_pct": 12.5}, tech_good, bert_high)
        self.assertTrue(sold_tp)

        sold_sl, _ = bot.evaluate_exit({"pnl_pct": -8.0}, tech_good, bert_high)
        self.assertTrue(sold_sl)

    def test_aggressive_breakout_rules(self):
        bot = AggressiveBreakoutBot()
        tech_breakout = {"valid": True, "price": 150.0, "sma20": 140.0, "above_sma20": True, "momentum_5d": 3.2}
        tech_flat = {"valid": True, "price": 150.0, "sma20": 140.0, "above_sma20": True, "momentum_5d": 0.5}
        bert_pos = {"net_score": 0.25, "info": "FinBERT[+]"}

        # Strong velocity (>= 2.0%) triggers breakout
        should_enter, _, _ = bot.evaluate_entry("TSLA", tech_breakout, bert_pos, [], [])
        self.assertTrue(should_enter)

        # Flat velocity does not trigger
        should_enter_flat, _, _ = bot.evaluate_entry("TSLA", tech_flat, bert_pos, [], [])
        self.assertFalse(should_enter_flat)

        # Exits on 18% target or 5% stop
        self.assertTrue(bot.evaluate_exit({"pnl_pct": 18.5}, tech_breakout, bert_pos)[0])
        self.assertTrue(bot.evaluate_exit({"pnl_pct": -5.1}, tech_breakout, bert_pos)[0])

    def test_contrarian_dip_buyer_rules(self):
        bot = ContrarianDipBuyerBot()
        tech_dip = {"valid": True, "price": 98.0, "sma20": 100.0, "above_sma20": False, "momentum_5d": -3.5}
        tech_crash = {"valid": True, "price": 80.0, "sma20": 100.0, "above_sma20": False, "momentum_5d": -15.0}
        bert_stable = {"net_score": 0.05, "info": "FinBERT[neutral]"}

        # Dip between -1.5% and -8.0% triggers
        should_enter, _, _ = bot.evaluate_entry("AAPL", tech_dip, bert_stable, [], [])
        self.assertTrue(should_enter)

        # Freefall crash (-15%) does not trigger
        should_enter_crash, _, _ = bot.evaluate_entry("AAPL", tech_crash, bert_stable, [], [])
        self.assertFalse(should_enter_crash)

    def test_finbert_catalyst_hunter_rules(self):
        bot = FinBERTCatalystHunterBot()
        tech = {"valid": True, "price": 50.0, "sma20": 55.0, "above_sma20": False, "momentum_5d": -1.0}
        bert_strong = {"positive": 0.85, "net_score": 0.75, "info": "FinBERT[Strong Bull]", "headline": "New Contract"}
        arts = [{"title": "Massive multi-billion deal"}]

        # Triggers even if technicals are lagging due to pure sentiment catalyst
        should_enter, _, _ = bot.evaluate_entry("AMD", tech, bert_strong, arts, [])
        self.assertTrue(should_enter)

        # Exits if sentiment flips bearish
        sold_flip, _ = bot.evaluate_exit({"pnl_pct": 1.0}, tech, {"net_score": -0.6, "info": "FinBERT[-]"})
        self.assertTrue(sold_flip)

    def test_trend_follower_rules(self):
        bot = TrendFollowerBot()
        tech_trend = {"valid": True, "price": 200.0, "sma20": 190.0, "above_sma20": True, "momentum_5d": 1.0}
        bert_neutral = {"net_score": 0.05, "info": "Neutral"}

        should_enter, _, _ = bot.evaluate_entry("MSFT", tech_trend, bert_neutral, [], [])
        self.assertTrue(should_enter)

        # Trend break exit
        tech_broken = {"valid": True, "price": 185.0, "sma20": 190.0, "above_sma20": False, "momentum_5d": -3.0}
        sold, _ = bot.evaluate_exit({"pnl_pct": 2.0}, tech_broken, bert_neutral)
        self.assertTrue(sold)


class TestPortfolioIsolationAndStorage(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(__file__).resolve().parent / "test_data"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.mock_file = self.test_dir / "test_state.json"

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_independent_cash_and_positions(self):
        # Create an engine with no remote client (pure local isolated storage)
        with patch("src.paper_trader.LOCAL_STATE_FILE", self.mock_file):
            engine = MultiBotTournamentEngine(client=None)

            # Check initial state: all bots have $5,000 cash
            for bot_id in engine.bots:
                cash, init = engine.storage.get_portfolio_balances(bot_id)
                self.assertEqual(cash, 5000.00)
                self.assertEqual(init, 5000.00)

            # Bot 1 (Conservative Swing) buys 1 share of NVDA for $100
            success = engine.execute_buy(
                bot_id="conservative_swing",
                symbol="NVDA",
                amount_dollars=500.00,
                rationale="Test buy",
                current_price=100.00
            )
            self.assertTrue(success)

            # Check Bot 1 cash decreased
            cash_b1, _ = engine.storage.get_portfolio_balances("conservative_swing")
            self.assertEqual(cash_b1, 4500.00)
            pos_b1 = engine.storage.get_positions("conservative_swing")
            self.assertEqual(len(pos_b1), 1)
            self.assertEqual(pos_b1[0]["symbol"], "NVDA")

            # Check Bot 2 (Aggressive Breakout) cash is COMPLETELY untouched ($5,000.00)
            cash_b2, _ = engine.storage.get_portfolio_balances("aggressive_breakout")
            self.assertEqual(cash_b2, 5000.00)
            pos_b2 = engine.storage.get_positions("aggressive_breakout")
            self.assertEqual(len(pos_b2), 0)

            # Bot 1 sells NVDA for $120 (+20%)
            sell_success = engine.execute_sell(
                bot_id="conservative_swing",
                symbol="NVDA",
                shares=5.0,
                rationale="Take profit test",
                current_price=120.00
            )
            self.assertTrue(sell_success)

            # Bot 1 cash is now $4,500 + $600 = $5,100
            cash_b1_after, _ = engine.storage.get_portfolio_balances("conservative_swing")
            self.assertEqual(cash_b1_after, 5100.00)
            self.assertEqual(len(engine.storage.get_positions("conservative_swing")), 0)

            # Status check verifies 100% win rate (1/1)
            status = engine.get_bot_status("conservative_swing")
            self.assertEqual(status["total_equity"], 5100.00)
            self.assertEqual(status["total_pnl"], 100.00)
            self.assertEqual(status["wins"], 1)
            self.assertEqual(status["win_rate_pct"], 100.0)


class TestLeaderboardAndEvaluation(unittest.TestCase):
    def test_leaderboard_rendering(self):
        engine = MultiBotTournamentEngine(client=None)
        standings = engine.get_leaderboard_data()
        self.assertEqual(len(standings), 5)
        # Verify medals assigned
        self.assertEqual(standings[0]["rank"], 1)
        self.assertEqual(standings[0]["medal"], "🥇")
        self.assertEqual(standings[1]["rank"], 2)
        self.assertEqual(standings[1]["medal"], "🥈")
        self.assertEqual(standings[2]["rank"], 3)
        self.assertEqual(standings[2]["medal"], "🥉")

        # Must run print_leaderboard with zero exceptions
        try:
            engine.print_leaderboard()
        except Exception as e:
            self.fail(f"print_leaderboard raised an exception: {e}")

    def test_backward_compatibility_adapter(self):
        adapter = PaperTradingEngine()
        port = adapter.get_portfolio_status()
        self.assertIn("cash", port)
        self.assertIn("total_equity", port)
        self.assertIn("positions", port)


class TestBorderlistGuardrail(unittest.TestCase):
    def setUp(self):
        self.guardrail = BorderlistGuardrail()

    def test_price_floor_below_five(self):
        # Stocks below $5.00 must be disqualified regardless of volume or cap
        is_safe, reason = self.guardrail.evaluate("PENY", price=4.99, volume=5_000_000, market_cap=1_000_000_000)
        self.assertFalse(is_safe)
        self.assertIn("penny stock risk", reason.lower())

        is_safe2, _ = self.guardrail.evaluate("TRASH", price=0.75, volume=10_000_000, market_cap=500_000_000)
        self.assertFalse(is_safe2)

    def test_price_floor_at_or_above_five(self):
        # Price at exactly $5.00 passes floor
        is_safe, _ = self.guardrail.evaluate("FLOOR", price=5.00, volume=300_000, market_cap=300_000_000)
        self.assertTrue(is_safe)

        is_safe2, _ = self.guardrail.evaluate("SOLID", price=15.50, volume=500_000, market_cap=500_000_000)
        self.assertTrue(is_safe2)

    def test_liquidity_and_market_cap_rules(self):
        # Disqualified if volume < 200,000 AND market cap < $250M
        is_safe, reason = self.guardrail.evaluate("ILLQ", price=10.00, volume=50_000, market_cap=50_000_000)
        self.assertFalse(is_safe)
        self.assertIn("illiquid micro-cap risk", reason.lower())

        # Passes if volume >= 200,000 even if market cap < $250M
        is_safe_vol, _ = self.guardrail.evaluate("VOL_PASS", price=10.00, volume=250_000, market_cap=100_000_000)
        self.assertTrue(is_safe_vol)

        # Passes if market cap >= $250M even if volume < 200,000
        is_safe_cap, _ = self.guardrail.evaluate("CAP_PASS", price=50.00, volume=80_000, market_cap=800_000_000)
        self.assertTrue(is_safe_cap)

        # Boundary volume 200,000 passes
        is_safe_boundary, _ = self.guardrail.evaluate("BND", price=12.00, volume=200_000, market_cap=50_000_000)
        self.assertTrue(is_safe_boundary)

    def test_unverified_liquidity_and_nan_handling(self):
        # Unknown ticker with missing volume and market cap MUST be disqualified
        is_safe, reason = self.guardrail.evaluate("UNKNOWN_TICKER", price=10.00, volume=None, market_cap=None)
        self.assertFalse(is_safe)
        self.assertIn("unverified liquidity", reason.lower())

        # Known mega-cap index/stock with missing volume/cap passes
        is_safe_nvda, _ = self.guardrail.evaluate("NVDA", price=120.00, volume=None, market_cap=None)
        self.assertTrue(is_safe_nvda)
        is_safe_spy, _ = self.guardrail.evaluate("SPY", price=550.00, volume=None, market_cap=None)
        self.assertTrue(is_safe_spy)

        # NaN volume and NaN market cap sanitized and rejected
        is_safe_nan, _ = self.guardrail.evaluate("NANTICK", price=15.00, volume=float("nan"), market_cap=float("nan"))
        self.assertFalse(is_safe_nan)

    def test_hot_trending_skipped_logging(self):
        # Disqualifying hot trending ticker records is_hot=True and logs skipped notice
        self.guardrail.disqualify("HOT_PENY", price=3.50, volume=1_000_000, market_cap=50_000_000, reason="Price below $5 floor", is_hot=True)
        self.assertIn("HOT_PENY", self.guardrail.disqualified)
        entry = self.guardrail.disqualified["HOT_PENY"]
        self.assertTrue(entry["is_hot"])
        self.assertEqual(entry["price"], 3.50)

    def test_execute_buy_guardrail_rejections(self):
        test_dir = Path(__file__).resolve().parent / "test_guardrail_data"
        test_dir.mkdir(parents=True, exist_ok=True)
        mock_file = test_dir / "guardrail_state.json"

        try:
            with patch("src.paper_trader.LOCAL_STATE_FILE", mock_file):
                engine = MultiBotTournamentEngine(client=None)

                # 1. Attempt to buy stock under $5.00 with hot rationale
                rejected_penny = engine.execute_buy(
                    bot_id="conservative_swing",
                    symbol="SUB5",
                    amount_dollars=300.00,
                    rationale="Hot surge catalyst breakout!",
                    current_price=4.50
                )
                self.assertFalse(rejected_penny)
                borderlist = engine.get_borderlist()
                self.assertIn("SUB5", borderlist)
                self.assertTrue(borderlist["SUB5"]["is_hot"])

                # 2. Attempt to buy illiquid micro-cap
                rejected_illiquid = engine.execute_buy(
                    bot_id="conservative_swing",
                    symbol="ILLQ_STOCK",
                    amount_dollars=300.00,
                    rationale="Try illiquid micro-cap",
                    current_price=20.00,
                    volume=30_000,
                    market_cap=60_000_000
                )
                self.assertFalse(rejected_illiquid)
                self.assertIn("ILLQ_STOCK", engine.get_borderlist())

                # 3. Legitimate stock passes and clears any stale borderlist status
                engine.storage.record_borderlist_item("GOOD_STOCK", {"reason": "Old transient error"})
                self.assertIn("GOOD_STOCK", engine.get_borderlist())

                bought = engine.execute_buy(
                    bot_id="conservative_swing",
                    symbol="GOOD_STOCK",
                    amount_dollars=300.00,
                    rationale="Legitimate purchase",
                    current_price=25.00,
                    volume=500_000,
                    market_cap=2_000_000_000
                )
                self.assertTrue(bought)
                # Must be removed from borderlist once it passes and executes
                self.assertNotIn("GOOD_STOCK", engine.get_borderlist())
        finally:
            if test_dir.exists():
                shutil.rmtree(test_dir)


class TestProbabilityWeightedUpside(unittest.TestCase):
    def test_small_cap_high_catalyst_ranks_above_stagnant_megacap(self):
        # Small-cap with strong FinBERT, breakout velocity, and real catalyst
        small_cap_tech = {"valid": True, "price": 18.0, "sma20": 16.5, "above_sma20": True, "momentum_5d": 4.5}
        small_cap_bert = {"net_score": 0.72, "positive": 0.85, "negative": 0.13, "info": "FinBERT[+]"}
        small_cap_arts = [{"title": "Company secures record contract beat revenue expectations"}]

        small_metrics = calculate_probability_weighted_upside(
            symbol="SMALL_ROCKET",
            price=18.0,
            tech=small_cap_tech,
            bert=small_cap_bert,
            articles=small_cap_arts,
            market_cap=800_000_000,
            volume=600_000
        )

        # Stagnant Mega-cap with flat news and flat technicals
        mega_cap_tech = {"valid": True, "price": 250.0, "sma20": 249.0, "above_sma20": True, "momentum_5d": 0.1}
        mega_cap_bert = {"net_score": 0.05, "positive": 0.35, "negative": 0.30, "info": "FinBERT[neutral]"}
        mega_cap_arts = [{"title": "General market overview"}]

        mega_metrics = calculate_probability_weighted_upside(
            symbol="MEGACAP_FLAT",
            price=250.0,
            tech=mega_cap_tech,
            bert=mega_cap_bert,
            articles=mega_cap_arts,
            market_cap=3_000_000_000_000,
            volume=35_000_000
        )

        # Small-cap with catalyst should have higher upside probability and expected return
        self.assertGreater(small_metrics["upside_probability"], mega_metrics["upside_probability"])
        self.assertGreater(small_metrics["expected_return_pct"], mega_metrics["expected_return_pct"])
        self.assertEqual(small_metrics["cap_tier"], "Small-Cap (<$2B)")
        self.assertEqual(mega_metrics["cap_tier"], "Large/Mega-Cap")

    def test_small_cap_without_catalyst_does_not_get_inflated_upside(self):
        # Small cap with NO real catalyst keyword or headlines
        small_no_cat_tech = {"valid": True, "price": 12.0, "sma20": 11.8, "above_sma20": True, "momentum_5d": 0.8}
        small_no_cat_bert = {"net_score": 0.05, "positive": 0.30, "negative": 0.25}
        small_no_cat_arts = [{"title": "Routine weekly filing"}]

        small_no_cat = calculate_probability_weighted_upside(
            symbol="SMALL_NOCAT",
            price=12.0,
            tech=small_no_cat_tech,
            bert=small_no_cat_bert,
            articles=small_no_cat_arts,
            market_cap=600_000_000,
            volume=400_000
        )

        # Mega cap with strong earnings catalyst
        mega_cat_tech = {"valid": True, "price": 300.0, "sma20": 290.0, "above_sma20": True, "momentum_5d": 2.5}
        mega_cat_bert = {"net_score": 0.65, "positive": 0.80, "negative": 0.15}
        mega_cat_arts = [{"title": "Record earnings beat and massive AI revenue growth"}]

        mega_with_cat = calculate_probability_weighted_upside(
            symbol="MEGA_CAT",
            price=300.0,
            tech=mega_cat_tech,
            bert=mega_cat_bert,
            articles=mega_cat_arts,
            market_cap=2_500_000_000_000,
            volume=40_000_000
        )

        # Without catalyst, small cap potential upside is standard (10%) rather than 22%
        self.assertEqual(small_no_cat["potential_upside_pct"], 10.0)
        # Mega cap with catalyst must comfortably beat the catalyst-free small cap
        self.assertGreater(mega_with_cat["expected_return_pct"], small_no_cat["expected_return_pct"])

    def test_catalyst_clarity_scoring(self):
        tech = {"valid": True, "price": 50.0, "sma20": 48.0, "above_sma20": True, "momentum_5d": 1.5}
        bert = {"net_score": 0.50, "positive": 0.70, "negative": 0.20}

        # News with catalyst keywords vs news without
        arts_with_catalyst = [{"title": "Company reports record earnings beat and partnership"}]
        arts_plain = [{"title": "Weekly summary report"}]

        res_cat = calculate_probability_weighted_upside("CAT1", 50.0, tech, bert, arts_with_catalyst)
        res_plain = calculate_probability_weighted_upside("CAT2", 50.0, tech, bert, arts_plain)

        self.assertGreater(res_cat["catalyst_clarity"], res_plain["catalyst_clarity"])
        self.assertEqual(res_cat["catalyst_label"], "Strong Catalyst")

    def test_ranked_candidates_in_snapshot(self):
        prices = {"HIGH_EXP": 15.0, "MID_EXP": 50.0, "LOW_EXP": 100.0, "BORDER_FAIL": 2.50}
        upside_metrics = {
            "HIGH_EXP": {"expected_return_pct": 16.5, "upside_probability_pct": 82.0},
            "MID_EXP": {"expected_return_pct": 8.2, "upside_probability_pct": 65.0},
            "LOW_EXP": {"expected_return_pct": 2.1, "upside_probability_pct": 52.0},
            "BORDER_FAIL": {"expected_return_pct": 25.0, "upside_probability_pct": 90.0}
        }
        borderlist = {"BORDER_FAIL": {"reason": "Below $5 floor"}}

        snapshot = MarketSnapshot(
            prices=prices,
            technicals={},
            finbert_scores={},
            ticker_articles={},
            upside_metrics=upside_metrics,
            borderlist=borderlist
        )

        ranked = snapshot.get_ranked_candidates()
        # BORDER_FAIL must be excluded (both via borderlist and defense-in-depth price floor)
        self.assertNotIn("BORDER_FAIL", ranked)
        # Remaining must be ordered by expected return descending
        self.assertEqual(ranked, ["HIGH_EXP", "MID_EXP", "LOW_EXP"])

    def test_extract_candidate_tickers_from_articles(self):
        articles = [
            {"title": "Why $PLTR and $SOFI are surging today (US)", "summary": "Growth in AI contracts (ALL)"},
            {"title": "Rocket Lab (RKLB) signs multi-launch deal in Europe (EU)", "summary": "Aerospace momentum"}
        ]
        found = extract_candidate_tickers_from_articles(articles)
        self.assertIn("PLTR", found)
        self.assertIn("SOFI", found)
        self.assertIn("RKLB", found)
        # Common stopwords in parentheses must NOT be extracted
        self.assertNotIn("US", found)
        self.assertNotIn("ALL", found)
        self.assertNotIn("EU", found)


if __name__ == "__main__":
    unittest.main()
