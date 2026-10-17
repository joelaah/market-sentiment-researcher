"""
Adversarial Stress Test Harness — Challenger 2
Empirical verification of:
1. Guardrail Boundary & Currency Stress (INR ₹19.99 vs ₹20.00, USD $4.99 vs $5.00, off-hours liquidity, NIFTYBEES currency)
2. Paper Trading Execution (Index ^NSEI routing, price poisoning, position sizing, portfolio accounting)
3. Database Resilience & In-Memory Vector Search (Network cutoff, RPC errors, zero vectors, mismatched dimensions)
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.paper_trader import (
    BorderlistGuardrail,
    is_inr_symbol,
    is_benchmark_index,
    INDEX_TRADEABLE_MAP,
    TRUSTED_SYMBOLS,
    MultiBotTournamentEngine,
)
from src.db import search_similar_news, _cosine_similarity, upsert_market_news


class AdversarialGuardrailAndCurrencyStress(unittest.TestCase):
    def setUp(self):
        self.guardrail = BorderlistGuardrail()

    def test_inr_exact_boundary_rejection(self):
        """₹19.99 must be rejected as penny stock for Indian security."""
        passed, reason = self.guardrail.evaluate("TEST.NS", price=19.99, volume=500_000)
        self.assertFalse(passed, "₹19.99 must be rejected")
        self.assertIn("penny stock risk", reason.lower())
        self.assertIn("₹20.00", reason)

    def test_inr_exact_boundary_acceptance(self):
        """₹20.00 exact must be accepted for Indian security."""
        passed, reason = self.guardrail.evaluate("TEST.NS", price=20.00, volume=500_000)
        self.assertTrue(passed, f"₹20.00 must be accepted, got reason: {reason}")
        self.assertIsNone(reason)

    def test_usd_exact_boundary_rejection(self):
        """$4.99 must be rejected as penny stock for US security."""
        passed, reason = self.guardrail.evaluate("TEST_US", price=4.99, volume=500_000)
        self.assertFalse(passed, "$4.99 must be rejected")
        self.assertIn("penny stock risk", reason.lower())
        self.assertIn("$5.00", reason)

    def test_usd_exact_boundary_acceptance(self):
        """$5.00 exact must be accepted for US security."""
        passed, reason = self.guardrail.evaluate("TEST_US", price=5.00, volume=500_000)
        self.assertTrue(passed, f"$5.00 must be accepted, got reason: {reason}")
        self.assertIsNone(reason)

    def test_niftybees_price_not_falsely_converted(self):
        """NIFTYBEES.NS at ~₹264 must pass and never be rejected as a penny stock."""
        passed, reason = self.guardrail.evaluate("NIFTYBEES.NS", price=264.0, volume=500_000)
        self.assertTrue(passed, f"NIFTYBEES.NS at 264.0 must pass: {reason}")
        self.assertIsNone(reason)

    def test_niftybees_off_hours_missing_liquidity_passes(self):
        """NIFTYBEES.NS with volume=None, market_cap=None must pass via trusted bypass."""
        passed, reason = self.guardrail.evaluate("NIFTYBEES.NS", price=264.0, volume=None, market_cap=None)
        self.assertTrue(passed, f"NIFTYBEES.NS off-hours must pass: {reason}")
        self.assertIsNone(reason)

    def test_nsei_off_hours_missing_liquidity_passes(self):
        """^NSEI with volume=None, market_cap=None must pass via trusted bypass."""
        passed, reason = self.guardrail.evaluate("^NSEI", price=23146.0, volume=None, market_cap=None)
        self.assertTrue(passed, f"^NSEI off-hours must pass: {reason}")
        self.assertIsNone(reason)

    def test_unknown_ticker_missing_liquidity_rejected(self):
        """Unknown ticker with volume=None, market_cap=None must be rejected."""
        passed, reason = self.guardrail.evaluate("UNKNOWN.NS", price=264.0, volume=None, market_cap=None)
        self.assertFalse(passed, "Unknown ticker with missing volume/mcap must be rejected")
        self.assertIn("unverified liquidity", reason.lower())

    def test_unknown_ticker_zero_volume_rejected(self):
        """Unknown ticker with zero volume must be rejected."""
        passed, reason = self.guardrail.evaluate("UNKNOWN.NS", price=264.0, volume=0.0, market_cap=None)
        self.assertFalse(passed, "Unknown ticker with 0 volume must be rejected")
        self.assertIn("illiquid", reason.lower())

    def test_unknown_ticker_low_volume_rejected(self):
        """Unknown ticker with sub-threshold volume (<200k) and sub-threshold mcap (<$250M) rejected."""
        passed, reason = self.guardrail.evaluate("UNKNOWN.NS", price=264.0, volume=10_000, market_cap=50_000_000)
        self.assertFalse(passed, "Sub-threshold ticker must be rejected")
        self.assertIn("illiquid micro-cap risk", reason.lower())

    def test_niftybees_sub_threshold_volume_passes_as_trusted(self):
        """NIFTYBEES.NS even with low volume off-hours must pass via trusted bypass."""
        passed, reason = self.guardrail.evaluate("NIFTYBEES.NS", price=264.0, volume=100, market_cap=1000)
        self.assertTrue(passed, "Trusted symbol must pass even with low off-hours volume")


class AdversarialOrderRoutingAndExecutionStress(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(__file__).resolve().parent / "test_adversarial_data"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.mock_file = self.test_dir / "adversarial_state.json"
        if self.mock_file.exists():
            self.mock_file.unlink()

    def tearDown(self):
        if self.mock_file.exists():
            self.mock_file.unlink()
        if self.test_dir.exists():
            try:
                self.test_dir.rmdir()
            except Exception:
                pass

    def test_unmapped_index_rejected_cleanly(self):
        """Non-tradeable index without mapping (^GSPC, ^IXIC) must be rejected without crashing."""
        with patch("src.paper_trader.LOCAL_STATE_FILE", self.mock_file):
            engine = MultiBotTournamentEngine(client=None)
            res = engine.execute_buy(
                bot_id="conservative_swing",
                symbol="^GSPC",
                amount_dollars=500.0,
                rationale="S&P rally"
            )
            self.assertFalse(res)
            positions = engine.storage.get_positions("conservative_swing")
            self.assertEqual(len(positions), 0)

    def test_index_routing_retains_index_price_bug(self):
        """
        ADVERSARIAL PROOF:
        When execute_buy receives symbol='^NSEI' and current_price=23146.0 (as passed by run_daily_tournament),
        does it route to NIFTYBEES.NS at the ETF price or does it execute at the index price (23146.0)?
        """
        with patch("src.paper_trader.LOCAL_STATE_FILE", self.mock_file):
            engine = MultiBotTournamentEngine(client=None)
            with patch("src.paper_trader.get_current_price", return_value=264.0):
                with patch("src.paper_trader.get_stock_market_metrics", return_value={"valid": True, "volume": 1_000_000, "market_cap": 500_000_000}):
                    engine.execute_buy(
                        bot_id="conservative_swing",
                        symbol="^NSEI",
                        amount_dollars=500.0,
                        rationale="Macro rally in Nifty 50",
                        current_price=23146.0
                    )
            positions = engine.storage.get_positions("conservative_swing")
            self.assertEqual(len(positions), 1)
            pos = positions[0]
            self.assertEqual(pos["symbol"], "NIFTYBEES.NS")
            # If bug is present, average_buy_price is 23146.0 instead of 264.0
            print(f"[TEST AUDIT] Routed buy price recorded: {pos['average_buy_price']}")
            # We record whether the bug occurs:
            bug_present = (pos["average_buy_price"] == 23146.0)
            self.assertTrue(bug_present, "Expected to reproduce index price poisoning bug in execute_buy")

    def test_index_routing_clean_when_current_price_none(self):
        """When current_price is None, execute_buy correctly fetches and uses NIFTYBEES.NS price."""
        with patch("src.paper_trader.LOCAL_STATE_FILE", self.mock_file):
            engine = MultiBotTournamentEngine(client=None)
            with patch("src.paper_trader.get_current_price", return_value=264.0):
                with patch("src.paper_trader.get_stock_market_metrics", return_value={"valid": True, "volume": 1_000_000, "market_cap": 500_000_000}):
                    res = engine.execute_buy(
                        bot_id="conservative_swing",
                        symbol="^NSEI",
                        amount_dollars=500.0,
                        rationale="Macro rally in Nifty 50",
                        current_price=None
                    )
            self.assertTrue(res)
            positions = engine.storage.get_positions("conservative_swing")
            self.assertEqual(len(positions), 1)
            pos = positions[0]
            self.assertEqual(pos["symbol"], "NIFTYBEES.NS")
            self.assertEqual(pos["average_buy_price"], 264.0)
            self.assertAlmostEqual(pos["shares"], round(500.0 / 264.0, 4))


class AdversarialDatabaseResilienceStress(unittest.TestCase):
    def test_search_similar_news_simulated_network_cutoff(self):
        """Simulates network timeout/cutoff during Supabase RPC call."""
        failing_client = MagicMock()
        failing_client.rpc.side_effect = TimeoutError("Connection to Supabase timed out after 30s")

        fallback = [
            {"title": "RBI Keeps Rates Steady", "link": "https://rbi", "embedding": [1.0, 0.0, 0.0]},
            {"title": "Global Tech Rally", "link": "https://tech", "embedding": [0.0, 1.0, 0.0]},
        ]
        results = search_similar_news(
            client=failing_client,
            query_embedding=[1.0, 0.0, 0.0],
            match_threshold=0.5,
            fallback_articles=fallback
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "RBI Keeps Rates Steady")
        self.assertAlmostEqual(results[0]["similarity"], 1.0)

    def test_search_similar_news_invalid_sql_or_missing_rpc(self):
        """Simulates PostgreSQL syntax error or missing RPC function."""
        failing_client = MagicMock()
        failing_client.rpc.side_effect = Exception("PGRST202: Could not find the function public.match_market_news")

        fallback = [
            {"title": "NIFTY Surges", "link": "https://nifty", "embedding": [0.8, 0.6]}
        ]
        results = search_similar_news(
            client=failing_client,
            query_embedding=[0.8, 0.6],
            match_threshold=0.5,
            fallback_articles=fallback
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "NIFTY Surges")

    def test_search_similar_news_zero_vector_query(self):
        """Zero vector query must not raise ZeroDivisionError and return empty list."""
        zero_query = [0.0] * 384
        fallback = [
            {"title": "Article 1", "link": "https://1", "embedding": [0.5] * 384}
        ]
        results = search_similar_news(
            client=None,
            query_embedding=zero_query,
            match_threshold=0.1,
            fallback_articles=fallback
        )
        self.assertEqual(results, [])

    def test_search_similar_news_zero_vector_in_fallback_article(self):
        """Zero vector article in fallback store must yield 0.0 similarity without ZeroDivisionError."""
        query = [1.0] * 384
        fallback = [
            {"title": "Zero Vector Article", "link": "https://zero", "embedding": [0.0] * 384},
            {"title": "Valid Article", "link": "https://valid", "embedding": [1.0] * 384}
        ]
        results = search_similar_news(
            client=None,
            query_embedding=query,
            match_threshold=0.5,
            fallback_articles=fallback
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Valid Article")

    def test_search_similar_news_dimension_mismatch(self):
        """Mismatched embedding dimensions must not crash and yield 0.0 similarity."""
        query_384 = [1.0] * 384
        fallback = [
            {"title": "128d Article", "link": "https://128", "embedding": [1.0] * 128}
        ]
        results = search_similar_news(
            client=None,
            query_embedding=query_384,
            match_threshold=0.1,
            fallback_articles=fallback
        )
        self.assertEqual(results, [])

    def test_upsert_market_news_network_cutoff(self):
        """upsert_market_news with network cutoff returns 0 inserted without unhandled exception."""
        failing_client = MagicMock()
        mock_table = MagicMock()
        failing_client.table.return_value = mock_table
        mock_upsert = MagicMock()
        mock_table.upsert.return_value = mock_upsert
        mock_upsert.execute.side_effect = ConnectionError("Supabase unreachable")

        articles = [{"title": "Art", "link": "https://art", "embedding": [0.1]}]
        count = upsert_market_news(client=failing_client, articles_with_embeddings=articles)
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
