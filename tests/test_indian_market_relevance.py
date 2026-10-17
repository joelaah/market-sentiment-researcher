"""
Unit and integration tests for Indian Market Extension (NIFTY 50 / NIFTYBEES.NS).
Covers:
1. Instrument & Guardrail Validation (R1)
2. Indian Feed Ingestion & In-Memory Deduplication (R2)
3. Database Resilience & Pure-Python Vector Cosine Fallback (R2, R4)
4. Targeted News Relevance Evaluator, FinBERT Sentiment & Materiality (R3)
5. CLI Invocation & End-to-End Brief Generation (R3, R4)
All tests run offline-safe with deterministic mocks and fixtures.
"""

import unittest
from unittest.mock import patch, MagicMock
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

from src.paper_trader import (
    BorderlistGuardrail,
    is_inr_symbol,
    is_benchmark_index,
    extract_candidate_tickers_from_articles,
    MIN_SHARE_PRICE,
    MIN_SHARE_PRICE_INR,
    TRUSTED_SYMBOLS,
    MultiBotTournamentEngine,
)
from src.scraper import (
    FEEDS,
    clean_html,
    normalize_url,
    sanitize_title,
    fetch_financial_news,
)
from src.db import (
    search_similar_news,
    upsert_market_news,
    _cosine_similarity as db_cosine_similarity,
)
from src.finbert_analyzer import FinBERTAnalyzer
from src.relevance import (
    NiftyRelevanceChecker,
    format_intelligence_brief,
    run_news_relevance_check,
    get_fallback_news_fixtures,
    _cosine_similarity as rel_cosine_similarity,
)


class TestInstrumentAndGuardrails(unittest.TestCase):
    """
    Test suite for R1: Indian instrument tracking, currency detection,
    and currency-aware borderlist guardrails.
    """

    def setUp(self):
        self.guardrail = BorderlistGuardrail()

    def test_niftybees_with_volume_passes(self):
        """NIFTYBEES.NS ETF with normal price and volume must pass guardrail evaluation."""
        passed, reason = self.guardrail.evaluate("NIFTYBEES.NS", price=263.9, volume=12000000)
        self.assertTrue(passed, f"Expected NIFTYBEES.NS to pass but failed: {reason}")
        self.assertIsNone(reason)

    def test_niftybees_without_volume_off_hours_passes(self):
        """NIFTYBEES.NS without volume/market cap (off-hours/weekends) must pass via trusted bypass."""
        passed, reason = self.guardrail.evaluate("NIFTYBEES.NS", price=263.9, volume=None, market_cap=None)
        self.assertTrue(passed, f"Expected NIFTYBEES.NS off-hours bypass to pass but failed: {reason}")
        self.assertIsNone(reason)

    def test_nsei_index_without_volume_passes(self):
        """^NSEI benchmark index without volume/market cap must pass via trusted bypass."""
        passed, reason = self.guardrail.evaluate("^NSEI", price=23146.0, volume=None, market_cap=None)
        self.assertTrue(passed, f"Expected ^NSEI trusted bypass to pass but failed: {reason}")
        self.assertIsNone(reason)

    def test_nsei_index_with_volume_passes(self):
        """^NSEI benchmark index with volume must pass guardrail evaluation."""
        passed, reason = self.guardrail.evaluate("^NSEI", price=23146.0, volume=391600)
        self.assertTrue(passed, f"Expected ^NSEI to pass but failed: {reason}")
        self.assertIsNone(reason)

    def test_indian_penny_stock_rejected(self):
        """Indian equities below ₹20.00 floor must be disqualified with penny-stock risk."""
        passed, reason = self.guardrail.evaluate("PENY.NS", price=5.00, volume=500000)
        self.assertFalse(passed)
        self.assertIsNotNone(reason)
        self.assertIn("penny stock risk", reason.lower())
        self.assertIn("₹20.00", reason)

    def test_us_penny_stock_rejected(self):
        """US equities below $5.00 floor must be disqualified with penny-stock risk."""
        passed, reason = self.guardrail.evaluate("PENY", price=1.50, volume=500000)
        self.assertFalse(passed)
        self.assertIsNotNone(reason)
        self.assertIn("penny stock risk", reason.lower())
        self.assertIn("$5.00", reason)

    def test_boundary_indian_price_floor(self):
        """Exact boundary tests for INR minimum price floor (₹20.00)."""
        # ₹20.00 exact passes
        passed_exact, _ = self.guardrail.evaluate("TEST.NS", price=20.00, volume=500000)
        self.assertTrue(passed_exact)

        # ₹19.99 fails
        passed_sub, reason_sub = self.guardrail.evaluate("TEST.NS", price=19.99, volume=500000)
        self.assertFalse(passed_sub)
        self.assertIn("penny stock risk", reason_sub.lower())

    def test_boundary_us_price_floor(self):
        """Exact boundary tests for USD minimum price floor ($5.00)."""
        # $5.00 exact passes
        passed_exact, _ = self.guardrail.evaluate("TEST_US", price=5.00, volume=500000)
        self.assertTrue(passed_exact)

        # $4.99 fails
        passed_sub, reason_sub = self.guardrail.evaluate("TEST_US", price=4.99, volume=500000)
        self.assertFalse(passed_sub)
        self.assertIn("penny stock risk", reason_sub.lower())

    def test_currency_detection(self):
        """Validates INR currency symbol detection logic."""
        self.assertTrue(is_inr_symbol("NIFTYBEES.NS"))
        self.assertTrue(is_inr_symbol("^NSEI"))
        self.assertTrue(is_inr_symbol("TCS.NS"))
        self.assertTrue(is_inr_symbol("RELIANCE.BO"))
        self.assertFalse(is_inr_symbol("NVDA"))
        self.assertFalse(is_inr_symbol("AAPL"))
        self.assertFalse(is_inr_symbol("SPY"))
        self.assertFalse(is_inr_symbol("QQQ"))

    def test_benchmark_index_detection(self):
        """Validates detection of benchmark index symbols."""
        self.assertTrue(is_benchmark_index("^NSEI"))
        self.assertTrue(is_benchmark_index("^GSPC"))
        self.assertTrue(is_benchmark_index("^IXIC"))
        self.assertTrue(is_benchmark_index("^DJI"))
        self.assertFalse(is_benchmark_index("NIFTYBEES.NS"))
        self.assertFalse(is_benchmark_index("NVDA"))

    def test_ticker_extraction_and_alias_mapping(self):
        """
        Validates ticker extraction supporting .NS suffixes, non-word ^NSEI,
        and financial keyword alias mapping ("Nifty 50", "Sensex").
        """
        articles = [
            {"title": "Nifty 50 hits record high as FII inflows return", "summary": "Sensex gains 600 points"},
            {"title": "Investors accumulate $NIFTYBEES.NS and $AAPL on dip", "summary": "ETF trading active"},
            {"title": "Quarterly earnings beat at (INFY.NS) and TCS.NS", "summary": "Tech sector surges"},
            {"title": "RBI and SEBI update guidance for NSE and BSE markets", "summary": "INR remains stable"}
        ]
        candidates = extract_candidate_tickers_from_articles(articles)

        # Alias mappings
        self.assertIn("^NSEI", candidates)
        self.assertIn("NIFTYBEES.NS", candidates)

        # Dollar and parenthetical symbols
        self.assertIn("AAPL", candidates)
        self.assertIn("INFY.NS", candidates)
        self.assertIn("TCS.NS", candidates)

        # Stopwords must not be extracted
        self.assertNotIn("RBI", candidates)
        self.assertNotIn("SEBI", candidates)
        self.assertNotIn("NSE", candidates)
        self.assertNotIn("BSE", candidates)
        self.assertNotIn("INR", candidates)

    def test_benchmark_order_routing_executes_at_etf_price(self):
        """
        Verifies that routing a ^NSEI buy order in MultiBotTournamentEngine.execute_buy()
        executes at the tradeable ETF's market price (~₹264) rather than using the index price (~₹23,146).
        """
        temp_dir = Path(__file__).resolve().parent / "test_temp_routing"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file = temp_dir / "routing_state.json"
        if temp_file.exists():
            temp_file.unlink()

        try:
            with patch("src.paper_trader.LOCAL_STATE_FILE", temp_file):
                engine = MultiBotTournamentEngine(client=None)
                with patch("src.paper_trader.get_current_price", return_value=264.0):
                    with patch("src.paper_trader.get_stock_market_metrics", return_value={"valid": True, "volume": 1_000_000, "market_cap": 500_000_000}):
                        success = engine.execute_buy(
                            bot_id="conservative_swing",
                            symbol="^NSEI",
                            amount_dollars=500.0,
                            rationale="Bullish Nifty breakout",
                            current_price=23146.0  # Index price passed by caller
                        )
                self.assertTrue(success)
                positions = engine.storage.get_positions("conservative_swing")
                self.assertEqual(len(positions), 1)
                pos = positions[0]
                self.assertEqual(pos["symbol"], "NIFTYBEES.NS")
                self.assertEqual(pos["average_buy_price"], 264.0)
                self.assertNotEqual(pos["average_buy_price"], 23146.0)
                self.assertAlmostEqual(pos["shares"], round(500.0 / 264.0, 4))
        finally:
            if temp_file.exists():
                temp_file.unlink()
            if temp_dir.exists():
                try:
                    temp_dir.rmdir()
                except Exception:
                    pass



class TestIndianFeedIngestionAndDeduplication(unittest.TestCase):
    """
    Test suite for R2: Indian financial feed configuration,
    HTML entity & anomaly cleaning, URL normalization, and in-memory deduplication.
    """

    def test_feeds_configuration_includes_indian_sources(self):
        """Verifies FEEDS in src/scraper.py includes all designated Indian financial feeds."""
        sources = [f["source"] for f in FEEDS]
        expected_feeds = [
            "Google News India Business",
            "Google News Nifty/Sensex",
            "Economic Times Markets",
            "Economic Times Stocks",
            "Livemint Markets",
            "Moneycontrol Latest",
        ]
        for expected in expected_feeds:
            self.assertIn(expected, sources, f"FEEDS must include '{expected}'")

        # Every feed must have valid structure and URL
        for feed in FEEDS:
            self.assertIn("source", feed)
            self.assertIn("url", feed)
            self.assertTrue(feed["url"].startswith("http://") or feed["url"].startswith("https://"))

    def test_clean_html_decoding_and_typography(self):
        """Tests HTML entity unescaping, unicode artifact cleanup, and whitespace normalization."""
        # 1. HTML entities unescaping
        raw_entities = "<span>F&amp;O Talk: Nifty&#39;s rally &quot;record&quot; price &lt; 20 and volume &gt; 100</span>"
        cleaned_entities = clean_html(raw_entities)
        self.assertEqual(cleaned_entities, "F&O Talk: Nifty's rally \"record\" price < 20 and volume > 100")

        # 2. Publisher typographic anomalies (\ufffd, curly quotes, en/em dashes)
        raw_typo = "Nifty\ufffds rally \u201cAll-Time High\u201d \u2013 Dalal Street\u2014surge \u2018gains\u2019"
        cleaned_typo = clean_html(raw_typo)
        self.assertEqual(cleaned_typo, "Nifty's rally \"All-Time High\" - Dalal Street-surge 'gains'")

        # 3. Non-breaking and zero-width spaces normalization
        raw_spaces = "NIFTY\xa050\u200b   surges   \ufeff 250 points \n\n on Dalal Street"
        cleaned_spaces = clean_html(raw_spaces)
        self.assertEqual(cleaned_spaces, "NIFTY 50 surges 250 points on Dalal Street")

        # 4. Edge cases: empty string and None
        self.assertEqual(clean_html(""), "")
        self.assertEqual(clean_html(None), "")

    def test_normalize_url_strips_tracking_params(self):
        """Tests stripping of marketing/analytics query parameters, fragments, and trailing slashes."""
        raw_url = "https://news.google.com/articles/12345?utm_source=rss&utm_medium=feed&oc=5&ved=2ahUKEwj#top"
        normalized = normalize_url(raw_url)
        self.assertEqual(normalized, "https://news.google.com/articles/12345")

        # Legitimate query parameters are preserved
        search_url = "https://news.google.com/rss/search?q=nifty+sensex&hl=en-IN&gl=IN&utm_campaign=daily"
        norm_search = normalize_url(search_url)
        self.assertNotIn("utm_campaign", norm_search)
        self.assertIn("q=nifty+sensex", norm_search)
        self.assertIn("hl=en-IN", norm_search)

        # Trailing slash normalization
        self.assertEqual(normalize_url("https://example.com/markets/"), "https://example.com/markets")
        self.assertEqual(normalize_url(""), "")
        self.assertEqual(normalize_url(None), "")

    def test_sanitize_title(self):
        """Tests title sanitization to alphanumeric lowercase."""
        self.assertEqual(sanitize_title("Nifty 50: Index hits 24,000!"), "nifty50indexhits24000")
        self.assertEqual(sanitize_title("   RBI Repo Rate -- Unchanged?   "), "rbireporateunchanged")
        self.assertEqual(sanitize_title(""), "")

    def test_in_memory_deduplication(self):
        """
        Tests that fetch_financial_news() deduplicates articles in-memory
        by normalized URL and sanitized title across feeds.
        """
        sample_rss = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <title>Mock Indian Financial News</title>
            <item>
              <title>NIFTY 50 hits record high &amp; surges</title>
              <link>https://economictimes.indiatimes.com/art1?utm_source=rss&amp;oc=5</link>
              <description>The NIFTY 50 index surged 200 points</description>
            </item>
            <item>
              <title>NIFTY 50 hits record high &amp; surges</title>
              <link>https://economictimes.indiatimes.com/art1?utm_medium=twitter</link>
              <description>Duplicate URL differing only by tracking query</description>
            </item>
            <item>
              <title>NIFTY 50 hits record high &amp; surges</title>
              <link>https://livemint.com/syndicated-art1-diff-url</link>
              <description>Syndicated duplicate with identical title</description>
            </item>
            <item>
              <title>RBI Keeps Repo Rate Steady at 6.5%</title>
              <link>https://moneycontrol.com/rbi-policy-decision</link>
              <description>Monetary policy committee decides rates</description>
            </item>
          </channel>
        </rss>
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = sample_rss.encode("utf-8")

        mock_feeds = [
            {"source": "Mock Feed 1", "url": "https://example.com/feed1.xml"},
            {"source": "Mock Feed 2", "url": "https://example.com/feed2.xml"},
        ]

        with patch("src.scraper.requests.get", return_value=mock_response):
            with patch("src.scraper.FEEDS", mock_feeds):
                articles = fetch_financial_news(limit_per_feed=10)

        # Out of 4 items across 2 feeds, only the 2 distinct articles must be collected
        self.assertEqual(len(articles), 2)
        titles = [a["title"] for a in articles]
        self.assertIn("NIFTY 50 hits record high & surges", titles)
        self.assertIn("RBI Keeps Repo Rate Steady at 6.5%", titles)

        # Validate schema contract
        for art in articles:
            for field in ("title", "link", "source", "summary", "published_at", "content"):
                self.assertIn(field, art)
                self.assertTrue(art[field])


class TestDatabaseResilienceAndFallback(unittest.TestCase):
    """
    Test suite for R2 & R4: Supabase vector store resilience, error handling,
    and pure-Python in-memory cosine similarity fallback.
    """

    def test_search_similar_news_none_client_no_fallback(self):
        """search_similar_news with client=None and no fallback must return empty list without crashing."""
        results = search_similar_news(client=None, query_embedding=[0.1] * 384)
        self.assertEqual(results, [])

    def test_search_similar_news_none_client_with_fallback(self):
        """search_similar_news with client=None and fallback_articles calculates pure-Python cosine similarity."""
        fallback = [
            {"title": "High Match", "link": "https://1", "embedding": [1.0, 0.0, 0.0]},
            {"title": "Mid Match", "link": "https://2", "embedding": [0.6, 0.8, 0.0]},
            {"title": "Low Match", "link": "https://3", "embedding": [0.1, 0.0, 0.99]},
        ]
        query = [1.0, 0.0, 0.0]
        results = search_similar_news(
            client=None,
            query_embedding=query,
            match_threshold=0.5,
            match_count=2,
            fallback_articles=fallback
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["title"], "High Match")
        self.assertAlmostEqual(results[0]["similarity"], 1.0, places=3)
        self.assertEqual(results[1]["title"], "Mid Match")
        self.assertAlmostEqual(results[1]["similarity"], 0.6, places=3)

    def test_search_similar_news_failing_client_falls_back(self):
        """If Supabase RPC raises an exception, search_similar_news falls back gracefully to in-memory search."""
        failing_client = MagicMock()
        failing_client.rpc.side_effect = ConnectionError("Supabase pgvector connection timed out")

        fallback = [
            {"title": "Resilient Fallback Article", "link": "https://fallback", "embedding": [1.0, 0.0]}
        ]
        results = search_similar_news(
            client=failing_client,
            query_embedding=[1.0, 0.0],
            match_threshold=0.5,
            fallback_articles=fallback
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Resilient Fallback Article")

    def test_cosine_similarity_edge_cases(self):
        """Validates mathematical properties and edge cases of vector cosine similarity."""
        # Parallel identical vectors
        self.assertAlmostEqual(db_cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        # Orthogonal vectors
        self.assertAlmostEqual(db_cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)
        # Empty or mismatched vectors return 0.0 without errors
        self.assertEqual(db_cosine_similarity([], []), 0.0)
        self.assertEqual(db_cosine_similarity([1.0], [1.0, 2.0]), 0.0)
        # Zero vectors do not trigger divide-by-zero
        self.assertEqual(db_cosine_similarity([0.0, 0.0], [0.0, 0.0]), 0.0)

    def test_upsert_market_news_resilience(self):
        """Validates upsert_market_news error handling and client=None guard."""
        # 1. client=None safely returns 0
        self.assertEqual(upsert_market_news(client=None, articles_with_embeddings=[{"title": "T", "link": "L"}]), 0)

        # 2. Empty articles list returns 0
        mock_client = MagicMock()
        self.assertEqual(upsert_market_news(client=mock_client, articles_with_embeddings=[]), 0)

        # 3. Handling invalid items and per-record database exceptions
        mock_table = MagicMock()
        mock_client.table.return_value = mock_table
        mock_upsert = MagicMock()
        mock_table.upsert.return_value = mock_upsert
        mock_upsert.execute.side_effect = [True, Exception("Primary key conflict")]

        articles = [
            {"title": "Valid 1", "link": "https://1", "embedding": [0.1]},
            {"title": "Valid 2 (will fail in db)", "link": "https://2", "embedding": [0.2]},
            {"title": "Missing link article"}  # Skips cleanly
        ]
        inserted = upsert_market_news(client=mock_client, articles_with_embeddings=articles)
        self.assertEqual(inserted, 1)


class TestTargetedNewsRelevanceChecker(unittest.TestCase):
    """
    Test suite for R3: NiftyRelevanceChecker entity matching, semantic similarity,
    FinBERT sentiment classification, materiality attribution, and tabular brief formatting.
    """

    @classmethod
    def setUpClass(cls):
        cls.checker = NiftyRelevanceChecker()
        cls.finbert = FinBERTAnalyzer()
        cls.fixtures = get_fallback_news_fixtures()

    def test_match_entities_direct_index_and_heavyweights(self):
        """Tests entity recognition patterns for direct index, macro terms, and constituent heavyweights."""
        # Direct index mentions
        matched, score, has_index = self.checker.match_entities("NIFTY 50 surges above 24,000 on Dalal Street")
        self.assertIn("NIFTY 50", matched)
        self.assertIn("Dalal Street", matched)
        self.assertTrue(has_index)
        self.assertGreaterEqual(score, 0.95)

        # Macro institutions and terms
        matched_rbi, score_rbi, has_index_rbi = self.checker.match_entities(
            "Reserve Bank of India (RBI) Monetary Policy Committee keeps repo rate steady"
        )
        self.assertIn("Reserve Bank of India", matched_rbi)
        self.assertIn("RBI", matched_rbi)
        self.assertIn("Monetary Policy Committee", matched_rbi)
        self.assertIn("repo rate", matched_rbi)
        self.assertFalse(has_index_rbi)
        self.assertGreaterEqual(score_rbi, 0.80)

        # Union Budget and Finance Ministry
        matched_budget, _, _ = self.checker.match_entities(
            "Union Budget: Finance Ministry and Nirmala Sitharaman announce tax reforms"
        )
        self.assertIn("Union Budget", matched_budget)
        self.assertIn("Finance Ministry", matched_budget)
        self.assertIn("Nirmala Sitharaman", matched_budget)

        # Constituent heavyweights
        matched_hw, _, _ = self.checker.match_entities(
            "Reliance Industries, HDFC Bank, and TCS lead morning gains"
        )
        self.assertIn("Reliance", matched_hw)
        self.assertIn("HDFC Bank", matched_hw)
        self.assertIn("TCS", matched_hw)

        # Completely unrelated non-Indian content
        matched_unrel, score_unrel, has_index_unrel = self.checker.match_entities(
            "Seattle metropolitan residential real estate zoning evaluation"
        )
        self.assertEqual(matched_unrel, [])
        self.assertEqual(score_unrel, 0.0)
        self.assertFalse(has_index_unrel)

    def test_relevance_discrimination_indian_vs_unrelated(self):
        """
        Validates discrimination: Indian market articles are flagged as relevant (True),
        while unrelated international/local articles are rejected (False).
        """
        # 1. NIFTY Record High + RBI fixture -> Relevant
        res_nifty = self.checker.evaluate_article(self.fixtures[0])
        self.assertTrue(res_nifty["is_relevant"])
        self.assertGreaterEqual(res_nifty["relevance_score"], 0.50)
        self.assertIn(res_nifty["materiality"], ("HIGH", "MEDIUM"))

        # 2. Reliance & HDFC Bank fixture -> Relevant
        res_reliance = self.checker.evaluate_article(self.fixtures[1])
        self.assertTrue(res_reliance["is_relevant"])
        self.assertGreaterEqual(res_reliance["relevance_score"], 0.50)

        # 3. Union Budget fixture -> Relevant
        res_budget = self.checker.evaluate_article(self.fixtures[2])
        self.assertTrue(res_budget["is_relevant"])

        # 4. Seattle residential housing fixture -> Unrelated / Not Relevant
        res_seattle = self.checker.evaluate_article(self.fixtures[4])
        self.assertFalse(res_seattle["is_relevant"])
        self.assertLessEqual(res_seattle["relevance_score"], 0.42)
        self.assertEqual(res_seattle["materiality"], "LOW")

        # 5. European Central Bank symposium fixture -> Unrelated / Not Relevant
        res_ecb = self.checker.evaluate_article(self.fixtures[5])
        self.assertFalse(res_ecb["is_relevant"])
        self.assertLessEqual(res_ecb["relevance_score"], 0.42)
        self.assertEqual(res_ecb["materiality"], "LOW")

    def test_deceptive_keyword_false_positives_rejected(self):
        """
        Adversarial test verifying that deceptive non-financial articles mentioning
        'nifty' (gadget, pet dog, tips), international acronym collisions ('BSE' mad cow,
        'NSE' Nigerian exchange, 'SBI' Tokyo crypto, 'ITC' solar tariff), and celestial/utility
        names ('Titan' Saturn moon, 'Power Grid' Texas freeze) evaluate strictly to is_relevant: False.
        """
        deceptive_articles = [
            {
                "id": "gadget",
                "title": "A nifty new tech gadget in Silicon Valley revolutionizes photography",
                "summary": "Startups in California unveil nifty smartphone accessories featuring AI lens zoom.",
                "content": "A nifty new tech gadget in Silicon Valley revolutionizes photography."
            },
            {
                "id": "pet_dog",
                "title": "Meet Nifty, the rescued golden retriever who loves playing fetch in Seattle",
                "summary": "An adorable dog named Nifty won hearts at the local park agility championship.",
                "content": "Meet Nifty, the rescued golden retriever who loves playing fetch in Seattle."
            },
            {
                "id": "cleaning_tips",
                "title": "Ten nifty tips for spring cleaning your house on a sunny weekend",
                "summary": "Use baking soda and vinegar for nifty kitchen hacks that remove stubborn grease.",
                "content": "Ten nifty tips for spring cleaning your house on a sunny weekend."
            },
            {
                "id": "bse_mad_cow",
                "title": "Outbreak of Bovine Spongiform Encephalopathy (BSE) confirmed by UK board",
                "summary": "Agricultural inspectors quarantine cattle farm in Yorkshire after BSE mad cow disease symptoms.",
                "content": "Outbreak of Bovine Spongiform Encephalopathy (BSE) confirmed by UK board."
            },
            {
                "id": "nse_nigeria",
                "title": "Nigerian Stock Exchange (NSE) welcomes new telecommunications listing in Lagos",
                "summary": "Equities on the Nigerian Stock Exchange (NSE) rallied as domestic pension funds bought shares.",
                "content": "Nigerian Stock Exchange (NSE) welcomes new telecommunications listing in Lagos."
            },
            {
                "id": "titan_moon",
                "title": "NASA discovers liquid methane lakes on Saturn's moon Titan",
                "summary": "Planetary scientists analyze radar data from the Cassini mission studying Titan atmosphere.",
                "content": "NASA discovers liquid methane lakes on Saturn's moon Titan."
            },
            {
                "id": "power_grid_texas",
                "title": "Severe winter storm knocks out Texas power grid leaving millions freezing",
                "summary": "Freezing rain crippled power grid infrastructure and energy distribution across Texas.",
                "content": "Severe winter storm knocks out Texas power grid leaving millions freezing."
            },
            {
                "id": "budget_hotel_paris",
                "title": "European travel guide: How to find a charming budget hotel in Paris",
                "summary": "Travelers can save euros by booking boutique budget accommodation near the Seine river.",
                "content": "European travel guide: How to find a charming budget hotel in Paris."
            },
            {
                "id": "itc_solar",
                "title": "US International Trade Commission (ITC) issues tariff ruling on solar panels",
                "summary": "The ITC determined that imported solar modules injured domestic manufacturing companies.",
                "content": "US International Trade Commission (ITC) issues tariff ruling on solar panels."
            },
            {
                "id": "sbi_japan",
                "title": "Tokyo financial firm SBI Holdings expands cryptocurrency exchange platform",
                "summary": "Japanese fintech group SBI Holdings reported increased crypto trading volumes in Tokyo.",
                "content": "Tokyo financial firm SBI Holdings expands cryptocurrency exchange platform."
            }
        ]

        for case in deceptive_articles:
            res = self.checker.evaluate_article(case)
            self.assertFalse(
                res["is_relevant"],
                f"Deceptive case '{case['id']}' falsely evaluated to is_relevant: True (score={res['relevance_score']}, reason={res['reason']})"
            )
            self.assertEqual(res["materiality"], "LOW")
            self.assertLessEqual(res["relevance_score"], 0.45)

    def test_genuine_indian_market_news_accepted(self):
        """Verifies that authentic Indian market news items evaluate strictly to is_relevant: True."""
        genuine_cases = [
            {
                "id": "nifty_rally",
                "title": "NIFTY 50 surges 350 points to cross 24,500 on Dalal Street as foreign investors return",
                "summary": "Benchmark index NIFTY posted strong gains as blue-chip equities rallied across sectors on NSE.",
                "content": "NIFTY 50 surges 350 points to cross 24,500 on Dalal Street as foreign investors return."
            },
            {
                "id": "rbi_repo",
                "title": "RBI Monetary Policy Committee keeps repo rate steady at 6.5%, raises GDP outlook",
                "summary": "Reserve Bank of India Governor announced policy stance supporting economic expansion and rupee stability.",
                "content": "RBI Monetary Policy Committee keeps repo rate steady at 6.5%, raises GDP outlook."
            },
            {
                "id": "heavyweight_earnings",
                "title": "Reliance Industries and TCS drive Sensex and NSE gains following record quarterly profits",
                "summary": "Indian heavyweight firms Reliance and Tata Consultancy Services lifted market indices in Mumbai trading.",
                "content": "Reliance Industries and TCS drive Sensex and NSE gains following record quarterly profits."
            }
        ]

        for case in genuine_cases:
            res = self.checker.evaluate_article(case)
            self.assertTrue(
                res["is_relevant"],
                f"Genuine case '{case['id']}' failed relevance check (score={res['relevance_score']}, reason={res['reason']})"
            )
            self.assertGreaterEqual(res["relevance_score"], 0.50)
            self.assertIn(res["materiality"], ("HIGH", "MEDIUM"))

    def test_finbert_analyzer_outputs(self):
        """Validates FinBERTAnalyzer returns polarity, confidence, probabilities, and bounded net_score."""
        # 1. Bullish statement
        res_pos = self.finbert.analyze("NIFTY 50 posts historic rally as corporate revenues and profits surge 30%")
        self.assertIn("polarity", res_pos)
        self.assertIn("confidence", res_pos)
        self.assertIn("net_score", res_pos)
        self.assertIn("positive", res_pos)
        self.assertIn("negative", res_pos)
        self.assertIn("neutral", res_pos)

        self.assertEqual(res_pos["polarity"], "positive")
        self.assertGreaterEqual(res_pos["confidence"], 0.50)
        self.assertGreater(res_pos["net_score"], 0.0)
        self.assertTrue(-1.0 <= res_pos["net_score"] <= 1.0)

        # 2. Bearish statement
        res_neg = self.finbert.analyze("Heavyweight firm reports disastrous loss and faces insolvency proceedings")
        self.assertEqual(res_neg["polarity"], "negative")
        self.assertLess(res_neg["net_score"], 0.0)

        # 3. Empty text handling
        res_empty = self.finbert.analyze("")
        self.assertEqual(res_empty["polarity"], "neutral")
        self.assertEqual(res_empty["confidence"], 1.0)
        self.assertEqual(res_empty["net_score"], 0.0)

    def test_materiality_scoring_rules(self):
        """Validates that high catalysts yield HIGH/MEDIUM materiality, and irrelevant items strictly yield LOW."""
        # High impact catalyst with positive sentiment
        mat_high = self.checker.determine_materiality(
            text="RBI surprises with repo rate cut and forecasts GDP beat",
            entity_score=0.85,
            sentiment={"net_score": 0.65}
        )
        self.assertIn(mat_high, ("HIGH", "MEDIUM"))

        # Routine announcement without catalysts
        mat_low = self.checker.determine_materiality(
            text="Standard administrative documentation filed with regulatory office",
            entity_score=0.20,
            sentiment={"net_score": 0.05}
        )
        self.assertEqual(mat_low, "LOW")

        # Irrelevant article evaluation contract: strictly LOW materiality
        unrelated_art = {
            "title": "Local municipal council in Bavaria discusses bridge restoration",
            "summary": "Engineering assessment completed for historical stone bridge",
            "content": "Local municipal council in Bavaria discusses bridge restoration"
        }
        res_unrel = self.checker.evaluate_article(unrelated_art)
        self.assertFalse(res_unrel["is_relevant"])
        self.assertEqual(res_unrel["materiality"], "LOW")

    def test_filter_relevant_news_sorting(self):
        """Tests that filter_relevant_news eliminates irrelevant items and sorts by materiality & score."""
        results = self.checker.filter_relevant_news(self.fixtures, symbol="NIFTY", top_n=10)

        # Only relevant articles must be retained
        for item in results:
            self.assertTrue(item["is_relevant"])

        titles = [item["article"]["title"] for item in results]
        self.assertFalse(any("Seattle" in t for t in titles))
        self.assertFalse(any("Frankfurt" in t for t in titles))

        # Check descending order of materiality ranks
        materiality_ranks = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        ranks = [materiality_ranks[item["materiality"]] for item in results]
        self.assertEqual(ranks, sorted(ranks, reverse=True))

    def test_format_intelligence_brief_structure(self):
        """Tests intelligence brief formatting produces expected ASCII headers, metrics, and tabular columns."""
        relevant = self.checker.filter_relevant_news(self.fixtures, symbol="NIFTY", top_n=5)
        brief = format_intelligence_brief(relevant, symbol="NIFTY", total_analyzed=len(self.fixtures))

        # Check mandatory brief headers and sections
        self.assertIn("TARGETED MARKET INTELLIGENCE BRIEF: NIFTY", brief)
        self.assertIn("[EXECUTIVE SCAN SUMMARY]", brief)
        self.assertIn("Total Articles Scanned:", brief)
        self.assertIn("Relevant Articles Identified:", brief)
        self.assertIn("Sentiment Breakdown:", brief)
        self.assertIn("Aggregate Net Sentiment:", brief)
        self.assertIn("High-Impact Catalysts Detected:", brief)
        self.assertIn("[KEY CATALYSTS & BREAKING HEADLINES]", brief)

        # Check table columns
        for col in ("Title", "Source", "Entity Matches", "Semantic Sim", "Sentiment (Pol + Conf)", "Net Score", "Materiality", "Reason"):
            self.assertIn(col, brief)

        # Empty scan results handling
        empty_brief = format_intelligence_brief([], symbol="NIFTY", total_analyzed=5)
        self.assertIn("No market news items met the relevance criteria", empty_brief)

    def test_fallback_news_fixtures_validity(self):
        """Verifies fallback news fixtures contain realistic items with complete schemas."""
        fixtures = get_fallback_news_fixtures()
        self.assertGreaterEqual(len(fixtures), 5)
        for item in fixtures:
            for field in ("title", "link", "source", "summary", "published_at", "content"):
                self.assertIn(field, item)
                self.assertTrue(item[field])


class TestCLIAndEndToEndExecution(unittest.TestCase):
    """
    Test suite for R3 & R4: Orchestrated news relevance pipeline
    and command-line interface execution (python run_pipeline.py --check-news --symbol NIFTY).
    """

    def test_run_news_relevance_check_orchestration(self):
        """Tests direct programmatic invocation of run_news_relevance_check with mocked feed input."""
        fixtures = get_fallback_news_fixtures()
        with patch("src.scraper.fetch_financial_news", return_value=fixtures):
            results = run_news_relevance_check(symbol="NIFTY", top_n=3)

        self.assertIsInstance(results, list)
        self.assertGreaterEqual(len(results), 1)
        for r in results:
            self.assertTrue(r["is_relevant"])
            self.assertIn("relevance_score", r)
            self.assertIn("sentiment", r)
            self.assertIn("materiality", r)

    def test_run_news_relevance_check_offline_exception_fallback(self):
        """Tests that when scraper encounters network exception, the pipeline falls back to fixtures."""
        with patch("src.scraper.fetch_financial_news", side_effect=RuntimeError("DNS resolution failed")):
            results = run_news_relevance_check(symbol="NIFTY", top_n=2)

        self.assertIsInstance(results, list)
        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(all(r["is_relevant"] for r in results))

    def test_run_pipeline_cli_subprocess(self):
        """
        Executes 'python run_pipeline.py --check-news --symbol NIFTY' via subprocess.
        Verifies returncode 0 and structured console output.
        """
        project_root = Path(__file__).resolve().parent.parent
        cmd = [sys.executable, "run_pipeline.py", "--check-news", "--symbol", "NIFTY"]

        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=300
        )

        self.assertEqual(proc.returncode, 0, f"CLI exited with error: {proc.stderr}")
        self.assertIn("TARGETED MARKET INTELLIGENCE BRIEF: NIFTY", proc.stdout)
        self.assertIn("[EXECUTIVE SCAN SUMMARY]", proc.stdout)
        self.assertIn("Relevant Articles Identified:", proc.stdout)


if __name__ == "__main__":
    unittest.main()
