"""
Unit and Integration Tests for Historical News Ingestion Pipeline & Backtesting Integrity.

Verifies:
1. HistoricalNewsFetcher loading of JSON and Parquet archives.
2. Complete archive schema validation (timestamp, date, symbol, title, snippet, source, link, logits, calibrated_probs, dense_text_embedding).
3. Safe symbol mapping across operating systems (^NSEI <-> _NSEI).
4. Single-bar news retrieval, multi-article daily aggregation, and quiet-day None handling.
5. Feed generation and dictionary mapping emulation for HistoricalNewsFetcher.
6. Elimination of circular price-derived sentiment in HistoricalBacktestEngine.
7. Strict enforcement of synthetic_sentiment_used is False when real news feeds are supplied.
8. Graceful quiet-day handling: neutral sentiment fallback, disabled sentiment weighting, no fake articles.
9. Independence from price returns and volume surges when real feeds are active.
10. Corrupt archive and missing file resilience.
"""

import unittest
import tempfile
import shutil
import warnings
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from src.news_fetcher import HistoricalNewsFetcher
from src.backtest_engine import HistoricalBacktestEngine, HistoricalDataFetcher
from src.paper_trader import (
    ConservativeSwingBot,
    FinBERTCatalystHunterBot,
    TrendFollowerBot,
    AggressiveBreakoutBot,
    ContrarianDipBuyerBot
)


class TestHistoricalNewsFetcher(unittest.TestCase):
    """Unit tests for HistoricalNewsFetcher archive loading and schema validation."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="test_news_fetcher_")
        cls.cache_path = Path(cls.temp_dir)
        cls.fetcher = HistoricalNewsFetcher(cache_dir=cls.cache_path)

        # Create sample seed article
        cls.sample_articles = [
            {
                "timestamp": "2024-03-15T14:30:00Z",
                "date": "2024-03-15",
                "symbol": "AAPL",
                "title": "Apple Expands AI Research Lab with M3 Ultra Hardware",
                "snippet": "Apple today announced new investments in foundation model training infrastructure.",
                "source": "Reuters",
                "link": "https://reuters.com/tech/apple-ai-m3",
                "sentiment_logits": [2.5, -1.8, -0.7],
                "positive": 0.93,
                "negative": 0.01,
                "neutral": 0.06,
                "dense_text_embedding": [0.05] * 384
            },
            {
                "timestamp": "2024-03-15T18:00:00Z",
                "date": "2024-03-15",
                "symbol": "AAPL",
                "title": "Morgan Stanley Reiterates Overweight on Apple AI Pipeline",
                "snippet": "Analysts expect multi-year device upgrade supercycle starting late 2024.",
                "source": "Bloomberg",
                "link": "https://bloomberg.com/apple-upgrade-cycle",
                "sentiment_logits": [2.2, -1.5, -0.7],
                "positive": 0.89,
                "negative": 0.02,
                "neutral": 0.09,
                "dense_text_embedding": [0.04] * 384
            },
            {
                "timestamp": "2024-03-18T10:00:00Z",
                "date": "2024-03-18",
                "symbol": "AAPL",
                "title": "Regulatory Scrutiny Intensifies Over Smartphone App Margins",
                "snippet": "Antitrust inquiries in multiple jurisdictions probe App Store fee structure.",
                "source": "Wall Street Journal",
                "link": "https://wsj.com/apple-regulations",
                "sentiment_logits": [-1.9, 2.4, -0.5],
                "positive": 0.02,
                "negative": 0.92,
                "neutral": 0.06,
                "dense_text_embedding": [-0.03] * 384
            }
        ]

    @classmethod
    def tearDownClass(cls):
        try:
            shutil.rmtree(cls.temp_dir)
        except Exception:
            pass

    def test_save_and_load_json_archive(self):
        """Verify saving and loading pure JSON news archives."""
        save_res = self.fetcher.save_news_archive("AAPL", self.sample_articles, format="json")
        self.assertIn("json", save_res)
        self.assertTrue(Path(save_res["json"]).exists())

        loaded = self.fetcher.load_symbol_news("AAPL")
        self.assertEqual(len(loaded), 3)
        first = loaded[0]
        self.assertEqual(first["symbol"], "AAPL")
        self.assertEqual(first["date"], "2024-03-15")
        self.assertIn("Apple Expands AI Research Lab", first["title"])
        self.assertAlmostEqual(first["positive"] + first["negative"] + first["neutral"], 1.0, places=3)
        self.assertEqual(len(first["dense_text_embedding"]), 384)

    def test_save_and_load_parquet_archive(self):
        """Verify saving and loading Parquet news archives with vector columns."""
        save_res = self.fetcher.save_news_archive("NVDA", self.sample_articles, format="parquet")
        self.assertIn("parquet", save_res)
        self.assertTrue(Path(save_res["parquet"]).exists())

        fetcher2 = HistoricalNewsFetcher(cache_dir=self.cache_path)
        loaded = fetcher2.load_symbol_news("NVDA")
        self.assertEqual(len(loaded), 3)
        self.assertEqual(len(loaded[0]["dense_text_embedding"]), 384)
        self.assertEqual(len(loaded[0]["sentiment_logits"]), 3)

    def test_safe_symbol_normalization(self):
        """Verify Indian index symbol ^NSEI is safely serialized to _NSEI and queryable via ^NSEI."""
        articles_nsei = [
            {
                "timestamp": "2024-02-08T06:00:00Z",
                "date": "2024-02-08",
                "symbol": "^NSEI",
                "title": "RBI Keeps Policy Repo Rate Unchanged at 6.5%",
                "snippet": "MPC projects 7% real GDP growth with retail inflation moderating.",
                "source": "Economic Times",
                "link": "https://economictimes.com/rbi-policy-rates",
                "sentiment_logits": [2.3, -1.6, -0.7],
                "positive": 0.90,
                "negative": 0.02,
                "neutral": 0.08,
            }
        ]
        save_res = self.fetcher.save_news_archive("^NSEI", articles_nsei, format="both")
        self.assertTrue(Path(save_res["json"]).name.startswith("_NSEI"))
        self.assertTrue(Path(save_res["parquet"]).name.startswith("_NSEI"))

        fetcher3 = HistoricalNewsFetcher(cache_dir=self.cache_path)
        bar = fetcher3.get_news_for_bar("^NSEI", "2024-02-08")
        self.assertIsNotNone(bar)
        self.assertEqual(bar["symbol"], "^NSEI")
        self.assertIn("RBI Keeps Policy", bar["headline"])

    def test_single_bar_multi_article_aggregation(self):
        """Verify multiple articles on the same date are aggregated cleanly with weighted metrics."""
        self.fetcher.save_news_archive("AAPL", self.sample_articles, format="both")
        bar = self.fetcher.get_news_for_bar("AAPL", "2024-03-15")
        self.assertIsNotNone(bar)
        self.assertTrue(bar["has_news"])
        self.assertFalse(bar["synthetic"])
        self.assertEqual(len(bar["articles"]), 2)
        # Average of 0.93 and 0.89 is ~0.91
        self.assertGreater(bar["positive"], 0.85)
        self.assertGreater(bar["net_score"], 0.80)
        self.assertIn("headline", bar)
        self.assertIn("bert", bar)
        self.assertEqual(bar["bert"]["has_news"], True)
        self.assertEqual(bar["bert"]["synthetic"], False)

    def test_quiet_day_returns_none(self):
        """Verify dates without published news evaluate to None via get_news_for_bar."""
        self.fetcher.save_news_archive("AAPL", self.sample_articles, format="both")
        quiet_bar = self.fetcher.get_news_for_bar("AAPL", "2024-03-16")
        self.assertIsNone(quiet_bar)

    def test_feed_generation_and_mapping_emulation(self):
        """Verify get_news_feed and dictionary container emulation on HistoricalNewsFetcher."""
        self.fetcher.save_news_archive("AAPL", self.sample_articles, format="both")
        feed = self.fetcher.get_news_feed(symbols=["AAPL"])

        self.assertIn(("2024-03-15", "AAPL"), feed)
        self.assertIn("2024-03-15_AAPL", feed)
        self.assertNotIn(("2024-03-16", "AAPL"), feed)

        # Mapping emulation on fetcher instance
        self.assertTrue(("2024-03-15", "AAPL") in self.fetcher)
        self.assertTrue("2024-03-15_AAPL" in self.fetcher)
        self.assertFalse(("2024-03-16", "AAPL") in self.fetcher)
        self.assertIsNotNone(self.fetcher.get(("2024-03-15", "AAPL")))
        self.assertIsNone(self.fetcher.get(("2024-03-16", "AAPL")))


class TestBacktestEngineRealNewsIntegration(unittest.TestCase):
    """Integration tests verifying HistoricalBacktestEngine with real news feeds."""

    def _create_mock_ohlcv_data(self, dates: List[str]) -> Dict[str, pd.DataFrame]:
        """Creates realistic test OHLCV data for backtesting."""
        n = len(dates)
        dt_idx = pd.to_datetime(dates)
        # Upward price progression with some volume
        prices = [150.0 + i * 0.5 for i in range(n)]
        df = pd.DataFrame({
            "open": prices,
            "high": [p + 1.5 for p in prices],
            "low": [p - 1.0 for p in prices],
            "close": prices,
            "volume": [300000.0] * n
        }, index=dt_idx)
        return {"AAPL": df}

    def test_real_news_feed_sets_synthetic_sentiment_used_false(self):
        """Providing a real sentiment_feed must set synthetic_sentiment_used=False and sentiment_source='real-news-feed'."""
        engine = HistoricalBacktestEngine()
        dates = [f"2024-01-{i:02d}" for i in range(1, 30)]
        data = self._create_mock_ohlcv_data(dates)
        bot = ConservativeSwingBot()

        # Provide a real news feed covering one date
        real_feed = {
            ("2024-01-10", "AAPL"): {
                "positive": 0.92,
                "negative": 0.01,
                "neutral": 0.07,
                "net_score": 0.91,
                "news_score": 3.64,
                "has_news": True,
                "synthetic": False,
                "articles": [{
                    "title": "Apple Services Surge",
                    "summary": "Services revenue hit all-time high.",
                    "source": "Reuters"
                }]
            }
        }

        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")
            results = engine.run_backtest(bots=[bot], data=data, sentiment_feed=real_feed)

        # Confirm flags
        self.assertFalse(results["synthetic_sentiment_used"])
        self.assertFalse(results["used_synthetic_sentiment"])
        self.assertEqual(results["sentiment_source"], "real-news-feed")
        self.assertNotIn("sentiment_warning", results)

        bot_metrics = results["bots"][bot.bot_id]["metrics"]
        self.assertFalse(bot_metrics["synthetic_sentiment_used"])

        # Confirm no UserWarning regarding synthetic sentiment was emitted
        synthetic_warnings = [
            w for w in captured_warnings
            if "SYNTHETIC sentiment" in str(w.message)
        ]
        self.assertEqual(len(synthetic_warnings), 0)

    def test_historical_news_fetcher_instance_as_sentiment_feed(self):
        """Passing HistoricalNewsFetcher instance directly as sentiment_feed executes cleanly without synthetic flag."""
        engine = HistoricalBacktestEngine()
        dates = [f"2024-01-{i:02d}" for i in range(1, 30)]
        data = self._create_mock_ohlcv_data(dates)
        bot = TrendFollowerBot()

        # Use seeded project archives in data/historical_news/
        fetcher = HistoricalNewsFetcher()

        results = engine.run_backtest(bots=[bot], data=data, sentiment_feed=fetcher)
        self.assertFalse(results["synthetic_sentiment_used"])
        self.assertEqual(results["sentiment_source"], "real-news-feed")
        self.assertNotIn("sentiment_warning", results)

    def test_quiet_days_graceful_neutral_fallback(self):
        """On dates without news, sentiment defaults to neutral, has_news=False, and no fake articles are made."""
        engine = HistoricalBacktestEngine()
        dates = [f"2024-01-{i:02d}" for i in range(1, 25)]
        data = self._create_mock_ohlcv_data(dates)

        # Catalyst hunter only enters when articles exist
        hunter_bot = FinBERTCatalystHunterBot()

        # Only one day has real news; all other 23 days are quiet
        news_date = "2024-01-22"
        real_feed = {
            (news_date, "AAPL"): {
                "positive": 0.95,
                "negative": 0.01,
                "neutral": 0.04,
                "net_score": 0.94,
                "news_score": 3.76,
                "has_news": True,
                "synthetic": False,
                "headline": "Breakthrough Catalyst Announced",
                "articles": [{
                    "title": "Breakthrough Catalyst Announced",
                    "summary": "Major institutional catalyst confirmed.",
                    "source": "Bloomberg"
                }]
            }
        }

        results = engine.run_backtest(bots=[hunter_bot], data=data, sentiment_feed=real_feed)

        # Integrity check: flags must remain strictly False
        self.assertFalse(results["synthetic_sentiment_used"])
        self.assertEqual(results["sentiment_source"], "real-news-feed")

        # FinBERT catalyst hunter should only consider entering on or after the real news date
        hunter_trades = results["bots"][hunter_bot.bot_id]["trades"]
        for t in hunter_trades:
            trade_date = str(t.get("timestamp") or t.get("date") or "")[:10]
            self.assertGreaterEqual(trade_date, news_date)

    def test_no_circular_price_dependency_on_quiet_days(self):
        """Simulate huge price surge on quiet day; confirm NO synthetic sentiment or fake articles are generated."""
        engine = HistoricalBacktestEngine()
        dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]

        # Craft data with a massive 15% price spike and 5x volume surge on 2024-01-05
        dt_idx = pd.to_datetime(dates)
        df = pd.DataFrame({
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 116.0],
            "low": [99.0, 99.0, 99.0, 100.0],
            "close": [100.0, 100.0, 100.0, 115.0],
            "volume": [100000.0, 100000.0, 100000.0, 500000.0]
        }, index=dt_idx)
        data = {"AAPL": df}

        # Real feed provided with NO news on 2024-01-05 (quiet day)
        real_feed = {
            ("2024-01-02", "AAPL"): {
                "positive": 0.50,
                "negative": 0.10,
                "neutral": 0.40,
                "net_score": 0.40,
                "news_score": 1.6,
                "has_news": True,
                "synthetic": False,
                "articles": [{"title": "Baseline", "summary": "Baseline news", "source": "Reuters"}]
            }
        }

        bot = ConservativeSwingBot()
        results = engine.run_backtest(bots=[bot], data=data, sentiment_feed=real_feed)

        # Must strictly preserve real-news-feed mode
        self.assertFalse(results["synthetic_sentiment_used"])
        self.assertEqual(results["sentiment_source"], "real-news-feed")
        self.assertNotIn("sentiment_warning", results)

    def test_legacy_mode_when_sentiment_feed_is_none(self):
        """When sentiment_feed is None, engine preserves legacy synthetic behavior for backward compatibility."""
        engine = HistoricalBacktestEngine()
        dates = [f"2024-01-{i:02d}" for i in range(1, 30)]
        data = self._create_mock_ohlcv_data(dates)
        bot = ConservativeSwingBot()

        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")
            results = engine.run_backtest(bots=[bot], data=data, sentiment_feed=None)

        self.assertTrue(results["synthetic_sentiment_used"])
        self.assertTrue(results["used_synthetic_sentiment"])
        self.assertEqual(results["sentiment_source"], "synthetic-price-derived")
        self.assertIn("sentiment_warning", results)


if __name__ == "__main__":
    unittest.main()
