"""
Historical News Ingestion & Archival Module.
Provides offline-capable, bidirectional JSON and Parquet news archive ingestion
keyed by ticker and trading date without lookahead leakage.
"""

from typing import List, Dict, Any, Optional, Tuple, Union, Set
from datetime import datetime, timezone
from pathlib import Path
import os
import sys
import json
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("HistoricalNewsFetcher")

# Optional PyArrow for high-performance Parquet serialization
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


class HistoricalNewsFetcher:
    """
    Ingests, standardizes, caches, and indexes historical financial news archives
    supporting both JSON and Parquet formats in `data/historical_news/`.
    Keys news by symbol/ticker and trading calendar date (YYYY-MM-DD).
    """

    DEFAULT_UNIVERSE: List[str] = ["^NSEI", "NIFTYBEES.NS", "NVDA", "AAPL", "MSFT", "SPY"]
    REQUIRED_FIELDS: List[str] = ["date", "symbol", "title"]

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        offline: bool = True
    ) -> None:
        """
        Initializes the fetcher with a persistent archive directory and offline mode flag.

        Args:
            cache_dir: Path to directory containing historical news archives.
                       Defaults to `data/historical_news/` under project root.
            offline: If True, operates strictly from local archive files without external network calls.
        """
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir).resolve()
        else:
            self.cache_dir = (Path(__file__).resolve().parent.parent / "data" / "historical_news").resolve()

        env_offline = os.environ.get("BACKTEST_OFFLINE", "").strip().lower() in ("1", "true", "yes")
        self.offline = bool(offline or env_offline)

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create news cache directory {self.cache_dir}: {e}")

        # In-memory index of parsed articles: {symbol: {date: [articles]}}
        self._cache: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        # Pre-computed bar aggregates: { (date, sym): aggregated_dict, f"{date}_{sym}": aggregated_dict }
        self._feed_cache: Dict[Any, Dict[str, Any]] = {}
        self._loaded_symbols: Set[str] = set()

    @staticmethod
    def get_safe_symbol(symbol: str) -> str:
        """
        Normalizes symbols containing special characters (e.g. '^' in '^NSEI')
        for safe filesystem operations on Windows and POSIX systems.
        """
        safe = str(symbol).strip().replace("^", "_").replace(":", "_").replace("/", "_").replace("\\", "_")
        return safe

    def get_candidate_paths(self, symbol: str) -> List[Path]:
        """
        Returns possible file paths for a symbol's archives in priority order:
        1. {safe_symbol}_news.parquet
        2. {safe_symbol}_news.json
        3. {symbol}_news.parquet
        4. {symbol}_news.json
        5. {symbol_without_caret}_news.parquet
        6. {symbol_without_caret}_news.json
        """
        safe_sym = self.get_safe_symbol(symbol)
        raw_sym = str(symbol).strip()
        stripped_sym = raw_sym.lstrip("^")

        candidates: List[Path] = [
            self.cache_dir / f"{safe_sym}_news.parquet",
            self.cache_dir / f"{safe_sym}_news.json",
        ]

        if raw_sym != safe_sym:
            candidates.extend([
                self.cache_dir / f"{raw_sym}_news.parquet",
                self.cache_dir / f"{raw_sym}_news.json",
            ])

        if stripped_sym not in (safe_sym, raw_sym):
            candidates.extend([
                self.cache_dir / f"{stripped_sym}_news.parquet",
                self.cache_dir / f"{stripped_sym}_news.json",
            ])

        # Also check prefixed variants if not already added (e.g. symbol passed as 'NSEI')
        clean_stripped = raw_sym.lstrip("^").lstrip("_")
        for pref in ("_", "^"):
            pref_sym = f"{pref}{clean_stripped}"
            if pref_sym not in (safe_sym, raw_sym, stripped_sym):
                candidates.extend([
                    self.cache_dir / f"{pref_sym}_news.parquet",
                    self.cache_dir / f"{pref_sym}_news.json",
                ])

        return candidates

    def _standardize_record(self, raw: Dict[str, Any], default_sym: str) -> Dict[str, Any]:
        """
        Normalizes and validates a single news record schema.
        Ensures standard keys: timestamp, date, symbol, title, snippet, source, link,
        sentiment_logits, positive, negative, neutral, net_score, news_score, dense_text_embedding.
        """
        sym = raw.get("symbol")
        if sym is None:
            sym = raw.get("ticker")
        if sym is None:
            sym = default_sym
        sym = str(sym).strip()

        raw_date = raw.get("date")
        timestamp = raw.get("timestamp")

        if raw_date is None and timestamp is not None:
            try:
                raw_date = str(timestamp)[:10]
            except Exception:
                raw_date = None

        if not raw_date:
            raw_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        else:
            raw_date = str(raw_date)[:10]

        if not timestamp:
            timestamp = f"{raw_date}T12:00:00Z"
        else:
            timestamp = str(timestamp)

        title_val = raw.get("title")
        if title_val is None:
            title_val = raw.get("headline")
        title = str(title_val or "").strip()

        snippet_val = raw.get("snippet")
        if snippet_val is None:
            snippet_val = raw.get("text")
        if snippet_val is None:
            snippet_val = raw.get("summary")
        snippet = str(snippet_val or "").strip()

        source = str(raw.get("source") or "Financial-Wire").strip()
        link_val = raw.get("link")
        if link_val is None:
            link_val = raw.get("url")
        link = str(link_val or "").strip()

        # Extract sentiment probabilities
        calibrated_probs = raw.get("calibrated_probs")
        pos = raw.get("positive")
        neg = raw.get("negative")
        neu = raw.get("neutral")

        if isinstance(calibrated_probs, dict):
            if pos is None:
                pos = calibrated_probs.get("positive")
            if neg is None:
                neg = calibrated_probs.get("negative")
            if neu is None:
                neu = calibrated_probs.get("neutral")

        pos = float(pos) if pos is not None and not (isinstance(pos, float) and np.isnan(pos)) else None
        neg = float(neg) if neg is not None and not (isinstance(neg, float) and np.isnan(neg)) else None
        neu = float(neu) if neu is not None and not (isinstance(neu, float) and np.isnan(neu)) else None

        # Extract sentiment logits safely without evaluating truth value of numpy arrays
        logits = raw.get("sentiment_logits")
        if logits is None:
            logits = raw.get("logits")

        parsed_logits: Optional[List[float]] = None
        if logits is not None:
            if isinstance(logits, (list, tuple, np.ndarray)):
                if len(logits) >= 3:
                    parsed_logits = [float(logits[0]), float(logits[1]), float(logits[2])]
            elif isinstance(logits, str):
                try:
                    parsed_list = json.loads(logits)
                    if isinstance(parsed_list, list) and len(parsed_list) >= 3:
                        parsed_logits = [float(parsed_list[0]), float(parsed_list[1]), float(parsed_list[2])]
                except Exception:
                    parsed_logits = None

        # Derive probabilities from logits if probabilities missing
        if pos is None or neg is None or neu is None:
            if parsed_logits is not None:
                exp_z = np.exp(np.array(parsed_logits) - np.max(parsed_logits))
                probs = exp_z / np.sum(exp_z)
                pos, neg, neu = float(probs[0]), float(probs[1]), float(probs[2])
            else:
                net_in = float(raw.get("net_score", 0.0) if raw.get("net_score") is not None else 0.0)
                if net_in > 0:
                    pos = 0.50 + net_in * 0.40
                    neg = max(0.01, (1.0 - pos) * 0.3)
                    neu = max(0.0, 1.0 - (pos + neg))
                elif net_in < 0:
                    neg = 0.50 + abs(net_in) * 0.40
                    pos = max(0.01, (1.0 - neg) * 0.3)
                    neu = max(0.0, 1.0 - (pos + neg))
                else:
                    pos, neg, neu = 0.20, 0.20, 0.60

        total_p = pos + neg + neu
        if total_p > 0 and abs(total_p - 1.0) > 1e-4:
            pos /= total_p
            neg /= total_p
            neu /= total_p

        net_score = raw.get("net_score")
        if net_score is None:
            net_score = pos - neg
        else:
            net_score = float(net_score)

        net_score = max(-1.0, min(1.0, round(net_score, 4)))
        news_score = raw.get("news_score")
        if news_score is None:
            news_score = round(net_score * 4.0, 4)
        else:
            news_score = float(news_score)

        # Dense text embedding (384-dimensional BGE-Small)
        dense_emb = raw.get("dense_text_embedding")
        if dense_emb is None:
            dense_emb = raw.get("embedding")

        parsed_emb: Optional[List[float]] = None
        if dense_emb is not None:
            try:
                if isinstance(dense_emb, str):
                    dense_emb = json.loads(dense_emb)
                arr = np.asarray(dense_emb, dtype=np.float64).flatten()
                if arr.shape[0] == 384:
                    parsed_emb = [float(v) for v in arr]
            except Exception:
                parsed_emb = None

        return {
            "timestamp": timestamp,
            "date": raw_date,
            "symbol": sym,
            "ticker": sym,
            "title": title,
            "snippet": snippet,
            "source": source,
            "link": link,
            "sentiment_logits": parsed_logits,
            "positive": round(pos, 4),
            "negative": round(neg, 4),
            "neutral": round(neu, 4),
            "net_score": net_score,
            "news_score": news_score,
            "dense_text_embedding": parsed_emb,
            "embedding": parsed_emb,
            "calibrated_probs": {
                "positive": round(pos, 4),
                "negative": round(neg, 4),
                "neutral": round(neu, 4),
            }
        }

    def load_symbol_news(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Loads all archived news articles for a given symbol from Parquet or JSON.
        Caches in-memory for subsequent fast bar-by-bar queries.
        """
        if symbol in self._cache:
            articles: List[Dict[str, Any]] = []
            for date_arts in self._cache[symbol].values():
                articles.extend(date_arts)
            return articles

        candidates = self.get_candidate_paths(symbol)
        loaded_records: List[Dict[str, Any]] = []

        for path in candidates:
            if not path.exists():
                continue

            try:
                if path.suffix.lower() == ".parquet":
                    df = pd.read_parquet(path)
                    records = df.to_dict(orient="records")
                    loaded_records = [self._standardize_record(r, symbol) for r in records]
                    logger.info(f"Loaded {len(loaded_records)} articles for {symbol} from {path.name}")
                    break
                elif path.suffix.lower() == ".json":
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        loaded_records = [self._standardize_record(r, symbol) for r in data]
                    elif isinstance(data, dict) and "articles" in data:
                        loaded_records = [self._standardize_record(r, symbol) for r in data["articles"]]
                    logger.info(f"Loaded {len(loaded_records)} articles for {symbol} from {path.name}")
                    break
            except Exception as e:
                logger.warning(f"Error loading news archive {path}: {e}")
                continue

        # Organize into symbol -> date -> list of articles
        sym_dict: Dict[str, List[Dict[str, Any]]] = {}
        for rec in loaded_records:
            d = rec["date"]
            if d not in sym_dict:
                sym_dict[d] = []
            sym_dict[d].append(rec)

        self._cache[symbol] = sym_dict
        safe_sym = self.get_safe_symbol(symbol)
        self._cache[safe_sym] = sym_dict
        clean_stripped = symbol.lstrip("^").lstrip("_")
        self._cache[clean_stripped] = sym_dict
        self._cache[f"_{clean_stripped}"] = sym_dict
        self._cache[f"^{clean_stripped}"] = sym_dict

        self._loaded_symbols.add(symbol)
        self._loaded_symbols.add(safe_sym)
        self._loaded_symbols.add(clean_stripped)
        return loaded_records

    def load_all(self, symbols: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Loads news archives for all requested symbols or discovers all available archives
        in `cache_dir`.
        """
        sym_list = symbols or self.DEFAULT_UNIVERSE
        results: Dict[str, List[Dict[str, Any]]] = {}

        # Load specified universe
        for sym in sym_list:
            arts = self.load_symbol_news(sym)
            if arts:
                results[sym] = arts

        # Also discover any extra files present in cache_dir
        if self.cache_dir.exists():
            for f in self.cache_dir.iterdir():
                if f.is_file() and (f.suffix.lower() in (".json", ".parquet")):
                    name = f.stem.replace("_news", "")
                    if name not in results and name not in sym_list:
                        arts = self.load_symbol_news(name)
                        if arts:
                            results[name] = arts

        return results

    def get_news_for_bar(self, symbol: str, trade_date: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves and aggregates all real news events for a symbol on a specific calendar trading date.

        Returns:
            Aggregated dictionary if news exists on that date; None if no news is present (quiet day).
        """
        date_str = str(trade_date)[:10]

        # Ensure symbol is loaded
        if symbol not in self._cache:
            self.load_symbol_news(symbol)

        date_map = self._cache.get(symbol, {})
        if not date_map:
            safe_sym = self.get_safe_symbol(symbol)
            date_map = self._cache.get(safe_sym, {})
            if not date_map:
                clean_stripped = symbol.lstrip("^").lstrip("_")
                date_map = (
                    self._cache.get(clean_stripped, {})
                    or self._cache.get(f"_{clean_stripped}", {})
                    or self._cache.get(f"^{clean_stripped}", {})
                )
        articles = date_map.get(date_str) if date_map else None
        if not articles:
            return None

        # Aggregate multi-article days
        pos_list = [a["positive"] for a in articles]
        neg_list = [a["negative"] for a in articles]
        neu_list = [a["neutral"] for a in articles]

        avg_pos = round(float(np.mean(pos_list)), 4)
        avg_neg = round(float(np.mean(neg_list)), 4)
        avg_neu = round(float(np.mean(neu_list)), 4)

        # Re-normalize to 1.0
        tot = avg_pos + avg_neg + avg_neu
        if tot > 0:
            avg_pos = round(avg_pos / tot, 4)
            avg_neg = round(avg_neg / tot, 4)
            avg_neu = round(avg_neu / tot, 4)

        net_score = round(avg_pos - avg_neg, 4)
        news_score = round(net_score * 4.0, 4)

        # Average logits if available
        logits_list = [a["sentiment_logits"] for a in articles if a.get("sentiment_logits") is not None]
        avg_logits: Optional[List[float]] = None
        if logits_list:
            avg_logits = [round(float(v), 4) for v in np.mean(logits_list, axis=0)]

        # Average embeddings if available
        emb_list = [a["dense_text_embedding"] for a in articles if a.get("dense_text_embedding") is not None]
        avg_emb: Optional[List[float]] = None
        if emb_list:
            mean_vec = np.mean(emb_list, axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 1e-12:
                mean_vec = mean_vec / norm
            avg_emb = [round(float(v), 6) for v in mean_vec]

        # Best headline: highest absolute polarity differential
        top_art = max(articles, key=lambda a: abs(a.get("net_score", 0.0)))
        top_headline = top_art.get("title", "")

        bert_dict = {
            "news_score": news_score,
            "net_score": net_score,
            "positive": avg_pos,
            "negative": avg_neg,
            "neutral": avg_neu,
            "info": f"FinBERT[Pos={avg_pos:.2f}, Neg={avg_neg:.2f}, Net={net_score:.2f}]",
            "headline": top_headline,
            "synthetic": False,
            "has_news": True,
            "polarity": "positive" if net_score > 0.05 else ("negative" if net_score < -0.05 else "neutral"),
            "confidence": round(max(avg_pos, avg_neg, avg_neu), 4),
            "sentiment_logits": avg_logits,
            "dense_text_embedding": avg_emb,
            "embedding": avg_emb,
        }

        snapshot = {
            "has_news": True,
            "synthetic": False,
            "symbol": symbol,
            "date": date_str,
            "positive": avg_pos,
            "negative": avg_neg,
            "neutral": avg_neu,
            "net_score": net_score,
            "news_score": news_score,
            "sentiment_logits": avg_logits,
            "dense_text_embedding": avg_emb,
            "embedding": avg_emb,
            "calibrated_probs": {
                "positive": avg_pos,
                "negative": avg_neg,
                "neutral": avg_neu
            },
            "headline": top_headline,
            "articles": articles,
            "bert": bert_dict,
            "info": bert_dict["info"]
        }

        return snapshot

    def get_news_feed(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[Any, Dict[str, Any]]:
        """
        Builds a comprehensive fast-lookup sentiment feed dictionary keyed by:
        - `(date_str, symbol)`
        - `f"{date_str}_{symbol}"`
        for direct injection into `HistoricalBacktestEngine.run_backtest(..., sentiment_feed=feed)`.
        """
        sym_list = symbols or self.DEFAULT_UNIVERSE
        feed: Dict[Any, Dict[str, Any]] = {}

        for sym in sym_list:
            if sym not in self._cache:
                self.load_symbol_news(sym)
            date_map = self._cache.get(sym, {})
            for d in date_map.keys():
                if start_date and d < start_date:
                    continue
                if end_date and d > end_date:
                    continue
                bar_news = self.get_news_for_bar(sym, d)
                if bar_news:
                    feed[(d, sym)] = bar_news
                    feed[f"{d}_{sym}"] = bar_news

        self._feed_cache.update(feed)
        return feed

    # Alias for build_sentiment_feed
    build_sentiment_feed = get_news_feed

    # Mapping emulation: allows passing HistoricalNewsFetcher directly as sentiment_feed
    def __contains__(self, key: Any) -> bool:
        if isinstance(key, tuple) and len(key) == 2:
            date_str, sym = key
            return self.get_news_for_bar(str(sym), str(date_str)[:10]) is not None
        elif isinstance(key, str) and "_" in key:
            parts = key.split("_", 1)
            date_str, sym = parts[0], parts[1]
            return self.get_news_for_bar(sym, date_str[:10]) is not None
        return False

    def __getitem__(self, key: Any) -> Dict[str, Any]:
        item = self.get(key)
        if item is None:
            raise KeyError(f"No news for key: {key}")
        return item

    def get(self, key: Any, default: Any = None) -> Any:
        if isinstance(key, tuple) and len(key) == 2:
            date_str, sym = key
            res = self.get_news_for_bar(str(sym), str(date_str)[:10])
            return res if res is not None else default
        elif isinstance(key, str) and "_" in key:
            parts = key.split("_", 1)
            date_str, sym = parts[0], parts[1]
            res = self.get_news_for_bar(sym, date_str[:10])
            return res if res is not None else default
        return default

    def save_news_archive(
        self,
        symbol: str,
        articles: List[Dict[str, Any]],
        format: str = "both"
    ) -> Dict[str, str]:
        """
        Saves a list of article dictionaries to JSON and/or Parquet in `cache_dir`.

        Args:
            symbol: Ticker symbol (e.g. 'AAPL', '^NSEI').
            articles: List of article dictionaries.
            format: 'json', 'parquet', or 'both'.

        Returns:
            Dictionary of saved file paths: {'json': path, 'parquet': path}.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        safe_sym = self.get_safe_symbol(symbol)
        saved_paths: Dict[str, str] = {}

        # Standardize records first
        standardized = [self._standardize_record(a, symbol) for a in articles]

        # 1. Save JSON
        if format.lower() in ("json", "both"):
            json_path = self.cache_dir / f"{safe_sym}_news.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(standardized, f, indent=2, ensure_ascii=False)
            saved_paths["json"] = str(json_path)

        # 2. Save Parquet
        if format.lower() in ("parquet", "both"):
            parquet_path = self.cache_dir / f"{safe_sym}_news.parquet"
            # Format dataframe cleanly: serialize list fields as json or pyarrow list
            flat_rows = []
            for item in standardized:
                row = dict(item)
                # Keep sentiment_logits and dense_text_embedding as lists for pyarrow
                flat_rows.append(row)
            df = pd.DataFrame(flat_rows)
            try:
                df.to_parquet(parquet_path, index=False, engine="pyarrow")
                saved_paths["parquet"] = str(parquet_path)
            except Exception as e:
                logger.warning(f"Could not save Parquet archive {parquet_path}: {e}")

        # Update cache in memory
        sym_dict: Dict[str, List[Dict[str, Any]]] = {}
        for rec in standardized:
            d = rec["date"]
            if d not in sym_dict:
                sym_dict[d] = []
            sym_dict[d].append(rec)
        self._cache[symbol] = sym_dict
        self._loaded_symbols.add(symbol)

        return saved_paths
