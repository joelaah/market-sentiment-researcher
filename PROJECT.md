# Project: Market Sentiment Research & Paper Trading Platform — Indian Market Extension

## Architecture
The platform is an autonomous market intelligence and paper trading engine.
- **Market Data & Trading Layer (`src/paper_trader.py`)**: Uses `yfinance` for price, volume, and momentum metrics. Evaluates risk via `BorderlistGuardrail`. Runs a multi-bot tournament managing isolated portfolios with local state fallback.
- **News Ingestion & Cleaning Layer (`src/scraper.py`)**: Ingests RSS feeds across global and Indian financial sources, cleans HTML, normalizes unicode, deduplicates articles, and parses timestamps.
- **Vector Embeddings & Semantic Search (`src/embeddings.py`, `src/db.py`)**: Embeds article texts using FastEmbed (`BAAI/bge-small-en-v1.5`, 384 dimensions). Indexes articles into Supabase PostgreSQL pgvector (`public.market_news`) with HNSW index and RPC `match_market_news`. Includes in-memory fallback if Supabase is offline.
- **Sentiment & Relevance Analysis (`src/finbert_analyzer.py`, `src/relevance.py`)**: Employs `ProsusAI/finbert` for domain-specific financial sentiment (probabilities, polarity label, confidence score, and net score). Evaluates article relevance across Entity/Ticker matching, Semantic Similarity against NIFTY macro anchors, and Sentiment Materiality.
- **CLI & Orchestration (`run_pipeline.py`)**: Exposes pipeline operations including scraping, research queries, bot tournament runs, and the automated news relevance checker (`--check-news --symbol NIFTY`) with structured tabular briefs.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Indian Instrument Tracking | Add `^NSEI` (NIFTY 50 index) and `NIFTYBEES.NS` (ETF) to `DEFAULT_WATCHLIST` and `WATCHLIST`. Support price and technical indicators fetching via yfinance. | M1 | Survey 1, ORIGINAL_REQUEST R1 |
| 2 | Currency-Aware & Off-Hours Guardrails | Update `BorderlistGuardrail.evaluate()` with trusted bypass for `^NSEI` and `NIFTYBEES.NS` to avoid off-hours/missing-volume rejections. Ensure currency-specific floors (INR vs USD) prevent false penny-stock rejections of `NIFTYBEES.NS` while blocking genuine penny stocks. | M1 | Survey 1, ORIGINAL_REQUEST R1 |
| 3 | Indian Ticker & Alias Discovery | Update `extract_candidate_tickers_from_articles` and `MarketSnapshot.capture` to support `.NS` suffixes, lengths > 5 chars, non-word prefix `^NSEI`, and keyword-to-ticker mapping (Nifty, Sensex -> `^NSEI`, `NIFTYBEES.NS`). | M1 | Survey 1, ORIGINAL_REQUEST R1 |
| 4 | Indian Financial RSS Ingestion | Add reputable Indian financial feeds (Google News India Business, Google News Nifty/Sensex, Economic Times Markets/Stocks, Livemint, Moneycontrol) to `FEEDS` in `src/scraper.py`. | M2 | Survey 2, ORIGINAL_REQUEST R2 |
| 5 | Article Cleaning & In-Memory Deduplication | Enhance `clean_html()` with `html.unescape()`, whitespace normalization, and url/title-based deduplication in `fetch_financial_news()` to prevent duplicate vector indexing. | M2 | Survey 2, ORIGINAL_REQUEST R2 |
| 6 | Resilient Vector Store Operations & Fallback | Wrap `search_similar_news()` and database operations in `src/db.py` with robust try/except error handling and in-memory cosine similarity fallback when Supabase is unreachable. | M2 | Survey 2, ORIGINAL_REQUEST R2, R4 |
| 7 | FinBERT Polarity & Confidence Output | Enhance `FinBERTAnalyzer.analyze()` in `src/finbert_analyzer.py` to return explicit `polarity` ("positive", "negative", "neutral") and `confidence` (float probability) alongside existing probabilities and net score. | M3 | Survey 3, ORIGINAL_REQUEST R3 |
| 8 | Targeted News Relevance Evaluator | Implement `NiftyRelevanceChecker` in `src/relevance.py` evaluating (1) Entity/Ticker matches (NIFTY, Sensex, RBI, Budget, heavyweights), (2) Semantic cosine similarity against NIFTY macro anchors, (3) FinBERT sentiment & materiality. | M3 | Survey 3, ORIGINAL_REQUEST R3 |
| 9 | CLI Relevance Command & Intelligence Brief | Add `--check-news` flag and `--symbol` argument (default `NIFTY`) in `run_pipeline.py`. Format and print structured intelligence brief via `tabulate`. Ensure offline fallback. | M3 | Survey 3, ORIGINAL_REQUEST R3 |
| 10 | Comprehensive Test Suite & Unittest Discovery | Implement `tests/test_indian_market_relevance.py` covering Indian feed parsing, deduplication, relevance scoring, FinBERT outputs, guardrail evaluation, and CLI execution. All tests pass with `python -m unittest discover tests`. | M4 | Survey 1-3, ORIGINAL_REQUEST R4 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | M1: NIFTY Market Data & Trading Universe | `src/paper_trader.py`: Watchlist expansion, `BorderlistGuardrail` trusted whitelist & currency awareness, ticker extraction & alias matching | none | DONE |
| 2 | M2: Indian Financial News Ingestion & pgvector Storage | `src/scraper.py`, `src/db.py`: Indian feeds addition, HTML unescaping & deduplication, robust db error handling & offline fallback | none | DONE |
| 3 | M3: Targeted News Relevance Checker & CLI Interface | `src/finbert_analyzer.py`, `src/relevance.py`, `run_pipeline.py`: FinBERT polarity/confidence, relevance evaluator, CLI command & brief formatter | M1, M2 | DONE |
| 4 | M4: Comprehensive Verification & Test Suite | `tests/test_indian_market_relevance.py`: Full test coverage, mock fallbacks, unittest discovery verification, independent review and audit | M1, M2, M3 | DONE |

## Interface Contracts
### `src/paper_trader.py` (M1)
- `BorderlistGuardrail.evaluate(symbol: str, price: float, volume: Optional[float] = None, market_cap: Optional[float] = None) -> Tuple[bool, Optional[str]]`:
  - Returns `(True, None)` for `NIFTYBEES.NS` and `^NSEI` even when `volume=None` and `market_cap=None`.
  - Enforces currency-aware price floor (INR >= 20.0 for Indian equities, USD >= 5.0 for US equities). Rejects Indian penny stocks under ₹20 while allowing `NIFTYBEES.NS` (~₹264).
- `get_stock_market_metrics(symbol: str) -> Dict[str, Any]`:
  - Returns dictionary with `price`, `volume`, `market_cap`, `sma20`, `momentum_5d`, `currency`, `valid`.

### `src/scraper.py` ↔ `src/db.py` (M2)
- `fetch_financial_news(limit_per_feed: int = 10) -> List[Dict[str, Any]]`:
  - Returns cleaned, deduplicated article dicts with keys: `title`, `link`, `source`, `summary`, `published_at`, `content`.
- `search_similar_news(client, query_embedding: List[float], match_threshold: float = 0.3, match_count: int = 5, fallback_articles: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]`:
  - Never raises unhandled exceptions. Returns matching article dicts with `similarity`.

### `src/finbert_analyzer.py` ↔ `src/relevance.py` (M3)
- `FinBERTAnalyzer.analyze(text: str) -> Dict[str, Any]`:
  - Returns dict with: `positive`, `negative`, `neutral`, `net_score`, `polarity` (str: "positive"|"negative"|"neutral"), `confidence` (float: 0.0 to 1.0).
- `NiftyRelevanceChecker.evaluate_article(article: Dict[str, Any]) -> Dict[str, Any]`:
  - Returns dict with: `is_relevant` (bool), `relevance_score` (float 0.0-1.0), `entity_matches` (List[str]), `semantic_similarity` (float 0.0-1.0), `sentiment` (`polarity`, `confidence`, `net_score`), `materiality` (str: "HIGH"|"MEDIUM"|"LOW"), `reason` (str).
- `NiftyRelevanceChecker.check_news(articles: Optional[List[Dict[str, Any]]] = None, symbol: str = "NIFTY", top_n: int = 10) -> List[Dict[str, Any]]`:
  - Ingests or receives articles, filters for relevant news, sorts by relevance and materiality.
- `run_news_relevance_check(symbol: str = "NIFTY") -> None`:
  - Formats results into a table via `tabulate` and prints the structured intelligence brief.

## Code Layout
```
D:\market-sentiment-researcher\
├── src\
│   ├── config.py             # Environment configurations
│   ├── scraper.py            # Financial news RSS ingestion & cleaning
│   ├── embeddings.py         # FastEmbed dense vector generation (384-d)
│   ├── db.py                 # Supabase pgvector client, upsert, and similarity RPC
│   ├── finbert_analyzer.py   # FinBERT financial sentiment model
│   ├── relevance.py          # Targeted NIFTY news relevance checker & brief
│   ├── researcher.py         # Semantic research query utility
│   └── paper_trader.py       # Paper trading engine, guardrails, multi-bot tournament
├── tests\
│   ├── test_multi_bot.py     # Existing 20 tests for tournament & guardrails
│   └── test_indian_market_relevance.py # New unit & integration tests
├── data\                     # Local tournament state fallbacks
├── schema.sql                # PostgreSQL table and HNSW index definitions
├── requirements.txt          # Python dependencies
├── run_pipeline.py           # Top-level CLI entry point
└── PROJECT.md                # Project architecture, inventory, and milestones
```
