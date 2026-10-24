# Project: Machine-Learned Multi-Modal Fusion Engine & Real Historical News Ingestion

## Architecture
The platform is an autonomous multi-bot trading arena with market sentiment research, historical backtesting, and walk-forward parameter calibration. This project enhances the architecture with a learned multi-modal fusion engine and real historical news ingestion:
- **Multi-Modal Fusion Layer (`src/fusion_engine.py`)**:
  - Implements `MultiModalFusionEngine` predicting forward trade edge probability $P(\text{edge}) \in [0.0, 1.0]$.
  - 3-tier fallback hierarchy: Tier 1 LightGBM with monotonic constraints (`monotone_constraints` with +1 on sentiment), Tier 2 Scikit-Learn (LogisticRegression / Isotonic), Tier 3 Pure NumPy calibrated logistic fallback with strictly positive sentiment weight ($w_{\text{sent}} > 0$).
  - Features: FinBERT sentiment logits (pos/neg/neu), BGE-Small dense text components (384-d), and technical indicators (14-period RSI, 5-day momentum, 20-day Volume Z-Score, 14-period normalized ATR volatility).
- **Execution & Candidate Selection Layer (`src/paper_trader.py`)**:
  - Replaces static heuristic blending formula ($0.40 \cdot \text{sentiment} + 0.35 \cdot \text{tech} + 0.25 \cdot \text{catalyst}$) in `calculate_probability_weighted_upside()` with `MultiModalFusionEngine.predict_edge_probability()`.
  - Preserves all 5 bot archetypes (`ConservativeSwingBot`, `AggressiveBreakoutBot`, `ContrarianDipBuyerBot`, `FinBERTCatalystHunterBot`, `TrendFollowerBot`) and their distinct $[0, 10]$ conviction formulas in `evaluate_entry()`.
- **Historical News & Backtest Pipeline (`src/backtest_engine.py`, `data/historical_news/`)**:
  - Implements `HistoricalNewsFetcher` ingesting Parquet/JSON archives keyed by symbol and trading date.
  - Eliminates circular price-derived sentiment generation (`daily_ret`, `vol_surge`, `mom5d_val`).
  - Implements engine-level mode: `is_real_feed_mode = sentiment_feed is not None`.
  - Quiet trading days gracefully fall back to neutral sentiment (`neutral=1.0, net_score=0.0, has_news=False, articles=[]`) with sentiment weighting disabled.
  - Strictly guarantees `synthetic_sentiment_used = False` when historical news feeds are provided.
- **Probabilistic Calibration, Reliability & Parameter Remediation (`src/finbert_analyzer.py`, `src/calibration.py`, `data/bot_calibrated_parameters.json`)**:
  - Isotonic and Platt scaling calibration routines with pure-NumPy fallback.
  - Enforces `FinBERTAnalyzer` singleton (`_instance`) via dependency injection (`analyzer: Optional[FinBERTAnalyzer] = None`).
  - Fixes `HistoricalDataFetcher(offline_only=offline)` bug in `src/calibration.py:540` to `offline=offline`.
  - Re-calibrates and populates missing out-of-sample metrics for `conservative_swing` in `data/bot_calibrated_parameters.json`.
  - Applies sample weighting / confidence tiers to prevent degraded/borderline strategies from over-weighting fusion training.
- **Verification, Adversarial Hardening & Forensic Integrity (`tests/`, `tests/test_fusion_engine.py`)**:
  - Full regression protection across all 221 existing tests.
  - New test suite in `tests/test_fusion_engine.py` testing monotonicity, [0.0, 1.0] range, and fallback mechanics.
  - Walk-forward strategy evaluation verifying deployed strategy satisfies $\text{OOS Return} > 0.0\%$ and $\text{OOS Profit Factor} \ge 1.0$.
  - Adversarial stress testing (Challengers) and Forensic Integrity Audit (Auditor).

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Learned Multi-Modal Fusion Engine | `src/fusion_engine.py` with 3-tier fallback (LightGBM, Sklearn, pure NumPy) and monotonic constraint on net sentiment. | M1 | Survey 1, ORIGINAL_REQUEST R1 |
| 2 | Technical Indicator Expansion | 14-period RSI, 20-day Volume Z-Score, 14-period normalized ATR volatility in `paper_trader.py` and `backtest_engine.py`. | M1 | Survey 1, ORIGINAL_REQUEST R1 |
| 3 | Decoupled Candidate Ranking & Bot Conviction Interface | Replace 0.40/0.35/0.25 heuristic in `calculate_probability_weighted_upside()` while preserving 5 bot conviction formulas in `evaluate_entry()`. | M1 | Survey 1, ORIGINAL_REQUEST R1 |
| 4 | Real Historical News Archive Ingestion | `HistoricalNewsFetcher` supporting JSON and Parquet archives in `data/historical_news/` keyed by ticker and date. | M2 | Survey 2, ORIGINAL_REQUEST R2 |
| 5 | Elimination of Circular Synthetic Sentiment | Remove price-derived sentiment fabrication; guarantee `synthetic_sentiment_used = False` when news is provided; graceful quiet-day fallback. | M2 | Survey 2, ORIGINAL_REQUEST R2 |
| 6 | Probabilistic Calibration & FinBERT Singleton | Pure NumPy / Platt / Isotonic probability calibration; enforce `FinBERTAnalyzer` singleton dependency injection with temperature scaling $T=1.5$. | M3 | Survey 3, ORIGINAL_REQUEST R3 |
| 7 | Offline Compatibility & Dependency Updates | Update `requirements.txt` (`lightgbm>=4.3.0`, `scikit-learn>=1.4.0`, `numpy>=1.26.0`); verify offline mode (`offline=True`). | M4 | Survey 3, ORIGINAL_REQUEST R4 |
| 8 | 221 Regression Tests & New Fusion Suite | Zero regressions across existing 221 tests; comprehensive new test suite in `tests/test_fusion_engine.py`. | M4 | Survey 1-3, ORIGINAL_REQUEST R4 |
| 9 | Calibration Bug Fix & Parameter Remediation | Fix `src/calibration.py:540` bug; populate missing `out_of_sample_metrics` for `conservative_swing` in `data/bot_calibrated_parameters.json` with confidence tiers. | M3 | Survey 3, ORIGINAL_REQUEST R5 |
| 10 | Strategy Verification, Adversarial Hardening & Audit | Verify OOS Return > 0.0%, OOS PF >= 1.0; Challenger empirical testing; Forensic Integrity Audit. | M5 | Survey 1-3, Acceptance Criteria |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | M1: Learned Multi-Modal Fusion Engine & Candidate Ranking | `src/fusion_engine.py`, `src/paper_trader.py`: LightGBM/Sklearn/NumPy fallback, monotonic constraints, technical indicators (RSI, Vol Z, ATR), replace 0.40/0.35/0.25 formula. | none | DONE |
| 2 | M2: Real Historical News Pipeline & Elimination of Synthetic Sentiment | `src/backtest_engine.py`, `data/historical_news/`: `HistoricalNewsFetcher`, Parquet/JSON ingestion, quiet-day neutral fallback, ensure `synthetic_sentiment_used = False`. | none | PLANNED |
| 3 | M3: Calibration, FinBERT Singleton DI & Parameter Remediation | `src/finbert_analyzer.py`, `src/calibration.py`, `data/bot_calibrated_parameters.json`: Temperature scaling DI, calibration bug fix, re-calibrate `conservative_swing`, confidence weighting. | none | PLANNED |
| 4 | M4: Offline Compatibility, Dependency Updates & Fusion Test Suite | `requirements.txt`, `tests/test_fusion_engine.py`: Add dependencies, ensure offline execution, achieve 100% pass on new tests and 221 existing tests. | M1, M2, M3 | PLANNED |
| 5 | M5: Strategy Verification, Adversarial Stress & Forensic Audit | Verification of OOS Return > 0.0% and OOS PF >= 1.0, 2 Challengers adversarial tests, Forensic Integrity Audit. | M1, M2, M3, M4 | PLANNED |

## Interface Contracts
### `src/fusion_engine.py` ↔ `src/paper_trader.py` (M1)
- `MultiModalFusionEngine.predict_edge_probability(features: Dict[str, Any]) -> float`:
  - Returns continuous float in $[0.0, 1.0]$.
  - Features dict contains:
    - `sentiment_logits`: Tuple/List[float] (pos, neg, neu) or `net_sentiment`: float in $[-1.0, 1.0]$
    - `dense_text_embedding`: Optional[np.ndarray] (384-d BGE-Small)
    - `rsi_14`: float in $[0.0, 100.0]$
    - `momentum_5d`: float (e.g. percentage return)
    - `volume_z_score`: float
    - `atr_volatility`: float
  - Monotonicity guarantee: For identical technical features, $\text{net\_sentiment}_A > \text{net\_sentiment}_B \implies P(\text{edge})_A > P(\text{edge})_B$.
- `calculate_probability_weighted_upside(...)` in `src/paper_trader.py`:
  - Retains existing signature and return dict format:
    `{'upside_probability_pct', 'expected_return_pct', 'prob_weighted_upside', 'p_sentiment', 'p_tech', 'c_clarity', 'edge_probability'}`.
  - Replaces heuristic line 314 with `p_upside = fusion_engine.predict_edge_probability(...)`.

### `src/backtest_engine.py` (M2)
- `HistoricalNewsFetcher.get_news_for_bar(symbol: str, trade_date: str) -> Dict[str, Any]`:
  - Ingests from JSON/Parquet archives in `data/historical_news/`.
  - Returns dict with keys `articles`, `has_news`, `net_sentiment`, `finbert_logits`, `calibrated_probs`.
- In `HistoricalBacktestEngine.run_backtest(...)`:
  - `is_real_feed_mode = sentiment_feed is not None`
  - If `is_real_feed_mode` is True:
    - If date has news: use feed news, `synthetic_sentiment_used = False`, `sentiment_source = "real-news-feed"`.
    - If date has NO news: graceful neutral fallback (`net_score = 0.0, neutral = 1.0, articles = [], has_news = False`), disable sentiment weighting, `synthetic_sentiment_used = False`.

### `src/finbert_analyzer.py` & `src/calibration.py` (M3)
- `FinBERTAnalyzer.get_instance(temperature: float = 1.5) -> FinBERTAnalyzer`:
  - Thread-safe singleton returning `_instance`.
  - Callers accept `analyzer: Optional[FinBERTAnalyzer] = None` and default to `FinBERTAnalyzer.get_instance()`.
- `src/calibration.py`:
  - Line 540 fixed to `HistoricalDataFetcher(offline=offline)`.
  - Populates `data/bot_calibrated_parameters.json` with valid `out_of_sample_metrics` for `conservative_swing`.

## Code Layout
```
D:\market-sentiment-researcher\
├── src\
│   ├── config.py                  # Configurations
│   ├── scraper.py                 # News ingestion
│   ├── embeddings.py              # FastEmbed BGE-Small vector embeddings
│   ├── db.py                      # Supabase client and local storage
│   ├── finbert_analyzer.py        # FinBERT sentiment analyzer (singleton + temperature scaling)
│   ├── probability_calibration.py # Platt & Isotonic calibration
│   ├── fusion_engine.py           # Multi-Modal Fusion Engine (LightGBM/Sklearn/NumPy) (NEW - M1)
│   ├── news_fetcher.py            # Historical news archive fetcher (JSON/Parquet) (NEW/M2)
│   ├── relevance.py               # Targeted NIFTY relevance
│   ├── backtest_engine.py         # Historical bar replay, indicators, and backtesting (M1, M2)
│   ├── calibration.py             # Walk-forward parameter calibration & optimization (M3)
│   └── paper_trader.py            # Bot archetypes & candidate selection (M1, M3)
├── tests\
│   ├── test_multi_bot.py          # Existing tournament tests
│   ├── test_indian_market_relevance.py # Existing Indian market tests
│   ├── test_backtest_calibration.py    # Existing backtest & calibration tests
│   ├── test_critical_fixes.py     # Existing critical fix tests
│   └── test_fusion_engine.py      # New unit tests for fusion engine & news pipeline (NEW - M4)
├── data\
│   ├── historical_bars\           # Cached OHLCV historical bars
│   ├── historical_news\           # Cached historical news archives (JSON/Parquet) (NEW - M2)
│   └── bot_calibrated_parameters.json # Calibrated bot parameters with OOS metrics (M3)
├── requirements.txt               # Updated with lightgbm, scikit-learn, numpy (M4)
├── run_pipeline.py                # CLI entry point
└── PROJECT.md                     # Project architecture and milestones
```
