# Project: 3-Year Historical Backtesting, Training Calibration, and Persistence Engine

## Architecture
The platform is an autonomous multi-bot trading arena with historical backtesting, walk-forward calibration, and Supabase knowledge persistence.
- **Market Data & Historical Replay Layer (`src/backtest_engine.py`, `src/paper_trader.py`)**:
  - Ingests and caches 3 years of daily OHLCV bars across US (`NVDA`, `AAPL`, `MSFT`, `SPY`) and Indian (`^NSEI`, `NIFTYBEES.NS`) universe instruments.
  - Replays market history bar-by-bar strictly chronologically without lookahead bias.
  - Computes standard quantitative metrics: Win Rate %, Profit Factor, Max Drawdown (MDD) %, Net Return %, and CAGR %.
- **Walk-Forward Calibration & Optimization Layer (`src/calibration.py`, `src/paper_trader.py`)**:
  - Refactors the 5 bot archetypes (`ConservativeSwingBot`, `AggressiveBreakoutBot`, `ContrarianDipBuyerBot`, `FinBERTCatalystHunterBot`, `TrendFollowerBot`) to accept dynamic parameter configurations while preserving default backwards compatibility.
  - Implements walk-forward validation (in-sample calibration and out-of-sample testing) to optimize stop-loss, take-profit, momentum thresholds, and conviction cutoffs without curve-fitting.
- **Supabase Persistence & Resilient Offline Fallback (`src/db.py`, `schema.sql`, `src/paper_trader.py`)**:
  - Persists training run histories to `bot_training_runs` and optimal parameters to `bot_calibrated_parameters`.
  - Implements 3-tier fallback resolution: Supabase Cloud -> Local JSON Cache (`data/bot_calibrated_parameters.json`) -> Hardcoded Persona Defaults.
  - Dynamically synchronizes `MultiBotTournamentEngine` with active calibrated parameters.
- **CLI & Orchestration (`run_pipeline.py`)**:
  - Exposes `python run_pipeline.py --train --years 3` to execute historical calibration and persist parameters.
  - Exposes `python run_pipeline.py --backtest-report` to display comparative performance metrics across all 5 bot archetypes in a formatted table.
- **Automated Verification (`tests/`)**:
  - Fast, offline-safe unit and integration tests covering replay accuracy, metric calculations, walk-forward splits, parameter loading, and CLI execution.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Historical Data Ingestion & Caching | Ingest 3-year daily OHLCV bars for US and Indian assets (`^NSEI`, `NIFTYBEES.NS`, `NVDA`, `AAPL`, `MSFT`, `SPY`) with persistent local disk cache in `data/historical_bars/`. | M1 | Survey 2, ORIGINAL_REQUEST R1 |
| 2 | Lookahead-Free Bar-by-Bar Replay Engine | Chronological simulation stepping bar-by-bar, maintaining rolling technical indicators (SMA20, 5d momentum) computed strictly up to bar $t$. | M1 | Survey 2, ORIGINAL_REQUEST R1 |
| 3 | Multi-Bot Simulation & Position Execution | Replay 5 bot personas over historical bars, evaluating entry rules, position sizing, stop-loss triggers, take-profit exits, and dynamic exits. | M1 | Survey 1, Survey 2, ORIGINAL_REQUEST R1 |
| 4 | Quantitative Performance Metrics | Accurate calculation of Win Rate %, Profit Factor, Max Drawdown %, Net Return %, with safe zero-division handling. | M1 | Survey 2, ORIGINAL_REQUEST R1, R4 |
| 5 | Bot Archetype Parameter Customization | Parameterize all 5 bot archetypes (`ConservativeSwingBot`, `AggressiveBreakoutBot`, `ContrarianDipBuyerBot`, `FinBERTCatalystHunterBot`, `TrendFollowerBot`) to accept custom parameters (`stop_loss_pct`, `take_profit_pct`, `allocation_pct`, `max_positions`, `min_momentum_5d`, thresholds) with default fallback. | M2 | Survey 1, ORIGINAL_REQUEST R2 |
| 6 | Walk-Forward Parameter Calibration & Optimization | Grid search and walk-forward validation (in-sample calibration and out-of-sample validation windows) preventing curve-fitting. | M2 | Survey 2, ORIGINAL_REQUEST R2 |
| 7 | Supabase Schema & Persistence | PostgreSQL tables `bot_training_runs` and `bot_calibrated_parameters` in `schema.sql` and persistence functions in `src/db.py`. | M3 | Survey 3, ORIGINAL_REQUEST R3 |
| 8 | Resilient Offline Local Fallback | Local JSON fallback in `data/bot_calibrated_parameters.json` and `data/bot_training_runs.json` with 3-tier parameter resolution. | M3 | Survey 3, ORIGINAL_REQUEST R3 |
| 9 | Dynamic Parameter Sync in Tournament Engine | `MultiBotTournamentEngine` loads active calibrated parameters from Supabase (or local cache) on initialization. | M4 | Survey 1, Survey 3, ORIGINAL_REQUEST R3 |
| 10 | Pipeline CLI Commands (`--train`, `--backtest-report`) | Implement `--train --years <N>` and `--backtest-report` in `run_pipeline.py` with tabular reporting. | M4 | Survey 3, ORIGINAL_REQUEST R4 |
| 11 | Comprehensive Test Suite & Unittest Discovery | Unit & integration tests in `tests/test_backtest_calibration.py`. Ensure all tests pass via `python -m unittest discover tests`. | M5 | Survey 1-3, ORIGINAL_REQUEST R4 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | M1: 3-Year Historical Data Replay & Backtest Engine | `src/backtest_engine.py`: Data ingestion, persistent local cache, bar-by-bar lookahead-free simulation, trade execution, and performance metrics (Win Rate, Profit Factor, MDD, Net Return) | none | DONE |
| 2 | M2: Walk-Forward Parameter Calibration & Optimization | `src/calibration.py`, `src/paper_trader.py`: Bot parameterization, parameter search grids, walk-forward train/test validation, persona profile generation | M1 | DONE |
| 3 | M3: Supabase Knowledge Persistence & Offline Fallback | `src/db.py`, `schema.sql`, `data/`: `bot_training_runs` and `bot_calibrated_parameters` tables, persistence routines, local JSON fallbacks, and 3-tier parameter resolution | M2 | DONE |
| 4 | M4: Dynamic Tournament Parameter Sync & CLI Interface | `src/paper_trader.py`, `run_pipeline.py`: `MultiBotTournamentEngine` dynamic parameter sync, `--train --years 3` execution, `--backtest-report` tabular reporting | M2, M3 | DONE |
| 5 | M5: Comprehensive Verification & Forensic Integrity Audit | `tests/test_backtest_calibration.py`: Unit and integration test suite, full test discovery (`python -m unittest discover tests`), independent review and forensic integrity audit | M1, M2, M3, M4 | DONE |

## Interface Contracts
### `src/backtest_engine.py` (M1)
- `HistoricalDataFetcher.fetch_universe_bars(symbols: List[str], years: int = 3) -> Dict[str, pd.DataFrame]`:
  - Fetches daily OHLCV bars using `yfinance` with local caching in `data/historical_bars/`.
  - Returns sanitized DataFrames with columns `['open', 'high', 'low', 'close', 'volume']` indexed by chronological date.
- `HistoricalBacktestEngine.run_backtest(bots: List[BotArchetype], data: Dict[str, pd.DataFrame], start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]`:
  - Steps bar-by-bar across calendar dates.
  - Updates rolling indicators (`sma20`, `momentum_5d`) using only past and current bar data.
  - Evaluates exits then entries for each bot.
  - Returns dict with per-bot performance metrics, equity curves, and trade logs.
- `calculate_performance_metrics(trades: List[Dict[str, Any]], equity_curve: List[float], initial_balance: float = 5000.0) -> Dict[str, float]`:
  - Returns dict with `win_rate_pct`, `profit_factor`, `max_drawdown_pct`, `net_return_pct`, `total_trades`, `gross_profit`, `gross_loss`.

### `src/calibration.py` ↔ `src/paper_trader.py` (M2)
- `BotArchetype.__init__(..., params: Optional[Dict[str, Any]] = None)`:
  - Supports keyword/dict parameter overrides for all 5 bot subclasses (`stop_loss_pct`, `take_profit_pct`, `allocation_pct`, `max_positions`, `momentum_threshold`, etc.).
  - Retains default behavior if `params=None`.
- `WalkForwardCalibrator.calibrate(bot_class: Type[BotArchetype], data: Dict[str, pd.DataFrame], param_grid: Dict[str, List[Any]], in_sample_pct: float = 0.67) -> Dict[str, Any]`:
  - Evaluates parameter combinations on in-sample training split.
  - Evaluates top candidates on out-of-sample validation split to prevent curve-fitting.
  - Returns dict containing `best_params`, `in_sample_metrics`, `out_of_sample_metrics`, and `full_period_metrics`.
- `calibrate_all_personas(data: Dict[str, pd.DataFrame], years: int = 3) -> Dict[str, Dict[str, Any]]`:
  - Runs calibration across all 5 bot archetypes and returns calibrated parameter profiles.

### `src/db.py` ↔ `src/paper_trader.py` (M3 & M4)
- `save_training_run(client, run_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]`:
  - Persists training run record to Supabase `bot_training_runs` or falls back to `data/bot_training_runs.json`.
- `save_calibrated_parameters(client, bot_id: str, parameters: Dict[str, Any], metrics: Dict[str, float]) -> Tuple[bool, Optional[str]]`:
  - Persists optimal parameters to Supabase `bot_calibrated_parameters` or falls back to `data/bot_calibrated_parameters.json`.
- `load_calibrated_parameters(client, bot_id: str) -> Optional[Dict[str, Any]]`:
  - 3-tier resolution: Supabase Cloud -> Local JSON Cache -> `None` (triggers persona default).
- `MultiBotTournamentEngine.sync_calibrated_parameters() -> Dict[str, Dict[str, Any]]`:
  - Dynamically updates active bots with loaded calibrated parameters.

### `run_pipeline.py` (M4)
- CLI Flags:
  - `--train`: triggers walk-forward historical training.
  - `--years`: integer number of historical years (default: 3).
  - `--backtest-report`: displays comparative tabular performance summary for all 5 bots.

## Code Layout
```
D:\market-sentiment-researcher\
├── src\
│   ├── config.py                  # Configurations (SUPABASE_URL, SUPABASE_KEY, SERVICE_KEY)
│   ├── scraper.py                 # News ingestion
│   ├── embeddings.py              # FastEmbed vector embeddings
│   ├── db.py                      # Supabase client, pgvector, and calibration storage
│   ├── finbert_analyzer.py        # FinBERT sentiment
│   ├── relevance.py               # Targeted NIFTY relevance
│   ├── backtest_engine.py         # 3-year historical bar replay & metrics (M1)
│   ├── calibration.py             # Walk-forward parameter calibration & optimization (M2)
│   ├── paper_trader.py            # Bot archetypes with parameterization & tournament dynamic sync (M2, M4)
│   └── researcher.py              # Research query utility
├── tests\
│   ├── test_multi_bot.py          # Existing tournament & guardrail tests
│   ├── test_indian_market_relevance.py # Existing Indian market tests
│   └── test_backtest_calibration.py    # Unit & integration tests for backtesting & calibration (M5)
├── data\
│   ├── historical_bars\           # Cached OHLCV historical parquet/csv files
│   ├── bot_tournament_state.json  # Existing tournament state fallback
│   ├── bot_calibrated_parameters.json # Calibrated parameters offline fallback
│   └── bot_training_runs.json     # Training runs offline fallback
├── schema.sql                     # PostgreSQL table definitions
├── requirements.txt               # Dependencies
├── run_pipeline.py                # CLI entry point (--train, --backtest-report, etc.)
└── PROJECT.md                     # Project architecture, inventory, and milestones
```
