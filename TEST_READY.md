# E2E Test Suite Ready

## Test Runner
- Command: `python -m unittest discover tests`
- Expected: all 54 tests pass with exit code 0

## Coverage Summary
| Tier | Count | Description |
|------|------:|-------------|
| 1. Feature Coverage | 22 | Indian instrument metrics, guardrail evaluation, feed parsing, deduplication, FinBERT polarity/confidence, entity matching |
| 2. Boundary & Corner | 13 | Exact boundary checks (₹20.00 / ₹19.99 INR, $5.00 / $4.99 USD), off-hours volume/market cap missing, zero vectors, empty text |
| 3. Cross-Feature | 11 | Semantic similarity separation, materiality assignment, fallback database searching, alias to ticker mapping |
| 4. Real-World Application | 8 | End-to-end CLI execution (`run_pipeline.py --check-news --symbol NIFTY`), offline exception fallbacks, multi-bot tournament regression |
| **Total** | **54** | **Comprehensive end-to-end, unit, and adversarial test coverage** |

## Feature Checklist
| Feature | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---------|:------:|:------:|:------:|:------:|
| Instrument & Watchlist (`^NSEI`, `NIFTYBEES.NS`) | ✓ | ✓ | ✓ | ✓ |
| Currency-Aware Guardrails (INR ₹20 vs USD $5) | ✓ | ✓ | ✓ | ✓ |
| Index Buy Order Routing to ETF (`NIFTYBEES.NS`) | ✓ | ✓ | ✓ | ✓ |
| Indian Financial RSS Feeds & Cleaning | ✓ | ✓ | ✓ | ✓ |
| In-Memory Deduplication | ✓ | ✓ | ✓ | ✓ |
| Supabase pgvector Fallback & Resilience | ✓ | ✓ | ✓ | ✓ |
| FinBERT Polarity & Confidence Output | ✓ | ✓ | ✓ | ✓ |
| Targeted News Relevance Evaluator | ✓ | ✓ | ✓ | ✓ |
| Deceptive Keyword & Homonym Immunity | ✓ | ✓ | ✓ | ✓ |
| Formatted Intelligence Brief | ✓ | ✓ | ✓ | ✓ |
| CLI `--check-news --symbol NIFTY` | ✓ | ✓ | ✓ | ✓ |
