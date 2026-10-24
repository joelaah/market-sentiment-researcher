import argparse
import sys
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from tabulate import tabulate

# Ensure UTF-8 / safe character encoding for Windows consoles
if sys.platform.startswith("win"):
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from src.config import check_config
from src.scraper import fetch_financial_news
from src.embeddings import EmbeddingService
from src.db import get_supabase_client, upsert_market_news
from src.researcher import MarketResearcher

def run_scraper_and_indexing():
    missing = check_config()
    if missing:
        print(f"[Error] Missing Supabase configuration in .env: {', '.join(missing)}")
        print("Please copy .env.example to .env and provide your Supabase URL & Key.")
        sys.exit(1)

    print("\n--- Step 1: Scraping Live Financial Feeds ---")
    articles = fetch_financial_news(limit_per_feed=10)
    if not articles:
        print("[Pipeline] No articles collected.")
        return

    print(f"\n--- Step 2: Generating Vector Embeddings ({len(articles)} articles) ---")
    embedder = EmbeddingService()
    # Embed based on title and summary
    texts_to_embed = [art["content"] for art in articles]
    vectors = embedder.embed_texts(texts_to_embed)

    for art, vec in zip(articles, vectors):
        art["embedding"] = vec

    print(f"\n--- Step 3: Upserting into Supabase pgvector ---")
    client = get_supabase_client()
    inserted = upsert_market_news(client, articles)
    print(f"[Pipeline] Successfully processed and stored {inserted} articles in Supabase!")

def run_interactive_research():
    missing = check_config()
    if missing:
        print(f"[Error] Missing Supabase configuration in .env: {', '.join(missing)}")
        print("Please copy .env.example to .env and set your Supabase URL & Key.")
        sys.exit(1)

    researcher = MarketResearcher()
    print("\n--- AI Market Research Assistant Ready ---")
    print("Type a financial topic, ticker, or catalyst to search historical news (or 'exit' to quit).")
    
    while True:
        try:
            query = input("\nEnter research topic (e.g., 'Fed interest rate cuts', 'Nvidia AI revenue'): ").strip()
            if not query or query.lower() in ("exit", "quit"):
                break
            results = researcher.research(query)
            researcher.print_brief(query, results)
        except (KeyboardInterrupt, EOFError):
            break

def run_paper_trading():
    missing = check_config()
    if missing:
        print(f"[Error] Missing Supabase configuration in .env: {', '.join(missing)}")
        sys.exit(1)
    from src.paper_trader import MultiBotTournamentEngine
    engine = MultiBotTournamentEngine()
    engine.evaluate_tournament()
    engine.print_leaderboard()

def run_portfolio_view(bot_id: str = None):
    missing = check_config()
    if missing:
        print(f"[Error] Missing Supabase configuration in .env: {', '.join(missing)}")
        sys.exit(1)
    from src.paper_trader import MultiBotTournamentEngine
    engine = MultiBotTournamentEngine()
    engine.print_leaderboard()

def run_borderlist_view():
    missing = check_config()
    if missing:
        print(f"[Error] Missing Supabase configuration in .env: {', '.join(missing)}")
        sys.exit(1)
    from src.paper_trader import MultiBotTournamentEngine
    engine = MultiBotTournamentEngine()
    engine.print_borderlist()

def safe_print(*args, **kwargs):
    """Safely prints strings without failing on limited console encodings (e.g. cp1252)."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        try:
            encoding = getattr(sys.stdout, "encoding", None) or "ascii"
            clean_args = [
                str(arg).encode(encoding, errors="replace").decode(encoding)
                for arg in args
            ]
            print(*clean_args, **kwargs)
        except Exception:
            clean_args = [
                str(arg).encode("ascii", errors="replace").decode("ascii")
                for arg in args
            ]
            print(*clean_args, **kwargs)


def format_key_params(bot_id: str, params: Optional[Dict[str, Any]]) -> str:
    """Formats calibrated bot parameters into a concise, aligned string for tabular display."""
    if not isinstance(params, dict) or not params:
        return "Defaults"

    parts = []
    if "stop_loss_pct" in params:
        parts.append(f"SL: {params['stop_loss_pct']:.1f}%")
    if "take_profit_pct" in params:
        parts.append(f"TP: +{params['take_profit_pct']:.1f}%")

    if bot_id == "conservative_swing":
        if "min_momentum_5d" in params:
            parts.append(f"Mom: {params['min_momentum_5d']:.1f}%")
        if "max_positions" in params:
            parts.append(f"MaxPos: {params['max_positions']}")
    elif bot_id == "aggressive_breakout":
        if "min_momentum_5d" in params:
            parts.append(f"Mom: {params['min_momentum_5d']:.1f}%")
        if "allocation_pct" in params:
            parts.append(f"Alloc: {int(params['allocation_pct'] * 100)}%")
    elif bot_id == "contrarian_dip_buyer":
        if "dip_threshold_pct" in params:
            parts.append(f"Dip: {params['dip_threshold_pct']:.1f}%")
    elif bot_id == "finbert_catalyst_hunter":
        if "conviction_threshold" in params:
            parts.append(f"Conv: {params['conviction_threshold']:.2f}")
    elif bot_id == "trend_follower":
        if "min_momentum_5d" in params:
            parts.append(f"Mom: {params['min_momentum_5d']:.1f}%")
    else:
        for k, v in params.items():
            if k not in ("stop_loss_pct", "take_profit_pct"):
                if isinstance(v, float):
                    parts.append(f"{k}: {v:.2f}")
                else:
                    parts.append(f"{k}: {v}")

    return ", ".join(parts) if parts else "Defaults"


def load_all_calibrated_metrics(client: Optional[Any] = None) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """
    Retrieves the latest calibration profiles across all personas with 3-tier fallback.
    Returns (personas_dict, data_source_description).
    """
    # Tier 1: Supabase Cloud query
    if client is not None:
        try:
            res = (
                client.table("bot_calibrated_parameters")
                .select("*")
                .eq("active", True)
                .order("calibrated_at", desc=True)
                .execute()
            )
            if res and hasattr(res, "data") and res.data:
                profiles = {}
                for row in res.data:
                    b_id = row.get("bot_id")
                    if b_id and b_id not in profiles:
                        profiles[b_id] = row
                if profiles:
                    return profiles, "Supabase Cloud (bot_calibrated_parameters)"
        except Exception:
            pass

    # Tier 2: Local JSON cache (data/bot_calibrated_parameters.json)
    try:
        from src.db import _get_calibrated_params_file, _safe_read_json
        cache_file = _get_calibrated_params_file()
        cache = _safe_read_json(cache_file, default=None)
        if isinstance(cache, dict) and cache:
            if "personas" in cache and isinstance(cache["personas"], dict):
                return cache["personas"], f"Local Cache ({cache_file.name})"
            profiles = {k: v for k, v in cache.items() if isinstance(v, dict)}
            if profiles:
                return profiles, f"Local Cache ({cache_file.name})"
    except Exception:
        pass

    # Tier 3: Local JSON training runs fallback (data/bot_training_runs.json)
    try:
        from src.db import _get_training_runs_file, _safe_read_json
        runs_file = _get_training_runs_file()
        runs = _safe_read_json(runs_file, default=None)
        if isinstance(runs, list) and len(runs) > 0:
            latest_run = runs[0]
            if isinstance(latest_run, dict):
                personas = latest_run.get("personas") or latest_run.get("training_metrics")
                if isinstance(personas, dict) and personas:
                    return personas, f"Local Training Runs ({runs_file.name})"
    except Exception:
        pass

    return {}, "None (No calibration records found)"


_DEFAULT_CLIENT = object()


def run_backtest_report(client: Optional[Any] = _DEFAULT_CLIENT, cached_data: Optional[Dict[str, Any]] = None) -> bool:
    """
    Renders an aligned comparative tabular summary comparing all 5 bots across:
    Bot Persona | Win Rate % | Profit Factor | Max Drawdown % | Net Return % | Total Trades | Status | Key Calibrated Params
    """
    safe_print("\n" + "=" * 125)
    safe_print("                              📊 3-YEAR HISTORICAL BACKTEST & BOT CALIBRATION REPORT 📊                              ")
    safe_print("=" * 125)

    if cached_data is not None:
        profiles = cached_data
        source_desc = "Current Training Pipeline Execution"
    else:
        if client is _DEFAULT_CLIENT:
            try:
                from src.db import get_supabase_client
                client = get_supabase_client()
            except Exception:
                client = None
        profiles, source_desc = load_all_calibrated_metrics(client)

    if not profiles:
        safe_print("\n⚠️  No historical backtest calibration records found in Supabase Cloud or local cache.")
        safe_print("To run historical training and generate calibrated parameters, execute:")
        safe_print("    python run_pipeline.py --train --years 3\n")
        safe_print("=" * 125)
        return False

    standard_bots = [
        ("conservative_swing", "Conservative Swing"),
        ("aggressive_breakout", "Aggressive Breakout"),
        ("contrarian_dip_buyer", "Contrarian Dip Buyer"),
        ("finbert_catalyst_hunter", "FinBERT Catalyst Hunter"),
        ("trend_follower", "Trend Follower"),
    ]

    table_rows = []
    for bot_id, default_name in standard_bots:
        entry = profiles.get(bot_id, {})
        persona_name = entry.get("name") or entry.get("persona_name") or default_name

        params = entry.get("parameters") or entry.get("best_params") or {}
        metrics = entry.get("full_period_metrics") or entry.get("full_metrics") or entry.get("metrics") or {}
        status = entry.get("status")
        if not status or status == "active":
            oos = entry.get("out_of_sample_metrics") or {}
            oos_ret = float(oos.get("net_return_pct", 0.0))
            oos_pf = float(oos.get("profit_factor", 0.0))
            if oos and (oos_ret != 0.0 or oos_pf != 0.0):
                try:
                    from src.calibration import check_overfitting_gate
                    is_m = entry.get("in_sample_metrics") or {}
                    is_overfit, _ = check_overfitting_gate(is_m, oos)
                    status = "degraded" if is_overfit else "optimal"
                except Exception:
                    is_m = entry.get("in_sample_metrics") or {}
                    is_ret = float(is_m.get("net_return_pct", 0.0))
                    if oos_ret <= 0.0 or oos_pf < 1.0 or (is_ret > 5.0 and oos_ret < 0.25 * is_ret):
                        status = "degraded"
                    else:
                        status = "optimal"
            elif not status:
                status = "active" if entry else "not calibrated"
        status = str(status)

        win_rate = metrics.get("win_rate_pct")
        win_rate_str = f"{float(win_rate):.1f}%" if win_rate is not None else "N/A"

        pf = metrics.get("profit_factor")
        pf_str = f"{float(pf):.2f}" if pf is not None else "N/A"

        mdd = metrics.get("max_drawdown_pct")
        mdd_str = f"{float(mdd):.1f}%" if mdd is not None else "N/A"

        net_ret = metrics.get("net_return_pct")
        if net_ret is not None:
            sign = "+" if float(net_ret) >= 0 else ""
            net_ret_str = f"{sign}{float(net_ret):.1f}%"
        else:
            net_ret_str = "N/A"

        trades = metrics.get("total_trades")
        trades_str = str(int(trades)) if trades is not None else "0"

        params_str = format_key_params(bot_id, params)

        table_rows.append([
            persona_name,
            win_rate_str,
            pf_str,
            mdd_str,
            net_ret_str,
            trades_str,
            status,
            params_str
        ])

    headers = [
        "Bot Persona",
        "Win Rate %",
        "Profit Factor",
        "Max Drawdown %",
        "Net Return %",
        "Total Trades",
        "Status",
        "Key Calibrated Params"
    ]

    safe_print(tabulate(table_rows, headers=headers, tablefmt="grid"))
    safe_print("-" * 125)
    safe_print("🏛️  Dataset: Historical Daily OHLCV Bars (Lookahead-Free Bar Replay)")
    safe_print("🌐  Universe: ^NSEI, NIFTYBEES.NS, NVDA, AAPL, MSFT, SPY (US & Indian Equities/ETFs)")
    safe_print("⚙️  Validation: Walk-Forward Chronological Split (67% In-Sample / 33% Out-of-Sample)")
    safe_print(f"💾  Storage Origin: {source_desc}")
    safe_print("=" * 125 + "\n")
    return True


def run_calibration_training(
    years: int = 3,
    data: Optional[Dict[str, Any]] = None,
    client: Optional[Any] = _DEFAULT_CLIENT,
    offline: bool = False
) -> Dict[str, Any]:
    """
    Executes historical training & walk-forward parameter calibration across all 5 bot archetypes:
    1. Loads universe daily OHLCV bars (^NSEI, NIFTYBEES.NS, NVDA, AAPL, MSFT, SPY).
    2. Runs walk-forward optimization in src.calibration.
    3. Persists training run and optimal parameters to Supabase and local JSON storage.
    4. Displays formatted comparative report table.
    """
    safe_print("\n" + "=" * 90)
    safe_print(f"  🚀 HISTORICAL TRAINING & WALK-FORWARD PARAMETER CALIBRATION ({years} YEARS) 🚀")
    safe_print("=" * 90)

    if years <= 0:
        safe_print(f"[Error] Invalid years parameter: {years}. Must be >= 1.")
        sys.exit(1)

    # 1. Resolve database client with graceful fallback
    if client is _DEFAULT_CLIENT:
        if not offline:
            try:
                from src.db import get_supabase_client
                client = get_supabase_client()
                safe_print("🔗 Connected to Supabase Cloud for calibration persistence.")
            except Exception as exc:
                client = None
                safe_print(f"ℹ️  Supabase offline ({exc}). Operating with local JSON storage resilience.")
        else:
            client = None

    # 2. Ingest historical universe daily OHLCV bars
    if data is None:
        from src.backtest_engine import HistoricalDataFetcher
        safe_print(f"\n--- Step 1: Loading {years}-Year Daily OHLCV Bars for Universe ---")
        fetcher = HistoricalDataFetcher(offline=offline)
        universe_symbols = ["^NSEI", "NIFTYBEES.NS", "NVDA", "AAPL", "MSFT", "SPY"]
        safe_print(f"Universe: {', '.join(universe_symbols)}")
        try:
            data = fetcher.fetch_universe_bars(symbols=universe_symbols, years=years)
            for sym, df in data.items():
                start_dt = str(df.index[0])[:10] if len(df) > 0 else "N/A"
                end_dt = str(df.index[-1])[:10] if len(df) > 0 else "N/A"
                safe_print(f"  • {sym:14s}: {len(df):4d} daily bars ({start_dt} to {end_dt})")
        except Exception as exc:
            safe_print(f"[Error] Failed to fetch universe historical bars: {exc}")
            raise
    else:
        safe_print(f"\n--- Step 1: Using provided historical dataset ({len(data)} instruments) ---")

    # 3. Execute walk-forward calibration
    safe_print(f"\n--- Step 2: Executing Walk-Forward Calibration Across 5 Bot Archetypes ---")
    from src.calibration import calibrate_all_personas
    import time
    start_t = time.time()
    calibration_output = calibrate_all_personas(data=data, years=years)
    elapsed = round(time.time() - start_t, 2)
    safe_print(f"✅ Calibration completed in {elapsed}s.")

    # 4. Persist master training run & persona parameter profiles
    safe_print(f"\n--- Step 3: Persisting Calibration State & Learned Parameters ---")
    from src.db import save_training_run, save_calibrated_parameters

    run_ok, run_err = save_training_run(client, calibration_output)
    if run_ok:
        run_id = calibration_output.get("run_id", "N/A")
        safe_print(f"✅ Master training run persisted (ID: {run_id})")
    else:
        safe_print(f"⚠️ Master training run persistence warning: {run_err}")

    personas = calibration_output.get("personas", {})
    for bot_id, persona_data in personas.items():
        best_params = persona_data.get("best_params", {})
        full_metrics = persona_data.get("full_period_metrics", {})
        in_sample = persona_data.get("in_sample_metrics", {})
        out_of_sample = persona_data.get("out_of_sample_metrics", {})
        status = persona_data.get("status", "active")
        run_id = calibration_output.get("run_id")

        p_ok, p_err = save_calibrated_parameters(
            client=client,
            bot_id=bot_id,
            parameters=best_params,
            metrics=full_metrics,
            in_sample_metrics=in_sample,
            out_of_sample_metrics=out_of_sample,
            full_metrics=full_metrics,
            status=status,
            training_run_id=run_id
        )
        if p_ok:
            safe_print(f"  • Persisted profile for '{bot_id}' [Status: {status}]")
        else:
            safe_print(f"  ⚠️ Profile persistence failed for '{bot_id}': {p_err}")

    # 5. Output comparative tabular performance report
    safe_print(f"\n--- Step 4: Calibrated Parameter Profiles ---")
    run_backtest_report(client=client, cached_data=personas)
    return calibration_output


def run_rolling_retrain_cli(window_days: int = 90, offline: bool = True):
    """Executes the automated rolling retraining engine from CLI."""
    from src.rolling_retrainer import run_rolling_retraining
    safe_print("\n" + "=" * 90)
    safe_print(f"       🔄 AUTOMATED ROLLING RETRAINING & OVERFITTING GATE AUDIT ({window_days} BARS)       ")
    safe_print("=" * 90)
    res = run_rolling_retraining(window_days=window_days, offline=offline)
    safe_print(f"Status: {res.get('status')} | Elapsed: {res.get('elapsed_seconds')}s | Offline: {res.get('offline')}")
    safe_print(f"Evaluated Universe: {res.get('universe')}")
    safe_print(f"Accepted Candidates: {res.get('accepted_count')} | Rejected (Overfitting Gate): {res.get('rejected_count')}\n")

    table_rows = []
    for bot_id, b_res in res.get("bot_evaluations", {}).items():
        table_rows.append([
            bot_id,
            b_res.get("action", "unknown").upper(),
            b_res.get("status", "unknown"),
            b_res.get("reason", "")[:55]
        ])
    safe_print(tabulate(table_rows, headers=["Bot Persona", "Decision", "Status", "Gate Decision Rationale"], tablefmt="grid"))
    f_res = res.get("fusion_retraining", {})
    safe_print(f"\nMultiModalFusionEngine Retraining: Samples={f_res.get('samples_trained', 0)}, Monotonic w_sent={f_res.get('monotonic_weight_sent', 'N/A')}")
    safe_print("=" * 90 + "\n")
    return res


def main():
    parser = argparse.ArgumentParser(description="Financial News Scraper, Supabase pgvector & Multi-Bot Trading Tournament")
    parser.add_argument("--scrape", action="store_true", help="Scrape latest financial news and index into Supabase")
    parser.add_argument("--query", type=str, help="Perform a semantic search query against indexed news")
    parser.add_argument("--trade", action="store_true", help="Evaluate all 5 AI bots simultaneously in Multi-Bot Tournament")
    parser.add_argument("--portfolio", "--leaderboard", dest="portfolio", action="store_true", help="View Live Multi-Bot Tournament Leaderboard")
    parser.add_argument("--borderlist", action="store_true", help="View Safety Borderlist & Guardrails status (disqualified tickers)")
    parser.add_argument("--bot", type=str, default=None, help="Specific bot ID to inspect (optional)")
    parser.add_argument("--check-news", action="store_true", help="Automated news relevance checker for NIFTY / Indian markets")
    parser.add_argument("--symbol", type=str, default="NIFTY", help="Target market symbol for relevance check (default: NIFTY)")
    parser.add_argument("--test-telegram", action="store_true", help="Send a test trade & NIFTY alert to verify Telegram integration")
    parser.add_argument("--train", action="store_true", help="Execute 3-year historical backtesting, training calibration, and walk-forward parameter optimization")
    parser.add_argument("--years", type=int, default=3, help="Historical lookback period in years for backtesting & calibration (default: 3)")
    parser.add_argument("--backtest-report", action="store_true", help="Display formatted historical backtesting performance comparison table across all 5 bot archetypes")
    parser.add_argument("--rolling-retrain", action="store_true", help="Execute automated weekly rolling retraining & bot parameter recalibration")
    parser.add_argument("--rolling-window-days", type=int, default=90, help="Rolling lookback window in trading days (default: 90)")
    parser.add_argument("--offline", action="store_true", default=True, help="Operate strictly offline using cached bars (default: True)")

    args = parser.parse_args()

    if args.test_telegram:
        from src.telegram_notifier import is_telegram_configured, send_telegram_message, notify_trade, notify_nifty_catalyst
        print("\n--- Testing Telegram Integration ---")
        if not is_telegram_configured():
            print("❌ Telegram credentials missing or set to defaults in .env.")
            print("Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
            sys.exit(1)
        print("Sending test greeting, trade alert, and NIFTY catalyst alert...")
        s1 = send_telegram_message("🚀 <b>Market Sentiment Researcher: Telegram Alerts Online!</b>\nConnected to DeepMind Antigravity Multi-Bot Arena.")
        s2 = notify_trade(
            bot_name="FinBERT Catalyst Hunter",
            action="BUY",
            symbol="NIFTYBEES.NS",
            shares=3.7893,
            price=263.90,
            total_val=1000.00,
            rationale="Test alert: Nifty soars 2.8% on strong macroeconomic catalyst"
        )
        s3 = notify_nifty_catalyst(
            title="Nifty 50 approaches record highs as FII inflows surge",
            source="Economic Times Markets",
            polarity="positive",
            confidence=0.92,
            materiality="HIGH",
            relevance_score=1.45,
            link="https://economictimes.indiatimes.com"
        )
        if s1 and s2 and s3:
            print("✅ All test alerts successfully delivered to Telegram!")
        else:
            print("⚠️ Some alerts could not be delivered. Check bot permissions or chat ID.")
        sys.exit(0)
    elif args.check_news:
        from src.relevance import run_news_relevance_check
        run_news_relevance_check(symbol=args.symbol)
        sys.exit(0)
    elif args.train:
        run_calibration_training(years=args.years)
        sys.exit(0)
    elif args.rolling_retrain:
        run_rolling_retrain_cli(window_days=args.rolling_window_days, offline=args.offline)
        sys.exit(0)
    elif args.backtest_report:
        run_backtest_report()
        sys.exit(0)
    elif args.scrape:
        run_scraper_and_indexing()
    elif args.trade:
        run_paper_trading()
    elif args.portfolio:
        run_portfolio_view(args.bot)
    elif args.borderlist:
        run_borderlist_view()
    elif args.query:
        missing = check_config()
        if missing:
            print(f"[Error] Missing Supabase configuration in .env: {', '.join(missing)}")
            sys.exit(1)
        researcher = MarketResearcher()
        results = researcher.research(args.query)
        researcher.print_brief(args.query, results)
    else:
        # Default menu
        print("=====================================================")
        print("   AI MARKET RESEARCH & MULTI-BOT TOURNAMENT ARENA   ")
        print("=====================================================")
        print("1. Scrape latest market feeds and index into Supabase")
        print("2. Ask research questions / Semantic vector search")
        print("3. Run Multi-Bot Tournament evaluation & paper trading")
        print("4. View Live Multi-Bot Tournament Leaderboard")
        print("5. View Safety Borderlist & Guardrails status")
        print("6. Run Targeted News Relevance Checker (NIFTY)")
        print("7. Run Historical Training & Parameter Calibration")
        print("8. View Backtest Performance Report")
        print("9. Run Weekly Rolling Retraining & Overfitting Gate Audit")
        print("10. Exit")
        choice = input("Select an option (1-10): ").strip()

        if choice == "1":
            run_scraper_and_indexing()
        elif choice == "2":
            run_interactive_research()
        elif choice == "3":
            run_paper_trading()
        elif choice == "4":
            run_portfolio_view()
        elif choice == "5":
            run_borderlist_view()
        elif choice == "6":
            from src.relevance import run_news_relevance_check
            run_news_relevance_check(symbol="NIFTY")
        elif choice == "7":
            run_calibration_training(years=3)
        elif choice == "8":
            run_backtest_report()
        elif choice == "9":
            run_rolling_retrain_cli(window_days=90, offline=True)
        else:
            print("Exiting.")

if __name__ == "__main__":
    main()
