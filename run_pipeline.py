import argparse
import sys
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

from src.paper_trader import PaperTradingEngine

def run_paper_trading():
    missing = check_config()
    if missing:
        print(f"[Error] Missing Supabase configuration in .env: {', '.join(missing)}")
        sys.exit(1)
    engine = PaperTradingEngine()
    engine.evaluate_news_and_trade()
    engine.print_dashboard()

def run_portfolio_view():
    missing = check_config()
    if missing:
        print(f"[Error] Missing Supabase configuration in .env: {', '.join(missing)}")
        sys.exit(1)
    engine = PaperTradingEngine()
    engine.print_dashboard()

def main():
    parser = argparse.ArgumentParser(description="Financial News Scraper & Supabase Vector Research Assistant")
    parser.add_argument("--scrape", action="store_true", help="Scrape latest financial news and index into Supabase")
    parser.add_argument("--query", type=str, help="Perform a semantic search query against indexed news")
    parser.add_argument("--trade", action="store_true", help="Evaluate latest news and execute paper trades on $5,000 portfolio")
    parser.add_argument("--portfolio", action="store_true", help="View current paper trading portfolio performance")
    
    args = parser.parse_args()

    if args.scrape:
        run_scraper_and_indexing()
    elif args.trade:
        run_paper_trading()
    elif args.portfolio:
        run_portfolio_view()
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
        print("   AI MARKET RESEARCH & $5,000 PAPER TRADING BOT     ")
        print("=====================================================")
        print("1. Scrape latest market feeds and index into Supabase")
        print("2. Ask research questions / Semantic vector search")
        print("3. Run AI Paper Trading strategy evaluation")
        print("4. View $5,000 Paper Trading portfolio performance")
        print("5. Exit")
        choice = input("Select an option (1-5): ").strip()

        if choice == "1":
            run_scraper_and_indexing()
        elif choice == "2":
            run_interactive_research()
        elif choice == "3":
            run_paper_trading()
        elif choice == "4":
            run_portfolio_view()
        else:
            print("Exiting.")

if __name__ == "__main__":
    main()
