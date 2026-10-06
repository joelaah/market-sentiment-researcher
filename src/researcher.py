from typing import List, Dict, Any
from tabulate import tabulate
from src.embeddings import EmbeddingService
from src.db import get_supabase_client, search_similar_news

class MarketResearcher:
    def __init__(self):
        self.embedder = EmbeddingService()
        self.db_client = get_supabase_client()

    def research(self, topic_query: str, top_k: int = 5, threshold: float = 0.25) -> List[Dict[str, Any]]:
        """
        Takes a natural language market question or asset symbol,
        embeds it, and retrieves the most semantically relevant market news from Supabase pgvector.
        """
        print(f"\n[Researcher] Generating vector embedding for query: '{topic_query}'...")
        query_vec = self.embedder.embed_query(topic_query)

        print(f"[Researcher] Querying Supabase pgvector for top {top_k} matches...")
        matches = search_similar_news(
            client=self.db_client,
            query_embedding=query_vec,
            threshold=threshold,
            count=top_k
        )
        return matches

    def print_brief(self, topic_query: str, matches: List[Dict[str, Any]]):
        """Displays formatted research findings."""
        print("=" * 80)
        print(f"MARKET RESEARCH INTELLIGENCE BRIEF: '{topic_query}'")
        print("=" * 80)

        if not matches:
            print("No relevant news matches found above the similarity threshold.")
            print("Tip: Run the scraper pipeline first or lower the threshold.")
            return

        table_data = []
        for i, item in enumerate(matches, 1):
            title = (item['title'][:55] + '...') if len(item['title']) > 55 else item['title']
            sim = f"{item['similarity'] * 100:.1f}%"
            source = item['source']
            table_data.append([i, title, source, sim, item.get('published_at', '')[:10]])

        print(tabulate(table_data, headers=["#", "Title", "Source", "Similarity", "Date"], tablefmt="grid"))
        print("\nKey Article Snippets:")
        for i, item in enumerate(matches, 1):
            print(f"\n[{i}] {item['title']} ({item['source']})")
            print(f"    Similarity: {item['similarity'] * 100:.1f}% | URL: {item['link']}")
            if item.get('summary'):
                print(f"    Summary: {item['summary']}")
        print("=" * 80)
