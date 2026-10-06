from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from src.config import SUPABASE_URL, SUPABASE_KEY

def get_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY must be set in your .env file."
        )
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def upsert_market_news(client: Client, articles_with_embeddings: List[Dict[str, Any]]) -> int:
    """
    Upserts articles and their vectors into Supabase.
    Uses 'link' as the conflict resolution key to prevent duplicate articles.
    """
    if not articles_with_embeddings:
        return 0

    inserted_count = 0
    # Batch upsert or item-by-item with upsert
    for item in articles_with_embeddings:
        record = {
            "title": item["title"],
            "link": item["link"],
            "source": item["source"],
            "published_at": item["published_at"],
            "summary": item["summary"],
            "content": item.get("content", ""),
            "embedding": item["embedding"]
        }
        try:
            # ON CONFLICT (link) DO UPDATE SET embedding, summary, published_at
            client.table("market_news").upsert(record, on_conflict="link").execute()
            inserted_count += 1
        except Exception as e:
            print(f"[Database] Error upserting article '{item['title'][:30]}...': {e}")

    return inserted_count

def search_similar_news(
    client: Client,
    query_embedding: List[float],
    threshold: float = 0.25,
    count: int = 5
) -> List[Dict[str, Any]]:
    """
    Calls the match_market_news RPC function in Supabase to perform cosine similarity search.
    """
    response = client.rpc(
        "match_market_news",
        {
            "query_embedding": query_embedding,
            "match_threshold": threshold,
            "match_count": count
        }
    ).execute()
    return response.data or []
