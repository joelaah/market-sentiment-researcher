from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from src.config import SUPABASE_URL, SUPABASE_KEY

def get_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY must be set in your .env file."
        )
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def upsert_market_news(client: Optional[Client], articles_with_embeddings: List[Dict[str, Any]]) -> int:
    """
    Upserts articles and their vectors into Supabase.
    Uses 'link' as the conflict resolution key to prevent duplicate articles.
    Resilient to database disconnects and individual record errors.
    """
    if not articles_with_embeddings:
        return 0

    if not client:
        print("[Database] Warning: Cannot upsert articles because Supabase client is None.")
        return 0

    inserted_count = 0
    # Batch upsert or item-by-item with upsert
    for item in articles_with_embeddings:
        link = item.get("link")
        title = item.get("title")
        if not link or not title:
            continue

        record = {
            "title": title,
            "link": link,
            "source": item.get("source", "Unknown"),
            "published_at": item.get("published_at"),
            "summary": item.get("summary", ""),
            "content": item.get("content", ""),
            "embedding": item.get("embedding")
        }
        try:
            # ON CONFLICT (link) DO UPDATE SET embedding, summary, published_at
            client.table("market_news").upsert(record, on_conflict="link").execute()
            inserted_count += 1
        except Exception as e:
            print(f"[Database] Error upserting article '{title[:30]}...': {e}")

    return inserted_count


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Computes cosine similarity between two float vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return float(dot / ((norm_a ** 0.5) * (norm_b ** 0.5)))


def search_similar_news(
    client: Optional[Client],
    query_embedding: List[float],
    threshold: Optional[float] = None,
    count: Optional[int] = None,
    match_threshold: float = 0.3,
    match_count: int = 5,
    fallback_articles: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Calls the match_market_news RPC function in Supabase to perform cosine similarity search.
    If the Supabase client is None, network fails, or credentials are unavailable,
    logs a warning and provides an in-memory cosine similarity fallback against
    fallback_articles (if provided) or returns an empty list gracefully.
    """
    effective_threshold = threshold if threshold is not None else match_threshold
    effective_count = count if count is not None else match_count

    if client is not None:
        try:
            response = client.rpc(
                "match_market_news",
                {
                    "query_embedding": query_embedding,
                    "match_threshold": effective_threshold,
                    "match_count": effective_count
                }
            ).execute()
            if response and hasattr(response, "data") and response.data is not None:
                return response.data
        except Exception as e:
            print(f"[Database] Warning: Supabase RPC error during similarity search: {e}")
    else:
        print("[Database] Warning: Supabase client is None. Proceeding with fallback if available.")

    # In-memory cosine similarity fallback
    if fallback_articles:
        results = []
        for art in fallback_articles:
            emb = art.get("embedding")
            if emb:
                sim = _cosine_similarity(query_embedding, emb)
                if sim >= effective_threshold:
                    art_copy = dict(art)
                    art_copy["similarity"] = sim
                    results.append(art_copy)
        results.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
        return results[:effective_count]

    return []
