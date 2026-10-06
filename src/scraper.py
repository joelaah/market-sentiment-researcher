from typing import List, Dict, Any
from datetime import datetime
import feedparser
from bs4 import BeautifulSoup
import requests

FEEDS = [
    {
        "source": "MarketWatch Top Stories",
        "url": "https://feeds.content.dowjones.io/public/rss/mw_topstories"
    },
    {
        "source": "WSJ Markets",
        "url": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"
    },
    {
        "source": "Google Finance & Economy",
        "url": "https://news.google.com/rss/search?q=stock+market+economy&hl=en-US&gl=US&ceid=US:en"
    }
]


def clean_html(html_text: str) -> str:
    """Removes HTML tags and normalizes whitespace."""
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())

def fetch_financial_news(limit_per_feed: int = 10) -> List[Dict[str, Any]]:
    """
    Scrapes live financial news items across verified financial feeds.
    Returns structured list of articles with title, link, source, summary, and published_at.
    """
    articles = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for feed_info in FEEDS:
        source_name = feed_info["source"]
        feed_url = feed_info["url"]
        print(f"[Scraper] Fetching {source_name} ({feed_url})...")

        try:
            resp = requests.get(feed_url, headers=headers, timeout=10)
            if resp.status_code != 200:
                print(f"[Scraper] Warning: Failed to fetch {feed_url} (HTTP {resp.status_code})")
                continue

            parsed = feedparser.parse(resp.content)
            count = 0

            for entry in parsed.entries:
                if count >= limit_per_feed:
                    break

                title = entry.get("title", "").strip()
                link = entry.get("link", "").strip()
                if not title or not link:
                    continue

                raw_summary = entry.get("summary") or entry.get("description", "")
                summary = clean_html(raw_summary)

                # Parse date if available
                published_parsed = entry.get("published_parsed")
                if published_parsed:
                    published_at = datetime(*published_parsed[:6]).isoformat() + "Z"
                else:
                    published_at = datetime.utcnow().isoformat() + "Z"

                articles.append({
                    "title": title,
                    "link": link,
                    "source": source_name,
                    "summary": summary,
                    "published_at": published_at,
                    "content": f"{title}. {summary}".strip()
                })
                count += 1

            print(f"[Scraper] Successfully collected {count} items from {source_name}")

        except Exception as e:
            print(f"[Scraper] Error scraping {source_name}: {e}")

    return articles
