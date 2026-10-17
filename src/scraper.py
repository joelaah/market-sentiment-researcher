from typing import List, Dict, Any, Optional
from datetime import datetime
import html
import re
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
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
        "source": "Google Finance Global Markets",
        "url": "https://news.google.com/rss/search?q=global+markets+stocks+europe+asia&hl=en-US&gl=US&ceid=US:en"
    },
    {
        "source": "CNBC International Markets",
        "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"
    },
    {
        "source": "Google Finance & Economy",
        "url": "https://news.google.com/rss/search?q=stock+market+economy&hl=en-US&gl=US&ceid=US:en"
    },
    {
        "source": "Google News India Business",
        "url": "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-IN&gl=IN&ceid=IN:en"
    },
    {
        "source": "Google News Nifty/Sensex",
        "url": "https://news.google.com/rss/search?q=nifty+sensex+indian+stock+market&hl=en-IN&gl=IN&ceid=IN:en"
    },
    {
        "source": "Economic Times Markets",
        "url": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"
    },
    {
        "source": "Economic Times Stocks",
        "url": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms"
    },
    {
        "source": "Livemint Markets",
        "url": "https://www.livemint.com/rss/markets"
    },
    {
        "source": "Moneycontrol Latest",
        "url": "https://www.moneycontrol.com/rss/latestnews.xml"
    }
]


def clean_html(html_text: str) -> str:
    """
    Removes HTML tags, decodes HTML entities, and normalizes whitespaces and anomalies.
    """
    if not html_text:
        return ""
    unescaped = html.unescape(str(html_text))
    soup = BeautifulSoup(unescaped, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    text = html.unescape(text)

    # Normalize publisher formatting anomalies and unicode replacements
    text = re.sub(r"(\w)\ufffd(\w)", r"\1'\2", text)
    text = text.replace("\xa0", " ").replace("\u200b", "").replace("\ufeff", "")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\ufffd", "")

    return " ".join(text.split())


def normalize_url(url: str) -> str:
    """
    Normalizes URL by stripping tracking query parameters, fragments, and trailing slashes.
    """
    if not url:
        return ""
    parsed = urlparse(url.strip())
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    tracking_params = {
        "utm_source", "utm_medium", "utm_campaign",
        "utm_term", "utm_content", "oc", "ved", "usqp"
    }
    filtered_query = [
        (k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=False)
        if k.lower() not in tracking_params
    ]
    filtered_query.sort()
    normalized = urlunparse((
        parsed.scheme.lower(),
        netloc,
        path,
        parsed.params,
        urlencode(filtered_query),
        ""
    ))
    return normalized or url.strip()


def sanitize_title(title: str) -> str:
    """
    Sanitizes title to alphanumeric lowercase for in-memory deduplication.
    """
    return re.sub(r"[^a-zA-Z0-9]", "", title).lower()


def fetch_financial_news(limit_per_feed: int = 10) -> List[Dict[str, Any]]:
    """
    Scrapes live financial news items across verified financial feeds.
    Deduplicates incoming articles in-memory by normalized URL and sanitized title.
    Returns structured list of articles with title, link, source, summary, published_at, and content.
    """
    articles = []
    seen_urls = set()
    seen_titles = set()

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

                raw_title = entry.get("title", "")
                title = clean_html(raw_title)

                raw_link = entry.get("link", "").strip()
                if not raw_link and entry.get("links"):
                    raw_link = entry["links"][0].get("href", "").strip()

                link = normalize_url(raw_link)
                if not title or not link:
                    continue

                # In-memory deduplication checks
                if link in seen_urls:
                    continue

                san_title = sanitize_title(title)
                if len(san_title) >= 12 and san_title in seen_titles:
                    continue

                seen_urls.add(link)
                if len(san_title) >= 12:
                    seen_titles.add(san_title)

                raw_summary = entry.get("summary") or entry.get("description", "")
                summary = clean_html(raw_summary)

                # Parse date if available
                published_parsed = entry.get("published_parsed")
                if published_parsed:
                    published_at = datetime(*published_parsed[:6]).isoformat() + "Z"
                else:
                    published_at = datetime.utcnow().isoformat() + "Z"

                content = f"{title}. {summary}".strip() if summary else title

                articles.append({
                    "title": title,
                    "link": link,
                    "source": source_name,
                    "summary": summary,
                    "published_at": published_at,
                    "content": content
                })
                count += 1

            print(f"[Scraper] Successfully collected {count} items from {source_name}")

        except Exception as e:
            print(f"[Scraper] Error scraping {source_name}: {e}")

    return articles
