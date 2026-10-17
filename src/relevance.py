from typing import List, Dict, Any, Optional, Tuple
import re
from tabulate import tabulate

from src.embeddings import EmbeddingService
from src.finbert_analyzer import FinBERTAnalyzer


# Anchor representations for NIFTY 50 and Indian macroeconomic catalysts
NIFTY_MACRO_ANCHORS = [
    "NIFTY 50 Indian benchmark equity index, Dalal Street stock market rally crash, Sensex movements, NSE trading",
    "Reserve Bank of India RBI Monetary Policy Committee MPC repo interest rates, inflation CPI, Indian economic GDP growth",
    "Union Budget, Finance Ministry fiscal reforms, Indian economy, foreign institutional investors FII flows, Rupee",
    "Corporate earnings quarterly results, profits and revenues of NIFTY 50 heavyweight Indian companies, Reliance, HDFC Bank, TCS, Infosys"
]

# Comprehensive entity recognition patterns for Indian financial markets
ENTITY_PATTERNS: List[Tuple[str, str, float]] = [
    # Benchmark indices and Indian equity aliases
    (r"\bNIFTY\s*50\b|\bNifty\s*50\b|\bNSE\s+NIFTY\b|\bNifty\s+index\b|\bNifty\s+futures\b", "NIFTY 50", 1.0),
    (r"\bNIFTYBEES(?:\.NS)?\b", "NIFTYBEES", 1.0),
    (r"\b\^NSEI\b", "^NSEI", 1.0),
    (r"\bNIFTY\b", "NIFTY", 0.95),  # Bare NIFTY requires case-sensitive uppercase match & financial context
    (r"\bSensex\b", "Sensex", 0.85),
    (r"\bBSE\b|\bBombay\s+Stock\s+Exchange\b", "BSE", 0.70),
    (r"\bNSE\b|\bNational\s+Stock\s+Exchange\b", "NSE", 0.70),
    (r"\bDalal\s+Street\b", "Dalal Street", 0.80),
    
    # Macro institutions, regulatory bodies, and officials
    (r"\bReserve\s+Bank\s+of\s+India\b", "Reserve Bank of India", 0.85),
    (r"\bRBI\b", "RBI", 0.80),
    (r"\bMonetary\s+Policy\s+Committee\b", "Monetary Policy Committee", 0.80),
    (r"\bMPC\b", "MPC", 0.75),
    (r"\brepo\s+rate\b", "repo rate", 0.75),
    (r"\breverse\s+repo\b", "reverse repo", 0.70),
    (r"\bUnion\s+Budget\b", "Union Budget", 0.85),
    (r"\bFinance\s+Ministry\b|\bMinistry\s+of\s+Finance\b", "Finance Ministry", 0.80),
    (r"\bNirmala\s+Sitharaman\b", "Nirmala Sitharaman", 0.80),
    (r"\bSEBI\b|\bSecurities\s+and\s+Exchange\s+Board\s+of\s+India\b", "SEBI", 0.80),
    
    # Indian market flows and economic metrics
    (r"\bFII(?:s)?\b|\bFPI(?:s)?\b", "FII / FPI", 0.65),
    (r"\bDII(?:s)?\b", "DII", 0.65),
    (r"\bIndian\s+economy\b|\bIndia(?:'s)?\s+GDP\b", "Indian Economy / GDP", 0.75),
    (r"\bRupee\b|\bINR\b", "Indian Rupee", 0.60),

    # Top NIFTY 50 heavyweights & key constituents
    (r"\bReliance\s+Industries\b|\bReliance\b", "Reliance", 0.60),
    (r"\bHDFC\s+Bank\b|\bHDFC\b", "HDFC Bank", 0.60),
    (r"\bICICI\s+Bank\b|\bICICI\b", "ICICI Bank", 0.60),
    (r"\bInfosys\b|\bINFY\b", "Infosys", 0.60),
    (r"\bTata\s+Consultancy\s+Services\b|\bTCS\b", "TCS", 0.60),
    (r"\bITC(?:\s+Limited)?\b", "ITC", 0.55),
    (r"\bLarsen\s+&\s+Toubro\b|\bL&T\b", "Larsen & Toubro", 0.55),
    (r"\bState\s+Bank\s+of\s+India\b|\bSBI\b|\bSBIN\b", "SBI", 0.60),
    (r"\bBharti\s+Airtel\b|\bAirtel\b", "Bharti Airtel", 0.55),
    (r"\bKotak\s+Mahindra\s+Bank\b|\bKotak(?:\s+Bank)?\b", "Kotak Mahindra Bank", 0.55),
    (r"\bAxis\s+Bank\b", "Axis Bank", 0.55),
    (r"\bHindustan\s+Unilever\b|\bHUL\b", "Hindustan Unilever", 0.55),
    (r"\bTata\s+Motors\b", "Tata Motors", 0.55),
    (r"\bSun\s+Pharma\b", "Sun Pharma", 0.50),
    (r"\bMaruti\s+Suzuki\b|\bMaruti\b", "Maruti Suzuki", 0.50),
    (r"\bMahindra\s+&\s+Mahindra\b|\bM&M\b", "M&M", 0.50),
    (r"\bBajaj\s+Finance\b", "Bajaj Finance", 0.50),
    (r"\bAdani\s+Enterprises\b|\bAdani\s+Ports\b|\bAdani\b", "Adani", 0.50),
    (r"\bWipro\b", "Wipro", 0.50),
    (r"\bHCL\s+Tech\b|\bHCLTech\b", "HCL Tech", 0.50),
    (r"\bNTPC\b", "NTPC", 0.50),
    (r"\bPower\s+Grid\b|\bPower\s+Grid\s+Corporation\b", "Power Grid", 0.50),
    (r"\bTitan\s+Company\b|\bTitan\b", "Titan", 0.50),
    (r"\bUltraTech\s+Cement\b|\bUltraTech\b", "UltraTech", 0.50),
    (r"\bTata\s+Steel\b", "Tata Steel", 0.50)
]

# Disambiguation keywords for common English and international homonyms
INDIAN_CONTEXT_TERMS = {
    "india", "indian", "rupee", "rupees", "inr", "dalal street", "mumbai",
    "sensex", "sebi", "rbi", "delhi", "bengaluru", "new delhi", "bombay",
    "reserve bank"
}

FINANCIAL_CONTEXT_TERMS = {
    "shares", "stock", "stocks", "market", "markets", "equity", "equities",
    "points", "rally", "rallied", "fall", "fell", "trading", "investor", "investors",
    "percent", "%", "quarterly", "earnings", "revenue", "profit", "dividend",
    "yield", "index", "indices", "etf", "funds", "securities", "broker", "brokerage"
}

AMBIGUOUS_ENTITIES = {"BSE", "NSE", "SBI", "ITC", "Titan", "Power Grid"}


def is_homonym_false_positive(label: str, text_lower: str) -> bool:
    """Disambiguates Indian financial entities from international homonyms and common words."""
    if label == "BSE":
        if any(w in text_lower for w in ["bovine", "encephalopathy", "mad cow", "cattle", "quarantine"]):
            return True
    elif label == "NSE":
        if any(w in text_lower for w in ["nigeria", "nigerian", "lagos", "kenya", "nairobi"]):
            return True
    elif label == "SBI":
        if any(w in text_lower for w in ["tokyo", "japanese", "japan", "sbi holdings"]):
            return True
    elif label == "ITC":
        if any(w in text_lower for w in ["trade commission", "tariff", "solar panels", "us international"]):
            return True
    elif label == "Titan":
        if any(w in text_lower for w in ["saturn", "moon", "nasa", "cassini", "methane", "atmosphere", "space"]):
            return True
    elif label == "Power Grid":
        if any(w in text_lower for w in ["texas", "ercot", "winter storm", "freezing", "blackout"]):
            return True
    return False


CATALYST_HIGH = [
    "rate cut", "rate hike", "repo rate", "reverse repo",
    "gdp beat", "gdp miss", "record high", "all-time high",
    "crash", "plunge", "surge", "rally", "union budget",
    "tax reform", "stimulus", "fii massive", "fii inflow",
    "fii outflow", "bear market", "bull run", "sanctions",
    "monetary policy"
]

CATALYST_MEDIUM = [
    "earnings", "profit jump", "profit beat", "revenue beat",
    "quarterly results", "q1", "q2", "q3", "q4", "inflation",
    "cpi", "target hike", "target cut", "dividend", "acquisition",
    "outperform", "downgrade", "upgrade", "guidance", "order win"
]


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Computes cosine similarity between two float vectors bounded between 0.0 and 1.0."""
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
    val = dot / ((norm_a ** 0.5) * (norm_b ** 0.5))
    return float(max(0.0, min(1.0, val)))


def get_fallback_news_fixtures() -> List[Dict[str, Any]]:
    """Curated realistic news fixtures used for testing or when external RSS feeds are unreachable."""
    return [
        {
            "title": "NIFTY 50 hits record high above 24,000 as RBI holds repo rate steady and projects 7.2% GDP growth",
            "link": "https://example.com/nifty-record-high-rbi",
            "source": "Google News Nifty/Sensex",
            "summary": "The NIFTY 50 index surged 250 points to touch an all-time peak following the Reserve Bank of India's monetary policy announcement keeping benchmark interest rates unchanged.",
            "published_at": "2026-10-17T06:00:00Z",
            "content": "NIFTY 50 hits record high above 24,000 as RBI holds repo rate steady and projects 7.2% GDP growth. The NIFTY 50 index surged 250 points to touch an all-time peak following the Reserve Bank of India's monetary policy announcement keeping benchmark interest rates unchanged."
        },
        {
            "title": "Reliance Industries and HDFC Bank rally on Dalal Street after stellar quarterly earnings results",
            "link": "https://example.com/reliance-hdfc-earnings",
            "source": "Economic Times Markets",
            "summary": "Heavyweights Reliance Industries and HDFC Bank lifted benchmark indices Sensex and NIFTY as institutional investors poured funds into blue-chip Indian equities.",
            "published_at": "2026-10-17T06:15:00Z",
            "content": "Reliance Industries and HDFC Bank rally on Dalal Street after stellar quarterly earnings results. Heavyweights Reliance Industries and HDFC Bank lifted benchmark indices Sensex and NIFTY as institutional investors poured funds into blue-chip Indian equities."
        },
        {
            "title": "Union Budget 2026: Finance Ministry announces manufacturing tax incentives, FII inflows accelerate",
            "link": "https://example.com/union-budget-fii-inflows",
            "source": "Livemint Markets",
            "summary": "Finance Minister Nirmala Sitharaman unveiled budgetary incentives for domestic capital expenditure, sparking positive foreign institutional investor flows into Indian markets.",
            "published_at": "2026-10-17T06:30:00Z",
            "content": "Union Budget 2026: Finance Ministry announces manufacturing tax incentives, FII inflows accelerate. Finance Minister Nirmala Sitharaman unveiled budgetary incentives for domestic capital expenditure, sparking positive foreign institutional investor flows into Indian markets."
        },
        {
            "title": "TCS reports double-digit cloud revenue growth; State Bank of India expands corporate loan book",
            "link": "https://example.com/tcs-sbi-corporate",
            "source": "Google News India Business",
            "summary": "Tata Consultancy Services (TCS) posted operating margins of 26% while State Bank of India (SBI) reported lower net NPAs, reinforcing banking sector health.",
            "published_at": "2026-10-17T06:45:00Z",
            "content": "TCS reports double-digit cloud revenue growth; State Bank of India expands corporate loan book. Tata Consultancy Services (TCS) posted operating margins of 26% while State Bank of India (SBI) reported lower net NPAs, reinforcing banking sector health."
        },
        {
            "title": "Seattle residential home prices rise 1.4% in local county regional market assessment",
            "link": "https://example.com/seattle-housing-report",
            "source": "MarketWatch Top Stories",
            "summary": "Single family home inventory across Seattle and suburban King County registered slight gains as mortgage rates remained elevated.",
            "published_at": "2026-10-17T05:30:00Z",
            "content": "Seattle residential home prices rise 1.4% in local county regional market assessment. Single family home inventory across Seattle and suburban King County registered slight gains as mortgage rates remained elevated."
        },
        {
            "title": "European Central Bank policymakers debate inflation trajectory in Frankfurt symposium",
            "link": "https://example.com/ecb-frankfurt-symposium",
            "source": "WSJ Markets",
            "summary": "Eurozone monetary authorities discussed wage trends and potential interest rate adjustments during meetings in Frankfurt.",
            "published_at": "2026-10-17T05:45:00Z",
            "content": "European Central Bank policymakers debate inflation trajectory in Frankfurt symposium. Eurozone monetary authorities discussed wage trends and potential interest rate adjustments during meetings in Frankfurt."
        }
    ]


class NiftyRelevanceChecker:
    """
    Automated targeted news relevance checker for NIFTY 50 (^NSEI / NIFTYBEES.NS)
    and Indian financial market assets.

    Evaluates articles across three core dimensions:
    1. Entity / Ticker Relevance: Regex token detection of NIFTY, Sensex, RBI, Budget, and constituent heavyweights.
    2. Semantic Similarity: Dense vector cosine similarity against NIFTY macro/catalyst anchor references.
    3. Sentiment & Materiality: FinBERT sentiment classification (polarity, confidence, net score) and catalyst intensity.
    """

    def __init__(
        self,
        embedder: Optional[EmbeddingService] = None,
        finbert: Optional[FinBERTAnalyzer] = None
    ):
        self._embedder = embedder
        self._finbert = finbert
        self._anchor_embeddings: Optional[List[List[float]]] = None

    def _get_embedder(self) -> EmbeddingService:
        if self._embedder is None:
            self._embedder = EmbeddingService()
        return self._embedder

    def _get_finbert(self) -> FinBERTAnalyzer:
        if self._finbert is None:
            self._finbert = FinBERTAnalyzer()
        return self._finbert

    def _get_anchor_embeddings(self) -> List[List[float]]:
        if self._anchor_embeddings is None:
            embedder = self._get_embedder()
            self._anchor_embeddings = embedder.embed_texts(NIFTY_MACRO_ANCHORS)
        return self._anchor_embeddings

    def match_entities(self, text: str, title: str = "") -> Tuple[List[str], float, bool]:
        """
        Scans text for Indian market entities and returns:
        - matched_entities: sorted unique list of entity labels
        - entity_score: float score from 0.0 to 1.0
        - has_direct_index_match: bool indicating explicit mention of NIFTY/Sensex/^NSEI
        """
        combined = f"{title or ''} {text or ''}".strip()
        if not combined:
            return [], 0.0, False

        combined_lower = combined.lower()
        has_indian_context = any(term in combined_lower for term in INDIAN_CONTEXT_TERMS)
        has_financial_context = any(term in combined_lower for term in FINANCIAL_CONTEXT_TERMS)

        matched = []
        max_weight = 0.0
        has_direct_index = False

        for pattern, label, weight in ENTITY_PATTERNS:
            # Standalone NIFTY requires case-sensitive uppercase match to avoid English adjective "nifty"
            flags = 0 if label == "NIFTY" else re.IGNORECASE
            if re.search(pattern, combined, flags):
                if is_homonym_false_positive(label, combined_lower):
                    continue

                if label == "NIFTY":
                    # Bare uppercase NIFTY must be accompanied by financial or Indian context
                    if not has_financial_context and not has_indian_context:
                        continue
                    has_direct_index = True
                elif label in {"NIFTY 50", "NIFTYBEES", "^NSEI", "Sensex", "Dalal Street"}:
                    has_direct_index = True
                elif label in {"BSE", "NSE"}:
                    # Bare BSE/NSE must have Indian context or full exchange name
                    is_full_name = ("bombay stock" in combined_lower) or ("national stock" in combined_lower)
                    if not has_indian_context and not is_full_name:
                        continue
                    has_direct_index = True
                elif label in AMBIGUOUS_ENTITIES:
                    # Ambiguous company acronyms without full company name require Indian context
                    full_names = {
                        "SBI": "state bank of india",
                        "ITC": "itc limited",
                        "Titan": "titan company",
                        "Power Grid": "power grid corporation"
                    }
                    if full_names.get(label, "") not in combined_lower and not has_indian_context:
                        continue

                matched.append(label)
                if weight > max_weight:
                    max_weight = weight

        # Deduplicate while preserving order
        unique_matches = []
        for m in matched:
            if m not in unique_matches:
                unique_matches.append(m)

        if not unique_matches:
            return [], 0.0, False

        # Compute composite entity score with multi-match & title boosts
        bonus = min(0.20, (len(unique_matches) - 1) * 0.05)
        entity_score = min(1.0, max_weight + bonus)

        # Title prominence boost
        if title:
            for pattern, label, _ in ENTITY_PATTERNS:
                flags = 0 if label == "NIFTY" else re.IGNORECASE
                if re.search(pattern, title, flags):
                    if not is_homonym_false_positive(label, title.lower()):
                        if label == "NIFTY" and not (has_financial_context or has_indian_context):
                            continue
                        entity_score = min(1.0, entity_score + 0.10)
                        break

        return unique_matches, round(entity_score, 4), has_direct_index

    def compute_semantic_similarity(self, article: Dict[str, Any]) -> float:
        """
        Computes the maximum cosine similarity of the article text
        against NIFTY macro/catalyst anchor references.
        """
        title = str(article.get("title") or "")
        summary = str(article.get("summary") or "")
        content = str(article.get("content") or "")
        text_to_embed = f"{title}. {summary or content}".strip()
        if not text_to_embed:
            return 0.0

        embedder = self._get_embedder()
        art_vector = embedder.embed_query(text_to_embed)
        anchor_vectors = self._get_anchor_embeddings()

        similarities = [_cosine_similarity(art_vector, anchor_vec) for anchor_vec in anchor_vectors]
        max_sim = max(similarities) if similarities else 0.0
        return round(float(max_sim), 4)

    def determine_materiality(
        self,
        text: str,
        entity_score: float,
        sentiment: Dict[str, Any]
    ) -> str:
        """
        Assigns materiality rating: 'HIGH', 'MEDIUM', 'LOW' based on:
        - Catalyst presence (repo rate, GDP, record high/crash, Union Budget, FII flows)
        - Sentiment intensity (abs(net_score))
        - Entity prominence
        """
        text_lower = text.lower()
        has_high_cat = any(cat in text_lower for cat in CATALYST_HIGH)
        has_med_cat = any(cat in text_lower for cat in CATALYST_MEDIUM)
        abs_net = abs(sentiment.get("net_score", 0.0))

        if (has_high_cat and (abs_net >= 0.40 or entity_score >= 0.50)) or (abs_net >= 0.80 and entity_score >= 0.50):
            return "HIGH"
        elif has_high_cat or (has_med_cat and entity_score >= 0.35) or abs_net >= 0.50 or entity_score >= 0.70:
            return "MEDIUM"
        else:
            return "LOW"

    def evaluate_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates an individual article and produces a structured relevance assessment.
        Returns:
            is_relevant: bool
            relevance_score: float (0.0 to 1.0)
            entity_matches: List[str]
            semantic_similarity: float (0.0 to 1.0)
            sentiment: Dict with polarity, confidence, net_score, positive, negative, neutral
            materiality: str ("HIGH" | "MEDIUM" | "LOW")
            reason: str
            article: Dict
        """
        title = str(article.get("title") or "")
        summary = str(article.get("summary") or "")
        content = str(article.get("content") or "")
        full_text = f"{title}. {summary or content}".strip()
        full_text_lower = full_text.lower()

        # 1. Entity Matching
        matched_entities, entity_score, has_direct_index = self.match_entities(text=full_text, title=title)

        # 2. Semantic Similarity
        semantic_sim = self.compute_semantic_similarity(article)

        # Context checks for financial and Indian market content
        has_financial_ctx = any(term in full_text_lower for term in FINANCIAL_CONTEXT_TERMS)
        has_indian_ctx = any(term in full_text_lower for term in INDIAN_CONTEXT_TERMS)

        # 3. Fast Candidate Pre-Filtering
        # If an article has 0 matched entities AND semantic similarity < 0.52 (or fails candidate viability),
        # immediately short-circuit without invoking expensive FinBERT transformer inference on CPU.
        is_candidate = False
        if has_direct_index and semantic_sim >= 0.50 and (has_financial_ctx or has_indian_ctx or semantic_sim >= 0.65):
            is_candidate = True
        elif matched_entities and semantic_sim >= 0.52:
            is_candidate = True
        elif semantic_sim >= 0.70:
            is_candidate = True

        if not is_candidate:
            # Irrelevant article: bypass expensive FinBERT transformer inference
            relevance_score = min(0.42, round(0.30 * entity_score + 0.70 * (semantic_sim * 0.50), 4))
            reasons = []
            if matched_entities:
                reasons.append(f"Entity match: {', '.join(matched_entities[:3])}")
            else:
                reasons.append("No Indian market entity matches")
            reasons.append(f"SemSim: {semantic_sim * 100:.1f}%")
            reasons.append("neutral (50.0%)")

            return {
                "is_relevant": False,
                "relevance_score": relevance_score,
                "entity_matches": matched_entities,
                "semantic_similarity": semantic_sim,
                "sentiment": {
                    "polarity": "neutral",
                    "confidence": 0.50,
                    "net_score": 0.0,
                    "positive": 0.25,
                    "negative": 0.25,
                    "neutral": 0.50
                },
                "materiality": "LOW",
                "reason": "; ".join(reasons),
                "article": article
            }

        # 4. FinBERT Sentiment Analysis (only for candidate articles)
        finbert = self._get_finbert()
        sentiment = finbert.analyze(full_text)

        # 5. Final Relevance Determination
        is_relevant = False
        if has_direct_index and semantic_sim >= 0.50 and (has_financial_ctx or has_indian_ctx or semantic_sim >= 0.65):
            is_relevant = True
        elif matched_entities and semantic_sim >= 0.55:
            if all(e in AMBIGUOUS_ENTITIES for e in matched_entities):
                if has_indian_ctx or semantic_sim >= 0.65:
                    is_relevant = True
            else:
                is_relevant = True
        elif semantic_sim >= 0.72:
            is_relevant = True

        # 6. Materiality Assessment
        if not is_relevant:
            materiality = "LOW"
        else:
            materiality = self.determine_materiality(full_text, entity_score, sentiment)

        # Compute composite relevance score
        if is_relevant:
            composite = 0.45 * entity_score + 0.40 * semantic_sim + 0.15 * min(1.0, abs(sentiment.get("net_score", 0.0)) + 0.20)
            relevance_score = max(0.50, min(1.0, round(composite, 4)))
        else:
            # Scaled down for irrelevant items
            composite = 0.30 * entity_score + 0.70 * (semantic_sim * 0.50)
            relevance_score = min(0.42, round(composite, 4))

        # Build clear explanation reason
        reasons = []
        if matched_entities:
            reasons.append(f"Entity match: {', '.join(matched_entities[:3])}")
        else:
            reasons.append("No Indian market entity matches")

        reasons.append(f"SemSim: {semantic_sim * 100:.1f}%")

        pol = sentiment.get("polarity", "neutral")
        conf = sentiment.get("confidence", 0.0)
        reasons.append(f"{pol} ({conf * 100:.1f}%)")

        if materiality == "HIGH":
            reasons.append("high-impact catalyst")

        reason = "; ".join(reasons)

        return {
            "is_relevant": is_relevant,
            "relevance_score": relevance_score,
            "entity_matches": matched_entities,
            "semantic_similarity": semantic_sim,
            "sentiment": sentiment,
            "materiality": materiality,
            "reason": reason,
            "article": article
        }

    def filter_relevant_news(
        self,
        articles: List[Dict[str, Any]],
        symbol: str = "NIFTY",
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Evaluates articles, filters out irrelevant news, and sorts results by
        materiality, relevance score, and confidence.
        """
        evaluated = [self.evaluate_article(art) for art in articles]
        relevant = [res for res in evaluated if res["is_relevant"]]

        materiality_ranks = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        relevant.sort(
            key=lambda x: (
                materiality_ranks.get(x["materiality"], 0),
                x["relevance_score"],
                x["sentiment"].get("confidence", 0.0)
            ),
            reverse=True
        )

        return relevant[:top_n]

    def check_news(
        self,
        articles: Optional[List[Dict[str, Any]]] = None,
        symbol: str = "NIFTY",
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Higher-level method: ingests articles from feeds if not provided,
        filters for relevant news impacting symbol, and returns top results.
        """
        if articles is None:
            try:
                from src.scraper import fetch_financial_news
                articles = fetch_financial_news(limit_per_feed=10)
            except Exception as e:
                print(f"[NiftyRelevanceChecker] Warning: Failed to fetch live news: {e}")
                articles = []

        if not articles:
            articles = get_fallback_news_fixtures()

        return self.filter_relevant_news(articles, symbol=symbol, top_n=top_n)

    def format_intelligence_brief(
        self,
        relevant_results: List[Dict[str, Any]],
        symbol: str = "NIFTY",
        total_analyzed: int = 0
    ) -> str:
        """Instance method wrapper for format_intelligence_brief."""
        return format_intelligence_brief(
            relevant_results=relevant_results,
            symbol=symbol,
            total_analyzed=total_analyzed
        )


def format_intelligence_brief(
    relevant_results: List[Dict[str, Any]],
    symbol: str = "NIFTY",
    total_analyzed: int = 0
) -> str:
    """
    Formats news relevance evaluation results into a clean, executive intelligence brief table using tabulate.
    Includes executive summary metrics (total articles scanned, relevant articles found, bullish/bearish balance).
    """
    lines = []
    lines.append("=" * 115)
    lines.append(f"              TARGETED MARKET INTELLIGENCE BRIEF: {symbol.upper()} (^NSEI / NIFTYBEES.NS)             ")
    lines.append("=" * 115)

    num_relevant = len(relevant_results)
    effective_total = max(total_analyzed, num_relevant)

    # Calculate sentiment distribution
    bullish_count = sum(1 for r in relevant_results if r["sentiment"]["polarity"] == "positive")
    bearish_count = sum(1 for r in relevant_results if r["sentiment"]["polarity"] == "negative")
    neutral_count = sum(1 for r in relevant_results if r["sentiment"]["polarity"] == "neutral")

    net_scores = [r["sentiment"]["net_score"] for r in relevant_results]
    avg_net_score = (sum(net_scores) / len(net_scores)) if net_scores else 0.0

    if avg_net_score >= 0.15:
        overall_stance = "[BULLISH]"
    elif avg_net_score <= -0.15:
        overall_stance = "[BEARISH]"
    else:
        overall_stance = "[NEUTRAL]"

    high_mat_count = sum(1 for r in relevant_results if r["materiality"] == "HIGH")

    lines.append("[EXECUTIVE SCAN SUMMARY]")
    lines.append(f"  * Total Articles Scanned: {effective_total}")
    lines.append(f"  * Relevant Articles Identified: {num_relevant}")
    lines.append(f"  * Sentiment Breakdown: Bullish: {bullish_count} | Bearish: {bearish_count} | Neutral: {neutral_count}")
    lines.append(f"  * Aggregate Net Sentiment: {overall_stance} ({avg_net_score:+.4f})")
    lines.append(f"  * High-Impact Catalysts Detected: {high_mat_count}")
    lines.append("-" * 115)

    if not relevant_results:
        lines.append("No market news items met the relevance criteria for the target symbol.")
        lines.append("=" * 115)
        return "\n".join(lines)

    table_data = []
    for i, item in enumerate(relevant_results, 1):
        art = item.get("article", {})
        raw_title = art.get("title", "")
        # Truncate title cleanly
        title_disp = (raw_title[:42] + "..") if len(raw_title) > 44 else raw_title
        source_disp = art.get("source", "Unknown")[:18]
        entities_disp = ", ".join(item.get("entity_matches", []))
        if len(entities_disp) > 20:
            entities_disp = entities_disp[:18] + ".."
        elif not entities_disp:
            entities_disp = "N/A"

        sim_disp = f"{item.get('semantic_similarity', 0.0) * 100:.1f}%"
        sent_info = item.get("sentiment", {})
        polarity_disp = f"{sent_info.get('polarity', 'neutral').upper()} ({sent_info.get('confidence', 0.0) * 100:.0f}%)"
        net_disp = f"{sent_info.get('net_score', 0.0):+.4f}"
        mat_disp = item.get("materiality", "LOW")
        reason_disp = item.get("reason", "")
        if len(reason_disp) > 30:
            reason_disp = reason_disp[:28] + ".."

        table_data.append([
            i,
            title_disp,
            source_disp,
            entities_disp,
            sim_disp,
            polarity_disp,
            net_disp,
            mat_disp,
            reason_disp
        ])

    headers = [
        "#",
        "Title",
        "Source",
        "Entity Matches",
        "Semantic Sim",
        "Sentiment (Pol + Conf)",
        "Net Score",
        "Materiality",
        "Reason"
    ]
    lines.append(tabulate(table_data, headers=headers, tablefmt="grid"))

    lines.append("\n[KEY CATALYSTS & BREAKING HEADLINES]")
    for i, item in enumerate(relevant_results[:5], 1):
        art = item.get("article", {})
        title = art.get("title", "Untitled")
        source = art.get("source", "Unknown")
        sent = item.get("sentiment", {})
        mat = item.get("materiality", "LOW")
        reason = item.get("reason", "")
        link = art.get("link", "")
        lines.append(f"  [{i}] {title}")
        lines.append(f"      Source: {source} | Materiality: {mat} | Polarity: {sent.get('polarity', '').upper()} (Net: {sent.get('net_score', 0.0):+.4f})")
        lines.append(f"      Analysis: {reason}")
        if link:
            lines.append(f"      URL: {link}")

    lines.append("=" * 115)
    return "\n".join(lines)


def run_news_relevance_check(
    symbol: str = "NIFTY",
    top_n: int = 10
) -> List[Dict[str, Any]]:
    """
    Main orchestration function for the news relevance check CLI and pipeline.
    Ingests financial news feeds (with graceful fallback fixtures), evaluates
    relevance and sentiment via NiftyRelevanceChecker, prints the structured
    intelligence brief to stdout, and returns the relevant results.
    """
    print(f"\n[News Relevance Pipeline] Initiating targeted intelligence scan for symbol: '{symbol}'...")
    checker = NiftyRelevanceChecker()

    articles: List[Dict[str, Any]] = []
    try:
        from src.scraper import fetch_financial_news
        print("[News Relevance Pipeline] Fetching latest market feeds...")
        articles = fetch_financial_news(limit_per_feed=10)
    except Exception as e:
        print(f"[News Relevance Pipeline] Warning: External feed fetch error: {e}")

    if not articles:
        print("[News Relevance Pipeline] Note: Live feeds returned 0 articles or unavailable. Using standard market news fixtures...")
        articles = get_fallback_news_fixtures()

    total_scanned = len(articles)
    print(f"[News Relevance Pipeline] Evaluating {total_scanned} articles for NIFTY relevance and FinBERT sentiment...")
    relevant_results = checker.filter_relevant_news(articles, symbol=symbol, top_n=top_n)

    brief_text = format_intelligence_brief(
        relevant_results=relevant_results,
        symbol=symbol,
        total_analyzed=total_scanned
    )
    print("\n" + brief_text)
    return relevant_results
