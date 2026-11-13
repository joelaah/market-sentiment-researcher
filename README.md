# AI Market Research Assistant (Python + Supabase pgvector)

An end-to-end Python pipeline designed for financial news ingestion, vector semantic indexing with Supabase `pgvector`, and market intelligence retrieval.

---

## 📌 Architecture & Design Philosophy

1. **Scraper (`src/scraper.py`):** Ingests real-time financial market news from high-authority financial RSS feeds (Yahoo Finance, CNBC, MarketWatch). It normalizes HTML, strips tracking boilerplate, and structures article metadata.
2. **Vector Embeddings (`src/embeddings.py`):** Uses FastEmbed (`BAAI/bge-small-en-v1.5`) to convert market headlines and article summaries into 384-dimensional dense vectors locally without needing heavy GPU hardware or paid API keys.
3. **Database (`src/db.py` & `schema.sql`):** Stores articles and high-dimensional vectors in Supabase with an HNSW cosine distance index. Includes a PostgreSQL RPC function (`match_market_news`) for vector similarity search.
4. **Researcher Agent (`src/researcher.py`):** Takes a natural language market question or ticker (e.g., *"Federal reserve interest rate cuts"* or *"NVIDIA artificial intelligence chip demand"*), projects the question into vector space, and finds relevant historical market developments.

---

## ⚠️ Financial Reality Check: Why Quantitative Traders Focus on Edge, Not "Zero Losses"

* **No "Zero-Loss" or "100% Win Rate":** Financial markets operate under partial information and non-deterministic behavior. Claims that a trading bot "never loses" are marketing myths.
* **The Goal of Quantitative Systems:** Systematic traders focus on **asymmetric risk-reward** (e.g. cutting losses quickly and letting winners run) and **statistical edge** (e.g., winning 53% of trades with a positive expectancy).
* **Role of AI in Modern Markets:** Modern hedge funds use AI and vector stores for **information extraction, sentiment summarization, and hypothesis generation**, not for blind automatic trade execution without human oversight and risk limits.

---

## 🚀 Setup & Getting Started

### 1. Install Dependencies
```bash

python -m pip install -r requirements.txt
```

### 2. Configure Supabase pgvector
1. Log in to your [Supabase Dashboard](https://supabase.com).
2. Open the **SQL Editor**.
3. Run the SQL script found in `schema.sql`. This:
   - Enables the `vector` extension.
   - Creates the `market_news` table.
   - Creates the HNSW index for high-speed similarity search.
   - Deploys the `match_market_news` RPC search function.

### 3. Set Environment Variables
Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```
Update `.env` with your Supabase Project URL and Service Role Key (from Supabase **Project Settings -> API**).

---

## 💻 Usage

### 1. Scrape & Index Market News
Scrape the latest headlines from financial feeds and automatically embed & upsert them into Supabase:
```bash
python run_pipeline.py --scrape
```

### 2. Query Market Intelligence
Ask market questions or search by ticker:
```bash
python run_pipeline.py --query "Fed interest rate inflation outlook"
```
### 3. Run Multi-Bot Tournament Evaluation
Simultaneously evaluate all 5 bot archetypes on live market prices and news sentiment:
```bash
python run_pipeline.py --trade
```

### 4. View Live Tournament Leaderboard & Borderlist
Inspect current standings, total equity, net P&L, win rates, and open holdings across all bots:
```bash
python run_pipeline.py --leaderboard
```
View safety guardrails and tickers disqualified by the borderlist filter:
```bash
python run_pipeline.py --borderlist
```
Or launch interactive mode:
```bash
python run_pipeline.py
```

---

## 🛡️ Safety Border / Guardrail Filter ("Borderlist")

To protect bots from penny stock manipulation, reverse splits, and illiquid pump-and-dump traps:
1. **Minimum Share Price Floor:** Enforces a strict **$5.00 minimum price**. No bot is permitted to purchase stocks trading below $5.00.
2. **Minimum Liquidity & Scale:** Requires legitimate daily trading volume (**$\ge$ 200,000 volume**) OR institutional-grade market cap (**$\ge$ $250M**).
3. **Automated Skipped Logging:** Any hot-trending or candidate ticker disqualified by the borderlist logs a skipped notice with the specific violation reason and is persisted on the borderlist audit registry.

---

## 📈 Probability-Weighted Upside & Expected Return Ranking

Rather than scanning tickers arbitrarily or focusing only on stagnant mega-caps:
1. **Upside Probability ($P_{\text{upside}}$):** Blends FinBERT institutional NLP sentiment (+0.40 weight), 5-day breakout momentum above 20d SMA (+0.35 weight), and catalyst clarity (+0.25 weight).
2. **Small/Mid-Cap Discovery:** High-potential small-cap and mid-cap stocks with verified catalysts (e.g. contracts, earnings beats, FDA approvals) can achieve expected return rankings higher than sluggish mega-caps, allowing bots to capture explosive rate increases while staying guarded by the $5 floor.
3. **Ranked Candidate Evaluation:** All bots evaluate candidates ordered by expected return descending, deploying capital into the highest expected-value opportunities first.

---

## ⚔️ Multi-Bot Tournament Architecture

The tournament arena features 5 distinct autonomous bot archetypes competing simultaneously with **independent $5,000 paper cash pools** (total arena pool: $25,000):

| Bot Archetype | Strategy Philosophy | Risk Controls | Target Upside |
| :--- | :--- | :--- | :--- |
| **Conservative Swing** | High-conviction AI sentiment (FinBERT $\ge$ 1.5) + confirmed 20d SMA uptrend | Stop Loss -7.5% | Take Profit +12.0% |
| **Aggressive Breakout** | High-velocity momentum expansion (5d $\ge$ +2.0%) + tight risk limits | Stop Loss -5.0% | Take Profit +18.0% |
| **Dip Buyer / Contrarian** | Mean-reversion buying oversold pullbacks (-1.5% to -8.0%) with stabilizing sentiment | Stop Loss -6.0% | Take Profit +8.0% |
| **FinBERT Catalyst Hunter** | Pure AI NLP sentiment sniper pouncing on institutional-grade catalyst spikes | Stop Loss -7.0% | Take Profit +10.0% |
| **Trend Follower** | Systematic moving-average channel following riding persistent upward drift | Stop Loss -8.0% | Take Profit +15.0% |

### Evaluation Cycle
- Live news, real-time prices, and volume/market-cap metrics are captured into a shared **`MarketSnapshot`**.
- Candidates are filtered by `BorderlistGuardrail` and ranked by expected return.
- Results and standings are published to GitHub Actions logs and local CLI output via a formatted ASCII leaderboard.
