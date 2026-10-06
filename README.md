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
cd "D:\market-sentiment-researcher"
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
Or launch interactive mode:
```bash
python run_pipeline.py
```
