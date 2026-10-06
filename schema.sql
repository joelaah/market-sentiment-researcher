-- Enable the pgvector extension to work with embedding vectors
create extension if not exists vector with schema extensions;

-- Table to store scraped financial articles and their vector embeddings
create table if not exists public.market_news (
    id uuid primary key default gen_random_uuid(),
    title text not null,
    link text unique not null,
    source text not null,
    published_at timestamptz,
    summary text,
    content text,
    -- 384 dimensions matches standard sentence-transformers (all-MiniLM-L6-v2)
    embedding vector(384),
    created_at timestamptz default now()
);

-- Enable Row Level Security (Supabase Best Practice)
alter table public.market_news enable row level security;

-- Allow read access for authenticated and anonymous users
create policy "Allow read access to market news"
on public.market_news
for select
to authenticated, anon
using (true);

-- Allow insert/update for backend services, authenticated, and anon scripts
create policy "Allow insert and update for market news"
on public.market_news
for all
to anon, authenticated, service_role
using (true)
with check (true);

-- HNSW Index for fast cosine similarity vector search
create index if not exists market_news_embedding_hnsw_idx
on public.market_news
using hnsw (embedding vector_cosine_ops);

-- Similarity search function callable via Supabase RPC (client.rpc('match_market_news', ...))
create or replace function public.match_market_news (
    query_embedding vector(384),
    match_threshold float default 0.3,
    match_count int default 5
)
returns table (
    id uuid,
    title text,
    link text,
    source text,
    published_at timestamptz,
    summary text,
    similarity float
)
language sql stable
as $$
    select
        market_news.id,
        market_news.title,
        market_news.link,
        market_news.source,
        market_news.published_at,
        market_news.summary,
        1 - (market_news.embedding <=> query_embedding) as similarity
    from public.market_news
    where 1 - (market_news.embedding <=> query_embedding) > match_threshold
    order by market_news.embedding <=> query_embedding
    limit match_count;
$$;

-- ============================================================================
-- PAPER TRADING SIMULATION TABLES ($5,000 Starting Balance)
-- ============================================================================

-- Portfolio cash and equity tracking
create table if not exists public.paper_portfolio (
    id uuid primary key default gen_random_uuid(),
    cash_balance numeric not null default 5000.00,
    initial_balance numeric not null default 5000.00,
    updated_at timestamptz default now()
);

-- Active open positions
create table if not exists public.paper_positions (
    id uuid primary key default gen_random_uuid(),
    symbol text not null unique,
    shares numeric not null check (shares >= 0),
    average_buy_price numeric not null,
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);

-- Audit log of simulated trades with AI rationale
create table if not exists public.paper_trades (
    id uuid primary key default gen_random_uuid(),
    symbol text not null,
    action text not null check (action in ('BUY', 'SELL')),
    shares numeric not null,
    price numeric not null,
    total_value numeric not null,
    rationale text,
    created_at timestamptz default now()
);

-- Enable RLS for all paper trading tables
alter table public.paper_portfolio enable row level security;
alter table public.paper_positions enable row level security;
alter table public.paper_trades enable row level security;

-- Policies for paper trading tables
create policy "Allow all on paper_portfolio" on public.paper_portfolio for all to anon, authenticated, service_role using (true) with check (true);
create policy "Allow all on paper_positions" on public.paper_positions for all to anon, authenticated, service_role using (true) with check (true);
create policy "Allow all on paper_trades" on public.paper_trades for all to anon, authenticated, service_role using (true) with check (true);

-- Seed initial $5,000 portfolio record if none exists
insert into public.paper_portfolio (cash_balance, initial_balance)
select 5000.00, 5000.00
where not exists (select 1 from public.paper_portfolio);

