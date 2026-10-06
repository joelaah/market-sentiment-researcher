from typing import List, Dict, Any, Optional
from decimal import Decimal
import re
from datetime import datetime
from tabulate import tabulate
import yfinance as yf
from src.db import get_supabase_client
from src.finbert_analyzer import FinBERTAnalyzer

# Tickers to monitor for sentiment catalysts
WATCHLIST = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AMD", "SPY", "QQQ"]

# Sentiment keywords
BULLISH_SIGNALS = ["surge", "record high", "beat", "rally", "growth", "jump", "bull", "upgrade", "comeback", "outperform", "rebound"]
BEARISH_SIGNALS = ["sink", "slump", "plunge", "rout", "miss", "crash", "bear", "downgrade", "warning", "investigation", "lawsuit", "layoff"]

def get_current_price(symbol: str) -> Optional[float]:
    """Fetches real-time / latest market price for a given ticker."""
    try:
        ticker = yf.Ticker(symbol)
        fast_info = ticker.fast_info
        if fast_info and fast_info.last_price:
            return round(float(fast_info.last_price), 2)
        hist = ticker.history(period="1d")
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]), 2)
    except Exception as e:
        print(f"[PaperTrader] Error fetching price for {symbol}: {e}")
    return None

def get_stock_technical_signal(symbol: str) -> Dict[str, Any]:
    """
    Analyzes technical momentum (20-day Simple Moving Average & 5-day return)
    to confirm whether price trend confirms positive news (avoids catching falling knives).
    """
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="1mo")
        if len(hist) < 10:
            return {"valid": False, "price": 0.0, "above_sma20": False, "momentum_5d": 0.0}
        
        cur_price = float(hist["Close"].iloc[-1])
        sma20 = float(hist["Close"].tail(min(20, len(hist))).mean())
        price_5d_ago = float(hist["Close"].iloc[-min(5, len(hist))])
        momentum_5d = round(((cur_price - price_5d_ago) / price_5d_ago) * 100, 2)
        
        return {
            "valid": True,
            "price": round(cur_price, 2),
            "sma20": round(sma20, 2),
            "above_sma20": cur_price >= sma20,
            "momentum_5d": momentum_5d
        }
    except Exception as e:
        return {"valid": False, "price": 0.0, "above_sma20": False, "momentum_5d": 0.0}


class PaperTradingEngine:
    def __init__(self):
        self.client = get_supabase_client()
        self._ensure_portfolio_initialized()

    def _ensure_portfolio_initialized(self):
        """Ensures the paper_portfolio table has the initial $5,000 cash balance."""
        res = self.client.table("paper_portfolio").select("*").limit(1).execute()
        if not res.data:
            self.client.table("paper_portfolio").insert({
                "cash_balance": 5000.00,
                "initial_balance": 5000.00
            }).execute()

    def get_portfolio_status(self) -> Dict[str, Any]:
        """Returns cash, holdings, current market value, and total equity."""
        port_res = self.client.table("paper_portfolio").select("*").limit(1).execute()
        portfolio_row = port_res.data[0]
        cash = float(portfolio_row["cash_balance"])
        initial = float(portfolio_row["initial_balance"])

        pos_res = self.client.table("paper_positions").select("*").execute()
        positions = pos_res.data or []

        holdings_value = 0.0
        enriched_positions = []

        for pos in positions:
            symbol = pos["symbol"]
            shares = float(pos["shares"])
            avg_price = float(pos["average_buy_price"])
            cur_price = get_current_price(symbol) or avg_price
            market_val = round(shares * cur_price, 2)
            cost_basis = round(shares * avg_price, 2)
            pnl = round(market_val - cost_basis, 2)
            pnl_pct = round((pnl / cost_basis * 100), 2) if cost_basis > 0 else 0.0

            holdings_value += market_val
            enriched_positions.append({
                "symbol": symbol,
                "shares": shares,
                "buy_price": avg_price,
                "cur_price": cur_price,
                "market_value": market_val,
                "pnl": pnl,
                "pnl_pct": pnl_pct
            })

        total_equity = round(cash + holdings_value, 2)
        total_pnl = round(total_equity - initial, 2)
        total_pnl_pct = round((total_pnl / initial * 100), 2)

        return {
            "cash": cash,
            "initial": initial,
            "holdings_value": round(holdings_value, 2),
            "total_equity": total_equity,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "positions": enriched_positions
        }

    def execute_buy(self, symbol: str, amount_dollars: float, rationale: str):
        """Simulates buying stock with paper cash."""
        price = get_current_price(symbol)
        if not price or price <= 0:
            print(f"[PaperTrader] Cannot buy {symbol}: invalid price {price}")
            return

        shares = round(amount_dollars / price, 4)
        if shares <= 0:
            return

        actual_cost = round(shares * price, 2)
        port = self.get_portfolio_status()

        if port["cash"] < actual_cost:
            print(f"[PaperTrader] Insufficient paper cash to buy ${actual_cost:.2f} of {symbol}. Available: ${port['cash']:.2f}")
            return

        # 1. Update Portfolio Cash
        new_cash = round(port["cash"] - actual_cost, 2)
        self.client.table("paper_portfolio").update({
            "cash_balance": new_cash,
            "updated_at": datetime.utcnow().isoformat() + "Z"
        }).eq("id", self.client.table("paper_portfolio").select("id").limit(1).execute().data[0]["id"]).execute()

        # 2. Update or Insert Position
        pos_res = self.client.table("paper_positions").select("*").eq("symbol", symbol).execute()
        if pos_res.data:
            existing = pos_res.data[0]
            old_shares = float(existing["shares"])
            old_price = float(existing["average_buy_price"])
            new_shares = round(old_shares + shares, 4)
            blended_price = round(((old_shares * old_price) + (shares * price)) / new_shares, 2)
            self.client.table("paper_positions").update({
                "shares": new_shares,
                "average_buy_price": blended_price,
                "updated_at": datetime.utcnow().isoformat() + "Z"
            }).eq("symbol", symbol).execute()
        else:
            self.client.table("paper_positions").insert({
                "symbol": symbol,
                "shares": shares,
                "average_buy_price": price
            }).execute()

        # 3. Log Trade Audit
        self.client.table("paper_trades").insert({
            "symbol": symbol,
            "action": "BUY",
            "shares": shares,
            "price": price,
            "total_value": actual_cost,
            "rationale": rationale
        }).execute()

        print(f"✅ [PAPER BUY] {shares:.4f} shares of {symbol} at ${price:.2f} (${actual_cost:.2f})")
        print(f"   Rationale: {rationale}")

    def execute_sell(self, symbol: str, shares: float, rationale: str):
        """Simulates selling a held stock back to paper cash."""
        price = get_current_price(symbol)
        if not price or price <= 0:
            return

        pos_res = self.client.table("paper_positions").select("*").eq("symbol", symbol).execute()
        if not pos_res.data:
            return

        held_shares = float(pos_res.data[0]["shares"])
        shares_to_sell = min(shares, held_shares)
        gross_proceeds = round(shares_to_sell * price, 2)

        port = self.get_portfolio_status()
        new_cash = round(port["cash"] + gross_proceeds, 2)

        # 1. Update Portfolio Cash
        self.client.table("paper_portfolio").update({
            "cash_balance": new_cash,
            "updated_at": datetime.utcnow().isoformat() + "Z"
        }).eq("id", self.client.table("paper_portfolio").select("id").limit(1).execute().data[0]["id"]).execute()

        # 2. Update Position
        remaining_shares = round(held_shares - shares_to_sell, 4)
        if remaining_shares <= 0.0001:
            self.client.table("paper_positions").delete().eq("symbol", symbol).execute()
        else:
            self.client.table("paper_positions").update({
                "shares": remaining_shares,
                "updated_at": datetime.utcnow().isoformat() + "Z"
            }).eq("symbol", symbol).execute()

        # 3. Log Trade Audit
        self.client.table("paper_trades").insert({
            "symbol": symbol,
            "action": "SELL",
            "shares": shares_to_sell,
            "price": price,
            "total_value": gross_proceeds,
            "rationale": rationale
        }).execute()

        print(f"🔴 [PAPER SELL] {shares_to_sell:.4f} shares of {symbol} at ${price:.2f} (${gross_proceeds:.2f})")
        print(f"   Rationale: {rationale}")

    def evaluate_news_and_trade(self):
        """
        Scans recent news in market_news, matches against WATCHLIST,
        scores sentiment, and makes paper allocation decisions.
        """
        print("\n--- Running AI Paper Trading Strategy Evaluation ---")
        port = self.get_portfolio_status()
        print(f"Current Portfolio: Cash ${port['cash']:.2f} | Total Equity: ${port['total_equity']:.2f}")

        # Check existing holdings for Stop Loss (-5%) or Take Profit (+8%)
        for pos in port["positions"]:
            if pos["pnl_pct"] >= 8.0:
                self.execute_sell(pos["symbol"], pos["shares"], f"Take Profit triggered: Position reached +{pos['pnl_pct']:.2f}% gain.")
            elif pos["pnl_pct"] <= -5.0:
                self.execute_sell(pos["symbol"], pos["shares"], f"Stop Loss triggered: Cut loss at {pos['pnl_pct']:.2f}%.")

        # Fetch latest 25 articles from Supabase
        news_res = self.client.table("market_news").select("title, summary, link").order("created_at", desc=True).limit(25).execute()
        articles = news_res.data or []

        for ticker in WATCHLIST:
            # Check if we already hold max positions
            updated_port = self.get_portfolio_status()
            if len(updated_port["positions"]) >= 5 and ticker not in [p["symbol"] for p in updated_port["positions"]]:
                continue  # Max 5 positions for diversification

            ticker_articles = []
            for art in articles:
                text = f"{art['title']} {art.get('summary', '')}".lower()
                # Check for ticker symbol or company name match
                if re.search(r'\b' + re.escape(ticker.lower()) + r'\b', text):
                    ticker_articles.append(art)

            if not ticker_articles:
                continue

            # Calculate sentiment score
            bullish_count = 0
            bearish_count = 0
            latest_headline = ticker_articles[0]["title"]

            for art in ticker_articles:
                combined_text = f"{art['title']} {art.get('summary', '')}".lower()
                bullish_count += sum(1 for kw in BULLISH_SIGNALS if kw in combined_text)
                bearish_count += sum(1 for kw in BEARISH_SIGNALS if kw in combined_text)

            # Multi-Factor Prediction & Conviction Analysis
            tech = get_stock_technical_signal(ticker)
            if not tech["valid"]:
                continue

            # 1. FinBERT Financial Neural Classification Score (0 to 4 points)
            try:
                finbert = FinBERTAnalyzer()
                combined_sample = f"{ticker_articles[0]['title']}. {ticker_articles[0].get('summary', '')}"
                bert_res = finbert.analyze(combined_sample)
                # Map net_score (-1.0 to +1.0) into news_score (-4.0 to +4.0)
                news_score = round(bert_res["net_score"] * 4.0, 2)
                finbert_info = f"FinBERT[Pos={bert_res['positive']:.2f}, Neg={bert_res['negative']:.2f}, Net={bert_res['net_score']:.2f}]"
            except Exception as e:
                news_score = 0.0
                if bullish_count > bearish_count:
                    news_score = min(4.0, bullish_count * 1.5)
                elif bearish_count > bullish_count:
                    news_score = -min(4.0, bearish_count * 1.5)
                finbert_info = f"Keywords(+{bullish_count}/-{bearish_count})"

            # 2. Trend Confirmation Score (0 to 3 points)
            # Above 20-day SMA confirms upward momentum (protects against falling knives)
            trend_score = 3.0 if tech["above_sma20"] else 0.0

            # 3. Short-term Momentum Score (0 to 2 points)
            momentum_score = 2.0 if tech["momentum_5d"] > 0.5 else (1.0 if tech["momentum_5d"] >= 0 else 0.0)

            # 4. Risk / Reward Profile (1 point)
            # Strict stop loss -4%, target +8% -> 2.0x ratio
            rr_score = 1.0

            # Total Conviction Score out of 10
            conviction_score = round(max(0.0, news_score) + trend_score + momentum_score + rr_score, 1)

            print(f"[Conviction Analysis] {ticker}: Score {conviction_score}/10 | {finbert_info} | Price: ${tech['price']} (SMA20: ${tech['sma20']}) | 5d: {tech['momentum_5d']}%")

            # HIGH-CONVICTION BUY TRIGGER (Must score >= 7.0 / 10 to enter trade)
            if conviction_score >= 7.0 and news_score >= 1.5 and tech["above_sma20"]:
                max_trade_size = round(min(updated_port["total_equity"] * 0.15, 750.0), 2)
                if updated_port["cash"] >= max_trade_size and ticker not in [p["symbol"] for p in updated_port["positions"]]:
                    rationale = (
                        f"High-Conviction BUY (Score {conviction_score}/10). "
                        f"FinBERT Signal: {finbert_info}. "
                        f"Catalyst: '{latest_headline[:50]}...'. "
                        f"Trend: Price ${tech['price']} > 20d SMA ${tech['sma20']} (5d: +{tech['momentum_5d']}%)."
                    )
                    self.execute_buy(ticker, max_trade_size, rationale)
            elif conviction_score < 7.0 and news_score > 0:
                print(f"   ↳ ⏭️ Skipped BUY for {ticker}: Conviction {conviction_score}/10 is below 7.0 threshold (needs technical trend confirmation).")

            # HIGH-CONVICTION SELL TRIGGER (Exit on severe bearish news or breakdown)
            elif news_score <= -2.0 or (not tech["above_sma20"] and tech["momentum_5d"] < -3.0):
                for p in updated_port["positions"]:
                    if p["symbol"] == ticker:
                        rationale = f"Exit triggered: Bearish FinBERT signal ({finbert_info}, 5d: {tech['momentum_5d']}%)"
                        self.execute_sell(ticker, p["shares"], rationale)

    def print_dashboard(self):
        """Displays formatted paper trading portfolio summary and trade log."""
        port = self.get_portfolio_status()
        print("\n" + "=" * 80)
        print("          AI PAPER TRADING 30-DAY CHALLENGE: $5,000 STARTING TEST          ")
        print("=" * 80)
        print(f"Initial Capital:   ${port['initial']:.2f}")
        print(f"Available Cash:    ${port['cash']:.2f}")
        print(f"Holdings Value:    ${port['holdings_value']:.2f}")
        print(f"Total Equity:      ${port['total_equity']:.2f}")
        pnl_sign = "+" if port['total_pnl'] >= 0 else ""
        print(f"Net Profit/Loss:   {pnl_sign}${port['total_pnl']:.2f} ({pnl_sign}{port['total_pnl_pct']:.2f}%)")
        print("-" * 80)

        if port["positions"]:
            print("Current Open Positions:")
            pos_table = []
            for p in port["positions"]:
                pos_table.append([
                    p["symbol"],
                    f"{p['shares']:.4f}",
                    f"${p['buy_price']:.2f}",
                    f"${p['cur_price']:.2f}",
                    f"${p['market_value']:.2f}",
                    f"{'+' if p['pnl'] >= 0 else ''}${p['pnl']:.2f} ({'+' if p['pnl_pct'] >= 0 else ''}{p['pnl_pct']:.2f}%)"
                ])
            print(tabulate(pos_table, headers=["Symbol", "Shares", "Avg Buy", "Current", "Value", "Unrealized P&L"], tablefmt="grid"))
        else:
            print("No open positions. 100% in cash.")

        # Show last 5 trades
        trade_res = self.client.table("paper_trades").select("*").order("created_at", desc=True).limit(5).execute()
        trades = trade_res.data or []
        if trades:
            print("\nRecent Executed Paper Trades:")
            trade_table = []
            for t in trades:
                trade_table.append([
                    t["created_at"][:16].replace("T", " "),
                    t["action"],
                    t["symbol"],
                    f"{float(t['shares']):.4f}",
                    f"${float(t['price']):.2f}",
                    f"${float(t['total_value']):.2f}",
                    t.get("rationale", "")[:40] + "..."
                ])
            print(tabulate(trade_table, headers=["Date", "Action", "Symbol", "Shares", "Price", "Total", "Rationale"], tablefmt="grid"))
        print("=" * 80)
