"""
Telegram Notification Engine for Market Sentiment & Paper Trading.

Features:
1. Trade Alerts: Instant notification on bot BUY / SELL executions with P&L and rationale.
2. News Catalyst Alerts: High-materiality NIFTY / Indian stock market breaking news alerts.
3. Tournament Leaderboard Briefs: Periodic status updates across all 5 bot personas.
4. Graceful Fallback: Operates silently if TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is unset.
"""

import os
from typing import Optional, Dict, Any, List
import requests
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

def is_telegram_configured() -> bool:
    """Checks whether valid Telegram credentials are provided."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    return bool(token and chat_id and not token.startswith("your_") and not chat_id.startswith("your_"))


def send_telegram_message(text: str, parse_mode: str = "HTML") -> bool:
    """
    Sends an HTML-formatted message to the configured Telegram chat.
    Returns True if sent successfully, False otherwise without raising exceptions.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token or not chat_id:
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": False
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            return True
        else:
            print(f"[Telegram] Failed to send alert: HTTP {resp.status_code} - {resp.text}")
            return False
    except Exception as e:
        print(f"[Telegram] Error sending message: {e}")
        return False


def notify_trade(
    bot_name: str,
    action: str,
    symbol: str,
    shares: float,
    price: float,
    total_val: float,
    rationale: str,
    pnl_str: Optional[str] = None
) -> bool:
    """Sends a formatted trading alert to Telegram."""
    if not is_telegram_configured():
        return False

    is_inr = symbol.endswith((".NS", ".BO")) or symbol == "^NSEI"
    curr_sym = "₹" if is_inr else "$"

    if action.upper() == "BUY":
        icon = "🟢 <b>BUY ORDER EXECUTED</b>"
        action_header = f"<b>{bot_name}</b> entered a new position"
    else:
        icon = "🔴 <b>SELL ORDER EXECUTED</b>"
        action_header = f"<b>{bot_name}</b> closed a position"

    lines = [
        f"{icon}",
        f"{action_header}",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"<b>Asset:</b> <code>{symbol}</code>",
        f"<b>Shares:</b> <code>{shares:.4f}</code>",
        f"<b>Execution Price:</b> {curr_sym}{price:.2f}",
        f"<b>Total Value:</b> {curr_sym}{total_val:.2f}"
    ]

    if pnl_str:
        pnl_icon = "📈" if "+" in pnl_str else "📉"
        lines.append(f"<b>Realized P&L:</b> {pnl_icon} <code>{pnl_str}</code>")

    # Clean rationale for HTML display
    clean_rationale = (rationale or "").replace("<", "&lt;").replace(">", "&gt;")
    lines.append(f"<b>Strategy Rationale:</b> <i>{clean_rationale}</i>")

    message = "\n".join(lines)
    return send_telegram_message(message)


def notify_nifty_catalyst(
    title: str,
    source: str,
    polarity: str,
    confidence: float,
    materiality: str,
    relevance_score: float,
    link: str
) -> bool:
    """Sends a breaking NIFTY / Indian stock news catalyst alert."""
    if not is_telegram_configured():
        return False

    if polarity.lower() == "positive":
        pol_badge = "🟢 <b>BULLISH</b>"
    elif polarity.lower() == "negative":
        pol_badge = "🔴 <b>BEARISH</b>"
    else:
        pol_badge = "⚪ <b>NEUTRAL</b>"

    mat_badge = f"🔥 <b>{materiality.upper()} IMPACT</b>" if materiality.upper() == "HIGH" else f"ℹ️ <b>{materiality.upper()}</b>"

    clean_title = (title or "").replace("<", "&lt;").replace(">", "&gt;")
    clean_source = (source or "Market News").replace("<", "&lt;").replace(">", "&gt;")

    lines = [
        f"🇮🇳 <b>NIFTY CATALYST ALERT</b>",
        f"{mat_badge} | {pol_badge}",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"<b>Headline:</b> {clean_title}",
        f"<b>Source:</b> {clean_source}",
        f"<b>FinBERT Confidence:</b> {confidence * 100:.1f}%",
        f"<b>Relevance Score:</b> {relevance_score:.2f}",
        f"🔗 <a href='{link}'>Read Full Article</a>"
    ]

    message = "\n".join(lines)
    return send_telegram_message(message)


def notify_tournament_leaderboard(leaderboard_rows: List[Dict[str, Any]]) -> bool:
    """Sends a summary of the current bot tournament standings."""
    if not is_telegram_configured():
        return False

    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    lines = [
        "🏆 <b>MULTI-BOT TOURNAMENT LEADERBOARD</b>",
        "━━━━━━━━━━━━━━━━━━━━━━"
    ]

    for i, bot in enumerate(leaderboard_rows):
        rank_icon = medals[i] if i < len(medals) else f"{i+1}."
        name = bot.get("name", "Bot")
        equity = bot.get("equity", 5000.0)
        pnl = bot.get("pnl", 0.0)
        pnl_pct = bot.get("pnl_pct", 0.0)
        win_rate = bot.get("win_rate", 0.0)
        open_pos = bot.get("open_positions", 0)

        pnl_sign = "+" if pnl >= 0 else ""
        lines.append(
            f"{rank_icon} <b>{name}</b>\n"
            f"   Equity: ${equity:,.2f} | P&L: {pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)\n"
            f"   Win Rate: {win_rate:.1f}% | Open Positions: {open_pos}"
        )

    lines.append("━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("<i>Automated Market Intelligence by DeepMind Antigravity</i>")

    message = "\n".join(lines)
    return send_telegram_message(message)
