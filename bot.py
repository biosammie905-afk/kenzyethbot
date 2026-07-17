"""
Telegram bot with two features:
  1. Word/character counter (with extra stats: sentences, reading time, top words)
  2. Forex trading signals (RSI + MACD + SMA crossover), with automatic
     fallback from Alpha Vantage to Twelve Data if the primary source fails
     or hits its rate limit, plus short-term caching to conserve API calls.

Setup:
  1. pip install -r requirements.txt
  2. Create a .env file (see .env.example) with:
       TELEGRAM_BOT_TOKEN=...
       ALPHA_VANTAGE_API_KEY=...
       TWELVE_DATA_API_KEY=...       (optional but recommended fallback)
  3. python bot.py

Get a Telegram bot token from @BotFather on Telegram.
Get a free Alpha Vantage key: https://www.alphavantage.co/support/#api-key
Get a free Twelve Data key:   https://twelvedata.com/pricing
"""

import os
import re
import time
import logging
from collections import Counter

import requests
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")  # optional fallback

# Pairs to auto-scan in the background job (edit as you like, or set
# WATCHLIST=EURUSD,GBPUSD,USDJPY in .env to override)
WATCHLIST = [p.strip().upper() for p in os.getenv("WATCHLIST", "EURUSD,GBPUSD,USDJPY").split(",") if p.strip()]
SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", 15 * 60))

# Cache candle data for this many seconds so /signal, /watch, and repeated
# requests for the same pair don't burn through the free API quota.
CACHE_TTL_SECONDS = 5 * 60

VALID_CURRENCY_CODES = {
    "USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "CNY", "SEK",
    "NOK", "MXN", "SGD", "HKD", "ZAR", "TRY", "INR", "BRL", "PLN", "DKK",
}

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

_candle_cache: dict[str, tuple[float, pd.DataFrame]] = {}


# ---------------------------------------------------------------------------
# Word counter feature
# ---------------------------------------------------------------------------

def _count_stats(text: str) -> dict:
    text = text.strip()
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    char_count_no_spaces = len(text.replace(" ", "").replace("\n", ""))

    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_count = len(sentences)

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    paragraph_count = max(len(paragraphs), 1 if text else 0)

    avg_word_length = (sum(len(w.strip(".,!?;:\"'()")) for w in words) / word_count) if word_count else 0

    # ~200 words/minute average adult reading speed
    reading_time_seconds = round((word_count / 200) * 60) if word_count else 0

    normalized = [w.strip(".,!?;:\"'()").lower() for w in words]
    normalized = [w for w in normalized if w]
    top_words = Counter(normalized).most_common(5)

    return {
        "words": word_count,
        "chars": char_count,
        "chars_no_spaces": char_count_no_spaces,
        "sentences": sentence_count,
        "paragraphs": paragraph_count,
        "avg_word_length": avg_word_length,
        "reading_time_seconds": reading_time_seconds,
        "top_words": top_words,
    }


def _format_count_reply(stats: dict) -> str:
    minutes, seconds = divmod(stats["reading_time_seconds"], 60)
    reading_time = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
    top_words = ", ".join(f"{w} ({c})" for w, c in stats["top_words"]) or "n/a"

    return (
        f"Words: {stats['words']}\n"
        f"Characters (with spaces): {stats['chars']}\n"
        f"Characters (no spaces): {stats['chars_no_spaces']}\n"
        f"Sentences: {stats['sentences']}\n"
        f"Paragraphs: {stats['paragraphs']}\n"
        f"Avg word length: {stats['avg_word_length']:.1f}\n"
        f"Est. reading time: {reading_time}\n"
        f"Top words: {top_words}"
    )


async def count_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = " ".join(context.args)
    if not text:
        await update.message.reply_text(
            "Send some text after the command, e.g.\n/count The quick brown fox jumps over the lazy dog."
        )
        return
    await update.message.reply_text(_format_count_reply(_count_stats(text)))


async def count_any_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Counts words for any plain text message sent directly to the bot (not a command)."""
    text = update.message.text
    if text and not text.startswith("/"):
        await update.message.reply_text(_format_count_reply(_count_stats(text)))


# ---------------------------------------------------------------------------
# Forex signal feature
# ---------------------------------------------------------------------------

class RateLimitError(Exception):
    """Raised when a data provider signals its rate limit has been hit."""


def _validate_pair(pair: str) -> tuple[str, str]:
    if len(pair) != 6 or not pair.isalpha():
        raise ValueError("Pair must be 6 letters, e.g. EURUSD")
    from_symbol, to_symbol = pair[:3].upper(), pair[3:].upper()
    if from_symbol not in VALID_CURRENCY_CODES or to_symbol not in VALID_CURRENCY_CODES:
        raise ValueError(
            f"Unrecognized currency code in '{pair}'. Example of a valid pair: EURUSD"
        )
    return from_symbol, to_symbol


def _fetch_from_alpha_vantage(from_symbol: str, to_symbol: str, interval: str) -> pd.DataFrame:
    if not ALPHA_VANTAGE_API_KEY:
        raise RuntimeError("ALPHA_VANTAGE_API_KEY not configured")

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "FX_INTRADAY",
        "from_symbol": from_symbol,
        "to_symbol": to_symbol,
        "interval": interval,
        "outputsize": "compact",
        "apikey": ALPHA_VANTAGE_API_KEY,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if "Note" in data or "Information" in data:
        raise RateLimitError(data.get("Note") or data.get("Information"))

    key = f"Time Series FX ({interval})"
    if key not in data:
        raise RuntimeError(f"Unexpected Alpha Vantage response: {data}")

    df = pd.DataFrame.from_dict(data[key], orient="index")
    df = df.rename(columns={
        "1. open": "open", "2. high": "high", "3. low": "low", "4. close": "close",
    })
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _fetch_from_twelve_data(from_symbol: str, to_symbol: str, interval: str) -> pd.DataFrame:
    if not TWELVE_DATA_API_KEY:
        raise RuntimeError("TWELVE_DATA_API_KEY not configured")

    interval_map = {"1min": "1min", "5min": "5min", "15min": "15min", "30min": "30min", "60min": "1h"}
    td_interval = interval_map.get(interval, "1h")

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": f"{from_symbol}/{to_symbol}",
        "interval": td_interval,
        "outputsize": 100,
        "apikey": TWELVE_DATA_API_KEY,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") == "error":
        message = data.get("message", "")
        if "run out of API credits" in message.lower() or "limit" in message.lower():
            raise RateLimitError(message)
        raise RuntimeError(f"Twelve Data error: {message}")

    values = data.get("values")
    if not values:
        raise RuntimeError(f"Unexpected Twelve Data response: {data}")

    df = pd.DataFrame(values)
    df = df.rename(columns={"datetime": "time"})
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    return df[["open", "high", "low", "close"]]


def fetch_forex_candles(pair: str, interval: str = "60min") -> pd.DataFrame:
    """
    Fetch intraday FX candles for a pair like 'EURUSD'.
    Tries Alpha Vantage first, falls back to Twelve Data on rate limit or
    failure, and serves from a short-lived cache when available.
    """
    from_symbol, to_symbol = _validate_pair(pair)
    cache_key = f"{from_symbol}{to_symbol}:{interval}"

    cached = _candle_cache.get(cache_key)
    if cached and (time.time() - cached[0]) < CACHE_TTL_SECONDS:
        return cached[1]

    errors = []
    for source_name, fetch_fn in (
        ("Alpha Vantage", _fetch_from_alpha_vantage),
        ("Twelve Data", _fetch_from_twelve_data),
    ):
        try:
            df = fetch_fn(from_symbol, to_symbol, interval)
            _candle_cache[cache_key] = (time.time(), df)
            return df
        except Exception as e:
            logger.warning(f"{source_name} failed for {pair}: {e}")
            errors.append(f"{source_name}: {e}")

    # Nothing worked — serve stale cache if we have it rather than failing outright
    if cached:
        logger.warning(f"Serving stale cached data for {pair} after all sources failed")
        return cached[1]

    raise RuntimeError("All data sources failed:\n" + "\n".join(errors))


def compute_signal(df: pd.DataFrame) -> dict:
    """
    Apply RSI(14), MACD, and SMA20/SMA50 crossover to candle data.
    Returns indicator readings, an overall Buy/Sell/Neutral call, and a
    confidence percentage based on how many indicators agree.
    """
    if len(df) < 50:
        raise RuntimeError("Not enough candle history to compute reliable indicators (need 50+ candles)")

    df = df.copy()
    df["rsi"] = ta.rsi(df["close"], length=14)
    macd = ta.macd(df["close"])
    df = pd.concat([df, macd], axis=1)
    df["sma20"] = ta.sma(df["close"], length=20)
    df["sma50"] = ta.sma(df["close"], length=50)

    latest = df.iloc[-1]
    rsi = latest["rsi"]
    macd_hist = latest.get("MACDh_12_26_9")
    sma20 = latest["sma20"]
    sma50 = latest["sma50"]

    votes = []
    if pd.notna(rsi):
        votes.append(1 if rsi < 30 else -1 if rsi > 70 else 0)
    if pd.notna(macd_hist):
        votes.append(1 if macd_hist > 0 else -1)
    if pd.notna(sma20) and pd.notna(sma50):
        votes.append(1 if sma20 > sma50 else -1)

    score = sum(votes) if votes else 0
    max_possible = len(votes) if votes else 1
    confidence = round(abs(score) / max_possible * 100) if votes else 0

    if score >= 2:
        overall = "BUY"
    elif score <= -2:
        overall = "SELL"
    else:
        overall = "NEUTRAL"

    return {
        "price": latest["close"],
        "rsi": rsi,
        "macd_hist": macd_hist,
        "sma20": sma20,
        "sma50": sma50,
        "overall": overall,
        "confidence": confidence,
        "timestamp": df.index[-1],
    }


def format_signal(pair: str, result: dict) -> str:
    return (
        f"Pair: {pair}\n"
        f"Time: {result['timestamp']}\n"
        f"Price: {result['price']:.5f}\n"
        f"RSI(14): {result['rsi']:.1f}\n"
        f"MACD hist: {result['macd_hist']:.5f}\n"
        f"SMA20 vs SMA50: {'above' if result['sma20'] > result['sma50'] else 'below'}\n"
        f"Signal: {result['overall']} (confidence: {result['confidence']}%)\n\n"
        f"This is a technical reading, not financial advice."
    )


async def signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "Usage: /signal EURUSD\n"
            f"Watchlist pairs available: {', '.join(WATCHLIST)}"
        )
        return

    pair = context.args[0].upper().replace("/", "")
    await update.message.reply_text(f"Fetching data for {pair}...")

    try:
        df = fetch_forex_candles(pair)
        result = compute_signal(df)
        await update.message.reply_text(format_signal(pair, result))
    except ValueError as e:
        await update.message.reply_text(str(e))
    except Exception as e:
        logger.exception("Error computing signal")
        await update.message.reply_text(
            f"Couldn't get a signal for {pair} right now. ({e})\n"
            f"This can happen if the free API's daily limit was hit — try again later."
        )


async def scan_watchlist_job(context: ContextTypes.DEFAULT_TYPE):
    """Background job: scans WATCHLIST and pushes alerts to the chat that started it."""
    chat_id = context.job.chat_id
    for pair in WATCHLIST:
        try:
            df = fetch_forex_candles(pair)
            result = compute_signal(df)
            if result["overall"] != "NEUTRAL":
                await context.bot.send_message(chat_id=chat_id, text=format_signal(pair, result))
        except Exception:
            logger.exception(f"Error scanning {pair}")


async def watch_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    for job in context.job_queue.get_jobs_by_name(str(chat_id)):
        job.schedule_removal()

    context.job_queue.run_repeating(
        scan_watchlist_job, interval=SCAN_INTERVAL_SECONDS, first=5, chat_id=chat_id, name=str(chat_id),
    )
    await update.message.reply_text(
        f"Watching {', '.join(WATCHLIST)} every {SCAN_INTERVAL_SECONDS // 60} minutes. "
        f"Send /unwatch to stop."
    )


async def unwatch_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    for job in context.job_queue.get_jobs_by_name(str(chat_id)):
        job.schedule_removal()
    await update.message.reply_text("Stopped watching the watchlist.")


# ---------------------------------------------------------------------------
# Bot setup
# ---------------------------------------------------------------------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hi! I can:\n"
        "/count <text> - word/character/reading-time stats\n"
        "/signal EURUSD - get a forex trading signal\n"
        "/watch - auto-scan the watchlist and alert on signals\n"
        "/unwatch - stop the auto-scan\n"
        "/help - show this again"
    )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Unhandled exception", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text(
            "Something went wrong handling that — please try again."
        )


def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN in your .env file")
    if not ALPHA_VANTAGE_API_KEY and not TWELVE_DATA_API_KEY:
        raise RuntimeError("Set at least one of ALPHA_VANTAGE_API_KEY / TWELVE_DATA_API_KEY in your .env file")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", start_command))
    app.add_handler(CommandHandler("count", count_command))
    app.add_handler(CommandHandler("signal", signal_command))
    app.add_handler(CommandHandler("watch", watch_command))
    app.add_handler(CommandHandler("unwatch", unwatch_command))
    # Uncomment to also count any plain text message sent to the bot:
    # app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, count_any_message))
    app.add_error_handler(error_handler)

    logger.info("Bot starting...")
    app.run_polling()


if __name__ == "__main__":
    main()
