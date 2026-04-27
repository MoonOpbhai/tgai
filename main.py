import os
import asyncio
import threading
import time
import json
import requests
import logging
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, HTTPServer

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ───────────────── LOGGING ───────────────── #

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgentBot")

# ───────────────── CONFIG ───────────────── #

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
NVIDIA_API_KEY     = os.getenv("NVIDIA_API_KEY", "").strip()

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

MODEL = "openai/gpt-oss-120b"

MAX_HISTORY = 20
STREAM_TIMEOUT = 90
UPDATE_INTERVAL = 0.5

if not TELEGRAM_BOT_TOKEN or not NVIDIA_API_KEY:
    raise SystemExit("Missing API keys")

# ───────────────── MEMORY (FIXED) ───────────────── #

conversation_history = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
user_locks = defaultdict(asyncio.Lock)
user_stop = set()

SYSTEM_PROMPT = """
You are a highly intelligent AI assistant.
Remember conversation context and respond naturally like a human.
"""

# ───────────────── STREAM AI (FIXED SAFE) ───────────────── #

def stream_ai(messages):
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.6,
        "max_tokens": 900,
        "stream": True,
    }

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        with requests.post(API_URL, json=payload, headers=headers,
                           stream=True, timeout=STREAM_TIMEOUT) as r:

            if r.status_code != 200:
                yield f"API Error {r.status_code}"
                return

            for line in r.iter_lines():
                if not line:
                    continue

                try:
                    line = line.decode().replace("data: ", "")
                    if line == "[DONE]":
                        return

                    data = json.loads(line)

                    delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        yield delta

                except:
                    continue

    except Exception as e:
        yield f"Error: {e}"

# ───────────────── MEMORY BUILDER (FIXED) ───────────────── #

def build_messages(chat_id, text):
    history = list(conversation_history[chat_id])

    return (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": text}]
    )

# ───────────────── SAVE MEMORY (FIXED) ───────────────── #

def save(chat_id, user_text, bot_text):
    conversation_history[chat_id].append({"role": "user", "content": user_text})
    conversation_history[chat_id].append({"role": "assistant", "content": bot_text})

# ───────────────── START ───────────────── #

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Stable Memory Agent Online")

# ───────────────── STOP ───────────────── #

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_stop.add(update.effective_chat.id)
    await update.message.reply_text("🛑 Stopped")

# ───────────────── MAIN HANDLER (FIXED MEMORY + STREAM) ───────────────── #

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text.strip()

    if user_locks[chat_id].locked():
        await update.message.reply_text("⏳ Wait, still replying...")
        return

    async with user_locks[chat_id]:

        sent = await update.message.reply_text("⏳ thinking...")

        messages = build_messages(chat_id, text)

        buffer = ""
        last_update = time.time()

        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(None, lambda: list(stream_ai(messages)))

        for chunk in chunks:

            if chat_id in user_stop:
                user_stop.discard(chat_id)
                await sent.edit_text("🛑 Stopped")
                return

            buffer += chunk

            # 🔥 SAFE UPDATE (NO BREAKING TEXT)
            if time.time() - last_update > UPDATE_INTERVAL:
                try:
                    await sent.edit_text(buffer[:4000])
                except:
                    pass
                last_update = time.time()

        # final output
        if buffer.strip():
            await sent.edit_text(buffer[:4000])
        else:
            await sent.edit_text("⚠️ Empty response")

        # 🧠 MEMORY SAVE (IMPORTANT FIX)
        save(chat_id, text, buffer)

# ───────────────── WEB SERVER (RENDER SAFE) ───────────────── #

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, *args):
        pass

def run_web():
    port = int(os.environ.get("PORT", 10000))
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()

# ───────────────── BOT RUN ───────────────── #

async def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stop", stop))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)

    logger.info("Bot Running with Memory + Stable Stream")
    await asyncio.Event().wait()

# ───────────────── ENTRY ───────────────── #

if __name__ == "__main__":
    threading.Thread(target=run_web, daemon=True).start()
    asyncio.run(main())
