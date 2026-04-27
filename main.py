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

# ───────────────── CONFIG ───────────────── #

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
NVIDIA_API_KEY     = os.getenv("NVIDIA_API_KEY", "").strip()

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"

MAX_HISTORY = 0  # 🚀 RAW MODE → no memory
STREAM_TIMEOUT = 90

if not TELEGRAM_BOT_TOKEN or not NVIDIA_API_KEY:
    raise SystemExit("Missing API keys")

# ───────────────── STATE ───────────────── #

user_locks = defaultdict(asyncio.Lock)
user_stop = set()

# ───────────────── SYSTEM PROMPT (MINIMAL RAW MODE) ───────────────── #

SYSTEM_PROMPT = "You are a helpful assistant."

# ───────────────── STREAM AI (RAW OUTPUT) ───────────────── #

def stream_ai(messages):
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1000,
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

# ───────────────── MESSAGE BUILDER (RAW ONLY USER INPUT) ───────────────── #

def build_messages(text):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text}
    ]

# ───────────────── START ───────────────── #

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 RAW API Bot Online")

# ───────────────── STOP ───────────────── #

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_stop.add(update.effective_chat.id)
    await update.message.reply_text("🛑 Stopped")

# ───────────────── MAIN HANDLER (RAW OUTPUT ONLY) ───────────────── #

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text.strip()

    if user_locks[chat_id].locked():
        await update.message.reply_text("⏳ Busy...")
        return

    async with user_locks[chat_id]:

        sent = await update.message.reply_text("⏳ ...")

        messages = build_messages(text)

        buffer = ""
        last_update = time.time()

        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(None, lambda: list(stream_ai(messages)))

        for chunk in chunks:

            if chat_id in user_stop:
                user_stop.discard(chat_id)
                await sent.edit_text("🛑 Stopped")
                return

            if not chunk:
                continue

            buffer += chunk

            # 🚀 only safe update (no cutting logic)
            if time.time() - last_update > 0.5:
                try:
                    await sent.edit_text(buffer[:4000])
                except:
                    pass
                last_update = time.time()

        # FINAL RAW OUTPUT (NO MODIFICATION)
        if buffer.strip():
            try:
                await sent.edit_text(buffer[:4000])
            except:
                await update.message.reply_text(buffer[:4000])
        else:
            await sent.edit_text("⚠️ Empty response")

# ───────────────── WEB SERVER ───────────────── #

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

    await asyncio.Event().wait()

# ───────────────── ENTRY ───────────────── #

if __name__ == "__main__":
    threading.Thread(target=run_web, daemon=True).start()
    asyncio.run(main())
