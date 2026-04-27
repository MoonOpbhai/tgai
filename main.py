import os
import asyncio
import threading
import time
import json
import requests
import logging
import sqlite3
from collections import defaultdict
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
logger = logging.getLogger("Bot")

# ───────────────── CONFIG ───────────────── #

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
NVIDIA_API_KEY     = os.getenv("NVIDIA_API_KEY", "").strip()

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

DEFAULT_MODEL = "moonshotai/kimi-k2-thinking"

DB_FILE = "memory.db"

if not TELEGRAM_BOT_TOKEN or not NVIDIA_API_KEY:
    raise SystemExit("Missing API keys")

# ───────────────── DATABASE ───────────────── #

conn = sqlite3.connect(DB_FILE, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS messages (
    chat_id TEXT,
    role TEXT,
    content TEXT,
    timestamp INTEGER
)
""")
conn.commit()

def save_msg(chat_id, role, content):
    cursor.execute(
        "INSERT INTO messages VALUES (?, ?, ?, ?)",
        (str(chat_id), role, content, int(time.time()))
    )
    conn.commit()

def get_history(chat_id, limit=8):
    cursor.execute(
        "SELECT role, content FROM messages WHERE chat_id=? ORDER BY timestamp DESC LIMIT ?",
        (str(chat_id), limit * 2)
    )
    rows = cursor.fetchall()
    rows.reverse()
    return [{"role": r[0], "content": r[1]} for r in rows]

def clear_memory(chat_id):
    cursor.execute("DELETE FROM messages WHERE chat_id=?", (str(chat_id),))
    conn.commit()

# ───────────────── STATE ───────────────── #

user_model = defaultdict(lambda: DEFAULT_MODEL)
user_mode  = defaultdict(lambda: "clean")  # clean / raw
user_lock  = defaultdict(asyncio.Lock)
user_stop  = set()

SYSTEM_PROMPT = "You are a helpful assistant."

# ───────────────── STREAM AI ───────────────── #

def stream_ai(messages, model):
    payload = {
        "model": model,
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
                           stream=True, timeout=120) as r:

            if r.status_code != 200:
                yield f"API Error: {r.text}"
                return

            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    yield chunk.decode("utf-8", errors="ignore")

    except Exception as e:
        yield f"Stream Error: {e}"

# ───────────────── BUILD MESSAGES ───────────────── #

def build_messages(chat_id, text):
    history = get_history(chat_id)

    return (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": text}]
    )

# ───────────────── COMMANDS ───────────────── #

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Bot Online\n\n"
        "/setmodel <model>\n"
        "/mode clean | raw\n"
        "/clearmem\n"
        "/stop"
    )

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_stop.add(update.effective_chat.id)
    await update.message.reply_text("🛑 Stopped")

async def clearmem(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    clear_memory(chat_id)
    await update.message.reply_text("🗑️ Memory cleared")

async def setmodel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id

    if not context.args:
        await update.message.reply_text("Usage: /setmodel <model>")
        return

    model = " ".join(context.args).strip()
    user_model[chat_id] = model

    await update.message.reply_text(f"✅ Model set:\n{model}")

async def mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id

    if not context.args:
        await update.message.reply_text("Usage: /mode clean or raw")
        return

    m = context.args[0].lower()
    if m not in ["clean", "raw"]:
        await update.message.reply_text("Use: clean or raw")
        return

    user_mode[chat_id] = m
    await update.message.reply_text(f"✅ Mode set: {m}")

# ───────────────── MAIN CHAT ───────────────── #

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text.strip()

    if user_lock[chat_id].locked():
        await update.message.reply_text("⏳ Wait...")
        return

    async with user_lock[chat_id]:

        sent = await update.message.reply_text("⏳ thinking...")

        model = user_model[chat_id]
        mode  = user_mode[chat_id]

        messages = build_messages(chat_id, text)

        buffer = ""
        last_update = time.time()

        loop = asyncio.get_event_loop()

        chunks = await loop.run_in_executor(
            None,
            lambda: list(stream_ai(messages, model))
        )

        for chunk in chunks:

            if chat_id in user_stop:
                user_stop.discard(chat_id)
                await sent.edit_text("🛑 Stopped")
                return

            buffer += chunk

            if time.time() - last_update > 0.6:
                try:
                    await sent.edit_text(buffer[-4000:])
                except:
                    pass
                last_update = time.time()

        final = buffer.strip()

        # ───────── MODE HANDLING ───────── #

        if mode == "clean":
            # try to extract only assistant content
            try:
                data_lines = []
                for line in final.split("\n"):
                    if "content" in line:
                        data_lines.append(line)
                final_output = "\n".join(data_lines) if data_lines else final
            except:
                final_output = final
        else:
            # RAW MODE
            final_output = final

        # send result
        try:
            await sent.edit_text(final_output[:4000])
        except:
            await update.message.reply_text(final_output[:4000])

        # save memory
        save_msg(chat_id, "user", text)
        save_msg(chat_id, "assistant", final_output)

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
    app.add_handler(CommandHandler("clearmem", clearmem))
    app.add_handler(CommandHandler("setmodel", setmodel))
    app.add_handler(CommandHandler("mode", mode))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)

    logger.info("Bot running (clean + raw mode)...")
    await asyncio.Event().wait()

# ───────────────── ENTRY ───────────────── #

if __name__ == "__main__":
    threading.Thread(target=run_web, daemon=True).start()
    asyncio.run(main())
