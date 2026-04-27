import os
import asyncio
import threading
import time
import json
import requests
import logging
import sqlite3
import re
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Bot")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"

DB_FILE = "memory.db"

if not TELEGRAM_BOT_TOKEN or not NVIDIA_API_KEY:
    raise SystemExit("Missing API keys")

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

user_model = defaultdict(lambda: DEFAULT_MODEL)
user_lock = defaultdict(asyncio.Lock)
user_stop = set()

SYSTEM_PROMPT = """
You are a calm, practical, human-like technical assistant.

Talk naturally in the user's language style.
If the user speaks Hinglish, reply in Hinglish.
Answer the actual question directly.
Do not give moral lectures for casual slang or frustration.
Stay calm even if the user is rude.
Only set a short boundary if the user gives direct threats or asks harmful content.
Do not use markdown headings like ###.
Do not use decorative bold like **text**.
Do not use excessive emojis.
Do not call yourself Nemo unless the user asks.
Do not say "Real Talk", "Ab kya karna hai", or robotic assistant lines.
For code or VPS commands, give the exact working code/command first.
Keep replies short unless the user asks for full detail.
"""

def clean_output(text):
    if not text:
        return ""

    text = text.replace("###", "")
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = re.sub(r"^\s*[-]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    bad_lines = [
        "real talk:",
        "ab kya karna hai?",
        "mera naam:",
        "mera kaam:",
        "mera limit:",
    ]

    lines = []
    for line in text.splitlines():
        low = line.strip().lower()
        if any(bad in low for bad in bad_lines):
            continue
        lines.append(line)

    return "\n".join(lines).strip()

def stream_ai(messages, model):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.35,
        "top_p": 0.85,
        "max_tokens": 900,
        "stream": True,
    }

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        with requests.post(
            API_URL,
            json=payload,
            headers=headers,
            stream=True,
            timeout=120
        ) as r:

            if r.status_code != 200:
                yield f"API Error {r.status_code}: {r.text[:500]}"
                return

            for line in r.iter_lines():
                if not line:
                    continue

                line = line.decode("utf-8", errors="ignore").strip()

                if not line.startswith("data:"):
                    continue

                line = line[5:].strip()

                if line == "[DONE]":
                    break

                try:
                    data = json.loads(line)
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except Exception:
                    continue

    except Exception as e:
        yield f"Stream Error: {e}"

def build_messages(chat_id, text):
    history = get_history(chat_id)
    return [{"role": "system", "content": SYSTEM_PROMPT}] + history + [
        {"role": "user", "content": text}
    ]

async def smooth_stream(message_obj, generator, chat_id):
    buffer = ""
    last_update = 0
    update_delay = 0.35

    for chunk in generator:
        if chat_id in user_stop:
            user_stop.discard(chat_id)
            await message_obj.edit_text("Stopped")
            return None

        buffer += chunk

        if time.time() - last_update > update_delay:
            try:
                visible = clean_output(buffer)
                if visible:
                    await message_obj.edit_text(visible[-4000:])
            except Exception:
                pass
            last_update = time.time()

    return clean_output(buffer)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Bot online. Message bhejo.")

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_stop.add(update.effective_chat.id)
    await update.message.reply_text("Stopping...")

async def clearmem(update: Update, context: ContextTypes.DEFAULT_TYPE):
    clear_memory(update.effective_chat.id)
    await update.message.reply_text("Memory cleared.")

async def setmodel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /setmodel <model_name>")
        return

    model = " ".join(context.args).strip()
    user_model[update.effective_chat.id] = model
    await update.message.reply_text(f"Model set:\n{model}")

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text.strip()

    if not text:
        return

    if user_lock[chat_id].locked():
        await update.message.reply_text("Wait, pehle wala response complete hone do.")
        return

    async with user_lock[chat_id]:
        sent = await update.message.reply_text("Thinking...")

        model = user_model[chat_id]
        messages = build_messages(chat_id, text)

        gen = stream_ai(messages, model)
        final = await smooth_stream(sent, gen, chat_id)

        if not final:
            final = "Empty response mila."

        final = clean_output(final)

        try:
            await sent.edit_text(final[:4000])
        except Exception:
            pass

        save_msg(chat_id, "user", text)
        save_msg(chat_id, "assistant", final)

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

async def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stop", stop))
    app.add_handler(CommandHandler("clearmem", clearmem))
    app.add_handler(CommandHandler("setmodel", setmodel))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)

    logger.info("Bot running...")
    await asyncio.Event().wait()

if __name__ == "__main__":
    threading.Thread(target=run_web, daemon=True).start()
    asyncio.run(main())
