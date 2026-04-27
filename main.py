import os
import asyncio
import threading
import time
import json
import requests
import logging
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, HTTPServer

from telegram import Update, constants
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ───────────────────────── LOGGING ───────────────────────── #

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgentBot")

# ───────────────────────── CONFIG ───────────────────────── #

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
NVIDIA_API_KEY     = os.getenv("NVIDIA_API_KEY", "").strip()

API_URL    = "https://integrate.api.nvidia.com/v1/chat/completions"
MODELS_URL = "https://integrate.api.nvidia.com/v1/models"

DEFAULT_MODEL = "stepfun-ai/step-3.5-flash"

MAX_HISTORY = 15
STREAM_TIMEOUT = 90
UPDATE_INTERVAL = 0.35
RATE_LIMIT = 10
RATE_WINDOW = 60

if not TELEGRAM_BOT_TOKEN or not NVIDIA_API_KEY:
    raise SystemExit("Missing API keys")

# ───────────────────────── STATE ───────────────────────── #

conversation_history = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
user_locks = defaultdict(asyncio.Lock)
user_stop = set()
rate_tracker = defaultdict(lambda: deque(maxlen=RATE_LIMIT))

user_profile = defaultdict(dict)
user_tone = defaultdict(lambda: "friendly")
user_persona = defaultdict(lambda: "default")

PERSONAS = {
    "default": "You are a helpful intelligent AI assistant.",
    "jarvis": "You are JARVIS from Iron Man: calm, precise, genius-level assistant.",
    "teacher": "You explain everything like a great teacher.",
    "coder": "You are a senior software engineer writing production code."
}

# ───────────────────────── RATE LIMIT ───────────────────────── #

def is_rate_limited(chat_id):
    now = time.time()
    dq = rate_tracker[chat_id]

    while dq and now - dq[0] > RATE_WINDOW:
        dq.popleft()

    if len(dq) >= RATE_LIMIT:
        return True

    dq.append(now)
    return False

# ───────────────────────── PROMPT ───────────────────────── #

def build_prompt(chat_id):
    name = user_profile[chat_id].get("name", "")
    tone = user_tone[chat_id]
    persona = PERSONAS.get(user_persona[chat_id], PERSONAS["default"])

    return f"""
{persona}

STYLE:
- Tone: {tone}
- Natural human-like responses

RULES:
- Be accurate
- No hallucination
- Be structured and clear

MEMORY:
User name: {name if name else "unknown"}
"""

# ───────────────────────── STREAM AI ───────────────────────── #

def stream_ai(messages, model):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.6,
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

                line = line.decode().replace("data: ", "")

                if line == "[DONE]":
                    return

                try:
                    delta = json.loads(line)["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except:
                    continue

    except Exception as e:
        yield f"Error: {e}"

# ───────────────────────── HELPERS ───────────────────────── #

def build_messages(chat_id, text, model):
    history = list(conversation_history[chat_id])

    return (
        [{"role": "system", "content": build_prompt(chat_id)}]
        + history
        + [{"role": "user", "content": text}]
    )

def save(chat_id, user_text, bot_text):
    conversation_history[chat_id].append({"role": "user", "content": user_text})
    conversation_history[chat_id].append({"role": "assistant", "content": bot_text})

# ───────────────────────── COMMANDS ───────────────────────── #

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Full Agent Bot Online")

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_stop.add(update.effective_chat.id)
    await update.message.reply_text("🛑 Stopping response...")

async def persona(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("jarvis / teacher / coder / default")
        return
    user_persona[update.effective_chat.id] = context.args[0]
    await update.message.reply_text("🧠 Persona updated")

async def tone(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("friendly / formal / casual")
        return
    user_tone[update.effective_chat.id] = " ".join(context.args)
    await update.message.reply_text("🎭 Tone updated")

async def setmodel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /setmodel <name>")
        return
    context.user_data["model"] = " ".join(context.args)
    await update.message.reply_text("✅ Model changed")

# ───────────────────────── MAIN CHAT ───────────────────────── #

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text.strip()
    model = context.user_data.get("model", DEFAULT_MODEL)

    # learning name
    if "my name is" in text.lower():
        user_profile[chat_id]["name"] = text.split("is")[-1].strip()

    # rate limit
    if is_rate_limited(chat_id):
        await update.message.reply_text("⏱️ Slow down please")
        return

    # lock
    if user_locks[chat_id].locked():
        await update.message.reply_text("⏳ Wait, I am still replying...")
        return

    async with user_locks[chat_id]:

        sent = await update.message.reply_text("⏳ thinking...")

        messages = build_messages(chat_id, text, model)

        buffer = ""
        last = time.time()

        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(
            None, lambda: list(stream_ai(messages, model))
        )

        for c in chunks:

            if chat_id in user_stop:
                user_stop.discard(chat_id)
                await sent.edit_text("🛑 Stopped")
                return

            buffer += c

            if time.time() - last > UPDATE_INTERVAL:
                try:
                    await sent.edit_text(buffer[-3500:])
                except:
                    pass
                last = time.time()

        await sent.edit_text(buffer[:4000])

        save(chat_id, text, buffer)

# ───────────────────────── WEB SERVER ───────────────────────── #

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

def run_web():
    port = int(os.environ.get("PORT", 10000))
    HTTPServer(("0.0.0.0", port), H).serve_forever()

# ───────────────────────── BOT RUN ───────────────────────── #

async def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stop", stop))
    app.add_handler(CommandHandler("persona", persona))
    app.add_handler(CommandHandler("tone", tone))
    app.add_handler(CommandHandler("setmodel", setmodel))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

    await app.bot.delete_webhook(drop_pending_updates=True)
    await app.run_polling()

# ───────────────────────── ENTRY ───────────────────────── #

if __name__ == "__main__":
    threading.Thread(target=run_web, daemon=True).start()
    asyncio.run(main())
