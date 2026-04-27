import os
import asyncio
import threading
import json
import requests
from http.server import BaseHTTPRequestHandler, HTTPServer

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

# ---------------- CONFIG ---------------- #

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODELS_URL = "https://integrate.api.nvidia.com/v1/models"

DEFAULT_MODEL = "stepfun-ai/step-3.5-flash"

if not TELEGRAM_BOT_TOKEN or not NVIDIA_API_KEY:
    print("❌ Missing ENV variables")
    exit(1)

# ---------------- MODELS ---------------- #

def get_models():
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}"}
    try:
        res = requests.get(MODELS_URL, headers=headers, timeout=20)
        if res.status_code != 200:
            return []
        return [m["id"] for m in res.json().get("data", []) if "id" in m]
    except:
        return []

# ---------------- STREAMING NVIDIA ---------------- #

def stream_nvidia(user_text, model):
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": user_text}
        ],
        "temperature": 0.6,
        "max_tokens": 800,
        "stream": True
    }

    with requests.post(API_URL, headers=headers, json=payload, stream=True, timeout=60) as res:
        if res.status_code != 200:
            yield f"API Error {res.status_code}"
            return

        for line in res.iter_lines():
            if not line:
                continue

            line = line.decode("utf-8")

            if line.startswith("data: "):
                line = line[6:]

            if line == "[DONE]":
                break

            try:
                chunk = json.loads(line)
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta
            except:
                continue

# ---------------- TELEGRAM COMMANDS ---------------- #

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    model = context.user_data.get("model", DEFAULT_MODEL)
    await update.message.reply_text(
        f"🤖 AI Bot Ready\n\n🧠 Model:\n{model}\n\n"
        "/models\n/setmodel <name>\n/current"
    )

async def models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Loading models...")
    models_list = get_models()

    if not models_list:
        await update.message.reply_text("❌ No models found")
        return

    text = "📦 Models:\n\n"
    for i, m in enumerate(models_list[:50]):
        text += f"{i+1}. {m}\n"

    await update.message.reply_text(text)

async def set_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /setmodel <model_name>")
        return

    model = " ".join(context.args)
    context.user_data["model"] = model

    await update.message.reply_text(f"✅ Model set:\n{model}")

async def current(update: Update, context: ContextTypes.DEFAULT_TYPE):
    model = context.user_data.get("model", DEFAULT_MODEL)
    await update.message.reply_text(f"🧠 Current Model:\n{model}")

# ---------------- LIVE CHAT (STREAMING) ---------------- #

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    model = context.user_data.get("model", DEFAULT_MODEL)

    sent = await update.message.reply_text("⏳ thinking...")

    full_text = ""

    def run_stream():
        return list(stream_nvidia(user_text, model))

    loop = asyncio.get_event_loop()
    chunks = await loop.run_in_executor(None, run_stream)

    for chunk in chunks:
        full_text += chunk

        try:
            await sent.edit_text(full_text + " ▌")
        except:
            pass

    try:
        await sent.edit_text(full_text)
    except:
        pass

# ---------------- WEB SERVER (RENDER FIX) ---------------- #

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot running")

def run_server():
    port = int(os.environ.get("PORT", 10000))
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()

# ---------------- RUN BOT ---------------- #

async def run_bot():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("models", models))
    app.add_handler(CommandHandler("setmodel", set_model))
    app.add_handler(CommandHandler("current", current))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    await app.initialize()
    await app.start()
    await app.updater.start_polling()

    print("🤖 Bot Running with Streaming...")

    await asyncio.Event().wait()

# ---------------- MAIN ---------------- #

if __name__ == "__main__":
    threading.Thread(target=run_server).start()
    asyncio.run(run_bot())
