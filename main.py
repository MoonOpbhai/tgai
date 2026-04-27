import os
import asyncio
import threading
import requests
from http.server import BaseHTTPRequestHandler, HTTPServer

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# ---------------- CONFIG ---------------- #

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODELS_URL = "https://integrate.api.nvidia.com/v1/models"

DEFAULT_MODEL = "stepfun-ai/step-3.5-flash"

# ---------------- SAFETY ---------------- #

if not TELEGRAM_BOT_TOKEN or not NVIDIA_API_KEY:
    print("❌ Missing ENV variables")
    exit(1)

# ---------------- MODEL HELPERS ---------------- #

def get_models():
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}"}

    try:
        res = requests.get(MODELS_URL, headers=headers, timeout=20)

        if res.status_code != 200:
            return []

        data = res.json().get("data", [])

        models = []
        for m in data:
            mid = m.get("id", "")
            if mid:
                models.append(mid)

        return models

    except Exception as e:
        print("Model error:", e)
        return []

# ---------------- NVIDIA CHAT ---------------- #

def ask_nvidia(user_text, model):
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Be concise and intelligent."
            },
            {"role": "user", "content": user_text}
        ],
        "temperature": 0.6,
        "max_tokens": 800,
        "stream": False
    }

    try:
        res = requests.post(API_URL, headers=headers, json=payload, timeout=60)

        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]

        return f"API Error {res.status_code}: {res.text}"

    except Exception as e:
        return f"Error: {str(e)}"

# ---------------- TELEGRAM HANDLERS ---------------- #

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    model = context.user_data.get("model", DEFAULT_MODEL)

    await update.message.reply_text(
        f"🤖 Bot Ready!\n\n"
        f"🧠 Default Model:\n{model}\n\n"
        f"/models - list models\n"
        f"/setmodel <name> - change model\n"
        f"/current - current model\n"
    )

# -------- MODELS -------- #

async def models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Fetching models...")

    models_list = get_models()

    if not models_list:
        await update.message.reply_text("❌ No models found")
        return

    text = "📦 Available NVIDIA Models:\n\n"

    for i, m in enumerate(models_list[:50]):
        text += f"{i+1}. {m}\n"

    await update.message.reply_text(text)

# -------- SET MODEL -------- #

async def set_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /setmodel <model_name>")
        return

    model = " ".join(context.args).strip()
    context.user_data["model"] = model

    await update.message.reply_text(f"✅ Model set:\n\n🧠 {model}")

# -------- CURRENT MODEL -------- #

async def current(update: Update, context: ContextTypes.DEFAULT_TYPE):
    model = context.user_data.get("model", DEFAULT_MODEL)
    await update.message.reply_text(f"🧠 Current Model:\n{model}")

# -------- CHAT -------- #

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text

    model = context.user_data.get("model", DEFAULT_MODEL)

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )

    reply = ask_nvidia(user_text, model)

    await update.message.reply_text(reply)

# ---------------- WEB SERVER (RENDER FIX) ---------------- #

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is running")

def run_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), Handler)
    server.serve_forever()

# ---------------- BOT RUN ---------------- #

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

    print("🤖 StepFun Agent Bot Running...")

    await asyncio.Event().wait()

# ---------------- MAIN ---------------- #

if __name__ == "__main__":
    threading.Thread(target=run_server).start()
    asyncio.run(run_bot())
