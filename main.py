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
MODEL = "abacusai/dracarys-llama-3.1-70b-instruct"

# ---------------- SAFETY ---------------- #

if not TELEGRAM_BOT_TOKEN:
    print("❌ TELEGRAM_BOT_TOKEN missing")
    exit(1)

if not NVIDIA_API_KEY:
    print("❌ NVIDIA_API_KEY missing")
    exit(1)

# ---------------- NVIDIA ---------------- #

def ask_nvidia(user_text):
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Keep replies short."},
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

        return f"API Error {res.status_code}"

    except Exception as e:
        return f"Error: {str(e)}"

# ---------------- TELEGRAM BOT ---------------- #

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Bot ready!")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )

    reply = ask_nvidia(update.message.text)
    await update.message.reply_text(reply)

# ---------------- TELEGRAM RUN ---------------- #

async def run_bot():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    await app.initialize()
    await app.start()
    await app.updater.start_polling()

    print("🤖 Bot running...")

    await asyncio.Event().wait()

# ---------------- WEB SERVER (FOR RENDER) ---------------- #

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is running")

def run_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), Handler)
    server.serve_forever()

# ---------------- MAIN ---------------- #

if __name__ == "__main__":
    # start web server (Render requirement)
    threading.Thread(target=run_server).start()

    # start telegram bot
    asyncio.run(run_bot())
