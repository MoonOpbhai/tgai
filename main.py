import os
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# ---------------- CONFIG ---------------- #

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "abacusai/dracarys-llama-3.1-70b-instruct"

# ---------------- NVIDIA ---------------- #

def ask_nvidia(user_text):
    if not NVIDIA_API_KEY:
        return "❌ NVIDIA_API_KEY missing"

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

        return f"API Error {res.status_code}: {res.text}"

    except Exception as e:
        return f"Error: {str(e)}"


# ---------------- HANDLERS ---------------- #

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Bot ready! Just send message.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text

    # ✅ correct typing indicator
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )

    reply = ask_nvidia(user_text)

    await update.message.reply_text(reply)


# ---------------- MAIN ---------------- #

def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot running...")
    app.run_polling()


if __name__ == "__main__":
    main()
