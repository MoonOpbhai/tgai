import os
import asyncio
import requests
from flask import Flask, request
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ---------------------------------------- #
# 🔐 ENV
# ---------------------------------------- #

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODELS_URL = "https://integrate.api.nvidia.com/v1/models"
DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"

# ---------------------------------------- #
# 🧠 Agent
# ---------------------------------------- #

class HermesAgent:
    def __init__(self):
        self.tools = {"calculator": self.calculator}

    def calculator(self, expression):
        try:
            return str(eval(expression, {"__builtins__": None}, {}))
        except Exception:
            return "Calculation error"

    def think(self, text):
        text_lower = text.lower()
        if "calculate" in text_lower:
            expr = text_lower.replace("calculate", "").strip()
            return self.tools["calculator"](expr)
        return None


agent = HermesAgent()

# ---------------------------------------- #
# 🌐 Flask App
# ---------------------------------------- #

app = Flask(__name__)

# ---------------------------------------- #
# 🤖 Telegram — Lazy Init
# ---------------------------------------- #

_ptb_app = None
_ptb_initialized = False


def get_ptb_app():
    global _ptb_app
    if _ptb_app is None:
        _ptb_app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
        _ptb_app.add_handler(CommandHandler("start", start))
        _ptb_app.add_handler(CommandHandler("models", list_models))
        _ptb_app.add_handler(CommandHandler("setmodel", set_model))
        _ptb_app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
        )
    return _ptb_app

# ---------------------------------------- #
# 🌐 NVIDIA Helpers
# ---------------------------------------- #

def get_models():
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}"}
    try:
        res = requests.get(MODELS_URL, headers=headers, timeout=20)
        if res.status_code == 200:
            return [m["id"] for m in res.json().get("data", [])]
        return [f"API error {res.status_code}"]
    except Exception as e:
        return [str(e)]


def chat_with_nvidia(messages, model):
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 500,
    }
    try:
        res = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]
        return f"API Error {res.status_code}: {res.text}"
    except Exception as e:
        return f"Request Error: {e}"

# ---------------------------------------- #
# 📬 Telegram Handlers
# ---------------------------------------- #

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Bot ready!\n\n"
        "/models → list models\n"
        "/setmodel <n> → change model\n"
        "Try: calculate 5*10"
    )


async def list_models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    models = get_models()
    if not models:
        await update.message.reply_text("No models found.")
        return
    text = "📦 Available Models:\n\n"
    for m in models[:30]:
        text += f"• {m}\n"
    text += f"\nTotal: {len(models)}"
    await update.message.reply_text(text)


async def set_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /setmodel <model_name>")
        return
    context.user_data["model"] = context.args[0]
    await update.message.reply_text(f"✅ Model set to:\n{context.args[0]}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text

    tool_result = agent.think(user_text)
    if tool_result:
        await update.message.reply_text(f"🧮 {tool_result}")
        return

    if "history" not in context.user_data:
        context.user_data["history"] = [
            {"role": "system", "content": "You are a helpful AI assistant."}
        ]

    context.user_data["history"].append({"role": "user", "content": user_text})
    history = context.user_data["history"]
    if len(history) > 11:
        context.user_data["history"] = [history[0]] + history[-10:]

    model = context.user_data.get("model", DEFAULT_MODEL)
    reply = chat_with_nvidia(context.user_data["history"], model)

    context.user_data["history"].append({"role": "assistant", "content": reply})
    await update.message.reply_text(reply)

# ---------------------------------------- #
# 🔗 Flask Routes
# ---------------------------------------- #

@app.route("/")
def home():
    return "Bot is running!", 200


@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json(force=True)
        if not data:
            return "bad request", 400

        ptb = get_ptb_app()

        async def process():
            await ptb.initialize()
            update = Update.de_json(data, ptb.bot)
            await ptb.process_update(update)

        asyncio.run(process())

    except Exception as e:
        print("Webhook error:", e)

    return "ok", 200


# ---------------------------------------- #
# ▶️  Entry point
# ---------------------------------------- #

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
