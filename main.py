import os
import asyncio
import json
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
DEFAULT_MODEL = "qwen/qwen2.5-72b-instruct"

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


def get_ptb_app():
    global _ptb_app
    if _ptb_app is None:
        _ptb_app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
        _ptb_app.add_handler(CommandHandler("start", start))
        _ptb_app.add_handler(CommandHandler("models", list_models))
        _ptb_app.add_handler(CommandHandler("setmodel", set_model))
        _ptb_app.add_handler(CommandHandler("search", search_models))
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
        res = requests.get(MODELS_URL, headers=headers, timeout=10)
        if res.status_code == 200:
            return [m["id"] for m in res.json().get("data", [])]
        return []
    except Exception:
        return []


def stream_nvidia(messages, model):
    """Generator — yields text chunks as they arrive from NVIDIA API."""
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.6,
        "max_tokens": 1024,
        "stream": True,
    }
    with requests.post(API_URL, headers=headers, json=payload, timeout=60, stream=True) as res:
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
                return
            try:
                chunk = json.loads(line)
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta
            except Exception:
                continue

# ---------------------------------------- #
# 📬 Telegram Handlers
# ---------------------------------------- #

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    model = context.user_data.get("model", DEFAULT_MODEL)
    await update.message.reply_text(
        f"🤖 Bot ready!\n\n"
        f"🧠 Model: {model}\n\n"
        f"/models → puri list\n"
        f"/search <keyword> → model dhundo\n"
        f"/setmodel <name> → model badlo\n"
        f"calculate 5*10 → calculator"
    )


async def list_models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Fetching models...")
    models = get_models()
    if not models:
        await update.message.reply_text("❌ Models nahi mile.")
        return
    chunk_size = 50
    for i in range(0, len(models), chunk_size):
        chunk = models[i:i + chunk_size]
        text = f"📦 Models ({i+1}-{i+len(chunk)}):\n\n"
        for m in chunk:
            text += f"• {m}\n"
        await update.message.reply_text(text)
    await update.message.reply_text(f"✅ Total: {len(models)} models")


async def search_models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /search <keyword>\nExample: /search qwen")
        return
    keyword = " ".join(context.args).lower()
    models = get_models()
    matched = [m for m in models if keyword in m.lower()]
    if not matched:
        await update.message.reply_text(f"❌ '{keyword}' se koi model nahi mila.")
        return
    text = f"🔍 '{keyword}' results:\n\n"
    for m in matched:
        text += f"• {m}\n"
    text += f"\nTotal: {len(matched)}"
    await update.message.reply_text(text)


async def set_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /setmodel <model_name>")
        return
    context.user_data["model"] = context.args[0]
    await update.message.reply_text(f"✅ Model set:\n{context.args[0]}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text

    # 🧠 Tool check
    tool_result = agent.think(user_text)
    if tool_result:
        await update.message.reply_text(f"🧮 {tool_result}")
        return

    # 💾 History
    if "history" not in context.user_data:
        context.user_data["history"] = [
            {"role": "system", "content": "You are a fast, helpful AI assistant. Keep responses concise."}
        ]
    context.user_data["history"].append({"role": "user", "content": user_text})
    history = context.user_data["history"]
    if len(history) > 11:
        context.user_data["history"] = [history[0]] + history[-10:]

    model = context.user_data.get("model", DEFAULT_MODEL)

    # Send placeholder message — will be edited live
    sent_msg = await update.message.reply_text("⏳")

    full_text = ""
    last_sent = ""
    UPDATE_EVERY = 20  # update message every 20 new characters

    def do_stream():
        """Run blocking generator — called in thread executor."""
        return list(stream_nvidia(context.user_data["history"], model))

    loop = asyncio.get_event_loop()
    chunks = await loop.run_in_executor(None, do_stream)

    for delta in chunks:
        full_text += delta
        # Update message every UPDATE_EVERY chars to avoid flood limits
        if len(full_text) - len(last_sent) >= UPDATE_EVERY:
            try:
                await sent_msg.edit_text(full_text + " ▌")
                last_sent = full_text
            except Exception:
                pass

    # Final edit — clean, no cursor
    if full_text:
        try:
            await sent_msg.edit_text(full_text)
        except Exception:
            await update.message.reply_text(full_text)
    else:
        await sent_msg.edit_text("❌ No response")

    context.user_data["history"].append({"role": "assistant", "content": full_text})

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
