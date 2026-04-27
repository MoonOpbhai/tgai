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

DEFAULT_MODEL = "abacusai/dracarys-llama-3.1-70b-instruct"

# ---------------- SAFETY ---------------- #

if not TELEGRAM_BOT_TOKEN or not NVIDIA_API_KEY:
    print("❌ Missing ENV variables")
    exit(1)

# ---------------- MEMORY ---------------- #

user_memory = {}

def get_history(user_id):
    if user_id not in user_memory:
        user_memory[user_id] = []
    return user_memory[user_id]

def add_history(user_id, role, content):
    history = get_history(user_id)
    history.append({"role": role, "content": content})
    if len(history) > 12:
        user_memory[user_id] = history[-12:]

# ---------------- TOOLS ---------------- #

def calculator(expr):
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))
    except:
        return "❌ Calculation error"

def agent_think(text):
    t = text.lower()

    if t.startswith("calculate"):
        return "tool", calculator(t.replace("calculate", "").strip())

    return "llm", None

# ---------------- NVIDIA ---------------- #

def ask_nvidia(messages, model):
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
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

def get_models():
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}"}
    try:
        res = requests.get(MODELS_URL, headers=headers, timeout=20)
        if res.status_code == 200:
            return [m["id"] for m in res.json().get("data", [])]
        return []
    except:
        return []

# ---------------- HANDLERS ---------------- #

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Hermes Agent Bot Ready!\n\n"
        "/models - list models\n"
        "/setmodel <name> - change model\n"
        "/current - current model\n\n"
        "💡 Try: calculate 5*10"
    )

async def models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Fetching models...")

    m = get_models()
    if not m:
        await update.message.reply_text("❌ No models found")
        return

    text = "📦 NVIDIA Models:\n\n"
    for x in m[:50]:
        text += f"• {x}\n"

    await update.message.reply_text(text)

async def set_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /setmodel <model>")
        return

    context.user_data["model"] = " ".join(context.args)
    await update.message.reply_text("✅ Model updated")

async def current(update: Update, context: ContextTypes.DEFAULT_TYPE):
    model = context.user_data.get("model", DEFAULT_MODEL)
    await update.message.reply_text(f"🧠 Current model:\n{model}")

# ---------------- MAIN CHAT (AGENT CORE) ---------------- #

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )

    mode, result = agent_think(text)

    # TOOL MODE
    if mode == "tool":
        await update.message.reply_text(f"🧮 {result}")
        return

    # MODEL
    model = context.user_data.get("model", DEFAULT_MODEL)

    history = get_history(user_id)

    # system prompt (HERMES STYLE)
    system = {
        "role": "system",
        "content": (
            "You are Hermes-style AI agent. "
            "Be intelligent, concise, and reason step-by-step when needed. "
            "Use context and respond like a smart assistant."
        )
    }

    messages = [system] + history + [{"role": "user", "content": text}]

    reply = ask_nvidia(messages, model)

    add_history(user_id, "user", text)
    add_history(user_id, "assistant", reply)

    await update.message.reply_text(reply)

# ---------------- WEB SERVER (RENDER FIX) ---------------- #

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot running")

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

    print("🤖 Hermes Agent Running...")

    await asyncio.Event().wait()

# ---------------- MAIN ---------------- #

if __name__ == "__main__":
    threading.Thread(target=run_server).start()
    asyncio.run(run_bot())
