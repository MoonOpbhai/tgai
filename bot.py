import os
import logging
from typing import Dict, Any
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)
from ollamafreeapi import OllamaFreeAPI

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ─── Config ─────────────────────────────────────────────────────────────────
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "llama3:8b-instruct")
# Render automatically RENDER_EXTERNAL_URL set karta hai
WEBHOOK_URL = os.environ.get("RENDER_EXTERNAL_URL", "")
PORT = int(os.environ.get("PORT", 8080))

MODELS = {
    "llama3:8b-instruct": "🦙 LLaMA 3 (8B)",
    "mistral:7b-v0.2":    "🌪️ Mistral (7B)",
    "deepseek-r1:7b":     "🔍 DeepSeek R1 (7B)",
    "qwen:7b-chat":       "🦄 Qwen (7B)",
}

client = OllamaFreeAPI()
user_state: Dict[int, Dict[str, Any]] = {}


def get_user_state(user_id: int) -> Dict[str, Any]:
    if user_id not in user_state:
        user_state[user_id] = {
            "model": DEFAULT_MODEL,
            "history": [],
        }
    return user_state[user_id]


# ─── Handlers ────────────────────────────────────────────────────────────────

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    state = get_user_state(user.id)
    await update.message.reply_text(
        f"👋 Namaste {user.first_name}!\n\n"
        f"Main ek free AI bot hoon powered by OllamaFreeAPI.\n\n"
        f"🤖 Current model: {MODELS.get(state['model'], state['model'])}\n\n"
        f"Commands:\n"
        f"/start - Bot restart karo\n"
        f"/model - Model change karo\n"
        f"/clear - Chat history clear karo\n"
        f"/help - Help dekho\n\n"
        f"Bas kuch bhi type karo aur main jawab dunga! 🚀",
    )


async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 AI Telegram Bot - Help\n\n"
        "Commands:\n"
        "• /start - Bot start karo\n"
        "• /model - AI model change karo\n"
        "• /clear - Conversation history clear karo\n\n"
        "Tips:\n"
        "• Seedha apna sawaal type karo\n"
        "• Bot conversation yaad rakhta hai (session me)\n"
        "• /clear se fresh start karo\n\n"
        "Powered by OllamaFreeAPI - No API key needed!",
    )


async def select_model(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton(label, callback_data=f"model:{key}")]
        for key, label in MODELS.items()
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    state = get_user_state(update.effective_user.id)
    current = MODELS.get(state["model"], state["model"])
    await update.message.reply_text(
        f"Model Select Karo\n\nAbhi: {current}\n\nNaya model chuno:",
        reply_markup=reply_markup,
    )


async def model_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    model_key = query.data.split(":", 1)[1]
    state = get_user_state(query.from_user.id)
    state["model"] = model_key
    state["history"] = []
    label = MODELS.get(model_key, model_key)
    await query.edit_message_text(
        f"✅ Model set: {label}\n\nHistory clear ho gayi. Ab baat karo!",
    )


async def clear_history(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    state = get_user_state(update.effective_user.id)
    state["history"] = []
    await update.message.reply_text("🗑️ Chat history clear ho gayi! Fresh start karo.")


async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text.strip()
    user_id = update.effective_user.id
    state = get_user_state(user_id)

    await ctx.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    history_text = ""
    for msg in state["history"][-6:]:
        history_text += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n\n"

    full_prompt = f"{history_text}User: {user_text}\nAssistant:"

    try:
        full_response = ""
        for chunk in client.stream_chat(state["model"], full_prompt):
            full_response += chunk

        if not full_response.strip():
            full_response = "Koi response nahi mila. /model se doosra model try karo."

        state["history"].append({"user": user_text, "assistant": full_response})

        if len(full_response) > 4000:
            for i in range(0, len(full_response), 4000):
                await update.message.reply_text(full_response[i: i + 4000])
        else:
            await update.message.reply_text(full_response)

    except Exception as e:
        logger.error(f"API Error: {e}", exc_info=True)
        await update.message.reply_text(
            f"⚠️ Error: {str(e)[:200]}\n\nDoosra model try karo: /model"
        )


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    if not BOT_TOKEN:
        raise ValueError("BOT_TOKEN set nahi hai!")

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("model", select_model))
    app.add_handler(CommandHandler("clear", clear_history))
    app.add_handler(CallbackQueryHandler(model_callback, pattern=r"^model:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    if WEBHOOK_URL:
        # ✅ Render pe webhook mode (free web service ke liye)
        logger.info(f"Webhook mode: {WEBHOOK_URL}")
        app.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            webhook_url=f"{WEBHOOK_URL}/webhook",
            url_path="/webhook",
        )
    else:
        # Local test ke liye polling mode
        logger.info("Polling mode (local)...")
        app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
