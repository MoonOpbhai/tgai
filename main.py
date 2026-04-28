import os
import asyncio
import threading
import time
import json
import requests
import logging
import sqlite3
import re
import html
import base64
from io import BytesIO
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer

from telegram import Update
from telegram.constants import ParseMode, ChatAction
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
OWNER_ID = int(os.getenv("OWNER_ID", "0").strip() or "0")

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"

DB_FILE = "memory.db"
MAX_CONTEXT_MESSAGES = 100

SKILL_CACHE = {}
chat_skill = {}

# ---------------- DB ----------------
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""CREATE TABLE IF NOT EXISTS messages (
chat_id TEXT, role TEXT, content TEXT, timestamp INTEGER)""")

cursor.execute("""CREATE TABLE IF NOT EXISTS settings (
chat_id TEXT PRIMARY KEY, model TEXT)""")

cursor.execute("""CREATE TABLE IF NOT EXISTS approved_users (
user_id TEXT PRIMARY KEY, approved_by TEXT, timestamp INTEGER)""")

conn.commit()

# ---------------- IMAGE HELP ----------------
def image_to_base64(image_bytes: bytes):
    return base64.b64encode(image_bytes).decode()

async def get_telegram_image(update: Update):
    if not update.message.photo:
        return None

    photo = update.message.photo[-1]
    file = await photo.get_file()
    bio = BytesIO()
    await file.download_to_memory(out=bio)
    return bio.getvalue()

# ---------------- MEMORY ----------------
def save_msg(chat_id, role, content):
    cursor.execute("INSERT INTO messages VALUES (?, ?, ?, ?)",
                   (str(chat_id), role, content, int(time.time())))
    conn.commit()

def get_history(chat_id):
    cursor.execute("SELECT role, content FROM messages WHERE chat_id=? ORDER BY timestamp ASC",
                   (str(chat_id),))
    return [{"role": r[0], "content": r[1]} for r in cursor.fetchall()]

def save_model(chat_id, model):
    cursor.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (str(chat_id), model))
    conn.commit()

def get_model(chat_id):
    cursor.execute("SELECT model FROM settings WHERE chat_id=?", (str(chat_id),))
    row = cursor.fetchone()
    return row[0] if row else DEFAULT_MODEL

# ---------------- APPROVAL ----------------
def is_owner(uid): return uid == OWNER_ID

def is_approved(uid):
    if is_owner(uid): return True
    cursor.execute("SELECT 1 FROM approved_users WHERE user_id=?", (str(uid),))
    return cursor.fetchone() is not None

# ---------------- AI CALL ----------------
def call_ai(messages, model):
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 1500
    }

    r = requests.post(API_URL, json=payload, headers=headers, timeout=120)
    if r.status_code == 200:
        return r.json()["choices"][0]["message"]["content"]
    return f"Error: {r.text}"

# ---------------- BUILD MESSAGE ----------------
def build_messages(chat_id, text, image_b64=None):
    msgs = [
        {"role": "system", "content": "You are a smart assistant."}
    ]

    history = get_history(chat_id)
    msgs.extend(history)

    if image_b64:
        msgs.append({
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        })
    else:
        msgs.append({"role": "user", "content": text})

    return msgs

# ---------------- HANDLER ----------------
async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    if not is_approved(user_id):
        await update.message.reply_text("Access denied")
        return

    text = update.message.text or ""

    img_bytes = await get_telegram_image(update)
    img_b64 = image_to_base64(img_bytes) if img_bytes else None

    model = get_model(chat_id)
    messages = build_messages(chat_id, text, img_b64)

    sent = await update.message.reply_text("Thinking...")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, call_ai, messages, model)

    await sent.edit_text(result[:4000])

    save_msg(chat_id, "user", text)
    save_msg(chat_id, "assistant", result)

# ---------------- COMMANDS ----------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Bot ready 🚀")

# ---------------- MAIN ----------------
async def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT | filters.PHOTO, handle))

    await app.initialize()
    await app.start()
    await app.updater.start_polling()

    await asyncio.Event().wait()

if __name__ == "__main__":
    threading.Thread(target=lambda: HTTPServer(("0.0.0.0", 10000), BaseHTTPRequestHandler).serve_forever(), daemon=True).start()
    asyncio.run(main())
