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
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer

from telegram import Update
from telegram.constants import ParseMode
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

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"

DB_FILE = "memory.db"

if not TELEGRAM_BOT_TOKEN or not NVIDIA_API_KEY:
    raise SystemExit("Missing API keys")

conn = sqlite3.connect(DB_FILE, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS messages (
    chat_id TEXT,
    role TEXT,
    content TEXT,
    timestamp INTEGER
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS settings (
    chat_id TEXT PRIMARY KEY,
    model TEXT
)
""")

conn.commit()

def save_msg(chat_id, role, content):
    cursor.execute(
        "INSERT INTO messages VALUES (?, ?, ?, ?)",
        (str(chat_id), role, content, int(time.time()))
    )
    conn.commit()

def get_history(chat_id, limit=6):
    cursor.execute(
        "SELECT role, content FROM messages WHERE chat_id=? ORDER BY timestamp DESC LIMIT ?",
        (str(chat_id), limit * 2)
    )
    rows = cursor.fetchall()
    rows.reverse()
    return [{"role": r[0], "content": r[1]} for r in rows]

def clear_memory(chat_id):
    cursor.execute("DELETE FROM messages WHERE chat_id=?", (str(chat_id),))
    conn.commit()

def save_model(chat_id, model):
    cursor.execute(
        "INSERT OR REPLACE INTO settings (chat_id, model) VALUES (?, ?)",
        (str(chat_id), model)
    )
    conn.commit()

def get_saved_model(chat_id):
    cursor.execute(
        "SELECT model FROM settings WHERE chat_id=?",
        (str(chat_id),)
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else DEFAULT_MODEL

user_model = defaultdict(lambda: DEFAULT_MODEL)
user_lock = defaultdict(asyncio.Lock)
user_stop = set()

SYSTEM_PROMPT = """
You are a smart, calm, agentic AI assistant.

Core behavior:
Understand the user's latest message carefully.
Think about what the user actually wants.
Answer the exact request directly.
Do not answer the opposite of what the user asked.
Do not give generic advice when the user gives code or asks for a fix.
If the user asks for full code, provide full working code.
If the user asks for only code, provide only code.
Preserve the user's original intention.

Language behavior:
Always reply in the same language and typing style as the user's latest message.
If the user writes English, reply in English.
If the user writes Hindi, reply in Hindi.
If the user writes Hinglish, reply in Hinglish.
If the user mixes Hindi and English, reply in natural Hinglish.
Do not force Hindi.
Do not force English.
Match the user's tone naturally.

Tone:
Be practical, direct, and human-like.
Do not sound robotic.
Do not over-explain unless needed.
Do not lecture the user for casual slang, anger, or frustration.
Stay calm and keep helping.
Set a short boundary only for serious threats, hate, or harmful requests.
Avoid fake assistant lines.
Avoid excessive emojis.

Formatting:
Use **bold** for important words when useful.
Use `inline code` for short commands, filenames, variables, model names, errors, and APIs.
Use triple backtick fenced code blocks for full code, terminal commands, JSON, Python, Bash, JavaScript, HTML, CSS, etc.

Correct:
```python
print("hello")
