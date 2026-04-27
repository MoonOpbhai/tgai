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

user_model = defaultdict(lambda: DEFAULT_MODEL)
user_lock = defaultdict(asyncio.Lock)
user_stop = set()

SYSTEM_PROMPT = """
You are a smart, calm, human-like AI agent.

Your job:
Understand the user's message carefully, think about what they actually want, then answer directly.

Language style:
Always reply in the same language and typing style as the user's latest message.
If the user writes English, reply in English.
If the user writes Hindi, reply in Hindi.
If the user writes Hinglish, reply in Hinglish.
Do not force Hindi.
Do not force English.

Tone:
Be natural, practical, and direct.
Do not sound robotic.
Do not over-explain unless needed.
Do not lecture the user for casual slang, anger, or frustration.
Stay calm and continue helping.
Set a short boundary only for serious threats or harmful requests.

Formatting:
Use **bold** for important words when useful.
Use `inline code` for short commands, filenames, variables, model names, and API names.
Use triple backtick code blocks for full code or terminal commands.
Do not use ### headings unless the user specifically asks for long formatted documentation.
Avoid excessive emojis.
Avoid decorative lines.
Avoid fake friendliness.
Do not say "Real Talk" or "Ab kya karna hai?" randomly.

Technical answers:
For VPS, coding, APIs, bots, and errors, give exact working commands or code first.
Preserve the user's original intention.
If code is requested, provide full working code.
If the user asks to fix code, fix the actual issue instead of giving generic advice.
"""

def detect_language_instruction(text):
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", text))
    english_letters = len(re.findall(r"[A-Za-z]", text))
    hindi_words = len(re.findall(
        r"\b(kya|hai|nhi|nahin|kaise|kese|kar|karo|kr|mujhe|tum|aap|bhai|bata|bolo|hona|chahiye|thek|sahi|galat|code|vps)\b",
        text.lower()
    ))

    if has_devanagari:
        return "The user's latest message is Hindi/Devanagari. Reply in Hindi or natural Hinglish matching them."
    if hindi_words >= 2:
        return "The user's latest message is Hinglish. Reply in natural Hinglish matching their style."
    if english_letters > 0:
        return "The user's latest message is English or mostly English. Reply in English."
    return "Reply in the same language style as the user's latest message."

def apply_inline_markdown(segment):
    segment = html.escape(segment)

    segment = re.sub(
        r"`([^`\n]+)`",
        lambda m: f"<code>{m.group(1)}</code>",
        segment
    )

    segment = re.sub(
        r"\*\*(.+?)\*\*",
        lambda m: f"<b>{m.group(1)}</b>",
        segment,
        flags=re.DOTALL
    )

    return segment

def markdown_to_telegram_html(text):
    if not text:
        return ""

    text = text.replace("\r\n", "\n")

    parts = []
    pos = 0

    pattern = re.compile(r"```([a-zA-Z0-9_+\-.]*)?\n?(.*?)```", re.DOTALL)

    for match in pattern.finditer(text):
        before = text[pos:match.start()]
        if before:
            parts.append(apply_inline_markdown(before))

        code = match.group(2).strip("\n")
        code = html.escape(code)
        parts.append(f"<pre><code>{code}</code></pre>")
        pos = match.end()

    rest = text[pos:]
    if rest:
        parts.append(apply_inline_markdown(rest))

    out = "".join(parts)

    out = out.replace("###", "")
    out = re.sub(r"\n{4,}", "\n\n\n", out)
    out = out.strip()

    if not out:
        out = html.escape(text.strip())

    return out

def plain_cleanup(text):
    if not text:
        return ""

    text = text.replace("###", "")
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()

def safe_raw_tail(text, limit=3400):
    text = text or ""
    if len(text) <= limit:
        return text
    return text[-limit:]

def stream_ai(messages, model):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.45,
        "top_p": 0.9,
        "max_tokens": 1200,
        "stream": True,
    }

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        with requests.post(
            API_URL,
            json=payload,
            headers=headers,
            stream=True,
            timeout=120
        ) as r:

            if r.status_code != 200:
                yield f"API Error {r.status_code}: {r.text[:700]}"
                return

            for line in r.iter_lines():
                if not line:
                    continue

                line = line.decode("utf-8", errors="ignore").strip()

                if not line.startswith("data:"):
                    continue

                line = line[5:].strip()

                if line == "[DONE]":
                    break

                try:
                    data = json.loads(line)
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except Exception:
                    continue

    except Exception as e:
        yield f"Stream Error: {e}"

def build_messages(chat_id, text):
    history = get_history(chat_id)

    current_rule = {
        "role": "system",
        "content": detect_language_instruction(text)
    }

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        current_rule,
    ] + history + [
        {"role": "user", "content": text}
    ]

async def edit_telegram_text(message_obj, raw_text):
    raw_text = plain_cleanup(raw_text)
    raw_text = safe_raw_tail(raw_text)
    html_text = markdown_to_telegram_html(raw_text)

    try:
        await message_obj.edit_text(
            html_text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True
        )
    except Exception:
        try:
            await message_obj.edit_text(
                plain_cleanup(raw_text).replace("**", "").replace("`", "")[:4000],
                disable_web_page_preview=True
            )
        except Exception:
            pass

async def send_telegram_text(update, text):
    raw = plain_cleanup(text)
    html_text = markdown_to_telegram_html(raw)

    try:
        return await update.message.reply_text(
            html_text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True
        )
    except Exception:
        return await update.message.reply_text(
            raw.replace("**", "").replace("`", "")[:4000],
            disable_web_page_preview=True
        )

async def smooth_stream(message_obj, generator, chat_id):
    buffer = ""
    last_update = 0
    update_delay = 0.45

    for chunk in generator:
        if chat_id in user_stop:
            user_stop.discard(chat_id)
            await edit_telegram_text(message_obj, "Stopped.")
            return None

        buffer += chunk

        if time.time() - last_update > update_delay:
            await edit_telegram_text(message_obj, buffer)
            last_update = time.time()

    final = plain_cleanup(buffer)
    await edit_telegram_text(message_obj, final)
    return final

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_telegram_text(update, "**Bot online.** Message bhejo.")

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_stop.add(update.effective_chat.id)
    await send_telegram_text(update, "Stopping...")

async def clearmem(update: Update, context: ContextTypes.DEFAULT_TYPE):
    clear_memory(update.effective_chat.id)
    await send_telegram_text(update, "**Memory cleared.**")

async def setmodel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await send_telegram_text(update, "Usage: `/setmodel model_name`")
        return

    model = " ".join(context.args).strip()
    user_model[update.effective_chat.id] = model
    await send_telegram_text(update, f"Model set:\n`{model}`")

async def modelcmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    model = user_model[update.effective_chat.id]
    await send_telegram_text(update, f"Current model:\n`{model}`")

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = (update.message.text or "").strip()

    if not text:
        return

    if user_lock[chat_id].locked():
        await send_telegram_text(update, "Wait, pehle wala response complete hone do.")
        return

    async with user_lock[chat_id]:
        sent = await update.message.reply_text("Thinking...")

        model = user_model[chat_id]
        messages = build_messages(chat_id, text)

        gen = stream_ai(messages, model)
        final = await smooth_stream(sent, gen, chat_id)

        if not final:
            final = "Empty response mila."

        final = plain_cleanup(final)

        await edit_telegram_text(sent, final)

        save_msg(chat_id, "user", text)
        save_msg(chat_id, "assistant", final)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, *args):
        pass

def run_web():
    port = int(os.environ.get("PORT", 10000))
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()

async def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stop", stop))
    app.add_handler(CommandHandler("clearmem", clearmem))
    app.add_handler(CommandHandler("setmodel", setmodel))
    app.add_handler(CommandHandler("model", modelcmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)

    logger.info("Bot running...")
    await asyncio.Event().wait()

if __name__ == "__main__":
    threading.Thread(target=run_web, daemon=True).start()
    asyncio.run(main())
