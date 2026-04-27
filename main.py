import os
import asyncio
import threading
import time
import json
import requests
import logging
import sqlite3
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# ───────────────── LOG ───────────────── #

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("AgentBot")

# ───────────────── CONFIG ───────────────── #

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API_KEY = os.getenv("NVIDIA_API_KEY")

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "z-ai/glm-5.1"

# ───────────────── MEMORY ───────────────── #

conn = sqlite3.connect("memory.db", check_same_thread=False)
cur = conn.cursor()

cur.execute("""CREATE TABLE IF NOT EXISTS mem(
chat_id TEXT, role TEXT, content TEXT, ts INT
)""")
conn.commit()

def save(cid, role, content):
    cur.execute("INSERT INTO mem VALUES(?,?,?,?)",
                (str(cid), role, content, int(time.time())))
    conn.commit()

def history(cid):
    cur.execute("SELECT role,content FROM mem WHERE chat_id=? ORDER BY ts DESC LIMIT 10", (str(cid),))
    rows = cur.fetchall()
    rows.reverse()
    return [{"role": r[0], "content": r[1]} for r in rows]

# ───────────────── TOOL SYSTEM ───────────────── #

def web_search(query):
    # simple placeholder (you can upgrade later with real API)
    return f"[WEB SEARCH RESULT for: {query}]"

def run_python(code):
    try:
        # SAFE limited execution
        local = {}
        exec(code[:1000], {"__builtins__": {}}, local)
        return str(local)
    except Exception as e:
        return str(e)

def tool_router(text):
    text_lower = text.lower()

    if text_lower.startswith("search:"):
        return web_search(text[7:].strip())

    if text_lower.startswith("python:"):
        return run_python(text[7:].strip())

    return None

# ───────────────── STREAM FIXED ───────────────── #

def stream(messages):
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "temperature": 0.7
    }

    headers = {"Authorization": f"Bearer {API_KEY}"}

    with requests.post(API_URL, json=payload, headers=headers, stream=True) as r:
        buffer = ""

        for chunk in r.iter_content(decode_unicode=True):

            if not chunk:
                continue

            buffer += chunk

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                if "data:" in line:
                    line = line.replace("data:", "").strip()

                if line == "[DONE]":
                    return

                try:
                    data = json.loads(line)
                    delta = data["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except:
                    continue

# ───────────────── HANDLER ───────────────── #

lock = defaultdict(asyncio.Lock)
stop_flag = set()

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    text = update.message.text

    # TOOL FIRST CHECK
    tool_result = tool_router(text)
    if tool_result:
        await update.message.reply_text(tool_result)
        return

    if lock[cid].locked():
        await update.message.reply_text("⏳ Busy...")
        return

    async with lock[cid]:

        msg = await update.message.reply_text("⚡ thinking...")

        msgs = [{"role": "system", "content": "You are a helpful assistant."}]
        msgs += history(cid)
        msgs += [{"role": "user", "content": text}]

        buffer = ""
        last = time.time()

        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(None, lambda: list(stream(msgs)))

        for c in chunks:

            if cid in stop_flag:
                stop_flag.remove(cid)
                await msg.edit_text("🛑 stopped")
                return

            buffer += c

            # 💥 SMOOTH TYPING EFFECT
            if time.time() - last > 0.2:
                try:
                    await msg.edit_text(buffer[-4000:])
                except:
                    pass
                last = time.time()

        await msg.edit_text(buffer[:4000])

        save(cid, "user", text)
        save(cid, "assistant", buffer)

# ───────────────── COMMANDS ───────────────── #

async def stop(update, context):
    stop_flag.add(update.effective_chat.id)
    await update.message.reply_text("🛑 stopping...")

# ───────────────── WEB ───────────────── #

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

def run():
    HTTPServer(("0.0.0.0", int(os.getenv("PORT", 10000))), H).serve_forever()

# ───────────────── MAIN ───────────────── #

async def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))
    app.add_handler(CommandHandler("stop", stop))

    await app.initialize()
    await app.start()
    await app.updater.start_polling()

    await asyncio.Event().wait()

if __name__ == "__main__":
    threading.Thread(target=run, daemon=True).start()
    asyncio.run(main())
