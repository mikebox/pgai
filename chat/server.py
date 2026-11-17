"""Minimal local web chat – Python standard library only, no accounts or extra packages.
Run from the project root:   python -m chat.server      then open http://localhost:8000
"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .config import CHAT_DIR
from .graph import LoanChatbot

INDEX = CHAT_DIR / "static" / "index.html"
bot: LoanChatbot | None = None


def _status(sid: str) -> dict:
    return {**bot.progress(sid), "placeholder": bot.model.name == "placeholder"}


class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path not in ("/", "/index.html"):
            return self._json(404, {"error": "not found"})
        body = INDEX.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/api/session":
                sid, greeting = bot.new_session()
                return self._json(200, {"session_id": sid, "message": greeting, **_status(sid)})
            if self.path == "/api/message":
                sid = data.get("session_id")
                if not sid or not bot.state(sid):
                    return self._json(400, {"error": "unknown session – start a new application"})
                if data.get("action") == "submit":
                    msg = bot.send(sid, action="submit")
                elif isinstance(data.get("form"), dict):
                    form = {str(k): str(v)[:200] for k, v in data["form"].items()}
                    msg = bot.send(sid, form=form)
                else:
                    text = str(data.get("text", "")).strip()[:2000]
                    if not text:
                        return self._json(400, {"error": "send text, a form, or action=submit"})
                    msg = bot.send(sid, text)
                return self._json(200, {"message": msg, **_status(sid)})
            return self._json(404, {"error": "not found"})
        except Exception as e:  # keep the server up; surface the error to the page
            return self._json(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):  # quiet access log (audit log covers the content)
        pass


def main() -> None:
    global bot
    bot = LoanChatbot()
    port = int(os.getenv("CHAT_PORT", "8000"))
    print(f"Loan chatbot on http://localhost:{port}  (model: {bot.model.name})")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
