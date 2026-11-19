"""Minimal local web chat – Python standard library only, no accounts or extra packages.
Run from the project root:   python -m chat.server      then open http://localhost:8000
"""
from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qsl, unquote, urlsplit

from . import admin, audit
from .config import CHAT_DIR, DECISION_SLA_S
from .graph import LoanChatbot

INDEX = CHAT_DIR / "static" / "index.html"
ADMIN = CHAT_DIR / "static" / "admin.html"
REPORT = CHAT_DIR / "static" / "report.html"
bot: LoanChatbot | None = None
_started: dict[str, float] = {}      # session_id -> wall-clock start (for the total-time readout)


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
        url = urlsplit(self.path)
        path = url.path.rstrip("/") or "/"
        try:
            if path == "/api/admin/applications":
                return self._json(200, admin.query(bot.collect, dict(parse_qsl(url.query))))
            if path.startswith("/api/admin/applications/"):
                d = admin.detail(bot.collect, unquote(path.rsplit("/", 1)[1]))
                return self._json(200, d) if d else self._json(404, {"error": "application not found"})
            if path == "/api/admin/report":
                return self._json(200, admin.report(bot.collect))
            if path.startswith("/reports/figures/"):
                fig = admin.figure_path(unquote(path.rsplit("/", 1)[1]))
                if fig is None:
                    return self._json(404, {"error": "not found"})
                return self._send(fig.read_bytes(), "image/png", cache=True)
        except Exception as e:
            return self._json(500, {"error": f"{type(e).__name__}: {e}"})
        page = {"/": INDEX, "/index.html": INDEX, "/admin": ADMIN, "/admin/report": REPORT}.get(path)
        if page is None:
            return self._json(404, {"error": "not found"})
        return self._send(page.read_bytes(), "text/html; charset=utf-8")

    def _send(self, body: bytes, ctype: str, cache: bool = False) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/api/session":
                sid, greeting = bot.new_session()
                _started[sid] = time.time()
                return self._json(200, {"session_id": sid, "message": greeting, **_status(sid)})
            if self.path == "/api/message":
                sid = data.get("session_id")
                if not sid or not bot.state(sid):
                    return self._json(400, {"error": "unknown session – start a new application"})
                was_done = bot.state(sid).get("stage") == "done"
                t0 = time.perf_counter()
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
                extra = {}
                if not was_done and bot.state(sid).get("stage") == "done":
                    # Final input -> decision, measured around the whole graph run on the server.
                    decision_ms = round((time.perf_counter() - t0) * 1000, 1)
                    total_s = round(time.time() - _started[sid], 1) if sid in _started else None
                    timing = {"decision_ms": decision_ms, "total_s": total_s, "sla_s": DECISION_SLA_S,
                              "within_sla": decision_ms <= DECISION_SLA_S * 1000,
                              "model_ms": (bot.state(sid).get("model_output") or {}).get("latency_ms")}
                    audit.log_event(sid, "decision_timing", **timing)
                    extra["timing"] = timing
                return self._json(200, {"message": msg, **_status(sid), **extra})
            return self._json(404, {"error": "not found"})
        except Exception as e:  # keep the server up; surface the error to the page
            return self._json(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):  # quiet access log (audit log covers the content)
        pass


def main() -> None:
    global bot
    bot = LoanChatbot()
    port = int(os.getenv("CHAT_PORT", "8000"))
    print(f"Loan chatbot on http://localhost:{port}  (model: {bot.model.name})   admin: http://localhost:{port}/admin")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
