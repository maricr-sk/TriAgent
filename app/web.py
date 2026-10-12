"""TriAgent web page: the same chat as triagent.py, in your browser.
Suggestions only, not medical advice.
"""
import json
import os
import secrets
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import triagent  # noqa: E402  (reuses the prompt, tools and triage rules)

PAGE = (Path(__file__).resolve().parent / "static" / "index.html").read_bytes()
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
MAX_MESSAGE_CHARS = 4000

SYSTEM_PROMPT = triagent.PROMPT_PATH.read_text()
TOOLS = triagent.load_tools()
_client = None
_sessions = {}  # session id -> {"messages": [...], "intake": {...}, "lock": Lock}
_sessions_lock = threading.Lock()


def client():
    global _client
    if _client is None:
        _client = triagent.anthropic.Anthropic()
    return _client


def get_session(session_id):
    with _sessions_lock:
        if session_id not in _sessions:
            _sessions[session_id] = {"messages": [], "intake": {}, "lock": threading.Lock()}
        return _sessions[session_id]


def take_turn(session, text):
    """One user message in; Angie's reply text and any answer chips out.
    Same loop as triagent.chat()."""
    messages, intake = session["messages"], session["intake"]
    triagent.add_user_text(messages, text)
    replies, choices = [], None
    while True:
        response = client().messages.create(
            model=triagent.MODEL, max_tokens=1500, system=SYSTEM_PROMPT,
            tools=TOOLS, messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})
        replies += [b.text for b in response.content if b.type == "text" and b.text.strip()]
        if response.stop_reason != "tool_use":
            break

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                output = triagent.run_tool(block.name, block.input, intake)
                is_error = "error" in output
            except Exception as error:  # report tool failures back to Claude
                output, is_error = {"error": str(error)}, True
            if block.name == "offer_choices":
                choices = {"question": str(block.input.get("question", "")).strip()[:200],
                           "options": [str(o)[:40] for o in block.input.get("options", [])][:6],
                           "multiple": bool(block.input.get("multiple"))}
            if is_error:  # e.g. a Maps API that isn't enabled; shows in the server log
                print(f"Tool {block.name} failed: {output['error']}")
            if os.environ.get("TRIAGENT_DEBUG"):
                print(f"  [{block.name}] {json.dumps(output)[:300]}")
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": json.dumps(output), "is_error": is_error})
        messages.append({"role": "user", "content": results})
        if triagent.only_choices(response):
            break  # the question and its chips are ready; wait for the person
    return "\n\n".join(replies), choices


class Handler(BaseHTTPRequestHandler):
    def send(self, status, body, content_type="application/json"):
        if isinstance(body, dict):
            body = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self.send(200, PAGE, "text/html; charset=utf-8")
        if self.path == "/api/config":
            return self.send(200, {"fake_clinics": not triagent.MAPS_KEY})
        self.send(404, {"error": "Not found"})

    def do_POST(self):
        if self.path == "/api/new":
            return self.send(200, {"session": secrets.token_urlsafe(16)})
        if self.path != "/api/chat":
            return self.send(404, {"error": "Not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length) or b"{}")
            session_id, text = str(data["session"]), str(data["message"]).strip()
        except (ValueError, KeyError):
            return self.send(400, {"error": "Bad request."})
        if not text or len(text) > MAX_MESSAGE_CHARS:
            return self.send(400, {"error": f"Messages must be 1 to {MAX_MESSAGE_CHARS} characters."})

        session = get_session(session_id)
        with session["lock"]:  # one turn at a time per conversation
            before = len(session["messages"])
            try:
                reply, choices = take_turn(session, text)
            except Exception as error:
                del session["messages"][before:]  # drop the half-finished turn
                print(f"Error: {error!r}")
                return self.send(502, {"error": "Angie couldn't reach Claude. "
                                       "Check ANTHROPIC_API_KEY and try again."})
        self.send(200, {"reply": reply, "choices": choices})

    def log_message(self, fmt, *args):  # keep the terminal quiet; no message text logged
        pass


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Note: ANTHROPIC_API_KEY is not set, so replies will fail until you set it.")
    if not triagent.MAPS_KEY:
        print("No MAPS_API_KEY set: using fake clinics.")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"TriAgent is running at http://localhost:{PORT}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
