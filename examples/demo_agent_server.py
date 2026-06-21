"""A toy banking agent — a REAL, separate HTTP service, not mock code inside the Studio.

Implements the contract core.HttpAgentModel expects:
    POST /  {"prompt": "<text>", "tools": [<tool schemas>]}
    ->     {"text": "<reply>", "tool_calls": [{"name": "...", "arguments": {...}}, ...]}

It has ONE deliberate, realistic bug: when asked to transfer money, it fires
transfer_funds immediately, without ever calling get_balance first to check
the account actually holds that much. A real agent should verify the
precondition before acting — this one doesn't, which is exactly the failure
mode the Studio's Agent-loop check ("check-before-transfer") is built to catch.

Run it:
    python examples/demo_agent_server.py [port]      # default port 8800

Then in the Studio sidebar:
    Backend: "Your deployed agent (HTTP)"
    Agent endpoint URL: http://127.0.0.1:8800/
    Tick "Allow private / localhost addresses" (it's a local test agent)

IMPORTANT — this only works if the Studio is running on the SAME machine:
  - Run locally (`streamlit run app.py`) alongside this server: works, both
    are on your machine's localhost.
  - The hosted Streamlit Cloud app CANNOT reach your machine's localhost —
    "127.0.0.1" from its container means itself, not your PC. To test this
    server from the hosted app, expose it publicly first (e.g. `ngrok http
    8800`) and use the ngrok URL instead.

Standard library only — no extra dependencies, consistent with the rest of
this project (see prompt_regression.models.HttpModel).
"""

from __future__ import annotations

import json
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

_MAX_BODY_BYTES = 1_000_000


def decide(prompt: str, tools: list[dict]) -> tuple[str, list[dict]]:
    """The agent's one decision per request — this is the whole 'brain.'

    Reads whichever tool names were offered, so it plays along with any
    toolset that includes get_balance / transfer_funds (the Studio's
    built-in banking scenarios). Real money never moves; this is a
    standalone process with no real backend behind it.
    """
    names = {t.get("name") for t in tools}
    q = prompt.lower()
    accounts = re.findall(r"account\s+(\d+)", q)

    if "transfer_funds" in names and re.search(r"transfer|send|move|pay", q):
        frm = accounts[0] if accounts else ""
        to = accounts[1] if len(accounts) > 1 else ""
        amount_m = re.search(r"\$\s*([\d,.]+)", prompt)
        amount = float(amount_m.group(1).replace(",", "")) if amount_m else 0.0
        # BUG (deliberate): a safe agent checks get_balance before transferring.
        # This one doesn't — it just does it.
        return "Transfer complete.", [
            {"name": "transfer_funds",
             "arguments": {"from_account": frm, "to_account": to, "amount": amount}}
        ]

    if "get_balance" in names and re.search(r"balance|how much.*have", q):
        acct = accounts[0] if accounts else ""
        return f"Checking the balance of account {acct}.", [
            {"name": "get_balance", "arguments": {"account_id": acct}}
        ]

    return "I'm not sure which action to take here.", []


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = min(int(self.headers.get("Content-Length", 0) or 0), _MAX_BODY_BYTES)
        try:
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw or b"{}")
            text, calls = decide(str(body.get("prompt", "")), list(body.get("tools") or []))
            self._send(200, {"text": text, "tool_calls": calls})
        except json.JSONDecodeError as exc:
            self._send(400, {"error": f"invalid JSON: {exc}"})
        except Exception as exc:                              # noqa: BLE001 - demo server
            self._send(500, {"error": str(exc)})

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:   # quiet by default
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8800
    print(f"Toy banking agent listening on http://127.0.0.1:{port}/  (Ctrl+C to stop)")
    print("Point the Studio's 'Your deployed agent (HTTP)' backend at this URL to test it.")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
