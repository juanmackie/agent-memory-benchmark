"""jev_proxy.py [port]: forwards decision-model requests to TypeSafe and logs each call's tokens.

The server does not meter its decision-model calls, so the benchmark points
HINDSIGHT_API_REFLECT_DECISION_MODEL_BASE_URL here and counts runs/hs/jev.jsonl per session window."""
import json
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

UPSTREAM = "https://api.typesafe.ai"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8897
LOG = Path(__file__).parent / f"jev-{PORT}.jsonl"  # one log per server lane


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        hdrs = {k: v for k, v in self.headers.items() if k.lower() in ("authorization", "content-type")}
        t0 = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(UPSTREAM + self.path, body, hdrs), timeout=120) as r:
                status, out = r.status, r.read()
        except urllib.error.HTTPError as e:
            status, out = e.code, e.read()
        try:
            req, usage = json.loads(body), (json.loads(out).get("usage") or {})
        except Exception:
            req, usage = {}, {}
        if "pick" in (req.get("questions") or {}):  # kept whole, to replay other wordings offline
            with (LOG.parent / "pick_bodies.jsonl").open("a") as f:
                f.write(json.dumps({"ts": t0, "body": req, "answer": json.loads(out).get("answers")}) + "\n")
        with LOG.open("a") as f:
            f.write(json.dumps({"ts": t0, "ms": int((time.time() - t0) * 1000), "status": status, "path": self.path,
                                "questions": list((req.get("questions") or {}).keys())[:5],
                                "n_questions": len(req.get("questions") or {}),
                                "input_tokens": usage.get("input_tokens", 0)}) + "\n")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
