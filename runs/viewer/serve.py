"""Session viewer: what Claude did and what the memory watcher did, on one timeline.

    python runs/viewer/serve.py [--port 8765]   then open http://localhost:8765

Reads /tmp/sdebench/run/<run>/: events.jsonl (agent + harness, stamped live), watcher.jsonl
(the watcher's own log) and trace.json. Runs from before events.jsonl existed get an
approximate agent timeline rebuilt from trace.json (tools spread evenly over each round).
"""

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

RUNS = Path("/tmp/sdebench/run")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sdebench" / "watcher"))
import watcher as W  # noqa: E402  the watcher's own step reading, so the tags match what it does
HERE = Path(__file__).parent


def jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass  # a line still being written
    return out


def agent_events(d: Path) -> tuple[list[dict], bool]:
    """(events, exact). Tool results are folded into their tool call."""
    raw = jsonl(d / "events.jsonl")
    if raw:
        out, by_id, tools = [], {}, {}
        for e in raw:
            if e.get("kind") != "call":
                _, trigger, _ = W.read_step(e, tools)
                if e.get("kind") == "result" and e.get("id") in by_id:
                    by_id[e["id"]]["watch"] = trigger or by_id[e["id"]].get("watch")
                else:
                    e["watch"] = trigger or "context"
            if e.get("kind") == "result" and e.get("id") in by_id:
                t = by_id[e["id"]]
                t.update(result=e.get("text", ""), error=e.get("error", False), end=e["ts"])
                continue
            ev = {"ts": e["ts"], "lane": {"harness": "harness", "user": "harness", "watcher": "watcher"}.get(e.get("src"), "agent"), **e}
            if e.get("kind") == "tool":
                by_id[e.get("id")] = ev
            out.append(ev)
        return out, True
    return approx_from_trace(d), False


def approx_from_trace(d: Path) -> list[dict]:
    tp = d / "trace.json"
    if not tp.exists():
        return []
    tr = json.loads(tp.read_text())
    w = jsonl(d / "watcher.jsonl")
    t = w[0]["ts"] if w else tp.stat().st_mtime - tr.get("wall_s", 0)
    out = []
    for r in tr["trace"]:
        out.append({"ts": t, "lane": "harness", "kind": "round", "role": r["role"], "text": r.get("prompt", "")})
        steps = r.get("trajectory") or []
        dur = r.get("wall_s") or 1
        for i, s in enumerate(steps):
            ts = t + dur * (i + 0.5) / max(1, len(steps))
            if s.get("k") == "tool":
                out.append({"ts": ts, "lane": "agent", "kind": "tool", "role": r["role"], "tool": s.get("tool"),
                            "input": json.dumps({"arg": s.get("arg", "")})})
            elif s.get("k") == "say":
                out.append({"ts": ts, "lane": "agent", "kind": "say", "role": r["role"], "text": s.get("text", "")})
        t += dur
        if "grade_passed" in r:
            out.append({"ts": t, "lane": "harness", "kind": "grade", "role": r["role"],
                        "passed": r["grade_passed"], "text": r.get("grade_pytest", "")})
            t += 5  # grading time is not recorded in old traces
    return out


def run_view(name: str) -> dict:
    d = RUNS / name
    events, exact = agent_events(d)
    for e in jsonl(d / "watcher.jsonl"):
        k = e.get("event")
        ev = {"ts": e["ts"], "lane": "watcher", "kind": k, **e}
        if k == "call":
            ev["end"] = e["ts"] + e.get("ms", 0) / 1000
        if k == "review":  # logged when the review ENDS; show it from when it started
            ev["end"] = e["ts"]
            ev["ts"] = e["ts"] - (e.get("recall_ms", 0) + e.get("ms", 0)) / 1000
        events.append(ev)
    events.sort(key=lambda e: e["ts"])
    meta = {}
    if (d / "result.json").exists():
        r = json.loads((d / "result.json").read_text())
        meta = {k: r.get(k) for k in ("solved", "interventions", "cost_usd", "wall_s", "model")}
    task = {}
    if (d / "trace.json").exists():
        tr = json.loads((d / "trace.json").read_text())
        task = {"bug_report": tr.get("bug_report", "")}
    newest = max((p.stat().st_mtime for p in d.glob("*.json*")), default=0)
    return {"name": name, "exact": exact, "live": time.time() - newest < 60 and not meta,
            "meta": meta, "task": task, "events": events}


def list_runs() -> list[dict]:
    out = []
    for d in RUNS.iterdir() if RUNS.exists() else []:
        if not ((d / "events.jsonl").exists() or (d / "watcher.jsonl").exists()):
            continue
        m = max((p.stat().st_mtime for p in d.glob("*.json*")), default=0)
        r = json.loads((d / "result.json").read_text()) if (d / "result.json").exists() else None
        out.append({"name": d.name, "mtime": m, "exact": (d / "events.jsonl").exists(),
                    "live": r is None and time.time() - m < 60,
                    "solved": r and r.get("solved"), "interventions": r and r.get("interventions")})
    return sorted(out, key=lambda r: -r["mtime"])


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        path = unquote(self.path.split("?")[0])
        if path == "/":
            body, ctype = (HERE / "index.html").read_bytes(), "text/html; charset=utf-8"
        elif path == "/api/runs":
            body, ctype = json.dumps(list_runs()).encode(), "application/json"
        elif path.startswith("/api/run/") and "/" not in path[9:] and (RUNS / path[9:]).is_dir():
            body, ctype = json.dumps(run_view(path[9:])).encode(), "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    port = ap.parse_args().port
    print(f"session viewer on http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
