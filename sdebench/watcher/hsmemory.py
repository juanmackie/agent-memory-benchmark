"""The watcher's memory on Hindsight: observations only.

One bank per project. Past chats and graded-session feedback are retained as plain memories, each
with its time, and Hindsight's consolidation turns them into observations: one current belief per
topic, rewritten when a newer session changes its mind (so a superseded rule does not come back).
The watcher recalls observations only; the source document is fetched for the verbatim quote.

    ingest(bank, task, repo)   past chats in session order, then wait for consolidation
    recall(bank, diff)         -> observation Notes for the judge (proof = newest source text)
    learn(bank, task, trace)   grader feedback retained back; consolidation folds it in
"""

from __future__ import annotations

import functools
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import watcher as W

HS = os.environ.get("WATCHER_HS_URL", "http://localhost:8899")
MISSION = ("Track what this project decided about its code: exact values, rules, exceptions, and options "
           "tried and rejected (with why). When a later session changes a decision, the observation must "
           "state only the current decision and say what it replaced.")


def _req(method: str, path: str, body: dict | None = None, timeout: float = 120, quiet: bool = False) -> dict:
    t0 = time.time()
    req = urllib.request.Request(f"{HS}/v1/default/banks/{path}", method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    if not quiet:
        W.CALLS.append({"ts": t0, "service": "hindsight", "name": f"{method} /{path.split('/', 1)[-1]}",
                        "bank": path.split("/", 1)[0], "request": json.dumps(body, indent=1)[:12000] if body else "",
                        "response": json.dumps(out, indent=1)[:16000], "ms": int((time.time() - t0) * 1000)})
    return out


def setup_bank(bank: str) -> None:
    _req("PATCH", f"{bank}/config", {"updates": {
        "enable_observations": True,
        "observations_mission": MISSION,
        "enable_graph_retrieval": False,
        "enable_temporal_retrieval": False,
        "enable_reranking": False,  # 3.3s -> 60ms per recall on this box; the judge does the precision
    }})


def retain(bank: str, items: list[dict]) -> None:
    _req("POST", f"{bank}/memories", {"items": items, "async": False}, timeout=600)


def wait_consolidated(bank: str, timeout: float = 900) -> None:
    """Retain is sync, consolidation runs after it in the background: wait until nothing is queued."""
    t0 = time.time()
    deadline = t0 + timeout
    while time.time() < deadline:
        busy = sum(len(_req("GET", f"{bank}/operations?status={st}&limit=1", quiet=True).get("operations") or [])
                   for st in ("pending", "processing"))
        if not busy:
            W.CALLS.append({"ts": t0, "service": "hindsight", "name": "consolidation (waited for observations)",
                            "bank": bank, "request": "", "response": "", "ms": int((time.time() - t0) * 1000)})
            return
        time.sleep(3)
    raise TimeoutError(f"consolidation of {bank} still running after {timeout}s")


def _item(kind: str, text: str, where: str, doc: str, when: str) -> dict:
    return {"content": text, "document_id": doc, "timestamp": when, "tags": [kind],
            "metadata": {"kind": kind, "where": where}}


def _session_time(i: int) -> str:
    """Chats carry no dates; their order is the history, so session N is day N."""
    return f"2026-01-{i + 1:02d}T12:00:00Z"


# ---------------------------------------------------------------- ingest / recall / learn


def ingest(bank: str, task: dict, repo: Path) -> dict:
    setup_bank(bank)
    chats = W.load_chats(task)
    try:
        _req("GET", f"{bank}/documents/chat-0", quiet=True)  # a reused bank already has the chats (and what it learned)
    except urllib.error.HTTPError:
        retain(bank, [_item("chat", n.text, n.where, f"chat-{i}", _session_time(i)) for i, n in enumerate(chats)])
    wait_consolidated(bank)
    obs = _req("GET", f"{bank}/memories/list?type=observation&limit=1").get("total", "?")
    return {"chats": len(chats), "observations": obs}


def query_from_diff(diff: str, cap: int = 1500) -> str:
    """A recall query in words: the files and functions touched, then the changed lines."""
    files, funcs, lines = [], [], []
    for l in diff.splitlines():
        if l.startswith("+++ b/"):
            files.append(l[6:])
        elif l.startswith("@@"):
            funcs.append(l.split("@@")[-1].strip())
        elif l[:1] in "+-" and not l.startswith(("+++", "---")) and l[1:].strip():
            lines.append(l.strip())
    q = f"Change to {', '.join(files)} ({'; '.join(f for f in funcs if f)}):\n" + "\n".join(lines)
    return q[:cap]


def where_for(doc: str, md: dict) -> str:
    """How the agent can open a source: our own items say so; the coding-agents plugin's ids
    (chat:session-NN, git:<sha>, gitlog:repo) map to the transcript file or a git command."""
    if md.get("where"):
        return md["where"]
    if doc.startswith("chat:session-"):
        return f"/root/project-history/{doc[5:]}.md"
    if doc.startswith("git:"):
        return f"git show {doc[4:14]}"
    if doc.startswith("gitlog:"):
        return "git log"
    return "project memory"


def _by_subject(where: str, text: str) -> str:
    """A commit by its message, not its hash: the repo the agent has can be a rebuild with other hashes
    (claude could not find the cited sha and called the note fabricated)."""
    m = re.search(r"Message:\n(.+)", text) if where.startswith("git show") else None
    return f'git log -1 -p --fixed-strings --grep "{m.group(1).strip()[:80].split(chr(34))[0].strip()}"' if m else where


@functools.lru_cache(maxsize=256)
def _doc_text(bank: str, doc: str) -> str:
    return _req("GET", f"{bank}/documents/{doc}").get("original_text") or ""


def moment_query(m: "W.Moment") -> str:
    """Search with what the agent just did: the diff in words for an edit, else its own words."""
    return query_from_diff(m.diff) if m.trigger == "edit" else m.text[:1500]


def search(bank: str, m: "W.Moment") -> list[tuple[float, W.Note]]:
    """What the watcher recalls for a moment. A request is searched paragraph by paragraph: as one query,
    the harness's boilerplate (role, rules, "work efficiently") outweighed the actual request and
    recalled unrelated memories, so the task-start heads-up never saw the decision."""
    if m.trigger != "task":
        return recall(bank, moment_query(m))
    paras = [p.strip() for p in re.split(r"\n\s*\n", m.text) if len(p.strip()) >= 40][:6]
    # a sentence naming code ("Fix append_event so ...") is the sharpest query; ranked ahead on ties
    named = [x.strip() for x in re.split(r"(?<=[.!?])\s+", m.text)
             if re.search(r"\b[a-z]+_[a-z_]+\b|\w\(\)|`[^`]+`", x) and "test" not in x.lower()][:4]
    best: dict[str, tuple[float, int, W.Note]] = {}
    for prio, q in [(0, x) for x in named] + [(1, x) for x in paras]:
        for score, n in recall(bank, q[:1500]):
            if n.id not in best or (score, -prio) > (best[n.id][0], -best[n.id][1]):
                best[n.id] = (score, prio, n)
    # every query's top 4, not a merged top 6: the decision sat 3rd for the bug report and was cut
    return [(s, n) for s, _, n in sorted(best.values(), key=lambda x: (-x[0], x[1]))[:10]]


def recall(bank: str, query: str, k: int = 4) -> list[tuple[float, W.Note]]:
    """Observations only. Each Note is the current belief plus its source documents (newest first); the
    judge's quote must come from one of them, which is what the agent can open and read."""
    r = _req("POST", f"{bank}/memories/recall",
             {"query": query, "types": ["observation"], "budget": "low", "max_tokens": 4000,
              "include": {"entities": None, "source_facts": {}}}, timeout=30)
    facts = r.get("source_facts") or {}
    out = []
    for i, x in enumerate((r.get("results") or [])[:k]):
        src = [facts[f] for f in x.get("source_fact_ids") or [] if f in facts and facts[f].get("document_id")]
        if not src:
            continue
        src.sort(key=lambda f: f.get("mentioned_at") or f.get("occurred_start") or "", reverse=True)
        docs = {}  # newest first, one entry per source document
        for f in src:
            docs.setdefault(f["document_id"], where_for(f["document_id"], f.get("metadata") or {}))
        sources = tuple((_by_subject(where, t), t) for where, t in ((w, _doc_text(bank, d)) for d, w in docs.items()))
        text = f"CURRENT BELIEF (consolidated, newest wins): {x['text']}"
        out.append((1.0 / (i + 1), W.Note(x["id"], "observation", text, sources[0][0], sources)))
    return out


def learn(bank: str, task: dict, trace: dict) -> int:
    """After grading: retain the test feedback the session got; consolidation folds it into the
    observations, so the next attempt at this code starts from it. Returns 1 if retained."""
    fb = [r["prompt"] for r in trace["trace"] if r["role"].startswith("intervention")]
    if not fb:
        return 0  # nothing went wrong, nothing new learned
    where = f"graded session {trace.get('run_id', '')} (test feedback)".strip()
    text = (f"TASK: {task['bug_report']}\n\nTEST FEEDBACK (solved={trace['solved']}):\n"
            + "\n\n".join(fb)[-8000:])
    retain(bank, [_item("feedback", text, where, f"learned-{trace['task_id']}",
                        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))])
    return 1


if __name__ == "__main__":
    # smoke: python hsmemory.py <task.json> <repo> <bank>  -> ingest, then recall on the working diff
    import sys
    t = json.loads(Path(sys.argv[1]).read_text())
    print(ingest(sys.argv[3], t, Path(sys.argv[2])))
    d = subprocess.run(["git", "-C", sys.argv[2], "diff", "HEAD"], capture_output=True, text=True).stdout
    for s, n in recall(sys.argv[3], d):
        print(round(s, 2), n.id, n.source, n.where, n.text[:160].replace("\n", " "))
