"""Offline scoring: replay every recorded agent attempt through the watcher, no agent needed.

For each round of each trace (its submitted patch + whether grading passed):
  failed round -> the watcher should speak, and say the task's recorded rule  (hit / wrong / miss)
  passed round -> the watcher should stay silent                               (ok / false alarm)
Whether a message states the rule is judged by a separate model call against task.json `policy`.

    python replay_eval.py <trace.json>... [--out results.jsonl]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import watcher as W

AGREES = """A memory watcher sent this message to a coding agent fixing a bug:
MESSAGE: {message}

The project's recorded rule for this code is: {policy} (it concerns `{function}` in {module}).
Does the message point the agent toward that rule (right value/behaviour, or clearly says the current
change disagrees with it)? A message about something else, or one that implies a different value, is NO.
Reply JSON only: {{"agrees": true|false}}"""


def build_repo(task: dict, dest: Path) -> None:
    ds = W.DATASETS / task["codebase"]
    subprocess.run(["python3", str(ds / task["build"]), str(dest)], check=True, capture_output=True,
                   env={**os.environ, "SDEBENCH_BOLTONS_HOST": os.path.expanduser("~/dev/_sdebench_hosts/boltons")})


def task_for(trace: dict) -> dict:
    p = W.DATASETS / trace["codebase"] / "tasks" / "main" / "task.json"
    return json.loads(p.read_text())


def score_trace(path: Path) -> list[dict]:
    tr = json.loads(path.read_text())
    task = task_for(tr)
    rows = []
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td) / "repo"
        build_repo(task, repo)
        if os.environ.get("WATCHER_BANK_PREFIX"):  # one Hindsight bank per project, ingested here
            import hsmemory
            os.environ["WATCHER_BANK"] = bank = f"{os.environ['WATCHER_BANK_PREFIX']}-{task['codebase']}"
            hsmemory.ingest(bank, task, repo)
            w = W.Watcher(task.get("bug_report", ""), lambda m: hsmemory.search(bank, m))
        else:
            w = W.build_watcher(task, repo)
        for rnd in tr["trace"]:
            patch = rnd.get("patch")
            if patch is None:
                continue
            v, hits = w.review(patch)
            row = {"trace": path.parent.name, "task": task["task_id"], "source": task["source"],
                   "round": rnd["role"], "passed": rnd.get("grade_passed"), "speak": v.speak,
                   "message": v.message, "evidence": v.evidence, "hits": [n.id for _, n in hits],
                   "gold_in_hits": any(n.source == "chat" for _, n in hits), "quote": v.quote,
                   "tokens_in": v.tokens_in, "tokens_out": v.tokens_out, "ms": v.ms, "decision": v.decision}
            if v.speak:
                out, _, _ = W.gemini(model="gemini-3.8-flash", prompt=AGREES.format(message=v.message, policy=task.get("policy", "?"),
                                                   function=task.get("function"), module=task.get("module")))
                row["agrees"] = bool(out.get("agrees"))
            if rnd.get("grade_passed"):
                row["outcome"] = "false_alarm" if v.speak else "ok_silent"
            else:
                row["outcome"] = ("hit" if row.get("agrees") else "wrong") if v.speak else "miss"
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out", default="replay.jsonl")
    ap.add_argument("--par", type=int, default=6)
    args = ap.parse_args()
    rows = []
    with cf.ThreadPoolExecutor(args.par) as ex:
        for rs in ex.map(lambda p: score_trace(Path(p)), args.traces):
            rows += rs
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    c = Counter(r["outcome"] for r in rows)
    failed = sum(not r["passed"] for r in rows)
    print(f"rounds={len(rows)} failed={failed} passed={len(rows) - failed}")
    print(f"  failed rounds: hit={c['hit']} wrong={c['wrong']} miss={c['miss']}")
    print(f"  passed rounds: silent={c['ok_silent']} false_alarm={c['false_alarm']}")
    print(f"  gold chat in top-{W.TOP_K}: {sum(r['gold_in_hits'] for r in rows)}/{len(rows)}")
    print(f"  watcher tokens in/out: {sum(r['tokens_in'] for r in rows)}/{sum(r['tokens_out'] for r in rows)}")
    for r in rows:
        if r["outcome"] in ("wrong", "miss", "false_alarm"):
            print(f"  [{r['outcome']}] {r['task']} {r['round']}: {r['message'][:160]}")


if __name__ == "__main__":
    main()
