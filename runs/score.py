"""score.py <label> [<label> ...]: per task and overall, plugin vs watcher from duel.jsonl rows.
Later labels win for the same task+arm (a rerun of the watcher replaces its older rows)."""
import json
import sys
from collections import defaultdict
from pathlib import Path

rows: dict = {}
for label in sys.argv[1:]:
    seen: dict = defaultdict(list)
    for line in (Path(__file__).parent / label / "duel.jsonl").read_text().splitlines():
        r = json.loads(line)
        r.setdefault("task", label)
        seen[(r["task"], r["arm"])].append(r)
    rows.update(seen)


def stats(rs: list[dict]) -> dict:
    n = len(rs)
    return {"n": n, "solved": sum(bool(r["solved"]) for r in rs), "first_try": sum(bool(r["solved"]) and not r["corrections"] for r in rs),
            "agent": sum(r["agent_cost"] or 0 for r in rs) / n, "time": sum(r["agent_wall_s"] or 0 for r in rs) / n,
            "mem": sum(r["memory_cost"] for r in rs) / n}


tasks = sorted({t for t, _ in rows})
tot = {"plugin": [], "watcher": []}
print(f"{'task':28} {'arm':8} solved first-try  claude$   time  memory$")
for t in tasks:
    for arm in ("plugin", "watcher"):
        rs = rows.get((t, arm))
        if not rs:
            continue
        tot[arm] += rs
        s = stats(rs)
        print(f"{t:28} {arm:8} {s['solved']}/{s['n']}    {s['first_try']}/{s['n']}     {s['agent']:.3f}  {s['time']:5.0f}s  {s['mem']:.4f}")
print()
for arm, rs in tot.items():
    if rs:
        s = stats(rs)
        print(f"TOTAL {arm:8} solved {s['solved']}/{s['n']}  first-try {s['first_try']}/{s['n']}  claude ${s['agent']:.3f}/run  "
              f"time {s['time']:.0f}s/run  memory ${s['mem']:.4f}/run")
both = [t for t in tasks if (t, "plugin") in rows and (t, "watcher") in rows]
print(f"tasks with both arms: {len(both)}")
