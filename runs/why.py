"""why.py <run dir name or glob>: one timeline of what the agent did and what the watcher decided/said."""
import json
import sys
from pathlib import Path

for d in sorted(Path("/tmp/sdebench/run").glob(sys.argv[1])):
    print(f"===== {d.name}")
    ev = []
    for line in (d / "watcher.jsonl").read_text().splitlines() if (d / "watcher.jsonl").exists() else []:
        e = json.loads(line)
        if e.get("event") == "review":
            ev.append((e["ts"], f"  W {e.get('trigger'):8} speak={e.get('speak')} {e.get('decision')} {e.get('ms')}ms | "
                                f"{(e.get('message') or '')[:160]}"))
        elif e.get("event") in ("delivered", "error"):
            ev.append((e["ts"], f"  W {e['event'].upper()} {e.get('error', '')[:200]}"))
    for line in (d / "events.jsonl").read_text().splitlines() if (d / "events.jsonl").exists() else []:
        e = json.loads(line)
        k = e.get("kind")
        if k == "round":
            txt = e.get("text", "")
            ev.append((e["ts"], f"ROUND {e.get('role')} {txt[txt.find('FAILURES'):][:300] if 'FAILURES' in txt else ''}".replace("\n", " ")))
        elif k == "say":
            ev.append((e["ts"], f"  say {e.get('text', '')[:220]}".replace("\n", " ")))
        elif k == "tool" and e.get("tool") in ("Edit", "Write", "MultiEdit"):
            ev.append((e["ts"], f"  EDIT {str(e.get('input'))[:160]}".replace("\n", " ")))
        elif k == "grade":
            ev.append((e["ts"], f"GRADE passed={e.get('passed')}"))
    t0 = min((t for t, _ in ev), default=0)
    for t, s in sorted(ev):
        print(f"{t - t0:6.1f} {s}")
