"""compare.py <label>...: corrections / cost / time per arm, plus what the watcher said.
Reads /tmp/sdebench/run/<task>_full_<label>/trace.json."""
import glob, json, sys
from pathlib import Path


def load(label):
    out = {}
    for p in glob.glob(f"/tmp/sdebench/run/*_{label}/trace.json"):
        t = json.loads(Path(p).read_text())
        out[t["codebase"]] = t
    return out


arms = {l: load(l) for l in sys.argv[1:]}
common = sorted(set.intersection(*(set(a) for a in arms.values())))
print(f"tasks finished in every arm: {len(common)}\n")
print(f"{'arm':14} {'solved':>6} {'corrections':>11} {'tasks w/ corr':>13} {'cost $':>7} {'wall min':>8}")
for l, a in arms.items():
    rs = [a[c] for c in common]
    print(f"{l:14} {sum(r['solved'] for r in rs):>6} {sum(r['interventions'] for r in rs):>11} "
          f"{sum(r['interventions'] > 0 for r in rs):>13} {sum(r['cost_usd'] for r in rs):>7.2f} "
          f"{sum(r['wall_s'] for r in rs) / 60:>8.1f}")

for l, a in arms.items():
    ws = [a[c]["watcher"] for c in common if a[c].get("watcher")]
    if not ws:
        continue
    msgs = sum(len(w["messages"]) for w in ws)
    words = sum(len(m.split()) for w in ws for m in w["messages"])
    print(f"\n[{l}] watcher: reviews={sum(w['reviews'] for w in ws)} messages={msgs} "
          f"tasks with a message={sum(bool(w['messages']) for w in ws)}/{len(ws)} "
          f"end-of-turn deliveries={sum(w['end_of_turn_deliveries'] for w in ws)} errors={sum(w['errors'] for w in ws)}")
    print(f"  words added to the agent: {words} total, {words / max(1, len(ws)):.0f}/task; "
          f"watcher tokens in/out: {sum(w['tokens_in'] for w in ws)}/{sum(w['tokens_out'] for w in ws)}")

print("\nper task (corrections per arm):")
names = list(arms)
for c in common:
    row = [arms[l][c]["interventions"] for l in names]
    w = next((arms[l][c]["watcher"] for l in names if arms[l][c].get("watcher")), None)
    said = (" | " + w["messages"][0][:110]) if w and w["messages"] else ""
    flag = "  <-" if len(set(row)) > 1 else ""
    print(f"  {c:32} {row}{flag}{said}")
