"""summarize.py <replay.jsonl>: per task and total, against v5."""
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
for r in rows:
    print(f"{r['task'][8:]:14} agree {r['agree']}/{r['moments']} missed {r['missed']} extra {len(r['extra'])} "
          f"ms/moment {r['ms'] // r['moments']} (v5 {r['v5_ms'] // r['moments']}) cost ${r['cost']:.4f}")
t = {k: sum(r[k] for r in rows) for k in ('moments', 'v5_speak', 'speak', 'agree', 'ms', 'v5_ms', 'llm_in', 'llm_out', 'jev_tok', 'cost')}
print(t)
print(f"per moment: {t['ms'] // t['moments']}ms (v5 {t['v5_ms'] // t['moments']}ms), ${t['cost'] / t['moments']:.5f}")
