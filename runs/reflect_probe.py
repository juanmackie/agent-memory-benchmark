"""reflect_probe.py <task> [--run final-w1] [--only task,edit] [--extra '{"json": "reflect fields"}']

Replays the v5 watcher's logged moments for one task against reflect on a fresh copy of the
task's bank, and prints per moment: what v5 decided, what reflect answered, time, LLM tokens,
and the decision-model calls the server made (from the metering proxy's log)."""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import cost_compare as C

sys.path.insert(0, str(C.ROOT / "sdebench/watcher"))
import reflectwatch as RW  # noqa: E402

JEV_LOG = C.ROOT / "runs/hs/jev-8897.jsonl"


def jev_since(t0: float) -> tuple[int, int]:
    n = tok = 0
    for line in JEV_LOG.read_text().splitlines() if JEV_LOG.exists() else []:
        e = json.loads(line)
        if e["ts"] >= t0:
            n, tok = n + 1, tok + e.get("input_tokens", 0)
    return n, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--run", default="final-w1")
    ap.add_argument("--only", default="")
    ap.add_argument("--extra", default="{}")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--model", default="gemini-3.8-flash", help="the server's reflect model, for the cost estimate")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out", default="", help="append one summary row (json) here")
    a = ap.parse_args()
    bank = f"probe-{a.task}"
    try:
        have = C.api("GET", f"/v1/default/banks/{bank}/stats").get("total_nodes")
    except Exception:
        have = 0
    if a.fresh or not have:
        C.import_bank(C.ROOT / "runs/banks" / f"{a.task}.zip", bank)
        time.sleep(5)
        C.wait_idle(bank)
    run = next(Path("/tmp/sdebench/run").glob(f"{a.task}-001_full_{a.run}"))
    goal = json.loads((C.ROOT / f"sdebench/datasets/{a.task}/tasks/main/task.json").read_text()).get("bug_report", "")
    w = RW.ReflectWatcher(bank, goal)
    extra = json.loads(a.extra)
    sent: list[str] = []
    recent: list[str] = []
    tot = {"moments": 0, "v5_speak": 0, "speak": 0, "agree": 0, "ms": 0, "llm_in": 0, "llm_out": 0, "jev_tok": 0,
           "v5_ms": 0, "missed": [], "extra": []}
    for line in (run / "watcher.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e.get("event") == "call" and e.get("service") == "decision":
            # the agent's recent steps as v5 saw them at this moment
            st = json.loads(e["request"]).get("state", "") if e["request"].endswith("}") else ""
            mm = re.search(r"AGENT'S RECENT STEPS:\n(.*?)(\n\nCODE CHANGE SO FAR:|\n\nPROJECT MEMORY:|$)", st, re.S)
            recent = [x for x in mm.group(1).splitlines() if x.startswith(("agent ", "the user"))] if mm else []
            continue
        if e.get("event") != "review" or (a.only and e["trigger"] not in a.only.split(",")):
            continue
        m = RW.Moment(e["trigger"], e.get("trigger_text") or "", recent=recent, diff=e.get("diff") or "")
        t0 = time.time()
        v, _ = w.review(m)
        d = v.decision or {}
        r = {"text": v.message if v.speak else "", "usage": {"input_tokens": d.get("llm_in") or 0,
                                                              "output_tokens": d.get("llm_out") or 0}}
        ms = int((time.time() - t0) * 1000)
        time.sleep(0.3)
        n, tok = jev_since(t0)
        u = r.get("usage") or {}
        spoke, v5 = bool((r.get("text") or "").strip()), str(e["speak"]) == "True"
        tot["moments"] += 1; tot["v5_speak"] += v5; tot["speak"] += spoke; tot["agree"] += spoke == v5
        tot["ms"] += ms; tot["v5_ms"] += int(e.get("ms") or 0) + int(e.get("recall_ms") or 0)
        tot["llm_in"] += u.get("input_tokens") or 0
        tot["llm_out"] += (u.get("output_tokens") or 0) + (u.get("thoughts_tokens") or 0)
        tot["jev_tok"] += tok
        if a.out:  # one row per moment, joined to the logged pick request by time
            with open(a.out.replace(".jsonl", ".moments.jsonl"), "a") as f:
                f.write(json.dumps({"task": a.task, "run": a.run, "trigger": e["trigger"], "t0": t0, "t1": time.time(),
                                    "v5_speak": v5, "v5_none": (e.get("decision") or {}).get("none"),
                                    "spoke": spoke, "text": (r.get("text") or "")[:300], "ms": ms}) + "\n")
        if v5 and not spoke:
            tot["missed"].append(e["trigger"])
        if spoke and not v5:
            tot["extra"].append(f"{e['trigger']}: {r['text'][:200]}")
        if a.quiet:
            continue
        print(f"--- {e['trigger']:8} v5: speak={e['speak']} {e.get('decision')} {e.get('ms')}ms+{e.get('recall_ms')}ms recall")
        print(f"    reflect {ms}ms llm in={u.get('input_tokens')} out={u.get('output_tokens')} jev={n} calls/{tok} tok"
              f" | {(r.get('text') or '').strip()[:400]!r}")
        print(f"      steps {d.get('steps')} quote={v.quote[:100]!r} where={v.where}")
    tot["cost"] = round(C.price(a.model, tot["llm_in"], 0, tot["llm_out"]) + C.price("jev", tot["jev_tok"], 0, 0), 5)
    tot["task"], tot["run"] = a.task, a.run
    print("SUMMARY", json.dumps(tot))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps({**tot, "extra_args": a.extra}) + "\n")


if __name__ == "__main__":
    main()
