"""duel.py <label> <codebase> <bank.zip> [--runs 3]: the plugin and the watcher on ONE task and ONE bank.

Both arms read the same Hindsight bank (built once by the plugin's own deepen ingest, exported, and
restored fresh for every session with no model calls), so setup is
identical and the only difference is how memory reaches the agent: the plugin reflects at the prompt,
the watcher follows the agent and speaks up. Neither writes back to the bank, so every session starts
from the same memory. Sessions run one at a time so each one's memory-side cost can be counted.
Rows go to runs/<label>/duel.jsonl.
"""
import argparse
import json
import time
from pathlib import Path

import cost_compare as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("codebase")
    ap.add_argument("archive", help="an exported bank zip (cost_compare.export_bank); each session gets a fresh copy")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--arms", default="plugin,watcher")
    a = ap.parse_args()
    out = C.ROOT / "runs" / a.label
    out.mkdir(parents=True, exist_ok=True)
    e = C.env()
    tj = C.ROOT / f"sdebench/datasets/{a.codebase}/tasks/main/task.json"
    task = json.loads(tj.read_text())
    if not Path(a.archive).exists():  # build once with the plugin's own ingest, then export
        bank, t0, before = f"{a.label}-build", time.time(), C.counters()
        C.plugin_ingest(task, tj, bank, e, out / "build.log")
        usage = C.server_usage(before, C.counters())
        C.export_bank(bank, Path(a.archive))
        print(f"built {a.archive} setup=${C.server_cost(usage):.4f} ({round(time.time() - t0)}s)", flush=True)
    for i in range(1, a.runs + 1):
        for arm in a.arms.split(","):
            run_id = f"{a.label}-{'q' if arm == 'pwatch' else arm[0]}{i}"  # p, w, r, q (pwatch)
            bank = f"{a.label}-{run_id}"
            C.import_bank(Path(a.archive), bank)  # same memory every session, no model calls
            time.sleep(5)  # a restore can queue consolidation a moment later; it must not land in the session
            C.wait_idle(bank)
            t0, before = time.time(), C.counters()
            if arm == "reflect":  # the watcher on reflect: Hindsight decides and writes, the client only asks
                r = C.run_task(tj, run_id, ["--history", "full", "--watcher", "--watcher-follow",
                                            "--watcher-memory", "hindsight"],
                               {**e, "WATCHER_SHARED_BANK": bank, "WATCHER_DECIDER": "reflect"}, out / f"{run_id}.log")
                client = C.watcher_client_usage(Path("/tmp/sdebench/run") / f"{task['task_id']}_full_{run_id}")
            elif arm == "watcher":
                r = C.run_task(tj, run_id, ["--history", "full", "--watcher", "--watcher-follow",
                                            "--watcher-memory", "hindsight"],
                               {**e, "WATCHER_SHARED_BANK": bank}, out / f"{run_id}.log")
                client = C.watcher_client_usage(Path("/tmp/sdebench/run") / f"{task['task_id']}_full_{run_id}")
            else:  # plugin: first-prompt reflect; pwatch: the plugin's own watcher (autoInject "watch")
                r = C.run_task(tj, run_id, ["--history", "hscoding"],
                               {**e, "SDE_HSCODING_PLUGIN_DIR": str(C.PLUGIN), "SDE_HSCODING_BANK": bank,
                                "SDE_HINDSIGHT_URL": C.HS,
                                **({"HINDSIGHT_AUTO_INJECT": "watch"} if arm == "pwatch" else {})},
                               out / f"{run_id}.log")
                client = {"cost": 0.0}
            C.wait_idle(bank)
            usage = C.server_usage(before, C.counters())
            jev = C.server_jev(t0, time.time())
            row = {"task": a.codebase, "run": run_id, "arm": arm, "solved": r.get("solved"), "corrections": r.get("interventions"),
                   "agent_cost": r.get("cost_usd"), "agent_wall_s": r.get("wall_s"), "session_server": usage,
                   "session_client": client, "session_jev": jev,
                   "memory_cost": round(C.server_cost(usage) + client["cost"] + jev["cost"], 4),
                   "elapsed_s": round(time.time() - t0)}
            with (out / "duel.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{run_id:14} solved={row['solved']} corrections={row['corrections']} agent=${row['agent_cost']} "
                  f"agent_time={row['agent_wall_s']}s memory=${row['memory_cost']}", flush=True)


if __name__ == "__main__":
    main()
