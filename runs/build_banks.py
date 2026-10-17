"""build_banks.py <task,task,...> [--jobs 3]: build each task's bank once with the plugin's own ingest
on the BUILD server (WATCHER_HS, e.g. :8898) and export it to runs/banks/<task>.zip. Kept apart from
the measured server so builds never leak into a session's cost. Existing zips are skipped."""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cost_compare as C


def build(cb: str) -> str:
    zp = C.ROOT / "runs/banks" / f"{cb}.zip"
    if zp.exists():
        return f"{cb}: already built"
    tj = C.ROOT / f"sdebench/datasets/{cb}/tasks/main/task.json"
    bank, t0 = f"build-{cb}", time.time()
    log = C.ROOT / "runs/banks" / f"{cb}.build.log"
    try:
        C.plugin_ingest(json.loads(tj.read_text()), tj, bank, C.env(), log)
        C.export_bank(bank, zp)
    except Exception as e:
        return f"{cb}: FAILED {e}"
    return f"{cb}: built in {round(time.time() - t0)}s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks")
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    assert "8899" not in C.HS, "build on the build server (WATCHER_HS), never on the measured one"
    with ThreadPoolExecutor(a.jobs) as ex:
        for line in ex.map(build, a.tasks.split(",")):
            print(line, flush=True)
