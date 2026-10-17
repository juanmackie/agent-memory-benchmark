"""cost_compare.py <label> <codebase,...> [--arms watcher,plugin]

Each task runs once per arm, ONE AT A TIME on the local Hindsight (port 8899), so the server's token
counters can be read before and after: every model call the memory side makes is counted, whoever
makes it (ingest, consolidation, knowledge pages, reflect, learning). Client-side calls (the watcher's
Jev decisions and note writer) come from its own log. Rows go to runs/<label>/costs.jsonl.

  watcher: proactive watcher, follows the agent, Jev decides, flash-lite writes, never blocks the agent
  plugin:  the existing hindsight-coding-agents plugin (deepen ingest of git history + chats,
           knowledge pages refreshed, reflect + inject at the prompt), as AMB's hscoding arm runs it
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path.home() / "dev/amb-watcher"
HS = os.environ.get("WATCHER_HS", "http://localhost:8899")
PLUGIN = Path.home() / "dev/hindsight-release-053/hindsight-integrations/coding-agents"
# $ per 1M tokens: input, cached input, output (third-party listings, Oct 2026; Jev bills input only)
PRICE = {"gemini-3.8-flash": (0.75, 0.075, 3.75), "gemini-3.5-flash-lite": (0.30, 0.03, 2.50), "jev": (0.042, 0.0, 0.0)}
METRIC = re.compile(r'^hindsight_llm_tokens_(input|output|thoughts|cached_input)_tokens_total\{(.*)\} ([0-9.e+]+)$')


def counters() -> dict:
    """Server token counters, summed per (model, scope, kind)."""
    out: dict = {}
    for line in urllib.request.urlopen(f"{HS}/metrics", timeout=30).read().decode().splitlines():
        m = METRIC.match(line)
        if m:
            labels = dict(re.findall(r'(\w+)="([^"]*)"', m.group(2)))
            key = (labels.get("model", "?"), labels.get("scope", "?"), m.group(1))
            out[key] = out.get(key, 0) + float(m.group(3))
    return out


def server_usage(before: dict, after: dict) -> dict:
    """{model: {scope: {kind: tokens}}} spent between two snapshots."""
    out: dict = {}
    for (model, scope, kind), v in after.items():
        d = v - before.get((model, scope, kind), 0)
        if d:
            out.setdefault(model, {}).setdefault(scope, {})[kind] = int(d)
    return out


def price(model: str, inp: int, cached: int, out: int) -> float:
    p = PRICE.get(model, PRICE["gemini-3.8-flash"])
    return ((inp - cached) * p[0] + cached * p[1] + out * p[2]) / 1e6


def server_cost(usage: dict) -> float:
    return sum(price(model, k.get("input", 0), k.get("cached_input", 0), k.get("output", 0) + k.get("thoughts", 0))
               for model, scopes in usage.items() for k in scopes.values())


def api(method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(f"{HS}/v1/default/banks/{path}", method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read() or b"{}")


def wait_idle(bank: str, timeout: float = 1800) -> None:
    """Background work (consolidation, page refreshes, the session retained after the run) counts too."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        busy = sum(len(api("GET", f"{bank}/operations?status={s}&limit=1").get("operations") or [])
                   for s in ("pending", "processing"))
        if not busy:
            return
        time.sleep(5)
    print(f"  ! {bank} still busy after {timeout}s", flush=True)


def wait_op(bank: str, op: str, timeout: float = 1800) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        d = api("GET", f"{bank}/operations/{op}")
        if d.get("status") in ("completed", "failed"):
            if d["status"] == "failed":
                raise RuntimeError(f"operation {op} on {bank} failed: {d.get('error_message')}")
            return d
        time.sleep(2)
    raise TimeoutError(f"operation {op} on {bank} still running")


def export_bank(bank: str, path: Path) -> Path:
    """Save a built bank as a transfer zip (memories, observations, pages, config): built once, reused."""
    op = api("POST", f"{bank}/transfer/export")["operation_id"]
    url = wait_op(bank, op)["result_metadata"]["download_url"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(urllib.request.urlopen(f"{HS}{url}", timeout=300).read())
    return path


HUB = "transfer-hub"  # the import endpoint lives under an existing bank; restore creates the target


def import_bank(path: Path, bank: str) -> None:
    """A fresh copy of an exported bank: re-embedded locally, no model calls."""
    try:
        api("DELETE", bank)
    except Exception:
        pass
    api("PATCH", f"{HUB}/config", {"updates": {}})
    boundary = "----bankarchive"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
            f"Content-Type: application/zip\r\n\r\n").encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{HS}/v1/default/banks/{HUB}/transfer/import?mode=restore&target_bank_id={bank}",
                                 data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    op = json.loads(urllib.request.urlopen(req, timeout=300).read())["operation_id"]
    wait_op(HUB, op)


JEV_LOG = ROOT / "runs/hs" / f"jev-{os.environ.get('WATCHER_JEV_PORT', '8897')}.jsonl"


def server_jev(t0: float, t1: float) -> dict:
    """The server's own decision-model calls in a session window, from its metering proxy's log
    (the server does not meter them). Each lane has its own server and proxy, so nothing mixes."""
    calls = tokens = 0
    for line in JEV_LOG.read_text().splitlines() if JEV_LOG.exists() else []:
        e = json.loads(line)
        if t0 <= e["ts"] <= t1:
            calls, tokens = calls + 1, tokens + e.get("input_tokens", 0)
    return {"calls": calls, "input": tokens, "cost": price("jev", tokens, 0, 0)}


def env() -> dict:
    e = {**os.environ, "PATH": f"{ROOT}/runs/bin:" + os.environ["PATH"],
         "SDEBENCH_BOLTONS_HOST": str(Path.home() / "dev/_sdebench_hosts/boltons")}
    tok = Path.home() / ".sdebench/claude_oauth_token"
    if tok.exists():
        e["CLAUDE_CODE_OAUTH_TOKEN"] = tok.read_text().strip()
    for line in (Path.home() / "nicolo-secrets.env").read_text().splitlines():
        if line.startswith("JEV_KEY="):
            e["JEV_KEY"] = line.split("=", 1)[1].strip().strip('"')
    return e


def run_task(task_json: Path, run_id: str, extra: list[str], e: dict, log: Path) -> dict:
    with log.open("w") as f:
        subprocess.run([sys.executable, "sdebench/harness/run.py", "--task", str(task_json), "--run-id", run_id,
                        "--agent", "claude-code", "--max-interventions", "3", *extra],
                       cwd=ROOT, env=e, stdout=f, stderr=subprocess.STDOUT)
    t = json.loads(task_json.read_text())
    res = Path("/tmp/sdebench/run") / f"{t['task_id']}_{extra[1]}_{run_id}" / "result.json"
    return json.loads(res.read_text()) if res.exists() else {}


def watcher_client_usage(run_dir: Path) -> dict:
    """The watcher's own model calls: Jev decisions and the note writer."""
    jev_in = jev_n = w_in = w_out = j_in = j_out = j_n = 0
    log = run_dir / "watcher.jsonl"
    for line in log.read_text().splitlines() if log.exists() else []:
        e = json.loads(line)
        if e.get("event") == "call" and e.get("service") == "decision":
            u = json.loads(e["response"]).get("usage") or {}
            jev_in += u.get("input_tokens", 0)
            jev_n += 1
        elif e.get("event") == "review" and (e.get("decision") or {}).get("escalated"):
            j_in += e.get("tokens_in", 0)  # Jev was unsure: the flash judge decided
            j_out += e.get("tokens_out", 0)
            j_n += 1
        elif e.get("event") == "review" and e.get("decision") and e.get("tokens_in"):
            w_in += e["tokens_in"]
            w_out += e["tokens_out"]
    return {"jev_calls": jev_n, "jev_input": jev_in, "writer_input": w_in, "writer_output": w_out,
            "judge_calls": j_n, "judge_input": j_in, "judge_output": j_out,
            "cost": price("jev", jev_in, 0, 0) + price("gemini-3.5-flash-lite", w_in, 0, w_out)
            + price("gemini-3.8-flash", j_in, 0, j_out)}


def plugin_ingest(task: dict, task_json: Path, bank: str, e: dict, log: Path, fresh: bool = True) -> int:
    """What AMB's hscoding provider does before the agent starts (memory/hscoding.py).
    fresh=False resumes an interrupted build: deepen catches up on what the bank is missing."""
    base = Path("/tmp/sdebench/cost-ingest") / bank
    src = base / "repo"
    if fresh or not (src / ".git").exists():
        try:
            api("DELETE", bank)
        except Exception:
            pass
        shutil.rmtree(base, ignore_errors=True)
        base.mkdir(parents=True)
        subprocess.run(["python", str(task_json.parents[2] / task.get("build", "build.py")), str(src)],
                       check=True, capture_output=True, env=e)
    cfg = base / "coding-agent.json"
    cfg.write_text(json.dumps({"pageTriggerType": "manual"}))
    conv = task.get("conversations") or []
    chats = conv if conv and isinstance(conv[0], list) else ([conv] if conv else [])
    cf = base / "conversations.json"
    cf.write_text(json.dumps([{"id": f"session-{i + 1:02d}", "turns": c} for i, c in enumerate(chats)]))
    # The plugin caps every request at 15s; a busy local server misses some enqueues. A user's next
    # session re-runs deepen and catches up, so do the same (items already in the bank are kept).
    logs, failed = [], 0
    for _ in range(3):
        p = subprocess.run(["node", str(PLUGIN / "dist/deepen.js"), "--repo", str(src), "--bank", bank, "--api-url", HS,
                            "--git-ingest", "full", "--config", str(cfg), "--conversations", str(cf)],
                           capture_output=True, text=True, env=e, timeout=3600)
        logs.append(p.stdout + "\n--- stderr ---\n" + p.stderr)
        log.write_text("\n=== pass ===\n".join(logs))
        if p.returncode != 0:
            raise RuntimeError(f"deepen failed for {bank}: {(p.stderr or p.stdout)[-400:]}")
        m = re.search(r"(\d+) items failed to enqueue", p.stdout + p.stderr)
        failed = int(m.group(1)) if m else 0
        if not failed:
            break
        wait_idle(bank)
    deadline = time.time() + 900
    while time.time() < deadline:
        sp = subprocess.run(["node", str(PLUGIN / "dist/status.js"), "--repo", str(src), "--bank", bank,
                             "--api-url", HS, "--config", str(cfg)], capture_output=True, text=True, env=e)
        try:
            if json.loads(sp.stdout.strip().splitlines()[-1]).get("synced"):
                break
        except Exception:
            pass
        time.sleep(5)
    pages = api("GET", f"{bank}/mental-models?limit=100").get("items") or []
    if not pages:  # same rule as AMB: a plugin arm without pages measures no memory
        raise RuntimeError(f"bank {bank} has no knowledge pages after deepen")
    for m in pages:
        api("POST", f"{bank}/mental-models/{m['id']}/refresh")
    wait_idle(bank)
    return failed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("tasks")
    ap.add_argument("--arms", default="watcher,plugin")
    a = ap.parse_args()
    out = ROOT / "runs" / a.label
    out.mkdir(parents=True, exist_ok=True)
    e = env()
    sys.path.insert(0, str(ROOT / "sdebench/watcher"))
    import hsmemory  # the watcher's own ingest, run here so its setup is counted apart from the session
    banks: list[str] = []
    for cb in a.tasks.split(","):
        task_json = ROOT / f"sdebench/datasets/{cb}/tasks/main/task.json"
        task = json.loads(task_json.read_text())
        for arm in a.arms.split(","):
            bank = f"{a.label}-{arm[0]}-{cb}"
            try:
                api("DELETE", bank)  # always a fresh bank: setup cost is part of the count
            except Exception:
                pass
            for b in banks:  # nothing from an earlier bank may run into this count
                wait_idle(b)
            banks.append(bank)
            t0, before = time.time(), counters()
            not_enqueued = 0
            if arm == "watcher":
                hsmemory.ingest(bank, task, Path("/nonexistent"))  # chats only; waits for consolidation
            else:
                not_enqueued = plugin_ingest(task, task_json, bank, e, out / f"{cb}-deepen.log")
            wait_idle(bank)
            t1, mid = time.time(), counters()
            if arm == "watcher":
                r = run_task(task_json, f"{a.label}-w", ["--history", "full", "--watcher", "--watcher-follow",
                                                         "--watcher-memory", "hindsight"],
                             {**e, "WATCHER_BANK_PREFIX": f"{a.label}-w"}, out / f"{cb}-watcher.log")
                client = watcher_client_usage(Path("/tmp/sdebench/run") / f"{task['task_id']}_full_{a.label}-w")
            else:
                r = run_task(task_json, f"{a.label}-p", ["--history", "hscoding"],
                             {**e, "SDE_HSCODING_PLUGIN_DIR": str(PLUGIN), "SDE_HSCODING_BANK": bank,
                              "SDE_HINDSIGHT_URL": HS}, out / f"{cb}-plugin.log")
                client = {"cost": 0.0}
            wait_idle(bank)
            setup, session = server_usage(before, mid), server_usage(mid, counters())
            row = {"task": cb, "arm": arm, "solved": r.get("solved"), "corrections": r.get("interventions"),
                   "agent_cost": r.get("cost_usd"), "agent_wall_s": r.get("wall_s"),
                   "setup_server": setup, "setup_cost": round(server_cost(setup), 4), "setup_s": round(t1 - t0),
                   "session_server": session, "session_client": client, "not_enqueued": not_enqueued,
                   "session_cost": round(server_cost(session) + client["cost"], 4), "session_s": round(time.time() - t1)}
            with (out / "costs.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{cb:28} {arm:8} solved={row['solved']} corrections={row['corrections']} agent=${row['agent_cost']} "
                  f"memory setup=${row['setup_cost']} ({row['setup_s']}s) session=${row['session_cost']}", flush=True)


if __name__ == "__main__":
    main()
