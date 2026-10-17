"""batch.py <label> <parallel> [--only id,id] -- <run.py args>: run every chat-backed task."""
import concurrent.futures as cf, glob, json, os, subprocess, sys
from pathlib import Path
ROOT = Path.home() / "dev/amb-watcher"
label, par = sys.argv[1], int(sys.argv[2])
rest = sys.argv[3:]
only = None
if rest[:1] == ["--only"]:
    only, rest = set(rest[1].split(",")), rest[2:]
rest = rest[1:] if rest[:1] == ["--"] else rest
env = {**os.environ, "PATH": f"{ROOT}/runs/bin:" + os.environ["PATH"],
       "SDEBENCH_BOLTONS_HOST": str(Path.home() / "dev/_sdebench_hosts/boltons")}
_tok = Path.home() / ".sdebench/claude_oauth_token"  # from `claude setup-token`
if _tok.exists():
    env["CLAUDE_CODE_OAUTH_TOKEN"] = _tok.read_text().strip()
tasks = [p for p in sorted(glob.glob(str(ROOT / "sdebench/datasets/boltons-*/tasks/*/task.json")))
         if json.load(open(p)).get("conversations")]
if only:
    tasks = [p for p in tasks if Path(p).parents[2].name in only]
out = ROOT / "runs" / label
out.mkdir(parents=True, exist_ok=True)

def one(p):
    cb = Path(p).parents[2].name
    with open(out / f"{cb}.log", "w") as log:
        r = subprocess.run([sys.executable, "sdebench/harness/run.py", "--task", p, "--run-id", label, *rest],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return cb, r.returncode

with cf.ThreadPoolExecutor(par) as ex:
    for cb, rc in ex.map(one, tasks):
        print(cb, "exit", rc, flush=True)
