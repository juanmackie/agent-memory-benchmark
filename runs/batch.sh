#!/bin/bash
# usage: batch.sh <arm-label> <parallel> <extra run.py args...>   (tasks: every chat-backed task)
source ~/dev/amb-watcher/runs/env.sh
cd ~/dev/amb-watcher
LABEL=$1; PAR=$2; shift 2
mkdir -p runs/$LABEL
python3 - <<PY > runs/$LABEL/tasks.txt
import json,glob
for p in sorted(glob.glob('sdebench/datasets/boltons-*/tasks/*/task.json')):
    if json.load(open(p)).get('conversations'): print(p)
PY
cat runs/$LABEL/tasks.txt | xargs -P $PAR -I{} sh -c '
  id=$(basename $(dirname $(dirname $(dirname {}))))
  python3 sdebench/harness/run.py --task {} --run-id '"$LABEL"' "$@" > runs/'"$LABEL"'/$id.log 2>&1
  echo "$id exit=$?"' _ "$@"
