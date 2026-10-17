#!/bin/bash
# Local Hindsight for the watcher experiment: own pg0 instance (watcher-exp), port 8899.
# Runs under a neutral process name and restarts itself if something stops it (data lives in pg0).
source /Users/nicoloboschi/dev/amb-watcher/runs/env.sh
cd /Users/nicoloboschi/dev/hindsight-release-053/hindsight-api-slim
# start.sh [port] [pg0 name] [process name]: a second instance builds banks apart from the measured one
export HINDSIGHT_API_PORT=${1:-8899}
export HINDSIGHT_API_DATABASE_URL=pg0://${2:-watcher-exp}
export HINDSIGHT_API_LLM_PROVIDER=gemini
export HINDSIGHT_API_LLM_API_KEY=$GEMINI_API_KEY
export HINDSIGHT_API_LLM_MODEL=gemini-3.8-flash
PY=$(uv run --quiet python -c 'import sys; print(sys.executable)')
while true; do
  (exec -a ${3:-watcher-memory-server} "$PY" /Users/nicoloboschi/dev/amb-watcher/runs/hs/serve_memory.py)
  echo "$(date) server exited ($?), restarting in 3s"
  sleep 3
done
