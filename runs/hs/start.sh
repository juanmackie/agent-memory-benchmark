#!/bin/bash
# Local Hindsight for the watcher experiment: own pg0 instance (watcher-exp), port 8899.
# Runs under a neutral process name and restarts itself if something stops it (data lives in pg0).
source /Users/nicoloboschi/dev/amb-watcher/runs/env.sh
cd /Users/nicoloboschi/dev/hindsight-release-053/hindsight-api-slim
# start.sh [port] [pg0 name] [process name] [jev proxy port]: a second instance builds banks, or runs a second lane
export HINDSIGHT_API_PORT=${1:-8899}
export HINDSIGHT_API_DATABASE_URL=pg0://${2:-watcher-exp}
export HINDSIGHT_API_LLM_PROVIDER=gemini
export HINDSIGHT_API_LLM_API_KEY=$GEMINI_API_KEY
export HINDSIGHT_API_LLM_MODEL=gemini-3.8-flash
# reflect only gets the decision model (Jev), through the metering proxy (jev_proxy.py on 8897)
export HINDSIGHT_API_REFLECT_DECISION_MODEL_PROVIDER=typesafe
# reflect answers with low thinking, as v5's judge did (default thinking took 6-20s per answer)
# flash-lite answers without thinking by default; a thinking_level makes it think (1.7k tokens, 14s)
[ "${WATCHER_REFLECT_MODEL:-gemini-3.8-flash}" = gemini-3.5-flash-lite ] || export HINDSIGHT_API_REFLECT_LLM_EXTRA_BODY="{\"thinking_config\": {\"thinking_level\": \"${WATCHER_THINKING:-low}\"}}"
export HINDSIGHT_API_REFLECT_PROMPT_CACHE_ENABLED=false
export HINDSIGHT_API_REFLECT_LLM_MODEL=${WATCHER_REFLECT_MODEL:-gemini-3.8-flash}
export HINDSIGHT_API_REFLECT_ABSTAIN_WRITER_LLM_MODEL=${WATCHER_ABSTAIN_WRITER-gemini-3.5-flash-lite}
export HINDSIGHT_API_REFLECT_DECISION_MODEL_API_KEY=$(grep '^JEV_KEY=' ~/nicolo-secrets.env | cut -d= -f2- | tr -d '"')
export HINDSIGHT_API_REFLECT_DECISION_MODEL_BASE_URL=http://127.0.0.1:${4:-8897}
PY=$(uv run --quiet python -c 'import sys; print(sys.executable)')
while true; do
  (exec -a ${3:-watcher-memory-server} "$PY" /Users/nicoloboschi/dev/amb-watcher/runs/hs/serve_memory.py)
  echo "$(date) server exited ($?), restarting in 3s"
  sleep 3
done
