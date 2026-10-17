#!/bin/bash
# restart.sh: stop the measured server's restart loop and its process, then start both again so
# start.sh is re-read (killing only the process restarts it with the loop's old environment).
cd "$(dirname "$0")"
for p in $(pgrep -f "^/bin/bash ./start.sh"); do
  [ "$(lsof -a -p $p -d cwd -Fn 2>/dev/null | sed -n 's/^n//p')" = "$HOME/dev/hindsight-release-053/hindsight-api-slim" ] && kill $p
done
pkill -f watcher-memory-server
while lsof -iTCP:8899 -sTCP:LISTEN >/dev/null 2>&1; do sleep 1; done
nohup ./start.sh > server.log 2>&1 &
until curl -sf localhost:8899/health >/dev/null; do sleep 2; done
ps eww -p $(pgrep -f watcher-memory-server) | tr ' ' '\n' | grep -E "^HINDSIGHT_API_REFLECT_LLM" | sed -E 's/(KEY=).*/\1<redacted>/'
