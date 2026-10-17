#!/bin/bash
# lane.sh <label> <hindsight port> <jev proxy port> <task...>: duel the reflect arm on each task, on
# this lane's own server, so each session's memory cost (server metrics + Jev proxy) is its own.
cd "$(dirname "$0")"
label=$1; port=$2; jev=$3; shift 3
export WATCHER_HS=http://localhost:$port WATCHER_HS_URL=http://localhost:$port WATCHER_JEV_PORT=$jev
for t in "$@"; do
  echo "== $t $(date +%H:%M)"
  bin/python duel.py $label-$port $t banks/$t.zip --runs 3 --arms ${ARMS:-reflect}
done
echo "lane done $(date +%H:%M)"
