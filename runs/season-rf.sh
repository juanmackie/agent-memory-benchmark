#!/bin/bash
# season-rf.sh <label>: the 33 tasks, alternately on lane A (8899) and lane B (8898).
cd "$(dirname "$0")"
T=(boltons-budget boltons-cachekey boltons-csvquote boltons-dedupe boltons-dedupe-amended boltons-deploywave boltons-discount boltons-drainplan boltons-featflag boltons-findhashtags boltons-hostallow boltons-listmerge boltons-mimemap boltons-overage boltons-overage-amended boltons-parseflag boltons-pluralize boltons-redact boltons-retryjitter boltons-retryjitter-amended boltons-rounding boltons-sched boltons-seqledger boltons-slalog boltons-slalog-amended boltons-slugify boltons-statetrans boltons-tagmerge boltons-tagmerge-amended boltons-trimstats boltons-under2camel boltons-unitparse boltons-unitparse-amended)
A=(); B=()
for i in "${!T[@]}"; do if (( i % 2 == 0 )); then A+=("${T[$i]}"); else B+=("${T[$i]}"); fi; done
./lane.sh "$1" 8899 8897 "${A[@]}" > "lane-$1-a.out" 2>&1 &
./lane.sh "$1" 8898 8896 "${B[@]}" > "lane-$1-b.out" 2>&1 &
wait
echo "season $1 done $(date +%H:%M)"
