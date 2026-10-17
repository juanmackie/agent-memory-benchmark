#!/bin/bash
# replay_eval.sh <out.jsonl> [--model m]: reflect probe over the 8 tasks with v5's mid-task notes
cd "$(dirname "$0")"
out=$1; shift
for tr in boltons-unitparse:final-w1 boltons-dedupe:final-w1 boltons-featflag:final-w1 boltons-pluralize:final-w1 \
          boltons-statetrans:final-w1 boltons-seqledger:final-w3 boltons-discount:final-w2 boltons-rounding:final-w2; do
  bin/python reflect_probe.py ${tr%%:*} --run ${tr##*:} --quiet --out "$out" "$@" 2>&1 | grep SUMMARY | cut -c1-400
done
