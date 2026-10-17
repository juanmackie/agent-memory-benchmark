#!/bin/bash
# season.sh <label> <runs> <arms> <task...>: duel each task as soon as its bank zip exists (built by build_banks.py).
cd "$(dirname "$0")"
label=$1; runs=$2; arms=$3; shift 3
for t in "$@"; do
  until [ -f banks/$t.zip ] || grep -q "^$t: FAILED" banks/build-all.out 2>/dev/null; do sleep 30; done
  [ -f banks/$t.zip ] || { echo "$t: no bank, skipped"; continue; }
  echo "== $t"
  bin/python duel.py $label $t banks/$t.zip --runs $runs --arms $arms
done
echo "season done"
