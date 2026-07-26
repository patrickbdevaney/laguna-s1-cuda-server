#!/bin/bash
# Resume the capability sweep, skipping configs that already have results.
#
# WHY THIS EXISTS SEPARATELY from run_capability_sweep.sh. Two reasons, both learned the hard
# way on this box:
#
#  1. This box rebooted once mid-run and took a whole eval with it. The sweep is detached
#     (setsid, ppid=1) so it survives an ssh drop or the agent session ending, but nothing
#     survives a reboot. Re-running the full sweep to recover one missing config wastes hours,
#     so completion is checked per config against the JSON that config would have written.
#
#  2. run_capability_sweep.sh must NOT be edited while it is running. bash reads a script
#     incrementally by byte offset, so editing in place can make a running shell resume at the
#     wrong offset and execute garbage. Resume logic therefore lives in its own file rather
#     than as a flag on the original.
#
# Usage:  setsid bash tools/resume_sweep.sh > resume.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
OUT=${OUT:-eval_out}
LIMIT_GSM=${LIMIT_GSM:-200}
MAXTOK_GSM=${MAXTOK_GSM:-1024}
PY=oracle-venv/bin/python
mkdir -p "$OUT"

done_cfg() {   # a config counts as done only if its JSON parses and holds the expected n
  local f="$OUT/$1_gsm8k.json"
  [ -s "$f" ] || return 1
  $PY -c "
import json,sys
try:
    s=json.load(open('$f'))['summary']
except Exception:
    sys.exit(1)
sys.exit(0 if s.get('n',0) >= $LIMIT_GSM else 1)" 2>/dev/null
}

kill_server() {
  pkill -x lgserve 2>/dev/null
  for _ in $(seq 1 60); do pgrep -x lgserve >/dev/null || break; sleep 2; done
  pgrep -x lgserve >/dev/null && { pkill -9 -x lgserve; sleep 10; }
  sleep 20
  # Reading the 57 GB checkpoint leaves ~68 GB in a cache MemAvailable does not count. Without
  # this the next boot sees ~48 GB "available", the kernel hands out free pages instead of
  # reclaiming, and the box OOMs -- which is exactly how it died once.
  sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
  sleep 3
}

free_gb() { awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo; }

run_cfg() {
  local tag=$1 levels=$2
  if done_cfg "$tag"; then echo "#### $tag already complete, skipping"; return 0; fi
  echo "#### $tag (LG_EXPERT_LEVELS=${levels:-unset})  free=$(free_gb)GB"
  kill_server
  if [ -n "$levels" ]; then export LG_EXPERT_LEVELS=$levels; else unset LG_EXPERT_LEVELS; fi
  CTX=4096 PORT=8080 ./build/lgserve > "$OUT/server_$tag.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 180); do
    curl -s -m 3 http://127.0.0.1:8080/healthz >/dev/null 2>&1 && break
    kill -0 $pid 2>/dev/null || { echo "  SERVER DIED, see $OUT/server_$tag.log"; return 1; }
    sleep 5
  done
  curl -s -m 3 http://127.0.0.1:8080/healthz >/dev/null 2>&1 || { echo "  NO HEALTHZ"; kill_server; return 1; }
  echo "  up, free=$(free_gb)GB"
  $PY tools/eval_capability.py --suite gsm8k --limit "$LIMIT_GSM" --max-tokens "$MAXTOK_GSM" \
      --timeout 600 --tag "$tag" --out "$OUT/${tag}_gsm8k.json"
  kill_server
  echo "  done, free=$(free_gb)GB"
}

run_cfg baseline ""
run_cfg lv7 7
run_cfg lv5 5
echo "#### SUMMARY"
$PY tools/eval_compare.py "$OUT"
