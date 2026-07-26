#!/bin/bash
# Capability sweep: baseline vs LG_EXPERT_LEVELS=7 vs 5, scored as time-to-correct-answer.
#
# MEMORY DISCIPLINE. One server at a time, hard-killed and given a settle gap before the next
# boots. The box has 122 GB unified and a server peaks near 71 GB, so two alive at once is an
# OOM, and an OOM here takes ssh and rustdesk with it. The wait loop below is on the PROCESS
# being gone, not on a timer.
#
# CTX is deliberately small. Prompts here are a few hundred tokens and the budget is 4096, so
# 8192 is plenty and it keeps the KV allocation off the memory ceiling.
set -u
cd "$(dirname "$0")/.."
OUT=${OUT:-eval_out}
# GSM8K is the primary discriminator (n=150 -> +-3% error bars, short traces); AIME is the
# hard-reasoning secondary (n=30 -> +-9%, only detects a large drop). Budgets are sized from
# measured trace lengths on this model: AIME ran 588-3459 tokens with a tail past 14k, GSM8K
# is roughly a tenth of that.
LIMIT_GSM=${LIMIT_GSM:-150}
LIMIT_AIME=${LIMIT_AIME:-30}
MAXTOK_GSM=${MAXTOK_GSM:-2048}
MAXTOK_AIME=${MAXTOK_AIME:-12288}
PY=oracle-venv/bin/python
mkdir -p "$OUT"

kill_server() {
  pkill -x lgserve 2>/dev/null
  for _ in $(seq 1 60); do pgrep -x lgserve >/dev/null || break; sleep 2; done
  pgrep -x lgserve >/dev/null && { pkill -9 -x lgserve; sleep 10; }
  sleep 20   # settle: let the driver actually release the unified pages
  # Reclaim page cache before the next 57 GB load. Reading the checkpoint leaves ~68 GB in a
  # cache that MemAvailable does NOT count, so the next boot sees ~48 GB "available" and the
  # kernel hands out free pages rather than reclaiming -- which is how this box OOM'd once
  # already. drop_caches is safe (clean pages only) and takes it straight back to ~118 GB.
  sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
  sleep 3
}

free_gb() { awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo; }

run_cfg() {
  local tag=$1 levels=$2
  echo "################ $tag (LG_EXPERT_LEVELS=${levels:-unset})  free=$(free_gb)GB"
  kill_server
  if [ -n "$levels" ]; then export LG_EXPERT_LEVELS=$levels; else unset LG_EXPERT_LEVELS; fi
  CTX=16384 PORT=8080 ./build/lgserve > "$OUT/server_$tag.log" 2>&1 &
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
  $PY tools/eval_capability.py --suite aime --limit "$LIMIT_AIME" --max-tokens "$MAXTOK_AIME" \
      --timeout 1800 --tag "$tag" --out "$OUT/${tag}_aime.json"
  kill_server
  echo "  done, free=$(free_gb)GB"
}

run_cfg baseline ""
run_cfg lv7 7
run_cfg lv5 5

echo "################ SUMMARY"
$PY - "$OUT" <<'PY'
import json, sys, os, glob
d = sys.argv[1]
print(f"{'config':10s} {'suite':6s} {'acc':>8s} {'s/correct':>10s} {'mean_gen':>9s} {'tok/s':>7s} {'trunc':>6s}")
for f in sorted(glob.glob(os.path.join(d, "*_*.json"))):
    if os.path.basename(f).startswith("server"): continue
    s = json.load(open(f))["summary"]
    print(f"{s['tag']:10s} {s['suite']:6s} {s['correct']:3d}/{s['n']:<4d} "
          f"{s['s_per_correct']:10.1f} {s['mean_gen_tokens']:9.0f} {s['mean_tok_s']:7.1f} {s['truncated']:6d}")
PY
