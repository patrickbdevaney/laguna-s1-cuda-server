#!/bin/bash
# Does forcing a fixed speculation length beat our adaptive bandit?
#
# WHY. The spec-decode research claims that under our own cost model M=1 is optimal for every
# alpha < 0.819, and that shipping M>=5 loses ~17% on code and ~45% on prose. We do not ship a
# fixed M -- SpecPolicy runs a per-position acceptance EWMA and picks k* = argmax
# E[tokens|k]/cost(k), with an AR (k=0) arm -- so the bandit SHOULD already select M=1 when it
# wins. But the bandit's choice is only as good as its cost model, and the same research argues
# cost(M) is misattributed. If the model is wrong the bandit is confidently wrong.
#
# This bypasses the model entirely and measures served tok/s at each forced k against the
# adaptive policy, on both traffic classes, because acceptance on code is roughly double
# acceptance on prose and a single number would hide the effect being tested.
set -u
cd "$(dirname "$0")/.."
OUT=${OUT:-kswee_out}; mkdir -p "$OUT"
PY=oracle-venv/bin/python
MAX=${MAX:-300}

CODE='Write a complete C implementation of a red-black tree with insert, delete and in-order traversal. Code only.'
PROSE='Write a detailed technical explanation of how memory bandwidth limits transformer inference at batch size one.'

kill_server() {
  pkill -x lgserve 2>/dev/null
  for _ in $(seq 1 60); do pgrep -x lgserve >/dev/null || break; sleep 2; done
  sleep 12; sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true; sleep 3
}

run_one() { # $1 label, $2 extra env
  kill_server
  env $2 CTX=4096 PORT=8080 tools/gpuexcl.sh ./build/lgserve > "$OUT/srv_$1.log" 2>&1 &
  for _ in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:8080/healthz >/dev/null 2>&1 && break; sleep 4; done
  curl -s -m 3 http://127.0.0.1:8080/healthz >/dev/null 2>&1 || { echo "$1: NO HEALTHZ"; return 1; }
  for kind in code prose; do
    p="$CODE"; [ "$kind" = prose ] && p="$PROSE"
    for rep in 1 2; do
      body=$($PY -c "import json,sys;print(json.dumps({'messages':[{'role':'user','content':sys.argv[1]}],'max_tokens':$MAX,'temperature':0.0}))" "$p")
      curl -s -m 600 http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d "$body" \
        | $PY -c "
import json,sys
d=json.load(sys.stdin); t=d['timings']
print(f\"  $1 $kind r$rep  {t['decode_tokens_per_second']:6.2f} tok/s  acc/fwd {t.get('accepted_per_forward',0):.2f}  arms[{t.get('spec_arms','')}]\")"
    done
  done
}

# SPEC_K caps the bandit's range; SPEC_K=0 disables speculation outright, which is the true
# "M=1, no draft at all" baseline and the exact configuration the research claims should win on
# prose. There is no fixed-k override in the server, so k<max is still bandit-selected -- that
# is fine, because the question is which CAP produces the best served throughput.
run_one adaptive ""
run_one nospec "SPEC_K=0"
for k in 1 2 3 5 8; do run_one "k$k" "SPEC_K=$k"; done
kill_server
