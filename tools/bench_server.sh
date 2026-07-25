#!/bin/bash
# Served end-to-end decode rate, by workload. The adaptive speculation bandit is
# content-sensitive by design -- acceptance on code is roughly double acceptance on prose --
# so a single number is not a useful summary of this server. Run both.
#
#   tools/bench_server.sh [host] [port] [max_tokens] [reps]
HOST=${1:-127.0.0.1}; PORT=${2:-8080}; MAX=${3:-500}; REPS=${4:-3}
until curl -s -m 3 "http://$HOST:$PORT/healthz" >/dev/null 2>&1; do sleep 5; done

run() {
  curl -s -m 600 "http://$HOST:$PORT/v1/chat/completions" \
       -H 'Content-Type: application/json' -d "$1" | python3 -c "
import json,sys
d=json.load(sys.stdin); t=d['timings']
print(f\"  DECODE={t['decode_tokens_per_second']:6.2f} tok/s   prefill={t['prefill_seconds']:5.2f}s\"
      f\"   acc/fwd={t['accepted_per_forward']:.2f}   arms[{t['spec_arms']}]\")"
}
mk() { python3 -c "
import json,sys
print(json.dumps({'messages':[{'role':'user','content':sys.argv[1]}],'max_tokens':int(sys.argv[2])}))
" "$1" "$MAX"; }

CODE=$(mk "Write a complete C implementation of a red-black tree with insert, delete and in-order traversal. Code only.")
PROSE=$(mk "Write a detailed technical explanation of how memory bandwidth limits transformer inference at batch size one.")

echo "=== code  (high acceptance) ==="; for i in $(seq 1 $REPS); do run "$CODE"; done
echo "=== prose (low acceptance)  ==="; for i in $(seq 1 $REPS); do run "$PROSE"; done
