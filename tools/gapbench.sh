#!/bin/bash
# Fire the trellis microbenchmark in the window where the sweep has torn one server down and
# has not finished loading the next. The GPU is idle there (model load is I/O bound), so the
# benchmark neither contends with a measured decode nor biases one config's tok/s against
# another's -- which it would if run mid-config.
cd /home/patrickd/laguna-s1-cuda-server
while true; do
  if ! pgrep -x lgserve >/dev/null 2>&1; then
    # No server: either a gap, or the sweep is finished. Either way it is safe to measure.
    sleep 2
    if ! pgrep -x lgserve >/dev/null 2>&1; then
      ./build/bench_trellis_v2 > trellis_bench.log 2>&1
      echo "ran at $(date -Is)" >> trellis_bench.log
      exit 0
    fi
  fi
  sleep 5
done
