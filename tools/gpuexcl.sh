#!/bin/bash
# Serialize everything that loads the model. Run as: tools/gpuexcl.sh <cmd> [args...]
#
# WHY THIS IS A LOCK AND NOT A CONVENTION. A server holds ~70 GB of a 122 GB pool, so two live
# instances is an OOM -- and this box has already died that way once, taking ssh and rustdesk
# with it. But OOM is not even the worst outcome: two models resident push the box into reclaim
# and every tok/s number measured in that window is quietly wrong. A poisoned measurement that
# still looks plausible is more expensive than a crash, because a crash is obvious.
#
# flock makes the invariant structural. Any process that forgets to take the lock is the bug;
# any process that takes it cannot overlap another, no matter what order things get launched in
# or which background watcher fires when.
LOCK=/tmp/laguna_model.lock
exec 9>"$LOCK" || { echo "cannot open $LOCK" >&2; exit 1; }
if ! flock -w "${GPUEXCL_WAIT:-14400}" 9; then
  echo "gpuexcl: timed out waiting for the model lock (held by another run)" >&2
  exit 1
fi
# Reclaim before loading: reading the 57 GB checkpoint leaves ~68 GB in a cache MemAvailable does
# not count, so a fresh load otherwise starts against ~48 GB and the kernel hands out free pages
# instead of reclaiming.
sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
sleep 2
avail=$(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo)
if [ "$avail" -lt 100 ]; then
  echo "gpuexcl: only ${avail}GB available after reclaim, refusing to load" >&2
  exit 1
fi
"$@"
rc=$?
# Hold the lock until the child is gone AND its pages are actually released.
for _ in $(seq 1 60); do pgrep -x lgserve >/dev/null || break; sleep 2; done
sleep 5
exit $rc
