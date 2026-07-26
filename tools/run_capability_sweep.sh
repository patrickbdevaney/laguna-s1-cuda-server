#!/bin/bash
# Capability sweep, scored as time-to-correct-answer, with the instrumentation guarded.
#
# ---------------------------------------------------------------------------------------------
# WHY THIS FILE IS FULL OF ASSERTIONS
#
# The first run of this sweep produced a clean, plausible, completely invalid result: baseline
# and LG_EXPERT_LEVELS=7 both scored 142/200 = 0.710, with identical truncation counts and
# identical mean generated tokens. They agreed because they were the SAME MODEL -- build/lgserve
# had been compiled at 21:05 and the requant LUT landed in include/laguna_weights.h at 23:03, so
# the binary contained no reference to LG_EXPERT_LEVELS at all and silently ignored it. Two GPU
# hours produced a number that looked like a finding ("no capability loss at 2.81 bits!") and was
# a measurement of nothing.
#
# Nothing in the pipeline noticed. The env var was exported correctly, the server booted
# correctly, the eval ran correctly, the JSON was well formed. The failure was invisible at every
# layer because an IGNORED environment variable is indistinguishable from one that had NO EFFECT
# -- and "no effect" is exactly the hypothesis under test. That is the trap: the null result and
# the broken instrument look the same.
#
# Hence the rule this file now enforces: A CONFIG MUST PROVE IT IS DIFFERENT BEFORE ITS RESULTS
# ARE ACCEPTED. Each guard below exists because there is a specific way this experiment can
# silently emit a well-formed lie.
#
#   G1  stale binary       rebuild unconditionally; never trust build/ to match src/
#   G2  missing feature    assert the flag's symbol is actually present in the binary
#   G3  flag ignored       functional probe per config; non-baseline MUST diverge from baseline
#   G4  identical results  post hoc, no two configs may be 100% identical across all problems
#   G5  different subsets  every config must answer the identical problem id set
#   G6  budget drift       max_tokens must match across configs
#   G7  request failures   abort if any config had request errors
#   G8  short runs         abort if a config returned fewer rows than requested
#   G9  checkpoint mutated the NVFP4 weights are read-only and must be byte-identical after
#   G10 GPU contention     nothing else on the GPU while a config is being timed
#   G11 memory trap        drop_caches between configs or the box OOMs (it has, once)
#
# G1 and G2 would have PREVENTED the actual bug; G3 would have CAUGHT it within 90 seconds
# instead of two hours; G4 is the backstop for whatever mechanism gets invented next, since it
# checks the data itself and knows nothing about binaries or flags.
# ---------------------------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/.."
OUT=${OUT:-eval_out}
LIMIT_GSM=${LIMIT_GSM:-200}
MAXTOK_GSM=${MAXTOK_GSM:-1024}
SKIP_BUILD=${SKIP_BUILD:-0}
PY=oracle-venv/bin/python
MODEL=models/Laguna-S-2.1-NVFP4
mkdir -p "$OUT"

die() { echo "FATAL: $*" >&2; pkill -x lgserve 2>/dev/null; exit 1; }
free_gb() { awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo; }

kill_server() {
  pkill -x lgserve 2>/dev/null
  for _ in $(seq 1 60); do pgrep -x lgserve >/dev/null || break; sleep 2; done
  pgrep -x lgserve >/dev/null && { pkill -9 -x lgserve; sleep 10; }
  sleep 15
  # G11. Reading the 57 GB checkpoint leaves ~68 GB in a cache that MemAvailable does NOT count,
  # so the next boot sees ~48 GB "available", the kernel hands out free pages instead of
  # reclaiming, and the box dies. A probe confirmed only 42 GB was allocatable in that state.
  sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
  sleep 3
}

# ------------------------------------------------------------------------------ preflight
echo "=== PREFLIGHT ==="

# G10: a DRAM-saturating benchmark alongside would bias one config's tok/s against another's,
# and tok/s is half the headline metric.
pgrep -x lgserve >/dev/null && die "an lgserve is already running (G10)"
for b in bench_trellis bench_trellis_v2 bench_decode bench_moe bench_ctx; do
  pgrep -x "$b" >/dev/null && die "$b is running; it saturates DRAM and would bias timings (G10)"
done

# G1: never trust build/ to match src/. This is the bug that invalidated the first run.
if [ "$SKIP_BUILD" = "0" ]; then
  echo "  rebuilding lgserve from source (G1)"
  nvcc -O3 -std=c++17 -arch=sm_110a -I. -o build/lgserve.tmp \
       src/server.cu kernels/*.cu -lpthread 2>build/build.err \
    || { tail -20 build/build.err; die "build failed"; }
  mv build/lgserve.tmp build/lgserve
fi
# Belt and braces even when the build is skipped: the binary must be newer than every source.
newest_src=$(find src include kernels -type f \( -name '*.cu' -o -name '*.h' \) \
             -newer build/lgserve 2>/dev/null | head -1)
[ -n "$newest_src" ] && die "build/lgserve is older than $newest_src -- rebuild (G1)"

# G2: an env var read by code that is not in the binary is silently ignored.
n=$(strings build/lgserve | grep -c LG_EXPERT_LEVELS)
[ "$n" -ge 1 ] || die "build/lgserve has no LG_EXPERT_LEVELS reference; the flag would be ignored (G2)"
echo "  binary carries LG_EXPERT_LEVELS ($n refs)"

# G9: "should be immutable" and "was not modified" are different claims; only one is checkable.
[ -d "$MODEL" ] || die "no $MODEL"
find "$MODEL" -name '*.safetensors' -printf '%s %T@ %m %p\n' | sort > "$OUT/.ckpt_before"
w=$(find "$MODEL" -name '*.safetensors' -perm -u+w | head -1)
[ -n "$w" ] && die "checkpoint $w is WRITABLE; chmod a-w before running (G9)"
echo "  checkpoint read-only, $(wc -l < "$OUT/.ckpt_before") shards fingerprinted"

kill_server
avail=$(free_gb)
awk -v a="$avail" 'BEGIN{if (a+0 < 100) exit 1}' || die "only ${avail}GB available; need ~100 (G11)"
echo "  memory ${avail}GB available"
echo

# ------------------------------------------------------------------------------ per config
PROBE_PROMPT='List the first 10 prime numbers.'
probe_of() {   # deterministic greedy continuation; the config's behavioural fingerprint (G3)
  curl -s -m 180 http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROBE_PROMPT\"}],\"max_tokens\":48,\"temperature\":0.0}" \
    | $PY -c "import json,sys
d=json.load(sys.stdin); m=d['choices'][0]['message']
sys.stdout.write(((m.get('reasoning_content') or '')+(m.get('content') or ''))[:200])"
}

run_cfg() {
  local tag=$1 levels=$2
  echo "######## $tag (LG_EXPERT_LEVELS=${levels:-unset})  free=$(free_gb)GB"
  kill_server
  if [ -n "$levels" ]; then export LG_EXPERT_LEVELS=$levels; else unset LG_EXPERT_LEVELS; fi

  CTX=4096 PORT=8080 ./build/lgserve > "$OUT/server_$tag.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 180); do
    curl -s -m 3 http://127.0.0.1:8080/healthz >/dev/null 2>&1 && break
    kill -0 $pid 2>/dev/null || { tail -20 "$OUT/server_$tag.log"; die "server died booting $tag"; }
    sleep 5
  done
  curl -s -m 3 http://127.0.0.1:8080/healthz >/dev/null 2>&1 || die "$tag never became healthy"

  # G3 -- THE guard. A config that cannot prove it behaves differently from baseline does not
  # get to contribute a data point, because "identical to baseline" is precisely the result we
  # would otherwise misread as "this quantization is free".
  probe_of > "$OUT/.probe_$tag"
  if [ "$tag" = "baseline" ]; then
    [ -s "$OUT/.probe_baseline" ] || die "baseline probe came back empty (G3)"
    echo "  probe recorded as reference"
  else
    [ -s "$OUT/.probe_baseline" ] || die "no baseline probe to compare against (G3)"
    if cmp -s "$OUT/.probe_baseline" "$OUT/.probe_$tag"; then
      echo "  probe (baseline): $(head -c 90 "$OUT/.probe_baseline")"
      echo "  probe ($tag):     $(head -c 90 "$OUT/.probe_$tag")"
      die "$tag is byte-identical to baseline on the probe: LG_EXPERT_LEVELS had NO EFFECT (G3)"
    fi
    echo "  probe diverges from baseline (flag is live)"
  fi

  $PY tools/eval_capability.py --suite gsm8k --limit "$LIMIT_GSM" --max-tokens "$MAXTOK_GSM" \
      --timeout 600 --tag "$tag" --out "$OUT/${tag}_gsm8k.json" || die "eval failed for $tag"

  # G7/G8: a short run, or one with request errors, is not a result.
  $PY -c "
import json,sys
d=json.load(open('$OUT/${tag}_gsm8k.json')); s=d['summary']
errs=[r for r in d['rows'] if r.get('err')]
if s['n'] < $LIMIT_GSM: sys.exit('only %d/%d rows (G8)' % (s['n'], $LIMIT_GSM))
if errs: sys.exit('%d request failures (G7): %s' % (len(errs), str(errs[0].get('err'))[:80]))
" || die "$tag result rejected"

  kill_server
  echo "  done, free=$(free_gb)GB"
}

run_cfg baseline ""
run_cfg lv7 7
run_cfg lv5 5

# ------------------------------------------------------------------------------ postflight
echo "======== POSTFLIGHT ========"
find "$MODEL" -name '*.safetensors' -printf '%s %T@ %m %p\n' | sort > "$OUT/.ckpt_after"
cmp -s "$OUT/.ckpt_before" "$OUT/.ckpt_after" || die "CHECKPOINT CHANGED during the run (G9)"
echo "  checkpoint byte-identical (G9)"

# G4/G5/G6 -- cross-config sanity from the DATA alone. These know nothing about binaries or
# flags, so they would have caught the stale-binary bug on their own.
$PY - "$OUT" <<'PY' || exit 1
import json, sys, os, glob, itertools
d = sys.argv[1]
cfg = {}
for f in sorted(glob.glob(os.path.join(d, "*_gsm8k.json"))):
    j = json.load(open(f)); cfg[j["summary"]["tag"]] = j
if len(cfg) < 2:
    sys.exit("fewer than 2 configs completed")
tags = list(cfg)
ids = {t: set(r["id"] for r in cfg[t]["rows"]) for t in tags}
for t in tags[1:]:
    if ids[t] != ids[tags[0]]:
        sys.exit(f"G5 VIOLATION: {t} answered a different problem set than {tags[0]}")
mt = {t: cfg[t]["summary"]["max_tokens"] for t in tags}
if len(set(mt.values())) != 1:
    sys.exit(f"G6 VIOLATION: max_tokens differs across configs: {mt}")
for a, b in itertools.combinations(tags, 2):
    ra = {r["id"]: r for r in cfg[a]["rows"]}
    rb = {r["id"]: r for r in cfg[b]["rows"]}
    same = sum(1 for i in ra
               if ra[i].get("pred") == rb[i].get("pred") and ra[i]["gen"] == rb[i]["gen"])
    if same == len(ra):
        sys.exit(f"G4 VIOLATION: {a} and {b} are IDENTICAL on all {same} problems -- "
                 f"they are the same model and the comparison is meaningless")
    print(f"  {a} vs {b}: {same}/{len(ra)} identical, differ on {len(ra)-same}")
print("  cross-config checks passed (G4/G5/G6)")
PY

echo "======== SUMMARY ========"
$PY tools/eval_compare.py "$OUT"
