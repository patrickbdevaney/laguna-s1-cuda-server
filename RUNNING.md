# RUNNING.md — long-pole jobs, where they live, and how to pick them up

These jobs outlive any agent session. They are started with `setsid` so they become their own
session leader with `ppid=1` and no controlling terminal: an ssh disconnect, a closed Claude Code
session, or a `SIGHUP` will not touch them. **A reboot will**, and this box has rebooted mid-run
once already — see "After a reboot" below.

## What is running

| job | started as | writes | how to check |
|---|---|---|---|
| capability sweep | `setsid bash tools/run_capability_sweep.sh` | `eval_out_sweep.log`, `eval_out/<cfg>_gsm8k.json` | `tail -2 eval_out_sweep.log` |
| trellis benchmark | `setsid tools/gapbench.sh` | `trellis_bench.log` | `cat trellis_bench.log` |

Confirm they are alive and detached:

```bash
for p in $(pgrep bash); do
  tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q 'run_capability\|gapbench' &&
    echo "pid=$p ppid=$(awk '{print $4}' /proc/$p/stat) tty=$(ps -o tty= -p $p)"
done
```

`ppid=1` and `tty=?` mean detached. A `ppid` pointing at a shell means it will die with that
shell.

## What each job is answering

**Capability sweep** — the gate on the whole 3 bpw expert path. GSM8K n=200 at a 1024-token
budget, three configs (`baseline`, `LG_EXPERT_LEVELS=7`, `=5`), scored as
seconds-per-correct-answer. Roughly one hour per config. See `EXPERT_BITS_EVAL.md` for why τ and
the golden gate could not answer this and why the metric is in seconds.

**Trellis benchmark** — whether a 3-bit trellis decoder stays memory-bound on 20 SMs. It waits
for a window where no `lgserve` is running and fires there, because it saturates DRAM and would
otherwise bias one config's tok/s against another's. Ready-built at `build/bench_trellis`.

## After a reboot

Nothing restarts itself. Resume with:

```bash
setsid bash tools/resume_sweep.sh > resume.log 2>&1 < /dev/null &
```

`resume_sweep.sh` skips any config whose JSON already parses and holds the full problem count,
so it costs only the missing configs. It is a **separate file on purpose**: bash reads a script
incrementally by byte offset, so editing `run_capability_sweep.sh` while it runs can make the
running shell resume at the wrong offset and execute garbage.

## Reading the results

```bash
oracle-venv/bin/python tools/eval_compare.py eval_out
```

Compares configs **paired** (McNemar, exact) rather than as independent proportions — every
config answers the identical problems, so problem difficulty cancels and the test resolves far
smaller shifts than the ±10 points an unpaired comparison could manage at n=200. It also reports
token inflation on problems *both* configs got right, which isolates "did it start rambling"
with correctness and difficulty held fixed.

## The memory trap on this box

Reading the 57 GB checkpoint leaves ~68 GB in a cache that `MemAvailable` does **not** count. The
next server boot then sees ~48 GB "available", the kernel hands out free pages rather than
reclaiming, and the box OOMs — which is how it died once. A probe confirmed only 42 GB was
allocatable in that state.

```bash
sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'   # ~48 GB -> ~118 GB free
```

Both sweep scripts do this between every config. Do it manually before launching anything large
by hand.
