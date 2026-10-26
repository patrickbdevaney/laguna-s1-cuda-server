# EXPERT_BITS_EVAL.md — requantizing the routed experts, and why τ cannot grade it

## What was built

`tools/requant_experts.py` (offline) and an in-loader path (`LG_EXPERT_LEVELS`) that coarsen the
routed experts onto a subset of the E2M1 grid **inside the NVFP4 container**. The scale is
inherited and the container is unchanged, so the CUDA kernels are untouched and the *capability*
delta is isolated from any kernel effect.

| setting | grid | payload bits | mean \|Δ\|/mean\|v\| |
|---|---|---:|---:|
| baseline | {0,±0.5,±1,±1.5,±2,±3,±4,±6} | 3.91 | — |
| `LG_EXPERT_LEVELS=7` | {0,±2,±4,±6} | 2.81 | **0.188** |
| `LG_EXPERT_LEVELS=5` | {0,±3,±6} | 2.32 | **0.301** |

The E2M1 magnitude histogram over real expert tensors is close to uniform (7.8 % … 15.0 % per
index), so the coarser grid really is the more distorting one — 55.1 % of codes move at 7 levels
against 68.0 % at 5.

**Safety:** `models/Laguna-S-2.1-NVFP4/*.safetensors` were made `chmod a-w` before any of this
ran, and verified unchanged (size, mtime, mode) afterwards. The loader transform is applied to a
scratch copy on the way in; the checkpoint is never an output path of anything.

## Measured

Gate D1, ctx 4096, 128 generated tokens, greedy:

| experts | k=2 τ | k=3 τ | k=4 τ | tok/s @k=2 | golden match |
|---|---:|---:|---:|---:|---|
| baseline (3.91 b) | 2.396 | 2.822 | 3.175 | 32.96 | 8/8 |
| 7 levels (2.81 b) | **2.612** | **2.977** | **3.368** | **35.47** | **0/8** |
| 5 levels (2.32 b) | 2.462 | 2.708 | 3.024 | 33.60 | 8/8 |

## Two findings, both about measurement

### 1. τ rises when the target is degraded — so τ cannot be a quality proxy

Requantizing the experts **increased** acceptance at every k (2.396 → 2.612 at k=2), and the
apparent +7.6 % tok/s is entirely that: 49 target forwards instead of 53. The container is
unchanged, so **not one byte was saved** — the speed came from the draft agreeing more often.

The draft was not modified. A drafter agreeing *more* with a target whose weights were just
mangled is the signature of the target becoming **more predictable**, which is what degradation
looks like from the outside. τ measures how easy the target is to imitate, and a worse model is
easier to imitate.

**Consequence for the plan:** the break-even framing — "3.0 bpw is worth +15.2 % if τ retention
exceeds 86.9 %" — cannot be checked with τ alone. τ will happily *rise* as quality falls. It has
to be gated on a capability benchmark, and any τ retention measured alongside is a lower bound on
the damage, not a measure of it.

### 2. The golden-continuation gate cannot grade a modified model

7 levels scores 0/8 and 5 levels — objectively 1.6× more distorting — scores 8/8. Under FP8
attention both fail-or-pass differently than under BF16 (`gate_forward` passes 8/8 at *both*).
That is not a quality ordering; it is a coin flip.

It is exactly what `README.md` fact 13 predicts: this model turns 1 ulp into a different token,
because the 256-way sigmoid router puts top-10 on a knife edge. An 8-token golden match answers
**"is this the same model?"** — for which the correct answer here is *no, by construction* — and
says nothing about **"is this model good?"**.

The gate is right for what it was built for (verifying that a kernel change is bit-exact) and
wrong for this. Using it here would have produced a confident, meaningless ranking.

## What this does and does not establish

**Does:** the machinery works and is safe; the distortion is quantified; and both instruments we
had to hand are unfit for the question. That is worth knowing before spending a day on an
EXL3 kernel.

**Does not:** anything about capability. The in-container emulation is also *not* a byte saving —
it measures quality at constant bandwidth. A real 3.0 bpw win requires the trellis container and
kernel, and the case for building that rests on a capability number that does not exist yet for
any MoE at these bit-widths.

## The distortion bridge: how pessimistic is the RTN proxy? (`tools/trellis_vs_rtn.py`)

"RTN is a pessimistic proxy for trellis" was an assertion. Without a number it is useless: a pass
on RTN could not be converted into a statement about trellis, and a failure could not be
attributed — we would not know whether 3 bits is too few or whether RTN is simply a bad code.

So the trellis was actually built: a Viterbi-optimal trellis-coded quantizer with a hashed
Gaussian codebook over a 10-bit state (the QTIP shape — the state shifts in `bits` new bits per
weight, so reconstruction levels depend on neighbours and the effective codebook is far larger
than 2^bits). Run on **real expert matrices**, against the same per-16 E4M3 scale structure, so
the only thing varying is the code:

| code | payload bits | rel. Frobenius error | sd (n=6) |
|---|---:|---:|---:|
| NVFP4 as shipped | 4.00 | 0 (reference) | — |
| **trellis** | **3.00** | **0.1396** | 0.0006 |
| RTN 7 levels | 2.81 | 0.2204 | 0.0106 |
| trellis | 2.00 | 0.2760 | 0.0013 |
| RTN 5 levels | 2.32 | 0.3146 | 0.0163 |

Two things fall out.

**The proxy is pessimistic by a wide margin, and now it is quantified.** Trellis at 3.00 bits has
**37 % lower error** than the RTN-7 the sweep is scoring, at only 0.19 bits more rate. Fitting the
RTN points gives 6.32 dB/bit; the trellis is 3.97 dB better at +0.19 bits, so the code itself is
worth **≈0.43 bits equivalent** — trellis@3.0 lands where RTN would need ~3.2–3.4 bits.

**Ordering is what matters operationally:** `trellis@3.0 < RTN@2.81 < trellis@2.0 < RTN@2.32`.
So if the sweep says RTN-7 holds up, **trellis at 3.0 bpw holds up with room to spare** — that is
the shipping configuration, and it is strictly the better code at strictly higher rate. Note also
that trellis@**2.0** beats RTN@2.32, which puts a genuine 2-bit expert path on the map.

0.43 bits is a **lower bound on the trellis advantage**, deliberately: the state is only 10 bits
(short constraint length), no Hadamard incoherence transform was applied, and the source is
already-NVFP4-discretized values, which is a poorly-behaved source for a Gaussian codebook. EXL3
in its real configuration has all three working in its favour. Theory says the recoverable gap is
nearer a full bit.

## Next

The capability sweep (`tools/run_capability_sweep.sh`, `tools/eval_capability.py`) — GSM8K n=150
as the primary discriminator (±3 % error bars, short traces) and AIME-2024 n=30 as the
hard-reasoning secondary (±9 %, only detects a large drop) — scored as **time-to-correct-answer**:

    s_per_correct = total_wall_seconds / n_correct

Seconds, not accuracy alone, because the whole point of dropping bits is speed: a config 15 %
faster per token that needs 30 % more reasoning tokens to arrive is a **loss**, and an
accuracy-only score hides that. Truncated traces are scored wrong on purpose.

Budgets were sized from measured trace lengths on this model (AIME ran 588 → 3459 tokens with a
tail past 14 k), not guessed — the first attempt used 2048 and truncated everything.

**Operational note — the box OOM'd once during this work.** Reading the 57 GB checkpoint leaves
~68 GB in a cache that `MemAvailable` does **not** count, so the next server boot sees ~48 GB
"available", the kernel hands out free pages instead of reclaiming, and the box dies. A probe
confirmed only 42 GB was allocatable in that state; `echo 3 > /proc/sys/vm/drop_caches` takes it
straight back to ~118 GB. The sweep now does this between every config.
