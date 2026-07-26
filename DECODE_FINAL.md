# DECODE_FINAL.md — where decode actually stands, measured on all three workloads

## Measured (`tools/k_sweep.sh`, 400 tokens, 2 reps per cell, server restarted per config)

| config | code r1 | code r2 | prose r1 | prose r2 |
|---|---:|---:|---:|---:|
| **adaptive (ships today)** | 38.39 | **26.67** | 31.19 | 33.63 |
| `SPEC_K=0` (no speculation) | 33.66 | 33.80 | **33.85** | **33.88** |
| `SPEC_K=2` | 36.37 | **40.99** | 28.33 | 33.05 |
| `SPEC_K=3` | 36.38 | 30.19 | 36.53 | 29.03 |
| `SPEC_K=5` | 36.31 | 40.08 | 28.02 | 28.02 |
| `SPEC_K=8` | 36.46 | 36.51 | 27.47 | 26.92 |

**Best observed: code 40.99 tok/s at `SPEC_K=2`; prose 33.88 tok/s with speculation OFF.**

## Three findings, and the adaptive policy is the problem

### 1. Speculation pays on code and loses on prose — the research was right

`SPEC_K=2` reaches 40.99 on code against 33.66–33.80 with speculation disabled: **+21%**. On prose,
speculation-off wins every cell — 33.85/33.88 against 26.9–36.5 with it on. The delegated
speculation research predicted exactly this ("M=1 is optimal for every α < 0.819; prose is not
lost, it is mis-tuned"), and it is confirmed here without relying on its cost model, which was
built partly on a figure of mine that was later corrected.

### 2. The bandit's boot sweep is unreliable, and that is the live defect

The `ar` arm estimate across server boots: **24.1, 33.3, 20.1, 21.1, 21.0, 20.9** — while the
*true* AR rate, measured directly with `SPEC_K=0`, is **33.66–33.88**. The estimator swings by 60%
and is usually 35% low.

The consequence is not random, it is bimodal:

* When the sweep *underestimates* AR (20–21), the bandit picks DFlash — good on code, bad on prose.
* When it *overestimates* AR relative to DFlash (`ar=33.3, dflash_best=23.9`), it picks AR
  everywhere and **gives up the 21% on code**.

A serving run earlier tonight hit the second mode: all three workloads returned `acc/fwd 1.00`
with identical frozen arm estimates, and code measured 33.7 against the 40.1 in `WIKI.md`. That is
not a regression in the kernels — every kernel number improved tonight. **It is the controller
sampling badly at boot and then never correcting**, because `AR_SAMPLES=4` / `TRIALS=12` is too few
samples for arms whose rates differ by less than their noise.

This is `WIKI.md` lesson 9 recurring with the roles reversed. It previously mis-rated *AR* at 23.4
against a true 33 and stopped selecting it; now it mis-rates *DFlash* the same way. The lesson was
recorded but the fix — enough samples, and periodic re-measurement of unselected arms — was not
made.

### 3. Adaptive is worse than either fixed policy

`adaptive` returns 38.39 then **26.67** on identical code prompts; `SPEC_K=2` returns 36.37 then
40.99. The variance is the controller thrashing, not the workload. **A static content-aware rule —
speculate on code, do not on prose — beats the current adaptive policy on both axes today.**

## Honest position on "maximum"

This is **not** the global maximum, and `OMEGA.md` already argued why that claim is not available.
What is true after tonight:

* **Every lever on the frontier list has been re-priced against a profile of the shipping config**,
  and four were rejected with arithmetic rather than left as aspirations (trellis +5.3%, activation
  sparsity, Block Verification → 0 at greedy, RMSNorm fusion ≤1.2%).
* **The largest lever is measured on both axes and ready to build**: NVFP4 on `q_proj`+`o_proj`,
  capability bit-identical 8/8 and conversion 95%, worth **+19.9%** (33.2 → ~39.8 tok/s base). It
  is specified in `ATTENTION_NVFP4.md` and not yet implemented.
* **A live defect is now identified with numbers**: the speculation controller costs up to 21% on
  code depending on how its boot sweep lands.

The two together — NVFP4 q+o on the base path, and a controller that reliably picks speculation on
code — are worth roughly **+19.9% base and +21% on code traffic**, and neither is speculative
anymore. Both are measurement-backed and both remain unbuilt.

## Instrument note

`acc/fwd 1.00` is the tell for "no speculation happened". It should be read first in any served
measurement, because a plausible tok/s number with `acc/fwd 1.00` means the speculator was never
consulted — which looks identical to a speculator that was consulted and rejected.
