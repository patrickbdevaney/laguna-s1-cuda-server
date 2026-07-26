# HARDWARE_PROBE.md — what `sm_110a` actually has, settled by compilation

`HARDWARE.md:38` says "TMEM + tcgen05 + TMA present". `RESEARCH_PROMPT_V5.md` says tcgen05 is
absent. Both cannot be right, and every kernel decision downstream depends on which. This is a
compile-only probe, so it needed no GPU time and did not disturb the running sweep.

## Method, and why the first attempt was worthless

The first pass wrote inline PTX with no operand constraints and reported four instructions
"ABSENT". **Three of those were false.** Running the identical probe against `sm_90a` and
`sm_100a` — architectures that certainly have these instructions — produced the identical
failure, which proves the probe was testing my assembly syntax, not the hardware.

**A capability probe without a positive control is not a measurement.** The control is what
separated the one real finding from three artifacts.

## Results

| feature | sm_90a | sm_100a | **sm_110a** | note |
|---|:--:|:--:|:--:|---|
| `tcgen05` (5th-gen tensor core) | — | — | **ABSENT** | explicit `not supported`, distinct from a syntax error |
| `barrier.cluster` (thread-block clusters) | — | — | **PRESENT** | |
| `cp.async.bulk` (**TMA**) | YES | YES | **PRESENT** | |
| `cvt.rn.satfinite.e4m3x2.f32` (FP8) | YES | YES | **PRESENT** | hardware FP8 pack, 16-bit dest |
| `cvt.rn.satfinite.e2m1x2.f32` (FP4) | no | no | **unresolved** | still an operand-width mismatch; dest is likely `.b8`, or it is only reachable via `cuda_fp4.h`. Not concluded either way. |

## What this changes

* **`HARDWARE.md:38` is wrong about tcgen05** and should be corrected. It is right about TMA.
* **TMA and thread-block clusters are available and we use neither.** Both are relevant to the
  attention GEMV work: TMA can stage weight tiles without burning registers on addressing, and
  cluster barriers are a cheaper synchronisation primitive than the `grid.sync` that was blamed
  for blocking the megakernel — though the latency research independently found `grid.sync` was
  never the real blocker, and that the megakernel ceiling here is ~1.8% regardless.
* **Hardware FP8 conversion exists**, which matters if the W4A16 attention kernel needs to
  produce FP8 intermediates.
* The FP4 conversion question stays **open**, and is recorded as open rather than guessed. It
  bears directly on the NVFP4 attention kernel's dequant cost.
