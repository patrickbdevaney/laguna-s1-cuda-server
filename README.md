# laguna-s1-cuda-server

Pure-CUDA/C++ inference for **`poolside/Laguna-S-2.1-NVFP4`** (117.6 B total / 8.5 B activated)
with the **`Laguna-S-2.1-DFlash-NVFP4`** speculative draft, on a **Jetson AGX Thor**
(Blackwell `sm_110a`, 20 SMs, 122 GB LPDDR5X unified). No Python on the hot path.

Ported from `~/gemma-cuda-hybrid` (read-only reference). Operating directive: `DIRECTIVE.md`;
corrections to its factual premises: `RESCOPE.md`.

## State

| gate | status |
|---|---|
| **H1** capacity | ✅ 122 GB / 115 GB available — viable |
| **R1** roofline | ✅ `B_tok` = 10.04 GB/token, AR wall 22.6 tok/s |
| **A1** oracle + arch delta | ✅ bit-exact vs shipped `modeling_laguna.py`; golden tensors from real weights |
| **L1** loader | ✅ 71.899 GB in one arena, 44.6 s cold, 4.92 GB peak RSS |
| **B1** kernels G1–G9 | ✅ 13/13 kernel gates; full forward **greedy-exact**; CUDA graph capture |
| **D1** DFlash + k-sweep | ⏳ |
| **S1** server layer | ⏳ |

**Current decode: 18.90 tok/s median** (ctx 4096, greedy, no speculation) = 189.8 GB/s effective
= **84 % of the measured 227 GB/s ceiling**. Prefill 55 tok/s.
Reference point: poolside measured **13–14 tok/s** for this model on a DGX Spark (same
bandwidth class) under vLLM, and **22–24 tok/s** with DFlash.

## Layout

```
DIRECTIVE.md        the original operating directive, verbatim (authority on intent)
RESCOPE.md          corrections to its factual premises, with evidence
ROOFLINE.md         byte budget, k-sweep, KV policy, measured E_frac
MODEL_INVENTORY.md  every tensor, the NVFP4 layout, tokenizer, chat grammar
ARCH_DELTA.md       gemma-4 vs Laguna, per subsystem, with port verdicts
HARDWARE.md         Gate H1, measured chip/toolchain state
LOOP_LOG.md         gate-by-gate record: what was checked, against what, verdict
OPTIMIZATION_LOG.md won/lost/neutral ledger + EV-ranked backlog
STATUS.md           resume point

include/            laguna_config.h (all constants read from config.json)
                    laguna_weights.h (arena loader), safetensors.h, third_party/
kernels/            laguna_kernels.cuh, gemm.cu, elementwise.cu, attention.cu, moe.cu
src/                forward.cu (the model)
oracle/             reference transcription + golden-tensor generation (validation only)
tests/              gate_* (correctness), bench_* (performance), bw/h2d probes
tools/              roofline.py, captured safetensors headers
```

## Build and run

```bash
nvcc -O3 -std=c++17 -arch=sm_110a -o build/gate_kernels \
  tests/gate_kernels.cu kernels/*.cu
./build/gate_kernels                       # 13 kernel gates

nvcc -O3 -std=c++17 -arch=sm_110a -o build/bench_decode \
  tests/bench_decode.cu kernels/*.cu
CTX=4096 N=40 ./build/bench_decode         # median decode + greedy re-check
CTX=4096 N=40 LG_PROF=1 ./build/bench_decode   # per-category profile
```

The oracle (validation only, never on the serving path) needs its venv:
`oracle-venv/bin/python oracle/validate_matrix.py`.

## Facts worth knowing before changing anything

Each of these cost a failed gate or a measurement to establish; details in `LOOP_LOG.md` and
`OPTIMIZATION_LOG.md`.

1. **Only the routed experts are NVFP4.** Attention, shared experts, router, layer-0 MLP and
   `lm_head` are BF16 — 8.03 GB of the checkpoint, and 5.61 GB of every decode step. `B_tok`
   is 10.04 GB, not the 4.5 GB a uniformly-quantized model would give.
2. **`weight_global_scale` is a reciprocal** (`2688/amax`). Dequant **divides**. Multiplying
   yields `absmax ≈ 3e7` and silently garbage experts.
3. **`lm_head` is not tied** to the embedding, and `vocab` is 100 352, not 262 144.
4. **`num_experts_per_tok` is 10**, and the router is **sigmoid**, with
   `e_score_correction_bias` affecting **selection only** — never the returned weights.
5. **Head counts differ per layer**: 48 on the 12 global layers, 72 on the 36 sliding ones,
   so GQA groups are 6 and 9. Rotary is per layer *type*: yarn θ=5e5 on 64 of 128 dims
   (global), default θ=1e4 on all 128 (sliding).
6. **The deployed model is bf16.** Kernel gates compare against a bf16-activation reference,
   not the fp32 oracle — feeding fp32 activations flips 15/540 router selections between
   experts whose scores differ by 1e-5.
7. **A small LUT indexed by a runtime value inside an inner loop is local memory.** Replacing
   `e2m1f`'s `float t[8]` with the hardware FP4 converter was worth 2.5× on that kernel.
8. **Any per-lane accumulator crossing a shared-memory reduction must keep its lane axis**, or
   31 of 32 lanes are silently discarded.
9. **`cudaMalloc` streams at 227 GB/s; a `cudaHostRegister`'d mmap only 160.** Weights are
   never mapped.
10. **The profile outranks the byte budget, and the bottleneck moves.** Attention is 56 % of
    `B_tok`; the MoE is 25 %. Before the expert repack the MoE was **81.5 %** of the step and
    attention 11.6 %; after it they are **37 % each**. Rank levers by bytes, then re-rank by
    profile after every win.
11. **Repack trades warp count for iterations-per-warp.** It won +70 % on the MoE (3
    iterations/warp) and LOST 33 % on the BF16 GEMMs (12 iterations/warp, and small-N ones
    collapse to 8–32 warps for the whole GPU).
12. **`__device__` globals are per-module without `-rdc=true`.** An `extern __device__` in a
    second translation unit silently becomes its own copy; nvcc's warning `#20044-D` is the
    only sign. Pass device state by pointer instead.

## KV sizing

Sliding layers use a 512-slot ring, so 36 of 48 layers cost a constant **1.05 MB/sequence**
forever. Only the 12 global layers scale, at **24 576 B/token** — a 4× structural win.

**Allocation is free with respect to decode speed; only *used* context costs bandwidth**, and
speculation halves even that (KV is read once per verify block). At 175 GB/s: 128 K context
keeps **89 %** of peak decode, 256 K keeps 80 %. Budget after weights is ~52 GB → **~1.4–1.9 M
tokens** of capacity, roughly 2× what poolside reports at the 256 K setting.

Default `CTX` = 262 144; ceiling 1 048 576 behind the documented rope override.
