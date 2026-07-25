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
| **B1c** batched ≡ sequential | ✅ 10/10 **bit-exact** (MAXTOK 4/16/64 × P 4…1200, above and below the window) |
| **D1** DFlash + k-sweep | ✅ greedy-exact at every k; τ 4.32; k\* = 3–4; E_frac measured |
| **S1** server layer | ✅ OpenAI endpoint, prefix cache, WebUI, C++ client, adaptive speculation |

**Current decode: 27.35 tok/s median** (FP8 attention, ctx 4096, greedy, no speculation).
Session arc 21.65 → 27.35 (+26 %) — see `OPTIMIZATION_LOG.md` #18–#23. The streaming ceiling
is **~254 GB/s** measured on a real weight shape, not the 227 GB/s `bw_probe` reported.
Prefill 60 tok/s.
**Served, end to end** (`lgserve`, adaptive speculation): **33.6 tok/s on code**,
**28–32 tok/s on prose**, with a hard floor at the autoregressive rate.

Reference point: poolside measured **13–14 tok/s** for this model on a DGX Spark (same
bandwidth class) under vLLM, and **22–24 tok/s** with DFlash.

## Run the server

```bash
nvcc -O3 -std=c++17 -arch=sm_110a -I. -o build/lgserve src/server.cu kernels/*.cu -lpthread
g++  -O2 -std=c++17 -I. -o build/lgchat tools/lgchat.cc -lpthread

CTX=262144 ./build/lgserve            # OpenAI endpoint + WebUI on :8080
./build/lgchat                        # terminal client
```

`POST /v1/chat/completions` (streaming and not), `GET /v1/models`, `GET /healthz`, WebUI at
`/`. Reasoning arrives as `reasoning_content` deltas until the model closes `</think>`, then
as `content`; tool calls are parsed with the same `poolside_v1` grammar the prompt is rendered
with. `timings.spec_arms` reports what the speculation bandit currently believes about each
arm.

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
12. **The attention split count must not depend on `M`.** It chooses the partition of the
    key axis, and the flash-decoding combine is a non-associative chain of fp32 rescales — so
    a decode step and a prefill step of the same token got different roundings. 1 ulp at
    layer 0, amplified ~1.4× per layer, flips the argmax by layer 48. Sizing the split from
    key length alone makes decode, prefill and speculative verify bit-identical, which is what
    lets a DFlash acceptance rate mean anything.
13. **On uniform-random token IDs this model turns 1 ulp into a different token.** The 256-way
    sigmoid router is near-uniform off-distribution, so top-10 is a coin toss. Gates on random
    IDs are the strictest form of the test; two paths that merely round differently must be
    compared against the oracle, never against each other.
14. **`__device__` globals are per-module without `-rdc=true`.** An `extern __device__` in a
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
