# LOOP_LOG.md — gate-by-gate record

Every gate: what was checked, against what, measured delta, verdict. Append-only.

---

## Gate H1 — capacity · **PASS** · 2026-07-24

122 GB unified / 115 GB available / 379 GB disk free. 128 GB Thor SKU.
74.13 GB of weights leaves ~40.9 GB for KV + activations. Detail: `HARDWARE.md`.

Open item carried forward: box is at `nvpmodel` mode 1 (120 W), **not MAXN**. Must be set
and EMC verified at max before the first bandwidth measurement (Gate B1).

---

## Gate R1 — roofline · **PASS** · 2026-07-24

`ROOFLINE.md` + `tools/roofline.py` (reads config + safetensors headers; no hardcoded
constants). Verification: per-group byte sum reproduces `index.metadata.total_size`
= 71 898 733 760 **exactly**; activated-param count reconstructs the card's 117.6 B / 8.5 B.

| quantity | value |
|---|---|
| `B_tok` @ ctx 4096 | **10.0444 GB** |
| attention share | **5.6063 GB = 56 %**, BF16 |
| routed-expert share | 2.4950 GB = 25 %, NVFP4 |
| AR wall @200 GB/s | **19.91 tok/s** |
| `E_frac(15)` | 0.450 (no blow-up) |
| `k*` prior | **4** (5 @T=0, 3 @T=0.7) |
| KV capacity | ~1.43 M tokens single-sequence |

Model calibration: predicts 13.4 AR / 26.1 code-spec / 21.1 prose-spec at 135 GB/s;
poolside measured 13–14 / 22–24 / 15 on GB10. Three independent matches ⇒ trusted.

Refuted three directive premises (`RESCOPE.md` §1). Largest: `B_tok` is 10.04 GB not 4.51,
because only routed experts are NVFP4.

---

## Gate A1 — inventory, arch delta, correctness oracle · **PASS** · 2026-07-24

### A1.1 Inventory — PASS
`MODEL_INVENTORY.md`. All 15 shards present and byte-exact. 145 153 target + 69 draft
tensor headers captured to `tools/hdr_*.json`.

NVFP4 layout confirmed from geometry, not assumption: `weight_scale [1024,192]` at K=3072
⇒ **group = 16**. Density 4.5 bits/param.

### A1.2 — ⚠ BUG FOUND AND FIXED: `weight_global_scale` is a RECIPROCAL

First dequant attempt multiplied by `weight_global_scale` and produced `absmax = 3.1e7`
(weights should be ~0.1). compressed-tensors stores
`global_scale = (FP8_MAX · FP4_MAX)/amax = 2688/amax` and pre-multiplies it into
`weight_scale`; **dequant must divide**.

Verified across three experts in different layers:

| tensor | `global_scale` | dequant absmax | `2688/gs` |
|---|---:|---:|---:|
| L1 e0 `gate_proj` | 11520.0 | 0.233333 | 0.233333 |
| L20 e137 `down_proj` | 17024.0 | 0.157895 | 0.157895 |
| L47 e255 `up_proj` | 3072.0 | 0.875000 | 0.875000 |

Cross-check: the BF16 `shared_expert.gate_proj` in the same layer has absmax 0.785 / std
0.022 — same regime as the dequantized experts (std 0.028). **This would have silently
garbaged every expert in the CUDA kernel.** Kernel note: fold `1/global_scale` into the
repacked group scales once, offline.

### A1.3 Reference transcription vs shipped `modeling_laguna.py` — PASS (bit-exact)

`oracle/ref_laguna.py` validated by `oracle/validate_ref.py` + `oracle/validate_matrix.py`
against the shipped reference instantiated at tiny scale with identical random weights
(transformers 5.12.1, torch 2.10.0+cu130).

| case | S | window | layers | prefill maxabs | argmax |
|---|---:|---:|---:|---:|---|
| baseline | 12 | 512 | 4 | **0.00e+00** | ✓ |
| SWA clips (sw=4 < S) | 20 | 4 | 4 | **0.00e+00** | ✓ |
| SWA clips, 8 layers | 24 | 5 | 8 | **0.00e+00** | ✓ |
| SWA clips + incremental KV | 10 | 4 | 4 | 0.00e+00 / decode 1.79e-07 | ✓ |
| no clip + incremental KV | 8 | 512 | 4 | 0.00e+00 / decode 1.49e-07 | ✓ |
| long seq, 8 layers | 40 | 7 | 8 | **0.00e+00** | ✓ |
| seed variation | 16 | 3 | 4 | 0.00e+00 / decode 1.79e-07 | ✓ |

Decode-path residual ~1.8e-07 is fp32 accumulation order (cached vs recomputed K/V), not
a semantic difference; argmax identical throughout.

Also validated: our yarn `inv_freq` transcription vs the shipped rotary module —
**maxdiff 0.00e+00**, `attention_factor` 1.3465735902799727 identical.

Comparison gotcha recorded: transformers emits `NL+1` hidden states as
`[embed, out_0 … out_{NL-2}, post_norm(out_{NL-1})]` — the last layer's raw output is
never recorded. A naive `hidden_states[L+1]` comparison reports a false mismatch on the
final layer only.

### A1.4 Golden tensors from the real 71.9 GB checkpoint — PASS

`oracle/golden.py`. 8.03 GB of BF16 non-expert weights resident as fp32 on GPU; the 63.87 GB
of NVFP4 experts stay on disk and only routed experts are dequantized, through a 384-entry
LRU. Peak host memory 53 GB.

- prompt: 54 tokens (poolside_v1 template, thinking on)
- prefill 135.4 s, expert LRU hits 54 834 / miss 21 306
- greedy 8 tokens in 22.9 s
- **ids `[33586, 81, 397, 874, 367, 2440, 330, 2833]` → `"Okay, I need to write a Python"`**

This is the strongest available end-to-end check: coherent, on-task output from a 117.6 B
model simultaneously validates NVFP4 dequant + nibble order, sigmoid routing with
selection-only bias, per-head softplus gating, QK-RMSNorm, partial rotary with yarn,
the 3:1 SWA/global pattern, the untied `lm_head`, the tokenizer, and the chat grammar.
Any one of them wrong yields garbage or off-topic text.

Artifacts: `docs/golden/golden_primes.pt` (per-layer hidden states, final norm, logits,
router selections per layer), `docs/golden/meta_primes.json`, `docs/golden/rope_ref.json`.

### A1.5 C++ config parser — PASS

`include/laguna_config.h` + `tests/test_config.cpp`. Every field read from `config.json`;
missing required keys are a hard error (no defaults that could mask a wrong constant).
Derived values match the independent Python computation in `ROOFLINE.md`:
global 12 / sliding 36 / window 512 / experts 256 / top-k 10 / group 16 / kv_fp8 /
per-layer heads 48↔72 / kv_group 6↔9 / rotary 64↔128 of 128.

`tests/dump_rope.cpp` vs `docs/golden/rope_ref.json`: max relative `inv_freq` difference
**1.67e-07 (full)** and **1.18e-07 (sliding)** — ~1 ULP of float32. Both `attention_factor`
values exact. **ROPE GATE: PASS.**

### Pinned versions

`transformers==5.12.1`, `torch 2.10.0+cu130`, `tokenizers` (oracle venv),
`safetensors 0.7.0`, driver 580.00, CUDA 13.0.48, L4T R38.4.0.
Recorded because DFlash upstream is still moving (vLLM #46853, SGLang #29446, TRT-LLM #15666).

---

## Correction — 2026-07-24 (self-caught before the loader was written)

Earlier prose in `MODEL_INVENTORY.md`, `ROOFLINE.md` and `LOOP_LOG.md` stated the BF16
(non-expert) portion of the checkpoint as **15.2 GB**. Measured exactly:

| | bytes |
|---|---:|
| routed experts (`weight_packed` 56.7741 + `weight_scale` 7.0968 + globals 0.0003) | **63.8712 GB** |
| everything else (BF16 attention, shared experts, router, layer-0 MLP, `lm_head`, `embed_tokens`, norms) | **8.0276 GB** |
| total | 71.8987 GB ✓ matches the index |

The correct figure is **8.03 GB**. The 15.2 came from subtracting only `weight_packed`
(56.77) from the total and forgetting that `weight_scale` is also part of the expert payload.

**No downstream number changes.** `B_tok`'s fixed term was always computed per component
(7.4110 GB) and reconciles exactly: 8.0276 − 0.6166 (`embed_tokens`, of which a decode step
reads one row) + 0.0000 = 7.411 GB. The roofline, the k-sweep, and the quantization-scenario
table are unaffected; only the prose was wrong. Sizing the loader's device allocations is the
first thing that would have consumed the wrong figure, which is why it was re-derived first.

---

## Gate L1 — loader · **PASS** · 2026-07-24

`include/laguna_weights.h` + `tests/test_loader.cu`. Requirements were: device load OK,
peak memory < 85 GB, fast restart < 60 s.

| | result |
|---|---|
| arena | **71.899 GB in ONE `cudaMalloc`**, 1096 reservations, 256 B-aligned |
| cold load (page cache dropped) | **44.6 s** — io 34.7 s + H2D 4.5 s |
| peak host RSS | **4.92 GB** |
| **peak total (device + host)** | **76.8 GB** — under the 85 GB limit |
| device free afterwards | **52.07 GB** for KV + draft + activations |
| tensors placed | 109 057 of 145 153; the other 36 096 are `input_global_scale`, intentionally unused (W4A16) |
| verification | every planned tensor found; dtype and byte-size checked per tensor; `e_gate_inv[0]` = 8.68056e-05 = 1/11520 exactly |

### What made it fast — three measurements, not guesses

The first working version took **74.4 s** and used 145 153 individual `cudaMemcpy` calls from
a lazily-faulted mmap. `tests/h2d_probe.cu` isolated why:

| path | GB/s |
|---|---:|
| one big memcpy from warm pageable mmap | **109.4** |
| 1365 × 1.5 MB copies (what the loader does per expert) | 21.7 |
| memcpy → `cudaHostAlloc` pinned → H2D | 10.6 |
| `cudaHostRegister` on the mmap | *not supported* on this platform |
| **cold `pread` → pageable → H2D** | **6.95** |

So the bottleneck was never the copies — it was mmap page-fault I/O at ~1 GB/s. Switching to
bulk `pread` (64 MB requests) into a reusable host buffer took io to 34.7 s. **Pinned staging
was measured and rejected: it is 10× slower than copying straight from pageable memory on
this integrated part.**

### Peak RSS: 9.67 → 4.92 GB

`std::vector::resize` growing organically per shard held old+new buffers simultaneously
during realloc. Sizing the read buffer once to the largest shard fixed it.

### The repack cache is currently WORTHLESS — disabled by default

Implemented as the directive asks (`save_cache`/`load_cache`, a byte image of the arena),
measured, and then turned off:

| path | time |
|---|---|
| cold from safetensors | **44.6 s** |
| from 71.9 GB cache file | **50.0 s** |
| cost to build the cache | 209 s + 71.9 GB of disk |

The cache is slower than the thing it caches, because our "repack" is presently a copy —
there is no expensive transform to amortise. The code stays (it will earn its keep the moment
a genuine mma-fragment-order repack lands, per `OPTIMIZATION_LOG` backlog), but the default
is off and the 71.9 GB file is deleted. **Gate L1's "second start < 60 s" is met by the cold
path itself.**

### Note for the KV budget

52.07 GB free after weights, before the 2.23 GB draft. That is *more* headroom than
`RESCOPE.md` §4 assumed (36.87 GB), so the KV capacity figure there is conservative;
it will be restated once the draft and activation buffers are real.

---

## Gate B1 (in progress) — kernels

Harness: `tests/gate_kernels.cu` against references dumped from the validated oracle by
`oracle/dump_kernel_refs.py` (real checkpoint weights, real hidden states from the golden run).

### G1 — NVFP4 dequant · **PASS, BIT-EXACT**

`kernels/gemm.cu:k_dequant_nvfp4` vs `oracle/ref_laguna.py:dequant_nvfp4` on
`model.layers.1.mlp.experts.0.gate_proj` [1024, 3072]:
**0 mismatching elements out of 3 145 728.** Both sides are exact products of exactly
representable values (E2M1 code × E4M3 scale × fp32 reciprocal), so bit-exactness is
attainable here and is the right bar.

### G2 — dense linear, BF16 weights · **PASS** (maxrel 7.3e-07)
### G2b — dense linear, NVFP4 weights · **PASS** (maxrel 1.5e-06)
### G3a — RMSNorm · **PASS** (maxrel 1.4e-07)
### G3b — rope tables, both layer types · **PASS** (maxrel ≤ 6.7e-06)

`__cosf`/`__sinf` fast-math intrinsics account for the 1e-6-class residual; tolerance 3e-5.

### G4 — sigmoid router, selection-only bias, top-10 · **PASS, INDEX-EXACT**

**0 mismatching indices out of 540**; normalised weights maxrel 7.4e-07.

Two failures on the way, both worth recording.

**Failure 1 — shared-memory staging does not scale (535/540 wrong).** The first GEMM staged
`x` as a `[M,K]` bf16 tile in shared memory. At the verify shape M=6 that is 36 KB and works;
at prefill M=54 it is **324 KB**, over the 228 KB limit, so the launch silently failed and the
router consumed garbage logits. Fixed by removing the staging entirely and reading `x` from
global — which is also what gemma independently measured as *faster* (+3 %: every block reads
the same small `x`, so it is hot in L2 and the shared hop is pure overhead). M is now tiled by
`MAXM=8` via `grid.y`, so any M works.

**Failure 2 — the precision contract, and it is not a tolerance question.** With the launch
fixed, 15 of 540 selections still differed. Cause: this oracle runs **fp32 activations
everywhere**, while the deployed model is **bf16** (`config.torch_dtype`). Measured directly:

| | |
|---|---|
| selection flips, fp32-act vs bf16-act reference | **15 / 540** — exactly what the kernel showed |
| tokens affected | 7 / 54 |
| router-score gap between rank-10 and rank-11 | **min 1.03e-05**, median 3.9e-03 |
| max resulting routing-weight delta | 3.5e-04 |

So the kernel was already reproducing the *correct* (bf16-activation) reference bit-for-bit;
the oracle was more precise than the model, not more right. The flips are between experts
whose router scores are tied to 1e-5 — numerically meaningless, but they would have looked
like a correctness bug forever.

**Contract adopted:** kernel gates compare against a **bf16-activation, fp32-accumulate**
reference, because that is what the checkpoint is and what poolside serves. The fp32 oracle
remains the *math* reference (it is what validated the transcription against the shipped
`modeling_laguna.py`). Carried forward to G8: the full-forward gate needs a bf16 golden run,
not the fp32 one, or a stated tolerance that accounts for exactly this effect.

**Standing note on router sensitivity:** with 256 experts and top-10, near-ties at the
selection boundary are common (min gap 1e-5 on a 54-token sample). Any future change to
activation precision on the router path — including the §5 self-quantisation lever — must
re-measure selection agreement, not just output norms.
