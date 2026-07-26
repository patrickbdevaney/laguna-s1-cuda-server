# WIKI.md — the whole project, indexed

Pure-CUDA/C++ inference for **`poolside/Laguna-S-2.1-NVFP4`** (117.6 B total / 8.5 B active MoE)
with the **`Laguna-S-2.1-DFlash-NVFP4`** speculator, on a **Jetson AGX Thor** (Blackwell
`sm_110a`, 20 SMs, 122 GB unified LPDDR5X). No Python on the hot path.

This file is the map. Every claim below points at the document that carries the evidence.

---

## 1. Where it ended up

| | value |
|---|---:|
| base decode | **33.2 tok/s** (from 3.66 at the first working forward) |
| `B_tok` | **6.251 GB/token** (from 10.044) |
| served — prose | 33.6 tok/s |
| served — code | 40.1 tok/s |
| served — repetitive / edit-style | 49.7 tok/s at 14.6 accepted/forward |
| context | 262 144 default, KV 6.48 GB, process 78 GB of 122 |

**Reference point:** poolside's own measurement on the closest hardware analogue (DGX Spark
GB10) is 13–14 tok/s unspeculated and 22–24 with DFlash. Published full-step batch-1 bandwidth
efficiency tops out at 82 % (FlashFormer, H100); we are at ~81–85 %.

All gates green: **H1 · R1 · A1 · L1 · B1 · B1c · D1 · S1**. Greedy output is bit-exact against
the oracle at every stage.

## 2. Documents

| file | what it holds |
|---|---|
| `DIRECTIVE.md` | the original operating directive, verbatim |
| `RESCOPE.md` | 12 corrections to its factual premises, with evidence |
| `ROOFLINE.md` | byte budget, k-sweep, KV policy |
| `MODEL_INVENTORY.md` | every tensor, the NVFP4 layout, tokenizer, chat grammar |
| `ARCH_DELTA.md` | gemma-4 vs Laguna per subsystem, with port verdicts |
| `LOOP_LOG.md` | **gate-by-gate record — what was checked, against what, verdict** |
| `OPTIMIZATION_LOG.md` | **the won/lost/neutral ledger, 33 entries** |
| `COST_MODEL_CORRECTION.md` | **a model of mine that was physically impossible, and its fix** |
| `RESEARCH_PROMPT_V2/V3/V4.md` | the research prompts, each built from what the last one taught |
| `RESEARCH_FINDINGS_V2/V3/V4.md` | what came back, including what refuted us |
| `IMPLEMENTATION_PLAN.md` | the ranked plan and what has been executed |
| `OMEGA.md` | **honest answer to "is this the global maximum?" — no, and why** |

## 3. The arc, in order

| stage | result |
|---|---|
| first working forward | 3.66 tok/s |
| FP4 dequant via the HW converter, flash-decoding attention split, MoE grid sized by actual M | 9.23 (+152 %) |
| offline expert repack, thread-per-output | 16.57 (+70 %) |
| whole-step CUDA graph + doubling re-capture bound | 18.74 |
| FP8 attention (W8A16, per-output-row scale) | 20.44 |
| **4 warps/block** — the 24-CTA limit made half the warp slots unreachable | 24.59 (+13 %) |
| 16-byte FP8 weight loads; segmented GEMM (q\|k\|v\|g, gate\|up) | 27.35 |
| **NVFP4 scale layout `[grp][lane]` → `[grp/8][lane][8]`** | 30.8 (+11 %) |
| FP8 `lm_head`, shared experts, layer-0 dense | **33.0 (+52 % cumulative)** |
| fused `moe_invert` (7 launches → 1) | 33.2 |

## 4. Things that cost a gate or a measurement to learn

The full list is in `README.md`; these are the ones that generalise beyond this project.

1. **A tuning constant justified by a measurement is only valid under that measurement's
   conditions.** Moving to 4 warps/block silently invalidated the FP8 load-width choice made at
   1 warp/block, and nothing in the code said so.
2. **Never hardcode a byte budget a build flag can change.** `B_tok` was pinned at the BF16
   figure, so we kept crediting ourselves with attention bytes FP8 had already removed — 83 % of
   roofline reported where the truth was 66 %.
3. **Short microbenchmarks on this part measure the idle clock.** The same shape gave 0.194 ms
   and 0.627 ms in two processes with identical code. Spin 300 ms first, and discard the first
   sample of every A/B.
4. **The first kernel over a fresh allocation is not measurable** — page-table warmup made a
   182 GB/s shape read as 56.
5. **A microbenchmark that varies one launch parameter at fixed grid is measuring total
   threads**, not threads per block. Check whether the target kernel's thread count is free or
   fixed by its problem shape before transferring the result.
6. **`extern "C"` does no type checking.** A stale declaration in one translation unit linked
   against a changed definition and put the CUDA stream in the wrong argument slot — segfault,
   no diagnostic. All 38 declarations now live in one header.
7. **On an issue-bound kernel, GB/s is a misleading metric.** FP8 at M=6 reads half the bytes of
   BF16 in less wall-clock time and therefore *reads* as half the bandwidth. Compare times.
8. **A correctness test cannot see a broken speculator.** A scratch-buffer overflow made the
   draft emit one constant token forever and the correctness gate *passed*, because every draft
   token was rejected. τ = 1.000 was the only signal.
9. **An arm that is never selected is never re-measured.** A single cold sample rated plain
   decode at 23.4 tok/s against a true 33; the controller then stopped choosing it, so the
   estimate could never correct itself.
10. **An online cost model must only be fed measurements from the regime it decides in.** One
    sampled request — which pays a full softmax in the same branch — permanently mis-priced every
    greedy request after it.
11. **A model fitted to times and read as bytes will assert traffic the hardware does not
    contain.** See `COST_MODEL_CORRECTION.md`.

## 5. Closed by measurement — do not re-propose without a contradicting number

**Hardware / platform:** wave quantization and persistent CTAs (bandwidth is flat from 240 to
20 000 blocks); compressible memory (`GENERIC_COMPRESSION_SUPPORTED = 0`); EMC clock pinning
(29.46 pinned vs 29.37 default); MAXN (~0 %, we draw 62.7 W of a 120 W cap); L2 persistence
(≤0.33 % ceiling); GEMV access-pattern tuning (`uint4` + `__ldcs` already optimal, `__ldcg` is a
21 % regression); zero-copy mapped weights (160 vs 254 GB/s).

**Kernels:** shared-memory activation staging (1.8× slower); N-blocking (neutral); MoE K-split
(neutral — partial traffic cancels the occupancy gain); gate/up on separate warps (−1.3 %);
MoE threads/block 128→256 (neutral, thread count is fixed by the problem); grid-cooperative
megakernel with `grid.sync` (~35 % of token time in the barrier); FlashNorm folding (not
bit-exact); tcgen05/TMA/DSMEM/split-K.

**Quantization:** attention below FP8 (five 2026 production recipes exclude it; damage scales
down with active params and we are at 8.5 B); `g_proj` in anything; lossless entropy coding of
the weight streams — refuted in **both** the variable-length and fixed-length forms (break-even
needs <72 % compression; our FP8 weights measure 82.2 % entropy).

**Speculation:** tree/multi-candidate verify; drafter quantization *as a replacement drafter*;
EAGLE-3/Medusa/MTP swap; expert budgeting (lossy); expert prefetch (assumes PCIe offload);
routing prediction (95.0 % → 57.6 % GSM8K); staged verify; Lookahead/Jacobi (0.66×); a
throughput bandit; **certified expert skipping** — killed not by Lipschitz blow-up but by
routing flips, since a sound certificate must cover 48 × 246 = 11 808 discrete top-10-of-256
gate comparisons per token.

## 6. Measurements nobody else has

> **Three claims in this section were wrong and are corrected below.** They were repeated across
> several documents and into research prompts, where they steered effort for weeks. Two were
> priority claims ("nobody else has this") that a literature check refuted; one was a measurement
> generalised past the regime it was taken in.

* **`E_frac(M)`** — the fraction of 256 experts a forward touches: 0.039 / 0.141 / 0.264 at
  M = 1 / 5 / 16, i.e. **22 % below independent routing at M=5 and 44 % below at M=16**.
  ~~The most position-correlated MoE routing measured anywhere, and no paper publishes this curve
  for any model.~~ **Both halves of that are false.** The curve *is* published (Cohere, EcoSpec,
  MoE-Spec, ST-MoE), and Cohere measures 20–31 % below independence against our 21.9 % — we are
  ordinary, not extreme. **What survives is narrower and still useful: no published *cost model*
  incorporates the correlation.** MoESD's `N(t) = E(1−(1−ρ)^t)` overstates our M=16 union by 79 %.
* **τ against the two length axes** — acceptance degrades with **prompt length** (0.817 → 0.533
  from 67 to 15 267 tokens) and is **flat-to-rising** with generated position. The literature
  conflicts on this; we measured it. *(Stands.)*
* **The dequant compute budget on Thor** — 16 ALU instructions per 32-bit word absorbed with zero
  bandwidth loss. *(Stands, in the regime it was measured: the NVFP4 shape, nibble-aligned and
  independent.)* ~~…and ≥10 shared-memory codebook lookups, which makes trellis/codebook decode
  compute-free here.~~ **Refuted at 3 bits**, which is exactly ~10.7 lookups per 32-bit word —
  the regime the claim covered. Measured: arithmetic codebook 395 Gweight/s vs shared-memory
  codebook 271, a **31 % loss**. Trellis decode is compute-**bound** here, not compute-free; it
  sits at 91 % of its own ALU ceiling. See `TRELLIS_VERDICT.md`.
  The original measurement was taken where the kernel had bandwidth stall cycles to hide lookups
  in. The 3-bit path has none — a budget measured under stalls does not transfer to a kernel
  without them.

## 7. The open frontier

| lever | estimate | state |
|---|---:|---|
| ~~Routed experts → 3.0 bpw trellis~~ | ~~+15.2 %~~ → **+5.3 %** | **REJECTED.** Measured: production NVFP4 MoE 332 Gweight/s vs trellis_B 395, applied to a 31.3 % term. The +15.2 % assumed the term stays bandwidth-bound and converts the 1.406× byte saving in full; it does not, because trellis_B sits at 91 % of its own ALU ceiling. See `TRELLIS_VERDICT.md`. |
| **Attention `q_proj`+`o_proj` → NVFP4** | **+21 %** | **capability MEASURED**: all five attention projections leave greedy output bit-identical, 8/8. Largest term in the profile (44.9 % of `B_tok`) and absent from every prior version of this table. Conversion is the open part. See `REPRICED.md`. |
| ~~Intra-expert activation sparsity~~ | ~~+24.9 %~~ | **REJECTED on this model.** At the published 87 % operating point, zeroing costs 43.7 % relative error unstructured / 83 % block-32; the bit-exact permutation rescue buys 0.9 pp. Our experts are dynamically, not chronically, sparse — every neuron is in the top-13 % for 9–18 % of tokens. `moe_intermediate`=1024 was the right thing to worry about. See `ACTIVATION_SPARSITY.md`. |
| Close the M>1 kernel inefficiency | lowers spec break-even τ 3.36 → 2.40 | **would make speculation profitable on prose** |
| Kernel fusion beyond `moe_invert` | +3.2 % ceiling | ~770 small kernels remain |
| Cassandra-style draft from truncated *target* weights | re-litigate | our DOA entry answered a different question |
| ~~Block Verification (joint accept/reject)~~ | ~~5–8 %~~ → **0 %** | **Worth exactly zero at greedy**, proven two independent ways. The 5–8 % holds only at γ=8, which the cost model says we cannot afford. |
| CSV-Decode certified `lm_head` skip | ~~+4.3 %~~ → **≤+3.1 %** | estimate predates the FP8 work; `lm_head` is now 3.0 % of the step, so a *full* skip is the ceiling |

**Out of scope rather than unattractive:** every published fix for long-context acceptance is a
drafter retrain, and DFlash is the *shipped* head for this checkpoint.

`OMEGA.md` argues this is a strong **local** maximum — roughly 2.4× ahead of the published state
of the art on comparable hardware — and states plainly what would have to be true for it to be a
global one. None of those four things holds.

---

## 9. The model, and the facts that bit us

Every one of these cost a failed gate or a measurement.

**Architecture.** 48 layers, hidden 3072, head_dim 128, vocab **100 352** (not 262 144), and
**`lm_head` is NOT tied** to the embedding. 12 global attention layers (48 heads, GQA group 6)
and 36 sliding-512 layers (72 heads, GQA group 9) in a 3:1 pattern — so head counts differ *per
layer*. Rotary is per layer *type*: yarn θ=5e5 on 64 of 128 dims (global), default θ=1e4 on all
128 (sliding). QK-RMSNorm per head, **pre**-RoPE. A per-head **softplus** gate — unbounded —
driven by the post-layernorm input and applied before `o_proj`. 256 experts, top-10,
`moe_intermediate` 1024, shared expert 1024, layer 0 is a **dense** MLP (intermediate 12288),
routed output scaled 2.5.

**Quantization as shipped.** Only the **routed experts** are NVFP4. Attention, shared experts,
router, layer-0 MLP and `lm_head` all ship BF16 — which is why `B_tok` started at 10.04 GB rather
than the 4.5 a uniformly-quantized model would give, and why self-quantizing the remainder was
the single largest lever in the project.

**Traps.**
* `weight_global_scale` is a **reciprocal** (2688/amax). Dequant **divides**. Multiplying yields
  `absmax ≈ 3e7` and silently garbage experts.
* The router is **sigmoid**, and `e_score_correction_bias` affects **selection only** — never the
  returned weights.
* **The deployed model is bf16.** Kernel gates must compare against a bf16-activation reference:
  feeding fp32 activations flips 15/540 router selections between experts whose scores differ by
  1e-5.
* **On uniform-random token IDs this model turns 1 ulp into a different token.** The 256-way
  router is near-uniform off-distribution, so top-10 is a coin toss. Gates on random IDs are the
  strictest form of the test — and two paths that merely round differently must be compared
  against the oracle, never against each other.

**Kernel-level traps.**
* A small LUT indexed by a runtime value inside an inner loop **is local memory** until proven
  otherwise. Replacing `e2m1f`'s `float[8]` with the hardware FP4 converter was worth 2.5×.
* Any per-lane accumulator crossing a shared-memory reduction **must keep its lane axis**, or
  31 of 32 lanes are silently discarded.
* `__device__` globals are **per-module** without `-rdc=true`; an `extern __device__` in a second
  translation unit becomes its own copy, and nvcc's `#20044-D` is the only sign.
* **The attention split count must not depend on `M`.** It chooses the partition of the key axis,
  and the flash-decoding combine is a non-associative chain of fp32 rescales — 1 ulp at layer 0,
  amplified ~1.4×/layer, flips the argmax by layer 48. Sizing it from key length alone makes
  decode, prefill and speculative verify bit-identical, which is what lets a DFlash acceptance
  rate mean anything (Gate B1c).
* **The sliding ring must be larger than the window.** With `cap == window` the read arc covers
  all slots, so a token written later in the same batch aliases into it — 63 of 64 collide at
  M=64. Decode is immune; prefill and verify are not.

## 10. DFlash, as reconstructed and then corrected

DFlash is **parallel block-diffusion drafting**, not a small autoregressive model. Its K/V for
everything before the block are projected from the **target's** fused hidden states — six tapped
residual vectors from target layers [1,10,19,29,38,47], concatenated through `fc [3072,18432]`.
Every draft layer projects K/V from the *same* fused vector. Its queries are
`[bonus_token, MASK × (block_size−1)]`, and one forward denoises them all.

Laguna's differs from the gemma-era one: `causal: true` (gemma's block was bidirectional), six
`aux_hidden_norms` applied **per tap before concatenation**, a fused `qkv_proj [11264,3072]`, and
all six layers sliding-512 — so the draft's KV is 6.5 MB and **constant in conversation length**.

**Corrected against poolside's own upstream integration:** the context path must apply **each
draft layer's `input_layernorm`** to the fused vector before projecting K/V. We were projecting
from the un-normed vector. Invisible to a correctness test — the output stays exact either way,
because rejected drafts are simply replaced — and visible only as depressed acceptance.
τ 3.06 → 3.12.

**Settled empirically:** the NVFP4 draft config says `rope_theta` 1e4 while poolside's BF16, FP8
and INT4 draft checkpoints all say 5e5. We measured: 1e4 gives τ 3.12 against 5e5's 2.91. Follow
the config you load.

**Still resting on inference, not evidence:** whether the checkpoint was trained with
`sliding_window_base = moving_query` or the legacy `fixed_anchor` mask. Not recorded anywhere in
the shipped artifact.

## 11. The server

One session at a time on a mutex — deliberate, not a shortcut: decode is bandwidth-bound at
batch 1 on a 69 GB weight set, so a second concurrent sequence would not add throughput, it would
halve both sequences' latency.

**Prefix cache.** Only the tokens past the longest common prefix are prefilled. Correct because
the sliding ring's read arc only ever looks *backward*, so slots holding stale future positions
are never read, and the global layers are indexed by absolute position so a rewind just
overwrites.

**Speculation policy.** A cost-aware controller over {AR, DFlash, suffix}: per-position
acceptance `p[i]` by EWMA (five Bernoulli observations per k=5 step, versus one high-variance
throughput sample), expected yield in closed form, and `k* = argmax_k E[tokens|k]/cost(k)` with
plain decode scored as k=0. A step at *any* k informs the estimate for *every* k, so
steady-state exploration cost is zero.

**SuffixDecoding.** A 3–12-gram index over prompt + output whose marginal cost is a hash lookup.
It only has to beat 1.0 accepted tokens where DFlash must clear τ ≈ 3.

**Sampling.** Greedy uses longest-prefix. Under temperature the correct rule needs the draft's
distribution q — except DFlash drafts greedily, so q is a point mass and it collapses to
`accept ⟺ u < p_target(x)` with recovery from the residual. Distribution-preserving, and it needs
only target logits. **Implemented and correct; the policy on top of it is off by default**
(`LG_SPEC_SAMPLED=1`) because it oscillated 22.9/19.7/14.6 tok/s and would trade a reliable
30 tok/s for an unreliable 15–25.

**Also in:** streaming SSE with incomplete-UTF-8 held back (token boundaries do not respect
codepoint boundaries), `poolside_v1` reasoning separation and tool calling, a self-contained
WebUI, and a C++ terminal client.

## 12. Correctness invariants — do not break these

1. **Greedy output must be bit-identical to autoregressive decode**, and sampled output
   distribution-preserving. The speculative verify path is built on it.
2. **Decode, prefill and speculative verify must be the same arithmetic** (Gate B1c). Anything
   that repartitions a reduction differently by `M` breaks it.
3. **Never invent a model constant.** Read it from disk or do not write it. Every shape in the
   draft loader is checked against what the config implies, and a mismatch is a hard error.
4. **Byte accounting must follow the build flags**, or every effective-bandwidth number lies.
5. **One change per measurement**, back-to-back A/B/A, first sample discarded.

---

## 13. Build and run

```bash
nvcc -O3 -std=c++17 -arch=sm_110a -I. -o build/lgserve src/server.cu kernels/*.cu -lpthread
g++  -O2 -std=c++17 -I. -o build/lgchat tools/lgchat.cc -lpthread

CTX=262144 ./build/lgserve      # OpenAI endpoint + WebUI on :8080
./build/lgchat                  # terminal client
tools/bench_server.sh           # served decode by workload
```

Gates: `build/gate_kernels` (13), `gate_forward`, `gate_longctx` (B1c), `gate_dflash` (D1),
`gate_tokenizer`, `gate_chat`. Benches: `bench_decode`, `bench_kernels`, `bench_moe`,
`tools/tau_sweep.py` + `tools/tau_analyze.py`.

---

## 14. Corrections to this wiki

**The attention line of the byte table was wrong by exactly 2×** — recorded as 1.403 GB where
the true figure is **2.803 GB**. Verified from the shapes: 12 global layers at q/o 6144-wide plus
36 sliding at 9216-wide, with k/v 1024 and g 48/72, is `2.803 G` parameters, and at FP8 that is
2.803 GB. The components then sum to 6.163 GB, which with KV/norms/router reaches the 6.251 that
`bytes_per_token()` computes and that `206/33.0 = 6.242` independently anchors.

So **the total was always right and only the line item was wrong** — a transcription error made
while writing the research prompt, propagated into this file. It matters because it made
attention look like 22 % of the budget when it is **45 %**, which would mis-rank any future work
that used the table.

Caught by an external reviewer reconstructing the tensor shapes from the safetensors headers
rather than trusting the table. That is the second time in this project a byte-table error
survived several passes (the first was the hardcoded BF16 `B_tok`), and the lesson is the same
one, sharpened: **a total that validates does not validate its parts.**
