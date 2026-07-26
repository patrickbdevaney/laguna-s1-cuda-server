# RESEARCH_V5_MOE.md — the MoE expert path, priced in microseconds

Axis: `moe_experts` (31.3 % of the profiled step, **39.9 % of `B_tok`** — and **40.4 % of the real
step** by absolute streaming time, see §0) and quantization beyond what is already settled. Written
to the §6 output contract of `RESEARCH_PROMPT_V5.md`.

**Top line.** The largest untried lever on this axis is not a format. It is **intra-expert
activation sparsity**, published two months ago with **87.4 % sparsity measured on the closest
published analogue to this model**, priced here at **+12 % to +25 % end-to-end** with device-
measured warp-occupancy correction (§1.3). It is gated by the same capability sweep the project
already accepts for precision changes (§1.4), and its enabling refactor is a **bit-exact +3.5 %**
that also fixes a known warp-starvation defect (§1.6). Second: the `router` category is **4 % of
the step unaccounted for in three trivial kernels** (§4). Third and only if the first two fail: a
**+2.3 %-at-equal-error** requantization (§5.4). Everything else on this axis — expert pruning,
expert caching, MXFP4, AQLM/QuIP#, post-hoc AWQ — is **dead**, with arithmetic, in §3 and §5.

---

## 0. The arithmetic frame everything below is priced against

All of this is derived from numbers already in this repo, so it is auditable, not imported.

```
B_tok                       6.251 GB          bytes_per_token(), FP8 attn/lm_head/dense
step                        30.3  ms          33.0 tok/s production
effective BW                206.3 GB/s        = 91 % of the 227 GB/s measured ceiling

routed expert weights       4.435 Gweight     10 experts x 3 mats x 1024 x 3072 x 47 layers
NVFP4 cost                  0.5625 B/weight   4 bits + one E4M3 per 16 = 4.5 bits
expert bytes per token      2.4947 GB         = 39.90 % of B_tok   <- reproduces the profile exactly
   gate / up / down          0.8316 GB each
```

**Sub-rates measured on this device** (the comment block above `k_moe_gu_split` in
`kernels/moe.cu`, lines 663-671 — these are *our* measurements, not citations):

| kernel | resident warps | occupancy | measured |
|---|---:|---:|---:|
| gate/up, both streams per warp | 320 | 33 % | 159.6 GB/s |
| gate/up, **split** (shipping) | 640 | 67 % | **195.6 GB/s** |
| down (same inner loop) | 960 | 100 % | **222.3 GB/s** |

```
gate+up time = 1.6632 GB / 195.6 GB/s = 8.503 ms
down    time = 0.8316 GB / 222.3 GB/s = 3.741 ms
MoE weight-streaming time             = 12.24 ms   of the 30.3 ms step  = 40.4 %
average MoE rate                      = 203.8 GB/s = 89.8 % of ceiling
```

### The single most important fact on this axis

**The MoE expert term is genuinely BANDWIDTH-BOUND at 89.8 % of the measured ceiling.** This is
the exact opposite of the trellis finding, and it is *why* the trellis verdict does not
generalise. `TRELLIS_VERDICT.md` recommended against 3 bpw because the *decoder* hit an ALU
ceiling at 91 %, so fewer bytes bought nothing. Here the *streaming* is at 90 % of the memory
ceiling with a 3-ALU-op-per-weight NVFP4 unpack that is nowhere near saturated. **Any lever that
removes expert BYTES without adding ALU work converts at essentially 1:1.**

Conversion rule used throughout (the same one `REPRICED.md` used for `o_proj`):
`speedup = 1 / (1 − ΔGB / 6.251)`, with a warp-count correction where parallelism changes.

**Note on the 31.3 % figure.** `LG_PROF` serialises, so shares are distorted. By *absolute*
streaming time the MoE is 12.24 ms of 30.3 ms = **40.4 %**, not 31.3 %. Every microsecond number
below uses the absolute 12.24 ms, which is the conservative, auditable quantity. The MoE axis is
worth more than the prompt's framing assumes.

---

## 1. THE HEADLINE: intra-expert activation sparsity is the largest untried lever on this axis

### 1.1 The result

**Claim.** Pre-trained MoE experts are internally dormant: 87–91 % of each *selected* expert's
intermediate neurons can be zeroed with 95 % benchmark-score retention, with no retraining and no
weight modification.

**Source.** Park, Kim, Gu, Stoica, Cheung, *Uncovering Intra-expert Activation Sparsity for
Efficient Mixture-of-Expert Model Execution*, arXiv:2605.08575 (9 May 2026, Berkeley/Sky).
<https://arxiv.org/abs/2605.08575>

**Measured, with hardware and batch size stated:**

| model | routed params | intra-expert sparsity @ 95 % score retention |
|---|---:|---:|
| Granite-1B-A400M | 1 B | 26.5 % |
| OLMoE-1B-7B | 7 B | ~50 % |
| DeepSeek-V2-Lite | 16 B | ~75 % |
| Qwen3.5-35B-A3B | 35 B | 84.5 % |
| **Qwen3.5-122B-A10B** | **122 B** | **87.4 %** |
| Qwen3.5-397B-A17B | 397 B | ~88 % |
| Llama-4-Maverick | 400 B | 90.8 % |

Speedups: MoE-layer 1.55× (RTX 4090, batch 1–128), peak **2.5×** (RTX 4090, batch 16–64), 1.8–2.0×
(H200), 1.5–1.8× (MI355X). End-to-end **1.11× at batch 1** (H200, 1 in / 4 out) and **1.18× at
batch 1** (H200, 1 in / 1024 out), for Qwen3.5-35B-A3B.

**Our model sits exactly on the 122 B row.** 256 experts × 3 × 1024 × 3072 × 47 = **113.6 G routed
parameters**, 4.435 G active per token. Structurally this is a 114B-A5B model — the same class as
Qwen3.5-122B-A10B, which measured 87.4 %.

### 1.2 Why this converts *better* on our device than in the paper

The paper's end-to-end numbers (1.11–1.18× at batch 1) are limited by *their* hardware, not by the
method. On an H200, HBM3e delivers ~4.8 TB/s and the MoE is only ~45 % of execution; removing
expert bytes there mostly exposes other bottlenecks. Here:

* the MoE is **40.4 %** of the step,
* it is streaming at **89.8 % of a 227 GB/s ceiling**,
* and the unpack ALU is idle enough that byte removal converts nearly 1:1.

This is the rare case where our hardware makes an imported technique *more* valuable. Flag it as
such rather than discounting it.

### 1.3 Microsecond arithmetic against the 12.24 ms MoE term

`gate_proj` must stay dense (its output is what decides the sparsity), so the cost model is
`gate + (1−s)·up + (1−s)·down`. Skipping neurons also removes gate/up warps, and we have a measured
warp→rate curve for exactly this kernel, so the estimate is corrected for it
(`rate(W) ≈ 159.6 + 0.1125·(W−320)`, from the two device measurements above; down keeps its 960
warps because its parallelism is over `H`, not `MI`).

| sparsity `s` | gate/up warps | modelled rate | gate+up | down | MoE total | **saved** | **end-to-end** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (today) | 640 | 195.6 GB/s | 8.50 ms | 3.74 ms | 12.24 ms | — | — |
| 0.25 | 560 | 186.6 | 7.80 | 2.81 | 10.60 | 1.64 ms | **+5.7 %** |
| 0.50 | 480 | 177.6 | 7.02 | 1.87 | 8.89 | 3.35 ms | **+12.4 %** |
| 0.75 | 400 | 168.6 | 6.17 | 0.94 | 7.10 | 5.14 ms | **+20.4 %** |
| 0.87 | 362 | 164.3 | 5.72 | 0.49 | 6.21 | 6.04 ms | **+24.9 %** |
| 1.00 (ceiling) | 320 | 159.6 | 5.21 | 0.00 | 5.21 | 7.03 ms | **+30.2 %** |

**+24.9 % end-to-end at the sparsity level measured on the closest-sized published model.** For
comparison, the entire trellis subsystem — a Viterbi CUDA encoder, a new container, a new decode
kernel, and re-validation of eight gates — was priced at **+5.3 % to +7 %** and recommended
against. This costs no new container and no re-quantization.

### 1.4 Exactness — and the reframe that unlocks it

This is **lossy**. It is not bit-identical to the unmodified model.

But read §1 of `RESEARCH_PROMPT_V5.md` precisely: *"greedy output must be bit-identical to
autoregressive decode; sampled output must be distribution-preserving. The speculative verify path
is built on this."* That invariant constrains the **draft↔verify relationship**, not the definition
of the target. If the sparse expert path is used consistently in both the reference AR pass and the
verify pass, speculative decoding remains exactly correct — the target is simply a different (and
faster) function.

**Therefore intra-expert sparsity sits in exactly the same gate class as the expert-bit reduction
the project already contemplates.** It is not a new category of risk. This reframing is the most
valuable sentence in this document, because it moves a +25 % lever from "forbidden" to "gated the
same way as things we already do".

With the caveat `EXPERT_BITS_EVAL.md` established the hard way: **the gate must be the capability
sweep, not τ and not golden-continuation.** τ *rose* (2.396 → 2.612) when the experts were
degraded, and 5-level (more distorting) scored 8/8 golden while 7-level scored 0/8. Because "this
model turns 1 ulp into a different token — the 256-way sigmoid router puts top-10 on a knife edge",
any non-bit-exact change to an expert output propagates into a *different expert selection* in the
next layer, so short-horizon match tests are coin flips. Budget for the capability sweep, not for a
greedy diff.

A partially-*certified* variant that keeps bit-exactness is developed in §2.

### 1.5 The blocker nobody outside this repo would find: our repack layout

`k_moe_gateup_rp` / `k_moe_gu_split` address weights as

```c
wbase + ((size_t)c * RPNB + lane) * 16      // RPNB = 32 output rows per block
```

i.e. **32 output rows are lane-interleaved at 16-byte granularity**. For a fixed k-chunk `c`, lanes
0..31 read 32 consecutive 16-byte chunks = 512 contiguous bytes, one per neuron. Consequences:

* **Skipping one arbitrary neuron saves zero bytes.** You would fetch a 512-byte group and use
  16 bytes of it. Unstructured 87 % sparsity is worth **nothing** in the shipping layout.
* The layout-native sparsity unit is a **32-neuron block** (`blockIdx.y`), because skipping lanes
  16..31 of a chunk still leaves 256 contiguous bytes — coalescing survives down to 128 B.
* For `down_proj` the alignment is even better: a lane's 16-byte chunk *is* 32 consecutive
  intermediate neurons, so a skipped 32-neuron block is exactly one skipped `c` iteration in
  `k_moe_down_rp`. **No transpose is needed** — which is the usual killer for CATS-style methods
  and does not apply to us.
* NVFP4's per-16 E4M3 blocking also forces ≥16-element granularity along any reduction axis. A
  32-neuron unit is 2 whole scale groups. Everything lines up.

But at 87 % *unstructured* sparsity, P(a random 32-block contains ≥1 active neuron) = 1 − 0.87³² =
**98.8 %**. Block-structured sparsity is therefore near-worthless *unless the active neurons
cluster*. Two ways out, both cheap:

1. **Offline neuron permutation (bit-exact).** Permute the 1024 rows of `gate` and `up` and the
   1024 columns of `down` by the *same* permutation. This is an exact identity on the dense
   computation — it changes nothing except accumulation order within `down` (and not even that if
   the permutation is applied at repack time and the k-loop order follows it). Sort neurons by mean
   |SiLU(gate)·up| over a calibration set so that chronically-dormant neurons concentrate into the
   same blocks. The permutation costs zero at inference and is folded into the existing repack.
2. **Row-contiguous repack (`RPNB = 1`, warp-per-row).** A warp reads one neuron's 3072 FP4 = 1536 B
   as 3 × (32 lanes × 16 B) — still perfectly coalesced, and now arbitrary neurons can be skipped.
   This also **fixes the warp starvation**: at 13 % density there are 133 active rows × 10 experts ×
   2 matrices = 2660 warps instead of 640, so the rate correction in §1.3 flips from a penalty to a
   bonus and the +24.9 % becomes a floor rather than a ceiling.

### 1.6 A free 3.5 % that falls out of the same refactor

`gate/up` runs at 195.6 GB/s because it is **warp-starved** — `moe_intermediate` (1024) is 3× smaller
than `hidden` (3072), so it has a third of `down`'s output rows to spread over 20 SMs. This is
already documented in `moe.cu`. If the row-contiguous repack lifts gate/up to `down`'s 222.3 GB/s:

```
gate+up  1.6632 / 222.3 = 7.482 ms   (was 8.503)
MoE      11.22 ms                     (was 12.24)
saved    1.02 ms of 30.3             -> 30.3 / 29.28 = +3.5 %
```

**+3.5 %, bit-exact, no capability risk, no re-quantization** — just a different repack and a
warp-per-row GEMV. This is the cheapest item in the entire document and it is a prerequisite for
the +25 % item.

### 1.7 Cheapest falsifying experiment (no kernel work, ~1 afternoon)

Dump `h = SiLU(gate)·up` for the 10 selected experts of ~4 layers over ~500 decode tokens (a short
instrumented run, then pure numpy). Produce three curves of relative output error vs sparsity:

1. unstructured top-`s` magnitude threshold (the paper's number — does 87 % reproduce at
   `moe_intermediate = 1024`?),
2. block-32 structured,
3. block-32 after sorting neurons by mean |h|.

If curve (1) collapses below ~50 %, the whole lever is worth ≤ +12 % and the priority drops. If
curve (3) tracks curve (1) within a few points, the shipping layout needs no change at all and this
is a two-week project worth +25 %.

**Open risk we could not resolve from the literature:** every published measurement is on models
with `moe_intermediate` ≥ 1408 (OLMoE 1024 is the one exception, and it measured only ~50 %). Our
1024 is at the narrow end, and *Sparsing Law* (arXiv:2411.02335) reports that narrower FFNs carry
*higher* activation sparsity — but we could not extract its SwiGLU-specific numbers before the
search budget ran out. This single uncertainty is worth resolving first because it multiplies a
+25 % item.

### 1.8 The dense-model cousin, and why it is blocked for us

**TEAL** (Liu et al., ICLR 2025, arXiv:2408.14690, <https://www.together.ai/blog/teal-training-free-activation-sparsity-in-large-language-models>)
magnitude-thresholds the *hidden state* before **every** projection — q, k, v, o, gate, up, down.
Measured **at batch size 1**:

| sparsity | A6000 | A100 | Llama-3-8B WikiText PPL (base 5.87) |
|---:|---:|---:|---:|
| 25 % | 1.22–1.28× | 1.11–1.16× | 5.94 |
| 40 % | 1.42–1.53× | 1.25–1.34× | 6.21 |
| 50 % | 1.63–1.80× | 1.33–1.45× | 6.67 |

Calibration: 10 samples × 2048 tokens, < 1 GPU-hour. **Composes with quantization**: Figure 4 shows
TEAL stacking on 8-bit RTN, 4-bit AWQ and 2/3-bit QuIP#, with "errors from sparsity and quantization
compounding somewhat independently". No MoE evaluation.

**Why it is largely DEAD here as stated.** TEAL sparsifies along the *reduction* axis (input
channels). Our weights are NVFP4 with a per-16 E4M3 scale along that axis, so skipping an input
channel does not skip a scale group: with 25 % random channel sparsity the probability that a whole
16-group is skippable is 0.25¹⁶ ≈ 10⁻¹⁰. Unstructured input-channel sparsity is worth **exactly
zero bytes** on an NVFP4 checkpoint.

The salvageable form is **group-structured input sparsity**: threshold on ‖x_group‖ over the 16-wide
NVFP4 groups (192 groups for H = 3072) and skip whole groups. Achievable sparsity will be well below
TEAL's unstructured figures, and the same 512-byte lane-interleave issue applies. This is the
correct framing of "why our quantized checkpoint makes half the sparsity literature inapplicable",
and it is a *structural* result, not an accident of implementation.

Note also that TEAL, unlike the intra-expert method, would reach `attn_qkvo_gemm` (34.7 % / 44.9 %
of bytes) — the largest term in the budget. Flagged for the attention axis; the NVFP4/FP8 group
blocking objection applies there too.

### 1.9 Deja Vu — the predictor route, and why we should not take it

**Deja Vu** (Liu et al., ICML 2023 oral, arXiv:2310.17157) trains small MLP predictors that, from
block `i`'s input, predict block `i+1`'s contextual sparsity. OPT-175B holds accuracy to **75 %
sparsity**, up to 85 % parameter deactivation, **>2× latency reduction vs FasterTransformer**
(A100, batch 1, 175 B model).

**Verdict for us: do not build.** (a) It needs *trained* predictors — small, but training, and
`RESEARCH_PROMPT_V5.md` §1 rules out anything requiring training hardware we do not have; (b) the
2× is against FasterTransformer on a 175 B dense model where the MLP dominates far more than 40 %;
(c) the intra-expert method gets the same sparsity *from the gate itself*, with a lookup table
generated at engine startup and no predictor at all. Deja Vu's contribution here is evidence that
the 75–85 % regime is real, not a technique to import.

**R-Sparse** (Zhang et al., ICLR 2025, arXiv:2504.19449) is the training-free alternative: it
replaces linear layers with a rank-aware sparse form (input-channel sparsity + singular-value
components), reporting **43 % end-to-end improvement at 50 % model-level sparsity** on Llama-2/3
and Mistral with custom kernels. It needs an SVD-derived low-rank term per matrix — extra weights
and extra bytes per token — which is the wrong direction for a bandwidth-bound decoder. **Dead
here** unless the low-rank term can be made smaller than the bytes it saves; at 50 % sparsity of
0.83 GB it would have to cost < 0.4 GB per token, which an SVD term over 47 × 256 experts will not.

---

## 2. Certified lazy expert evaluation — can `B_tok` go sublinear *exactly*?

`RESEARCH_PROMPT_V5.md` §5 asks for a mechanism that reads **less than the active parameter set per
token, exactly**. The literature has the machinery; nobody has pointed it at an MoE.

### 2.1 What exists, and where it comes from (this is the "outside LLM serving" answer)

The canonical exact-with-certificate algorithms are from **information retrieval**, not ML:

* **MaxScore** (Turtle & Flood 1995) and **WAND** (Broder et al. 2003) — per-term upper bounds on
  score contributions let you *provably* skip documents that cannot enter the top-k. Exact:
  identical output to exhaustive scoring.
* **Block-Max WAND** (Ding & Suel, SIGIR 2011, <https://dl.acm.org/doi/10.1145/2009916.2010048>) —
  per-block precomputed maxima allow skipping whole segments. Still exact.
* **GAIPS** (SIGIR 2021, <https://dl.acm.org/doi/10.1145/3404835.3462997>) — the same idea on GPU
  for maximum-inner-product search, using norm-based, residue-based and hash-based pruning.

The pattern is exactly the one §5 asks for: *predict, bound, verify cheaply, fall back exactly*.

**Nobody has applied it to MoE expert evaluation.** That is a genuine gap, not a gap in my search.

### 2.2 The scheme that actually fits our checkpoint: a scale-only certificate

NVFP4 hands us a free two-level representation that no other format does. Per weight we store
4 mantissa bits **plus** a per-16 E4M3 scale. The scales alone are

```
scale bytes   = 4.435 G / 16          = 277.2 MB   (11.1 % of expert bytes)
mantissa bytes= 4.435 G x 0.5 B       = 2.218 GB   (88.9 %)
```

and E2M1's maximum magnitude is exactly **6.0**, so **reading only the scales gives a rigorous,
data-independent upper bound on any dot product**:

```
| Σ_i w_i x_i |  ≤  6 · Σ_blocks  scale_b · Σ_{i∈b} |x_i|
```

`Σ_{i∈b}|x_i|` is 192 numbers shared by every neuron of every expert in the layer — computed once.
So the certificate costs **one ninth** of the bytes of the thing it certifies.

**The procedure.**

1. Read `gate` densely (0.832 GB) → exact `gate_j` for all 1024 neurons.
2. Read `up` **scales only** (0.092 GB) → certified bound `U_j ≥ |up_j|`.
3. Read `down` **scales only** (0.092 GB) → certified bound `D_j ≥ ‖W_down[:,j]‖_∞`.
4. Neuron `j`'s contribution to *any* output coordinate is bounded by
   `C_j = |SiLU(gate_j)| · U_j · D_j`, all three factors exact or rigorous upper bounds.
5. Skip the mantissas of every neuron whose accumulated `Σ C_j` stays below half an ulp of the
   bf16 output. **The skip is then provably invisible in the emitted bf16 tensor** — bit-identical
   greedy output, no capability gate needed at all.

**Arithmetic** (`f` = surviving mantissa fraction):

| `f` | expert bytes | saved | % of `B_tok` | end-to-end |
|---:|---:|---:|---:|---:|
| 1.0 | 2.495 GB | — | — | — |
| 0.8 | 2.199 GB | 0.296 GB | 4.7 % | +5.0 % |
| 0.5 | 1.756 GB | 0.739 GB | 11.8 % | **+13.4 %** |
| 0.2 | 1.312 GB | 1.183 GB | 18.9 % | **+23.3 %** |

**All of it exact.** This is the only mechanism in this document that satisfies §5's "sublinear in
the model, *exactly*" without touching the capability gate.

**Why bit-exactness is worth more on this model than on most.** `EXPERT_BITS_EVAL.md` records that
this model "turns 1 ulp into a different token, because the 256-way sigmoid router puts top-10 on a
knife edge". A scheme that is exact *at the bf16 expert output* therefore cannot perturb the next
layer's routing at all; a scheme that is merely close will flip expert selections and diverge. On a
model with a softmax router and k = 2 this distinction would be academic. Here it is the whole
difference between "ship it" and "run the capability sweep and hope".

### 2.3 The honest problem with it

The bound is an **L1 bound with worst-case signs**. Against a true dot product over n = 3072 with
roughly random sign alignment, the slack is ≈ √n ≈ 55×, and E2M1 values average well below their
6.0 max, so realistically **~100–200× loose**, i.e. ~7 bits of slack. Since a bf16 output only has
8 mantissa bits, the certificate can only fire on neurons whose true contribution is ~2⁻¹⁵ of the
output — the extreme tail.

Tightenings, in increasing order of promise:

* Store a precomputed `‖W_up[j,:]‖₂` per neuron (one fp16 per 3072 weights ⇒ +0.03 % bytes) and use
  Cauchy–Schwarz instead of L1. Slack drops from ~150× to ~55×.
* Bound per 16-group and use `‖x_b‖₂` per group: turns one √3072 into 192 independent √16 = 4×
  factors, which partially cancel. Slack ~15–25×.
* **Two-sided:** compute a *lower* bound on ‖o‖ from the top few neurons first (they are already
  read), so the denominator of the relative test is not a worst case.
* Relax "bit-identical bf16" to "bit-identical fp32 accumulator to within the rounding of the
  existing kernel" — the shipping kernel already sums in a lane-partitioned order, so it is not
  the exact real-number sum either; the *existing* rounding error is a legitimate budget to spend.
  This last point is worth more than it sounds: the certificate only has to be as good as the
  kernel's own float error, not as good as exact arithmetic.

**Expected verdict: `f` in the 0.7–0.9 range, i.e. +3 % to +6 % exact.** Modest, but it is
*exact*, it composes with §1 (use the certificate to decide the guaranteed-free part of the skip
and the threshold for the rest), and it costs one numpy script to price.

**Cheapest falsifying experiment (~2 hours, CPU only):** on one dumped layer, compute for every
neuron both the true |h_j·W_down| contribution and the scale-only bound, and histogram the ratio.
The 10th/25th/50th percentiles of `bound/true` immediately give `f` for each certificate variant.
No CUDA, no GPU time.

### 2.4 Bit-exact top-k with a guess-verify certificate — the Blackwell precedent

**Claim.** A guess-from-the-previous-decode-step + threshold-verify + exact-refine pipeline
produces **bit-exact** top-K (identical index sets to `torch.topk`) while touching far less data,
at batch 1, on Blackwell.

**Source.** *Guess-Verify-Refine: Data-Aware Top-K for Sparse-Attention Decoding on Blackwell via
Temporal Correlation*, arXiv:2604.22312. <https://arxiv.org/html/2604.22312v1>

**Measured:** NVIDIA **B200 (sm_100)**, **single-batch, single-row**. K = 2048, N ≈ 70 690.
Top-K latency **25.3 µs (GVR) vs 43.7 µs (radix)** = 1.88× average, up to 2.42× per layer;
end-to-end TPOT up to 7.52 % at 100 K context. Temporal overlap between consecutive decode steps
measured at **35–50 %** on DeepSeek-V3.2 layers 20–60. Exactness comes from their Lemma 1: once a
threshold `T` with `K ≤ count(x ≥ T)` is found, `S* ⊆ {i : x_i ≥ T}`, so the refine stage is exact.
Implementation: single CTA, 512 threads, ~60 KB smem, `__ldg + redux.sync`, **no `__ballot_sync`,
no `__shfl_sync`, no `atomicAdd`**.

The paper explicitly does **not** evaluate MoE routing and states that porting requires evidence of
temporal correlation in router scores. **We have that evidence and nobody else does**: `E_frac`
= 0.039 / 0.141 / 0.264 at M = 1/5/16, 22 % below independent routing at M = 5 and 44 % below at
M = 16. Our router is *more* temporally correlated than the 35–50 % attention overlap GVR is built
on.

**But the honest pricing is negative.** GVR's win comes from N ≈ 70 690; ours is N = 256, where a
full scan is 8 elements per lane and the "verify" machinery costs more than it saves. The
transferable content is the **certificate shape**, not the kernel. See §4 for what the router
actually needs.

---

## 3. Levers that DO NOT work here — stated as negatives, with the arithmetic

A negative with arithmetic is the deliverable §4 of the prompt asks for.

### 3.1 Expert pruning and expert merging are worth exactly ZERO bytes per token

`REAP` (arXiv:2510.13999), `HC-SMoE` (arXiv:2410.08589), `DERN` (arXiv:2509.10377),
`MoE-Pruner` (arXiv:2410.12013), `PreMoE` (arXiv:2505.17639) all reduce the number of experts
*stored*. At batch 1 we already read only 10 of 256. Pruning 256 → 128 changes `B_tok` by
**0.000 GB**. It reduces the 122 GB footprint, which is not a constraint — the model fits.

**These papers are entirely about a resource we are not short of.** The only way any of them helps
is if the recovery step permits a *lower k*, which is §3.2. Do not let their large reported
"compression ratios" enter the ranking.

### 3.2 Reducing k is real but small, and lossy

| k | expert bytes saved | % of `B_tok` | end-to-end |
|---:|---:|---:|---:|
| 10 → 9 | 0.249 GB | 4.0 % | +4.2 % |
| 10 → 8 | 0.499 GB | 8.0 % | +8.7 % |
| 10 → 7 | 0.748 GB | 12.0 % | +13.6 % |

**LExI** (*Layer-Adaptive Active Experts*, arXiv:2509.02753) is training-free, sets a per-layer `k`
from a sensitivity sweep, and finds large layer-to-layer variation in how many experts a layer
needs. **Ada-K** (learnable allocator, updates only a linear projection) reports **1.22× on
Qwen1.5-MoE-14.3B-A2.7B** — but it *trains* the allocator, so it is ruled out by §1 of the prompt.
Threshold/top-p routing (cumulative sigmoid mass) is the training-free variant.

Verdict: worth pricing, strictly dominated by §1 (same lossiness class, one third the payoff), but
it is *far* cheaper to implement — a per-layer `k` table in the router is a ten-line change. **It is
the correct first experiment on this axis** precisely because it costs nothing and immediately tells
you how much slack the routing has.

### 3.2b Expert caching and prefetch: structurally dead here, and the argument is short

The prompt asks for the state of the art in overlapping expert fetch with compute given 256 experts
and top-10, noting we have unified memory. The honest answer is that **the entire expert-offload
literature optimises a transfer that does not exist for us**, and what remains dies on cache
arithmetic. Three lines:

1. **There is no transfer.** Mixtral-offloading, MoE-Infinity, Pre-gated MoE, SiDA-MoE, EdgeMoE,
   Fiddler, HOBBIT, ExpertFlow, Klotski, ProMoE, SwapMoE all assume expert weights live in CPU DRAM
   and cross PCIe. Our arena is 68.2 GB in a 122 GB unified pool; every expert is already in the
   address space the SMs read. Their entire measured speedup is PCIe latency they removed. **DEAD.**
2. **L2 cannot help.** One expert is `3 × 1024 × 3072 × 0.5625 B = 5.31 MB`, so **6.0 experts fit
   in the 32 MB L2**. One layer's top-10 is **53.1 MB — 1.66× the whole L2** before any reuse is
   even possible. And at batch 1 every expert weight is touched exactly *once* per token, so there
   is no intra-token reuse to capture. Across tokens, the gap between reading layer L at token `t`
   and at token `t+1` is the entire 6.251 GB of the rest of the model — L2 turns over ~195 times in
   between. **DEAD.** `cudaAccessPropertyPersisting` cannot pin 53 MB into 32 MB.
3. **The one real reuse is already implemented.** Within a speculative window the reuse is over the
   *expert union*, and `k_moe_invert1` already groups assignments expert-major (`elist`/`eoff`) so
   each expert in the union is streamed once for all tokens routed to it. **Our `E_frac`
   measurement is already fully cashed in by the shipping kernel.** This is the concrete answer to
   premise §4.5 of the prompt on this axis.

**Unified memory as an advantage — the negative that follows.** CPU/GPU co-execution (CPU computes
some experts while the GPU computes others) is **zero-sum or worse on a bandwidth-bound decoder**:
the CPU reads the same LPDDR5X at the same ~227 GB/s aggregate, so it does not add bandwidth, it
contends for it. Copy-engine "prefetch" is DRAM→DRAM and consumes the bandwidth it is trying to
hide. `cudaMemPrefetchAsync` on an integrated Tegra pool has no physical transfer to perform.

The only shape where the CPU helps is **latency-bound work**, not bandwidth-bound: the 20.1 % of
the step (`attn_core` + `norm+cast` + `router`) that moves ~1.3 % of the bytes. That is not this
axis, but it is where a heterogeneous-execution idea would have to be aimed, and §4 shows the
`router` third of it is 4 % of the step sitting in three trivial kernels.

### 3.3 The 3 bpw trellis path stays dead, and §1 makes it deader

At `s = 0.87` intra-expert sparsity the expert term falls to 6.21 ms and the *mantissa* share of it
falls to 13 %. A 3 bpw code would then be saving 25 % of a 0.94 GB term instead of 25 % of a
2.49 GB term. **The two levers are anti-synergistic**: taking the sparsity first removes most of
what the trellis was going to save. If §1 lands, the trellis project's ceiling drops from +7 % to
roughly +2 %.

Conversely — and this is the condition `TRELLIS_VERDICT.md` asked to be revisited on — the row-
contiguous repack of §1.5/§1.6 lifts gate/up toward its bandwidth limit, which *would* make the
term more bandwidth-bound. But sparsity removes more bytes than the trellis does, so it wins the
race regardless.

---

## 4. The router: 6.0 % of the step for 1.2 % of the bytes

### 4.1 It is ours that is slow, not the problem that is hard

The `router` profile category covers three launches per layer (`src/forward.cu:353-356`):
`gemm_bf16` router logits → `k_router` (sigmoid + top-10) → `moe_invert`.

```
router weights   256 x 3072 x 2 B (bf16) x 47 layers = 73.9 MB/token = 1.18 % of B_tok  <- matches
byte floor       73.9 MB / 227 GB/s                  = 0.326 ms/step = 6.9 us/layer
launch floor     3 launches x 1.6 us x 47            = 0.226 ms/step
--------------------------------------------------------------------------------
achievable floor                                     ~ 0.55 ms/step
measured (LG_PROF)                                     2.603 ms/step  = 55.4 us/layer
```

Even subtracting the ~10 µs/layer `cudaDeviceSynchronize` that `acc()`/`mark()` inject, roughly
**1.5–2.0 ms/step, i.e. 5–6 % of the whole decode step, is unaccounted for in three tiny kernels.**

Where the time is not:

* `k_moe_invert1` is already fused to **one** launch, one block, `nass = 10` assignments and an
  8-step Hillis-Steele over 256. Microseconds at most.
* `k_router` launches `<<<rows, 256>>>` — at decode `rows = 1`, so **one block of 256 threads on a
  20-SM GPU**, of which only warp 0 does the top-k: 10 sequential argmax passes, each a strided
  8-element smem scan plus a 5-step shuffle reduction. A few hundred cycles.

Where the time probably is — and it is the **same disease as §1.6**:

```
gemm_bf16(M=1, N=256, K=3072)  ->  GEMM_DISPATCH(k_gemm_bf16, 1, ...)
grid = ((N + WPB - 1)/WPB, 1) = (32, 1) at WPB=8  ->  32 blocks x 8 warps = 256 warps
```

**256 resident warps of the 960 the device holds — 27 % occupancy**, one warp per output row
streaming 6144 B. On the measured warp→rate curve from `moe.cu` that is ≈ 145 GB/s, so
1.57 MB / 145 GB/s ≈ **10.8 µs/layer** for the GEMV alone. K-splitting it 4 ways (the same move
that fixed flash-decoding and gate/up) gives 1024 warps → ~222 GB/s → **7.1 µs**. That accounts for
~11 µs of the ~45 µs; **roughly 25 µs/layer = 1.2 ms/step = 4 % of the whole decode step is still
unexplained by any kernel in the category.** That gap is the finding: it is not an algorithm
problem, it is unmeasured overhead in three kernels that should be trivial.

**Answer to the prompt's question "is a 256-way sigmoid router really 6 % of a decode step, or is
ours slow?" — ours is slow, by roughly 4×, and the fix is in our own code, not in the literature.**

### 4.2 Arithmetic

```
router today                       ~1.82 ms/step (6.0 % of 30.3)
router at the 0.55 ms floor         saved 1.27 ms  ->  30.3/29.03 = +4.4 %
router weights bf16 -> FP8          saved 36.9 MB = 0.59 % of B_tok  ->  +0.6 %  (bit-inexact, but
                                    the router is a selection function; FP8 router logits are
                                    standard and the top-10 is stable to far more than FP8 noise)
```

**+4.4 % from making three small kernels not be slow** is the second-cheapest item in this
document and it needs no research at all.

**Cheapest falsifying experiment (~30 min, GPU):** `ncu --set speedlight` (or `nsys` with kernel
tracing) on one decode step, and read off the three router kernels individually. If they sum to
< 15 µs/layer, the 55.4 µs is a `LG_PROF` artefact and this item collapses to +1 %; if they sum to
40 µs+, the fusion is worth the +4.4 % as priced. Either answer is worth having, and the current
profiler cannot distinguish them because `acc()` lumps all three under one `cudaDeviceSynchronize`.

### 4.3 What the literature offers, and it is not much

* **RTop-K** (ICLR 2025, arXiv:2409.00822, <https://xiexi51.github.io/assets/pdf/RTopK.pdf>) —
  binary-search row-wise top-k, one warp per row, with a theoretical analysis of early stopping.
  Directly the right shape (one warp, one row of 256), but it targets large N and its win over a
  10-pass argmax at N = 256 is small.
* **RadiK** (arXiv:2501.14336) and **Dr. Top-k** — radix/bucket selection; benchmarks show the best
  algorithm is regime-dependent. At N = 256 all of them are dominated by launch overhead.
* Bitonic top-k is 15× faster than a sort and 4× faster than alternatives for k ≤ 256, but is smem
  limited to N ≤ 256 — exactly our size, so it is a candidate for a single-pass replacement of the
  10-iteration argmax.

**No published top-k-of-256 microsecond number at k = 10, batch 1 exists.** The honest conclusion is
that at this size the algorithm is irrelevant and the launch count is everything: fuse
`gemm_bf16 + k_router + moe_invert` into one kernel (all three fit in one block's working set —
256 logits + 256 scores + a 256-bin histogram is under 4 KB) and the category collapses to its
0.55 ms floor. **1 launch/layer instead of 3 saves 47 × 2 × 1.6 µs = 0.15 ms on launches alone**,
with the rest coming from not round-tripping 256 floats through DRAM twice.

---

## 5. Quantization beyond NVFP4

Full delegated survey in §9. This section keeps only what survives contact with the device
arithmetic. **Sources are named inline; every ppl/MSE number below is offline PTQ with no batch size
or hardware, which is the correct regime for a weight-format question.**

### 5.0 The fact that reopens the whole axis

**poolside publishes `Laguna-S-2.1` (BF16, 118 B), `Laguna-S-2.1-FP8` (118 B) and
`Laguna-S-2.1-INT4` (121 B) alongside the NVFP4 we run** (<https://huggingface.co/poolside>).
The FP8 variant is ~118 GB and **fits the 122 GB unified pool**. Every requantization idea in this
section is therefore *available* — we are not stuck improving an already-quantized artifact.

Which matters, because the delegated research returned a clean impossibility result for the thing
the prompt asked about directly:

> **Applying AWQ/SmoothQuant-style diagonal rescaling to an already-NVFP4 checkpoint cannot reduce
> error. It is not hard, it is information-theoretically dead.** For any diagonal `D`,
> `xD⁻¹ · (Q(W)D)ᵀ = x·Q(W)ᵀ` exactly — the error is unchanged, only re-expressed. Re-rounding
> `Q(W)·D` onto a fresh NVFP4 grid *strictly adds* error, because `D` is per-K and NVFP4's blocks
> run along K, so `D` perturbs every block's amax and forces a second rounding. Zero papers attempt
> it; the family requires the pre-rounding weights.

So: **question 3 of the brief is answered NO for the shipped artifact, and YES only via
requantization from `Laguna-S-2.1-FP8`.**

### 5.1 The bits-vs-error curve is FLAT at 4.5 bpw — which independently confirms our trellis verdict

QTIP's own table (arXiv:2406.11235, Table 5), Llama-2 WikiText-2:

| | FP16 | QTIP | QuIP# | AQLM |
|---|---:|---:|---:|---:|
| 7B | 5.12 | 5.17 | 5.19 | 5.21 |
| 70B | 3.12 | 3.16 | 3.18 | 3.19 |

At **2 bits** QTIP beats QuIP# by 0.21 ppl on 70B. At **4 bits** the gap is **0.02**. The entire
lattice-codebook / incoherence-processing apparatus is a *low-bit* technology. We are at 4.5 bpw.
`TRELLIS_VERDICT.md` reached the same conclusion from the speed side; this is the accuracy side of
the same wall, from an independent direction. **AQLM is separately dead here**: a 2¹⁶×8 fp16
codebook = 1 MiB *per linear layer*, and we have 47 × 256 × 3 of them.

### 5.2 MXFP4 is strictly worse than what we already run — do not consider it

| format | group | scale | bpw | weight MSE (×10⁻³) |
|---|---:|---|---:|---:|
| MXFP4 | 32 | UE8M0 | 4.25 | **13.2** |
| NVFP4 | 16 | E4M3 | 4.50 | 9.0 |
| NVFP4 + 4/6 | 16 | E4M3 | 4.50 | 7.5 |
| NVINT4 | 16 | E4M3 | 4.50 | 7.4 |
| **IF4** | 16 | UE4M3 | 4.50 | **6.2** |

*(arXiv:2603.28765 Table 1.)* On **MoE models** at RTN, WikiText-2 (Table 2), with
**Qwen3.5-122B-A10B as the closest published analogue to Laguna-S at ~113 B routed / ~5 B active**:

| model | BF16 | MXFP4 | NVFP4 | NVINT4 | 4/6 | IF4 |
|---|---:|---:|---:|---:|---:|---:|
| Nemotron 3 30B-A3B | 7.33 | 10.51 | 9.02 | **8.35** | 9.12 | **8.26** |
| Qwen3.5 35B-A3B | 7.70 | 8.83 | 8.14 | 8.31 | 8.11 | **8.07** |
| **Qwen3.5 122B-A10B** | **5.72** | 6.78 | **6.20** | 6.28 | 6.16 | **6.10** |

MXFP4 costs 0.58 ppl on the 122 B row where NVFP4 costs 0.48. **gpt-oss-120b ships MXFP4 MoE but
published no BF16 control**, so it is evidence of deployability, not of accuracy.

### 5.3 The only lever here that removes BYTES: group size

This is the one quantization change that shows up in `B_tok`. E4M3 scale, `4 + 8/G` bpw:

| G | bpw | expert bytes | saved | % of `B_tok` | **end-to-end** |
|---:|---:|---:|---:|---:|---:|
| 16 (today) | 4.500 | 2.4947 GB | — | — | — |
| 32 | 4.250 | 2.3561 GB | 0.139 GB | 2.22 % | **+2.27 %** |
| 64 | 4.125 | 2.2868 GB | 0.208 GB | 3.33 % | **+3.44 %** |
| 128 | 4.0625 | 2.2521 GB | 0.243 GB | 3.88 % | **+4.04 %** |

It also thins the scale traffic in the warp-starved `gate/up` kernel (the `uint2` 8-byte scale load
at `moe.cu:438` covers twice as many k-chunks at G = 32), so it should convert slightly better
than the byte count alone suggests.

**Where is the knee?** The delegated search found **no clean sweep** that holds the scale format at
E4M3 and varies G — every published comparison confounds G with E4M3-vs-E8M0. The one usable proxy
is within-block crest factor (arXiv:2510.25602 Table 2): per-channel **11.97**, block-32 **2.96**,
block-16 **2.39**. **The 32 → 16 step buys 19 % for +0.25 bpw; the per-channel → 32 step buys 75 %
for the same +0.25 bpw. The knee is above 32, not between 32 and 16.** That is a real, exploitable
gap in the literature and it is ours to close in numpy.

### 5.4 The best idea on this axis: buy the group-size bits back with a better element grid

This is the synthesis the delegated survey does not make, and it is the most actionable thing in §5.

MSE scales roughly as 4⁻ᵇ, so an MSE ratio `r` is worth `−log₄ r` bits:

| change | MSE ratio vs NVFP4 | **equivalent bits gained** | decode ops added |
|---|---:|---:|---:|
| Four-Over-Six (4/6) | 7.5/9.0 = 0.833 | **0.13 bits** | **exactly zero** |
| NVINT4 | 7.4/9.0 = 0.822 | 0.14 bits | +1 to +2 (loses `cvt.rn.f16x2.e2m1x2`) |
| **IF4** | 6.2/9.0 = 0.689 | **0.27 bits** | ~0.06 (warp-uniform branch, see below) |

**And the G = 16 → 32 step costs exactly 0.25 bits.**

> **IF4 at G = 32 lands at 4.25 bpw with the same reconstruction error as NVFP4 at G = 16 and
> 4.50 bpw. That is +2.27 % end-to-end at zero accuracy cost, from a pure re-quantization with no
> calibration data and no new decode instruction.**

Two supporting facts that make it plausible rather than cute:

* **IF4's selector costs no bits.** The per-block choice between the E2M1 grid and an INT4 grid
  rides in the **sign bit of the E4M3 scale**, which NVFP4 leaves unused because every element
  carries its own sign (arXiv:2603.28765; MixFP4 arXiv:2605.31035 does the same with E1M2).
* **IF4's papers say it needs new silicon — that conclusion does not bind us.** They cost it as a
  tensor-core MAC datapath (+66.6 % area, −4.6 % throughput in 28 nm). At batch 1 we dequantize to
  fp16 in registers anyway. The selector is **warp-uniform over 16 consecutive weights**; lay the
  repack out so a lane's 16-byte chunk sits inside one block and the branch is divergence-free at
  ~1 predicated op per 16 weights = **0.06 ops/weight** against a ~3 ops/weight budget. If that
  layout cannot be had, the fallback is compute-both-and-select at 2× unpack cost, which is fatal —
  so **the layout is the whole experiment.**

**4/6 alone** (0.13 bits, literally zero decode change) is the risk-free floor of this idea, but
⚠️ IF4's Table 2 shows 4/6 *hurting* Nemotron-30B-A3B (9.02 → 9.12) while helping both Qwen3.5
MoEs. It is not unconditionally good on MoE. Measure it, do not assume it.

**NVINT4 is the trap.** Better error curve, same bits, and `poolside/Laguna-S-2.1-INT4` already
exists so the accuracy comparison may be free. But sm_110a has a hardware `cvt.rn.f16x2.e2m1x2`
FP4-pair unpack and no INT4 equivalent; +1 op/weight against a 434 Gweight/s ALU ceiling at
332 achieved is roughly **−25 % decode throughput**, which erases a 0.14-bit accuracy win several
times over. **Gate it with a 20-line dequant microbenchmark before any accuracy work** — half a
day, and it either kills the idea or clears it.

### 5.5 MoE-specific quantization findings that map onto our config

* **MoE is genuinely easier to quantize than dense.** MoQE (arXiv:2310.02410, Microsoft): expert
  FFN weights have a far tighter range than dense FFN. 5.3 B MoE MT model, expert-weights-only,
  channel-wise scales — BLEU vs dense baseline: fp16 +2.87 %, int4 +2.49 %, int3 +2.11 %, int2
  (QAT) +1.88 %, while the **dense model collapses at 2-bit (−42.96 BLEU)**. ⚠️ the 1.24× speedup
  is A100 **batch 24**, encoder-decoder MT — do not port it.
* **Attention is more sensitive than experts.** QuantMoE-Bench (arXiv:2406.08155): raising
  attention to 4/8 bits gives **> 5 % gains** versus spending the same bit budget on experts.
  **This is a direct warning against `REPRICED.md`'s current top item.** `o_proj → NVFP4` is priced
  at +9.2 % on bytes; the MoE quantization literature says attention projections are the *worst*
  place in an MoE to spend precision. The two are not contradictory — one is speed, one is
  accuracy — but the +9.2 % should not be treated as low-risk just because it is bandwidth-obvious.
* **Shared experts and first layers need protection.** Same paper: 4-bit on shared experts beats
  4-bit on randomly-chosen routed experts, and early layers are the most sensitive. Laguna-S has
  **one shared expert (1024) per layer and a layer-0 dense at intermediate 12288** — both are in
  the `shared+dense` category (10.8 % of the step, 8.9 % of bytes) and both are exactly the tensors
  this literature says to keep at higher precision. Cheap and directionally safe.
* **Expert-activation frequency is a good bit-allocation heuristic — but only under imbalanced
  routing.** Strong on DeepSeek, weak on Mixtral. **Our routing is the most position-correlated
  measured anywhere** (`E_frac` 0.039/0.141/0.264, 22–44 % below independent), which is exactly the
  regime where the heuristic works. ⚠️ But per-expert bit-widths mean a divergent branch across the
  10 experts in one MoE launch, on 20 SMs. **Structurally unaffordable for us.** The only member of
  this family that avoids it is EAQuant (arXiv:2506.13329), whose contribution is a *shared*
  per-layer smoothing vector — uniform format, folded offline, zero decode cost, and reported
  +1.15 to +1.37 average accuracy over DuQuant on OLMoE / DeepSeek-MoE / Mixtral at W4A4.
* **Rotations: use exactly 16 points or none.** arXiv:2509.23202 finds Hadamard *hurts* NVFP4 at
  RTN unless it is group-aligned (Had16); NVIDIA (arXiv:2509.25149) independently chose 16×16 and
  found 4×4 degraded and 128×128 only marginal. The residual-stream rotation (QuaRot R1) folds
  offline for free; the intra-MLP rotation (R4, between SwiGLU and `down_proj`) cannot fold. Cost
  of the online part here: (10 routed + 1 shared) × 1024 × log₂16 ≈ 45 k add/sub per layer,
  ≈ 2.7 Mop/token against ~13.3 Gop/token of expert dequant = **0.02 % of the ALU budget**. It is a
  `__shfl_xor_sync` butterfly on registers — no shared memory, **no codebook**, so our measured
  −31 % lookup penalty does not apply. But it **must** be fused into the down-proj prologue: 47
  separate launches × 1.6 µs = 75 µs/token = 0.25 % of the step, which is the entire benefit.

### 5.6 The risk this section surfaces that has nothing to do with speed

Quantization-Aware Distillation (arXiv:2601.20088) measures NVFP4 **PTQ** holes that perplexity
does not see:

| model | benchmark | BF16 | **NVFP4 PTQ** | NVFP4 QAD |
|---|---|---:|---:|---:|
| Llama Nemotron Super 49B | AIME25 | 46.0 | **32.3** | 45.6 |
| AceReason 7B | LiveCodeBench | 54.3 | 52.0 | 53.3 |

And NVIDIA's own pretraining paper (arXiv:2509.25149) reports NVFP4 vs FP8 on a 12 B model:
MMLU-Pro 62.58 vs 62.62 (no change) but **HumanEval++ 59.93 → 57.43 and MBPP++ 59.11 → 55.91 —
code average −2.85 points.** *Code is the one benchmark family FP4 reliably damages, and this is a
code model.*

**`Laguna-S-2.1-FP8` fits in 122 GB and needs no new kernels** (the FP8 dense path already exists).
Running LiveCodeBench or HumanEval+ on FP8 vs NVFP4 at batch 1 is a ~1-day experiment that could
reframe the project: if poolside shipped a PTQ NVFP4 checkpoint with this hole, the available win
is not +5 % tok/s, it is recovering capability points. **This is the highest-information experiment
in the whole document and it is not a speed experiment.**

### 5.7 The strategic conclusion: every bits-per-weight lever is anti-synergistic with §1

At `s = 0.75` intra-expert sparsity the expert term is already down to 1.247 GB, of which 0.832 GB
is the *dense gate*. Applying G = 32 on top then saves 0.069 GB instead of 0.139 GB — **half the
value**. At `s = 0.87` it saves 0.052 GB. The same applies to IF4, to 3 bpw trellis (§3.3), and to
every format change.

**Ordering therefore matters more than selection.** If §1 lands, the entire quantization axis
collapses to roughly a third of its standalone value. If §1 fails, §5.4 (IF4 @ G=32, +2.3 % free)
becomes the best remaining item on this axis. **Price §1 first — one afternoon of numpy — because
its answer determines whether the rest of §5 is worth building at all.**

---

## 6. Ranked list — (payoff × uncertainty) / cost

Uncertainty is scored as "how much would we learn", so a high-payoff item we cannot currently
predict ranks above a certain small one.

| # | lever | § | payoff | exact? | retrain? | cost | uncertainty | rank score |
|---:|---|---|---:|---|---|---|---|---:|
| 0 | **FP8-vs-NVFP4 capability check on a code benchmark** | 5.6 | not speed — could reframe the project | n/a | no | **1 d, no new kernels** | **very high** | **run it first** |
| 1 | **Intra-expert activation sparsity, block-32, layout-native** | 1.3 | **+12 % … +25 %** | lossy (same gate class as expert bits) | no | 2 wk (repack + 2 kernels + capability sweep) | **high** — depends entirely on whether 87 % survives `MI = 1024` | **highest** |
| 2 | **Row-contiguous repack / fix gate-up warp starvation** | 1.6 | **+3.5 %** | **bit-exact** | no | 3–4 d | low | **very high** (cheap, certain, prerequisite for #1) |
| 3 | **Fuse the three router kernels; find the missing 1.2 ms** | 4.2 | **+4.4 %** | bit-exact | no | 2–3 d (+30 min to price with `ncu`) | medium — we do not know where the time is | **very high** |
| 4 | Layer-adaptive `k` (10 → 8/9), LExI-style table | 3.2 | +4 % … +9 % | lossy | no | **1 d** | medium | **high** (cheapest probe of routing slack) |
| 5 | Scale-only certified skip (exact sublinear `B_tok`) | 2.2 | +3 % … +13 % | **bit-exact** | no | 2 h to *price*, 1 wk to build | **very high** — nobody has done this | **high on the pricing step alone** |
| 6 | **IF4 (or 4/6) element grid at G = 32** | 5.4 | **+2.3 % at equal error** | lossy | no (requantize from `-FP8`) | 1 d numpy + 1 wk kernel | medium-high | **high** (best of the quantization axis) |
| 7 | NVFP4 group 16 → 64 / 128 alone | 5.3 | +3.4 % … +4.0 % | lossy | no (requantize) | 1 wk | **high** — no published `MSE(G)` curve at fixed E4M3 | medium-high |
| 8 | Router GEMV K-split + bf16 → FP8 | 4.1–4.2 | +1.2 % | ~exact in selection | no | 1 d | low | medium |
| 9 | Shared expert + layer-0 dense to higher precision | 5.5 | negative on speed, positive on capability | lossy→less lossy | no | 1 d | low | medium (buy accuracy to spend elsewhere) |
| 10 | 16-point Hadamard fused into down-proj prologue | 5.5 | 0 % alone; enables NVINT4 | lossy | no | 2 d | medium | only after #6's numpy sweep |
| 11 | Group-structured (16-wide) input sparsity, TEAL-shape | 1.8 | ? | lossy | no | 2 wk | high but likely small | low |
| 12 | NVINT4 | 5.4 | −25 % decode risk vs +0.14 bits | lossy | no | half-day to *kill* | high | **gate with a microbenchmark, expect to kill** |
| 13 | 3 bpw trellis | 3.3 | +7 % alone, **+2 % after #1** | lossy | no | 4–6 wk | settled twice over (§3.3, §5.1) | **do not build** |
| 14 | MXFP4 | 5.2 | negative | lossy | no | — | none | **dead — strictly worse than today** |
| 15 | AWQ/GPTQ/SmoothQuant on the shipped NVFP4 | 5.0 | **impossible** | — | — | — | none | **dead in principle** |
| 16 | Expert pruning / merging (REAP, HC-SMoE, DERN, PreMoE) | 3.1 | **+0.0 GB, +0.0 %** | — | — | — | none | **dead — solves a problem we do not have** |
| 17 | AQLM / QuIP# / GPTVQ / VPTQ | 5.1 | 0.02 ppl at 4 bits | lossy | no | — | none | **dead — low-bit technology** |
| 18 | Per-expert mixed bit-widths (MxMoE, MC-MoE, GEMQ) | 5.5 | — | lossy | no | — | — | **structurally unaffordable** (divergent branch, 20 SMs) |
| 19 | Deja Vu predictors | 1.9 | — | lossy | **yes** | — | — | **ruled out by prompt §1** |
| 20 | R-Sparse low-rank term | 1.9 | negative (adds bytes) | lossy | no | — | — | **dead** |

**Recommended sequence: 0 → [1-price, 5-price, 6-price in numpy, all one week total] → 2 → 3 → 4 →
build whichever of 1/5/6 the pricing cleared.**

Rationale: items 2, 3 and 4 total **+12 %**, are cheap, and two of the three are bit-exact — they
pay for the week spent pricing the big ones. Each also de-risks item 1: #2 is its enabling
refactor, #4 measures how much slack the routing actually has, and #5 tells you what fraction of #1
is free. And **§5.7 means the pricing order is not arbitrary**: if #1 lands at +25 %, items 6, 7 and
13 lose two thirds of their value, so building them first would be wasted work.

---

## 7. Premises from §4 of the prompt that I could not refute

* **§4.5, "`E_frac`'s position correlation cannot be exploited at inference time."** I could not
  refute it *on this axis*. The expert-major grouping in `moe_invert` already converts window
  overlap into single reads of the union, which is the whole of the available gain. GVR
  (arXiv:2604.22312) shows temporal correlation can be turned into a bit-exact top-k certificate,
  and ours is stronger than theirs — but at N = 256 there is nothing to save. **The premise
  survives.**
* **§4.4, "`N_k · c_k` is irreducible without a megakernel."** §4.2 shows one category where three
  kernels can become one for +4.4 %, which weakens the premise but does not overturn it: that is
  fusion, not a megakernel.
* The trellis premise (§2.1) **survives and is strengthened** — §3.3 shows the trellis is
  anti-synergistic with the larger sparsity lever.

## 8. What I could not answer

* **Search budget was exhausted** (200/200 WebSearch calls, shared across this session) before I
  could resolve two things that matter: (a) *Sparsing Law*'s SwiGLU-specific sparsity-vs-FFN-width
  numbers, which directly govern whether §1's 87 % reproduces at `moe_intermediate = 1024`, and
  (b) whether anyone has published intra-expert sparsity numbers for a model with experts as narrow
  as ours. Both are answerable with a handful of targeted fetches next session, and (a) is worth
  doing *first* because it multiplies the largest item in the ranking.
* One directional hint I did retrieve but could not pin to numbers: *Sparsing Law* reports that
  **narrower FFNs carry *higher* activation sparsity**, which if it holds would make our
  `moe_intermediate = 1024` experts *more* sparse than the published models, not less. I am
  deliberately not leaning on this — it is a paraphrase, not an extracted table, and §1.7's
  experiment settles it on our own weights in an afternoon regardless.
* No published microsecond figure exists for top-k-of-256 at k = 10, batch 1. §4 answers the
  question by arithmetic instead.
* I could not determine where the ~25 µs/layer of unexplained `router` time goes without running
  `ncu`, which was out of scope (GPU busy). §4.2 gives the 30-minute experiment that settles it.
* The scale-only certificate of §2.2 is, as far as I can establish, **novel** — I found the
  bounding machinery (WAND/BMW/GAIPS) and the NVFP4 structure separately, and nobody has joined
  them. That means there is no external evidence for or against it, so its uncertainty in the
  ranking is genuinely maximal rather than a hedge.

---

## 9. Appendix — delegated research

### 9.A Quantization: what the delegated pass could NOT find

Stating the gaps is the useful part; the numbers are already integrated into §5.

1. **No MXFP4-vs-NVFP4 head-to-head on MoE at matched calibration strength.** The only MoE
   comparison is RTN-only (IF4 Table 2) plus a KL-divergence table under direct-cast
   (arXiv:2510.25602). MR-GPTQ, DuQuant++, ARCQuant, SOAR, ScaleSweep, MixFP4, Four-Over-Six PTQ
   and the FP4-sensitivity study **all evaluated dense models only**.
2. **No clean group-size sweep at fixed scale format.** There is no published `MSE(G)` curve for
   FP4 with E4M3 scales. Every comparison confounds `G` with E4M3-vs-E8M0. (This is the gap §5.3
   proposes to close.)
3. **No work on improving an already-quantized checkpoint without the original weights** — zero
   hits across ~6 query formulations, and §5.0 argues it is impossible in principle for the
   diagonal-rescale family.
4. **No decode-ops-per-weight accounting for any new 4-bit format on a GPU.** IF4 and MixFP4 report
   *ASIC MAC* area/power; MX+ reports a software slowdown ratio (1.13×) but not ops/weight. Nobody
   publishes the number we actually need — which is why §5.4's kill-gate is a microbenchmark.
5. **MxMoE's 3.4× and +29.4 % have no stated hardware or batch size** anywhere reachable.
6. **No MoE with a sigmoid router + selection-only bias + 2.5× routed-output scale has been studied
   under quantization at all.** Laguna-S's router is architecturally unlike
   Mixtral/DeepSeek/Qwen3-MoE, so EAQuant's router-alignment result is a hypothesis for us, not a
   transfer.
7. **No batch-1, single-stream, unified-memory decode measurement for anything in §5.** Every
   throughput number is discrete-GPU-with-PCIe and mostly batch ≥ 4. The sole batch-1 exception is
   QTIP on an RTX 6000 Ada (Llama-2-7B 140 tok/s at 4-bit, 70B 16.3 tok/s).

### 9.B Sources

Activation sparsity and contextual sparsity:
- [Uncovering Intra-expert Activation Sparsity for Efficient MoE Model Execution (arXiv:2605.08575)](https://arxiv.org/abs/2605.08575) — **the headline source**
- [TEAL: Training-Free Activation Sparsity in LLMs, ICLR 2025 (arXiv:2408.14690)](https://arxiv.org/abs/2408.14690) · [Together AI writeup](https://www.together.ai/blog/teal-training-free-activation-sparsity-in-large-language-models)
- [CATS: Contextually-Aware Thresholding for Sparsity (arXiv:2404.08763)](https://arxiv.org/abs/2404.08763) · [code](https://github.com/ScalingIntelligence/CATS)
- [Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time, ICML 2023 (arXiv:2310.17157)](https://arxiv.org/abs/2310.17157)
- [R-Sparse: Rank-Aware Activation Sparsity, ICLR 2025 (arXiv:2504.19449)](https://arxiv.org/pdf/2504.19449)
- [Sparsing Law (arXiv:2411.02335)](https://arxiv.org/pdf/2411.02335) — *unresolved, see §8*
- [SparseInfer (arXiv:2411.12692)](https://arxiv.org/pdf/2411.12692)

Exact top-k with certificates (the "outside LLM serving" family):
- [Guess-Verify-Refine: Data-Aware Top-K for Sparse-Attention Decoding on Blackwell (arXiv:2604.22312)](https://arxiv.org/html/2604.22312v1) — **bit-exact, B200, batch 1**
- [Faster Top-k Document Retrieval Using Block-Max Indexes, SIGIR 2011](https://dl.acm.org/doi/10.1145/2009916.2010048) — WAND / BMW
- [GAIPS: Accelerating Maximum Inner Product Search with GPU, SIGIR 2021](https://dl.acm.org/doi/10.1145/3404835.3462997)
- [RTop-K, ICLR 2025 (arXiv:2409.00822)](https://xiexi51.github.io/assets/pdf/RTopK.pdf) · [RadiK (arXiv:2501.14336)](https://arxiv.org/pdf/2501.14336)

MoE structure, pruning, routing:
- [LExI: Layer-Adaptive Active Experts (arXiv:2509.02753)](https://www.arxiv.org/pdf/2509.02753)
- [REAP: Why Pruning Prevails for One-Shot MoE (arXiv:2510.13999)](https://arxiv.org/pdf/2510.13999) · [HC-SMoE (arXiv:2410.08589)](https://arxiv.org/pdf/2410.08589) · [DERN (arXiv:2509.10377)](https://arxiv.org/html/2509.10377v1) · [MoE-Pruner (arXiv:2410.12013)](https://arxiv.org/pdf/2410.12013) · [PreMoE (arXiv:2505.17639)](https://arxiv.org/html/2505.17639) — all §3.1 dead
- [Survey on Inference Optimization for MoE (arXiv:2412.14219)](https://arxiv.org/pdf/2412.14219)

Quantization:
- [Adaptive Block-Scaled Data Types / IF4 (arXiv:2603.28765)](https://arxiv.org/html/2603.28765v1) — **the MoE format table**
- [Four Over Six: adaptive NVFP4 block scaling (arXiv:2512.02010)](https://arxiv.org/html/2512.02010v5)
- [MixFP4 (arXiv:2605.31035)](https://arxiv.org/html/2605.31035v1)
- [MR-GPTQ / QuTLASS, ICLR 2026 (arXiv:2509.23202)](https://arxiv.org/abs/2509.23202)
- [INT vs FP: fine-grained low-bit formats (arXiv:2510.25602)](https://arxiv.org/html/2510.25602v1)
- [Pretraining LLMs with NVFP4, NVIDIA (arXiv:2509.25149)](https://arxiv.org/html/2509.25149v1)
- [Quantization-Aware Distillation for NVFP4 recovery (arXiv:2601.20088)](https://arxiv.org/html/2601.20088v3)
- [QTIP (arXiv:2406.11235)](https://arxiv.org/html/2406.11235v3) · [QuIP# (arXiv:2402.04396)](https://arxiv.org/pdf/2402.04396) · [VPTQ (arXiv:2409.17066)](https://arxiv.org/abs/2409.17066) · [GPTVQ (arXiv:2402.15319)](https://arxiv.org/abs/2402.15319)
- [QuaRot (arXiv:2404.00456)](https://arxiv.org/pdf/2404.00456) · [DuQuant++ (arXiv:2604.17789)](https://arxiv.org/html/2604.17789v1)
- [MoQE (arXiv:2310.02410)](https://arxiv.org/html/2310.02410) · [QuantMoE-Bench (arXiv:2406.08155)](https://arxiv.org/html/2406.08155v2) · [MC-MoE (arXiv:2410.06270)](https://arxiv.org/abs/2410.06270) · [MxMoE (arXiv:2505.05799)](https://arxiv.org/abs/2505.05799) · [EAQuant (arXiv:2506.13329)](https://arxiv.org/html/2506.13329v2) · [GEMQ (arXiv:2605.23078)](https://arxiv.org/pdf/2605.23078) · [AlphaQ (arXiv:2606.04980)](https://arxiv.org/pdf/2606.04980)
- [MX+ , MICRO'25 (arXiv:2510.14557)](https://arxiv.org/html/2510.14557v1) · [ARCQuant (arXiv:2601.07475)](https://arxiv.org/html/2601.07475v1) · [SOAR (arXiv:2605.12245)](https://arxiv.org/pdf/2605.12245) · [ScaleSweep (arXiv:2606.07618)](https://arxiv.org/pdf/2606.07618)
- [Red Hat: NVFP4 quantization recovery rates](https://developers.redhat.com/articles/2026/02/04/accelerating-large-language-models-nvfp4-quantization) · [gpt-oss-120b card (arXiv:2508.10925)](https://arxiv.org/pdf/2508.10925)
- [poolside on Hugging Face — BF16 / FP8 / INT4 / NVFP4 variants](https://huggingface.co/poolside)

### 9.C Expert caching / prefetch on unified memory

*(delegated; appended below when received)*
