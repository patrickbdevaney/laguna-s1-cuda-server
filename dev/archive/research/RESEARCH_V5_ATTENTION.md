# RESEARCH_V5_ATTENTION.md — the `attn_qkvo_gemm` axis

Research pass against `RESEARCH_PROMPT_V5.md` §3 row 1 (`attn_qkvo_gemm`, 34.7% of step,
44.9% of `B_tok`) and §4 premises 1 and 2. Desk research only — no GPU was touched.

---

## 0. Arithmetic baseline (derived, then cross-checked against the measured 44.9%)

Everything below is priced against this. It is worth checking first because it reveals a
term the prompt did not name.

Per-layer projection parameter counts from the stated shape (hidden 3072, head_dim 128):

| | global ×12 | sliding ×36 |
|---|---:|---:|
| heads | 48 | 72 |
| KV heads (group 6 / 9) | 8 | 8 |
| `q_proj` 3072×(h·128) | 18.874 M | 28.311 M |
| `k_proj` 3072×1024 | 3.146 M | 3.146 M |
| `v_proj` 3072×1024 | 3.146 M | 3.146 M |
| `o_proj` (h·128)×3072 | 18.874 M | 28.311 M |
| gate 3072×h | 0.147 M | 0.221 M |
| **layer total** | **44.19 M** | **63.14 M** |

Total = 12·44.19 + 36·63.14 = **2803.2 M params = 2.803 GB at FP8** (row scales add 2.0 MB).

**2.803 / 6.251 = 44.84%.** This reproduces the measured 44.9% to within rounding, so the
shape model is right and the per-projection split below can be trusted.

Step T = 1/33.0 s = **30.30 ms**. Category time = 0.347·T = **10.52 ms**.
Effective category rate = 2.803 GB / 10.52 ms = **267 GB/s** — *above* the 227 GB/s streaming
ceiling, which is only explicable by L2 hits (a sliding layer's `o_proj` is 28.3 MB against a
32 MB L2). Note this: the category is already running at 118% of the streaming ceiling, so any
model that prices it at 227 GB/s will over-predict the win.

Per-projection split of the category:

| projection | bytes (FP8) | % of `B_tok` | % of category |
|---|---:|---:|---:|
| `o_proj` | 1.2457 GB | **19.93%** | 44.4% |
| `q_proj` | 1.2457 GB | **19.93%** | 44.4% |
| `k_proj`+`v_proj` | 0.3020 GB | 4.83% | 10.8% |
| gate | 0.0097 GB | 0.16% | 0.35% |

### Finding 0 — `q_proj` is exactly the same size as `o_proj`.

Because head_dim·heads is the same on both sides of the attention block, `q_proj` and `o_proj`
are *byte-identical* (1.2457 GB each). The prompt frames this axis as "`o_proj` is 42.7% of it
and has never been evaluated below FP8". **There are two equal prizes here, not one.** And
`q_proj` carries none of the gating objection — its input is the post-layernorm residual, the
best-conditioned activation in the block, and its output passes through QK-RMSNorm which
*rescales away* per-head magnitude error before RoPE. `q_proj` is strictly the easier of the two
and is worth the same +7.3%.

Conversely: `k_proj`+`v_proj` together are 4.83% of `B_tok`. Even taking them to 2 bits buys
under +2%. **Do not spend effort on k/v.** This retires priority question 5 (GQA structure): GQA
has already shrunk k/v to irrelevance; there is nothing left to exploit there.

### Conversion table used throughout

Category-local model (the category's time scales with its own bytes at the measured 267 GB/s,
provided the kernel stays bandwidth-bound):

| change | category bytes | category time | step | tok/s | **end-to-end** |
|---|---:|---:|---:|---:|---:|
| baseline FP8 | 2.803 GB | 10.52 ms | 30.30 ms | 33.0 | — |
| `o_proj` → NVFP4 (4.5 bpw) | 2.258 GB | 8.47 ms | 28.25 ms | 35.4 | **+7.3%** |
| `q_proj` → NVFP4 | 2.258 GB | 8.47 ms | 28.25 ms | 35.4 | **+7.3%** |
| `q`+`o` → NVFP4 | 1.713 GB | 6.43 ms | 26.21 ms | 38.2 | **+15.6%** |
| all five → NVFP4 | 1.577 GB | 5.92 ms | 25.70 ms | 38.9 | **+17.9%** |
| all five → INT4 g128 (4.125 bpw) | 1.445 GB | 5.42 ms | 25.20 ms | 39.7 | **+20.2%** |

**ALU-capped floor.** §2.1 records an ALU ceiling of 434 Gweight/s on this device. If NVFP4
GEMV were pinned there rather than by bandwidth, the all-five case gives
2803.2 Mweight / 434 Gweight/s = **6.46 ms** (vs 5.92 ms bandwidth-bound), i.e. **+15.5%**
instead of +17.9%. **The ALU risk is worth at most 2.4 points.** That is the single most
important de-risking fact in this document: even the pessimistic assumption leaves this the
largest available win in the whole project, larger than 3 bpw experts (+5.3%) by 3×.

The prompt's stated "+9.2%" for `o_proj` corresponds to assuming the *whole step* is
byte-proportional (0.545 GB / 6.251 GB = 8.7% → +9.6%). The category-local +7.3% is the
defensible number. I use +7.3%.

---

## 1. Priority question 1 — can `o_proj` go below FP8 under an unbounded softplus gate?

### Claim 1.1 — For **weight-only** quantization the gate cannot degrade SNR at all. It is exactly scale-invariant. *(derivation, not a citation)*

`o_proj` computes `y = Σ_h W_hᵀ (g_h · x_h)` where `g_h > 0` is the per-head softplus gate,
constant across the 128 channels of head `h`. Write the gate as a positive diagonal `D` acting
on the input channels. With `W̃ = W + ΔW` from round-to-nearest or GPTQ:

* signal `s = W D x`
* error `e = ΔW D x`

`ΔW` is data-independent (it is a property of the weight and its group scale). Scaling `D` by any
positive amount scales `s` and `e` **by the identical factor**. The signal-to-quantization-noise
ratio `‖s‖/‖e‖` is therefore *exactly invariant* to the gate, for any gate value, bounded or not.

Premise §4.1 says the gate "multiplies quantization error". It does — and it multiplies the
signal by the same number. The premise is a category error: it is true for **activation**
quantization (W4A4/W8A8, where a per-tensor activation scale is blown out by one large gate) and
false for **weight-only** quantization, which is what W8A16 → W4A16 is. Laguna's shipped attention
is W8A16. **Premise §4.1 does not apply to the change actually being contemplated.**

*Exact?* No — it is a lossy weight change. *Retraining?* No; PTQ only.
*Category:* `attn_qkvo_gemm`, −2.05 ms (o) / −4.09 ms (q+o).
*Cheapest falsifying experiment:* RTN-quantize `o_proj` weights to NVFP4 in-place (no
calibration, no Hessian, ~30 min of numpy), then sweep an artificial per-head gain `α·g_h` for
α ∈ {0.1, 1, 10, 100} on a fixed prompt and measure the relative L2 error of the layer output
against FP8. Claim 1.1 predicts the relative error is **flat in α**. If it rises with α, the
premise survives and the rest of §1 collapses. This is one afternoon and needs no GPU time
beyond a single forward pass.

### Claim 1.2 — Even under *activation* quantization, the gate is absorbed exactly by the block scale, because gate boundaries and quantization-group boundaries coincide.

head_dim = 128. NVFP4 groups 16 elements along K; MXFP4 groups 32; INT4-g128 groups 128. All
three divide 128 exactly. Since `g_h` is constant within a head, **the gate is constant within
every quantization block of `o_proj`, for all three formats.** A per-block activation scale
therefore represents `g_h` with zero residual error — you would simply fold `g_h` into
`scale_block`. For INT4-g128 the group *is* the head, one-to-one.

This is a structural property of this architecture, not a general one, and it means the
worst-case version of the objection (W4A4) is also defused. AWQ's own error analysis
(arXiv 2306.00978 §2.2) gives the mechanism: `Err' = Δ'·RoundErr·(1/s)` with ratio
`(Δ'/Δ)·(1/s)` — an input channel scaled up by `s` has *smaller* relative error. A large gate is
the AWQ-salient case, and AWQ protects it automatically **provided calibration is run with the
gate applied**. That is the one real engineering requirement: calibrate post-gate.

*Cheapest falsifying experiment:* dump `g_h` over 200 tokens; confirm within-head variance is
zero (it must be, by construction) and check the cross-head dynamic range. If max/min `g_h`
exceeds ~10⁴ within a layer, per-head scaling becomes numerically awkward in fp16 accumulate and
you want fp32 accumulation. Cost: one hook, 10 minutes.

### Claim 1.3 — Empirically, `o_proj` is among the **least** 4-bit-sensitive projections; `v_proj` and `down_proj` are the sensitive ones.

**Source:** *From Signal Degradation to Computation Collapse: Uncovering the Two Failure Modes of
LLM Quantization*, arXiv:2604.19884, Table 5. Single-projection-to-4-bit ablation, everything else
FP16, accuracy on a failure subset:

| projection at 4-bit | Llama | Mistral |
|---|---:|---:|
| **O** | **88.61%** | **89.74%** |
| Gate / Up | 70.65% | 86.59% |
| V | 54.44% | 65.68% |
| Down | 49.57% | 45.76% |

`o_proj` is the *most robust* projection measured. The paper also separates the regimes: at 4-bit
the failure mode is "signal degradation" (cumulative noise, patterns intact); "computation
collapse" only appears at 2-bit. It notes Qwen/Gemma degrade more uniformly (50–81% range) with no
dominant failure point — relevant, since Laguna is closer to that family architecturally.

Corroborating: layer-sensitivity work (arXiv:2503.06518, SensiBoost/KurtBoost, Llama-2-7B/13B,
Llama-3-8B) reports K and V projections as the most sensitive weight layers, Q and O
intermediate, MLP most robust, with recommended allocations of ~8 bits to K/V, 6–7 to Q and O,
4–5 to MLP. *Hardware/regime:* accuracy studies, no hardware dependence; transfers.

**Do not overstate this.** The same paper (2503.06518) reports that `self_attn.o_proj` weight
magnitudes "differ significantly between the second layer and the last layer", with the final
layer showing substantial outliers — i.e. **`o_proj` in the last layer specifically may need
protection**. Cheapest mitigation: leave layer 47's `o_proj` at FP8. That costs 28.3 MB of the
1245.7 MB budget — 2.3% of the win. Buy the insurance.

### Claim 1.4 — The gate *removes* the outliers that make quantization hard. The mechanism in premise §4.1 runs backwards.

**Source:** *Gated Attention for LLMs: Non-linearity, Sparsity, and Attention-Sink-Free*,
arXiv:2505.06708 (NeurIPS 2025 best paper), 15B MoE and 1.7B dense, 3.5T tokens. Head-wise gate
applied to the SDPA output — **the same position as Laguna's gate**. Measured:

| | no gate | with head-wise gate |
|---|---:|---:|
| hidden-state mean magnitude, deep layers | ~4.0 | **0.05** |
| first-token attention, averaged over layers | 0.467 | **0.048** |
| first-token attention at layer 21 | 83% | **4%** |
| gate-value sparsity ratio @1e-2 | 0.03 | 0.44 |

Massive activations and attention sinks are *the* documented cause of activation-quantization
difficulty (arXiv:2406.12016; arXiv:2603.05498). A head-wise output gate is a known,
measured **suppressor** of both. The Qwen3-Next quantization analysis states this directly: "the
gated-attention architecture acts as a natural regulator of feature magnitudes… making the model
more robust to the precision loss induced by quantization", alongside QK-Norm constraining the
dynamic range of q/k (Qwen3 shows markedly lower activation kurtosis than Qwen2.5).

**Caveat that must be stated:** the paper's gate is *sigmoid* (bounded 0–1); Laguna's is
*softplus* (unbounded above). The sparsity/sink-suppression half of the argument carries over
(softplus is a soft ReLU: it still drives low-drive heads to ≈0). The magnitude-bounding half does
not. So Laguna keeps the sink-suppression benefit but retains an unbounded upper tail — which
Claim 1.1 shows is irrelevant for W4A16 and Claim 1.2 shows is absorbable for W4A4.

### Claim 1.5 — Published models shipping a 4-bit attention output projection: essentially all of them.

The standard `llm-compressor` W4A16 and NVFP4 recipes are `GPTQModifier(targets="Linear",
scheme="W4A16", ignore=["lm_head"])` — **`o_proj`, `q_proj`, `k_proj`, `v_proj` are all quantized
to 4 bits; only `lm_head` is excluded.** This is the default across the AWQ, GPTQ, and NVFP4
ecosystems. Measured cost, W4A16 NVFP4 (Red Hat / vLLM, evaluated on Open LLM Leaderboard tasks):

| model class | accuracy recovery vs BF16 |
|---|---:|
| MoE (Llama-4, Qwen3-235B-A22B) | **>99%** |
| dense 70B–235B | ~99% |
| ~30B | 97–99% |
| 7B–14B | 95–98% (Llama-3.1-8B worst) |

Laguna S 2.1 is a MoE — the class with the *strongest* reported recovery. For comparison, INT4
weight-only GPTQ/AWQ on Llama-3-8B lands at WikiText-2 ppl ≈ 8.5–9.0 (g128), a small fraction of
a point over FP16. Gemma-4-31B: MMLU 99.2% of BF16 at 8-bit → 97.1% at 4-bit.

Note the NVIDIA `Llama-3.3-70B-Instruct-FP4` card (MMLU 83.3→81.1, GSM8K 95.3→92.6) is **W4A4**,
not W4A16 — do not use it to price this change.

### Claim 1.6 — The strongest counter-evidence, and why it does not apply here.

`bullpoint/Qwen3-Coder-Next-AWQ-4bit` — a 4-bit checkpoint of a **gated-attention** model —
explicitly excludes `self_attn.q_proj`, `k_proj`, `v_proj`, `o_proj` from 4-bit "to preserve
model quality". This is the closest thing in the wild to a practitioner refusing to 4-bit a gated
`o_proj`.

Three reasons it is weak evidence:

1. The README supplies **no measured justification and no ablation** — it is folklore, exactly as
   premise §4.2 suspects.
2. Qwen3-Next-80B-A3B is an MoE where attention is a few percent of parameters. Excluding it is
   nearly free, so nobody had to measure whether it was necessary. **In Laguna, attention is
   44.9% of `B_tok`** — the trade is completely different and has never been faced.
3. The genuinely-required exclusions in that family are the *linear-attention* (Gated DeltaNet)
   projections `in_proj_a` / `in_proj_b` / `in_proj_ba`, which are state-space gains, not the
   softmax-attention `o_proj` (vLLM issue #40252; RedHatAI/Qwen3.6-35B-A3B-NVFP4 discussion #4).
   The literature's "don't quantize the gate projections" rule is about **DeltaNet state
   decay**, and it has been sloppily generalised to gated softmax attention.

Laguna's own artifact is `Laguna-S-2.1-NVFP4`: experts NVFP4, attention FP8. That is the
llm-compressor MoE default, chosen when attention is small. **Laguna's attention is not small.**
This is the best available explanation of why the question was never asked.

### Verdict on premise §4.1

**Refuted as stated**, on mechanism (Claim 1.1: scale-invariance for weight-only), on structure
(1.2: gate aligns with every block boundary), on direct measurement of the projection's relative
sensitivity (1.3), on the physics of what gating does to outliers (1.4), and on shipping practice
(1.5). The only surviving concerns are (a) last-layer `o_proj` weight outliers — insure for 2.3%
of the win; (b) calibration must be run post-gate.

---

## 2. Priority question 2 — is "q/k/v never below FP8" supported?

**No.** It is folklore of exactly the kind §4.2 suspects. Evidence:

* The default W4A16/NVFP4 recipe quantizes all four to 4 bits with `ignore=["lm_head"]` only.
* Per-projection ablation (2604.19884 Table 5) puts O at 88–90% and V at 54–66% — i.e. the rule,
  if it existed, should read "protect V and Down", not "protect all of q/k/v/o".
* KV-cache literature (KIVI, arXiv:2402.02750; KVTuner, arXiv:2502.04420) shows the K/V
  *sensitivity* is a **channel-wise outlier** phenomenon in the K states, cured by per-channel
  granularity — not an argument for higher bit width. Laguna's K path already has **QK-RMSNorm
  applied per head pre-RoPE**, which normalises away exactly the per-head magnitude variation that
  KIVI is fighting. QK-Norm is independently credited (Qwen3 vs Qwen2.5 kurtosis) with making
  q/k the *easiest* tensors to quantize in a modern architecture.

**The architecture-specific consequence:** Laguna's `q_proj` output is immediately RMSNormed per
head. Any per-head gain error introduced by weight quantization of `q_proj` is **divided out** by
QK-RMSNorm before RoPE. `q_proj` is therefore *more* forgiving of 4-bit than a model without
QK-Norm, and it is worth exactly as many bytes as `o_proj`. **`q_proj` at 4 bits is the
lowest-risk +7.3% in this document and it is not in the prompt.**

k/v are not worth attacking at 4.83% of `B_tok` regardless of how they behave.

---

## 3. Priority question 3 — fastest batch-1 W4A16 / W8A16 skinny-GEMM kernels

Ranked by transferability to 20 SMs / 227 GB/s / batch 1.

### 3.1 Transferable: instruction count, not dequantization math, is the limiter.

**Source:** an ncu-instrumented reimplementation of llama.cpp's Q4_K `mul_mat_vec_q`, Mistral-7B,
**A100 80GB PCIe** (vijayprabhas9.github.io/gemv_optimization). Arithmetic intensity 2 ops/byte.

| variant | time | memory throughput | instructions |
|---|---:|---:|---:|
| activation-tiled | 80.58 µs | 19.2% | 7.33 M |
| weight-tiled | 32.67 µs | 38.4% | 6.43 M |
| weight-tile + 4 warps/CTA | 29.82 µs | 44.6% | 7.33 M |
| custom "2pack" | 22.27 µs | 47.5% | 7.33 M |
| **llama.cpp upstream** | **20.03 µs** | 36.5% | **4.78 M** |

The production kernel wins on *fewest instructions* despite higher register pressure and lower
apparent memory throughput. Author's conclusion: "instruction count dominates performance at low
arithmetic intensity"; a 53% instruction excess produced the 11% gap. Also flagged: uncoalesced
activation loads cost ~60% excess sectors across all variants — an untapped ~20%.

This is the single most directly transferable result, because it is the same failure mode as the
already-measured 16-ALU-ops-per-word budget on this device. **The design rule for a Laguna NVFP4
`o_proj` GEMV is: minimise instructions per 32-bit word, not maximise anything else.**

An NVFP4 word holds 8 e2m1 values. With hardware `cvt.rn.f16x2.e2m1x2` (§5 memory: FP4x2-unpack
is present on sm_110a) that is 4 cvt + 4 `hfma2` = **~1.0–1.5 instructions per weight**, against
QTIP 3INST's ~3 ops/weight which measured 395 Gweight/s. Linear extrapolation puts NVFP4 GEMV
ALU-capped at ~800–1200 Gweight/s, far above the 404 Gweight/s that 227 GB/s allows at 4.5 bpw.
**Prediction: NVFP4 `o_proj` is bandwidth-bound, not ALU-bound, on this device.** This is the
whole ballgame and it is a one-hour microbenchmark.

### 3.2 Transferable-with-caveat: FLUTE.

*Fast Matrix Multiplications for Lookup Table-Quantized LLMs*, arXiv:2407.10960, **A6000 and
A100**. 2–4× over existing GEMM kernels at **batch ≤ 32, group size 128** — exactly the decode
regime — and 1.5–2× end-to-end on LLaMA-3. Techniques worth stealing: (a) **offline restructuring
of the packed weight matrix** so unpacking costs minimal bit manipulation at runtime; (b)
**vectorised + duplicated lookup tables** to dodge shared-memory bandwidth limits. The paper notes
other LUT kernels match it only at batch=1 before degrading — so at Laguna's M=1..16 speculation
range, the simpler competitors are fine at M=1 and FLUTE's advantage appears as M grows.
**Caveat:** §2.2 already measured on *this* device that a shared-memory codebook costs 31% at 3
bits (271 vs 395 Gweight/s). FLUTE's LUT approach therefore imports a known-bad primitive here.
Take the offline-restructuring idea; leave the LUT.

### 3.3 Transferable: dequantization is not the bottleneck.

arXiv:2601.16536 (Ascend 910 NPU): an INT4×FP16 kernel issues **64.66% more instructions** than
FP16×FP16 but costs only **2.89% more cycles / 2.45% more time** — ILP hides the dequant latency
entirely. Their actual bottleneck was an architecture-specific global-memory round trip between
vector and cube cores, which **does not exist on CUDA** and does not transfer. Also: Split-K beat
data-parallel by 1.01–1.74× when K ≫ N — but Laguna's `o_proj` is K=9216, N=3072, i.e. K ≈ 3N, a
mild case; and with 20 SMs and N=3072 rows there is already 150 rows/SM of parallelism, so
**Split-K is unlikely to be needed**. Stream-K / Split-K exist to fix wave quantization when
M·N/tile < SM count; that is not this shape.

### 3.4 Not transferable — flag these when they are cited at you.

* **Marlin / AWQ-Marlin / Machete.** Designed for batch 16–64 on A100/H100 with tensor cores as
  the target. At M=1 there is no tensor-core work to feed. Machete explicitly targets Hopper+.
* **LiquidGEMM** (2.90× kernel, 4.94× system): W4A8 GEMM, **datacenter serving, large batch**.
  Does not transfer.
* **arXiv:2605.30571** (44-cell cross-GPU batch-1 study, **H100 / A100 / L40S / L4, no Jetson**)
  is the most honest general result and it cuts *against* naive quantization optimism: on L4,
  baseline 62.32 ms/step; bnb-nf4 **59.36** (a 5% gain, not 4×); AutoAWQ+Marlin 45.24;
  GPTQ+ExLlamaV2 17.36. The paper's own summary is that "common quantized approaches fail to
  deliver the expected 4× weight-traffic reduction." Its bandwidth-limited GPU (L4, 81% of peak)
  is the closest analogue to Thor; its H100 (27% of peak, 1.259× from CUDA graphs alone) is not.
  **Read this as: the win is real but only if the kernel is good; a bad W4 kernel gives 5%.**
  ExLlamaV2's Ada-tuned int4 kernels got 3.6× on the same GPU, which is the existence proof that
  the kernel — not the format — is the variable.

---

## 4. Priority question 4 — fusing RMSNorm + cast into the consumer GEMV prologue

### Claim 4.1 — "Lazy Pre-Norm": at M=1 the pre-RMSNorm becomes a **scalar** epilogue multiply and the norm kernel can disappear entirely.

**Source:** PyTorch blog, *Towards Free Normalization: Fusing Normalization into GEMM and
Attention Kernels*, **B200, bf16, Meta production ads shapes**. The identity is:

```
(A * rstd[:, None]) @ B  ==  (A @ B) * rstd[:, None]
```

so the prologue norm is computed *concurrently with* the GEMM's K-loop (`square_sum +=
(tile_A*tile_A).sum(-1)`) and applied in the epilogue. Measured **17–32% latency saving vs
torch-inductor's normalization kernel at N = 64 and 128**, degrading to negative at N ≥ 256
because `square_sum` is redundantly recomputed across CTAs. Their separate multi-CTA epilogue
variant gets 30–40% at M=256K.

**Why Laguna's regime is the best case, not the degrading case.** Their redundancy penalty scales
with the number of CTAs times the cost of re-reducing A. At M=1, A is a single 3072-element
vector (6 KB). Every CTA can redundantly compute its sum of squares in 3072 FMAs, against
3072×9216/n_CTA ≈ 177 K MACs of real work per CTA at 160 CTAs — **1.7% extra ALU**. The blog's
"suboptimal as K/N grows" caveat is about *large-M* redundancy and does not bind here.

### Claim 4.2 — The affine weight is not a blocker; fold it into the consumer weight matrix offline.

The blog lists "cannot support elementwise affines" as a hard limitation. That is true for a
general kernel and false for this deployment, because every consumer of a normalized vector here
is a linear layer with a static weight:

```
((x * rstd) ⊙ w) @ B  =  (x * rstd) @ (diag(w)·B)  =  (x @ (diag(w)·B)) * rstd
```

`diag(w)·B` is precomputed once at load time — **zero runtime cost, zero extra memory**. This
works for `input_layernorm` (consumers: the already-fused qkv+gate GEMV) and for
`post_attention_layernorm` (consumers: router and experts). It does **not** work where the
normalized vector is consumed non-linearly or by more than one differently-scaled path without
duplicating the fold, and it does **not** apply to QK-RMSNorm (whose output goes into RoPE, not a
GEMM).

*Category:* `norm+cast` (6.8% of step = **2.06 ms**), plus a slice of `attn_qkvo_gemm` launch
overhead. Removing the two per-layer full-width RMSNorms removes 96 launches (96 × 1.6 µs =
**154 µs floor**) plus their DRAM round trips.

*Exact?* **No — this reorders floating-point operations.** Folding `diag(w)` into `B` and moving
`rstd` past the accumulation both change rounding. The §1 exactness invariant is that the
*speculative* path match this system's own AR decode; since both use the same fused kernels, that
invariant is preserved. Bit-identity with the reference HF implementation is **lost**. Flag this
to the owner before implementing.

### Claim 4.3 — But profile `norm+cast` first, because its shape does not match the fusion story.

2.06 ms / 96 launches = **21.5 µs per launch**, which is **13× the 1.6 µs launch floor** for an
operation touching 6 KB. A 3072-element RMSNorm should take ~2 µs. Either the launch count is not
what it appears, or some of those 96 kernels are doing something much larger — most plausibly
per-token **NVFP4/FP8 activation quantization** for the expert path, which is a different problem
with a different fix (fuse it into the *producer's* epilogue, not the consumer's prologue).

**This is the cheapest high-information experiment in this document** and it gates everything in
§4: one `ncu` pass over one decode step, bucketing the 96 `norm+cast` kernels by name and
duration. Thirty minutes. If they are 96 × 21.5 µs of genuine RMSNorm, Lazy Pre-Norm is worth up
to +3.5% (removing half the category) and possibly +7.3% (all of it). If they are 48 fast norms
plus 48 slow quantize kernels, the fusion target is completely different and §4.1–4.2 are worth
under +1%.

### Claim 4.4 — Megakernel results are from GPUs 6–7× larger and should be discounted.

* Hazy Research: whole Llama-1B decoder in one persistent kernel, **78% of H100 memory
  bandwidth** at batch 1, vs vLLM 50% / SGLang 51%.
* Mirage Persistent Kernel (CMU Catalyst): SM-level dependency graph, **1.0–1.7× over
  SGLang/vLLM on A100/H100**; up to 6.7× in some configurations.
* Deep Kernel Fusion (ACL 2026 short): up to **13.2% on H100, 9.7% on A100** over SGLang, from
  fusing SwiGLU blocks.

All three attack the *H100 regime* where launch overhead dominates because a kernel cannot fill
132 SMs. Thor has 20 SMs and this category already runs at 267 GB/s = 118% of streaming ceiling
— i.e. `attn_qkvo_gemm` is **not** launch-starved. The measured cross-GPU study makes the same
point from the other side: CUDA graphs bought 1.259× on H100 but only **1.028× on L4**. Thor is
an L4-like device. Expect the L4 number.

**Corollary that retires part of question 5:** 96 launches × 1.6 µs = 154 µs = **0.51% of the
step**. Launch overhead is not the problem in this category. The launch count also tells us
qkv+gate is **already fused** (2 launches/layer × 48 = 96), so the obvious GQA/fusion win has
already been taken. Bit width is the only remaining lever here.

---

## 5. Priority question 5 — GQA / per-layer head counts

Nothing worth doing. Three reasons, all arithmetic:

1. GQA has already reduced `k_proj`+`v_proj` to **4.83% of `B_tok`**. Both layer types emit
   1024 KV dims (48/6 = 8 and 72/9 = 8 heads — a nice invariant, but one with no exploitable
   consequence at batch 1).
2. `q`/`k`/`v`/gate are already fused into one launch per layer (see §4.4).
3. The differing head counts (48 vs 72) mean two GEMV shapes rather than one. The only cost is
   two specializations or one dynamic-N kernel; at N ∈ {6144, 9216} with 20 SMs both are far
   above the wave-quantization threshold, so there is no tail effect to fix.

The one thing the per-layer structure *does* buy is under §1.2: head_dim 128 divides every
quantization group size cleanly in both layer types, so no group ever straddles a head boundary
and the per-head gate is always exactly representable as a block scale.

---

## 6. Ranked recommendations — (payoff × uncertainty) / cost

| # | action | payoff | uncertainty | cost | score |
|---|---|---:|---|---|---|
| **1** | **`ncu` bucket of the 96 `norm+cast` kernels** | gates up to +7.3% | high — the 21.5 µs/launch number is unexplained | 30 min, no code | **highest** |
| **2** | **Dense NVFP4 GEMV microbenchmark at 3072×9216, M=1** | decides +15.5% vs +17.9% vs +5% | high — is it BW- or ALU-bound? | 1 hr, existing MoE kernel reshaped | **highest** |
| **3** | **RTN NVFP4 `o_proj` + gate-gain sweep** (falsifies §4.1) | unlocks +7.3% | high — the premise is untested | ½ day, CPU-only | **very high** |
| **4** | **`q_proj` → NVFP4** | **+7.3%** | low — QK-Norm downstream, no gate | same PTQ pipeline | **very high** |
| **5** | `o_proj` → NVFP4, layer 47 held at FP8 | +7.1% | medium | same pipeline | high |
| **6** | `q`+`o` → INT4-g128 instead of NVFP4 | +16.9% | medium — cheaper unpack, group = head exactly | one more kernel | high |
| **7** | Lazy Pre-Norm fold of the two per-layer RMSNorms | ≤ +3.5% | medium, gated on #1 | 2 days CUDA; breaks bit-identity vs HF | medium |
| **8** | Fix uncoalesced activation loads in the GEMV | ~20% of the kernel (blog) → ≲+1% here | medium | 1 day | low |
| **9** | k/v/gate → 4-bit | +1.7% | low | free once #4 exists | low |
| **10** | Split-K / Stream-K for the projections | ~0 | low — shape is not wave-quantized | — | **do not** |

**Combined realistic target for this axis: +15.6% end-to-end** (`q_proj` + `o_proj` → NVFP4,
last-layer `o_proj` held at FP8, k/v/gate left alone), with a pessimistic ALU-capped floor of
+13% and an optimistic INT4-g128 ceiling of +20%. No retraining. Not exact — it is a quality
trade, and quality must be measured directly (§2.5: τ is *not* a quality proxy; a degraded target
raises τ). Use held-out perplexity plus a greedy-match rate against the FP8 baseline.

---

## 7. Premises I could not refute

* **§4.2, partially.** I refuted the general form ("q/k/v cannot go below FP8 at all") — the
  default 4-bit recipes quantize them and the ablation evidence puts O and Q among the most
  robust. But I found **no measurement of `v_proj` at 4 bits in a model whose attention is 45% of
  its bytes**, and 2604.19884 puts V at 54–66% — the second-worst projection. `v_proj` at 4 bits
  is *not* cleared by this pass. It is also only 2.4% of `B_tok`, so this does not matter.
* **The last-layer `o_proj` outlier finding (2503.06518)** stands unrefuted and I have no
  measurement on Laguna. Mitigation is cheap (hold one layer at FP8, 2.3% of the win) so I did
  not pursue it further.
* **Nothing in the literature measures a *softplus* (unbounded) attention output gate under
  quantization.** All the gating evidence — Qwen3-Next, arXiv:2505.06708, the GLU
  activation-spike literature — is sigmoid or SiLU, i.e. bounded or self-bounding. Claim 1.1's
  scale-invariance argument is a derivation of mine, not a citation, and it is the load-bearing
  element of this whole document. **Experiment #3 exists precisely to falsify it and should be
  run before anything is committed.**
* **The 267 GB/s effective rate of this category exceeds the 227 GB/s streaming ceiling.** I
  attribute it to L2 (a sliding `o_proj` is 28.3 MB against a 32 MB L2) but did not verify. If
  the true mechanism is something else, the conversion table in §0 is wrong by up to 18%.

## 8. Sources

- [From Signal Degradation to Computation Collapse (arXiv:2604.19884)](https://arxiv.org/abs/2604.19884)
- [Towards Superior Quantization Accuracy: A Layer-Sensitive Approach (arXiv:2503.06518)](https://arxiv.org/pdf/2503.06518)
- [Gated Attention for LLMs: Non-linearity, Sparsity, Attention-Sink-Free (arXiv:2505.06708)](https://arxiv.org/abs/2505.06708)
- [AWQ: Activation-aware Weight Quantization (arXiv:2306.00978)](https://ar5iv.labs.arxiv.org/html/2306.00978)
- [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs (arXiv:2404.00456)](https://arxiv.org/abs/2404.00456)
- [Prefixing Attention Sinks Mitigates Activation Outliers (arXiv:2406.12016)](https://arxiv.org/abs/2406.12016)
- [KIVI: Asymmetric 2-bit KV Cache Quantization (arXiv:2402.02750)](https://arxiv.org/abs/2402.02750)
- [Memory-Bound but Not Bandwidth-Limited: 44-Cell Batch-1 Decode Study (arXiv:2605.30571)](https://arxiv.org/abs/2605.30571)
- [FLUTE: Fast Matrix Multiplications for LUT-Quantized LLMs (arXiv:2407.10960)](https://arxiv.org/abs/2407.10960)
- [W4A16 on Decoupled Architecture / Ascend (arXiv:2601.16536)](https://arxiv.org/html/2601.16536)
- [Towards Free Normalization: Fusing Normalization into GEMM and Attention Kernels (PyTorch blog)](https://pytorch.org/blog/towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/)
- [Accelerating LLMs with NVFP4 Quantization (Red Hat Developer)](https://developers.redhat.com/articles/2026/02/04/accelerating-large-language-models-nvfp4-quantization)
- [INT4 W4A16 recipe — vLLM / llm-compressor docs](https://docs.vllm.ai/en/stable/features/quantization/llm_compressor/int4/)
- [I Built a Quantized GEMV Kernel from Scratch (A100 ncu study)](https://vijayprabhas9.github.io/gemv_optimization/)
- [We Bought the Whole GPU (Hazy Research megakernel)](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)
- [Mirage Persistent Kernel (CMU Catalyst)](https://catalyst.cs.cmu.edu/projects/mpk.html)
- [Deep Kernel Fusion for Transformers (ACL 2026)](https://aclanthology.org/2026.acl-short.15/)
- [vLLM issue #40252 — Qwen3-Next NVFP4 linear_attn ignore list](https://github.com/vllm-project/vllm/issues/40252)
- [bullpoint/Qwen3-Coder-Next-AWQ-4bit README (excludes gated-attn projections)](https://huggingface.co/bullpoint/Qwen3-Coder-Next-AWQ-4bit)
- [nvidia/Llama-3.3-70B-Instruct-FP4 (W4A4 reference)](https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-FP4)
