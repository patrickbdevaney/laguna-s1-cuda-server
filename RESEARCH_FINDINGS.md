# RESEARCH_FINDINGS.md — synthesis of the deep-research pass

Commissioned via `RESEARCH_PROMPT.md`, five axes, grounded in the measured ledger so agents
could not re-propose tested losers. This file records what survived scrutiny, what was
refuted, and — where it happened — where the research **corrected our own ledger**.

Status: Axis D complete. Axis B partially in (two sub-results below). A, C, E in flight.

---

## ★ THE DECISIVE FINDING — quantizing attention is not free, and the industry knows it

The single most important result of the whole pass. IST-DASLab shipped **two otherwise
identical** DeepSeek-R1 W4A16 checkpoints (GPTQ, 4-bit, group 128, symmetric) differing only
in whether attention is quantized — a clean controlled A/B on a large MoE:

| checkpoint | what is INT4 | reasoning avg | recovery |
|---|---|---:|---:|
| DeepSeek-R1 reference | — | 82.99 | 100 % |
| `ISTA-DASLab/DeepSeek-R1-GPTQ-4b-128g` | experts **+ attention** | 80.10 | **96.52 %** |
| `ISTA-DASLab/DeepSeek-R1-GPTQ-4b-128g-experts` | experts **only** | 82.00 | **98.81 %** |

**Quantizing attention costs ~2.3 points of recovery.** Corroborated across two Red Hat
releases: DeepSeek-R1-0528 w4a16 (excludes `self_attn`, `shared_experts`, dense MLP) reaches
99.82 % with AIME 88.66 → 87.33; GLM-4.6 w4a16 (excludes only `lm_head`) drops AIME
96.67 → 90.00, i.e. 93.10 %.

**Poolside's choice was deliberate and is the industry norm**, not an oversight we can simply
correct. Every large MoE in this survey that ships a careful quantization excludes attention:

| model | attention quantized? |
|---|---|
| Kimi-K2-Instruct, DeepSeek-R1-0528 | **No** — only routed experts are INT4 |
| Llama-4 Scout, Llama-4 Maverick | **No** — excludes all q/k/v/o across 48 layers, routers, `lm_head` |
| Qwen3-Next-80B-A3B, Qwen3-30B-A3B, Mixtral, GLM-4.6 | Yes |

Most telling: `nvidia/DeepSeek-R1-NVFP4-v2`'s entire headline change over v1 is that it
"additionally quantizes the **`wo`** module in attention layers." Even NVIDIA, pushing NVFP4
hardest, adds only `o_proj` and leaves Q/K/V in higher precision.

**Scale makes it worse for us.** Qwen's own GPTQ-Int4 (attention quantized) loses 0.8 GPQA
points at 22 B active but **5.7 points at 3.3 B active**. Laguna is **8.5 B active** — between
them, nearer the fragile end.

### What this does to `ROOFLINE.md` §5

The lever survives but its staging is now evidence-driven rather than guessed:

| stage | target | `B_tok` | AR @227 | expected quality | confidence |
|---|---|---:|---:|---|---|
| 0 | today (stock) | 10.04 | 22.6 | — | — |
| **1** | **FP8 (e4m3) on all attention** | **7.24** | **31.4** | near-lossless | **high** |
| 2 | + NVFP4 on `o_proj` only | ~6.9 | 33 | small, NVIDIA-validated | medium-high |
| 3 | + NVFP4 on q/k/v | 6.02 | 37.7 | **~2 pts recovery, measured** | needs eval |
| 4 | + shared experts, router, `lm_head` | 4.72 | 48.1 | compounding | needs eval |

**Stage 1 is the one to build.** INT8 weight-only (W8A16) is *measured* lossless — llama.cpp
Q8_0 on Llama-3.1-8B: ppl 7.32 → 7.33, benchmark avg 69.47 → 69.41. FP8 E4M3 weight-only has
no isolated published study, but its error is a strict subset of W8A8 (~99.7–100 % recovery
with plain RTN), so it is at least that good. It is a **1.39× reduction in `B_tok` for
near-zero risk**, and it needs no calibration data.

Stage 3 is where the 2.3-point cliff is. Do not do it without an eval harness.

## Format evidence (if we ever go below 8 bits on attention)

Cleanest weight-only head-to-head (arXiv 2509.23202 Table 3, Llama-3.1-8B, **W4A16**,
FP16 base 79.70):

| format | RTN | GPTQ | AWQ |
|---|---:|---:|---:|
| INT4-g32 | 77.12 | 76.60 | — |
| **NVFP4** | 77.37 | 77.33 | **77.40** |
| MXFP4 | 75.44 | 76.37 | 75.18 |

NVFP4 ≈ INT4-g32; MXFP4 is 1–2 points behind. **And the reason is the scale format, not the
block size** — arXiv 2507.17417 Table 13 (Llama-3.2-1B, W4A16, WikiText-2, BF16 = 9.76):
E8M0 → E4M3 at fixed g16 is **14.52 → 11.54 (−2.98 ppl)**, while g32 → g16 at fixed E4M3 is
only **12.04 → 11.54 (−0.50)**. **~82 % of the NVFP4/MXFP4 gap is the E4M3 scale.** Our
checkpoint already uses E4M3 per-16 — the good configuration.

Two traps recorded: plain FP4/E2M1 at g128 is the *worst* 4-bit format (Llama-3-70B WikiText-2:
FP16 2.86, INT4 3.63, **FP4 3.94**, NF4 3.43) — NVFP4's advantage is entirely g16 + the
two-level E4M3/FP32 scale. And **Hadamard rotation helps INT4 W4A4 but HURTS NVFP4
weight-only** (77.37 → 76.41): do not import W4A4 recipes into a weight-only setting.

## Evaluation methodology — a trap we would have fallen into

**OpenLLM v1 recovery is not a usable signal.** Every recipe in the survey, including the bad
ones, scores 98–100 % on it. Only **AIME, GPQA-Diamond and BBH** discriminate. If we quantize
attention and validate on an easy benchmark, we will conclude it was free when it was not.

Also flagged: several HF model cards order columns `Recovery | Baseline | Quantized`, which
markdown-to-text conversion silently reorders. Verify against raw markdown.

## Evidence holes (stated, not papered over)

- **No measured W4A16 accuracy for GLM-4.5-Air 106B-A12B** — the closest architectural
  analogue to a 117 B/8.5 B MoE. Community quants publish no eval tables.
- No published INT4 group-size sweep on any model ≥ 70 B; all controlled sweeps are ≤ 7 B.
- No isolated FP8-weight-only measurement anywhere.
- NVIDIA has published **no** inference PTQ NVFP4-vs-MXFP4 accuracy table; their MXFP4 claims
  in the marketing blog carry no numbers.

---

## Axis D — launch, scheduling, fusion · **EXHAUSTED**

Ran its own on-box probes. Measured CPU-side enqueue **2.06 µs** empty / **3.01 µs** for a
12-arg kernel — the ARM cores are 2–3× a modern x86 host, which is why graphs matter *more*
here than on a discrete part.

**It corrected two errors in our ledger** (both now fixed):
1. We listed FlashNorm weight-folding as bit-exact. It is not — rounding `g·W` to bf16 offline
   is a real precision change across 5.6 GB of attention weights, and it targets a
   latency-bound kernel. Superseded by the fused add+RMSNorm+cast, which *is* bit-exact.
2. Our CUDA-graph estimate of +10–20 % was too generous; measured **+7.8 %**. Of the 4.15 µs
   per launch only ~2.6 µs was recoverable; ~1.5 µs is irreducible null-kernel execution.

**Megakernel: quantitatively killed.** In-graph launch residual is ~0.7 µs × 1665 ≈ 1.2 ms
= **2.1 % of the step** — that is the hard ceiling on every launch-count lever. A
sentinel-poll persistent kernel would spend ~0.68 ms on ~800 dependency edges to recover it,
netting <1 %, and would force **global register allocation** across all stages — which would
undo the two biggest wins (the `TM=1` MoE at 50 registers instead of 127, and the
lane-per-output repack, which is warp-count-hungry). Widen the DEAD entry: sentinel polling is
8× cheaper than the `grid.sync()` we measured, and still not worth it.

**Remaining Axis D levers (~+3 % total), all bit-exact:**
- ✅ fused add+RMSNorm+cast — **built, +2.4 %**
- concatenate q\|k\|v\|g into one GEMM: +0.7 %. `g_proj` alone is 3.34 µs for 0.44 MB and
  launches **72 blocks onto 20 SMs** — a rounding error of bytes costing a full kernel slot.
- fuse `moe_invert`'s 3 memsets + 4 kernels into one at M=1: +0.5 %, and *more* deterministic
  (the current `atomicAdd` cursors have scheduling-dependent ordering).
- producers write bf16 directly; fuse QK-norm + RoPE + `store_kv`: ~+0.7 %.

**Multi-stream concurrency measured a LOSS** (298.9 vs 297.3 µs in-graph vs concatenation).
Rule: *if two kernels read the same activation and write disjoint outputs, concatenate the
weights — never stream them.*

**Design constraint for Gate D1:** conditional graph nodes (`cudaGraphCondTypeWhile/If`,
available on CUDA 12.8+/Blackwell; we have 13.0.48) can keep the whole draft→verify→accept
loop on-device. At 2–3 µs enqueue on these ARM cores, a per-block host round-trip is expensive
in exactly the regime speculation is supposed to win. Decide this before building the verify path.
