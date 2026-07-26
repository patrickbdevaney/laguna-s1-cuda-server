# RESEARCH_FINDINGS_V3.md — explore pass (UCB over the hypothesis space)

Two parallel passes framed as *explore*, not exploit: high-uncertainty × high-payoff edges, with
the tried-and-failed list supplied up front so nothing already paid for came back. Several
questions turned out to be answerable **on this device** rather than from papers, and those
measurements are the most valuable part.

---

## A. The 19 % base-decode gap is now decomposed, on our own silicon

A synthetic replica of our step structure — 6.251 GB of DRAM reads split into *NK* dependent
kernels in one CUDA graph, perfect `uint4` coalescing, no writes, no math — is a hard upper
bound on what our step can reach at a given kernel count:

| kernels/step | tok/s |
|---:|---:|
| 1 | 39.6 |
| 240 | **40.56** |
| 480 | 40.16 |
| **1665 (ours)** | **36.96** |
| 3330 | 33.76 |

Marginal cost ≈ **1.6–1.7 µs per kernel**, and the byte model reproduces our predicted 40.6
tok/s almost exactly. So:

* **~9.0 points of the gap is kernel-count structure alone** — a floor, not an estimate. At
  1665 kernels we cannot exceed 36.96 tok/s even with zero compute and zero writes.
* **~10.7 points is everything else** — real access patterns, activation round-trips,
  occupancy shortfalls, ramp/drain.

### Two things this kills outright

* **Wave quantization / tail effect is worth ~0 %.** Bandwidth is flat from 240 blocks to
  20 000 on our 20 SMs. The published tail-effect pathology is a *compute*-bound GEMM effect.
  This is why our split-K and stream-K results were neutral, and it retires persistent-CTA
  redesigns as a direction.
* **Compressible memory is unavailable.** Measured on this device:
  `CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 0`, and `cuMemCreate` with
  `CU_MEM_ALLOCATION_COMP_GENERIC` fails outright. The `lts__gcomp_*` metrics exist because the
  namespace is shared across Blackwell, not because the iGPU has the feature. NVFP4 weights are
  maximum-entropy anyway.

### One immediate lever

At **grid = 80 blocks** (our MoE gate/up shape), bandwidth goes **213 GB/s at 128 threads/block
→ 249 GB/s at 256**. The binding constraint is *resident threads per SM*, not block count. Also
worth noting the peak observed was **262 GB/s**, ~3 % above our 254 assumption.

---

## B. Trellis quantization of the routed experts is compute-free here — measured

The routed experts are 2.495 GB, 40 % of the remaining byte budget and the last big block. The
open question was whether a more expensive codec (trellis, codebook) eats the byte win at batch 1.

**Measured on this device:** Thor absorbs up to **16 ALU instructions per 32-bit word — 6.0
instructions per weight at 3.0 bpw — with zero bandwidth loss, even on a fully serial
dependency chain**, and **≥10 shared-memory codebook lookups per word for free**.

QTIP's `3INST` decode is 3 ALU instructions per weight; `1MAD` is 4. Both sit inside that budget
with ~2× headroom. This confirms with numbers the asymmetry we suspected when removing the FP4
decode and the entire FMA chain from the MoE kernel bought nothing.

| target | Δ `B_tok` | projected |
|---|---:|---:|
| 3.5 bpw | −0.554 GB | 36.2 tok/s (+9.6 %) |
| **3.0 bpw** | **−0.832 GB** | **38.0 tok/s (+15.2 %)** |

Accuracy is the remaining risk and the evidence is favourable *for routed experts specifically*:
MoQE finds "expert layers are much more robust to quantization than conventional FFN layers"
(2-bit expert-only viable); GEMQ reaches 1.5 bits/expert on Mixtral / DeepSeek-V2-Lite / OLMoE /
Qwen3-30B-A3B. The contrary result — that MLP up/down projections dominate FP4 sensitivity —
was measured on *dense* MLPs, which is an argument for leaving our shared experts and layer-0
dense at FP8, exactly where they are.

**Decisive cheap experiment:** quantize routed experts only to 3.0 bpw offline, evaluate
AIME/GPQA-D/LiveCodeBench in PyTorch. The hardware question is already settled, so this
decouples the risky question from the expensive one.

---

## C. The correction that matters most: our framing of speculation was wrong

Fitting a cost model to our own three E_frac points gives, in units of one AR decode step:

```
cost(M) = 0.235 + 19.6 · E_frac(M)        cost(1)=1.00, cost(5)=3.00, cost(16)=5.41
drafter cost c = 2.23/6.251 = 0.357
gain = τ(M) / (c + cost(M))
```

We had concluded "everything that raises τ is fighting the wrong variable." **That is
quantitatively wrong.** Raising τ *at fixed M* is a 1:1 multiplier at **zero** cost. Raising τ
*by raising M* costs 0.50 decode-steps per slot. Our controller only searches over k — so it can
only ever buy τ the expensive way, which is why it looked like the wrong variable.

At M=5, τ 3.67 → 5.04 reaches 1.5× (~49.5 tok/s on code) **with no change to the verify at all.**

Three corollaries:

1. **k\* = 3–4 is correct, not a mis-tuning.** The model reproduces it independently, and
   verifying the full 16-token block is a *loss* (needs τ ≥ 8.65 to reach 1.5×). Our rejection
   of tree/multi-candidate verify was right for a reason we hadn't quantified.
2. **Our drafter cost ratio c = 0.357 is ~7× the entire literature's.** Every published DFlash
   speedup (5.5×, 6.7×, 9.6×) is against a *dense* target where a 2 GB drafter is 1–5 % of a
   forward; against an 8.5 B-active MoE it is 36 %. **No published speedup number transfers to
   us.**
3. **Our τ is at published baseline, not misconfigured.** DFlash on Qwen3.5-122B (≈10 B active
   MoE — functionally our model) measures τ 2.4–3.4 across AIME/GPQA-D/LiveCodeBench/MATH-500.
   Our 3.06 math / 2.26 prose sits squarely inside that band.

### The one asset nobody else has

Our expert union is **22 % below independent routing at M=5 and 44 % below at M=16**. The
independent model predicts E_frac 0.181 and 0.468; we measure 0.141 and 0.264. Published MoEs
are at or near independent (DeepSeek-V3.1 essentially exactly). **No paper publishes an E_frac(M)
curve for any model.** Laguna's routing is the most position-correlated measured anywhere, and
every number in this section is downstream of it.

---

## D. Ranked next actions

| # | action | payoff | cost | exact? |
|---|---|---|---|---|
| 1 | Log τ against **prompt length** vs **generated position** separately | gates everything below | hours | — |
| 2 | Certified lazy expert evaluation — offline probe over logged gate weights | up to +14 %, unknown | one afternoon | **yes by construction** |
| 3 | 256 threads/block on kernels with ≲240 blocks | up to +17 % on those | ~1 hour | yes |
| 4 | Routed experts → 3.0 bpw trellis (accuracy eval first) | **+15.2 %** | days | changes target, not correctness |
| 5 | Long-context drafter fine-tune (1.6 K samples, 3 epochs) | τ 3.61 → 6.05 published | hours on one GPU | yes |
| 6 | Retrain drafter with chain + first-error focal loss | **τ +21–76 %**, free at inference | 8×H100-days | yes |
| 7 | Test-time drafter adaptation (TTS) | **1.66× measured on Qwen3.5-122B** | high | yes |
| 8 | Fuse 1665 → ~400 kernels via SM-level deps (not `grid.sync`) | +8.0 % measured ceiling | weeks | yes |

**Items 5–7 are all pure drafter-side, cost nothing at inference, and preserve exactness.** They
are the only interventions with the magnitude to clear 1.5× on their own, and both TTS and the
loss-shaping work were measured on architectures close to ours.

### Newly retired

Wave quantization / persistent CTAs (measured ~0); compressible memory (measured unsupported);
cross-layer weight sharing (needs retraining); MLX-derived techniques (we already have them);
block > 16 without retraining at that block size (τ collapses 5.35 → 3.66); HiSpec two-pass
verify (not distribution-preserving); sparse-computation verify (lossy); LoRA/PEFT drafting
(published negative: good prefixes, zero speedup).

### Our own correction, recorded

The earlier rejection of EcoSpec may have tested the wrong thing: its rule is a *selection* rule
needing a surplus of candidates over slots, and a linear 4-token draft into 4 slots makes it a
no-op. Re-testing it properly means proposing 12–16 and selecting 4–5.
