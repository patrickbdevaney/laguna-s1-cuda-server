# RESEARCH_FINDINGS_V4.md — "can anything be exact AND read fewer bytes at batch 1?"

A parallel session ran the exact-byte-reduction question to ground across four families. The
answer is a clean, well-supported **no** for our current weight format, and the negatives are
worth more than another ranking would have been — one of them closes the item I had ranked
highest on (payoff × uncertainty)/cost.

## The bottom line

**Nothing is both exact and reads fewer bytes at batch 1 for a group-scaled 4-bit MoE.**

| family | verdict |
|---|---|
| lossless weight compression | only bit-exact family; **NVFP4 compresses ~5 %** |
| certified / bounded expert skipping | **does not exist** — zero papers |
| exact-zero activation sparsity | requires retraining into a different model |
| speculative decoding | exact-in-distribution but reads **1.10–1.54× MORE** bytes/token |

## A. Lossless compression — dead for NVFP4, and the distinction matters

The ZipNN extension (arXiv 2508.19263) measures FP4 streams as **statistically uniform**:
NVFP4 DeepSeek-R1 achieves only **5 % overall compression**, and the only part that compresses
at all is the FP8 scaler stream (~10 % of bytes). An earlier estimate of 1.2–1.4× extrapolated
from AWQ was too optimistic — AWQ's 128-element groups are a poor proxy for NVFP4's 16-element
blocks, which flatten the code histogram further.

Decompressor throughput as a secondary check: DFloat11 measures 122–262 GB/s (A5000/A100),
nvCOMP ANS 18–22 GB/s, against A100 HBM at 1555–2039 GB/s — a 6–17× loss. NVIDIA's dedicated
Blackwell Decompression Engine is "up to 600 GB/s" against B200's ~8 TB/s. End to end, DFloat11's
own repo states batch-1 inference is **~2× slower** than BF16; NeuZip measures 33–49 % slower at
bit-identical perplexity.

**The distinction to preserve, and the reason this section is not simply "compression is dead":**
Thor has roughly **19× more compute per byte of bandwidth than an A100**, which is precisely the
gradient along which fused-decompression wins increase (ZipServ wins on RTX 4090/L40S and "may
not always match" on A100/H800). DFloat11's 122–262 GB/s is *the same order as Thor's entire
273 GB/s bus. On this machine the usual "the decompressor is slower than HBM" objection is much
weaker than in any published evaluation.* It is the **FP4 uniformity result** that closes this
off, not the bandwidth argument.

⇒ **Do not generalise "lossless compression is dead" to a future BF16 or FP8 component.** For a
BF16 draft head it would still be live. It is dead for NVFP4 specifically.

## B. Certified expert skipping — no prior art whatsoever

I had ranked this the highest (payoff × uncertainty)/cost item on the grounds that nobody had
tried it. That is now confirmed rigorously — **zero papers across ~20 distinct queries** give a
provable bound on the emitted token from skipping low-gate experts.

The nearest miss is **WINA** (2505.19427), which claims "optimal approximation error bounds with
theoretical guarantees" — but it bounds the **layer-output perturbation norm, not the emitted
token**, and it gates activation *neurons*, not experts. Everything else is heuristic with no
certificate: AdapMoE (−25 % experts, 1.35×), SERE (2.0×), MoDES, HOBBIT (9.93× but swaps in
low-precision experts, so lossy), SliceMoE, and the dynamic-k family (Ada-K, DTop-p, BEAM) which
all need training.

Worth recording separately so it is never re-proposed: **MoE-Infinity, Fiddler, Pre-gated MoE,
ExpertFlow, SwapMoE, MoE-Lightning, Klotski, eMoE and DALI all reduce CPU↔GPU PCIe traffic.**
That is a non-problem here — our active set is already resident in one unified pool. They are
I/O-hiding, not byte-reduction.

**Status: still unexplored, now with the confirmation that being first carries all the risk of
being first.** No one has shown the bound stays tight through 48 sequential routing layers.

## C. Exact-zero activation sparsity — the exact variants are all retrains

* **Approximate** (a skip is not exact — you must read the weights to learn they are
  sub-threshold, unless gating by input channel): TEAL 40–50 % model-wide, but the reasoning
  cost is real — Llama-3-8B 6-task average including 5-shot MMLU and GSM8K falls **68.07 → 66.21
  at 40 %** (−1.86 pts); decode 1.53×/1.8× on A6000, 1.25–1.45× on A100, and **no MoE
  evaluation at all**. CATS: 50 % sparsity for only ~15 % wall-clock. R-Sparse: explicitly a
  rank approximation.
* **Genuinely exact zeros, but of a *different, retrained* model**: Q-Sparse (top-K with a
  straight-through estimator at training time); ProSparse (89.32 % sparsity, up to 4.52×);
  TurboSparse/dReLU (Mixtral-47B → 4.3 B active, 2–5× decode — architecturally the most
  relevant to us since it exploits sparsity *inside* MoE experts, and still a retrain).

This is consistent with what we measured on gemma: activations are magnitude-dense.

## D. Speculative decoding reads MORE bytes, not fewer — and this validates our cost model

EcoSpec's Table 3 gives unique experts activated per MoE layer per verification step (E) and
mean accepted tokens (α) at batch 1, γ=4, T=0. E/α is experts read **per accepted token**,
directly comparable to autoregressive top-k:

| model | AR | best SD | ratio |
|---|---:|---:|---:|
| Qwen3-235B-A22B | 8.0 | 8.84 | **1.10× more** |
| GPT-OSS-120B | 4.0 | 5.70 | **1.42× more** |
| DeepSeek-V3.1 | 8.0 | 11.02 | **1.38× more** |

**Speculative decoding on an MoE always reads 1.10–1.54× more bytes per emitted token.** It is
distribution-exact and it helps wall-clock, but purely by amortising *more* bytes across one
fatter, better-utilised pass. ELMoE-3D states the mechanism outright: "verification must load
experts even for rejected tokens, severely limiting its benefit in MoE especially at low batch
sizes."

This is independent confirmation of our own fitted cost model — `cost(M) = 0.235 + 19.6·E_frac(M)`,
with a verify at M=5 costing ~3× a decode step — arrived at from entirely different data. It also
explains why our break-even sits at τ ≈ 3 while dense-target papers report break-even near 1.

Supporting: "Lossless but Not Free" (2607.17283) audits distribution equivalence properly
(χ² 162.5, dof 200, p = 0.976, plus exact greedy agreement) and finds a best case of 1.61× at
K=6 — but **3 of 5 configurations decelerate**. That is the same shape as our own result that
speculation loses on prose.

## E. Two guarantees that read stronger than they are

* **CALM** (2207.07061) — the only rigorous guarantee in the early-exit family — is
  `E[D(Y_early, Y_full)] ≤ δ`, valid at least 95 % of the time, **over a calibration set**. A
  statistical bound on an *expected dissimilarity*: not per-token, not exact. "Provably
  maintaining high performance" in the abstract will be over-read by anyone who does not check.
* **ZipMoE** (2601.21198) says "**semantically** lossless", not bit-exact, and its "provable
  performance guarantee" is a constant-factor-of-optimum result on **cache scheduling**, not on
  output error. Its 72.77 % latency reduction is entirely against a 1–5 GB/s Jetson Orin
  UMA/PCIe transport — irrelevant to a resident model.

## What this changes for us

1. **Compression is closed for NVFP4** — and explicitly *not* closed for BF16/FP8 components.
2. **Certified expert skipping remains the only unexplored exact byte-reduction mechanism**, and
   we now know for certain that no one has done it. That raises its value and its risk together.
3. **Stop looking for speculation to reduce bytes.** It never does on an MoE. Its entire value is
   wall-clock amortisation, which is exactly how our controller already prices it.
4. The remaining live byte lever is unchanged: **routed experts 4.5 → 3.0 bpw**, where the
   hardware question is already settled (Thor absorbs 16 ALU ops/word free) and only accuracy is
   open.
