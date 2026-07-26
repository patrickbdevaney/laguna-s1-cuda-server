# RESEARCH_PROMPT_V5.md — the unexplored edges, priced in microseconds

## Meta-critique of V1–V4, and why this prompt is shaped differently

Four research passes are on record. Reviewing what actually changed the code:

| changed our behaviour | did not |
|---|---|
| Measurements taken **on this device** (1.6 µs/kernel floor, 16-ALU-ops/word budget, `GENERIC_COMPRESSION_SUPPORTED = 0`) | Rankings of known techniques |
| **Corrections to a premise we held** (τ-vs-M framing, missing per-layer `input_layernorm`, `ncu` not actually blocked) | Speedup numbers from other hardware |
| **Negative results with arithmetic** (EcoSpec's 0.6% union reduction on our architecture class) | "This might help" |

V4's flaw, visible only in hindsight: **it aimed every question at the terms we already knew
about.** It asked how to reduce `B_tok`, raise `BW_eff`, cut `N_k·c_k`. A profile of the shipping
config then showed that **20.1% of the decode step moves ~1.3% of the bytes** — latency-bound work
that the roofline framing renders *invisible*, and that four research passes never asked about
because the framing had no term for it. The prompt inherited the blind spot of the model it was
built on.

So V5 is organised by **where the microseconds actually are**, measured, not by the identity we
happen to have written down. Each section names its measured share of the step and asks what can
attack it. A proposal that does not name a category below, with its microsecond budget, is not
actionable.

---

## 1. Invariants — do not spend effort here

* **The model artifact is fixed**: `poolside/Laguna-S-2.1-NVFP4` and its *shipped* DFlash head.
  Anything requiring the drafter to be retrained forks poolside's artifact and needs training
  hardware we do not have. Rule these out explicitly rather than proposing them as if free.
* **Exactness**: greedy output must be bit-identical to autoregressive decode; sampled output must
  be distribution-preserving. The speculative verify path is built on this.
* **One device, one stream, batch 1.** Jetson AGX Thor, `sm_110a`, **20 SMs**, 122 GB unified
  LPDDR5X, ~227 GB/s measured streaming ceiling, 32 MB L2, 228 KB smem/SM, 1536 threads/SM.
* **No system-wide changes** (root, clocks, module options) without the owner's decision.

## 2. Settled on this device — do not re-derive, but DO attack the premise

Each of these is a measurement we made. Bring a contradicting measurement and it is the most
valuable thing you can return; bring a citation and it is not.

1. **Trellis decode is compute-BOUND here, not free.** QTIP 3INST (two weights per hash, ~3 ALU
   ops/weight) reaches 395 Gweight/s against a 434 Gweight/s ALU ceiling — 91% of it. Production
   NVFP4 MoE runs at 332 Gweight/s. So 3 bpw experts are worth **+5.3% end-to-end, not +15.2%**.
2. **A shared-memory codebook is NOT free at 3 bits.** 271 vs 395 Gweight/s, a 31% loss at ~10.7
   lookups per 32-bit word. This *refutes* our own earlier "≥10 lookups absorbed with zero
   bandwidth loss".
3. **`E_frac`**: expert-union fraction 0.039 / 0.141 / 0.264 at M = 1/5/16 — **22% below
   independent routing at M=5 and 44% below at M=16**. The most position-correlated MoE routing
   measured anywhere; no paper publishes this curve for any model.
4. **τ degrades with prompt length** (0.817 → 0.533 from 67 to 15 267 tokens) and is
   flat-to-rising with generated position. The literature conflicts; we measured it.
5. **τ is not a quality proxy** — it *rose* when the target was degraded, because a worse target
   is easier to imitate.
6. Production decode: 33.0 tok/s, `B_tok` 6.251 GB, 206/227 GB/s = **91% of ceiling** — but that
   only applies to the 79% of the step that moves bytes.

## 3. Where the microseconds are — attack a category, with arithmetic

Measured shares of the decode step. **Name one, or the proposal is not actionable.**

| category | % step | % bytes | what it is | the obvious question |
|---|---:|---:|---|---|
| `attn_qkvo_gemm` | **34.7** | **44.9** | q/k/v/o/gate projections, FP8 W8A16, 96 launches | biggest term; `o_proj` is 42.7% of it and has never been evaluated below FP8 |
| `moe_experts` | **31.3** | 39.9 | NVFP4 top-10-of-256 GEMV | 3 bpw is worth only +5.3%; what else? |
| `shared+dense` | 10.8 | 8.9 | shared expert + layer-0 dense, FP8 | |
| `attn_core` | 7.3 | **0.1** | flash-decoding over FP8 KV | **latency-bound: 7.3% of time for 0.1% of bytes** |
| `norm+cast` | 6.8 | ~0 | RMSNorm + dtype casts, 96 launches | **pure overhead** |
| `router` | 6.0 | 1.2 | sigmoid router, top-10 of 256, 47 launches | **5× more time than bytes** |
| `lm_head` | 3.0 | 4.9 | FP8, vocab 100 352 | |

**The 20.1% question.** `attn_core` + `norm+cast` + `router` = 20.1% of the step for ~1.3% of the
bytes. This is the least-examined region of the whole project. At 20 SMs with a 1.6 µs launch
floor and ~770 residual kernels, what is the actual achievable floor? Specifically:

* Persistent-kernel / megakernel designs where the *whole decoder layer* is one launch. Our
  recorded blocker is that `grid.sync` costs 35% of the step — is that avoidable with
  producer-consumer queues, `cudaTriggerProgrammaticLaunchCompletion`, programmatic dependent
  launch, or cluster-wide barriers on Blackwell? What has anyone measured on a *small* GPU
  (≤32 SMs) where occupancy per launch is not the limiter?
* Is a 256-way sigmoid router + top-10 selection really 6% of a decode step, or is ours slow?
  What is the fastest published top-k-of-256 for k=10 at batch 1?
* RMSNorm + cast fusion into the *consumer* GEMV's prologue — who has done this and what did it
  buy?

## 4. Premises to attack — a refutation is worth more than a survey

1. `o_proj` cannot go below FP8 because the unbounded per-head softplus gate immediately precedes
   it and multiplies quantization error. *(Untested. This is the single highest-value question in
   the document: it is worth +9.2%.)*
2. q/k/v cannot go below FP8 at all.
3. The KV cache is already FP8 and irrelevant at short context (0.1% of bytes) — but what about
   at 16k–128k, where `attn_core` grows and τ collapses to 0.533?
4. `N_k · c_k` is irreducible without a megakernel, and the megakernel is blocked by `grid.sync`.
5. `E_frac`'s position correlation is a property of routing that cannot be exploited at inference
   time. *(We have the only measurements of this curve. If it can be exploited, nobody has.)*
6. Speculation cannot be made profitable on prose without retraining the drafter.

## 5. Black-swan register — mechanisms with a different SHAPE

Percentage work is well covered. Ask specifically for:

* Something making `B_tok` **sublinear in the model** — reading less than the active parameter set
  per token, *exactly*. (Certified lazy expert evaluation is our candidate. Has anyone made
  partial-computation acceptance certificates work?)
* Something making **verify cost not scale with M** — the whole speculative gain is capped by
  `cost(M)` growing at ~0.50 decode-steps per slot.
* Something exploiting **unified memory as an advantage** rather than a constraint. Every
  technique we have imported treats it as a limitation. Zero-copy, `cudaMemAdvise`, CPU/GPU
  co-execution of different layers, DMA engines idle during decode?
* **Blackwell `sm_110a` specifics**: we know `tcgen05` is absent and `GENERIC_COMPRESSION_SUPPORTED
  = 0`. What *is* present that we are not using — cluster launch, distributed shared memory, TMA,
  `cp.async.bulk`, programmatic dependent launch, FP4/FP6 conversion instructions?
* Anything from **outside LLM serving** whose structure maps onto "predict, verify cheaply, fall
  back exactly": database query planning, CPU speculative execution, branch prediction,
  approximate query processing with exactness certificates.

## 6. Output contract

For every claim: the sentence, the source **with hardware**, the measured number, **which category
in §3 it attacks and how many microseconds it removes, with the arithmetic shown**, whether it is
exact, whether it needs retraining, and the cheapest experiment that would falsify it. Rank by
(payoff × uncertainty) / cost.

**Run experiments on the device where the question is answerable there.** Three of the most
valuable results in the last pass were microbenchmarks, not citations. If a question can be
settled in an hour of CUDA, settle it.

State plainly which of §4's premises you could not refute. A premise that survives an adversarial
attempt is worth as much as one that falls.
