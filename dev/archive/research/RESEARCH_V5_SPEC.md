# RESEARCH_V5_SPEC.md — speculative decoding axis, priced in microseconds

Scope: axis = speculative decoding on Laguna S 2.1 NVFP4 MoE (256 experts, top-10, 48 layers),
Jetson AGX Thor `sm_110a`, 20 SMs, ~227 GB/s, **batch 1**. Base decode 33.0 tok/s = **30.30 ms/step**.
Drafter = poolside's shipped DFlash block-diffusion head (block 16, one forward per block).
Written against the §6 output contract of `RESEARCH_PROMPT_V5.md`. No GPU was touched.

Notation used throughout, because the prompt overloads `tau`:
* **α** = per-token acceptance probability (the 0.817 / 0.533 numbers).
* **τ(M)** = mean accepted tokens from a block of M drafted slots. Under a geometric model
  τ(M) = (1 − α^(M+1))/(1 − α); asymptote τ_max = 1/(1−α).
* `gain = τ(M) / (c + cost(M))`, c = 0.357, `cost(M) = 0.235 + 19.6·E_frac(M)`.

---

## 0. Executive summary — the five things that changed

1. **The binding constraint is misattributed.** `cost(M) = 0.235 + 19.6·E_frac(M)` charges *all*
   M-growth to the MoE expert union. Byte arithmetic says the expert union explains only **~60% of
   it at M=5**. The other **1.18 decode-steps (35.8 ms)** is the §3 latency-bound region
   (`attn_core` + `norm+cast` + `router` + `lm_head` = 23.1% of the step, ~1.3% of the bytes)
   replicating once per draft position. **The 20.1% question and the speculation axis are the same
   question.** (§2)
   The decisive evidence: every published MoE verify-cost measurement at batch 1 lands **below** its
   own Amdahl bandwidth prediction (Cohere 0.70×, Cascade/Mixtral 0.82×). **Ours lands 1.65× above.**
   Cascade's Mixtral "verify costs 2–3×" *looks* like our 3.0× but is fully explained by bandwidth
   (Mixtral is ~90% expert bytes); ours is 31.3% expert bytes and should cost 1.82×. (§2)
2. **Under the current cost model, M=1 is optimal for every α < 0.819, and M=5 needs α > 0.902.**
   Our best measured α is 0.817. So on this device DFlash's 16-token block is currently
   ~unusable — M=16 cannot pay at any α below 0.827 because τ_max = 1/(1−α) < break-even 5.766.
   The entire upside of the shipped block drafter is locked behind making `cost(M)` sublinear. (§1)
3. **The one exact mechanism that makes verify cost sublinear in M**: prune the draft suffix
   *mid-forward*, at an intermediate layer of the verify pass. Causal masking makes this
   **bit-exact** (§3.1). Arithmetic: pruning 5→2 at layer 12/48 takes `cost(5)` from 2.999 to
   **1.762 steps**, break-even τ from 3.356 to **2.119**, and at α=0.75 flips M=5 from a 0.98
   (losing) to a 1.55 gain. Published precedent: FASER (rejected-suffix predictor at intermediate
   layers, up to 95% accuracy, 19% latency cut) — but on a *dense* model at batch 32, where the
   term it attacks is far smaller than ours. **This is the highest-value item on this axis.**
4. **Two of our premises are refuted.** "No paper publishes the E_frac curve" is false (Cohere,
   EcoSpec, MoE-Spec, ST-MoE all publish it now), and "the most position-correlated MoE routing
   measured anywhere" is false — Cohere measures 20–31% below independence at K=4 on 128-expert
   top-8, versus our 21.9% below at M=5. We are *typical*, not exceptional. What **is** still
   unpublished and still ours: the correlation is absent from every published *cost model*
   (MoESD uses `N(t) = E·(1−(1−ρ)^t)`, pure independence, which overstates our M=16 union by 79%). (§4)
5. **Block Verification is worth exactly zero at greedy — proven two ways** (§5.1), and only
   1.0–2.8% at T = 0.2–1.0. Meanwhile our prose problem is now a *published* curve: WhiFlash
   measures DFlash acceptance length at **4.26 on MT-Bench and 3.04 on Alpaca versus 6.48 on GSM8K
   and 6.52 on HumanEval** — block-diffusion drafting loses on chat as a property of the drafter
   class, on someone else's hardware. Their fix is a **32–64-prompt-calibrated entropy router at
   0.10 ms/step**, and we have the arms to route between. (§7.1)

---

## 1. The cost model, and what it implies about M — arithmetic first

Recomputing from the given constants (1 decode step = 30.303 ms):

| M | experts touched (E_frac·256) | independent-routing baseline `1−(1−10/256)^M` | ratio | `cost(M)` steps | ms | break-even τ = 0.357+cost |
|---|---|---|---|---|---|---|
| 1 | 9.98 | 10.0 | 0.998 | 0.999 | 30.3 | 1.356 |
| 5 | 36.1 | 46.2 | 0.781 (**21.9% below**) | 2.999 | 90.9 | 3.356 |
| 16 | 67.6 | 120.7 | 0.560 (**44.0% below**) | 5.409 | 163.9 | 5.766 |

`cost(1) = 0.999` — the model is self-consistently calibrated to one decode step. Good.

**Optimal M as a function of α** (geometric τ; E_frac(2)=0.0645 interpolated — *unmeasured*, see §5.1):

| α | gain @M=1 | @M=2 | @M=5 | @M=16 | best |
|---|---|---|---|---|---|
| 0.533 (long prompt) | **1.130** | 0.979 | 0.624 | 0.371 | M=1 |
| 0.65 | **1.216** | 1.117 | 0.787 | 0.495 | M=1 |
| 0.75 | **1.290** | 1.246 | 0.980 | 0.688 | M=1 |
| 0.817 (short prompt) | **1.340** | 1.338 | 1.144 | 0.917 | M=1 |
| 0.90 | 1.401 | **1.460** | 1.396 | 1.445 | M=2/16 |
| 0.95 | 1.438 | 1.537 | 1.579 | **2.018** | M=16 |

Crossovers: **M=2 beats M=1 only above α = 0.819. M=5 beats M=1 only above α = 0.902.**

Three consequences that should be checked against production behaviour immediately:

* **Even on code, if α ≤ 0.82, we should be drafting M=1.** If the shipping config submits more
  than 1–2 slots at α≈0.8, we are losing ~17% versus M=1 (1.144 → 1.340). *Cheapest falsification:
  run the existing harness with M pinned to 1, 2, 3, 5 on the code corpus and compare tok/s. One
  hour, no new code.*
* **M=16 is arithmetically dead at our α.** Break-even 5.766 > τ_max = 1/(1−0.817) = 5.46. The
  block drafter's block size cannot pay at *any* achievable acceptance rate unless `cost` falls.
* **Prose is not hopeless today — it is being run at the wrong M.** At α=0.533 speculation at M=1
  still yields gain **1.130** (33.0 → 37.3 tok/s). The claim "speculation loses on prose" is true
  for large M and false for M=1. *This is the cheapest win in the document and needs no research.*

**Caveat on the geometric model.** DFlash is a block-diffusion drafter: it emits all 16 positions
in one shot, so per-position acceptance is *not* obviously geometric — a non-autoregressive drafter
typically degrades faster with position than a chain drafter (it conditions on no realized prefix).
If so, τ(M) is *below* geometric and every conclusion above is strengthened, not weakened. Measuring
the empirical τ(M) curve directly (not α) removes this assumption entirely.

---

## 2. Where the M-growth actually is — the misattribution (attacks §3 `attn_core`/`norm+cast`/`router`, 20.1% of step)

**Claim.** `cost(M) = 0.235 + 19.6·E_frac(M)` is a one-term fit that attributes all M-growth to the
expert union. Byte arithmetic says that attribution is wrong by roughly a factor of two at M=5.

**Arithmetic.** In the verify forward at M draft slots, weights are read *once* regardless of M.
Therefore, from the §3 table:
* `moe_experts` (31.3% of step, 39.9% of bytes) scales with `r(M) = E_frac(M)/E_frac(1)`:
  r(5) = 3.615, r(16) = 6.769.
* `attn_qkvo_gemm` (34.7%) and `shared+dense` (10.8%): bytes constant, FLOPs ×M, bandwidth-bound →
  approximately **flat** in M.
* `attn_core` (7.3%, 0.1% bytes), `norm+cast` (6.8%, ~0 bytes), `router` (6.0%, 1.2% bytes),
  `lm_head` (3.0%): work ×M, essentially no extra bytes → **latency-bound, replicates per position.**

Bandwidth-only prediction:

    cost_bw(M) = (1 − 0.313) + 0.313·r(M)
    cost_bw(5)  = 0.687 + 1.132 = 1.819 steps   (model says 2.999 → excess 1.180 steps = 35.8 ms)
    cost_bw(16) = 0.687 + 2.119 = 2.806 steps   (model says 5.409 → excess 2.604 steps = 78.9 ms)

Two-term fit adding the latency-bound categories at ×(M−1):

    cost₂(M) = 0.687 + 0.313·r(M) + 0.231·(M−1)
    cost₂(5)  = 0.687 + 1.132 + 0.924 = 2.743   vs 2.999   → 91% of the growth explained
    cost₂(16) = 0.687 + 2.119 + 3.465 = 6.271   vs 5.409   → overshoots, i.e. the per-position
                term saturates at large M (those kernels finally get enough rows to amortise).

**Marginal excess per added draft slot over M∈[1,5]: 0.295 decode-steps = 8.94 ms/slot.** The
latency-bound categories at ×M account for 0.231 steps = 7.0 ms/slot, i.e. **78% of it**. The
per-slot penalty is *worst at small M* — exactly the regime §1 says we must operate in.

**Which §3 category this attacks and how many microseconds:** `attn_core` + `norm+cast` + `router`
+ `lm_head` = 23.1% of the step = 7.0 ms. At M=5 that is replicated 4 extra times = **28.0 ms of the
90.9 ms verify forward, moving ~1.3% of the bytes.** If a fused/persistent decoder layer or a
batched-over-M formulation of those four categories reduced their M-scaling to ×1.5 instead of ×M,
`cost(5)` falls 2.999 → 2.24 and break-even τ falls 3.356 → 2.60.

**Exact?** Yes — this is pure kernel restructuring, no algorithmic change.
**Retraining?** No.
**Cheapest falsifying experiment (≈1 GPU-hour, do this first):** `nsys profile` a single verify
forward at M = 1, 2, 3, 5, 8, 16 and attribute the wall-clock delta to the seven §3 categories.
If `moe_experts` accounts for ≥90% of the M=5 delta, this section is refuted and the one-term cost
model is right. If it accounts for ≤65%, the highest-value work on this axis is not MoE work at all.

**Corroboration from outside — this is the strongest single argument in the document.** Take every
published MoE verify-cost measurement at batch 1 and compare it to its *own* Amdahl bandwidth
prediction `(1−f) + f·r(M)`, where `f` is the routed-expert share of the step and `r(M)` the expert
union ratio:

| model | experts / top-k | M | union ratio `r` | `f` | Amdahl predicts | **measured** | measured / Amdahl |
|---|---|---|---|---|---|---|---|
| Cohere MoE (vLLM) | 128 / 8 | 4 | 3.64 | 0.30 | 1.79× | **1.25×** | **0.70×** |
| Cohere MoE (vLLM) | 128 / 8 | 4 | 3.64 | 0.55 | 2.45× | **1.25×** | **0.51×** |
| Mixtral-8×7B (Cascade) | 8 / 2 | 7 | 3.47 | 0.90 | 3.22× | **2.3–3.0×** | **0.82×** |
| **Laguna S 2.1 (ours)** | **256 / 10** | **5** | **3.615** (measured) | **0.313** | **1.82×** | **2.999×** | **1.65×** |

**Every published MoE datapoint comes in *below* its bandwidth prediction. Ours comes in 65%
above.** Cascade's Mixtral number — which reads alarmingly like ours in absolute terms ("verification
time 2–3×", RTX 6000 Ada, batch 1) — is *fully explained by bandwidth*: Mixtral is ~90% routed-expert
bytes, so a 3.47× union legitimately costs ~3.2×. Ours is only 31.3% routed-expert bytes, so a
3.615× union should cost 1.82×. **The absolute similarity of the numbers is a coincidence; the
mechanism is different.** Sources:
[cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding],
Cascade arXiv 2506.20675 ("draft tokens collectively activate more weights, increasing data movement
and verification time by 2–3×… slowdowns up to 1.5×"; dense LLaMA-3 contrast: **5–10% overhead
regardless of K**).

That 1.65× is a 20-SM, 1.6 µs-launch-floor artefact, not an MoE property — and **it is the single
largest unexplained term on this axis.**

---

## 3. Making verify cost NOT scale with M — the black-swan item (§5)

### 3.1 Mid-forward suffix pruning — EXACT, and the top-ranked proposal

**Mechanism.** Run the verify forward over all M draft positions. At an intermediate layer `L_c`,
use a cheap estimator on the target's own hidden states to predict the accepted prefix length `n`.
Drop positions `n+1..M` from the residual stream for layers `L_c+1..48`. Treat the dropped positions
as rejected.

**Proof of exactness (one line).** Attention is causal: the layer-(L_c+1..48) hidden states of
positions `1..n` depend only on positions `≤ n`. Deleting positions `> n` therefore leaves the
logits at positions `1..n` **bit-identical**. The result is exactly equivalent to having drafted `n`
tokens instead of `M`. A wrong prediction costs accepted tokens; it can **never** change the output
distribution. Greedy stays bit-identical; sampled stays distribution-preserving. This is stronger
than the usual "lossless" claim in the literature, which is about the accept/reject rule.

**Arithmetic (prune 5 → 2 at layer `L_c`):**

    cost(M=5, prune at L_c) = 0.687 + 0.313·[f·r(5) + (1−f)·r(2)] + 0.231·[f·4 + (1−f)·1],  f = L_c/48

| `L_c` | effective MoE ratio | cost(5) steps | vs 2.999 | break-even τ |
|---|---|---|---|---|
| 8/48 | 1.981 | **1.653** | −45% | 2.010 |
| 12/48 | 2.144 | **1.762** | −41% | 2.119 |
| 16/48 | 2.308 | 1.871 | −38% | 2.228 |
| 24/48 | 2.635 | 2.089 | −30% | 2.446 |

Effect on gain at `L_c = 12`:

| α | M=5 today | M=5 pruned | M=1 today |
|---|---|---|---|
| 0.533 | 0.624 | 0.987 | 1.130 |
| 0.65 | 0.787 | **1.247** | 1.216 |
| 0.75 | 0.980 | **1.552** | 1.290 |
| 0.817 | 1.144 | **1.812** | 1.340 |

At α = 0.75 this is **+58%** over unpruned M=5 and **+20%** over the M=1 fallback. It is the only
mechanism found that unlocks the shipped 16-token block.

**The estimator.** Three options, in increasing cost:
1. **Linear probe** (d_model → 1 logistic regression) on the layer-`L_c` hidden state of each draft
   position, trained offline on collected accept/reject traces. Cost ≈ M·d_model MACs ≈ nothing.
   **This is NOT retraining the drafter** — it is a few thousand parameters fitted by logistic
   regression on CPU from logged traces. Label accordingly.
2. **Free signal, zero training:** the layer-`L_c` *router* gate distribution. Draft positions whose
   routing diverges sharply from the position-1 (known-correct) routing are likely off-trajectory.
   Costs nothing — the router already runs. Unvalidated; worth a 30-minute correlation check on
   logged traces before building anything.
3. **Re-run the DFlash head on partial hidden states.** DFlash already taps the target at layers
   [1,10,19,29,38,47]; layer 10 is almost exactly the `L_c = 12` sweet spot and the tap already
   exists in the implementation. Re-invoking the head on the *verify pass's* layer-10 states gives a
   second opinion on the block. But c = 0.357 steps is a large fraction of the 1.24 steps saved, so
   only a truncated/partial head evaluation would pay. Note as a fallback.

**Published precedent.** FASER (arXiv 2604.20503, Chen/Lu/Lin/Ustiugov, vLLM prototype): "token-wise
early-exit mechanism … after each verification layer the system uses the intermediate hidden states
… returns a mask to prune tokens predicted to belong to the rejected suffix." Reported: rejected-suffix
predictor reaches **up to 95% accuracy** at several intermediate layers, giving **up to 19% latency
reduction** (their Fig. 5a/5b, speculative length 6, **batch size 32**, dense Qwen3/Llama3, vLLM).
End-to-end FASER: up to 53% throughput / 1.92× latency versus SpecInfer/AdaSpec/Smurfs.
**Why our payoff should be much larger than their 19%:** at batch 32 on a dense model, pruning a
suffix position saves only FLOPs on an already compute-saturated GEMM. At batch 1 on a 256-expert
MoE, pruning a suffix position also shrinks the **expert union** for every remaining layer — a
bandwidth saving that simply does not exist in their setting. Our arithmetic above says 30–45%.

**Cheapest falsifying experiment:** offline only, no new kernels. From logged verify forwards, dump
layer-12 hidden states and the ground-truth accepted length, fit the logistic probe, and report the
ROC. If a probe cannot reach ≥85% accuracy at a false-prune rate below ~5%, the expected accepted
length lost exceeds the cost saved and the idea dies for ~£0.

**Risk.** Ragged/variable sequence length inside a single forward pass. On 20 SMs with 770 residual
kernels, the recompaction itself may cost more than it saves. Prune at exactly one layer, not at
every layer, and recompact once.

### 3.2 What does NOT transfer — a whole literature branch ruled out

MoE-Infinity (2401.14361), MoE-SpeQ (2511.14102), ST-MoE (2606.15453), Klotski/HOBBIT-class expert
prefetchers, and Pre-gated MoE all optimise **expert transfer over PCIe from host DRAM to GPU HBM**.
MoE-SpeQ's headline (2.34–3.3×, A100 40GB, PCIe 4.0 x16 at 32 GB/s, batch 1) is *entirely* the
hiding of PCIe latency behind draft compute; its beautiful result that a 4-bit draft model predicts
the FP16 target's top-4 experts with **90.9% accuracy** (44.1% exact-set, 46.8% correct-set-wrong-order)
buys us nothing.

**Thor has 122 GB of unified LPDDR5X. There is no PCIe hop and no offload.** Every expert is already
"resident"; the only cost is the ~227 GB/s stream, which prefetching cannot reduce. **Prefetch does
not reduce bytes; it hides latency we do not have.** Rule out the entire prefetch/caching branch
explicitly. (Corollary check that also fails: L2 reuse. One expert's weights at one layer are
~5.2 MB (2.494 GB of MoE bytes / 480 expert-instances). 32 MB L2 holds ~6 experts; we need 10 per
layer and stream 6.25 GB between visits to the same layer. L2 residency across positions or steps
is arithmetically impossible.)

### 3.3 Certified lazy expert evaluation (§5 candidate) — could not make it work

The idea: evaluate only the top-j of 10 routed experts, and accept the partial result when a
certificate bounds the omitted contribution below the decision margin. For **greedy** verification
only the argmax is needed, so the certificate is `logit_top1 − logit_top2 > bound(omitted experts)`.
**The blocker is bound propagation:** truncating at layer ℓ perturbs the residual stream, and that
perturbation must be carried through 48 − ℓ layers of attention and MoE to reach the logits.
Per-layer Lipschitz bounds compound multiplicatively and are vacuous after ~5 layers. The only
tractable version truncates in the last 1–2 layers, where the saving is 2/48 = 4% of the MoE term =
1.3% of the step. **Not worth it.** I found no paper that makes partial-computation acceptance
certificates work for transformer MoE; the nearest neighbours (MoE-Spec below, expert budgeting) all
give up exactness. Reporting this as a negative result, per §2's instruction.

### 3.4 Expert budgeting (MoE-Spec) — real gains, but approximate; rule out

MoE-Spec (arXiv 2602.16052) aggregates router probabilities across the whole draft tree,
`s_i = Σ_t g_i(h_t)`, keeps the top-B experts, and handles tokens routed outside the shortlist by
**truncation** or **substitution**. Measured on A100 80GB, **batch 1**, FP16:

| model | experts / top-k | tree | EAGLE experts | budget B | reduction | EAGLE spd | MoE-Spec spd |
|---|---|---|---|---|---|---|---|
| OLMoE-1B-7B | 64 / 8 | 31 | 39 | 32 | 18% | — | — |
| OLMoE-1B-7B | 64 / 8 | 255 | 54 | 32 | 41% | 2.1× (MATH500) | 2.5× |
| Qwen3-30B-A3B | 128 / 8 | 63 | ~80 | 64 | 20%+ | 1.9× (GSM8K) | 2.4× |
| Mixtral-8x7B | 8 / 2 | 63 | 7–8 | 4–6 | 25–50% | 1.7× (MBPP) | 2.3× |

10–30% throughput over EAGLE-3, and acceptance length barely moves (−1.4%), confirming the gain is
pure bandwidth. **But it is explicitly approximate** — "tokens whose natural routing falls outside
the shortlist are handled by truncation or substitution," with quality retention of ~95%
(Qwen3-30B), ~78% (Mixtral), worst on OLMoE. **This violates our exactness invariant and is ruled
out.** Recorded here because it is the strongest published evidence that the expert union *is* a
first-order lever at batch 1, and because one of its side-measurements is directly useful to us:
experts needed for 90% router-mass coverage, OLMoE — MBPP 8, HumanEval ~12, MATH500 ~16, GSM8K ~18,
**CNN/DailyMail 24**. **Code concentrates routing; summarisation/prose disperses it.** That means
`E_frac(M)` is itself workload-dependent and is probably *worse on prose* than the numbers we
measured — a second, independent reason prose loses. *Cheap experiment: re-measure E_frac(5) on a
prose corpus rather than a mixed one.*

### 3.5 The other bound: EVICT (adaptive verification prefix)

EVICT / "Making Every Verified Token Count" (arXiv 2605.00342), A100 80GB, **batch 1**, tree
steps=4/topk=8/draft_tokens=32, on Qwen3-30B-A3B (128 exp, top-8), Qwen3-235B-A22B, Ling-flash-2.0
(**256 experts**, top-8): selects `k* = argmax_k Ê[A(T_k)]/C(k)` — *the identical rule we already
run*. Reported: **32.5% fewer activated experts, 26.6% lower verification latency, 1.21× over
EAGLE-3, up to 2.35× over AR**, training-free, hyperparameter-free, lossless. Verifies **74.7% fewer
tokens** than EAGLE-3 on average.
**Read this as confirmation, not opportunity:** our adaptive-k already implements the winning rule
and their 26.6% is the ceiling of that mechanism. The delta available to us from EVICT is the
*estimator* (§5.2), not the objective.

---

## 4. Routing correlation — two of our claims are wrong, one survives

### 4.1 REFUTED: "no paper publishes this curve for any model"

| source | model | experts / top-k | measurement | hardware / batch |
|---|---|---|---|---|
| Cohere blog (MoESD analysis) | Cohere MoE | 128 / 8 | **20.36 unique experts across 4 tokens**; per-step overlap 0.381 / 0.329 / 0.301 / 0.299 vs independence baseline 0.118 and uniform 0.0625; "temporal correlation reduces unique experts by **20–31%** vs the independence baseline"; stable across 13 Spec-Bench categories (0.377–0.385) and 7 languages (0.378–0.386) | vLLM, **batch 1** reported separately per BS |
| MoE-Spec (2602.16052) Fig. 4 | OLMoE-1B-7B | 64 / 8 | union 39 experts @ tree 31, 54 @ tree 255 (vs 8 for one token) | A100, batch 1 |
| EcoSpec (2607.12696) Fig. 1(b) | Qwen3-235B / GPT-OSS-120B / DeepSeek-V3.1 | 128,·/8, ·/4 | expert count vs verification budget γ; active-expert counts tabulated (23.7 → 20.5 etc.) | A100, batch 1 |
| ST-MoE (2606.15453) | Qwen1.5/2.0, DeepSeek-V2/MoE | 60–160 / 4–8 | "the number of activated experts does not increase linearly as the number of neighbouring tokens grows"; consecutive-token overlap "**nearly 2×**" the K²/N random baseline; cross-layer chi-squared p < 0.01 | 40 nm ASIC sim |
| Layer-wise Routing Locality (2604.17182) | Qwen3.5-35B-A3B-FP8 | **256 / 8** | Jaccard 0.649 same-token (40× random) vs 0.175 different-token (11× random), per-layer; L0 ≈ 0.828 | GH200 |

The curve is published. Withdraw the claim.

### 4.2 REFUTED: "the most position-correlated MoE routing measured anywhere"

Ours: **21.9% below independence at M=5**, 44.0% below at M=16. Cohere's: **20–31% below
independence at K=4** on 128/top-8. MoE-Spec's OLMoE at tree 31: union 39 of 64 = 0.609 versus an
independent baseline of 1−(1−8/64)^31 = 0.984 → **38% below** at M=31. We are squarely inside the
published band, and *less* correlated than Cohere at comparable M. Withdraw the superlative.

### 4.3 SURVIVES, and is the real contribution: no published *cost model* contains the correlation

MoESD (arXiv 2505.19645), the reference cost model for MoE speculative decoding, models the union as
`N(t) = E·(1 − (1 − ρ)^t)` — **exact independence, no correlation term.** Applied to us that
predicts 46.2 experts at M=5 (we measure 36.1, **28% over**) and 120.7 at M=16 (we measure 67.6,
**79% over**). Every downstream conclusion drawn from that model about when speculation pays for MoE
at small batch is therefore *pessimistic by 28–79% on the expert term*. MoESD's headline conclusion —
"SD is ineffective at batch size 1 for MoE; under small B, `T_T(B,γ)` is notably greater than
`T_T(B,1)`" — is the conclusion our data most directly qualifies. **That is a publishable
correction and it is ours.** It is also, unfortunately, not a speedup.

### 4.4 Routing-aware draft selection — why our 0.6% was structural, not a measurement failure

EcoSpec (arXiv 2607.12696) is exactly "choose draft candidates that route to already-loaded
experts": `S(t_i) = P(t_i) / (ΔCost(t_i | B) + ε)` where `ΔCost` counts experts not yet in the
dynamic buffer `B`. It is lossless (it changes only *which* drafts are proposed, never the verify
rule). Measured on A100, **batch 1**: Qwen3-235B 1.22× → **1.36×**, GPT-OSS-120B 1.14× → **1.31×**,
DeepSeek-V3.1 1.10× → 1.15×; HBM traffic −11.3% / −8.0% / −1.0%; latency breakdown shows verify
0.832 s → 0.730 s on Qwen3 with a 0.004 s predictor. Ablation: the *marginal* cost formulation beats
a global-cost variant (1.39× vs 1.28×).

**Why we got 0.6%.** Three structural reasons, all checkable:
1. **We have a linear draft, not a tree.** EcoSpec re-ranks *sibling candidates* at each tree node.
   With one candidate per position there is nothing to re-rank and the mechanism is a no-op. Its
   gain is proportional to branching factor.
2. **DeepSeek-V3.1 — the closest published analogue to a very large, very sparse top-k MoE — got
   only 1.10× → 1.15× and −1.0% HBM traffic.** The mechanism's own numbers degrade as the model gets
   sparser and larger, i.e. toward us. Our 0.6% is consistent with their trend, not anomalous.
3. **It needs a routing predictor**: EcoSpec fine-tunes a 0.6B–1.5B model (Qwen3-0.6B,
   DeepSeek-R1-Distill-Qwen-1.5B) on routing traces. At 30.3 ms/step and 227 GB/s, a 0.6B FP8
   predictor costs ~0.6 GB = 2.6 ms = **0.087 decode steps** per invocation — affordable, but it
   must beat 0.087 steps of saving to break even, and 0.6% of the MoE term is 0.06 ms.

**The one version of this we have NOT tried, and should.** DFlash emits a block of 16 *positions*,
each with a full distribution. That is a latent tree we currently flatten. Building a 2-wide tree at
positions 1–3 and applying EcoSpec's marginal-cost score would give the mechanism something to
choose between for the first time. **But** §1 says our optimal M is 1–2, and a tree only helps if
extra breadth is cheaper than extra depth — which it is not, since breadth grows the expert union
just as depth does. **Recommend: do not pursue *routing-aware re-ranking*. Rank low.** Documented so
it is not re-proposed.
*(Note the distinction from §5.3: building a tree at all is a separate question with a much larger
prize — DDTree gets +25–60% acceptance length at T=0 with zero extra draft traffic — and is worth
one more budget sweep. What is dead is EcoSpec-style re-ranking of the tree's siblings by expert
cost, not tree breadth itself.)*

### 4.5 Routing correlation via retraining — the ceiling, for calibration only

**REQUIRES RETRAINING (of the target's router — worse than retraining the drafter). Out of scope.**
ReMoE (arXiv 2605.27081) fine-tunes the router with a temporal-locality loss to *increase* expert
reuse across adjacent tokens. On DeepSeek-V2-Lite (64 routed experts, top-6): Expert Overlap Ratio
**27.3% → 34.5%** (+26.4% relative), cache hit 31.87% → 36.87% at C=6, quality roughly neutral
(MMLU +0.09, HumanEval +2.44, GSM8K −0.76). Speedups: **Jetson Orin NX 1.77–1.99× decode**, vLLM
GPU-CPU offload +8.4%, Iluvatar 1.67×.
**Calibration point:** the Jetson number is an *offload* number (edge SSD) and does not transfer to
Thor's unified memory for the reasons in §3.2. More usefully: **even a dedicated router fine-tune
only moves expert overlap by 7.2 percentage points.** That caps how much any inference-time routing
trick can possibly buy, and it is small. Treat 4.4's 0.6% as being near the achievable ceiling, not
far below it.

---

## 5. Block verification and the accept/reject rule (priority question 1)

### 5.1 Block Verification — the exact rule, and the bad news

Sun et al., "Block Verification Accelerates Speculative Decoding", arXiv 2403.10444 (NeurIPS 2024).
Full algorithm, verbatim in substance (their Algorithm 2 and Fig. 2):

```
p_0 = 1,  τ = 0
for i = 1..γ:
    p_i = min( p_{i-1} · M_b(X_i | c, X^{i-1}) / M_s(X_i | c, X^{i-1}), 1 )
    h_γ = p_γ ;  and for i < γ:
        h_i = A_i / (A_i + 1 − p_i),   where
        A_i = Σ_{x∈X} max{ p_i · M_b(x | c, X^i) − M_s(x | c, X^i), 0 }
    if η_i ≤ h_i:  τ = i
    else:          CONTINUE          # ← the whole difference: token verification does `break`
if τ == γ: sample Y ~ M_b(· | c, X^γ)
else:      sample Y ~ p_res^block(· | c, X^τ),
           p_res^block(x) ∝ max{ p_τ · M_b(x | c, X^τ) − M_s(x | c, X^τ), 0 }
return X^τ, Y
```

The mechanism: token verification stops at the first rejection; block verification tests every
sub-block prefix and takes the **longest accepted prefix**, with `p_i` a clipped running likelihood
ratio that couples the decisions. Theorem 1: it is a valid verification algorithm (output
distribution identical to the target). Theorem 2: it is **optimal among all valid verification
algorithms**, achieving expected acceptance `Σ_ℓ Σ_{x^ℓ} min{M_s(x^ℓ), M_b(x^ℓ)}` — it optimises
`min` of products where token verification gets the product of `min`s. Cannot be worse. Same
model calls, no extra traffic.

**Measured (PaLM-2-S target, PaLM-2-XXS drafter, γ=8, 1000 prompts × 8 datasets, 3 seeds):**

| dataset | block eff. TokenV → BlockV | Δ% | wall-clock TokenV → BlockV | Δ% |
|---|---|---|---|---|
| LM1B | 3.21 → 3.49 | +8.68 | 2.17 → 2.32 | +6.85 |
| GPT Prompt | 3.41 → 3.76 | +10.06 | 2.30 → 2.48 | +8.14 |
| WebQA | 3.44 → 3.70 | +7.53 | 2.32 → 2.45 | +5.75 |
| PIQA | 3.40 → 3.68 | +8.30 | 2.29 → 2.44 | +6.52 |
| ShareGPT | 3.34 → 3.62 | +8.45 | 2.25 → 2.40 | +6.68 |
| XSum | 3.49 → 3.76 | +7.63 | 2.35 → 2.49 | +5.82 |
| GSM8K | 3.81 → 4.15 | +8.74 | 2.55 → 2.73 | +6.84 |
| WMT-DeEn | 3.19 → 3.41 | +7.00 | 2.15 → 2.27 | +5.36 |
| **average** | **3.41 → 3.70** | **+8.30** | **2.30 → 2.45** | **+6.49** |

**The gain scales with γ, and we cannot use large γ.** Their Fig. 3: γ=4 → 2.89→2.99 (**+3.46%**);
γ=6 → 3.23→3.43 (**+6.19%**); γ=8 → 3.41→3.70 (+8.50%). At γ=1 the two algorithms are identical.
**At our operating point of M = 1–2 (§1), Block Verification is worth 0% to ~2%.** The 5–8% estimate
in the brief is correct only at γ=8, which §1 shows we cannot afford.

*(Table above extracted verbatim from the v2 PDF via `pdftotext`; it reconciles with the abstract's
"5%–8% wall-clock". A parallel extraction pass produced a lower table — +1.48% to +2.83% block
efficiency — which does not reconcile with the abstract and is probably the v1 paper
("Optimal Block-Level Draft Verification"). If this matters, diff v1 against v2. Sun et al. never
state their temperature, hardware, or batch size — a real weakness of the paper. Use the open-model
replication below instead.)*

**Better evidence: Traversal Verification's chain column is a direct measurement of Block
Verification on a linear draft, on open models.** Traversal Verification (arXiv 2505.12398)
**Theorem 3.4** proves `E[N_traversal] = E[N_block] ≥ E[N_token]` — on a single chain the two are
*exactly equivalent*. Its Chain column is therefore the number we actually wanted, at **batch 1** on
a single **RTX A6000**, T = 1.0:

| task | Llama-68M → Llama2-7B | Llama3.2-1B → Llama3.1-8B |
|---|---|---|
| multi-turn | 2.05 → 2.16 (**+5.5%**) | 3.95 → 4.09 (+3.5%) |
| translation | 1.97 → 2.10 (**+6.3%**) | 3.50 → 3.53 (+0.9%) |
| summarization | 1.77 → 1.86 (+4.9%) | 3.66 → 3.76 (+2.7%) |
| QA | 2.07 → 2.19 (+5.6%) | 3.51 → 3.68 (+4.8%) |
| math (GSM8K) | 2.01 → 2.15 (**+7.0%**) | 4.61 → 4.70 (+2.0%) |
| RAG | 2.09 → 2.19 (+4.8%) | 4.05 → 4.17 (+3.0%) |
| **chain average** | **+5.7%** | **+2.8%** |

**The gain is roughly 2× larger for the weaker, more misaligned drafter** (68M→7B vs 1B→8B). A
6-layer factorized block head is a strongly misaligned drafter, so we should expect the favourable
end of this range — *at a given temperature*.

**And the temperature dependence, measured** (their Table 4, Llama3.2-1B → Llama3.1-8B, chain):

| T | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|---|
| Δ acceptance length | **+1.0%** | +1.4% | +1.5% | +2.2% | **+2.8%** |

Their own words: "as the temperature decreases … the performance gap between token-level
verification and Traversal Verification narrows." They declined to run T→0.

**THE KILLER: at temperature 0 the gain is provably exactly zero.** Proof from the rule above. With
`M_b` a point mass at the greedy token `x*`: if `X_i = x*_i` then `M_b/M_s = 1/M_s(X_i) ≥ 1`, so
`p_i = min(p_{i-1}/M_s, 1) = 1` while the draft tracks the greedy path. At the first mismatch
`M_b(X_i) = 0` ⇒ `p_i = 0`, and thereafter `A_j = Σ_x max{0 − M_s(x), 0} = 0`, so
`h_j = 0/(0 + 1 − 0) = 0` — no position after the first mismatch can ever be accepted. Identical to
token verification, token for token.

Cleaner proof, straight from the two theorems. Let `g^τ` be the greedy prefix; at T=0,
`M_b(x^τ) = 1[x^τ = g^τ]`.
* Block (Thm 1): `E[τ] = Σ_τ Σ_{x^τ} min{M_s(x^τ), M_b(x^τ)} = Σ_τ min{M_s(g^τ), 1} = Σ_τ M_s(g^τ)`.
* Token (Thm 2): `E[τ] = Σ_τ ∏_i min{M_s(g_i|g^{i−1}), 1} = Σ_τ ∏_i M_s(g_i|g^{i−1}) = Σ_τ M_s(g^τ)`.

**Identical.** The min-of-products vs product-of-mins gap collapses because every `M_b` factor is
either 1 (nothing to clip) or 0 (kills the term on both sides). At T=0, acceptance is pure prefix
string-matching against the greedy sequence and there is no target surplus mass anywhere except on
`g` — and redistributing target surplus is the *entire* mechanism of block verification.
**If Laguna S serves greedy, Block Verification is worth exactly nothing.** Their own related-work
section notes Stern et al.'s draft-and-verify was "for the greedy decoding case (zero temperature)"
and all their experiments use non-zero temperature. Two independent derivations agree; no paper
states this outright, and Traversal Verification's monotone T-table above is the strongest published
corroboration.

**Decision rule: if we serve at T ≥ 0.6, build it (free ~1.5–3%). If we serve at T = 0, do not.**

**One genuinely good structural fit, if we do serve at T > 0.** Block Verification needs the
drafter's per-position conditionals `M_s(X_i | c, X^{i−1})`. DFlash is block-diffusion: it samples
the block's positions from marginals in a single forward, so its joint is `∏_i q_i(X_i)` — and the
conditionals of an independent joint **are** the marginals. Substituting `M_s(X_i|c,X^{i−1}) := q_i(X_i)`
is therefore *exactly* correct, not an approximation, and Theorems 1 and 2 apply verbatim. Block
drafters are the natural home for block verification; nobody appears to have said this. Extra cost:
one vocab-sized reduction (`A_i`, 100 352 elements) per position per iteration ≈ 0.5 M ops at M=5 —
microseconds. **Implementation risk:** `A_i` needs the drafter's *full* vocab distribution, not a
top-k; if the DFlash head only materialises top-k logits this needs a full softmax per position.

**Rank:** high confidence, near-zero cost, but payoff **conditional on T > 0** and small at our M.
*Cheapest falsification: instrument logged draft/target distributions offline and replay both
verification rules over recorded blocks. No GPU, no kernel, exact answer in an afternoon.*

### 5.2 Traversal Verification and SpecTr — for completeness

* **Traversal Verification** (arXiv 2505.12398, NeurIPS 2025): leaf-to-root traversal, accepts the
  whole sequence from a node to the root, preserving subsequences that top-down schemes discard.
  Proven identical to the target distribution. **+2.2% to +5.7% acceptance length** across tasks and
  tree architectures. **Requires a tree** — inapplicable to a linear draft, and §4.4 argues we
  should not build a tree.
* **SpecTr** (arXiv 2310.15141, NeurIPS 2023): optimal transport with membership cost; k candidates
  per position; the LP-optimal plan is exponential in k, so they give a `(1 − 1/e)`-optimal scheme
  in near-linear time. 2.13× wall-clock, **1.37× over vanilla SD**. **Requires k parallel draft
  candidates per position**, i.e. k× the draft cost, and its verify batch is k×M — which for us
  multiplies the expert union. Rule out on cost grounds.

* **SpecTr-GBV** (arXiv 2604.25925) combines SpecTr's OT with greedy block verification over `K`
  i.i.d. draft sequences; Theorem 5.3 gives `E[τ] = Σ q(x^t)[1 − (1 − min{p/q,1})^K]`, monotone in
  `K`. Measured on DeepSeek-33B, L=12, T=0.4, K=3: **+12.4% block efficiency / +29.3% speedup over
  token-level**, +2.3% BE over SpecTr with −53.5% verification overhead. Hardware and batch not
  stated. Needs `K` independent draft sequences — for us nearly free to *sample* from the DFlash
  marginals, but the verify batch becomes `K·M`, which multiplies the expert union. Same objection
  as SpecTr.

**Head-to-head "more tokens from the SAME draft with zero extra traffic":** within the *accept/reject
rule* family, Block Verification requires no extra candidates and is *proven optimal* in that class
(Theorem 2). That question is closed. But the rule family is not the only way to get free tokens —
see §5.3, which is a different and larger mechanism.

### 5.3 Tree breadth from the marginals we already have — and why our own test failed

**DDTree** (arXiv 2604.12989) is built for exactly our drafter. It states the structural fact
explicitly: a one-pass block-diffusion drafter "provides only per-position marginals `{q_i}`, …
**without conditioning on realized tokens at earlier positions within the same block**", so the
draft joint is `Q(y_{1:L}) = ∏_i q_i(y_i)`. From that one forward pass it builds a *tree* of the
top-`B` highest-probability prefixes under `Q` via a best-first heap over the top-`K` tokens per
depth — **zero extra draft traffic** — and verifies with tree attention. Lossless. Measured on
**8× H200**, block 16, bf16, DFlash → DDTree:

| model / task | T | DFlash | DDTree |
|---|---|---|---|
| Qwen3-4B MATH-500 | 0.0 | 5.54× / τ=7.72 | **7.50× / τ=10.71** |
| Qwen3-4B HumanEval | 0.0 | 4.81× / τ=6.62 | **6.81× / τ=9.44** |
| Qwen3-8B MATH-500 | 0.0 | 5.56× / τ=7.79 | **7.52× / τ=10.73** |
| Qwen3-Coder-30B HumanEval | 0.0 | 6.09× / τ=8.02 | **8.22× / τ=10.72** |
| Qwen3-8B MATH-500 | 1.0 | 4.56× / τ=6.46 | **6.59× / τ=9.54** |

**+25–60% acceptance length, +20–40% speedup, and — crucially — it works at T = 0**, where Block
Verification is provably worth zero (§5.1). Tree breadth and block verification are orthogonal
mechanisms: breadth adds *candidates* (helps at any temperature); block verification redistributes
*probability mass* (helps only against a stochastic target). **For a greedy workload, breadth is the
only one of the two that does anything.**

**We already implemented DDTree and it did not beat linear on this device** (prior pass: "DDTree
correct but doesn't beat linear — depth-dominated"). The arithmetic in §1 explains why, and the
explanation is worth writing down because it will keep coming up:

* DDTree's node-budget sweep **peaks at 256–512 nodes on H200**. Our `E_frac` curve prices a
  16-slot *linear* block at 67.6 experts and `cost = 5.409` decode steps. A 256-node tree presents
  256 rows to every MoE layer; the expert union saturates toward 256 of 256 and `cost` heads toward
  `0.235 + 19.6 = 19.8` decode steps. **Breadth grows the expert union exactly as depth does** —
  the union does not care whether two rows are siblings or successors, only that they are distinct
  positions. On an 80 GB H200 with ~140 SMs that cost is hidden; on 20 SMs at 227 GB/s it is the
  whole budget.
* Sequoia's hardware-aware tree optimiser independently selects **64–128 nodes on-device** versus
  768 when offloading — i.e. the published on-device optimum is already 4–8× smaller than DDTree's
  H200 optimum, before any MoE correction. Ours should be smaller still, plausibly 4–16 nodes.

**So the correct reading is not "DDTree fails" but "DDTree was run at an H200-tuned budget".**
*Cheapest experiment: re-sweep the DDTree node budget over {2, 4, 8, 16, 32, 64} rather than the
paper's {16 … 1024}, with the per-node cost taken from our own `E_frac` curve rather than a flat
per-token cost.* If the optimum is at 4–8 nodes and still loses to linear, the mechanism is dead
here and we should record that as a device-specific refutation of a published result — which is
exactly the kind of return §2 of the prompt asks for.

**Also relevant:** an earlier pass found **Medusa-style typical acceptance on linear DFlash = +11%
tok/s at T=0.3**. That is *approximate* (it relaxes exactness) and therefore violates our invariant,
but it is the largest measured verification-side win we have on this device, and it is ~4× larger
than anything exact in this section. Worth restating to the owner as an explicit
exactness-vs-11% trade, rather than leaving it filed as done.

---

## 6. Adaptive M (priority question 5) — one concrete improvement

Our rule: per-position acceptance EWMA, `k* = argmax_k E[tokens|k]/cost(k)`. EVICT (§3.5) uses
literally the same objective and is state of the art, so the objective is right. **The improvement
available is the estimator.**

**D-cut** (arXiv 2607.14647), and it is the most directly comparable paper found — **its baseline is
DFlash itself** ("DFlash: block diffusion for flash speculative decoding", Chen et al. 2026,
"block-parallel diffusion drafter … generates an entire block of draft tokens in one forward pass",
block sizes 8 and 16). Mechanism:

    s_{i,k} = ∏_{t=1..k} c_{i,t}          # prefix product of the drafter's OWN confidences
    Â_i(n)  = Σ_{k=0..n} s_{i,k}          # expected marginal token advance from verifying to depth n
    ρ*      = argmax_ρ  [ Σ_{(i,k) ∈ S_{Kρ}(B)} s_{i,k} ] / C(B, ρ)

with `C(B,ρ)` a profiled hardware cost table (their Table 3: H20, B=64, D=15 — 100% budget 133.91 ms
vs 25% budget 55.16 ms). Lossless: "D-cut only drops low-utility suffixes and leaves the target
model's accept/reject logic unaltered." Results at **concurrency 32** on H20/H800 over Llama-3.1-8B,
Qwen3-4B/8B, Qwen3.5-27B, Qwen3.5-35B-A3B, Hy3-295B-A21B: **DFlash(16) 1.26× → D-cut(16) 1.65×**
average over AR; 1.68× relative under stochastic sampling; verification latency **−23.7% to −38.3%**
at B=64; MoE targets up to 3.0× over AR.

**What transfers to batch 1 and what does not.** The *cross-request global top-K packing* is a
batching mechanism and is worth nothing at batch 1 — do not expect 1.26 → 1.65. What transfers is
the **estimator**: `s_{i,k}` is a **per-instance** survival probability computed from *this block's*
drafter confidences, whereas our EWMA is a *positional average* over history. The EWMA cannot tell a
confident block from a hesitant one; the prefix product can. Concretely:

    replace   E[tokens | k] = Σ_{t≤k} ᾱ_t              (positional EWMA)
    with      E[tokens | k] = Σ_{t≤k} ∏_{u≤t} c_u      (this block's DFlash confidences)
    optionally calibrate:    ∏_{u≤t} g(c_u),  g fitted offline (Platt scaling) on logged traces

**Which §3 category / how many µs:** it does not remove work; it *chooses* less work. Given §1's
crossover table (M=1 optimal below α=0.819, M=2 above), the whole value is in making the right
per-block call near the crossover. If 30% of blocks are misclassified and the gain difference at the
crossover is ~10%, this is worth ~3% end-to-end. **Cost: ~20 lines, no new kernels, no training
beyond an offline Platt fit.** Best payoff-per-effort ratio in the document.

**Other adaptive-M work, ranked by what it adds over what we already do:**

* ⭐ **Cascade / Utility-Driven SD for MoE** (arXiv 2506.20675, **RTX 6000 Ada 48 GB, single GPU,
  batch 1**, vLLM, 5 MoE models, lossless, no training) — *the closest published setting to ours.*
  Their own words: "In single-batch serving, which is the focus of this work, the compute units are
  underutilised, and execution time of each decoding step is governed by data movement."
  Utility `U = ETR_spec / (t_iter,spec / t_iter,base)`; test 4 K-values × 4 iterations, apply best-K
  for 16, and **disable speculation entirely when U < 1**. Results: worst-case slowdown capped at
  **5% instead of 1.5×** (Mixtral/math at static K=3 is **−54%**; Cascade ≤ 5%); **+7–14% over static
  K**. **The lesson to internalise: gating buys downside protection (~50 points of avoided loss on
  adverse workloads), not upside (7–14%).** Our EWMA does not currently express "M = 0".
* ⭐ **AdaEDL** (arXiv 2410.18351, Qualcomm, A100 FP32) — the only classic adaptive-length method
  that is **both training-free and parameter-free**. Stop drafting when
  `1 − √(γ·H_draft(x)) < λ`, an entropy *lower bound* on acceptance probability (via
  `β = 1 − TVD`, Pinsker, and a linear cross-entropy approximation); γ = 0.2, λ auto-adapted toward
  a 0.9 target acceptance. Computable from the DFlash head's per-position logits **before** the
  verify forward. Llama2-7B tok/s vs best static: CNN-DM 51.50 → **56.90**, Dolly-15k 47.60 →
  **57.10**, WMT-19 32.70 → **45.20**. **The regime where it shines is ours**: with an expensive
  drafter (Pythia-6.9B/1B) it delivers **+56% vs AR where static SD gets +8%**; with TinyLlama-1B it
  converts a **−16% regression into +43%**. ⚠️ Batch size not stated. Against a *tuned* max-confidence
  baseline the margin narrows to ~2–5% — treat 2–5% as our realistic delta, not 56%.
* ⭐ **CaDDTree** (arXiv 2606.01813) — structurally identical to our `k* = argmax`, but replaces the
  exhaustive argmax with a **one-dimensional search plus greedy stopping, justified by unimodality
  of the throughput function under convex verification cost**, using the current block's marginals
  and **no training**. Claims it "matches or surpasses DDTree with **oracle** budget selection on
  nearly all tasks" (Qwen3-4B/8B, 8 benchmarks). Our `cost(M) = 0.235 + 19.6·E_frac(M)` is convex in
  M via `E_frac`, so the unimodality precondition plausibly holds. **Only the abstract was
  retrievable — get the full text.**
* **SpecDec++** (arXiv 2405.19715, 2×A100, lossless): proves the optimal policy is a **threshold
  policy on `P(∃ rejection)`**, with the threshold set by the draft/verify cost ratio; implements it
  with a trained ResNet acceptance head. **+7.2% Alpaca / +11.1% HumanEval / +9.4% GSM8K over best
  static.** ⚠️ Their measured `c₂/c₁ ≈ 3.76` — *verification dominates drafting*. **Ours is the
  inverse regime** (`c = 0.357` drafting vs `cost(1) ≈ 1.0` verify, with a steep MoE term). Direction
  transfers, magnitude does not.
* **DISCO** (arXiv 2405.04304, Intel, A100-80GB, lossless): 2-layer FFN on drafter distribution +
  entropy + position; **+10.3% over optimal static**; trained on only **~500 traces** with the
  drafter untouched — cheap enough that it does not violate our constraint.
* **BanditSpec** (arXiv 2505.15141, ICML'25, A100): training-free UCB/EXP3 over
  {drafter, γ ∈ 0..4} with a proven **stopping-time regret** bound; LLaMA3-8B EAGLE-2 98.15 →
  **UCBSpec 105.72 tok/s (+7.7%)**. ⚠️ **Batch size is 11–50, not 1.** The exploration bonus is still
  a ~10-line upgrade to our exploit-only argmax and the regret bound is the right formalism, but the
  measured number does not transfer.
* **DSDE** (arXiv 2509.01083, 8×A100, training-free, lossless) — **honest negative:** at T=0, batch-1
  latency it is *worse* than both static-optimal and AdaEDL (13.97 s vs 13.44 s vs 13.83 s). Its
  advantage is straggler robustness at batch 64, and it floors speculation length at 2 — it never
  skips. Do not adopt.
* ⚠️ **SmartSpec / TurboSpec** (arXiv 2406.14066) — **a continuous-batching goodput paper, not a
  batch-1 paper.** Its 3.2× comes from degrading to k=0 under compute contention. The goodput
  formalism transfers; the numbers do not.
* ⚠️ **PEARL** (arXiv 2408.11850, up to 3.79×) requires draft and target to run **concurrently on
  separate compute**. On 20 SMs at batch 1 they contend for the same SMs. Premise does not hold.
* **FASER** (§3.1) and **EVICT** (§3.5): already covered.

**Measurement gap worth knowing about:** *no paper anywhere publishes a skip-rate histogram* —
"X% of positions skipped → Y% latency reduction" does not exist in the literature. Cascade gates on
`U < 1` but reports only aggregate effect. If we instrument it, that number is ours.

---

## 7. Prose (priority question 6) and drafter-free arms (priority question 4)

### 7.0 The published ceiling: nothing drafter-free reaches τ = 3.4 on prose

SuffixDecoding's own appendix (arXiv 2411.04975, NeurIPS'25 Spotlight) contains the only 13-way
per-subtask mean-accepted-token table measured in one harness. **Llama-3.1-8B-Instruct, single H100,
batch 1, greedy, Spec-Bench:**

| system | coding | math | humanities | **roleplay** | **writing** | summ | transl | overall |
|---|---|---|---|---|---|---|---|---|
| EAGLE-3 *(trained)* | 5.98 | 5.75 | 5.18 | 4.66 | 5.09 | 4.77 | 2.91 | 4.65 |
| Token Recycling | 3.04 | 3.13 | 2.54 | **2.34** | **2.42** | 2.61 | 2.31 | 2.55 |
| **SuffixDecoding** | 1.98 | 2.16 | 1.45 | **1.25** | **1.44** | 1.73 | 1.71 | 1.77 |
| PLD | 1.91 | 1.96 | 1.39 | **1.21** | **1.39** | 1.82 | 1.36 | 1.61 |

Cross-checked against the independent Spec-Bench leaderboard (Vicuna-7B, **batch 1**, RTX 3090 /
A100; `#Tokens` is hardware-independent): Token Recycling 2.73, PLD 1.73, Lookahead 1.64, REST 1.63.

**Drafter-free ceiling = SAM-Decoding[Token Recycling], #MAT 3.03 aggregate** (arXiv 2411.10666,
RTX A6000, batch 1, lossless, training-free) — and prose sub-categories sit below the aggregate.
**Our break-even of 3.36 is above the entire drafter-free frontier for prose. This is not a tuning
problem and no amount of n-gram engineering fixes it.** Confirmed independently by Hybrid Verified
Decoding (arXiv 2606.01019, H100/RTX 5090/B200, batch 1, lossless), which gates a cache-draft arm on
a predicted accepted length ≥ 6 and finds the cache arm is worth selecting in only **8.8% / 9.7% /
18.6%** of decoding states, with payoff concentrated in **4.8–8.9%** of them; on prose it is *worse*
than a model drafter (MT-Bench 0.87–1.18× vs EAGLE-3, Alpaca 0.83–0.97×).

**Our SuffixDecoding arm scores 1.25 on roleplay / 1.055× speedup in its own authors' harness.**
Keeping it for repetitive/agentic traffic is right; expecting anything from it on prose is not.
SuffixDecoding Fig. 8: acceptance on WildChat stays **flat at ~0.20** as the suffix tree grows from
256 to 10 000 examples — more history does not help on open-ended chat. REST shows the same
sublinearity from the other direction: datastore 0.9 GB → M = 1.96, 27 GB → M = 2.65 (**30× the data
buys +0.7 tokens**), and code-vs-chat splits cleanly: HumanEval M = 2.65 (2.36×) vs MT-Bench
M = 1.99 (1.69×).

**Two actionable items from this literature:**
* **Token Recycling** (arXiv 2408.08696, A100-80GB, **batch 1**, lossless, training-free, **< 2 MB**
  adjacency matrix, k=8) is the best *untried* weight-free arm, and it is uniquely **task-robust**
  because it keys on logit structure rather than textual repetition — its worst case (roleplay 2.34)
  is ~2× better than SuffixDecoding's or PLD's worst case. Memory footprint comparison from the
  paper: Token Recycling 1.95 MB / 2.03×, Lookahead 105 MB / 1.27×, REST 465 MB / 1.22×, Medusa
  > 800 MB / 1.65×. On a 122 GB unified-memory device the 2 MB is free.
* **DO NOT deploy Lookahead Decoding on Thor.** Its authors state it "requires surplus FLOPs" and
  targets "powerful GPUs (e.g. A100)"; it measurably degrades **1.34× (A100) → 1.13× (RTX 3090)**
  and scores **1.00× (literally zero gain) on translation**. On 20 SMs running a memory-bound NVFP4
  MoE it is the worst possible fit. Rule out.

### 7.1 The published DFlash prose curve — and the router that fixes it

**WhiFlash** (arXiv 2606.07710, H100, single-sequence, **lossless**) routes per-token between an
autoregressive drafter and a **DFlash-style block-diffusion drafter**, and in doing so publishes the
per-dataset acceptance length of DFlash itself — the published version of exactly our problem:

| Qwen3-8B | EAGLE-3 AL | **DFlash AL** | WhiFlash AL |
|---|---|---|---|
| MT-Bench (chat) | 5.17 | **4.26** | 5.68 |
| Alpaca (chat) | 5.26 | **3.04** | 5.34 |
| GSM8K | 6.21 | **6.48** | 7.35 |
| HumanEval | 5.65 | **6.52** | 7.42 |

**Block-diffusion drafting wins on math/code and loses badly on open-ended chat — as an independent
measurement on other hardware.** Our prose problem is a property of the drafter class, not of our
integration. WhiFlash's fix is +37.3% throughput over DFlash alone on MT-Bench, via two routers:
* **entropy router** — route away from the diffusion drafter when target entropy `H_t > τ`;
  calibrated on **32–64 prompts per category**, costs **0.10 ms/step**;
* **neural router** — 2-layer GELU MLP, **2.1 M params**, predicts the acceptance-length *difference*
  ΔAL, costs **0.30 ms/step = 0.78% of round latency**.

**We do not have an AR drafter to route to — but we have SuffixDecoding, and we have "no
speculation".** A three-way entropy router over {DFlash block, SuffixDecoding, plain decode} is the
direct adaptation, costs ~0.10 ms/step (0.003 decode steps), and needs 32–64 calibration prompts and
no training. Compare **MetaSD** (arXiv 2604.05417, A5000/A6000/A100, single-batch, lossless,
training-free): a UCB bandit over 5 drafters with reward = block divergence, which "matches or beats
the best *single* specialised drafter on every task", biggest wins on mixed workloads.

Also worth stealing: **SuffixDecoding's entropy metric** (their Appendix A.2.3) as a zero-GPU-cost
*request-class* router — build a suffix tree from ~100 sample outputs and compute the weighted
average per-node child-access entropy. It separates workloads sharply: AgenticSQL-Extract 0.086 →
11.8 accepted tokens; Magicoder 2.95; **WildChat 3.43 → ~1.3**. Strictly better than a binary
code-vs-prose heuristic, and it runs on the CPU at request admission.

### 7.2 The four things that are true about prose

Only the first is a research finding:

1. **We are probably running the wrong M on prose.** §1: at α = 0.533 the gain at M=1 is **1.130**
   and at M=5 is 0.624. "Speculation loses on prose" is a statement about M ≥ 2, not about
   speculation. Verify this before anything else.
2. **The prose penalty is partly a routing penalty, and it compounds.** MoE-Spec's per-benchmark
   coverage (§3.4): experts needed for 90% router mass — MBPP 8, HumanEval ~12, MATH500 ~16,
   GSM8K ~18, **CNN/DailyMail 24** on a 64-expert model. Prose disperses routing, so `E_frac(M)` is
   plausibly *larger* on prose than the mixed-corpus numbers we fitted, making `cost(M)` larger
   exactly where α is smallest. **We have never measured E_frac on a prose-only corpus.** If prose
   E_frac(5) is, say, 0.17 instead of 0.141, `cost(5)` is 3.57 and break-even τ is 3.93 — the losses
   are worse than we think. *Cheap: re-run the existing E_frac instrumentation on a prose corpus.*
3. **Mid-forward pruning (§3.1) is the only mechanism that changes the break-even rather than the
   acceptance rate.** Everything else in the literature raises τ; a fixed shipped drafter puts a
   hard ceiling on that. Lowering break-even from 3.356 to ~2.1 is the only lever that does not
   route through the drafter. Note it still does not rescue M=5 at α=0.533 (0.987 < 1.130 at M=1) —
   it rescues the *middle* band α ∈ [0.62, 0.85], which is where most real prose likely sits.
4. **The literature's consensus is bluntly against us and should be taken at face value.**
   Practitioner guidance converges on "below ~0.5 acceptance rate, speculation adds latency — turn
   it off," and "open-ended creative generation and multilingual outputs yield low acceptance."
   One dissent worth chasing: "Acceptance Dynamics Across Cognitive Domains in Speculative Decoding"
   (arXiv 2604.14682, TinyLlama-1.1B drafter / Llama-2-7B-Chat-GPTQ target, 99 768 speculative nodes
   from 200 prompts, 4 domains) reports that **"only the chat domain consistently yields an expected
   accepted length exceeding 1.0 token per step"** and that **task type is a stronger predictor of
   acceptance than tree depth** — i.e. *chat beat code* in their setup, the opposite of our
   measurement. Their target is a 7B GPTQ model and their drafter is 1.1B, so the divergence is
   plausibly a drafter-family artefact rather than a domain fact. **I could not retrieve their
   per-domain numeric table** (PDF text extraction failed); flagging as the one piece of unresolved
   contradicting evidence on this axis.

---

## 8. Ranked actions — (payoff × uncertainty) / cost

| # | action | §3 category attacked | payoff (arithmetic) | uncertainty | cost | exact? | retrain? |
|---|---|---|---|---|---|---|---|
| **1** | **Sweep M ∈ {1,2,3,5,8,16} on prose and code with the existing harness.** §1 predicts M=1 wins below α=0.819 | — (measurement) | up to **+17%** on code, **+13%** on prose if we are currently at M≥5 | high | ~1 h, no code | yes | no |
| **2** | **`nsys` the verify forward at M=1,2,3,5,8,16; attribute the delta across the seven §3 categories.** Settles whether the M-growth is expert bytes or the latency-bound 20.1% | `attn_core`+`norm+cast`+`router`+`lm_head` (23.1%) | reframes the whole axis; if ≥35% is latency-bound, break-even τ 3.36 → 2.60 | **very high** | ~1 GPU-h | yes | no |
| **3** | **Measure `E_frac` at M = 2, 3, 4, and separately on a prose-only corpus.** Every conclusion near the M=1/M=2 crossover depends on E_frac(2), which is interpolated | `moe_experts` | gain at M=2 ranges **1.19 → 1.49** at α=0.817 depending on E_frac(2) ∈ [0.055, 0.077] | high | ~2 h instrumentation | yes | no |
| **4** | **Mid-forward suffix pruning at layer ~12/48, exact (§3.1).** Offline probe fit first | `moe_experts` + the 23.1% | `cost(5)` 2.999 → **1.762** (−41%); break-even τ 3.356 → **2.119**; α=0.75 gain 0.98 → **1.55** | medium-high | offline probe ≈ 1 day; kernel work 1–2 weeks | **yes, provably** | **no** (linear probe ≠ drafter retrain) |
| **5** | **Per-instance survival scores + entropy gate replacing the positional EWMA (§6): D-cut prefix product × AdaEDL's `1−√(0.2·H)`, plus a Cascade-style `U<1` skip** | — (policy) | ~2–5% upside; **and ~50 points of downside protection** — Cascade turns a −54% adverse case into ≤5% | medium | **~30 lines**, no training | yes | no |
| **6** | **Three-way entropy router over {DFlash block, SuffixDecoding, no speculation}** (§7.1, WhiFlash recipe) | — (policy) | WhiFlash: **+37.3% over DFlash alone on MT-Bench**; router costs 0.10 ms/step = 0.003 steps | medium-high (they route to an AR drafter; we route to a weaker arm) | 32–64 calibration prompts, no training | yes | no |
| **7** | **Re-sweep the DDTree node budget over {2,4,8,16,32,64}** with per-node cost from our own `E_frac` curve (§5.3) — the prior test used the paper's H200-tuned 256–512 | `moe_experts` | DDTree publishes **+25–60% τ and +38% at T=0**; Sequoia's on-device optimum is 64–128 nodes, ours should be far smaller | high | ~1 day (code exists) | yes | no |
| **8** | **Try Token Recycling as the weight-free arm** (§7.0): <2 MB, lossless, training-free, **task-robust** — worst case 2.34 on roleplay vs SuffixDecoding's 1.25 | — | replaces the arm that scores 1.055× on prose in its authors' own harness | medium | small | yes | no |
| **9** | **Block Verification (§5.1)** — replay offline first to price it | — (accept rule) | **0% at greedy (proven twice)**; +1.0% at T=0.2, +1.5% at T=0.6, +2.8% at T=1.0 | low (numbers are solid) | ~50 lines + offline replay | yes (Thm 1) | no |
| 10 | Publish the E_frac correction to MoESD's `N(t) = E(1−(1−ρ)^t)` independence model (§4.3) | — | 0 tok/s; genuine research contribution | — | writing only | — | — |
| — | ~~EcoSpec routing-aware draft selection~~ (§4.4) | `moe_experts` | 0.6% measured; structurally capped — linear draft has no siblings to re-rank | — | — | yes | needs a 0.6B predictor |
| — | ~~MoE-Spec expert budgeting~~ (§3.4) | `moe_experts` | 18–41% expert reduction, +10–30% throughput | — | — | **NO — approximate** | no |
| — | ~~Expert prefetch/caching family~~ (§3.2) | `moe_experts` | **zero** — optimises a PCIe hop Thor does not have | — | — | — | — |
| — | ~~Certified lazy expert evaluation~~ (§3.3) | `moe_experts` | ≤1.3% — Lipschitz bounds vacuous after ~5 of 48 layers | — | — | yes | no |
| — | ~~SpecTr / SpecTr-GBV / Traversal Verification~~ (§5.2) | — | need k× candidates or a tree; the verify batch becomes k·M, multiplying the expert union | — | — | yes | no |
| — | ~~Lookahead / Jacobi decoding~~ (§7.0) | — | **negative** — authors state it "requires surplus FLOPs"; degrades 1.34× (A100) → 1.13× (RTX 3090); **1.00× on translation**. Worst possible fit for 20 SMs + memory-bound MoE | — | — | yes | no |
| — | ~~PLD / REST / SuffixDecoding as the prose fix~~ (§7.0) | — | drafter-free ceiling is **3.03 aggregate** (SAM-Decoding[TR]); prose sub-scores 1.2–2.4. Below our 3.36 break-even by construction | — | — | yes | no |
| — | ~~SmartSpec / PEARL / BanditSpec magnitudes~~ (§6) | — | premises fail at batch 1: continuous-batching goodput, concurrent draft/target compute, and batch 11–50 respectively. Formalisms transfer; numbers do not | — | — | — | — |
| — | ~~DSDE~~ (§6) | — | **published negative at T=0, batch 1**: worse than both static-optimal and AdaEDL | — | — | yes | no |
| — | ~~ReMoE router fine-tune~~ (§4.5) | `moe_experts` | ceiling datum only: expert overlap 27.3% → 34.5% | — | — | — | **YES — retrains the target's router** |

---

## 9. §4 premises — status after adversarial attempt

* **§4.5 "`E_frac`'s position correlation cannot be exploited at inference time."**
  **SURVIVES, with a sharpened reason.** It *has* been exploited (EcoSpec, lossless, +0.14 speedup on
  Qwen3-235B at batch 1), so the premise is false in general. But it survives *for us*: the exploit
  requires sibling candidates to re-rank, we run a linear draft, its own published gains collapse
  toward the sparse/large end (DeepSeek-V3.1: 1.10× → 1.15×, −1.0% HBM traffic), and ReMoE shows a
  dedicated router fine-tune moves expert overlap by only 7.2 points. Our 0.6% is near the ceiling,
  not far below it.
* **§4.6 "Speculation cannot be made profitable on prose without retraining the drafter."**
  **REFUTED, three ways.** (a) At M=1 it is already profitable on prose today (gain 1.130 at
  α=0.533) — the losses are an M-selection artefact. (b) Mid-forward suffix pruning is exact,
  needs no drafter retraining, and lowers break-even τ from 3.356 to ~2.12, which flips the whole
  α ∈ [0.62, 0.85] band. (c) WhiFlash gets **+37.3% over DFlash alone on MT-Bench** with a
  32–64-prompt-calibrated entropy router and no retraining of anything.
  **But the premise's spirit survives in one place:** no *drafter-free* arm can rescue prose. The
  published ceiling is 3.03 aggregate, below our 3.36 break-even, and more n-gram history provably
  does not help (SuffixDecoding: WildChat acceptance flat at 0.20 from 256 to 10 000 examples).
* **§5 "something making verify cost not scale with M."**
  **FOUND: §3.1**, with an exactness proof from causal masking and published precedent (FASER, 19%
  on dense at batch 32). Predicted 30–45% here because it shrinks the expert union too, which the
  dense precedent does not.
* **"Our E_frac curve is unpublished and uniquely correlated."** **REFUTED** (§4.1, §4.2). What
  survives is narrower and still valuable: no published *cost model* carries the correlation (§4.3).
* **Could not refute:** that `cost(M)` genuinely is ~3.0 at M=5 (it is a measurement; §2 attacks its
  *attribution*, not its value). Also could not resolve the contradiction in §7.2 item 4 — one paper
  (arXiv 2604.14682) reports chat outperforming code on acceptance length, the opposite of our
  measurement and of WhiFlash's DFlash numbers, and I could not extract its table. Its target is a
  7B GPTQ model with a 1.1B drafter, so it is probably a drafter-family artefact, but it is
  unresolved.

**Two further novelty claims that survived checking:**
* **Nobody models MoE expert-activation cost in an adaptive-draft-length or verification-rule
  paper.** Every adaptive-M cost model found (SpecDec++, DISCO, AdaEDL, BanditSpec, SmartSpec, D-cut,
  CaDDTree) is a flat draft-vs-verify latency ratio. Our `19.6·E_frac(M)` term has no counterpart.
  Corollary: every published magnitude was measured in the regime `verify ≫ draft`
  (SpecDec++ measures `c₂/c₁ ≈ 3.76`); ours is the inverse. **Directions transfer, magnitudes do not.**
* **Nobody has applied Block Verification to a factorised block-diffusion drafter.** DDTree states
  the factorisation `Q = ∏ q_i` explicitly and does not cite Sun et al. at all. §5.1 shows the
  substitution is exact rather than approximate, and the `min(∏) ≥ ∏(min)` gap should be *largest*
  for a factorised drafter (frequent `q−b` sign flips across positions, plus the weak-drafter effect
  in Traversal Verification's Table 3). Gated on T > 0.

**Provenance warning.** Two independent extraction passes disagreed on Sun et al.'s Table 1
(mine, from `pdftotext` on the v2 PDF: +8.30% block efficiency / +6.49% wall-clock average; the
other: +1.48–2.83%). Mine reconciles with the abstract's "5%–8%"; the other does not. Separately,
early PDF-only extractions in the parallel pass produced at least one *fabricated* range that the
HTML later contradicted. **Re-verify any number in this document before building on it**, and prefer
the open-model replications (Traversal Verification's A6000 chain columns, the Spec-Bench
leaderboard) over the PaLM-2 numbers, which state neither hardware, batch size, nor temperature.

**Hardware caveat that applies to the whole document.** Nothing here was measured on Jetson /
Orin / Thor-class silicon. Every number is A100 / H100 / H200 / H20 / B200 / A6000 / A800 /
RTX 3090 / RTX 6000 Ada. The two nearest analogues are Cascade (RTX 6000 Ada, single GPU, batch 1,
explicitly data-movement-bound) and Lookahead's measured A100 → RTX 3090 degradation. **On 20 SMs,
expect every tree/breadth optimum to be far smaller and every FLOP-surplus technique to fail.**

---

## 10. Sources

Block Verification — https://arxiv.org/abs/2403.10444 ·
Traversal Verification — https://arxiv.org/abs/2505.12398 ·
SpecTr — https://arxiv.org/abs/2310.15141 ·
EcoSpec — https://arxiv.org/abs/2607.12696 ·
EVICT — https://arxiv.org/html/2605.00342 ·
MoE-Spec — https://arxiv.org/html/2602.16052 ·
MoESD — https://arxiv.org/abs/2505.19645 ·
Cascade (utility-driven SD for MoE) — https://arxiv.org/abs/2506.20675 ·
D-cut — https://arxiv.org/html/2607.14647v1 ·
FASER — https://arxiv.org/pdf/2604.20503 ·
MoE-SpeQ — https://arxiv.org/abs/2511.14102 ·
ST-MoE prefetching — https://arxiv.org/abs/2606.15453 ·
ReMoE — https://arxiv.org/html/2605.27081 ·
MoE-Infinity — https://arxiv.org/abs/2401.14361 ·
Layer-wise MoE routing locality — https://arxiv.org/html/2604.17182v1 ·
Acceptance dynamics across cognitive domains — https://arxiv.org/abs/2604.14682 ·
AdaEDL — https://arxiv.org/pdf/2410.18351 ·
HiSpec — https://arxiv.org/abs/2510.01336 ·
Cohere, "Why MoE Models Get More From Speculative Decoding" — https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding

**Block-diffusion drafters (our drafter class):**
DFlash — https://arxiv.org/html/2602.06036v2 · poolside/Laguna-S-2.1-DFlash on HF ·
DDTree — https://arxiv.org/html/2604.12989 ·
CaDDTree — https://arxiv.org/abs/2606.01813 ·
BlockPilot — https://arxiv.org/abs/2606.31315 ·
WhiFlash (AR ↔ diffusion routing) — https://arxiv.org/abs/2606.07710 ·
Speculative Diffusions via Block Verification — https://arxiv.org/abs/2606.13426 *(continuous
diffusion, not discrete — title is a false friend; 6.3% headline)*

**Verification rules / multi-draft:**
SpecTr-GBV — https://arxiv.org/html/2604.25925 ·
Sequoia — https://arxiv.org/abs/2402.12374 ·
EAGLE-2 — https://arxiv.org/html/2406.16858v1

**Adaptive draft length:**
SpecDec++ — https://arxiv.org/html/2405.19715v2 ·
DISCO — https://arxiv.org/abs/2405.04304 ·
BanditSpec — https://arxiv.org/abs/2505.15141 ·
SVIP — https://arxiv.org/html/2411.18462v2 ·
SmartSpec/TurboSpec — https://arxiv.org/abs/2406.14066 ·
PEARL — https://arxiv.org/abs/2408.11850 ·
DSDE — https://arxiv.org/abs/2509.01083 ·
MetaSD — https://arxiv.org/abs/2604.05417 ·
Nightjar — https://arxiv.org/abs/2512.22420

**Drafter-free / weight-free:**
Spec-Bench leaderboard — https://github.com/hemingkx/Spec-Bench/blob/main/Leaderboard.md ·
SuffixDecoding — https://arxiv.org/abs/2411.04975 ·
Token Recycling — https://arxiv.org/abs/2408.08696 ·
SAM-Decoding — https://arxiv.org/abs/2411.10666 ·
REST — https://arxiv.org/abs/2311.08252 ·
PLD+ — https://arxiv.org/abs/2412.01447 ·
Lookahead — https://arxiv.org/abs/2402.02057 *(ruled out)* ·
Hierarchy Drafting — https://arxiv.org/abs/2502.05609 ·
Hybrid Verified Decoding — https://arxiv.org/abs/2606.01019 ·
Graft — https://arxiv.org/abs/2605.20104 ·
CLLM — https://arxiv.org/abs/2403.00835 *(fine-tunes the target)* ·
ELMoE-3D — https://arxiv.org/abs/2604.14626
