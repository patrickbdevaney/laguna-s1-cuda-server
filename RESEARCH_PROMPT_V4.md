# RESEARCH_PROMPT_V4.md — constructed from first principles

## Why this prompt is shaped the way it is

Three research passes are now on record. Reviewing which findings actually changed the code:

| what changed our behaviour | what did not |
|---|---|
| **Measurements taken on our device** — the 1.6 µs/kernel floor, the 16-ALU-ops-per-word budget, `GENERIC_COMPRESSION_SUPPORTED = 0` | Rankings of known techniques |
| **Corrections to a premise we held** — the τ-vs-M framing, the missing per-layer `input_layernorm`, `ncu` not actually being blocked | Speedup numbers from other hardware |
| **Negative results with arithmetic** — EcoSpec's 0.6 % union reduction on our architecture class | "This might help" |

So the objective is not literature coverage. It is **premise refutation and on-device
measurement**. A prompt that asks "what techniques exist for X" gets a survey; a prompt that
says "here is the identity that governs our performance, attack a term" gets an answer.

The structure below follows from that.

---

## 1. Invariants — things that cannot be traded

State these first so no proposal wastes itself on them.

* **The model artifact is fixed.** `poolside/Laguna-S-2.1-NVFP4` and its *shipped* DFlash head.
  Anything requiring the drafter to be retrained forks poolside's artifact and needs training
  hardware we do not have. Rule those out unless the payoff is large enough to justify the fork,
  and say so explicitly rather than proposing them as if they were free.
* **Exactness.** Greedy output must be bit-identical to autoregressive decode; sampled output
  must be distribution-preserving. This is not a preference — the speculative verify path is
  built on it.
* **One device, one stream, batch 1.** 20 SMs, ~254 GB/s achievable, 122 GB unified.
* **No system-wide changes** without the owner's decision (root, clocks, module options).

## 2. The identities — attack a TERM, not a topic

Everything we can do is a change to one of these. A proposal that does not name the term it
attacks is not actionable.

```
Base decode:      t_tok  =  B_tok / BW_eff  +  N_k · c_k
                    B_tok  = 6.251 GB      bytes that must cross DRAM per token
                    BW_eff = 206 GB/s      achieved, against 254 achievable
                    N_k    = 1665          kernels per step
                    c_k    = 1.6-1.7 µs    measured marginal cost per kernel

Speculation:      gain   =  τ(M) / (c + cost(M))
                    cost(M) = 0.235 + 19.6 · E_frac(M)     fitted to our own measurements
                    c       = 0.357         drafter cost, in units of one decode step
                    E_frac  = 0.039 / 0.141 / 0.264  at M = 1 / 5 / 16
```

For each of the seven terms — `B_tok`, `BW_eff`, `N_k`, `c_k`, `τ`, `c`, `E_frac` — ask:

1. What is the **theoretical floor** for this term given the invariants?
2. What is the **largest published reduction** of this term on any hardware, and what did it cost?
3. Is there a reduction that is **exact** and does not require retraining?
4. What **cheap decisive experiment** would falsify it?

`E_frac` deserves special attention: ours is 22 % below independent routing at M=5 and 44 %
below at M=16, which is the most position-correlated MoE routing measured anywhere, and no
paper publishes this curve for any model. Anything exploiting routing correlation is unexplored
territory where our numbers are the only data.

## 3. Premises to attack

These are the beliefs currently steering the work. **The most valuable answer this prompt can
return is a measurement that falsifies one of them.** Take them adversarially.

1. `B_tok` cannot go below ~5.4 GB without quality loss, because the routed experts are already
   NVFP4 and q/k/v must stay at FP8.
2. `BW_eff` of 206 against 254 achievable is ~81 %, and the published frontier for a full decode
   step is 82 %, so there is almost nothing left in the kernels.
3. `N_k · c_k` = 2.8 ms is irreducible without a megakernel, and the megakernel is blocked
   because `grid.sync` costs 35 % of the step.
4. τ cannot be raised without retraining the drafter, which forks the shipped artifact.
5. `c` = 0.357 is fixed by the drafter's 2.23 GB, and quantizing the drafter loses more τ than
   it saves bytes (measured: 13.33 → 11.14).
6. `E_frac` is a property of the model's routing and cannot be influenced at inference time.
7. Acceptance degrades with prompt length because the draft's 512-position window covers only
   the tail of a long prompt.

For each: what evidence would show it false, and does that evidence exist?

## 4. Black-swan register — what would a step change even look like?

Percentage work is well covered. Ask specifically for mechanisms with a *different shape*:

* Something that makes `B_tok` **sublinear in the model** — reading less than the active
  parameter set per token, exactly. (Certified lazy expert evaluation is our candidate; is
  there anything else, and has anyone made partial-computation acceptance certificates work?)
* Something that makes the **verify cost not scale with M** — the whole speculative gain is
  capped by `cost(M)` growing at 0.50 decode-steps per slot.
* Something that **removes the drafter's byte cost entirely** while keeping model-quality
  proposals (our suffix arm does this but only on repetitive traffic).
* Something that exploits **unified memory as an advantage** rather than a constraint — every
  technique we have imported treats it as a limitation.
* Anything from outside LLM serving — database query planning, speculative execution in CPUs,
  branch prediction, approximate query processing with exactness certificates — whose structure
  maps onto "predict, verify cheaply, fall back exactly".

## 5. Output contract

For every claim: the sentence, the source with hardware, the measured number, **which term of §2
it attacks and by how much with the arithmetic shown**, whether it is exact, whether it needs
retraining, and the cheapest experiment that would falsify it. Rank by
(payoff × uncertainty) / cost.

**Run experiments on the device where the question is answerable there.** Three of the most
valuable results in the last pass were microbenchmarks, not citations. If a question can be
settled in an hour of CUDA, settle it.

State plainly which of §3's premises you could not refute — a premise that survives an
adversarial attempt is worth as much as one that falls.
