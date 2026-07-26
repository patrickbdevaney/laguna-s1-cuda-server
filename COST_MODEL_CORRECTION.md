# COST_MODEL_CORRECTION.md — the fitted verify-cost model was physically impossible

## The falsification

`RESEARCH_FINDINGS_V3.md` fitted, from two measured points:

```
cost(M) = 0.235 + 19.6 · E_frac(M)      in units of one decode step (6.251 GB)
```

Two independent research passes falsified it the same way. At `E_frac = 1.0` — every expert
touched — that model asserts `19.8 × 6.251 = 123.8 GB` of traffic. **The entire routed-expert
set is 63.9 GB** (`3 × 1024 × 3072 × 4.5 bits × 256 experts × 47 layers`). The model claims
**1.92× more bytes than the weights contain.**

It was fitted from `cost(1) = 1.00` and `cost(5) = 3.00`, both of which are **times**, and then
read as though it were a **byte** model. That conflation is the whole error.

## The corrected byte model

```
cost_bytes(M) = 0.601 + 10.22 · E_frac(M)
   0.601 = dense share  (6.251 − 2.495 = 3.756 GB)
  10.22  = full expert sweep / decode step  (63.9 / 6.251)
```

It reproduces the measurement it was not fitted to: at `E_frac = 0.039` it predicts 2.491 GB of
expert traffic against **2.495 GB measured**.

| M | E_frac | cost by BYTES | cost by TIME (measured) | gap |
|---:|---:|---:|---:|---:|
| 1 | 0.039 | 1.00 | 1.00 | — |
| 5 | 0.141 | **2.04** | **3.00** | **0.96** |
| 16 | 0.264 | 3.30 | 5.41 | 2.11 |

## The consequence, and it is the useful part

**The gap is not bytes. It is our own M>1 kernel inefficiency**, and it is the thing that sets
our speculative break-even.

```
break-even τ = c + cost(M)          c = drafter cost ≈ 0.36–0.41
   at M=5, by bytes:  2.04 + 0.36 = 2.40
   at M=5, measured:  3.00 + 0.36 = 3.36
```

We measure τ = 2.26 on prose and 3.67 on code. **At the byte-model break-even of 2.40, prose is
borderline and code is a clear win. At our actual 3.36, prose is a loss** — which is exactly what
the server measures, and exactly why the controller declines to speculate there.

So ~0.96 decode-steps per verify at M=5 is being spent on nothing but the M>1 path being less
efficient per byte than the M=1 path. That is consistent with what we already measured directly:
the dense GEMM at M=6 costs 2.25× the M=1 time for **identical** weight traffic
(`OPTIMIZATION_LOG.md` #24/#25, where shared-memory staging and N-blocking both failed to fix it).

**This relocates the single highest-value speculative lever from the drafter to our own kernels.**
Closing the M>1 efficiency gap would move break-even from τ 3.36 to 2.40 and make speculation
profitable on prose — the workload where we currently get nothing. It needs no retraining, no
new model artifact, and no change to the acceptance rule.

## What else the pass established

* **P6 (certified expert skipping) is closed, by a stronger argument than mine.** I worried about
  Lipschitz blow-up over 48 layers. The real killer is discrete: skipping experts perturbs the
  residual stream, which perturbs every downstream router's **argmax**, so a sound certificate
  must certify **48 × 246 = 11 808 top-10-of-256 gate comparisons per token**. With 256
  fine-grained experts the boundary gap is small while the perturbation is 1–3 % of gate mass.
  Falsifiable in an hour of logging. (Supporting: dot-product attention is not Lipschitz on
  unbounded domains; measured bound decay is ~3×/layer for CROWN and ~570×/layer for IBP, and
  GPT-2 XL's forward-error bound is called "practically vacuous" by its own authors.)
* **Our EcoSpec 0.6 % result was not a bad test** — it numerically reproduces EcoSpec's own
  DeepSeek-V3.1 number, and DeepSeek is the one model they drafted with a *chain* rather than a
  tree. So the published 256-expert experiment is itself the low-surplus configuration. Genuinely
  confounded upstream; resolvable offline from logged M=16 trees.
* **Block Verification** (arXiv 2403.10444, NeurIPS 2024) — verifies a block *jointly* rather than
  token-by-token, provably optimal in expected tokens per iteration, lossless, **zero extra
  compute or memory traffic**, 5–8 % wall-clock. It is for *linear* drafts, so it survives our
  tree-verify rejection, and it raises τ at fixed M, which is a 1:1 multiplier at zero cost. This
  is a change to the accept/reject rule only.
* **CSV-Decode** (arXiv 2511.21702) — certified `lm_head` skipping via ~2000 spherical k-means
  clusters and a one-line Cauchy–Schwarz bound: 18.4 % of the vocabulary read, 98.2 %
  certification. `lm_head` is 24 % of the drafter's cost and 4.9 % of every target step, so ≈
  +4.3 % at M=5, exact. Caveat: their headline mixes an exact mode with a non-exact ε-mode — take
  the math, not the speedup.
* **Two clean negatives.** Freivalds' algorithm gives exactly 1.00× at batch 1 (matrix–*vector* is
  already O(n²) and the check re-reads the same weights). And unbiased-estimator exactness is
  impossible in general: `min(1, p̂/q)` attains 1 and so is not simulable by any Bernoulli factory
  (Keane–O'Brien). The one live crack is Barker's rule `p/(p+q)`, simulable precisely because it
  never attains 0 or 1.

## The lesson worth keeping

A model fitted to **times** and then interpreted as **bytes** will assert traffic the hardware
does not contain. **Every fitted coefficient should be checked against a physical bound before it
is used to rank anything.** This one survived two research passes and steered several conclusions
before a third caught it — including my claim that tree verify needs τ ≥ 8.65, which was computed
from the inflated curve.
