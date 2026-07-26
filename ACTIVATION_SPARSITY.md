# ACTIVATION_SPARSITY.md — measured on our model, and the +25% lever is dead

## What was claimed

arXiv:2605.08575 (Park, Kim, Gu, Stoica, Cheung — citation verified real) reports that pre-trained
MoE experts are internally dormant: **up to 90% of a selected expert's intermediate neurons can be
zeroed without significant accuracy loss**, 2.5× MoE-layer and **1.2× end-to-end**. Priced against
our 12.24 ms MoE term that was **+24.9%** — the largest single lever on the whole list, and larger
than the trellis subsystem we rejected.

The research flagged one risk that multiplied the entire item: **every published measurement is on
models with `moe_intermediate` ≥ 1408**, except OLMoE at 1024, which reached only ~50%. **Ours is
1024.** So it was measured here rather than inherited.

## Method

`LG_DUMP_MOEH` dumps `h = SiLU(gate)·up` for the 10 selected experts of layer 24 during a real
decode run — 353 tokens, **3530 expert-token rows** of 1024 neurons. Then three curves of mean
relative L2 error of `h` against sparsity, exactly as the research specified: unstructured top-`s`,
block-32 structured (our layout's native unit), and block-32 after permuting neurons by mean |h|
(the proposed bit-exact fix).

## Result

| sparsity | (1) unstructured | (2) block-32 | (3) block-32, sorted |
|---:|---:|---:|---:|
| 0.25 | 0.0297 | 0.3245 | 0.3183 |
| 0.50 | 0.1145 | 0.5229 | 0.5140 |
| 0.75 | 0.2873 | 0.7196 | 0.7095 |
| **0.87** | **0.4374** | 0.8314 | 0.8220 |
| 0.90 | 0.4910 | 0.8635 | 0.8548 |

**At the published operating point of 87%, zeroing costs 43.7% relative error unstructured and 83%
block-structured.** For a model that "turns 1 ulp into a different token" because the 256-way
sigmoid router puts top-10 on a knife edge, neither is survivable.

## Why it fails here, and why no fix applies

The proposed rescue was an offline neuron permutation — bit-exact, free at inference — to
concentrate chronically-dormant neurons into the same 32-blocks. **It buys 0.9 percentage points**
(0.8314 → 0.8220), and the concentration statistics say why:

    neurons active in >90% of rows:   0 of 1024
    neurons active in <10% of rows:   4 of 1024
    per-neuron activity rate:         min 0.092, max 0.182

If dormancy were *structural*, activity rates would be bimodal — some neurons near 1.0, many near
0.0. Instead **every neuron is in the top-13% for between 9% and 18% of tokens.** Participation is
almost perfectly uniform across the 1024 neurons.

**Our experts are not internally dormant. They are dynamically sparse** — a different subset of
neurons matters for each token. There is nothing chronic to permute, so:

* the offline permutation has nothing to concentrate (0.9 pp, confirmed);
* block-32 structured sparsity is dead regardless of layout (0.83 error);
* and our repack — 32 output rows lane-interleaved at 16-byte granularity — means **unstructured
  sparsity saves zero bytes anyway**, since skipping one arbitrary neuron still fetches its
  512-byte group.

Dynamic per-token sparsity would need the active set *predicted before the weights are read*, which
is the Deja Vu route: a trained predictor per layer, i.e. new training, and a predictor whose
mistakes are silent. That is a different project with a different risk profile.

## Verdict

**T2 is rejected.** The lever is worth ~+12% at s=0.50 where error is already 11.5%, and nothing
near the +24.9% priced. `moe_intermediate = 1024` was the right thing to worry about: our model
behaves like OLMoE (the one published model at this width), not like the wide-FFN models the
headline number came from.

Cost of finding out: one instrumented decode run and twenty minutes of numpy, against a two-week
kernel project. This is the same discipline that killed the trellis — measure the thing your payoff
multiplies before building the thing that spends it.

## Note on the instrument

The first dump run **dumped core** after 540 rows: it called `cudaStreamSynchronize` and
`cudaMemcpy` inside a CUDA-graph-captured region, which is illegal. Fixed by asking
`cudaStreamIsCapturing` rather than assuming. The partial data it produced gave 47.4% sparsity at
99% energy retention — consistent with the full result, but the metric is not the paper's, so it
was not reported as a finding until the proper curves existed.
