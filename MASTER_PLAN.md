# MASTER_PLAN.md — the exhaustive path to maximum decode speed

Synthesis of four parallel research streams (attention, latency, speculation, MoE/quantization)
against a profile of the config we actually ship. Ordered by measured or best-estimated payoff per
unit of work. **Everything here is priced against `B_tok` = 6.251 GB and a 30.3 ms step at
33.0 tok/s.**

## The single most important structural fact

**The two big terms fail in opposite ways, and that decides everything.**

| term | absolute | state | consequence |
|---|---:|---|---|
| MoE experts | 12.24 ms (40.4%) | **bandwidth-bound at 89.8% of ceiling**, NVFP4 unpack ~3 ALU ops/weight nowhere near saturated | any lever removing expert **bytes** converts ~1:1 |
| trellis decoder | — | **ALU-bound at 91% of its own ceiling** | fewer bytes buy nothing → subsystem rejected |

This is why `TRELLIS_VERDICT.md` does not generalise: it was a statement about a *decoder*, not
about the expert term. The expert term still has ~1:1 byte→speed conversion available — just not
via a more expensive decoder.

## Ranked plan

### Tier 1 — measured, or gated by a cheap measurement

**T1. NVFP4 attention weights (q + o).** `attn_qkvo_gemm` is the largest term (~39.7% of step,
44.9% of bytes) and nothing on the old list attacked it.
*Status: **capability MEASURED*** — all five projections at NVFP4 leave greedy output bit-identical,
8/8, via loader simulation. `q_proj` and `o_proj` are byte-identical at 1.2457 GB each.
q+o saves 1.090 GB → **+21%**; all five saves 1.226 GB → **+24.4%**. k/v are only 4.83% of `B_tok`
and the most quantization-sensitive, so **q+o is the target**.
*Remaining risk:* whether the saving **converts**. Must be settled with a NOMEM twin before the
kernel is written — exactly the discipline that caught the trellis. Precedent is good: production's
NVFP4 MoE GEMV runs at 82% of ceiling, so NVFP4 dequant stays bandwidth-bound.
*Work:* W4A16 dense GEMV + loader quantization. **No new container, no encoder.**

**T2. Intra-expert activation sparsity.** 87–91% of a *selected* expert's intermediate neurons are
dormant in pre-trained MoEs (arXiv:2605.08575, verified real; abstract confirms "up to 90% sparsity
without significant accuracy loss", 2.5× MoE layer, **1.2× end-to-end**). Priced here at
**+24.9%** at s=0.87; the paper's own end-to-end is +20%.
*Two repo-specific facts that decide it:*
- **Our repack layout defeats unstructured sparsity.** 32 output rows are lane-interleaved at
  16-byte granularity, so skipping one arbitrary neuron saves **zero bytes**. At 87% unstructured
  sparsity, P(a 32-block has ≥1 active neuron) = **98.8%** — block sparsity is worthless unless
  active neurons *cluster*.
- **The fix is bit-exact and free**: permute the 1024 rows of `gate`/`up` and columns of `down` by
  the same permutation, sorted by mean |SiLU(gate)·up| over a calibration set. Exact identity on the
  dense computation, folded into the existing repack, zero inference cost.
*This is lossy* (95% score retention ≠ free), but it sits in the same gate class as the expert-bit
reduction already contemplated — and `EXPERT_BITS_EVAL.md` proved the gate must be the **capability
sweep**, not τ (which *rose* when quality fell) and not golden-continuation (a coin flip on a
modified model).
*Open risk that multiplies the whole item:* every published measurement is on `moe_intermediate`
≥ 1408, except OLMoE at 1024 which measured only **~50%**. **Ours is 1024.** If our sparsity is 50%
not 87%, this is +12.4%, not +24.9%.
→ **T2a (doing now): measure our own sparsity curve.** Dump `h = SiLU(gate)·up` for the 10 selected
experts over ~500 decode tokens; produce error-vs-sparsity curves for (1) unstructured, (2)
block-32, (3) block-32 after sorting. Cheap, no kernel work, and it gates a +25% item.

**T3. Speculation retune.** Research claims M=1 is optimal for α < 0.819 and that M≥5 loses ~17% on
code / ~45% on prose. **I do not accept this yet**: its foundational term came from my own 20.1%
figure, which the latency stream independently corrected to ~10.5%, so the misattribution argument
is roughly halved. `tools/k_sweep.sh` bypasses the cost model and measures served tok/s at each
`SPEC_K` cap (including `SPEC_K=0`) on code and prose. **Queued, sequenced after the capability
sweep.** Possibly a free win from a config change.

### Tier 2 — real but smaller

**T4. Mid-forward suffix pruning** (speculation): drop suffix positions at layer ~12/48, provably
**bit-exact** via causal masking. `cost(5)` 2.999 → 1.762, break-even τ 3.356 → **2.119**. Flips
M=5 from loss to win at α=0.75. Also shrinks the expert union.
**T5. Split-K router GEMV** with last-block fused top-k: **+2.1%**, exact. The router is an
*occupancy* problem (one warp per output row → 27% warp occupancy), not a top-k problem; our warp
top-k is already at the published floor.
**T6. NSP sweep** on `attn_core` (`by_len` 64→24): up to **+3.3%**, one line, needs a greedy re-gate.

### Tier 3 — rejected, with the arithmetic

| lever | verdict |
|---|---|
| 3 bpw trellis experts | **+5.3–7%** for a Viterbi CUDA encoder, new container, new kernel, 8 re-gates. Rejected; T1/T2 are 4× the payoff for less work. |
| RMSNorm→GEMV fusion | **≤1.2%**. `norm+cast`'s 6.8% was a profiling artifact — `add_rms_cast` moves 48 KB (0.21 µs at 227 GB/s) but is charged 30.5 µs/call. Dead on arithmetic. |
| Megakernel | ceiling **~1.8%**; CUDA graphs already capture 82% of removable launch cost. `grid.sync` was never the real blocker. |
| Block Verification | **exactly zero at greedy**, proven two ways. The 5–8% held only at γ=8, which the cost model says we cannot afford. |
| Expert pruning / merging | **zero bytes per token** — we already read only the top-10. |
| Expert caching / prefetch | structurally dead: unified memory, no PCIe transfer to overlap. |
| AWQ/SmoothQuant on the NVFP4 artifact | **information-theoretically dead**: for diagonal `D`, `xD⁻¹·(Q(W)D)ᵀ = x·Q(W)ᵀ` exactly. Re-rounding strictly *adds* error. Only reachable by requantizing from `Laguna-S-2.1-FP8`. |
| Certified lazy experts | Lipschitz bounds go vacuous after ~5 of 48 layers. |

## Corrections this synthesis forced on our own record

1. `WIKI.md`: "no paper publishes the `E_frac` curve" — **false** (Cohere, EcoSpec, MoE-Spec, ST-MoE).
2. `WIKI.md`: "the most position-correlated MoE routing measured anywhere" — **false** (Cohere 20–31% below independence vs our 21.9%). What survives: no published *cost model* incorporates the correlation.
3. `WIKI.md` §6: "≥10 smem codebook lookups absorbed with zero bandwidth loss" — **refuted** at 3 bits (271 vs 395 Gweight/s).
4. `HARDWARE.md:38`: "tcgen05 present" — **false**. TMA, cluster barriers and hardware FP8 conversion *are* present and unused.
5. `REPRICED.md` v1: "20.1% of the step moves 1.3% of bytes" — **mine, and wrong**; ~10.5%.
6. The softplus-gate objection to quantizing `o_proj` — **mine, and a category error**. For weight-only quantization the gate scales signal and error identically, so SNR is invariant.
