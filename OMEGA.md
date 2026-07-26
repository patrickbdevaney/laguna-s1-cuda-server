# OMEGA.md — have we reached the global maximum?

**No. And it would be dishonest to say otherwise.**

This file exists because the question was asked directly, and the answer that would be most
flattering — "yes, this is the ceiling" — is not the one the evidence supports. What follows is
the honest position: what is genuinely finished, what is genuinely open with numbers attached,
and what would have to be true for this to be the end.

## Where the work actually stands

| | start | now |
|---|---:|---:|
| base decode | 3.66 tok/s | **33.0** |
| `B_tok` | 10.044 GB | **6.251 GB** |
| served, prose | — | 33.6 tok/s |
| served, code | — | 40.1 tok/s |
| served, repetitive | — | 49.7 tok/s |

For scale: poolside's own measurement on the closest hardware analogue (DGX Spark GB10) is
13–14 tok/s unspeculated and 22–24 with DFlash. Published full-step batch-1 bandwidth efficiency
tops out at 82 % (FlashFormer, H100); we are at ~81–85 %.

All eight gates are green: H1, R1, A1, L1, B1, B1c, D1, S1. Greedy output is bit-exact against
the oracle at every stage.

## What is genuinely closed

Each of these was closed by a measurement, not an argument:

* **Wave quantization / persistent CTAs** — bandwidth is flat from 240 to 20 000 blocks on 20 SMs.
* **Compressible memory** — `GENERIC_COMPRESSION_SUPPORTED = 0`; `cuMemCreate` fails.
* **EMC clock pinning** — 29.46 pinned vs 29.37 default. Null.
* **MAXN** — EMC max is identical in both modes; we draw 62.7 W of a 120 W cap.
* **L2 persistence** — 24 MB against 6.25 GB of read-once traffic; ≤0.33 % ceiling.
* **GEMV access-pattern tuning** — `__ldcs`/`__ldg`/`ld.global.nc` within noise; `uint4` optimal.
* **Zero-copy weights** — 160 vs 227 GB/s.
* **Attention below FP8** — five 2026 production recipes exclude it; damage scales down with
  active params and we are at 8.5 B.
* **Tree / multi-candidate verify** — the fitted cost model requires τ ≥ 8.65 at M=16 to reach
  1.5×; published τ at our target scale is 2.4–3.4.
* **Drafter quantization** — measured τ 13.33 → 11.14; `c` is in the denominator alongside a
  much larger `cost(M)`, so halving it cannot pay for a 16 % τ loss.
* **Megakernel with `grid.sync`** — ~35 % of token time in the barrier, independently reproduced
  by others at the same figure.
* **Acceptance collapsing with generated position** — measured flat-to-rising. It is prompt
  length that matters.

## What is open, with numbers

These are not speculative. Each has an estimate and a stated uncertainty.

| lever | estimate | why it is not done |
|---|---:|---|
| Routed experts → 3.0 bpw trellis | **+15.2 %** | Hardware question **settled** (Thor absorbs 16 ALU ops/word free). Accuracy on AIME/GPQA-D is the open part — needs an offline quantize + eval. |
| Kernel fusion 1665 → ~400 with SM-level deps | **+8.0 %** (measured ceiling) | Weeks of work. The `grid.sync` failure does not condemn it; the published fix is sentinel/fine-grained sync. |
| 256 threads/block on kernels with ≲240 blocks | up to +17 % on those | Measured on a synthetic; not yet applied to the MoE shape. |
| Draft context selection for long prompts | unquantified, motivated by our own τ data | Needs a larger tap ring and a relevance ranking. **Exact and inference-side.** |
| Certified lazy expert evaluation | up to +14 %, high variance | Nobody has tried it. Cheap offline probe not yet run. |
| EcoSpec re-test at candidate surplus | ~+3 % | Our first test may have measured a no-op: the rule needs more candidates than slots. |

Two more that are **out of scope rather than unattractive**: every published fix for
long-context acceptance is a drafter retrain, and DFlash is the *shipped* head for this
checkpoint — retraining forks poolside's artifact and needs hardware we do not have. Test-time
adaptation measured 1.66× on a functionally identical model. That is the largest single number
in any of the research, and it is behind a door we cannot open here.

## What would have to be true for this to be the omega point

1. `B_tok` at its floor — it is not; the routed experts are 40 % of the budget at 4.5 bpw and
   3.0 bpw is compute-free on this hardware.
2. `BW_eff` at the achievable ceiling — it is not; ~9 of the 19-point gap is kernel-count
   structure with a measured 8 % recoverable.
3. τ at the drafter's ceiling — unknown, and the one intervention that would answer it is out
   of scope here.
4. No unexplored mechanism — false; certified lazy expert evaluation has never been attempted
   by anyone, and our routing-correlation curve is data nobody else has.

None of the four holds.

## The honest summary

This is a **strong local maximum, reached by exhausting every lever we could measure**, and it
is ahead of the published state of the art on comparable hardware by roughly 2.4× on the
unspeculated path. It is not a global maximum, and the largest remaining item is gated by
training hardware rather than by ideas.

The correct thing to record is the frontier, not a victory. `RESEARCH_PROMPT_V4.md` is written
to attack it, and it is deliberately built to invite refutation of the seven premises this
position rests on — because in three research passes, every finding that changed the code was
either a device measurement or a corrected premise, and never a survey.
