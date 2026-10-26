# TRELLIS_VERDICT.md — should the 3 bpw expert subsystem be built?

Two inputs decide it: **does it go faster** and **does the model survive**. This file settles the
first from end-to-end measurement. The second is the capability sweep.

## The speed answer: +6% to +11%, not the +15.2% that was projected

Everything below is measured on this device, not scaled from another part.

| quantity | value | how obtained |
|---|---:|---|
| production `moe_experts` | **13.345 ms/step** | `LG_PROF=1 build/bench_decode`, 47 launches/step |
| expert weights per token | 4.435 G | 10 experts × 3 mats × 1024 × 3072 × 47 layers |
| **production NVFP4 MoE rate** | **332 Gweight/s** | 4.435 G ÷ 13.345 ms |
| corrected for LG_PROF's 47 syncs | ~344 Gweight/s | 47 × ~10 µs ≈ 0.47 ms of overhead removed |
| **trellis_B measured** | **395 Gweight/s** | `bench_tr3`, real GEMV, coalesced repack |
| trellis_B ALU ceiling | 434 Gweight/s | NOMEM twin: same arithmetic, zero weight traffic |

Decode step at 33.0 tok/s is 30.3 ms, of which the MoE is 13.3 ms — **44%**.

    swap in trellis_B at 395 Gw/s   MoE 13.3 -> 11.2 ms   step 28.2 ms   35.5 tok/s   +7.5%
    at the ALU ceiling  434 Gw/s    MoE 13.3 -> 10.2 ms   step 27.2 ms   36.8 tok/s   +11.4%
    against the sync-corrected 344  MoE 12.9 -> 11.2 ms                               +5.8%

**Realistically ~+7%.** The original +15.2% assumed the expert term stays bandwidth-bound and
converts the 1.406× byte saving in full. It does not: trellis_B runs at **91% of its own ALU
ceiling**, so it is compute-bound, and reading fewer bytes cannot buy what the ALU will not give.

## Why the first benchmark said the opposite, twice

Worth recording, because both errors were mine and both produced confident wrong answers.

**v1 said trellis was 18% of ceiling.** It measured its own harness: every weight did a separate
scalar `x[...]` global load, so activation traffic was 8–11× the weight traffic being reported,
and `bytes_moved` counted only W. The E2M1 LUT sat in `__constant__` indexed by a runtime value,
which diverges across a warp and serializes into up to 8 transactions instead of broadcasting —
that alone put NVFP4 *below* trellis at 14.6 GB/s against production's 206.

**v2 said trellis loses at 0.845×.** Harness fixed, but the decoder was bad: it summed the two
half-precision lanes of each hash into one Gaussian sample, ~8 ALU ops per weight, and sat at
**94% of its ALU ceiling**. It was measuring my arithmetic, not the code.

**v3 keeps both lanes**, as QTIP's 3INST actually does — one hash, two weights, ~3 ops each. The
ceiling moves from 279 to 434 Gweight/s and the measured rate from 262 to 395.

The lesson is the NOMEM twin. A decoder pinned at its own ALU ceiling and a decoder that is
memory-bound look identical in GB/s, and the difference is the entire verdict: only the second
one can be helped by reading fewer bytes.

## A claim in our own wiki is refuted

`WIKI.md` §6 records "≥10 shared-memory codebook lookups per 32-bit word absorbed with **zero**
bandwidth loss — this makes trellis/codebook decode compute-free here."

At 3 bits a 32-bit word holds 10.67 weights, so a per-weight shared-memory codebook is ~10.7
lookups/word — precisely the regime that claim covers. Measured:

    trellis_B  arithmetic codebook, no lookups   395 Gweight/s
    trellis_C  shared-memory codebook            271 Gweight/s   (-31%)

The lookups are **not** free at this operating point, and "trellis decode is compute-free here"
is wrong as stated — trellis decode is compute-*bound* here. The original measurement was taken
where the kernel had bandwidth stall cycles to hide lookups in; the 3-bit path does not.

## What this means for the build decision

**Cost.** A GPU Viterbi encoder (numpy does ~3k weights/s; the checkpoint has 108 G expert
weights, so a CUDA encoder is mandatory, not optional), a new container format, a new decode
kernel, loader changes, and re-validation of all eight gates.

**Benefit.** ~+7% decode, 33.0 → ~35.4 tok/s, *conditional on capability holding at 3 bpw*.

**Comparison.** The open-frontier table lists "close the M>1 kernel inefficiency" as lowering the
speculation break-even from τ 3.36 to 2.40, which would make speculation profitable on prose —
where it currently is not. Given prose is the common case, that is plausibly worth more than 7%
and needs no new container, no encoder, and no re-quantization of the checkpoint.

**Recommendation: do not build the trellis subsystem for +7%.** The distortion bridge
(`EXPERT_BITS_EVAL.md`) shows trellis@3.0 is a strictly better code than the RTN the sweep is
scoring, so if capability holds it holds — but capability was never the reason to build this.
Speed was, and speed came in at half the projection because the term stopped being
bandwidth-bound the moment the bytes came down.

The measurement to revisit this on: if the M>1 work or anything else lifts the MoE kernel toward
its own bandwidth limit again, the expert term becomes bandwidth-bound once more and the trellis
saving would convert at closer to its full 1.406×.
