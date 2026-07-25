# OPTIMIZATION_LOG.md — Laguna edition

Every attempt: hypothesis → single change → measurement → **WON / LOST / NEUTRAL**.
Back-to-back A/B only (thermal drift is real on this box). Ideas not yet tried live in
§Backlog with an EV estimate, so the ledger stays as reusable as gemma's.

---

## Measured hardware facts (established before any kernel work)

### M1 — Thor streaming bandwidth is **227 GB/s**, not 200 · `tests/bw_probe.cu`

20 SMs × 8 blocks × 256 threads, `__ldcs` uint4 streaming over 4 GB, best-of-5:

| allocation | achieved | vs 273 peak |
|---|---:|---:|
| `cudaMalloc` | **227.4 GB/s** | 83 % |
| `mmap` + `cudaHostRegister(Mapped)` | 160.2 GB/s | 59 % |
| `malloc` + `cudaHostRegister(Mapped)` | 160.9 GB/s | 59 % |

Two consequences:

1. **The inherited "~200 GB/s achievable" from the gemma constitution is 12 % pessimistic.**
   The AR wall moves 19.9 → **22.6 tok/s**, and the speculative ceiling 38.6 → **43.8 tok/s**
   (HumanEval T=0 at k*=5). `ROOFLINE.md` §9 carries the corrected table.
2. **Zero-copy weight mapping costs 30 % of bandwidth.** A `cudaHostRegister`'d mmap of the
   repack cache would have been elegant — one copy of 74 GB, instant startup — but on a
   bandwidth-bound decode path it would cap us at 160 GB/s, giving up more than the entire
   speculative win. **Decision: weights live in `cudaMalloc` memory.** The repack cache is
   read into device buffers, not mapped. Recorded so it is not revisited.

### M2 — L2 is **32 MB**, `persistingL2CacheMaxSize` **24 MB**

Resolves the on-device TODO the gemma repo left open. This is large relative to Laguna's
sliding-window KV: **1.05 MB per SWA layer** (512 × 8 kv-heads × 128 × 2 × 1 B FP8), 37.7 MB
for all 36 layers. So one layer's entire window fits in L2 with room to spare, and up to ~22
layers' worth could be pinned via `cudaAccessPolicyWindow`. Promising for `RESCOPE.md`
priority 5 — logged in Backlog, not yet attempted.

Device also reports `integrated=1`, `pageableMemoryAccess=1`, `hostRegisterSupported=1`,
`unifiedAddressing=1`, host and device pointers identical under `cudaHostGetDevicePointer`.

---

## Attempts

Baseline for this arc: **`tests/bench_decode.cu`, median of N steps at ctx 4096**, greedy
output re-checked on the same run. Thermal drift is real on this box, so every entry below is
back-to-back A/B/A.

| # | change | before | after | verdict |
|---|---|---:|---:|---|
| 1 | FP4 dequant: `e2m1f` LUT → HW `__nv_cvt_fp4x2_to_halfraw2` | 11.0 GB/s | **27.1 GB/s** | **WON (2.5x on the kernel)** |
| 2 | Attention: flash-decoding split over the key range | 8 blocks/step | ~80 blocks | **WON** |
| 3 | MoE grid sized by actual M, not `MAXTOK` | 256-wide grid | 10-wide | **WON** |
| — | *(1+2+3 combined, end to end)* | **3.66 tok/s** | **9.23 tok/s** | **+152 %** |
| 4 | MoE accumulators templated on a compile-time token bound | 9.00 | 9.31 | marginal (+3 %, within noise) |
| 5 | MoE E4M3 scale rows staged in shared memory | 9.34 | 9.17 | **LOST, reverted** |
| 6 | MoE token loop specialised TM=1 at decode (registers 127→50 / 80→40) | 9.35 | **9.73** | **WON +4.1 %** |
| 7 | **Offline expert repack + thread-per-output MoE** | 9.71 | **16.57** | **WON +70.6 %** |
| 8 | Same repack applied to the BF16 attention/dense GEMMs | 16.57 | 11.14 | **LOST −33 %, reverted** |
| 9 | GEMM M-loop specialised at M=1 (registers 59→32) | 16.57 | 16.59 | NEUTRAL (kept: register headroom) |
| 10 | **G9 — whole-step CUDA graph** (device-side position counter) | 16.59 | **17.88** | **WON +7.8 %** |
| 11 | Doubling re-capture bound for the attention split count | 17.88 | **18.74** | **WON +4.8 %** |
| 12 | **Fused add + RMSNorm + f32→bf16 cast** (`D` a template constant) | 18.46 | **18.90** | **WON +2.4 %** |
| 13 | Sliding-ring `cap = window + MAXTOK` | — | — | **CORRECTNESS FIX** |
| 14 | MoE gate/up on separate warps (320→640 warps) | 18.90 | 18.62/18.68 | **LOST −1.3 %, default off** |
| 15 | **FP8 e4m3 attention weights, per-output-row scale** | 18.94 | **20.44** | **WON +8.1 %** |

### #1 — the LUT was in local memory (WON)
`e2m1f()` indexes `const float t[8]` by a runtime code. Inside a GEMM inner loop the compiler
cannot keep that in registers, so every weight cost a **local-memory** load. The hardware
converter decodes both nibbles of a byte in one instruction, register-only.
Generalised lesson: **any small lookup table indexed by a runtime value inside an inner loop
is local memory until proven otherwise.**

### #2 — attention was grid-starved by construction (WON)
One block per (query, kv_head) is 1×8 = **8 blocks on 20 SMs** at decode. It measured 1–2 % of
streaming roofline. Split the key range, combine with a second pass. Pure scheduling change,
deterministic combine.

### #3 — grid sized for the wrong M (WON)
The MoE grid used `min(E, MAXTOK·topk) = 256`, but at decode there are at most 10 active
experts, so ~96 % of blocks launched only to read `*nactive` and exit.

### #5 — shared-memory scale staging (LOST, −1.8 %)
Hypothesis: the E4M3 scale stream is read as 1-byte loads at stride 64, so one warp
instruction touches 16 cache lines — a 16× amplification on a stream that is 1/8 of the
payload. Staged the scale rows cooperatively into shared instead.

Measured A/B/A: **9.34 → 9.17 → 9.33**. Thermally stable, so the −1.8 % is real. The
amplification is evidently already absorbed by L2 (a 192-byte scale row is one or two lines
and is reused by all 32 lanes), and the `__syncwarp` plus shared round-trip costs more than it
saves. **Reverted.** Recorded so it is not retried: on this shape, scale reads are not the
problem.

### #6 — the token-unroll was a register wall (WON, +4.1 %)
`-Xptxas -v` said `k_moe_gateup<4>` used **127 registers**. At 128 threads/block that is 16 K
of the SM's 64 K register file, capping occupancy at **4 blocks/SM**. The cause is the
`#pragma unroll` over the 4-token loop: it replicates every temporary. At decode each expert
receives exactly one token, so dispatching a `TM=1` instantiation drops it to **50 registers**
(and `k_moe_down` 80 → 40). Experts with more tokens than `TM` just loop and re-read.

Also hoisted the `elist` load and `/topk` division out of the k-loop, where they were being
redone every iteration.

### #7 — the offline repack, and why it was worth 70 % (WON)

The diagnosis that mattered was not bandwidth or coalescing — the row-major kernel was
*already* perfectly coalesced (lane `c` strides 32, so 32 lanes read 512 contiguous bytes).
The problem was **iteration count**: with `C = H/32 = 96` chunks and a 32-lane stride, a warp
that owns one output row gets exactly **3 k-iterations**. Three dependent loads is no
memory-level parallelism *inside* a warp at all, so throughput depended entirely on warp
count — and warp count was register-capped (#6).

Fix: repack each expert projection from `[n][k]` to `[n_block][k_chunk][lane][16 B]` on the
host at load (`laguna_weights.h:repack_packed` / `repack_scale`), so a **lane** owns an output
row and streams all of k:

* 96 iterations per lane instead of 3, unrolled by `RPU=4` → several loads in flight
* still perfectly coalesced (32 lanes × 16 B = 512 contiguous bytes per step)
* **no warp reduction at all** — each lane's accumulator is its own output
* each expert's region stays contiguous and the same size, so the arena layout and every
  base pointer are unchanged

| | before | after |
|---|---:|---:|
| decode | 9.71 tok/s | **16.57 tok/s** |
| prefill | 15.4 tok/s | **55.0 tok/s** |
| effective bandwidth | 97.7 GB/s | **166.5 GB/s (73 % of 227)** |
| greedy vs oracle | 8/8 | **8/8** |

Registers rose (gateup 50 → 79) because `RPU=4` holds four `uint4` plus eight scales in
flight — that is the *point*, and it pays for itself 17×.

**This also finally makes the repack cache worth building** (`OPTIMIZATION_LOG` L1 note): the
repack is now a real host-side transform over 63.9 GB, not a memcpy.

### #8 — the same repack on BF16 GEMMs LOST 33 %, and the reason is the useful part

Having won 70 % on the MoE, applying the identical transform to the BF16 attention and dense
weights looked like free money. It was correct (greedy still 8/8) and **33 % slower**.

The repack trades **warp count for iterations per warp**. That is a win only when the
row-major form is *iteration*-starved, and a loss when it is *warp*-starved:

| GEMM | N | row-major warps | row-major iters/warp | repacked warps | repacked iters/warp |
|---|---:|---:|---:|---:|---:|
| MoE gate/up (per expert) | 1024 | 1024 | **3** | 32 | 96 |
| `q_proj` sliding | 9216 | 9216 | 12 | 288 | 384 |
| `k_proj` / shared expert | 1024 | 1024 | 12 | **32** | 384 |
| router | 256 | 256 | 12 | **8** | 384 |

The MoE at 3 iterations had no memory-level parallelism inside a warp and *needed* the trade.
The BF16 GEMMs already had 12 iterations — enough to pipeline — and giving up 32× of their
warps starved the machine, catastrophically so for the small-N ones (`k_proj`, `v_proj`,
shared expert, router all drop to 8–32 warps for the entire GPU).

**Rule extracted: repack when iterations-per-warp < ~8; keep row-major when N/32 would leave
fewer than ~500 warps.** A hybrid (repacked *and* split over k, with a combine pass) could in
principle get both, and is the obvious next experiment if the attention GEMMs stay hot.

### #9 — GEMM M-loop specialisation (NEUTRAL, kept)
Same trick that won on the MoE token loop: templating `MAXM` and dispatching `MM=1` at decode
cut registers 59 → 32. Measured 16.57 → 16.59, i.e. nothing. The GEMM is evidently not
register-limited. Kept only because the register headroom is free and may matter once CUDA
graphs and the verify shapes raise pressure; **logged as neutral so it is not mistaken for a
win.**

### #10/#11 — CUDA graphs, and the position counter that makes them possible (WON, +12.9 % combined)

~1665 launches per decode step at 4.15 µs measured is 11.5 % of a 60 ms step — but the
measurement showed the real cost is worse than that: without a graph the step time is wildly
variable (median 12.6 tok/s, **best 17.5**), because 1665 host-side launches per token expose
every CPU scheduling hiccup. The graph removes the variance as well as the overhead.

To capture at all, nothing in the step may depend on a host-side position. Every
position-dependent kernel (`rope_tables`, `store_kv`, `attend`, `attend_split`) now reads the
decode position from **device memory** via a pointer.

Note the pointer, not an `extern __device__` global: `__device__` globals are **per-module**
without `-rdc=true`, so `attention.cu` and `elementwise.cu` would each have silently gotten
their own copy. nvcc warns (`#20044-D: extern declaration ... treated as a static definition`)
and the warning is easy to miss.

**#11:** the attention split count becomes `grid.z`, so it is baked in at capture. Sizing it
from full KV *capacity* made it correct but wasteful (10 splits per global layer from token
one). Sizing it from a **doubling context bound**, and re-capturing when the conversation
outgrows it, recovered another 4.8 %. Capture costs one step, and doubling makes re-captures
logarithmically rare.

### #12 — fused add+RMSNorm+cast (WON, +2.4 %)

The unfused `k_rmsnorm` at `rows=1` is `<<<1,256>>>`: one block on one of twenty SMs, and it
is **latency**-bound, not bandwidth-bound — 12 strided loads per thread behind a runtime loop
bound the compiler cannot unroll. Making `D` a template constant unrolls it into 12
independent loads issued up front, and the residual add and the `f32→bf16` cast fold into the
same pass. Three kernels → one at 48 sites, two → one at 96.

**Bit-exact by construction**: thread `t` still accumulates elements `t, t+T, t+2T, …` in the
same order, the warp/cross-warp reduction is untouched, and the cast applies the identical
`f2bf` to the identical fp32 register value. Gates: 13/13 kernel, 8/8 greedy. Registers 33/40,
no spill.

### ⚠ MEASUREMENT HAZARD — the first benchmark after a 71.9 GB process exits reads ~50 % low

This change was very nearly reverted. Its first measurement was **8.96 tok/s** against a
control of 18.50 — an apparent 2× regression on a change that had just passed every
correctness gate. Re-running the *same binary* immediately afterwards gave **18.65**.

Cause: the run started seconds after a previous 71.9 GB process exited, while the kernel was
still reclaiming those pages. That reclaim competes for exactly the resource being measured.

**Protocol, now mandatory for every A/B on this box:** allow ~25 s of settling after any
model-sized process exits, alternate A/B/A/B rather than running A then B, and never trust a
single first-run number. The final measurement — G 18.37 / 18.54 versus H 18.90 / 18.90,
alternating with settling — is stable to 0.1 % on the H side across both rounds.

This sits alongside thermal drift as a second, independent reason absolute numbers lie here.

### #13 — the sliding ring aliased future tokens into the read window (CORRECTNESS)

`cap[L] = sliding_window` exactly means the read arc `[p−window+1, p]` covers **all** `cap`
slots, so any token written later in the same batch lands inside it. Verified independently:
at `base=1000, M=64`, **63 of 64 future tokens collide** with positions the first query reads.

Decode (M=1) is immune — 512 consecutive positions map to 512 distinct slots. It bites
multi-token forwards only: prefill, and the speculative verify path. **Every prompt over 512
tokens was silently wrong on 36 of 48 layers**, and the 54-token greedy gate could not see it.
Fix: `cap = window + MAXTOK`, making the read and look-ahead arcs disjoint. Cost 4.7 MB.

Found by the Axis E research pass reading the code, not by any test we had. The gate that
would have caught it — a >512-token prompt checked against sequential single-token decode,
which is provably alias-free — is the one to add.

### #14 — MoE gate/up on separate warps (LOST −1.3 %, despite a measured +22.5 % in isolation)

A synthetic benchmark at full 2.5 GB scale showed the gate/up kernel is warp-starved
(320 warps / 33 % occupancy → 159.6 GB/s) against `k_moe_down_rp`'s identical inner loop at
960 warps → 222.3 GB/s, and that splitting the two weight streams onto separate warps takes it
to 195.6 GB/s. End-to-end it measured **18.62 / 18.68 against an 18.90 control**.

The isolated kernel win did not survive integration. Most likely the extra `gbuf`/`ubuf`
round-trip, and that in the full step the MoE shares the machine with other resident work, so
the spare warp slots the synthetic benchmark found were not actually spare. **Kept behind
`LG_SPLIT`, default off.** Lesson: a synthetic kernel benchmark measures the kernel, not the
step — the only number that counts is end-to-end.

### #15 — FP8 attention weights (WON +8.1 %, the first `B_tok` reduction)

Per-output-row e4m3 with the scale applied once after the dot product, so the inner loop is
the BF16 loop with half the bytes and no extra math. `B_tok` 10.044 → **7.242 GB**; arena
71.9 → **69.1 GB**. A/B/A: 18.90 → **20.44** → 18.97, greedy 8/8 throughout.

**One instructive failure first.** The obvious 16-byte (`uint4`) load covers 16 fp8 weights,
which at K=3072 leaves only `K/16/32 = 6` iterations per lane — half what the BF16 kernel gets,
and straight into the latency-bound regime our own repack rule describes. It measured **17.23
tok/s, slower than the BF16 baseline it was supposed to beat**. Narrowing to 8-byte `uint2`
loads restores 12 iterations per lane and gives the +8.1 %. Same lesson as #7 and #8:
**iterations per lane is the variable, and halving the bytes per weight halves it.**

**Honest accounting: this path is less byte-efficient, not more.** At 7.242 GB and 20.44 tok/s
the FP8 attention path runs at **148 GB/s effective**, against the BF16 path's 190. It wins on
bytes and loses on efficiency, netting +8.1 %. If the FP8 GEMM reached the BF16 kernel's
efficiency the same byte budget would give **26 tok/s** — so roughly two thirds of this lever
is still unclaimed.

Quality: greedy-exact on the 8-token gate, which is a *weak* signal. The research (see
`RESEARCH_FINDINGS.md`) puts weight-only 8-bit at measured-lossless — llama.cpp Q8_0 on
Llama-3.1-8B moves ppl 7.32 → 7.33 — so confidence is high, but the flag stays opt-in and the
BF16 path remains the bit-exact gate oracle.

### Where the time actually goes (per-category profile, `LG_PROF=1`)

| category | share of step |
|---|---:|
| category | before #7 | **after #7** |
|---|---:|---:|
| routed-expert MoE | 81.5 % | **37.1 %** |
| attention q/k/v/o GEMMs | 11.6 % | **36.1 %** |
| shared expert + dense | 2.7 % | 9.9 % |
| attention core | 1.3 % | 5.3 % |
| norms + casts | 0.9 % | 4.1 % |
| router | 0.9 % | 4.0 % |
| lm_head | 1.1 % | 3.5 % |

The bottleneck moved, as it always does. MoE and the BF16 attention GEMMs are now roughly
equal, so neither is a clear single target — which is itself the signal that the easy
structural wins are spent and the next real multiplier is speculation (Gate D1), not another
kernel rewrite.

The MoE is the bottleneck by a wide margin and the next arc belongs to it. Note this
*contradicts* `RESCOPE.md` §2's byte-based priority order, which put the attention path first
because it is 56 % of `B_tok`: attention is now running at ~150 GB/s while the MoE runs at
~30 GB/s, so the MoE's 25 % of the bytes costs 81 % of the time. **Bytes ranked the levers;
the profile re-ranked them.** RESCOPE §2 priority 1 (attention) is banked as "efficient
enough for now"; priority 3 (MoE) is promoted to first.

---

## Backlog, ranked by expected value

EV = expected gain × P(works) ÷ cost. Derived from `ROOFLINE.md` and `RESCOPE.md` §2.

### 1. Self-quantize the BF16 remainder — **the largest lever, now evidence-staged**
Poolside quantized only the routed experts; 7.41 GB of every decode step is BF16.

| variant | `B_tok` | AR @227 | gain | quality evidence |
|---|---:|---:|---:|---|
| stock | 10.044 | 22.6 | — | — |
| **FP8 attention only** | 7.241 | **31.4** | **+39 %** | **near-lossless (high confidence)** |
| + NVFP4 `o_proj` | ~6.9 | ~33 | +46 % | small; NVIDIA validated exactly this |
| NVFP4 all attention | 6.015 | 37.7 | +67 % | **−2.3 pts recovery, MEASURED** |
| NVFP4 all non-expert | 4.718 | 48.1 | +113 % | compounding, unmeasured |

**The research pass changed this entry materially** — see `RESEARCH_FINDINGS.md`. A controlled
A/B on DeepSeek-R1 (identical GPTQ recipe, attention in vs out) measures **98.81 % → 96.52 %**
recovery when attention is included. Kimi-K2, DeepSeek-R1-0528 and both Llama-4 models all
exclude attention; NVIDIA's most aggressive NVFP4 variant adds only `o_proj`. Poolside's choice
was the industry norm, not an oversight. And the damage scales *down* with active params —
Qwen loses 0.8 GPQA points at 22 B active but 5.7 at 3.3 B; Laguna is 8.5 B.

⇒ **Build stage 1 (FP8 attention) only.** INT8 weight-only is measured lossless and FP8's error
is a strict subset; it needs no calibration data and gives 1.39× on `B_tok` for near-zero risk.
Stages 3–4 need an eval harness first, and **not OpenLLM v1** — every recipe in the survey,
good and bad, scores 98–100 % there. Only AIME / GPQA-Diamond / BBH discriminate.

### 2. `k*` selection — free, already modelled
Both model cards recommend k=7/15; the byte model says k*=3–5 on this hardware and k=15 is
worst at every temperature. Costs one sweep (Gate D1). **Gain +15–30 % vs shipping k=15.**

### 3. ~~Whole-step CUDA graph~~ — **DONE, measured +7.8 % (not the +10–20 % predicted)**
Implemented as #10/#11. The prediction was too generous: of the 4.15 µs per launch, only
~2.6 µs was recoverable and ~1.5 µs is irreducible null-kernel execution. Residual in-graph
launch cost is ~0.7 µs × 1665 ≈ 1.2 ms = **2.1 % of the step**, which is the hard ceiling on
every remaining launch-count lever.

### 4. SWA KV residency in L2 (`cudaAccessPolicyWindow`)
See M2. 1.05 MB per layer against a 32 MB L2. Bounded by the sliding layers' share of KV
traffic, which is small in absolute terms (37.7 MB/step) — so the gain is on *latency*, not
bytes. **Gain low-single-digit %, P medium, cost low.** Do it when profiling says attention
is latency-bound.

### 5. Sub-FP8 KV on the 12 global layers only
The global layers are 100 % of the context-scaling KV cost (24 576 B/token; the 36 sliding
layers are a constant 37.7 MB). 4-bit KV there would halve the long-context tax: at 262 K,
6.48 → 3.24 GB/step, restoring ~9 points of throughput. Quality risk concentrated in exactly
the layers that carry long-range information, so this is a *long-context-only* option behind
a flag. **Gain +15 % at 256 K, 0 % at 4 K. P medium, cost medium.**

### 6. Fuse `g_proj` into the QKV launch
`g_proj` consumes the same post-layernorm hidden state as Q/K/V and is tiny
(`[48|72, 3072]`, ~0.4 MB/layer). Fusing removes 48 kernel launches per step. **Gain
low-single-digit %, P high, cost low.** Bit-exact.

### 7. ~~FlashNorm fold~~ — **NOT bit-exact, and superseded. Do not build.**
I claimed this was bit-exact. It is not. Proposition 1 of arXiv:2407.09577 (`W*_ij = g_i·W_ij`)
is exact *in real arithmetic*; in our pipeline the GEMM today computes
`Σ bf2f(W)·bf2f(f2bf(bf2f(g)·x·inv))` and folded it computes
`Σ bf2f(f2bf(g·W))·bf2f(f2bf(x·inv))` — rounding `g·W` to bf16 offline is a real precision
change across 5.6 GB of attention weights. It also buys almost nothing: it removes a 6 KB
weight read from a kernel that is latency-bound, not bandwidth-bound. **Superseded by #12**,
which took the same sites and is genuinely bit-exact.

Proposition 2 (deferring the `1/RMS` scalar past the matmul) *is* exact but targets hardware
where the norm blocks the matrix unit — irrelevant on a CUDA-core GEMV. Proposition 3
(dropping the first RMSNorm because QK-norm follows) does **not** apply: the same normed
hidden also feeds `v_proj` (no downstream norm) and `g_proj` (nonlinear softplus), so the
scalar cannot cancel.

### 8. ~~Fused add+RMSNorm~~ — **DONE as #12, +2.4 %.** Remaining in this family: fuse the
`swiglu`+cast pair, have `k_moe_gateup_rp` write bf16 directly, and fuse `moe_finalize`+`add`.
Each is bit-exact and worth a few tenths of a percent; ~+0.3 % combined.

### 9. Routing-aware draft truncation
Only if measured `E_frac(k*)` > 0.6. The model says 0.18 at k=5, so this is almost certainly
unnecessary — logged because `DIRECTIVE.md` §7.5 asks for it explicitly.

---

## Dead on arrival — inherited from the gemma ledger, not to be re-litigated

Carried from `~/gemma-cuda-hybrid/CUDA_ENGINEERING_CONSTITUTION.md` §4 unless Laguna's shapes
give a *documented* reason to revisit:

- **tcgen05 / TMEM at small M.** Block-scaled NVFP4 locks MMA_M ∈ {128, 256}; crossover is
  M ≈ 900. Gemma was at M=15; Laguna's k*≈5 puts us at **M≤6**, further away still.
- **FP8/FP4 draft.** Collapsed gemma's τ 13.33 → 11.14. The BF16 draft is the moat.
- **Tree / multi-candidate verify.** Expert-union grows per branch; depth-dominated.
- **Drafter swap (EAGLE-3/Medusa/MTP).** All τ downgrades vs block-diffusion DFlash.
- **PowerInfer / hot-cold expert tiering.** Needs a two-tier bandwidth hierarchy; Thor is one
  unified pool. Refuted, not assumed.
- **Activation sparsity (TEAL/CATS).** Measured on gemma: activations are magnitude-dense
  (≈0 % hard zeros), so it is inherently lossy — incompatible with a bit-exact verify.
  Re-measure on Laguna only if the verify path stops needing to be bit-exact.
- **cp.async pipelining on a max-grid-fill GEMM.** Block-level parallelism already hides the
  latency; async win < shared-hop + pipeline-fill cost.
- **MoE ILP levers (2-way accumulator split, RB=2 multi-output).** Both lost to the register
  wall on gemma's 4-weight-stream gateup. Laguna's `moe_intermediate` is 1024 vs gemma's 704,
  so the register pressure is *worse*, not better.
- **Grid-cooperative megakernel with `grid.sync()`.** ~35 % of token time in the barrier.
- **DSMEM clusters** (Thor caps `cta_group` at 2), **split-K/stream-K**, **TMA** (none are
  bandwidth levers).
- **NEW for Laguna — zero-copy mapped weights.** See M1: 160 vs 227 GB/s.
