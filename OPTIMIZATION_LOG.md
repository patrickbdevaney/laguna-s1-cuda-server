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
| 16 | Attention kernel: template `G`, Q in registers (stack 224 B → **0**) | 20.79 | 20.75 / 21.16 | NEUTRAL at short ctx, **kept** — see below |
| 17 | `attend_nsplit` no longer depends on `M` (split sized by key length alone) | — | — | **CORRECTNESS FIX**, decode shape unchanged |
| 18 | **FP8 GEMM: load `x` as one `uint4` instead of 8 scalar 2-byte loads** | 0.809 ms | **0.292 ms** | **WON 2.8x at M=6**, 1.5x at M=1 |
| 19 | **GEMM warps per block 1 → 4** (occupancy, bit-exact) | 21.65/21.70 | **24.59/24.50** | **WON +13.4 %** |
| 20 | **MoE: activation row as `uint4`, not 32 scalar 2-byte loads** (bit-exact) | 24.55 | **25.62/25.51** | **WON, with #21** |
| 21 | **Router top-k on a warp, not on `threadIdx.x == 0`** | — | — | (folded into #20's measurement) |
| 22 | **FP8 GEMM weight loads 8 B → 16 B**, now that 4 warps hide the latency | 25.63 | **26.59/26.66** | **WON +4.0 %** |
| 23 | **Segmented GEMM: q\|k\|v\|g, sh_gate\|sh_up, mlp_gate\|mlp_up each one launch** | 26.63 | **27.18/27.35** | **WON +2.7 %**, bit-exact |
| 24 | Shared-memory `x` staging at the verify shapes | 0.271 ms | 0.492 ms | **LOST 1.8x, default off** |
| 25 | N-blocking the M>1 GEMM (2 or 4 outputs per warp) | 0.270 ms | 0.271 / 0.270 | **NEUTRAL, default 1** |
| 26 | MoE gate/up K-split (grid.z over the K axis + deterministic combine) | 27.30/27.49 | 26.9/26.7 (KS=2), 27.41/27.47 (KS=3) | **NEUTRAL, default off** |

### #23 — the loss was in the SMALL projections, not the big ones

Per-shape, the dense GEMM was already near the ceiling where it mattered most and terrible
where it looked unimportant:

| projection | GB/s |
|---|---:|
| q_proj [9216,3072] | 236 |
| o_proj [3072,9216] | 228 |
| k/v_proj [1024,3072] | 168 |
| g_proj [72,3072] | **25** |

`g_proj` is 0.22 MB — easy to dismiss, and it was costing more per byte than anything else in
the model. The fix is not a better kernel but a better *launch*: q, k, v and g share K and
share their input, so they are one GEMM over N = qd + 2·kvd + heads. Same for the shared
expert's gate/up and layer 0's dense gate/up.

Requirement: the weights must be contiguous in the arena. `reserve()` had them interleaved
with their scale vectors, so the reservation order changed to `q8,k8,v8,g8` then
`q8s,k8s,v8s,g8s` — every size here is already a multiple of the 256 B reservation quantum, so
"consecutive" really is "contiguous". (`g8s` at 288 B is the one that is not, which is why it
is last in its group.)

Each segment keeps its own output buffer and row stride, so nothing downstream learns about
the fusion, and one warp still reduces one whole output row in the same order — **bit-exact**.
Greedy 8/8 on both the FP8 and BF16 paths and Gate B1c still 0.0000e+00.

### #22 — a correct decision that expired

`k_gemm_fp8` deliberately used 8-byte weight loads, with a comment explaining why: 16-byte
loads leave only K/16/32 = 6 iterations per lane, which measured 125 GB/s against the BF16
kernel's 256. That reasoning was right **at one warp per block**. Once #19 put four warps on
each SM there are enough warps to hide the longer-latency, fewer-iteration form, and the
decision inverts:

| shape | 8 B | 16 B |
|---|---:|---:|
| q_proj sliding [9216,3072] | 230.5 | 235.7 |
| o_proj sliding [3072,9216] | 193.4 | **227.6** |
| q_proj global [6144,3072] | 225.4 | **244.4** |
| k/v_proj [1024,3072] | 138.9 | **168.3** |
| end to end | 25.63 | **26.59 / 26.66** |

Unlike #19 and #20 this one is **not bit-exact** — it repartitions K across lanes, so the
dot product sums in a different order. It is applied identically at every M, so decode,
prefill and verify still agree with each other (Gate B1c's invariant), and greedy stayed 8/8
against the oracle on both the FP8 and BF16 paths. Kept as a knob (`LG_FP8_VEC`) rather than
hardcoded, because the next occupancy change could invert it again.

**The general lesson is the one this entry is really for:** a tuning constant justified by a
measurement is only valid under the conditions of that measurement. When #19 changed
occupancy it silently invalidated #15's load-width choice, and nothing in the code said so.
Re-run the width/unroll sweeps after any occupancy change.

### #20/#21 — the same scalar-load defect, and a "negligible" kernel that was 3.2 %

`LG_PROF` categories were too coarse and its own `cudaDeviceSynchronize` per category
inflated the small ones — it reported attention core at 7.8 % of the step when the truth is
1.4 %. `nsys` per-kernel is the right instrument. Note it must be run with `LG_NOGRAPH=1`:
with the CUDA graph on, every decode kernel is inside the graph and the report shows only the
prefill forward.

Decode, 29 steps, ms/step:

| kernel | ms/step | % | GB/step | GB/s | % of ~254 |
|---|---:|---:|---:|---:|---:|
| `k_gemm_fp8` q/k/v/o/g | 14.19 | 34 | 2.806 | 198 | 78 % |
| `k_moe_gateup_rp` | 9.31 | 22 | 1.663 | 179 | 70 % |
| `k_gemm_bf16` | 8.37 | 20 | 1.800 | 215 | 85 % |
| `k_moe_down_rp` | 5.61 | 14 | 0.832 | **148** | 58 % |
| `k_router` | 1.32 | 3.2 | ~0 | — | pure latency |
| attention core (all kernels) | 0.60 | 1.4 | 0.090 | — | not a problem |

**#20.** Both repacked MoE kernels read the activation row as `xh[2*j]` off a `uint16_t*`.
The row is broadcast to all 32 lanes, which makes the scalar form look harmless — it is not:
32 scalar loads per 16-byte code chunk where 4 vector loads do. Rewritten as four `uint4`
loads with each 32-bit word split into its two bf16 halves in registers. The code-byte order
is untouched, so both accumulators sum in the same sequence — **bit-exact**, and greedy stayed
8/8. (Naming the words matters: taking the address of a local `uint4[4]` would put it straight
back in local memory, which is defect #1 all over again.)

**#21.** `k_router` ran its top-10-of-256 under `if (threadIdx.x == 0)` — 2560 serial
comparisons with 255 threads idle, 27 µs per launch × 47 layers = **1.32 ms, 3.2 % of the
step**. The comment justifying it said "negligible next to the expert GEMMs". Moved to a
warp-wide strided scan plus a shuffle reduction. The tie-break is the subtle part: torch.topk
gives the lowest index, so each lane scans ascending with a strict `>` and the shuffle
reduction breaks value ties by index explicitly. Gate G4 still reports 0/540 mismatching
indices.

**Lesson worth carrying:** "negligible" has to mean negligible *measured*. Both of these were
justified in comments by flop counts, and both were costing more than several of the wins in
this table.

### #18/#19 — the dense GEMMs were issue-bound, and half the warp slots were unreachable

Found by making `bench_decode`'s byte budget honest. It had `B_tok` hardcoded at 10.044 GB —
the *stock BF16* figure — so it kept crediting us with the attention bytes FP8 had already
removed and reported 83 % of roofline where the truth was 66 %. **Never hardcode a byte budget
a build flag can change.**

With the real denominator, a decode-only profile (the old one silently included the 2048-token
prefill, which dominates it) gave achieved bandwidth per category:

| category | GB/tok | % bytes | % time | GB/s | % of ceiling |
|---|---:|---:|---:|---:|---:|
| attn q/k/v/o/g GEMM | 2.806 | 39.0 | 37.1 | 124.1 | 55 % |
| MoE experts | 2.495 | 34.7 | 27.8 | 147.0 | 65 % |
| shared+dense | 1.114 | 15.5 | 12.1 | 151.4 | 67 % |
| lm_head | 0.617 | 8.6 | 4.1 | **245.3** | 108 % |
| attn_core + router + norms | 0.165 | 2.3 | **18.9** | ~19 | latency, not bytes |

`lm_head` and `q_proj` run the **same kernel** on the **same K**. 245 vs 124 GB/s is exactly
2×, and FP8 reads exactly half the bytes per instruction — so the kernel was issue-bound, not
bandwidth-bound, and the only difference was how many bytes each instruction happened to move.

Two independent causes, both invisible in the byte model:

**(a) `WARPS=1` made half the machine unreachable.** This part allows 24 CTAs *and* 48 warps
per SM. One warp per block hits the CTA limit at 24 warps — **half the warp slots cannot be
filled at any grid size.** The comment justifying it ("maximises grid fill on a 20-SM part")
was inherited from gemma and is simply wrong here: at N=9216 the grid is 100× the machine
either way, so grid fill was never the binding constraint; resident warps were. One warp still
owns one whole output row, so this is **bit-exact at any value** — greedy stayed 8/8.

| warps/block | q_proj BF16 | q_proj FP8 | end-to-end |
|---:|---:|---:|---:|
| 1 | 140.4 GB/s | 155.0 | 21.65 / 21.70 tok/s |
| 2 | 253.6 | 231.3 | 24.59 / 24.50 |
| **4** | 253.6 | 229.0 | **24.52 / 24.59** |
| 8 | 229.7 | 193.4 | — |

**(b) The FP8 kernel loaded `x` with eight scalar 2-byte loads** where the BF16 kernel used one
`uint4`. At MM=8 the inner loop issued 64 loads where 8 would do. It cost 2.8× **at M=6** —
the speculative-verify shape — and would have surfaced as "DFlash does not help on this box"
rather than as a GEMM bug. Worth stating generally: **a microbenchmark that only covers the
decode shape cannot see a defect that only bites the verify shape.** Add M=k+1 rows to every
kernel bench.

Note both microbench corrections that made this measurable: the harness reused a single weight
buffer, so anything under the 32 MB L2 (which is *every* attention tensor once FP8 halves it)
was measuring L2, not DRAM; and the tail crashed on a stale `attend` declaration that passed
an `int` where the kernel now takes `const int*`.

Also: `q_proj` reaches **253.6 GB/s**, above the 227.4 GB/s that `bw_probe` reported as the
streaming ceiling. The probe was a low sample; the real achievable figure is ~254 GB/s
(93 % of the 273 GB/s spec peak), and every "% of roofline" before this entry is correspondingly
optimistic.

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

### #16 — attention accumulator was in local memory (kept; the win is at long context)

`k_attn_split` declared `acc[9][4] + mx[9] + ls[9]` with **runtime** `G`, so the compiler
could not unroll the g-loops, the indices were dynamic, and ptxas placed the accumulator in
**local memory** — a measured **224-byte stack frame**, executing ~72 local loads and ~72
local stores per key position. This is the third instance of the same failure mode in this
project (after the `e2m1f` LUT and the MoE token loop).

Templating `G` on its only two values (6 global, 9 sliding) and moving Q from shared memory to
registers gives **0 bytes of stack frame** on both instantiations, and drops G=6 shared memory
from 23 328 to 12 480 B. Bit-identical: 13/13 kernel gates, attention at 1.5e-07, greedy 8/8.

**Measured NEUTRAL at the benchmark's context** (20.75 / 21.16 against a 20.79 control) — and
that is the expected result, not a disappointment: at 83 live tokens the attention core is a
small share of the step. Kept because it is free, bit-exact, and removes a structural defect
whose cost grows linearly with context. **Its value has to be demonstrated on the context
curve, not here** — which is exactly the measurement hygiene point below.

### ⚠ MEASUREMENT SCOPE — every number in this log was taken at ~83 tokens of context

`bench_decode` defaults `PRE=0`; `CTX` sizes KV *capacity*, not live context. So the profile
that ranked every lever in this file was taken at a live position of a few dozen tokens, where
the 12 global layers read ~130 KB instead of the 100 MB they read at ctx 4096. Attention's
share of the step is ~12 % here and materially larger at real context.

This does not invalidate the wins — the MoE repack, the CUDA graph, the fused norm and FP8
attention are all context-independent — but it does mean **the ranking is only valid for short
context**, and levers whose payoff scales with context (#16, the split heuristic, sub-FP8 KV)
are systematically under-valued by it. `tests/bench_ctx.cu` measures the curve properly.

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

### 0. ~~MoE gate/up is grid-starved~~ — **tested three ways, it is not. Gap unexplained.**

The theory was clean: `k_moe_gateup_rp` sits at 177 GB/s (70 % of the ~254 ceiling) and
launches only `nact_max × MI/(RPNB·WY)` = 80 blocks = 320 warps of the 960 the part holds,
while `k_moe_down_rp` launches 240 blocks and responded to the vector-load fix (#20) exactly
as an instruction-starved kernel should. So gate/up looked grid-starved.

It is not. Three independent attacks, all negative:

| attempt | result |
|---|---|
| #24 stage `x` in shared memory (verify shapes) | **1.8× slower** |
| #25 N-block: 2 or 4 output rows per warp | exactly neutral (0.270 / 0.271 / 0.270 ms) |
| #26 K-split: grid.z over the K axis, 320 → 960 warps, deterministic combine | 27.44 vs 27.40 tok/s — **nothing** |

And #14 in the table above had already doubled the warp count a different way and *lost* 1.3 %.
Four experiments now say the same thing: **this kernel's 177 GB/s is not limited by warp
occupancy, by load width, or by K-parallelism.** Compute is not the limit either — it issues
about 2 FMAs per weight byte against roughly 6× headroom.

Recorded as unexplained rather than papered over with a fifth guess. The next person should
start from a hardware-counter profile (`ncu`) of this one kernel rather than from another
structural hypothesis; every structural hypothesis available from first principles has now
been tried and priced. The K-split survives behind `LG_MOE_KS` so the measurement is
reproducible, and costs nothing at the default of 1.

### 1. Self-quantize the BF16 remainder

`k_moe_gateup_rp` sits at 177 GB/s (70 % of the ~254 ceiling) and 9.4 ms/step, the worst of
the four big kernels. The vector-load fix (#20) moved `k_moe_down_rp` 148 → 169 GB/s and did
**nothing** for gate/up, which localises the cause: down launches 240 blocks, gate/up launches
only `nact_max × MI/32/4` = **80 blocks = 320 warps = 16 of the 48 warp slots per SM**. It is
not instruction-starved, it is grid-starved, and the grid is capped by the problem shape
(10 experts × 1024 outputs, one thread per output).

The fix is a split along K (H=3072) with a deterministic combine, the same move that made
attention #2 the biggest early win. Note the counter-evidence: #14 doubled the warp count by
putting gate and up on separate warps and LOST 1.3 % — but that split also duplicated the x
reads and halved per-warp ILP, neither of which a K-split does. **Gain ~+4 %, P medium.**
Only helps M=1; at verify M=k+1 the active-expert count already fills the grid.

Second-order: `o_proj` runs at 228 GB/s against `q_proj`'s 236 for identical bytes — N=3072
with K=9216 gives 36 iterations per lane where q gives 12. Worth one sweep of the load width
for that shape specifically.

### 2. Self-quantize the BF16 remainder — **the largest lever, now evidence-staged**
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
