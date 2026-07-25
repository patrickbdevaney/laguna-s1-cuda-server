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

### Where the time actually goes (per-category profile, `LG_PROF=1`)

| category | share of step |
|---|---:|
| **routed-expert MoE** | **81.5 %** |
| attention q/k/v/o GEMMs | 11.6 % |
| shared expert + dense | 2.7 % |
| attention core | 1.3 % |
| lm_head | 1.1 % |
| router | 0.9 % |
| norms + casts | 0.9 % |

The MoE is the bottleneck by a wide margin and the next arc belongs to it. Note this
*contradicts* `RESCOPE.md` §2's byte-based priority order, which put the attention path first
because it is 56 % of `B_tok`: attention is now running at ~150 GB/s while the MoE runs at
~30 GB/s, so the MoE's 25 % of the bytes costs 81 % of the time. **Bytes ranked the levers;
the profile re-ranked them.** RESCOPE §2 priority 1 (attention) is banked as "efficient
enough for now"; priority 3 (MoE) is promoted to first.

---

## Backlog, ranked by expected value

EV = expected gain × P(works) ÷ cost. Derived from `ROOFLINE.md` and `RESCOPE.md` §2.

### 1. Self-quantize the BF16 remainder — **the largest lever in the project**
Poolside quantized only the routed experts; 7.41 GB of every decode step is BF16.

| variant | `B_tok` | AR @227 | gain |
|---|---:|---:|---:|
| stock | 10.044 | 22.6 | — |
| FP8 attention only | 7.241 | 31.4 | **+39 %** |
| NVFP4 attention only | 6.015 | 37.7 | **+67 %** |
| NVFP4 all non-expert | 4.718 | 48.1 | **+113 %** |

Not bit-exact ⇒ quality-gated, not equality-gated, and staged: FP8 `o_proj` → FP8 all
attention → evaluate → NVFP4 attention. Must come after the bit-exact stock path passes B1
so the comparison has a fixed reference. Also frees ~10 GB of resident memory.
**P(works) high, cost medium, gain very high.**

### 2. `k*` selection — free, already modelled
Both model cards recommend k=7/15; the byte model says k*=3–5 on this hardware and k=15 is
worst at every temperature. Costs one sweep (Gate D1). **Gain +15–30 % vs shipping k=15.**

### 3. Whole-step CUDA graph
48 layers × ~14 kernels on 20 SMs. Gemma measured +11 % for graphing the M=1 chain at 30
layers; Laguna has 1.6× the layers. **Gain +10–20 %, P high, cost low-medium.**

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

### 7. FlashNorm fold
Laguna uses plain RMSNorm (`g · x̂`), not gemma's zero-centred `(1+g)` — so folding the norm
weight into the following GEMM's rows is a direct scale, simpler than gemma's case. Removes
a kernel and an activation round-trip per norm, ~4 norms/layer × 48. **Bit-exact, +1–3 %.**

### 8. Fused add+RMSNorm and fused gate+up SwiGLU
~35 % MLP traffic cut per arXiv:2602.11808. Match the reduction dtype to hold τ.
**+3–6 %, P medium.**

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
