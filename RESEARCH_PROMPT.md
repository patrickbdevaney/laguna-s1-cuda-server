# DEEP RESEARCH PROMPT — reaching the physical ceiling on Laguna S 2.1 base decode

**Purpose.** Surface every remaining lever — academic and industry, 2023→2026 — that could move
autoregressive (non-speculative) decode of a 117.6 B/8.5 B-active NVFP4 MoE from its current
**73 % of measured streaming bandwidth** toward, and past, the physical ceiling.

**This prompt is deliberately grounded.** The field data below is *measured on this machine*.
Recommendations that contradict it, or that re-propose something in the DEAD list, are worse
than useless — they burn the scarce resource, which is serial GPU wall-clock, not tokens.
Every recommendation must state: expected gain × P(works) ÷ cost, whether it preserves
bit-exactness, and a citation to real code, PTX, a paper, or a measured number.

---

## 1. The machine (measured, not from a spec sheet)

| | |
|---|---|
| Chip | Jetson AGX Thor, Blackwell **`sm_110a`**, **20 SMs**, 2560 CUDA cores |
| Memory | 122 GB LPDDR5X **unified**, 273 GB/s peak |
| **Measured streaming BW** | **227.4 GB/s** via `cudaMalloc` + `__ldcs` uint4 (83 % of peak) |
| Zero-copy penalty | `cudaHostRegister`'d mmap streams at only **160 GB/s** (−30 %) |
| **L2** | **32 MB**, `persistingL2CacheMaxSize` 24 MB |
| Shared memory | 228 KB/SM |
| Kernel launch overhead | **4.15 µs** measured |
| Power mode | **120 W (`nvpmodel` mode 1), NOT MAXN** — deliberate, do not assume MAXN |
| Toolchain | CUDA 13.0.48, `-arch=sm_110a`, driver 580.00, L4T R38.4 |
| Notable | `integrated=1`, `pageableMemoryAccess=1`, TMEM + tcgen05 + TMA present |

## 2. The model and its byte budget (read from the checkpoint, not the model card)

Laguna S 2.1: 48 layers, hidden 3072, head_dim 128, 8 KV heads.
**48 attention heads on the 12 global layers, 72 on the 36 sliding layers** (GQA groups 6 and
9). Sliding window 512. 256 experts, **top-10**, `moe_intermediate` 1024, plus one shared
expert; layer 0 is a dense MLP. Vocab 100 352, `lm_head` **not tied**. FP8 KV with static
per-layer scales.

**Critical and non-obvious: only the routed experts are NVFP4.** Attention, shared experts,
router, layer-0 MLP and `lm_head` are **BF16**.

Bytes read per decode step at ctx 4096 — this is the whole game:

| component | GB | dtype | share |
|---|---:|---|---:|
| attention q,k,v,o,g × 48 | **5.606** | BF16 | **56 %** |
| shared experts × 47 | 0.887 | BF16 | 9 % |
| `lm_head` | 0.617 | BF16 | 6 % |
| layer-0 dense MLP | 0.227 | BF16 | 2 % |
| routers × 47 | 0.074 | BF16 | 1 % |
| routed experts, top-10 × 47 | 2.495 | NVFP4 | 25 % |
| FP8 KV | 0.138 | FP8 | 1 % |
| **B_tok** | **10.044** | | |

Measured expert union (47 layers × all contiguous k-windows): `E_frac(5) = 0.129`,
`E_frac(15) = 0.279` — routing is far more correlated between adjacent tokens than
independence predicts.

## 3. Where we are, and why the framing matters

| | |
|---|---|
| decode | **16.59 tok/s** median, ctx 4096, greedy, bit-exact vs oracle |
| effective bandwidth | **166.7 GB/s = 73 % of the measured 227 GB/s** |
| prefill | 55 tok/s |
| AR ceiling at 227 GB/s | 22.6 tok/s |
| reference | poolside measured 13–14 tok/s for this model on DGX Spark (same BW class) under vLLM |

> **The central constraint: we are already at 73 % of streaming roofline. Kernel-efficiency
> work has at most 1.36× left in it, ever. Anything beyond ~22.6 tok/s requires READING FEWER
> BYTES — reducing `B_tok` — or exploiting a memory path faster than 227 GB/s.**

Answer both questions, and weight the second more heavily:
- **(A)** How do we capture the last 27 % of efficiency?
- **(B)** What legitimately reduces `B_tok`, or beats 227 GB/s, on *this* hardware?

Current per-category profile (after the wins in §4):

| category | share |
|---|---:|
| routed-expert MoE | 37.1 % |
| attention q/k/v/o BF16 GEMMs | 36.1 % |
| shared expert + dense MLP | 9.9 % |
| attention core (FP8 KV, head-packed GQA) | 5.3 % |
| norms + dtype casts | 4.1 % |
| router | 4.0 % |
| `lm_head` | 3.5 % |

## 4. WON on this machine (do not re-propose; do build on)

1. **HW FP4 converter over a LUT.** `e2m1f()`'s `const float t[8]` indexed by a runtime code
   lands in **local memory** inside a GEMM inner loop. `__nv_cvt_fp4x2_to_halfraw2`: 11 → 27 GB/s.
2. **Flash-decoding key-split in attention.** One block per (query, kv_head) is 8 blocks on
   20 SMs at decode; it ran at 1–2 % of roofline.
3. **Grid sized by actual M**, not a compile-time max (96 % of MoE blocks existed only to exit).
4. **Register discipline via template specialisation.** `k_moe_gateup<TM=4>` used 127 registers
   (4 blocks/SM); `TM=1` at decode uses 50. +4.1 %.
5. **★ Offline expert repack + thread-per-output MoE: +70.6 %.** Repacked `[n][k]` →
   `[n_block][k_chunk][lane][16 B]` so a *lane* owns an output row and streams all of k. The
   row-major form was already perfectly coalesced — the problem was that a warp got only
   **3 k-iterations**, i.e. no memory-level parallelism *inside* a warp. Now 96 iterations
   unrolled by 4, and **no warp reduction at all**.

## 5. DEAD — measured here or inherited with evidence. Re-proposing these is a failure.

**Measured LOST on this machine:**
- **The same repack applied to the BF16 GEMMs: −33 %.** Extracted rule: *the repack trades warp
  count for iterations-per-warp — it wins only when the row-major form is iteration-starved
  (<~8 iters/warp) and loses when N/32 leaves fewer than ~500 warps.* `k_proj`, shared expert
  and router collapse to 8–32 warps for the whole GPU.
- Staging E4M3 group scales in shared memory: −1.8 % (L2 already absorbs them).
- Zero-copy `cudaHostRegister` weights: 160 vs 227 GB/s.
- Pinned-staging H2D on this integrated part: 10.6 GB/s vs 109 GB/s copying straight from
  pageable.

**Inherited with evidence from a completed gemma-4 MoE/NVFP4 port on the same chip:**
- **tcgen05 / TMEM at small M** — block-scaled NVFP4 locks MMA_M ∈ {128, 256}; crossover
  M ≈ 900; we are at M = 1.
- **cp.async pipelining on a max-grid-fill GEMM** — block-level parallelism already hides the
  latency; async win < shared-hop + pipeline-fill cost.
- **MoE ILP levers** (2-way accumulator split, RB=2 multi-output-per-warp): both lost to the
  register wall.
- **Grid-cooperative megakernel using `grid.sync()`** — the barrier was ~35 % of token time.
- **PowerInfer / hot-cold expert tiering** — needs a two-tier bandwidth hierarchy; unified
  memory has none.
- **Activation sparsity (TEAL/CATS)** — *measured*: activations are magnitude-dense (≈0 % hard
  zeros), so it is inherently lossy.
- **DSMEM clusters** (Thor caps `cta_group` at 2), **split-K/stream-K**, **TMA** — none are
  bandwidth levers at M=1.

## 6. Constraints that bound any answer

1. **No Python on the hot path.** Pure C++/CUDA single binary. No PyTorch, no JIT at serve time.
2. The AR/verify path is gated **bit-exact against an oracle**. Lossy proposals are allowed but
   must be flagged and come with a quality-evaluation plan, not just a speed number.
3. Weights are 71.9 GB of a 122 GB unified pool. Anything needing a second full copy is out.
4. Decode is **M = 1**; speculative verify is M ≤ 16. No batching — this is single-stream
   interactive serving.
5. 20 SMs. Occupancy and grid-fill arguments that assume 100+ SMs do not transfer.
6. An **offline weight repack at load is cheap and already implemented** — layout changes are
   nearly free. Exploit this.

---

## 7. The axes to investigate

### Axis A — the last 27 % of streaming efficiency at M=1
Byte-for-byte GEMV/skinny-GEMM engineering on Blackwell and LPDDR-class unified memory.
Marlin / Machete / FLUTE / llama.cpp `mmvq` / vLLM / TensorRT-LLM / SGLang kernel structure at
batch 1. Weight layouts that beat `[n_block][k_chunk][lane][16 B]`. `__ldcs`/`ld.global.nc`
/ evict-priority hints. `cp.async.bulk` on integrated memory. Whether 227 GB/s is really the
wall or an artefact of the access pattern — is there a pattern that beats it? Interleaving
multiple weight streams. Instruction-level tricks for bf16→fp32 and FP4 unpack (bit-splice
vs hardware converter).

### Axis B — reading fewer bytes (weight side) — **weight this most heavily**
The BF16 remainder is 7.4 GB of every 10.04 GB step. What is the current SOTA for quantizing
**attention** projections specifically (they are known more sensitive than MLP)? FP8 vs NVFP4
vs INT4 with per-channel/group scales; AWQ/GPTQ/SmoothQuant/QuaRot/SpinQuant/QuIP# for
attention; rotation-based outlier suppression. What accuracy is actually retained on a
117 B MoE, and what is measured rather than claimed? Also: `lm_head` quantization; low-rank or
shared-basis factorisation of q/o projections; whether 72-head sliding layers admit head
pruning; and any 2026-era result on quantizing *only* the un-quantized remainder of a
partially-quantized checkpoint.

### Axis C — MoE at batch 1
Grouped/segmented GEMM structure when ~10 experts each get exactly one token. Expert weight
layout, fused router→gather→GEMM→scatter, avoiding the invert/scan glue. SGLang/vLLM/
TensorRT-LLM MoE-at-bs-1 code. Anything exploiting the *measured* routing correlation
(`E_frac(5) = 0.129`). Shared-expert fusion into the routed path.

### Axis D — launch, scheduling and fusion on a 20-SM part
~1665 kernel launches per decode step at 4.15 µs each. CUDA graphs (whole-step capture with
device-dependent grid sizes), persistent/megakernel designs using **sentinel-poll counters
rather than `grid.sync()`** (Hazy/Kog/MPK-style), stream-level parallelism across independent
projections, norm+cast+GEMM fusion, FlashNorm-style folding of RMSNorm weights into the next
GEMM. Quantify what is actually available at 20 SMs.

### Axis E — attention, KV, and the memory hierarchy
Head-packed GQA at groups of 6 and 9 with FP8 KV. Flash-decoding split tuning. **A 32 MB L2
with 24 MB persisting against a 1.05 MB-per-layer sliding KV window** — is `cudaAccessPolicyWindow`
worth it, and what is the measured effect on integrated parts? Sub-FP8 KV on the 12 global
layers only (they are 100 % of the context-scaling KV cost). Any 2026 result on KV layouts for
mixed sliding/global models.

---

## 8. Required output format

An **EV-ranked table**: lever · expected gain · P(works) · effort · bit-exact? · citation.
Then, for the top items, enough implementation detail to build from: exact PTX/intrinsics,
memory layout, tile shape, and the specific failure mode to watch for.

Explicitly separate:
- **incremental grinds** (single-digit %, cheap, compose), and
- **step changes** (something that moves `B_tok` or beats 227 GB/s).

State plainly when an axis is exhausted. "Nothing further here, and here is why" is a valuable
answer and will be trusted; a padded list will not.
