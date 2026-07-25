# Deep-research prompt — pushing Laguna S 2.1 decode past 27.4 tok/s on Jetson AGX Thor

**Purpose.** We have a working pure-CUDA/C++ inference server for `poolside/Laguna-S-2.1-NVFP4`
with `Laguna-S-2.1-DFlash-NVFP4` speculation on a Jetson AGX Thor. Base decode is at **27.4
tok/s**, which is **197 GB/s effective against a measured ~254 GB/s achievable ceiling**. We
have exhausted every optimization we could derive from first principles, including four
consecutive negative results on the single largest remaining gap. This prompt asks for
external evidence — published kernels, papers, vendor documentation, production inference
stacks, and measured numbers — to find the levers we cannot see from inside our own repo.

Answer with **specific, checkable claims**: name the kernel/file/paper, give the measured
number and the hardware it was measured on, and state the conditions under which it transfers
to our shapes. **A plausible-sounding idea with no measurement attached is worse than
useless to us** — we have already spent four experiments on plausible-sounding ideas.

---

## 1. The hardware, exactly

| property | value | how we know |
|---|---|---|
| part | Jetson AGX Thor, Blackwell, `sm_110a` | — |
| SMs | **20** | device query |
| memory | 122 GB LPDDR5X **unified** (117 GB addressable), one pool | device query |
| peak BW | 273 GB/s spec | vendor |
| **achievable BW** | **~254 GB/s** | measured on a real 56.6 MB weight GEMV, not a synthetic probe |
| `cudaMalloc` stream | 227 GB/s | `bw_probe`, now known to be a low sample |
| `cudaHostRegister`'d mmap | **160 GB/s** | measured — zero-copy weights are a 30 % loss, rejected |
| L2 | 32 MB, `persistingL2CacheMaxSize` 24 MB | device query |
| shared memory | 228 KB/SM | device query |
| occupancy limits | **1536 threads / 48 warps / 24 CTAs per SM** → 960 resident warps | device query |
| registers | 64 K per SM | device query |
| power | **120 W, standard nvpmodel — NOT MAXN** (deliberate; MAXN is a separate free win later) | — |
| `integrated` | 1; `pageableMemoryAccess` 1; host and device pointers identical | device query |

The 24-CTA-per-SM limit against 48 warps per SM is load-bearing: **one-warp blocks can only
ever reach half the warp slots**, which was worth +13.4 % when we found it.

## 2. The model, exactly

`poolside/Laguna-S-2.1-NVFP4` — 117.6 B total, 8.5 B activated.

```
layers            48       hidden 3072      head_dim 128     vocab 100352 (lm_head NOT tied)
attention         12 global (48 heads, GQA 6) + 36 sliding (72 heads, GQA 9), window 512, 3:1
rotary            global: yarn theta 5e5 on 64 of 128 dims;  sliding: theta 1e4 on all 128
norms             QK-RMSNorm per head, PRE-rope; per-head softplus gate on attn out (unbounded)
MoE               256 experts, top-10, moe_intermediate 1024, sigmoid router,
                  e_score_correction_bias affects SELECTION ONLY, routed output scaled 2.5
shared expert     1024, added before the routed sum;  layer 0 is DENSE (intermediate 12288)
quantization      ONLY the routed experts are NVFP4 (E2M1 + E4M3 per-16 group + fp32
                  per-tensor RECIPROCAL global scale). Attention, shared experts, router,
                  layer-0 MLP and lm_head are all BF16.
KV                FP8 e4m3, static per-layer k_scale/v_scale shipped in the checkpoint
```

Byte budget per decode token, **measured from the checkpoint layout in use** (FP8 attention on):

| component | GB/token | share |
|---|---:|---:|
| attention q/k/v/o/g (FP8 weight-only, per-row scale) | 2.806 | 39 % |
| MoE routed experts (10 of 256, NVFP4) | 2.495 | 35 % |
| shared expert + layer-0 dense (BF16) | 1.114 | 15 % |
| `lm_head` (BF16, 100352×3072) | 0.617 | 9 % |
| KV read + norms + router | ~0.16 | 2 % |
| **total `B_tok`** | **7.20** | |

Draft: `Laguna-S-2.1-DFlash-NVFP4`, 2.230 GB BF16, 6 layers, all sliding-512, fused
`qkv_proj [11264,3072]`, `fc [3072,18432]` fusing six target taps from layers
`[1,10,19,29,38,47]`, six `aux_hidden_norms`, `block_size` 16, `mask_token_id` 12,
`causal: true`. No embed, no lm_head — both shared with the target. **Block-diffusion
drafting: one draft forward denoises the whole block.**

## 3. Where the time actually goes (nsys, per decode step, 29 steps, ctx 4096)

| kernel | ms/step | % | GB/step | GB/s | % of ~254 |
|---|---:|---:|---:|---:|---:|
| `k_gemm_fp8` (attention q/k/v/o/g) | 13.12 | 34 | 2.806 | 214 | 84 % |
| `k_moe_gateup_rp` (NVFP4 experts) | 9.40 | 25 | 1.663 | **177** | **70 %** |
| `k_gemm_bf16` (shared, dense, lm_head) | 8.80 | 23 | 1.800 | 205 | 81 % |
| `k_moe_down_rp` (NVFP4 experts) | 4.93 | 13 | 0.832 | **169** | **67 %** |
| `k_router` | 0.39 | 1 | ~0 | — | latency |
| all attention-core kernels | 0.60 | 1.4 | 0.090 | — | not a problem |

Kernel structure:
* **Dense GEMV** — one warp owns one whole output row and reduces over K with `__ldcs` 16-byte
  loads; 4 warps per block; M-tiled by 8 for the verify shape; segmented so q\|k\|v\|g issue as
  one launch.
* **MoE** — weights **offline-repacked** to `[n_block][k_chunk][lane][16 B]` so one thread owns
  one output element and reads 16 contiguous bytes per step (this repack was worth **+70 %**).
  Per 16-byte code chunk a lane also loads 2 E4M3 group scales per matrix. Gate and up are
  computed together in one kernel; `down` is separate. FP4 decode uses
  `__nv_cvt_fp4x2_to_halfraw2` (the hardware converter; a `float[8]` LUT was 2.5× slower
  because it landed in local memory).

## 4. THE CENTRAL UNSOLVED QUESTION

**Why do the NVFP4 MoE GEMV kernels top out at 169–177 GB/s when the BF16/FP8 dense GEMVs on
the same device reach 205–254 GB/s?**

We have tested and *falsified* every structural hypothesis available from first principles:

| hypothesis | experiment | result |
|---|---|---|
| warp-starved (320 of 960 warps at decode) | gate and up on separate warps, 320→640 | **LOST 1.3 %** |
| activation-load-bound | stage `x` in shared memory, K-tiled | **LOST 1.8×** |
| too few outputs per warp | N-block 2 and 4 outputs/warp | **exactly NEUTRAL** |
| K-parallelism | `grid.z` K-split, 320→960 warps, deterministic combine | **NEUTRAL** (27.44 vs 27.40) |

It is also not compute: ~2 FMA per weight byte against ~6× headroom. `ncu` hardware counters
would settle it but return `ERR_NVGPUCTRPERM` at our privilege level.

**What we want:** the actual limiter for register-dequant 4-bit grouped GEMV on Blackwell-class
hardware, with evidence. Candidate directions we want checked against real kernels and real
numbers:
* Does the **two-stream** access pattern (codes + per-16-group E4M3 scales) cost sector
  efficiency or LSU issue slots? What layouts do production kernels use — interleaved scales
  inside the code block, scales in a separate pass, scales promoted to shared/constant?
* Is there an **LSU / address-generation** throughput ceiling that a 1-byte-per-lane scale load
  every 16 code bytes runs into on a 20-SM part?
* What do **Marlin / Machete (vLLM), Bitblas, EXL3, llama.cpp `MUL_MAT_ID`, TensorRT-LLM's
  NVFP4 MoE, FlashInfer, TileLang, sglang's FusedMoE** actually achieve as a *fraction of
  achievable bandwidth* for grouped 4-bit GEMV at batch 1, on any hardware? A published
  "% of roofline" for this exact kernel shape is the single most valuable number this
  research can return.
* Blackwell-specific: `cvt.e2m1x2`, `cvt.e4m3x2`, `prmt`/`lop3` dequant tricks,
  `ld.global.nc.L2::128B`, `createpolicy`/`discard`, `cp.async.bulk`. Which of these matter for
  a *memory*-bound GEMV rather than a compute-bound GEMM?
* Is 70 % of achievable simply **what this kernel shape costs** on a narrow part, and the
  remaining 30 % is not recoverable? A credible negative answer is a valuable result — say so
  plainly if that is what the evidence shows.

## 5. The second unsolved question: MoE × speculation

Our DFlash speculation is **correct** (greedy-exact output, τ = 4.32, inside poolside's
published 4.02–6.44 band) but only pays **1.07×**, where the byte budget predicts 1.48×.

Cause, measured: verifying k+1 tokens routes (k+1)·10 assignments, and the MoE pays for the
**distinct** experts. `E_frac` (fraction of the 256 experts touched by one forward):

| shape | E_frac | `k_moe_gateup_rp` |
|---|---:|---:|
| decode M=1 | 0.039 | 200 µs/layer |
| verify M=5 (k=4) | 0.141 | **815 µs/layer** |
| verify M=16 (k=15) | 0.264 | — |

So the verify forward costs ~3× a decode step, and speculation only wins when τ > ~3. τ is
content-dependent — we measure 3.06 on a math prompt and **2.26 on open prose** — so a fixed k
loses 25 % on prose. We now run a bandit over {AR, k=2, k=3, k=5} choosing by measured
throughput, giving 33.6 tok/s on code and 23–30 on prose.

**What we want:**
* Published work on the **MoE-expert-union problem in speculative decoding**. Anything that
  reduces the distinct-expert count of a verify block: routing-aware draft truncation,
  expert-aligned draft selection, reordering the block so accepted-prefix experts dominate,
  partial verification, expert prefetch overlapping the draft forward, or accepting a
  *bounded* expert budget per verify.
* Does any production stack (vLLM, SGLang, TensorRT-LLM) special-case speculative verify for
  MoE targets? What do they measure?
* **Is there a drafting scheme whose verify block reuses the same experts?** E.g. drafting
  conditioned on the target's router state, or a draft that predicts expert IDs alongside
  tokens.
* Empirical acceptance-length data for **block-diffusion / DFlash-class** drafters by workload
  (code vs prose vs math vs agentic tool-calling), to calibrate what τ we should expect.
* Alternatives that sidestep the union problem entirely: **prompt-lookup / n-gram drafting**,
  **retrieval-based drafting**, **self-speculation with layer skipping**, and their measured
  acceptance on code and agentic workloads. Note we have already rejected tree/multi-candidate
  verify (expert union grows per branch) and drafter swaps to EAGLE-3/Medusa/MTP (τ downgrades
  vs block diffusion) — **argue against those rejections only with measurements**.

## 6. The third question: the BF16 remainder

39 % of every decode token is attention weights, currently **FP8 e4m3 weight-only with a
per-output-row scale** (near-lossless, and worth +8.1 % when we built it). `lm_head` is another
9 % at BF16. The shared expert and layer-0 dense are 15 % at BF16.

Prior research (ours) established: a controlled A/B on DeepSeek-R1 with an identical recipe
measured **98.81 % → 96.52 % recovery** when attention was included in 4-bit quantization;
Kimi-K2, DeepSeek-R1-0528 and both Llama-4 models all exclude attention; NVIDIA's most
aggressive NVFP4 variant adds only `o_proj`. Damage scales *down* with active parameters, and
Laguna is 8.5 B active — the risky end.

**What we want:**
* Current (2025–2026) evidence on **4-bit attention weights in MoE models at small active
  parameter counts**, especially anything measuring AIME / GPQA-Diamond / BBH rather than
  OpenLLM v1 (every recipe scores 98–100 % on v1, so it does not discriminate).
* `o_proj`-only NVFP4: how much quality, how much speed, and does the **unbounded softplus
  attention gate** in Laguna make `o_proj` riskier than in the models NVIDIA validated?
* **`lm_head` quantization** — 0.617 GB/token, 9 % of the budget, and it is the one tensor
  whose error goes straight into the sampled token. What do production stacks do?
* **Sub-FP8 KV on the 12 global layers only.** Global KV is 24 576 B/token and 100 % of the
  context-scaling cost; the 36 sliding layers are a constant 37.7 MB. What is the measured
  quality of 4-bit KV *on long-range layers specifically*, and are there published per-layer
  KV precision schedules?

## 7. The fourth question: platform levers we may be leaving on the table

We run at **120 W standard nvpmodel, deliberately not MAXN**. Beyond that:
* **EMC / memory clock pinning** — we estimated +3–8 % but it needs root. What is the actual
  measured DRAM-clock behaviour on Thor under a sustained bandwidth-bound load? Does the
  memory controller downclock during a 40 ms kernel chain, and does `jetson_clocks` /
  `nvpmodel` / `/sys/kernel/debug/bpmp/debug/clk/emc` pinning measurably change GEMV
  throughput?
* **L2 persistence** (`cudaAccessPolicyWindow`, 24 MB persisting). Our sliding-window KV is
  1.05 MB/layer, 37.7 MB total — just over L2. We measured L2 persistence as null once. Is
  there a published methodology for making it pay on a repeated-weight-stream workload?
* **Unified-memory specifics on an integrated part**: does `cudaMemAdvise`,
  `cudaMemPrefetchAsync`, hugepages, or NUMA/interleave configuration change achieved GEMV
  bandwidth on Thor/Orin-class hardware? Is 254 GB/s of 273 actually the ceiling, or do
  published Thor/Orin microbenchmarks reach higher with a different access pattern (e.g.
  wider vectors, more outstanding loads per thread, `__ldg` vs `__ldcs` vs `ld.global.nc`)?
* Anything Thor-specific published since its release: SM architecture details, LSU width,
  L2 sectoring, DRAM burst behaviour, and any inference numbers for large MoE models on it.

## 8. What is already ruled out — do not return these

Each cost us a measurement. Re-propose one **only** with a number that contradicts ours.

* **tcgen05 / TMEM at small M** — block-scaled NVFP4 locks MMA_M ∈ {128,256}; crossover M≈900,
  we are at M≤6.
* **FP8/FP4 draft** — collapsed a comparable τ 13.33 → 11.14. The BF16 draft is the moat.
* **Tree / multi-candidate verify** — expert union grows per branch.
* **Drafter swap to EAGLE-3 / Medusa / MTP** — τ downgrades vs block diffusion.
* **PowerInfer / hot-cold expert tiering** — needs a two-tier bandwidth hierarchy; Thor is one
  unified pool.
* **Activation sparsity (TEAL / CATS)** — measured magnitude-dense (~0 % hard zeros), and
  inherently lossy, which breaks a bit-exact verify.
* **cp.async pipelining on a max-grid-fill GEMV** — block parallelism already hides latency.
* **Grid-cooperative megakernel with `grid.sync()`** — ~35 % of token time in the barrier.
* **DSMEM clusters** (Thor caps `cta_group` at 2), **split-K / stream-K**, **TMA**.
* **Zero-copy mapped weights** — 160 vs 227 GB/s.
* **FlashNorm weight folding** — not bit-exact in our pipeline, and it targets a
  latency-bound kernel.
* **Shared-memory `x` staging, N-blocking, MoE K-split, gate/up warp split** — our four
  negatives above.

## 9. Deliverable

For each axis, return:
1. **The claim**, in one sentence.
2. **The evidence** — paper/repo/file/line, hardware, measured number.
3. **Transfer analysis** — does it hold at 20 SMs, 254 GB/s, M=1–6, 256-expert top-10,
   `moe_intermediate` 1024? What breaks it?
4. **Expected value for us** — % of decode step, with the arithmetic shown against the byte
   table in §3.
5. **Cost and risk** — implementation size, and whether it preserves bit-exact greedy output
   (our verify path requires it; see `LOOP_LOG.md` Gate B1c).

Rank everything by expected value ÷ cost. **Call out explicitly anything that is a dead end,
and say why** — a well-supported negative saves us a week.
