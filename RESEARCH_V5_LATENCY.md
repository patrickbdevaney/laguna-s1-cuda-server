# RESEARCH_V5_LATENCY.md — the 20.1%, priced in microseconds

Axis: `attn_core` + `norm+cast` + `router` — the part of the step that moves ~1.3% of the bytes.
Research pass only; nothing was run on the GPU (it was busy). Every on-device number below is
re-derived from `prof_production.log`, `MODEL_INVENTORY.md`, `kernels/*.cu` and the V1–V4 ledger.

---

## 0. Headline: the 20.1% is not 20.1%. It is ~9.4%, and `norm+cast` is ~0.

This is the one finding in this document that changes what you should do next, so it goes first.

`REPRICED.md` states the caveat itself and then the number escaped it: *"`LG_PROF` serialises and
disables graph capture, so the total (43.2 ms) is inflated against the real 30.3 ms step and
**per-launch-heavy categories are over-represented**."* V5's prompt then carries `20.1%` forward as
a measured share of the shipping step. It is not. It is a share of a **42-millisecond** step.
`prof_production.log` line 3 says so directly: **DECODE median 23.78 tok/s = 42.05 ms/step**, and
line 5: **effective BW 148.7 GB/s = 65% of ceiling**, against production's 206 GB/s = 91%.

### The correction, with the arithmetic

The `launches` column is **timing regions**, not kernels (`attn_qkvo_gemm` = 96 = 2 regions/layer,
but that category contains q, k, v, g and o — five GEMMs per layer). Regions total
1+96+96+48+48+47+47+1 = **384**. The profiled total is 43.182 ms; the shipping step is 30.3 ms.
A uniform per-region instrumentation cost δ that reconciles them:

```
δ = (43.182 − 30.3) ms / 384 regions = 33.5 µs/region
```

That number is not assumed — it is **independently confirmed by `norm+cast`**, whose true cost is
computable from bytes and therefore known:

> `add_rms_cast` at hidden = 3072 reads h (12 KB fp32) + residual (12 KB) + weight (6 KB bf16) and
> writes hb (6 KB bf16) + h (12 KB) ≈ **48 KB**. At 227 GB/s that is **0.21 µs**. Grid is 12 blocks
> of 256. Even charging a full in-graph launch and zero overlap, the kernel cannot cost more than
> ~2.5 µs. But the profile charges it **2.924 ms / 96 = 30.5 µs per region**.

So `norm+cast`'s entire profiled cost, to within 3 µs, *is* the instrumentation. δ measured two
independent ways agrees: **30.5 µs (from norm+cast's byte floor) vs 33.5 µs (from the totals)**.

Applying δ = 30.6 µs (the value that also reproduces the known 206 GB/s effective bandwidth):

| category | profiled ms | regions | −δ·N | **corrected ms** | **% of 30.3 ms** | implied GB/s | cross-check |
|---|---:|---:|---:|---:|---:|---:|---|
| `attn_qkvo_gemm` | 14.973 | 96 | 2.938 | **12.04** | **39.7** | 233 | at ceiling ✓ |
| `moe_experts` | 13.501 | 47 | 1.438 | **12.06** | **39.8** | 207 | **matches the known 206 GB/s exactly** ✓ |
| `shared+dense` | 4.667 | 48 | 1.469 | **3.20** | **10.6** | 174 | narrow GEMVs, plausible ✓ |
| `attn_core` | 3.167 | 48 | 1.469 | **1.70** | **5.6** | — | latency-bound |
| `lm_head` | 1.304 | 1 | 0.031 | **1.27** | **4.2** | 241 | one big GEMM ✓ |
| `router` | 2.603 | 47 | 1.438 | **1.17** | **3.8** | — | see §4 |
| `norm+cast` | 2.924 | 96 | 2.938 | **≈0.0** (floor 0.2–0.35) | **0.7–1.2** | — | byte floor 0.21 µs/call |
| `embed+rope` | 0.042 | 1 | 0.031 | 0.01 | 0.0 | — | |
| **sum** | 43.18 | 384 | 11.75 | **31.4** | | | 3.8% over 30.3 — within drift |

**The latency-bound trio is 1.70 + 1.17 + ~0.3 = 3.17 ms = 10.5% of the shipping step, not 20.1%.**
Take the most generous reading of the residual and it is still under 13%. The prize is **half the
advertised size**, and its composition is completely different from what the prompt assumes:

* `norm+cast` — **the entire category is a profiling artifact.** Real cost ≤ 0.35 ms (1.2%),
  and it is already the *fused* add+RMSNorm+cast kernel built in V1 (+2.4%). §5 kills the
  RMSNorm-prologue-fusion line of research on arithmetic alone.
* `attn_core` — **1.70 ms (5.6%) is real** and is the single largest latency-bound term. §6.
* `router` — **1.17 ms (3.8%) is real**, and §4 shows it is an *occupancy* problem, not a top-k
  problem.

Two side effects of the correction worth noting: **`attn_qkvo_gemm` is 39.7% of the step, not
34.7%** — which makes `o_proj → NVFP4` worth *more* than `REPRICED.md`'s +9.2%, not less — and
`lm_head` is 4.2%, not 3.0%.

**Cheapest falsifying experiment (do this before anything else in this document).** Do not trust
δ; measure the categories directly *in the shipping graph-captured config* by the doubling
ablation, which needs no profiler and no new kernels: run the category's kernels **twice** per
layer, second call writing to a scratch buffer, capture the graph, take median-of-N tok/s. The
delta *is* the marginal cost of the category with graphs on and overlap intact. Three A/B runs,
under an hour, zero risk. If `norm+cast`-doubled costs ≥ 2.9 ms of extra step time, this entire
section is wrong and the megakernel case is back on. **Predicted: +0.25 ms (norm), +1.7 ms
(attn_core), +1.2 ms (router).**

---

## 1. Q1 — persistent/megakernel decoders. Confirmed dead, now with the published decomposition.

`RESEARCH_FINDINGS.md` already killed this ("in-graph launch residual ~0.7 µs × 1665 ≈ 1.2 ms =
2.1% of the step — that is the hard ceiling on every launch-count lever"). The literature now
supplies the decomposition that explains *why* the published 1.5–3.5× wins cannot transfer here.

### Who has shipped it, and what it bought

| system | hardware | measured | notes |
|---|---|---|---|
| **Mirage MPK** ([arXiv:2512.22219](https://arxiv.org/abs/2512.22219), [CMU](https://catalyst.cs.cmu.edu/projects/mpk.html)) | A100 **108 SM**, H100 132, B200 148 | Qwen3-8B decode **14.5 → 12.5 ms/token (1.16×)**; 1.0–1.7× vs vLLM/SGLang | §6.6: 293 launches/token; **eager 3.8 µs/launch = 1.1 ms, with CUDA graphs 0.8 µs/launch = 0.2 ms (1.4% of step)**. MPK's own scheduler overhead 0.28% |
| **Hazy Research "Look Ma, No Bubbles"** ([blog](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)) | H100 **132 SM**, B200 148 | Llama-3.2-1B: H100 <1 ms/forward, **78% of HBM BW**, 2.5× vs vLLM | B200 600 µs breakdown: activation store/load 250, RMSNorm+matvec 200, weight wait 30, **sync 40, setup 80** |
| **Ada-MK** ([arXiv:2605.11581](https://arxiv.org/html/2605.11581v1)) | L20 (Ada, **92 SM**) | Qwen3-1.7B BS=1: **+23.6% vs TRT-LLM, +50.2% vs vLLM** | justified by "launch overhead ≈14.6% of e2e" — that is *eager* TRT-LLM, not graph-replay. **No ablation separating the three levers; authors say so explicitly** |
| **FlashDMoE** ([arXiv:2506.04667](https://arxiv.org/html/2506.04667v2)) | 8× H100 | 6.4× latency, 93% SM util vs 9.7–59% | ⚠ **4K–16K-token prefill, multi-GPU. Does not transfer.** |
| **Fleet** ([arXiv:2604.15379](https://arxiv.org/abs/2604.15379)) | AMD MI350 (chiplet) | Qwen3-8B batch 1–8: **1.3–1.5×** | win is per-chiplet L2-aware scheduling; no analogue on a monolithic 20-SM part |
| **FlashInfer persistent / POD-Attention** ([ASPLOS'25](https://akkamath.github.io/files/ASPLOS25_POD.pdf)) | A100/H100 | up to 59% on attention | ⚠ **multi-request continuous batching. Does not transfer.** |
| **TileLang** | — | **no megakernel claims found** | tile-authoring DSL; described as orthogonal |

### The critical sub-question: nobody has done this on ≤32 SMs

**Explicit negative result.** No published megakernel or whole-step persistent-kernel result exists
on any GPU with ≤32 SMs — no Jetson Orin, no Thor, no embedded part. Every system above is ≥92 SMs.
Anything you build here is unprecedented, not a reproduction.

### The decomposition, and why each term is ~0 for us

Both groups decompose the win into (a) launch overhead, (b) memory pipelining across former kernel
boundaries, (c) tail/wave-quantization. Priced against this device:

* **(a) launch overhead — ceiling +1.8%.** ~770 in-graph kernels × 0.7 µs residual = 0.54 ms of
  30.3 ms. MPK's own numbers say CUDA graphs already capture ~82% of the removable launch cost
  (1.1 ms → 0.2 ms); we already have graphs. Corroborated externally: **"Memory-Bound but Not
  Bandwidth-Limited"** ([arXiv:2605.30571](https://arxiv.org/abs/2605.30571)) A/B-tests CUDA graphs
  on/off across H100 / A100 / L40S / L4 and finds the graph win is **1.259× on H100 but only 1.028×
  on L4** — *"launch-side overhead becomes visible on fast GPUs but remains mostly hidden on
  slower, bandwidth-bound GPUs."* Thor is slower and smaller than their L4 datapoint. The trend
  lands on ≤2.8%, and our own null-kernel sweep already put it at 1.8–4.1%.
* **(c) tail / wave quantization — measured at 0 on this device.** `RESEARCH_FINDINGS_V3.md`:
  bandwidth is flat from 240 blocks to 20 000 on our 20 SMs. Hazy Research names this as a large
  contributor precisely because *"up to 80 of 132 SMs idle during kernel tails"* — a big-GPU
  artifact by construction. We have no spare-SM pool to reclaim. **Term = 0.**
* **(b) cross-boundary memory pipelining — the only live term, and it is capped by our own
  measurement.** This is what MPK (1.2–1.3× from preloading the next chunk while the current task
  runs) and Hazy Research both name as the residual win once graphs are in play. But we are at
  **91% of the 227 GB/s ceiling for the 79% of the step that moves bytes**. The maximum recoverable
  from perfect cross-boundary overlap on the byte-moving portion is 9% of 79% = **7.1%**, and that
  assumes the remaining 9% gap is *entirely* pipeline-fill/drain rather than access pattern.

**Verdict: unchanged, DEAD, with a firmer number.** Ceiling ≈ 1.8% (a) + 0% (c) + an unproven
fraction of ≤7.1% (b), against weeks of work, a global register allocation across all stages that
would undo the TM=1 MoE register win (50 vs 127 registers) and the lane-per-output repack, and no
precedent on ≤32-SM hardware anywhere.

**One thing the ledger got wrong, though**, and it is worth correcting: `OPTIMIZATION_LOG.md` lists
**TMA** as dead because "none are bandwidth levers". That is the right verdict on the wrong axis —
TMA/`cp.async.bulk` is the *pipelining* primitive that makes term (b) mechanically possible
(async weight fetch for stage N+1 into shared-memory pages freed by stage N). It was never
evaluated as a latency lever. This does not resurrect the megakernel; it does mean the recorded
reason for rejecting TMA is not the reason that matters.

---

## 2. Q2 — is `grid.sync()` necessary? No, and the measured alternatives are known.

Our recorded blocker (35% of token time in the barrier) is **independently reproduced**, on a
*much larger* GPU:

* **[Alpin, RTX 5090 decode optimization](https://blog.alpindale.net/posts/5090_decode_optimization/)** (Blackwell consumer, **170 SM**):
  *"Every per-layer `grid.sync()` costs ~3 µs, and with 6 barriers across 28 layers, that's ~500 µs
  of pure synchronization overhead per step."* Per-layer barriers+overhead = **9.5 µs / 28.3 µs =
  33.6%** — our 35%, on 8.5× the SM count. Their hand-rolled `AtomicGridSync` was **2.2 µs**, only
  27% cheaper; a lightweight flag-based barrier saved ~2 µs/barrier.
* **[Benchmarking Thread Block Cluster (TUHH)](https://tore.tuhh.de/dspace-cris-server/api/core/bitstreams/f19e2bfe-f059-499b-b473-d9f00f78d01d/content)**, H100, CUDA 12.3, cycles:
  `__syncthreads()` **14**; `cluster.sync()` **~1100–1300**; **`grid.sync()` ~2200–2800** — grid.sync
  is **157×** an intra-block barrier and cluster.sync is only 41% cheaper. ⚠ their grid sweep only
  went 1→16 blocks, so this is *not* evidence that grid.sync scales well.

### The measured alternatives

| primitive | measured | hardware | verdict here |
|---|---|---|---|
| `grid.sync()` | 2200–2800 cy; ~3 µs real | H100 114 SM / RTX 5090 170 SM | dead, reproduced |
| custom atomic grid barrier | **2.2 µs** | RTX 5090 | 27% better than dead |
| `cluster.sync()` | 1100–2100 cy | H100 | **`cta_group` caps at 2 on Thor** (our own measurement, `OPTIMIZATION_LOG.md`) → a 2-CTA barrier cannot serve a 20-SM grid |
| DSMEM inter-SM access | **181 cy** (cluster 2) vs L2 near-hit 258 cy, global 656 cy ([Luo et al., arXiv:2501.12084](https://arxiv.org/html/2501.12084v1)) | H800 | 30% better than L2, but only across 2 CTAs |
| **global-counter semaphore (MPK / Hazy)** | MPK: **0.28% of total runtime**; Hazy: async barrier ~60 ns even when passed, **40 µs of a 600 µs B200 step (6.7%)** | B200 / H100 | **the answer**: ~8× cheaper than grid.sync (our own V1 estimate: 0.68 ms over ~800 edges) — and still not worth 0.54 ms of prize |
| **PDL** (`cudaTriggerProgrammaticLaunchCompletion`) | ⚠ **zero overlap observed on GB10** (DGX Spark, small integrated Blackwell) despite correct API use — [NVIDIA forum #359590](https://forums.developer.nvidia.com/t/issue-with-programmatic-dependent-launch-pdl-no-overlapping-execution-observed-on-gb10/359590). NVIDIA's own docs: *"opportunistic and not guaranteed… reliance on concurrent execution in this manner is unsafe and can lead to deadlock."* Hazy Research rejected it as *"very coarse, still introduces unnecessary stalls."* | GB10 = closest small-Blackwell proxy to Thor | **do not build on this.** Works in graphs via `cudaGraphDependencyTypeProgrammatic` (CUDA 12.3+), but the one small-Blackwell datapoint is a null result |
| global flag poll roundtrip | L2 273 cy / global 659 cy (GH100); **L2 358 cy / global 877 cy (GB203, consumer Blackwell)** ([Jarmusch et al., arXiv:2507.10789](https://arxiv.org/pdf/2507.10789)) | GB203 is the best small-die Blackwell proxy | ~0.2–0.5 µs per dependency edge |
| CUDA graph per-node cost | **~2.5 µs + ~1 ns/node** in CUDA 12.6 (was 2 µs + 200 ns/node in 11.8) — [NVIDIA blog](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-graphs-and-other-performance-enhancements) | discrete Ampere | **no Jetson graph-launch number exists publicly.** Our 1.6–1.7 µs marginal is the only datapoint on this hardware class anywhere |
| `__nanosleep` backoff vs busy poll | **no published microbenchmark found on any architecture** | — | genuine literature gap |

**Answer to Q2: `grid.sync()` is not necessary — the global-atomic-counter/semaphore pattern
(MPK's circular task/event queues with `atomicAdd`; Hazy's zero-initialised counter array with
paged 16 KiB shared memory) replaces it at ~8× lower cost and with no cooperative-launch occupancy
restriction.** That restriction is the real reason grid.sync is catastrophic at 20 SMs: a
cooperative launch requires every block resident simultaneously, so on 20 SMs the legal grid is a
tiny fraction of the parallelism a normal multi-wave kernel uses — you pay in lost parallelism
*before* you pay for the barrier. But removing the blocker does not create a prize: §1 prices the
whole megakernel at ≤1.8% for (a) and 0% for (c).

---

## 3. Q6 — `sm_110a` feature inventory, and a contradiction in our own ledger

**Flagging an unresolved contradiction inside this repository.** `HARDWARE.md` line 38 states,
inherited verbatim from the gemma constitution and never re-derived here: **"TMEM + tcgen05 + TMA
present."** The V5 prompt states **"tcgen05 is ABSENT."** A community post on the
[Jetson Thor benchmarking thread](https://forums.developer.nvidia.com/t/performance-benchmarking-on-jetson-thor/349494)
also claims *"Thor's 20 SMs with tcgen05+tmem beat the Spark's 48 SMs on tensor core GEMM."*
These cannot all be true. It is a **20-minute `ptxas -arch=sm_110a` compile probe** to settle, and
it gates an entire family of techniques. Do it.

What the public record supports:

* **Identity**: `sm_110a`, CC **11.0**. Was `sm_101/101a` under CUDA 12.8/12.9, **renamed to sm_110
  in CUDA 13.0** ([vLLM #26791](https://github.com/vllm-project/vllm/issues/26791)). NVIDIA's own
  public docs — CUDA C Programming Guide compute-capability appendix and the
  [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) —
  **document only CC 10.0 and 12.0. CC 11.0 is absent from NVIDIA's public documentation.**
* **Lineage → sm_100, not sm_120.** Consumer Blackwell (sm_120/121, RTX 50 / GB10) has **cluster
  size 1 only, no tcgen05, no TMEM, 99–128 KB smem/SM**. Datacenter sm_100 has clusters to 16,
  tcgen05, **228 KB smem/SM**. **Our measured 228 KB/SM matches sm_100 exactly.** That is the
  strongest single piece of evidence, and it is ours.
* **Clusters/DSMEM**: present but **`cta_group` caps at 2 on this device** (our measurement). No
  public microbenchmark of clusters on sm_110a exists. Blackwell portable cluster max is 8; size-16
  is B200-only opt-in. On 20 SMs a size-16 cluster would consume 80% of the machine, and the TUHH
  paper measures up to **22% execution-time overhead** at cluster size 16 from GPC fragmentation.
* **TMA / `cp.async.bulk`**: [CCCL documents `cp.async.bulk` for SM_90a, SM_100a **and SM_110a**](https://nvidia.github.io/cccl/unstable/libcudacxx/ptx/instructions/cp_async_bulk.html)
  — PTX-legal on this target. Worth noting: PTX's `.scale-vectorsize` MXFP4 modifier is documented
  valid for `sm_100a/100f/110f` but **not `sm_110a`**, which is a concrete instance of Thor being
  feature-swept relative to full sm_100.
* **FP4**: NVIDIA claims 2070 FP4 TFLOPS theoretical for Thor; community CUTLASS runs report
  **>800 TFLOPS FP4 achieved** but only **0.67 TFLOPS FP8 sparse on 8192³**
  ([forum](https://forums.developer.nvidia.com/t/jetson-thor-cutlass-fp4-fp8-fp16-performance-test/372014)) —
  i.e. the FP4 tensor path executes, and CUTLASS for this target is immature.
* **L2 persistence** (`cudaAccessPolicyWindow`, CC 8.0+): generic wins elsewhere are **1.096–1.16×**
  on the persistent working set. **Relevant to §6**: our entire KV working set at pos 93 is
  **9.1 MB, which fits in the 32 MB L2**, so this is not the lever for `attn_core` — the data is
  already resident.
* **CUDA 13.0 on Thor** is the first release with **true hardware-coherent UVM**: the GPU can access
  pageable host memory through host page tables, **cached in GPU L2**
  ([NVIDIA blog](https://developer.nvidia.com/blog/whats-new-in-cuda-toolkit-13-0-for-jetson-thor-unified-arm-ecosystem-and-more/)).
  ⚠ This directly touches our recorded fact that *"`cudaMallocManaged` is not GPU-L2-cached on Thor"* —
  that was inherited from the **gemma/Orin-era** constitution. On CUDA 13.0 + Thor it may no longer
  hold. Cheap to re-probe, and it gates the zero-copy result (160 vs 227 GB/s).

---

## 4. Q3 — the router. 3.8%, and it is an occupancy problem, not a top-k problem.

**Our top-k is not slow. Our router GEMV is warp-starved.**

### What the router actually costs

Corrected: **1.17 ms / 47 layers = 24.8 µs per layer**, in two kernels:

1. `gemm_bf16(rlogit, w.router, hb, M=1, N=256, K=3072)` — reads **256 × 3072 × 2 B = 1.573 MB**.
   Floor at 227 GB/s = **6.9 µs**.
2. `k_router<<<1, 256, 2·256·4 B>>>` — one block, 256 threads, sigmoid + softcap + 10 sequential
   warp-argmax passes.

### Why the GEMV is the problem

`k_gemm_bf16` gives **one warp one output row**. At N = 256 that is **256 warps total**. This part
holds 20 SMs × 48 warps = **960 resident warps**, so the router GEMV runs at **26.7% warp
occupancy** — and the file's own comment says these GEMVs are *"issue/latency-bound, not
bandwidth-bound… resident warps are the throughput knob."* Concurrency check: sustaining 227 GB/s
at ~880 cycles of Blackwell global latency (0.59 µs at 1.5 GHz) needs **133 KB in flight**; 256
warps × 32 lanes × 16 B (one `uint4` per lane) = **131 KB** — exactly one outstanding load per lane,
i.e. the kernel is balanced on a knife edge and any dependency stall halves it. That predicts
**~14 µs**, and 24.8 − 14 ≈ 11 µs is left for `k_router` plus two launches.

**This generalises into a rule for this device**: a GEMV's parallelism *is* N. Sorted by starvation:
`g_proj` N=48/72 → **5–7% occupancy** (already flagged in V1: "3.34 µs for 0.44 MB, 72 blocks onto
20 SMs"); `router` N=256 → 27%; `k/v_proj` N=1024 → 107%; everything else ≥ 100%. **The router is
the second most warp-starved GEMV in the model.** Fix: split-K over K = 3072 into 4 → 1024 warps.

### Is the top-k itself slow? No.

`k_router` does 10 sequential argmax passes; each lane scans 8 of 256 shared floats, then a 5-step
shuffle butterfly + broadcast + `__syncwarp`, then `ss[best] = −1e30`. Dependent chain ≈ 10 ×
(8 smem loads + 5 shuffles @ ~30 cy + broadcast + syncwarp) ≈ **3500 cycles ≈ 2.3 µs**, plus the
sigmoid pass. The code comment records that the *serial* version cost 27 µs/launch and the warp
version fixed it. **~2–5 µs for a 256-way sigmoid + top-10 is at or near the published floor.**

Against the literature:

* **vLLM on GB200/GB300, DeepSeek-V3.2** ([blog](https://vllm.ai/blog/2026-05-11-vllm-tops-artificial-analysis)):
  *"At low batch sizes DeepSeek V3.2 was bound by GPU kernel launch overhead, not compute. Each
  transformer layer was issuing dozens of separate kernels… each carrying a fixed launch cost that
  dominated."* They cut **~33 → ~10 kernels/layer**, and a **specialised small-batch router GEMM
  kernel bought a further 6% at batch = 1**. ⚠ Datacenter Blackwell, but the diagnosis and the fix
  are ours exactly — and note their fix was the **GEMM**, not the top-k.
* **SGLang** ships `moe_fused_gate.cu` **templated for exactly 256 experts / 8 groups**, and has
  since consolidated onto a single Triton router (#26771). **LMDeploy**'s equivalent fusion:
  **>30% on the isolated operator, 1.85–3.00% end-to-end.** ⚠ big-GPU serving.
* Academic top-k (**RadiK**, **Dr. Top-k** at n≈2²⁰, **RTop-K** [arXiv:2409.00822](https://arxiv.org/pdf/2409.00822)
  on A6000) all target large n or large batch. **None are applicable to one row of 256 floats.**

### Proposal R1 — split-K router GEMV with last-block fused sigmoid+top-k, one launch

Split K = 3072 into S = 4 chunks (1024 warps, 107% occupancy). The last block to finish — detected
with the standard `__threadfence()` + `atomicInc` ticket pattern, **not** `grid.sync`, so no
cooperative launch and no occupancy restriction — sums the S fixed partials in a fixed order,
applies sigmoid + bias, and runs the existing warp top-10. Determinism is preserved because the
partition and the summation order are fixed, exactly as `attend_nsplit`'s comment demands.

* **µs removed**: 24.8 → (1.573 MB / 200 GB/s = 7.9) + (2.3 top-k) + (1.6 launch) ≈ **11.8 µs/layer**.
  13 µs × 47 = **0.61 ms = 2.0% of the step → ≈ +2.1% tok/s.**
* Also deletes 47 launches (−0.08 ms) and the 1 KB logits round-trip.
* **Exact**: yes, if the split is sized from K alone and combined in fixed order (same discipline as
  `attend_nsplit`). **Retraining**: no.
* **Cheapest falsification**: before writing any kernel, run the router GEMV alone in
  `bench_kernels` at N=256, K=3072 with S = 1, 2, 4, 8 fixed partials. If S=4 is not ≥1.7× S=1, the
  occupancy diagnosis is wrong and R1 is worth nothing. **~1 hour.**

### Non-proposal R2 — router weights to FP8

The router `gate.weight` is **still bf16** while everything around it went FP8: 47 × 256 × 3072 × 2 B
= **73.9 MB = 1.18% of `B_tok`** (this is the "1.2% of bytes" line). FP8 halves it → **+0.6%**.
**Not exact**: an FP8 perturbation of the logits can flip a near-tie in the top-10, changing which
experts run. Cheap but it must clear the greedy gate for a 0.6% return. **Rank: low.**

---

## 5. Q4 — RMSNorm fusion into the consumer GEMV prologue. Refuted on arithmetic. Do not build it.

The prize is not 6.8% (2.06 ms). It is **at most 0.35 ms (1.2%)**, and prologue fusion recovers at
most part of that. §0 gives the byte arithmetic: 48 KB per call, 0.21 µs at 227 GB/s, 96 calls,
and the kernel is *already* the fused add+RMSNorm+cast (built in V1, +2.4%, bit-exact).

The literature is real and the technique is sound — it is the *size of our term* that kills it:

* **"Towards Free Normalization: Fusing Normalization into GEMM and Attention Kernels"**
  ([PyTorch blog, 2026](https://pytorch.org/blog/towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/)),
  **B200, bf16, 750 W**. "Lazy Pre-Norm" exploits `(A · rstd) @ B = (A @ B) · rstd`: the CTA already
  scans the full row of A for the GEMM, so the sum-of-squares is free in the K-loop and the rescale
  is deferred to the epilogue. **~17% at K/N=64, ~32% at K/N=128, vanishing above K/N≈128.** They
  cite un-fused norm at ~20% of latency for one production model, ~10% "typical LLM".
  ⚠ **Large-M training regime; no batch-1 numbers.** Hard cap of ~512 N per CTA on Blackwell,
  RMSNorm-only, no elementwise-affine.
* **vLLM** ships `fused_add_rms_norm` and a `RMSNormQuantFusionPass` ([PR #10906](https://github.com/vllm-project/vllm/pull/10906))
  — **norm+quant, stopping short of folding into the GEMM**. We already have the equivalent.
* **TensorRT-LLM** ships AllReduce+Residual+Norm+Quant fusion and explicitly recommends it **for
  small batch where decode dominates** — the right regime, but no isolated µs published, and the
  AllReduce term does not exist for us (one device).
* **TokenWeave**: 1.40× — communication-bound, multi-GPU. ⚠ does not transfer.

**Arithmetic**: even a *perfect* fusion that makes all 96 norm calls literally free removes
**≤0.35 ms = 1.2% → ≤+1.2% tok/s**, at the cost of a custom non-CUTLASS prologue in five different
GEMV kernels, each of which would then need re-gating for bit-exactness. **Payoff/cost is the worst
in this document.** The one thing worth keeping from the PyTorch work is the identity itself: it is
bit-exact-adjacent and free *if* someone is already rewriting a GEMV for another reason (e.g. R1).

**Cheapest falsification**: the norm-doubling ablation in §0. If doubling `add_rms_cast` costs
≥ 2 ms of step, this section is wrong.

---

## 6. Q5 — `attn_core`. 1.70 ms (5.6%) is real, and it is the best remaining latency target.

This is now the largest genuinely latency-bound term in the step. **35.4 µs per layer to move
190 KB** (KV at pos 93: 2 × 8 kv-heads × 128 × 93 × 1 B FP8), whose byte floor is **0.84 µs**.
97.6% of it is not bytes. And the whole 48-layer KV working set is **9.1 MB — L2-resident in 32 MB**,
so it is not a DRAM problem either.

### What is actually launched

`attend_nsplit(M=1, nkv=8, len=93)`: `want = 80/8 = 10`, `by_len = ceil(93/64) = 2` → **NSP = 2**.
So the decode path is `k_attn_split<6|9>` + `k_attn_combine` — **2 kernels per layer, 96 total**
(the profile's 48 is regions). Grid = (1, 8, 2) = **16 blocks × NW=4 warps = 64 warps on a
960-warp machine = 6.7% warp occupancy.** The combine is 48 blocks × 128 threads reading ~48 KB.

**Two candidate explanations, both testable, and they are not exclusive:**

1. **Warp starvation (same disease as §4).** 64 of 960 warp slots. The `by_len` heuristic —
   *"≥64 keys per block, else it is all overhead"* — was written for a bandwidth-bound framing.
   At 93 keys it forces NSP=2 when the machine has room for 8–12.
2. **The per-(key, head) `warp_sum` dependency chain.** Each warp strides keys by NW=4 and, for
   each key, loops g = 0..G−1 doing a 4-FMA partial then a **full 5-step shuffle butterfly**, then
   two `__expf` and a rescale of `acc[g][4]`. Per split, per warp: ~47/2/4 ≈ 6 keys × G=9 heads =
   54 reductions. At ~150 cycles per dependent butterfly + ~40 for the exp/rescale, that is
   ~10 000 cycles ≈ **6.7 µs** *if the 9 head-chains do not overlap*. `k_attn_split` templates G
   (6 or 9), so the compiler *can* unroll and interleave the 9 independent chains — `k_attn`
   (the NSP=1 path) does **not** template G, and its `for (int g = 0; g < G; ++g)` is a runtime
   loop the compiler cannot unroll. Worth checking which one the profile actually exercised.

Plus one small certain cost: **`int slot = j % cap;` is an integer modulo in the innermost key
loop** with `cap` a runtime value — ~20+ cycles per key per warp that the compiler cannot strength-
reduce. Make `cap` a power of two and mask, or hoist (j is monotone; the wrap happens at most once
per warp stride).

### What the literature says the floor is

* **[Flash-Decoding](https://pytorch.org/blog/flash-decoding/)** (A100, **108 SM**, batch 1):
  the split path floor is **~56–64 µs and essentially flat from seqlen 4k to 64k** — i.e. at
  moderate context the cost is the split+reduce structure, not the KV volume. ⚠ A100 numbers; the
  transferable part is the *shape* of the curve. Their heuristic: `batch × num_splits ≈ num_SMs`,
  which for us at 20 SMs / 8 kv-heads says **NSP ≈ 2–3** — consistent with what we do.
* **[TensorRT-LLM XQA](https://nvidia.github.io/TensorRT-LLM/blogs/XQA-kernel.html)** (H200):
  2.4× at 1 GPU. NVIDIA's `multi_block_mode` guidance: use it when `batch × num_heads < num_SMs`
  (for us: 1 × 8 < 20 → **yes, split**), and *"there is a minimum number of tokens required for the
  multi-block version to become more efficient than the vanilla single-CTA-per-head
  implementation."* ⚠ throughput/occupancy story on 132 SMs.
* **FlashInfer** explicitly states **split-KV does not improve performance on RTX Ada 6000 / 4090
  "because they have relatively smaller memory bandwidth and stronger CUDA Cores"** — the closest
  published analogue to our position, and it argues the *opposite* of (1) above. Worth knowing
  before spending a day.
* **No published attention-decode kernel timing exists for Jetson Orin/Thor or any ≤32-SM GPU.**
  Our 35.4 µs/layer may be the most granular public datapoint on this hardware class.

### Proposal A1 — NSP sweep (one line, one rebuild)

Change `by_len` from `(len + 63) / 64` to 32 / 24 / 16 and A/B the decode. At len = 93 that gives
NSP = 3 / 4 / 6 → 24 / 32 / 48 blocks → 10% / 13% / 20% warp occupancy.

* **µs removed**: if attention is warp-starved and scales even sub-linearly to 20% occupancy,
  35.4 → ~15 µs/layer removes **0.98 ms = 3.2% → ≈ +3.3% tok/s**. If FlashInfer's small-GPU finding
  governs instead, it removes 0 and may regress.
* **Exact**: ⚠ **no** — a different key partition is a different rounding, and `attend_nsplit`'s
  comment records that this exact class of change flipped an argmax by layer 48 (LOOP_LOG B1c). It
  is *self*-consistent (decode and verify use the same NSP, sized from `len` alone, so speculative
  verify stays bit-identical to decode), but greedy output will not match the current build
  token-for-token and **must be re-gated against the oracle**. **Retraining**: no.
* **Cost**: one line + `gate_attn_split` + the 8/8 greedy gate. **~1 hour.** Highest payoff-per-hour
  in this document after §0.

### Proposal A2 — fuse `k_attn_combine` into `k_attn_split` (threadfence last-block)

Same ticket pattern as R1: last block of the (m, kh) group does the online-softmax combine. Removes
48 launches and the pacc/pml round-trip. **µs removed: 48 × ~1.6 = 0.077 ms = 0.25%.** Exact if the
combine order is fixed by split index rather than arrival order — which is *more* deterministic than
the current arrangement, not less. Cheap, small, do it opportunistically.

### Proposal A3 — hoist the `% cap` modulo

**µs removed**: ~20 cy × ~12 key-iterations × 4 warps... ≈ 1–2 µs/layer = **0.05–0.10 ms (0.2–0.3%)**.
Exact, trivial, 30 minutes. Bundle with A1.

---

## 7. Ranked: (payoff × uncertainty) / cost

Uncertainty is scored as *information value* — a high-uncertainty item earns its rank by resolving
something, not by being risky.

| # | item | payoff | uncertainty | cost | exact? | retrain? |
|---|---|---:|---|---|---|---|
| **1** | **§0 doubling ablation of the three categories under CUDA graphs** | decides whether the remaining 10.5% (not 20.1%) is worth any of the work below; re-ranks the whole project | **very high** — δ is a model, not a measurement | **~1 h**, no new kernels, zero risk | n/a | no |
| **2** | **§6 A1 — NSP sweep (`by_len` 64 → 32/24/16)** | **up to +3.3%** (0.98 ms) | high — FlashInfer's small-GPU finding argues the other way | **~1 h** + greedy re-gate | ⚠ no (re-gate) | no |
| **3** | **§3 — `ptxas -arch=sm_110a` probe: tcgen05 / TMEM / TMA / `cp.async.bulk`** | resolves a direct contradiction between `HARDWARE.md` and the V5 premise; gates a whole family | **very high** | **~20 min** | n/a | no |
| **4** | **§4 R1 — split-K router GEMV + fused last-block sigmoid/top-k** | **+2.1%** (0.61 ms) | medium — occupancy diagnosis is arithmetic, not measured | 1 h to falsify, ~1 d to build | yes | no |
| **5** | §3 — re-probe `cudaMallocManaged` L2-caching on **CUDA 13.0 + Thor** (the recorded fact is Orin-era) | re-opens zero-copy (160 vs 227 GB/s) if it changed | high | ~1 h | n/a | no |
| **6** | §6 A3 — hoist `% cap` out of the attention inner loop | +0.2–0.3% | low | 30 min | yes | no |
| **7** | §6 A2 — fuse `k_attn_combine` via threadfence ticket | +0.25% | low | ~half day | yes | no |
| **8** | §4 R2 — router weights bf16 → FP8 | +0.6% | low | ~2 h + gate | ⚠ no | no |
| — | **§5 RMSNorm → GEMV prologue fusion** | **≤+1.2%, probably ≤+0.5%** | low | days, 5 kernels, 5 re-gates | yes | no | **DO NOT BUILD** |
| — | **§1/§2 megakernel (any sync mechanism)** | **≤+1.8%** from launches, **0%** from tail | resolved | **weeks**, undoes the TM=1 register win | yes | no | **DEAD, reconfirmed** |
| — | PDL / `cudaTriggerProgrammaticLaunchCompletion` | unknown | — | — | — | — | **null result on GB10, the closest small-Blackwell proxy; NVIDIA calls overlap "not guaranteed"** |
| — | cluster barriers / DSMEM as a grid barrier | 0 | resolved | — | — | — | **`cta_group` caps at 2 on Thor — cannot span 20 SMs** |

---

## 8. §4 premises: what fell, what survived

**Refuted:**

* **"`norm+cast` is 6.8% of the step."** No — it is ~0.7–1.2%, and the 6.8% is instrumentation.
  Byte arithmetic (48 KB/call, 0.21 µs) and the δ model agree to within 3 µs. **This is the most
  valuable line in the document: it deletes a work item, not adds one.**
* **"20.1% of the step is latency-bound."** No — **~10.5%**, in a step that is 30.3 ms, not 42 ms.
* **"`attn_qkvo_gemm` is 34.7%."** It is **39.7%** after the correction, which *raises* the value of
  `o_proj → NVFP4`.
* **Premise §4.4, "`N_k·c_k` is irreducible without a megakernel, and the megakernel is blocked by
  `grid.sync`."** Half-refuted: `grid.sync` is **not** the blocker (global-atomic counters cost ~8×
  less — MPK 0.28% of runtime, Hazy 6.7%). The blocker is that **the prize is 1.8%**. The
  conclusion survives; the stated reason does not.
* **"TMA is dead because it is not a bandwidth lever."** True but irrelevant — TMA is the
  *pipelining* primitive for cross-boundary prefetch, the only live megakernel term. It was rejected
  on the wrong axis. (It still does not clear the bar.)

**Survived adversarial attempt:**

* **§4.6, speculation on prose needs a retrained drafter.** Nothing found.
* **§4.5, `E_frac` position correlation is inexploitable at inference.** Nothing found; still the
  only such curve measured anywhere.
* **The megakernel verdict.** Attacked from three directions (small-GPU precedent, sync primitives,
  win decomposition) and it got *stronger*: term (c) is measured at 0 on this device, term (a) is
  bounded at 1.8% by our own null-kernel sweep and corroborated by arXiv:2605.30571's
  1.259× (H100) → 1.028× (L4) gradient, and term (b) is capped by our own 91%-of-ceiling figure.
* **"Our top-k is slow."** Refuted in our favour: ~2–5 µs for 256-way sigmoid + top-10 in one warp
  is at the published floor. vLLM's own DeepSeek-V3.2 fix at batch 1 was the **router GEMM**, worth
  6% — not the top-k.

**Could not resolve:**

* **tcgen05 / TMEM presence on `sm_110a`.** `HARDWARE.md` says present, V5 says absent, a forum post
  says present, and **NVIDIA's public docs do not document CC 11.0 at all**. Needs the 20-minute
  on-device probe (ranked #3).
* **Jetson CUDA-graph launch overhead.** No public number exists for Orin or Thor. Our 1.6–1.7 µs
  is, as far as the literature goes, the only datapoint on this hardware class.
* **`__nanosleep` backoff vs busy-poll.** No published microbenchmark on any architecture.

---

## Sources

[Mirage MPK arXiv:2512.22219](https://arxiv.org/abs/2512.22219) ·
[MPK project page](https://catalyst.cs.cmu.edu/projects/mpk.html) ·
[Hazy Research, Look Ma No Bubbles](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) ·
[Ada-MK arXiv:2605.11581](https://arxiv.org/html/2605.11581v1) ·
[FlashDMoE arXiv:2506.04667](https://arxiv.org/html/2506.04667v2) ·
[Fleet arXiv:2604.15379](https://arxiv.org/abs/2604.15379) ·
[POD-Attention ASPLOS'25](https://akkamath.github.io/files/ASPLOS25_POD.pdf) ·
[Memory-Bound but Not Bandwidth-Limited arXiv:2605.30571](https://arxiv.org/abs/2605.30571) ·
[Alpin, RTX 5090 decode optimization](https://blog.alpindale.net/posts/5090_decode_optimization/) ·
[Benchmarking Thread Block Cluster, TUHH](https://tore.tuhh.de/dspace-cris-server/api/core/bitstreams/f19e2bfe-f059-499b-b473-d9f00f78d01d/content) ·
[Dissecting Hopper, arXiv:2501.12084](https://arxiv.org/html/2501.12084v1) ·
[GH100 vs GB203 microbenchmarks, arXiv:2507.10789](https://arxiv.org/pdf/2507.10789) ·
[NVIDIA: constant-time graph launch](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-graphs-and-other-performance-enhancements) ·
[CUTLASS dependent kernel launch](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/dependent_kernel_launch.html) ·
[PDL null result on GB10](https://forums.developer.nvidia.com/t/issue-with-programmatic-dependent-launch-pdl-no-overlapping-execution-observed-on-gb10/359590) ·
[Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) ·
[CCCL cp.async.bulk (lists SM_110a)](https://nvidia.github.io/cccl/unstable/libcudacxx/ptx/instructions/cp_async_bulk.html) ·
[CUDA 13.0 for Jetson Thor](https://developer.nvidia.com/blog/whats-new-in-cuda-toolkit-13-0-for-jetson-thor-unified-arm-ecosystem-and-more/) ·
[sm100 vs sm120 feature table](https://0xsero.github.io/blackwell-gpu-wiki/blackwell/sm100-vs-sm120/) ·
[Jetson Thor CUTLASS FP4/FP8 tests](https://forums.developer.nvidia.com/t/jetson-thor-cutlass-fp4-fp8-fp16-performance-test/372014) ·
[Jetson Thor benchmarking thread](https://forums.developer.nvidia.com/t/performance-benchmarking-on-jetson-thor/349494) ·
[vLLM tops Artificial Analysis (DSv3.2 batch-1 router work)](https://vllm.ai/blog/2026-05-11-vllm-tops-artificial-analysis) ·
[vLLM RMSNorm+quant fusion PR #10906](https://github.com/vllm-project/vllm/pull/10906) ·
[vLLM fusion passes](https://docs.vllm.ai/en/stable/design/fusions/) ·
[PyTorch: Towards Free Normalization](https://pytorch.org/blog/towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/) ·
[Flash-Decoding](https://pytorch.org/blog/flash-decoding/) ·
[TensorRT-LLM XQA](https://nvidia.github.io/TensorRT-LLM/blogs/XQA-kernel.html) ·
[FlashInfer](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html) ·
[RTop-K arXiv:2409.00822](https://arxiv.org/pdf/2409.00822) ·
[vLLM sm_110 issue #26791](https://github.com/vllm-project/vllm/issues/26791) ·
[TokenWeave arXiv:2505.11329](https://arxiv.org/html/2505.11329v4)
