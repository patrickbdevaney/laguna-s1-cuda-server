# DIRECTIVE — `laguna-s1-cuda-server`

> **Archival copy of the original operating directive, verbatim, as handed over 2026-07-24.**
> This file is the authority on *intent and scope*. Where its factual premises were refuted by
> reading the checkpoint, `RESCOPE.md` records the correction and the reason — **`RESCOPE.md`
> supersedes this document on facts, never on goals.** Do not edit the text below.

---

**Port `gemma-cuda-hybrid` to `poolside/Laguna-S-2.1-NVFP4` + `Laguna-S-2.1-DFlash-NVFP4` on Jetson Thor (`sm_110a`).**

You are working from `~` as top level. `~/gemma-cuda-hybrid` is the complete reference implementation — **read-only**. `~/laguna-s1-cuda-server` is empty and is where everything you write goes. Never edit the gemma repo.

Deliverable: a single lean binary that serves Laguna S 2.1 as an OpenAI-compatible endpoint with NVFP4 W4A16 decode, FP8 KV, DFlash speculative decoding, prefix caching, reasoning separation, tool calling, WebUI, and a terminal client — no Python on the hot path — at the highest decode tok/s the hardware permits.

---

## 0. Read this before you touch anything

Three documents in `~/gemma-cuda-hybrid` are the operating manual for this work. Read them in full first:

1. `CUDA_ENGINEERING_CONSTITUTION.md` — the champion kernel stack and the full won/lost/neutral ledger. **Everything in the "lost" and "neutral" columns is already-paid tuition. Do not re-pay it.**
2. `AGENTIC_OPTIMIZATION_METHODOLOGY.md` — the profile-first loop, map-vs-territory discipline, black-swan budget.
3. `RESEARCH_FINDINGS.md` — what was researched, won, and refuted with evidence (Marlin, FlashInfer, tcgen05, roofline).

Then read `README.md`, `SERVER_PLAN.md`, `MEGAKERNEL_PLAN.md`, `TC_FEASIBILITY.md`, and skim `src/forward.cu`, `src/draft.cu`, and `kernels/`.

**Standing rule for this entire project: never invent a model constant.** Every architectural number for Laguna comes from the downloaded `config.json` and the `custom_code` modeling file in the HF repo, read directly. If you find yourself writing a hidden size, head count, head dim, top-k, or rope theta that you did not read out of a file on disk in this session, stop and go read it.

---

## 1. Ground truth — hardware

From the gemma repo's measured constitution, do not re-derive:

| | |
|---|---|
| Chip | Jetson AGX Thor, Blackwell `sm_110a` |
| SMs | 20 |
| Memory | LPDDR5X unified, **273 GB/s peak** |
| Achievable streaming BW | **~200 GB/s** |
| Achieved BW on gemma MoE decode path | **~91 GB/s** (45 tok/s × 2.02 GB/token) |
| Shared memory | 228 KB |
| Toolchain | CUDA 13, `-arch=sm_110a` |

**Gate H1 — capacity.** Before anything else, run `free -g` and `nvidia-smi` (or `tegrastats`). Laguna's NVFP4 weights are ~71 GB resident plus a ~2 GB BF16 draft. **If this box has less than ~110 GB of usable unified memory, this project is not viable and you must stop and report that.** The 64 GB Thor SKU cannot hold this model at any quantization it ships in. Write the result to `HARDWARE.md` as the first line of state.

---

## 2. Ground truth — model

Verified from the model cards (2026-07-24). Treat these as fixed; everything else you must read from `config.json`.

**`poolside/Laguna-S-2.1-NVFP4`** (target)
- 117.6B total / **8.5B activated per token**
- 48 layers: **12 global attention, 36 sliding-window attention** (3:1 ratio)
- **Sliding window = 512 tokens**
- **256 experts + 1 shared expert** (top-k not stated on the card — read it from config)
- Softplus per-head gating, **per-layer rotary scales** — this is not a uniform-RoPE model
- KV cache natively FP8
- Context 262,144 as shipped; native 1M checkpoint, restorable via `rope_parameters.full_attention.factor = 128.0` + `max_position_embeddings = 1048576`
- NVFP4 weights ≈ **71 GB on disk/resident**
- Repo is tagged `custom_code` → **there is a reference modeling implementation in the repo. It is your correctness oracle. Read it.**
- Recommended sampling: `temperature 0.7`, `top_p 0.95`, `top_k 20` (from `generation_config.json`)
- vLLM parsers: `--tool-call-parser poolside_v1 --reasoning-parser poolside_v1`

**`poolside/Laguna-S-2.1-DFlash-NVFP4`** (draft)
- **6-layer Laguna-style draft, ~1B params, weights are BF16** despite the repo name (the "NVFP4" refers to the target it's matched to). Model size field reads `1B params / BF16`.
- **The draft staying BF16 is the acceptance moat.** This is the same finding as the gemma port. Do not quantize the draft.
- Poolside's own recommendation on the draft card: **`num_speculative_tokens = 7`**. The NVFP4 card's DGX Spark recipe says 15. **These conflict. Treat 7 as the prior and sweep k empirically — see §5.**
- Reported acceptance length at concurrency 1, temp 0, TP=2 datacenter: HumanEval **6.44**, MBPP 4.17, MT-Bench 4.02. Throughput speedup 3.69× / 2.43× / 2.34× respectively.
- Poolside measured only **2.9–3.1 accepted/step on a DGX Spark at temp 0.7**. Sampling temperature is the likely cause of the gap vs the temp-0 numbers. Note this when you interpret your own acceptance measurements.

**Reference point you must not ignore.** Poolside measured this exact pair on a **DGX Spark (GB10, 128 GB unified, same 273 GB/s class as Thor)** under vLLM 0.25.1:
- Prefill 600–800 tok/s
- Decode **13–14 tok/s without speculation** — poolside calls this "the memory-bandwidth ceiling for this model"
- Decode **15 tok/s prose / 22–24 tok/s code with DFlash**
- 830–870K KV tokens at the 256K setting

---

## 3. Phase 0 — roofline before code

**Do not write a kernel until `ROOFLINE.md` exists and is defended with arithmetic.** This phase is the highest-value hour in the project because it determines whether the target is 25 tok/s or 50 tok/s, and those imply different engineering.

Produce `ROOFLINE.md` containing:

### 3.1 Bytes per token

Read `config.json` and the safetensors index. Compute exactly, not approximately:
- Non-expert bytes read per decode step (embeddings touched, attention QKVO across 48 layers, norms, router, shared expert, lm_head).
- Expert bytes read per decode step at top-k (single token).
- FP8 KV bytes read per step at a given context length, split between the 12 global layers and the 36 SWA(512) layers. **The SWA layers read a bounded 512-token window regardless of context — this is a large and permanent win that gemma-4 only partly had. Quantify it.**
- Total → **B_tok** (GB/token).

Sanity anchor: 8.5B active × 0.531 B/param (NVFP4 = 4.25 bits) ≈ **4.51 GB**. Your computed B_tok should land near this. If it doesn't, you read something wrong.

### 3.2 Autoregressive decode projections

| Scenario | Effective BW | Projected AR decode |
|---|---|---|
| Same effective BW as the gemma champion path | ~91 GB/s | **~20 tok/s** |
| Improved (larger per-layer work amortizes launch overhead better than gemma) | ~135 GB/s | **~30 tok/s** |
| Hard ceiling at Thor's achievable streaming BW | ~200 GB/s | **~44 tok/s** |

**44 tok/s is the autoregressive wall.** Nothing gets past it except speculation. Write that sentence in `ROOFLINE.md`.

Note that poolside's GB10 measurement of 13–14 tok/s implies **~59–63 GB/s effective** — roughly a third of Thor's achievable, and well below what the gemma repo already demonstrates on this chip class. That gap is the actual opportunity here, and it is bigger than the +10%-over-vLLM heuristic from the gemma work. **State explicitly in `ROOFLINE.md` that the +10% heuristic does not transfer**: it was measured against a vLLM path that was already near-roofline for gemma-4, whereas vLLM's Laguna MoE path on unified-memory Blackwell appears to be leaving 1.5–2× on the floor.

### 3.3 The expert-union problem — the dominant unknown

This is the crux of the whole project and you must model it before you build.

Speculative verification of `k` draft tokens in one forward pass activates the **union** of experts routed to by all `k` positions, not the intersection. Gemma-4 had 128 experts; Laguna has **256**. At the same top-k, doubling the expert count roughly doubles the union's spread for a given `k`, so **the verify step gets disproportionately more expensive on Laguna than it did on gemma.**

Model it:

- Let `E_frac(k)` = fraction of the ~55 GB of expert weights touched by a k-token verify block.
- Verify step cost ≈ `(non-expert bytes × 1) + (E_frac(k) × expert bytes)`.
- Effective decode = `τ(k) / verify_step_time(k)`, where `τ(k)` is accepted tokens per block.
- `τ` grows sublinearly in `k`; `E_frac` grows toward 1. **There is an interior optimum and it is very likely well below k=15.**

Produce a table of projected tok/s over `k ∈ {3, 5, 7, 9, 11, 15}` under `E_frac ∈ {0.3, 0.5, 0.7}`, using `τ(k)` interpolated from poolside's published acceptance numbers. Then state your headline projection as a **band, not a point**:

> Expected realistic band: **32–52 tok/s**, centered near 40–45, contingent on `E_frac(k*) ≤ 0.5` at the optimal `k*`.

**Do not carry forward a 50–70 tok/s expectation.** 70 tok/s would require B_tok under 2.9 GB at full achievable bandwidth with perfect speculation, and B_tok is ~4.5 GB. The high end of the band is reachable; 70 is not.

### 3.4 Gate R1

`ROOFLINE.md` is complete when it contains: measured B_tok, the AR wall, the k-sweep projection table, the `E_frac` sensitivity, and a named `k*` prior. Only then proceed.

---

## 4. Phase 1 — acquisition, inventory, and the reference oracle

1. `hf download poolside/Laguna-S-2.1-NVFP4` and `hf download poolside/Laguna-S-2.1-DFlash-NVFP4`. ~73 GB combined. Verify free disk before starting; a failed 71 GB download is an expensive mistake.
2. Dump `config.json`, `generation_config.json`, the tokenizer files, and the safetensors index into `MODEL_INVENTORY.md`: every tensor name, shape, dtype, and the NVFP4 scale layout (E2M1 block + E4M3 group scale + FP32 global — confirm the group size, gemma's was 16).
3. Read the `custom_code` modeling file end to end. Write `ARCH_DELTA.md` — a per-subsystem diff of gemma-4 vs Laguna S 2.1:

| Subsystem | gemma-4-26B-A4B | Laguna S 2.1 | Port verdict |
|---|---|---|---|
| Layers | 30 | 48 | constant change |
| Total / active | 25.2B / 3.8B | 117.6B / 8.5B | constant change |
| Experts | 128, top-8, +1 shared | 256, +1 shared (read top-k) | **retile grouped MoE GEMM** |
| Attention mix | sliding hd=256/nkv=8 + full hd=512/nkv=2 | 36 SWA(512) + 12 global, 3:1 | **new mask/pattern logic** |
| Gating | gemma router | **softplus per-head gating** | **new kernel** |
| RoPE | single θ | **per-layer rotary scales** | **new: per-layer scale table** |
| Norms | gemma double-norm | read from config | verify |
| Vocab | 262,144 | read from config | constant change |
| Tokenizer | gemma-4 BPE | poolside — **full rewrite** | **new** |
| Chat grammar | `<\|turn\|>` / `<\|channel\|>` / `<\|tool_call\|>` | poolside_v1 — **full rewrite** | **new** |
| Draft | 5-layer qwen3-style, k=14 | 6-layer Laguna-style, k≈7 | **new, same BF16 discipline** |
| KV | FP8 e4m3, 64K default | FP8 e4m3, 256K target | retune sizing |

4. **Stand up the correctness oracle.** Install vLLM 0.25.1 (or Transformers with `trust_remote_code`) in a venv *outside* the build, purely as a reference. Capture golden tensors: per-layer hidden states and logits for a fixed prompt at temp 0. Every gate in §6 is checked bit-exact or within a stated ULP tolerance against these. **This is non-negotiable — the gemma port's entire discipline was correctness-first, and skipping it here on a model 4.7× larger would be unrecoverable.**

**Gate A1.** Reference oracle produces reproducible golden tensors on disk. `ARCH_DELTA.md` complete.

---

## 5. Phase 2 — the loader (do this before kernels)

At 71 GB on a 128 GB unified-memory part, **loading is a first-class engineering problem, not plumbing.**

- The gemma repo already contains the **unified-memory double-copy fix**. Port it first and port it exactly. A naive load that stages 71 GB in host buffers before copying to device buffers needs 142 GB and will take the machine down.
- mmap the safetensors; do the NVFP4 offline repack into mma-fragment order **streaming, shard by shard**, never materializing the full model twice.
- Write the repacked weights to a cache file so subsequent starts skip the repack. First load off NVMe will be minutes; make the second load fast.
- Instrument peak RSS + peak device allocation during load and record it in `STATUS.md`.

**Gate L1.** Model loads to device, peak memory during load stays under 85 GB, and a second start from the repack cache completes in under 60 s.

---

## 6. Phase 3 — gated kernel build

Mirror the gemma repo's gate discipline. Each gate is bit-exact (or stated-tolerance) against the oracle from §4.4 before the next begins. Log every gate to `LOOP_LOG.md` with pass/fail and the measured delta.

- **G1 — NVFP4 dequant.** E2M1 + E4M3 group scale + FP32 global → bf16/fp16, using HW `__nv_cvt_fp4x2` intrinsics. Port from `kernels/nvfp4_quant.cu`. Confirm the group size from the checkpoint rather than assuming 16.
- **G2 — dense linear / tensor-core GEMM.** Port the Marlin-class raw `mma.sync.m16n8k16` path from `kernels/tc_verify_gemm.cu` wholesale: in-register FP4→fp16 dequant, offline repack, 16-byte `int4` coalesced loads, `__ldcs` evict-first, max-grid-fill. **Retile for Laguna's shapes.** The tcgen05 rejection from the gemma work still holds at M≤16 — do not revisit it.
- **G3 — attention.** Head-packed GQA (KV read once per kv-head). New work: the **3:1 SWA/global layer pattern with a 512-token window**, and **per-layer rotary scales** (a per-layer scale table, not a single θ). The SWA window is small enough that its KV working set may stay resident in L2 — check this, it's a real optimization.
- **G4 — softplus per-head gating + router.** New kernel, no gemma analog. Validate router top-k selection index-exact against the oracle; a single mis-ordered expert selection silently destroys quality without crashing.
- **G5 — grouped weight-resident MoE.** Port the no-atomics, U-unroll-prefetch, HW-decode SwiGLU structure, then **retile for 256 experts + 1 shared**. Instrument achieved bandwidth per call. This kernel is where the project is won or lost.
- **G6 — FP8 KV cache.** e4m3, sized for 256K context with the SWA/global split. Target the ~830–870K KV token capacity poolside reports at 256K on a 128 GB box; if you can beat it because SWA layers only need 512, say so with numbers.
- **G7 — lm_head.** Quantize the tied embedding to NVFP4 as gemma did. Confirm Laguna actually ties the embedding before assuming it.
- **G8 — full autoregressive forward, bit-exact greedy vs oracle.** Record base decode tok/s. **Compare against the §3.2 projection and reconcile any gap over 20% before proceeding.**
- **G9 — CUDA graph capture** of the decode/verify step.

**Gate B1.** G1–G9 pass. `STATUS.md` records base AR tok/s and achieved effective bandwidth.

---

## 7. Phase 4 — DFlash, and the k-sweep

1. Port `src/draft.cu` structure. Load the 6-layer draft **in BF16**. Do not quantize it. Share the frozen embed + lm_head with the target.
2. Implement the propose/verify loop with Gumbel-max target sampling and sample-match acceptance, so temp=0 reduces to exact greedy (gemma's lossless sampling — port it, it is subtle and already correct).
3. **Run the k-sweep. This is the single most important experiment in the project.** For `k ∈ {3, 5, 7, 9, 11, 15}`, on a code-generation workload at both temp 0 and temp 0.7, measure:
   - τ (accepted tokens/block)
   - verify step wall time
   - **measured `E_frac(k)` — instrument the actual count of distinct experts activated per verify block.** This is the number your §3.3 model guessed at. Replace the guess with the measurement and rewrite `ROOFLINE.md`.
   - end-to-end tok/s
4. Pick `k*` from the measurement, not from either model card. Expect `k*` to land nearer 7 than 15.
5. If `E_frac(k*) > 0.6`, the naive verify is reading most of the expert weights and you should evaluate **routing-aware draft scheduling**: truncating the verify block at the position where the expert union crosses a byte budget. Log the idea in `OPTIMIZATION_LOG.md` even if you don't build it.

**Gate D1.** DFlash decode tok/s recorded at `k*`, with measured τ and `E_frac`. Head-to-head against vLLM on the same box, same prompt, back-to-back, in `BENCHMARK_COMPARISON.md`.

---

## 8. Phase 5 — server layer

Port feature-for-feature from the gemma repo. This is the part that makes it an agent substrate rather than a benchmark.

- `POST /v1/chat/completions` — streaming SSE + non-streaming; `GET /v1/models`; WebUI at `GET /`.
- **Prefix caching** — LCP KV reuse across turns. **This is the single highest-leverage feature for the long-horizon agentic use case**: a large CLAUDE.md-style constitution plus accumulated project-state markdown is prefilled once and reused every turn. At 600–800 tok/s prefill, a 40K-token constitution costs ~60 s cold and ~0 s warm. Make sure the cache survives tool-call round-trips, which is where naive implementations break.
- **Reasoning separation** — hand-written equivalent of `--reasoning-parser poolside_v1`. Laguna does **interleaved thinking between tool calls with preserved thinking**, which is materially different from gemma's single thought channel. Read the poolside chat template and the reference parser before writing this; get the preserved-thinking semantics right or multi-turn agentic traces will degrade.
- **Tool calling** — equivalent of `--tool-call-parser poolside_v1`. Emit OpenAI `tool_calls` with valid-JSON `arguments` and `finish_reason: "tool_calls"`.
- **`enable_thinking` per-request**, defaulting per poolside's guidance (reasoning on by default).
- **Sampling defaults baked in**: `temperature 0.7`, `top_p 0.95`, `top_k 20`. Many agent clients send no sampling params, and raw defaults measurably degrade NVFP4 output. Do not accept `min_p` or `logit_bias` under speculation.
- Self-contained WebUI (no CDN, no npm, offline) and the pure-C++ terminal client.
- Env surface consistent with gemma: `CTX`, `FP8KV`, `DK`, `SERVE`, `PORT`.
- **Also expose the 1M context path** behind an env flag, applying the `rope_parameters` override from the model card, with a documented quality caveat.

**Gate S1.** `pool` CLI and a generic OpenAI-SDK agent loop both drive the server through a multi-turn tool-calling session with reasoning preserved. Prefix cache hit rate logged and non-trivial.

---

## 9. Phase 6 — optimization loop

Now, and only now, run the `AGENTIC_OPTIMIZATION_METHODOLOGY.md` loop: profile → hypothesis → single change → measure → gate → log. Every attempt goes in `OPTIMIZATION_LOG.md` with a won/lost/neutral verdict, so the ledger for Laguna becomes as reusable as gemma's is.

Priority order, derived from the roofline:
1. MoE grouped GEMM achieved bandwidth (G5) — biggest single lever.
2. `E_frac` reduction in verify.
3. SWA KV residency in L2.
4. Kernel launch count per decode step (48 layers is 1.6× gemma's; launch overhead scales with it on a 20-SM part).
5. Everything else.

---

## 10. Invariants

- **No Python on the hot path.** No PyTorch, no framework, no JIT at serve time.
- **Correctness gates before speed gates. Always.** A faster wrong kernel is worth zero.
- **The draft stays BF16.**
- **No invented model constants.** Read from disk or don't write it.
- **Do not re-litigate the gemma ledger.** tcgen05 at M≤16, and everything else in the "lost" column, stays lost unless Laguna's shapes give a specific documented reason to revisit.
- **One change per measurement.**
- **Report bands, not points**, for anything not yet measured.
- **If a gate fails, stop and report. Do not build on an unverified layer.** On a 117B model, an error introduced at layer 12 and discovered at the server layer is days of work.

---

## 11. State files to maintain in `~/laguna-s1-cuda-server`

`HARDWARE.md` · `ROOFLINE.md` · `MODEL_INVENTORY.md` · `ARCH_DELTA.md` · `STATUS.md` · `LOOP_LOG.md` · `OPTIMIZATION_LOG.md` · `BENCHMARK_COMPARISON.md` · `CUDA_ENGINEERING_CONSTITUTION.md` (Laguna edition) · `README.md`

Update `STATUS.md` at every gate. It is the resume point if this session ends.

---

## 12. Known risks, stated up front

1. **Capacity.** 71 GB weights + 2 GB draft + KV + activations on 128 GB unified. Workable but not comfortable. Loader discipline (§5) is mandatory, not optional.
2. **Expert-union blowup.** 256 experts makes speculation structurally less profitable than it was on gemma's 128. This is the most likely reason the final number lands at 35 rather than 50.
3. **Speed expectation.** The realistic band is **32–52 tok/s**, not 50–70. Poolside's own same-bandwidth-class measurement is 22–24 tok/s under vLLM. Beating that by 1.5–2× is an excellent, publishable result. Beating it by 3× is not on the table.
4. **Scope.** Laguna is 48 layers, 256 experts, a new gating scheme, per-layer rotary scales, a new tokenizer, and a new chat/tool/reasoning grammar. This is meaningfully more new work than the gemma port implies — perhaps 60% port, 40% net-new. Plan and pace accordingly.
5. **Draft acceptance at temperature.** Published τ is temp-0. Poolside saw 2.9–3.1 at temp 0.7. Agentic coding at temp 0.7 is the real workload; measure τ there, not at temp 0, when quoting the headline number.
6. **DFlash upstream support is still in progress** (vLLM #46853, SGLang #29446, TRT-LLM #15666). The reference implementations you're gating against may themselves be moving. Pin versions and record them.

---

## First action

Read the three gemma methodology docs. Then run Gate H1. Then produce `ROOFLINE.md`. Report back with the roofline before writing a single kernel.

---

## Standing amendment (2026-07-24, post-roofline)

> proceed with Gate A1 when the downloads required are complete. rescope against the corrected
> roofline. write the original directive in the repo. we want a continual operating loop with no
> handback of setting up the max KV size that doesn't affect decode speed that this model and
> DFlash draft head can run. our goal as you recall is a good large enough long-horizon agentic
> KV context window but optimal decode speed too for this model in a pure CUDA server like
> gemma-cuda-hybrid (with its C++ as needed). keep looping indefinitely with no handback. use
> every lever possible.

This amendment adds: (a) **KV sizing is a first-class objective**, co-equal with decode tok/s —
see `RESCOPE.md` §4; (b) the loop runs **autonomously to completion** without checkpointing back;
(c) **every lever is in scope**, including levers the original directive did not name (the
self-quantization lever in `ROOFLINE.md` §5 is the first of these).
