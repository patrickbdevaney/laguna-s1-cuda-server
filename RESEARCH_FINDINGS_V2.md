# RESEARCH_FINDINGS_V2.md — six parallel deep-research passes, and what they changed

Prompt: `RESEARCH_PROMPT_V2.md`. Six axes run in parallel against the published literature,
production inference stacks, vendor documentation and poolside's own upstream PRs.

Two of the findings **contradicted things this repo asserted**, and one contradicted a
conclusion reached three hours earlier in the same session. Those are listed first, because a
research pass that only confirms you is a research pass that was not worth running.

---

## A. Corrections to this repo

### A1. `ncu` hardware counters are NOT blocked. `ERR_NVGPUCTRPERM` is just non-root.

`OPTIMIZATION_LOG.md` §0 said counters were unavailable and treated that as terminal. It is
not: `sudo -E env PATH=$PATH ncu --csv -k regex:... ./build/bench_moe` works on this box today.

Caveat that matters: **`dram__*` metrics return `n/a` on Tegra** — there are no DRAM
performance counters — so DRAM throughput still cannot be read directly. But `l1tex__*`,
`smsp__*` and `gpu__compute_memory_throughput` all work, and they answer the questions that
five falsified hypotheses could not.

**Counters measured on `bench_moe`:**

| variant | ld requests | sectors/req | wavefronts/req | bytes/wavefront | `mio_throttle` |
|---|---:|---:|---:|---:|---:|
| FULL | 1,167,360 | 1.79 | 1.16 | 49.4 | 0.04 % |
| NODEQ | 61,440 | **16.00** | 4.03 | **127.1** | 0.04 % |

Two results, both negative for hypotheses we had been carrying:

* **The weight stream is already perfect** — 16 sectors/request, 127.1 of 128 bytes per
  wavefront. The offline repack did its job completely; there is no coalescing defect left.
* **We are not LSU/MIO issue-throttled** — `mio_throttle` 0.04 %, `lg_throttle` 0.00 %. That
  kills the load-issue-ceiling theory empirically, without needing to research it.

### A2. The first `bench_moe` harness had a defect, so its attribution was partly wrong

The harness read the activation row as 32 scalar 2-byte loads — the exact defect entry #20
removed from production — and `MODE 2/3` skipped the activation stream entirely, so FULL was
being compared against variants carrying no activation traffic at all. **93 % of the harness's
load requests came from an artifact production does not have.**

Fixed: every variant now issues the production `uint4` loads. **The conclusion survives, at a
smaller magnitude:**

| variant | GB/s (corrected harness) |
|---|---:|
| FULL (old scalar-scale form) | 187 |
| NOSCALE | 217 |
| NODEQ (+ FP4 decode removed) | 223 |
| STREAM (+ FMA chain removed) | 225 |

Scales cost **16 %**, not the 24 % first reported; FP4 decode and the FMA chain remain
essentially free. The **end-to-end** result was never in doubt — the layout change measured
27.3/27.5 → 30.8/29.5 tok/s in the real model with greedy 8/8 — but the *explanation* was
cleaner than the evidence supported, and that distinction is the whole point of the ledger.

### A3. Our DFlash context path was missing a per-layer `input_layernorm`

**DFlash is published**: arXiv 2602.06036 (ICML 2026, z-lab), reference code at
`github.com/z-lab/dflash`, and — decisively — poolside's own upstream integration in
`vllm/model_executor/models/laguna_dflash.py` (PR #46853).

That code stacks the six draft layers' `input_layernorm` weights and RMS-norms the fused
context states **per layer** before the K/V projection. We projected from the un-normed fused
vector. Independently corroborated by TensorRT-LLM PR #15666, which describes the same step as
"folds input layer normalization into KV weights".

This is invisible to a correctness test — the output stays exactly right either way, because
rejected drafts are simply replaced — and shows up only as a depressed acceptance rate.
**Fixed; τ 3.06 → 3.12 at k=3 and 3.30 → 3.37 at k=4.**

Confirmed unchanged by the same reading: per-tap `aux_hidden_norms` before concatenation,
`hidden_norm` in the context branch only, bonus token at P0 with masks at P0+1…P0+15, and
longest-prefix greedy acceptance.

### A4. Undocumented choices we were treating as known

* **Draft `rope_theta`.** The NVFP4 draft config says 1e4; poolside's BF16, FP8 and INT4 draft
  checkpoints — same architecture, same size, separately trained — all say **5e5**. The configs
  were hand-edited upstream. Settled empirically: 1e4 (our config value) gives τ 3.12 vs 2.91
  at k=3, so **the config we load is the right one**. Kept as `LG_DRAFT_THETA`.
* **`sliding_window_base`** (`moving_query` vs the legacy `fixed_anchor` training mask) is not
  recorded in the shipped checkpoint. We use standard per-query SWA, which matches vLLM and
  TRT-LLM inference.

---

## B. Where we actually stand against the published field

**We are at or above every published number for a full batch-1 decode step.**

| system | hardware | full-step bandwidth efficiency |
|---|---|---:|
| vLLM / SGLang (kernel-per-op) | H100 | ~50 % |
| GPT-Fast | H100 | 68–75 % |
| Stanford/Hazy megakernel | H100 | 78 % |
| **FlashFormer** (whole-model kernel) | H100 | **82 %** — highest published |
| **this repo** | Thor | **~85 %** at 30.8 tok/s |

And on the closest published hardware analogue — poolside's *own* DGX Spark GB10 measurement,
which is the same Blackwell/unified-LPDDR5X class at a comparable bandwidth:

| | poolside on GB10 | this repo on Thor |
|---|---:|---:|
| decode, no speculation | 13–14 tok/s | **30.8** |
| decode with DFlash | 22–24 tok/s | **33.6** (code) |
| acceptance τ | 2.9–3.1 | 3.1–4.3 |

Independent GB10 k-sweeps (classmethod, NVIDIA forums, and a single RTX PRO 6000) all land on
**k\* = 3–7**, confirming our k\* = 3–4 and confirming that the model card's
`num_speculative_tokens=15` is wrong for a bandwidth-bound part.

For the MoE kernel specifically, no published NVFP4 GEMV beats us: the winning entries in a
B200 NVFP4 GEMV competition reach **46 % of speed-of-light**, and Marlin, BitBLAS, FLUTE,
Atom, QUICK, T-MAC and Machete publish **no bandwidth number at all**. The one stronger
single-stream datapoint is `zeux/calm`'s `gf4` at 87 % on an MoE — bought with a weaker format
(3-bit, group 8) that packs codes and scale into the *same 32-bit word*.

---

## C. The ranked backlog this produced

### C1. Pin the EMC clock — **needs root, biggest single free lever**
`/sys/class/devfreq/bwmgr/` reads `cur_freq` **3200 MHz against `max_freq` 4266 MHz**. This
applies to *every* kernel, not just the MoE. `ROOFLINE.md` §11 estimated +3–8 %; the measured
clock ratio suggests the ceiling is higher. **Not attempted — it is a system-wide change on a
box running remote-access services, and it is the owner's call.**

### C2. FP8 the BF16 remainder — **+13.6 %, reuses machinery we already have**
Every 2026 production MoE recipe from NVIDIA moves shared experts and dense layers off BF16;
the generic `nvfp4_experts_only` preset leaves them BF16 only as a side effect of a glob.

| move | Δ bytes | new `B_tok` | precedent |
|---|---:|---:|---|
| shared experts BF16 → FP8 | −0.443 | | Nemotron-3-Super/Ultra ship exactly this |
| `lm_head` BF16 → FP8 | −0.308 | | Nemotron-3-Nano-4B ships exactly this |
| layer-0 dense BF16 → FP8 | −0.113 | | Mistral-Medium-3.5 gives edge layers FP8 |
| | **−0.864** | **7.20 → 6.34 GB** | **+13.6 %** |

All three reuse the **existing, already-gated** FP8 W8A16 per-output-row kernel. No new kernel,
no new numerics to validate. NVFP4 on the shared experts would give +9.7 % instead of +6.6 %,
but needs a quantizer we do not have and a kernel that must be independently proven to hit
bandwidth — two GB10 measurements show NVFP4 paths landing at **24 % utilization where FP8 hits
72 %**, so a byte-count win at 24 % utilization is a real-world loss.

**Do not** put q/k/v in NVFP4: five 2026 production recipes all exclude attention, and fresh
evidence (Qwen3-30B-A3B at 3B active loses 1–2 points on AIME/GPQA where Qwen3.5-27B at 27B
active loses <0.7) confirms the damage scales down with active parameters. Laguna is 8.5 B
active — the risky end. Our FP8-with-per-row-scales attention is now literally NVIDIA's written
recommendation for this case.

**Never quantize `g_proj`** — 0.010 GB, and gate projections are the module class with the most
pronounced outlier spikes in the NVFP4 literature.

### C3. Replace the throughput bandit with a verify-budget controller — **+8 % / +17–20 % mixed**
Our bandit ranks whole arms by realized throughput. The literature (EVICT, arXiv 2605.00342 —
training-free, lossless, and measured on **Ling-flash-2.0, a 103B/6.1B 256-expert MoE**, 1.28×
→ 1.59×) does something strictly better:

```
E[tokens | k] = 1 + p0 + p0·p1 + …      p[i] = EWMA per-position acceptance
cost(k)       = profiled table, built once at init
k*            = argmax_k E[tokens|k] / cost(k)        (score AR as k=0)
```

Why this beats what we built, in exactly our three observed failure modes:
1. **Thrashing** — a throughput sample spans the whole acceptance distribution; one k=5 step
   instead yields **five Bernoulli observations**, one per position. ~5× better signal per step.
2. **Exploration cost** — goes to **zero**. A step at any k updates `p[0…a]`, which informs
   every k. There are no arms to probe.
3. **Non-stationarity** — an EWMA on `p[i]` re-converges in 10–30 steps against our 32-step
   block × 4 arms × probe-every-10.

Cohere measured adjacent-token expert overlap at **0.381 across all 13 Spec-Bench categories
and seven languages** — "a structural property of MoE routing, not of the input distribution".
That is the licence to profile `cost(k)` **once, offline**, and reuse it for every workload.

Prerequisite, and it is a two-for-one: this needs the draft's **per-position probabilities**,
which we do not currently plumb — and which is also why sampled requests fall back to AR.
Exposing them unlocks both the controller and the rejection sampler for temperature > 0.

### C4. SuffixDecoding in front of DFlash for agentic traffic
A suffix automaton over prompt + prior output. **Zero extra weight reads**, bit-exact, and
**τ ≈ 7.8 on SWE-Bench-class agentic traffic** because agentic output massively re-copies
context (paths, diffs, tool schemas, repeated JSON). Its *linear* draft performs about as well
as its tree, so it sidesteps the expert-union problem by construction. Run suffix-match first,
fall through to DFlash when the match is short; both are bit-exact so the fallback is free.

### C5. Cheap and zero numerical risk
`cudaAccessPolicyWindow` with `hitProp=cudaAccessPropertyStreaming` on the weight range —
weights are strictly single-use per token and currently evict KV and activations from a 32 MB
L2 for nothing. Plus `prefetch.global.L2` (sm_20+, unlike the TMA-family
`cp.async.bulk.prefetch.L2` which is unavailable here).

---

## D. Explicitly rejected, with the measurement

* **EcoSpec / expert-aware token selection** — on DeepSeek-V3.1 (256 experts, top-8, our class)
  it reduces the expert union by **0.6 %**. Modelled on our byte table: **+2.9 %**. And it needs
  a *candidate tree* to re-rank, which a linear block-diffusion chain does not have.
* **Expert budgeting** (MoE-Spec, XShare, opportunistic activation) — 7–8 % reconstruction
  error, up to −10.8 % accuracy. Fails the exactness requirement.
* **Expert prefetch** (SP-MoE, MoE-SpeQ) — every one assumes host↔GPU offload over PCIe. Thor
  is one unified pool with no faster tier. **Zero transfer.**
* **Routing prediction** (*Speculating Experts*) — Qwen3-30B-A3B GSM8K **95.0 % → 57.6 %**.
* **Staged / chunked verify** — modelled −2 to −4 %: the fixed 7.4 GB term gets paid twice.
* **Lookahead / Jacobi** — measured **0.66×** (a slowdown) on an MoE target.
* **Full auto-megakernel compilers** (Mirage/MPK) — their own best case is 80 % of roofline,
  and their gains come from the launch overhead we already deleted with the CUDA graph.
* **Approximate `lm_head`** (HNSW / vector-index) — fails bit-exactness, and capped at +9.4 %
  anyway since `lm_head` is 8.6 % of the byte stream.
* **NVFP4 KV via vLLM** — gated to SM100 family and FlashInfer-only; Thor is SM110.
* **MXFP4 anywhere** — 4.1 pt MMLU gap vs NVFP4 on Llama-3.3-70B.
* **More denoising steps** — DFlash is one-step by construction; extra steps degrade.

---

## E. Two methodology results worth more than any single lever

1. **Short microbenchmarks on this part measure the idle clock.** The same shape gave 0.194 ms
   and 0.627 ms in two processes with identical code — a 3.2× swing — purely on whether the
   preceding variant ran long enough to ramp DVFS. `bench_moe` now spins 300 ms before any
   measurement. **Several earlier "neutral" results in `OPTIMIZATION_LOG.md` were taken with
   20-iteration warmups and are worth re-running.**
2. **The first kernel over a freshly allocated arena is not measurable** — page-table and TLB
   warmup made a 182 GB/s shape read as 56 GB/s, which pointed at entirely the wrong conclusion.
