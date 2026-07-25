# IMPLEMENTATION_PLAN.md — what the research pass says to build, in order

Derived from `RESEARCH_FINDINGS_V2.md`. Ranked by expected value ÷ cost. Items marked **DONE**
were implemented and measured during the pass itself.

Baseline at the start of the pass: **27.4 tok/s** base decode, 33.6 tok/s served on code.
Baseline now: **29.4–30.8 tok/s** base decode.

---

## DONE — landed and measured

| # | change | result |
|---|---|---|
| 1 | NVFP4 scale layout `[grp][lane]` → `[grp/8][lane][8]` | **27.4 → 30.8 tok/s (+11 %)**, bit-exact, greedy 8/8 |
| 2 | DFlash context K/V: per-layer `input_layernorm` | τ 3.06 → 3.12 (k=3), 3.30 → 3.37 (k=4) |
| 3 | Draft `rope_theta` ambiguity settled by measurement | config's 1e4 beats the siblings' 5e5 (τ 3.12 vs 2.91) — keep config |
| 4 | All 38 `extern "C"` kernel decls consolidated into one header | closes a silent-miscall class of bug |
| 5 | `bench_moe` DVFS spin-up + arena warmup + production activation loads | makes every future microbenchmark here trustworthy |
| 6 | **`lm_head` BF16 → FP8** (N1, first step) | **29.5 → 30.6 tok/s (+3.6 %)**, `B_tok` 7.115 → 6.807 GB, greedy 8/8 |

## MEASURED AND REJECTED

| lever | measurement | verdict |
|---|---|---|
| **EMC clock pin** (`bwmgr/min_freq=4266000000`) | 29.46/29.55 pinned vs 29.37/29.42 default | **NULL for this workload.** A synthetic streaming kernel gains 32–39 % from the pin, and 1 Hz samples of `cur_freq` during our decode do show 2750 MHz — but the decode step is 1665 dependent launches, not a stream, and it already sustains 212 GB/s effective, which 2750 MHz (176 GB/s theoretical) cannot produce. The samples were catching the clock between steps. **Do not make this system change.** Setting was pinned, measured, and restored. |
| **MAXN power mode** | EMC max is 4266 MHz in *both* MAXN and 120 W per NVIDIA's own table; measured peak draw 62.7 W against the 120 W cap at 47 °C | **~0 %.** We are bandwidth-limited with ~57 W of headroom, not power-limited. MAXN only raises GPC 1386 → 1575 MHz. Permanently off the list. |
| **L2 persistence** | 24 MB of persisting cache against 7.24 GB of read-once weights = **≤0.33 %** ceiling | Dead, now quantified. Our earlier null was correct. |
| **GEMV access-pattern tuning** | `__ldcs`/`__ldg`/`ld.global.nc` within noise; `uint4` + 4–8 CTAs/SM already optimal; `__ldcg` is a 21 % *regression* | Already optimal. Stop here. |
| **Allocator alternatives** | `cudaMalloc` 252 GB/s vs managed+prefetch 248, `cudaHostRegister` 217, THP 214 | `cudaMalloc` stays. |

---

## NEXT — ranked

### N1. FP8 the BF16 remainder — **+13.6 %, reuses machinery already gated**
`lm_head` (0.617 GB), shared experts (0.887 GB) and the layer-0 dense MLP (0.227 GB) are still
BF16. All three can go through the **existing** FP8 W8A16 per-output-row kernel — the one that
already gave +8.1 % on attention and is covered by Gate B1.

| move | Δ `B_tok` | precedent |
|---|---:|---|
| `lm_head` → FP8 | −0.308 GB | Nemotron-3-Nano-4B ships exactly this |
| shared experts → FP8 | −0.443 GB | Nemotron-3-Super/Ultra ship exactly this |
| layer-0 dense → FP8 | −0.113 GB | Mistral-Medium-3.5 gives edge layers FP8 |
| | **7.20 → 6.34 GB** | **≈ +13.6 %** |

No new kernel, no new numerics. Quantization happens at load time exactly as `put_q8` already
does for attention. **Order: `lm_head` first** — it is one tensor and validates the path.

Explicitly **not** in scope: q/k/v in NVFP4 (five 2026 production recipes exclude attention;
damage scales down with active params and Laguna is 8.5 B active), and `g_proj` in anything
(gate projections are the worst outlier class in the NVFP4 literature, and it is 0.010 GB).

### N2. Replace the throughput bandit with a verify-budget controller — **+8 %, +17–20 % mixed**
Our bandit ranks whole arms by realized throughput, which is why it thrashed, why exploration
cost 30 % of a block, and why it lags mid-generation acceptance collapse. EVICT's rule fixes
all three at once:

```
E[tokens | k] = 1 + p0 + p0·p1 + …        p[i] = EWMA per-position acceptance
cost(k)       = profiled once at init, offline
k*            = argmax_k E[tokens|k] / cost(k)          (AR scored as k=0)
```

One k=5 step yields **five Bernoulli observations** instead of one throughput sample, and a step
at *any* k informs the estimate for *every* k — so exploration cost goes to zero. Cohere's
measurement that adjacent-token expert overlap is 0.381 across all 13 Spec-Bench categories and
seven languages is the licence to profile `cost(k)` once and reuse it everywhere.

**Prerequisite, and it is a two-for-one:** this needs the draft's per-position probabilities,
which we do not plumb today — and which is also the reason sampled requests fall back to AR.
Exposing them unlocks the controller *and* the rejection sampler for temperature > 0.

### N3. SuffixDecoding ahead of DFlash for agentic traffic
A suffix automaton over prompt + prior output: **zero extra weight reads**, bit-exact, τ ≈ 7.8
on SWE-Bench-class traffic because agentic output re-copies context heavily. Its linear draft
performs about as well as its tree, so it sidesteps the expert-union problem entirely. Run the
suffix match first and fall through to DFlash when the match is short — both paths are exact,
so the fallback is free. This is the single best lever for the agentic workload the directive
actually targets.

### N4. Cheap, zero numerical risk
* `cudaAccessPolicyWindow` with `hitProp=cudaAccessPropertyStreaming` on the weight range —
  weights are single-use per token and currently evict KV and activations for nothing.
* Investigate whether mid-generation acceptance collapse correlates with **position** rather
  than content: the DFlash paper measures τ 4.53 → 2.09 at 32 K context. If it is position, the
  fix is a drafter finetune, not a controller. **One-hour experiment, changes the diagnosis.**

### N5. Counter-driven work, now that `sudo ncu` is known to work
`sudo /usr/local/cuda-13.0/bin/ncu --metrics lts__d_sectors_fill_sysmem.sum,...`. Note
`dram__*` returns `n/a` on Tegra — use `lts__d_sectors_fill_sysmem.sum × 32` for DRAM read
bytes and `lts__t_request_hit_rate.pct` for L2 hit rate.

---

## Where the ceiling actually is

Published full-step batch-1 bandwidth efficiency tops out at **82 %** (FlashFormer, H100); the
Stanford megakernel reaches 78 %, GPT-Fast 68–75 %, vLLM/SGLang ~50 %. **We are at ~85 %.**

So the remaining headroom is not in the kernels — it is in the **byte budget** (N1) and in
**not spending target forwards on tokens that will be rejected** (N2, N3). That is a change of
strategy from the first two-thirds of this project, and it is what the research pass bought.
