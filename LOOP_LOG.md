# LOOP_LOG.md — gate-by-gate record

Every gate: what was checked, against what, measured delta, verdict. Append-only.

---

## Gate H1 — capacity · **PASS** · 2026-07-24

122 GB unified / 115 GB available / 379 GB disk free. 128 GB Thor SKU.
74.13 GB of weights leaves ~40.9 GB for KV + activations. Detail: `HARDWARE.md`.

Open item carried forward: box is at `nvpmodel` mode 1 (120 W), **not MAXN**. Must be set
and EMC verified at max before the first bandwidth measurement (Gate B1).

---

## Gate R1 — roofline · **PASS** · 2026-07-24

`ROOFLINE.md` + `tools/roofline.py` (reads config + safetensors headers; no hardcoded
constants). Verification: per-group byte sum reproduces `index.metadata.total_size`
= 71 898 733 760 **exactly**; activated-param count reconstructs the card's 117.6 B / 8.5 B.

| quantity | value |
|---|---|
| `B_tok` @ ctx 4096 | **10.0444 GB** |
| attention share | **5.6063 GB = 56 %**, BF16 |
| routed-expert share | 2.4950 GB = 25 %, NVFP4 |
| AR wall @200 GB/s | **19.91 tok/s** |
| `E_frac(15)` | 0.450 (no blow-up) |
| `k*` prior | **4** (5 @T=0, 3 @T=0.7) |
| KV capacity | ~1.43 M tokens single-sequence |

Model calibration: predicts 13.4 AR / 26.1 code-spec / 21.1 prose-spec at 135 GB/s;
poolside measured 13–14 / 22–24 / 15 on GB10. Three independent matches ⇒ trusted.

Refuted three directive premises (`RESCOPE.md` §1). Largest: `B_tok` is 10.04 GB not 4.51,
because only routed experts are NVFP4.

---

## Gate A1 — inventory, arch delta, correctness oracle · **PASS** · 2026-07-24

### A1.1 Inventory — PASS
`MODEL_INVENTORY.md`. All 15 shards present and byte-exact. 145 153 target + 69 draft
tensor headers captured to `tools/hdr_*.json`.

NVFP4 layout confirmed from geometry, not assumption: `weight_scale [1024,192]` at K=3072
⇒ **group = 16**. Density 4.5 bits/param.

### A1.2 — ⚠ BUG FOUND AND FIXED: `weight_global_scale` is a RECIPROCAL

First dequant attempt multiplied by `weight_global_scale` and produced `absmax = 3.1e7`
(weights should be ~0.1). compressed-tensors stores
`global_scale = (FP8_MAX · FP4_MAX)/amax = 2688/amax` and pre-multiplies it into
`weight_scale`; **dequant must divide**.

Verified across three experts in different layers:

| tensor | `global_scale` | dequant absmax | `2688/gs` |
|---|---:|---:|---:|
| L1 e0 `gate_proj` | 11520.0 | 0.233333 | 0.233333 |
| L20 e137 `down_proj` | 17024.0 | 0.157895 | 0.157895 |
| L47 e255 `up_proj` | 3072.0 | 0.875000 | 0.875000 |

Cross-check: the BF16 `shared_expert.gate_proj` in the same layer has absmax 0.785 / std
0.022 — same regime as the dequantized experts (std 0.028). **This would have silently
garbaged every expert in the CUDA kernel.** Kernel note: fold `1/global_scale` into the
repacked group scales once, offline.

### A1.3 Reference transcription vs shipped `modeling_laguna.py` — PASS (bit-exact)

`oracle/ref_laguna.py` validated by `oracle/validate_ref.py` + `oracle/validate_matrix.py`
against the shipped reference instantiated at tiny scale with identical random weights
(transformers 5.12.1, torch 2.10.0+cu130).

| case | S | window | layers | prefill maxabs | argmax |
|---|---:|---:|---:|---:|---|
| baseline | 12 | 512 | 4 | **0.00e+00** | ✓ |
| SWA clips (sw=4 < S) | 20 | 4 | 4 | **0.00e+00** | ✓ |
| SWA clips, 8 layers | 24 | 5 | 8 | **0.00e+00** | ✓ |
| SWA clips + incremental KV | 10 | 4 | 4 | 0.00e+00 / decode 1.79e-07 | ✓ |
| no clip + incremental KV | 8 | 512 | 4 | 0.00e+00 / decode 1.49e-07 | ✓ |
| long seq, 8 layers | 40 | 7 | 8 | **0.00e+00** | ✓ |
| seed variation | 16 | 3 | 4 | 0.00e+00 / decode 1.79e-07 | ✓ |

Decode-path residual ~1.8e-07 is fp32 accumulation order (cached vs recomputed K/V), not
a semantic difference; argmax identical throughout.

Also validated: our yarn `inv_freq` transcription vs the shipped rotary module —
**maxdiff 0.00e+00**, `attention_factor` 1.3465735902799727 identical.

Comparison gotcha recorded: transformers emits `NL+1` hidden states as
`[embed, out_0 … out_{NL-2}, post_norm(out_{NL-1})]` — the last layer's raw output is
never recorded. A naive `hidden_states[L+1]` comparison reports a false mismatch on the
final layer only.

### A1.4 Golden tensors from the real 71.9 GB checkpoint — PASS

`oracle/golden.py`. 8.03 GB of BF16 non-expert weights resident as fp32 on GPU; the 63.87 GB
of NVFP4 experts stay on disk and only routed experts are dequantized, through a 384-entry
LRU. Peak host memory 53 GB.

- prompt: 54 tokens (poolside_v1 template, thinking on)
- prefill 135.4 s, expert LRU hits 54 834 / miss 21 306
- greedy 8 tokens in 22.9 s
- **ids `[33586, 81, 397, 874, 367, 2440, 330, 2833]` → `"Okay, I need to write a Python"`**

This is the strongest available end-to-end check: coherent, on-task output from a 117.6 B
model simultaneously validates NVFP4 dequant + nibble order, sigmoid routing with
selection-only bias, per-head softplus gating, QK-RMSNorm, partial rotary with yarn,
the 3:1 SWA/global pattern, the untied `lm_head`, the tokenizer, and the chat grammar.
Any one of them wrong yields garbage or off-topic text.

Artifacts: `docs/golden/golden_primes.pt` (per-layer hidden states, final norm, logits,
router selections per layer), `docs/golden/meta_primes.json`, `docs/golden/rope_ref.json`.

### A1.5 C++ config parser — PASS

`include/laguna_config.h` + `tests/test_config.cpp`. Every field read from `config.json`;
missing required keys are a hard error (no defaults that could mask a wrong constant).
Derived values match the independent Python computation in `ROOFLINE.md`:
global 12 / sliding 36 / window 512 / experts 256 / top-k 10 / group 16 / kv_fp8 /
per-layer heads 48↔72 / kv_group 6↔9 / rotary 64↔128 of 128.

`tests/dump_rope.cpp` vs `docs/golden/rope_ref.json`: max relative `inv_freq` difference
**1.67e-07 (full)** and **1.18e-07 (sliding)** — ~1 ULP of float32. Both `attention_factor`
values exact. **ROPE GATE: PASS.**

### Pinned versions

`transformers==5.12.1`, `torch 2.10.0+cu130`, `tokenizers` (oracle venv),
`safetensors 0.7.0`, driver 580.00, CUDA 13.0.48, L4T R38.4.0.
Recorded because DFlash upstream is still moving (vLLM #46853, SGLang #29446, TRT-LLM #15666).

---

## Correction — 2026-07-24 (self-caught before the loader was written)

Earlier prose in `MODEL_INVENTORY.md`, `ROOFLINE.md` and `LOOP_LOG.md` stated the BF16
(non-expert) portion of the checkpoint as **15.2 GB**. Measured exactly:

| | bytes |
|---|---:|
| routed experts (`weight_packed` 56.7741 + `weight_scale` 7.0968 + globals 0.0003) | **63.8712 GB** |
| everything else (BF16 attention, shared experts, router, layer-0 MLP, `lm_head`, `embed_tokens`, norms) | **8.0276 GB** |
| total | 71.8987 GB ✓ matches the index |

The correct figure is **8.03 GB**. The 15.2 came from subtracting only `weight_packed`
(56.77) from the total and forgetting that `weight_scale` is also part of the expert payload.

**No downstream number changes.** `B_tok`'s fixed term was always computed per component
(7.4110 GB) and reconciles exactly: 8.0276 − 0.6166 (`embed_tokens`, of which a decode step
reads one row) + 0.0000 = 7.411 GB. The roofline, the k-sweep, and the quantization-scenario
table are unaffected; only the prose was wrong. Sizing the loader's device allocations is the
first thing that would have consumed the wrong figure, which is why it was re-derived first.

---

## Gate L1 — loader · **PASS** · 2026-07-24

`include/laguna_weights.h` + `tests/test_loader.cu`. Requirements were: device load OK,
peak memory < 85 GB, fast restart < 60 s.

| | result |
|---|---|
| arena | **71.899 GB in ONE `cudaMalloc`**, 1096 reservations, 256 B-aligned |
| cold load (page cache dropped) | **44.6 s** — io 34.7 s + H2D 4.5 s |
| peak host RSS | **4.92 GB** |
| **peak total (device + host)** | **76.8 GB** — under the 85 GB limit |
| device free afterwards | **52.07 GB** for KV + draft + activations |
| tensors placed | 109 057 of 145 153; the other 36 096 are `input_global_scale`, intentionally unused (W4A16) |
| verification | every planned tensor found; dtype and byte-size checked per tensor; `e_gate_inv[0]` = 8.68056e-05 = 1/11520 exactly |

### What made it fast — three measurements, not guesses

The first working version took **74.4 s** and used 145 153 individual `cudaMemcpy` calls from
a lazily-faulted mmap. `tests/h2d_probe.cu` isolated why:

| path | GB/s |
|---|---:|
| one big memcpy from warm pageable mmap | **109.4** |
| 1365 × 1.5 MB copies (what the loader does per expert) | 21.7 |
| memcpy → `cudaHostAlloc` pinned → H2D | 10.6 |
| `cudaHostRegister` on the mmap | *not supported* on this platform |
| **cold `pread` → pageable → H2D** | **6.95** |

So the bottleneck was never the copies — it was mmap page-fault I/O at ~1 GB/s. Switching to
bulk `pread` (64 MB requests) into a reusable host buffer took io to 34.7 s. **Pinned staging
was measured and rejected: it is 10× slower than copying straight from pageable memory on
this integrated part.**

### Peak RSS: 9.67 → 4.92 GB

`std::vector::resize` growing organically per shard held old+new buffers simultaneously
during realloc. Sizing the read buffer once to the largest shard fixed it.

### The repack cache is currently WORTHLESS — disabled by default

Implemented as the directive asks (`save_cache`/`load_cache`, a byte image of the arena),
measured, and then turned off:

| path | time |
|---|---|
| cold from safetensors | **44.6 s** |
| from 71.9 GB cache file | **50.0 s** |
| cost to build the cache | 209 s + 71.9 GB of disk |

The cache is slower than the thing it caches, because our "repack" is presently a copy —
there is no expensive transform to amortise. The code stays (it will earn its keep the moment
a genuine mma-fragment-order repack lands, per `OPTIMIZATION_LOG` backlog), but the default
is off and the 71.9 GB file is deleted. **Gate L1's "second start < 60 s" is met by the cold
path itself.**

### Note for the KV budget

52.07 GB free after weights, before the 2.23 GB draft. That is *more* headroom than
`RESCOPE.md` §4 assumed (36.87 GB), so the KV capacity figure there is conservative;
it will be restated once the draft and activation buffers are real.

---

## Gate B1 (in progress) — kernels

Harness: `tests/gate_kernels.cu` against references dumped from the validated oracle by
`oracle/dump_kernel_refs.py` (real checkpoint weights, real hidden states from the golden run).

### G1 — NVFP4 dequant · **PASS, BIT-EXACT**

`kernels/gemm.cu:k_dequant_nvfp4` vs `oracle/ref_laguna.py:dequant_nvfp4` on
`model.layers.1.mlp.experts.0.gate_proj` [1024, 3072]:
**0 mismatching elements out of 3 145 728.** Both sides are exact products of exactly
representable values (E2M1 code × E4M3 scale × fp32 reciprocal), so bit-exactness is
attainable here and is the right bar.

### G2 — dense linear, BF16 weights · **PASS** (maxrel 7.3e-07)
### G2b — dense linear, NVFP4 weights · **PASS** (maxrel 1.5e-06)
### G3a — RMSNorm · **PASS** (maxrel 1.4e-07)
### G3b — rope tables, both layer types · **PASS** (maxrel ≤ 6.7e-06)

`__cosf`/`__sinf` fast-math intrinsics account for the 1e-6-class residual; tolerance 3e-5.

### G4 — sigmoid router, selection-only bias, top-10 · **PASS, INDEX-EXACT**

**0 mismatching indices out of 540**; normalised weights maxrel 7.4e-07.

Two failures on the way, both worth recording.

**Failure 1 — shared-memory staging does not scale (535/540 wrong).** The first GEMM staged
`x` as a `[M,K]` bf16 tile in shared memory. At the verify shape M=6 that is 36 KB and works;
at prefill M=54 it is **324 KB**, over the 228 KB limit, so the launch silently failed and the
router consumed garbage logits. Fixed by removing the staging entirely and reading `x` from
global — which is also what gemma independently measured as *faster* (+3 %: every block reads
the same small `x`, so it is hot in L2 and the shared hop is pure overhead). M is now tiled by
`MAXM=8` via `grid.y`, so any M works.

**Failure 2 — the precision contract, and it is not a tolerance question.** With the launch
fixed, 15 of 540 selections still differed. Cause: this oracle runs **fp32 activations
everywhere**, while the deployed model is **bf16** (`config.torch_dtype`). Measured directly:

| | |
|---|---|
| selection flips, fp32-act vs bf16-act reference | **15 / 540** — exactly what the kernel showed |
| tokens affected | 7 / 54 |
| router-score gap between rank-10 and rank-11 | **min 1.03e-05**, median 3.9e-03 |
| max resulting routing-weight delta | 3.5e-04 |

So the kernel was already reproducing the *correct* (bf16-activation) reference bit-for-bit;
the oracle was more precise than the model, not more right. The flips are between experts
whose router scores are tied to 1e-5 — numerically meaningless, but they would have looked
like a correctness bug forever.

**Contract adopted:** kernel gates compare against a **bf16-activation, fp32-accumulate**
reference, because that is what the checkpoint is and what poolside serves. The fp32 oracle
remains the *math* reference (it is what validated the transcription against the shipped
`modeling_laguna.py`). Carried forward to G8: the full-forward gate needs a bf16 golden run,
not the fp32 one, or a stated tolerance that accounts for exactly this effect.

**Standing note on router sensitivity:** with 256 experts and top-10, near-ties at the
selection boundary are common (min gap 1e-5 on a 54-token sample). Any future change to
activation precision on the router path — including the §5 self-quantisation lever — must
re-measure selection agreement, not just output norms.

### G5 — head-packed GQA over FP8 KV, both tilings · **PASS**

`kernels/attention.cu`. maxrel **1.67e-07** (global, G=6) and **1.49e-07** (sliding, G=9),
against a reference that FP8-round-trips K/V with the checkpoint's static scales first, so the
gate isolates attention from the deliberate FP8 loss.

**Failure on the way — a silent 31/32 data loss.** The cross-warp softmax combine buffered
each warp's partial as `red[NW][G][{m,l,acc[4]}]`. But `acc[4]` is *per-lane* (each lane owns
4 distinct head-dim slots), so all 32 lanes of a warp wrote the same 4 shared floats: a race
that keeps one lane's partial and discards the other 31. Output was ~170 % wrong. Fixed by
carrying the head-dim axis: `red_acc[NW][G][HD]`, which also forced `NW` 8 → 4 to keep shared
memory at 23 KB. The lesson generalises — **any per-lane accumulator crossing a shared-memory
reduction must keep its lane axis.**

Known and deliberate limitation: at decode (M=1) the grid is (1, nkv) = **8 blocks on 20 SMs**,
so attention is grid-starved in exactly the shape that matters most. Logged for the
optimisation loop (split the key range across blocks with a second-stage combine); not fixed
now because correctness gates precede speed gates.

### G6 — grouped routed-expert MoE · **PASS**

`kernels/moe.cu`. maxrel **3.00e-05** on real layer-1 experts and a real hidden state,
8 tokens × top-10. Structure: invert → weight-resident grouped gate/up → weight-resident down
→ deterministic finalize. **No atomics** (float atomics would make the verify step disagree
with itself run to run), and the finalize sums in **ascending expert index** to match the
reference's `expert_hit` iteration order.

Tolerance is looser here (3e-5) than elsewhere because `h = silu(gate)·up` is rounded to bf16
before `down_proj`, which the reference mirrors — that rounding is the dominant residual.

**It also produced a real measurement:** 44 active experts of 256 for 8 tokens, against 70.5
from the independent-uniform union model. That prompted the full `E_frac` measurement now in
`ROOFLINE.md` §10, which raises `k*` and improves every speculative projection.

### G7/G8 — full autoregressive forward · **PASS**

`src/forward.cu` + `tests/gate_forward.cu`. The pure-CUDA forward reproduces the oracle's
greedy continuation **exactly, 8/8 tokens**:
`33586 81 397 874 367 2440 330 2833` → `"Okay, I need to write a Python"`.

G7 note: `lm_head` is **not** tied to the embedding (`tie_word_embeddings: false`), so
gemma's tied-embed quantization trick does not apply. It is a separate BF16 `[100352, 3072]`
tensor and is loaded and used as one. Quantizing it is part of the §5 lever, not a free port.

### G8 reconciliation — the directive requires closing any >20 % gap before proceeding

First working version: **3.66 tok/s** against a 22.6 tok/s roofline — 16 % of it, ~37 GB/s
effective. Profiled before touching anything (`tests/bench_kernels.cu`, real decode shapes):

| kernel @ decode shape | bytes | ms | GB/s | % of 227 |
|---|---:|---:|---:|---:|
| `q_proj` sliding [9216,3072] | 56.6 M | 0.415 | 136.5 | 60 % |
| `o_proj` sliding [3072,9216] | 56.6 M | 0.405 | 139.9 | 62 % |
| `lm_head` [100352,3072] | 616.6 M | 2.518 | 244.8 | — |
| **expert gate FP4 [1024,3072]** | 1.8 M | 0.162 | **11.0** | **5 %** |
| **attn sliding win=512 M=1** | 1.0 M | 0.574 | **1.8** | **1 %** |
| **attn global ctx=4096 M=1** | 8.4 M | 2.206 | **3.8** | **2 %** |
| per-launch overhead | — | 4.15 µs | — | — |

Three fixes, each from the profile rather than from guessing:

**1. FP4 dequant was hitting local memory.** `e2m1f()` indexes a `const float t[8]` by a
runtime code; in a GEMM inner loop that array lands in **local memory**, so every weight cost
a stack load. Replaced with the hardware converter `__nv_cvt_fp4x2_to_halfraw2` (both nibbles
per call). **11.0 → 27.1 GB/s, 2.5×**, correctness unchanged. This is gemma's
"C_LUT → HW cvt" win, but worth 2.5× here rather than 0.5 % because the LUT was on the stack.

**2. Attention was grid-starved by construction.** One block per (query, kv_head) means
decode launches **8 blocks on 20 SMs** — the GPU is 60 % idle by design, and it measured 1–2 %
of roofline. Added a flash-decoding split over the key range (`k_attn_split` +
`k_attn_combine`), with the split chosen to land at ~4 blocks/SM. Pure scheduling change; the
combine is still deterministic.

**3. The MoE grid was sized by `MAXTOK`, not by the actual token count.** At decode M=1 there
are at most 10 active experts, but the grid was `min(E, MAXTOK·topk) = 256` wide — so ~96 % of
blocks existed only to read `*nactive` and exit.

| | before | after |
|---|---:|---:|
| decode | 3.66 tok/s | **9.00 tok/s** |
| prefill | 4.5 tok/s | **16.0 tok/s** |
| greedy match vs oracle | 8/8 | **8/8** |
| effective bandwidth | ~37 GB/s | **~90 GB/s** |

**2.46× with correctness held.** 90 GB/s is now level with the effective bandwidth gemma's
champion path achieved (91 GB/s), and 40 % of the 227 GB/s ceiling — so ~2.5× of headroom
remains. The gap is no longer "unreconciled": it is the MoE FP4 GEMM (27 GB/s = 12 %), the
BF16 GEMM (136 GB/s = 60 %), and ~1665 kernel launches per step with no CUDA graph yet.

---

## Gate B1c — batched prefill ≡ sequential decode · **PASS** · 2026-07-25

`tests/gate_longctx.cu` prefills P tokens in batches of `MAXTOK`, then feeds the *same* P
tokens one at a time, and requires the last logits to agree. Single-token decode is a
trustworthy oracle here with no Python and no golden tensors: 512 consecutive positions map
to 512 distinct sliding-ring slots, so M=1 is provably free of the aliasing the batched path
can suffer.

It failed, and the failure was instructive.

| P | MAXTOK | last-logit maxabs | argmax batched / sequential |
|---:|---:|---:|---|
| 2 | 2 | 0.0000e+00 | 66721 / 66721 |
| 4 | 4 | 8.5791e-01 | **57413 / 29046** |
| 8 | 8 | 5.6977e+00 | 56869 / 56869 |

Ruled out first, each by direct measurement rather than argument:

* **The two attention kernels.** `gate_attn_split` compares `attend` against `attend_split`
  bit-exactly across G ∈ {6,9}, len ∈ {64,200,512,900}, NSP ∈ {2,4,8,16} — 0.000e+00.
* **The sliding-window ring.** P=300 is entirely below the 512 window and still failed.
* **The `attb` overflow** (sized `MAXTOK·maxq`, used for the layer-0 dense intermediate).
  Real bug, fixed, gate still failed.

A per-layer capture of the hidden state for the *same* token then localised it:

```
layer        maxabs          rel
0        7.4506e-09   1.0912e-08  <-- diverges
```

**7.45e-09 at layer 0 is fp32 rounding, not an index error.** An indexing bug produces O(1)
garbage. What the table above actually shows is a 1-ulp seed amplified by the residual stream
at roughly 1.4× per layer — 1.4^48 ≈ 4e7, which takes 1e-8 to 1e-1 by the time it reaches the
logits, and flips a knife-edge argmax.

### Root cause

`attend_nsplit(M, nkv, len)` sized the key split as `ceil(80/(M·nkv))` — targeting ~80 blocks
for the *current* shape. So the same token was computed with **10 key splits at decode M=1 and
3 at a batched M=4**. The split count chooses the partition of the key axis, and the
flash-decoding combine is a chain of fp32 online-softmax rescales, which is not associative:
a different partition is a different rounding of identical inputs. (P=2 passed by luck — 5
splits happened to round the same way.)

Confirmed by pinning it, which is the cleanest possible A/B:

| forced NSP | layers 0–47 | last logits | argmax |
|---|---|---|---|
| 1 | all 0.0000e+00 | 0.0000e+00 | 350 / 350 |
| 8 | all 0.0000e+00 | 0.0000e+00 | 29046 / 29046 |

**Verdict: no bug in the batched path.** It was arithmetically correct the whole time; the two
paths were simply given different reduction orders to compare.

### Fix

`attend_nsplit` now ignores `M` and sizes the split from the key length alone
(`clamp(ceil(80/nkv), 1, ceil(len/64), 32)`), which is exactly the old value at the decode
shape — so decode numerics and decode speed are untouched — and makes every other shape match
it. Decode, batched prefill and speculative verify are now the same arithmetic.

This matters well beyond the gate: **a DFlash acceptance rate is only meaningful if the verify
pass at M=k+1 reproduces the decode pass at M=1 exactly.** Otherwise the speculative path and
the autoregressive path are different models and τ measures their disagreement as much as the
draft's quality.

### Result — 10/10 bit-exact

| MAXTOK | P | verdict |
|---:|---|---|
| 4 | 4, 8, 300, 700 | 0.0000e+00 ×4 |
| 16 (DFlash verify shape) | 17, 300, 700 | 0.0000e+00 ×3 |
| 64 (bulk prefill) | 300, 700, 1200 | 0.0000e+00 ×3 |

700 and 1200 sit above the 512 sliding window, so the `cap = window + MAXTOK` ring fix is
exercised at all three batch widths.

### Second finding, recorded because it constrains how every later gate is written

On **uniform-random token IDs** this model amplifies 1 ulp into an argmax flip. That is not a
defect: random IDs are maximally off-distribution, the 256-way sigmoid router sits near
uniform, and the top-10 selection is then a coin toss between experts whose scores differ in
the last bits. It does mean a bit-exactness gate on random tokens is the *strictest possible*
form of the test — which is why it is worth keeping — and that any future gate comparing two
numerically-different-but-both-valid paths must compare against the oracle, never against each
other. `LG_TEXT=tests/data/sample.txt` switches the same gate to real prose for the
in-distribution case.

---

## Gate D1 — DFlash speculative decoding · **PASS (correctness), QUALIFIED (speedup)** · 2026-07-25

`tests/gate_dflash.cu`. Two independent things had to hold, and the second is the one that
does the work:

* **Correctness.** Greedy speculative decoding must emit the *exact* token stream greedy
  autoregressive decoding emits. **8/8 against the golden continuation at every k from 1 to
  15.** Gate B1c is what makes this meaningful: the verify pass runs at M=k+1 and decode at
  M=1, and those only became bit-identical once `attend_nsplit` stopped depending on M.
* **Acceptance.** τ = accepted tokens per target forward. **4.32 at k=15**, inside poolside's
  published 4.02–6.44 band. Nothing in the shipped checkpoint documents where the six
  `aux_hidden_norms` apply or which end of the layer the taps come from; τ is the only
  evidence that those choices are right, and it is strong evidence.

### The bug that only τ could catch

`Drafter::init` sized every scratch buffer for the 16-row query block, while `context_kv` ran
over the whole 512-wide draft window. `rope_tables`, `hb`, `fcx`, `kk` and `vv` all wrote past
their ends. It does not crash. The draft simply emits **one constant token forever** — and the
correctness gate **passes**, because every draft token is rejected and the target's own token
is committed instead.

> **A correctness test cannot see a broken speculator.** Rejection is indistinguishable from a
> conservative draft. τ = 1.000 was the only signal, which is why it is a gate and not a
> statistic.

### The k sweep, and the finding that matters

96 generated tokens, ctx 4096, `E_frac` = mean fraction of the 256 experts a forward touches:

| k | tok/s | τ | E_frac | GB/accepted token |
|---:|---:|---:|---:|---:|
| 1 | 25.42 | 1.863 | 0.078 | 6.70 |
| 2 | 28.70 | 2.500 | 0.103 | 5.61 |
| **3** | **29.28** | 2.969 | 0.123 | 5.15 |
| 4 | 28.65 | 3.393 | 0.141 | **4.86** |
| 6 | 24.27 | 3.654 | 0.169 | 5.00 |
| 8 | 20.36 | 4.130 | 0.197 | 4.86 |
| 15 | 14.29 | 4.318 | 0.264 | 5.64 |

**k\* = 3–4, not the 7 the model card recommends or the 15 poolside benchmark at.** And the
speedup is **1.07×** (29.28 against 27.35 base), where the byte budget says 1.48×
(7.20 → 4.86 GB per accepted token).

Two things cause that, and the byte model in `ROOFLINE.md` §4 missed both:

1. **The MoE expert blow-up on the verify side.** Verifying k+1 tokens routes (k+1)·10
   assignments, and the MoE kernel pays for the *distinct* experts. E_frac goes 0.039 (decode)
   → 0.141 at k=4 → 0.264 at k=15. Measured in the profile: `k_moe_gateup_rp` costs 200 µs
   per layer at M=1 and **815 µs at M=5**. MoE alone is 62 ms of the ~119 ms verify+propose
   step. This is why τ rising past k=4 does not buy throughput — the experts are eating the
   tokens the draft is winning. The roofline modelled E_frac (0.18 predicted at k=5, 0.155
   measured) but treated the verify pass as costing one decode step.
2. **The dense GEMM is issue-bound at M>1.** The fused q/k/v/g reads 34.8 MB in 293 µs at
   M=5 (119 GB/s) against 230 GB/s at M=1. Per token that is still 2.6× better, but it means
   the dense half of a verify costs ~2× a decode step, not 1×.

Net: the spec loop runs at **151 GB/s** where the M=1 CUDA graph runs at 197.

### A hypothesis that was wrong, recorded because the reasoning was plausible

The M>1 GEMM looked x-bound: each lane reads VEC bytes of weight against M·32 of activation,
and all four warps in a block re-read the same x. Staging x in shared memory, tiled over K to
avoid the 324 KB blow-up that killed staging at prefill, measured **1.8× slower** —
0.271 → 0.492 ms on FP8 q_proj at M=6. The barriers cost more than the L2 re-reads they save.

The hypothesis also rested on a misread: BF16 at M=6 is already at 196 GB/s (87 % of ceiling),
and only FP8 looked bad at 104. But this kernel is *issue*-bound at M=6, so FP8 moving half
the bytes in the same time reads as half the bandwidth. In wall clock, FP8 at M=6 (0.271 ms)
is **faster** than BF16 (0.288) for identical N and K. There was no defect.

> **On an issue-bound kernel, GB/s is a misleading metric.** Compare times.

### Status

D1 passes on correctness and on acceptance. The speedup is real but small, and the reason is
structural rather than a bug: a 256-expert top-10 MoE is close to the worst case for
speculation, because every extra speculative token widens the expert set the verify pass must
read. Recorded in the backlog: N-blocking the dense GEMM at M>1 (amortise the x loads over 2–4
output rows per warp) is the concrete ~+12 % lever, and capturing the verify forward in its own
CUDA graph is worth the ~8 % the decode graph is already measured to give.

---

## Gate S1 — server layer · **PASS** · 2026-07-25

`src/server.cu` (`lgserve`), `tools/lgchat.cc`, `include/webui.h`. One process, no Python on
any path.

| surface | state |
|---|---|
| `POST /v1/chat/completions` | streaming SSE and non-streaming |
| `GET /v1/models`, `GET /healthz` | ✅ |
| WebUI at `/` | 6.8 KB, self-contained, light/dark |
| reasoning separation | `reasoning_content` deltas until `</think>`, then `content` |
| tool calling | verified end to end: `finish_reason: tool_calls`, OpenAI-shaped |
| prefix cache | 2nd turn of a conversation: 0.79 s vs 19.8 s cold |
| terminal client | `lgchat`, same SSE stream, reasoning dimmed |
| context | `CTX=262144` → KV 6.48 GB, process 78 GB of 122 |

### Adaptive speculation — the part that took the measurements to get right

D1 established that a verify forward costs ~3× a decode step, so speculation needs τ > ~3 to
pay. τ is content-dependent — 3.06 on a math prompt, **2.26 on open prose** — so a fixed k is
wrong in both directions. At k=3 on prose the server ran **20.7 tok/s against 27.4 for plain
decode: a 25 % loss**, while the same k on code was a 1.3× win.

So the mode is chosen by measurement. A bandit over {ar, k=2, k=3, k=5} keeps an EWMA of each
arm's achieved tokens/second.

Three iterations, each fixing something the previous one exposed:

1. **Per-step ranking thrashed.** A speculative step yields 1…k+1 tokens, so a single sample
   spans the entire acceptance distribution. Ranking on it cost 12 % on exactly the workload
   where speculation wins (code: 33.3 → 29.5 tok/s). Fixed by holding each arm for a **block
   of 32 steps** and ranking on the block average.
2. **Exploration was too expensive.** A block on a bad arm costs ~30 % of its tokens. Probes
   went from every 6 blocks to every 10, gated on the arm being within 0.65× of the best.
3. **Acceptance is non-stationary *within* a generation** — the same k=5 arm measured 37.4 then
   22.6 tok/s as one code answer drifted into prose. So a gated probe alone would strand a
   recovered arm; a full sweep of every arm runs every 50 blocks.

Measured, served, 500-token generations:

| workload | tok/s | vs AR floor |
|---|---:|---:|
| code (red-black tree) | **33.6** | 1.28× |
| prose (technical explanation) | 28.3 – 32.0 | 1.02 – 1.16× |
| AR arm alone | 26.1 – 28.1 | 1.00 |

### The autoregressive arm had to be graphed, and that needed a kernel change

The bandit picks AR on prose, so AR *is* the chat path. It was running ungraphed at 24.0 tok/s
where `bench_decode` gets 27.4. The blocker was `tap_store`, which took the position as a host
`int`: a host int is frozen into a CUDA graph at capture, so every DFlash tap would have landed
in the captured position's ring slot forever. Changed to a device pointer, exactly as
`store_kv` already did. **24.0 → 26.1 tok/s, +8.5 %.** `base` is now unused inside `forward()`
— every position-dependent kernel reads `dbase`.

### Two things deliberately not done

* **Speculation under sampling.** Longest-prefix acceptance is only valid against an argmax
  target; the correct rule under temperature is the rejection sampler, which needs the draft's
  own distribution. Rather than silently change the output distribution, sampled requests
  decode autoregressively.
* **Concurrency.** One session at a time, on a mutex. Decode is bandwidth-bound at batch 1 on
  a 69 GB weight set; a second concurrent sequence would not add throughput, it would halve
  both sequences' latency. Requests queue.

---

## N2 + N3 — cost-aware speculation policy and a zero-weight drafter · 2026-07-25

### N2 — the throughput bandit is gone

The bandit ranked whole arms by *realised tokens/second*. That is the highest-variance signal
available: one speculative step yields between 1 and k+1 tokens, so a single sample spans the
entire acceptance distribution. Averaging it needed 32-step blocks, which made exploration cost
~30 % of a block and left the controller lagging acceptance that changes mid-generation.

The replacement measures the *low*-variance thing instead. Every verify already reports, for
free, whether position i was accepted **given** that 0..i−1 were — Bernoulli observations, one
per position, five per k=5 step. From per-position acceptance the expected yield of any k
follows in closed form, and the choice is a cost-weighted argmax:

```
E[tokens | k] = 1 + p0 + p0·p1 + … + Π_{i<k} p_i
k*            = argmax_k  E[tokens | k] / cost(k)          (AR scored as k = 0)
```

A step at **any** k updates the estimate for **every** k, so steady-state exploration cost is
zero. `cost(k)` is established by a one-time boot sweep and then tracked by EWMA.

### Three bugs, each of which made the controller worse than the thing it replaced

Worth recording in order, because each looked like a tuning problem and was actually a
measurement problem.

1. **`cost_ar` from a single sample.** Measured once during the boot sweep and never updated.
   It rated AR at 23.4 tok/s against a true 33, so a k=1 verify with **zero** acceptance looked
   cheaper than decoding, and the controller chose it on prose.
2. **The estimate could never recover.** Fixing (1) with an EWMA made it *worse* — 19.7 —
   because once the controller stopped choosing AR there were no AR steps to update it from.
   **An arm that is never selected is never re-measured.** Any cost feeding an argmax has to be
   established before the argmax can starve it. Fixed with a bounded boot sweep that samples AR
   four times, skipping the first, because the first AR step also captures the CUDA graph and
   is ~1.7× a steady one.
3. **A boolean where a position belonged.** `draft_ctx_current` meant every AR step invalidated
   the draft's context, so the next DFlash step rebuilt the entire 512-wide window — four
   chunks, each an fc GEMM of 113 MB plus six layers of K/V. Harmless when arms were held for
   32 steps; ruinous when the mode can change every token. Replaced with `draft_ctx_pos`, which
   bounds the rebuild to the tokens actually committed while the draft was idle.

After (3), the controller's own AR estimate reads **33.0 tok/s** — matching standalone
`bench_decode` exactly. That agreement is the signal that the cost model is finally calibrated.

### N3 — SuffixDecoding

A suffix automaton over prompt + generated tokens, indexing 3- to 12-grams. Marginal cost is a
hash lookup; it reads no weights at all. That matters here because Gate D1 established a verify
forward costs ~3× a decode step, so DFlash must clear τ ≈ 3 to pay — and after the FP8 work
raised base decode to 33 tok/s, that bar went up again. A drafter with no weight reads only has
to beat 1.0 accepted tokens.

**Two bugs, both of which made it silently never fire:**

* **Cold start.** An unobserved position priced at a 0.5 prior put the suffix arm at 31.5 tok/s
  against AR's 33.3 — just below, forever, so it was never chosen and never learned. Fixed with
  a bounded number of forced trials per source, per server lifetime.
* **Self-match.** `extend()` indexed *through* the current position, so `draft()` found the
  current position as its own "earlier occurrence" and fell through every length. Twelve forced
  trials, zero accepted tokens, `p[0]` pinned at 0. The index now stops one short.

With both fixed, `p_sfx` reads 0.92 / 1.00 / 1.00 on repetitive output.

### Measured

| workload | tok/s | accepted / forward | chosen arm |
|---|---:|---:|---|
| prose | **33.6** | 1.00 | AR — correctly declines to speculate |
| code | **40.1** | 3.67 | DFlash k=3 |
| repetitive / edit-style | **49.7** | **14.6** | DFlash k=15, suffix competitive at 47–50 |

All gates green throughout: 13/13 kernels, B1c bit-exact, D1 greedy 8/8 at every k.

**One honest limitation.** `p[]` is global across requests, so a hard workload switch costs
tens of steps to re-converge — visible as a first-request number below steady state. The EWMA
decay (0.92) sets that timescale deliberately: faster tracking would reintroduce the variance
the bandit suffered from.

---

## Per-position probabilities and the sampled path · 2026-07-25

The research pass listed "plumb the draft's per-position probabilities" as the prerequisite for
speculation under temperature. **It turned out not to be needed.** DFlash drafts *greedily*, so
the draft distribution q is a point mass on the drafted token, and the general rejection rule
collapses:

```
accept x  ⟺  u < p_target(x),   u ~ U[0,1)          (because q(x) = 1)
on reject, draw from norm((p − q)_+) = p with p(x) zeroed and renormalised
```

Only the **target's** probability of the drafted token is required, and we already read those
logits. This is exactly vLLM's `NO_DRAFT_PROBS` path and it is distribution-preserving: the
emitted stream has the same law as sampling from the target directly.

### Implemented, correct, and **off by default**

The rule works — measured acceptance 1.56–1.72 tokens/forward at T=0.7, where greedy
longest-prefix would be invalid. But the *policy* on top of it does not beat plain sampled
decode: acceptance is inherently lower and noisier at T>0, which pollutes the per-position
estimates, and successive runs oscillated 22.9 / 19.7 / 14.6 tok/s with the AR estimate
swinging 20.6 / 14.4 / 28.7. Shipping that on by default would trade a reliable ~30 tok/s for
an unreliable 15–25. `LG_SPEC_SAMPLED=1` enables it for further work.

### Three bugs, all in the sampler rather than the algorithm

1. **Sorting the vocabulary per token.** `top_p` filtering sorted all 100 352 logits on every
   sampled token — ~23 ms on a 30 ms step, dragging sampled decode to 18.9 tok/s. The
   acceptance test needs only `p(x)` for one token, which needs no ordering at all; ordering is
   now built only when a sample must actually be drawn, and then only over a bounded candidate
   set via `nth_element`.
2. **`double` and a separate normalise pass.** Rewritten single-pass in `float`, with
   normalisation folded into the consumer.
3. **The cost model was polluted across sampling regimes.** A sampled request takes the same AR
   branch but pays a full softmax on top of the model, and folding that into `cost_ar` made the
   policy believe plain decode costs 1/22.7 s when greedy decode costs 1/33 — so **one sampled
   request permanently mis-priced every greedy request that followed it on the same server**.
   Costs are now learned only from steps the policy actually governs.

That third one is the general lesson: an online cost model must only be fed measurements from
the regime it will be used to make decisions in.

### Greedy path, re-verified after all of the above

| workload | tok/s | accepted / forward |
|---|---:|---:|
| prose | 33.6 | 1.00 (declines to speculate; `ar` estimate 33.1 matches `bench_decode`) |
| code | 40.1 | 3.67 |
