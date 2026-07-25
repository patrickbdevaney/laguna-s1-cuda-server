# ROOFLINE.md — Gate R1

**Status: COMPLETE.** Contains measured `B_tok`, the AR wall, the k-sweep projection table, the
`E_frac` sensitivity, and a named `k*` prior.

Everything here is computed by `tools/roofline.py`, which reads `config.json`,
`model.safetensors.index.json`, and the actual safetensors tensor headers
(`tools/hdr_target.json`, `tools/hdr_draft.json`, 145 153 + 69 tensors). **No model constant is
hardcoded.** Raw output: `docs/roofline_ctx4096.txt`, `docs/roofline_ctx32768.txt`.

Verification anchor: the per-group byte sum reproduces the index's `total_size` of
**71 898 733 760 B exactly**, and the activated-parameter count reconstructs the model card's
**117.6 B total / 8.5 B activated** to three significant figures. The accounting is right.

---

## 0. Headline — read this first

> **Three of the directive's Phase-0 premises are wrong, and the corrections are large.**

1. **`B_tok` is 10.04 GB/token, not ~4.5 GB.** The directive's anchor (`8.5 B active ×
   0.531 B/param`) assumed the whole model is NVFP4. It is not. `quantization_config.ignore`
   excludes attention, the shared experts, the router, the layer-0 dense MLP, and `lm_head` —
   **only the 256 routed experts are quantized.** 3.71 B of the 8.5 B activated parameters are
   BF16. (The 4.5 GB figure is, however, exactly what we would get if *we* quantized the rest —
   see §5, which is the single biggest lever in this project.)

2. **The AR wall is ~20 tok/s, not 44.** `200 GB/s ÷ 10.04 GB = 19.9 tok/s`.

3. **vLLM is not leaving 1.5–2× on the floor.** With the correct `B_tok`, poolside's GB10
   measurement of 13–14 tok/s implies **~135 GB/s effective**, i.e. **~68 % of achievable
   streaming bandwidth** — a competent, near-roofline implementation. The directive computed
   ~59–63 GB/s only because it divided by the wrong `B_tok`. The stated opportunity of the
   project as originally framed does not exist in that form.

The **+10 %-over-vLLM heuristic from the gemma work does not transfer**, but for the opposite
reason to the one the directive gives: not because vLLM's Laguna path is weak, but because it is
already strong. Our headroom on the *stock* checkpoint is the gap from ~135 GB/s to ~175–200 GB/s
effective (≈ +30–48 %), plus a better choice of `k` — not a 2× algorithmic gap.

**Corrected expected band: 25–40 tok/s on the stock checkpoint** (code workload, at `k*`),
centred near **30**. With self-quantized attention (§5) the band moves to **32–48**, centred near
**38**. The directive's 32–52 band is reachable only *via* §5, and 50–70 remains off the table.

---

## 1. Ground truth read from disk

From `config.json` (nothing here is assumed):

| | value |
|---|---|
| `hidden_size` | 3072 |
| `num_hidden_layers` | 48 |
| `head_dim` | 128, `num_key_value_heads` = 8 |
| `num_attention_heads_per_layer` | **48 on global layers, 72 on sliding layers** |
| `layer_types` | 12 `full_attention` (layers 0,4,8,…,44) + 36 `sliding_attention`, 3:1 |
| `sliding_window` | 512 |
| `num_experts` / `num_experts_per_tok` | 256 / **10** |
| `moe_intermediate_size` | 1024 (shared expert also 1024) |
| `mlp_only_layers` | `[0]` → layer 0 is dense, `intermediate_size` 12288; 47 MoE layers |
| `vocab_size` | **100 352** (not 262 144) |
| `tie_word_embeddings` | **false** — `lm_head` and `embed_tokens` are separate BF16 tensors |
| `moe_routed_scaling_factor` | 2.5, `norm_topk_prob` true |
| NVFP4 `group_size` | **16**, confirmed: `weight_scale` `[1024,192]` for K = 3072 |
| KV cache | FP8 e4m3, per-tensor static scale (`k_scale`/`v_scale` shipped per layer) |
| DFlash `block_size` | **16** → `k` ranges 1…15, and the draft is **one forward per block** |

Quantization coverage, from `quantization_config`:

| BF16 (unquantized) | NVFP4 |
|---|---|
| `q_proj`, `k_proj`, `v_proj`, `o_proj`, `g_proj` (all 48 layers) | `experts.{0..255}.{gate,up,down}_proj` (47 layers) |
| `mlp.gate` (router), `shared_expert.*`, layer-0 dense MLP, `lm_head`, `embed_tokens`, all norms | — |
| **15.2 GB of checkpoint** | **63.9 GB of checkpoint (88.8 %)** |

Effective expert weight density: 4.5 bits/param (4-bit E2M1 + one E4M3 scale per 16) =
0.5625 B/param — same as gemma's NVFP4, **not** the 0.531 the directive assumed.

---

## 2. Bytes per token (§3.1)

Per decode step, at ctx = 4096:

| component | GB | dtype |
|---|---:|---|
| attention `q,k,v,o,g` × 48 layers | **5.6063** | BF16 |
| norms (input/post/q/k + final) | 0.0006 | BF16 |
| layer-0 dense MLP | 0.2265 | BF16 |
| MoE routers ×47 + `e_score_correction_bias` | 0.0740 | BF16/F32 |
| shared experts ×47 | 0.8871 | BF16 |
| `lm_head` (not tied) | 0.6166 | BF16 |
| embedding row | ~0 | BF16 |
| **k-independent FIXED subtotal** | **7.4110** | |
| routed experts, top-10 × 47 layers | 2.4950 | NVFP4 |
| FP8 KV @ ctx 4096 | 0.1384 | FP8 |
| **B_tok** | **10.0444** | |

One routed expert = 5.308 MB (packed FP4 + E4M3 group scales + FP32 globals). Whole routed pool
= 63.87 GB.

### ⚑ The structural finding: Laguna is attention-bound at bs = 1, not expert-bound

Attention weights are **5.61 GB — 56 % of `B_tok`** and 2.2× the entire top-10 expert read.
Per sliding layer, `q_proj` + `o_proj` alone is `2 × 9216 × 3072 × 2 B` = **113 MB**, more than
double that layer's 53 MB of routed experts. This is a direct consequence of 72 attention heads
on 36 of 48 layers with everything left in BF16.

This inverts the directive's threat model. The expert-union blow-up (§4) is a second-order
effect; the first-order cost is a BF16 attention stack that no amount of MoE kernel work touches.

### FP8 KV and the SWA win (§3.1)

`2 × 8 kv-heads × 128 head_dim × 1 B` = **2048 B per token per layer**.
- 12 global layers scale with context: **24 576 B/token**.
- 36 sliding layers are capped at the 512-token window: **37.7 MB per sequence, constant, forever.**

| ctx | KV read/step | `B_tok` | @91 | @135 | @200 GB/s |
|---:|---:|---:|---:|---:|---:|
| 2 048 | 0.088 | 9.994 | 9.11 | 13.51 | **20.01** |
| 4 096 | 0.138 | 10.044 | 9.06 | 13.44 | **19.91** |
| 8 192 | 0.239 | 10.145 | 8.97 | 13.31 | 19.71 |
| 32 768 | 0.843 | 10.749 | 8.47 | 12.56 | 18.61 |
| 131 072 | 3.259 | 13.165 | 6.91 | 10.25 | 15.19 |
| 262 144 | 6.480 | 16.386 | 5.55 | 8.24 | 12.21 |

**Quantifying the SWA win as the directive asks:** if all 48 layers were global, KV would be
98 304 B/token → 25.8 GB/step at 262 K. The 3:1 split cuts that to 6.48 GB — a **4.0× reduction
in both KV bandwidth and KV capacity**, and it is why long context degrades gracefully
(262 K costs only 1.6× the `B_tok` of 4 K instead of 3.6×). It is permanent and free.

---

## 3. Autoregressive decode projections (§3.2)

| scenario | effective BW | projected AR decode |
|---|---|---|
| gemma champion path's achieved BW | ~91 GB/s | **9.1 tok/s** |
| poolside's measured GB10 efficiency | ~135 GB/s | **13.4 tok/s** |
| Thor achievable streaming ceiling | ~200 GB/s | **19.9 tok/s** |

> **19.9 tok/s is the autoregressive wall on the stock checkpoint. Nothing gets past it except
> speculation — or reading fewer bytes (§5).**

**Calibration.** At 135 GB/s the model predicts 13.4 tok/s AR; poolside measured **13–14** on
GB10. At the same 135 GB/s it predicts 26.1 tok/s for HumanEval-class code with DFlash at `k*`;
poolside measured **22–24**. And 21.1 for MT-Bench-class prose at `k*`; poolside measured **15**
(they ran `k=15`, where the model predicts 13.9 — see §4). The byte model reproduces all three
independent measurements. It is trustworthy.

Note the gemma path's 91 GB/s is a *pessimistic* import: gemma's per-layer work was far smaller
(H = 2816 vs 3072 but 30 layers vs 48, and 128 experts × top-8 at `MOE_INT` 704). Laguna's larger
per-call GEMMs amortise launch overhead better. Planning number: **150–175 GB/s achievable**.

---

## 4. The expert-union problem (§3.3)

Model: with `k` verify positions each selecting top-10 of 256, the expected distinct experts per
MoE layer under independent-uniform routing is `U(k) = 256 · (1 − (1 − 10/256)^k)`. This is an
**upper bound** — real routing is correlated between adjacent tokens.

| k | U(k) of 256 | `E_frac(k)` | expert GB | fixed GB | block GB (incl. draft) |
|---:|---:|---:|---:|---:|---:|
| 1 | 10.0 | 0.039 | 2.495 | 7.411 | 12.891 |
| 3 | 28.8 | 0.113 | 7.196 | 7.411 | 17.592 |
| 5 | 46.2 | 0.181 | 11.538 | 7.411 | 21.934 |
| 7 | 62.3 | 0.243 | 15.546 | 7.411 | 25.942 |
| 9 | 77.1 | 0.301 | 19.248 | 7.411 | 29.644 |
| 11 | 90.8 | 0.355 | 22.666 | 7.411 | 33.062 |
| 15 | 115.2 | **0.450** | 28.737 | 7.411 | 39.133 |

**`E_frac` never exceeds 0.45 even at `k`=15** — well below the directive's feared 0.7, because
10/256 is a narrow slice. The directive's premise that "doubling the expert count roughly doubles
the union's spread" is not how the union behaves: what matters is `top_k/N` = 0.039, and at that
ratio the union grows nearly *linearly* in `k` over the whole usable range. So there is no
saturation knee to exploit — but also no blow-up.

The real cost driver is absolute, not fractional: expert bytes go from 2.50 GB at `k`=1 to
28.74 GB at `k`=15, while the 7.41 GB fixed term amortises. Speculation pays only while the
amortisation of `FIXED` outruns the linear growth of expert bytes — and because `FIXED` is only
7.41 GB, that runs out early.

**Draft cost.** The DFlash draft is 2.230 GB of BF16 weights and has **no `embed_tokens` and no
`lm_head`** — it shares the target's, exactly as the gemma draft did. Charging the target
`lm_head` (0.617 GB) for draft logits, the propose step is **2.847 GB per block**, one forward
regardless of `k` (`block_size` semantics: "size of the draft block predicted with a forward pass
of the model"). That is a 28 %-of-an-AR-step fixed tax on every block, and it is why very small
`k` does not win.

### k-sweep projections, ctx = 4096

`τ(k)` fitted as `(1−α^(k+1))/(1−α)` from poolside's published acceptance lengths. **Those are
measured at `num_speculative_tokens=15`, not 7** — the directive misread this; the draft card's
benchmark header says `TP=2, temperature=0, num_speculative_tokens=15`. Fitting at `k`=15 gives
α = 0.858 (HumanEval), 0.837 (GSM8K), 0.763 (MBPP), 0.754 (MT-Bench).

**@ 135 GB/s (poolside-equivalent efficiency), AR baseline 13.4:**

| workload | α | k=1 | k=3 | k=5 | k=7 | k=9 | k=11 | k=15 | k* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HumanEval T=0 | 0.858 | 19.5 | 24.8 | **26.1** | 25.9 | 25.1 | 24.2 | 22.2 | **5** |
| GSM8K T=0 | 0.837 | 19.2 | 24.0 | **24.8** | 24.2 | 23.2 | 22.1 | 19.9 | **5** |
| MBPP T=0 | 0.763 | 18.5 | **21.4** | 20.9 | 19.5 | 18.0 | 16.6 | 14.4 | **3** |
| MT-Bench T=0 | 0.754 | 18.4 | **21.1** | 20.4 | 18.9 | 17.4 | 16.0 | 13.9 | **3** |
| code T=0.7 (τ 3.0) | 0.682 | 17.6 | **18.9** | 17.4 | 15.6 | 14.0 | 12.7 | 10.8 | **3** |

**@ 200 GB/s (hard ceiling), AR baseline 19.9:**

| workload | k=3 | k=5 | k=7 | k=15 | k* | best |
|---|---:|---:|---:|---:|---:|---:|
| HumanEval T=0 | 36.7 | **38.6** | 38.4 | 32.9 | 5 | 38.6 |
| MT-Bench T=0 | **31.3** | 30.2 | 28.0 | 20.5 | 3 | 31.3 |
| code T=0.7 | **28.0** | 25.8 | 23.1 | 16.1 | 3 | 28.0 |

**@ 91 GB/s (gemma-equal, pessimistic):** best is 17.6 (HumanEval, k=5) down to 12.8 (code T=0.7, k=3).

### `k*` prior — the named deliverable

> **`k*` = 5 for temp-0 code, `k*` = 3 for temp-0.7 code and for prose. Enter the sweep with
> `k*` = 4 as the single-value prior; test {2,3,4,5,7} densely and {9,11,15} sparsely.**

Both model cards are wrong for this hardware. The NVFP4 card's `k`=15 is a *datacenter TP=2*
setting where the AR baseline is latency-bound and speculation converts idle time into work; on
a bandwidth-bound unified-memory part `k`=15 is **the worst usable choice at every temperature**
(it costs 39.1 GB/block for τ ≤ 6.4). The draft card's `k`=7 is the better prior of the two and is
within 1 % of optimum at temp 0 — but at temp 0.7, the workload that actually matters, `k`=7 is
17 % worse than `k`=3.

This also explains poolside's own numbers: their 3.69× HumanEval speedup at `k`=15 is a
datacenter result; on GB10 the same pair gave only 22–24 vs 13–14 = **1.6–1.75×**, which is what
this model predicts. **Quote 1.6–1.9× as the achievable speculative multiplier on Thor, not 3.7×.**

### `E_frac` sensitivity to routing correlation

The independent-uniform union is the pessimistic end. Shrinking the spread toward the top-10
floor (code T=0.7, 135 GB/s):

| correlation | k=3 | k=5 | k=7 | k=11 | k=15 |
|---:|---:|---:|---:|---:|---:|
| 0.0 (independent) | 18.9 | 17.4 | 15.6 | 12.7 | 10.8 |
| 0.2 | 20.0 | 19.0 | 17.4 | 14.5 | 12.5 |
| 0.4 | 21.2 | 20.9 | 19.5 | 16.8 | 14.8 |

Correlation is worth up to +37 % at `k`=7 and moves `k*` upward. **This is the largest single
unknown in the projection**, and it is exactly what §7.3 of the directive requires be replaced
with a measurement: instrument the count of distinct experts per verify block and rewrite this
section. Until then all speculative numbers are quoted at correlation 0 (pessimistic).

---

## 5. The lever the directive does not name: self-quantizing the BF16 remainder

Poolside quantized only the routed experts. The other 15.2 GB of the checkpoint — and 7.41 GB of
every decode step — is BF16. We already own a validated NVFP4 path from the gemma work.

| scenario | FIXED | `B_tok` | AR@135 | AR@200 | spec@135 | spec@200 |
|---|---:|---:|---:|---:|---:|---:|
| stock (poolside) | 7.411 | 10.044 | 13.4 | 19.9 | 26.1 | 38.6 |
| FP8 attention only | 4.608 | 7.241 | 18.6 | 27.6 | 29.9 | 44.3 |
| NVFP4 attention only | 3.381 | 6.015 | 22.4 | 33.3 | 32.1 | 47.6 |
| FP8 everything non-expert | 3.706 | 6.339 | 21.3 | 31.5 | 31.4 | 46.5 |
| **NVFP4 everything non-expert** | 2.085 | **4.718** | **28.6** | **42.4** | **35.5** | **52.6** |

(spec column = HumanEval T=0 at that scenario's `k*`.)

Note that full self-quantization lands `B_tok` at **4.72 GB — the directive's 4.51 GB anchor.**
The anchor was right for a *uniformly* NVFP4 Laguna; poolside simply did not ship one.

**This is a +39 % to +113 % lever on AR decode and it is the highest-EV item in the project.** It
is not bit-exact against the oracle, so it must be gated on quality, not on bit-equality, and it
must come *after* the bit-exact stock path passes G1–G9. Attention is more quantization-sensitive
than MLP, so the staged order is: FP8 `o_proj` → FP8 all attention → evaluate → NVFP4 attention.
Logged as the leading entry in `OPTIMIZATION_LOG.md`. It also frees ~10 GB of resident memory.

---

## 6. KV capacity

With the 36 sliding layers held in fixed 512-entry rings, KV costs **24 576 B/token** (global
layers only) plus a constant **37.7 MB/sequence**.

| KV budget | context capacity |
|---|---|
| 30 GB | 1.22 M tokens |
| 40 GB | 1.63 M tokens |
| available 40.9 GB | **~1.66 M tokens** |

Poolside reports 830–870 K KV tokens at the 256 K setting on a 128 GB box. **We project ~1.66 M —
roughly 2×** — because the SWA ring makes 36 of 48 layers cost nothing beyond 512 entries. Even
holding the full 262 144-token context, one sequence needs only 6.48 GB, so the box supports ~6
concurrent max-length sequences, or the 1 M-token restored-rope path at 24.6 GB for a single
sequence with room to spare.

---

## 7. What this means for the plan

1. **Priority 1 is not the MoE GEMM.** The directive's §9 ordering puts MoE grouped-GEMM
   bandwidth first. At bs = 1 the routed experts are 25 % of `B_tok` and attention is 56 %. The
   correct first lever is the attention path — its BF16 GEMV/GEMM efficiency, and then §5.
   MoE remains the priority *within* the verify step at larger `k`, where expert bytes dominate.
2. **`k*` ≈ 3–5, not 7 or 15.** Design the verify path and its CUDA graphs for `M` ≤ 6, which is
   comfortably inside the `mma.sync m16n8k16` M ≤ 16 regime the gemma tc kernel already targets.
   The tcgen05 rejection holds a fortiori.
3. **Speculation is worth 1.4–1.9×, not 3×.** Combined with §5 the realistic serving target is
   **32–48 tok/s** on code, versus poolside's 22–24 on equivalent hardware — a 1.4–2.0× win that
   is defensible and publishable. Anything quoting 50–70 is not supported by the byte budget.
4. **Long context is cheap here.** The 4× SWA KV win means the 262 K path is genuinely usable and
   the prefix-cache feature (§8 of the directive) sits on top of an unusually favourable KV
   budget. Prefix caching remains the highest-leverage *feature*, independent of tok/s.
5. **48 layers × ~14 kernels is a launch-overhead risk on 20 SMs.** Gemma's whole-step CUDA graph
   is mandatory here, not optional.

## 8. Assumptions to retire with measurements

| assumption | how it gets replaced | section to rewrite |
|---|---|---|
| routing correlation = 0 | instrument distinct experts per verify block (Gate D1) | §4 |
| draft = one forward per block | confirm against the reference DFlash loop (Gate A1) | §4 |
| τ at temp 0.7 ≈ 3.0 | measure τ on a code workload at 0.7 (Gate D1) | §4 |
| effective BW 135–175 GB/s | measure achieved BW per kernel (Gate B1) | §3 |
| `k*` = 4 prior | the k-sweep (Gate D1) | §4 |
