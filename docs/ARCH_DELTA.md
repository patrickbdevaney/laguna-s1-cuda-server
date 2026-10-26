# ARCH_DELTA.md — gemma-4-26B-A4B vs Laguna S 2.1, per subsystem

Gemma column from `~/gemma-cuda-hybrid/src/forward.cu:54-58` and its constitution §2.
Laguna column read from `config.json` + `modeling_laguna.py` + safetensors headers. Nothing inferred.

Verdict key: **PORT** = gemma kernel works after a constant change · **RETILE** = same kernel
structure, new shapes/loop bounds · **NEW** = no gemma analog, write from scratch ·
**INVERT** = gemma treated it as secondary, here it is primary.

---

## Summary table

| Subsystem | gemma-4-26B-A4B | Laguna S 2.1 | verdict |
|---|---|---|---|
| Layers | 30 | **48** | PORT (constant) |
| Total / active | 25.2 B / 3.8 B | **117.6 B / 8.5 B** | PORT |
| Hidden | 2816 | **3072** | PORT |
| Vocab | 262 144 | **100 352** | PORT |
| Attention heads | 16 uniform | **48 global / 72 sliding**, per-layer list | **RETILE ×2** |
| KV heads / head_dim | 8 (sliding) / 2 (full), hd 256 / 512 | **8 uniform, hd 128** → GQA groups **6 and 9** | **RETILE** |
| Attention mix | 25 sliding(1024) + 5 full, `is_full(L)` = L∈{5,11,17,23,29} | **36 sliding(512) + 12 global**, global at L ≡ 0 mod 4 | RETILE |
| Attention dtype | NVFP4 | **BF16** | **INVERT** (see §2) |
| QK norm | none | **RMSNorm per head, pre-RoPE** | **NEW** |
| Attention gating | none | **per-head `softplus(g_proj(x))`, pre-`o_proj`** | **NEW** |
| RoPE | single θ, full rotary | **2 tables**: global yarn θ=5e5 partial **0.5**; sliding default θ=1e4 partial 1.0 | **NEW** |
| Logit softcap | 30.0 on lm_head | **none** (`moe_router_logit_softcapping` 0.0) | delete |
| Embedding scale | `sqrt(2816)` = 53.066 | **none** in `modeling_laguna.py` | delete |
| Norms | gemma double-norm, zero-centred `(1+g)` | **plain RMSNorm `g · x̂`**, ~4/layer | simplify |
| Router | gemma top-8 softmax | **sigmoid + `e_score_correction_bias` (selection only)**, top-**10**, sum-normalised | **NEW** |
| Experts | 128, top-8, `MOE_INT` 704, +1 shared | **256, top-10, `moe_intermediate_size` 1024, +1 shared** | RETILE |
| MoE output | — | `expert_out × 2.5 + shared_out` (`moe_routed_scaling_factor`) | NEW constant |
| Dense layers | none | **layer 0 only**, `intermediate_size` 12288 | NEW special case |
| lm_head | tied to embed, quantized to NVFP4 (the v1.0 breakthrough) | **NOT tied**, separate BF16 `[100352,3072]` | **INVERT** |
| KV cache | FP8 e4m3, dynamic, 64 K | FP8 e4m3, **static per-layer scales in checkpoint**, 256 K target | PORT + use shipped scales |
| Tokenizer | `▁`-normalised BPE, byte_fallback, 262 k | **ByteLevel BPE, 100 352, 2-stage `\p{L}`/`\p{N}` pre-tokenizer** | **NEW** |
| Chat grammar | `<start_of_turn>`/`<end_of_turn>` | `<system>/<user>/<assistant>/<think>/<tool_call>` + preserved thinking | **NEW** |
| Draft | 5-layer qwen3-style, BLK 16, k=14 | **6-layer Laguna-style, BLK 16, k*≈5**, fused `qkv_proj`, `fc[3072,18432]` over 6 aux taps | RETILE, same discipline |
| Draft taps | `TAP_LAYERS[6] = {1,6,11,17,22,27}` | **`target_layer_ids = [1,10,19,29,38,47]`** | PORT (constant) |

---

## 1. Attention — two tilings, not one

The per-layer head count is the surprise. `num_attention_heads_per_layer` gives **48 heads on the
12 global layers and 72 on the 36 sliding layers**, with `num_key_value_heads` = 8 and
`head_dim` = 128 throughout. So:

| | global (12 layers) | sliding (36 layers) |
|---|---|---|
| heads | 48 | 72 |
| GQA group | **6** | **9** |
| `q_proj` | `[6144, 3072]` | `[9216, 3072]` |
| `o_proj` | `[3072, 6144]` | `[3072, 9216]` |
| `g_proj` | `[48, 3072]` | `[72, 3072]` |
| KV span | full context | **512** |
| rotary | yarn, θ 5e5, **64 of 128 dims** | default, θ 1e4, **128 of 128 dims** |

Both GQA groups are non-powers-of-2 (6 and 9). Gemma's head-pack kernel assumes one block per
(query, kv_head) covering all `G` siblings with `red[G*hd]` in shared memory — that structure
ports, but `G` becomes a runtime parameter and the shared-memory footprint is `G·128` floats
(3 KB and 4.5 KB), far under the 228 KB budget. Easier than gemma's hd=512 case, not harder.

The 512-token sliding window is small: **36 layers × 512 × 8 kv-heads × 128 × 2 (K,V) × 1 B
(FP8) = 37.7 MB total**, i.e. **1.05 MB per layer**. If Thor's L2 holds a layer's window, sliding
attention is nearly free. Check `cudaDeviceProp::l2CacheSize` and `persistingL2CacheMaxSize`
(the gemma repo left this as an open on-device TODO — resolve it here).

### Order of operations (from `modeling_laguna.py:401-465`), exact

```
h      = input_layernorm(x)              # plain RMSNorm
q,k,v  = q_proj(h), k_proj(h), v_proj(h) # no bias
q      = q_norm(q.view(...,128))         # RMSNorm over head_dim, per head
k      = k_norm(k.view(...,128))
q,k    = rope(q,k)                       # partial: first rotary_dim dims rotate, rest pass through
attn   = softmax(q·kᵀ · 128^-0.5 + mask) · v
gate   = softplus(g_proj(h).float())     # NOTE: g_proj takes h, the *normed* input, not attn
attn   = attn.view(...,heads,128) * gate.unsqueeze(-1)
out    = o_proj(attn.view(...,heads*128))
```

Two traps: **`g_proj` consumes the post-layernorm hidden state, not the attention output**, so it
can be computed concurrently with QKV; and the gate is computed in **fp32** before the cast back.
The softplus is unbounded above, so the gate is not a 0–1 scale — it can amplify.

Partial rotary uses `rotary_dim = cos.shape[-1]`; the first `rotary_dim` dims rotate and the tail
is concatenated through untouched. For global layers `rotary_dim` = 64, and the rotate-half split
is `x1 = x[..:32], x2 = x[32:64]` — **within the rotary slice**, not the head.

The yarn `attention_factor` = **1.3465735902799727** is applied to cos/sin at table build
(`cos = emb.cos() * attention_scaling`), i.e. it can be folded into the table once. Global layers
only; sliding layers use scaling 1.0.

## 2. The BF16 inversion

Gemma's whole optimisation arc was FP4-first: the NVFP4 lm_head was the v1.0 breakthrough (+35 %)
and routing the FP4 lm_heads through the Marlin tc kernel was the v2.0 one (+9.7 %). **On Laguna
the dominant dense weights are BF16**, because poolside quantized only the routed experts.

So gemma's *secondary* kernel — `tc_bf16` (`mma.f16.f16.f32`, bf16→f16 weights, no dequant), built
for the draft linears and worth +9.3 % there — becomes Laguna's **primary dense kernel**. It
carries 5.61 GB/step of attention plus 0.89 GB of shared experts plus 0.62 GB of `lm_head`.
The FP4 path serves the experts only.

This also means gemma's alignment gotcha inverts: per-expert FP4 pointers are not 16 B-aligned
(safetensors packing) so `uint4` loads crash, but BF16 attention tensors are large, contiguous,
and will be re-allocated by our loader anyway — 16 B `int4` loads are safe there after repack.

## 3. Router — sigmoid with a selection-only bias

`modeling_laguna.py:169-184`, exactly:

```
logits  = W_gate · h                       # [256], fp32
scores  = sigmoid(logits)                  # NO softmax, NO softcap (softcapping = 0.0)
sel     = topk(scores + e_score_correction_bias, k=10).indices   # bias affects SELECTION only
w       = scores.gather(sel)               # UNBIASED scores
w       = w / w.sum()                      # norm_topk_prob = true
out     = Σ w_i · expert_i(h) · 2.5 + shared_expert(h)
```

Three ways to get this silently wrong, all of which produce fluent-but-degraded output with no
crash: using softmax; letting the bias leak into `w`; forgetting the 2.5 scaling. **G4 must
validate the top-10 index set *and their order* against the oracle**, plus the normalised weights
to a stated ULP tolerance.

`e_score_correction_bias` is F32 `[256]` per layer and is the aux-loss-free load-balancing bias
of arXiv:2408.15664. It ships under `mlp.experts.e_score_correction_bias` and the reference remaps
it onto the router module at load (`_checkpoint_conversion_mapping`).

`moe_apply_router_weight_on_input` is **false** and the reference *raises* if true — so the
routing weight multiplies the expert **output**. Matches gemma's `k_moe_down`/`finalize` structure.

## 4. MoE shapes — retile targets

| | gemma | Laguna |
|---|---|---|
| experts / top-k | 128 / 8 | **256 / 10** |
| intermediate | 704 | **1024** |
| per-expert bytes | — | **5.308 MB** (gate 1.77 + up 1.77 + down 1.77) |
| assignments per token | 8 | **10** |
| MoE layers | 30 | **47** (layer 0 dense) |
| shared expert | 1 | 1, intermediate **1024** |

Gemma's grouped path — `k_moe_invert` (expert→token map) → `k_moe_gateup_grouped` →
`k_moe_down_bw` (weight-resident, no atomics, per-assignment partials) → `k_moe_finalize` — ports
structurally. `EL` (expert list capacity) and the invert bookkeeping resize from 128 to 256.
At k*≈5 the union is ~46 experts of 256 per layer with ~1.1 tokens per expert — the same
one-token-per-expert regime that made gemma's TC grouped MoE lose to padding waste. **That
dead-end stays dead** (constitution §4).

Layer 0's dense MLP (`intermediate_size` 12288, BF16, 226 MB) is a one-off code path: no router,
no experts, plain SwiGLU. Cheaper to special-case than to generalise.

## 5. Draft — same discipline, new shapes

| | gemma DFlash | Laguna DFlash |
|---|---|---|
| layers | 5, qwen3-style | **6, Laguna-style** (sliding 512, 72 heads, gated) |
| BLK | 16 | **16** (`dflash_config.block_size`) |
| k used | 14 | **k\* ≈ 5** (see `ROOFLINE.md` §4) |
| mask token | — | **12** (`〈\|MASK\|〉`) |
| taps | target layers {1,6,11,17,22,27} | **{1,10,19,29,38,47}** (`dflash_config.target_layer_ids`) |
| tap fusion | — | `fc [3072, 18432]` over 6 concatenated aux states + 6 `aux_hidden_norms` |
| qkv | separate | **fused `qkv_proj [11264, 3072]`** = 9216 q + 1024 k + 1024 v |
| embed / lm_head | shared with target (frozen) | **shared with target** — draft ships neither | 
| dtype | BF16 (the moat) | **BF16 (the moat)** |
| weights | — | **2.230 GB**, + 0.617 GB target `lm_head` per propose = 2.847 GB/block |
| within-block attn | causal for sliding, non-causal for full | `dflash_config.causal: true`, all 6 layers sliding | 

`config.py` documents `block_size` as "size of the draft block predicted with **a forward pass**
of the model" — confirming the block-diffusion structure: **one draft forward per block**,
not `k` sequential forwards. This is what made the gemma ceiling-agent's "draft is 76 % of the
step" analysis wrong, and it holds here.

Note `eagle_aux_hidden_state_layer_ids` = `[2,11,20,30,39,48]` in the draft config while
`dflash_config.target_layer_ids` = `[1,10,19,29,38,47]` — off by one, the former being 1-indexed
"after layer N". **Use `target_layer_ids` (0-indexed), which matches gemma's `TAP_LAYERS`
convention.** Confirm against the oracle by comparing tapped hidden states.

## 6. What ports unchanged

- The whole Marlin memory recipe: offline repack to mma-fragment order, 16 B `int4` loads,
  `__ldcs` evict-first, U-unroll register prefetch, WARPS=1 max-grid-fill.
- `mma.sync.m16n8k16` fragment layout (constitution §5.3) — verified, reuse verbatim.
- Head-pack GQA structure (one block per (query, kv_head), KV read once).
- HW FP4/FP8 decode intrinsics `__nv_cvt_fp4x2_to_halfraw2`, `__nv_cvt_fp8_to_halfraw`.
- CUDA-graph capture with `g_base` as a device-side position counter.
- Lossless Gumbel-max target sampling / sample-match acceptance.
- `safetensors.h`, `httplib.h`, `json.hpp`, `webui.h` scaffolding.
- Every entry in the gemma LOST column. tcgen05 at M≤16 stays rejected — and `k*`≈5 puts us at
  M≤6, even further from the M≈900 crossover than gemma's M=15 was.

## 7. What must be written from scratch

1. ByteLevel BPE with the 2-stage Unicode-property pre-tokenizer (§6 of `MODEL_INVENTORY.md`).
2. Sigmoid router + selection-only bias + top-10.
3. Per-head softplus attention gate.
4. QK-RMSNorm.
5. Partial rotary with two tables and a folded yarn `attention_factor`.
6. Layer-0 dense-MLP special case.
7. poolside_v1 chat rendering, reasoning separation with **preserved thinking**, and the
   non-JSON `<tool_call>/<arg_key>/<arg_value>` grammar.
8. Untied `lm_head` handling (and, for the §5 lever, our own quantization of it).
