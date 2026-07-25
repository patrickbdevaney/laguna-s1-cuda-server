# RESCOPE.md — the plan, corrected against the measured roofline

Supersedes `DIRECTIVE.md` **on facts only**. Goals, deliverable, and invariants are unchanged.
Every correction below traces to a file on disk, cited inline. Generated evidence:
`ROOFLINE.md`, `tools/roofline.py`, `docs/roofline_ctx*.txt`.

---

## 1. What changed, and why it matters

| # | directive said | checkpoint says | consequence |
|---|---|---|---|
| 1 | `B_tok` ≈ 4.51 GB (whole model NVFP4) | **10.04 GB** — `quantization_config.ignore` exempts attention, shared experts, router, layer-0 MLP, `lm_head` | AR wall 44 → **19.9 tok/s** |
| 2 | vLLM leaves 1.5–2× on the floor (~59–63 GB/s) | poolside's 13–14 tok/s ÷ 10.04 GB = **~135 GB/s = 68 % of achievable** | no 2× algorithmic gap exists; the gap is ~135→175 GB/s of kernel efficiency |
| 3 | MoE grouped GEMM is the biggest lever | attention = **5.61 GB = 56 %** of the step; routed experts = 2.50 GB = 25 % | **priority order inverts** |
| 4 | expert-union blow-up is the crux | `E_frac` ≤ 0.45 even at k=15 (top-10 of 256 = 3.9 % slice) | risk #2 in the directive is a non-risk |
| 5 | `k*` nearer 7 than 15 | `k*` = **5 at T=0, 3 at T=0.7**; k=15 is worst at every temperature | design verify for M ≤ 6 |
| 6 | acceptance table measured at k=7 | draft card header: `num_speculative_tokens=15` | α refit: 0.858 / 0.837 / 0.763 / 0.754 |
| 7 | vocab 262 144, tied embedding | **vocab 100 352**, `tie_word_embeddings: false` | G7 premise void — `lm_head` is its own BF16 tensor |
| 8 | top-k "read it from config" | **`num_experts_per_tok` = 10** | MoE tiling target |
| 9 | per-layer rotary scale table | per-layer-**type**: 2 tables (global yarn θ=500 k partial 0.5; sliding default θ=10 k partial 1.0) | simpler than feared |
| 10 | uniform head count | **48 heads global / 72 heads sliding**, `num_key_value_heads` = 8 → GQA groups **6 and 9** | two attention tilings, both non-power-of-2 groups |
| 11 | 3.69× speculative speedup | that is TP=2 datacenter; same pair on GB10 gave **1.6–1.75×** | quote 1.4–1.9× |
| 12 | band 32–52 tok/s | **25–40 stock**, **32–48 with §5 self-quantization** | 52 requires the new lever |

Unchanged and still binding: no Python on the hot path; correctness before speed; **the draft
stays BF16**; no invented constants; the gemma lost-column stays lost; one change per
measurement; bands not points; a failed gate stops the build.

---

## 2. Revised priority order (replaces `DIRECTIVE.md` §9)

Derived from the byte budget, not from the gemma profile:

1. **Attention weight path** — 5.61 GB/step, 56 %. BF16 GEMV at M=1 and `mma.sync` at M≤6.
   Two tilings (48-head/group-6 global, 72-head/group-9 sliding).
2. **Self-quantization of the BF16 remainder** (`ROOFLINE.md` §5) — the single largest lever in
   the project: −40 % `B_tok` for FP8 attention, −53 % for full NVFP4 non-expert. Quality-gated,
   staged, after the bit-exact stock path passes B1.
3. **MoE grouped GEMM** — 2.50 GB at k=1 but **28.7 GB at k=15**; it dominates *inside* the
   verify step and its share grows with `k`. Still the #1 kernel for the served path at k*≈5,
   where it is 11.5 GB of a 21.9 GB block (53 %).
4. **Kernel launch count** — 48 layers × ~14 kernels on 20 SMs. Whole-step CUDA graph is
   mandatory, not optional.
5. **SWA KV residency in L2** — 36 layers × 512 × 2048 B = 37.7 MB total. Check `l2CacheSize`;
   if the per-layer 1.05 MB window pins, the sliding attention becomes nearly free.
6. `E_frac` reduction / routing-aware draft scheduling — only if measurement contradicts §4.

Deleted from the priority list: expert-union mitigation as a *primary* lever (item 4 above).

---

## 3. Revised gate definitions

Changes only; everything unstated is as `DIRECTIVE.md` §6.

- **G1 NVFP4 dequant** — group size **confirmed = 16** from `weight_scale [1024,192]` at K=3072.
  Scale layout confirmed: `weight_packed` U8 (2×E2M1/byte) + `weight_scale` F8_E4M3 per 16 +
  `weight_global_scale` F32 scalar. Also present and unused by us: `input_global_scale` (we do
  W4A16, no activation quant — the gemma finding that dodges vLLM's bs=1 quant trap).
- **G2 dense GEMM** — must handle **BF16 weights**, not just FP4. The dominant dense shapes are
  BF16 `q/k/v/o`. Port gemma's `tc_bf16` path (the +9.3 % draft-linear win) as the *primary*
  dense kernel, with the FP4 path for experts. This reverses gemma's emphasis.
- **G3 attention** — new: two head counts, partial rotary (**64 of 128 dims rotate on global
  layers**, all 128 on sliding), **QK-RMSNorm per head before RoPE**, yarn scaling with
  `attention_factor` 1.3465735902799727 applied to cos/sin, and per-head **softplus gating on
  the *pre-attention* hidden state**, applied before `o_proj`.
- **G4 router** — **sigmoid, not softmax**. `e_score_correction_bias` is added for *selection
  only*; the returned weights are the **unbiased** sigmoid scores, then top-10 normalized by
  their sum, then the expert output is scaled by `moe_routed_scaling_factor` = 2.5 and added to
  the shared expert. Getting the bias-in-selection-but-not-in-weights split wrong is silent.
- **G5 MoE** — 256 experts + 1 shared, `moe_intermediate_size` 1024, 47 layers (layer 0 dense).
- **G6 FP8 KV** — per-layer **static** per-tensor scales ship in the checkpoint as
  `self_attn.k_scale` / `v_scale`. Use them; do not compute dynamic scales.
- **G7 lm_head** — **not tied.** Separate BF16 `[100352, 3072]`. Quantizing it is part of lever
  §2, not a free port of gemma's tied-embed trick.
- **G8** — reconcile against **19.9 tok/s @200 GB/s**, not 44.

---

## 4. KV sizing policy — the standing-amendment objective

**Finding: KV capacity is free with respect to decode speed. Only *used* context costs
bandwidth, and only on the 12 global layers.**

The 36 sliding layers read a bounded 512-token window forever — 37.7 MB per sequence, constant,
no matter how long the conversation gets. Only the 12 global layers scale, at **24 576 B/token**.
That is a 4.0× structural KV win over a uniformly-global model and it is what makes a
long-horizon agentic window affordable here.

Second finding: **speculation halves the long-context penalty.** KV is read once per verify
block, so the per-token KV tax divides by τ. Going 4 K → 262 K costs AR **−39 %** but costs
DFlash only **−20 %**.

| used ctx | KV/step | KV as % of `B_tok` | AR @175 | DFlash @175 | k* | vs 4 K |
|---:|---:|---:|---:|---:|---:|---:|
| 4 K | 0.14 GB | 1.4 % | 17.4 | **33.8** | 6 | 100 % |
| 32 K | 0.84 GB | 7.8 % | 16.3 | 32.8 | 6 | 97 % |
| 64 K | 1.65 GB | 14.3 % | 15.1 | 31.8 | 6 | 94 % |
| **128 K** | 3.26 GB | 24.8 % | 13.3 | **30.0** | 7 | **89 %** |
| 256 K | 6.48 GB | 39.5 % | 10.7 | 27.0 | 8 | 80 % |
| 512 K | 12.92 GB | 56.6 % | 7.7 | 22.8 | 9 | 67 % |
| 1 M | 25.81 GB | 72.3 % | 4.9 | 17.7 | 11 | 52 % |

Free-zone knees (AR, the strict case): KV is 5 % of `B_tok` at **19 K**, 10 % at **42 K**,
20 % at **97 K**, 33 % at **192 K**.

Memory budget: `115 − 71.90 (weights) − 2.23 (draft) − 4.0 (runtime) = 36.87 GB` for KV →
**1.43 M tokens of single-sequence capacity**, or 5 concurrent 256 K sequences.

### Policy adopted

1. **Paged KV, allocated on demand.** Pages are claimed as the conversation grows, so declaring
   a large maximum costs nothing until it is used. This is what makes "max KV that doesn't
   affect decode speed" a well-posed request with a clean answer: **declare the maximum, pay
   only for what you fill.**
2. **`CTX` default = 262 144** (the shipped `max_position_embeddings`), which keeps **80 % of
   peak decode** at full occupancy and needs 6.48 GB.
3. **`CTX` ceiling = 1 048 576** behind the documented rope override
   (`rope_parameters.full_attention.factor` 32 → 128, `max_position_embeddings` → 1048576),
   fitting in the 36.87 GB budget at 25.81 GB with a stated quality caveat.
4. **The agentic sweet spot is 128 K**: 89 % of peak decode, 3.26 GB. A 40 K-token constitution
   plus accumulated project state sits inside the ≤10 %-tax zone with room for a long trace.
5. **Do not shrink the window to buy speed.** Halving 256 K → 128 K buys +11 % decode; the
   `ROOFLINE.md` §5 quantization lever buys +39–113 % without touching context. Spend effort
   there instead.
6. **Prefix caching saves prefill, not decode.** The constitution's KV is *computed* once but
   still *read* every step. This is worth stating loudly because it is the intuitive trap: the
   long-horizon win is latency-to-first-token, and decode cost is set by total context length
   regardless of cache hits.
7. **Open lever:** the 12 global layers are 100 % of the scaling KV cost. Sub-FP8 KV *on global
   layers only* would halve the long-context tax (262 K: 6.48 → 3.24 GB, restoring ~9 points of
   throughput). Logged in `OPTIMIZATION_LOG.md`, not yet scoped.

---

## 5. Revised targets

| metric | poolside on GB10 (same BW class) | stock-checkpoint target | with §2 lever |
|---|---|---|---|
| AR decode @ 4 K | 13–14 | 15–19 | 24–31 |
| DFlash code @ 4 K | 22–24 | **26–34** | **32–48** |
| DFlash code @ 128 K | not reported | 23–30 | 29–43 |
| KV tokens | 830–870 K @256 K | **~1.43 M** | ~1.43 M |
| prefill | 600–800 tok/s | ≥ 800 | ≥ 800 |

Headline commitment: **beat 22–24 tok/s by 1.4–2.0× on code at temp 0.7 while holding a 256 K
window**, and keep 1 M reachable.

---

## 6. Scope revision

The directive estimated 60 % port / 40 % net-new. Reading the checkpoint moves that to roughly
**45 % port / 55 % net-new**, because: two attention tilings instead of one; BF16 dense GEMM
promoted from a side path to the primary path; sigmoid+bias routing with no gemma analog;
per-head softplus gating; QK-norm; partial rotary with yarn `attention_factor`; an untied
`lm_head`; a new tokenizer; and a new chat/tool/reasoning grammar with interleaved preserved
thinking. The gemma kernel *patterns* all transfer; the gemma kernel *shapes* mostly do not.
