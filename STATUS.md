# STATUS.md — resume point

Last updated: 2026-07-24. Repo: `~/laguna-s1-cuda-server` (all writes here).
Reference, **read-only**: `~/gemma-cuda-hybrid`.

## Gates

| gate | state | evidence |
|---|---|---|
| **H1 — capacity** | ✅ **PASS** | 122 GB unified / 115 GB available / 379 GB disk → `HARDWARE.md` |
| **R1 — roofline** | ✅ **PASS** | `ROOFLINE.md`, generator `tools/roofline.py`, raw `docs/roofline_ctx*.txt` |
| **A1 — oracle + arch delta** | ✅ **PASS** | `MODEL_INVENTORY.md`, `ARCH_DELTA.md`, `oracle/`, golden tensors in `docs/golden/` → `LOOP_LOG.md` |
| L1 — loader | ⏳ next | |
| B1 — kernels G1–G9 | 🟡 **G1–G8 PASS** | `gate_kernels` 13/13; `gate_forward` 8/8 greedy exact; **16.59 tok/s decode / 166.7 GB/s / 73 % of ceiling**; G9 CUDA graph pending |
| D1 — DFlash + k-sweep | ▫ | |
| S1 — server | ▫ | |

## Current state

Working pure-CUDA Laguna forward pass: **greedy-exact vs the oracle**, **16.59 tok/s median**
decode at ctx 4096 (166.7 GB/s effective, **73 % of the 227 GB/s ceiling**), prefill 55.0 tok/s.
For reference, poolside measured **13–14 tok/s** for this model on a DGX Spark (same bandwidth
class) under vLLM — so the autoregressive path is already ahead of that, before speculation.
Everything from the checkpoint on disk through to logits is C++/CUDA; Python exists only in
`oracle/` for validation and never on the serving path.

Built and gated so far: config parser, arena loader, NVFP4 dequant, BF16 and FP4 dense GEMMs,
RMSNorm, QK-norm, two rope tables with partial rotary, sigmoid router with selection-only
bias, per-head softplus gating, head-packed GQA with FP8 KV and SWA rings, grouped
weight-resident MoE, and the full 48-layer forward.

## Immediate next actions (in priority order)

1. **Gate D1 — DFlash.** Now the biggest remaining multiplier (1.6–1.9×) and a deliverable in
   its own right. The kernel arc has spent its easy structural wins: MoE and the BF16
   attention GEMMs are now ~37 % each, so no single kernel dominates.
2. **G9 — whole-step CUDA graph.** ~1665 launches/step at 4.15 µs each.
3. **Gate D1 — DFlash.** Load the 6-layer draft in BF16 (never quantize it), share the
   target's embed + `lm_head`, implement propose/verify with lossless sampling, then run the
   k-sweep and re-measure `E_frac` on real draft blocks (the current measurement is from
   prompt tokens — `ROOFLINE.md` §10 caveat).
4. **Gate S1 — server layer.** Tokenizer (ByteLevel BPE, 2-stage Unicode pre-tokenizer),
   poolside_v1 chat/tool/reasoning grammar with preserved thinking, prefix cache, SSE.
5. **The §5 self-quantization lever** — the largest single win available (+39 % to +113 % on
   `B_tok`), but quality-gated and correctly sequenced after the bit-exact path is banked.

## Standing corrections to the directive (carry forward)
1. `B_tok` ≈ 10.04 GB, not 4.5 → AR wall 20 tok/s, not 44.
2. Only routed experts are NVFP4; attention/shared/router/`lm_head`/layer-0 MLP are BF16.
3. `vocab_size` = 100 352, not 262 144. `tie_word_embeddings` = **false** — G7's premise
   ("quantize the tied embedding") does not hold; `lm_head` is a separate BF16 tensor.
4. `num_experts_per_tok` = **10**.
5. Rotary is per-layer-**type** (two tables: yarn θ=500 k partial 0.5 for global; default
   θ=10 k partial 1.0 for sliding), not a per-layer scale table.
6. Poolside's acceptance table is measured at `num_speculative_tokens=15`, not 7.
7. Per-layer head counts differ: 48 heads on global layers, **72 on sliding layers**.
8. Priority order inverts: attention path before MoE grouped GEMM at bs = 1.
