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
| B1 — kernels G1–G9 | ▫ | |
| D1 — DFlash + k-sweep | ▫ | |
| S1 — server | ▫ | |

## Current state

- Metadata for both repos downloaded and read: `config.json`, `generation_config.json`,
  tokenizer, `chat_template.jinja`, `modeling_laguna.py`, `configuration_laguna.py`, draft
  `config.py`.
- All 145 153 target tensor headers + 69 draft tensor headers captured to `tools/hdr_*.json`
  (fetched by HTTP range before the shards finished). Byte sum reproduces the index
  `total_size` = 71 898 733 760 exactly.
- **Weight download running in background** (74.1 GB total). Logs:
  `$SCRATCH/dl_target.log`, `$SCRATCH/dl_draft.log`; sentinel `$SCRATCH/dl_done`.
  Resume with the same `hf download ... --local-dir models/<repo>` command — it is idempotent.

## Headline numbers established (all from disk, none invented)

- `B_tok` = **10.04 GB/token** at ctx 4096 — attention is **5.61 GB (56 %)** and BF16;
  routed experts only 2.50 GB (25 %). **Laguna is attention-bound at bs = 1.**
- **AR wall = 19.9 tok/s** at 200 GB/s. Poolside's GB10 13–14 tok/s ⇒ ~135 GB/s effective,
  i.e. vLLM is already near-roofline; the "1.5–2× on the floor" premise is refuted.
- `k*` prior = **4** (5 at temp 0, 3 at temp 0.7). Both model cards' values (7, 15) are
  datacenter settings and are wrong for this hardware.
- `E_frac(k)` ≤ 0.45 even at k = 15 — no expert-union blow-up.
- KV: 24 576 B/token (global layers only) + 37.7 MB/seq constant ⇒ **~1.66 M KV tokens**, ~2×
  poolside's reported capacity, thanks to the 512-token SWA rings.
- Expected band: **25–40 tok/s** stock, **32–48 tok/s** with self-quantized attention.

## Immediate next actions

1. Wait out the download; verify shard checksums.
2. Write `MODEL_INVENTORY.md` (tensor/shape/dtype/scale-layout dump — material already in
   `tools/hdr_target.json`) and `ARCH_DELTA.md` (gemma-4 vs Laguna per subsystem).
3. Stand up the correctness oracle in a venv outside the build (Transformers +
   `trust_remote_code`, pinned); capture golden per-layer hidden states + logits at temp 0.
   **Pin and record versions** — DFlash upstream is in flux (vLLM #46853, SGLang #29446).
4. Confirm the DFlash propose loop is one forward per block (affects `ROOFLINE.md` §4).

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
