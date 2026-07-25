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

`oracle/golden.py`. 15.2 GB of BF16 non-expert weights resident as fp32 on GPU; the 63.9 GB
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
