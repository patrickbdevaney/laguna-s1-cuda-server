# APPENDIX — cross-model decode comparison, and where Laguna actually stands

Companion appendix. The same document exists in the DeepSeek engine repo as
`wiki/cross-model-decode-comparison.md`; both are maintained together because the comparison is only
meaningful with all three models on one axis. Numbers here for Laguna come from this repo's own
`README.md`, `ROOFLINE.md`, `OPTIMIZATION_LOG.md` and `COST_MODEL_CORRECTION.md`; numbers for the
DeepSeek engine come from its measured checkpoint headers and evidence logs.

## The three, on one Jetson AGX Thor

| | total / active | quantisation | `B_tok` | decode | engine |
|---|---|---|---|---|---|
| **Laguna S-2.1-NVFP4** (this repo) | 117.6 B / **8.5 B active** | routed experts NVFP4; attention, shared experts, router, layer-0 MLP, `lm_head` were BF16 → self-quantised to FP8 (#15, #30) | **6.251 GB** | **33.0 AR** / 40.1 code / **49.7 edit** (DFlash) | pure CUDA |
| **DeepSeek V4-Flash-0731-REAP** | 284 B → REAP ~180 B / **13 B active** | routed experts MXFP4; MLA+dense FP8; **1.74 GB/token still BF16** | **12.26 GB** | 13.8 AR / 24.5 spec (DSpark MTP) | pure CUDA |
| **Qwen 3.5 122 B A10B** | 122 B / **10 B active** | NVFP4 throughout | ~5.3 GB (est.) | ~50 spec | vLLM + DFlash |

## Byte efficiency, compared like with like

    decode x B_tok = useful weight traffic

| mode | model | tok/s | `B_tok` | effective |
|---|---|---|---|---|
| **spec** | **Laguna, edit-style** | 49.7 | 6.251 | **311 GB/s** |
| **spec** | DeepSeek engine | 24.5 | 12.26 | 301 GB/s |
| **spec** | Qwen + vLLM/DFlash | 50.0 | ~5.3 | ~265 GB/s |
| **spec** | **Laguna, code** | 40.1 | 6.251 | **251 GB/s** |
| **AR** | **Laguna** | 33.0 | 6.251 | **206 GB/s** |
| **AR** | DeepSeek engine | 13.8 | 12.26 | 169 GB/s |

**Laguna's AR is the most byte-efficient measurement any of the three engines has produced** — 206
GB/s, 91 % of its own byte wall, against a published best-in-class full-step batch-1 efficiency of
82 % (FlashFormer, H100). It also exceeds the DeepSeek engine's *optimistic* AR floor (196 GB/s,
every byte mark at full DRAM rate), which is a useful external check on that project's roofline: its
internal ceiling is too conservative by at least 5 %.

## What this says about Laguna's remaining headroom

**Kernels are done. What is left is bytes.** The 70 tok/s target was not missed for want of
optimisations — at 91 % of the byte wall there is no meaningful kernel headroom to find. The
arithmetic:

    B_tok 6.251 GB = dense 3.756 + routed experts 2.495   (experts already NVFP4)
    dense FP8 -> FP4:   B_tok -> 4.37 GB   =>  AR ~47 tok/s
    x measured edit-style speculation (49.7/33.0 = 1.51x)  =>  ~71 tok/s

**70 was reachable through precision on the dense path**, and only there. That is consistent with
this repo's own history: every large win was a `B_tok` reduction, not a kernel — #15 (attention
BF16→FP8, 10.044 → 7.242) and #30 (shared experts + layer-0 dense, → 6.251), **+52 % cumulative**.
The dense share is now 60 % of `B_tok`, so it remains the only term with real volume behind it.

**One caveat that cuts against it:** dense here is attention, and FP4 on attention weights is a
different risk class from FP4 on experts. `ATTENTION_NVFP4.md` is the right place to settle that,
not this appendix.

## The one measured lever that may transfer in the other direction

`ACTIVATION_SPARSITY.md` killed the published 2.5×-MoE-layer claim (arXiv:2605.08575) **on these
weights**: at the paper's 87 % operating point, zeroing costs 43.7 % relative L2 error unstructured
and 83 % block-32, and the offline neuron permutation rescue buys 0.9 points. The stated reason was
that every published measurement is on models with `moe_intermediate` ≥ 1408, and **Laguna's is
1024**.

**The DeepSeek checkpoint's `moe_intermediate` is 2048.** So the one lever that failed here on a
size argument is the one lever that might survive there — it is the only item in this comparison
that could go the other way, and it is cheap to test with the same `LG_DUMP_MOEH`-style dump.

## The synthesis

All three engines sit near their byte walls, so **the decode ranking is a quantisation ranking**:
Qwen fully NVFP4 and fastest, Laguna NVFP4-experts + FP8-dense second, the DeepSeek REAP
MXFP4-experts + FP8-dense + 1.74 GB BF16 third — while the last of those extracts more useful
bandwidth per byte under speculation than either of the others.

For Laguna specifically the ranking is flattering and the conclusion is narrow: **it is the most
efficient of the three per byte at AR, and its next tier is a precision decision, not an engineering
one.**
