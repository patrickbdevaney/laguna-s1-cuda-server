# ATTENTION_NVFP4.md — the largest remaining lever, measured on both axes

The frontier list never contained an entry for the largest term in the byte budget. Re-pricing
against a profile of the shipping config exposed that: `attn_qkvo_gemm` is **~39.7% of the step and
44.9% of `B_tok`**, and every catalogued lever targeted MoE or speculation.

Two questions had to be answered separately, because the trellis work proved they are independent:
**does the model survive it**, and **does the byte saving convert**.

## 1. Capability — measured, passes

`LG_NVFP4_SIM` rounds attention weights onto the exact NVFP4 grid (E2M1 codes, per-16 E4M3 group
scale, per-tensor global scale) inside the loader, then hands them to the existing FP8 path. The
kernels are untouched, so the delta is purely capability.

| config | greedy vs oracle | tok/s |
|---|---|---:|
| baseline FP8 | 8/8 | 33.04 |
| `o_proj` NVFP4 | **8/8** | 33.16 |
| `q/k/v` NVFP4 | **8/8** | 32.97 |
| gate NVFP4 | **8/8** | 33.06 |
| **all five** NVFP4 | **8/8** | 33.20 |

Greedy output is **bit-identical**. The same gate failed the 2.81-bit expert requant **0/8**, so it
discriminates — it simply does not fire here.

### The premise this refutes was mine

I argued `o_proj` could not go below FP8 because the unbounded per-head softplus gate sits
immediately upstream and would multiply quantization error. **That is a category error.** For
*weight-only* quantization the gate scales signal and error by the identical factor, so SNR is
**exactly invariant** to gate magnitude; the argument holds only for *activation* quantization,
which is not what is being done. Two independent confirmations: `head_dim` = 128 divides the NVFP4
group of 16 exactly, so the gate is **constant within every quantization block**; and `o_proj` is
empirically the **most** 4-bit-robust projection measured (88.6% retention vs `v_proj` at 54.4%).

## 2. Conversion — measured, passes at 95%

The question the trellis failed. `tests/bench_attnq.cu`, real `o_proj` shape (3072 reduction),
production `[row][chunk][lane]` repack, activations staged in shared memory:

```
stream (ceiling)   499.72 Gweight/s   249.9 GB/s
fp8                174.82 Gweight/s   175.0 GB/s    65.7% of its own ALU ceiling
nvfp4              295.62 Gweight/s   166.3 GB/s    78.8% of its own ALU ceiling

byte ratio fp8/nvfp4              1.780x
measured Gweight/s nvfp4/fp8      1.691x
conversion efficiency             95.0%
```

**Both encodings achieve essentially the same GB/s (175 vs 166), so the weights/s gap is purely
bytes-per-weight.** That is the signature of a bandwidth-bound term. Neither kernel is pinned at
its ALU ceiling — which is exactly what the trellis was, at 91%, and why its 1.406× byte saving
evaporated.

**Projected: +19.9% end-to-end for q+o** (1.090 GB of 6.251), i.e. 33.2 → ~39.8 tok/s.

### The NOMEM twin caught its own bug

The first run of this benchmark reported nvfp4 at **310.9% of its own ALU ceiling** — impossible, a
kernel cannot exceed its no-memory twin. The twin was not structurally identical to the real
kernel: it lacked the `#pragma unroll` on the chunk loop and substituted a constant for the
per-chunk scale arithmetic, so it was a *different* kernel whose ceiling meant nothing. Made
identical, it reads 375.15 Gweight/s and the whole picture becomes coherent.

Worth stating because the headline number barely moved (97.6% → 95.0% conversion, +20.5% →
+19.9%). **The right conclusion was reachable from a broken instrument** — which is exactly when a
self-inconsistency is easiest to wave through, and exactly why it must not be.

## 3. Why q+o and not all five

`q_proj` and `o_proj` are byte-identical at **1.2457 GB each** — two equal prizes. `k`/`v` together
are only **4.83% of `B_tok`** (GQA already shrank them) *and* are the most quantization-sensitive
projections in the literature (`v_proj` 54.4% retention vs `o_proj` 88.6%). So q+o takes ~89% of
the available bytes at the lowest risk. All five would be +24.4%; q+o is +19.9%.

## 4. What remains to build

Capability and conversion are both settled. The implementation is:

1. **Loader**: quantize attention weights BF16 → NVFP4 into the production repack layout. The
   expert path already does exactly this (`repack_scale`, `[nb][ng/8][lane][8]`); this is the same
   transform on a different tensor set.
2. **Kernel**: a W4A16 dense GEMV. `k_nvfp4` in `tests/bench_attnq.cu` is a working one at the real
   shape — it is the benchmark kernel, already measured.
3. **Integration**: `o_proj` is a standalone GEMM and is the low-risk first step. `q_proj` lives in
   the fused `q|k|v|g` segmented path, so it needs the segment descriptor to carry per-segment
   encoding — more invasive, and worth doing second.
4. **Gates**: greedy 8/8 (already passing under simulation), then the capability sweep, which
   `EXPERT_BITS_EVAL.md` established is the only instrument that can grade a modified model — not
   τ, which *rose* when quality fell, and not golden-continuation alone, which is a coin flip.

Estimated: `o_proj` alone is **+9.6%**, q+o is **+19.9%**.
