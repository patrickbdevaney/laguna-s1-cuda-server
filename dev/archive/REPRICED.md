# REPRICED.md — the frontier list re-priced against the production profile

Every estimate on the old open-frontier list was computed when `B_tok` was 10.04 GB. FP8 took it
to 6.251 GB, which changed every term's share of the step. This re-prices them against a profile
of the config we actually ship, and the ranking inverts.

## The production profile (FP8 attention, lm_head, dense)

`LG_FP8ATTN=1 LG_FP8LMHEAD=1 LG_FP8DENSE=1 LG_PROF=1 build/bench_decode`

| category | ms/step | % time | % of B_tok | launches |
|---|---:|---:|---:|---:|
| attn_qkvo_gemm | 14.973 | **34.7** | **44.9** | 96 |
| moe_experts | 13.501 | **31.3** | **39.9** | 47 |
| shared+dense | 4.667 | 10.8 | 8.9 | 48 |
| attn_core | 3.167 | 7.3 | 0.1 | 48 |
| norm+cast | 2.924 | 6.8 | ~0 | 96 |
| router | 2.603 | 6.0 | 1.2 | 47 |
| lm_head | 1.304 | 3.0 | 4.9 | 1 |
| embed+rope | 0.042 | 0.1 | ~0 | 1 |

`LG_PROF` serialises and disables graph capture, so the total (43.2 ms) is inflated against the
real 30.3 ms step and per-launch-heavy categories are over-represented. **Shares are usable;
absolute times are not.**

Byte shares come from `bytes_per_token()` re-derived independently — it reproduces 6.251 GB
exactly, so the split is trustworthy.

## The two things this reveals

**1. Attention weights are the largest term, and nothing on the list attacked them.**
44.9% of bytes and 34.7% of time. Every catalogued lever targeted the MoE side or speculation.
The biggest line in the budget had no entry against it.

**2. ~10% of the step moves almost no data.** *(Corrected — see below.)*
`norm+cast` + `attn_core` + `router` looked like **20.1%** of time for ~1.3% of bytes. That was
wrong, and the error was mine: I said "shares are usable, absolute times are not", but `LG_PROF`'s
serialization cost is **per timing region**, not proportional to time, so it inflates
launch-heavy categories far more than others. With ~30.6 µs of serialization on each of 384
regions, the real figure is **~10.5%**.

The clearest casualty is `norm+cast`: `add_rms_cast` moves 48 KB, which is 0.21 µs at 227 GB/s,
yet is charged 30.5 µs per call. Its true cost is **≤1.2% of the step, not 6.8%** — almost
entirely profiling artifact. `attn_core` deflates to ~5.6% and is the largest *real* latency term.

**This kills RMSNorm→GEMV prologue fusion on arithmetic**: perfect fusion buys ≤1.2% for days of
work across five kernels and five re-gates. The technique is real (PyTorch "Lazy Pre-Norm", B200,
17–32% of the norm kernel) — our term is simply too small to be worth it.

The deflation also moves `attn_qkvo_gemm` **up** to ~39.7%, so the attention-precision lever below
is worth *more* than the +9.2% first computed, not less.

## Re-priced ranking

| lever | old claim | **re-priced** | basis |
|---|---:|---:|---|
| **`o_proj` → NVFP4** | *(absent)* | **+9.2%** | `o_proj` is 42.7% of attention weights = 1.198 GB; NVFP4 saves 0.524 GB = 8.4% of `B_tok` |
| Routed experts → 3 bpw trellis | +15.2% | **+5.3%** | measured: trellis_B 395 vs production 332 Gweight/s, applied to a 31.3% term |
| Kernel fusion beyond `moe_invert` | +3.2% | **+3–4%** | ~770 residual kernels × 1.6 µs = 1.23 ms of 30.3 |
| Shared expert → NVFP4 | *(absent)* | **+3.2%** | 7.1% of bytes × 0.4375 |
| CSV-Decode `lm_head` skip | +4.3% | **≤+3.1%** | `lm_head` is now only 3.0% of the step; a full skip is the ceiling |
| Block Verification | 5–8% | unchanged | speculation axis, not base decode |
| Close M>1 inefficiency | τ 3.36→2.40 | unchanged | speculation axis; still the highest-value item overall |

**The trellis subsystem drops from first to second**, and its measured +5.3% is now below an
untried lever that needs no new container, no encoder, and no re-quantization.

### MEASURED: NVFP4 attention passes the greedy gate

Simulated in the loader (`LG_NVFP4_SIM`) — weights rounded onto the exact NVFP4 grid (E2M1,
per-16 E4M3 group scale, per-tensor global scale) and then handed to the existing FP8 path, so
the kernels are untouched and the delta is purely capability:

| config | greedy vs oracle | tok/s |
|---|---|---:|
| baseline FP8 | 8/8 | 33.04 |
| `o_proj` NVFP4 | **8/8** | 33.16 |
| `q/k/v` NVFP4 | **8/8** | 32.97 |
| gate NVFP4 | **8/8** | 33.06 |
| **all five** NVFP4 | **8/8** | 33.20 |

All five attention projections at NVFP4 leave greedy output **bit-identical**. For contrast, the
expert requant at 2.81 bits failed this same gate 0/8 — so the gate does discriminate; it simply
does not fire here.

**My softplus premise was wrong, and it was a category error.** I argued the unbounded per-head
gate would multiply quantization error. For *weight-only* quantization the gate scales signal and
error by the identical factor, so SNR is **exactly invariant** to gate magnitude — the argument
holds for activation quantization, which is not what is being done. Two independent confirmations:
`head_dim` = 128 divides the NVFP4 group of 16 exactly, so the gate is *constant* within every
quantization block; and `o_proj` is empirically the **most** 4-bit-robust projection measured
(88.6% retention vs `v_proj` at 54.4%).

### Sizing, and the right target

Attention weights are 2.803 G elements = 2.803 GB at FP8, 1.577 GB at NVFP4 → saves 1.226 GB.
`B_tok` 6.251 → 5.025 GB = **+24.4%**, i.e. 33.0 → 41.1 tok/s, if the term stays bandwidth-bound.

But `q_proj` and `o_proj` are byte-identical at 1.2457 GB **each** — two equal prizes — while k/v
together are only 4.83% of `B_tok` (GQA already shrank them) *and* are the most
quantization-sensitive projections in the literature. So **q+o is the right target**: most of the
prize, least of the risk.

**The open question is no longer capability, it is whether the byte saving converts.** The trellis
work showed exactly how this fails: a compute-bound kernel cannot spend a byte saving. Production's
NVFP4 MoE GEMV runs at 332 Gweight/s ≈ 187 GB/s ≈ 82% of ceiling, so NVFP4 dequant at ~1–1.5
ALU ops/weight should stay bandwidth-bound — but that must be measured with a NOMEM twin before
the kernel is built, exactly as it was for the trellis.
