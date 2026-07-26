# HARDWARE.md — Gate H1

**GATE H1: PASS.** 122 GB total unified memory, 115 GB available. This is the 128 GB Thor SKU.
Laguna S 2.1 NVFP4 (71.90 GB) + DFlash draft (2.23 GB) = 74.13 GB resident, leaving ~41 GB for
KV cache, activations, and the repack working set. **Project is viable.**

Measured 2026-07-24 on this box.

## Capacity

| | measured |
|---|---|
| Unified memory total | **122 GB** (`free -g`) |
| Available at idle | **115 GB** |
| Swap | 31 GB (must never be touched during decode) |
| Disk free (`/`, nvme0n1p1) | **379 GB** of 936 GB |
| Baseline GPU allocation | ~460 MB (Xorg, gnome-shell, rustdesk, nautilus) |

Budget after weights: `115 − 74.13 = 40.9 GB` for KV + activations + repack scratch.
See `ROOFLINE.md` §6 — that is ~1.6 M KV tokens, far more than the 262 K context needs.

## Chip / toolchain

| | |
|---|---|
| Chip | Jetson AGX Thor, Blackwell **`sm_110a`** |
| SMs | 20 (2560 CUDA cores) |
| Shared memory | 228 KB/SM |
| Memory | LPDDR5X unified, **273 GB/s peak** |
| Achievable streaming BW | **~200 GB/s** (~73% of peak; shared CPU/GPU arbiter) |
| Driver / CUDA | 580.00 / **CUDA 13.0** (`nvcc` V13.0.48) |
| L4T | R38 rev 4.0 (2025-12-31) |
| Host gcc | 13.3.0 |
| CPU cores | 14 |
| Build flag | `-arch=sm_110a` |

Inherited from `~/gemma-cuda-hybrid/CUDA_ENGINEERING_CONSTITUTION.md` §1 — and note that
"inherited, not re-derived" is how the error below survived:
~~TMEM + tcgen05 + TMA present~~ → **`tcgen05` is ABSENT on `sm_110a`**; TMA (`cp.async.bulk`),
`barrier.cluster` and hardware FP8 `cvt.e4m3x2` **are** present, and we use none of them.
Settled by compilation in `HARDWARE_PROBE.md` — with a positive control against `sm_90a`/`sm_100a`,
because the first probe reported four features absent and three of those were artifacts of my own
inline-asm syntax. **A capability probe without a positive control is not a measurement.**
CUTLASS needs `-DCUTLASS_NVCC_ARCHS=110a` and arch-guard patching;
`cudaMallocManaged` is **not** GPU-L2-cached on Thor while `cudaMalloc` is — use `cudaMalloc` for weights.

## ⚠ Open item — power mode is NOT at MAXN

```
NV Power Mode: 120W   (nvpmodel mode 1)
```

The gemma constitution lists `nvpmodel -m 0` (MAXN) + `jetson_clocks` as a **mandatory** baseline —
without EMC pinned to max you never reach the ~200 GB/s achievable figure, and every bandwidth
number in `ROOFLINE.md` is quoted against that figure. This must be set before the first
benchmark in Gate B1, and EMC `CurrentFreq == MaxFreq` verified. Not changed yet: it is a
system-level change and the constitution warns that **rebooting kills the session**, so it is
deferred to benchmark time and will be confirmed first.

Thermal throttling is real on this box (gemma saw the same code drift 94↔108 tok/s). **All
A/B measurements must be back-to-back, median-of-N** — never compare absolute numbers across time.

## Reference point on same-bandwidth-class hardware

Poolside measured this exact model+draft pair on a **DGX Spark (GB10, 128 GB unified, 273 GB/s
class)** under vLLM 0.25.1: prefill 600–800 tok/s, decode **13–14 tok/s** autoregressive,
**15 tok/s prose / 22–24 tok/s code** with DFlash, 830–870 K KV tokens at 256 K context.

`ROOFLINE.md` reproduces all three of those numbers from first principles at ~135 GB/s effective
bandwidth, which is what calibrates our own projections.
