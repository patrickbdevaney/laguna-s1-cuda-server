// laguna_kernels.cuh — device primitives shared by every Laguna kernel.
//
// Conventions:
//   * weights are BF16 stored as raw uint16_t, or NVFP4 (E2M1 codes + E4M3 per-16 group
//     scale + a PRE-INVERTED fp32 per-tensor scale, see LOOP_LOG A1.2)
//   * activations are fp32 in memory, bf16 in the GEMM inner loop (W4A16/W16A16: no
//     activation quantisation, which is what let the gemma port stay bit-exact and dodges
//     vLLM's per-step activation-quant cost at batch 1)
//   * reductions are fp32, matching the reference's `.float()` upcasts
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace lgk {

// ---------------------------------------------------------------- scalar conversions
__device__ __forceinline__ float bf2f(uint16_t b) {
    unsigned u = (unsigned)b << 16; float f; __builtin_memcpy(&f, &u, 4); return f;
}
__device__ __forceinline__ uint16_t f2bf(float f) {
    // round-to-nearest-even, matching torch's .bfloat16()
    unsigned u; __builtin_memcpy(&u, &f, 4);
    unsigned lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (uint16_t)(u >> 16);
}
__device__ __forceinline__ float e4m3f(uint8_t b) {
    __half_raw r = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3);
    return __half2float(*reinterpret_cast<__half*>(&r));
}
__device__ __forceinline__ uint8_t f2e4m3(float f) {
    return (uint8_t)__nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3);
}

// NVFP4 E2M1: sign in bit 3, magnitude index in bits 0..2 -> {0,.5,1,1.5,2,3,4,6}.
// Hardware path; the gemma port measured this beating a shared-memory LUT.
__device__ __forceinline__ __half2 fp4x2_to_h2(uint8_t byte) {
    __half2_raw r = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)byte, __NV_E2M1);
    return *reinterpret_cast<__half2*>(&r);
}
__device__ __forceinline__ float e2m1f(uint8_t code) {
    const float t[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float v = t[code & 7];
    return (code & 8) ? -v : v;
}

// ---------------------------------------------------------------- warp reduction
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// ---------------------------------------------------------------- activations
__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }
// softplus, computed the numerically stable way torch uses (threshold 20)
__device__ __forceinline__ float softplus(float x) {
    return x > 20.f ? x : log1pf(__expf(x));
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + __expf(-x)); }

} // namespace lgk
