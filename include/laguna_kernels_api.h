// laguna_kernels_api.h — the ONE declaration of every kernel entry point.
//
// These are `extern "C"`, so the linker does no type checking: a stale declaration in one
// translation unit links happily against a changed definition in another and the arguments
// land in the wrong registers. That is not hypothetical -- adding a parameter to
// moe_gateup_rp left tests/gate_kernels.cu declaring the old 19-argument form, so the CUDA
// stream was read out of the slot the new parameter occupied, and the gate segfaulted with no
// compiler or linker diagnostic whatsoever. It is the same hazard as the `__device__`-globals
// one in README fact 14, and the same fix: one definition, included everywhere.
//
// Add a kernel here, not in the file that happens to call it.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

extern "C" {
void dequant_nvfp4(float*, const uint8_t*, const uint8_t*, float, int, int, int, cudaStream_t);
void gemm_bf16(float*, const uint16_t*, const uint16_t*, int, int, int, cudaStream_t);
void gemm_fp8(float*, const uint8_t*, const float*, const uint16_t*, int, int, int, cudaStream_t);
void gemm_bf16_seg(float* const*, const int*, int, const uint16_t*, const uint16_t*, int, int, cudaStream_t);
void gemm_fp8_seg(float* const*, const int*, int, const uint8_t*, const float*, const uint16_t*, int, int, cudaStream_t);
void gemm_fp4(float*, const uint8_t*, const uint8_t*, float, const uint16_t*, int, int, int, int, cudaStream_t);
void f32_to_bf16(uint16_t*, const float*, long, cudaStream_t);
void rmsnorm(float*, const float*, const uint16_t*, int, int, float, cudaStream_t);
void add_rms_cast(uint16_t*, float*, const float*, const uint16_t*, int, int, float, int, cudaStream_t);
void rmsnorm_heads(float*, const float*, const uint16_t*, int, int, int, float, cudaStream_t);
void rope_tables(float*, float*, const float*, int, const int*, int, float, cudaStream_t);
void rope_apply(float*, const float*, const float*, int, int, int, int, cudaStream_t);
void router(int*, float*, float*, const float*, const float*, int, int, int, float, int, cudaStream_t);
void gate_softplus(float*, const float*, int, int, int, cudaStream_t);
void swiglu(float*, const float*, const float*, long, cudaStream_t);
void add_inplace(float*, const float*, long, cudaStream_t);
void embed_rows(float*, const uint16_t*, const int*, int, int, cudaStream_t);
void tap_store(float*, const float*, int, int, const int*, int, int, cudaStream_t);
void tap_concat_cast(uint16_t*, const float*, int, int, int, int, int, cudaStream_t);
void rmsnorm_tap(float*, const uint16_t*, int, int, int, int, int, float, cudaStream_t);
void store_kv(uint8_t*, uint8_t*, const float*, const float*, float, float, int, int, int, int, const int*, cudaStream_t);
void attend(float*, const float*, const uint8_t*, const uint8_t*, float, float, int, int, int, int, int, const int*, float, cudaStream_t);
int  attend_nsplit(int, int, int);
void attend_split(float*, float*, float*, const float*, const uint8_t*, const uint8_t*, float, float, int, int, int, int, int, const int*, float, int, cudaStream_t);
void set_base(int*, int, cudaStream_t);
void inc_base(int*, int, cudaStream_t);
void moe_invert(int*, int*, int*, int*, int*, int*, const int*, int, int, int, cudaStream_t);
void moe_gateup_rp(float*, const uint8_t*, const uint8_t*, const float*, const uint8_t*, const uint8_t*,
                const float*, const uint16_t*, const int*, const int*, const int*, const int*,
                const int*, int, int, int, int, int, int, float*, cudaStream_t);
void moe_down_rp(float*, const uint8_t*, const uint8_t*, const float*, const uint16_t*, const int*,
              const int*, const int*, const int*, const int*, int, int, int, int, int, cudaStream_t);
void moe_finalize(float*, const float*, const float*, const int*, int, int, int, float, cudaStream_t);
void moe_gateup_split(float*, float*, uint16_t*, const uint8_t*, const uint8_t*, const float*,
                      const uint8_t*, const uint8_t*, const float*, const uint16_t*, const int*,
                      const int*, const int*, const int*, const int*, int, int, int, int, int,
                      int, long, cudaStream_t);
void tap_fuse(uint16_t*, const float*, const uint16_t* const*, int, int, int, int, int, float, cudaStream_t);
}
