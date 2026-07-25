// gemm.cu — G1 (NVFP4 dequant) and G2 (dense linear).
//
// RESCOPE.md §2 inverts gemma's emphasis: Laguna's dominant dense weights are **BF16**
// (attention q/k/v/o/g = 5.61 GB/step = 56 % of B_tok), not FP4. So the BF16 path is the
// primary kernel here and the FP4 path serves the routed experts.
//
// Shape regime: decode is M=1, speculative verify is M<=6 (k*≈5 per ROOFLINE.md §4). Both
// are deep in the memory-bound, weight-stationary regime — the weight is read once and
// reused across all M rows, which is where gemma measured +21.8 % on its batched lm_head.
#include "laguna_kernels.cuh"
#include <cstdio>

using namespace lgk;

// ---------------------------------------------------------------------------------------
// G1 — NVFP4 dequant.  packed[N][K/2] (2 E2M1 codes/byte, low nibble = even k)
//                      scale [N][K/G]  E4M3
//                      inv_gs           1/weight_global_scale (pre-inverted at load)
// out[N][K] fp32.  Reference: oracle/ref_laguna.py:dequant_nvfp4.
// ---------------------------------------------------------------------------------------
__global__ void k_dequant_nvfp4(float* __restrict__ out,
                                const uint8_t* __restrict__ packed,
                                const uint8_t* __restrict__ scale,
                                float inv_gs, int N, int K, int G) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;      // one thread = one BYTE = 2 vals
    long total = (long)N * (K / 2);
    if (idx >= total) return;
    int n = (int)(idx / (K / 2));
    int kh = (int)(idx % (K / 2));
    uint8_t b = packed[idx];
    float s = e4m3f(scale[(long)n * (K / G) + (2 * kh) / G]) * inv_gs;
    // NOTE: a 16-wide group spans 8 bytes, so both nibbles of a byte share one scale.
    out[(long)n * K + 2 * kh + 0] = e2m1f(b & 0x0F) * s;
    out[(long)n * K + 2 * kh + 1] = e2m1f(b >> 4)  * s;
}

extern "C" void dequant_nvfp4(float* out, const uint8_t* packed, const uint8_t* scale,
                              float inv_gs, int N, int K, int G, cudaStream_t st) {
    long total = (long)N * (K / 2);
    int T = 256;
    k_dequant_nvfp4<<<(int)((total + T - 1) / T), T, 0, st>>>(out, packed, scale, inv_gs, N, K, G);
}

// ---------------------------------------------------------------------------------------
// G2 — dense linear, BF16 weights, fp32 activations.
//     out[M][N] = x[M][K] @ W[N][K]^T
// One warp owns one output column n and streams W[n][:] with 16-byte (uint4 = 8 bf16)
// loads through __ldcs (evict-first: weights are read once, activations are reused).
//
// x is read straight from global, NOT staged in shared. Two reasons, one measured and one
// structural: gemma measured "no-shared-A, read direct from L2-cached global x" at +3 %
// (every block reads the same x, so it is hot in L2 and shared adds a pointless hop); and
// staging does not scale — at prefill M=54 a [M,K] bf16 tile is 324 KB, over the 228 KB
// limit, so the launch simply fails. This cost one failed gate to learn.
//
// WARPS_PER_BLOCK=1 maximises grid fill on a 20-SM part — gemma measured +3 % for this over
// wider blocks, because block-level parallelism is what hides the weight-load latency here
// (and is also why cp.async lost there; see OPTIMIZATION_LOG "Dead on arrival").
//
// M is tiled by MAXM so any M works: grid.y = ceil(M/MAXM).
// ---------------------------------------------------------------------------------------
#define MAXM 8

template <int WARPS>
__global__ __launch_bounds__(WARPS * 32)
void k_gemm_bf16(float* __restrict__ out, const uint16_t* __restrict__ W,
                 const uint16_t* __restrict__ xb, int M, int N, int K) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    int n = blockIdx.x * WARPS + warp;
    if (n >= N) return;
    const int m0 = blockIdx.y * MAXM;
    const int mn = min(MAXM, M - m0);

    float acc[MAXM];
    #pragma unroll
    for (int m = 0; m < MAXM; ++m) acc[m] = 0.f;

    const uint4* wrow = (const uint4*)(W + (long)n * K);
    const int K8 = K / 8;                          // 8 bf16 per uint4
    for (int c = lane; c < K8; c += 32) {
        uint4 wv = __ldcs(wrow + c);
        const uint16_t* wh = (const uint16_t*)&wv;
        #pragma unroll
        for (int m = 0; m < MAXM; ++m) {
            if (m >= mn) break;
            const uint4 xv = *(const uint4*)(xb + (long)(m0 + m) * K + c * 8);
            const uint16_t* xh = (const uint16_t*)&xv;
            float s = 0.f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) s = fmaf(bf2f(wh[j]), bf2f(xh[j]), s);
            acc[m] += s;
        }
    }
    #pragma unroll
    for (int m = 0; m < MAXM; ++m) {
        if (m >= mn) break;
        float v = warp_sum(acc[m]);
        if (lane == 0) out[(long)(m0 + m) * N + n] = v;
    }
}

extern "C" void gemm_bf16(float* out, const uint16_t* W, const uint16_t* xb,
                          int M, int N, int K, cudaStream_t st) {
    dim3 g(N, (M + MAXM - 1) / MAXM);
    k_gemm_bf16<1><<<g, 32, 0, st>>>(out, W, xb, M, N, K);
}

// fp32 activations -> bf16 staging (the GEMM's x input)
__global__ void k_f32_to_bf16(uint16_t* __restrict__ o, const float* __restrict__ i, long n) {
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (t < n) o[t] = f2bf(i[t]);
}
extern "C" void f32_to_bf16(uint16_t* o, const float* i, long n, cudaStream_t st) {
    int T = 256; k_f32_to_bf16<<<(int)((n + T - 1) / T), T, 0, st>>>(o, i, n);
}

// ---------------------------------------------------------------------------------------
// G2b — dense linear with NVFP4 weights (routed experts, and later the §5 self-quantised
// attention).  out[M][N] = x[M][K] @ dequant(W)[N][K]^T, dequantised in-register.
// ---------------------------------------------------------------------------------------
template <int WARPS>
__global__ __launch_bounds__(WARPS * 32)
void k_gemm_fp4(float* __restrict__ out,
                const uint8_t* __restrict__ packed, const uint8_t* __restrict__ scale,
                float inv_gs, const uint16_t* __restrict__ xb, int M, int N, int K, int G) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    int n = blockIdx.x * WARPS + warp;
    if (n >= N) return;
    const int m0 = blockIdx.y * MAXM;
    const int mn = min(MAXM, M - m0);

    float acc[MAXM];
    #pragma unroll
    for (int m = 0; m < MAXM; ++m) acc[m] = 0.f;

    // 16 bytes of codes = 32 weights = 2 groups of 16
    const uint4* prow = (const uint4*)(packed + (long)n * (K / 2));
    const uint8_t* srow = scale + (long)n * (K / G);
    const int C = K / 32;
    for (int c = lane; c < C; c += 32) {
        uint4 pv = __ldcs(prow + c);
        const uint8_t* pb = (const uint8_t*)&pv;
        float s0 = e4m3f(srow[(c * 32) / G]) * inv_gs;
        float s1 = e4m3f(srow[(c * 32 + 16) / G]) * inv_gs;
        #pragma unroll
        for (int m = 0; m < MAXM; ++m) {
            if (m >= mn) break;
            const uint16_t* xh = xb + (long)(m0 + m) * K + c * 32;
            float h0 = 0.f, h1 = 0.f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {            // first group of 16 = bytes 0..7
                uint8_t b = pb[j];
                h0 = fmaf(e2m1f(b & 0x0F), bf2f(xh[2 * j]), h0);
                h0 = fmaf(e2m1f(b >> 4),   bf2f(xh[2 * j + 1]), h0);
            }
            #pragma unroll
            for (int j = 8; j < 16; ++j) {           // second group of 16 = bytes 8..15
                uint8_t b = pb[j];
                h1 = fmaf(e2m1f(b & 0x0F), bf2f(xh[2 * j]), h1);
                h1 = fmaf(e2m1f(b >> 4),   bf2f(xh[2 * j + 1]), h1);
            }
            acc[m] += h0 * s0 + h1 * s1;
        }
    }
    #pragma unroll
    for (int m = 0; m < MAXM; ++m) {
        if (m >= mn) break;
        float v = warp_sum(acc[m]);
        if (lane == 0) out[(long)(m0 + m) * N + n] = v;
    }
}

extern "C" void gemm_fp4(float* out, const uint8_t* packed, const uint8_t* scale,
                         float inv_gs, const uint16_t* xb, int M, int N, int K, int G,
                         cudaStream_t st) {
    dim3 g(N, (M + MAXM - 1) / MAXM);
    k_gemm_fp4<1><<<g, 32, 0, st>>>(out, packed, scale, inv_gs, xb, M, N, K, G);
}
