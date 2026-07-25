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
#include <cstdlib>

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

template <int WARPS, int MM>
__global__ __launch_bounds__(WARPS * 32)
void k_gemm_bf16(float* __restrict__ out, const uint16_t* __restrict__ W,
                 const uint16_t* __restrict__ xb, int M, int N, int K) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    int n = blockIdx.x * WARPS + warp;
    if (n >= N) return;
    const int m0 = blockIdx.y * MM;
    const int mn = min(MM, M - m0);

    float acc[MM];
    #pragma unroll
    for (int m = 0; m < MM; ++m) acc[m] = 0.f;

    const uint4* wrow = (const uint4*)(W + (long)n * K);
    const int K8 = K / 8;                          // 8 bf16 per uint4
    for (int c = lane; c < K8; c += 32) {
        uint4 wv = __ldcs(wrow + c);
        const uint16_t* wh = (const uint16_t*)&wv;
        #pragma unroll
        for (int m = 0; m < MM; ++m) {
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
    for (int m = 0; m < MM; ++m) {
        if (m >= mn) break;
        float v = warp_sum(acc[m]);
        if (lane == 0) out[(long)(m0 + m) * N + n] = v;
    }
}

// Warps per block. One warp owns one whole output row, so this changes NOTHING about the
// arithmetic — it is bit-exact at any value — but it changes occupancy: this part allows 24
// CTAs and 48 warps per SM, so 1-warp blocks leave HALF the warp slots unusable. These
// GEMVs are issue/latency-bound, not bandwidth-bound (measured: the FP8 and BF16 kernels run
// the same instruction stream at the same rate, and FP8 therefore reaches exactly half the
// GB/s), so resident warps are the throughput knob. Overridable for sweeps.
static int gemm_warps() {
    static int w = -1;
    if (w < 0) {
        const char* e = getenv("LG_GEMM_WARPS");
        w = e ? atoi(e) : 4;
        if (w != 1 && w != 2 && w != 4 && w != 8) w = 4;
    }
    return w;
}

#define GEMM_DISPATCH(KERN, MMV, ...)                                                       \
    do {                                                                                    \
        const int WPB = gemm_warps();                                                       \
        dim3 g_((N + WPB - 1) / WPB, (M + (MMV) - 1) / (MMV));                              \
        switch (WPB) {                                                                      \
        case 8: KERN<8, MMV><<<g_, 8 * 32, 0, st>>>(__VA_ARGS__); break;                    \
        case 4: KERN<4, MMV><<<g_, 4 * 32, 0, st>>>(__VA_ARGS__); break;                    \
        case 2: KERN<2, MMV><<<g_, 2 * 32, 0, st>>>(__VA_ARGS__); break;                    \
        default: KERN<1, MMV><<<g_, 32, 0, st>>>(__VA_ARGS__); break;                       \
        }                                                                                   \
    } while (0)

extern "C" void gemm_bf16(float* out, const uint16_t* W, const uint16_t* xb,
                          int M, int N, int K, cudaStream_t st) {
    // Specialise the M-loop the way the MoE token loop was specialised (OPTIMIZATION_LOG #6):
    // at decode M=1, and unrolling an 8-wide loop that only ever runs once burns registers
    // and issue slots for nothing.
    if (M == 1) { GEMM_DISPATCH(k_gemm_bf16, 1, out, W, xb, M, N, K); return; }
    GEMM_DISPATCH(k_gemm_bf16, MAXM, out, W, xb, M, N, K);
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
                float2 w = fp4x2_f2(pb[j]);
                h0 = fmaf(w.x, bf2f(xh[2 * j]), h0);
                h0 = fmaf(w.y, bf2f(xh[2 * j + 1]), h0);
            }
            #pragma unroll
            for (int j = 8; j < 16; ++j) {           // second group of 16 = bytes 8..15
                float2 w = fp4x2_f2(pb[j]);
                h1 = fmaf(w.x, bf2f(xh[2 * j]), h1);
                h1 = fmaf(w.y, bf2f(xh[2 * j + 1]), h1);
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

// ---------------------------------------------------------------------------------------
// FP8 (e4m3) weight-only linear with a PER-OUTPUT-ROW scale.
//   out[m][n] = scale[n] * Σ_k e4m3(q[n][k]) · x[m][k]
// The row scale factors out of the dot product entirely, so it is applied ONCE at the end —
// the inner loop is the bf16 loop with half the bytes and no extra math.
//
// Why per-row and not per-tensor: weight-only INT8/FP8 with per-channel scales is measured
// lossless (llama.cpp Q8_0: ppl 7.32 -> 7.33), and per-row costs 4 bytes per output channel
// (37 KB on the largest attention tensor) against 28 MB saved.
// RESEARCH_FINDINGS.md: quantizing attention to 4 bits costs ~2.3 pts of recovery on a large
// MoE; at 8 bits it is near-lossless. This is the stage-1 lever and deliberately not stage 3.
// ---------------------------------------------------------------------------------------
__global__ void k_quant_fp8_rows(uint8_t* __restrict__ q, float* __restrict__ scale,
                                 const uint16_t* __restrict__ w, int N, int K) {
    int n = blockIdx.x;
    if (n >= N) return;
    const uint16_t* row = w + (long)n * K;
    float amax = 0.f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) amax = fmaxf(amax, fabsf(bf2f(row[k])));
    amax = warp_max(amax);
    __shared__ float red[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = amax;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < (blockDim.x + 31) / 32) ? red[lane] : 0.f;
        v = warp_max(v);
        if (lane == 0) red[0] = (v > 0.f) ? v / 448.f : 1.f;   // 448 = e4m3 max finite
    }
    __syncthreads();
    float s = red[0], inv = 1.f / s;
    if (threadIdx.x == 0) scale[n] = s;
    for (int k = threadIdx.x; k < K; k += blockDim.x)
        q[(long)n * K + k] = f2e4m3(bf2f(row[k]) * inv);
}
extern "C" void quant_fp8_rows(uint8_t* q, float* scale, const uint16_t* w, int N, int K,
                               cudaStream_t st) {
    k_quant_fp8_rows<<<N, 256, 0, st>>>(q, scale, w, N, K);
}

template <int WARPS, int MM>
__global__ __launch_bounds__(WARPS * 32)
void k_gemm_fp8(float* __restrict__ out, const uint8_t* __restrict__ W,
                const float* __restrict__ rs, const uint16_t* __restrict__ xb,
                int M, int N, int K) {
    const int lane = threadIdx.x & 31;
    int n = blockIdx.x * WARPS + (threadIdx.x >> 5);
    if (n >= N) return;
    const int m0 = blockIdx.y * MM;
    const int mn = min(MM, M - m0);

    float acc[MM];
    #pragma unroll
    for (int m = 0; m < MM; ++m) acc[m] = 0.f;

    // 8-byte (uint2) loads, NOT 16. One byte per weight means a 16-byte load covers 16
    // weights, which at K=3072 leaves only K/16/32 = 6 iterations per lane — half what the
    // BF16 kernel gets, and squarely in the latency-bound regime. Measured: 16-byte loads
    // gave 125 GB/s on this path against the BF16 kernel's 256. uint2 restores 12
    // iterations per lane and keeps 32x8 = 256 contiguous bytes per warp step.
    const uint2* wrow = (const uint2*)(W + (long)n * K);
    const int C = K / 8;                                   // 8 fp8 per uint2
    for (int c = lane; c < C; c += 32) {
        uint2 wv = __ldcs(wrow + c);
        const uint8_t* wb = (const uint8_t*)&wv;
        #pragma unroll
        for (int m = 0; m < MM; ++m) {
            if (m >= mn) break;
            // One 16-byte vector load, exactly as the BF16 kernel does. Indexing
            // `xb + ...` as a uint16_t* here instead compiled to EIGHT scalar 2-byte loads
            // per row, so at MM=8 the inner loop issued 64 loads where 8 would do: measured
            // 0.809 ms vs 0.146 at M=1 on the same weights, and it would have shown up as
            // "speculation does not help" rather than as a GEMM bug. (K and c*8 are both
            // multiples of 8, so the address is 16-byte aligned.)
            const uint4 xv = *(const uint4*)(xb + (long)(m0 + m) * K + c * 8);
            const uint16_t* xh = (const uint16_t*)&xv;
            float s = 0.f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) s = fmaf(e4m3f(wb[j]), bf2f(xh[j]), s);
            acc[m] += s;
        }
    }
    const float sc = rs[n];
    #pragma unroll
    for (int m = 0; m < MM; ++m) {
        if (m >= mn) break;
        float v = warp_sum(acc[m]);
        if (lane == 0) out[(long)(m0 + m) * N + n] = v * sc;   // row scale applied once
    }
}

extern "C" void gemm_fp8(float* out, const uint8_t* W, const float* rs, const uint16_t* xb,
                         int M, int N, int K, cudaStream_t st) {
    if (M == 1) { GEMM_DISPATCH(k_gemm_fp8, 1, out, W, rs, xb, M, N, K); return; }
    GEMM_DISPATCH(k_gemm_fp8, MAXM, out, W, rs, xb, M, N, K);
}
