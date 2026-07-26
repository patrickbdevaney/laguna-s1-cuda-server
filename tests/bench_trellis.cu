// bench_trellis.cu — can a 3-bit trellis decoder keep the MoE GEMV memory-bound on 20 SMs?
//
// WHAT IS ACTUALLY IN DOUBT. The wiki records that this part absorbs 16 ALU instructions per
// 32-bit word and >=10 shared-memory codebook lookups with zero bandwidth loss, which is why
// trellis decode was called "compute-free here". That was measured on the NVFP4 shape, where
// every field is nibble-aligned and every unpack is independent. A 3-bit trellis is not covered
// by it, for one specific reason:
//
//   * The trellis state is SERIAL. Weight i's reconstruction depends on the state after weight
//     i-1, so the decode chain cannot be ILP'd away inside a lane the way independent nibble
//     unpacks can. Latency, not throughput, is the risk.
//
// (Bit-straddling is NOT a real cost and this benchmark does not pretend it is: the offline
// repack gives each lane a whole 16 B chunk, so weights straddle only inside a lane's own
// registers, never across a memory boundary. That is how the production NVFP4 path is already
// laid out.)
//
// ---------------------------------------------------------------------------------------
// WHY THE FIRST VERSION OF THIS FILE WAS THROWN AWAY. It reported nvfp4 at 14.6 GB/s against a
// production kernel doing 206 GB/s -- a 14x discrepancy that was entirely the harness:
//
//   1. Every weight did its own scalar `x[...]` global load, so activation traffic was 8-11x
//      the weight traffic the benchmark claimed to be measuring, and `bytes_moved` counted only
//      W. Both the GB/s figures and the ratio to the ceiling were computed against the wrong
//      denominator.
//   2. The E2M1 LUT lived in `__constant__` and was indexed by a runtime value. That diverges
//      across a warp, so constant memory serializes into up to 8 transactions instead of
//      broadcasting -- which is why nvfp4 came in BELOW trellis despite doing less work, since
//      trellis computes its codebook arithmetically and never touches a LUT. The repo's own
//      wiki warns about exactly this trap.
//
// Both are fixed here: x is staged once in shared memory and reused across every row the block
// touches, the LUT is in shared memory as production does it, and the layout is the real
// [row][chunk][lane] repack so a warp's 32 lanes read 32 consecutive 16 B chunks (512 B, fully
// coalesced).
//
// ---------------------------------------------------------------------------------------
// THE MEASUREMENT. Same GEMV, same row count, same access pattern; only the code differs.
//   nvfp4    32 weights + 2 E4M3 scales per 16 B lane-chunk  = 0.5625 B/weight (ships today)
//   trellis  40 weights per 16 B lane-chunk                  = 0.4000 B/weight (3.2 bits)
//   stream   reads the trellis-sized buffer, decodes nothing = the bandwidth ceiling
//
// The honest headline is WEIGHTS PER SECOND, not GB/s: the two codes move different byte counts
// for identical work, so weights/s is the only figure that answers "does the byte saving turn
// into speed". If trellis/nvfp4 weights-per-second approaches the 1.406x byte ratio, the
// decoder is free and the saving converts. If it falls short, the serial chain is the wall and
// the EXL3 payoff shrinks by exactly that gap.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1);} } while(0)

// 5120 weights per row is chosen so BOTH layouts divide evenly into whole 16 B lane-chunks:
//   nvfp4   5120 / (32 weights * 32 lanes) = 5 chunks
//   trellis 5120 / (40 weights * 32 lanes) = 4 chunks
// and so the staged activation (20 KB) sits comfortably inside the 228 KB/SM shared budget.
#define K_LEN   5120
#define CH_NV   (K_LEN / (32 * 32))
#define CH_TR   (K_LEN / (40 * 32))

// QTIP-style hashed codebook: state -> approximately-Gaussian float with no table and no memory
// traffic. Mix the state, force the bits into two half-precision mantissas, add them. This is
// the property the whole approach rests on, so it is what gets benchmarked, not a stand-in.
__device__ __forceinline__ float cb3inst(uint32_t s) {
    uint32_t x = s * 0x9E3779B9u;
    x ^= x >> 15;
    uint32_t y = (x & 0x8FFF8FFFu) | 0x3B603B60u;
    __half2 h = *reinterpret_cast<__half2*>(&y);
    return __half2float(h.x) + __half2float(h.y);
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, o);
    return v;
}

// ---------------------------------------------------------------- NVFP4: 32 weights / 16 B
__global__ void k_nvfp4(const uint4* __restrict__ W, const uchar2* __restrict__ S,
                        const float* __restrict__ xg, float* __restrict__ out, int nrows) {
    extern __shared__ float sm[];
    float* sx = sm;              // K_LEN activations, staged once per block
    float* lut = sm + K_LEN;     // 8-entry E2M1 table in SHARED, not __constant__
    for (int i = threadIdx.x; i < K_LEN; i += blockDim.x) sx[i] = xg[i];
    if (threadIdx.x < 8) {
        const float e[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
        lut[threadIdx.x] = e[threadIdx.x];
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int nwarp = (gridDim.x * blockDim.x) >> 5;

    for (int row = warp; row < nrows; row += nwarp) {
        float acc = 0.f;
        #pragma unroll
        for (int c = 0; c < CH_NV; ++c) {
            size_t idx = ((size_t)row * CH_NV + c) * 32 + lane;   // warp reads 32*16B coalesced
            uint4 w = W[idx];
            uchar2 sc = S[idx];
            const uint32_t* wp = &w.x;
            int kb = c * (32 * 32) + lane * 32;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t v = wp[j];
                float s = (j < 2) ? (float)sc.x : (float)sc.y;   // one E4M3 scale per 16
                #pragma unroll
                for (int n = 0; n < 8; ++n) {
                    uint32_t c4 = (v >> (4 * n)) & 0xFu;
                    float m = lut[c4 & 7u] * ((c4 & 8u) ? -1.f : 1.f);
                    acc = fmaf(m * s, sx[kb + j * 8 + n], acc);
                }
            }
        }
        acc = warp_sum(acc);
        if (lane == 0) out[row] = acc;
    }
}

// ---------------------------------------------------------------- trellis: 40 weights / 16 B
__global__ void k_trellis(const uint4* __restrict__ W, const float* __restrict__ xg,
                          float* __restrict__ out, int nrows) {
    extern __shared__ float sm[];
    float* sx = sm;
    for (int i = threadIdx.x; i < K_LEN; i += blockDim.x) sx[i] = xg[i];
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int nwarp = (gridDim.x * blockDim.x) >> 5;

    for (int row = warp; row < nrows; row += nwarp) {
        float acc = 0.f;
        #pragma unroll
        for (int c = 0; c < CH_TR; ++c) {
            size_t idx = ((size_t)row * CH_TR + c) * 32 + lane;
            uint4 w = W[idx];
            const uint32_t* wp = &w.x;
            int kb = c * (40 * 32) + lane * 40;
            // The trellis restarts per 16 B chunk. Real EXL3 does the same -- an unbounded
            // chain would make the encoder's Viterbi pass unparallelisable and give the decoder
            // no entry point. So the serial run measured here is 40 weights long, which is the
            // length the shipping design would actually have.
            uint32_t state = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t v = wp[j];
                #pragma unroll
                for (int n = 0; n < 10; ++n) {     // 10 weights per 32-bit word, 2 bits spare
                    state = (state << 3) | ((v >> (3 * n)) & 0x7u);   // SERIAL dependency
                    acc = fmaf(cb3inst(state), sx[kb + j * 10 + n], acc);
                }
            }
        }
        acc = warp_sum(acc);
        if (lane == 0) out[row] = acc;
    }
}

// ---------------------------------------------------------------- ceiling: read, do not decode
__global__ void k_stream(const uint4* __restrict__ W, float* __restrict__ out, int nrows) {
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int nwarp = (gridDim.x * blockDim.x) >> 5;
    uint32_t a = 0;
    for (int row = warp; row < nrows; row += nwarp) {
        #pragma unroll
        for (int c = 0; c < CH_TR; ++c) {
            uint4 w = W[((size_t)row * CH_TR + c) * 32 + lane];
            a ^= w.x ^ w.y ^ w.z ^ w.w;
        }
    }
    if (lane == 0) out[warp] = (float)a;
}

int main(int argc, char** argv) {
    // Row count is set so both working sets are far larger than the 32 MB L2 -- otherwise this
    // measures cache, which is how an earlier microbenchmark on this box read 2x high.
    int nrows = argc > 1 ? atoi(argv[1]) : 262144;
    size_t bytes_nv = (size_t)nrows * CH_NV * 32 * 16;
    size_t scal_nv  = (size_t)nrows * CH_NV * 32 * 2;
    size_t bytes_tr = (size_t)nrows * CH_TR * 32 * 16;

    uint4 *Wnv, *Wtr; uchar2* S; float *x, *out;
    CK(cudaMalloc(&Wnv, bytes_nv));  CK(cudaMalloc(&S, scal_nv));
    CK(cudaMalloc(&Wtr, bytes_tr));
    CK(cudaMalloc(&x, K_LEN * sizeof(float)));
    CK(cudaMalloc(&out, (size_t)nrows * sizeof(float) + (1 << 20)));
    CK(cudaMemset(Wnv, 0x5A, bytes_nv)); CK(cudaMemset(S, 0x38, scal_nv));
    CK(cudaMemset(Wtr, 0x5A, bytes_tr)); CK(cudaMemset(x, 0x3C, K_LEN * sizeof(float)));

    int threads = 256, blocks = 20 * 8;
    size_t shm_nv = (K_LEN + 8) * sizeof(float), shm_tr = K_LEN * sizeof(float);

    // Spin the clocks up: short kernels on this part otherwise measure the idle clock, recorded
    // in the wiki as a repeated source of wrong numbers here.
    for (int i = 0; i < 200; ++i) k_stream<<<blocks, threads>>>(Wtr, out, nrows);
    CK(cudaDeviceSynchronize());

    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    auto timeit = [&](const char* name, auto fn, double bytes, double weights) {
        for (int i = 0; i < 3; ++i) fn();
        CK(cudaDeviceSynchronize());
        CK(cudaGetLastError());
        CK(cudaEventRecord(a));
        const int R = 20;
        for (int i = 0; i < R; ++i) fn();
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
        double s = (ms / R) * 1e-3;
        printf("  %-9s %8.3f ms  %7.1f GB/s  %8.2f Gweight/s\n",
               name, ms / R, bytes / s / 1e9, weights / s / 1e9);
        return weights / s;
    };

    double W_total = (double)nrows * K_LEN;
    printf("GEMV over %d rows x %d weights (%.2f Gweight), x staged in shared, LUT in shared\n",
           nrows, K_LEN, W_total / 1e9);
    printf("  nvfp4 working set %.2f GB (0.5625 B/w) | trellis %.2f GB (0.4000 B/w)\n\n",
           (bytes_nv + scal_nv) / 1e9, bytes_tr / 1e9);

    double w_stream = timeit("stream", [&]{ k_stream<<<blocks,threads>>>(Wtr, out, nrows); },
                             (double)bytes_tr, W_total);
    double w_nvfp4  = timeit("nvfp4",  [&]{ k_nvfp4<<<blocks,threads,shm_nv>>>(Wnv, S, x, out, nrows); },
                             (double)(bytes_nv + scal_nv), W_total);
    double w_trell  = timeit("trellis",[&]{ k_trellis<<<blocks,threads,shm_tr>>>(Wtr, x, out, nrows); },
                             (double)bytes_tr, W_total);

    const double byte_ratio = 0.5625 / 0.4000;     // 1.406x fewer bytes at 3.2 bits
    printf("\n  byte ratio nvfp4/trellis      = %.3fx   (the saving that is theoretically there)\n",
           byte_ratio);
    printf("  measured weights/s trellis/nvfp4 = %.3fx   (the saving that actually arrives)\n",
           w_trell / w_nvfp4);
    printf("  conversion efficiency            = %.1f%%\n", 100.0 * (w_trell / w_nvfp4) / byte_ratio);
    printf("  trellis vs pure-read ceiling     = %.1f%%   (100%% => decoder is free)\n",
           100.0 * w_trell / w_stream);
    return 0;
}
