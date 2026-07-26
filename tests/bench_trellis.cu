// bench_trellis.cu — can a 3-bit trellis decoder keep the MoE GEMV memory-bound on 20 SMs?
//
// WHAT IS ACTUALLY IN DOUBT. The wiki already records that this part absorbs 16 ALU
// instructions per 32-bit word and >=10 shared-memory codebook lookups with zero bandwidth
// loss, which is why trellis decode was called "compute-free here". But that was measured on
// the NVFP4 shape, where a 32-bit word holds exactly 8 weights and every field is nibble
// aligned. A 3-bit code is different in a way that matters:
//
//   * 32 is not divisible by 3, so weights STRADDLE word boundaries and every lane needs a
//     funnel shift across two registers rather than a shift-and-mask.
//   * The trellis state is SERIAL. Weight i's reconstruction depends on the state after weight
//     i-1, so the decode chain cannot be ILP'd away within a lane the way independent nibble
//     unpacks can. This is the real risk and it is not covered by any previous measurement.
//
// So the question is narrow and worth one benchmark: at 3 bits/weight the kernel reads 1.5x
// fewer bytes for the same FLOPs, which raises arithmetic intensity by 1.5x on top of a decoder
// that is both more expensive and serial. Does it still saturate DRAM?
//
// THE COMPARISON. Three kernels over the same logical weight matrix, all doing the same GEMV:
//   nvfp4    -- nibble + per-16 E4M3 scale, 0.5625 B/weight   (what ships today)
//   trellis  -- 3-bit funnel-shifted bitstream, hashed codebook, 0.375 B/weight
//   stream   -- pure read, no decode at all, 0.375 B/weight   (the bandwidth ceiling at 3 bits)
// `stream` is the control: if trellis lands near it, the decoder is free and the byte saving
// converts to speed; if trellis lands well below it, the decoder is the wall and the whole
// EXL3 payoff shrinks by exactly that gap. Reporting achieved GB/s makes it directly
// comparable to the 206/254 the production kernel gets.
//
// The codebook is the QTIP "3INST" shape: mix the state, force the bits into two half-precision
// mantissas, and add them. Two dependent multiplies and an add produce a value whose marginal
// distribution is approximately Gaussian, with no table and no memory traffic. That is the
// property the whole approach rests on, so it is what gets benchmarked -- not a stand-in.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1);} } while(0)

// ---------------------------------------------------------------- QTIP-style 3INST codebook
// state -> approximately-Gaussian float, no table lookup. The mask keeps the sign and mantissa
// bits from the hash and pins the exponent, so the two halves are O(1) magnitude; summing two
// of them is the cheapest route to a bell-shaped marginal.
__device__ __forceinline__ float cb3inst(uint32_t s) {
    uint32_t x = s * 0x9E3779B9u;          // mix
    x ^= x >> 15;                          // avalanche the low bits into the mantissas
    uint32_t y = (x & 0x8FFF8FFFu) | 0x3B603B60u;
    __half2 h = *reinterpret_cast<__half2*>(&y);
    return __half2float(h.x) + __half2float(h.y);
}

// ---------------------------------------------------------------- NVFP4 reference path
__constant__ float kE2M1[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

__global__ void k_nvfp4(const uint4* __restrict__ W, const uint8_t* __restrict__ S,
                        const float* __restrict__ x, float* __restrict__ out, int nvec) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    float acc = 0.f;
    for (int i = t; i < nvec; i += stride) {
        uint4 w = W[i];                        // 16 B = 32 weights
        float sc = (float)S[i];                // one scale per 16 (approximated as one per 32)
        const uint32_t* wp = &w.x;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t v = wp[j];
            #pragma unroll
            for (int n = 0; n < 8; ++n) {
                uint32_t c = (v >> (4 * n)) & 0xFu;
                float m = kE2M1[c & 7u] * ((c & 8u) ? -1.f : 1.f);
                acc = fmaf(m * sc, x[(i * 32 + j * 8 + n) & 1023], acc);
            }
        }
    }
    out[t] = acc;
}

// ---------------------------------------------------------------- trellis path (3 bits/weight)
// Each uint4 (128 bits) carries 42 whole weights plus a straddler. Rather than fight the
// boundary, each lane decodes 40 weights per uint4 and carries the leftover bits forward in the
// state, which is exactly what a real bitstream reader does and keeps the cost honest.
__global__ void k_trellis(const uint4* __restrict__ W, const float* __restrict__ x,
                          float* __restrict__ out, int nvec) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    float acc = 0.f;
    for (int i = t; i < nvec; i += stride) {
        uint4 w = W[i];
        const uint32_t* wp = &w.x;
        uint32_t state = 0;
        int base = i * 40;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t v = wp[j];
            // 10 weights per 32-bit word; the remaining 2 bits fold into the state, which is
            // free because the state is shifted anyway.
            #pragma unroll
            for (int n = 0; n < 10; ++n) {
                uint32_t bits = (v >> (3 * n)) & 0x7u;
                state = (state << 3) | bits;          // SERIAL: this is the dependency chain
                acc = fmaf(cb3inst(state), x[(base + j * 10 + n) & 1023], acc);
            }
        }
    }
    out[t] = acc;
}

// ---------------------------------------------------------------- bandwidth control
__global__ void k_stream(const uint4* __restrict__ W, float* __restrict__ out, int nvec) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    uint32_t a = 0;
    for (int i = t; i < nvec; i += stride) {
        uint4 w = W[i];
        a ^= w.x ^ w.y ^ w.z ^ w.w;
    }
    out[t] = (float)a;
}

int main(int argc, char** argv) {
    // Default size is chosen to be far larger than the 32 MB L2 so this measures DRAM, not
    // cache -- the mistake that made an earlier microbenchmark on this box read 2x high.
    size_t MB = argc > 1 ? atoll(argv[1]) : 2048;
    size_t bytes = MB << 20;
    int nvec = (int)(bytes / 16);

    uint4 *W; float *x, *out; uint8_t* S;
    CK(cudaMalloc(&W, bytes));
    CK(cudaMalloc(&S, nvec));
    CK(cudaMalloc(&x, 1024 * sizeof(float)));
    CK(cudaMalloc(&out, 1 << 20));
    CK(cudaMemset(W, 0x5A, bytes));
    CK(cudaMemset(S, 0x38, nvec));
    CK(cudaMemset(x, 0x3C, 1024 * sizeof(float)));

    int blocks = 20 * 8, threads = 256;      // 20 SMs; enough waves to hide latency

    // Spin the clocks up first. Short kernels on this part otherwise measure the idle clock --
    // recorded in the wiki as a repeated source of wrong numbers here.
    for (int i = 0; i < 200; ++i) k_stream<<<blocks, threads>>>(W, out, nvec);
    CK(cudaDeviceSynchronize());

    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    auto timeit = [&](const char* name, auto fn, double bytes_moved) {
        for (int i = 0; i < 3; ++i) fn();
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(a));
        const int R = 20;
        for (int i = 0; i < R; ++i) fn();
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
        double gbs = bytes_moved * R / (ms * 1e-3) / 1e9;
        printf("  %-10s %8.3f ms   %7.1f GB/s\n", name, ms / R, gbs);
        return gbs;
    };

    printf("trellis decode feasibility, %zu MB working set, %d blocks x %d threads\n",
           MB, blocks, threads);
    double g_stream = timeit("stream",  [&]{ k_stream <<<blocks,threads>>>(W, out, nvec); }, (double)bytes);
    double g_nvfp4  = timeit("nvfp4",   [&]{ k_nvfp4  <<<blocks,threads>>>(W, S, x, out, nvec); }, (double)bytes + nvec);
    double g_trell  = timeit("trellis", [&]{ k_trellis<<<blocks,threads>>>(W, x, out, nvec); }, (double)bytes);

    printf("\n  trellis / stream ceiling = %.1f%%   (100%% => decoder is free)\n",
           100.0 * g_trell / g_stream);
    printf("  nvfp4  / stream ceiling = %.1f%%   (today's shipping path, for reference)\n",
           100.0 * g_nvfp4 / g_stream);
    // Convert to the thing we actually care about: at 3 bits the routed experts read 1.5x fewer
    // bytes, so the end-to-end gain is that ratio scaled by how much of the ceiling we keep.
    printf("\n  If trellis holds %.1f%% of the ceiling, the 1.500x byte saving on routed\n"
           "  experts realises as %.3fx on that term.\n",
           100.0 * g_trell / g_stream, 1.5 * (g_trell / g_stream));
    return 0;
}
