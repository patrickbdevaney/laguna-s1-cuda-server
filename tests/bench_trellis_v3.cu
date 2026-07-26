// bench_trellis_v3.cu — is the trellis loss intrinsic to the approach, or just to my decoder?
//
// v2 measured trellis at 220 Gweight/s against NVFP4's 260, i.e. 0.845x -- the 1.406x byte
// saving not merely failing to arrive but going backwards. Before accepting "trellis does not
// pay on Thor" as the verdict that kills a whole subsystem, the obvious objection has to be
// answered: THAT decoder was expensive. It cost roughly 8 ALU ops per weight
//
//     mul, shr, xor, and, or, cvt, cvt, add          -> ONE weight
//
// whereas QTIP's actual 3INST kernel produces TWO values per hash, because a 32-bit hash holds
// two half-precision lanes and there is no reason to throw one away by summing them. If the
// real cost is ~3 ops/weight rather than ~8, the entire conclusion flips.
//
// So this benchmark separates the CODE from the IMPLEMENTATION with four decoders over an
// identical layout, identical byte count, identical access pattern:
//
//   nvfp4    reference: 32 weights + 2 E4M3 scales per 16 B lane-chunk (0.5625 B/weight)
//   trellis_A  v2's decoder, one Gaussian per hash               ~8 ops/weight
//   trellis_B  QTIP shape, TWO weights per hash                  ~3 ops/weight
//   trellis_C  shared-memory codebook indexed by the state       ~3 ops + 1 smem load/weight
//
// trellis_C is worth measuring specifically because the repo's own on-device result says this
// part absorbs ">=10 shared-memory codebook lookups per 32-bit word with zero bandwidth loss".
// At 3 bits a 32-bit word holds 10.67 weights, so a per-weight smem lookup sits exactly at that
// measured boundary -- this either confirms the earlier finding at the real operating point or
// refutes it.
//
// Each also runs in a NOMEM variant (same arithmetic, weights synthesised in-register, no W
// traffic) to expose each decoder's pure ALU ceiling. If a decoder's real throughput is far
// below its NOMEM ceiling it is memory-bound and the byte saving should help; if it sits at its
// ceiling it is compute-bound and reading fewer bytes cannot help at all. That distinction is
// the whole verdict, and no amount of GB/s reporting substitutes for it.
//
// Headline stays WEIGHTS PER SECOND: per token the model must process a fixed number of
// weights, so time = weights / (weights per second). Bytes only matter insofar as they move
// that number.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1);} } while(0)

#define K_LEN   5120
#define CH_NV   (K_LEN / (32 * 32))     // 5 chunks: 32 weights/lane/chunk
#define CH_TR   (K_LEN / (40 * 32))     // 4 chunks: 40 weights/lane/chunk

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, o);
    return v;
}

// ---- A: v2's decoder. One hash -> one value, by summing the two half lanes.
__device__ __forceinline__ float cbA(uint32_t s) {
    uint32_t x = s * 0x9E3779B9u;
    x ^= x >> 15;
    uint32_t y = (x & 0x8FFF8FFFu) | 0x3B603B60u;
    __half2 h = *reinterpret_cast<__half2*>(&y);
    return __half2float(h.x) + __half2float(h.y);
}

// ---- B: QTIP shape. One hash -> TWO values, keeping both half lanes.
__device__ __forceinline__ float2 cbB(uint32_t s) {
    uint32_t x = s * 0x9E3779B9u;
    x ^= x >> 15;
    uint32_t y = (x & 0x8FFF8FFFu) | 0x3B603B60u;
    return __half22float2(*reinterpret_cast<__half2*>(&y));
}

__global__ void k_nvfp4(const uint4* __restrict__ W, const uchar2* __restrict__ S,
                        const float* __restrict__ xg, float* __restrict__ out, int nrows) {
    extern __shared__ float sm[];
    float* sx = sm; float* lut = sm + K_LEN;
    for (int i = threadIdx.x; i < K_LEN; i += blockDim.x) sx[i] = xg[i];
    if (threadIdx.x < 8) { const float e[8]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f}; lut[threadIdx.x]=e[threadIdx.x]; }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int nwarp = (gridDim.x*blockDim.x) >> 5;
    for (int row = warp; row < nrows; row += nwarp) {
        float acc = 0.f;
        #pragma unroll
        for (int c = 0; c < CH_NV; ++c) {
            size_t idx = ((size_t)row*CH_NV + c)*32 + lane;
            uint4 w = W[idx]; uchar2 sc = S[idx];
            const uint32_t* wp = &w.x; int kb = c*(32*32) + lane*32;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t v = wp[j]; float s = (j<2)?(float)sc.x:(float)sc.y;
                #pragma unroll
                for (int n = 0; n < 8; ++n) {
                    uint32_t c4 = (v >> (4*n)) & 0xFu;
                    acc = fmaf(lut[c4&7u]*((c4&8u)?-1.f:1.f)*s, sx[kb+j*8+n], acc);
                }
            }
        }
        acc = warp_sum(acc); if (lane==0) out[row]=acc;
    }
}

#define TRELLIS_BODY(DECODE)                                                     \
    extern __shared__ float sm[];                                                \
    float* sx = sm;                                                              \
    for (int i = threadIdx.x; i < K_LEN; i += blockDim.x) sx[i] = xg[i];         \
    DECLARE_CB                                                                   \
    __syncthreads();                                                             \
    const int lane = threadIdx.x & 31;                                           \
    const int warp = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;                  \
    const int nwarp = (gridDim.x*blockDim.x) >> 5;                                \
    for (int row = warp; row < nrows; row += nwarp) {                             \
        float acc = 0.f;                                                          \
        for (int c = 0; c < CH_TR; ++c) {                                          \
            uint4 w = W[((size_t)row*CH_TR + c)*32 + lane];                        \
            const uint32_t* wp = &w.x; int kb = c*(40*32) + lane*40;               \
            uint32_t state = 0; (void)state;                                       \
            DECODE                                                                 \
        }                                                                          \
        acc = warp_sum(acc); if (lane==0) out[row]=acc;                            \
    }

// ---- A
#define DECLARE_CB
__global__ void k_trA(const uint4* __restrict__ W, const float* __restrict__ xg,
                      float* __restrict__ out, int nrows) {
    TRELLIS_BODY(
        _Pragma("unroll") for (int j=0;j<4;++j) {
            uint32_t v = wp[j];
            _Pragma("unroll") for (int n=0;n<10;++n) {
                state = (state<<3) | ((v>>(3*n))&0x7u);
                acc = fmaf(cbA(state), sx[kb+j*10+n], acc);
            }
        }
    )
}
#undef DECLARE_CB

// ---- B: two weights per hash, state advances 6 bits at a time
#define DECLARE_CB
__global__ void k_trB(const uint4* __restrict__ W, const float* __restrict__ xg,
                      float* __restrict__ out, int nrows) {
    TRELLIS_BODY(
        _Pragma("unroll") for (int j=0;j<4;++j) {
            uint32_t v = wp[j];
            _Pragma("unroll") for (int n=0;n<5;++n) {
                state = (state<<6) | ((v>>(6*n))&0x3Fu);
                float2 f = cbB(state);
                int k = kb + j*10 + n*2;
                acc = fmaf(f.x, sx[k],   acc);
                acc = fmaf(f.y, sx[k+1], acc);
            }
        }
    )
}
#undef DECLARE_CB

// ---- C: shared-memory codebook, 256 entries indexed by the low state bits
#define DECLARE_CB                                                               \
    float* cbs = sm + K_LEN;                                                     \
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {                        \
        uint32_t h = (uint32_t)i * 0x9E3779B9u; h ^= h >> 15;                    \
        uint32_t y = (h & 0x8FFF8FFFu) | 0x3B603B60u;                            \
        __half2 hh = *reinterpret_cast<__half2*>(&y);                            \
        cbs[i] = __half2float(hh.x) + __half2float(hh.y);                        \
    }
__global__ void k_trC(const uint4* __restrict__ W, const float* __restrict__ xg,
                      float* __restrict__ out, int nrows) {
    TRELLIS_BODY(
        _Pragma("unroll") for (int j=0;j<4;++j) {
            uint32_t v = wp[j];
            _Pragma("unroll") for (int n=0;n<10;++n) {
                state = (state<<3) | ((v>>(3*n))&0x7u);
                acc = fmaf(cbs[state & 255u], sx[kb+j*10+n], acc);
            }
        }
    )
}
#undef DECLARE_CB

// ---- NOMEM ceilings: identical arithmetic, W synthesised in-register, zero weight traffic.
__global__ void k_nomemA(const float* __restrict__ xg, float* __restrict__ out, int nrows, uint32_t seed) {
    extern __shared__ float sm[]; float* sx = sm;
    for (int i = threadIdx.x; i < K_LEN; i += blockDim.x) sx[i] = xg[i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int nwarp = (gridDim.x*blockDim.x) >> 5;
    for (int row = warp; row < nrows; row += nwarp) {
        float acc = 0.f;
        for (int c = 0; c < CH_TR; ++c) {
            uint32_t v0 = seed ^ (uint32_t)(row*31 + c); int kb = c*(40*32) + lane*40;
            uint32_t state = 0;
            #pragma unroll
            for (int j=0;j<4;++j) { uint32_t v = v0*(j+1);
                #pragma unroll
                for (int n=0;n<10;++n) { state=(state<<3)|((v>>(3*n))&7u);
                    acc = fmaf(cbA(state), sx[kb+j*10+n], acc); } }
        }
        acc = warp_sum(acc); if (lane==0) out[row]=acc;
    }
}
__global__ void k_nomemB(const float* __restrict__ xg, float* __restrict__ out, int nrows, uint32_t seed) {
    extern __shared__ float sm[]; float* sx = sm;
    for (int i = threadIdx.x; i < K_LEN; i += blockDim.x) sx[i] = xg[i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int nwarp = (gridDim.x*blockDim.x) >> 5;
    for (int row = warp; row < nrows; row += nwarp) {
        float acc = 0.f;
        for (int c = 0; c < CH_TR; ++c) {
            uint32_t v0 = seed ^ (uint32_t)(row*31 + c); int kb = c*(40*32) + lane*40;
            uint32_t state = 0;
            #pragma unroll
            for (int j=0;j<4;++j) { uint32_t v = v0*(j+1);
                #pragma unroll
                for (int n=0;n<5;++n) { state=(state<<6)|((v>>(6*n))&0x3Fu);
                    float2 f = cbB(state); int k = kb+j*10+n*2;
                    acc = fmaf(f.x, sx[k], acc); acc = fmaf(f.y, sx[k+1], acc); } }
        }
        acc = warp_sum(acc); if (lane==0) out[row]=acc;
    }
}

int main(int argc, char** argv) {
    int nrows = argc > 1 ? atoi(argv[1]) : 262144;
    size_t bnv = (size_t)nrows*CH_NV*32*16, snv = (size_t)nrows*CH_NV*32*2;
    size_t btr = (size_t)nrows*CH_TR*32*16;

    uint4 *Wnv,*Wtr; uchar2* S; float *x,*out;
    CK(cudaMalloc(&Wnv,bnv)); CK(cudaMalloc(&S,snv)); CK(cudaMalloc(&Wtr,btr));
    CK(cudaMalloc(&x,K_LEN*sizeof(float)));
    CK(cudaMalloc(&out,(size_t)nrows*sizeof(float)+(1<<20)));
    CK(cudaMemset(Wnv,0x5A,bnv)); CK(cudaMemset(S,0x38,snv));
    CK(cudaMemset(Wtr,0x5A,btr)); CK(cudaMemset(x,0x3C,K_LEN*sizeof(float)));

    int threads=256, blocks=20*8;
    size_t shm_nv=(K_LEN+8)*sizeof(float), shm_tr=K_LEN*sizeof(float),
           shm_c=(K_LEN+256)*sizeof(float);

    for (int i=0;i<200;++i) k_trA<<<blocks,threads,shm_tr>>>(Wtr,x,out,nrows);
    CK(cudaDeviceSynchronize());

    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    double W_total=(double)nrows*K_LEN;
    auto run=[&](const char* name, auto fn, double bytes){
        for(int i=0;i<3;++i) fn();
        CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        CK(cudaEventRecord(a)); const int R=20;
        for(int i=0;i<R;++i) fn();
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms=0; CK(cudaEventElapsedTime(&ms,a,b)); double s=(ms/R)*1e-3;
        double gw=W_total/s/1e9;
        printf("  %-12s %8.3f ms  %7.1f GB/s  %8.2f Gweight/s\n", name, ms/R,
               bytes/s/1e9, gw);
        return gw;
    };

    printf("GEMV %d rows x %d weights (%.2f Gweight); x + codebook in shared\n\n",
           nrows, K_LEN, W_total/1e9);
    double nv = run("nvfp4",     [&]{k_nvfp4<<<blocks,threads,shm_nv>>>(Wnv,S,x,out,nrows);}, (double)(bnv+snv));
    double A  = run("trellis_A", [&]{k_trA  <<<blocks,threads,shm_tr>>>(Wtr,x,out,nrows);},   (double)btr);
    double B  = run("trellis_B", [&]{k_trB  <<<blocks,threads,shm_tr>>>(Wtr,x,out,nrows);},   (double)btr);
    double C  = run("trellis_C", [&]{k_trC  <<<blocks,threads,shm_c >>>(Wtr,x,out,nrows);},   (double)btr);
    printf("\n  ALU ceilings (no weight traffic at all):\n");
    double cA = run("nomem_A",   [&]{k_nomemA<<<blocks,threads,shm_tr>>>(x,out,nrows,0x1234);}, 0.0);
    double cB = run("nomem_B",   [&]{k_nomemB<<<blocks,threads,shm_tr>>>(x,out,nrows,0x1234);}, 0.0);

    printf("\n  ---- verdict ----\n");
    printf("  nvfp4 reference                 %8.2f Gweight/s\n", nv);
    printf("  best trellis                    %8.2f Gweight/s  (%.3fx nvfp4)\n",
           (B>C?(B>A?B:A):(C>A?C:A)), (B>C?(B>A?B:A):(C>A?C:A))/nv);
    printf("  trellis_B vs its own ALU ceiling  %.1f%%  (near 100%% => compute-bound,\n"
           "                                            byte savings CANNOT help)\n", 100.0*B/cB);
    printf("  trellis_A vs its own ALU ceiling  %.1f%%\n", 100.0*A/cA);
    printf("\n  Per token the model must process a FIXED weight count, so a trellis only pays\n"
           "  if its Gweight/s EXCEEDS nvfp4's. Byte savings are irrelevant when compute-bound.\n");
    return 0;
}
