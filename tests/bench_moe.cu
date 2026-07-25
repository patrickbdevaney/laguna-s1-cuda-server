// bench_moe.cu — bisect the NVFP4 MoE GEMV gap without hardware counters.
//
// The real kernel reaches 169–177 GB/s where the dense BF16/FP8 GEMVs on the same device reach
// 205–254. Four structural hypotheses have been falsified (OPTIMIZATION_LOG #14, #24, #25,
// #26) and `ncu` counters are unavailable (ERR_NVGPUCTRPERM), so this bisects the kernel by
// *removing one thing at a time* and measuring what the removal buys. Each variant reads the
// SAME code stream in the SAME layout at the SAME occupancy; only the work per byte changes.
//
//   FULL      codes + per-16-group E4M3 scales + hardware FP4 decode + FMA   (the real kernel)
//   NOSCALE   codes + decode + FMA, one constant scale                       (kills the scale stream)
//   NODEQ     codes only, accumulated as raw integers                        (kills decode + scale)
//   STREAM    codes only, one accumulate per 16 B                            (pure read bound)
//
// FULL→NOSCALE isolates the second memory stream (1 byte per lane per 16 code bytes, which is
// a 32 B sector per warp). NOSCALE→NODEQ isolates the FP4 decode arithmetic. NODEQ→STREAM
// isolates the FMA chain. Whatever step is large is the limiter; if none are, the kernel is
// already at the shape's ceiling and the remaining gap is not recoverable by these means.
//
// Weights are rotated through a >L2 arena so every timed iteration touches cold bytes — the
// mistake that made the first version of `bench_kernels` measure L2 instead of DRAM.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} }while(0)

__device__ __forceinline__ float e4m3f(uint8_t v) {
    __nv_fp8_e4m3 t; t.__x = v; return (float)t;
}
__device__ __forceinline__ float2 fp4x2_f2(uint8_t b) {
    __half2_raw h = __nv_cvt_fp4x2_to_halfraw2(*(const __nv_fp4x2_storage_t*)&b, __NV_E2M1);
    return __half22float2(*(__half2*)&h);
}
__device__ __forceinline__ float bf2f(uint16_t v) {
    uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4); return f;
}

#define RPNB 32
#define RPU  4
#define LG_XW4(P) const uint4 W0_ = *(const uint4*)(P),      W1_ = *(const uint4*)((P) + 8), \
                             W2_ = *(const uint4*)((P) + 16), W3_ = *(const uint4*)((P) + 24)
#define LG_XLO(w) bf2f((uint16_t)((w) & 0xffffu))
#define LG_XHI(w) bf2f((uint16_t)((w) >> 16))

// MODE: 0 FULL, 1 NOSCALE, 2 NODEQ, 3 STREAM, 4 VECSCALE
//
// VECSCALE keeps every scale byte but changes their LAYOUT. Today the array is
// [n_block][group][lane], so a lane reads ONE byte per group and needs 8 of them to cover the
// RPU=4 code chunks in flight -- eight separate load instructions for 8 bytes. Regrouped as
// [n_block][group/8][lane][8] a lane reads all eight as a single 8-byte load, and the warp
// still reads 32*8 = 256 contiguous bytes, so coalescing is unchanged. Same bytes, one
// instruction instead of eight.
template <int MODE>
__global__ __launch_bounds__(128)
void k_bisect(float* __restrict__ out, const uint8_t* __restrict__ gp,
              const uint8_t* __restrict__ gs, const uint8_t* __restrict__ up,
              const uint8_t* __restrict__ us, const uint16_t* __restrict__ xb,
              int H, int MI, int GRP, int nexp) {
    const int e    = blockIdx.x;
    const int lane = threadIdx.x;
    const int nb   = blockIdx.y * blockDim.y + threadIdx.y;
    const int nblocks = MI / RPNB;
    if (nb >= nblocks || e >= nexp) return;
    const int n = nb * RPNB + lane;

    const int C = H / 32, NG = H / GRP;
    const size_t eb = (size_t)e * nblocks;
    const uint8_t* gbase = gp + ((eb + nb) * C) * RPNB * 16;
    const uint8_t* ubase = up + ((eb + nb) * C) * RPNB * 16;
    const uint8_t* gsb   = gs + ((eb + nb) * NG) * RPNB;
    const uint8_t* usb   = us + ((eb + nb) * NG) * RPNB;

    float accg = 0.f, accu = 0.f;
    const uint16_t* xrow = xb;

    for (int c0 = 0; c0 < C; c0 += RPU) {
        uint4 gv[RPU], uv[RPU];
        float gsc[RPU][2], usc[RPU][2];
        #pragma unroll
        for (int u = 0; u < RPU; ++u) {
            const int c = c0 + u;
            if (c < C) {
                gv[u] = __ldcs((const uint4*)(gbase + ((size_t)c * RPNB + lane) * 16));
                uv[u] = __ldcs((const uint4*)(ubase + ((size_t)c * RPNB + lane) * 16));
                if (MODE == 0) {
                    const int g0 = (c * 32) / GRP, g1 = (c * 32 + 16) / GRP;
                    gsc[u][0] = e4m3f(gsb[(size_t)g0 * RPNB + lane]);
                    gsc[u][1] = e4m3f(gsb[(size_t)g1 * RPNB + lane]);
                    usc[u][0] = e4m3f(usb[(size_t)g0 * RPNB + lane]);
                    usc[u][1] = e4m3f(usb[(size_t)g1 * RPNB + lane]);
                } else if (MODE == 4) {
                    // one 8-byte load covers the 8 groups this RPU window needs
                    if (u == 0) {
                        const int gb = (c0 * 32) / GRP / 8;          // which block of 8 groups
                        const uint2 gv8 = *(const uint2*)(gsb + ((size_t)gb * RPNB + lane) * 8);
                        const uint2 uv8 = *(const uint2*)(usb + ((size_t)gb * RPNB + lane) * 8);
                        const uint8_t* gb8 = (const uint8_t*)&gv8;
                        const uint8_t* ub8 = (const uint8_t*)&uv8;
                        #pragma unroll
                        for (int q = 0; q < RPU; ++q) {
                            gsc[q][0] = e4m3f(gb8[2 * q]); gsc[q][1] = e4m3f(gb8[2 * q + 1]);
                            usc[q][0] = e4m3f(ub8[2 * q]); usc[q][1] = e4m3f(ub8[2 * q + 1]);
                        }
                    }
                } else {
                    gsc[u][0] = gsc[u][1] = usc[u][0] = usc[u][1] = 1.0f;
                }
            }
        }
        #pragma unroll
        for (int u = 0; u < RPU; ++u) {
            const int c = c0 + u;
            if (c >= C) break;
            const uint8_t* gbb = (const uint8_t*)&gv[u];
            const uint8_t* ubb = (const uint8_t*)&uv[u];
            // The activation row must be read the SAME way in every variant, or the
            // comparison is meaningless. Two defects were found here after the fact:
            //   * MODE 2 and 3 `continue`d before touching x at all, so they were compared
            //     against a FULL that carried the entire activation stream -- the residual
            //     that got attributed to "the scale stream" was partly activations.
            //   * x was read as 32 scalar 2-byte loads, which is exactly the defect entry #20
            //     removed from production. 93 % of this harness's load requests came from an
            //     artifact the real kernel does not have.
            // Both are fixed: every variant now issues the production uint4 loads.
            LG_XW4(xrow + c * 32);
            const float xs = LG_XLO(W0_.x) + LG_XHI(W0_.x) + LG_XLO(W3_.w) + LG_XHI(W3_.w);
            if (MODE == 3) {                        // pure read: one accumulate per 16 B
                accg += (float)gbb[0] * xs; accu += (float)ubb[0] * xs;
                continue;
            }
            if (MODE == 2) {                        // no FP4 decode, raw byte FMAs
                float g = 0.f, u2 = 0.f;
                #pragma unroll
                for (int j = 0; j < 16; ++j) { g += (float)gbb[j]; u2 += (float)ubb[j]; }
                accg += g * xs; accu += u2 * xs;
                continue;
            }
            const uint16_t* xh = xrow + c * 32;
            float g0 = 0.f, g1 = 0.f, u0 = 0.f, u1 = 0.f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float x0 = bf2f(xh[2 * j]), x1 = bf2f(xh[2 * j + 1]);
                float2 gw = fp4x2_f2(gbb[j]), uw = fp4x2_f2(ubb[j]);
                g0 = fmaf(gw.x, x0, g0); g0 = fmaf(gw.y, x1, g0);
                u0 = fmaf(uw.x, x0, u0); u0 = fmaf(uw.y, x1, u0);
            }
            #pragma unroll
            for (int j = 8; j < 16; ++j) {
                float x0 = bf2f(xh[2 * j]), x1 = bf2f(xh[2 * j + 1]);
                float2 gw = fp4x2_f2(gbb[j]), uw = fp4x2_f2(ubb[j]);
                g1 = fmaf(gw.x, x0, g1); g1 = fmaf(gw.y, x1, g1);
                u1 = fmaf(uw.x, x0, u1); u1 = fmaf(uw.y, x1, u1);
            }
            accg += g0 * gsc[u][0] + g1 * gsc[u][1];
            accu += u0 * usc[u][0] + u1 * usc[u][1];
        }
    }
    out[(size_t)e * MI + n] = accg + accu;
}

static uint8_t *gp, *gs, *up, *us; static uint16_t* xb; static float* out;
static int gH, gMI, gGRP, gNEXP, gMODE;
static size_t code_bytes_per_exp, scale_bytes_per_exp, arena_exp;
static int rot = 0;

template <int MODE>
static void launch() {
    // rotate the expert base so consecutive timed iterations read cold bytes
    const int off = (rot++ * gNEXP) % (int)(arena_exp - gNEXP);
    dim3 blk(RPNB, 4), grd(gNEXP, (gMI / RPNB + 3) / 4);
    k_bisect<MODE><<<grd, blk>>>(out,
        gp + (size_t)off * code_bytes_per_exp, gs + (size_t)off * scale_bytes_per_exp,
        up + (size_t)off * code_bytes_per_exp, us + (size_t)off * scale_bytes_per_exp,
        xb, gH, gMI, gGRP, gNEXP);
}
static void dispatch() {
    switch (gMODE) { case 0: launch<0>(); break; case 1: launch<1>(); break;
                     case 2: launch<2>(); break; case 3: launch<3>(); break;
                     default: launch<4>(); break; }
}

// The GPU idles at a low clock and a short burst measures that clock, not the sustained one.
// The same 35 MB shape measured 0.194 ms in one process and 0.627 in another -- a 3.2x swing
// with identical code -- purely on whether the preceding variant had run long enough to ramp
// DVFS. Spin for a fixed wall time before ANY measurement, and again between variants.
static void spin_up(double ms_target = 300.0) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    float ms = 0;
    CK(cudaEventRecord(a));
    do {
        for (int i = 0; i < 20; ++i) dispatch();
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        CK(cudaEventElapsedTime(&ms, a, b));
    } while (ms < ms_target);
}

static float bench(int it = 30) {
    spin_up();
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    rot = 0;
    CK(cudaEventRecord(a));
    for (int i = 0; i < it; ++i) dispatch();
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    CK(cudaGetLastError());
    return ms / it;
}

int main(int argc, char** argv) {
    gH = 3072; gMI = 1024; gGRP = 16;
    gNEXP = getenv("NEXP") ? atoi(getenv("NEXP")) : 10;    // top-10 at decode
    const double CEIL = 254.0;

    code_bytes_per_exp  = (size_t)gMI * gH / 2;            // 4 bits per weight
    scale_bytes_per_exp = (size_t)gMI * gH / gGRP;         // 1 E4M3 byte per 16
    // ~1.5 GB of codes so the rotation defeats the 32 MB L2 many times over
    arena_exp = 260;

    CK(cudaMalloc(&gp, code_bytes_per_exp * arena_exp));
    CK(cudaMalloc(&up, code_bytes_per_exp * arena_exp));
    CK(cudaMalloc(&gs, scale_bytes_per_exp * arena_exp));
    CK(cudaMalloc(&us, scale_bytes_per_exp * arena_exp));
    CK(cudaMemset(gp, 0x42, code_bytes_per_exp * arena_exp));
    CK(cudaMemset(up, 0x24, code_bytes_per_exp * arena_exp));
    CK(cudaMemset(gs, 0x3c, scale_bytes_per_exp * arena_exp));
    CK(cudaMemset(us, 0x3c, scale_bytes_per_exp * arena_exp));
    CK(cudaMalloc(&xb, (size_t)gH * 2));
    CK(cudaMemset(xb, 0x3c, (size_t)gH * 2));
    CK(cudaMalloc(&out, (size_t)arena_exp * gMI * 4));

    const double codes  = 2.0 * gNEXP * code_bytes_per_exp;      // gate + up
    const double scales = 2.0 * gNEXP * scale_bytes_per_exp;

    printf("NVFP4 MoE gate/up bisection — %d experts, H=%d MI=%d, arena %.2f GB, ceiling %.0f GB/s\n",
           gNEXP, gH, gMI, (codes + scales) * arena_exp / gNEXP / 1e9, CEIL);
    printf("  codes %.1f MB + scales %.1f MB = %.1f MB per launch\n\n",
           codes / 1e6, scales / 1e6, (codes + scales) / 1e6);
    printf("  %-9s %10s %12s %10s  %s\n", "variant", "ms", "GB/s", "% ceil", "what it removes");

    struct V { const char* n; int mode; double bytes; const char* what; } vs[] = {
        {"FULL",    0, codes + scales, "-- the real kernel"},
        {"NOSCALE", 1, codes,          "the per-16-group E4M3 scale stream"},
        {"NODEQ",   2, codes,          "+ the FP4 hardware decode"},
        {"STREAM",  3, codes,          "+ the FMA chain (pure read bound)"},
        {"VECSCALE",4, codes + scales, "nothing -- same scales, 8-byte loads (THE FIX)"},
    };
    double full_ms = 0;
    for (auto& v : vs) {
        gMODE = v.mode;
        float ms = bench();
        if (v.mode == 0) full_ms = ms;
        printf("  %-9s %10.3f %12.1f %9.0f%%  %s%s\n", v.n, ms, v.bytes / 1e9 / (ms / 1e3),
               v.bytes / 1e9 / (ms / 1e3) / CEIL * 100, v.what,
               v.mode ? "" : "");
    }
    printf("\n  FULL is %.2fx the pure-read time. If that ratio is ~1 the kernel is at the\n"
           "  shape's read ceiling and the gap is not recoverable by removing work.\n", full_ms / bench());
    return 0;
}
