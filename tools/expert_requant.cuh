// expert_requant.cuh -- offline requantizer for the ROUTED EXPERTS, in place, on device.
//
// WHAT THIS IS FOR
// ----------------
// Only `model.layers.{1..47}.mlp.experts.{0..255}.{gate,up,down}_proj` are NVFP4 in this
// checkpoint: E2M1 codes (4 bit) + one E4M3 scale per 16 contiguous K + an fp32 per-tensor
// RECIPROCAL global scale. That is 4 + 8/16 = 4.5 bits/weight and 2.495 GB of the 6.251 GB
// per-token byte budget. The question this file exists to answer is what capability is lost
// by taking those experts to ~3.0 and ~2.75 bits/weight.
//
// THE CONTAINER TRICK (option (a) of the brief)
// --------------------------------------------
// We do NOT change the file format or the kernels. We requantize the expert weights to a
// SMALLER alphabet and a COARSER scale group, then write the result back into the shipped
// NVFP4 container:
//
//   * a b-bit alphabet is expressed by using only 2^b of the 16 E2M1 codes;
//   * a scale group of G > 16 is expressed by writing the SAME E4M3 byte into all G/16
//     container slots it covers.
//
// Every value we produce is exactly representable as (E2M1 code) x (E4M3 group scale) x
// (global scale), so the existing `k_moe_gateup_rp` / `k_moe_down_rp` kernels reconstruct it
// bit-exactly and run at exactly the same speed. That is the point: the measured delta is a
// pure capability delta, with wall-clock held fixed, and the byte saving of the real codec is
// then a separate, already-measured hardware number (+15.2 % decode).
//
// The global scale cancels: reconstruction is code * e4m3 * ginv with ginv constant over a
// whole projection, so we requantize the product (code * e4m3) and never touch ginv.
//
// ARENA LAYOUT
// ------------
// The loader repacks each expert projection to [n_block][k_chunk][lane][16B] for the codes and
// [n_block][group/8][lane][8] for the scales (see laguna_weights.h). This file addresses the
// REPACKED form directly, matching kernels/moe.cu:k_moe_gateup_rp exactly:
//
//   code byte  = packed + ((( (e*NB + nb) * C + c ) * 32 + lane) * 16) + j
//   scale byte = scale  + ( (e*NB + nb) * NG ) * 32 + ((g/8) * 32 + lane) * 8 + (g%8)
//
// with NB = rows/32, C = K/32, NG = K/16, lane = the output row within its 32-row block.
// Weight k of output row n lives in chunk c = k/32, byte j = (k%32)/2, nibble k%2, and its
// container scale group is g = k/16.
//
// EFFECTIVE BITS PER WEIGHT
// -------------------------
// bpw = payload_bits + 8 / group. `payload_bits` is the EXACT cost of a dense packing of the
// alphabet, not ceil(log2 L):
//
//   16 levels -> 4.000  (1 per 4 bits)                     shipped NVFP4, group 16 -> 4.500
//    8 levels -> 3.000  (1 per 3 bits)
//    6 levels -> 2.667  (3 per 8 bits,  6^3 = 216 <= 256)
//    5 levels -> 2.500  (2 per 5 bits,  5^2 =  25 <=  32)
//    3 levels -> 1.600  (5 per 8 bits,  3^5 = 243 <= 256)
//
// SATURATION NOTE. The checkpoint's global scale is 2688/amax, so max(code * e4m3) reaches
// 2688 = 6 * 448 on the largest group, and 448 is E4M3's maximum finite value. A level set
// whose top magnitude is 6 can therefore always place that group's amax on the top level
// without saturating the E4M3 scale; a top magnitude of 4 would need a scale of 672 and would
// clip. We count clipped groups and report them.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "../kernels/laguna_kernels.cuh"
#include "../include/laguna_weights.h"

namespace lgrq {

// The 8 E2M1 magnitudes, indexed by the low 3 bits of the code. Bit 3 is the sign.
__device__ __host__ __forceinline__ float e2m1_mag(int i) {
    const float t[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    return t[i & 7];
}

// A requantization scheme. `mag_code[0..nmag-1]` are E2M1 code indices (1..7) giving the
// nonzero magnitudes of the alphabet, ascending; zero is always present. So the level count is
// 2*nmag + 1.
struct Scheme {
    int   nmag = 7;                  // 7 -> all of {0.5,1,1.5,2,3,4,6}: the shipped alphabet
    int   mag_code[7] = {1,2,3,4,5,6,7};
    int   group = 16;                // NEW scale group, a multiple of 16 (16/32/64/128)
    float payload_bits = 4.0f;       // exact dense-packing cost of the alphabet
    char  name[32] = "nvfp4";

    float bpw() const { return payload_bits + 8.0f / (float)group; }
    int   nlev() const { return 2 * nmag + 1; }
};

// Candidate scale multipliers, applied to amax/top_magnitude. Pure round-to-nearest at
// clip = 1.0 wastes resolution when a group has one outlier, which is exactly the regime a
// 5-level alphabet is in; the search is 7 extra passes over data that is already in L1.
#define LGRQ_NCAND 7
__device__ __constant__ float kClip[LGRQ_NCAND] = {1.00f, 0.95f, 0.90f, 0.85f, 0.80f, 0.72f, 0.65f};

struct DevScheme {
    int   nmag;
    float mag[7];                    // magnitudes themselves, ascending
    uint8_t code[7];                 // the E2M1 code index for each
    int   group;
};

// Nearest level of {0, +-mag[0..nmag)} to |v|/S; returns the packed 4-bit code (sign in bit 3).
__device__ __forceinline__ uint8_t quant_code(float v, float S, const DevScheme& sc, float* rec) {
    const float a = fabsf(v);
    const float t = (S > 0.f) ? a / S : 0.f;
    int best = -1;                                   // -1 == the zero level
    float bd = t * t;                                // (t - 0)^2
    #pragma unroll
    for (int i = 0; i < 7; ++i) {
        if (i >= sc.nmag) break;
        const float d = t - sc.mag[i];
        if (d * d < bd) { bd = d * d; best = i; }
    }
    const bool neg = (v < 0.f);
    if (best < 0) { *rec = 0.f; return 0; }
    *rec = (neg ? -sc.mag[best] : sc.mag[best]) * S;
    return (uint8_t)(sc.code[best] | (neg ? 8 : 0));
}

// ---------------------------------------------------------------------------------------
// One thread owns one output row of one expert projection and walks its whole K axis.
// `apply` false = measure only (nothing is written), true = requantize in place.
// ---------------------------------------------------------------------------------------
__global__ void k_requant(uint8_t* __restrict__ packed, uint8_t* __restrict__ scale,
                          int nexp, int rows, int K, DevScheme sc, int apply,
                          double* __restrict__ acc /* [0]=sse [1]=ref [2]=clipped */) {
    const int NB = rows / 32;
    const int C  = K / 32;
    const int NG = K / 16;
    const int lane = threadIdx.x;
    const int nb   = blockIdx.x * blockDim.y + threadIdx.y;
    const int e    = blockIdx.y;
    if (nb >= NB || e >= nexp) return;

    const size_t eb   = (size_t)e * NB + nb;
    uint8_t* pbase    = packed + (eb * C) * 32 * 16 + (size_t)lane * 16;
    uint8_t* sbase    = scale  + (eb * NG) * 32 + (size_t)lane * 8;
    // code byte of weight k: pbase + (k/32)*32*16 + ((k%32)/2)
    // scale byte of group g: sbase + (g/8)*32*8 + (g%8)
    auto pptr = [&](int k) { return pbase + (size_t)(k >> 5) * 512 + (size_t)((k & 31) >> 1); };
    auto sptr = [&](int g) { return sbase + (size_t)(g >> 3) * 256 + (size_t)(g & 7); };

    const float topmag = sc.mag[sc.nmag - 1];
    const int SG = sc.group;
    double sse = 0.0, ref = 0.0; int clipped = 0;

    for (int k0 = 0; k0 < K; k0 += SG) {
        // ---- pass A: amax over the super-group, in (code * e4m3) units
        float amax = 0.f;
        for (int k = k0; k < k0 + SG; ++k) {
            const uint8_t b = *pptr(k);
            const uint8_t code = (k & 1) ? (uint8_t)(b >> 4) : (uint8_t)(b & 0x0F);
            const float v = e2m1_mag(code) * lgk::e4m3f(*sptr(k >> 4));
            amax = fmaxf(amax, v);                    // magnitudes only; sign is separate
        }
        if (amax <= 0.f) {                            // an all-zero group: nothing to choose
            if (apply) {
                for (int g = k0 >> 4; g < (k0 + SG) >> 4; ++g) *sptr(g) = 0;
                for (int k = k0; k < k0 + SG; k += 2) *pptr(k) = 0;
            }
            continue;
        }

        // ---- pass B: pick the E4M3 scale that minimises SSE over the super-group
        float bestS = 0.f; double bestE = 1e300;
        for (int ci = 0; ci < LGRQ_NCAND; ++ci) {
            float S = lgk::e4m3f(lgk::f2e4m3(amax * kClip[ci] / topmag));
            if (S <= 0.f) continue;
            double E = 0.0;
            for (int k = k0; k < k0 + SG; ++k) {
                const uint8_t b = *pptr(k);
                const uint8_t code = (k & 1) ? (uint8_t)(b >> 4) : (uint8_t)(b & 0x0F);
                float v = e2m1_mag(code) * lgk::e4m3f(*sptr(k >> 4));
                if (code & 8) v = -v;
                float r; quant_code(v, S, sc, &r);
                const double d = (double)v - (double)r;
                E += d * d;
            }
            if (E < bestE) { bestE = E; bestS = S; }
        }
        if (bestS <= 0.f) bestS = lgk::e4m3f(lgk::f2e4m3(amax / topmag));
        if (amax / topmag > 448.f) ++clipped;

        // ---- pass C: quantize, accumulate error, and (optionally) write back
        const uint8_t sbyte = lgk::f2e4m3(bestS);
        const float   S     = lgk::e4m3f(sbyte);
        for (int k = k0; k < k0 + SG; k += 2) {
            const uint8_t b = *pptr(k);
            float v0 = e2m1_mag(b & 0x0F) * lgk::e4m3f(*sptr(k >> 4));
            if (b & 8) v0 = -v0;
            float v1 = e2m1_mag((b >> 4) & 0x0F) * lgk::e4m3f(*sptr((k + 1) >> 4));
            if (b & 0x80) v1 = -v1;
            float r0, r1;
            const uint8_t c0 = quant_code(v0, S, sc, &r0);
            const uint8_t c1 = quant_code(v1, S, sc, &r1);
            const double d0 = (double)v0 - (double)r0, d1 = (double)v1 - (double)r1;
            sse += d0 * d0 + d1 * d1;
            ref += (double)v0 * v0 + (double)v1 * v1;
            if (apply) *pptr(k) = (uint8_t)(c0 | (c1 << 4));
        }
        if (apply)
            for (int g = k0 >> 4; g < ((k0 + SG) >> 4); ++g) *sptr(g) = sbyte;
    }

    // block reduction then one atomic per block, not per thread
    __shared__ double sh[3][256];
    const int tid = threadIdx.y * 32 + lane;
    sh[0][tid] = sse; sh[1][tid] = ref; sh[2][tid] = (double)clipped;
    __syncthreads();
    const int nt = blockDim.x * blockDim.y;
    for (int s = nt >> 1; s > 0; s >>= 1) {
        if (tid < s) { sh[0][tid] += sh[0][tid + s]; sh[1][tid] += sh[1][tid + s];
                       sh[2][tid] += sh[2][tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { atomicAdd(acc, sh[0][0]); atomicAdd(acc + 1, sh[1][0]);
                    atomicAdd(acc + 2, sh[2][0]); }
}

inline DevScheme to_dev(const Scheme& s) {
    DevScheme d; d.nmag = s.nmag; d.group = s.group;
    for (int i = 0; i < 7; ++i) {
        d.code[i] = (uint8_t)(i < s.nmag ? s.mag_code[i] : 0);
        d.mag[i]  = i < s.nmag ? e2m1_mag(s.mag_code[i]) : 0.f;
    }
    return d;
}

struct Result { double rel_mse = 0, sse = 0, ref = 0, clipped = 0; double seconds = 0; };

// Requantize (or, with apply=false, just measure) every routed expert in the loaded arena.
// `layer_stride`: pass >1 to sample every Nth layer when only a measurement is wanted.
inline Result run(laguna::Weights& W, const Scheme& s, bool apply, int layer_stride = 1) {
    using namespace laguna;
    const Config& c = W.cfg;
    if (s.group % 16 || s.group > 1024) { fprintf(stderr, "requant: bad group %d\n", s.group); abort(); }
    DevScheme d = to_dev(s);
    double* acc = nullptr;
    CUDA_CHECK(cudaMalloc(&acc, 3 * sizeof(double)));
    CUDA_CHECK(cudaMemset(acc, 0, 3 * sizeof(double)));
    const double t0 = wall_now();
    const int E = c.n_experts, H = c.hidden, MI = c.moe_intermediate;
    for (int L = 0; L < c.n_layers; L += 1) {
        if (c.is_dense(L)) continue;                 // layer 0 is a dense MLP, not routed
        if (((L - 1) % layer_stride) != 0) continue;
        LayerW& w = W.L[L];
        struct T { const uint8_t* p; const uint8_t* q; int rows; int K; } ts[3] = {
            {w.e_gate_p, w.e_gate_s, MI, H},
            {w.e_up_p,   w.e_up_s,   MI, H},
            {w.e_down_p, w.e_down_s, H,  MI}};
        for (auto& t : ts) {
            const int NB = t.rows / 32;
            const int wy = (NB % 8 == 0) ? 8 : ((NB % 4 == 0) ? 4 : 1);
            dim3 blk(32, wy), grd((NB + wy - 1) / wy, E);
            k_requant<<<grd, blk>>>((uint8_t*)t.p, (uint8_t*)t.q, E, t.rows, t.K, d,
                                    apply ? 1 : 0, acc);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double h[3]; CUDA_CHECK(cudaMemcpy(h, acc, 3 * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(acc);
    Result r; r.sse = h[0]; r.ref = h[1]; r.clipped = h[2];
    r.rel_mse = h[1] > 0 ? h[0] / h[1] : 0.0;
    r.seconds = wall_now() - t0;
    return r;
}

// ---------------------------------------------------------------------------------------
// The configurations this project evaluates. Everything here is exactly representable in the
// shipped NVFP4 container, so the CUDA kernels are untouched and the speed is identical.
// ---------------------------------------------------------------------------------------
inline Scheme scheme_by_name(const std::string& n) {
    Scheme s;
    auto set = [&](const char* nm, int nmag, std::initializer_list<int> codes, int grp, float pb) {
        snprintf(s.name, sizeof s.name, "%s", nm);
        s.nmag = nmag; s.group = grp; s.payload_bits = pb;
        int i = 0; for (int c : codes) s.mag_code[i++] = c;
    };
    // baseline: the shipped alphabet, untouched. 4 + 8/16 = 4.500 bpw
    if (n == "baseline")  set("baseline", 7, {1,2,3,4,5,6,7}, 16, 4.0f);
    // 5 levels {0,+-m1,+-6}, group 16. 2.5 + 8/16 = 3.000 bpw
    else if (n == "L5g16_15") set("L5g16_15", 2, {3,7}, 16, 2.5f);   // {1.5, 6}   ratio 4
    else if (n == "L5g16_2")  set("L5g16_2",  2, {4,7}, 16, 2.5f);   // {2, 6}     ratio 3
    else if (n == "L5g16_3")  set("L5g16_3",  2, {5,7}, 16, 2.5f);   // {3, 6}     ratio 2
    else if (n == "L5g16_1")  set("L5g16_1",  2, {2,7}, 16, 2.5f);   // {1, 6}     ratio 6
    // the same 5-level alphabet at group 32. 2.5 + 8/32 = 2.750 bpw
    else if (n == "L5g32_15") set("L5g32_15", 2, {3,7}, 32, 2.5f);
    else if (n == "L5g32_2")  set("L5g32_2",  2, {4,7}, 32, 2.5f);
    else if (n == "L5g32_3")  set("L5g32_3",  2, {5,7}, 32, 2.5f);
    // 8-level (3 bit) alternatives on the same bpw ladder
    else if (n == "L7g16")  set("L7g16",  3, {2,4,7}, 16, 3.0f);     // 3.500 bpw
    else if (n == "L7g32")  set("L7g32",  3, {2,4,7}, 32, 3.0f);     // 3.250 bpw
    else if (n == "L7g64")  set("L7g64",  3, {2,4,7}, 64, 3.0f);     // 3.125 bpw
    else if (n == "L7g128") set("L7g128", 3, {2,4,7}, 128, 3.0f);    // 3.063 bpw
    else if (n == "L7g128b")set("L7g128b",3, {3,5,7}, 128, 3.0f);    // 3.063 bpw, {1.5,3,6}
    // 3-level ternary
    else if (n == "L3g16")  set("L3g16",  1, {7}, 16, 1.6f);         // 2.100 bpw
    else if (n == "L3g32")  set("L3g32",  1, {7}, 32, 1.6f);         // 1.850 bpw
    // 9-level (0 + 4 magnitudes), 5 codes per 16 bits: 9^5 = 59049 <= 65536 -> 3.2 bits
    else if (n == "L9g32")  set("L9g32",  4, {2,3,4,7}, 32, 3.2f);   // 3.450 bpw
    else if (n == "L9g64")  set("L9g64",  4, {2,3,4,7}, 64, 3.2f);   // 3.325 bpw
    else { fprintf(stderr, "requant: unknown scheme '%s'\n", n.c_str()); abort(); }
    return s;
}

} // namespace lgrq
