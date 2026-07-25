// attention.cu — G3c/G6: head-packed GQA over an FP8 KV cache, for both Laguna layer types.
//
// Laguna needs TWO tilings, not one (ARCH_DELTA.md §1):
//   global  layers: 48 heads, 8 kv-heads -> G=6, full-context KV
//   sliding layers: 72 heads, 8 kv-heads -> G=9, 512-token window
// Both are handled by the same kernel with G and the window as runtime parameters.
//
// Head-pack: one block per (query, kv_head) covering all G sibling query heads, so the KV
// for that head is read ONCE instead of G times. This is the gemma pattern; here it matters
// more, because G is 6 or 9 rather than 2 or 8.
//
// KV is FP8 e4m3 with the STATIC per-layer per-tensor scales shipped in the checkpoint
// (`self_attn.k_scale` / `v_scale`) — not dynamically computed
// (MODEL_INVENTORY.md §4: kv_cache_scheme is `dynamic: false`, `strategy: "tensor"`).
//
// Sliding layers use a ring buffer of exactly `window` slots, which is what makes the KV
// budget work: 36 of 48 layers cost a constant 1.05 MB per sequence no matter how long the
// conversation runs (ROOFLINE.md §6 — a 4.0x KV win over a uniformly-global model).
#include "laguna_kernels.cuh"
#include <cstdio>
#include <cstdlib>

using namespace lgk;

#define MAXG 9          // max GQA group: 72 heads / 8 kv-heads
#define HD   128         // head_dim, asserted against config at startup
#define NW   4           // warps per block (bounded by the cross-warp reduction's shared cost)

// ---------------------------------------------------------------------------------------
// Store K/V for `M` new positions. K/V arrive as fp32 [M][nkv][hd]; cache is [nkv][cap][hd].
// ---------------------------------------------------------------------------------------
__global__ void k_store_kv(uint8_t* __restrict__ Kc, uint8_t* __restrict__ Vc,
                           const float* __restrict__ K, const float* __restrict__ V,
                           float inv_ks, float inv_vs,
                           int M, int nkv, int hd, int cap, const int* __restrict__ dbase) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= M * nkv * hd) return;
    int d = t % hd;
    int kh = (t / hd) % nkv;
    int m = t / (hd * nkv);
    int slot = (*dbase + m) % cap;                     // == base+m when cap >= ctx
    long dst = ((long)kh * cap + slot) * hd + d;
    Kc[dst] = f2e4m3(K[t] * inv_ks);
    Vc[dst] = f2e4m3(V[t] * inv_vs);
}
extern "C" void store_kv(uint8_t* Kc, uint8_t* Vc, const float* K, const float* V,
                         float k_scale, float v_scale, int M, int nkv, int hd, int cap,
                         const int* dbase, cudaStream_t st) {
    int T = 256, n = M * nkv * hd;
    k_store_kv<<<(n + T - 1) / T, T, 0, st>>>(Kc, Vc, K, V, 1.f / k_scale, 1.f / v_scale,
                                              M, nkv, hd, cap, dbase);
}

// ---------------------------------------------------------------------------------------
// Attention. grid = (M, nkv), block = NW*32.
// Each lane owns 4 of the 128 head-dim lanes; each warp strides over key positions.
// Online (flash-style) softmax so a global layer never materialises a [ctx] score row.
// ---------------------------------------------------------------------------------------
__global__ __launch_bounds__(NW * 32)
void k_attn(float* __restrict__ out, const float* __restrict__ Q,
            const uint8_t* __restrict__ Kc, const uint8_t* __restrict__ Vc,
            float ks, float vs, int nkv, int G, int cap, int window, const int* __restrict__ dbase, float qscale) {
    const int m  = blockIdx.x;
    const int kh = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int d0 = lane * 4;                       // this lane's 4 head-dim slots

    __shared__ float qs[MAXG][HD];
    // Cross-warp softmax combine. The accumulator MUST carry the lane (head-dim) axis:
    // each lane owns 4 distinct dims, so a [NW][G][4] buffer would have all 32 lanes of a
    // warp writing the same 4 slots — a race that silently keeps one lane's partial and
    // discards the other 31. That cost one failed gate.
    __shared__ float red_acc[NW][MAXG][HD];        // per-warp, per-head, per-dim partial
    __shared__ float red_ml[NW][MAXG][2];          // per-warp {running max, running sum}

    // stage this block's G query heads
    for (int i = threadIdx.x; i < G * HD; i += NW * 32) {
        int g = i / HD, d = i % HD;
        qs[g][d] = Q[((long)m * (nkv * G) + kh * G + g) * HD + d];
    }
    __syncthreads();

    const int p = *dbase + m;                      // absolute position of this query
    const int jlo = (window > 0) ? max(0, p - window + 1) : 0;
    const int jhi = p;

    float mx[MAXG], ls[MAXG], acc[MAXG][4];
    #pragma unroll
    for (int g = 0; g < MAXG; ++g) {
        mx[g] = -1e30f; ls[g] = 0.f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[g][i] = 0.f;
    }

    for (int j = jlo + warp; j <= jhi; j += NW) {
        int slot = j % cap;
        const uint8_t* kp = Kc + ((long)kh * cap + slot) * HD + d0;
        const uint8_t* vp = Vc + ((long)kh * cap + slot) * HD + d0;
        uint32_t kraw = *(const uint32_t*)kp;
        uint32_t vraw = *(const uint32_t*)vp;
        const uint8_t* kb = (const uint8_t*)&kraw;
        const uint8_t* vb = (const uint8_t*)&vraw;
        float kv[4], vv[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) { kv[i] = e4m3f(kb[i]) * ks; vv[i] = e4m3f(vb[i]) * vs; }

        for (int g = 0; g < G; ++g) {
            float part = 0.f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) part = fmaf(qs[g][d0 + i], kv[i], part);
            float s = warp_sum(part) * qscale;
            float mnew = fmaxf(mx[g], s);
            float corr = __expf(mx[g] - mnew);
            float pj = __expf(s - mnew);
            ls[g] = ls[g] * corr + pj;
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[g][i] = acc[g][i] * corr + pj * vv[i];
            mx[g] = mnew;
        }
    }

    // combine the NW partial softmaxes
    #pragma unroll
    for (int g = 0; g < G; ++g) {
        if (lane == 0) { red_ml[warp][g][0] = mx[g]; red_ml[warp][g][1] = ls[g]; }
        #pragma unroll
        for (int i = 0; i < 4; ++i) red_acc[warp][g][d0 + i] = acc[g][i];
    }
    __syncthreads();

    if (warp == 0) {
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            float M0 = -1e30f;
            #pragma unroll
            for (int w = 0; w < NW; ++w) M0 = fmaxf(M0, red_ml[w][g][0]);
            float L = 0.f, a[4] = {0, 0, 0, 0};
            for (int w = 0; w < NW; ++w) {
                float c = __expf(red_ml[w][g][0] - M0);
                L += red_ml[w][g][1] * c;
                #pragma unroll
                for (int i = 0; i < 4; ++i) a[i] += red_acc[w][g][d0 + i] * c;
            }
            float inv = (L > 0.f) ? 1.f / L : 0.f;
            float* o = out + ((long)m * (nkv * G) + kh * G + g) * HD + d0;
            #pragma unroll
            for (int i = 0; i < 4; ++i) o[i] = a[i] * inv;
        }
    }
}

// ---------------------------------------------------------------------------------------
// SPLIT-KEY variant. The one-block-per-(query,kv_head) form leaves the GPU idle at decode:
// M=1 x nkv=8 is EIGHT blocks on 20 SMs, and it measured 1-2 % of streaming roofline
// (0.58 ms to read 1 MB). Splitting the key range over `NSP` blocks and combining restores
// occupancy. This is the standard flash-decoding split; it is a pure scheduling change, so
// the numerics are unchanged apart from a different (still deterministic) combine order.
// ---------------------------------------------------------------------------------------
// G is a TEMPLATE parameter, not a runtime one. With a runtime G the compiler cannot unroll
// the g-loops, so `acc[G][4] + mx[G] + ls[G]` is dynamically indexed and ptxas puts the whole
// accumulator in LOCAL memory -- measured 224 bytes of stack frame, and the kernel ran at
// 9-26 GB/s against 141-172 achievable for the identical access pattern. Same failure mode as
// the e2m1f LUT (WON #1) and the MoE token loop (WON #6). Verify with -Xptxas -v that the
// stack frame is 0 bytes; if it is not, an index is still dynamic somewhere.
//
// Q also moves from shared memory to registers: each lane owns 4 consecutive head-dim slots,
// so one float4 load per head replaces the staging loop, its __syncthreads, and 4608 B of
// shared memory.
template <int G>
__global__ __launch_bounds__(NW * 32)
void k_attn_split(float* __restrict__ pacc, float* __restrict__ pml,
                  const float* __restrict__ Q,
                  const uint8_t* __restrict__ Kc, const uint8_t* __restrict__ Vc,
                  float ks, float vs, int nkv, int cap, int window,
                  float qscale, int NSP, const int* __restrict__ dbase) {
    const int m  = blockIdx.x;
    const int kh = blockIdx.y;
    const int sp = blockIdx.z;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int d0 = lane * 4;

    __shared__ float red_acc[NW][G][HD];
    __shared__ float red_ml[NW][G][2];

    float qr[G][4];
    #pragma unroll
    for (int g = 0; g < G; ++g) {
        const float4 t = *(const float4*)(Q + ((long)m * (nkv * G) + kh * G + g) * HD + d0);
        qr[g][0] = t.x; qr[g][1] = t.y; qr[g][2] = t.z; qr[g][3] = t.w;
    }

    const int p = *dbase + m;
    const int lo = (window > 0) ? max(0, p - window + 1) : 0;
    const int len = p - lo + 1;
    const int chunk = (len + NSP - 1) / NSP;
    const int jlo = lo + sp * chunk;
    const int jhi = min(p, jlo + chunk - 1);

    float mx[G], ls[G], acc[G][4];
    #pragma unroll
    for (int g = 0; g < G; ++g) {
        mx[g] = -1e30f; ls[g] = 0.f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[g][i] = 0.f;
    }

    for (int j = jlo + warp; j <= jhi; j += NW) {
        int slot = j % cap;
        uint32_t kraw = *(const uint32_t*)(Kc + ((long)kh * cap + slot) * HD + d0);
        uint32_t vraw = *(const uint32_t*)(Vc + ((long)kh * cap + slot) * HD + d0);
        const uint8_t* kb = (const uint8_t*)&kraw;
        const uint8_t* vb = (const uint8_t*)&vraw;
        float kv[4], vv[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) { kv[i] = e4m3f(kb[i]) * ks; vv[i] = e4m3f(vb[i]) * vs; }
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            float part = 0.f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) part = fmaf(qr[g][i], kv[i], part);
            float s = warp_sum(part) * qscale;
            float mnew = fmaxf(mx[g], s);
            float corr = __expf(mx[g] - mnew);
            float pj = __expf(s - mnew);
            ls[g] = ls[g] * corr + pj;
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[g][i] = acc[g][i] * corr + pj * vv[i];
            mx[g] = mnew;
        }
    }
    #pragma unroll
    for (int g = 0; g < G; ++g) {
        if (lane == 0) { red_ml[warp][g][0] = mx[g]; red_ml[warp][g][1] = ls[g]; }
        #pragma unroll
        for (int i = 0; i < 4; ++i) red_acc[warp][g][d0 + i] = acc[g][i];
    }
    __syncthreads();

    if (warp == 0) {
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            float M0 = -1e30f;
            #pragma unroll
            for (int w = 0; w < NW; ++w) M0 = fmaxf(M0, red_ml[w][g][0]);
            float L = 0.f, a[4] = {0, 0, 0, 0};
            for (int w = 0; w < NW; ++w) {
                float c = __expf(red_ml[w][g][0] - M0);
                L += red_ml[w][g][1] * c;
                #pragma unroll
                for (int i = 0; i < 4; ++i) a[i] += red_acc[w][g][d0 + i] * c;
            }
            long hidx = ((long)m * (nkv * G) + kh * G + g) * NSP + sp;
            #pragma unroll
            for (int i = 0; i < 4; ++i) pacc[hidx * HD + d0 + i] = a[i];
            if (lane == 0) { pml[hidx * 2] = (jhi >= jlo) ? M0 : -1e30f; pml[hidx * 2 + 1] = L; }
        }
    }
}

__global__ void k_attn_combine(float* __restrict__ out, const float* __restrict__ pacc,
                               const float* __restrict__ pml, int nheads, int NSP) {
    int hh = blockIdx.x;                       // flattened (m, kv_head, g)
    if (hh >= nheads) return;
    int d = threadIdx.x;
    __shared__ float sM, sL;
    if (d == 0) {
        float M0 = -1e30f;
        for (int s = 0; s < NSP; ++s) M0 = fmaxf(M0, pml[((long)hh * NSP + s) * 2]);
        float L = 0.f;
        for (int s = 0; s < NSP; ++s)
            L += pml[((long)hh * NSP + s) * 2 + 1] * __expf(pml[((long)hh * NSP + s) * 2] - M0);
        sM = M0; sL = L;
    }
    __syncthreads();
    float a = 0.f;
    for (int s = 0; s < NSP; ++s)
        a += pacc[((long)hh * NSP + s) * HD + d] * __expf(pml[((long)hh * NSP + s) * 2] - sM);
    out[(long)hh * HD + d] = (sL > 0.f) ? a / sL : 0.f;
}

extern "C" void attend(float* out, const float* Q, const uint8_t* Kc, const uint8_t* Vc,
                       float k_scale, float v_scale, int M, int nkv, int G, int cap,
                       int window, const int* dbase, float qscale, cudaStream_t st) {
    dim3 g(M, nkv);
    k_attn<<<g, NW * 32, 0, st>>>(out, Q, Kc, Vc, k_scale, v_scale, nkv, G, cap, window,
                                  dbase, qscale);
}

// Choose the split so the grid lands at ~4 blocks/SM (20 SMs), capped by the key range.
//
// CORRECTNESS: this must NOT depend on M. The split count selects the partition of the key
// axis, and the flash-decoding combine is a chain of fp32 online-softmax rescales, so a
// different partition is a different rounding of the SAME token. The earlier form used
// ceil(80/(M*nkv)) and therefore gave 10 splits at decode M=1 but 3 at a batched M=4 — the
// two paths disagreed by ~1 ulp at layer 0 (7.4e-09), the residual stream amplified it about
// 1.4x per layer, and by layer 48 the argmax had flipped (LOOP_LOG B1c). Sizing the split
// from `len` alone makes decode, batched prefill and speculative verify bit-identical, which
// is what makes a DFlash acceptance rate mean anything: the verify pass at M=k+1 must
// reproduce the decode pass at M=1 exactly, or it is not verifying the same model.
//
// The target is the decode shape: ~80 blocks = 4 waves of 20 SMs at M=1. Larger M simply
// gets a larger grid, which costs a bigger partial buffer and a wider combine but no extra
// K/V traffic — the key range is partitioned, not replicated.
extern "C" int attend_nsplit(int M, int nkv, int len) {
    (void)M;
    int want = (80 + nkv - 1) / nkv;
    int by_len = (len + 63) / 64;               // >=64 keys per block, else it is all overhead
    int n = want < by_len ? want : by_len;
    if (n < 1) n = 1;
    if (n > 32) n = 32;
    return n;
}

extern "C" void attend_split(float* out, float* pacc, float* pml, const float* Q,
                             const uint8_t* Kc, const uint8_t* Vc, float k_scale, float v_scale,
                             int M, int nkv, int G, int cap, int window, const int* dbase,
                             float qscale, int NSP, cudaStream_t st) {
    if (NSP <= 1) {
        attend(out, Q, Kc, Vc, k_scale, v_scale, M, nkv, G, cap, window, dbase, qscale, st);
        return;
    }
    dim3 g(M, nkv, NSP);
    // Only two GQA groups exist in this model: 6 (global, 48 heads) and 9 (sliding, 72).
    if (G == 6)
        k_attn_split<6><<<g, NW * 32, 0, st>>>(pacc, pml, Q, Kc, Vc, k_scale, v_scale, nkv, cap,
                                               window, qscale, NSP, dbase);
    else if (G == 9)
        k_attn_split<9><<<g, NW * 32, 0, st>>>(pacc, pml, Q, Kc, Vc, k_scale, v_scale, nkv, cap,
                                               window, qscale, NSP, dbase);
    else
        { fprintf(stderr, "attend_split: unsupported GQA group %d\n", G); abort(); }
    k_attn_combine<<<M * nkv * G, HD, 0, st>>>(out, pacc, pml, M * nkv * G, NSP);
}
