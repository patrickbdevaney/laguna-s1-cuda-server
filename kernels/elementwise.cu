// elementwise.cu — G3 (RMSNorm, partial RoPE) and G4 (sigmoid router with selection-only bias).
//
// Every routine here is a transcription of a specific part of modeling_laguna.py, cited in
// the comment above it. The reference upcasts to fp32 in exactly these places; we match it.
#include "laguna_kernels.cuh"

using namespace lgk;

// Decode position, held in DEVICE MEMORY so a captured CUDA graph can replay across steps
// without re-capture. Passed as a pointer rather than an `extern __device__` global because
// __device__ globals are per-module without -rdc=true, so each translation unit would get its
// own silent copy. The pointer is stable, so baking it into a graph is safe; only the
// contents change per step.
__global__ void k_set_base(int* p, int v) { *p = v; }
__global__ void k_inc_base(int* p, int n) { *p += n; }
extern "C" void set_base(int* p, int v, cudaStream_t st) { k_set_base<<<1,1,0,st>>>(p, v); }
extern "C" void inc_base(int* p, int n, cudaStream_t st) { k_inc_base<<<1,1,0,st>>>(p, n); }

// ---------------------------------------------------------------------------------------
// G3a — LagunaRMSNorm (modeling_laguna.py:49-63):
//   x32 = x.float(); x32 *= rsqrt(mean(x32^2) + eps); out = w.float() * x32.to(dt).float()
// One block per row; fp32 reduction.
// ---------------------------------------------------------------------------------------
__global__ void k_rmsnorm(float* __restrict__ out, const float* __restrict__ x,
                          const uint16_t* __restrict__ w, int rows, int D, float eps) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const float* xr = x + (long)r * D;
    float* orow = out + (long)r * D;

    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) { float v = xr[i]; ss += v * v; }
    ss = warp_sum(ss);
    __shared__ float red[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < (blockDim.x + 31) / 32) ? red[lane] : 0.f;
        v = warp_sum(v);
        if (lane == 0) red[0] = rsqrtf(v / D + eps);
    }
    __syncthreads();
    float inv = red[0];
    for (int i = threadIdx.x; i < D; i += blockDim.x) orow[i] = bf2f(w[i]) * (xr[i] * inv);
}
extern "C" void rmsnorm(float* out, const float* x, const uint16_t* w,
                        int rows, int D, float eps, cudaStream_t st) {
    k_rmsnorm<<<rows, 256, 0, st>>>(out, x, w, rows, D, eps);
}

// ---------------------------------------------------------------------------------------
// Fused (optional add) + RMSNorm + f32->bf16 cast.
//
// The unfused `k_rmsnorm` at rows=1 is <<<1,256>>>: ONE block on one of twenty SMs, and it is
// latency-bound, not bandwidth-bound -- 12 strided loads per thread behind a runtime loop
// bound the compiler cannot unroll. Making D a template constant unrolls it into 12
// independent loads issued up front. Measured 12.0 -> 4.5 us (2.7-3.1x), 96 sites per step.
//
// BIT-EXACT by construction: thread `t` still accumulates elements t, t+T, t+2T, ... in that
// same order, the warp/cross-warp reduction is untouched, and the cast applies the identical
// `f2bf` to the identical fp32 register value the separate cast kernel would have seen.
// ---------------------------------------------------------------------------------------
template <int D, int T, bool ADD>
__global__ __launch_bounds__(T)
void k_add_rms_cast(uint16_t* __restrict__ ob, float* __restrict__ h,
                    const float* __restrict__ resid, const uint16_t* __restrict__ w,
                    float eps) {
    constexpr int R = D / T;
    const int row = blockIdx.x;
    float* hr = h + (long)row * D;
    uint16_t* obr = ob + (long)row * D;
    const float* rr = ADD ? resid + (long)row * D : nullptr;

    float v[R];
    #pragma unroll
    for (int i = 0; i < R; ++i) {
        int k = threadIdx.x + i * T;
        v[i] = ADD ? (hr[k] + rr[k]) : hr[k];
    }
    if (ADD) {
        #pragma unroll
        for (int i = 0; i < R; ++i) hr[threadIdx.x + i * T] = v[i];   // residual stays fp32
    }
    float ss = 0.f;
    #pragma unroll
    for (int i = 0; i < R; ++i) ss += v[i] * v[i];       // same order as the strided loop

    ss = warp_sum(ss);
    __shared__ float red[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float x = (lane < T / 32) ? red[lane] : 0.f;
        x = warp_sum(x);
        if (lane == 0) red[0] = rsqrtf(x / D + eps);
    }
    __syncthreads();
    const float inv = red[0];
    #pragma unroll
    for (int i = 0; i < R; ++i) {
        int k = threadIdx.x + i * T;
        obr[k] = f2bf(bf2f(w[k]) * (v[i] * inv));
    }
}

// Generic fallback: normalise in place into a scratch then cast. Only used if D != 3072.
extern "C" void f32_to_bf16(uint16_t*, const float*, long, cudaStream_t);
extern "C" void add_inplace(float*, const float*, long, cudaStream_t);
static float* g_fallback = nullptr; static size_t g_fb = 0;
static void rmsnorm_scratch(const float* h, const uint16_t* w, int rows, int D, float eps,
                            uint16_t* ob, cudaStream_t st) {
    size_t need = (size_t)rows * D * 4;
    if (need > g_fb) { cudaFree(g_fallback); cudaMalloc(&g_fallback, need); g_fb = need; }
    k_rmsnorm<<<rows, 256, 0, st>>>(g_fallback, h, w, rows, D, eps);
    f32_to_bf16(ob, g_fallback, (long)rows * D, st);
}

// Falls back to the generic path when D is not the compiled-for hidden size.
extern "C" void add_rms_cast(uint16_t* ob, float* h, const float* resid, const uint16_t* w,
                             int rows, int D, float eps, int do_add, cudaStream_t st) {
    if (D == 3072) {
        if (do_add) k_add_rms_cast<3072, 256, true ><<<rows, 256, 0, st>>>(ob, h, resid, w, eps);
        else        k_add_rms_cast<3072, 256, false><<<rows, 256, 0, st>>>(ob, h, resid, w, eps);
        return;
    }
    if (do_add) add_inplace(h, resid, (long)rows * D, st);
    rmsnorm_scratch(h, w, rows, D, eps, ob, st);
}

// Per-head QK RMSNorm: x is [rows, heads, head_dim], normalise over head_dim.
extern "C" void rmsnorm_heads(float* out, const float* x, const uint16_t* w,
                              int rows, int heads, int hd, float eps, cudaStream_t st) {
    k_rmsnorm<<<rows * heads, 128, 0, st>>>(out, x, w, rows * heads, hd, eps);
}

// ---------------------------------------------------------------------------------------
// G3b — rope tables. inv_freq is supplied by the host (laguna_config.h RopeSpec::inv_freq,
// gated to ~1 ULP against the reference in tests/dump_rope.cpp).
//   freqs = pos * inv_freq ; emb = cat(freqs,freqs) ; cos = cos(emb)*attn_scale
// ---------------------------------------------------------------------------------------
__global__ void k_rope_tables(float* __restrict__ cosT, float* __restrict__ sinT,
                              const float* __restrict__ inv, int npos, const int* __restrict__ dbase,
                              int half, float scale) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= npos * half) return;
    int p = t / half, i = t % half;
    float a = (float)(*dbase + p) * inv[i];
    float c = __cosf(a) * scale, s = __sinf(a) * scale;
    int rot = half * 2;
    cosT[(long)p * rot + i] = c;  cosT[(long)p * rot + i + half] = c;
    sinT[(long)p * rot + i] = s;  sinT[(long)p * rot + i + half] = s;
}
extern "C" void rope_tables(float* cosT, float* sinT, const float* inv,
                            int npos, const int* dbase, int half, float scale, cudaStream_t st) {
    int T = 256, n = npos * half;
    k_rope_tables<<<(n + T - 1) / T, T, 0, st>>>(cosT, sinT, inv, npos, dbase, half, scale);
}

// Partial rotary (modeling_laguna.py:271-306): only the first `rot` dims of each head
// rotate; rotate_half splits WITHIN that slice; the tail passes through untouched.
__global__ void k_rope_apply(float* __restrict__ x, const float* __restrict__ cosT,
                             const float* __restrict__ sinT,
                             int rows, int heads, int hd, int rot) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int half = rot / 2;
    if (t >= rows * heads * half) return;
    int i = t % half;  t /= half;
    int h = t % heads; int r = t / heads;
    float* v = x + ((long)r * heads + h) * hd;
    const float* c = cosT + (long)r * rot;
    const float* s = sinT + (long)r * rot;
    float a = v[i], b = v[i + half];
    v[i]        = a * c[i]        - b * s[i];
    v[i + half] = b * c[i + half] + a * s[i + half];
}
extern "C" void rope_apply(float* x, const float* cosT, const float* sinT,
                           int rows, int heads, int hd, int rot, cudaStream_t st) {
    int T = 256, n = rows * heads * (rot / 2);
    k_rope_apply<<<(n + T - 1) / T, T, 0, st>>>(x, cosT, sinT, rows, heads, hd, rot);
}

// ---------------------------------------------------------------------------------------
// G4 — LagunaTopKRouter (modeling_laguna.py:169-184). The three ways to get this silently
// wrong, all of which degrade quality without crashing:
//   1. using softmax instead of sigmoid
//   2. letting e_score_correction_bias leak into the returned WEIGHTS (it must affect
//      SELECTION only)
//   3. forgetting the sum-normalisation
// One block per token; 256 experts fit one block's worth of work.
// ---------------------------------------------------------------------------------------
__global__ void k_router(int* __restrict__ sel, float* __restrict__ wts,
                         float* __restrict__ scores_out,
                         const float* __restrict__ logits, const float* __restrict__ bias,
                         int rows, int E, int topk, float softcap, int norm_topk) {
    int r = blockIdx.x;
    if (r >= rows) return;
    extern __shared__ float sh[];               // [E] scores, [E] selection scores
    float* sc = sh;
    float* ss = sh + E;

    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        float lg = logits[(long)r * E + e];
        if (softcap > 0.f) lg = tanhf(lg / softcap) * softcap;
        float s = sigmoidf_(lg);
        sc[e] = s;
        ss[e] = s + (bias ? bias[e] : 0.f);
    }
    __syncthreads();
    if (scores_out)
        for (int e = threadIdx.x; e < E; e += blockDim.x) scores_out[(long)r * E + e] = sc[e];

    // top-k by iterative argmax. topk is 10 and E is 256: 2560 comparisons per token,
    // negligible next to the expert GEMMs, and it reproduces torch.topk's tie-breaking
    // (lowest index wins) exactly, which matters because a mis-ordered selection is silent.
    // The argmax scan runs on ONE warp, not one thread. The serial version ("negligible next
    // to the expert GEMMs") measured 27 us per launch x 47 layers = 1.32 ms, 3.2 % of the
    // decode step, with 255 of 256 threads idle -- a reminder that "small" means small
    // *measured*, not small in flops.
    //
    // Tie-breaking must stay torch.topk's: lowest index wins. Each lane scans its strided
    // slice ascending with a strict >, so within a lane the lowest index already wins; the
    // warp reduction then has to break value ties by index explicitly or the winner depends
    // on the shuffle order.
    if (threadIdx.x < 32) {
        const int lane = threadIdx.x;
        float sum = 0.f;
        for (int j = 0; j < topk; ++j) {
            float bv = -1e30f; int best = 0x7fffffff;
            for (int e = lane; e < E; e += 32)
                if (ss[e] > bv) { bv = ss[e]; best = e; }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = __shfl_down_sync(0xffffffffu, bv, off);
                const int   oi = __shfl_down_sync(0xffffffffu, best, off);
                if (ov > bv || (ov == bv && oi < best)) { bv = ov; best = oi; }
            }
            best = __shfl_sync(0xffffffffu, best, 0);
            if (lane == 0) {
                sel[(long)r * topk + j] = best;
                wts[(long)r * topk + j] = sc[best];
                sum += sc[best];
            }
            if (lane == 0) ss[best] = -1e30f;
            __syncwarp();
        }
        if (lane == 0 && norm_topk && sum != 0.f)
            for (int j = 0; j < topk; ++j) wts[(long)r * topk + j] /= sum;
    }
}
extern "C" void router(int* sel, float* wts, float* scores_out, const float* logits,
                       const float* bias, int rows, int E, int topk, float softcap,
                       int norm_topk, cudaStream_t st) {
    k_router<<<rows, 256, 2 * E * sizeof(float), st>>>(sel, wts, scores_out, logits, bias,
                                                       rows, E, topk, softcap, norm_topk);
}

// ---------------------------------------------------------------------------------------
// Attention output gating (modeling_laguna.py:451-461):
//   gate = softplus(g_proj(h).float())        <- h is the POST-LAYERNORM input, not attn out
//   attn = attn.view(...,heads,hd) * gate.unsqueeze(-1)
// ---------------------------------------------------------------------------------------
__global__ void k_gate_softplus(float* __restrict__ attn, const float* __restrict__ gproj,
                                int rows, int heads, int hd) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows * heads * hd) return;
    int h = (t / hd) % heads, r = t / (hd * heads);
    attn[t] *= softplus(gproj[(long)r * heads + h]);
}
extern "C" void gate_softplus(float* attn, const float* gproj, int rows, int heads, int hd,
                              cudaStream_t st) {
    int T = 256, n = rows * heads * hd;
    k_gate_softplus<<<(n + T - 1) / T, T, 0, st>>>(attn, gproj, rows, heads, hd);
}

// SwiGLU: h = silu(gate) * up
__global__ void k_swiglu(float* __restrict__ o, const float* __restrict__ g,
                         const float* __restrict__ u, long n) {
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (t < n) o[t] = silu(g[t]) * u[t];
}
extern "C" void swiglu(float* o, const float* g, const float* u, long n, cudaStream_t st) {
    int T = 256; k_swiglu<<<(int)((n + T - 1) / T), T, 0, st>>>(o, g, u, n);
}

__global__ void k_add(float* __restrict__ a, const float* __restrict__ b, long n) {
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (t < n) a[t] += b[t];
}
extern "C" void add_inplace(float* a, const float* b, long n, cudaStream_t st) {
    int T = 256; k_add<<<(int)((n + T - 1) / T), T, 0, st>>>(a, b, n);
}

__global__ void k_embed(float* __restrict__ o, const uint16_t* __restrict__ emb,
                        const int* __restrict__ ids, int rows, int H) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows * H) return;
    o[t] = bf2f(emb[(long)ids[t / H] * H + (t % H)]);
}
extern "C" void embed_rows(float* o, const uint16_t* emb, const int* ids, int rows, int H,
                           cudaStream_t st) {
    int T = 256, n = rows * H;
    k_embed<<<(n + T - 1) / T, T, 0, st>>>(o, emb, ids, rows, H);
}

// ---------------------------------------------------------------------------------------
// DFlash target taps. The draft's context K/V are projected from the TARGET's residual
// stream after layers `target_layer_ids` = [1,10,19,29,38,47] — six 3072-vectors per
// position, which the draft's `fc [3072,18432]` then fuses. `slot` is which of the six.
//
// Stored as [6][cap][H] rather than [cap][6*H]: each tap gets its own contiguous [C,H]
// block so it can be RMS-normed by `aux_hidden_norms.{slot}` with the plain rmsnorm kernel,
// and the concatenation into [C,18432] happens afterwards, in the cast to bf16.
// ---------------------------------------------------------------------------------------
__global__ void k_tap_store(float* __restrict__ taps, const float* __restrict__ h,
                            int M, int H, int base, int cap, int slot) {
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (t >= (long)M * H) return;
    const int m = (int)(t / H), d = (int)(t % H);
    // A ring, not a flat buffer: the draft's window is 512 on every layer, so only the last
    // `cap` positions can ever be attended and the tap store is O(1) in conversation length.
    const int p = (base + m) % cap;
    taps[((long)slot * cap + p) * H + d] = h[t];
}

extern "C" void tap_store(float* taps, const float* h, int M, int H, int base, int cap,
                          int slot, cudaStream_t st) {
    long n = (long)M * H; int T = 256;
    k_tap_store<<<(int)((n + T - 1) / T), T, 0, st>>>(taps, h, M, H, base, cap, slot);
}

// Norm each tap with its own `aux_hidden_norms.{slot}` and concatenate the six results into
// the [R][ntap*H] bf16 matrix `fc` consumes — one pass, no intermediate fp32 buffer.
//
// The reduction shape and the final expression are copied from k_rmsnorm exactly, so a tap
// normed here and the same vector normed there agree bit for bit.
struct TapNorms { const uint16_t* w[8]; };

__global__ void k_tap_fuse(uint16_t* __restrict__ out, const float* __restrict__ taps,
                           TapNorms wn, int R, int H, int cap, int r0, int ntap, float eps) {
    const int slot = blockIdx.x % ntap;
    const int r    = blockIdx.x / ntap;
    if (r >= R) return;
    const float* xr = taps + ((long)slot * cap + (r0 + r) % cap) * H;
    uint16_t* orow  = out + (long)r * ntap * H + (long)slot * H;
    const uint16_t* w = wn.w[slot];

    float ss = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) { float v = xr[i]; ss += v * v; }
    ss = warp_sum(ss);
    __shared__ float red[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < (blockDim.x + 31) / 32) ? red[lane] : 0.f;
        v = warp_sum(v);
        if (lane == 0) red[0] = rsqrtf(v / H + eps);
    }
    __syncthreads();
    float inv = red[0];
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        orow[i] = f2bf(bf2f(w[i]) * (xr[i] * inv));
}

extern "C" void tap_fuse(uint16_t* out, const float* taps, const uint16_t* const* wn,
                         int R, int H, int cap, int r0, int ntap, float eps, cudaStream_t st) {
    TapNorms t; for (int i = 0; i < 8; ++i) t.w[i] = i < ntap ? wn[i] : nullptr;
    k_tap_fuse<<<R * ntap, 256, 0, st>>>(out, taps, t, R, H, cap, r0, ntap, eps);
}
