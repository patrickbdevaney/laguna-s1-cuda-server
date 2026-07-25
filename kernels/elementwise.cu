// elementwise.cu — G3 (RMSNorm, partial RoPE) and G4 (sigmoid router with selection-only bias).
//
// Every routine here is a transcription of a specific part of modeling_laguna.py, cited in
// the comment above it. The reference upcasts to fp32 in exactly these places; we match it.
#include "laguna_kernels.cuh"

using namespace lgk;

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
                              const float* __restrict__ inv, int npos, int base,
                              int half, float scale) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= npos * half) return;
    int p = t / half, i = t % half;
    float a = (float)(base + p) * inv[i];
    float c = __cosf(a) * scale, s = __sinf(a) * scale;
    int rot = half * 2;
    cosT[(long)p * rot + i] = c;  cosT[(long)p * rot + i + half] = c;
    sinT[(long)p * rot + i] = s;  sinT[(long)p * rot + i + half] = s;
}
extern "C" void rope_tables(float* cosT, float* sinT, const float* inv,
                            int npos, int base, int half, float scale, cudaStream_t st) {
    int T = 256, n = npos * half;
    k_rope_tables<<<(n + T - 1) / T, T, 0, st>>>(cosT, sinT, inv, npos, base, half, scale);
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
    if (threadIdx.x == 0) {
        float sum = 0.f;
        for (int j = 0; j < topk; ++j) {
            int best = -1; float bv = -1e30f;
            for (int e = 0; e < E; ++e) if (ss[e] > bv) { bv = ss[e]; best = e; }
            sel[(long)r * topk + j] = best;
            wts[(long)r * topk + j] = sc[best];
            sum += sc[best];
            ss[best] = -1e30f;
        }
        if (norm_topk && sum != 0.f)
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
