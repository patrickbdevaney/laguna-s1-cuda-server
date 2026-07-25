// moe.cu — G5: grouped, weight-resident routed-expert MLP.
//
// Structure ported from gemma's champion MoE path (invert -> grouped gate/up -> weight-
// resident down -> finalize), retiled for Laguna: 256 experts (not 128), top-10 (not top-8),
// moe_intermediate 1024 (not 704), 47 MoE layers (layer 0 is dense).
//
// Two deliberate choices carried over from the gemma ledger:
//
//  * NO ATOMICS. Each (token, slot) assignment writes its own partial and a finalize pass
//    sums them in a FIXED order. Atomic float accumulation would make the output depend on
//    scheduling, which destroys run-to-run determinism and, under speculation, makes a
//    verify step disagree with itself.
//
//  * The finalize sums in ASCENDING EXPERT-INDEX order, because that is the order the
//    reference accumulates in (modeling_laguna.py:212-222 iterates `expert_hit`, which
//    torch.nonzero returns sorted). Summing in top-k slot order instead is a ~1e-7
//    difference — small, but free to avoid, and it keeps the gate honest.
//
// No tensor-core path here: at k*≈5 the union is ~46 experts per layer with ~1.3 tokens
// each, which is exactly the ~1-token-per-expert regime where gemma measured TC grouped MoE
// LOSING to tile padding. See OPTIMIZATION_LOG "Dead on arrival".
#include "laguna_kernels.cuh"

using namespace lgk;

// ---------------------------------------------------------------------------------------
// Inversion: assignments (token,slot) -> per-expert lists.
//   sel[rows][topk] -> ecount[E], eoff[E+1], elist[rows*topk] (assignment ids, expert-major)
// ---------------------------------------------------------------------------------------
__global__ void k_moe_count(int* __restrict__ ecount, const int* __restrict__ sel, int nass) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a < nass) atomicAdd(&ecount[sel[a]], 1);      // integer atomics: order-independent
}

__global__ void k_moe_scan(int* __restrict__ eoff, const int* __restrict__ ecount, int E) {
    // single block, E<=1024
    extern __shared__ int sh[];
    int t = threadIdx.x;
    sh[t] = (t < E) ? ecount[t] : 0;
    __syncthreads();
    for (int off = 1; off < blockDim.x; off <<= 1) {
        int v = (t >= off) ? sh[t - off] : 0;
        __syncthreads();
        sh[t] += v;
        __syncthreads();
    }
    if (t < E) eoff[t + 1] = sh[t];
    if (t == 0) eoff[0] = 0;
}

__global__ void k_moe_fill(int* __restrict__ elist, int* __restrict__ cursor,
                           const int* __restrict__ sel, const int* __restrict__ eoff, int nass) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= nass) return;
    int e = sel[a];
    int pos = atomicAdd(&cursor[e], 1);
    elist[eoff[e] + pos] = a;
}

// Compact the list of experts that actually have work, so the GEMM grid is over ACTIVE
// experts (≈46 at k*=5) rather than all 256.
__global__ void k_moe_active(int* __restrict__ active, int* __restrict__ nactive,
                             const int* __restrict__ ecount, int E) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < E && ecount[e] > 0) active[atomicAdd(nactive, 1)] = e;
}

extern "C" void moe_invert(int* ecount, int* eoff, int* elist, int* cursor,
                           int* active, int* nactive, const int* sel,
                           int rows, int topk, int E, cudaStream_t st) {
    int nass = rows * topk, T = 256;
    cudaMemsetAsync(ecount, 0, E * sizeof(int), st);
    cudaMemsetAsync(cursor, 0, E * sizeof(int), st);
    cudaMemsetAsync(nactive, 0, sizeof(int), st);
    k_moe_count<<<(nass + T - 1) / T, T, 0, st>>>(ecount, sel, nass);
    int P = 1; while (P < E) P <<= 1;
    k_moe_scan<<<1, P, P * sizeof(int), st>>>(eoff, ecount, E);
    k_moe_fill<<<(nass + T - 1) / T, T, 0, st>>>(elist, cursor, sel, eoff, nass);
    k_moe_active<<<(E + T - 1) / T, T, 0, st>>>(active, nactive, ecount, E);
}

// ---------------------------------------------------------------------------------------
// gate/up: one warp owns output column n of ONE expert and reuses that weight across every
// token routed to it (weight-resident). hbuf[assignment][MI] = silu(gate) * up.
// ---------------------------------------------------------------------------------------
#define MAXTOK 16           // tokens per expert handled with the weight held in registers

__global__ void k_moe_gateup(float* __restrict__ hbuf,
                             const uint8_t* __restrict__ gp, const uint8_t* __restrict__ gs,
                             const float* __restrict__ ginv,
                             const uint8_t* __restrict__ up, const uint8_t* __restrict__ us,
                             const float* __restrict__ uinv,
                             const uint16_t* __restrict__ xb,
                             const int* __restrict__ elist, const int* __restrict__ eoff,
                             const int* __restrict__ ecount, const int* __restrict__ active,
                             const int* __restrict__ nactive,
                             int H, int MI, int topk, int GRP) {
    int slot = blockIdx.x;
    if (slot >= *nactive) return;
    int e = active[slot];
    int cnt = ecount[e];
    int base = eoff[e];
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= MI) return;
    int lane = threadIdx.x;

    const uint4* grow = (const uint4*)(gp + ((long)e * MI + n) * (H / 2));
    const uint4* urow = (const uint4*)(up + ((long)e * MI + n) * (H / 2));
    const uint8_t* gsr = gs + ((long)e * MI + n) * (H / GRP);
    const uint8_t* usr = us + ((long)e * MI + n) * (H / GRP);
    float gi = ginv[e], ui = uinv[e];

    float accg[MAXTOK], accu[MAXTOK];
    int nt = min(cnt, MAXTOK);
    for (int i = 0; i < nt; ++i) { accg[i] = 0.f; accu[i] = 0.f; }

    const int C = H / 32;                              // 16 bytes of codes = 32 weights
    for (int c = lane; c < C; c += 32) {
        uint4 gv = __ldcs(grow + c), uv = __ldcs(urow + c);
        const uint8_t* gb = (const uint8_t*)&gv;
        const uint8_t* ub = (const uint8_t*)&uv;
        float gsa = e4m3f(gsr[(c * 32) / GRP]) * gi, gsb = e4m3f(gsr[(c * 32 + 16) / GRP]) * gi;
        float usa = e4m3f(usr[(c * 32) / GRP]) * ui, usb = e4m3f(usr[(c * 32 + 16) / GRP]) * ui;
        for (int i = 0; i < nt; ++i) {
            int a = elist[base + i];
            const uint16_t* xh = xb + (long)(a / topk) * H + c * 32;
            float g0 = 0.f, g1 = 0.f, u0 = 0.f, u1 = 0.f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float x0 = bf2f(xh[2 * j]), x1 = bf2f(xh[2 * j + 1]);
                g0 = fmaf(e2m1f(gb[j] & 0x0F), x0, g0);
                g0 = fmaf(e2m1f(gb[j] >> 4),   x1, g0);
                u0 = fmaf(e2m1f(ub[j] & 0x0F), x0, u0);
                u0 = fmaf(e2m1f(ub[j] >> 4),   x1, u0);
            }
            #pragma unroll
            for (int j = 8; j < 16; ++j) {
                float x0 = bf2f(xh[2 * j]), x1 = bf2f(xh[2 * j + 1]);
                g1 = fmaf(e2m1f(gb[j] & 0x0F), x0, g1);
                g1 = fmaf(e2m1f(gb[j] >> 4),   x1, g1);
                u1 = fmaf(e2m1f(ub[j] & 0x0F), x0, u1);
                u1 = fmaf(e2m1f(ub[j] >> 4),   x1, u1);
            }
            accg[i] += g0 * gsa + g1 * gsb;
            accu[i] += u0 * usa + u1 * usb;
        }
    }
    for (int i = 0; i < nt; ++i) {
        float gv = warp_sum(accg[i]), uv = warp_sum(accu[i]);
        if (lane == 0) hbuf[(long)elist[base + i] * MI + n] = silu(gv) * uv;
    }
}

// ---------------------------------------------------------------------------------------
// down: weight-resident over the same token group; writes per-assignment partials.
// ---------------------------------------------------------------------------------------
__global__ void k_moe_down(float* __restrict__ dpart,
                           const uint8_t* __restrict__ dp, const uint8_t* __restrict__ ds,
                           const float* __restrict__ dinv,
                           const uint16_t* __restrict__ hb,
                           const int* __restrict__ elist, const int* __restrict__ eoff,
                           const int* __restrict__ ecount, const int* __restrict__ active,
                           const int* __restrict__ nactive,
                           int H, int MI, int GRP) {
    int slot = blockIdx.x;
    if (slot >= *nactive) return;
    int e = active[slot];
    int cnt = ecount[e], base = eoff[e];
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= H) return;
    int lane = threadIdx.x;

    const uint4* drow = (const uint4*)(dp + ((long)e * H + n) * (MI / 2));
    const uint8_t* dsr = ds + ((long)e * H + n) * (MI / GRP);
    float di = dinv[e];

    float acc[MAXTOK];
    int nt = min(cnt, MAXTOK);
    for (int i = 0; i < nt; ++i) acc[i] = 0.f;

    const int C = MI / 32;
    for (int c = lane; c < C; c += 32) {
        uint4 dv = __ldcs(drow + c);
        const uint8_t* db = (const uint8_t*)&dv;
        float sa = e4m3f(dsr[(c * 32) / GRP]) * di, sb = e4m3f(dsr[(c * 32 + 16) / GRP]) * di;
        for (int i = 0; i < nt; ++i) {
            const uint16_t* hh = hb + (long)elist[base + i] * MI + c * 32;
            float h0 = 0.f, h1 = 0.f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                h0 = fmaf(e2m1f(db[j] & 0x0F), bf2f(hh[2 * j]), h0);
                h0 = fmaf(e2m1f(db[j] >> 4),   bf2f(hh[2 * j + 1]), h0);
            }
            #pragma unroll
            for (int j = 8; j < 16; ++j) {
                h1 = fmaf(e2m1f(db[j] & 0x0F), bf2f(hh[2 * j]), h1);
                h1 = fmaf(e2m1f(db[j] >> 4),   bf2f(hh[2 * j + 1]), h1);
            }
            acc[i] += h0 * sa + h1 * sb;
        }
    }
    for (int i = 0; i < nt; ++i) {
        float v = warp_sum(acc[i]);
        if (lane == 0) dpart[(long)elist[base + i] * H + n] = v;
    }
}

// ---------------------------------------------------------------------------------------
// finalize: out[token] = scaling * sum_over_slots( w_slot * dpart[assignment] ), summed in
// ASCENDING EXPERT INDEX to match the reference's accumulation order.
// ---------------------------------------------------------------------------------------
__global__ void k_moe_finalize(float* __restrict__ out, const float* __restrict__ dpart,
                               const float* __restrict__ wts, const int* __restrict__ sel,
                               int rows, int H, int topk, float scaling) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows * H) return;
    int r = t / H, d = t % H;

    // order the token's top-k slots by expert index (topk<=16, insertion sort in registers)
    int ord[16];
    for (int j = 0; j < topk; ++j) ord[j] = j;
    for (int a = 1; a < topk; ++a) {
        int key = ord[a], b = a - 1;
        while (b >= 0 && sel[(long)r * topk + ord[b]] > sel[(long)r * topk + key]) {
            ord[b + 1] = ord[b]; --b;
        }
        ord[b + 1] = key;
    }
    float acc = 0.f;
    for (int j = 0; j < topk; ++j) {
        int s = ord[j];
        long a = (long)r * topk + s;
        acc += dpart[a * H + d] * wts[a];
    }
    out[t] = acc * scaling;
}

extern "C" void moe_gateup(float* hbuf, const uint8_t* gp, const uint8_t* gs, const float* ginv,
                           const uint8_t* up, const uint8_t* us, const float* uinv,
                           const uint16_t* xb, const int* elist, const int* eoff,
                           const int* ecount, const int* active, const int* nactive,
                           int nact_max, int H, int MI, int topk, int GRP, cudaStream_t st) {
    dim3 blk(32, 4), grd(nact_max, (MI + 3) / 4);
    k_moe_gateup<<<grd, blk, 0, st>>>(hbuf, gp, gs, ginv, up, us, uinv, xb, elist, eoff,
                                      ecount, active, nactive, H, MI, topk, GRP);
}
extern "C" void moe_down(float* dpart, const uint8_t* dp, const uint8_t* ds, const float* dinv,
                         const uint16_t* hb, const int* elist, const int* eoff,
                         const int* ecount, const int* active, const int* nactive,
                         int nact_max, int H, int MI, int GRP, cudaStream_t st) {
    dim3 blk(32, 4), grd(nact_max, (H + 3) / 4);
    k_moe_down<<<grd, blk, 0, st>>>(dpart, dp, ds, dinv, hb, elist, eoff, ecount, active,
                                    nactive, H, MI, GRP);
}
extern "C" void moe_finalize(float* out, const float* dpart, const float* wts, const int* sel,
                             int rows, int H, int topk, float scaling, cudaStream_t st) {
    int T = 256, n = rows * H;
    k_moe_finalize<<<(n + T - 1) / T, T, 0, st>>>(out, dpart, wts, sel, rows, H, topk, scaling);
}
