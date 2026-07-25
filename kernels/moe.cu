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
// TMAX is a COMPILE-TIME bound so the per-token accumulators stay in registers. With a
// runtime bound (`for i < nt` over `float acc[16]`) the compiler cannot keep the array in
// registers and every accumulate becomes a LOCAL-memory round trip — the same failure mode
// as the e2m1f LUT. Experts with more than TMAX tokens are handled by looping token groups,
// which re-reads that expert's weight per group; at decode (1 token/expert) and verify
// (M<=8, so rarely >4 tokens on any one expert) that path is essentially never taken.
#define TMAX 4

template <int TM>
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
    float gi = ginv[e], ui = uinv[e];

    const uint8_t* gsr = gs + ((long)e * MI + n) * (H / GRP);
    const uint8_t* usr = us + ((long)e * MI + n) * (H / GRP);

    const int C = H / 32;                              // 16 bytes of codes = 32 weights
  for (int t0 = 0; t0 < cnt; t0 += TM) {
    const int nt = min(TM, cnt - t0);
    float accg[TM], accu[TM];
    const uint16_t* xrow[TM];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        accg[i] = 0.f; accu[i] = 0.f;
        // hoisted: the elist load and the /topk division were being redone every c-iteration
        int a = (i < nt) ? elist[base + t0 + i] : 0;
        xrow[i] = xb + (long)(a / topk) * H;
    }

    for (int c = lane; c < C; c += 32) {
        uint4 gv = __ldcs(grow + c), uv = __ldcs(urow + c);
        const uint8_t* gb = (const uint8_t*)&gv;
        const uint8_t* ub = (const uint8_t*)&uv;
        float gsa = e4m3f(gsr[(c * 32) / GRP]) * gi, gsb = e4m3f(gsr[(c * 32 + 16) / GRP]) * gi;
        float usa = e4m3f(usr[(c * 32) / GRP]) * ui, usb = e4m3f(usr[(c * 32 + 16) / GRP]) * ui;
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            if (i >= nt) break;
            const uint16_t* xh = xrow[i] + c * 32;
            float g0 = 0.f, g1 = 0.f, u0 = 0.f, u1 = 0.f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float x0 = bf2f(xh[2 * j]), x1 = bf2f(xh[2 * j + 1]);
                float2 gw = fp4x2_f2(gb[j]), uw = fp4x2_f2(ub[j]);
                g0 = fmaf(gw.x, x0, g0);  g0 = fmaf(gw.y, x1, g0);
                u0 = fmaf(uw.x, x0, u0);  u0 = fmaf(uw.y, x1, u0);
            }
            #pragma unroll
            for (int j = 8; j < 16; ++j) {
                float x0 = bf2f(xh[2 * j]), x1 = bf2f(xh[2 * j + 1]);
                float2 gw = fp4x2_f2(gb[j]), uw = fp4x2_f2(ub[j]);
                g1 = fmaf(gw.x, x0, g1);  g1 = fmaf(gw.y, x1, g1);
                u1 = fmaf(uw.x, x0, u1);  u1 = fmaf(uw.y, x1, u1);
            }
            accg[i] += g0 * gsa + g1 * gsb;
            accu[i] += u0 * usa + u1 * usb;
        }
    }
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        if (i >= nt) break;
        float gv = warp_sum(accg[i]), uv = warp_sum(accu[i]);
        if (lane == 0) hbuf[(long)elist[base + t0 + i] * MI + n] = silu(gv) * uv;
    }
  }
}

// ---------------------------------------------------------------------------------------
// down: weight-resident over the same token group; writes per-assignment partials.
// ---------------------------------------------------------------------------------------
template <int TM>
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
    float di = dinv[e];
    const uint8_t* dsr = ds + ((long)e * H + n) * (MI / GRP);

    const int C = MI / 32;
  for (int t0 = 0; t0 < cnt; t0 += TM) {
    const int nt = min(TM, cnt - t0);
    float acc[TM];
    const uint16_t* hrow[TM];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        acc[i] = 0.f;
        hrow[i] = hb + (long)((i < nt) ? elist[base + t0 + i] : 0) * MI;
    }

    for (int c = lane; c < C; c += 32) {
        uint4 dv = __ldcs(drow + c);
        const uint8_t* db = (const uint8_t*)&dv;
        float sa = e4m3f(dsr[(c * 32) / GRP]) * di, sb = e4m3f(dsr[(c * 32 + 16) / GRP]) * di;
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            if (i >= nt) break;
            const uint16_t* hh = hrow[i] + c * 32;
            float h0 = 0.f, h1 = 0.f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float2 dw = fp4x2_f2(db[j]);
                h0 = fmaf(dw.x, bf2f(hh[2 * j]), h0);
                h0 = fmaf(dw.y, bf2f(hh[2 * j + 1]), h0);
            }
            #pragma unroll
            for (int j = 8; j < 16; ++j) {
                float2 dw = fp4x2_f2(db[j]);
                h1 = fmaf(dw.x, bf2f(hh[2 * j]), h1);
                h1 = fmaf(dw.y, bf2f(hh[2 * j + 1]), h1);
            }
            acc[i] += h0 * sa + h1 * sb;
        }
    }
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        if (i >= nt) break;
        float v = warp_sum(acc[i]);
        if (lane == 0) dpart[(long)elist[base + t0 + i] * H + n] = v;
    }
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
                           int nact_max, int H, int MI, int topk, int GRP, int maxtok,
                           cudaStream_t st) {
    dim3 blk(32, 4), grd(nact_max, (MI + 3) / 4);
    // Unrolling the token loop replicates every temporary: TM=4 compiles to 128 registers,
    // which caps occupancy at 4 blocks/SM (128 threads x 128 regs = 16K of the SM's 64K).
    // At decode every expert receives exactly one token, so TM=1 is both correct and far
    // cheaper. Experts with more tokens than TM simply loop and re-read that weight.
    if (maxtok <= 1)
        k_moe_gateup<1><<<grd, blk, 0, st>>>(hbuf, gp, gs, ginv, up, us, uinv, xb, elist, eoff,
                                             ecount, active, nactive, H, MI, topk, GRP);
    else
        k_moe_gateup<TMAX><<<grd, blk, 0, st>>>(hbuf, gp, gs, ginv, up, us, uinv, xb, elist, eoff,
                                                ecount, active, nactive, H, MI, topk, GRP);
}
extern "C" void moe_down(float* dpart, const uint8_t* dp, const uint8_t* ds, const float* dinv,
                         const uint16_t* hb, const int* elist, const int* eoff,
                         const int* ecount, const int* active, const int* nactive,
                         int nact_max, int H, int MI, int GRP, int maxtok, cudaStream_t st) {
    dim3 blk(32, 4), grd(nact_max, (H + 3) / 4);
    if (maxtok <= 1)
        k_moe_down<1><<<grd, blk, 0, st>>>(dpart, dp, ds, dinv, hb, elist, eoff, ecount, active,
                                           nactive, H, MI, GRP);
    else
        k_moe_down<TMAX><<<grd, blk, 0, st>>>(dpart, dp, ds, dinv, hb, elist, eoff, ecount, active,
                                              nactive, H, MI, GRP);
}
extern "C" void moe_finalize(float* out, const float* dpart, const float* wts, const int* sel,
                             int rows, int H, int topk, float scaling, cudaStream_t st) {
    int T = 256, n = rows * H;
    k_moe_finalize<<<(n + T - 1) / T, T, 0, st>>>(out, dpart, wts, sel, rows, H, topk, scaling);
}
