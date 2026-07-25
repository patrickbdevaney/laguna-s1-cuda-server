// forward.cu — G7/G8: the full Laguna forward pass in pure CUDA.
//
// Layer order is a direct transcription of LagunaDecoderLayer.forward
// (modeling_laguna.py:490-519), with the Laguna-specific pieces called out where they differ
// from anything in the gemma port:
//   * per-layer head count (48 global / 72 sliding) and GQA group (6 / 9)
//   * QK-RMSNorm per head, BEFORE rope
//   * partial rotary: 64 of 128 dims on global (yarn), 128 of 128 on sliding
//   * per-head softplus gate driven by the POST-LAYERNORM input, applied before o_proj
//   * sigmoid router with a selection-only bias, top-10, sum-normalised
//   * routed-expert output scaled by 2.5 and added to a shared expert
//   * layer 0 is a dense MLP, not MoE
//   * lm_head is NOT tied to the embedding
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include "../include/laguna_config.h"
#include "../include/laguna_weights.h"

extern "C" {
void dequant_nvfp4(float*, const uint8_t*, const uint8_t*, float, int, int, int, cudaStream_t);
void gemm_bf16(float*, const uint16_t*, const uint16_t*, int, int, int, cudaStream_t);
void gemm_fp8(float*, const uint8_t*, const float*, const uint16_t*, int, int, int, cudaStream_t);
void gemm_bf16_seg(float* const*, const int*, int, const uint16_t*, const uint16_t*, int, int, cudaStream_t);
void gemm_fp8_seg(float* const*, const int*, int, const uint8_t*, const float*, const uint16_t*, int, int, cudaStream_t);
void gemm_fp4(float*, const uint8_t*, const uint8_t*, float, const uint16_t*, int, int, int, int, cudaStream_t);
void f32_to_bf16(uint16_t*, const float*, long, cudaStream_t);
void rmsnorm(float*, const float*, const uint16_t*, int, int, float, cudaStream_t);
void add_rms_cast(uint16_t*, float*, const float*, const uint16_t*, int, int, float, int, cudaStream_t);
void rmsnorm_heads(float*, const float*, const uint16_t*, int, int, int, float, cudaStream_t);
void rope_tables(float*, float*, const float*, int, const int*, int, float, cudaStream_t);
void rope_apply(float*, const float*, const float*, int, int, int, int, cudaStream_t);
void router(int*, float*, float*, const float*, const float*, int, int, int, float, int, cudaStream_t);
void gate_softplus(float*, const float*, int, int, int, cudaStream_t);
void swiglu(float*, const float*, const float*, long, cudaStream_t);
void add_inplace(float*, const float*, long, cudaStream_t);
void embed_rows(float*, const uint16_t*, const int*, int, int, cudaStream_t);
void tap_store(float*, const float*, int, int, int, int, int, cudaStream_t);
void tap_concat_cast(uint16_t*, const float*, int, int, int, int, int, cudaStream_t);
void rmsnorm_tap(float*, const uint16_t*, int, int, int, int, int, float, cudaStream_t);
void store_kv(uint8_t*, uint8_t*, const float*, const float*, float, float, int, int, int, int, const int*, cudaStream_t);
void attend(float*, const float*, const uint8_t*, const uint8_t*, float, float, int, int, int, int, int, const int*, float, cudaStream_t);
int  attend_nsplit(int, int, int);
void attend_split(float*, float*, float*, const float*, const uint8_t*, const uint8_t*, float, float, int, int, int, int, int, const int*, float, int, cudaStream_t);
void set_base(int*, int, cudaStream_t);
void inc_base(int*, int, cudaStream_t);
void moe_invert(int*, int*, int*, int*, int*, int*, const int*, int, int, int, cudaStream_t);
void moe_gateup_rp(float*, const uint8_t*, const uint8_t*, const float*, const uint8_t*, const uint8_t*,
                const float*, const uint16_t*, const int*, const int*, const int*, const int*,
                const int*, int, int, int, int, int, int, cudaStream_t);
void moe_down_rp(float*, const uint8_t*, const uint8_t*, const float*, const uint16_t*, const int*,
              const int*, const int*, const int*, const int*, int, int, int, int, int, cudaStream_t);
void moe_finalize(float*, const float*, const float*, const int*, int, int, int, float, cudaStream_t);
void moe_gateup_split(float*, float*, uint16_t*, const uint8_t*, const uint8_t*, const float*,
                      const uint8_t*, const uint8_t*, const float*, const uint16_t*, const int*,
                      const int*, const int*, const int*, const int*, int, int, int, int, int,
                      int, long, cudaStream_t);
}

namespace laguna {

// KV cache. Sliding layers get a ring of exactly `window` slots — 36 of 48 layers therefore
// cost a constant 1.05 MB per sequence regardless of conversation length (ROOFLINE.md §6).
struct Session {
    std::vector<uint8_t*> Kc, Vc;
    std::vector<int> cap;
    int pos = 0;
    size_t bytes = 0;

    void alloc(const Config& c, int ctx, int max_tok = 64) {
        Kc.resize(c.n_layers); Vc.resize(c.n_layers); cap.resize(c.n_layers);
        size_t per = (size_t)c.n_kv_heads * c.head_dim;
        for (int L = 0; L < c.n_layers; ++L) {
            // CORRECTNESS: the ring must be LARGER than the window. With cap == window the
            // read arc [p-window+1, p] covers all `cap` slots, so any token written later in
            // the SAME batch aliases into it: at base=1000, M=64, query m=0 reads j=489 and
            // gets position 1001's K/V. 63 of 64 tokens collide. Decode (M=1) is immune
            // because 512 consecutive positions map to 512 distinct slots; this only bites
            // multi-token forwards (prefill, and speculative verify) past position `window`.
            // cap = window + MAXTOK makes the read arc and the look-ahead arc disjoint.
            cap[L] = c.is_sliding(L) ? (c.sliding_window + max_tok) : ctx;
            size_t n = per * cap[L];
            CUDA_CHECK(cudaMalloc(&Kc[L], n));
            CUDA_CHECK(cudaMalloc(&Vc[L], n));
            CUDA_CHECK(cudaMemset(Kc[L], 0, n));
            CUDA_CHECK(cudaMemset(Vc[L], 0, n));
            bytes += 2 * n;
        }
    }
    void free_() { for (auto p : Kc) cudaFree(p); for (auto p : Vc) cudaFree(p); Kc.clear(); Vc.clear(); }
};

// Bytes a decode step at position `pos` MUST read, from the checkpoint layout actually in
// use. This is the denominator of every effective-bandwidth number in the logs. It used to be
// a hardcoded 10.044 GB — the stock BF16 figure — which quietly kept crediting us with the
// attention bytes that FP8 attention had already removed, and so reported 83 % of roofline
// where the truth was 66 %. Never hardcode a byte budget that a build flag can change.
inline double bytes_per_token(const Config& c, bool fp8_attn, int pos) {
    const double H = c.hidden, V = c.vocab;
    const double aw = fp8_attn ? 1.0 : 2.0;                   // attention weight element
    double b = H * 2;                                          // one embedding row
    b += H * 2 + V * H * 2;                                    // final norm + lm_head
    for (int L = 0; L < c.n_layers; ++L) {
        const double qd = c.q_dim(L), kvd = (double)c.n_kv_heads * c.head_dim, nh = c.heads[L];
        b += 2.0 * H * 2;                                      // in_ln + post_ln
        b += (qd * H + 2.0 * kvd * H + H * qd + nh * H) * aw;   // q, k, v, o, g
        if (fp8_attn) b += (qd + 2.0 * kvd + H + nh) * 4;       // one fp32 scale per out row
        b += 2.0 * c.head_dim * 2;                             // q_norm + k_norm
        const int klen = c.is_sliding(L) ? std::min(pos + 1, c.sliding_window) : pos + 1;
        b += (double)klen * kvd * 2;                           // FP8 K and V
        if (c.is_dense(L)) {
            b += 3.0 * c.intermediate * H * 2;
        } else {
            b += 3.0 * c.shared_intermediate * H * 2;
            b += (double)c.n_experts * H * 2 + c.n_experts * 4; // router + selection bias
            const double ew = 3.0 * c.moe_intermediate * H;     // gate + up + down elements
            b += c.top_k * (ew * 0.5 + ew / c.nvfp4_group);     // FP4 codes + E4M3 scales
        }
    }
    return b;
}

// Measured streaming ceiling for cudaMalloc'd memory on this part (OPTIMIZATION_LOG M1).
static const double THOR_BW_CEILING = 227.4e9;

struct Engine {
    Config c;
    Weights W;
    int MAXTOK = 64;

    // scratch
    float *h, *hn, *q, *kk, *vv, *att, *gp, *mlp_a, *mlp_b, *moe_h, *dpart, *rlogit, *rwts, *logits;
    uint16_t *hb, *attb, *moe_hb;
    int *sel, *ecount, *eoff, *elist, *cursor, *active, *nactive;
    float *cos_f, *sin_f, *cos_s, *sin_s, *inv_f, *inv_s;
    float *pacc, *pml;                 // split-attention partials
    float *moe_u = nullptr;            // second gate/up accumulator for the split-warp path
    bool  moe_split = false;   // LOST end-to-end: see OPTIMIZATION_LOG #14
    int   *dbase = nullptr;            // decode position, device-side (graph-safe)
    static const int MAXSPLIT = 32;

    // --- optional per-layer capture, for localising a prefill/decode divergence
    bool dbg = false; int dbg_row = 0;
    std::vector<std::vector<float>> dbg_h;
    void dbg_grab(int M) {
        if (!dbg) return;
        std::vector<float> row(c.hidden);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(row.data(), h + (size_t)std::min(dbg_row, M - 1) * c.hidden,
                              c.hidden * 4, cudaMemcpyDeviceToHost));
        dbg_h.push_back(row);
    }

    // --- CUDA graph for the M=1 decode step
    cudaGraphExec_t g_exec = nullptr;
    bool graph_ready = false;
    // Upper bound on context used to size the attention split count. It becomes grid.z, so it
    // is baked into a captured graph — hence a *bound*, not the live position. Sizing it from
    // full KV capacity would launch 10 splits per global layer from token one; instead we
    // capture against a doubling bound and re-capture when the conversation outgrows it.
    int  split_ctx = 0;
    // Diagnostic/determinism override. attend_nsplit() sizes the key split from M, so a
    // batched forward and a single-token decode of the SAME token combine their partials in
    // different orders. Pinning it makes the two paths bit-comparable.
    int  nsp_force = 0;

    // ---- E_frac instrumentation: how many of the 256 experts a forward actually touches.
    // At decode M=1 it is at most top_k=10. At the speculative-verify shape M=k+1 the block
    // routes M*top_k assignments, and the DISTINCT count is what the MoE kernel pays for.
    // This is the number that decides whether speculation is worth anything on a MoE target.
    bool efrac = false;
    long efrac_sum = 0, efrac_n = 0;

    // ---- DFlash target taps. Non-null enables capture of the residual stream after the
    // layers in `tap_of`, which is the ONLY thing the draft consumes from the target.
    float* taps = nullptr;
    int    tap_cap = 0;                    // ring length in positions
    std::vector<int> tap_of;               // tap_of[L] = slot, or -1

    void init(int maxtok) {
        MAXTOK = maxtok;
        int H = c.hidden, V = c.vocab, E = c.n_experts, TK = c.top_k, MI = c.moe_intermediate;
        int maxq = 0, maxheads = 0;
        for (int L = 0; L < c.n_layers; ++L) {
            maxq = std::max(maxq, c.q_dim(L));
            maxheads = std::max(maxheads, c.heads[L]);
        }
        int kvd = c.n_kv_heads * c.head_dim;
        auto A = [&](void** p, size_t n) { CUDA_CHECK(cudaMalloc(p, n)); };
        A((void**)&h,   (size_t)MAXTOK * H * 4);
        A((void**)&hn,  (size_t)MAXTOK * H * 4);
        A((void**)&hb,  (size_t)MAXTOK * std::max(H, maxq) * 2);
        A((void**)&q,   (size_t)MAXTOK * maxq * 4);
        A((void**)&kk,  (size_t)MAXTOK * kvd * 4);
        A((void**)&vv,  (size_t)MAXTOK * kvd * 4);
        A((void**)&att, (size_t)MAXTOK * maxq * 4);
        // attb is reused as the bf16 staging buffer for THREE different widths: the
        // attention output (q_dim, up to 9216), the layer-0 dense MLP intermediate (12288)
        // and the shared expert (1024). Sizing it by q_dim alone overflowed by M*3072
        // elements on every forward -- 6 KB at M=1, which landed in cudaMalloc slack and so
        // never showed up in the short greedy gate, and 393 KB at M=64, which corrupted
        // prefill. Found by tests/gate_longctx.cu.
        A((void**)&attb,(size_t)MAXTOK * std::max(maxq, c.intermediate) * 2);
        A((void**)&gp,  (size_t)MAXTOK * maxheads * 4);
        A((void**)&mlp_a, (size_t)MAXTOK * c.intermediate * 4);
        A((void**)&mlp_b, (size_t)MAXTOK * c.intermediate * 4);
        A((void**)&moe_h, (size_t)MAXTOK * TK * MI * 4);
        A((void**)&moe_hb,(size_t)MAXTOK * TK * MI * 2);
        A((void**)&moe_u, (size_t)MAXTOK * TK * MI * 4);
        A((void**)&dpart, (size_t)MAXTOK * TK * H * 4);
        A((void**)&rlogit,(size_t)MAXTOK * E * 4);
        A((void**)&rwts,  (size_t)MAXTOK * TK * 4);
        A((void**)&logits,(size_t)MAXTOK * V * 4);
        A((void**)&sel,   (size_t)MAXTOK * TK * 4);
        A((void**)&ecount, E * 4); A((void**)&eoff, (E + 1) * 4);
        A((void**)&elist, (size_t)MAXTOK * TK * 4);
        A((void**)&cursor, E * 4); A((void**)&active, E * 4); A((void**)&nactive, 4);
        // split-attention scratch: [M][nkv*Gmax][NSP][HD] plus {m,l} per partial
        int gmax = 0; for (int L = 0; L < c.n_layers; ++L) gmax = std::max(gmax, c.kv_group(L));
        size_t nh_max = (size_t)MAXTOK * c.n_kv_heads * gmax * MAXSPLIT;
        A((void**)&pacc, nh_max * c.head_dim * 4);
        A((void**)&pml,  nh_max * 2 * 4);
        A((void**)&dbase, 4);
        CUDA_CHECK(cudaMemset(dbase, 0, 4));

        // rope: one table per layer TYPE (not per layer) — Laguna has exactly two
        auto ivf = c.rope_full.inv_freq(c.head_dim);
        auto ivs = c.rope_slide.inv_freq(c.head_dim);
        A((void**)&inv_f, ivf.size() * 4); A((void**)&inv_s, ivs.size() * 4);
        CUDA_CHECK(cudaMemcpy(inv_f, ivf.data(), ivf.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(inv_s, ivs.data(), ivs.size() * 4, cudaMemcpyHostToDevice));
        A((void**)&cos_f, (size_t)MAXTOK * c.rope_full.rotary_dim * 4);
        A((void**)&sin_f, (size_t)MAXTOK * c.rope_full.rotary_dim * 4);
        A((void**)&cos_s, (size_t)MAXTOK * c.rope_slide.rotary_dim * 4);
        A((void**)&sin_s, (size_t)MAXTOK * c.rope_slide.rotary_dim * 4);
    }

    // ---- optional per-category profiling (LG_PROF=1). Serialises the step, so it is
    // strictly a diagnostic: never enable it while timing tok/s.
    static const int NCAT = 8;
    double cat_ms[NCAT] = {0};
    long   cat_n[NCAT] = {0};
    bool prof = false;
    const char* CATN[NCAT] = {"embed+rope","norm+cast","attn_qkvo_gemm","attn_core",
                              "shared+dense","router","moe_experts","lm_head"};
    double t_mark = 0;
    inline void mark() { if (prof) { cudaDeviceSynchronize(); t_mark = wall_now(); } }
    inline void acc(int i) {
        if (prof) { cudaDeviceSynchronize(); cat_ms[i] += (wall_now() - t_mark) * 1e3; ++cat_n[i]; }
    }
    // Zero the counters. The profile is per-DECODE-step; if the synthetic prefill is left in
    // it dominates completely (2048 tokens at M=64 vs 39 tokens at M=1) and the percentages
    // describe prefill's bottleneck, not decode's.
    void prof_reset() { for (int i = 0; i < NCAT; ++i) { cat_ms[i] = 0; cat_n[i] = 0; } }
    void prof_report(int steps) {
        if (!prof) return;
        double tot = 0; for (int i = 0; i < NCAT; ++i) tot += cat_ms[i];
        printf("\n  %-18s %10s %8s %8s\n","category","ms/step","%","launches");
        for (int i = 0; i < NCAT; ++i)
            printf("  %-18s %10.3f %7.1f%% %8ld\n", CATN[i], cat_ms[i]/steps,
                   100.0*cat_ms[i]/tot, cat_n[i]/steps);
        printf("  %-18s %10.3f\n","TOTAL",tot/steps);
    }

    // One forward over M tokens starting at absolute position `base`.
    void forward(Session& S, const int* d_ids, int M, int base, cudaStream_t st = 0) {
        const int H = c.hidden, E = c.n_experts, TK = c.top_k, MI = c.moe_intermediate;
        const float qscale = 1.0f / sqrtf((float)c.head_dim);

        mark();
        embed_rows(h, W.embed, d_ids, M, H, st);
        rope_tables(cos_f, sin_f, inv_f, M, dbase, c.rope_full.rotary_dim / 2,
                    (float)c.rope_full.attention_factor, st);
        rope_tables(cos_s, sin_s, inv_s, M, dbase, c.rope_slide.rotary_dim / 2,
                    (float)c.rope_slide.attention_factor, st);
        acc(0);

        for (int L = 0; L < c.n_layers; ++L) {
            const LayerW& w = W.L[L];
            const int nh = c.heads[L], G = c.kv_group(L), hd = c.head_dim;
            const int qd = c.q_dim(L), kvd = c.n_kv_heads * hd;
            const bool slide = c.is_sliding(L);
            const float* cosT = slide ? cos_s : cos_f;
            const float* sinT = slide ? sin_s : sin_f;
            const int rot = slide ? c.rope_slide.rotary_dim : c.rope_full.rotary_dim;

            // ---- attention
            mark();
            add_rms_cast(hb, h, nullptr, w.in_ln, M, H, (float)c.rms_eps, 0, st);
            acc(1); mark();
            if (W.fp8_attn) {
                // ONE launch for q|k|v|g. They share K and share the input, the arena keeps
                // them contiguous, and each keeps its own output buffer -- so this is
                // bit-exact and nothing downstream changes. On their own the small ones
                // measure 168 GB/s (k/v) and 25 (g) against q's 236; fused they all run at
                // q's rate. The gate comes from the NORMED input, same as q/k/v.
                float* outs[4] = {q, kk, vv, gp};
                int    Ns[4]   = {qd, kvd, kvd, nh};
                gemm_fp8_seg(outs, Ns, 4, w.q8, w.q8s, hb, M, H, st);
            } else {
                float* outs[4] = {q, kk, vv, gp};
                int    Ns[4]   = {qd, kvd, kvd, nh};
                gemm_bf16_seg(outs, Ns, 4, w.q, hb, M, H, st);
            }
            acc(2); mark();
            rmsnorm_heads(q,  q,  w.q_norm, M, nh,          hd, (float)c.rms_eps, st);
            rmsnorm_heads(kk, kk, w.k_norm, M, c.n_kv_heads, hd, (float)c.rms_eps, st);
            rope_apply(q,  cosT, sinT, M, nh,           hd, rot, st);
            rope_apply(kk, cosT, sinT, M, c.n_kv_heads, hd, rot, st);
            store_kv(S.Kc[L], S.Vc[L], kk, vv, w.k_scale, w.v_scale, M, c.n_kv_heads, hd,
                     S.cap[L], dbase, st);
            // The split count must NOT depend on the current position: it becomes grid.z and
            // would be baked into a captured graph. Size it from the cache CAPACITY instead,
            // so it is constant for the life of the session; blocks whose key range is empty
            // exit immediately and contribute a -inf partial that the combine ignores.
            int klen = slide ? c.sliding_window
                             : std::min(split_ctx > 0 ? split_ctx : S.cap[L], S.cap[L]);
            int nsp = nsp_force ? nsp_force : attend_nsplit(M, c.n_kv_heads, klen);
            if (nsp > MAXSPLIT) nsp = MAXSPLIT;
            attend_split(att, pacc, pml, q, S.Kc[L], S.Vc[L], w.k_scale, w.v_scale, M,
                         c.n_kv_heads, G, S.cap[L], slide ? c.sliding_window : 0, dbase,
                         qscale, nsp, st);
            gate_softplus(att, gp, M, nh, hd, st);
            acc(3); mark();
            f32_to_bf16(attb, att, (long)M * qd, st);
            if (W.fp8_attn) gemm_fp8(hn, w.o8, w.o8s, attb, M, H, qd, st);
            else            gemm_bf16(hn, w.o, attb, M, H, qd, st);
            acc(2); mark();

            // ---- MLP / MoE.  The attention residual add is folded into this norm.
            add_rms_cast(hb, h, hn, w.post_ln, M, H, (float)c.rms_eps, 1, st);
            acc(1); mark();
            if (c.is_dense(L)) {
                int I = c.intermediate;
                { float* o2[2] = {mlp_a, mlp_b}; int n2[2] = {I, I};
                  gemm_bf16_seg(o2, n2, 2, w.mlp_gate, hb, M, H, st); }
                swiglu(mlp_a, mlp_a, mlp_b, (long)M * I, st);
                f32_to_bf16(attb, mlp_a, (long)M * I, st);
                gemm_bf16(hn, w.mlp_down, attb, M, H, I, st);
                add_inplace(h, hn, (long)M * H, st);
                acc(4);
            } else {
                // shared expert
                int SI = c.shared_intermediate;
                { float* o2[2] = {mlp_a, mlp_b}; int n2[2] = {SI, SI};
                  gemm_bf16_seg(o2, n2, 2, w.sh_gate, hb, M, H, st); }
                swiglu(mlp_a, mlp_a, mlp_b, (long)M * SI, st);
                f32_to_bf16(attb, mlp_a, (long)M * SI, st);
                gemm_bf16(hn, w.sh_down, attb, M, H, SI, st);
                add_inplace(h, hn, (long)M * H, st);      // shared expert added first
                acc(4); mark();

                // routed experts
                gemm_bf16(rlogit, w.router, hb, M, E, H, st);
                router(sel, rwts, nullptr, rlogit, w.router_bias, M, E, TK,
                       (float)c.router_softcap, c.norm_topk_prob ? 1 : 0, st);
                moe_invert(ecount, eoff, elist, cursor, active, nactive, sel, M, TK, E, st);
                acc(5); mark();
                // Grid must be sized by the tokens in THIS call, not by MAXTOK: at decode
                // M=1 there are at most 10 active experts, and launching 256 x (MI/4) blocks
                // means 96 % of them exist only to read *nactive and exit.
                int nact = std::min(E, M * TK);
                if (moe_split) {
                    // gate and up on separate warps: 320 -> 640 warps. The kernel is
                    // warp-starved, not bandwidth-limited (measured 159.6 -> 195.6 GB/s),
                    // because moe_intermediate 1024 is 3x smaller than hidden 3072.
                    moe_gateup_split(moe_h, moe_u, moe_hb,
                                     w.e_gate_p, w.e_gate_s, w.e_gate_inv,
                                     w.e_up_p, w.e_up_s, w.e_up_inv, hb, elist, eoff, ecount,
                                     active, nactive, nact, H, MI, TK, c.nvfp4_group,
                                     (M == 1 ? 1 : 4), (long)M * TK * MI, st);
                } else {
                    moe_gateup_rp(moe_h, w.e_gate_p, w.e_gate_s, w.e_gate_inv,
                               w.e_up_p, w.e_up_s, w.e_up_inv, hb, elist, eoff, ecount,
                               active, nactive, nact, H, MI, TK, c.nvfp4_group, (M == 1 ? 1 : 4), st);
                    f32_to_bf16(moe_hb, moe_h, (long)M * TK * MI, st);
                }
                moe_down_rp(dpart, w.e_down_p, w.e_down_s, w.e_down_inv, moe_hb, elist, eoff,
                         ecount, active, nactive, nact, H, MI, c.nvfp4_group, (M == 1 ? 1 : 4), st);
                if (efrac) {
                    int na = 0;
                    CUDA_CHECK(cudaMemcpyAsync(&na, nactive, 4, cudaMemcpyDeviceToHost, st));
                    CUDA_CHECK(cudaStreamSynchronize(st));
                    efrac_sum += na; ++efrac_n;
                }
                moe_finalize(hn, dpart, rwts, sel, M, H, TK, (float)c.routed_scaling, st);
                add_inplace(h, hn, (long)M * H, st);
                acc(6);
            }
            if (taps && tap_of[L] >= 0)
                tap_store(taps, h, M, H, base, tap_cap, tap_of[L], st);
            dbg_grab(M);
        }
        mark();
        add_rms_cast(hb, h, nullptr, W.final_norm, M, H, (float)c.rms_eps, 0, st);
        gemm_bf16(logits, W.lm_head, hb, M, c.vocab, H, st);   // NOT tied to the embedding
        acc(7);
    }

    // ---- G9: capture the M=1 decode step once, replay it every token.
    // Everything position-dependent reads `dbase` from device memory, so one graph is valid
    // at every step. The only per-token host work is writing the new id into `d_tok` and
    // setting `dbase`. The attention split count is sized from cache CAPACITY, not the
    // current position, precisely so it does not get baked in wrong.
    // Re-capture when the live position outgrows the bound the graph was built against.
    bool needs_recapture(int pos) const { return !graph_ready || pos >= split_ctx; }

    void capture(Session& S, int* d_tok, int pos = 0) {
        if (prof) return;                     // profiling needs per-category syncs
        split_ctx = 256; while (split_ctx <= pos) split_ctx <<= 1;   // doubling bound
        cudaStream_t cap;
        CUDA_CHECK(cudaStreamCreate(&cap));
        CUDA_CHECK(cudaStreamBeginCapture(cap, cudaStreamCaptureModeThreadLocal));
        forward(S, d_tok, 1, 0, cap);
        cudaGraph_t g;
        CUDA_CHECK(cudaStreamEndCapture(cap, &g));
        if (g_exec) cudaGraphExecDestroy(g_exec);
        CUDA_CHECK(cudaGraphInstantiate(&g_exec, g, nullptr, nullptr, 0));
        cudaGraphDestroy(g);
        CUDA_CHECK(cudaStreamDestroy(cap));
        graph_ready = true;
    }

    void step_graph(int pos, cudaStream_t st = 0) {
        set_base(dbase, pos, st);
        CUDA_CHECK(cudaGraphLaunch(g_exec, st));
    }
};

} // namespace laguna
