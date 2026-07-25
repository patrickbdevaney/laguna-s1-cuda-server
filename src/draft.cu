// draft.cu — the DFlash speculator.
//
// DFlash is *parallel block-diffusion* drafting, not an autoregressive small model. One draft
// forward produces the whole block:
//
//   * Its Keys/Values for everything before the block are **not** computed from its own
//     residual stream. They are projected from the TARGET's fused hidden states — six tapped
//     residual vectors per position, from target layers [1,10,19,29,38,47], concatenated and
//     put through `fc [3072,18432]`. Every draft layer projects K/V from the SAME fused
//     vector; there is no per-layer context residual stream. This is the trick that makes the
//     draft cheap and that makes it track the target so closely.
//   * Its Queries are `[bonus_token, MASK, MASK, ...]` — the token the target just committed,
//     followed by `block_size-1` copies of `mask_token_id`. One forward denoises them all.
//
// So the cost of a propose is: 2.230 GB of draft weights + one `lm_head` (0.617 GB), read
// once for the entire block, against the target's ~7 GB per token. That ratio is the whole
// argument for speculation on a bandwidth-bound part.
//
// Laguna's draft differs from the gemma-era one in ways that matter:
//   * `dflash_config.causal = true` — the block is CAUSAL, not bidirectional. gemma's was
//     non-causal. This means the existing target attention kernel is exactly right as-is.
//   * Six `aux_hidden_norms`, one per tap, which gemma's draft did not have.
//   * Laguna-style layer: QK-RMSNorm pre-RoPE and a per-head softplus gate on the attention
//     output, fused `qkv_proj [11264,3072]` = 9216 q + 1024 k + 1024 v.
//   * All six layers slide over 512, so the draft's KV is 13 MB and CONSTANT in conversation
//     length — the speculator costs nothing in the long-context budget.
//
// The target applies no embedding scale (`k_embed` is a raw table lookup), so the
// "unscaled embeddings" ambiguity that the gemma spec flags does not arise here: the draft
// sees exactly what the target sees.
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include "../include/laguna_config.h"
#include "../include/draft_weights.h"

namespace laguna {

#include "../include/laguna_kernels_api.h"

struct Drafter {
    DraftConfig d;
    DraftWeights W;

    // The draft's KV ring. cap = window + block, for the same reason the target's does:
    // with cap == window a query can alias a key written LATER in the same block.
    std::vector<uint8_t*> Kc, Vc;
    int cap = 0;

    // FP8 KV, like the target. The target ships calibrated per-layer k_scale/v_scale; the
    // draft checkpoint ships none, and inventing one is exactly what the directive forbids.
    // 1.0 is not an invented model constant, it is the identity: K and V here are outputs of
    // an RMS-normalised, RoPE'd projection and are O(1), which is inside e4m3's range with
    // room to spare. If the acceptance rate ever comes in low, this is the first thing to
    // A/B against a BF16 KV path.
    float kv_scale = 1.0f;

    float *h, *hn, *q, *kk, *vv, *att, *gp, *mlp_a, *mlp_b, *fused, *logits;
    float *pacc, *pml;
    uint16_t *hb, *attb, *fcx;
    int *dpos, *d_ids;
    float *inv_freq;
    float *cosT, *sinT;
    int MAXROW = 0;
    static const int MAXSPLIT = 32;

    int qd() const { return d.n_heads * d.head_dim; }
    int kvd() const { return d.n_kv_heads * d.head_dim; }
    int ntap() const { return (int)d.target_layer_ids.size(); }

    // max_rows must cover BOTH the query block AND the context chunk size.
    void init(int max_rows) {
        MAXROW = std::max(max_rows, d.block_size);
        const int H = d.hidden, I = d.intermediate, V = d.vocab;
        cap = d.sliding_window + d.block_size;
        auto A = [&](void** p, size_t n) { CUDA_CHECK(cudaMalloc(p, n)); };
        Kc.resize(d.n_layers); Vc.resize(d.n_layers);
        for (int l = 0; l < d.n_layers; ++l) {
            size_t n = (size_t)kvd() * cap;
            A((void**)&Kc[l], n); A((void**)&Vc[l], n);
            CUDA_CHECK(cudaMemset(Kc[l], 0, n)); CUDA_CHECK(cudaMemset(Vc[l], 0, n));
        }
        A((void**)&h,     (size_t)MAXROW * H * 4);
        A((void**)&hn,    (size_t)MAXROW * H * 4);
        A((void**)&fused, (size_t)MAXROW * H * 4);
        A((void**)&q,     (size_t)MAXROW * qd() * 4);
        A((void**)&kk,    (size_t)MAXROW * kvd() * 4);
        A((void**)&vv,    (size_t)MAXROW * kvd() * 4);
        A((void**)&att,   (size_t)MAXROW * qd() * 4);
        A((void**)&gp,    (size_t)MAXROW * d.n_heads * 4);
        A((void**)&mlp_a, (size_t)MAXROW * I * 4);
        A((void**)&mlp_b, (size_t)MAXROW * I * 4);
        A((void**)&logits,(size_t)d.block_size * V * 4);
        A((void**)&hb,    (size_t)MAXROW * H * 2);
        A((void**)&attb,  (size_t)MAXROW * std::max(qd(), I) * 2);
        A((void**)&fcx,   (size_t)MAXROW * ntap() * H * 2);
        A((void**)&dpos,  4);
        A((void**)&d_ids, (size_t)d.block_size * 4);
        A((void**)&pacc,  (size_t)MAXROW * d.n_kv_heads * (d.n_heads / d.n_kv_heads) *
                          d.head_dim * MAXSPLIT * 4);
        A((void**)&pml,   (size_t)MAXROW * d.n_kv_heads * (d.n_heads / d.n_kv_heads) *
                          MAXSPLIT * 2 * 4);
        // The draft's rope: theta from its own config, full rotary over head_dim, no scaling.
        // Built here rather than reused from the target because the target's sliding rope has
        // the same theta but the full rope does not, and silently sharing one would be a bug
        // that only shows up as a slightly worse acceptance rate.
        int half = d.head_dim / 2;
        std::vector<float> iv(half);
        for (int i = 0; i < half; ++i)
            iv[i] = 1.0f / (float)std::pow(d.rope_theta, (double)(2 * i) / d.head_dim);
        A((void**)&inv_freq, half * 4);
        CUDA_CHECK(cudaMemcpy(inv_freq, iv.data(), half * 4, cudaMemcpyHostToDevice));
        A((void**)&cosT, (size_t)MAXROW * d.head_dim * 4);
        A((void**)&sinT, (size_t)MAXROW * d.head_dim * 4);
    }

    void free_() {
        for (auto p : Kc) cudaFree(p);
        for (auto p : Vc) cudaFree(p);
        Kc.clear(); Vc.clear();
    }

    // ---------------------------------------------------------------------------------
    // Context branch: turn the target's taps for positions [p0, p0+C) into draft K/V.
    //
    //   per tap i:  t_i = rmsnorm(tap_i, aux_hidden_norms[i])
    //   fused       = fc @ concat(t_0..t_5)
    //   fused_n     = rmsnorm(fused, hidden_norm)
    //   per layer:  K,V = qkv_proj[k,v rows] @ fused_n ; K = qk_norm(K) ; K = rope(K)
    //
    // Every layer projects from the same `fused_n`. `tap_cap` is the target-side tap ring.
    // ---------------------------------------------------------------------------------
    void context_kv(const float* taps, int tap_cap, int p0, int C, cudaStream_t st = 0) {
        // Chunked by MAXROW. The block forward only ever needs `block_size` rows, but the
        // context branch has to cover the whole draft window (512) after a prefill -- sizing
        // the scratch for the block and then running the context branch over 512 rows writes
        // past the end of every buffer here, and the symptom is not a crash: the draft simply
        // emits one constant token forever, which a correctness gate happily passes because
        // every draft token is rejected. tau is what catches it.
        for (int off = 0; off < C; off += MAXROW)
            context_chunk(taps, tap_cap, p0 + off, std::min(MAXROW, C - off), st);
    }

    void context_chunk(const float* taps, int tap_cap, int p0, int C, cudaStream_t st) {
        if (C <= 0) return;
        const int H = d.hidden;
        tap_fuse(fcx, taps, W.aux_norm.data(), C, H, tap_cap, p0, ntap(), (float)d.rms_eps, st);
        gemm_bf16(fused, W.fc, fcx, C, H, ntap() * H, st);
        rmsnorm(fused, fused, W.hidden_norm, C, H, (float)d.rms_eps, st);
        f32_to_bf16(hb, fused, (long)C * H, st);

        set_base(dpos, p0, st);
        rope_tables(cosT, sinT, inv_freq, C, dpos, d.head_dim / 2, 1.0f, st);
        for (int l = 0; l < d.n_layers; ++l) {
            const auto& w = W.L[l];
            // rows [qd, qd+2*kvd) of the fused qkv_proj are k then v
            float* outs[2] = {kk, vv};
            int    Ns[2]   = {kvd(), kvd()};
            gemm_bf16_seg(outs, Ns, 2, w.qkv + (size_t)qd() * H, hb, C, H, st);
            rmsnorm_heads(kk, kk, w.k_norm, C, d.n_kv_heads, d.head_dim, (float)d.rms_eps, st);
            rope_apply(kk, cosT, sinT, C, d.n_kv_heads, d.head_dim, d.head_dim, st);
            store_kv(Kc[l], Vc[l], kk, vv, kv_scale, kv_scale, C, d.n_kv_heads, d.head_dim,
                     cap, dpos, st);
        }
    }

    // ---------------------------------------------------------------------------------
    // Query branch: one forward over `block_size` positions starting at P0, then read the
    // argmax at the mask slots. Returns k ids on the host.
    // ---------------------------------------------------------------------------------
    void propose(int next_token, int P0, int k, const uint16_t* embed, const uint16_t* lm_head,
                 int* out_ids, cudaStream_t st = 0) {
        const int H = d.hidden, I = d.intermediate, BLK = d.block_size;
        const int M = k + 1;                       // bonus + k masks; never more than BLK
        std::vector<int> ids(M, d.mask_token_id);
        ids[0] = next_token;
        CUDA_CHECK(cudaMemcpyAsync(d_ids, ids.data(), M * 4, cudaMemcpyHostToDevice, st));

        set_base(dpos, P0, st);
        embed_rows(h, embed, d_ids, M, H, st);
        rope_tables(cosT, sinT, inv_freq, M, dpos, d.head_dim / 2, 1.0f, st);

        const float qscale = 1.0f / sqrtf((float)d.head_dim);
        const int G = d.n_heads / d.n_kv_heads;
        for (int l = 0; l < d.n_layers; ++l) {
            const auto& w = W.L[l];
            add_rms_cast(hb, h, nullptr, w.in_ln, M, H, (float)d.rms_eps, 0, st);
            // qkv_proj is already fused in the checkpoint; g_proj is separate and tiny, so it
            // rides along as a third segment off its own base pointer.
            float* outs[3] = {q, kk, vv};
            int    Ns[3]   = {qd(), kvd(), kvd()};
            gemm_bf16_seg(outs, Ns, 3, w.qkv, hb, M, H, st);
            gemm_bf16(gp, w.g, hb, M, d.n_heads, H, st);
            rmsnorm_heads(q,  q,  w.q_norm, M, d.n_heads,   d.head_dim, (float)d.rms_eps, st);
            rmsnorm_heads(kk, kk, w.k_norm, M, d.n_kv_heads, d.head_dim, (float)d.rms_eps, st);
            rope_apply(q,  cosT, sinT, M, d.n_heads,   d.head_dim, d.head_dim, st);
            rope_apply(kk, cosT, sinT, M, d.n_kv_heads, d.head_dim, d.head_dim, st);
            store_kv(Kc[l], Vc[l], kk, vv, kv_scale, kv_scale, M, d.n_kv_heads, d.head_dim,
                     cap, dpos, st);
            int nsp = attend_nsplit(M, d.n_kv_heads, d.sliding_window);
            if (nsp > MAXSPLIT) nsp = MAXSPLIT;
            // causal = true, so this is the target's kernel unchanged: query at position
            // P0+m attends to [P0+m-window+1, P0+m], which spans context and block alike.
            attend_split(att, pacc, pml, q, Kc[l], Vc[l], kv_scale, kv_scale, M,
                         d.n_kv_heads, G, cap, d.sliding_window, dpos, qscale, nsp, st);
            gate_softplus(att, gp, M, d.n_heads, d.head_dim, st);
            f32_to_bf16(attb, att, (long)M * qd(), st);
            gemm_bf16(hn, w.o, attb, M, H, qd(), st);

            add_rms_cast(hb, h, hn, w.post_ln, M, H, (float)d.rms_eps, 1, st);
            float* o2[2] = {mlp_a, mlp_b}; int n2[2] = {I, I};
            gemm_bf16_seg(o2, n2, 2, w.mlp_gate, hb, M, H, st);
            swiglu(mlp_a, mlp_a, mlp_b, (long)M * I, st);
            f32_to_bf16(attb, mlp_a, (long)M * I, st);
            gemm_bf16(hn, w.mlp_down, attb, M, H, I, st);
            add_inplace(h, hn, (long)M * H, st);
        }
        add_rms_cast(hb, h, nullptr, W.final_norm, M, H, (float)d.rms_eps, 0, st);
        // Only the k MASK slots produce draft tokens; slot 0 is the bonus token, already
        // known. Skipping it saves a full lm_head row-block per propose.
        gemm_bf16(logits, lm_head, hb + (size_t)H, k, d.vocab, H, st);
        CUDA_CHECK(cudaStreamSynchronize(st));

        std::vector<float> lg((size_t)k * d.vocab);
        CUDA_CHECK(cudaMemcpy(lg.data(), logits, (size_t)k * d.vocab * 4, cudaMemcpyDeviceToHost));
        for (int j = 0; j < k; ++j) {
            const float* row = lg.data() + (size_t)j * d.vocab;
            int b = 0; float bv = row[0];
            for (int i = 1; i < d.vocab; ++i) if (row[i] > bv) { bv = row[i]; b = i; }
            out_ids[j] = b;
        }
    }

};

} // namespace laguna
