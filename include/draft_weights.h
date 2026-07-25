// draft_weights.h — loader for the DFlash draft checkpoint.
//
// 2.230 GB, one shard, entirely BF16 — "the draft stays BF16" is a standing invariant of the
// directive, and it is also the cheap choice: the draft's whole weight set is 2.2 GB against
// the target's 69, so quantizing it would buy ~3 % of a decode step and put the acceptance
// rate at risk, which is the one number speculation actually lives on.
//
// Shapes, read from the checkpoint (never assumed):
//   fc.weight                     [3072, 18432]   fuses six 3072-wide target taps
//   aux_hidden_norms.{0..5}       [3072]          one RMSNorm weight per tap
//   hidden_norm.weight            [3072]          on the fused vector, context branch only
//   norm.weight                   [3072]          final norm
//   layers.N.self_attn.qkv_proj   [11264, 3072]   = 9216 q + 1024 k + 1024 v, FUSED
//   layers.N.self_attn.g_proj     [72, 3072]      per-head softplus gate (Laguna-style)
//   layers.N.self_attn.o_proj     [3072, 9216]
//   layers.N.self_attn.{q,k}_norm [128]           QK-RMSNorm, pre-RoPE
//   layers.N.mlp.{gate,up}_proj   [12288, 3072]   dense SiLU MLP, no MoE, no router
//   layers.N.mlp.down_proj        [3072, 12288]
//   layers.N.{input,post_attention}_layernorm [3072]
// There is no embed_tokens and no lm_head: both are SHARED with the target.
#pragma once
#include <string>
#include <vector>
#include "laguna_config.h"
#include "laguna_weights.h"   // CUDA_CHECK, wall_now
#include "safetensors.h"

namespace laguna {

struct DraftLayerW {
    const uint16_t *in_ln = nullptr, *post_ln = nullptr;
    const uint16_t *qkv = nullptr, *g = nullptr, *o = nullptr;
    const uint16_t *q_norm = nullptr, *k_norm = nullptr;
    const uint16_t *mlp_gate = nullptr, *mlp_up = nullptr, *mlp_down = nullptr;
};

struct DraftWeights {
    void* arena = nullptr;
    size_t arena_bytes = 0;
    const uint16_t *fc = nullptr, *hidden_norm = nullptr, *final_norm = nullptr;
    std::vector<const uint16_t*> aux_norm;      // one per tap
    std::vector<DraftLayerW> L;
    double load_seconds = 0;
};

class DraftLoader {
public:
    DraftLoader(std::string dir, DraftConfig cfg) : dir_(std::move(dir)), cfg_(cfg) {}

    DraftWeights load(bool verbose = true) {
        double t0 = wall_now();
        const auto& d = cfg_;
        const size_t H = d.hidden, I = d.intermediate, hd = d.head_dim;
        const size_t qd = (size_t)d.n_heads * hd, kvd = (size_t)d.n_kv_heads * hd;
        const size_t ntap = d.target_layer_ids.size();

        W_.L.resize(d.n_layers);
        W_.aux_norm.resize(ntap);
        reserve1(W_.fc, H * ntap * H * 2);
        reserve1(W_.hidden_norm, H * 2);
        reserve1(W_.final_norm, H * 2);
        for (size_t i = 0; i < ntap; ++i) reserve1(W_.aux_norm[i], H * 2);
        for (int l = 0; l < d.n_layers; ++l) {
            auto& w = W_.L[l];
            reserve1(w.in_ln, H * 2);   reserve1(w.post_ln, H * 2);
            reserve1(w.qkv, (qd + 2 * kvd) * H * 2);
            reserve1(w.g, (size_t)d.n_heads * H * 2);
            reserve1(w.o, H * qd * 2);
            reserve1(w.q_norm, hd * 2); reserve1(w.k_norm, hd * 2);
            // gate|up contiguous so the dense MLP is one segmented GEMM, as in the target
            reserve1(w.mlp_gate, I * H * 2); reserve1(w.mlp_up, I * H * 2);
            reserve1(w.mlp_down, H * I * 2);
        }

        CUDA_CHECK(cudaMalloc(&W_.arena, arena_bytes_));
        W_.arena_bytes = arena_bytes_;
        for (auto& f : fixups_) *f.slot = (const char*)W_.arena + f.off;

        ::st::SafeTensors sf(dir_ + "/model.safetensors");
        put(sf, "fc.weight", W_.fc, H * ntap * H * 2);
        put(sf, "hidden_norm.weight", W_.hidden_norm, H * 2);
        put(sf, "norm.weight", W_.final_norm, H * 2);
        for (size_t i = 0; i < ntap; ++i)
            put(sf, "aux_hidden_norms." + std::to_string(i) + ".weight", W_.aux_norm[i], H * 2);
        for (int l = 0; l < d.n_layers; ++l) {
            auto& w = W_.L[l];
            std::string p = "layers." + std::to_string(l) + ".";
            put(sf, p + "input_layernorm.weight", w.in_ln, H * 2);
            put(sf, p + "post_attention_layernorm.weight", w.post_ln, H * 2);
            put(sf, p + "self_attn.qkv_proj.weight", w.qkv, (qd + 2 * kvd) * H * 2);
            put(sf, p + "self_attn.g_proj.weight", w.g, (size_t)d.n_heads * H * 2);
            put(sf, p + "self_attn.o_proj.weight", w.o, H * qd * 2);
            put(sf, p + "self_attn.q_norm.weight", w.q_norm, hd * 2);
            put(sf, p + "self_attn.k_norm.weight", w.k_norm, hd * 2);
            put(sf, p + "mlp.gate_proj.weight", w.mlp_gate, I * H * 2);
            put(sf, p + "mlp.up_proj.weight",   w.mlp_up,   I * H * 2);
            put(sf, p + "mlp.down_proj.weight", w.mlp_down, H * I * 2);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        W_.load_seconds = wall_now() - t0;
        if (verbose)
            printf("[draft] arena %.3f GB, %zu tensors, %.2f s\n",
                   arena_bytes_ / 1e9, fixups_.size(), W_.load_seconds);
        return W_;
    }

private:
    struct Fix { const void** slot; size_t off; };
    template <class T> void reserve1(const T*& field, size_t bytes) {
        bytes = (bytes + 255) & ~(size_t)255;
        fixups_.push_back(Fix{(const void**)&field, arena_bytes_});
        arena_bytes_ += bytes;
    }

    // Every size is checked against what the CONFIG implies, not against the file: a shape
    // that disagrees is a wrong assumption about the architecture, and it must be a hard
    // error rather than a silently truncated copy.
    void put(::st::SafeTensors& sf, const std::string& name, const uint16_t* dst, size_t bytes) {
        if (!sf.has(name)) throw std::runtime_error("draft: missing tensor " + name);
        const auto& v = sf.get(name);
        if (v.dtype != "BF16")
            throw std::runtime_error("draft: " + name + " is " + v.dtype + ", expected BF16");
        if (v.nbytes != bytes)
            throw std::runtime_error("draft: " + name + " is " + std::to_string(v.nbytes) +
                                     " B, config implies " + std::to_string(bytes));
        CUDA_CHECK(cudaMemcpy((void*)dst, v.data, bytes, cudaMemcpyHostToDevice));
    }

    std::string dir_;
    DraftConfig cfg_;
    DraftWeights W_;
    std::vector<Fix> fixups_;
    size_t arena_bytes_ = 0;
};

} // namespace laguna
