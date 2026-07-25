// laguna_config.h — the ONLY source of Laguna architectural constants.
//
// Project invariant (DIRECTIVE.md §0): never invent a model constant. Every field below
// is read from the checkpoint's config.json at load time. Nothing here has a numeric
// default that could silently stand in for a missing field — a missing required key is
// a hard error, because a wrong constant on a 117B model is undetectable until it is
// expensive.
#pragma once
#include <string>
#include <vector>
#include <set>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include "third_party/json.hpp"

namespace laguna {

enum class LayerType { Full, Sliding };

struct RopeSpec {
    std::string type;            // "yarn" | "default"
    double theta = 0.0;
    double factor = 1.0;
    int    orig_max_pos = 0;
    double beta_fast = 0.0, beta_slow = 0.0;
    double attention_factor = 1.0;   // multiplies cos/sin
    double partial = 1.0;            // fraction of head_dim that rotates
    int    rotary_dim = 0;           // = head_dim * partial

    // transformers' _compute_yarn_parameters, transcribed. Validated bit-exact against
    // the shipped rotary module by oracle/validate_ref.py.
    std::vector<float> inv_freq(int head_dim) const {
        int dim = (int)std::llround(head_dim * partial);
        int n = dim / 2;
        std::vector<float> inv(n);
        if (type != "yarn") {
            for (int i = 0; i < n; ++i)
                inv[i] = (float)(1.0 / std::pow(theta, (double)(2 * i) / dim));
            return inv;
        }
        auto find_dim = [&](double nrot) {
            return (dim * std::log(orig_max_pos / (nrot * 2.0 * M_PI))) / (2.0 * std::log(theta));
        };
        double low = std::floor(find_dim(beta_fast));
        double high = std::ceil(find_dim(beta_slow));
        low = std::max(low, 0.0); high = std::min(high, (double)dim - 1.0);
        if (low == high) high += 0.001;
        for (int i = 0; i < n; ++i) {
            double pos_freq = std::pow(theta, (double)(2 * i) / dim);
            double extrap = 1.0 / pos_freq;
            double interp = 1.0 / (factor * pos_freq);
            double ramp = ((double)i - low) / (high - low);
            ramp = ramp < 0 ? 0 : (ramp > 1 ? 1 : ramp);
            double mask = 1.0 - ramp;                   // 1 => extrapolate (high freq)
            inv[i] = (float)(interp * (1.0 - mask) + extrap * mask);
        }
        return inv;
    }
};

struct Config {
    // --- core dims
    int hidden = 0, n_layers = 0, head_dim = 0, n_kv_heads = 0, vocab = 0;
    int intermediate = 0;                 // dense-MLP layers only
    double rms_eps = 0.0;
    bool tie_word_embeddings = false;

    // --- per-layer
    std::vector<int> heads;               // num_attention_heads_per_layer
    std::vector<LayerType> layer_type;
    std::set<int> dense_layers;           // mlp_only_layers
    int sliding_window = 0;

    // --- MoE
    int n_experts = 0, top_k = 0, moe_intermediate = 0, shared_intermediate = 0;
    bool norm_topk_prob = true;
    double routed_scaling = 1.0;
    double router_softcap = 0.0;
    bool apply_router_weight_on_input = false;

    // --- quantization
    int  nvfp4_group = 0;                 // weight_scale group size
    bool kv_fp8 = false;

    // --- rope, one per layer type
    RopeSpec rope_full, rope_slide;

    // --- tokens
    int bos = -1, pad = -1, mask_token = -1;
    std::vector<int> eos;

    // --- derived
    int n_global() const { int c = 0; for (auto t : layer_type) c += (t == LayerType::Full); return c; }
    int n_sliding() const { return n_layers - n_global(); }
    bool is_sliding(int L) const { return layer_type[L] == LayerType::Sliding; }
    bool is_dense(int L) const { return dense_layers.count(L) > 0; }
    int  kv_group(int L) const { return heads[L] / n_kv_heads; }
    int  q_dim(int L) const { return heads[L] * head_dim; }
    const RopeSpec& rope(int L) const { return is_sliding(L) ? rope_slide : rope_full; }

    // bytes of FP8 KV per token per layer (K and V)
    long kv_bytes_per_token_per_layer() const { return 2L * n_kv_heads * head_dim; }
};

namespace detail {
inline const nlohmann::json& req(const nlohmann::json& j, const char* k) {
    auto it = j.find(k);
    if (it == j.end())
        throw std::runtime_error(std::string("config.json: missing required key '") + k + "'");
    return *it;
}
inline RopeSpec parse_rope(const nlohmann::json& r, int head_dim) {
    RopeSpec s;
    s.type = req(r, "rope_type").get<std::string>();
    s.theta = req(r, "rope_theta").get<double>();
    s.partial = r.value("partial_rotary_factor", 1.0);
    s.rotary_dim = (int)std::llround(head_dim * s.partial);
    if (s.type == "yarn") {
        s.factor = req(r, "factor").get<double>();
        s.orig_max_pos = req(r, "original_max_position_embeddings").get<int>();
        s.beta_fast = req(r, "beta_fast").get<double>();
        s.beta_slow = req(r, "beta_slow").get<double>();
        // attention_factor ships explicitly; the yarn default is 0.1*ln(factor)+1
        s.attention_factor = r.value("attention_factor", 0.1 * std::log(s.factor) + 1.0);
    }
    return s;
}
} // namespace detail

inline Config load_config(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    nlohmann::json j; f >> j;
    using detail::req;
    Config c;

    c.hidden       = req(j, "hidden_size").get<int>();
    c.n_layers     = req(j, "num_hidden_layers").get<int>();
    c.head_dim     = req(j, "head_dim").get<int>();
    c.n_kv_heads   = req(j, "num_key_value_heads").get<int>();
    c.vocab        = req(j, "vocab_size").get<int>();
    c.intermediate = req(j, "intermediate_size").get<int>();
    c.rms_eps      = req(j, "rms_norm_eps").get<double>();
    c.tie_word_embeddings = req(j, "tie_word_embeddings").get<bool>();
    c.sliding_window      = req(j, "sliding_window").get<int>();

    // per-layer head counts: required here because Laguna is NOT uniform (48 vs 72)
    for (auto& v : req(j, "num_attention_heads_per_layer")) c.heads.push_back(v.get<int>());
    for (auto& v : req(j, "layer_types"))
        c.layer_type.push_back(v.get<std::string>() == "sliding_attention"
                               ? LayerType::Sliding : LayerType::Full);
    for (auto& v : req(j, "mlp_only_layers")) c.dense_layers.insert(v.get<int>());
    if ((int)c.heads.size() != c.n_layers || (int)c.layer_type.size() != c.n_layers)
        throw std::runtime_error("config.json: per-layer list length != num_hidden_layers");

    c.n_experts        = req(j, "num_experts").get<int>();
    c.top_k            = req(j, "num_experts_per_tok").get<int>();
    c.moe_intermediate = req(j, "moe_intermediate_size").get<int>();
    c.shared_intermediate = req(j, "shared_expert_intermediate_size").get<int>();
    c.norm_topk_prob   = req(j, "norm_topk_prob").get<bool>();
    c.routed_scaling   = req(j, "moe_routed_scaling_factor").get<double>();
    c.router_softcap   = j.value("moe_router_logit_softcapping", 0.0);
    c.apply_router_weight_on_input = j.value("moe_apply_router_weight_on_input", false);
    if (c.apply_router_weight_on_input)
        throw std::runtime_error("moe_apply_router_weight_on_input=true is unsupported "
                                 "(the reference implementation rejects it too)");

    const auto& rp = req(j, "rope_parameters");
    c.rope_full  = detail::parse_rope(req(rp, "full_attention"), c.head_dim);
    c.rope_slide = detail::parse_rope(req(rp, "sliding_attention"), c.head_dim);

    // quantization: group size is READ from the weight_scale geometry by the loader and
    // cross-checked against this field; kv scheme tells us FP8 KV is expected.
    const auto& q = req(j, "quantization_config");
    const auto& g0 = req(req(q, "config_groups"), "group_0");
    c.nvfp4_group = req(req(g0, "weights"), "group_size").get<int>();
    auto kvs = q.find("kv_cache_scheme");
    c.kv_fp8 = (kvs != q.end() && !kvs->is_null() && (*kvs).value("num_bits", 0) == 8);

    c.bos = j.value("bos_token_id", -1);
    c.pad = j.value("pad_token_id", -1);
    if (j.contains("eos_token_id")) {
        const auto& e = j["eos_token_id"];
        if (e.is_array()) for (auto& v : e) c.eos.push_back(v.get<int>());
        else c.eos.push_back(e.get<int>());
    }
    return c;
}

// The DFlash draft's config.json is a different, smaller schema.
struct DraftConfig {
    int hidden = 0, n_layers = 0, head_dim = 0, n_heads = 0, n_kv_heads = 0;
    int intermediate = 0, vocab = 0, sliding_window = 0, block_size = 0, mask_token_id = -1;
    double rms_eps = 0.0, rope_theta = 0.0;
    bool causal = true;
    std::vector<int> target_layer_ids;      // taps into the TARGET model
};

inline DraftConfig load_draft_config(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    nlohmann::json j; f >> j;
    using detail::req;
    DraftConfig d;
    d.hidden       = req(j, "hidden_size").get<int>();
    d.n_layers     = req(j, "num_hidden_layers").get<int>();
    d.head_dim     = req(j, "head_dim").get<int>();
    d.n_heads      = req(j, "num_attention_heads").get<int>();
    d.n_kv_heads   = req(j, "num_key_value_heads").get<int>();
    d.intermediate = req(j, "intermediate_size").get<int>();
    d.vocab        = req(j, "vocab_size").get<int>();
    d.sliding_window = req(j, "sliding_window").get<int>();
    d.rms_eps      = req(j, "rms_norm_eps").get<double>();
    d.rope_theta   = req(j, "rope_theta").get<double>();
    const auto& dc = req(j, "dflash_config");
    d.block_size    = req(dc, "block_size").get<int>();
    d.mask_token_id = req(dc, "mask_token_id").get<int>();
    d.causal        = dc.value("causal", true);
    for (auto& v : req(dc, "target_layer_ids")) d.target_layer_ids.push_back(v.get<int>());
    return d;
}

} // namespace laguna
