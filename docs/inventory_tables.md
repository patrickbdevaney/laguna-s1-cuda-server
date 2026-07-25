
### Target — `poolside/Laguna-S-2.1-NVFP4` (15 shards)

| tensor (N=layer, E=expert) | count | dtype | shape(s) | total |
|---|---:|---|---|---:|
| `model.layers.N.mlp.experts.e_score_correction_bias` | 47 | F32 | [256] | 0.05 MB |
| `model.layers.N.input_layernorm.weight` | 48 | BF16 | [3072] | 0.29 MB |
| `model.layers.N.mlp.down_proj.weight` | 1 | BF16 | [3072, 12288] | 75.50 MB |
| `model.layers.N.mlp.gate_proj.weight` | 1 | BF16 | [12288, 3072] | 75.50 MB |
| `model.layers.N.mlp.up_proj.weight` | 1 | BF16 | [12288, 3072] | 75.50 MB |
| `model.layers.N.post_attention_layernorm.weight` | 48 | BF16 | [3072] | 0.29 MB |
| `model.layers.N.self_attn.g_proj.weight` | 48 | BF16 | [48, 3072] / [72, 3072] | 19.46 MB |
| `model.layers.N.self_attn.k_norm.weight` | 48 | BF16 | [128] | 0.01 MB |
| `model.layers.N.self_attn.k_proj.weight` | 48 | BF16 | [1024, 3072] | 0.3020 GB |
| `model.layers.N.self_attn.k_scale` | 48 | BF16 | [1] | 0.00 MB |
| `model.layers.N.self_attn.o_proj.weight` | 48 | BF16 | [3072, 6144] / [3072, 9216] | 2.4914 GB |
| `model.layers.N.self_attn.q_norm.weight` | 48 | BF16 | [128] | 0.01 MB |
| `model.layers.N.self_attn.q_proj.weight` | 48 | BF16 | [6144, 3072] / [9216, 3072] | 2.4914 GB |
| `model.layers.N.self_attn.v_proj.weight` | 48 | BF16 | [1024, 3072] | 0.3020 GB |
| `model.layers.N.self_attn.v_scale` | 48 | BF16 | [1] | 0.00 MB |
| `model.layers.N.mlp.gate.weight` | 47 | BF16 | [256, 3072] | 73.92 MB |
| `model.layers.N.mlp.shared_expert.down_proj.weight` | 47 | BF16 | [3072, 1024] | 0.2957 GB |
| `model.layers.N.mlp.shared_expert.gate_proj.weight` | 47 | BF16 | [1024, 3072] | 0.2957 GB |
| `model.layers.N.mlp.shared_expert.up_proj.weight` | 47 | BF16 | [1024, 3072] | 0.2957 GB |
| `model.layers.N.mlp.experts.E.down_proj.input_global_scale` | 12032 | F32 | [] | 0.05 MB |
| `model.layers.N.mlp.experts.E.down_proj.weight_global_scale` | 12032 | F32 | [] | 0.05 MB |
| `model.layers.N.mlp.experts.E.gate_proj.input_global_scale` | 12032 | F32 | [] | 0.05 MB |
| `model.layers.N.mlp.experts.E.gate_proj.weight_global_scale` | 12032 | F32 | [] | 0.05 MB |
| `model.layers.N.mlp.experts.E.up_proj.input_global_scale` | 12032 | F32 | [] | 0.05 MB |
| `model.layers.N.mlp.experts.E.up_proj.weight_global_scale` | 12032 | F32 | [] | 0.05 MB |
| `lm_head.weight` | 1 | BF16 | [100352, 3072] | 0.6166 GB |
| `model.embed_tokens.weight` | 1 | BF16 | [100352, 3072] | 0.6166 GB |
| `model.norm.weight` | 1 | BF16 | [3072] | 0.01 MB |
| `model.layers.N.mlp.experts.E.down_proj.weight_scale` | 12032 | F8_E4M3 | [3072, 64] | 2.3656 GB |
| `model.layers.N.mlp.experts.E.gate_proj.weight_scale` | 12032 | F8_E4M3 | [1024, 192] | 2.3656 GB |
| `model.layers.N.mlp.experts.E.up_proj.weight_scale` | 12032 | F8_E4M3 | [1024, 192] | 2.3656 GB |
| `model.layers.N.mlp.experts.E.down_proj.weight_packed` | 12032 | U8 | [3072, 512] | 18.9247 GB |
| `model.layers.N.mlp.experts.E.gate_proj.weight_packed` | 12032 | U8 | [1024, 1536] | 18.9247 GB |
| `model.layers.N.mlp.experts.E.up_proj.weight_packed` | 12032 | U8 | [1024, 1536] | 18.9247 GB |

**total 71.8987 GB**, 145153 tensors

### Draft — `poolside/Laguna-S-2.1-DFlash-NVFP4` (1 shard)

| tensor (N=layer, E=expert) | count | dtype | shape(s) | total |
|---|---:|---|---|---:|
| `aux_hidden_norms.0.weight` | 1 | BF16 | [3072] | 0.01 MB |
| `aux_hidden_norms.1.weight` | 1 | BF16 | [3072] | 0.01 MB |
| `aux_hidden_norms.2.weight` | 1 | BF16 | [3072] | 0.01 MB |
| `aux_hidden_norms.3.weight` | 1 | BF16 | [3072] | 0.01 MB |
| `aux_hidden_norms.4.weight` | 1 | BF16 | [3072] | 0.01 MB |
| `aux_hidden_norms.5.weight` | 1 | BF16 | [3072] | 0.01 MB |
| `fc.weight` | 1 | BF16 | [3072, 18432] | 0.1132 GB |
| `hidden_norm.weight` | 1 | BF16 | [3072] | 0.01 MB |
| `layers.N.input_layernorm.weight` | 6 | BF16 | [3072] | 0.04 MB |
| `layers.N.mlp.down_proj.weight` | 6 | BF16 | [3072, 12288] | 0.4530 GB |
| `layers.N.mlp.gate_proj.weight` | 6 | BF16 | [12288, 3072] | 0.4530 GB |
| `layers.N.mlp.up_proj.weight` | 6 | BF16 | [12288, 3072] | 0.4530 GB |
| `layers.N.post_attention_layernorm.weight` | 6 | BF16 | [3072] | 0.04 MB |
| `layers.N.self_attn.g_proj.weight` | 6 | BF16 | [72, 3072] | 2.65 MB |
| `layers.N.self_attn.k_norm.weight` | 6 | BF16 | [128] | 0.00 MB |
| `layers.N.self_attn.o_proj.weight` | 6 | BF16 | [3072, 9216] | 0.3397 GB |
| `layers.N.self_attn.q_norm.weight` | 6 | BF16 | [128] | 0.00 MB |
| `layers.N.self_attn.qkv_proj.weight` | 6 | BF16 | [11264, 3072] | 0.4152 GB |
| `norm.weight` | 1 | BF16 | [3072] | 0.01 MB |

**total 2.2300 GB**, 69 tensors

### Per-layer attention shape map

| layers | type | heads | GQA group | q_proj | o_proj | g_proj | rotary |
|---|---|---:|---:|---|---|---|---|
| 0,4,…,44 (12) | full_attention | 48 | 6 | [6144, 3072] | [3072, 6144] | [48, 3072] | yarn θ=500000 partial=0.5 → 64/128 dims |
| 1,2,…,47 (36) | sliding_attention | 72 | 9 | [9216, 3072] | [3072, 9216] | [72, 3072] | default θ=10000 partial=1.0 → 128/128 dims |
