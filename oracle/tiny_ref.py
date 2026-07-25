"""Instantiate the SHIPPED modeling_laguna.py at tiny scale with random weights.
This is the math oracle: our CUDA kernels and our own reference transcription are
both validated against this, without needing the 71.9 GB checkpoint."""
import sys, json, torch, os
_MD = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'Laguna-S-2.1-NVFP4')
# Load the shipped custom code as a real package so its relative imports resolve.
import importlib.util, types
_pkg = types.ModuleType('laguna_ref'); _pkg.__path__ = [_MD]; sys.modules['laguna_ref'] = _pkg
def _load(name):
    spec = importlib.util.spec_from_file_location(f'laguna_ref.{name}', os.path.join(_MD, name + '.py'))
    mod = importlib.util.module_from_spec(spec); sys.modules[f'laguna_ref.{name}'] = mod
    spec.loader.exec_module(mod); return mod
LagunaConfig = _load('configuration_laguna').LagunaConfig
LagunaForCausalLM = _load('modeling_laguna').LagunaForCausalLM

REAL = json.load(open(os.path.join(_MD, 'config.json')))

def tiny_config(nl=4, hidden=64, heads_full=4, heads_slide=6, nkv=2, hd=16,
                nexp=8, topk=3, moe_int=16, vocab=128, inter=32):
    """Same STRUCTURE as the real config (3:1 layer pattern, per-layer heads,
    dense layer 0, two rope tables, per-head gating), tiny DIMENSIONS."""
    layer_types, heads = [], []
    for i in range(nl):
        if i % 4 == 0: layer_types.append("full_attention");     heads.append(heads_full)
        else:          layer_types.append("sliding_attention");  heads.append(heads_slide)
    rp = json.loads(json.dumps(REAL["rope_parameters"]))       # keep yarn/partial exactly
    rp["full_attention"]["original_max_position_embeddings"] = 64
    return LagunaConfig(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=inter,
        num_hidden_layers=nl, num_attention_heads=heads_full,
        num_key_value_heads=nkv, head_dim=hd,
        max_position_embeddings=256, rms_norm_eps=REAL["rms_norm_eps"],
        num_experts=nexp, num_experts_per_tok=topk,
        moe_intermediate_size=moe_int, shared_expert_intermediate_size=moe_int,
        norm_topk_prob=REAL["norm_topk_prob"],
        moe_routed_scaling_factor=REAL["moe_routed_scaling_factor"],
        moe_router_logit_softcapping=REAL["moe_router_logit_softcapping"],
        moe_apply_router_weight_on_input=REAL["moe_apply_router_weight_on_input"],
        decoder_sparse_step=REAL["decoder_sparse_step"], mlp_only_layers=REAL["mlp_only_layers"],
        gating=REAL["gating"], sliding_window=REAL["sliding_window"],
        layer_types=layer_types, num_attention_heads_per_layer=heads,
        rope_parameters=rp, tie_word_embeddings=False,
        attention_bias=False, hidden_act="silu", dtype=torch.float32,
        _attn_implementation="eager",
    )

def build(seed=0, **kw):
    torch.manual_seed(seed)
    cfg = tiny_config(**kw)
    m = LagunaForCausalLM(cfg).to(torch.float32).eval()
    # randomise the aux-loss-free routing bias so the selection-vs-weight split is exercised
    with torch.no_grad():
        for layer in m.model.layers:
            if hasattr(layer.mlp, "gate"):
                layer.mlp.gate.e_score_correction_bias.normal_(0, 0.5)
    return cfg, m

if __name__ == "__main__":
    cfg, m = build()
    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        out = m(input_ids=ids, use_cache=False, output_hidden_states=True)
    print("OK  logits", tuple(out.logits.shape), "hidden_states", len(out.hidden_states))
    print("layer_types", cfg.layer_types)
    print("heads/layer", cfg.num_attention_heads_per_layer)
    print("logit[0,0,:6]", out.logits[0, 0, :6].tolist())
    print("params", sum(p.numel() for p in m.parameters()))
