"""Gate A1 math check: our transcription (ref_laguna.RefLaguna) vs the SHIPPED
modeling_laguna.py, at tiny scale with identical random weights.

If this passes, RefLaguna's math is the reference's math, and we can trust the
golden tensors it produces on the real 71.9 GB checkpoint.
"""
import os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tiny_ref import build, _MD                      # noqa: E402
from ref_laguna import RefLaguna, yarn_inv_freq      # noqa: E402

torch.set_printoptions(precision=8)


def flatten_hf(m, cfg):
    """HF module tree -> the flat checkpoint-style names RefLaguna expects."""
    W, experts = {}, {}
    sd = dict(m.state_dict())
    for k, v in sd.items():
        W[k] = v.float()
    # fused expert tensors -> per-expert accessor
    for L, layer in enumerate(m.model.layers):
        if not hasattr(layer.mlp, "experts"):
            continue
        gu = layer.mlp.experts.gate_up_proj.float()   # [E, 2I, H]
        dn = layer.mlp.experts.down_proj.float()      # [E, H, I]
        I = gu.shape[1] // 2
        for e in range(gu.shape[0]):
            experts[(L, e, "gate_proj")] = gu[e, :I]
            experts[(L, e, "up_proj")] = gu[e, I:]
            experts[(L, e, "down_proj")] = dn[e]
        # router bias lives on the router module in the reference; expose it under the
        # checkpoint name RefLaguna looks up
        W[f"model.layers.{L}.mlp.experts.e_score_correction_bias"] = \
            layer.mlp.gate.e_score_correction_bias.float()
        W[f"model.layers.{L}.mlp.gate.weight"] = layer.mlp.gate.weight.float()
    return W, experts


def main():
    cfg_obj, m = build(seed=0)
    cfg = json.loads(cfg_obj.to_json_string())
    # to_json_string may omit fields that equal the class default; RefLaguna needs them
    for k, d in (("num_attention_heads_per_layer", cfg_obj.num_attention_heads_per_layer),
                 ("layer_types", cfg_obj.layer_types),
                 ("mlp_only_layers", cfg_obj.mlp_only_layers),
                 ("moe_routed_scaling_factor", cfg_obj.moe_routed_scaling_factor),
                 ("norm_topk_prob", cfg_obj.norm_topk_prob),
                 ("moe_router_logit_softcapping", cfg_obj.moe_router_logit_softcapping),
                 ("num_experts_per_tok", cfg_obj.num_experts_per_tok),
                 ("num_experts", cfg_obj.num_experts),
                 ("sliding_window", cfg_obj.sliding_window),
                 ("head_dim", cfg_obj.head_dim),
                 ("rope_parameters", cfg_obj.rope_parameters)):
        cfg[k] = d

    W, EX = flatten_hf(m, cfg)
    ref = RefLaguna(cfg, W, lambda L, e, which: EX[(L, e, which)], device="cpu")

    ids = torch.randint(0, cfg["vocab_size"], (1, 12), generator=torch.Generator().manual_seed(7))

    with torch.no_grad():
        hf = m(input_ids=ids, use_cache=False, output_hidden_states=True)
        mine, _ = ref.forward(ids, kv=None)

    ok = True

    # --- rope: our yarn transcription vs the shipped rotary module
    inv_hf = m.model.rotary_emb.inv_freq.float()
    sc_hf = float(m.model.rotary_emb.attention_scaling)
    fa = cfg["rope_parameters"]["full_attention"]
    inv_mine, sc_mine = yarn_inv_freq(cfg["head_dim"], fa.get("partial_rotary_factor", 1.0),
                                      fa["rope_theta"], fa["factor"],
                                      fa["original_max_position_embeddings"],
                                      fa["beta_fast"], fa["beta_slow"])
    d_inv = (inv_hf - inv_mine).abs().max().item()
    print(f"yarn inv_freq   maxabs diff = {d_inv:.3e}   (n={inv_hf.numel()})")
    print(f"yarn attn_scale hf={sc_hf!r} mine(formula)={sc_mine!r} "
          f"config={fa.get('attention_factor')!r}")
    ok &= d_inv < 1e-6

    # --- per-layer hidden states
    print(f"\n{'layer':>6s}{'maxabs':>14s}{'rel':>12s}")
    cap = {}
    with torch.no_grad():
        ref.forward(ids, kv=None, capture=cap)
    # NOTE: transformers records NL+1 entries as [embed, out_0 .. out_{NL-2}, post_norm(out_{NL-1})]
    # -- the last layer's raw output is never recorded, the final RMSNorm result is.
    NL = cfg["num_hidden_layers"]
    a = hf.hidden_states[0][0].float(); b = cap["h_embed"][0].float()
    print(f"{'embed':>6s}{(a-b).abs().max().item():14.3e}"
          f"{(a-b).abs().max().item()/max(a.abs().max().item(),1e-9):12.3e}")
    ok &= torch.allclose(a, b, atol=1e-6)
    for L in range(NL - 1):
        a = hf.hidden_states[L + 1][0].float()
        b = cap["h"][L][0].float()
        mx = (a - b).abs().max().item()
        rel = mx / max(a.abs().max().item(), 1e-9)
        flag = "" if rel < 2e-5 else "   <-- MISMATCH"
        print(f"{L:6d}{mx:14.3e}{rel:12.3e}{flag}")
        ok &= rel < 2e-5
    a = hf.hidden_states[NL][0].float(); b = cap["h_final"][0].float()
    mx = (a - b).abs().max().item(); rel = mx / max(a.abs().max().item(), 1e-9)
    print(f"{'final':>6s}{mx:14.3e}{rel:12.3e}" + ("" if rel < 2e-5 else "   <-- MISMATCH"))
    ok &= rel < 2e-5

    # --- logits
    a, b = hf.logits[0].float(), mine[0].float()
    mx = (a - b).abs().max().item()
    rel = mx / max(a.abs().max().item(), 1e-9)
    print(f"\nlogits          maxabs = {mx:.3e}   rel = {rel:.3e}")
    print(f"argmax match    {(a.argmax(-1) == b.argmax(-1)).all().item()}")
    ok &= rel < 2e-5
    ok &= bool((a.argmax(-1) == b.argmax(-1)).all())

    print("\n" + ("PASS - RefLaguna reproduces the shipped reference"
                  if ok else "FAIL - transcription diverges"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
