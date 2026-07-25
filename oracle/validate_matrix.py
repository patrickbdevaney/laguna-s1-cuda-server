"""Stronger Gate A1 check: sweep configs that actually exercise the tricky paths --
sliding-window clipping, incremental KV decode, both layer types, several seeds."""
import os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny_ref
from tiny_ref import build
from ref_laguna import RefLaguna
from validate_ref import flatten_hf

def cfg_dict(o):
    c = json.loads(o.to_json_string())
    for k in ("num_attention_heads_per_layer","layer_types","mlp_only_layers",
              "moe_routed_scaling_factor","norm_topk_prob","moe_router_logit_softcapping",
              "num_experts_per_tok","num_experts","sliding_window","head_dim",
              "rope_parameters","rms_norm_eps","hidden_size","num_hidden_layers",
              "vocab_size","num_key_value_heads"):
        c[k] = getattr(o, k)
    return c

def run_case(name, seed, S, sw, nl, decode_steps=0):
    # patch the sliding window so it actually clips within S
    orig = tiny_ref.REAL["sliding_window"]; tiny_ref.REAL["sliding_window"] = sw
    try:
        cfg_o, m = build(seed=seed, nl=nl)
    finally:
        tiny_ref.REAL["sliding_window"] = orig
    cfg_o.sliding_window = sw
    for lay in m.model.layers:
        if lay.self_attn.is_sliding: lay.self_attn.sliding_window = sw
    cfg = cfg_dict(cfg_o); cfg["sliding_window"] = sw
    W, EX = flatten_hf(m, cfg)
    ref = RefLaguna(cfg, W, lambda L,e,w: EX[(L,e,w)], device="cpu")
    g = torch.Generator().manual_seed(seed*31+7)
    ids = torch.randint(0, cfg["vocab_size"], (1, S), generator=g)
    with torch.no_grad():
        hf = m(input_ids=ids, use_cache=False)
        mine, _ = ref.forward(ids, kv=None)
    d = (hf.logits[0].float()-mine[0].float()).abs().max().item()
    am = bool((hf.logits[0].argmax(-1)==mine[0].argmax(-1)).all())
    res = [f"{name:34s} S={S:3d} sw={sw:3d} nl={nl}  prefill maxabs={d:.2e} argmax={am}"]
    ok = d < 2e-4 and am

    if decode_steps:                      # incremental decode through our KV path
        kv = [None]*cfg["num_hidden_layers"]
        with torch.no_grad():
            ref.forward(ids, kv=kv)
            cur = ids.clone(); worst = 0.0; amok = True
            for t in range(decode_steps):
                nxt = torch.randint(0, cfg["vocab_size"], (1,1), generator=g)
                cur = torch.cat([cur, nxt], 1)
                l_inc, _ = ref.forward(nxt, kv=kv, pos_offset=cur.shape[1]-1)
                l_full = m(input_ids=cur, use_cache=False).logits[:, -1:]
                worst = max(worst, (l_full[0].float()-l_inc[0].float()).abs().max().item())
                amok &= bool((l_full[0].argmax(-1)==l_inc[0].argmax(-1)).all())
        res.append(f"{'':34s} decode x{decode_steps}  maxabs={worst:.2e} argmax={amok}")
        ok &= worst < 2e-4 and amok
    return ok, res

CASES = [
    ("baseline",                     0, 12, 512, 4, 0),
    ("SWA clips (sw=4 < S)",         1, 20,   4, 4, 0),
    ("SWA clips, 8 layers",          2, 24,   5, 8, 0),
    ("SWA clips, incremental KV",    3, 10,   4, 4, 6),
    ("no clip, incremental KV",      4,  8, 512, 4, 6),
    ("long seq, 8 layers",           5, 40,   7, 8, 0),
    ("seed variation",               6, 16,   3, 4, 4),
]
allok = True
for c in CASES:
    ok, lines = run_case(*c)
    allok &= ok
    for l in lines: print(("  OK  " if ok else " FAIL ") + l)
print("\n" + ("ALL PASS" if allok else "FAILURES PRESENT"))
sys.exit(0 if allok else 1)
