"""Dump reference tensors for the C++ kernel gates (G1..G9), computed from the REAL
checkpoint with the validated reference math in oracle/ref_laguna.py.

Binary format per file: raw little-endian, shapes recorded in refs.json.
"""
import os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_laguna import dequant_nvfp4, rms_norm, rope_tables, yarn_inv_freq
from safetensors import safe_open

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(ROOT, "models", "Laguna-S-2.1-NVFP4")
OUT = os.path.join(ROOT, "docs", "kernel_refs")
os.makedirs(OUT, exist_ok=True)
cfg = json.load(open(os.path.join(MD, "config.json")))
wmap = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
meta = {}

def raw(name):
    f = safe_open(os.path.join(MD, wmap[name]), framework="pt", device="cpu")
    return f.get_tensor(name)

def save(tag, t, note=""):
    t = t.contiguous()
    p = os.path.join(OUT, tag + ".bin")
    t.cpu().numpy().tofile(p)
    meta[tag] = {"shape": list(t.shape), "dtype": str(t.dtype), "note": note,
                 "bytes": os.path.getsize(p)}
    print(f"{tag:24s} {str(list(t.shape)):20s} {t.dtype} {os.path.getsize(p)/1e6:.2f} MB")

torch.manual_seed(1234)

# ---- G1: NVFP4 dequant of a real expert
P = "model.layers.1.mlp.experts.0.gate_proj."
pk, sc = raw(P + "weight_packed"), raw(P + "weight_scale")
gs = float(raw(P + "weight_global_scale"))
deq = dequant_nvfp4(pk, sc, gs)
save("g1_packed", pk, "L1 e0 gate_proj weight_packed [1024,1536] u8")
save("g1_scale", sc.view(torch.uint8), "weight_scale e4m3 as u8 [1024,192]")
save("g1_out", deq, "expected dequant [1024,3072] fp32")
meta["g1_inv_gs"] = 1.0 / gs
meta["g1_gs"] = gs

# ---- G2: BF16 dense linear against a real attention weight
Wq = raw("model.layers.1.self_attn.q_proj.weight")          # [9216,3072] bf16
M, K, N = 6, Wq.shape[1], 512                                # N truncated to keep the file small
Wsub = Wq[:N].contiguous()
x = (torch.randn(M, K) * 0.5).bfloat16()
ref = x.float() @ Wsub.float().T
save("g2_w", Wsub.view(torch.uint16), "L1 q_proj[:512] bf16-as-u16 [512,3072]")
save("g2_x", x.view(torch.uint16), "activations bf16-as-u16 [6,3072]")
save("g2_out", ref, "expected [6,512] fp32")

# ---- G2b: FP4 linear (the expert path) with the same activation shape
xk = (torch.randn(M, cfg["hidden_size"]) * 0.5).bfloat16()
ref4 = xk.float() @ deq.T                                     # [6,1024]
save("g2b_x", xk.view(torch.uint16), "activations bf16-as-u16 [6,3072]")
save("g2b_out", ref4, "expected [6,1024] fp32")

# ---- G3a: RMSNorm on a real hidden state
gold = torch.load(os.path.join(ROOT, "docs", "golden", "golden_primes.pt"),
                  weights_only=False)
h = gold["h"][0][0].float()                                   # [S,H] after layer 0
w_in = raw("model.layers.1.input_layernorm.weight")
save("g3a_x", h, "layer-0 output hidden [S,3072] fp32")
save("g3a_w", w_in.view(torch.uint16), "L1 input_layernorm bf16-as-u16 [3072]")
save("g3a_out", rms_norm(h, w_in.float(), cfg["rms_norm_eps"]), "expected [S,3072] fp32")

# ---- G3b: rope tables for both layer types, at real positions
S = h.shape[0]
pos = torch.arange(S)[None]
for tag, key in (("full", "full_attention"), ("slide", "sliding_attention")):
    rp = cfg["rope_parameters"][key]
    if rp["rope_type"] == "yarn":
        inv, _ = yarn_inv_freq(cfg["head_dim"], rp.get("partial_rotary_factor", 1.0),
                               rp["rope_theta"], rp["factor"],
                               rp["original_max_position_embeddings"],
                               rp["beta_fast"], rp["beta_slow"])
        sc_ = rp["attention_factor"]
    else:
        d = int(cfg["head_dim"] * rp.get("partial_rotary_factor", 1.0))
        inv = 1.0 / (rp["rope_theta"] ** (torch.arange(0, d, 2).float() / d)); sc_ = 1.0
    cos, sin = rope_tables(inv, pos, sc_, "cpu")
    save(f"g3b_cos_{tag}", cos[0], f"{key} cos [S,rot]")
    save(f"g3b_sin_{tag}", sin[0], f"{key} sin [S,rot]")

# ---- G4: router — sigmoid + selection-only bias + top-10 + sum-normalise
#
# PRECISION CONTRACT: the deployed model is bf16 (config torch_dtype), so the router sees a
# bf16 hidden state and accumulates in fp32. This oracle otherwise runs fp32-everywhere,
# which is *more* precise than the real model, not more correct. Feeding fp32 activations
# here flips 15/540 top-10 selections versus bf16 — and the flipped pairs differ in router
# score by as little as 1.03e-05 (rank-10 vs rank-11 gap), i.e. genuinely tied experts.
# The reference must therefore round the activation to bf16, matching what the kernel and
# the real model both do. See LOOP_LOG "G4".
hn = rms_norm(gold["h"][0][0].float(), raw("model.layers.1.post_attention_layernorm.weight").float(),
              cfg["rms_norm_eps"]).bfloat16().float()
rw = raw("model.layers.1.mlp.gate.weight")
bias = raw("model.layers.1.mlp.experts.e_score_correction_bias")
logits = hn @ rw.float().T
scores = torch.sigmoid(logits.float())
sel = torch.topk(scores + bias.float(), cfg["num_experts_per_tok"], dim=-1).indices
wts = scores.gather(-1, sel)
wts = wts / wts.sum(-1, keepdim=True)
save("g4_x", hn, "post-attn normed hidden [S,3072] fp32")
save("g4_w", rw.view(torch.uint16), "router bf16-as-u16 [256,3072]")
save("g4_bias", bias.float(), "e_score_correction_bias [256] fp32")
save("g4_sel", sel.to(torch.int32), "expected top-10 indices [S,10] i32")
save("g4_wts", wts, "expected normalised weights [S,10] fp32")

meta["config"] = {k: cfg[k] for k in ("hidden_size", "num_experts", "num_experts_per_tok",
                                      "rms_norm_eps", "head_dim", "sliding_window")}
meta["seq_len"] = int(S)
json.dump(meta, open(os.path.join(OUT, "refs.json"), "w"), indent=2)
print("\nwrote", OUT)

# ---- G5: attention (head-packed GQA over an FP8 KV cache), both layer types.
# The reference quantises K/V to e4m3 with the checkpoint's static scales first, so the gate
# isolates attention correctness from FP8 rounding (which is a separate, deliberate loss).
import torch.nn.functional as F
def fp8_rt(t, scale):
    return (t / scale).to(torch.float8_e4m3fn).float() * scale

for tag, LREF in (("full", 0), ("slide", 1)):
    nh = cfg["num_attention_heads_per_layer"][LREF]
    nkv, hd = cfg["num_key_value_heads"], cfg["head_dim"]
    G = nh // nkv
    win = cfg["sliding_window"] if cfg["layer_types"][LREF] == "sliding_attention" else 0
    Sm = 24                                       # queries; positions 0..Sm-1
    torch.manual_seed(99 + LREF)
    q = torch.randn(Sm, nh, hd) * 0.3
    k = torch.randn(Sm, nkv, hd) * 0.3
    v = torch.randn(Sm, nkv, hd) * 0.3
    ks = float(raw(f"model.layers.{LREF}.self_attn.k_scale"))
    vs = float(raw(f"model.layers.{LREF}.self_attn.v_scale"))
    kq, vq = fp8_rt(k, ks), fp8_rt(v, vs)
    # reference: repeat_kv then masked softmax, fp32
    kr = kq.permute(1, 0, 2).repeat_interleave(G, 0)      # [nh,S,hd]
    vr = vq.permute(1, 0, 2).repeat_interleave(G, 0)
    qq = q.permute(1, 0, 2)                                # [nh,S,hd]
    att = (qq @ kr.transpose(1, 2)) * (hd ** -0.5)
    pos = torch.arange(Sm)
    allowed = pos[None, :] <= pos[:, None]
    if win: allowed &= pos[None, :] > (pos[:, None] - win)
    att = att.masked_fill(~allowed[None], float("-inf"))
    o = (F.softmax(att, -1, dtype=torch.float32) @ vr).permute(1, 0, 2)   # [S,nh,hd]
    save(f"g5_q_{tag}", q.reshape(Sm, nh * hd), f"queries [S,{nh}*{hd}] fp32")
    save(f"g5_k_{tag}", k.reshape(Sm, nkv * hd), "keys pre-quant fp32")
    save(f"g5_v_{tag}", v.reshape(Sm, nkv * hd), "values pre-quant fp32")
    save(f"g5_out_{tag}", o.reshape(Sm, nh * hd), "expected attention out fp32")
    meta[f"g5_{tag}"] = {"nh": nh, "nkv": nkv, "hd": hd, "G": G, "window": win,
                         "S": Sm, "k_scale": ks, "v_scale": vs}
json.dump(meta, open(os.path.join(OUT, "refs.json"), "w"), indent=2)
print("appended G5 attention refs")

# ---- G6: routed-expert MoE for a real layer, real hidden state, real experts.
#
# PRECISION: matched to the kernel's W4A16 choices, which are deliberately *more* precise
# than torch-bf16: activations enter each GEMM as bf16, accumulation is fp32, and the only
# intermediate rounding is h = silu(gate)*up -> bf16 before down_proj (because down_proj's
# activation operand must be bf16 too). torch's bf16 path would additionally round gate and
# up. The end-to-end greedy gate (G8) is what checks that this choice costs nothing.
L = 1
E, TK = cfg["num_experts"], cfg["num_experts_per_tok"]
MI, Hh = cfg["moe_intermediate_size"], cfg["hidden_size"]
xin = hn[:8].contiguous()                       # 8 tokens is the verify-shape regime
sel8 = sel[:8].contiguous(); w8 = wts[:8].contiguous()
xb = xin.bfloat16().float()

def expert_w(e, which):
    p = f"model.layers.{L}.mlp.experts.{e}.{which}."
    return dequant_nvfp4(raw(p + "weight_packed"), raw(p + "weight_scale"),
                         float(raw(p + "weight_global_scale")))

out = torch.zeros(xin.shape[0], Hh)
for t in range(xin.shape[0]):
    # accumulate in ASCENDING EXPERT INDEX, matching modeling_laguna's expert_hit iteration
    order = sorted(range(TK), key=lambda j: int(sel8[t, j]))
    for j in order:
        e = int(sel8[t, j])
        g = xb[t] @ expert_w(e, "gate_proj").T
        u = xb[t] @ expert_w(e, "up_proj").T
        h = (torch.nn.functional.silu(g) * u).bfloat16().float()
        out[t] += (h @ expert_w(e, "down_proj").T) * float(w8[t, j])
out *= cfg["moe_routed_scaling_factor"]
save("g6_x", xb, "bf16-rounded hidden [8,3072] fp32")
save("g6_sel", sel8.to(torch.int32), "top-10 expert ids [8,10] i32")
save("g6_wts", w8, "normalised routing weights [8,10] fp32")
save("g6_out", out, "expected routed-expert output (x2.5 scaling) [8,3072] fp32")
meta["g6"] = {"layer": L, "rows": int(xin.shape[0]), "E": E, "topk": TK, "MI": MI,
              "scaling": cfg["moe_routed_scaling_factor"]}
json.dump(meta, open(os.path.join(OUT, "refs.json"), "w"), indent=2)
print("appended G6 MoE refs")
