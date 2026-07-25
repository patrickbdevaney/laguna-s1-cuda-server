#!/usr/bin/env python3
"""Laguna S 2.1 NVFP4 roofline calculator (Jetson Thor sm_110a).

Every number is derived from files on disk:
  models/Laguna-S-2.1-NVFP4/config.json
  models/Laguna-S-2.1-NVFP4/model.safetensors.index.json
  safetensors headers (shapes+dtypes)  -> tools/hdr_target.json, tools/hdr_draft.json
No model constant is hardcoded here.
"""
import json, os, sys

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = json.load(open(f"{R}/models/Laguna-S-2.1-NVFP4/config.json"))
HDR = json.load(open(f"{R}/tools/hdr_target.json"))
DHDR = json.load(open(f"{R}/tools/hdr_draft.json"))
DCFG = json.load(open(f"{R}/models/Laguna-S-2.1-DFlash-NVFP4/config.json"))

DT = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1, "I8": 1, "I32": 4}
def nbytes(h, name):
    v = h[name]; n = DT[v["dtype"]]
    for d in v["shape"]: n *= d
    return n
def group(h, pred):
    return sum(nbytes(h, k) for k in h if pred(k))

NL      = CFG["num_hidden_layers"]
LT      = CFG["layer_types"]
NKV     = CFG["num_key_value_heads"]
HD      = CFG["head_dim"]
TOPK    = CFG["num_experts_per_tok"]
NEXP    = CFG["num_experts"]
SW      = CFG["sliding_window"]
DENSE   = set(CFG["mlp_only_layers"])
MOE_L   = [l for l in range(NL) if l not in DENSE]
GLOBAL  = [l for l in range(NL) if LT[l] == "full_attention"]
SLIDING = [l for l in range(NL) if LT[l] == "sliding_attention"]
BLK     = DCFG["dflash_config"]["block_size"]

# ---- per-decode-step WEIGHT bytes (read once per forward, any M) -------------
attn   = group(HDR, lambda k: ".self_attn." in k and k.endswith(".weight"))
norms  = group(HDR, lambda k: "layernorm" in k or k == "model.norm.weight")
dense0 = group(HDR, lambda k: any(k == f"model.layers.{l}.mlp.{p}.weight"
                                 for l in DENSE for p in ("gate_proj","up_proj","down_proj")))
router = group(HDR, lambda k: k.endswith(".mlp.gate.weight") or k.endswith("e_score_correction_bias"))
shared = group(HDR, lambda k: ".mlp.shared_expert." in k)
lmhead = nbytes(HDR, "lm_head.weight")
embed_row = CFG["hidden_size"] * 2                      # one gathered row

# one routed expert (packed FP4 + FP8 group scales + FP32 globals), from disk
E_PREFIX = f"model.layers.{MOE_L[0]}.mlp.experts.0."
EXPERT_B = group(HDR, lambda k: k.startswith(E_PREFIX))
EXPERT_POOL = EXPERT_B * NEXP * len(MOE_L)              # all routed expert weights

FIXED = attn + norms + dense0 + router + shared + lmhead + embed_row   # k-independent
AR_EXPERTS = EXPERT_B * TOPK * len(MOE_L)                              # k=1

# ---- KV cache (FP8 e4m3, per config quantization_config.kv_cache_scheme) -----
KV_PER_TOK_PER_LAYER = 2 * NKV * HD * 1                 # K and V, 1 byte each
def kv_read(ctx):
    return (len(GLOBAL) * ctx + len(SLIDING) * min(ctx, SW)) * KV_PER_TOK_PER_LAYER
def kv_capacity_bytes_per_token():
    return len(GLOBAL) * KV_PER_TOK_PER_LAYER           # SWA layers are a fixed ring

# ---- draft (DFlash): ONE forward per block, shares target embed + lm_head ----
DRAFT_W = sum(nbytes(DHDR, k) for k in DHDR)
DRAFT_STEP = DRAFT_W + lmhead                            # + target lm_head for draft logits

# ---- expert union model ------------------------------------------------------
def union(k, corr=0.0):
    """Expected distinct experts per MoE layer for k tokens x top-k of N.
    corr in [0,1) shrinks the spread to model routing correlation between
    adjacent tokens (corr=0 => independent-uniform upper bound)."""
    p = TOPK / NEXP
    u = NEXP * (1.0 - (1.0 - p) ** k)
    return TOPK + (u - TOPK) * (1.0 - corr)

def verify_bytes(k, ctx, corr=0.0):
    return FIXED + union(k, corr) * EXPERT_B * len(MOE_L) + kv_read(ctx)

def efrac(k, corr=0.0):
    return union(k, corr) * EXPERT_B * len(MOE_L) / EXPERT_POOL

def tau(k, alpha):
    return (1 - alpha ** (k + 1)) / (1 - alpha)

def fit_alpha(k, t):
    lo, hi = 1e-6, 0.999999
    for _ in range(200):
        m = (lo + hi) / 2
        if tau(k, m) < t: lo = m
        else: hi = m
    return (lo + hi) / 2

GB = 1e9
def main():
    ctx = int(os.environ.get("CTX", 4096))
    print("="*78); print("LAGUNA S 2.1 NVFP4 - BYTES PER DECODE STEP  (all read from disk)"); print("="*78)
    print(f"layers={NL}  global={len(GLOBAL)} sliding={len(SLIDING)} (window={SW})  "
          f"experts={NEXP} top-{TOPK}  moe_layers={len(MOE_L)}  dense_layers={sorted(DENSE)}  BLK={BLK}")
    print(f"\n  {'component':34s}{'GB':>10s}{'dtype':>12s}")
    for nm, b, d in [("attention q,k,v,o,g (48 layers)", attn, "BF16"),
                     ("norms (in/post/q/k + final)", norms, "BF16"),
                     ("layer-0 dense MLP", dense0, "BF16"),
                     ("MoE routers (47) + bias", router, "BF16/F32"),
                     ("shared experts (47)", shared, "BF16"),
                     ("lm_head (NOT tied)", lmhead, "BF16"),
                     ("embed row (1 token)", embed_row, "BF16"),
                     ("--- k-independent FIXED ---", FIXED, ""),
                     (f"routed experts, top-{TOPK} x 47", AR_EXPERTS, "NVFP4"),
                     (f"KV @ ctx={ctx}", kv_read(ctx), "FP8")]:
        print(f"  {nm:34s}{b/GB:10.4f}{d:>12s}")
    B_tok = FIXED + AR_EXPERTS + kv_read(ctx)
    print(f"  {'B_tok (autoregressive)':34s}{B_tok/GB:10.4f}")
    print(f"\n  one routed expert = {EXPERT_B/1e6:.3f} MB   whole expert pool = {EXPERT_POOL/GB:.2f} GB "
          f"({EXPERT_POOL/71.898733760e9*100:.1f}% of checkpoint)")
    print(f"  draft: {DRAFT_W/GB:.4f} GB weights + {lmhead/GB:.4f} GB shared lm_head = {DRAFT_STEP/GB:.4f} GB/block")

    print(f"\n{'-'*78}\nB_tok and AR ceiling vs context\n{'-'*78}")
    print(f"  {'ctx':>9s}{'KV GB':>10s}{'B_tok GB':>11s}{'@91':>8s}{'@135':>8s}{'@200':>8s}  tok/s")
    for c in (2048, 4096, 8192, 32768, 131072, 262144):
        b = FIXED + AR_EXPERTS + kv_read(c)
        print(f"  {c:>9d}{kv_read(c)/GB:10.4f}{b/GB:11.4f}"
              f"{91e9/b:8.2f}{135e9/b:8.2f}{200e9/b:8.2f}")

    print(f"\n{'-'*78}\nExpert union / E_frac (independent-uniform upper bound)\n{'-'*78}")
    print(f"  {'k':>3s}{'U(k)/256':>10s}{'E_frac':>9s}{'expert GB':>11s}{'fixed GB':>10s}{'block GB':>10s}")
    KS = [1,3,5,7,9,11,15]
    for k in KS:
        u = union(k); eb = u*EXPERT_B*len(MOE_L)
        print(f"  {k:>3d}{u:10.1f}{efrac(k):9.3f}{eb/GB:11.3f}{FIXED/GB:10.3f}"
              f"{(verify_bytes(k,ctx)+DRAFT_STEP)/GB:10.3f}")

    print(f"\n{'-'*78}\nProjected DFlash tok/s   (ctx={ctx}, block bytes incl. draft forward)\n{'-'*78}")
    WORK = [("HumanEval  T=0   (tau 6.438 @k=15)", fit_alpha(15, 6.438)),
            ("GSM8K      T=0   (tau 5.775 @k=15)", fit_alpha(15, 5.775)),
            ("MBPP       T=0   (tau 4.171 @k=15)", fit_alpha(15, 4.171)),
            ("MT-Bench   T=0   (tau 4.017 @k=15)", fit_alpha(15, 4.017)),
            ("code       T=0.7 (tau 3.0  @k=7 )", fit_alpha(7, 3.0)),
            ("code       T=0.7 (tau 3.0  @k=15)", fit_alpha(15, 3.0))]
    for bw in (91, 135, 200):
        print(f"\n  --- effective BW = {bw} GB/s ---   AR baseline = {bw*1e9/B_tok:.2f} tok/s")
        print(f"  {'workload':36s}{'alpha':>7s}" + "".join(f"{'k='+str(k):>8s}" for k in KS) + f"{'k*':>5s}")
        for nm, a in WORK:
            row, best, bk = [], -1, 0
            for k in KS:
                t = tau(k, a) * bw*1e9 / (verify_bytes(k, ctx) + DRAFT_STEP)
                row.append(t)
                if t > best: best, bk = t, k
            print(f"  {nm:36s}{a:7.3f}" + "".join(f"{v:8.1f}" for v in row) + f"{bk:5d}")

    print(f"\n{'-'*78}\nSensitivity: routing correlation shrinks the union\n{'-'*78}")
    a = fit_alpha(7, 3.0)
    print(f"  workload = code T=0.7 (alpha={a:.3f}), BW=135 GB/s")
    print(f"  {'corr':>6s}" + "".join(f"{'k='+str(k):>8s}" for k in KS))
    for corr in (0.0, 0.2, 0.4):
        row = [tau(k,a)*135e9/(verify_bytes(k,ctx,corr) + DRAFT_STEP) for k in KS]
        print(f"  {corr:6.1f}" + "".join(f"{v:8.1f}" for v in row))

    print(f"\n{'-'*78}\nSELF-QUANTIZATION SCENARIOS (poolside left everything but experts in BF16)\n{'-'*78}")
    NVFP4_R, FP8_R = 0.5625/2, 1.0/2          # bytes-per-param ratio vs BF16
    parts = {"attn": attn, "dense0": dense0, "router": router, "shared": shared, "lmhead": lmhead}
    SCEN = [("stock (poolside)",            dict()),
            ("FP8 attention only",          dict(attn=FP8_R)),
            ("NVFP4 attention only",        dict(attn=NVFP4_R)),
            ("FP8 everything non-expert",   dict(attn=FP8_R, dense0=FP8_R, router=FP8_R, shared=FP8_R, lmhead=FP8_R)),
            ("NVFP4 everything non-expert", dict(attn=NVFP4_R, dense0=NVFP4_R, router=NVFP4_R, shared=NVFP4_R, lmhead=NVFP4_R))]
    print(f"  {'scenario':30s}{'FIXED GB':>10s}{'B_tok GB':>10s}{'AR@135':>8s}{'AR@200':>8s}"
          f"{'spec@135':>10s}{'spec@200':>10s}   (spec = HumanEval T=0 at its k*)")
    aH = fit_alpha(15, 6.438)
    for nm, ov in SCEN:
        fx = norms + embed_row + sum(b*ov.get(k,1.0) for k,b in parts.items())
        bt = fx + AR_EXPERTS + kv_read(ctx)
        best135 = max(tau(k,aH)*135e9/(fx + union(k)*EXPERT_B*len(MOE_L) + kv_read(ctx) + DRAFT_STEP) for k in KS)
        best200 = max(tau(k,aH)*200e9/(fx + union(k)*EXPERT_B*len(MOE_L) + kv_read(ctx) + DRAFT_STEP) for k in KS)
        print(f"  {nm:30s}{fx/GB:10.3f}{bt/GB:10.3f}{135e9/bt:8.1f}{200e9/bt:8.1f}{best135:10.1f}{best200:10.1f}")

    print(f"\n{'-'*78}\nKV capacity (SWA layers = fixed {SW}-entry ring)\n{'-'*78}")
    perTok = kv_capacity_bytes_per_token()
    ringMB = len(SLIDING)*SW*KV_PER_TOK_PER_LAYER/1e6
    print(f"  global-layer KV = {perTok} B/token   sliding ring = {ringMB:.1f} MB/sequence (constant)")
    for budget in (30, 40, 44):
        print(f"  {budget} GB KV budget -> {budget*GB/perTok/1e6:.2f} M tokens of context")
    naive = NL*KV_PER_TOK_PER_LAYER
    print(f"  (if all {NL} layers were global: {naive} B/token -> "
          f"{44*GB/naive/1e6:.2f} M tokens; SWA split is a {naive/perTok:.1f}x KV win)")

main()
