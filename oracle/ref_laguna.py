"""Dependency-light Laguna reference — a direct transcription of modeling_laguna.py.

Why this exists: the shipped reference needs transformers v5 and would materialise the
117.6B model in BF16 (~235 GB) to run. This transcription reads the NVFP4 checkpoint
directly and dequantises only the ~10 experts per layer that a token actually routes to,
so a real-weights forward pass fits in memory. It is validated bit-close against the
shipped reference at tiny scale by oracle/validate_ref.py before it is trusted.

Contract: fp32 math throughout (the reference upcasts in RMSNorm and the softplus gate;
we go further and keep everything fp32 so the golden tensors carry no accumulated bf16
rounding of our own). Weights are cast to fp32 on use.
"""
import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- primitives
def rms_norm(x, w, eps):
    """LagunaRMSNorm: fp32 variance, weight applied after cast back."""
    dt = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (w.float() * x.to(dt).float())


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def apply_rope(q, k, cos, sin):
    """Partial rotary: only the first cos.shape[-1] dims rotate; the tail passes through."""
    cos = cos.unsqueeze(1)          # [B,1,S,rd]
    sin = sin.unsqueeze(1)
    rd = cos.shape[-1]
    qr, qp = q[..., :rd], q[..., rd:]
    kr, kp = k[..., :rd], k[..., rd:]
    q = torch.cat([qr * cos + rotate_half(qr) * sin, qp], dim=-1)
    k = torch.cat([kr * cos + rotate_half(kr) * sin, kp], dim=-1)
    return q, k


def repeat_kv(x, n):
    if n == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n, s, d).reshape(b, h * n, s, d)


# ---------------------------------------------------------------- rope tables
def rope_tables(inv_freq, positions, attention_scaling, device, dtype=torch.float32):
    """cos/sin of shape [B, S, rotary_dim].  emb = cat(freqs, freqs)."""
    inv = inv_freq.to(device=device, dtype=torch.float32)[None, :, None]   # [1,D/2,1]
    pos = positions.to(device=device, dtype=torch.float32)[:, None, :]     # [B,1,S]
    freqs = (inv @ pos).transpose(1, 2)                                    # [B,S,D/2]
    emb = torch.cat((freqs, freqs), dim=-1)
    return (emb.cos() * attention_scaling).to(dtype), (emb.sin() * attention_scaling).to(dtype)


def yarn_inv_freq(head_dim, partial, base, factor, orig_max_pos, beta_fast, beta_slow):
    """transformers' _compute_yarn_parameters, transcribed. Returns (inv_freq, attn_scale).

    Kept here so the C++ server can be checked against a formula rather than a dumped
    table; oracle/dump_rope.py asserts this matches the shipped implementation exactly.
    """
    dim = int(head_dim * partial)
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    inv_extrapolation = 1.0 / pos_freqs
    inv_interpolation = 1.0 / (factor * pos_freqs)

    def find_dim(num_rot):
        return (dim * math.log(orig_max_pos / (num_rot * 2 * math.pi))) / (2 * math.log(base))

    low = math.floor(find_dim(beta_fast))
    high = math.ceil(find_dim(beta_slow))
    low, high = max(low, 0), min(high, dim - 1)

    if low == high:
        high += 0.001
    ramp = (torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)
    mask = 1 - torch.clamp(ramp, 0, 1)          # 1 => extrapolate (high freq)
    inv_freq = inv_interpolation * (1 - mask) + inv_extrapolation * mask
    attn_scale = 0.1 * math.log(factor) + 1.0
    return inv_freq, attn_scale


# ---------------------------------------------------------------- NVFP4
_E2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float32)


def dequant_nvfp4(packed, scale_e4m3, global_scale, group=16):
    """packed [N, K/2] uint8 (two E2M1 codes/byte, low nibble = even k)
       scale_e4m3 [N, K/group] float8_e4m3fn
       global_scale scalar fp32   ->  [N, K] fp32

    compressed-tensors stores `weight_global_scale` as a RECIPROCAL:
        global_scale = (FP8_E4M3_MAX * FP4_E2M1_MAX) / amax = 2688 / amax
    and pre-multiplies it into `weight_scale` so the per-group scale fits e4m3.
    Dequant therefore DIVIDES:  w = e2m1_code * weight_scale / global_scale.
    Sanity: 6 * max(weight_scale) / global_scale == the tensor's original amax.
    """
    lo = (packed & 0x0F).long()
    hi = (packed >> 4).long()
    sgn_lo = torch.where((lo & 8) != 0, -1.0, 1.0)
    sgn_hi = torch.where((hi & 8) != 0, -1.0, 1.0)
    mag = _E2M1.to(packed.device)
    vals = torch.stack([sgn_lo * mag[lo & 7], sgn_hi * mag[hi & 7]], dim=-1)
    N, Kh, _ = vals.shape
    vals = vals.reshape(N, Kh * 2)                                   # [N,K]
    s = scale_e4m3.to(torch.float32) / float(global_scale)           # [N,K/group]
    s = s.repeat_interleave(group, dim=1)
    return vals * s


# ---------------------------------------------------------------- the model
class RefLaguna:
    """cfg: dict from config.json.  W: name -> torch tensor (fp32/bf16, on `device`).
       expert: callable(layer:int, expert:int, which:str) -> fp32 [out,in] tensor."""

    def __init__(self, cfg, W, expert, device="cpu", dtype=torch.float32):
        self.c, self.W, self.expert = cfg, W, expert
        self.dev, self.dt = device, dtype
        self.NL = cfg["num_hidden_layers"]
        self.H = cfg["hidden_size"]
        self.HD = cfg["head_dim"]
        self.NKV = cfg["num_key_value_heads"]
        self.EPS = cfg["rms_norm_eps"]
        self.NHL = cfg["num_attention_heads_per_layer"]
        self.LT = cfg["layer_types"]
        self.SW = cfg["sliding_window"]
        self.TOPK = cfg["num_experts_per_tok"]
        self.NEXP = cfg["num_experts"]
        self.SCALE = float(cfg["moe_routed_scaling_factor"])
        self.NORMTOPK = cfg["norm_topk_prob"]
        self.DENSE = set(cfg["mlp_only_layers"])
        self.softcap = float(cfg.get("moe_router_logit_softcapping", 0.0) or 0.0)

        rp = cfg["rope_parameters"]
        fa = rp["full_attention"]
        inv_f, sc_f = yarn_inv_freq(self.HD, fa.get("partial_rotary_factor", 1.0),
                                    fa["rope_theta"], fa["factor"],
                                    fa["original_max_position_embeddings"],
                                    fa["beta_fast"], fa["beta_slow"])
        if "attention_factor" in fa and fa["attention_factor"] is not None:
            sc_f = float(fa["attention_factor"])
        sa = rp["sliding_attention"]
        dim_s = int(self.HD * sa.get("partial_rotary_factor", 1.0))
        inv_s = 1.0 / (sa["rope_theta"] ** (torch.arange(0, dim_s, 2, dtype=torch.float32) / dim_s))
        self.rope = {"full_attention": (inv_f, sc_f), "sliding_attention": (inv_s, 1.0)}

    def w(self, name):
        return self.W[name].to(device=self.dev, dtype=torch.float32)

    # ---- one attention layer
    def attn(self, h_in, L, cos, sin, kv, pos):
        p = f"model.layers.{L}.self_attn."
        nh, hd, nkv = self.NHL[L], self.HD, self.NKV
        B, S, _ = h_in.shape

        q = (h_in @ self.w(p + "q_proj.weight").T).view(B, S, nh, hd).transpose(1, 2)
        k = (h_in @ self.w(p + "k_proj.weight").T).view(B, S, nkv, hd).transpose(1, 2)
        v = (h_in @ self.w(p + "v_proj.weight").T).view(B, S, nkv, hd).transpose(1, 2)

        q = rms_norm(q, self.w(p + "q_norm.weight"), self.EPS)
        k = rms_norm(k, self.w(p + "k_norm.weight"), self.EPS)
        q, k = apply_rope(q, k, cos, sin)

        if kv is not None:                       # append to cache, attend over all of it
            if kv[L] is None:
                kv[L] = (k, v)
            else:
                kv[L] = (torch.cat([kv[L][0], k], 2), torch.cat([kv[L][1], v], 2))
            k, v = kv[L]

        g = nh // nkv
        ks, vs = repeat_kv(k, g), repeat_kv(v, g)
        att = (q @ ks.transpose(2, 3)) * (hd ** -0.5)

        # causal (+ sliding-window) mask over absolute positions
        kpos = torch.arange(ks.shape[2], device=q.device)
        qpos = pos.view(-1)[-S:] if pos.numel() >= S else torch.arange(S, device=q.device)
        allowed = kpos[None, :] <= qpos[:, None]
        if self.LT[L] == "sliding_attention":
            allowed &= kpos[None, :] > (qpos[:, None] - self.SW)
        att = att.masked_fill(~allowed[None, None], float("-inf"))

        att = F.softmax(att, dim=-1, dtype=torch.float32)
        o = (att @ vs).transpose(1, 2).reshape(B, S, nh * hd)

        gate = F.softplus((h_in @ self.w(p + "g_proj.weight").T).float())      # per-head
        o = (o.view(B, S, nh, hd) * gate.unsqueeze(-1)).view(B, S, nh * hd)
        return o @ self.w(p + "o_proj.weight").T

    # ---- one MLP / MoE layer
    def mlp(self, x, L, capture=None):
        pre = f"model.layers.{L}.mlp."
        if L in self.DENSE:
            g = x @ self.w(pre + "gate_proj.weight").T
            u = x @ self.w(pre + "up_proj.weight").T
            return (F.silu(g) * u) @ self.w(pre + "down_proj.weight").T

        B, S, H = x.shape
        xf = x.reshape(-1, H)

        # shared expert
        sg = xf @ self.w(pre + "shared_expert.gate_proj.weight").T
        su = xf @ self.w(pre + "shared_expert.up_proj.weight").T
        shared = (F.silu(sg) * su) @ self.w(pre + "shared_expert.down_proj.weight").T

        # router: sigmoid, bias affects SELECTION ONLY
        logits = xf @ self.w(pre + "gate.weight").T
        if self.softcap > 0:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        scores = torch.sigmoid(logits.float())
        bias = self.W.get(pre + "experts.e_score_correction_bias")
        sel_scores = scores + (bias.to(scores.device, torch.float32) if bias is not None else 0.0)
        sel = torch.topk(sel_scores, self.TOPK, dim=-1).indices
        wts = scores.gather(-1, sel)
        if self.NORMTOPK:
            wts = wts / wts.sum(-1, keepdim=True)

        if capture is not None:
            capture.setdefault("router", {})[L] = {
                "sel": sel.cpu().clone(), "w": wts.cpu().clone(),
                "scores": scores.cpu().clone()}

        out = torch.zeros_like(xf)
        for t in range(xf.shape[0]):
            for j in range(self.TOPK):
                e = int(sel[t, j])
                gw = self.expert(L, e, "gate_proj")
                uw = self.expert(L, e, "up_proj")
                dw = self.expert(L, e, "down_proj")
                hgu = F.silu(xf[t] @ gw.T) * (xf[t] @ uw.T)
                out[t] += (hgu @ dw.T) * wts[t, j]
        return (out * self.SCALE + shared).view(B, S, H)

    # ---- full forward
    def forward(self, ids, kv=None, pos_offset=0, capture=None, tap_layers=()):
        B, S = ids.shape
        h = self.w("model.embed_tokens.weight")[ids]
        pos = torch.arange(pos_offset, pos_offset + S, device=self.dev)[None]

        cs = {}
        for t in ("full_attention", "sliding_attention"):
            inv, sc = self.rope[t]
            cs[t] = rope_tables(inv, pos, sc, self.dev)

        if capture is not None:
            capture["h_embed"] = h.cpu().clone()
        taps = {}
        for L in range(self.NL):
            pre = f"model.layers.{L}."
            hn = rms_norm(h, self.w(pre + "input_layernorm.weight"), self.EPS)
            cos, sin = cs[self.LT[L]]
            h = h + self.attn(hn, L, cos, sin, kv, pos)
            hn = rms_norm(h, self.w(pre + "post_attention_layernorm.weight"), self.EPS)
            h = h + self.mlp(hn, L, capture)
            if capture is not None:
                capture.setdefault("h", {})[L] = h.cpu().clone()
            if L in tap_layers:
                taps[L] = h.clone()

        h = rms_norm(h, self.w("model.norm.weight"), self.EPS)
        logits = h @ self.w("lm_head.weight").T
        if capture is not None:
            capture["h_final"] = h.cpu().clone()
            capture["logits"] = logits.cpu().clone()
        return logits, taps
