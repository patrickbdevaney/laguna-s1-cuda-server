#!/usr/bin/env python3
"""How pessimistic is the RTN proxy, in distortion, against a real trellis code?

WHY THIS IS THE RIGHT QUESTION RIGHT NOW. The capability sweep is scoring RTN-in-container at
2.81 and 2.32 bits of payload. Whatever it returns, it is a statement about ROUND-TO-NEAREST,
and the thing we would actually ship is an EXL3/QTIP trellis codec at 3.0 bpw. Without a
distortion bridge between the two, a pass tells us nothing quantitative about trellis and a
fail cannot be attributed -- we would not know whether 3 bits is too few or whether RTN is just
a bad code. This script builds that bridge on the real expert weights.

WHAT A TRELLIS BUYS, in principle. Scalar quantization of a Gaussian at rate R leaves a fixed
gap to the rate-distortion bound D = sigma^2 * 2^(-2R): about 4.35 dB at high rate (the
"space-filling" 1.53 dB plus the granular loss). A trellis-coded quantizer spends state, not
rate, to recover most of that gap -- the encoder runs Viterbi over a convolutional code so the
reconstruction levels for one weight depend on its neighbours, which buys back roughly 1 bit
of effective precision at practical constraint lengths. That is the entire reason EXL3 at 3.0
bpw is interesting when RTN at 3.0 bpw is not.

WHAT IS MEASURED. For real expert matrices out of the checkpoint, at matched payload bits:
    1. NVFP4 as shipped                       (4 bits payload, the reference)
    2. RTN onto a coarser E2M1 subset         (what the sweep is actually running)
    3. Trellis-coded quantization, Viterbi    (what EXL3 would ship)
reported as relative Frobenius error, which is the quantity that composes across a layer.

The comparison is deliberately run INSIDE the same per-16 E4M3 scale structure for all three,
so the only thing varying is the code, not the scaling. That understates the trellis slightly
(EXL3 also applies a Hadamard incoherence transform, which helps it further), which keeps this
a conservative bound in the direction we want: if trellis already wins here, it wins by more in
its real configuration.
"""
import argparse, json, os, struct, sys
import numpy as np

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
SUBSET = {7: np.array([0, 4, 6, 7]), 5: np.array([0, 5, 7]), 3: np.array([0, 7])}


def e4m3_to_f32(b):
    s = np.where(b >> 7, -1.0, 1.0).astype(np.float32)
    e = ((b >> 3) & 0x0F).astype(np.int32)
    m = (b & 0x07).astype(np.float32)
    return s * np.where(e == 0, (m / 8.0) * 2.0 ** -6, (1.0 + m / 8.0) * 2.0 ** (e - 7))


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def load(path, hdr, off0, name):
    t = hdr[name]
    a, b = t["data_offsets"]
    with open(path, "rb") as f:
        f.seek(off0 + a)
        raw = f.read(b - a)
    dt = {"U8": np.uint8, "F8_E4M3": np.uint8, "F32": np.float32, "BF16": np.uint16}[t["dtype"]]
    return np.frombuffer(raw, dtype=dt).reshape(t["shape"] or (1,))


def dequant_nvfp4(packed, scale, ginv, group=16):
    lo, hi = packed & 0x0F, packed >> 4
    v = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.float32)
    v[:, 0::2] = np.where(lo >> 3, -1.0, 1.0) * E2M1[lo & 7]
    v[:, 1::2] = np.where(hi >> 3, -1.0, 1.0) * E2M1[hi & 7]
    return v, v * np.repeat(e4m3_to_f32(scale).astype(np.float32), group, axis=1) * float(ginv)


def rtn_subset(v, levels):
    """Snap the E2M1 code magnitudes onto a coarser allowed subset (what the sweep runs)."""
    allowed = E2M1[SUBSET[levels]]
    mag = np.abs(v)
    j = np.abs(mag.reshape(-1, 1) - allowed.reshape(1, -1)).argmin(axis=1)
    return (np.sign(v) * allowed[j].reshape(v.shape)).astype(np.float32)


def trellis_quantize(x, bits=3, state_bits=10, cb_sigma=1.0):
    """Viterbi-optimal trellis-coded quantization of a 1-D signal.

    The code is the QTIP shape: a running state of `state_bits` bits shifts in `bits` new bits
    per weight, and the reconstruction level is a deterministic pseudo-random function of the
    whole state. So consecutive weights share state, the effective codebook is far larger than
    2^bits, and the encoder must search -- which is what Viterbi does here, exactly, in
    O(n * 2^state_bits * 2^bits).

    The codebook is a hash of the state mapped through an inverse Gaussian CDF, which is what
    makes the marginal distribution of reconstruction levels Gaussian and matches the weight
    distribution without storing any table. This mirrors QTIP's "3INST"/hashed codebook; the
    hash constant is arbitrary as long as it mixes.
    """
    n = len(x)
    S = 1 << state_bits
    K = 1 << bits
    # Hashed Gaussian codebook over the full state space.
    s = np.arange(S, dtype=np.uint64)
    h = (s * np.uint64(0x9E3779B97F4A7C15)) ^ (s >> np.uint64(29))
    h = (h * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    u = ((h >> np.uint64(11)).astype(np.float64) / float(1 << 53))
    u = np.clip(u, 1e-9, 1 - 1e-9)
    # Inverse normal CDF via the Beasley-Springer-Moro-free route: use erfinv.
    from scipy.special import erfinv
    cb = (np.sqrt(2.0) * erfinv(2 * u - 1) * cb_sigma).astype(np.float32)

    mask = S - 1
    # pred[s] : the 2^bits states that can precede state s is implicit; we go forward instead.
    # Forward transition: s' = ((s << bits) | w) & mask  for w in [0, K)
    prev_states = np.arange(S, dtype=np.int64)
    trans = np.empty((S, K), dtype=np.int64)
    for w in range(K):
        trans[:, w] = ((prev_states << bits) | w) & mask

    INF = np.float32(1e30)
    cost = np.zeros(S, dtype=np.float32)          # start: any state allowed
    back = np.empty((n, S), dtype=np.int32)

    for i in range(n):
        # cand[s_prev, w] = cost[s_prev] + (x[i] - cb[trans[s_prev, w]])^2
        d = (x[i] - cb[trans]) ** 2               # [S, K]
        cand = cost[:, None] + d
        # Scatter-min into the new state space. Several (s_prev, w) map to the same s_new;
        # np.minimum.at is the correct reduction and argmin has to be recovered separately.
        new = np.full(S, INF, dtype=np.float32)
        flat_t = trans.ravel()
        flat_c = cand.ravel()
        np.minimum.at(new, flat_t, flat_c)
        # Recover the predecessor achieving the min.
        hit = flat_c <= new[flat_t] + np.float32(0)
        src = np.full(S, -1, dtype=np.int32)
        idx = np.nonzero(hit)[0]
        src[flat_t[idx]] = (idx // K).astype(np.int32)
        back[i] = src
        cost = new

    # Trace back from the best final state.
    s_end = int(np.argmin(cost))
    rec = np.empty(n, dtype=np.float32)
    s_cur = s_end
    for i in range(n - 1, -1, -1):
        rec[i] = cb[s_cur]
        s_cur = int(back[i, s_cur])
        if s_cur < 0:
            s_cur = 0
    return rec


def relerr(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="models/Laguna-S-2.1-NVFP4")
    ap.add_argument("--tensors", type=int, default=3, help="expert tensors to sample")
    ap.add_argument("--rows", type=int, default=8, help="rows per tensor (Viterbi is O(n*2^s))")
    ap.add_argument("--state-bits", type=int, default=10)
    a = ap.parse_args()

    shards = sorted(f for f in os.listdir(a.src) if f.endswith(".safetensors"))
    picked = []
    for sh in shards:
        p = os.path.join(a.src, sh)
        hdr, off0 = read_header(p)
        names = sorted(k for k in hdr if k.endswith(".weight_packed") and ".experts." in k)
        for nm in names[:: max(1, len(names) // max(1, a.tensors))]:
            picked.append((p, hdr, off0, nm))
            if len(picked) >= a.tensors:
                break
        if len(picked) >= a.tensors:
            break

    agg = {}
    for p, hdr, off0, nm in picked:
        base = nm[: -len("weight_packed")]
        packed = load(p, hdr, off0, nm)
        scale = load(p, hdr, off0, base + "weight_scale")
        g = load(p, hdr, off0, base + "weight_global_scale")
        ginv = 1.0 / float(g[0])
        codes, ref = dequant_nvfp4(packed, scale, ginv)

        rows = min(a.rows, codes.shape[0])
        sc = np.repeat(e4m3_to_f32(scale[:rows]).astype(np.float32), 16, axis=1) * float(ginv)
        ref_r = ref[:rows]

        for lv in (7, 5):
            q = rtn_subset(codes[:rows], lv) * sc
            agg.setdefault(f"rtn{lv}", []).append(relerr(ref_r, q))

        # Trellis at 3 bits, run on the SAME scaled grid so the comparison is code-vs-code.
        # Operate in units of the per-group scale, i.e. on the E2M1-domain values, and let the
        # codebook sigma match that domain.
        dom = codes[:rows]
        sig = float(dom.std())
        for tb in (3, 2):
            tq = np.empty_like(ref_r)
            for r in range(rows):
                tq[r] = trellis_quantize(dom[r].astype(np.float32), bits=tb,
                                         state_bits=a.state_bits, cb_sigma=sig)
            agg.setdefault(f"trellis{tb}", []).append(relerr(ref_r, tq * sc))
        print(f"  {nm.split('.')[-3:][0]}... done", flush=True)

    print("\nrelative Frobenius error vs shipped NVFP4 (lower is better)")
    print(f"{'code':12s} {'payload bits':>12s} {'rel err':>10s}")
    for k, bits in (("rtn7", 2.81), ("rtn5", 2.32), ("trellis3", 3.00), ("trellis2", 2.00)):
        if k in agg:
            v = np.array(agg[k], dtype=float)
            print(f"{k:12s} {bits:12.2f} {v.mean():10.4f}   (n={len(v)}, sd={v.std():.4f})")


if __name__ == "__main__":
    main()
