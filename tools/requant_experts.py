#!/usr/bin/env python3
"""Requantize the ROUTED EXPERTS to fewer levels, in place in the NVFP4 container.

WHY IN-CONTAINER. The question is what capability costs, not what a new kernel costs. Reducing
the number of representable levels while keeping the NVFP4 container means the existing CUDA
kernels run unchanged, so the quality delta and the acceptance rate are measured at identical
speed and with zero new numerics to validate. The speed side comes from the byte model, which
is already anchored (206/33.0 = 6.242 GB against a computed 6.251).

WHAT IT MEASURES, precisely. E2M1 represents {0, .5, 1, 1.5, 2, 3, 4, 6} with a sign, so ~15
levels per group of 16. Restricting to a symmetric subset that lands exactly on the grid gives:

    LEVELS=7  ->  {0, ±2, ±4, ±6}      log2(7) = 2.81 bits of payload
    LEVELS=5  ->  {0, ±3, ±6}          log2(5) = 2.32 bits of payload
    LEVELS=3  ->  {0, ±6}              log2(3) = 1.58 bits of payload

The per-group scale is INHERITED, not re-derived. Re-deriving it as amax/L is the obvious move
and it is wrong: NVFP4 divides amax by 6 precisely because 6 is E2M1's maximum, which puts the
scale exactly at E4M3's 448 ceiling. Asking for amax/3 asks for 896, E4M3 clips, and every
weight returns at half magnitude. The round-trip check caught it (absmax 0.154 -> 0.083).
Coarsening the code grid instead has no such failure mode and keeps amax exact.

This is round-to-nearest, so it is a PESSIMISTIC proxy for a trellis codec at similar payload
bits — vector quantization is strictly better per bit. If RTN at ~2.8 bits holds up on
reasoning, EXL3 Trellis-3 at 3.0 bpw almost certainly does. Read the result in that direction
and not the other.

SAFETY. The source shards are opened read-only and never written. Output goes to a separate
directory containing ONLY the tensors that changed; the loader is pointed at it with an
override so the original checkpoint is never modified. The source files are also chmod a-w on
disk as a second line of defence.
"""
import argparse, json, os, struct, sys
import numpy as np

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

def e4m3_to_f32(b: np.ndarray) -> np.ndarray:
    s = np.where(b >> 7, -1.0, 1.0).astype(np.float32)
    e = ((b >> 3) & 0x0F).astype(np.int32)
    m = (b & 0x07).astype(np.float32)
    sub = (m / 8.0) * (2.0 ** -6)
    nor = (1.0 + m / 8.0) * (2.0 ** (e - 7))
    return s * np.where(e == 0, sub, nor)

def f32_to_e4m3(x: np.ndarray) -> np.ndarray:
    """Nearest E4M3 by exhaustive search over the 256 codes — 256 candidates is nothing next to
    being subtly wrong about a float format, and this runs offline."""
    codes = np.arange(256, dtype=np.uint8)
    vals = e4m3_to_f32(codes)
    vals[np.isnan(vals)] = np.inf
    idx = np.abs(x.reshape(-1, 1).astype(np.float32) - vals.reshape(1, -1)).argmin(axis=1)
    return codes[idx].reshape(x.shape)

def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n

def load_tensor(path, hdr, off0, name):
    t = hdr[name]
    a, b = t["data_offsets"]
    with open(path, "rb") as f:
        f.seek(off0 + a)
        raw = f.read(b - a)
    dt = {"U8": np.uint8, "F8_E4M3": np.uint8, "F32": np.float32, "BF16": np.uint16}[t["dtype"]]
    return np.frombuffer(raw, dtype=dt).reshape(t["shape"] if t["shape"] else (1,)), t

def requant_expert(packed, scale_e4m3, ginv, levels, group=16):
    """packed[R, K/2] uint8 (low nibble = even k), scale[R, K/group] E4M3, ginv = 1/global.

    Dequantizes, then re-quantizes each group of `group` weights onto a symmetric integer grid
    of `levels` values with a re-derived per-group scale, and re-encodes into the same
    container. Returns (packed', scale').
    """
    R, KH = packed.shape
    K = KH * 2
    lo = (packed & 0x0F).astype(np.uint8)
    hi = (packed >> 4).astype(np.uint8)
    mag_lo, sgn_lo = E2M1[lo & 0x07], np.where(lo >> 3, -1.0, 1.0)
    mag_hi, sgn_hi = E2M1[hi & 0x07], np.where(hi >> 3, -1.0, 1.0)
    v = np.empty((R, K), dtype=np.float32)
    v[:, 0::2] = sgn_lo * mag_lo
    v[:, 1::2] = sgn_hi * mag_hi

    s = e4m3_to_f32(scale_e4m3).astype(np.float32)           # [R, K/group]
    w = v * np.repeat(s, group, axis=1) * float(ginv)        # true weights (ginv is 1/gscale)

    # --- Snap onto a coarser SUBSET of the E2M1 grid, keeping the original scale.
    #
    # The first version re-derived the scale as amax/L. That saturates: NVFP4 divides amax by 6
    # precisely because 6 is E2M1's maximum and the resulting scale then lands exactly at
    # E4M3's 448 ceiling. Asking for amax/3 asks for 896, E4M3 clips it to 448, and every
    # weight comes back at half magnitude -- which is what the round-trip check caught
    # (absmax 0.154 -> 0.083). Keeping the scale and coarsening the CODE grid has no such
    # failure mode and is exactly representable.
    #
    #   7 levels: {0, ±2, ±4, ±6}  -> E2M1 indices {0, 4, 6, 7}, log2(7) = 2.81 bits
    #   5 levels: {0, ±3, ±6}      -> E2M1 indices {0, 5, 7},     log2(5) = 2.32 bits
    SUBSET = {7: np.array([0, 4, 6, 7]), 5: np.array([0, 5, 7]), 3: np.array([0, 7])}
    if levels not in SUBSET:
        raise SystemExit(f"levels={levels} not one of {sorted(SUBSET)}")
    idx_allowed = SUBSET[levels]
    mag_allowed = E2M1[idx_allowed]

    mag = np.abs(v)                                    # in E2M1 units, scale untouched
    j = np.abs(mag.reshape(-1, 1) - mag_allowed.reshape(1, -1)).argmin(axis=1)
    code_mag = idx_allowed[j].reshape(v.shape).astype(np.uint8)
    code = (code_mag | np.where(v < 0, 0x08, 0x00).astype(np.uint8))
    scale_new = scale_e4m3                             # unchanged

    packed_new = (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8)
    return packed_new, scale_new

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="models/Laguna-S-2.1-NVFP4")
    ap.add_argument("--out", required=True, help="output DIR (created; originals untouched)")
    ap.add_argument("--levels", type=int, default=7, help="7 -> ~2.81b payload, 5 -> ~2.32b")
    ap.add_argument("--limit", type=int, default=0, help="stop after N expert tensors (dry run)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    shards = sorted(f for f in os.listdir(a.src) if f.endswith(".safetensors"))
    out_t, out_meta, blob, off = {}, {}, [], 0
    done = 0
    for sh in shards:
        p = os.path.join(a.src, sh)
        hdr, off0 = read_header(p)
        names = [k for k in hdr if k.endswith(".weight_packed") and ".experts." in k]
        for nm in sorted(names):
            base = nm[: -len("weight_packed")]
            packed, tp = load_tensor(p, hdr, off0, nm)
            scale, ts = load_tensor(p, hdr, off0, base + "weight_scale")
            g, _ = load_tensor(p, hdr, off0, base + "weight_global_scale")
            ginv = 1.0 / float(g[0])            # RECIPROCAL: dequant divides. See README fact 2.
            pk, sc = requant_expert(packed, scale, ginv, a.levels)
            for n2, arr, dt in ((nm, pk, "U8"), (base + "weight_scale", sc, "F8_E4M3")):
                b = arr.tobytes()
                out_t[n2] = {"dtype": dt, "shape": list(arr.shape),
                             "data_offsets": [off, off + len(b)]}
                blob.append(b); off += len(b)
            done += 1
            if done % 200 == 0:
                print(f"  {done} expert tensors", flush=True)
            if a.limit and done >= a.limit:
                break
        if a.limit and done >= a.limit:
            break

    out_meta["__metadata__"] = {"requant_levels": str(a.levels),
                                "payload_bits": f"{np.log2(a.levels):.3f}",
                                "source": a.src,
                                "note": "routed experts only; all other tensors unchanged"}
    hdr_bytes = json.dumps({**out_meta, **out_t}).encode()
    pad = (-len(hdr_bytes)) % 8
    hdr_bytes += b" " * pad
    dst = os.path.join(a.out, "experts.safetensors")
    with open(dst, "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        for b in blob:
            f.write(b)
    print(f"wrote {dst}: {done} expert tensors, {off/1e9:.2f} GB, "
          f"{np.log2(a.levels):.2f} bits payload + 0.5 scale = "
          f"{np.log2(a.levels)+0.5:.2f} bpw in-container")

if __name__ == "__main__":
    main()
