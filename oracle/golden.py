"""Gate A1 deliverable: golden tensors from the REAL 71.9 GB NVFP4 checkpoint.

Strategy that makes this fit in memory: the 15.2 GB of BF16 non-expert weights are
resident (as fp32 on GPU); the 63.9 GB of NVFP4 routed experts stay on disk and only
the experts a token actually routes to are dequantised, through a bounded LRU.

Outputs (docs/golden/):
  golden_<tag>.pt   per-layer hidden states, final norm, logits, router selections
  meta_<tag>.json   prompt, token ids, top-k predictions, checksums
"""
import os, sys, json, argparse, time, collections
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_laguna import RefLaguna, dequant_nvfp4      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(ROOT, "models", "Laguna-S-2.1-NVFP4")


class Checkpoint:
    """Lazy safetensors access over the sharded checkpoint."""

    def __init__(self, path, device):
        self.dir, self.dev = path, device
        self.map = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self.files = {}
        self.cache = {}

    def _f(self, shard):
        if shard not in self.files:
            self.files[shard] = safe_open(os.path.join(self.dir, shard), framework="pt", device="cpu")
        return self.files[shard]

    def raw(self, name):
        return self._f(self.map[name]).get_tensor(name)

    def get(self, name):                       # resident, fp32, on device
        if name not in self.cache:
            self.cache[name] = self.raw(name).to(device=self.dev, dtype=torch.float32)
        return self.cache[name]

    def has(self, name):
        return name in self.map


class ResidentW(dict):
    """dict-like view that pulls BF16 (non-expert) tensors on demand."""

    def __init__(self, ck):
        super().__init__()
        self.ck = ck

    def __getitem__(self, k):
        return self.ck.get(k)

    def get(self, k, default=None):
        return self.ck.get(k) if self.ck.has(k) else default


class ExpertLRU:
    def __init__(self, ck, dev, maxn=384):
        self.ck, self.dev, self.maxn = ck, dev, maxn
        self.d = collections.OrderedDict()
        self.hits = self.miss = 0

    def __call__(self, L, e, which):
        key = (L, e, which)
        if key in self.d:
            self.hits += 1
            self.d.move_to_end(key)
            return self.d[key]
        self.miss += 1
        p = f"model.layers.{L}.mlp.experts.{e}.{which}."
        packed = self.ck.raw(p + "weight_packed").to(self.dev)
        scale = self.ck.raw(p + "weight_scale").to(self.dev)
        gs = float(self.ck.raw(p + "weight_global_scale").item())
        w = dequant_nvfp4(packed, scale, gs)
        self.d[key] = w
        if len(self.d) > self.maxn:
            self.d.popitem(last=False)
        return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Write a Python function that returns the first n prime numbers.")
    ap.add_argument("--tag", default="primes")
    ap.add_argument("--gen", type=int, default=8, help="greedy tokens to generate after prefill")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--raw", action="store_true", help="treat --prompt as raw text (no chat template)")
    args = ap.parse_args()

    dev = torch.device(args.device)
    cfg = json.load(open(os.path.join(MD, "config.json")))
    ck = Checkpoint(MD, dev)
    W, EX = ResidentW(ck), ExpertLRU(ck, dev)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(MD, "tokenizer.json"))

    if args.raw:
        text = args.prompt
    else:
        # poolside_v1 chat grammar, thinking ON (the shipped default)
        text = ("〈|EOS|〉<system>You are a helpful, conversationally-fluent assistant made by "
                "Poolside. You are here to be helpful to users through natural language "
                "conversations.</system>\n"
                f"<user>{args.prompt}</user>\n<assistant><think>")
    ids = tok.encode(text, add_special_tokens=False).ids
    print(f"prompt tokens: {len(ids)}")

    ref = RefLaguna(cfg, W, EX, device=dev)
    t0 = time.time()
    cap = {}
    kv = [None] * cfg["num_hidden_layers"]
    with torch.no_grad():
        logits, _ = ref.forward(torch.tensor([ids], device=dev), kv=kv, capture=cap)
    t_prefill = time.time() - t0
    print(f"prefill {len(ids)} tok in {t_prefill:.1f}s   expert LRU hits={EX.hits} miss={EX.miss}")

    nxt = int(logits[0, -1].argmax())
    gen = [nxt]
    t0 = time.time()
    with torch.no_grad():
        for i in range(args.gen - 1):
            lg, _ = ref.forward(torch.tensor([[gen[-1]]], device=dev), kv=kv,
                                pos_offset=len(ids) + i)
            gen.append(int(lg[0, -1].argmax()))
    print(f"greedy {args.gen} tok in {time.time()-t0:.1f}s")
    print("GREEDY IDS :", gen)
    print("GREEDY TEXT:", repr(tok.decode(gen, skip_special_tokens=False)))

    outdir = os.path.join(ROOT, "docs", "golden")
    os.makedirs(outdir, exist_ok=True)
    payload = {
        "h_embed": cap["h_embed"], "h_final": cap["h_final"], "logits": cap["logits"],
        "h": {L: v for L, v in cap["h"].items()},
        "router": cap.get("router", {}),
        "ids": torch.tensor(ids), "greedy": torch.tensor(gen),
    }
    torch.save(payload, os.path.join(outdir, f"golden_{args.tag}.pt"))

    top = torch.topk(cap["logits"][0, -1].float(), 10)
    meta = {
        "prompt": args.prompt, "raw": args.raw, "text": text,
        "n_prompt_tokens": len(ids), "ids": ids, "greedy_ids": gen,
        "greedy_text": tok.decode(gen, skip_special_tokens=False),
        "last_logit_top10": {"ids": top.indices.tolist(), "vals": top.values.tolist()},
        "h_final_sum": float(cap["h_final"].double().sum()),
        "logits_sum": float(cap["logits"].double().sum()),
        "per_layer_h_absmax": {str(L): float(v.abs().max()) for L, v in cap["h"].items()},
        "config_sha_fields": {k: cfg[k] for k in
                              ("hidden_size", "num_hidden_layers", "num_experts",
                               "num_experts_per_tok", "vocab_size", "sliding_window")},
        "torch": torch.__version__,
    }
    json.dump(meta, open(os.path.join(outdir, f"meta_{args.tag}.json"), "w"), indent=2)
    print(f"saved -> {outdir}/golden_{args.tag}.pt")


if __name__ == "__main__":
    main()
