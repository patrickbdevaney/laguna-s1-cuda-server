#!/usr/bin/env python3
"""Compare capability configs PAIRWISE, because the configs saw identical problems.

WHY NOT JUST COMPARE ACCURACIES. Two independent proportions at n=200 and p~0.6 have a
difference-of-proportions standard error near 4.9 points, so an unpaired test can only resolve a
~10-point drop. That is too coarse to decide whether 3 bpw is safe. But the configs are not
independent samples -- every config answered the SAME 200 problems, so the comparison can be
paired, and problem difficulty (which dominates the variance) cancels exactly.

McNemar's test uses only the DISCORDANT pairs: problems one config got right and the other got
wrong. If requantization is harmless, the two discordant counts should be symmetric; a
systematic capability loss shows up as an asymmetry. With ~20 discordant pairs this resolves
shifts far smaller than 10 points, which is the whole reason to run it.

Also reported, because accuracy alone hides the thing that matters here:
  * s_per_correct  -- the headline. Seconds of wall-clock per correct answer. A config that is
                      faster per token but needs more tokens to arrive is a LOSS and this is
                      the only number that shows it.
  * token inflation on the problems BOTH configs got right -- the cleanest possible measure of
                      "did it start rambling", with correctness and problem difficulty held
                      fixed.
"""
import json, sys, os, glob
from math import comb


def mcnemar_exact(b, c):
    """Two-sided exact binomial test on the discordant pairs. Exact rather than chi-square
    because the discordant count is small enough that the asymptotic approximation is not
    trustworthy, and this costs nothing."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def load(path):
    d = json.load(open(path))
    return d["summary"], {r["id"]: r for r in d["rows"]}


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "eval_out"
    files = sorted(glob.glob(os.path.join(d, "*_gsm8k.json")))
    if not files:
        sys.exit(f"no *_gsm8k.json in {d}")
    cfgs = {}
    for f in files:
        s, rows = load(f)
        cfgs[s["tag"]] = (s, rows)

    print(f"{'config':10s} {'acc':>10s} {'s/correct':>10s} {'wall':>8s} "
          f"{'mean_gen':>9s} {'tok/s':>7s} {'trunc':>6s}")
    for tag, (s, _) in cfgs.items():
        print(f"{tag:10s} {s['correct']:4d}/{s['n']:<5d} {s['s_per_correct']:10.1f} "
              f"{s['wall_s']:8.0f} {s['mean_gen_tokens']:9.0f} {s['mean_tok_s']:7.1f} "
              f"{s['truncated']:6d}")

    if "baseline" not in cfgs:
        return
    base_s, base_r = cfgs["baseline"]
    print("\npaired against baseline (McNemar, exact two-sided)")
    print(f"{'config':10s} {'base_only':>10s} {'cfg_only':>9s} {'p':>8s} "
          f"{'d_acc':>7s} {'tok infl':>9s} {'s/correct':>10s}")
    for tag, (s, rows) in cfgs.items():
        if tag == "baseline":
            continue
        ids = [i for i in base_r if i in rows]
        b = sum(1 for i in ids if base_r[i]["ok"] and not rows[i]["ok"])   # baseline only
        c = sum(1 for i in ids if rows[i]["ok"] and not base_r[i]["ok"])   # this config only
        p = mcnemar_exact(b, c)
        d_acc = (s["accuracy"] - base_s["accuracy"]) * 100

        # Token inflation on problems BOTH got right: difficulty and correctness held fixed, so
        # any change is the model needing more thinking to reach the same place.
        both = [i for i in ids if base_r[i]["ok"] and rows[i]["ok"]]
        gb = sum(base_r[i]["gen"] for i in both)
        gc = sum(rows[i]["gen"] for i in both)
        infl = (gc / gb - 1.0) * 100 if gb else 0.0
        d_spc = ((s["s_per_correct"] / base_s["s_per_correct"]) - 1.0) * 100

        print(f"{tag:10s} {b:10d} {c:9d} {p:8.4f} {d_acc:+6.1f}pp "
              f"{infl:+8.1f}% {d_spc:+9.1f}%")

    print("\n  base_only = baseline right, config wrong.  cfg_only = the reverse.")
    print("  p is on the discordant pairs only; a small p with base_only > cfg_only is a real loss.")
    print("  tok infl is measured on problems BOTH answered correctly.")
    print("  s/correct is the headline: negative is better (fewer seconds per correct answer).")


if __name__ == "__main__":
    main()
