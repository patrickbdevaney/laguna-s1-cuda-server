#!/usr/bin/env python3
"""Capability eval scored as TIME-TO-CORRECT-ANSWER.

WHY THIS EXISTS. Two instruments were tried on the expert-bits question and both are unfit:

  * tau (draft acceptance) went UP when the target was degraded -- 2.396 -> 2.612 at k=2 for
    7 levels -- because a worse target is an easier target to imitate. It measures how
    predictable the model is, which anti-correlates with the thing we want to know.
  * The golden-continuation gate scored 7 levels 0/8 and 5 levels 8/8, though 5 levels is
    objectively 1.6x more distorting (mean rel delta 0.301 vs 0.188). It answers "is this the
    same model", not "is this model good", and README fact 13 says why: 1 ulp flips a token
    because top-10-of-256 sits on a knife edge.

So the metric has to be an end-to-end task with a checkable answer. And it has to be scored in
SECONDS, not accuracy alone, because the entire point of dropping bits is speed: a config that
is 15% faster per token but needs 30% more reasoning tokens to arrive is a LOSS, and an
accuracy-only score would hide that. Hence seconds-per-correct-answer as the headline:

    s_per_correct = total_wall_seconds / n_correct

which is the quantity a user actually experiences, and it moves the right way for both terms.
Accuracy, mean generated tokens and mean tok/s are all reported alongside so a change in the
headline can be attributed.

TRUNCATION IS SCORED AS WRONG, deliberately. A reasoning model that cannot reach an answer
inside the budget has failed at the task; counting it as "incomplete" and excluding it would
flatter exactly the degradation we are hunting for.

The comparison is baseline vs LG_EXPERT_LEVELS=7 (2.81b payload) vs 5 (2.32b). RTN inside the
NVFP4 container is a PESSIMISTIC proxy for an EXL3 trellis codec at similar payload bits --
vector quantization is strictly better per bit -- so read a pass as strong evidence for trellis
and a fail as inconclusive against it, not the reverse.
"""
import argparse, json, os, re, sys, time
import urllib.request

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

BOXED = re.compile(r"\\boxed\s*\{([^{}]*)\}")
ANSWER_LINE = re.compile(r"(?:final answer|answer)\s*(?:is)?\s*[:=]?\s*\**\s*([A-D0-9][^\n]*)", re.I)


def extract_aime(text):
    """AIME answers are integers 0-999. Prefer \\boxed{}, else the last standalone integer."""
    for m in reversed(BOXED.findall(text)):
        digits = re.sub(r"[^0-9]", "", m)
        if digits:
            return str(int(digits))
    m = ANSWER_LINE.findall(text)
    if m:
        digits = re.sub(r"[^0-9]", "", m[-1])
        if digits:
            return str(int(digits))
    nums = re.findall(r"\b\d{1,3}\b", text)
    return str(int(nums[-1])) if nums else None


def extract_mc(text):
    """GPQA answers are a letter A-D."""
    for m in reversed(BOXED.findall(text)):
        g = re.search(r"[A-D]", m.upper())
        if g:
            return g.group(0)
    for m in reversed(ANSWER_LINE.findall(text)):
        g = re.match(r"\s*([A-D])\b", m.upper())
        if g:
            return g.group(1)
    g = re.findall(r"\b([A-D])\b", text.upper())
    return g[-1] if g else None


def load_tasks(which, limit):
    import datasets
    if which == "gsm8k":
        # The statistically sharp instrument. AIME has 30 problems, so a 30-problem run has
        # +-9% binomial error and can only detect a catastrophic drop. GSM8K at n=150 gives
        # +-3%, and its traces are ~10x shorter, so it costs a third of the wall-clock for
        # several times the resolving power. AIME is kept as the hard-reasoning signal, not as
        # the primary discriminator.
        d = datasets.load_dataset("openai/gsm8k", "main")["test"]
        return _sub([dict(id=f"gsm8k-{i}",
                          prompt=r["question"].strip() +
                          "\n\nSolve it. Put your final numeric answer in \\boxed{}.",
                          gold=r["answer"].split("####")[-1].strip().replace(",", ""),
                          kind="aime")            # same integer extractor
                     for i, r in enumerate(d)], limit)
    if which == "aime":
        d = datasets.load_dataset("Maxwell-Jia/AIME_2024")["train"]
        out = [dict(id=r["ID"],
                    prompt=r["Problem"].strip() +
                    "\n\nSolve the problem. Put your final integer answer in \\boxed{}.",
                    gold=str(int(r["Answer"])), kind="aime") for r in d]
    else:
        d = datasets.load_dataset("fingertap/GPQA-Diamond")["test"]
        out = [dict(id=f"gpqa-{i}",
                    prompt=r["question"].strip() +
                    "\n\nThink it through, then give the letter of the correct option in \\boxed{}.",
                    gold=r["answer"].strip().upper()[:1], kind="gpqa")
               for i, r in enumerate(d)]
    return _sub(out, limit)


def _sub(out, limit):
    """Deterministic stride, not head: a head slice of GSM8K or GPQA is front-loaded on one
    topic and one difficulty, and every config must see the IDENTICAL subset or the comparison
    is between samples rather than between configs."""
    if limit and limit < len(out):
        step = len(out) / limit
        out = [out[int(i * step)] for i in range(limit)]
    return out


def ask(url, prompt, max_tokens, timeout):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--suite", choices=["aime", "gpqa", "gsm8k"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tasks = load_tasks(a.suite, a.limit)
    rows, wall = [], 0.0
    print(f"[{a.tag}/{a.suite}] {len(tasks)} problems, max_tokens={a.max_tokens}", flush=True)
    for i, t in enumerate(tasks):
        try:
            d, sec = ask(a.url, t["prompt"], a.max_tokens, a.timeout)
        except Exception as e:
            print(f"  {i+1:3d}/{len(tasks)} {t['id']:12s} REQUEST FAILED {type(e).__name__}: {e}",
                  flush=True)
            rows.append(dict(id=t["id"], ok=False, err=str(e), sec=0.0, gen=0))
            continue
        msg = d["choices"][0]["message"]
        # The answer can land in either field: reasoning models often box inside the reasoning
        # block and then restate. Search content first, fall back to reasoning.
        text = (msg.get("content") or "")
        reasoning = (msg.get("reasoning_content") or "")
        pred = (extract_aime(text) if t["kind"] == "aime" else extract_mc(text))
        if pred is None:
            pred = (extract_aime(reasoning) if t["kind"] == "aime" else extract_mc(reasoning))
        gen = d["usage"]["completion_tokens"]
        tps = d["timings"]["decode_tokens_per_second"]
        trunc = gen >= a.max_tokens
        # A truncated trace is WRONG, and its "prediction" is digits scraped from an unfinished
        # thought -- recording it invites reading a coincidence as a correct answer.
        if trunc:
            pred = None
        ok = (pred is not None) and (pred == t["gold"])
        wall += sec
        rows.append(dict(id=t["id"], ok=ok, pred=pred, gold=t["gold"], sec=sec, gen=gen,
                         tps=tps, trunc=trunc))
        print(f"  {i+1:3d}/{len(tasks)} {t['id']:12s} {'OK ' if ok else '   '} "
              f"pred={str(pred):>4s} gold={t['gold']:>4s} {gen:5d}tok {sec:6.1f}s "
              f"{tps:5.1f}tok/s{' TRUNC' if trunc else ''}", flush=True)

    n = len(rows)
    nc = sum(r["ok"] for r in rows)
    gens = [r["gen"] for r in rows if r["gen"]]
    tpss = [r["tps"] for r in rows if r.get("tps")]
    summary = dict(
        tag=a.tag, suite=a.suite, n=n, correct=nc,
        accuracy=nc / n if n else 0.0,
        wall_s=wall,
        s_per_correct=(wall / nc) if nc else float("inf"),
        mean_gen_tokens=(sum(gens) / len(gens)) if gens else 0,
        total_gen_tokens=sum(gens),
        mean_tok_s=(sum(tpss) / len(tpss)) if tpss else 0,
        truncated=sum(1 for r in rows if r.get("trunc")),
        max_tokens=a.max_tokens,
    )
    with open(a.out, "w") as f:
        json.dump(dict(summary=summary, rows=rows), f, indent=1)
    print(f"[{a.tag}/{a.suite}] acc {nc}/{n} = {summary['accuracy']:.3f}   "
          f"wall {wall:.0f}s   S/CORRECT {summary['s_per_correct']:.1f}   "
          f"mean_gen {summary['mean_gen_tokens']:.0f}tok   "
          f"{summary['mean_tok_s']:.1f}tok/s   trunc {summary['truncated']}", flush=True)


if __name__ == "__main__":
    main()
