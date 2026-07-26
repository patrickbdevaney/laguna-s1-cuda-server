#!/usr/bin/env python3
"""Separate the two length axes for speculative acceptance.

Holds the TASK constant and varies only the prompt length, by prepending a neutral filler
document. Any single generation confounds the axes -- generated position rises while prompt
length is fixed -- so the same question is asked at several input lengths and acceptance is
then binned by generated position within each.
"""
import json, sys, urllib.request

HOST = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 400

FILLER = ("The memory subsystem of a modern accelerator is organised as a hierarchy. "
          "Each level trades capacity against latency and bandwidth. Understanding where a "
          "workload sits in that hierarchy is the first step in any optimisation. ")
TASK = ("\n\nNow, ignoring the text above, write a detailed technical explanation of how "
        "memory bandwidth limits transformer inference at batch size one.")

def ask(pad_reps):
    body = {"messages": [{"role": "user", "content": FILLER * pad_reps + TASK}],
            "max_tokens": MAXTOK, "temperature": 0}
    req = urllib.request.Request(HOST + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=900))
    return d["usage"]["prompt_tokens"], d["timings"]["decode_tokens_per_second"], \
           d["timings"]["accepted_per_forward"]

for reps in (0, 20, 120, 400):
    p, tps, acc = ask(reps)
    print(f"  prompt={p:6d} tok   decode={tps:6.2f} tok/s   accepted/forward={acc:.2f}", flush=True)
