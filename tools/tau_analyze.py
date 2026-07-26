#!/usr/bin/env python3
"""Bin the tau log by prompt length and by generated position, independently."""
import sys, collections
rows = []
for line in open(sys.argv[1]):
    if line.startswith("#"): continue
    try: plen, gpos, src, k, nacc = (int(x) for x in line.strip().split(","))
    except ValueError: continue
    rows.append((plen, gpos, src, k, nacc))
if not rows: sys.exit("no rows")

def plen_bucket(p):
    for b in (256, 1024, 4096, 16384):
        if p <= b: return b
    return 65536
def gpos_bucket(g): return (g // 100) * 100

print(f"{len(rows)} speculative steps\n")
print("acceptance by PROMPT LENGTH (all generated positions pooled)")
by_p = collections.defaultdict(list)
for plen, gpos, src, k, nacc in rows: by_p[plen_bucket(plen)].append(nacc)
for b in sorted(by_p):
    v = by_p[b]; print(f"  prompt<={b:6d}   n={len(v):5d}   mean accepted={sum(v)/len(v):.3f}")

print("\nacceptance by GENERATED POSITION, within each prompt-length bucket")
by_pg = collections.defaultdict(list)
for plen, gpos, src, k, nacc in rows: by_pg[(plen_bucket(plen), gpos_bucket(gpos))].append(nacc)
for pb in sorted(set(k[0] for k in by_pg)):
    cells = sorted((g, by_pg[(pb, g)]) for (p, g) in by_pg if p == pb)
    line = "  ".join(f"{g}:{sum(v)/len(v):.2f}(n={len(v)})" for g, v in cells if len(v) >= 5)
    print(f"  prompt<={pb:6d}  {line}")
