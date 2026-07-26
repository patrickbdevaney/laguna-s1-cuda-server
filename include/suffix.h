// suffix.h — SuffixDecoding: a drafter that reads no weights at all.
//
// The observation is that agentic and coding output massively re-copies its own context —
// file paths, diffs, tool schemas, repeated JSON, edited-and-restated code. If the next tokens
// have already appeared in this conversation, a suffix match proposes them for free.
//
// Why this matters here specifically: DFlash costs 2.23 GB of draft weights plus an lm_head
// read on every propose, and Gate D1 measured that a verify forward costs ~3x a decode step
// because a k+1 token block routes ~4x the distinct experts. So speculation only pays when
// tau clears ~3 — and after the FP8 work raised base decode to 33 tok/s, that bar went up
// again. A drafter whose marginal cost is a hash lookup changes the arithmetic completely:
// it only has to beat 1.0 accepted tokens, not 3.
//
// Published measurements put mean accepted length at ~7.8 on SWE-Bench-class traffic, and
// note that the linear draft performs about as well as the tree — which also sidesteps the
// expert-union problem, since a linear block is what we already verify.
//
// Exactness: this proposes tokens and nothing else. The target's longest-prefix greedy check
// is unchanged, so accepted output is bit-identical to autoregressive decode. A bad guess
// costs a rejected token, never a wrong one.
#pragma once
#include <cstdint>
#include <vector>
#include <unordered_map>

namespace lgsuffix {

// Longest n-gram we index. Longer matches are far more likely to continue correctly, so the
// lookup tries long before short.
static const int MAXL = 12;
static const int MINL = 3;

class SuffixIndex {
public:
    void reset(size_t reserve_hint = 4096) {
        for (int L = MINL; L <= MAXL; ++L) { idx_[L].clear(); idx_[L].reserve(reserve_hint); }
        indexed_ = 0;
    }

    // Index every n-gram ENDING at positions [indexed_, n). Called after tokens are committed.
    // Cost is O(new tokens x (MAXL-MINL)) hash inserts — a few hundred nanoseconds per token
    // against a 30 ms decode step.
    // Indexes up to but NOT including the last position. If the current suffix is in the
    // index, draft() finds the current position as its own "earlier occurrence" and falls
    // through every length, so the drafter silently never fires -- which is exactly what it
    // did until this was caught: 12 forced trials, zero accepted tokens, p[0] stuck at 0.
    void extend(const std::vector<int>& seq) {
        const size_t n = seq.size();
        if (n < 2) return;
        for (size_t p = indexed_; p + 1 < n; ++p) {
            for (int L = MINL; L <= MAXL; ++L) {
                if (p + 1 < (size_t)L) continue;
                idx_[L][hash(seq, p + 1 - L, L)] = (int)p;   // most recent occurrence wins
            }
        }
        indexed_ = n - 1;
    }

    // Propose up to `k` continuations of the current suffix. Returns the matched length (0 if
    // nothing usable). Longest match first: a 12-gram repeat is far stronger evidence than a
    // 3-gram one, and trying short first would drown the signal in coincidences.
    int draft(const std::vector<int>& seq, int k, int* out) const {
        const size_t n = seq.size();
        for (int L = MAXL; L >= MINL; --L) {
            if (n < (size_t)L) continue;
            auto it = idx_[L].find(hash(seq, n - L, L));
            if (it == idx_[L].end()) continue;
            const size_t q = (size_t)it->second;             // an earlier end of the same L-gram
            if (q + 1 >= n) continue;                        // that was the current position
            int m = 0;
            while (m < k && q + 1 + m < n) out[m] = seq[q + 1 + m], ++m;
            if (m > 0) return L;
        }
        return 0;
    }

private:
    static uint64_t hash(const std::vector<int>& s, size_t off, int L) {
        uint64_t h = 1469598103934665603ULL;                 // FNV-1a over the token ids
        for (int i = 0; i < L; ++i) {
            h ^= (uint64_t)(uint32_t)s[off + i];
            h *= 1099511628211ULL;
        }
        return h;
    }
    std::unordered_map<uint64_t, int> idx_[MAXL + 1];
    size_t indexed_ = 0;
};

} // namespace lgsuffix
