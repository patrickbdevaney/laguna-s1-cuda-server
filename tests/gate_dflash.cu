// Gate D1 — DFlash speculative decoding: correctness, acceptance rate, and the k sweep.
//
// Two things must both be true, and they are independent:
//
//   1. CORRECTNESS. Greedy speculative decoding must emit the EXACT token stream that greedy
//      autoregressive decoding emits. Speculation is only a scheduling trick; if the output
//      differs at all, the acceptance rule is wrong. This gate compares against the same
//      golden continuation the non-speculative gates use, so a regression here is
//      unambiguous. Gate B1c is what makes this test meaningful: the verify pass runs at
//      M=k+1 and decode runs at M=1, and those were only made bit-identical by the
//      attend_nsplit fix.
//
//   2. ACCEPTANCE. tau = mean accepted tokens per target forward. poolside publish 6.44 on
//      HumanEval and ~4.0-5.8 elsewhere at k=15. If our tau came out near 1 the draft
//      pipeline would be wired up wrong (most likely the tap placement or the aux-norm
//      order) even though correctness would still pass -- rejection makes a broken draft
//      invisible to a correctness test. So tau is a gate, not a statistic.
//
// The k sweep then answers the question the roofline poses: k=15 is what the model card
// recommends, but each extra draft token costs target bytes in the verify pass, and on a
// 254 GB/s part the optimum is much smaller than on a datacentre GPU.
#include "../src/forward.cu"
#include "../src/draft.cu"
#include "../include/tokenizer.h"
#include <algorithm>
#include <fstream>
#include <sstream>
using namespace laguna;

int main(int argc, char** argv) {
    std::string md = "models/Laguna-S-2.1-NVFP4";
    std::string dd = "models/Laguna-S-2.1-DFlash-NVFP4";
    int CTX = getenv("CTX") ? atoi(getenv("CTX")) : 4096;
    int NGEN = getenv("NGEN") ? atoi(getenv("NGEN")) : 64;
    bool FP8A = getenv("LG_FP8ATTN") != nullptr;
    std::vector<int> KS;
    { const char* e = getenv("K"); std::string t = e ? e : "1,2,3,4,5,6,7,9,11,15";
      size_t p = 0; while (p < t.size()) { size_t q = t.find(',', p);
        KS.push_back(atoi(t.substr(p, q == std::string::npos ? q : q - p).c_str()));
        if (q == std::string::npos) break; p = q + 1; } }

    Config c = load_config(md + "/config.json");
    DraftConfig dc = load_draft_config(dd + "/config.json");
    const int BLK = dc.block_size;
    for (int k : KS)
        if (k < 1 || k > BLK - 1) { printf("k=%d outside [1,%d]\n", k, BLK - 1); return 1; }

    Loader ld(md, c, FP8A); Weights W = ld.load("", false);
    DraftLoader dld(dd, dc); DraftWeights DW = dld.load(true);

    Engine E; E.c = c; E.W = W; E.init(BLK);
    // 128 rows of context scratch: enough to fill the 512-wide draft window in four
    // chunks without sizing the MLP buffers for 512 rows.
    Drafter D; D.d = dc; D.W = DW; D.init(getenv("DROW") ? atoi(getenv("DROW")) : 128);

    // Tap ring: the draft's window is 512 on every layer, so nothing older can ever be
    // attended and the ring is that plus one block.
    E.tap_cap = dc.sliding_window + BLK;
    E.tap_of.assign(c.n_layers, -1);
    for (size_t i = 0; i < dc.target_layer_ids.size(); ++i) {
        int L = dc.target_layer_ids[i];
        if (L < 0 || L >= c.n_layers) { printf("tap layer %d out of range\n", L); return 1; }
        E.tap_of[L] = (int)i;
    }
    CUDA_CHECK(cudaMalloc(&E.taps, (size_t)dc.target_layer_ids.size() * E.tap_cap * c.hidden * 4));
    printf("[D1] taps %.1f MB, draft KV %.1f MB (constant in context length)\n",
           (double)dc.target_layer_ids.size() * E.tap_cap * c.hidden * 4 / 1e6,
           (double)dc.n_layers * 2 * D.cap * dc.n_kv_heads * dc.head_dim / 1e6);

    std::ifstream mf("docs/golden/meta_primes.json"); nlohmann::json M; mf >> M;
    std::vector<int> prompt; for (auto& x : M["ids"]) prompt.push_back(x.get<int>());
    std::vector<int> want;   for (auto& x : M["greedy_ids"]) want.push_back(x.get<int>());

    int* d_tok; CUDA_CHECK(cudaMalloc(&d_tok, (size_t)BLK * 4));
    std::vector<float> lg(c.vocab);
    auto argmax_row = [&](int row) {
        CUDA_CHECK(cudaMemcpy(lg.data(), E.logits + (size_t)row * c.vocab, c.vocab * 4,
                              cudaMemcpyDeviceToHost));
        int b = 0; float bv = lg[0];
        for (int i = 1; i < c.vocab; ++i) if (lg[i] > bv) { bv = lg[i]; b = i; }
        return b;
    };

    printf("\n  %-4s %10s %10s %12s %10s %10s\n", "k", "tok/s", "tau", "target fwd", "gen", "match");
    double best_tps = 0; int best_k = 0;
    for (int k : KS) {
        // ---- fresh session, prefill the prompt (this also fills the tap ring) ----
        Session S; S.alloc(c, CTX, BLK);
        for (auto p : D.Kc) CUDA_CHECK(cudaMemset(p, 0, (size_t)D.kvd() * D.cap));
        for (auto p : D.Vc) CUDA_CHECK(cudaMemset(p, 0, (size_t)D.kvd() * D.cap));

        int pos = 0;
        for (size_t i = 0; i < prompt.size(); i += BLK) {
            int n = (int)std::min((size_t)BLK, prompt.size() - i);
            CUDA_CHECK(cudaMemcpy(d_tok, prompt.data() + i, n * 4, cudaMemcpyHostToDevice));
            set_base(E.dbase, pos, 0);
            E.forward(S, d_tok, n, pos);
            pos += n;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        int next = argmax_row((int)(prompt.size() - 1) % BLK == 0 && prompt.size() % BLK == 0
                              ? BLK - 1 : (int)((prompt.size() - 1) % BLK));
        std::vector<int> got{next};

        // Context K/V for every position the draft can still see.
        int c0 = std::max(0, pos - dc.sliding_window);
        D.context_kv(E.taps, E.tap_cap, c0, pos - c0);
        CUDA_CHECK(cudaDeviceSynchronize());

        // ---- speculative loop ----
        std::vector<int> draft(k), blk(BLK);
        long target_fwd = 0, accepted_total = 0;
        double t0 = wall_now();
        while ((int)got.size() < NGEN) {
            // propose k tokens conditioned on the committed token `next` at position `pos`
            D.propose(next, pos, k, W.embed, W.lm_head, draft.data());

            // verify: one target forward over [next, draft...] at positions pos..pos+k
            blk[0] = next;
            for (int i = 0; i < k; ++i) blk[1 + i] = draft[i];
            CUDA_CHECK(cudaMemcpy(d_tok, blk.data(), (k + 1) * 4, cudaMemcpyHostToDevice));
            set_base(E.dbase, pos, 0);
            E.forward(S, d_tok, k + 1, pos);
            CUDA_CHECK(cudaDeviceSynchronize());
            ++target_fwd;

            // Longest-prefix greedy acceptance. Row j of the verify forward is the target's
            // distribution for the token AFTER blk[j], so target_out[j] is what the target
            // itself would have emitted at position pos+j+1. Accept draft[j] iff it equals
            // target_out[j]; the first mismatch is replaced by the target's own token, which
            // is the "bonus" -- so at least one token is always committed and the loop cannot
            // stall.
            int nacc = 0;
            std::vector<int> tout(k + 1);
            for (int j = 0; j <= k; ++j) tout[j] = argmax_row(j);
            while (nacc < k && draft[nacc] == tout[nacc]) ++nacc;

            if (getenv("LG_DBG") && target_fwd <= 5) {
                printf("    [dbg] pos=%d next=%d  draft:", pos, next);
                for (int j = 0; j < k; ++j) printf(" %d", draft[j]);
                printf("   target:");
                for (int j = 0; j <= k; ++j) printf(" %d", tout[j]);
                printf("   nacc=%d\n", nacc);
            }
            for (int j = 0; j < nacc; ++j) got.push_back(draft[j]);
            got.push_back(tout[nacc]);              // the bonus / correction token
            accepted_total += nacc + 1;

            // Commit exactly nacc+1 positions. The verify forward wrote KV for all k+1, so
            // the rejected tail's KV is stale -- harmless, because the next forward starts at
            // the committed position and overwrites those slots before anything reads them.
            pos += nacc + 1;
            next = tout[nacc];

            // The draft's context for the newly committed positions comes from THIS forward's
            // taps: rows 0..nacc of the verify block are the accepted tokens.
            D.context_kv(E.taps, E.tap_cap, pos - (nacc + 1), nacc + 1);
            if ((int)got.size() >= NGEN) break;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        double secs = wall_now() - t0;

        size_t ncmp = std::min(got.size(), want.size());
        int match = 0; for (size_t i = 0; i < ncmp; ++i) { if (got[i] == want[i]) ++match; else break; }
        double tps = got.size() / secs;
        double tau = (double)accepted_total / (double)target_fwd;
        printf("  %-4d %10.2f %10.3f %12ld %10zu %6d/%zu%s\n", k, tps, tau, target_fwd,
               got.size(), match, ncmp, (size_t)match == ncmp ? "" : "  <-- MISMATCH");
        if (tps > best_tps) { best_tps = tps; best_k = k; }
        S.free_();
    }
    printf("\n[D1] best k = %d at %.2f tok/s\n", best_k, best_tps);
    return 0;
}
