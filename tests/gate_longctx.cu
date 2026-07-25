// Gate B1c: the long-prompt gate that WOULD have caught the sliding-ring aliasing bug.
//
// The bug: with cap == sliding_window, a batched forward past position `window` aliases
// tokens written later in the same batch into the read window. Decode (M=1) is provably
// immune -- 512 consecutive positions map to 512 distinct slots. So single-token decode is a
// trustworthy oracle for the batched path, with no Python and no golden tensors needed:
//
//   (a) prefill P tokens in batches of MAXTOK, then read the last logits
//   (b) feed the SAME P tokens one at a time, then read the last logits
//   they must agree.
//
// P is chosen > sliding_window so the aliasing window is actually exercised.
#include "../src/forward.cu"
#include "../include/tokenizer.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cmath>
using namespace laguna;

int main(int argc, char** argv) {
    std::string md = "models/Laguna-S-2.1-NVFP4";
    std::vector<int> PS;
    { const char* e = getenv("P"); std::string t = e ? e : "300,500,700";
      size_t p = 0; while (p < t.size()) { size_t q = t.find(',', p);
        PS.push_back(atoi(t.substr(p, q == std::string::npos ? q : q - p).c_str()));
        if (q == std::string::npos) break; p = q + 1; } }
    int P = PS[0];
    int CTX = getenv("CTX") ? atoi(getenv("CTX")) : 4096;
    Config c = load_config(md + "/config.json");
    bool FP8A = getenv("LG_FP8ATTN") != nullptr;
    Loader ld(md, c, FP8A); Weights W = ld.load("", false);
    int MT = getenv("MAXTOK") ? atoi(getenv("MAXTOK")) : 64;
    Engine E; E.c = c; E.W = W; E.init(MT);
    if (getenv("LG_NSP")) E.nsp_force = atoi(getenv("LG_NSP"));
    printf("[longctx] window=%d  MAXTOK=%d\n", c.sliding_window, E.MAXTOK);

    // Token stream. The default is uniform-random IDs, which is the WORST case for this
    // model: off-distribution input leaves the 256-way sigmoid router near-uniform, so the
    // top-10 selection sits on a knife edge and a 1-ulp perturbation flips experts. LG_TEXT
    // points at a file of real text and exercises the in-distribution case instead.
    int PMAX = 0; for (int x : PS) PMAX = std::max(PMAX, x);
    std::vector<int> ids;
    if (const char* tf = getenv("LG_TEXT")) {
        lgtok::Tokenizer tk(md + "/tokenizer.json");
        std::ifstream f(tf); std::stringstream ss; ss << f.rdbuf();
        std::string txt = ss.str();
        if (txt.empty()) { printf("[longctx] LG_TEXT file empty\n"); return 1; }
        while ((int)ids.size() < PMAX) {
            auto e = tk.encode(txt);
            if (e.empty()) break;
            ids.insert(ids.end(), e.begin(), e.end());
        }
        ids.resize(PMAX);
        printf("[longctx] token source: real text (%s)\n", tf);
    } else {
        ids.resize(PMAX);
        unsigned s = 12345;
        for (int i = 0; i < PMAX; ++i) { s = s * 1664525u + 1013904223u; ids[i] = (int)(s % c.vocab); }
        printf("[longctx] token source: uniform-random ids (worst case for routing)\n");
    }
    int* d; CUDA_CHECK(cudaMalloc(&d, (size_t)E.MAXTOK * 4));

    auto last_logits = [&](Session& S, bool batched) {
        int pos = 0;
        if (batched) {
            while (pos < P) {
                int n = std::min(E.MAXTOK, P - pos);
                CUDA_CHECK(cudaMemcpy(d, ids.data() + pos, n * 4, cudaMemcpyHostToDevice));
                set_base(E.dbase, pos, 0);
                E.forward(S, d, n, pos);
                pos += n;
            }
        } else {
            for (int i = 0; i < P; ++i) {
                CUDA_CHECK(cudaMemcpy(d, ids.data() + i, 4, cudaMemcpyHostToDevice));
                set_base(E.dbase, i, 0);
                E.forward(S, d, 1, i);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> lg(c.vocab);
        size_t off = batched ? (size_t)(P - 1 - (P - 1) / E.MAXTOK * E.MAXTOK) : 0;
        CUDA_CHECK(cudaMemcpy(lg.data(), E.logits + off * c.vocab, c.vocab * 4,
                              cudaMemcpyDeviceToHost));
        return lg;
    };

  bool allok = true;
  for (int Pi : PS) {
    P = Pi;
    printf("\n--- P=%d (%s window %d) ---\n", P, P > c.sliding_window ? "ABOVE" : "below",
           c.sliding_window);
    Session Sb; Sb.alloc(c, CTX, E.MAXTOK);
    printf("[longctx] batched prefill (M up to %d) ...\n", E.MAXTOK);
    E.dbg = getenv("LG_DBG") != nullptr; E.dbg_h.clear();
    E.dbg_row = (P - 1) - ((P - 1) / E.MAXTOK) * E.MAXTOK;
    auto lb = last_logits(Sb, true);
    auto hb_dbg = E.dbg_h;
    Sb.free_();

    Session Ss; Ss.alloc(c, CTX, E.MAXTOK);
    printf("[longctx] sequential single-token (the alias-free oracle) ...\n");
    E.dbg_h.clear(); E.dbg_row = 0;
    auto ls = last_logits(Ss, false);
    auto hs_dbg = E.dbg_h;
    Ss.free_();
    if (E.dbg) {
        // batched captured one row per layer per BATCH; keep only the final batch's layers
        size_t nl = c.n_layers;
        if (hb_dbg.size() >= nl && hs_dbg.size() >= nl) {
            auto B = std::vector<std::vector<float>>(hb_dbg.end() - nl, hb_dbg.end());
            auto S = std::vector<std::vector<float>>(hs_dbg.end() - nl, hs_dbg.end());
            printf("  %-6s %12s %12s\n", "layer", "maxabs", "rel");
            for (size_t L = 0; L < nl; ++L) {
                double m2 = 0, s2 = 0;
                for (size_t i = 0; i < B[L].size(); ++i) {
                    m2 = std::max(m2, (double)std::fabs(B[L][i] - S[L][i]));
                    s2 = std::max(s2, (double)std::fabs(S[L][i]));
                }
                double r2 = m2 / std::max(s2, 1e-9);
                printf("  %-6zu %12.4e %12.4e%s\n", L, m2, r2,
                       (m2 != 0.0 && L == 0) ? "  <-- first divergence" : "");
            }
        }
    }

    double mx = 0, scale = 0;
    for (int i = 0; i < c.vocab; ++i) { mx = std::max(mx, (double)std::fabs(lb[i] - ls[i]));
                                        scale = std::max(scale, (double)std::fabs(ls[i])); }
    int ab = (int)(std::max_element(lb.begin(), lb.end()) - lb.begin());
    int as = (int)(std::max_element(ls.begin(), ls.end()) - ls.begin());
    double rel = mx / std::max(scale, 1e-9);
    printf("[longctx] last-logit maxabs=%.4e rel=%.4e   argmax batched=%d sequential=%d\n",
           mx, rel, ab, as);
    bool ok = (ab == as) && rel < 5e-4;
    printf("[longctx] %s\n", ok ? "PASS" : "FAIL - the batched path disagrees with decode");
    allok &= ok;
  }
    printf("\n%s\n", allok ? "ALL PASS" : "FAILURES PRESENT");
    return allok ? 0 : 1;
}
