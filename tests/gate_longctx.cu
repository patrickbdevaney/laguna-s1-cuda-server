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
#include <algorithm>
#include <cmath>
using namespace laguna;

int main(int argc, char** argv) {
    std::string md = "models/Laguna-S-2.1-NVFP4";
    int P   = getenv("P")   ? atoi(getenv("P"))   : 700;    // > 512
    int CTX = getenv("CTX") ? atoi(getenv("CTX")) : 4096;
    Config c = load_config(md + "/config.json");
    bool FP8A = getenv("LG_FP8ATTN") != nullptr;
    Loader ld(md, c, FP8A); Weights W = ld.load("", false);
    Engine E; E.c = c; E.W = W; E.init(64);
    printf("[longctx] P=%d  window=%d  MAXTOK=%d\n", P, c.sliding_window, E.MAXTOK);

    // deterministic pseudo-random token stream
    std::vector<int> ids(P);
    unsigned s = 12345;
    for (int i = 0; i < P; ++i) { s = s * 1664525u + 1013904223u; ids[i] = (int)(s % c.vocab); }
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

    Session Sb; Sb.alloc(c, CTX, E.MAXTOK);
    printf("[longctx] batched prefill (M up to %d) ...\n", E.MAXTOK);
    auto lb = last_logits(Sb, true);
    Sb.free_();

    Session Ss; Ss.alloc(c, CTX, E.MAXTOK);
    printf("[longctx] sequential single-token (the alias-free oracle) ...\n");
    auto ls = last_logits(Ss, false);
    Ss.free_();

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
    return ok ? 0 : 1;
}
