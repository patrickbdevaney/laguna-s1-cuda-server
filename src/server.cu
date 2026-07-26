// server.cu — Gate S1. OpenAI-compatible HTTP endpoint, in one process, no Python anywhere.
//
// Everything the request path touches is already gated:
//   tokenizer.h   ByteLevel BPE, 18/18 against HF
//   chat.h        poolside_v1 render + reasoning/tool parsing, 10/10 byte-exact vs Jinja
//   forward.cu    the target, greedy-exact against the oracle
//   draft.cu      DFlash, tau 4.32, greedy-identical output
//
// Concurrency: one session at a time, serialised on a mutex. That is a deliberate choice, not
// a shortcut. The model is 69 GB of weights on a 117 GB unified-memory part and decode is
// bandwidth-bound at batch 1; a second concurrent sequence would not add throughput, it would
// halve the per-token latency of both. Requests queue.
//
// Prefix cache: the KV of the last conversation is kept, and a new request only prefills the
// tokens past the longest common prefix. This is what makes multi-turn chat and agentic
// loops usable -- the 40th turn of a conversation prefills the new turn, not the whole
// history. Correctness rests on two properties established earlier: the sliding ring's read
// arc only ever looks BACKWARD, so slots holding stale future positions are never read
// (the same argument as the cap = window + MAXTOK fix), and the global layers are indexed by
// absolute position so a rewind simply overwrites.
#include <cstdio>
#include <cstring>
#include <mutex>
#include <atomic>
#include <string>
#include <vector>
#include <chrono>
#include "../src/forward.cu"
#include "../src/draft.cu"
#include "../include/tokenizer.h"
#include "../include/chat.h"
#include "../include/suffix.h"
#include "../include/third_party/httplib.h"
#include "../include/webui.h"

using namespace laguna;
using lgchat::ojson;

// ------------------------------------------------------------------ incremental UTF-8
// Token boundaries do not respect UTF-8 boundaries. Streaming the raw decode of each token
// would emit broken bytes mid-codepoint, which browsers render as replacement characters and
// which corrupts any client doing its own UTF-8 handling. Hold back a trailing partial
// sequence until the continuation bytes arrive.
struct Utf8Stream {
    std::string pending;
    std::string feed(const std::string& s) {
        pending += s;
        size_t cut = pending.size();
        // walk back over at most 3 continuation bytes to find an incomplete lead byte
        for (size_t back = 1; back <= 4 && back <= pending.size(); ++back) {
            unsigned char c = (unsigned char)pending[pending.size() - back];
            if ((c & 0xC0) == 0x80) continue;                 // continuation, keep walking
            int need = (c & 0x80) == 0x00 ? 1 : (c & 0xE0) == 0xC0 ? 2
                     : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
            if ((int)back < need) cut = pending.size() - back; // incomplete: hold it back
            break;
        }
        std::string out = pending.substr(0, cut);
        pending.erase(0, cut);
        return out;
    }
    std::string flush() { std::string o = pending; pending.clear(); return o; }
};

// ------------------------------------------------------------------ speculation policy
//
// This replaces a 4-arm throughput bandit, and the reason is worth recording because the
// bandit was not obviously wrong.
//
// The bandit ranked whole arms by *realised tokens/second*. One speculative step yields
// between 1 and k+1 tokens, so a single sample spans the entire acceptance distribution --
// ranking on it thrashed, and blocking 32 steps per arm to average it out made exploration
// cost ~30 % of a block and left it lagging behind acceptance that changes mid-generation.
//
// The fix is to stop measuring the thing with high variance and start measuring the thing
// with low variance. Every verify already tells us, for free, whether position i was accepted
// GIVEN that 0..i-1 were. Those are Bernoulli observations, one per position, and a single
// k=5 step yields five of them. From per-position acceptance p[i] the expected yield of ANY k
// follows in closed form:
//
//     E[tokens | k] = 1 + p0 + p0·p1 + … + Π_{i<k} p_i
//     k*            = argmax_k  E[tokens | k] / cost(k)
//
// Three consequences, each fixing one of the bandit's observed failures:
//   * a step at ANY k updates the estimate for EVERY k, so exploration cost goes to zero;
//   * ~5x the signal per step, so no 32-step commitment is needed;
//   * an EWMA on p[i] re-converges in tens of steps rather than hundreds.
//
// `cost(k)` is profiled once at boot rather than learned. Cohere measured adjacent-token
// expert overlap at 0.381 across all 13 Spec-Bench categories and seven languages -- routing
// correlation is a structural property of MoE, not of the prompt -- so the cost of verifying
// k+1 tokens does not depend on the workload, only the acceptance does.
//
// Two drafters compete on the same footing. DFlash pays 2.23 GB of weights per propose;
// SuffixDecoding pays a hash lookup. They keep separate p[] tables because their acceptance
// is completely differently distributed, but they share cost_verify(k).
struct SpecPolicy {
    // block_size-1: DFlash emits the whole block in one forward, so a larger k costs the
    // draft nothing extra -- only the verify grows. On repetitive output tau reaches 8.5,
    // which was clipped by an earlier cap of 8.
    static const int KMAX = 15;
    enum Src { AR = 0, DFLASH = 1, SUFFIX = 2 };

    // per-position conditional acceptance, per drafter
    double p[3][KMAX] = {};
    long   np[3][KMAX] = {};

    // profiled costs, seconds
    double cost_ar = 0;                        // one M=1 graph step
    double cost_verify[KMAX + 1] = {};         // one M=k+1 target forward
    double cost_draft_dflash = 0;              // one propose (~flat in k)
    bool   profiled = false;
    int    boot = 0;                           // one-time cost sweep, KMAX+1 steps

    // Until every cost_verify[k] has been observed once, walk k deterministically instead of
    // choosing. This costs KMAX+1 steps ONCE per server lifetime -- against a bandit that paid
    // an exploration tax forever -- and after it the controller never needs to explore again,
    // because a step at any k informs the estimate for every k.
    // The boot sweep is [AR x AR_SAMPLES] then [k=1..KMAX]. Several AR steps, not one,
    // because the FIRST AR step also captures the CUDA graph and is therefore ~1.7x a steady
    // one. Measuring cost_ar from that single sample rated AR at 19.7 tok/s against a true 33,
    // and since the controller then never CHOSE AR, the estimate never got a chance to
    // correct itself -- an arm that is never selected is never re-measured. Any cost that
    // feeds an argmax has to be established before the argmax can starve it.
    static const int AR_SAMPLES = 4;
    bool bootstrapping() const { return !profiled; }
    int  boot_k() const { return boot < AR_SAMPLES ? 0 : boot - AR_SAMPLES + 1; }
    bool boot_record_ar() const { return boot > 0 && boot < AR_SAMPLES; }  // skip the capture

    void observe(Src s, int k, int nacc) {
        // positions 0..nacc-1 accepted; position nacc rejected (unless the block ran out)
        for (int i = 0; i < nacc && i < KMAX; ++i) obs(s, i, 1.0);
        if (nacc < k && nacc < KMAX) obs(s, nacc, 0.0);
    }
    void obs(Src s, int i, double v) {
        p[s][i] = np[s][i] ? 0.92 * p[s][i] + 0.08 * v : v;
        ++np[s][i];
    }

    double yield(Src s, int k) const {
        double e = 1.0, run = 1.0;             // the bonus token is always committed
        for (int i = 0; i < k; ++i) {
            // an unobserved position is optimistic-but-bounded, so a fresh k is tried once
            // and then judged on evidence rather than on a prior
            run *= np[s][i] ? p[s][i] : 0.5;
            e += run;
        }
        return e;
    }
    double rate(Src s, int k) const {
        if (s == AR) return cost_ar > 0 ? 1.0 / cost_ar : 0.0;
        const double c = cost_verify[k] + (s == DFLASH ? cost_draft_dflash : 0.0);
        return c > 0 ? yield(s, k) / c : 0.0;
    }

    // Best (source, k). `suffix_k` is how many tokens the suffix index can actually offer
    // this step -- 0 means that arm does not exist right now.
    // A source that is never chosen is never measured, and an unobserved position priced at
    // 0.5 put SuffixDecoding at 31.5 tok/s against AR's 33.3 -- just below, forever. Give each
    // source a bounded number of forced trials so the argmax runs on evidence rather than on
    // a prior. TRIALS is per server lifetime, not per request.
    static const int TRIALS = 12;
    bool needs_trial(Src s) const { return np[s][0] < TRIALS; }

    void choose(int suffix_k, Src* s_out, int* k_out) const {
        if (suffix_k > 0 && needs_trial(SUFFIX)) {
            *s_out = SUFFIX; *k_out = std::min(3, suffix_k); return;
        }
        Src bs = AR; int bk = 0; double br = rate(AR, 0);
        for (int k = 1; k <= KMAX; ++k) {
            if (cost_verify[k] <= 0) continue;
            double r = rate(DFLASH, k);
            if (r > br) { br = r; bs = DFLASH; bk = k; }
            if (k <= suffix_k) {
                r = rate(SUFFIX, k);
                if (r > br) { br = r; bs = SUFFIX; bk = k; }
            }
        }
        *s_out = bs; *k_out = bk;
    }

    std::string report() const {
        char b[256];
        Src s; int k; choose(KMAX, &s, &k);
        snprintf(b, sizeof b,
                 "ar=%.1f dflash_best=%.1f@k%d suffix_best=%.1f@k%d p_df=%.2f/%.2f/%.2f "
                 "p_sfx=%.2f/%.2f/%.2f",
                 rate(AR, 0), best_rate(DFLASH), best_k(DFLASH), best_rate(SUFFIX),
                 best_k(SUFFIX), p[DFLASH][0], p[DFLASH][1], p[DFLASH][2],
                 p[SUFFIX][0], p[SUFFIX][1], p[SUFFIX][2]);
        return b;
    }
    int best_k(Src s) const {
        int bk = 1; double br = 0;
        for (int k = 1; k <= KMAX; ++k)
            if (cost_verify[k] > 0 && rate(s, k) > br) { br = rate(s, k); bk = k; }
        return bk;
    }
    double best_rate(Src s) const { return rate(s, best_k(s)); }
};

// ------------------------------------------------------------------ engine wrapper
struct Server {
    Config c;
    DraftConfig dc;
    Weights W;
    DraftWeights DW;
    Engine E;
    Drafter D;
    Session S;
    lgtok::Tokenizer* tok = nullptr;
    std::mutex mu;

    int CTX = 262144;
    int SPEC_K = 3;                 // Gate D1 measured k* = 3-4 on this part
    bool use_spec = true;

    SpecPolicy policy;
    lgsuffix::SuffixIndex sfx;
    // How far the draft's context K/V have been projected. NOT a boolean: with a per-step
    // controller the mode can change every token, and a boolean meant every AR step
    // invalidated the whole thing, so the next DFlash step rebuilt the entire 512-wide window
    // (four chunks x an fc GEMM of 113 MB plus six layers of k/v). That made AR look 30 %
    // more expensive than it is and pushed the controller off it on prose. Tracking the
    // POSITION bounds the rebuild to exactly the tokens that were committed while the draft
    // was not being consulted -- usually a handful.
    //
    // The taps themselves are always fresh: they are written by the target forward and cost
    // nothing, so only the projection can fall behind.
    int draft_ctx_pos = 0;

    // prefix cache
    std::vector<int> cached;        // token ids currently represented in S's KV
    int pos = 0;

    std::vector<float> logits_host;

    void boot(const std::string& md, const std::string& dd, bool fp8a, int ctx, int k,
              bool fp8l, bool fp8d) {
        CTX = ctx; SPEC_K = k;
        c  = load_config(md + "/config.json");
        dc = load_draft_config(dd + "/config.json");
        use_spec = SPEC_K > 0;
        if (SPEC_K > dc.block_size - 1) SPEC_K = dc.block_size - 1;

        Loader ld(md, c, fp8a, fp8l, fp8d);
        W = ld.load("", false);
        E.c = c; E.W = W; E.init(dc.block_size);
        S.alloc(c, CTX, dc.block_size);

        if (use_spec) {
            DraftLoader dld(dd, dc);
            DW = dld.load(false);
            D.d = dc; D.W = DW; D.init(128);
            E.tap_cap = dc.sliding_window + dc.block_size;
            E.tap_of.assign(c.n_layers, -1);
            for (size_t i = 0; i < dc.target_layer_ids.size(); ++i)
                E.tap_of[dc.target_layer_ids[i]] = (int)i;
            CUDA_CHECK(cudaMalloc(&E.taps, (size_t)dc.target_layer_ids.size() *
                                           E.tap_cap * c.hidden * 4));
        }
        tok = new lgtok::Tokenizer(md + "/tokenizer.json");
        logits_host.resize(c.vocab);
        printf("[server] ctx=%d  KV=%.2f GB  spec k=%d  weights=%.2f GB%s\n",
               CTX, S.bytes / 1e9, use_spec ? SPEC_K : 0, W.arena_bytes / 1e9,
               use_spec ? "" : "  (speculation off)");
    }

    int argmax_row(int row) {
        CUDA_CHECK(cudaMemcpy(logits_host.data(), E.logits + (size_t)row * c.vocab,
                              c.vocab * 4, cudaMemcpyDeviceToHost));
        int b = 0; float bv = logits_host[0];
        for (int i = 1; i < c.vocab; ++i) if (logits_host[i] > bv) { bv = logits_host[i]; b = i; }
        return b;
    }

    // Sampling. Greedy when temperature <= 0, else top-p over a temperature-scaled softmax.
    // The temperature-scaled, top-p-filtered distribution for one logits row. `keep` is the
    // nucleus size and `order` is the descending permutation; both are needed by the
    // rejection sampler, not just by plain sampling.
    std::vector<float> pbuf;
    std::vector<int>    obuf;
    size_t nucleus_ = 0;

    // Temperature-scaled probabilities for one row. O(vocab), no ordering.
    //
    // The ordering is separated out deliberately: the acceptance test needs only p(x) for one
    // token, and sorting 100 352 logits to obtain it cost ~23 ms per token and dragged
    // sampled decode from 33 tok/s to 18.9 -- slower than not speculating at all. Sort only
    // when a sample actually has to be drawn, and even then only over a bounded candidate set.
    void row_probs(int row, double temp) {
        CUDA_CHECK(cudaMemcpy(logits_host.data(), E.logits + (size_t)row * c.vocab,
                              c.vocab * 4, cudaMemcpyDeviceToHost));
        // float, not double, and one pass: this runs on EVERY sampled token over a 100352
        // vocabulary, and in double with a separate assign+normalise pass it cost ~20 ms,
        // dragging the autoregressive arm from 33 tok/s to 19.8 -- so sampled requests were
        // paying more for the sampler than for the model. Normalisation folds into the
        // consumer instead of a second sweep.
        pbuf.resize(c.vocab);
        const float mx = *std::max_element(logits_host.begin(), logits_host.end());
        const float invT = 1.0f / (float)temp;
        float sum = 0.f;
        for (int i = 0; i < c.vocab; ++i) {
            const float e = __builtin_expf((logits_host[i] - mx) * invT);
            pbuf[i] = e; sum += e;
        }
        psum_ = sum;
    }
    float psum_ = 1.f;

    // Build the top-p nucleus over `pbuf`. With top_p ~ 1 there is nothing to do -- the whole
    // distribution is the nucleus -- which is the common case and costs nothing. Otherwise
    // partially select a bounded candidate set rather than sorting the vocabulary: a nucleus
    // wide enough to exceed CAND tokens is vanishingly rare, and the fallback covers it.
    void build_nucleus(double top_p) {
        if (top_p >= 0.999) { nucleus_ = 0; return; }              // 0 = "everything"
        // NOTE: only the CANDIDATE set is zeroed outside the nucleus; tokens outside the
        // candidate set keep stale weights but are never reachable, because every consumer
        // walks obuf[0..nucleus_) once nucleus_ != 0.
        const int CAND = 2048;
        obuf.resize(c.vocab);
        for (int i = 0; i < c.vocab; ++i) obuf[i] = i;
        std::nth_element(obuf.begin(), obuf.begin() + CAND, obuf.end(),
                         [&](int a, int b) { return pbuf[a] > pbuf[b]; });
        obuf.resize(CAND);
        std::sort(obuf.begin(), obuf.end(), [&](int a, int b) { return pbuf[a] > pbuf[b]; });
        double cum = 0; size_t keep = obuf.size();
        for (size_t i = 0; i < obuf.size(); ++i) {
            cum += pbuf[obuf[i]];
            if (cum >= top_p) { keep = i + 1; break; }
        }
        float z = 0;
        for (size_t i = 0; i < keep; ++i) z += pbuf[obuf[i]];
        for (size_t i = keep; i < obuf.size(); ++i) pbuf[obuf[i]] = 0.f;
        psum_ = z;
        nucleus_ = keep;
    }

    double uniform(uint64_t& rng) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        return (double)((rng >> 11) & ((1ULL << 53) - 1)) / (double)(1ULL << 53);
    }
    // Walks `pbuf` directly when the nucleus is everything, so no ordering is ever needed.
    // pbuf holds UNNORMALISED weights summing to psum_, so the draw scales by psum_ rather
    // than the array being normalised in a separate sweep.
    int sample_pbuf(uint64_t& rng) {
        const float r = (float)uniform(rng) * psum_;
        float acc = 0;
        if (nucleus_ == 0) {
            for (int i = 0; i < c.vocab; ++i) { acc += pbuf[i]; if (r <= acc) return i; }
            return c.vocab - 1;
        }
        for (size_t i = 0; i < nucleus_; ++i) { acc += pbuf[obuf[i]]; if (r <= acc) return obuf[i]; }
        return obuf[nucleus_ - 1];
    }
    float prob_of(int tok) const { return psum_ > 0 ? pbuf[tok] / psum_ : 0.f; }

    int sample_row(int row, double temp, double top_p, uint64_t& rng) {
        if (temp <= 0.0) return argmax_row(row);
        row_probs(row, temp);
        build_nucleus(top_p);
        return sample_pbuf(rng);
    }

    // Speculative acceptance under temperature — the distribution-preserving rejection rule.
    //
    // Longest-prefix matching is only correct against an argmax target, which is why sampled
    // requests used to fall back to plain decoding. The general rule needs the DRAFT's
    // distribution q as well as the target's p... except that DFlash drafts greedily, so q is
    // a point mass on the drafted token. That collapses the algebra:
    //
    //     accept x  ⟺  u < p(x),        u ~ U[0,1)            (since q(x) = 1)
    //     on reject, sample from  norm( (p − q)_+ )  =  p with p(x) zeroed, renormalised
    //
    // So no draft probabilities are needed at all — only the target logits we already read.
    // This is exactly vLLM's NO_DRAFT_PROBS path, and it is distribution-preserving: the
    // emitted token stream has the same law as sampling from the target directly.
    //
    // Returns the number of accepted draft tokens; `*next_out` receives the bonus token (on a
    // full accept) or the recovered token (on a rejection).
    int verify_stochastic(int k, const int* draft, double temp, double top_p,
                          uint64_t& rng, int* next_out) {
        for (int j = 0; j < k; ++j) {
            row_probs(j, temp);
            build_nucleus(top_p);
            const double px = prob_of(draft[j]);
            if (uniform(rng) < px) continue;                 // accepted
            // rejected: draw from the residual, which is just this token's mass removed
            psum_ -= pbuf[draft[j]];
            pbuf[draft[j]] = 0.f;
            *next_out = sample_pbuf(rng);
            return j;
        }
        *next_out = sample_row(k, temp, top_p, rng);          // all k accepted; bonus token
        return k;
    }

    // Prefill `ids`, reusing whatever prefix the KV already holds. Returns the position of the
    // last token (so the caller can read its logits).
    int prefill(const std::vector<int>& ids, int& reused) {
        size_t common = 0;
        while (common < ids.size() && common < cached.size() && ids[common] == cached[common])
            ++common;
        // Never reuse the entire prompt: we need a forward over at least the last token to
        // have logits to sample from.
        if (common == ids.size() && common > 0) --common;
        reused = (int)common;

        pos = (int)common;
        const int BLK = dc.block_size;
        int last_row = 0;
        for (size_t i = common; i < ids.size(); i += BLK) {
            int n = (int)std::min((size_t)BLK, ids.size() - i);
            CUDA_CHECK(cudaMemcpy(d_tok, ids.data() + i, n * 4, cudaMemcpyHostToDevice));
            set_base(E.dbase, pos, 0);
            E.forward(S, d_tok, n, pos);
            pos += n;
            last_row = n - 1;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        cached = ids;
        return last_row;
    }

    int* d_tok = nullptr;
    void alloc_io() { CUDA_CHECK(cudaMalloc(&d_tok, (size_t)dc.block_size * 4)); }
};

static Server* g = nullptr;

// ------------------------------------------------------------------ generation
struct GenOpts {
    int max_tokens = 512;
    double temperature = 0.0, top_p = 1.0;
    std::vector<std::string> stop;
    bool thinking = true;
};

// Emits each newly committed token to `on_tok`; returns false from on_tok to stop.
struct GenStats { double prefill_s = 0, decode_s = 0; int prompt = 0, gen = 0, steps = 0; };

template <class F>
static void generate(Server& s, std::vector<int>& ids, const GenOpts& o, F&& on_tok,
                     GenStats* stats = nullptr) {
    uint64_t rng = 0x9E3779B97F4A7C15ULL ^ (uint64_t)std::chrono::steady_clock::now()
                       .time_since_epoch().count();
    double tp0 = wall_now();
    int reused = 0;
    int last_row = s.prefill(ids, reused);
    int next = s.sample_row(last_row, o.temperature, o.top_p, rng);

    // The prefix cache means the draft's context may be stale for an arbitrary rewind, so it
    // is rebuilt lazily on the first speculative step rather than eagerly here.
    s.draft_ctx_pos = 0;

    if (stats) { stats->prefill_s = wall_now() - tp0; stats->prompt = (int)ids.size(); }
    const double td0 = wall_now();

    int emitted = 0;
    auto is_eos = [&](int t) {
        for (int e : s.c.eos) if (t == e) return true;
        return false;
    };
    const int KMAX = s.dc.block_size - 1;
    std::vector<int> draft(KMAX), blk(s.dc.block_size), tout(s.dc.block_size);

    while (emitted < o.max_tokens) {
        if (is_eos(next)) break;
        if (!on_tok(next)) break;
        s.cached.push_back(next);
        ++emitted;

        s.sfx.extend(s.cached);
        int sfx_avail = 0;
        std::vector<int> sfx_draft(SpecPolicy::KMAX);
        static const bool spec_sampled_ = getenv("LG_SPEC_SAMPLED") != nullptr;
        const bool policy_active = s.use_spec && (o.temperature <= 0.0 || spec_sampled_);
        if (policy_active)
            sfx_avail = s.sfx.draft(s.cached, SpecPolicy::KMAX, sfx_draft.data())
                        ? SpecPolicy::KMAX : 0;

        SpecPolicy::Src src = SpecPolicy::AR;
        int k = 0;
        // Speculation under sampling is OFF by default. The rejection rule below is correct
        // and distribution-preserving -- that part is done and tested -- but the *policy* on
        // top of it does not yet beat plain sampled decode: acceptance is inherently lower and
        // noisier at T>0, which pollutes the per-position estimates, and measured runs
        // oscillated (22.9 / 19.7 / 14.6 tok/s with the AR estimate swinging 20.6 / 14.4 /
        // 28.7). Shipping it on by default would trade a reliable 30 tok/s for an unreliable
        // 15-25. LG_SPEC_SAMPLED=1 enables it for further work.
        if (policy_active) {
            if (s.policy.bootstrapping()) {
                k = s.policy.boot_k();
                src = k ? SpecPolicy::DFLASH : SpecPolicy::AR;
                if (++s.policy.boot > SpecPolicy::AR_SAMPLES + SpecPolicy::KMAX - 1)
                    s.policy.profiled = true;
            } else {
                s.policy.choose(sfx_avail, &src, &k);
            }
        }
        const double ts0 = wall_now();

        if (k == 0) {
            // The M=1 step runs from a captured CUDA graph -- measured +7.8 % on its own, and
            // it is the arm chosen on prose, so it is the one that matters most for a chat
            // workload. The graph is only valid because `tap_store` and `store_kv` both take
            // the position as a device pointer; a host int would be frozen at capture time.
            CUDA_CHECK(cudaMemcpy(s.d_tok, &next, 4, cudaMemcpyHostToDevice));
            if (s.E.needs_recapture(s.pos)) s.E.capture(s.S, s.d_tok, s.pos);
            s.E.step_graph(s.pos);
            CUDA_CHECK(cudaDeviceSynchronize());
            s.pos += 1;
            next = s.sample_row(0, o.temperature, o.top_p, rng);
            // Only learn costs from steps the policy actually governs. A sampled request
            // takes this same AR branch but pays a full softmax over 100352 logits on top of
            // the model, and folding that into cost_ar made the policy believe plain decode
            // costs 1/22.7 s when greedy decode costs 1/33 -- so one sampled request
            // permanently mis-priced every greedy request that followed it on the same
            // server. Costs are only comparable within one sampling regime.
            if (policy_active)
            {   // EWMA, not a single sample. cost_ar was measured once during the boot
                // sweep -- on a cold clock, before DVFS had ramped -- and never revisited, so
                // AR was permanently underrated at 23.4 tok/s against a true ~33. That made a
                // k=1 verify with ZERO acceptance look cheaper than decoding, and the
                // controller picked it on prose. Every arm's cost must decay the same way, or
                // the argmax is comparing a stale number against fresh ones.
                const double ca = wall_now() - ts0;
                const bool boot_ok = !s.policy.bootstrapping() || s.policy.boot_record_ar();
                if (boot_ok)
                    s.policy.cost_ar = s.policy.cost_ar > 0 ? 0.9 * s.policy.cost_ar + 0.1 * ca
                                                            : ca;
            }
            if (stats) ++stats->steps;
            continue;
        }

        if (src == SpecPolicy::SUFFIX) {
            for (int i = 0; i < k; ++i) draft[i] = sfx_draft[i];
        } else {
            if (s.draft_ctx_pos < s.pos) {
                const int lo = std::max(std::max(0, s.pos - s.dc.sliding_window),
                                        s.draft_ctx_pos);
                s.D.context_kv(s.E.taps, s.E.tap_cap, lo, s.pos - lo);
                CUDA_CHECK(cudaDeviceSynchronize());
                s.draft_ctx_pos = s.pos;
            }
            s.D.propose(next, s.pos, k, s.W.embed, s.W.lm_head, s.W.lm_head8, s.W.lm_head8s,
                        draft.data());
        }
        const double tdraft = wall_now();

        blk[0] = next;
        for (int i = 0; i < k; ++i) blk[1 + i] = draft[i];
        CUDA_CHECK(cudaMemcpy(s.d_tok, blk.data(), (k + 1) * 4, cudaMemcpyHostToDevice));
        set_base(s.E.dbase, s.pos, 0);
        s.E.forward(s.S, s.d_tok, k + 1, s.pos);
        CUDA_CHECK(cudaDeviceSynchronize());

        const bool greedy = (o.temperature <= 0.0);
        int nacc;
        if (greedy) {
            for (int j = 0; j <= k; ++j) tout[j] = s.sample_row(j, o.temperature, o.top_p, rng);
            nacc = 0;
            while (nacc < k && draft[nacc] == tout[nacc]) ++nacc;
        } else {
            nacc = s.verify_stochastic(k, draft.data(), o.temperature, o.top_p, rng, &next);
        }

        const int nacc_full = nacc;
        for (int j = 0; j < nacc; ++j) {
            if (emitted >= o.max_tokens || is_eos(draft[j])) { nacc = j; break; }
            if (!on_tok(draft[j])) { nacc = j; break; }
            s.cached.push_back(draft[j]);
            ++emitted;
        }
        // `next` belongs to whichever position we actually stopped at. On the greedy path
        // tout[] is valid for every j so indexing it is enough. On the sampled path
        // verify_stochastic already produced the correct token for `nacc_full` -- a rejection
        // draws from the RESIDUAL distribution, which must not be re-drawn -- so only a
        // genuine truncation (EOS or the token budget) needs a fresh draw.
        if (greedy)                  next = tout[nacc];
        else if (nacc != nacc_full)  next = s.sample_row(nacc, o.temperature, o.top_p, rng);
        s.pos += nacc + 1;
        if (s.draft_ctx_pos == s.pos - (nacc + 1)) {   // contiguous: extend cheaply
            s.D.context_kv(s.E.taps, s.E.tap_cap, s.draft_ctx_pos, nacc + 1);
            s.draft_ctx_pos = s.pos;
        }
        // The observation is per POSITION, not per step -- this is the whole point.
        s.policy.observe(src, k, nacc);
        {   // EWMA the cost too: it drifts with context length as KV grows
            const double cv = wall_now() - tdraft;
            s.policy.cost_verify[k] = s.policy.cost_verify[k] > 0
                                    ? 0.9 * s.policy.cost_verify[k] + 0.1 * cv : cv;
            if (src == SpecPolicy::DFLASH) {
                const double cd = tdraft - ts0;
                s.policy.cost_draft_dflash = s.policy.cost_draft_dflash > 0
                                           ? 0.9 * s.policy.cost_draft_dflash + 0.1 * cd : cd;
            }
        }
        if (stats) ++stats->steps;
    }
    if (stats) { stats->decode_s = wall_now() - td0; stats->gen = emitted; }
}

// ------------------------------------------------------------------ HTTP
static std::string now_id() {
    static std::atomic<long> n{0};
    return "chatcmpl-" + std::to_string(++n) + "-" +
           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count() & 0xffffff);
}

static std::vector<lgchat::Message> parse_messages(const ojson& j) {
    std::vector<lgchat::Message> out;
    for (const auto& m : j.at("messages")) {
        lgchat::Message x;
        x.role = m.value("role", "user");
        if (m.contains("content") && m["content"].is_string()) x.content = m["content"];
        else if (m.contains("content") && m["content"].is_array())
            for (const auto& part : m["content"])
                if (part.value("type", "") == "text") x.content += part.value("text", "");
        x.reasoning = m.value("reasoning_content", m.value("reasoning", std::string()));
        if (m.contains("tool_calls"))
            for (const auto& tc : m["tool_calls"]) {
                lgchat::ToolCall t;
                t.id = tc.value("id", "");
                t.name = tc["function"].value("name", "");
                t.arguments = tc["function"].value("arguments", "");
                x.tool_calls.push_back(t);
            }
        out.push_back(x);
    }
    return out;
}

int main(int argc, char** argv) {
    std::string md = "models/Laguna-S-2.1-NVFP4";
    std::string dd = "models/Laguna-S-2.1-DFlash-NVFP4";
    int port = getenv("PORT") ? atoi(getenv("PORT")) : 8080;
    int ctx  = getenv("CTX")  ? atoi(getenv("CTX"))  : 262144;
    int k    = getenv("SPEC_K") ? atoi(getenv("SPEC_K")) : 3;
    // Everything the FP8 W8A16 path covers is ON by default: attention, lm_head, the shared
    // experts and the layer-0 dense MLP. Together they take B_tok 10.04 -> 6.25 GB. Each was
    // A/B'd with greedy output held 8/8 against the oracle, and each has a shipping NVIDIA
    // precedent (Nemotron-3 for lm_head and shared experts, Mistral-Medium for edge dense
    // layers). The routed experts stay NVFP4 as shipped, and q/k/v NEVER go below FP8.
    bool fp8a = getenv("LG_BF16ATTN") == nullptr;
    bool fp8l = getenv("LG_BF16LMHEAD") == nullptr;
    bool fp8d = getenv("LG_BF16DENSE") == nullptr;
    std::string host = getenv("HOST") ? getenv("HOST") : "0.0.0.0";

    Server s; g = &s;
    s.boot(md, dd, fp8a, ctx, k, fp8l, fp8d);
    s.alloc_io();

    httplib::Server http;
    http.set_read_timeout(600); http.set_write_timeout(600);

    http.Get("/healthz", [](const httplib::Request&, httplib::Response& r) {
        r.set_content("ok\n", "text/plain");
    });
    http.Get("/v1/models", [&](const httplib::Request&, httplib::Response& r) {
        ojson j = {{"object", "list"}, {"data", ojson::array({
            ojson{{"id", "Laguna-S-2.1-NVFP4"}, {"object", "model"}, {"owned_by", "poolside"}}})}};
        r.set_content(j.dump(), "application/json");
    });
    http.Get("/", [](const httplib::Request&, httplib::Response& r) {
        r.set_content(kWebUI, "text/html; charset=utf-8");
    });

    http.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        ojson body;
        try { body = ojson::parse(req.body); }
        catch (const std::exception& e) {
            res.status = 400;
            res.set_content(ojson{{"error", ojson{{"message", std::string("bad JSON: ") + e.what()}}}}.dump(),
                            "application/json");
            return;
        }
        GenOpts o;
        o.max_tokens = body.value("max_tokens", body.value("max_completion_tokens", 512));
        o.temperature = body.value("temperature", 0.0);
        o.top_p = body.value("top_p", 1.0);
        bool stream = body.value("stream", false);
        o.thinking = true;
        if (body.contains("chat_template_kwargs"))
            o.thinking = body["chat_template_kwargs"].value("enable_thinking", true);
        if (body.contains("reasoning_effort") && body["reasoning_effort"] == "none")
            o.thinking = false;

        auto msgs = parse_messages(body);
        ojson tools = body.contains("tools") ? body["tools"] : ojson::array();
        std::string prompt = lgchat::render(msgs, tools, o.thinking, true);

        std::lock_guard<std::mutex> lk(s.mu);
        std::vector<int> ids = s.tok->encode(prompt);
        if ((int)ids.size() + o.max_tokens > s.CTX) {
            res.status = 400;
            res.set_content(ojson{{"error", ojson{{"message", "context length exceeded"}}}}.dump(),
                            "application/json");
            return;
        }
        std::string id = now_id();

        if (!stream) {
            std::string text;
            GenStats gs; int n = 0;
            generate(s, ids, o, [&](int t) { text += s.tok->decode({t}); ++n; return true; }, &gs);
            lgchat::Parsed p = lgchat::parse_output(text, o.thinking);
            ojson msg{{"role", "assistant"}, {"content", p.content}};
            if (!p.reasoning.empty()) msg["reasoning_content"] = p.reasoning;
            if (!p.tool_calls.empty()) {
                ojson tc = ojson::array();
                for (auto& t : p.tool_calls)
                    tc.push_back(ojson{{"id", t.id}, {"type", "function"},
                                       {"function", ojson{{"name", t.name}, {"arguments", t.arguments}}}});
                msg["tool_calls"] = tc;
                msg["content"] = nullptr;
            }
            ojson j{{"id", id}, {"object", "chat.completion"}, {"model", "Laguna-S-2.1-NVFP4"},
                    {"choices", ojson::array({ojson{{"index", 0}, {"message", msg},
                        {"finish_reason", p.tool_calls.empty() ? "stop" : "tool_calls"}}})},
                    {"usage", ojson{{"prompt_tokens", (int)ids.size()},
                                    {"completion_tokens", n},
                                    {"total_tokens", (int)ids.size() + n}}},
                    // Decode rate EXCLUDING prefill is the number to compare against the
                    // benchmarks; the combined figure just tracks prompt length.
                    {"timings", ojson{
                        {"prefill_seconds", gs.prefill_s},
                        {"decode_seconds", gs.decode_s},
                        {"decode_tokens_per_second", gs.gen / std::max(gs.decode_s, 1e-9)},
                        {"tokens_per_second", gs.gen / std::max(gs.prefill_s + gs.decode_s, 1e-9)},
                        {"target_forwards", gs.steps},
                        {"accepted_per_forward", gs.steps ? (double)gs.gen / gs.steps : 0.0},
                        {"spec_arms", s.policy.report()}}}};
            res.set_content(j.dump(), "application/json");
            return;
        }

        // ---- streaming. Reasoning is emitted as `reasoning_content` deltas until the
        // model closes </think>, then as `content`. Tool calls need the whole body, so they
        // are parsed at the end and sent as one delta before the final chunk.
        res.set_chunked_content_provider("text/event-stream",
            [&s, ids, o, id](size_t, httplib::DataSink& sink) mutable {
                auto send = [&](const ojson& delta, const char* finish) {
                    ojson ch{{"id", id}, {"object", "chat.completion.chunk"},
                             {"model", "Laguna-S-2.1-NVFP4"},
                             {"choices", ojson::array({ojson{{"index", 0}, {"delta", delta},
                                 {"finish_reason", finish ? ojson(finish) : ojson(nullptr)}}})}};
                    std::string s2 = "data: " + ch.dump() + "\n\n";
                    return sink.write(s2.data(), s2.size());
                };
                send(ojson{{"role", "assistant"}}, nullptr);

                Utf8Stream u8;
                std::string raw;
                bool in_think = o.thinking, saw_toolmark = false;
                generate(s, ids, o, [&](int t) {
                    std::string piece = s.tok->decode({t});
                    raw += piece;
                    // Once a tool call starts, stop streaming text -- the call is structured
                    // and only makes sense once complete.
                    if (raw.find("<tool_call>") != std::string::npos) { saw_toolmark = true; return true; }
                    if (saw_toolmark) return true;
                    std::string out = u8.feed(piece);
                    if (out.empty()) return true;
                    if (in_think) {
                        size_t e = raw.find("</think>");
                        if (e != std::string::npos) {
                            // split this chunk at the tag
                            size_t cut = out.find("</think>");
                            if (cut != std::string::npos) {
                                if (cut) send(ojson{{"reasoning_content", out.substr(0, cut)}}, nullptr);
                                std::string rest = out.substr(cut + 8);
                                if (!rest.empty()) send(ojson{{"content", rest}}, nullptr);
                            }
                            in_think = false;
                            return true;
                        }
                        return send(ojson{{"reasoning_content", out}}, nullptr);
                    }
                    return send(ojson{{"content", out}}, nullptr);
                });
                std::string tail = u8.flush();
                if (!tail.empty() && !saw_toolmark)
                    send(in_think ? ojson{{"reasoning_content", tail}} : ojson{{"content", tail}}, nullptr);

                lgchat::Parsed p = lgchat::parse_output(raw, o.thinking);
                const char* finish = "stop";
                if (!p.tool_calls.empty()) {
                    ojson tc = ojson::array();
                    for (size_t i = 0; i < p.tool_calls.size(); ++i) {
                        auto& t = p.tool_calls[i];
                        tc.push_back(ojson{{"index", (int)i}, {"id", t.id}, {"type", "function"},
                            {"function", ojson{{"name", t.name}, {"arguments", t.arguments}}}});
                    }
                    send(ojson{{"tool_calls", tc}}, nullptr);
                    finish = "tool_calls";
                }
                send(ojson::object(), finish);
                const char* done = "data: [DONE]\n\n";
                sink.write(done, strlen(done));
                sink.done();
                return true;
            });
    });

    printf("[server] listening on http://%s:%d  (WebUI at /)\n", host.c_str(), port);
    if (!http.listen(host.c_str(), port)) { fprintf(stderr, "listen failed\n"); return 1; }
    return 0;
}
