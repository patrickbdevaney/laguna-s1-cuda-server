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

// ------------------------------------------------------------------ adaptive speculation
//
// Speculation is NOT free on this model, and on a bandwidth-bound MoE it is not even reliably
// positive. Gate D1 measured the break-even: a verify forward at M=k+1 costs about 3x a
// decode step (the MoE activates ~4x the experts, and the dense GEMMs are issue-bound at
// M>1), so speculation only wins when tau > ~3. And tau is content-dependent -- poolside
// publish 6.44 on HumanEval against 4.02 on MT-Bench, and this server measures 3.06 on a
// math prompt against 2.26 on open prose. At 2.26 with k=3 speculation runs at 20.7 tok/s
// against 27.4 for plain autoregressive decode: a 25 % LOSS.
//
// So the mode is chosen by measurement rather than by configuration. Each arm keeps an EWMA
// of its achieved tokens/second; the best arm runs, and every arm is re-probed periodically
// so the estimate tracks a conversation that switches from prose to code. The state lives on
// the server, not the request, because it is a property of the hardware far more than of any
// one prompt.
struct SpecBandit {
    static const int NM = 4;
    static const int RUN = 32;              // steps an arm holds before it is re-ranked
    static const int PROBE_EVERY = 10;      // blocks between exploration
    static const int SWEEP_EVERY = 50;      // blocks between a forced probe of EVERY arm

    int    k[NM]     = {0, 2, 3, 5};        // 0 = plain autoregressive
    double rate[NM]  = {0, 0, 0, 0};        // EWMA tokens/second
    long   tries[NM] = {0, 0, 0, 0};

    // An arm is held for a BLOCK of steps, not one step. A single speculative step yields
    // between 1 and k+1 tokens, so its instantaneous rate varies by the full width of the
    // acceptance distribution -- ranking arms on one sample made the bandit thrash and cost
    // 12 % on the workload where speculation actually wins. Averaging over RUN steps cuts the
    // variance by ~5x, which is the difference between converging and oscillating.
    int    cur = 0;
    int    left = 0;
    int    blk_tok = 0;
    double blk_s = 0;
    long   blocks = 0;

    int pick() {
        if (left > 0) return cur;
        if (blk_s > 0) commit();
        for (int i = 0; i < NM; ++i) if (tries[i] == 0) { cur = i; break; }
        if (tries[cur] != 0) {
            int best = 0;
            for (int i = 1; i < NM; ++i) if (rate[i] > rate[best]) best = i;
            ++blocks;
            // Exploration is not free: a block spent on a bad arm costs ~30 % of its tokens,
            // and acceptance is non-stationary WITHIN a generation (the same arm measured
            // 37.4 then 22.6 tok/s as a code answer drifted into prose). So probe rarely,
            // and skip arms that are not plausibly competitive -- unless a full sweep is due,
            // which is what lets an arm that has recovered come back.
            const bool sweep = (blocks % SWEEP_EVERY) < NM;
            if (sweep) cur = (int)(blocks % NM);
            else if (blocks % PROBE_EVERY == 0) {
                cur = best;
                for (int t = 1; t < NM; ++t) {
                    int cand = (int)((blocks / PROBE_EVERY + t) % NM);
                    if (rate[cand] >= 0.65 * rate[best]) { cur = cand; break; }
                }
            } else cur = best;
        }
        left = RUN;
        return cur;
    }
    void update(int, int toks, double secs) { blk_tok += toks; blk_s += secs; --left; }
    void commit() {
        double r = blk_tok / std::max(blk_s, 1e-9);
        rate[cur] = tries[cur] ? 0.6 * rate[cur] + 0.4 * r : r;
        ++tries[cur];
        blk_tok = 0; blk_s = 0;
    }
    void flush() { if (blk_s > 0) { commit(); left = 0; } }   // end of a request

    std::string report() const {
        std::string s;
        for (int i = 0; i < NM; ++i) {
            char b[64];
            snprintf(b, sizeof b, "%s%s=%.1f", i ? " " : "",
                     k[i] ? ("k" + std::to_string(k[i])).c_str() : "ar", rate[i]);
            s += b;
        }
        return s;
    }
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

    SpecBandit bandit;

    // prefix cache
    std::vector<int> cached;        // token ids currently represented in S's KV
    int pos = 0;

    std::vector<float> logits_host;

    void boot(const std::string& md, const std::string& dd, bool fp8a, int ctx, int k) {
        CTX = ctx; SPEC_K = k;
        c  = load_config(md + "/config.json");
        dc = load_draft_config(dd + "/config.json");
        use_spec = SPEC_K > 0;
        if (SPEC_K > dc.block_size - 1) SPEC_K = dc.block_size - 1;

        Loader ld(md, c, fp8a);
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
    int sample_row(int row, double temp, double top_p, uint64_t& rng) {
        if (temp <= 0.0) return argmax_row(row);
        CUDA_CHECK(cudaMemcpy(logits_host.data(), E.logits + (size_t)row * c.vocab,
                              c.vocab * 4, cudaMemcpyDeviceToHost));
        std::vector<int> idx(c.vocab);
        for (int i = 0; i < c.vocab; ++i) idx[i] = i;
        float mx = *std::max_element(logits_host.begin(), logits_host.end());
        std::vector<double> p(c.vocab);
        double sum = 0;
        for (int i = 0; i < c.vocab; ++i) { p[i] = std::exp((logits_host[i] - mx) / temp); sum += p[i]; }
        for (int i = 0; i < c.vocab; ++i) p[i] /= sum;
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return p[a] > p[b]; });
        double cum = 0; size_t keep = idx.size();
        for (size_t i = 0; i < idx.size(); ++i) { cum += p[idx[i]]; if (cum >= top_p) { keep = i + 1; break; } }
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        double r = (double)((rng >> 11) & ((1ULL << 53) - 1)) / (double)(1ULL << 53) * cum;
        double acc = 0;
        for (size_t i = 0; i < keep; ++i) { acc += p[idx[i]]; if (r <= acc) return idx[i]; }
        return idx[keep - 1];
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

    // Draft context for everything still inside the draft's 512-wide window.
    if (s.use_spec) {
        int c0 = std::max(0, s.pos - s.dc.sliding_window);
        s.D.context_kv(s.E.taps, s.E.tap_cap, c0, s.pos - c0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

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

        // Sampling has no cheap correct acceptance rule here -- longest-prefix is only valid
        // against an argmax target, and the rejection sampler needs the draft's own
        // distribution. Rather than silently change the output distribution, sampled requests
        // decode autoregressively.
        const int arm = (s.use_spec && o.temperature <= 0.0) ? s.bandit.pick() : 0;
        const int k = s.bandit.k[arm];
        const double ts0 = wall_now();

        if (k == 0) {
            // The M=1 step runs from a captured CUDA graph -- measured +7.8 % on its own, and
            // it is the arm the bandit picks on prose, so it is the one that matters most for
            // a chat workload. The graph is only valid because `tap_store` and `store_kv`
            // both take the position as a device pointer; a host int would be frozen at
            // capture time.
            CUDA_CHECK(cudaMemcpy(s.d_tok, &next, 4, cudaMemcpyHostToDevice));
            if (s.E.needs_recapture(s.pos)) s.E.capture(s.S, s.d_tok, s.pos);
            s.E.step_graph(s.pos);
            CUDA_CHECK(cudaDeviceSynchronize());
            s.pos += 1;
            next = s.sample_row(0, o.temperature, o.top_p, rng);
            if (s.use_spec) s.D.context_kv(s.E.taps, s.E.tap_cap, s.pos - 1, 1);
            s.bandit.update(arm, 1, wall_now() - ts0);
            if (stats) ++stats->steps;
            continue;
        }

        s.D.propose(next, s.pos, k, s.W.embed, s.W.lm_head, draft.data());
        blk[0] = next;
        for (int i = 0; i < k; ++i) blk[1 + i] = draft[i];
        CUDA_CHECK(cudaMemcpy(s.d_tok, blk.data(), (k + 1) * 4, cudaMemcpyHostToDevice));
        set_base(s.E.dbase, s.pos, 0);
        s.E.forward(s.S, s.d_tok, k + 1, s.pos);
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int j = 0; j <= k; ++j) tout[j] = s.sample_row(j, o.temperature, o.top_p, rng);
        int nacc = 0;
        while (nacc < k && draft[nacc] == tout[nacc]) ++nacc;

        for (int j = 0; j < nacc; ++j) {
            if (emitted >= o.max_tokens || is_eos(draft[j])) { nacc = j; break; }
            if (!on_tok(draft[j])) { nacc = j; break; }
            s.cached.push_back(draft[j]);
            ++emitted;
        }
        s.pos += nacc + 1;
        next = tout[nacc];
        s.D.context_kv(s.E.taps, s.E.tap_cap, s.pos - (nacc + 1), nacc + 1);
        s.bandit.update(arm, nacc + 1, wall_now() - ts0);
        if (stats) ++stats->steps;
    }
    s.bandit.flush();
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
    bool fp8a = getenv("LG_BF16ATTN") == nullptr;      // FP8 attention on by default
    std::string host = getenv("HOST") ? getenv("HOST") : "0.0.0.0";

    Server s; g = &s;
    s.boot(md, dd, fp8a, ctx, k);
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
                        {"spec_arms", s.bandit.report()}}}};
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
