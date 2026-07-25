// tokenizer.h — Laguna / poolside ByteLevel BPE, pure C++, no Python at serve time.
//
// Read from models/…/tokenizer.json (MODEL_INVENTORY.md §6). This is a FULL REWRITE relative
// to the gemma port: gemma was `▁`-normalised BPE with byte_fallback and a no-op
// pre-tokenizer; Laguna is ByteLevel with **no normalizer**, `byte_fallback=false`, and a
// two-stage pre-tokenizer whose second stage is a GPT-4-class regex with Unicode property
// classes. There is no `std::regex` that can express `\p{L}` / `\p{N}`, so the matcher is
// hand-rolled against generated category tables (`unicode_tables.h`).
//
// Pipeline, in order:
//   1. added tokens (70 of them) matched literally and never split
//   2. Split  (?:\r?\n)+(?!\r?\n)   behaviour MergedWithNext  — a newline run attaches to
//      the chunk that FOLLOWS it
//   3. Split  the GPT-4-class alternation, behaviour Isolated
//   4. ByteLevel map (add_prefix_space=false), then BPE by merge rank
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include "unicode_tables.h"
#include "third_party/json.hpp"

namespace lgtok {

// ---------------------------------------------------------------- UTF-8
inline uint32_t utf8_next(const std::string& s, size_t& i) {
    unsigned char c = s[i];
    if (c < 0x80) { ++i; return c; }
    if ((c >> 5) == 0x6 && i + 1 < s.size()) {
        uint32_t v = ((c & 0x1F) << 6) | (s[i + 1] & 0x3F); i += 2; return v;
    }
    if ((c >> 4) == 0xE && i + 2 < s.size()) {
        uint32_t v = ((c & 0x0F) << 12) | ((s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F);
        i += 3; return v;
    }
    if ((c >> 3) == 0x1E && i + 3 < s.size()) {
        uint32_t v = ((c & 0x07) << 18) | ((s[i + 1] & 0x3F) << 12) |
                     ((s[i + 2] & 0x3F) << 6) | (s[i + 3] & 0x3F);
        i += 4; return v;
    }
    ++i; return c;
}
inline void utf8_put(std::string& s, uint32_t cp) {
    if (cp < 0x80) s.push_back((char)cp);
    else if (cp < 0x800) { s.push_back((char)(0xC0 | (cp >> 6)));
                           s.push_back((char)(0x80 | (cp & 0x3F))); }
    else if (cp < 0x10000) { s.push_back((char)(0xE0 | (cp >> 12)));
                             s.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
                             s.push_back((char)(0x80 | (cp & 0x3F))); }
    else { s.push_back((char)(0xF0 | (cp >> 18)));
           s.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
           s.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
           s.push_back((char)(0x80 | (cp & 0x3F))); }
}

// ---------------------------------------------------------------- ByteLevel alphabet
// The GPT-2 byte<->printable-codepoint bijection that `ByteLevel` uses.
struct ByteMap {
    std::string enc[256];              // byte -> UTF-8 of its mapped code point
    int dec[0x200];                    // code point -> byte (-1 if none)
    ByteMap() {
        for (int i = 0; i < 0x200; ++i) dec[i] = -1;
        std::vector<int> bs;
        for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
        for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
        for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; ++b)
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b); cs.push_back(256 + n); ++n;
            }
        for (size_t i = 0; i < bs.size(); ++i) {
            std::string u; utf8_put(u, (uint32_t)cs[i]);
            enc[bs[i]] = u;
            if (cs[i] < 0x200) dec[cs[i]] = bs[i];
        }
    }
};

class Tokenizer {
public:
    explicit Tokenizer(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("cannot open " + path);
        nlohmann::json j; f >> j;

        const auto& model = j.at("model");
        if (model.at("type").get<std::string>() != "BPE")
            throw std::runtime_error("tokenizer.json: expected a BPE model");

        for (auto it = model.at("vocab").begin(); it != model.at("vocab").end(); ++it) {
            int id = it.value().get<int>();
            vocab_[it.key()] = id;
            if ((int)id2tok_.size() <= id) id2tok_.resize(id + 1);
            id2tok_[id] = it.key();
        }
        int rank = 0;
        for (const auto& m : model.at("merges")) {
            std::string a, b;
            if (m.is_array()) { a = m[0].get<std::string>(); b = m[1].get<std::string>(); }
            else {                                     // older "A B" string form
                std::string s = m.get<std::string>();
                size_t sp = s.find(' ');
                a = s.substr(0, sp); b = s.substr(sp + 1);
            }
            auto ia = vocab_.find(a), ib = vocab_.find(b);
            if (ia == vocab_.end() || ib == vocab_.end()) { ++rank; continue; }
            merge_[key(ia->second, ib->second)] = rank++;
        }
        if (j.contains("added_tokens"))
            for (const auto& t : j.at("added_tokens")) {
                Added a; a.content = t.at("content").get<std::string>();
                a.id = t.at("id").get<int>();
                added_.push_back(a);
                vocab_[a.content] = a.id;
                if ((int)id2tok_.size() <= a.id) id2tok_.resize(a.id + 1);
                id2tok_[a.id] = a.content;
            }
        // longest-first so "</assistant>" wins over any prefix of it
        std::sort(added_.begin(), added_.end(),
                  [](const Added& x, const Added& y) { return x.content.size() > y.content.size(); });
    }

    size_t vocab_size() const { return id2tok_.size(); }
    int id_of(const std::string& tok) const {
        auto it = vocab_.find(tok); return it == vocab_.end() ? -1 : it->second;
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> out;
        size_t i = 0;
        while (i < text.size()) {
            size_t hit = std::string::npos; int hid = -1;
            for (const auto& a : added_)                       // added tokens are never split
                if (text.compare(i, a.content.size(), a.content) == 0) { hit = a.content.size(); hid = a.id; break; }
            if (hit != std::string::npos) { out.push_back(hid); i += hit; continue; }
            size_t nxt = text.size();
            for (const auto& a : added_) {
                size_t p = text.find(a.content, i);
                if (p != std::string::npos && p < nxt) nxt = p;
            }
            encode_span(text.substr(i, nxt - i), out);
            i = nxt;
        }
        return out;
    }

    std::string decode(const std::vector<int>& ids, bool skip_special = false) const {
        std::string bytes;
        for (int id : ids) {
            if (id < 0 || id >= (int)id2tok_.size()) continue;
            bool is_added = false;
            for (const auto& a : added_) if (a.id == id) { is_added = true; break; }
            if (is_added) { if (!skip_special) bytes += id2tok_[id]; continue; }
            const std::string& t = id2tok_[id];
            size_t p = 0;
            while (p < t.size()) {                            // reverse the ByteLevel map
                uint32_t cp = utf8_next(t, p);
                int b = (cp < 0x200) ? bm_.dec[cp] : -1;
                bytes.push_back(b >= 0 ? (char)b : '?');
            }
        }
        return bytes;
    }

private:
    struct Added { std::string content; int id; };
    static uint64_t key(int a, int b) { return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b; }

    // ---- stage 2 matcher. Alternatives are tried in the written order, leftmost wins —
    // the same first-alternative-wins semantics tiktoken and llama.cpp implement.
    size_t match_piece(const std::vector<uint32_t>& cp, size_t i) const {
        const size_t n = cp.size();
        auto L = [&](size_t k) { return k < n && is_letter(cp[k]); };
        auto N = [&](size_t k) { return k < n && is_number(cp[k]); };
        auto S = [&](size_t k) { return k < n && is_space(cp[k]); };
        auto NL = [&](size_t k) { return k < n && (cp[k] == '\r' || cp[k] == '\n'); };

        // 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (cp[i] == '\'' && i + 1 < n) {
            auto lo = [](uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; };
            uint32_t a = lo(cp[i + 1]);
            uint32_t b = (i + 2 < n) ? lo(cp[i + 2]) : 0;
            if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
        }
        // 2. [^\r\n\p{L}\p{N}]?\p{L}+
        {
            size_t k = i;
            if (!NL(k) && !L(k) && !N(k)) ++k;        // optional single non-letter/digit prefix
            if (L(k)) { size_t s = k; while (L(k)) ++k; (void)s; return k - i; }
        }
        // 3. \p{N}   (exactly one — digits tokenize individually)
        if (N(i)) return 1;
        // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
        {
            size_t k = i;
            if (cp[k] == ' ') ++k;
            size_t s = k;
            while (k < n && !S(k) && !L(k) && !N(k)) ++k;
            if (k > s) { while (NL(k)) ++k; return k - i; }
        }
        // 5. \s*[\r\n]+
        {
            size_t k = i;
            while (S(k) && !NL(k)) ++k;
            if (NL(k)) { while (NL(k)) ++k; return k - i; }
        }
        // 6. \s+(?!\S)   — a whitespace run only if nothing non-space follows it
        {
            size_t k = i;
            while (S(k)) ++k;
            if (k > i && k >= n) return k - i;
            if (k > i + 1) return k - i - 1;          // leave the last space for the next piece
        }
        // 7. \s+
        {
            size_t k = i;
            while (S(k)) ++k;
            if (k > i) return k - i;
        }
        return 1;                                      // never stall
    }

    void encode_span(const std::string& text, std::vector<int>& out) const {
        if (text.empty()) return;
        // stage 1: newline runs merge with what FOLLOWS
        std::vector<std::string> chunks;
        {
            size_t i = 0, start = 0;
            while (i < text.size()) {
                if (text[i] == '\r' || text[i] == '\n') {
                    size_t j = i;
                    while (j < text.size() && (text[j] == '\r' || text[j] == '\n')) ++j;
                    if (i > start) chunks.push_back(text.substr(start, i - start));
                    start = i;                          // delimiter stays with the next chunk
                    i = j;
                } else ++i;
            }
            if (start < text.size()) chunks.push_back(text.substr(start));
        }
        for (const auto& ch : chunks) {
            std::vector<uint32_t> cp;
            for (size_t p = 0; p < ch.size();) cp.push_back(utf8_next(ch, p));
            size_t i = 0;
            while (i < cp.size()) {
                size_t len = match_piece(cp, i);
                std::string piece;
                for (size_t k = i; k < i + len; ++k) utf8_put(piece, cp[k]);
                bpe(piece, out);
                i += len;
            }
        }
    }

    // ByteLevel map then merge by rank.
    void bpe(const std::string& piece, std::vector<int>& out) const {
        std::string mapped;
        for (unsigned char b : piece) mapped += bm_.enc[b];

        std::vector<int> sym;                          // current symbols, as vocab ids
        {
            size_t p = 0;
            while (p < mapped.size()) {
                size_t q = p; utf8_next(mapped, q);
                auto it = vocab_.find(mapped.substr(p, q - p));
                if (it == vocab_.end()) { p = q; continue; }   // byte_fallback=false: cannot happen
                sym.push_back(it->second);
                p = q;
            }
        }
        while (sym.size() > 1) {
            int best = -1; size_t at = 0;
            for (size_t k = 0; k + 1 < sym.size(); ++k) {
                auto it = merge_.find(key(sym[k], sym[k + 1]));
                if (it != merge_.end() && (best < 0 || it->second < best)) { best = it->second; at = k; }
            }
            if (best < 0) break;
            auto it = vocab_.find(id2tok_[sym[at]] + id2tok_[sym[at + 1]]);
            if (it == vocab_.end()) break;
            sym[at] = it->second;
            sym.erase(sym.begin() + at + 1);
        }
        out.insert(out.end(), sym.begin(), sym.end());
    }

    std::unordered_map<std::string, int> vocab_;
    std::vector<std::string> id2tok_;
    std::unordered_map<uint64_t, int> merge_;
    std::vector<Added> added_;
    ByteMap bm_;
};

} // namespace lgtok
