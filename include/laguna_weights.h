// laguna_weights.h — device weight layout + streaming loader.
//
// Memory discipline (DIRECTIVE.md §5): the checkpoint is 71.9 GB on a 122 GB box. A naive
// load that lets the page cache retain every shard while device buffers fill would need
// ~144 GB. So: reserve every buffer in ONE arena up front, then make a single pass over the
// shards, reading each with bulk pread into a reusable host buffer, copying tensors into the
// arena, and dropping that shard's page cache before touching the next.
// Peak host RSS = one shard (~5 GB).
//
// Three measurements shaped this (tests/bw_probe.cu, tests/h2d_probe.cu):
//   * cudaMalloc streams at 227 GB/s vs 160 GB/s for a cudaHostRegister'd mmap
//       -> weights live in cudaMalloc, never in mapped memory.
//   * cold mmap page-fault I/O is ~1 GB/s; bulk pread is 6.95 GB/s
//       -> read with pread, not by faulting an mmap.
//   * a warm host->device memcpy on this integrated part is 109 GB/s, while staging through
//     cudaHostAlloc measured SLOWER (10.6 GB/s)
//       -> copy straight from the pageable read buffer, never stage.
//
// One arena also makes the repack cache trivial: it is a byte image of device memory.
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <fstream>
#include <algorithm>
#include <ctime>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include "laguna_config.h"
#include "third_party/json.hpp"

namespace laguna {

// ---------------------------------------------------------------------------------------
// Expert-weight repack (the "offline repack" of DIRECTIVE.md §5 / the gemma Marlin recipe).
//
// The checkpoint stores each expert projection row-major [n][k]. A warp-per-output-row GEMV
// over that layout gives each warp only H/32/32 = 3 k-iterations, so there is no
// memory-level parallelism *inside* a warp -- the kernel depends entirely on having many
// warps, and occupancy is register-limited.
//
// Repacking to  [n_block][k_chunk][lane][16B]  lets one LANE own an output row and stream
// all of k: 32 lanes read 32 x 16 B = 512 contiguous bytes per step (still perfectly
// coalesced), each lane accumulates privately (no warp reduction at all), and the loop is
// long enough to unroll and keep several loads in flight.
//
// Each expert's region stays contiguous and the same size, so the arena layout and the
// per-expert base pointers are unchanged -- only the addressing inside an expert changes.
constexpr int RP_NB = 32;      // output rows per repack block == warp width

// packed codes: src [rows][K/2] bytes -> dst [rows/NB][K/32][NB][16]
inline void repack_packed(uint8_t* dst, const uint8_t* src, int rows, int K) {
    const int chunks = K / 32;                       // 16 bytes of codes = 32 weights
    for (int n = 0; n < rows; ++n) {
        const int nb = n / RP_NB, l = n % RP_NB;
        const uint8_t* s = src + (size_t)n * (K / 2);
        for (int c = 0; c < chunks; ++c) {
            uint8_t* d = dst + (((size_t)nb * chunks + c) * RP_NB + l) * 16;
            memcpy(d, s + (size_t)c * 16, 16);
        }
    }
}

// e4m3 group scales: src [rows][K/GRP] -> dst [rows/NB][K/GRP][NB]
inline void repack_scale(uint8_t* dst, const uint8_t* src, int rows, int K, int GRP) {
    const int ng = K / GRP;
    for (int n = 0; n < rows; ++n) {
        const int nb = n / RP_NB, l = n % RP_NB;
        const uint8_t* s = src + (size_t)n * ng;
        for (int g = 0; g < ng; ++g)
            dst[((size_t)nb * ng + g) * RP_NB + l] = s[g];
    }
}

#define CUDA_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, \
            cudaGetErrorString(_e)); std::abort(); } } while (0)

struct LayerW {
    // attention (BF16 as raw uint16_t)
    const uint16_t *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr, *g = nullptr;
    const uint16_t *in_ln = nullptr, *post_ln = nullptr, *q_norm = nullptr, *k_norm = nullptr;
    float k_scale = 1.f, v_scale = 1.f;      // static FP8 KV scales shipped in the checkpoint

    // dense MLP (layer 0 only)
    const uint16_t *mlp_gate = nullptr, *mlp_up = nullptr, *mlp_down = nullptr;

    // MoE
    const uint16_t *router = nullptr;        // [E, H]
    const float   *router_bias = nullptr;    // [E]  e_score_correction_bias (SELECTION only)
    const uint16_t *sh_gate = nullptr, *sh_up = nullptr, *sh_down = nullptr;

    // routed experts, contiguous over E:
    //   gate/up packed [E][MI][H/2]    scale [E][MI][H/group]
    //   down   packed [E][H][MI/2]     scale [E][H][MI/group]
    const uint8_t *e_gate_p = nullptr, *e_up_p = nullptr, *e_down_p = nullptr;
    const uint8_t *e_gate_s = nullptr, *e_up_s = nullptr, *e_down_s = nullptr;
    // 1/weight_global_scale, PRE-INVERTED at load. LOOP_LOG A1.2: the checkpoint stores a
    // reciprocal (2688/amax) so dequant must divide; inverting once here lets every kernel
    // multiply instead.
    const float *e_gate_inv = nullptr, *e_up_inv = nullptr, *e_down_inv = nullptr;
};

struct Weights {
    Config cfg;
    void*  arena = nullptr;
    size_t arena_bytes = 0;
    const uint16_t *embed = nullptr, *lm_head = nullptr, *final_norm = nullptr;
    std::vector<LayerW> L;
    double load_seconds = 0.0;
    size_t peak_rss_kb = 0;
    bool   from_cache = false;
};

inline size_t rss_kb() {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256]; size_t v = 0;
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, "VmHWM:", 6)) { sscanf(line + 6, "%zu", &v); break; }
    fclose(f);
    return v;
}

inline double wall_now() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ---------------------------------------------------------------- a shard, read in bulk
class Shard {
public:
    Shard(const std::string& path, std::vector<char>& buf) {
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("open " + path + ": " + strerror(errno));
        struct stat sb; fstat(fd_, &sb);
        size_ = (size_t)sb.st_size;
        if (buf.size() < size_) buf.resize(size_);
        char* p = buf.data();
        size_t off = 0;
        while (off < size_) {
            size_t want = std::min<size_t>(64u << 20, size_ - off);
            ssize_t r = pread(fd_, p + off, want, off);
            if (r <= 0) throw std::runtime_error("pread " + path + ": " + strerror(errno));
            off += (size_t)r;
        }
        uint64_t hlen; memcpy(&hlen, p, 8);
        data_ = p + 8 + hlen;
        hdr_ = nlohmann::json::parse(p + 8, p + 8 + hlen);
    }
    ~Shard() { if (fd_ >= 0) { posix_fadvise(fd_, 0, 0, POSIX_FADV_DONTNEED); close(fd_); } }

    struct View { const char* p; size_t n; std::string dtype; };
    View get(const std::string& n) const {
        const auto& e = hdr_.at(n);
        const auto& off = e.at("data_offsets");
        size_t a = off[0].get<size_t>(), b = off[1].get<size_t>();
        return View{data_ + a, b - a, e.at("dtype").get<std::string>()};
    }
    size_t bytes() const { return size_; }

private:
    int fd_ = -1;
    size_t size_ = 0;
    const char* data_ = nullptr;
    nlohmann::json hdr_;
};

// ---------------------------------------------------------------- loader
class Loader {
public:
    Loader(std::string model_dir, Config cfg)
        : dir_(std::move(model_dir)), cfg_(std::move(cfg)) {}

    // cache_path: if present and matching, take the fast path; otherwise cold-load and write
    // it. Pass "" to disable the cache.
    Weights load(const std::string& cache_path = "", bool verbose = true) {
        double t0 = wall_now();
        W_.cfg = cfg_;
        W_.L.resize(cfg_.n_layers);
        read_index();
        reserve();
        CUDA_CHECK(cudaMalloc(&W_.arena, arena_bytes_));
        W_.arena_bytes = arena_bytes_;
        for (auto& f : fixups_) *f.slot = (const char*)W_.arena + f.off;
        if (verbose)
            printf("[loader] arena %.3f GB in 1 allocation, %zu reservations\n",
                   arena_bytes_ / 1e9, fixups_.size());

        if (!cache_path.empty() && cache_valid(cache_path)) {
            load_cache(cache_path, verbose);
            W_.from_cache = true;
        } else {
            plan();
            stream(verbose);
            if (!cache_path.empty()) save_cache(cache_path, verbose);
        }
        W_.load_seconds = wall_now() - t0;
        W_.peak_rss_kb = rss_kb();
        return W_;
    }

private:
    struct Fix { const void** slot; size_t off; };
    // rp: 0 = plain copy, 1 = repack packed codes, 2 = repack e4m3 scales.
    // rows/K describe the ORIGINAL [rows][K] logical shape of that projection.
    struct Slot { void* dst; size_t bytes; const char* dtype; float* host_scalar;
                  int rp = 0, rows = 0, K = 0; };

    template <class T>
    void reserve1(const T*& field, size_t bytes) {
        bytes = (bytes + 255) & ~(size_t)255;               // 256B align every buffer
        fixups_.push_back(Fix{(const void**)&field, arena_bytes_});
        arena_bytes_ += bytes;
    }

    void read_index() {
        std::ifstream f(dir_ + "/model.safetensors.index.json");
        if (!f) throw std::runtime_error("cannot open index in " + dir_);
        nlohmann::json j; f >> j;
        for (auto it = j["weight_map"].begin(); it != j["weight_map"].end(); ++it)
            index_[it.key()] = it.value().get<std::string>();
        if (index_.empty()) throw std::runtime_error("empty weight_map");
    }

    void reserve() {
        const auto& c = cfg_;
        const size_t H = c.hidden, V = c.vocab, E = c.n_experts;
        const size_t MI = c.moe_intermediate, GRP = c.nvfp4_group;
        reserve1(W_.embed,      (size_t)V * H * 2);
        reserve1(W_.lm_head,    (size_t)V * H * 2);
        reserve1(W_.final_norm, H * 2);
        for (int L = 0; L < c.n_layers; ++L) {
            auto& w = W_.L[L];
            size_t qd = c.q_dim(L), kvd = (size_t)c.n_kv_heads * c.head_dim;
            reserve1(w.q, qd * H * 2);   reserve1(w.k, kvd * H * 2);
            reserve1(w.v, kvd * H * 2);  reserve1(w.o, H * qd * 2);
            reserve1(w.g, (size_t)c.heads[L] * H * 2);
            reserve1(w.in_ln, H * 2);            reserve1(w.post_ln, H * 2);
            reserve1(w.q_norm, c.head_dim * 2);  reserve1(w.k_norm, c.head_dim * 2);
            if (c.is_dense(L)) {
                size_t I = c.intermediate;
                reserve1(w.mlp_gate, I * H * 2); reserve1(w.mlp_up, I * H * 2);
                reserve1(w.mlp_down, H * I * 2);
            } else {
                size_t SI = c.shared_intermediate;
                reserve1(w.router, E * H * 2);   reserve1(w.router_bias, E * 4);
                reserve1(w.sh_gate, SI * H * 2); reserve1(w.sh_up, SI * H * 2);
                reserve1(w.sh_down, H * SI * 2);
                reserve1(w.e_gate_p, E * MI * (H / 2));
                reserve1(w.e_up_p,   E * MI * (H / 2));
                reserve1(w.e_down_p, E * H * (MI / 2));
                reserve1(w.e_gate_s, E * MI * (H / GRP));
                reserve1(w.e_up_s,   E * MI * (H / GRP));
                reserve1(w.e_down_s, E * H * (MI / GRP));
                reserve1(w.e_gate_inv, E * 4);
                reserve1(w.e_up_inv,   E * 4);
                reserve1(w.e_down_inv, E * 4);
            }
        }
    }

    void put(const std::string& n, const void* dst, size_t b, const char* dt) {
        slots_[n] = Slot{(void*)dst, b, dt, nullptr, 0, 0, 0};
    }
    void put_rp(const std::string& n, const void* dst, size_t b, const char* dt,
                int rp, int rows, int K) {
        slots_[n] = Slot{(void*)dst, b, dt, nullptr, rp, rows, K};
    }
    void put_scalar(const std::string& n, float* h) {
        slots_[n] = Slot{nullptr, 2, "BF16", h, 0, 0, 0};
    }

    void plan() {
        const auto& c = cfg_;
        const size_t H = c.hidden, V = c.vocab, E = c.n_experts;
        const size_t MI = c.moe_intermediate, GRP = c.nvfp4_group;
        put("model.embed_tokens.weight", W_.embed,   (size_t)V * H * 2, "BF16");
        put("lm_head.weight",            W_.lm_head, (size_t)V * H * 2, "BF16");
        put("model.norm.weight",         W_.final_norm, H * 2, "BF16");
        for (int L = 0; L < c.n_layers; ++L) {
            auto& w = W_.L[L];
            std::string p = "model.layers." + std::to_string(L) + ".";
            size_t qd = c.q_dim(L), kvd = (size_t)c.n_kv_heads * c.head_dim;
            put(p + "self_attn.q_proj.weight", w.q, qd * H * 2, "BF16");
            put(p + "self_attn.k_proj.weight", w.k, kvd * H * 2, "BF16");
            put(p + "self_attn.v_proj.weight", w.v, kvd * H * 2, "BF16");
            put(p + "self_attn.o_proj.weight", w.o, H * qd * 2, "BF16");
            put(p + "self_attn.g_proj.weight", w.g, (size_t)c.heads[L] * H * 2, "BF16");
            put(p + "input_layernorm.weight",  w.in_ln, H * 2, "BF16");
            put(p + "post_attention_layernorm.weight", w.post_ln, H * 2, "BF16");
            put(p + "self_attn.q_norm.weight", w.q_norm, c.head_dim * 2, "BF16");
            put(p + "self_attn.k_norm.weight", w.k_norm, c.head_dim * 2, "BF16");
            put_scalar(p + "self_attn.k_scale", &w.k_scale);
            put_scalar(p + "self_attn.v_scale", &w.v_scale);
            if (c.is_dense(L)) {
                size_t I = c.intermediate;
                put(p + "mlp.gate_proj.weight", w.mlp_gate, I * H * 2, "BF16");
                put(p + "mlp.up_proj.weight",   w.mlp_up,   I * H * 2, "BF16");
                put(p + "mlp.down_proj.weight", w.mlp_down, H * I * 2, "BF16");
            } else {
                size_t SI = c.shared_intermediate;
                put(p + "mlp.gate.weight", w.router, E * H * 2, "BF16");
                put(p + "mlp.experts.e_score_correction_bias", w.router_bias, E * 4, "F32");
                put(p + "mlp.shared_expert.gate_proj.weight", w.sh_gate, SI * H * 2, "BF16");
                put(p + "mlp.shared_expert.up_proj.weight",   w.sh_up,   SI * H * 2, "BF16");
                put(p + "mlp.shared_expert.down_proj.weight", w.sh_down, H * SI * 2, "BF16");
                for (size_t e = 0; e < E; ++e) {
                    std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
                    put_rp(ep + "gate_proj.weight_packed", w.e_gate_p + e * MI * (H / 2),  MI * (H / 2), "U8", 1, (int)MI, (int)H);
                    put_rp(ep + "up_proj.weight_packed",   w.e_up_p   + e * MI * (H / 2),  MI * (H / 2), "U8", 1, (int)MI, (int)H);
                    put_rp(ep + "down_proj.weight_packed", w.e_down_p + e * H  * (MI / 2), H * (MI / 2), "U8", 1, (int)H, (int)MI);
                    put_rp(ep + "gate_proj.weight_scale",  w.e_gate_s + e * MI * (H / GRP),  MI * (H / GRP), "F8_E4M3", 2, (int)MI, (int)H);
                    put_rp(ep + "up_proj.weight_scale",    w.e_up_s   + e * MI * (H / GRP),  MI * (H / GRP), "F8_E4M3", 2, (int)MI, (int)H);
                    put_rp(ep + "down_proj.weight_scale",  w.e_down_s + e * H  * (MI / GRP), H * (MI / GRP), "F8_E4M3", 2, (int)H, (int)MI);
                    gs_[ep + "gate_proj.weight_global_scale"] = {(const float*)w.e_gate_inv, e};
                    gs_[ep + "up_proj.weight_global_scale"]   = {(const float*)w.e_up_inv,   e};
                    gs_[ep + "down_proj.weight_global_scale"] = {(const float*)w.e_down_inv, e};
                }
            }
        }
    }

    void stream(bool verbose) {
        std::map<std::string, std::vector<std::string>> by_shard;
        for (auto& kv : index_) by_shard[kv.second].push_back(kv.first);
        std::map<const float*, std::vector<float>> gs_host;
        // Size the read buffer to the LARGEST shard once. Letting std::vector grow
        // organically doubled peak RSS (9.67 -> the realloc holds old+new simultaneously).
        std::vector<char> buf;
        {
            size_t mx = 0;
            for (auto& sh : by_shard) {
                struct stat sb;
                if (stat((dir_ + "/" + sh.first).c_str(), &sb) == 0)
                    mx = std::max(mx, (size_t)sb.st_size);
            }
            buf.resize(mx);
            if (verbose) printf("[loader] read buffer %.2f GB (largest shard)\n", mx / 1e9);
        }
        size_t done = 0, total = index_.size();
        double tio = 0, tcp = 0;
        for (auto& sh : by_shard) {
            double ta = wall_now();
            Shard f(dir_ + "/" + sh.first, buf);
            double tb = wall_now(); tio += tb - ta;
            for (const auto& name : sh.second) {
                auto git = gs_.find(name);
                if (git != gs_.end()) {
                    auto v = f.get(name);
                    float x; memcpy(&x, v.p, 4);
                    auto& vec = gs_host[git->second.first];
                    if (vec.empty()) vec.assign(cfg_.n_experts, 0.f);
                    vec[git->second.second] = (x != 0.f) ? 1.0f / x : 0.f;
                    ++done; continue;
                }
                auto sit = slots_.find(name);
                if (sit == slots_.end()) { ++unmapped_; continue; }
                Slot& s = sit->second;
                auto v = f.get(name);
                if (v.dtype != s.dtype)
                    throw std::runtime_error("dtype mismatch " + name + ": " + v.dtype +
                                             " != " + s.dtype);
                if (s.host_scalar) {
                    uint16_t bits; memcpy(&bits, v.p, 2);
                    unsigned u = (unsigned)bits << 16; float fv; memcpy(&fv, &u, 4);
                    *s.host_scalar = fv; filled_.insert(name); ++done; continue;
                }
                if (v.n != s.bytes) throw std::runtime_error("size mismatch " + name);
                if (s.rp) {
                    if (rp_buf_.size() < s.bytes) rp_buf_.resize(s.bytes);
                    if (s.rp == 1) repack_packed(rp_buf_.data(), (const uint8_t*)v.p, s.rows, s.K);
                    else           repack_scale (rp_buf_.data(), (const uint8_t*)v.p, s.rows, s.K,
                                                 cfg_.nvfp4_group);
                    CUDA_CHECK(cudaMemcpy(s.dst, rp_buf_.data(), s.bytes, cudaMemcpyHostToDevice));
                } else {
                    CUDA_CHECK(cudaMemcpy(s.dst, v.p, s.bytes, cudaMemcpyHostToDevice));
                }
                filled_.insert(name); ++done;
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            tcp += wall_now() - tb;
            if (verbose)
                printf("[loader] %s  %zu/%zu  io %.1fs copy %.1fs  rss %.1f GB\n",
                       sh.first.c_str(), done, total, tio, tcp, rss_kb() / 1048576.0);
        }
        for (auto& kv : gs_host)
            CUDA_CHECK(cudaMemcpy((void*)kv.first, kv.second.data(), kv.second.size() * 4,
                                  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<std::string> missing;
        for (auto& kv : slots_) if (!filled_.count(kv.first)) missing.push_back(kv.first);
        if (!missing.empty()) {
            fprintf(stderr, "[loader] %zu planned tensors missing, e.g. %s\n",
                    missing.size(), missing[0].c_str());
            throw std::runtime_error("checkpoint incomplete");
        }
        if (verbose && unmapped_)
            printf("[loader] %zu checkpoint tensors intentionally unused "
                   "(input_global_scale — we do W4A16, no activation quant)\n", unmapped_);
    }

    // ---- repack cache: byte image of the arena, then the per-layer KV scalars
    size_t scalars_bytes() const { return (size_t)cfg_.n_layers * 8; }
    std::string cache_stamp() const {
        char b[256] = {0};
        snprintf(b, sizeof b, "LAGUNA-ARENA-v1 bytes=%zu layers=%d experts=%d H=%d V=%d",
                 arena_bytes_, cfg_.n_layers, cfg_.n_experts, cfg_.hidden, cfg_.vocab);
        return b;
    }
    bool cache_valid(const std::string& p) const {
        int fd = open(p.c_str(), O_RDONLY);
        if (fd < 0) return false;
        char hdr[256] = {0};
        bool ok = pread(fd, hdr, 256, 0) == 256 && cache_stamp() == std::string(hdr);
        struct stat sb; fstat(fd, &sb);
        ok = ok && (size_t)sb.st_size == 256 + arena_bytes_ + scalars_bytes();
        close(fd);
        return ok;
    }
    void save_cache(const std::string& p, bool verbose) {
        double t0 = wall_now();
        std::string tmp = p + ".tmp";
        int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) { fprintf(stderr, "[loader] cannot write %s\n", tmp.c_str()); return; }
        char hdr[256] = {0}; snprintf(hdr, sizeof hdr, "%s", cache_stamp().c_str());
        auto wr = [&](const void* d, size_t n) {
            size_t w = 0;
            while (w < n) { ssize_t r = write(fd, (const char*)d + w, n - w);
                            if (r <= 0) return false; w += (size_t)r; }
            return true;
        };
        if (!wr(hdr, 256)) { close(fd); return; }
        const size_t CH = 256u << 20;
        std::vector<char> buf(CH);
        for (size_t off = 0; off < arena_bytes_; off += CH) {
            size_t n = std::min(CH, arena_bytes_ - off);
            CUDA_CHECK(cudaMemcpy(buf.data(), (const char*)W_.arena + off, n, cudaMemcpyDeviceToHost));
            if (!wr(buf.data(), n)) { close(fd); return; }
        }
        for (auto& l : W_.L) { if (!wr(&l.k_scale, 4) || !wr(&l.v_scale, 4)) { close(fd); return; } }
        fsync(fd); close(fd);
        rename(tmp.c_str(), p.c_str());
        if (verbose) printf("[loader] wrote repack cache %.1f GB in %.1fs\n",
                            arena_bytes_ / 1e9, wall_now() - t0);
    }
    void load_cache(const std::string& p, bool verbose) {
        double t0 = wall_now();
        int fd = open(p.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cache vanished");
        posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
        const size_t CH = 256u << 20;
        std::vector<char> buf(CH);
        for (size_t off = 0; off < arena_bytes_; off += CH) {
            size_t n = std::min(CH, arena_bytes_ - off), got = 0;
            while (got < n) {
                ssize_t r = pread(fd, buf.data() + got, n - got, 256 + off + got);
                if (r <= 0) throw std::runtime_error("cache read failed");
                got += (size_t)r;
            }
            CUDA_CHECK(cudaMemcpy((char*)W_.arena + off, buf.data(), n, cudaMemcpyHostToDevice));
        }
        for (int i = 0; i < cfg_.n_layers; ++i) {
            if (pread(fd, &W_.L[i].k_scale, 4, 256 + arena_bytes_ + (size_t)i * 8) != 4 ||
                pread(fd, &W_.L[i].v_scale, 4, 256 + arena_bytes_ + (size_t)i * 8 + 4) != 4)
                throw std::runtime_error("cache scalar read failed");
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        close(fd);
        double dt = wall_now() - t0;
        if (verbose) printf("[loader] cache load %.3f GB in %.1fs (%.2f GB/s)\n",
                            arena_bytes_ / 1e9, dt, arena_bytes_ / 1e9 / dt);
    }

    std::string dir_;
    Config cfg_;
    Weights W_;
    std::map<std::string, std::string> index_;
    std::map<std::string, Slot> slots_;
    std::map<std::string, std::pair<const float*, size_t>> gs_;
    std::set<std::string> filled_;
    std::vector<Fix> fixups_;
    std::vector<uint8_t> rp_buf_;
    size_t arena_bytes_ = 0, unmapped_ = 0;
};

} // namespace laguna
