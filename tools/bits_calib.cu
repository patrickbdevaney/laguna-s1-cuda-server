// bits_calib.cu -- measure the relative weight error of every candidate expert-quantization
// scheme, against the shipped NVFP4 weights, WITHOUT modifying anything.
//
// This is the step that picks which codec goes into the end-to-end eval. It is cheap (one
// model load, then a few passes per scheme over a layer sample) and it makes the choice a
// measurement instead of an assumption.
//
// Reported per scheme:
//   bpw        exact effective bits/weight = payload_bits + 8/group   (payload plus scales)
//   rel_mse    sum (w_new - w_old)^2 / sum w_old^2 over the sampled routed experts.
//              w_old is the SHIPPED NVFP4 value, not the original bf16 -- we never had the
//              bf16 weights, so every number here is error relative to the 4.5-bit baseline
//              the server actually runs. That is the right reference for this question.
//
//   build:  nvcc -O3 -std=c++17 -arch=sm_110a -I. -o build/bits_calib tools/bits_calib.cu \
//                 kernels/*.cu -lpthread
//   run:    STRIDE=4 ./build/bits_calib
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "../include/laguna_config.h"
#include "../include/laguna_weights.h"
#include "../include/laguna_kernels_api.h"
#include "expert_requant.cuh"

using namespace laguna;

int main(int argc, char** argv) {
    std::string md = "models/Laguna-S-2.1-NVFP4";
    // Sample every STRIDE'th MoE layer. rel_mse is an average over ~113 G weights even at
    // stride 8, so the estimate is not the noisy part of this experiment.
    const int stride = getenv("STRIDE") ? atoi(getenv("STRIDE")) : 4;
    Config c = load_config(md + "/config.json");
    // FP8 flags do not touch the routed experts, but they change arena_bytes and hence the
    // repack-cache stamp. Match the server's defaults so we share its cache file.
    Loader ld(md, c, true, true, true);
    const char* cache = getenv("LG_CACHE") ? getenv("LG_CACHE") : "cache/laguna_arena_fp8.bin";
    Weights W = ld.load(cache, true);
    printf("[calib] arena %.3f GB  load %.1f s  from_cache=%d\n\n",
           W.arena_bytes / 1e9, W.load_seconds, (int)W.from_cache);

    std::vector<std::string> names;
    if (argc > 1) { for (int i = 1; i < argc; ++i) names.push_back(argv[i]); }
    else {
        // magnitude search at a fixed group, then the group ladder for the winners
        const char* m5[] = {"1,4","1.5,4","2,4","1,6","1.5,6","2,6","3,6","1,3","1.5,3","2,3"};
        const char* m7[] = {"1,2,4","1,2,6","1,3,6","1.5,3,6","2,4,6","1.5,4,6","0.5,1.5,4",
                            "1,2,3","0.5,1,2","1,1.5,2"};
        const char* m3[] = {"2","3","4","6"};
        names.push_back("baseline/16");
        for (auto m : m5) names.push_back(std::string(m) + "/16");
        for (auto m : m7) names.push_back(std::string(m) + "/16");
        for (auto m : m3) names.push_back(std::string(m) + "/16");
    }

    printf("%-12s %5s %6s %6s %10s %10s %9s %7s\n",
           "scheme", "lev", "grp", "bpw", "rel_mse", "rel_rms", "clipped", "sec");
    printf("%s\n", std::string(74, '-').c_str());
    for (auto& n : names) {
        lgrq::Scheme s = lgrq::scheme_by_name(n);
        lgrq::Result r = lgrq::run(W, s, /*apply=*/false, stride);
        printf("%-12s %5d %6d %6.3f %10.3e %10.4f %9.0f %7.1f\n",
               s.name, s.nlev(), s.group, s.bpw(), r.rel_mse, sqrt(r.rel_mse),
               r.clipped, r.seconds);
        fflush(stdout);
    }
    printf("\n(layer stride %d; rel_mse is against the SHIPPED NVFP4 weights)\n", stride);
    return 0;
}
