// Gate L1: load the full checkpoint to device, report peak memory and timing.
#include "../include/laguna_weights.h"
#include <cstdio>
int main(int argc,char**argv){
    using namespace laguna;
    std::string md = argc>1?argv[1]:"models/Laguna-S-2.1-NVFP4";
    Config c = load_config(md+"/config.json");
    size_t freeB,totB; CUDA_CHECK(cudaMemGetInfo(&freeB,&totB));
    printf("[gate L1] before: device free %.2f GB / total %.2f GB\n",freeB/1e9,totB/1e9);
    std::string cache = argc>2?argv[2]:"cache/laguna_arena.bin";
    Loader ld(md,c);
    Weights W = ld.load(cache,true);
    CUDA_CHECK(cudaMemGetInfo(&freeB,&totB));
    printf("\n[gate L1] arena %.3f GB  load %.1f s  peak RSS %.2f GB  from_cache=%d  device free now %.2f GB\n",
           W.arena_bytes/1e9, W.load_seconds, W.peak_rss_kb/1048576.0, (int)W.from_cache, freeB/1e9);
    // spot-check a few values landed
    uint16_t h[4]; CUDA_CHECK(cudaMemcpy(h,W.embed,8,cudaMemcpyDeviceToHost));
    printf("embed[0..3] bf16 bits: %04x %04x %04x %04x\n",h[0],h[1],h[2],h[3]);
    printf("L0 k_scale=%g v_scale=%g   L47 k_scale=%g v_scale=%g\n",
           W.L[0].k_scale,W.L[0].v_scale,W.L[47].k_scale,W.L[47].v_scale);
    float inv[3]; CUDA_CHECK(cudaMemcpy(inv,W.L[1].e_gate_inv,12,cudaMemcpyDeviceToHost));
    printf("L1 e_gate_inv[0..2]=%g %g %g  (expect 1/11520=%.9g for e0)\n",inv[0],inv[1],inv[2],1.0/11520.0);
    return 0;
}
