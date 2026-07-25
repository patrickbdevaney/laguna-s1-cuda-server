// Gate A1 sub-check: config.json parses, and every derived quantity matches what
// ROOFLINE.md computed independently in Python.
#include "../include/laguna_config.h"
#include <cstdio>
int main(int argc,char**argv){
    using namespace laguna;
    std::string md = argc>1?argv[1]:"models/Laguna-S-2.1-NVFP4";
    Config c = load_config(md+"/config.json");
    printf("hidden=%d layers=%d head_dim=%d n_kv=%d vocab=%d inter=%d eps=%g tied=%d\n",
           c.hidden,c.n_layers,c.head_dim,c.n_kv_heads,c.vocab,c.intermediate,c.rms_eps,(int)c.tie_word_embeddings);
    printf("global=%d sliding=%d window=%d\n",c.n_global(),c.n_sliding(),c.sliding_window);
    printf("experts=%d top_k=%d moe_int=%d shared_int=%d scaling=%g normtopk=%d softcap=%g\n",
           c.n_experts,c.top_k,c.moe_intermediate,c.shared_intermediate,c.routed_scaling,
           (int)c.norm_topk_prob,c.router_softcap);
    printf("nvfp4_group=%d kv_fp8=%d bos=%d pad=%d eos=[",c.nvfp4_group,(int)c.kv_fp8,c.bos,c.pad);
    for(size_t i=0;i<c.eos.size();++i) printf("%s%d",i?",":"",c.eos[i]); printf("]\n");
    printf("dense_layers={"); for(int L:c.dense_layers) printf("%d,",L); printf("}\n");
    for(int L : {0,1,4,47}) printf("  L%-2d %-8s heads=%-3d group=%d qdim=%-5d rope=%s theta=%g rot=%d/%d af=%.16g\n",
        L, c.is_sliding(L)?"sliding":"global", c.heads[L], c.kv_group(L), c.q_dim(L),
        c.rope(L).type.c_str(), c.rope(L).theta, c.rope(L).rotary_dim, c.head_dim, c.rope(L).attention_factor);
    auto ivf=c.rope_full.inv_freq(c.head_dim), ivs=c.rope_slide.inv_freq(c.head_dim);
    printf("inv_freq full n=%zu [0]=%.9g [1]=%.9g [last]=%.9g\n",ivf.size(),ivf[0],ivf[1],ivf.back());
    printf("inv_freq slide n=%zu [0]=%.9g [1]=%.9g [last]=%.9g\n",ivs.size(),ivs[0],ivs[1],ivs.back());
    printf("kv bytes/tok/layer=%ld  -> global-only per token=%ld\n",
           c.kv_bytes_per_token_per_layer(), c.kv_bytes_per_token_per_layer()*c.n_global());
    DraftConfig d = load_draft_config((argc>2?std::string(argv[2]):std::string("models/Laguna-S-2.1-DFlash-NVFP4"))+"/config.json");
    printf("DRAFT layers=%d heads=%d kv=%d hidden=%d inter=%d win=%d BLK=%d mask=%d causal=%d taps=[",
           d.n_layers,d.n_heads,d.n_kv_heads,d.hidden,d.intermediate,d.sliding_window,d.block_size,d.mask_token_id,(int)d.causal);
    for(size_t i=0;i<d.target_layer_ids.size();++i) printf("%s%d",i?",":"",d.target_layer_ids[i]); printf("]\n");
    return 0;
}
