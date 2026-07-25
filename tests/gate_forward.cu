// G8: full autoregressive forward vs the oracle's golden greedy output.
#include "../src/forward.cu"
#include <algorithm>
#include <fstream>
#include <cmath>
using namespace laguna;

static std::vector<int> load_ids(const std::string& p){
    std::ifstream f(p); nlohmann::json j; f>>j;
    std::vector<int> v; for(auto&x:j["ids"]) v.push_back(x.get<int>()); return v;
}
int main(int argc,char**argv){
    std::string md = "models/Laguna-S-2.1-NVFP4";
    int CTX = getenv("CTX")?atoi(getenv("CTX")):4096;
    int GEN = getenv("GEN")?atoi(getenv("GEN")):8;
    std::ifstream mf("docs/golden/meta_primes.json"); nlohmann::json META; mf>>META;
    std::vector<int> ids = load_ids("docs/golden/meta_primes.json");
    std::vector<int> want; for(auto&x:META["greedy_ids"]) want.push_back(x.get<int>());

    Config c = load_config(md+"/config.json");
    printf("[G8] loading weights...\n");
    Loader ld(md,c); Weights W = ld.load("",false);
    Engine E; E.c=c; E.W=W; E.init(std::max(64,(int)ids.size()));
    E.prof = getenv("LG_PROF")!=nullptr;
    Session S; S.alloc(c,CTX,E.MAXTOK);
    printf("[G8] KV %.3f GB for ctx=%d  (12 global full + 36 sliding rings of %d)\n",
           S.bytes/1e9,CTX,c.sliding_window);

    int* d_ids; CUDA_CHECK(cudaMalloc(&d_ids,ids.size()*4));
    CUDA_CHECK(cudaMemcpy(d_ids,ids.data(),ids.size()*4,cudaMemcpyHostToDevice));

    double t0=wall_now();
    set_base(E.dbase,0,0);
    E.forward(S,d_ids,(int)ids.size(),0);
    CUDA_CHECK(cudaDeviceSynchronize());
    double tp=wall_now()-t0;
    printf("[G8] prefill %zu tok in %.2fs (%.1f tok/s)\n",ids.size(),tp,ids.size()/tp);

    // greedy decode
    std::vector<float> lg(c.vocab);
    auto argmax_last=[&](int M){
        CUDA_CHECK(cudaMemcpy(lg.data(),E.logits+(size_t)(M-1)*c.vocab,c.vocab*4,cudaMemcpyDeviceToHost));
        int b=0; float bv=lg[0];
        for(int i=1;i<c.vocab;++i) if(lg[i]>bv){bv=lg[i];b=i;}
        return b;
    };
    std::vector<int> got;
    int nxt=argmax_last((int)ids.size());
    got.push_back(nxt);
    int pos=(int)ids.size();
    double td=0;
    for(int i=1;i<GEN;++i){
        CUDA_CHECK(cudaMemcpy(d_ids,&nxt,4,cudaMemcpyHostToDevice));
        double a=wall_now();
        if(E.graph_ready) E.step_graph(pos); else { set_base(E.dbase,pos,0); E.forward(S,d_ids,1,pos); }
        CUDA_CHECK(cudaDeviceSynchronize());
        td+=wall_now()-a;
        pos++; nxt=argmax_last(1); got.push_back(nxt);
    }
    printf("[G8] decode %d tok in %.3fs = %.2f tok/s\n",GEN-1,td,(GEN-1)/td);
    E.prof_report(GEN-1);
    printf("[G8] got : "); for(int x:got) printf("%d ",x); printf("\n");
    printf("[G8] want: "); for(size_t i=0;i<want.size()&&i<got.size();++i) printf("%d ",want[i]); printf("\n");
    size_t ncmp=std::min(got.size(),want.size());
    int match=0; for(size_t i=0;i<ncmp;++i){ if(got[i]==want[i]) ++match; else break; }
    printf("[G8] leading greedy match: %d / %zu compared (golden has %zu) %s\n",
           match,ncmp,want.size(),(size_t)match==ncmp?"PASS":"FAIL");
    return (size_t)match==ncmp?0:1;
}
