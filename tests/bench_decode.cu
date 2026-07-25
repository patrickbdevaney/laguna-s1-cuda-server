// Repeatable decode benchmark: median of N steps at a fixed context, plus a correctness
// re-check on the same run. Thermal drift on this box is real (gemma saw the same code
// swing 94<->108 tok/s), so every optimisation A/B must use this, back to back.
#include "../src/forward.cu"
#include <algorithm>
#include <fstream>
using namespace laguna;
int main(int argc,char**argv){
    std::string md="models/Laguna-S-2.1-NVFP4";
    int CTX=getenv("CTX")?atoi(getenv("CTX")):4096;
    int N=getenv("N")?atoi(getenv("N")):40;
    int PRE=getenv("PRE")?atoi(getenv("PRE")):0;     // synthetic prefill length
    Config c=load_config(md+"/config.json");
    bool FP8A = getenv("LG_FP8ATTN")!=nullptr;
    Loader ld(md,c,FP8A); Weights W=ld.load("",false);
    if(FP8A) printf("FP8 attention ON: arena %.3f GB\n", W.arena_bytes/1e9);
    Engine E; E.c=c; E.W=W; E.init(64);
    Session S; S.alloc(c,CTX,E.MAXTOK);
    E.prof = getenv("LG_PROF")!=nullptr;
    E.moe_split = getenv("LG_SPLIT")!=nullptr;

    std::ifstream mf("docs/golden/meta_primes.json"); nlohmann::json M; mf>>M;
    std::vector<int> ids; for(auto&x:M["ids"]) ids.push_back(x.get<int>());
    std::vector<int> want; for(auto&x:M["greedy_ids"]) want.push_back(x.get<int>());
    int* d; CUDA_CHECK(cudaMalloc(&d,64*4));

    // optional synthetic context so we can measure the long-context curve
    int pos=0;
    if(PRE>0){ std::vector<int> pad(64, ids[3]);
        for(int i=0;i<PRE;i+=64){ int n=std::min(64,PRE-i);
            CUDA_CHECK(cudaMemcpy(d,pad.data(),n*4,cudaMemcpyHostToDevice));
            set_base(E.dbase,pos,0); E.forward(S,d,n,pos); pos+=n; } CUDA_CHECK(cudaDeviceSynchronize()); }

    CUDA_CHECK(cudaMemcpy(d,ids.data(),ids.size()*4,cudaMemcpyHostToDevice));
    set_base(E.dbase,pos,0);
    double t0=wall_now(); E.forward(S,d,(int)ids.size(),pos); CUDA_CHECK(cudaDeviceSynchronize());
    double tpre=wall_now()-t0; pos+=ids.size();

    std::vector<float> lg(c.vocab);
    auto amax=[&](int M_){ CUDA_CHECK(cudaMemcpy(lg.data(),E.logits+(size_t)(M_-1)*c.vocab,c.vocab*4,cudaMemcpyDeviceToHost));
        int b=0; float bv=lg[0]; for(int i=1;i<c.vocab;++i) if(lg[i]>bv){bv=lg[i];b=i;} return b; };
    bool USEG = getenv("LG_NOGRAPH")==nullptr;
    if(USEG) E.capture(S,d,pos);
    int nxt=amax((int)ids.size());
    std::vector<int> got{nxt};
    std::vector<double> ts;
    for(int i=1;i<N;++i){
        CUDA_CHECK(cudaMemcpy(d,&nxt,4,cudaMemcpyHostToDevice));
        double a=wall_now();
        if(USEG){ if(E.needs_recapture(pos)) E.capture(S,d,pos); E.step_graph(pos); }
        else { set_base(E.dbase,pos,0); E.forward(S,d,1,pos); }
        CUDA_CHECK(cudaDeviceSynchronize());
        ts.push_back(wall_now()-a);
        pos++; nxt=amax(1); got.push_back(nxt);
    }
    std::sort(ts.begin(),ts.end());
    double med=ts[ts.size()/2], best=ts.front();
    size_t ncmp=std::min(got.size(),want.size());
    int match=0; for(size_t i=0;i<ncmp;++i){ if(got[i]==want[i]) ++match; else break; }
    double Btok=10.0444e9;
    printf("ctx=%d pre=%d  prefill %zu tok %.2fs (%.1f tok/s)\n",CTX,PRE,ids.size(),tpre,ids.size()/tpre);
    printf("DECODE median %.2f tok/s   best %.2f   (n=%zu, ctx@end=%d)\n",1.0/med,1.0/best,ts.size(),pos);
    printf("effective BW (median) = %.1f GB/s = %.0f%% of 250\n",Btok/med/1e9,Btok/med/1e9/250*100);
    printf("greedy match %d/%zu %s\n",match,ncmp,(size_t)match==ncmp?"OK":"MISMATCH");
    E.prof_report((int)ts.size());
    return 0;
}
