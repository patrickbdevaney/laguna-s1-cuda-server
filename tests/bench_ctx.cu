// Measured decode-vs-context curve. ROOFLINE.md 6 claims KV capacity is free and only USED
// context costs bandwidth; this measures it on the real engine instead of modelling it.
#include "../src/forward.cu"
#include <algorithm>
#include <fstream>
using namespace laguna;
int main(){
    std::string md="models/Laguna-S-2.1-NVFP4";
    int CTX=getenv("CTX")?atoi(getenv("CTX")):65536;
    int N=getenv("N")?atoi(getenv("N")):16;
    Config c=load_config(md+"/config.json");
    Loader ld(md,c); Weights W=ld.load("",false);
    Engine E; E.c=c; E.W=W; E.init(256);
    Session S; S.alloc(c,CTX);
    printf("KV alloc %.3f GB for CTX=%d\n",S.bytes/1e9,CTX);

    std::vector<int> pad(256,1000);
    int* d; CUDA_CHECK(cudaMalloc(&d,256*4));
    CUDA_CHECK(cudaMemcpy(d,pad.data(),256*4,cudaMemcpyHostToDevice));
    int pos=0;
    std::vector<int> pts={512,2048,8192,16384,32768,65536};
    printf("\n%10s %12s %12s %10s %8s\n","used ctx","tok/s","GB/s","KV GB/step","vs 512");
    double base=0;
    for(int target:pts){
        if(target>CTX) break;
        while(pos+256<=target){                      // prefill in 256-token chunks
            set_base(E.dbase,pos,0); E.forward(S,d,256,pos); pos+=256;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        E.capture(S,d,pos);
        std::vector<double> ts;
        for(int i=0;i<N;i++){
            double a=wall_now();
            if(E.needs_recapture(pos)) E.capture(S,d,pos);
            E.step_graph(pos); CUDA_CHECK(cudaDeviceSynchronize());
            ts.push_back(wall_now()-a); pos++;
        }
        std::sort(ts.begin(),ts.end());
        double med=ts[ts.size()/2], tps=1.0/med;
        // KV bytes read per step: 12 global layers x used ctx + 36 sliding x window
        double kv=(12.0*pos+36.0*c.sliding_window)*2*c.n_kv_heads*c.head_dim;
        double B=7.4110e9+2.4950e9+kv;
        if(base==0) base=tps;
        printf("%10d %12.2f %12.1f %10.3f %7.0f%%\n",pos,tps,B/med/1e9,kv/1e9,tps/base*100);
    }
    return 0;
}
