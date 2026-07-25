// Gate B1 harness: every kernel checked against oracle-derived references.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include "../include/third_party/json.hpp"
#include "../include/laguna_config.h"
#include "../include/laguna_weights.h"
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

extern "C" {
void dequant_nvfp4(float*,const uint8_t*,const uint8_t*,float,int,int,int,cudaStream_t);
void gemm_bf16(float*,const uint16_t*,const uint16_t*,int,int,int,cudaStream_t);
void gemm_fp4(float*,const uint8_t*,const uint8_t*,float,const uint16_t*,int,int,int,int,cudaStream_t);
void rmsnorm(float*,const float*,const uint16_t*,int,int,float,cudaStream_t);
void rope_tables(float*,float*,const float*,int,int,int,float,cudaStream_t);
void router(int*,float*,float*,const float*,const float*,int,int,int,float,int,cudaStream_t);
void f32_to_bf16(uint16_t*,const float*,long,cudaStream_t);
void store_kv(uint8_t*,uint8_t*,const float*,const float*,float,float,int,int,int,int,int,cudaStream_t);
void attend(float*,const float*,const uint8_t*,const uint8_t*,float,float,int,int,int,int,int,int,float,cudaStream_t);
void moe_invert(int*,int*,int*,int*,int*,int*,const int*,int,int,int,cudaStream_t);
void moe_gateup(float*,const uint8_t*,const uint8_t*,const float*,const uint8_t*,const uint8_t*,const float*,const uint16_t*,const int*,const int*,const int*,const int*,const int*,int,int,int,int,int,int,cudaStream_t);
void moe_down(float*,const uint8_t*,const uint8_t*,const float*,const uint16_t*,const int*,const int*,const int*,const int*,const int*,int,int,int,int,int,cudaStream_t);
void moe_finalize(float*,const float*,const float*,const int*,int,int,int,float,cudaStream_t);
}

static std::string DIR="docs/kernel_refs/";
static nlohmann::json REFS;
template<class T> std::vector<T> loadbin(const std::string& tag){
    std::string p=DIR+tag+".bin";
    std::ifstream f(p,std::ios::binary); if(!f){printf("missing %s\n",p.c_str());exit(1);}
    f.seekg(0,std::ios::end); size_t n=f.tellg(); f.seekg(0);
    std::vector<T> v(n/sizeof(T)); f.read((char*)v.data(),n); return v;
}
template<class T> T* todev(const std::vector<T>& h){ void* d; CK(cudaMalloc(&d,h.size()*sizeof(T)));
    CK(cudaMemcpy(d,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice)); return (T*)d; }

static int PASS=0, FAIL=0;
static void report(const char* name,double maxabs,double maxrel,double tol,bool extra=true){
    bool ok = maxrel<=tol && extra;
    printf("  %-28s maxabs=%9.3e  maxrel=%9.3e  tol=%7.1e  %s\n",name,maxabs,maxrel,tol,ok?"PASS":"FAIL");
    ok?++PASS:++FAIL;
}
static void cmp(const char* name,const std::vector<float>& got,const std::vector<float>& want,double tol){
    double ma=0,mr=0,scale=0;
    for(size_t i=0;i<want.size();++i) scale=std::max(scale,(double)std::fabs(want[i]));
    for(size_t i=0;i<want.size();++i){ double d=std::fabs((double)got[i]-want[i]); ma=std::max(ma,d); }
    mr = ma/std::max(scale,1e-30);
    report(name,ma,mr,tol);
}

int main(){
    { std::ifstream f(DIR+"refs.json"); if(!f){printf("run oracle/dump_kernel_refs.py first\n");return 1;} f>>REFS; }
    int H=REFS["config"]["hidden_size"], E=REFS["config"]["num_experts"];
    int TOPK=REFS["config"]["num_experts_per_tok"], S=REFS["seq_len"];
    float EPS=REFS["config"]["rms_norm_eps"];

    printf("\n=== G1  NVFP4 dequant ===\n");
    {
        auto pk=loadbin<uint8_t>("g1_packed"), sc=loadbin<uint8_t>("g1_scale");
        auto want=loadbin<float>("g1_out");
        int N=REFS["g1_out"]["shape"][0], K=REFS["g1_out"]["shape"][1];
        float inv=REFS["g1_inv_gs"];
        auto*dp=todev(pk); auto*ds=todev(sc); float*dout;
        CK(cudaMalloc(&dout,(size_t)N*K*4));
        dequant_nvfp4(dout,dp,ds,inv,N,K,16,0); CK(cudaDeviceSynchronize());
        std::vector<float> got((size_t)N*K); CK(cudaMemcpy(got.data(),dout,got.size()*4,cudaMemcpyDeviceToHost));
        // bit-exactness is achievable here: both sides are exact products of exact values
        size_t nbits=0; for(size_t i=0;i<got.size();++i) if(got[i]!=want[i]) ++nbits;
        printf("  %-28s mismatching elements = %zu / %zu  %s\n","bit-exact",nbits,got.size(),nbits?"FAIL":"PASS");
        nbits?++FAIL:++PASS;
    }

    printf("\n=== G2  dense linear (BF16 weights) ===\n");
    {
        auto w=loadbin<uint16_t>("g2_w"), x=loadbin<uint16_t>("g2_x");
        auto want=loadbin<float>("g2_out");
        int N=REFS["g2_w"]["shape"][0], K=REFS["g2_w"]["shape"][1], M=REFS["g2_x"]["shape"][0];
        auto*dw=todev(w); auto*dx=todev(x); float*dout; CK(cudaMalloc(&dout,(size_t)M*N*4));
        gemm_bf16(dout,dw,dx,M,N,K,0); CK(cudaDeviceSynchronize());
        std::vector<float> got((size_t)M*N); CK(cudaMemcpy(got.data(),dout,got.size()*4,cudaMemcpyDeviceToHost));
        cmp("vs torch fp32 matmul",got,want,2e-6);
    }

    printf("\n=== G2b dense linear (NVFP4 weights) ===\n");
    {
        auto pk=loadbin<uint8_t>("g1_packed"), sc=loadbin<uint8_t>("g1_scale");
        auto x=loadbin<uint16_t>("g2b_x"); auto want=loadbin<float>("g2b_out");
        int N=REFS["g2b_out"]["shape"][1], K=H, M=REFS["g2b_x"]["shape"][0];
        float inv=REFS["g1_inv_gs"];
        auto*dp=todev(pk); auto*ds=todev(sc); auto*dx=todev(x);
        float*dout; CK(cudaMalloc(&dout,(size_t)M*N*4));
        gemm_fp4(dout,dp,ds,inv,dx,M,N,K,16,0); CK(cudaDeviceSynchronize());
        std::vector<float> got((size_t)M*N); CK(cudaMemcpy(got.data(),dout,got.size()*4,cudaMemcpyDeviceToHost));
        cmp("vs torch fp32 matmul",got,want,2e-6);
    }

    printf("\n=== G3a RMSNorm ===\n");
    {
        auto x=loadbin<float>("g3a_x"); auto w=loadbin<uint16_t>("g3a_w"); auto want=loadbin<float>("g3a_out");
        auto*dx=todev(x); auto*dw=todev(w); float*dout; CK(cudaMalloc(&dout,x.size()*4));
        rmsnorm(dout,dx,dw,S,H,EPS,0); CK(cudaDeviceSynchronize());
        std::vector<float> got(x.size()); CK(cudaMemcpy(got.data(),dout,got.size()*4,cudaMemcpyDeviceToHost));
        cmp("vs reference",got,want,2e-6);
    }

    printf("\n=== G3b rope tables ===\n");
    for(const char* tag : {"full","slide"}){
        auto wc=loadbin<float>(std::string("g3b_cos_")+tag), ws=loadbin<float>(std::string("g3b_sin_")+tag);
        int rot=REFS[std::string("g3b_cos_")+tag]["shape"][1];
        // inv_freq + scale come from the C++ config path, dumped by tests/dump_rope
        std::ifstream rf("build/rope_cpp.json"); nlohmann::json RJ; rf>>RJ;
        std::string key = std::string(tag)=="full"?"full":"slide";
        std::vector<float> inv; for(auto&v:RJ[key]["inv"]) inv.push_back(v.get<float>());
        float scale=RJ[key]["scale"].get<float>();
        auto*dinv=todev(inv); float *dc,*ds; CK(cudaMalloc(&dc,wc.size()*4)); CK(cudaMalloc(&ds,ws.size()*4));
        rope_tables(dc,ds,dinv,S,0,rot/2,scale,0); CK(cudaDeviceSynchronize());
        std::vector<float> gc(wc.size()),gs(ws.size());
        CK(cudaMemcpy(gc.data(),dc,gc.size()*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(gs.data(),ds,gs.size()*4,cudaMemcpyDeviceToHost));
        cmp((std::string("cos ")+tag).c_str(),gc,wc,3e-5);
        cmp((std::string("sin ")+tag).c_str(),gs,ws,3e-5);
    }

    printf("\n=== G4  sigmoid router + selection-only bias + top-%d ===\n",TOPK);
    {
        auto x=loadbin<float>("g4_x"); auto w=loadbin<uint16_t>("g4_w"); auto bias=loadbin<float>("g4_bias");
        auto wsel=loadbin<int32_t>("g4_sel"); auto wwts=loadbin<float>("g4_wts");
        auto*dx=todev(x); auto*dw=todev(w); auto*db=todev(bias);
        uint16_t* dxb; CK(cudaMalloc(&dxb,x.size()*2)); f32_to_bf16(dxb,dx,x.size(),0);
        float* dlg; CK(cudaMalloc(&dlg,(size_t)S*E*4));
        gemm_bf16(dlg,dw,dxb,S,E,H,0); CK(cudaGetLastError());
        int* dsel; float* dwts; CK(cudaMalloc(&dsel,(size_t)S*TOPK*4)); CK(cudaMalloc(&dwts,(size_t)S*TOPK*4));
        router(dsel,dwts,nullptr,dlg,db,S,E,TOPK,0.f,1,0); CK(cudaDeviceSynchronize());
        std::vector<int32_t> gsel((size_t)S*TOPK); std::vector<float> gwts((size_t)S*TOPK);
        CK(cudaMemcpy(gsel.data(),dsel,gsel.size()*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(gwts.data(),dwts,gwts.size()*4,cudaMemcpyDeviceToHost));
        size_t bad=0; for(size_t i=0;i<gsel.size();++i) if(gsel[i]!=wsel[i]) ++bad;
        printf("  %-28s mismatching indices = %zu / %zu  %s\n","top-k selection INDEX-EXACT",bad,gsel.size(),bad?"FAIL":"PASS");
        bad?++FAIL:++PASS;
        cmp("normalised weights",gwts,wwts,5e-6);
    }

    printf("\n=== G5  head-packed GQA over FP8 KV ===\n");
    for(const char* tag : {"full","slide"}){
        auto& J=REFS[std::string("g5_")+tag];
        int nh=J["nh"],nkv=J["nkv"],hd=J["hd"],G=J["G"],win=J["window"],Sm=J["S"];
        float ks=J["k_scale"],vs=J["v_scale"];
        auto q=loadbin<float>(std::string("g5_q_")+tag);
        auto k=loadbin<float>(std::string("g5_k_")+tag);
        auto v=loadbin<float>(std::string("g5_v_")+tag);
        auto want=loadbin<float>(std::string("g5_out_")+tag);
        int cap = win? win : Sm;
        auto*dq=todev(q); auto*dk=todev(k); auto*dv=todev(v);
        uint8_t *dKc,*dVc; CK(cudaMalloc(&dKc,(size_t)nkv*cap*hd)); CK(cudaMalloc(&dVc,(size_t)nkv*cap*hd));
        CK(cudaMemset(dKc,0,(size_t)nkv*cap*hd)); CK(cudaMemset(dVc,0,(size_t)nkv*cap*hd));
        store_kv(dKc,dVc,dk,dv,ks,vs,Sm,nkv,hd,cap,0,0); CK(cudaGetLastError());
        float* dout; CK(cudaMalloc(&dout,q.size()*4));
        attend(dout,dq,dKc,dVc,ks,vs,Sm,nkv,G,cap,win,0,1.0f/sqrtf((float)hd),0);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<float> got(q.size()); CK(cudaMemcpy(got.data(),dout,got.size()*4,cudaMemcpyDeviceToHost));
        cmp((std::string("attention ")+tag+" (G="+std::to_string(G)+")").c_str(),got,want,5e-6);
    }

    printf("\n=== G6  grouped routed-expert MoE (256 experts, top-10) ===\n");
    {
        using namespace laguna;
        auto& J=REFS["g6"];
        int Lr=J["layer"],rows=J["rows"],E=J["E"],TK=J["topk"],MI=J["MI"];
        float scaling=J["scaling"];
        Config c = load_config("models/Laguna-S-2.1-NVFP4/config.json");
        Loader ld("models/Laguna-S-2.1-NVFP4", c);
        printf("  (loading real expert weights for layer %d ...)\n", Lr);
        Weights W = ld.load("", false);
        const LayerW& lw = W.L[Lr];

        auto x=loadbin<float>("g6_x"); auto sel=loadbin<int32_t>("g6_sel");
        auto wt=loadbin<float>("g6_wts"); auto want=loadbin<float>("g6_out");
        auto*dx=todev(x); auto*dsel=todev(sel); auto*dwt=todev(wt);
        uint16_t* dxb; CK(cudaMalloc(&dxb,x.size()*2)); f32_to_bf16(dxb,dx,x.size(),0);

        int nass=rows*TK;
        int *ec,*eo,*el,*cur,*act,*na;
        CK(cudaMalloc(&ec,E*4)); CK(cudaMalloc(&eo,(E+1)*4)); CK(cudaMalloc(&el,nass*4));
        CK(cudaMalloc(&cur,E*4)); CK(cudaMalloc(&act,E*4)); CK(cudaMalloc(&na,4));
        moe_invert(ec,eo,el,cur,act,na,dsel,rows,TK,E,0); CK(cudaGetLastError());
        int h_na; CK(cudaMemcpy(&h_na,na,4,cudaMemcpyDeviceToHost));
        printf("  active experts = %d of %d for %d tokens x top-%d\n",h_na,E,rows,TK);

        float* hbuf; CK(cudaMalloc(&hbuf,(size_t)nass*MI*4));
        moe_gateup(hbuf,lw.e_gate_p,lw.e_gate_s,lw.e_gate_inv,lw.e_up_p,lw.e_up_s,lw.e_up_inv,
                   dxb,el,eo,ec,act,na,h_na,c.hidden,MI,TK,c.nvfp4_group,4,0);
        CK(cudaGetLastError());
        uint16_t* hbf; CK(cudaMalloc(&hbf,(size_t)nass*MI*2));
        f32_to_bf16(hbf,hbuf,(long)nass*MI,0);
        float* dpart; CK(cudaMalloc(&dpart,(size_t)nass*c.hidden*4));
        moe_down(dpart,lw.e_down_p,lw.e_down_s,lw.e_down_inv,hbf,el,eo,ec,act,na,h_na,
                 c.hidden,MI,c.nvfp4_group,4,0);
        CK(cudaGetLastError());
        float* dout; CK(cudaMalloc(&dout,(size_t)rows*c.hidden*4));
        moe_finalize(dout,dpart,dwt,dsel,rows,c.hidden,TK,scaling,0);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<float> got((size_t)rows*c.hidden);
        CK(cudaMemcpy(got.data(),dout,got.size()*4,cudaMemcpyDeviceToHost));
        cmp("routed experts vs reference",got,want,3e-5);
        CK(cudaFree(W.arena));
    }

    printf("\n%d passed, %d failed\n",PASS,FAIL);
    return FAIL?1:0;
}
