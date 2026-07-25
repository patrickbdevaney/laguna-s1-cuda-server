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
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

extern "C" {
void dequant_nvfp4(float*,const uint8_t*,const uint8_t*,float,int,int,int,cudaStream_t);
void gemm_bf16(float*,const uint16_t*,const uint16_t*,int,int,int,cudaStream_t);
void gemm_fp4(float*,const uint8_t*,const uint8_t*,float,const uint16_t*,int,int,int,int,cudaStream_t);
void rmsnorm(float*,const float*,const uint16_t*,int,int,float,cudaStream_t);
void rope_tables(float*,float*,const float*,int,int,int,float,cudaStream_t);
void router(int*,float*,float*,const float*,const float*,int,int,int,float,int,cudaStream_t);
void f32_to_bf16(uint16_t*,const float*,long,cudaStream_t);
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

    printf("\n%d passed, %d failed\n",PASS,FAIL);
    return FAIL?1:0;
}
