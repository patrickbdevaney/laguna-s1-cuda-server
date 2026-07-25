// Does attend() (NSP=1) agree with attend_split() (NSP>1)? They are different kernels, and
// the engine silently picks between them by M: at M=64 the split heuristic returns NSP=1 so
// prefill uses k_attn, while decode at M=1 returns NSP=10 and uses k_attn_split. If they
// disagree, batched prefill and single-token decode cannot agree either.
// Synthetic inputs, no model load.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
extern "C" {
void attend(float*,const float*,const uint8_t*,const uint8_t*,float,float,int,int,int,int,int,const int*,float,cudaStream_t);
void attend_split(float*,float*,float*,const float*,const uint8_t*,const uint8_t*,float,float,int,int,int,int,int,const int*,float,int,cudaStream_t);
int  attend_nsplit(int,int,int);
}
int main(){
    const int HD=128, nkv=8;
    int fails=0;
    for (int G : {6, 9}) {
      for (int len : {64, 200, 512, 900}) {
        const int M=1, cap = 4096, window = (G==9)? 512 : 0;
        int base = len-1;                        // query at the end of `len` keys
        int nh = nkv*G;
        std::vector<float> q((size_t)M*nh*HD);
        for (auto& v : q) v = (float)((rand()%2000)-1000)/1000.f;
        std::vector<uint8_t> k((size_t)nkv*cap*HD), v8((size_t)nkv*cap*HD);
        for (auto& x : k)  x = (uint8_t)(rand()%256);
        for (auto& x : v8) x = (uint8_t)(rand()%256);
        float *dq,*o1,*o2,*pacc,*pml; uint8_t *dk,*dv; int* db;
        CK(cudaMalloc(&dq,q.size()*4)); CK(cudaMemcpy(dq,q.data(),q.size()*4,cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dk,k.size())); CK(cudaMemcpy(dk,k.data(),k.size(),cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dv,v8.size())); CK(cudaMemcpy(dv,v8.data(),v8.size(),cudaMemcpyHostToDevice));
        CK(cudaMalloc(&o1,q.size()*4)); CK(cudaMalloc(&o2,q.size()*4));
        CK(cudaMalloc(&pacc,(size_t)M*nh*32*HD*4)); CK(cudaMalloc(&pml,(size_t)M*nh*32*2*4));
        CK(cudaMalloc(&db,4)); CK(cudaMemcpy(db,&base,4,cudaMemcpyHostToDevice));
        const float qs = 1.f/sqrtf((float)HD);

        attend(o1,dq,dk,dv,0.03f,0.02f,M,nkv,G,cap,window,db,qs,0);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<float> h1(q.size()); CK(cudaMemcpy(h1.data(),o1,q.size()*4,cudaMemcpyDeviceToHost));

        for (int nsp : {2, 4, 8, 16}) {
            attend_split(o2,pacc,pml,dq,dk,dv,0.03f,0.02f,M,nkv,G,cap,window,db,qs,nsp,0);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            std::vector<float> h2(q.size()); CK(cudaMemcpy(h2.data(),o2,q.size()*4,cudaMemcpyDeviceToHost));
            double mx=0,sc=0;
            for(size_t i=0;i<h1.size();++i){ mx=fmax(mx,fabs((double)h1[i]-h2[i])); sc=fmax(sc,fabs((double)h1[i])); }
            double rel = mx/fmax(sc,1e-9);
            bool ok = rel < 1e-5;
            if(!ok) ++fails;
            printf("  G=%d len=%-4d NSP=%-3d  maxabs=%9.3e rel=%9.3e  %s\n",G,len,nsp,mx,rel,ok?"OK":"MISMATCH");
        }
        cudaFree(dq);cudaFree(dk);cudaFree(dv);cudaFree(o1);cudaFree(o2);
        cudaFree(pacc);cudaFree(pml);cudaFree(db);
      }
    }
    printf("\nattend() vs attend_split(): %s\n", fails? "DISAGREE - this is the prefill/decode gap" : "AGREE");
    return fails?1:0;
}
