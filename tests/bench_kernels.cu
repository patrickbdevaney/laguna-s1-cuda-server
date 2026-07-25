// Profile-first: which kernel is actually slow at the real decode shapes?
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cstdint>
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %s\n",cudaGetErrorString(e));exit(1);} }while(0)
extern "C" {
void gemm_bf16(float*,const uint16_t*,const uint16_t*,int,int,int,cudaStream_t);
void gemm_fp4(float*,const uint8_t*,const uint8_t*,float,const uint16_t*,int,int,int,int,cudaStream_t);
void rmsnorm(float*,const float*,const uint16_t*,int,int,float,cudaStream_t);
void attend(float*,const float*,const uint8_t*,const uint8_t*,float,float,int,int,int,int,int,int,float,cudaStream_t);
}
static float bench(void(*fn)(),int it=20){
    fn(); CK(cudaDeviceSynchronize());
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a)); for(int i=0;i<it;i++) fn(); CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b)); float ms; CK(cudaEventElapsedTime(&ms,a,b)); return ms/it;
}
static uint16_t *W,*X; static float*O; static uint8_t *P,*Sc; static float*Q; static uint8_t*Kc,*Vc; static float*Oa;
static int gM,gN,gK;
static void run_bf16(){ gemm_bf16(O,W,X,gM,gN,gK,0); }
static void run_fp4(){ gemm_fp4(O,P,Sc,1e-4f,X,gM,gN,gK,16,0); }
static int aM,aNKV,aG,aCAP,aWIN;
static void run_attn(){ attend(Oa,Q,Kc,Vc,1.f,1.f,aM,aNKV,aG,aCAP,aWIN,aCAP-1,0.088f,0); }

int main(){
    size_t big=(size_t)100352*3072;   // sized for lm_head, the largest single weight
    CK(cudaMalloc(&W,big*2)); CK(cudaMalloc(&X,(size_t)64*16384*2)); CK(cudaMalloc(&O,(size_t)8*131072*4));
    CK(cudaMalloc(&P,big/2)); CK(cudaMalloc(&Sc,big/16));
    CK(cudaMemset(W,0x3c,big*2)); CK(cudaMemset(X,0x3c,(size_t)64*16384*2));
    CK(cudaMemset(P,0x22,big/2)); CK(cudaMemset(Sc,0x38,big/16));
    printf("%-34s %6s %7s %7s %9s %9s\n","kernel / shape","M","bytes","ms","GB/s","%%of 227");
    struct C{const char*n;int M,N,K;int fp4;};
    C cs[]={
      {"q_proj sliding [9216,3072]",1,9216,3072,0},
      {"o_proj sliding [3072,9216]",1,3072,9216,0},
      {"q_proj global  [6144,3072]",1,6144,3072,0},
      {"k/v_proj       [1024,3072]",1,1024,3072,0},
      {"shared gate    [1024,3072]",1,1024,3072,0},
      {"router         [ 256,3072]",1,256,3072,0},
      {"lm_head        [100352,3072]",1,100352,3072,0},
      {"expert gate FP4[1024,3072]",1,1024,3072,1},
      {"expert down FP4[3072,1024]",1,3072,1024,1},
      {"q_proj sliding M=6",6,9216,3072,0},
      {"lm_head M=6",6,100352,3072,0},
    };
    for(auto&cc:cs){
        gM=cc.M;gN=cc.N;gK=cc.K;
        double bytes = cc.fp4 ? (double)cc.N*cc.K*(0.5+1.0/16) : (double)cc.N*cc.K*2;
        float ms = cc.fp4?bench(run_fp4):bench(run_bf16);
        printf("%-34s %6d %6.1fM %7.3f %9.1f %8.0f%%\n",cc.n,cc.M,bytes/1e6,ms,bytes/1e9/(ms/1e3),
               bytes/1e9/(ms/1e3)/227*100);
    }
    // attention at decode
    CK(cudaMalloc(&Q,(size_t)64*9216*4)); CK(cudaMalloc(&Oa,(size_t)64*9216*4));
    CK(cudaMalloc(&Kc,(size_t)8*4096*128)); CK(cudaMalloc(&Vc,(size_t)8*4096*128));
    CK(cudaMemset(Kc,0x38,(size_t)8*4096*128)); CK(cudaMemset(Vc,0x38,(size_t)8*4096*128));
    struct A{const char*n;int M,NKV,G,CAP,WIN;};
    A as[]={{"attn sliding win=512 M=1",1,8,9,512,512},{"attn global ctx=4096 M=1",1,8,6,4096,0},
            {"attn sliding win=512 M=6",6,8,9,512,512}};
    for(auto&a:as){
        aM=a.M;aNKV=a.NKV;aG=a.G;aCAP=a.CAP;aWIN=a.WIN;
        double bytes=(double)a.NKV*(a.WIN?a.WIN:a.CAP)*128*2;
        float ms=bench(run_attn);
        printf("%-34s %6d %6.1fM %7.3f %9.1f %8.0f%%\n",a.n,a.M,bytes/1e6,ms,bytes/1e9/(ms/1e3),
               bytes/1e9/(ms/1e3)/227*100);
    }
    // launch overhead
    cudaEvent_t x,y; CK(cudaEventCreate(&x)); CK(cudaEventCreate(&y));
    gM=1;gN=32;gK=128;
    CK(cudaEventRecord(x)); for(int i=0;i<2000;i++) gemm_bf16(O,W,X,1,32,128,0); CK(cudaEventRecord(y));
    CK(cudaEventSynchronize(y)); float ms; CK(cudaEventElapsedTime(&ms,x,y));
    printf("\nper-launch overhead (tiny kernel): %.2f us\n",ms*1000/2000);
    return 0;
}
