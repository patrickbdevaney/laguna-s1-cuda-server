// bench_attnq.cu — does NVFP4 on attention weights actually CONVERT to speed?
//
// Capability is already settled: all five attention projections at NVFP4 leave greedy output
// bit-identical (8/8), measured by loader simulation. The open question is the one the trellis
// work failed on, and it is not a capability question at all:
//
//     A byte saving only becomes a speed-up if the kernel is BANDWIDTH-bound.
//
// The trellis decoder read 1.406x fewer bytes and still lost, because it sat at 91% of its own
// ALU ceiling -- fewer bytes bought nothing the ALU would not give. So before writing a W4A16
// attention kernel, measure whether an NVFP4 dequant at attention shapes stays bandwidth-bound.
//
// THE INSTRUMENT THAT MAKES THIS DECIDABLE is the NOMEM twin: the identical arithmetic with the
// weights synthesised in-register and zero weight traffic. A kernel at its NOMEM ceiling is
// compute-bound and cannot be helped by reading less; a kernel far below it is memory-bound and
// converts. In GB/s these two look the same, which is exactly how v2 of the trellis benchmark
// reached a confident wrong answer.
//
// Shapes are the real ones. o_proj on a sliding layer is [3072 x 9216]; q_proj is the same size
// transposed, which is why they are byte-identical at 1.2457 GB each and why q+o is the target.
// Layout is the production repack: [row][chunk][lane], 16 B per lane, so a warp reads 32
// consecutive chunks = 512 B fully coalesced.
//
//   fp8      1 byte/weight + one fp32 scale per output row   (ships today)
//   nvfp4    0.5 byte payload + 1/16 byte E4M3 group scale = 0.5625 B/weight
//   stream   reads the nvfp4-sized buffer, decodes nothing   (the ceiling)
//
// Headline is WEIGHTS PER SECOND. Per token the model must process a fixed weight count, so
// time = weights / (weights per second); GB/s flatters whichever encoding reads more.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e_)); exit(1);} } while(0)

#define K_LEN 3072                    // hidden; the reduction axis of o_proj
#define CH_F8 (K_LEN/(16*32))         // fp8:   16 weights per 16 B lane-chunk -> 6 chunks
#define CH_F4 (K_LEN/(32*32))         // nvfp4: 32 weights per 16 B lane-chunk -> 3 chunks

__device__ __forceinline__ float warp_sum(float v){
    #pragma unroll
    for(int o=16;o;o>>=1) v+=__shfl_down_sync(0xFFFFFFFFu,v,o);
    return v;
}

// ---------------- FP8 W8A16, one fp32 scale per output row (what ships)
__global__ void k_fp8(const uint4* __restrict__ W, const float* __restrict__ rs,
                      const float* __restrict__ xg, float* __restrict__ out, int nrows){
    extern __shared__ float sm[]; float* sx=sm;
    for(int i=threadIdx.x;i<K_LEN;i+=blockDim.x) sx[i]=xg[i];
    __syncthreads();
    const int lane=threadIdx.x&31;
    const int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5;
    const int nwarp=(gridDim.x*blockDim.x)>>5;
    for(int row=warp;row<nrows;row+=nwarp){
        float acc=0.f; const float s=rs[row];
        #pragma unroll
        for(int c=0;c<CH_F8;++c){
            uint4 w=W[((size_t)row*CH_F8+c)*32+lane];
            const uint8_t* b=(const uint8_t*)&w;   // 16 e4m3 codes
            int kb=c*(16*32)+lane*16;
            #pragma unroll
            for(int j=0;j<16;++j){
                // e4m3 -> float by the standard bit surgery (no LUT: a __constant__ LUT indexed
                // by a runtime value serialises across a warp, which is a trap this repo has
                // already been burned by).
                uint32_t u=b[j]; uint32_t sg=(u>>7)&1u, e=(u>>3)&0xFu, m=u&7u;
                float v = e? __int_as_float(((e+120u)<<23)|(m<<20))
                           : (float)m * 1.52587890625e-05f;
                acc=fmaf(sg? -v*s : v*s, sx[kb+j], acc);
            }
        }
        acc=warp_sum(acc); if(lane==0) out[row]=acc;
    }
}

// ---------------- NVFP4 W4A16: E2M1 codes + per-16 E4M3 group scale
__global__ void k_nvfp4(const uint4* __restrict__ W, const uint16_t* __restrict__ gs,
                        const float* __restrict__ xg, float* __restrict__ out,
                        int nrows, float ginv){
    extern __shared__ float sm[]; float* sx=sm; float* lut=sm+K_LEN;
    for(int i=threadIdx.x;i<K_LEN;i+=blockDim.x) sx[i]=xg[i];
    if(threadIdx.x<8){ const float e[8]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f}; lut[threadIdx.x]=e[threadIdx.x]; }
    __syncthreads();
    const int lane=threadIdx.x&31;
    const int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5;
    const int nwarp=(gridDim.x*blockDim.x)>>5;
    for(int row=warp;row<nrows;row+=nwarp){
        float acc=0.f;
        #pragma unroll
        for(int c=0;c<CH_F4;++c){
            size_t idx=((size_t)row*CH_F4+c)*32+lane;
            uint4 w=W[idx];
            uint32_t sp=gs[idx];                    // two packed E4M3 group scales
            float s0=__int_as_float((((sp&0x7Fu)>>3)+120u)<<23) * ginv;
            float s1=__int_as_float(((((sp>>8)&0x7Fu)>>3)+120u)<<23) * ginv;
            const uint32_t* wp=&w.x; int kb=c*(32*32)+lane*32;
            #pragma unroll
            for(int j=0;j<4;++j){
                uint32_t v=wp[j]; float sc=(j<2)?s0:s1;
                #pragma unroll
                for(int n=0;n<8;++n){
                    uint32_t c4=(v>>(4*n))&0xFu;
                    acc=fmaf(lut[c4&7u]*((c4&8u)?-sc:sc), sx[kb+j*8+n], acc);
                }
            }
        }
        acc=warp_sum(acc); if(lane==0) out[row]=acc;
    }
}

// ---------------- ceiling: read the nvfp4-sized buffer, decode nothing
__global__ void k_stream(const uint4* __restrict__ W, float* __restrict__ out, int nrows){
    const int lane=threadIdx.x&31;
    const int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5;
    const int nwarp=(gridDim.x*blockDim.x)>>5;
    uint32_t a=0;
    for(int row=warp;row<nrows;row+=nwarp)
        #pragma unroll
        for(int c=0;c<CH_F4;++c){ uint4 w=W[((size_t)row*CH_F4+c)*32+lane]; a^=w.x^w.y^w.z^w.w; }
    if(lane==0) out[warp]=(float)a;
}

// ---------------- NOMEM twins: identical arithmetic, zero weight traffic
__global__ void k_nomem_f8(const float* __restrict__ xg, float* __restrict__ out, int nrows, uint32_t seed){
    extern __shared__ float sm[]; float* sx=sm;
    for(int i=threadIdx.x;i<K_LEN;i+=blockDim.x) sx[i]=xg[i];
    __syncthreads();
    const int lane=threadIdx.x&31;
    const int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5, nwarp=(gridDim.x*blockDim.x)>>5;
    for(int row=warp;row<nrows;row+=nwarp){
        float acc=0.f, s=1.0009f;
        #pragma unroll
        for(int c=0;c<CH_F8;++c){
            uint32_t r=seed^(uint32_t)(row*31+c); int kb=c*(16*32)+lane*16;
            #pragma unroll
            for(int j=0;j<16;++j){
                uint32_t u=(r>>((j&3)*8))&0xFFu;
                uint32_t sg=(u>>7)&1u,e=(u>>3)&0xFu,m=u&7u;
                float v=e?__int_as_float(((e+120u)<<23)|(m<<20)):(float)m*1.52587890625e-05f;
                acc=fmaf(sg?-v*s:v*s, sx[kb+j], acc);
            }
        }
        acc=warp_sum(acc); if(lane==0) out[row]=acc;
    }
}
__global__ void k_nomem_f4(const float* __restrict__ xg, float* __restrict__ out, int nrows, uint32_t seed){
    extern __shared__ float sm[]; float* sx=sm; float* lut=sm+K_LEN;
    for(int i=threadIdx.x;i<K_LEN;i+=blockDim.x) sx[i]=xg[i];
    if(threadIdx.x<8){ const float e[8]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f}; lut[threadIdx.x]=e[threadIdx.x]; }
    __syncthreads();
    const int lane=threadIdx.x&31;
    const int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5, nwarp=(gridDim.x*blockDim.x)>>5;
    for(int row=warp;row<nrows;row+=nwarp){
        float acc=0.f;
        #pragma unroll
        for(int c=0;c<CH_F4;++c){
            uint32_t r=seed^(uint32_t)(row*31+c);
            uint32_t sp=r&0xFFFFu;                       // stands in for the loaded group scales
            float s0=__int_as_float((((sp&0x7Fu)>>3)+120u)<<23);
            float s1=__int_as_float(((((sp>>8)&0x7Fu)>>3)+120u)<<23);
            int kb=c*(32*32)+lane*32;
            #pragma unroll
            for(int j=0;j<4;++j){ uint32_t v=r*(j+1); float sc=(j<2)?s0:s1;
                #pragma unroll
                for(int n=0;n<8;++n){ uint32_t c4=(v>>(4*n))&0xFu;
                    acc=fmaf(lut[c4&7u]*((c4&8u)?-sc:sc), sx[kb+j*8+n], acc); } }
        }
        acc=warp_sum(acc); if(lane==0) out[row]=acc;
    }
}

int main(int argc,char** argv){
    // Default rows give working sets far larger than the 32 MB L2, so this measures DRAM.
    int nrows = argc>1? atoi(argv[1]) : 262144;
    size_t bf8=(size_t)nrows*CH_F8*32*16, bf4=(size_t)nrows*CH_F4*32*16;
    size_t sf4=(size_t)nrows*CH_F4*32*2;

    uint4 *Wf8,*Wf4; uint16_t* GS; float *RS,*x,*out;
    CK(cudaMalloc(&Wf8,bf8)); CK(cudaMalloc(&Wf4,bf4)); CK(cudaMalloc(&GS,sf4));
    CK(cudaMalloc(&RS,(size_t)nrows*4)); CK(cudaMalloc(&x,K_LEN*4));
    CK(cudaMalloc(&out,(size_t)nrows*4+(1<<20)));
    CK(cudaMemset(Wf8,0x3A,bf8)); CK(cudaMemset(Wf4,0x5A,bf4));
    CK(cudaMemset(GS,0x38,sf4)); CK(cudaMemset(RS,0x3C,(size_t)nrows*4));
    CK(cudaMemset(x,0x3C,K_LEN*4));

    int th=256, bl=20*8;
    size_t sh_f8=K_LEN*4, sh_f4=(K_LEN+8)*4;
    for(int i=0;i<200;++i) k_stream<<<bl,th>>>(Wf4,out,nrows);
    CK(cudaDeviceSynchronize());

    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    double WT=(double)nrows*K_LEN;
    auto run=[&](const char* n, auto fn, double bytes){
        for(int i=0;i<3;++i) fn();
        CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        CK(cudaEventRecord(a)); const int R=20;
        for(int i=0;i<R;++i) fn();
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms=0; CK(cudaEventElapsedTime(&ms,a,b)); double s=(ms/R)*1e-3;
        printf("  %-11s %8.3f ms  %7.1f GB/s  %8.2f Gweight/s\n",n,ms/R,bytes/s/1e9,WT/s/1e9);
        return WT/s;
    };

    printf("o_proj-shaped GEMV: %d rows x %d (%.2f Gweight), production repack, x in shared\n",
           nrows,K_LEN,WT/1e9);
    printf("  fp8 %.2f GB (1.0000 B/w)   nvfp4 %.2f GB (0.5625 B/w)   ratio %.3fx\n\n",
           (bf8+(size_t)nrows*4)/1e9,(bf4+sf4)/1e9,
           (double)(bf8+(size_t)nrows*4)/(double)(bf4+sf4));

    double st=run("stream",[&]{k_stream<<<bl,th>>>(Wf4,out,nrows);},(double)bf4);
    double f8=run("fp8",   [&]{k_fp8   <<<bl,th,sh_f8>>>(Wf8,RS,x,out,nrows);},(double)(bf8+(size_t)nrows*4));
    double f4=run("nvfp4", [&]{k_nvfp4 <<<bl,th,sh_f4>>>(Wf4,GS,x,out,nrows,1.0f);},(double)(bf4+sf4));
    printf("\n  ALU ceilings (no weight traffic):\n");
    double c8=run("nomem_fp8",[&]{k_nomem_f8<<<bl,th,sh_f8>>>(x,out,nrows,0x1234);},0.0);
    double c4=run("nomem_nvfp4",[&]{k_nomem_f4<<<bl,th,sh_f4>>>(x,out,nrows,0x1234);},0.0);

    const double br=(double)(bf8+(size_t)nrows*4)/(double)(bf4+sf4);
    printf("\n  ---- verdict ----\n");
    printf("  byte ratio fp8/nvfp4              %.3fx  (the saving theoretically available)\n",br);
    printf("  measured Gweight/s nvfp4/fp8      %.3fx  (the saving that actually arrives)\n",f4/f8);
    printf("  conversion efficiency             %.1f%%\n",100.0*(f4/f8)/br);
    printf("  nvfp4 vs its OWN ALU ceiling      %.1f%%  (near 100%% => compute-bound, bytes cannot help)\n",100.0*f4/c4);
    printf("  fp8   vs its OWN ALU ceiling      %.1f%%\n",100.0*f8/c8);
    printf("  nvfp4 vs pure-read ceiling        %.1f%%\n",100.0*f4/st);
    printf("\n  Projected end-to-end for q+o (1.090 GB of 6.251): +%.1f%%\n",
           100.0*(6.251/(6.251-1.090*((f4/f8)/br))-1.0));
    return 0;
}
