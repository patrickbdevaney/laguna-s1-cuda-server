// Loader-architecture probe: on Thor's unified memory, is a cudaHostRegister'd mmap of a
// file as fast for the GPU to stream as cudaMalloc? If yes, the repack cache can be mapped
// zero-copy and we never hold two copies of 74 GB.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

__global__ void stream_sum(const uint4* __restrict__ p, size_t n4, unsigned long long* out){
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x*blockDim.x;
    unsigned long long acc=0;
    for(; i<n4; i+=stride){ uint4 v=__ldcs(p+i); acc += v.x+v.y+v.z+v.w; }
    atomicAdd(out, acc);
}

static double bench(const uint4* p, size_t bytes, unsigned long long* d_out, int iters=5){
    size_t n4 = bytes/16;
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    int blocks = 20*8, threads=256;                 // 20 SMs
    stream_sum<<<blocks,threads>>>(p,n4,d_out); CK(cudaDeviceSynchronize());
    double best=1e30;
    for(int i=0;i<iters;i++){
        CK(cudaMemsetAsync(d_out,0,8)); CK(cudaEventRecord(a));
        stream_sum<<<blocks,threads>>>(p,n4,d_out);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms,a,b));
        if(ms<best) best=ms;
    }
    return bytes/1e9/(best/1e3);
}

int main(int argc,char**argv){
    size_t GB = argc>1?atoll(argv[1]):4;
    size_t bytes = GB<<30;
    unsigned long long* d_out; CK(cudaMalloc(&d_out,8));
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr,0));
    printf("device %s  SMs=%d  L2=%.2f MB  persistL2max=%.2f MB  smem/SM=%zu KB\n",
        pr.name,pr.multiProcessorCount,pr.l2CacheSize/1048576.0,
        pr.persistingL2CacheMaxSize/1048576.0,pr.sharedMemPerMultiprocessor/1024);
    printf("unifiedAddressing=%d canMapHost=%d integrated=%d pageableAccess=%d hostRegisterSupported=%d\n",
        pr.unifiedAddressing,pr.canMapHostMemory,pr.integrated,
        pr.pageableMemoryAccess,pr.hostRegisterSupported);

    // 1) plain cudaMalloc
    void* dm; CK(cudaMalloc(&dm,bytes)); CK(cudaMemset(dm,1,bytes));
    printf("\ncudaMalloc            %5.1f GB/s\n", bench((uint4*)dm,bytes,d_out));
    CK(cudaFree(dm));

    // 2) mmap'd file + cudaHostRegister
    const char* path = argc>2?argv[2]:"build/bwprobe.bin";
    int fd = open(path,O_RDWR|O_CREAT,0644);
    if(fd<0){perror("open");return 1;}
    if(ftruncate(fd,bytes)){perror("ftruncate");return 1;}
    void* mp = mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    if(mp==MAP_FAILED){perror("mmap");return 1;}
    // fault the pages in
    for(size_t i=0;i<bytes;i+=4096) ((volatile char*)mp)[i]=1;
    cudaError_t e = cudaHostRegister(mp,bytes,cudaHostRegisterMapped);
    if(e){ printf("cudaHostRegister FAILED: %s\n",cudaGetErrorString(e)); }
    else {
        void* dp=nullptr; CK(cudaHostGetDevicePointer(&dp,mp,0));
        printf("mmap+HostRegister    %5.1f GB/s   (host=%p dev=%p same=%d)\n",
               bench((uint4*)dp,bytes,d_out), mp, dp, mp==dp);
        CK(cudaHostUnregister(mp));
    }
    // 3) plain malloc + register (no file backing)
    void* hm = aligned_alloc(4096,bytes);
    if(hm){ memset(hm,1,bytes);
        e=cudaHostRegister(hm,bytes,cudaHostRegisterMapped);
        if(e) printf("malloc+HostRegister FAILED: %s\n",cudaGetErrorString(e));
        else { void* dp; CK(cudaHostGetDevicePointer(&dp,hm,0));
               printf("malloc+HostRegister  %5.1f GB/s\n", bench((uint4*)dp,bytes,d_out));
               CK(cudaHostUnregister(hm)); }
        free(hm);
    }
    munmap(mp,bytes); close(fd); unlink(path);
    return 0;
}
