// Why is the cold load 966 MB/s? Measure H2D paths for a 2 GB shard-like source.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %s: %s\n",#x,cudaGetErrorString(e));exit(1);} }while(0)
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(int argc,char**argv){
    const char* path = argc>1?argv[1]:"models/Laguna-S-2.1-NVFP4/model-00001-of-00015.safetensors";
    int fd=open(path,O_RDONLY); if(fd<0){perror("open");return 1;}
    struct stat sb; fstat(fd,&sb);
    size_t n = sb.st_size > (2ull<<30) ? (2ull<<30) : (size_t)sb.st_size;
    void* mp=mmap(nullptr,n,PROT_READ,MAP_PRIVATE,fd,0);
    void* d; CK(cudaMalloc(&d,n));
    printf("source %s  %.2f GB\n",path,n/1e9);

    // fault in
    volatile char s=0; for(size_t i=0;i<n;i+=4096) s+=((char*)mp)[i];
    double t=now(); CK(cudaMemcpy(d,mp,n,cudaMemcpyHostToDevice)); CK(cudaDeviceSynchronize());
    printf("  one big memcpy from warm pageable mmap : %6.2f GB/s\n", n/1e9/(now()-t));

    // many small copies, like 145k tensors
    size_t chunk=1572864;  // 1.5 MB, the size of one expert gate_proj weight_packed
    t=now(); for(size_t off=0;off+chunk<=n;off+=chunk) CK(cudaMemcpy((char*)d+off,(char*)mp+off,chunk,cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());
    printf("  %zu x 1.5 MB copies (as the loader does): %6.2f GB/s\n", n/chunk, n/1e9/(now()-t));

    // pinned staging
    void* pin; CK(cudaHostAlloc(&pin,64u<<20,cudaHostAllocDefault));
    t=now(); for(size_t off=0;off<n;off+=(64u<<20)){ size_t c=(n-off)<(64u<<20)?(n-off):(64u<<20);
        memcpy(pin,(char*)mp+off,c); CK(cudaMemcpy((char*)d+off,pin,c,cudaMemcpyHostToDevice)); }
    CK(cudaDeviceSynchronize());
    printf("  memcpy->pinned->H2D 64MB chunks        : %6.2f GB/s\n", n/1e9/(now()-t));

    // hostRegister the mmap then copy
    cudaError_t e=cudaHostRegister(mp,n,cudaHostRegisterReadOnly);
    if(!e){ t=now(); CK(cudaMemcpy(d,mp,n,cudaMemcpyHostToDevice)); CK(cudaDeviceSynchronize());
        printf("  hostRegister(mmap) + one big memcpy     : %6.2f GB/s\n", n/1e9/(now()-t));
        CK(cudaHostUnregister(mp)); }
    else printf("  hostRegister failed: %s\n",cudaGetErrorString(e));

    // cold read from NVMe into pinned (drop cache first)
    posix_fadvise(fd,0,0,POSIX_FADV_DONTNEED);
    t=now(); size_t got=0;
    for(size_t off=0;off<n;off+=(64u<<20)){ size_t c=(n-off)<(64u<<20)?(n-off):(64u<<20);
        ssize_t r=pread(fd,pin,c,off); if(r>0){got+=r; CK(cudaMemcpy((char*)d+off,pin,r,cudaMemcpyHostToDevice));} }
    CK(cudaDeviceSynchronize());
    printf("  COLD pread->pinned->H2D 64MB chunks     : %6.2f GB/s\n", got/1e9/(now()-t));
    return 0;
}
