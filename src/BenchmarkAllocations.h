#pragma once
// Diagnostic build only: count C++ operator-new requests in the benchmark's
// single query thread. Direct malloc/Eigen/OpenMP runtime allocations are not
// counted. The uninstrumented performance executable has no replacements.
#ifdef TRIBASE_ENABLE_STATS
#include <cstdlib>
#include <new>
namespace benchmark_allocations {
struct Counts {size_t calls=0,bytes=0,deletes=0;};
inline thread_local bool enabled=false;
inline thread_local Counts counts;
inline void start(){counts={};enabled=true;}
inline Counts stop(){enabled=false;return counts;}
inline void record(size_t size){if(enabled){++counts.calls;counts.bytes+=size;}}
}
void* operator new(std::size_t size) {
    if(void* p=std::malloc(size?size:1)){benchmark_allocations::record(size);return p;}
    throw std::bad_alloc();
}
void* operator new[](std::size_t size){return ::operator new(size);}
void operator delete(void* p) noexcept {if(p&&benchmark_allocations::enabled)++benchmark_allocations::counts.deletes;std::free(p);}
void operator delete[](void* p) noexcept {::operator delete(p);}
void operator delete(void* p,std::size_t) noexcept {::operator delete(p);}
void operator delete[](void* p,std::size_t) noexcept {::operator delete(p);}
void* operator new(std::size_t size,std::align_val_t alignment) {
    void* p=nullptr;if(posix_memalign(&p,static_cast<size_t>(alignment),size?size:1))throw std::bad_alloc();
    benchmark_allocations::record(size);return p;
}
void* operator new[](std::size_t size,std::align_val_t alignment){return ::operator new(size,alignment);}
void operator delete(void* p,std::align_val_t) noexcept {::operator delete(p);}
void operator delete[](void* p,std::align_val_t) noexcept {::operator delete(p);}
void operator delete(void* p,std::size_t,std::align_val_t) noexcept {::operator delete(p);}
void operator delete[](void* p,std::size_t,std::align_val_t) noexcept {::operator delete(p);}
#else
namespace benchmark_allocations {
struct Counts {size_t calls=0,bytes=0,deletes=0;};
inline void start(){}
inline Counts stop(){return {};}
}
#endif
