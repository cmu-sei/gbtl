#ifndef UTILS_H
#define UTILS_H
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdbool>
#include <memory>
#include <cstdint>
#include <unistd.h>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <vector>
#include <omp.h>
// For mmap
#include <sys/mman.h>
#include <errno.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

int cmp_u32(const void* p1, const void* p2);
unsigned long long rdtsc();
void clean_caches();
char* truncate_fname(char* input_fname);

#define TIME_THIS_RDTSC(EXPR)\
{                      \
	unsigned long long __t1 = rdtsc(); \
	(EXPR);                            \
	unsigned long long __t2 = rdtsc(); \
	printf("TIMING cycles: %f\n", (double)__t2 - (double)__t1); \
}

#define TIME_THIS_OMP(EXPR)\
{                      \
	double __t1 = omp_get_wtime(); \
	(EXPR);                        \
	double __t2 = omp_get_wtime(); \
	_Pragma("omp single") 				 \
	{printf("TIMING seconds: %f\n", (double)__t2 - (double)__t1);}\
}

#define SILENT_TIME_THIS_OMP(total, EXPR)\
{                      \
	double __t1 = omp_get_wtime(); \
	(EXPR);                        \
	double __t2 = omp_get_wtime(); \
	_Pragma("omp single") 				 \
	{total += (double)__t2 - (double)__t1;}\
}

#define PRINT_TOTAL_TIME(total)\
	printf("TOTAL TIME: %f seconds\n", total);

// Create mmap to use hugeTLB memory. Must be freed with munmap(ptr, sz);
template<typename T = uint32_t>
T * create_mmap(size_t num_elems){
	T * mmapped = 
		(T*)mmap(nullptr, 
				num_elems*sizeof(T), 
				PROT_READ | PROT_WRITE, 
				MAP_HUGETLB | MAP_PRIVATE | MAP_ANONYMOUS, 
				-1, 0);
	if (mmapped == MAP_FAILED){
		perror("Could not mmap file");
		return NULL;
	}
	return mmapped;
}

#endif
