/*
 * bfdsp.cu
 *
 *  Created on: Feb 17, 2015
 *  modified: Apr 6, 15
 *      Author: yz79
 */

//we should force the use of texture mem
//https://stackoverflow.com/questions/19860094/is-1d-texture-memory-access-faster-than-1d-global-memory-access
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "graphDriver.cuh"


/*
 * Modes:
 * 0: pure dynamic parallelism, no optimization
 * 1: version 0+streams
 * 2: version 0+warp level cooperation
 */

extern "C"{
void tic();
double toc();
}

#define WARP_SIZE 32
#define LEFTOVER_SIZE 16
//#define KERNEL_TH 512
#define KERNEL_TH 1024
//#define KERNEL_TH 0
//#define KERNEL_TH 256

// Number of vertices to process per child kernel thread, if
// CONFIG_UNROLL_CHILD_KERNEL is set.
#define NPKT 2

using namespace gafa;
using namespace uusi;

__device__ __forceinline__ bool warp_cull(
	int neighborid)
{
	volatile __shared__ uint32_t scratch[WARP_SIZE][128];
    //uint16_t laneid=threadIdx.x & (WARP_SIZE-1);
    uint16_t warpid=threadIdx.x / (WARP_SIZE);
	uint32_t hash=neighborid&127;
	scratch[warpid][hash]=neighborid;
	uint32_t retrieved=scratch[warpid][hash];
	if(retrieved==neighborid){
		scratch[warpid][hash]=threadIdx.x;
		if(scratch[warpid][hash]!=threadIdx.x){
			return true;
		}
	}
	return false;
}

__global__ void kernel2(uint32_t parent,
                        uint32_t size,
                        GPUNode e,
                        BitMap * q,
                        int *parentMap,
                        int *count);

#if CONFIG_WARP_REAP
//warp reaping
__device__ __forceinline__ bool warp_reap(
    uint32_t parent,
    uint32_t neighbors,
    EdgeValue* e,
    BitMap * q,
    int *parentMap)

{
  //warpsize=warp count for 1024 threads.
  __shared__ uint32_t comm[WARP_SIZE][3];
  __shared__ EdgeValue* evs[WARP_SIZE];
  uint16_t laneid=threadIdx.x & (WARP_SIZE-1);
  uint16_t warpid=threadIdx.x / (WARP_SIZE);
  bool updated = false;
  while(__any(neighbors)){
    //per warp: one write will succeed
    if(neighbors){
      comm[warpid][0]=laneid;
    }
    //winner descr:
    if(comm[warpid][0]==laneid){
      comm[warpid][1]=neighbors;
      comm[warpid][2]=parent;
      evs[warpid]=e;
      neighbors=0;
    }
    while(comm[warpid][1]) {
	    if(comm[warpid][1]>=32) {
		    comm[warpid][1]-=32;
		    EdgeValue *edgevalue=evs[warpid];
		    uint32_t node=edgevalue[laneid].dst;
		    //if not marked in parentmap:
		    if(-1 == parentMap[node]) {
			    set_bitmap_atomic(q, node, 1);
			    parentMap[node] = comm[warpid][2];
			    updated = true;
		    }
		    //increment edgevalue pointer:
		    evs[warpid]+=32;
	    }
	    else if(laneid<comm[warpid][1]){
		    comm[warpid][1]=0;
		    EdgeValue *edgevalue=evs[warpid];
		    uint32_t node=edgevalue[laneid].dst;
		    //if not marked in parentmap:
		    if(-1 == parentMap[node]) {
			    set_bitmap_atomic(q, node, 1);
			    parentMap[node] = comm[warpid][2];
			    updated = true;
		    }
	    }
        __threadfence_block();
    }
  }
  return updated;
}
#endif

//very simple kernel for now.
__global__ void kernel1(GPUNode* nodes,
                        uint32_t size,
                        BitMap * q1,
                        BitMap * q2,
                        int *parentMap,
                        int *count)
  /*
     q1 is the current frontier.
     q2 is the next frontier.
   */
{
	int idx=threadIdx.x+blockDim.x*blockIdx.x;
#if CONFIG_WARP_REAP
	__shared__ EdgeValue* leftovers[1024];
	__shared__ uint32_t leftover_size[1024];
	//fillshmemaddrs:
	leftover_size[threadIdx.x]=0;
	leftovers[threadIdx.x]=(EdgeValue*)0x3;
	if(idx<size && get_bitmap_value_at(q1,idx)) {
		uint32_t num_edges=nodes[idx].neighbors;
		if(num_edges>=KERNEL_TH){
			//get remainder:
			int b,t;
#if CONFIG_UNROLL_CHILD_KERNEL
			int thread_count = (num_edges + NPKT - 1)/ NPKT;
#else
			int thread_count = num_edges;
#endif
			get_small_threads_blocks(thread_count, &b, &t, 1024);
			kernel2<<<b,t>>>(idx,num_edges,nodes[idx],q2,parentMap, count);
		}
		else{
			leftovers[threadIdx.x]=nodes[idx].ev;
			leftover_size[threadIdx.x]=nodes[idx].neighbors;
		}
	}
	__syncthreads();
	if(__any(warp_reap(
	                   idx, //parent
	                   leftover_size[threadIdx.x],
	                   leftovers[threadIdx.x],
	                   q2,
	                   parentMap))) {
		*count = 1;
	}
#else
	if(idx<size && get_bitmap_value_at(q1,idx)) {
		uint32_t num_edges=nodes[idx].neighbors;
		int b,t;
		get_small_threads_blocks(num_edges,&b,&t,1024);
		kernel2<<<b,t>>>(idx,num_edges,nodes[idx],q2,parentMap, count);
	}
#endif
}

//do all the visits in kernel2:
//use parentmap as visited map
__global__ void kernel2(uint32_t parent,
                        uint32_t size,
                        GPUNode e,
                        BitMap *q,
                        int *parentMap,
                        int *count)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	bool updated = false;
#if CONFIG_UNROLL_CHILD_KERNEL
	int thread_count = (size + NPKT - 1)/ NPKT;
#pragma unroll NPKT
	while(idx < size) {
#else
    if(idx < size) {
#endif
		uint32_t node = e.ev[idx].dst;
		//if not marked in parentmap:
		// We used to check if this is already in the queue, but it hurt
		// performance to do that.
		if(-1 == parentMap[node]) {
			set_bitmap_value_at(q, node, 1);
			parentMap[node] = parent;
			updated = true;
		}

#if CONFIG_UNROLL_CHILD_KERNEL
		idx += thread_count;
#endif
	}
	if(__any(updated)) {
		*count = 1;
	}
}

#if CONFIG_GPU_DRIVER
__global__ void bfs_driver(int t, int b, int *count,
                           BitMap *q1, BitMap *q2,
                           GPUNode* nodes,
                           uint32_t size,
                           int *parentMap)
{
	*count = 1;
	while(*count != 0) {
		*count = 0;
		kernel1<<<b,t>>>(nodes,
		                 size,
		                 q1,
		                 q2,
		                 parentMap,
		                 count);
		zero_queue_dyn(q1, size);
		// swap the queues
		auto temp = q1;
		q1 = q2;
		q2 = temp;
		cudaDeviceSynchronize();
	}
}
#endif

template <typename Int>
__global__ void pm(BitMap *data, Int size){
    for(int i=0;i<size;i++){
      if(get_bitmap_value_at(data,i))
        printf("%d ", i);
        blkSync();
    }
}

int64_t* run_bfs(GPUGraph n, int source){
    set_max_tpb();
    BitMap *q1,*q2;
    int t,b;
#if !CONFIG_GPU_DRIVER
    int count = 1,
	    k=0;
#endif
    int *count_d;
    int *parentMap, *parentMap_h;
    q1=new_device_bitmap(n.getNodes());
    q2=new_device_bitmap(n.getNodes());
    get_threads_blocks(n.getNodes(),&b,&t,1024);
    set_bitmap_from_host(q1, source, 1);
    cuMalloc(&parentMap,n.getNodes()*(sizeof(int)));
    cuMalloc(&count_d,4);
    set_value_d(parentMap,-1,n.getNodes());
    cuCopy(parentMap+source,&source,4,h2d);
    setChildLimit(256000);
    printf("blocks=%d,threads=%d\n",b,t);
    devSync();
    //use event timer instead
    //tic();
    start_timer();
#if CONFIG_GPU_DRIVER
    bfs_driver<<<1, 1>>>(t, b, count_d, q1, q2,
                         n.getGraph(),
                         n.getNodes(),
                         parentMap);
    devSync();
#else 
    while(count!=0) {
      kernel1<<<b,t>>>(n.getGraph(),
		       n.getNodes(),
		       q1,
		       q2,
               parentMap,
               count_d        
		       );
      devSync();
      cuCopy(&count,count_d,4,d2h);
      zero(q1,bitmapSize(n.getNodes()));
      zero(count_d,1);

      //debug:
      //pm<<<1,1>>>(q2,n.getNodes());
      //devSync();

      //printf("k=%d, count=%d\n",k, count);
      
      k++;
      std::swap(q1, q2);
    }
#endif
    stop_timer();
    elapsed_time=get_elapsed_time();
    elapsed_time/=1000;
    printf("cuda event timer: %f s, or %f ms\n",elapsed_time,elapsed_time*1000);
    parentMap_h=(int*)malloc(4*n.getNodes());
    int64_t *pm=(int64_t*)malloc(8*n.getNodes());
    devSync();
    cuCopy(parentMap_h,parentMap,sizeof(int)*n.getNodes(),d2h);
    devSync();
    for(uint32_t i=0;i<n.getNodes();i++){
        pm[i]=parentMap_h[i];
    }
    //dump parent map:
    //dump_array_to_file(parentMap, n.getNodes(), "pmdump.txt");
    
    cudaFree(parentMap);
    cudaFree(count_d);
    cudaFree(q1);
    cudaFree(q2);
    free(parentMap_h);
    return pm;
}
