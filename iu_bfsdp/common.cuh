/**
  * last modified Feb 16 2015
  */
#pragma once
#ifndef _COMMON_CUH_
#define _COMMON_CUH_
#include <sys/time.h>
#include <stdint.h>
#include <math.h>

#ifndef __cuda_cuda_h__
#include <cuda.h>
#endif

#ifndef  __CUDA_RUNTIME_H__
#include <cuda_runtime.h>
#endif
#include <stdlib.h>
#include <stdio.h>

#include "config.h"

//#include "graph.cuh"
//#include "dimacs.cuh"

//This is the bitmapped version

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

//error checking cuda apis:
#define devSync() (gpuErrchk(cudaDeviceSynchronize()))
#define cuMalloc(a,b) (gpuErrchk(cudaMalloc(a,b)))
#define cuCopy(a,b,c,d) (gpuErrchk(cudaMemcpy(a,b,c,d)))
#define cuMemset(a,b,c) (gpuErrchk(cudaMemset(a,b,c)))
#define cuFree(a) (gpuErrchk(cudaFree(a)))

//shorthand:
#define blkSync() (__syncthreads())
#define h2d cudaMemcpyHostToDevice
#define d2h cudaMemcpyDeviceToHost
#define d2d cudaMemcpyDeviceToDevice

#if CONFIG_USE_FAT_BITMAP
#include "bitmap-wordsize.cuh"
#else
#include "bitmap-compact.cuh"
#endif
#include "bitmap-common.cuh"

namespace gafa{
    //struct defs:

    typedef struct
    {
        uint32_t starting;
        uint32_t no_of_edges;
    }Node;

    //Namespace variables:
    uint32_t max_tpb;
    uint32_t num_of_nodes;
    uint32_t num_of_edges;
    uint32_t source;

    Node * d_nodes;
    uint32_t * d_edges;
    //BitMap *d_visited;


    cudaEvent_t start_event, stop_event;

    __host__ void setChildLimit(int limit){
        gpuErrchk(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount,
                                     limit));
    }
    //common functions:
    double get_time(){
        struct timeval t;
        gettimeofday(&t, NULL);
        return t.tv_sec + t.tv_usec*1e-6;
    }

    cudaDeviceProp get_prop(int device_id=0){
        cudaDeviceProp prop;
        gpuErrchk(cudaGetDeviceProperties(&prop, device_id));
        return prop;
    }

    void set_max_tpb(int device_id=0){
        cudaDeviceProp p=get_prop(device_id);
        gafa::max_tpb=p.maxThreadsPerBlock;
    }
    void start_timer(){
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        cudaEventRecord(start_event);
    }

    void stop_timer(){
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
    }

    float get_elapsed_time(){
        float ms=0;
        cudaEventElapsedTime(&ms, start_event, stop_event);
        return ms;
    }

    template <typename T, typename IntType>
    __host__ void zero(T *d_ds, IntType size){
        cudaMemset(d_ds,0,size*sizeof(T));
    }

    template <typename T, typename IntType>
    __global__ void set_value(T *d_ds ,T value, IntType size){
        uint64_t idx=threadIdx.x+blockDim.x*blockIdx.x;
        if(idx<size){
            d_ds[idx]=value;
        }
    }

    template <typename T, typename IntType>
    __global__ void set_value2(T *d_ds1, T* d_ds2, T value, IntType size){
        uint64_t idx=threadIdx.x+blockDim.x*blockIdx.x;
        if(idx<size){
            d_ds1[idx]=value;
            d_ds2[idx]=value;
        }
    }

    template <typename T, typename Int>
    __host__ void dump_array_to_file(T * d_arr, Int size, const char * filename){
        T* h_arr=(T*)malloc(sizeof(T)*size);
        gpuErrchk(cudaMemcpy(h_arr,d_arr,sizeof(T)*size,cudaMemcpyDeviceToHost));
        FILE *fp=fopen(filename,"w");
        for(int i=0;i<size;i++){
            fprintf(fp,"%d: %lu\n",i,(uint64_t) h_arr[i]);
        }
        fclose(fp);
        free(h_arr);
    }

    template <typename IntegerType1, typename IntegerType2>
    __host__ __device__ void get_threads_blocks
                        (IntegerType1 problem_size,
                         IntegerType2 * block_size,
                         IntegerType2 * thread_count,
                         uint32_t dev_max_tpb)
    {
        //bad scheduling:
        if((int64_t)problem_size>(int64_t)dev_max_tpb)
        {
            *block_size = (IntegerType2)ceil(problem_size/(double)dev_max_tpb);
            *thread_count = (IntegerType2) dev_max_tpb;
        }
        //better:
#if 0
        if((uint64_t)problem_size>(uint64_t)dev_max_tpb){
            uint64_t thd=problem_size;
            uint64_t blk=1;
            //TODO: consider loop expansion/better bitwise opr
            while(thd>dev_max_tpb){
                thd>>=1;
                blk<<=1;
            }
            *block_size=blk;
            *thread_count = thd;
        }
#endif
        else{
            *block_size = (IntegerType2)1;
            //make sure at least 1/2 warp is launched, for now.
            /*
            if(problem_size<=16){
                *thread_count = 16;
            }
            */
            if(problem_size<=1024){
                *thread_count = 1024;
            }
            else{
                //*thread_count = (ceil(((IntegerType2) problem_size)/32.0f))*32;
                *thread_count=problem_size;
            }
        }
    }
    template <typename IntegerType1, typename IntegerType2>
    __host__ __device__ void get_small_threads_blocks
                        (IntegerType1 problem_size,
                         IntegerType2 * block_size,
                         IntegerType2 * thread_count,
                         uint32_t dev_max_tpb)
    {
        if((int64_t)problem_size>(int64_t)dev_max_tpb)
        {
            *block_size = 1;
            *thread_count=problem_size;
            while(*thread_count>dev_max_tpb){
              (*thread_count)++;
              *thread_count /= 2;
              *block_size*=2;
            }
        }
        else{
          *block_size = (IntegerType2)1;
          *thread_count=problem_size;
        }
    }
    //sets value directly on host
    template <typename T, typename IntType>
    __host__ void set_value_d(T *d_ds ,const T value, IntType size){
    	int blks, thrds;
    	gafa::get_threads_blocks(size,&blks,&thrds,1024);
    	set_value<<<blks,thrds>>>(d_ds,value,size);
    }
}

#endif
