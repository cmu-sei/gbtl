#pragma once

#include <graphblas/header.hpp>
#include <graphblas/config.hpp>
//#include <moderngpu/include/moderngpu.cuh>
#include <cub/cub.cuh>
#include <cstdint>

namespace graphblas{
namespace backend{
namespace utility{

template <typename ScalarT, typename IndexType>
__global__ void sequence(ScalarT *output, IndexType items, ScalarT init=0)
{
    IndexType tid = threadIdx.x+blockDim.x*blockIdx.x;
    if (tid < items){
        output[tid] = tid + init;
    }
}

template <typename IntegerType1, typename IntegerType2>
__host__ __device__ void get_threads_blocks
                    (IntegerType1 problem_size,
                     IntegerType2 * block_size,
                     IntegerType2 * thread_count,
                     uint32_t dev_max_tpb)
{
    if((int64_t)problem_size>(int64_t)dev_max_tpb)
    {
        *block_size = (IntegerType2)ceil(problem_size/(double)dev_max_tpb);
        *thread_count = (IntegerType2) dev_max_tpb;
    }

    else{
        *block_size = (IntegerType2)1;

        if(problem_size<=32){
            *thread_count = 32;
        }

        else{
            *thread_count=problem_size;
        }
    }
}

//input cannot overlap with output
template <typename IndexType1, typename IndexType2, typename InputType, typename OutputType>
__global__ void gather_kernel(IndexType1 *map, InputType *input, OutputType* output, IndexType2 items, int folds)
{
    IndexType2 tid = threadIdx.x+blockDim.x*blockIdx.x;


    //total num of threads:
    IndexType2 t_count = gridDim.x*blockDim.x;

#pragma unroll
    for (int fold=0; fold<folds; fold++){
        IndexType2 next_location = tid + fold * t_count;
        if (next_location < items)
            output[map[next_location]] = static_cast<OutputType>(input[next_location]);
    }
}

//input cannot overlap with output
//gathers into 2 args
template <typename IndexType1,
          typename IndexType2,
          typename InputType1,
          typename InputType2,
          typename OutputType1,
          typename OutputType2>
__global__ void gather_kernel(
        IndexType1 *map,
        InputType1 *input1,
        InputType2 *input2,
        OutputType1 *output1,
        OutputType2 *output2,
        IndexType2 items)
{
    //TODO: multiple process per thread?
    //extern __shared__ InputType temp[];

    IndexType2 tid = threadIdx.x+blockDim.x*blockIdx.x;

    output1[map[tid]] = static_cast<OutputType1>(input1[tid]);
    output2[map[tid]] = static_cast<OutputType2>(input2[tid]);
}

template <typename IndexType1, typename IndexType2, typename InputType, typename OutputType>
inline void gather(IndexType1 *map, InputType *input, OutputType *output, IndexType2 items)
{
    //calls gather kernel
    int folds=1;
    while (items > (65535*1024*folds)){
        ++folds;
    }
    int threads=(items/folds)+1;
    int t,b;
    graphblas::backend::utility::get_threads_blocks(
            threads,
            &b,
            &t,
            1024);

    graphblas::backend::utility::gather_kernel<<<b,t>>>(
            map, input, output, items, folds);
}

template <typename InputType, typename OutputType, typename UnaryFunc, typename IndexType>
__global__ void transform(InputType *input, OutputType *output,IndexType items , UnaryFunc ufunc=UnaryFunc())
{
    IndexType tid = threadIdx.x+blockDim.x*blockIdx.x;
    if (tid<items)
        output[tid] = ufunc(input[tid]);
}



template <typename IndexType, typename ScalarT>
__host__ inline void dtod_matrix_member_async_memcpy(
        IndexType * dst1,
        IndexType * dst2,
        ScalarT *dst3,
        IndexType * src1,
        IndexType * src2,
        ScalarT *src3,
        IndexType size)
{
    cudaMemcpyAsync(dst1,
            src1,
            sizeof(IndexType)*size,
            DtoD,
            graphblas::backend::streams[0]);

    cudaMemcpyAsync(dst2,
            src2,
            sizeof(IndexType)*size,
            DtoD,
            graphblas::backend::streams[1]);

    cudaMemcpyAsync(dst3,
            src3,
            sizeof(ScalarT)*size,
            DtoD,
            graphblas::backend::streams[2]);

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    cudaStreamSynchronize(streams[2]);
}

template <typename IndexType, typename ScalarT>
__host__ inline void htod_matrix_member_async_memcpy(
        IndexType * dst1,
        IndexType * dst2,
        ScalarT *dst3,
        IndexType * src1,
        IndexType * src2,
        ScalarT *src3,
        IndexType size)
{
    cudaMemcpyAsync(dst1,
            src1,
            sizeof(IndexType)*size,
            HtoD,
            graphblas::backend::streams[0]);

    cudaMemcpyAsync(dst2,
            src2,
            sizeof(IndexType)*size,
            HtoD,
            graphblas::backend::streams[1]);

    cudaMemcpyAsync(dst3,
            src3,
            sizeof(ScalarT)*size,
            HtoD,
            graphblas::backend::streams[2]);

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    cudaStreamSynchronize(streams[2]);
}

template <typename IndexType, typename ScalarT>
inline void dtoh_matrix_member_async_memcpy(
        IndexType * dst1,
        IndexType * dst2,
        ScalarT *dst3,
        IndexType * src1,
        IndexType * src2,
        ScalarT *src3,
        IndexType size)
{
    cudaMemcpyAsync(dst1,
            src1,
            sizeof(IndexType)*size,
            DtoH,
            graphblas::backend::streams[0]);

    cudaMemcpyAsync(dst2,
            src2,
            sizeof(IndexType)*size,
            DtoH,
            graphblas::backend::streams[1]);

    cudaMemcpyAsync(dst3,
            src3,
            sizeof(ScalarT)*size,
            DtoH,
            graphblas::backend::streams[2]);

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    cudaStreamSynchronize(streams[2]);
}

struct Timer{
    cudaEvent_t start_event, stop_event;

    void start(){
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        cudaEventRecord(start_event);
    }

    void stop(){
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
    }

    float get(){
        float ms=0;
        cudaEventElapsedTime(&ms, start_event, stop_event);
        return ms;
    }
    void clear(){
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    //reset timer, different from clear
    //keeps timer running.
    void reset(){
        clear();
        start();
    }
};


} //end utility
} //end backend
} //end graphblas
