#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "header.hpp"

namespace graphblas
{
namespace backend
{

//memcpy streams:
cudaStream_t streams[3];

inline void init_system()
{
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);
    cudaStreamCreate(&streams[2]);
}

inline void close_system()
{
    cudaDeviceSynchronize();
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    cudaStreamDestroy(streams[2]);
}


}
}
