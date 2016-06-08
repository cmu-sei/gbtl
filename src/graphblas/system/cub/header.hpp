#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <graphblas/detail/config.hpp>

#define HtoD cudaMemcpyHostToDevice
#define DtoH cudaMemcpyDeviceToHost
#define DtoD cudaMemcpyDeviceToDevice
