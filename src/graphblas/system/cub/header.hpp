#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#define HtoD cudaMemcpyHostToDevice
#define DtoH cudaMemcpyDeviceToHost
#define DtoD cudaMemcpyDeviceToDevice
