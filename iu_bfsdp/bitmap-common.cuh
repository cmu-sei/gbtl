#pragma once

namespace gafa {
	__device__ void zero_queue_dyn(BitMap *map, int size)
	{
		int threads = 1024;
		int blocks = (bitmapSize(size) + threads - 1) / threads;
		zero_queue<<<blocks, threads>>>(map, size);
		//cudaMemsetAsync(map, 0, bitmapSize(size));
	}
}
