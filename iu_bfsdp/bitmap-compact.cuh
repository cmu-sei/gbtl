#pragma once

#define bitmapSize(count) ((count + 31) / 32)

namespace gafa {
	//bitmap
	typedef unsigned int BitMap;

	//return new bitmap:
	__host__ BitMap* new_device_bitmap(int size){
		size = bitmapSize(size);
		BitMap *b;
		gpuErrchk(cudaMalloc(&b,sizeof(BitMap)*size));
		gpuErrchk(cudaMemset(b,0,sizeof(BitMap)*size));
		return b;
	}

	__device__ __forceinline__ uint32_t get_bitmap_value_at(BitMap * map,int bit_seq){
		uint32_t major=bit_seq/32;
		uint32_t minor=bit_seq & (31);
		uint32_t bit=0x80000000>>minor;
		bit=map[major] & bit;
		bit=bit>>(31-minor);
		return (map[bit_seq/32] & (1<<(bit_seq&31)))!=0;
	}


	__device__ __forceinline__ void set_bitmap_value_at(BitMap * map,int bit_seq, uint32_t value){
		uint32_t major=bit_seq / 32;
		uint32_t minor=bit_seq & 31;
		value==0?
		    atomicAnd(map+major,~(1<<minor))
		    :
		    atomicOr(map+major,1<<minor);
	}

	__device__ __forceinline__ void set_bitmap_atomic(BitMap * map,
	                                                  int bit_seq,
	                                                  uint32_t value)
	{
		uint32_t major=bit_seq / 32;
		uint32_t minor=bit_seq & 31;
		value==0?
		    atomicAnd(map+major,~(1<<minor))
		    :
		    atomicOr(map+major,1<<minor);
	}
    

	void set_bitmap_from_host(BitMap *map, int bit_seq, uint32_t value) {
		cudaMemset((void*)((uint8_t*)(map)+(bit_seq/8)),
		           (0x01 << (bit_seq % 8)),1);
	}
  
	__host__ void free_device_bitmap(BitMap *b){
		cudaFree(b);
	}	

	__global__ void zero_queue(BitMap *map, int size)
	{
		auto i = threadIdx.x+blockDim.x*blockIdx.x;
		while(i < bitmapSize(size)) {
			map[i] = 0;
			i += blockDim.x * gridDim.x;
		}
	}
}
