/*
 * graphDriver.cuh
 *
 *  Created on: Feb 16, 2015
 *      Author: yz79
 */

#ifndef GRAPHDRIVER_CUH_
#define GRAPHDRIVER_CUH_

#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <omp.h>
#include "EdgeList.hpp"
#include "common.cuh"

#define OMP 1

//call EdgeListGenerator
namespace uusi{


typedef union {
    uint32_t dst; //destination
    uint32_t wt;  //weight;
}EdgeValue;

  bool operator<(const EdgeValue &lhs, const EdgeValue &rhs) {
    return lhs.dst < rhs.dst;
  }
  
struct GPUNode{
    uint32_t neighbors;
    EdgeValue * ev;
};

//code template from:
//http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/
__host__ cudaTextureObject_t initTextureMem(void *buffer, size_t channelSize, uint32_t count){
    // create texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = buffer;
    resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
    resDesc.res.linear.desc.x = channelSize; // bits per channel
    resDesc.res.linear.sizeInBytes = count*channelSize;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t tex=0;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
    return tex;
}

//temporary ds for cpu->gpu mem
/*
template <typename uint32_t>
using CPUNode = std::vector<EdgeValue<uint32_t> >;
*/

typedef std::vector<EdgeValue> CPUNode;

class GPUGraph{
private:
    uusi::GPUNode * d_nodes;
    //internal datastructure to avoid millions of cudaMalloc
    uusi::EdgeValue * d_edges;
    uint32_t edges;
    uint32_t nodes;
    void populateNodes(EdgeList el){
        if(this->edges!=el.size()){
            this->edges=el.size()*2;
        }
        std::vector<CPUNode> cnodes(nodes);
        EdgeValue * dedges=(EdgeValue*)malloc(sizeof(EdgeValue)*this->edges);
        GPUNode * dnodes=(GPUNode*)malloc(sizeof(GPUNode)*(nodes));
        uint64_t counter=0;
        cuMalloc(&d_nodes,sizeof(GPUNode)*nodes);
        cuMalloc(&d_edges,sizeof(GPUNode)*edges);

#if OMP
        //omp
        uint32_t i,tid,threads,start,end,chunk;
        omp_set_num_threads(omp_get_max_threads());
        omp_lock_t *lock=(omp_lock_t*)malloc(sizeof(omp_lock_t)*nodes);
        for(uint32_t j=0;j<nodes;j++){
        	omp_init_lock(&lock[j]);
        }
#pragma omp parallel shared(cnodes,lock) private(chunk,start,end,i,tid,threads)
{
        tid = omp_get_thread_num();
        threads=omp_get_num_threads();
        chunk=(uint32_t)floor((float)el.size()/(float)threads);
        start=chunk*tid;
        end=chunk*(tid+1);
        if(tid==(threads-1) && el.size()%threads){
                end+=el.size()%threads;
        }
    	if(end<=el.size()){
#pragma omp parallel for
            for(i=start;i<end;i++){
    		    Edge e=el[i];
                uint32_t src=(uint32_t)e.getStart();
                uint32_t dst=(uint32_t)e.getEnd();
                EdgeValue v1, v2;
                v1.dst=dst;
                //v1.wt=1;
                v2.dst=src;
                //v2.wt=1;
                if(dst>=nodes){
                    printf("error: %d>%d\n",dst,nodes);
                }
                omp_set_lock(&lock[src]);
                cnodes[src].push_back(v1);
                omp_unset_lock(&lock[src]);
                omp_set_lock(&lock[dst]);
                cnodes[dst].push_back(v2);
                omp_unset_lock(&lock[dst]);
            }
    	}
}
        for(uint32_t j=0;j<nodes;j++){
        	omp_destroy_lock(&lock[j]);
        }
        free(lock);
#else
        for(Edge e:el){
               uint32_t src=(uint32_t)e.getStart();
               uint32_t dst=(uint32_t)e.getEnd();
               EdgeValue v1, v2;
               v1.dst=dst;
               //v1.wt=1;
               v2.dst=src;
               //v2.wt=1;
               if(dst>=nodes){
                   printf("error: %d>%d\n",dst,nodes);
               }
               cnodes[src].push_back(v1);
               cnodes[dst].push_back(v2);
           }
#endif

        for(uint64_t i=0;i<nodes;i++){
#if CONFIG_SORT_EDGES
	        std::sort(cnodes[i].begin(), cnodes[i].end());
#endif
	        dnodes[i].neighbors=cnodes[i].size();
	        memcpy(dedges+counter,
	               cnodes[i].data(),
	               sizeof(EdgeValue)*cnodes[i].size());
	        //record offset from start of array
	        dnodes[i].ev=(EdgeValue*)(d_edges+counter);
	        counter+=cnodes[i].size();
        }
        cuCopy(d_nodes,dnodes,sizeof(GPUNode)*nodes,h2d);
        cuCopy(d_edges,dedges,sizeof(EdgeValue)*edges,h2d);
        free(dnodes);
        free(dedges);
    }
public:
    /**
     * There's really no reason NOT to use uint32 for all graph data
     */
    GPUGraph(EdgeList e){
        this->edges=e.size()*2;
        this->nodes=(uint32_t)getVertexCount(e);
        cuMalloc(&this->d_nodes, (sizeof(uusi::GPUNode)*this->nodes));
        populateNodes(e);
    }
    ~GPUGraph(){
        cudaFree(d_nodes);
        cudaFree(d_edges);
    }
    uusi::GPUNode* getGraph(){
        return this->d_nodes;
    }
    uusi::EdgeValue * getEdgeValues(){
    	return this->d_edges;
    }
    uint32_t getNodes(){
        return nodes;
    }
    uint32_t getEdges(){
        return edges;
    }
};

GPUGraph generateGPUGraph(int argc, char** argv, packed_edge** output){
    return GPUGraph(generateEdgeList(argc,argv,output));
}

class Queue{
protected:
	uint32_t * data;
	uint32_t * size;
	uint32_t * maxsize;
	uint32_t maxsize_h;
	uint32_t size_h;
public:
	Queue(uint32_t initsize){
		cuMalloc(&data,initsize*sizeof(uint32_t));
		cuMalloc(&size,sizeof(uint32_t));
		cuMalloc(&maxsize,sizeof(uint32_t));
		this->maxsize_h=initsize;
		setSize_h(0);
		cuCopy(maxsize,&initsize,4,h2d);
		clear();
	}
	__device__ uint32_t operator[](int seq){
		return this->data[seq];
	}
	__host__ void destroy(){
		cuFree(data);
		cuFree(size);
		cuFree(maxsize);
	}
	__host__ void setSize_h(uint32_t newsize){
		cuCopy(this->size,&newsize,sizeof(uint32_t),h2d);
		this->size_h=newsize;
	}
	__host__ void clear(){
		cuMemset(this->data,0,sizeof(uint32_t)*maxsize_h);
		this->size_h=0;
		setSize_h(0);
	}
	__host__ uint32_t getSize_h(){
		cuCopy(&this->size_h,this->size,sizeof(uint32_t),d2h);
		return this->size_h;
	}
	__host__ void set_h(uint32_t loc,uint32_t v){
		cuCopy(this->data+loc,&v,sizeof(uint32_t),h2d);
	}
	//non-atomic setting
	__device__ uint32_t setSize(uint32_t s){
		uint32_t temp=*this->size;
		*this->size=s;
		return temp;
	}
	__host__ __device__ uint32_t* getData(){
		return this->data;
	}
	//atomic setting
	__device__ uint32_t atomicSetSize(uint32_t s){
		return atomicExch(size,s);
	}
	__device__ uint32_t atomicIncrSize(uint32_t s){
		return atomicAdd(size,s);
	}
	//non-atomic getsize
	__device__ uint32_t getSize(){
		return *this->size;
	}
	//non-atomic setting
	__device__ uint32_t set(uint32_t loc, uint32_t data){
		uint32_t temp=this->data[loc];
		this->data[loc]=data;
		return temp;
	}
	__device__ uint32_t atomicSet(uint32_t loc, uint32_t data){
		return atomicExch(this->data+loc,data);
	}
};

	//slightly more "precise" scheduling"
    template <typename IntegerType1, typename IntegerType2>
    __host__ __device__ void setDim
                        (IntegerType1 problem_size,
                         IntegerType2 * block_size,
                         IntegerType2 * thread_count,
                         uint32_t dev_max_tpb)
    {
        if((int64_t)problem_size>(int64_t)dev_max_tpb)
        {
            *block_size = (IntegerType2)ceil(problem_size/(double)dev_max_tpb);
            *thread_count = (IntegerType2)ceil(problem_size/(double)(*block_size));
        }
        else{
            *block_size = (IntegerType2)1;
            //*thread_count = (ceil(((IntegerType2) problem_size)/32.0f))*32;
            *thread_count=problem_size;
        }
    }

}


#endif /* GRAPHDRIVER_CUH_ */
