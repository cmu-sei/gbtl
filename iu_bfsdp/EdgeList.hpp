/*
 * EdgeList.hpp
 *
 *  Created on: Feb 17, 2015
 *      Author: yz79
 */
#pragma once
#include <vector>
#include <stdint.h>
#include <generator/make_graph.h>
#include <algorithm>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#define MAX(a,b) (a>b?a:b)

#define rmdup 0
extern "C"{
void tic();
double toc();
}


#if rmdup
struct is_loop{
    __host__ __device__ bool operator()(const packed_edge e){
        return (e.v1==e.v0);
    }
};

__host__ __device__ bool operator==(const packed_edge &e1, const packed_edge &e2){
    return ((e1.v1==e2.v1) && (e1.v0==e2.v0));
}

struct pe_less2{
	__host__ __device__ bool operator()(const packed_edge & lhs, const packed_edge & rhs){
		return (lhs.v0<rhs.v0);
	}
};

struct pe_less1{
	__host__ __device__ bool operator()(const packed_edge & lhs, const packed_edge & rhs){
		return (lhs.v1<rhs.v1);
	}
};

#endif

namespace uusi{
double elapsed_time;
class Edge{
private:
    uint64_t s;
    uint64_t e;
public:
    Edge(uint64_t st, uint64_t end){
        this->s=st;
        this->e=end;
    }
    uint64_t getStart(){
        return this->s;
    }
    uint64_t getEnd(){
        return this->e;
    }
};
typedef std::vector<Edge> EdgeList;

EdgeList graph500ToEdgeList(const packed_edge* graph, int64_t edges){
    uusi::EdgeList el;
    for(int64_t i=0;i<edges;i++){
        el.push_back(Edge(graph[i].v0,graph[i].v1));
    }
    return el;
}
EdgeList generateEdgeList(int argc, char** argv, packed_edge** output){
    int log_numverts;
    int edge_factor = 16;
    int64_t nedges;
    packed_edge* result;
    log_numverts = 16; /* In base 2 */
    if (argc >= 2) log_numverts = atoi(argv[1]);
    if (argc >= 3) edge_factor = atoi(argv[2]);
    make_graph(log_numverts, edge_factor << log_numverts, 1, 2,
	       &nedges, &result);
    printf("nedges=%ld before\n",nedges);
    //remove loops:
#if rmdup
    tic();
    packed_edge* end=thrust::remove_if(result, result+nedges, is_loop());
    //sort packed_edge twice:
    thrust::sort(result,end,pe_less1());
    thrust::sort(result,end,pe_less2());
    //nedges=(end-result);
    //printf("nedges=%d after loop\n",nedges);
    //remove duplicates:
    end=thrust::unique(result,end);
    cudaDeviceSynchronize();
    elapsed_time+=toc();
    printf("remove dup took %f ms\n", elapsed_time);
    //TEMP: reset elapsed time:
    elapsed_time=0;
    nedges=(end-result);
    printf("nedges=%ld after dup\n",nedges);
    //
#endif
    uusi::EdgeList el=graph500ToEdgeList((const packed_edge*)result,nedges);
    *output=result;
    return el;
}
uint64_t getVertexCount(EdgeList el){
    uint64_t max=0;
    for(Edge e:el){
        max=MAX(max,(MAX(e.getEnd(),e.getStart())));
    }
    return max+1;
}

void toDimacs(int argc, char ** argv, char* filename){
	packed_edge* out;
	EdgeList e=generateEdgeList(argc, argv,&out);
	FILE *fp=fopen(filename,"w+");
	fprintf(fp,"p sp %lu %lu\n",getVertexCount(e),(e.size()*2));
	//default to 1:
	for(auto edge:e){
		fprintf(fp,"a %lu %lu %d\n",edge.getStart()+1,edge.getEnd()+1,1);
		fprintf(fp,"a %lu %lu %d\n",edge.getEnd()+1,edge.getStart()+1,1);
	}
	fclose(fp);
}
}




