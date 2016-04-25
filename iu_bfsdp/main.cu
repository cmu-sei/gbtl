/*
 * main.cu
 *
 *  Created on: Feb 16, 2015
 *      Author: yz79
 */

#include <cuda.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>
#include <stdio.h>
#include <omp.h>

#include "generator/make_graph.h"
#include "graphDriver.cuh"
#include "bfsdp.cuh"
//#include "bfsdp_queue.cuh"
//#include "depthctrl.cuh"
//#include "collapseBitmask.cuh"
//#include "collapseBitmask.cuh"

extern "C"{
int64_t verify_bfs_tree (int64_t *bfs_tree, int64_t max_bfsvtx,
      int64_t root,
      const struct packed_edge *IJ, int64_t nedge);
}

__global__ void printgraph(uusi::GPUNode *g, uint32_t size){
    for(int idx=0;idx<size;idx++){
        printf("node #%d has %d neighbors\n",idx, g[idx].neighbors);
        for(int i=0;i<g[idx].neighbors;i++){
            printf("%d ",g[idx].ev[i].dst);
        }
        printf("\n");
        __syncthreads();
    }
}

int main(int argc, char ** argv){
  if(argc<5){
    printf("wrong number of parameters. need 4, you have %d\n", argc-1);
    return -1;
  }
	  cudaSetDevice(atoi(argv[4]));
    packed_edge * IJ;
    uusi::GPUGraph g=uusi::generateGPUGraph(argc,argv,&IJ);
    std::cout<<"nodes:"<<g.getNodes()<<std::endl;
    std::cout<<"edges:"<<g.getEdges()<<std::endl;
    //printgraph<<<1,1>>>(g.getGraph(),g.getNodes());
    //devSync();
#if MODE<5
    int64_t *parentMap=run_bfs(g,atoi(argv[3]));
#elif MODE==5
    int64_t *parentMap=run_bfs_multigpu(g);
#endif
#if 0
    printf("reference:\n");
    for(int i=0;i<g.getNodes();i++){
        printf("%d ",parentMap[i]);
    }
    printf("\n");
#endif
#if 0
    printf("contents of IJ:\n");
    for(uint64_t i=0;i<g.getEdges()/2;i++){
        printf("%d, %d\n",IJ[i].v0,IJ[i].v1);
    }
#endif
    int64_t
        edges_traversed=verify_bfs_tree(parentMap,g.getNodes()-1,atoi(argv[3]),IJ,g.getEdges()/2);
    printf("traversed edges: %ld\n",edges_traversed);
    printf("TEPS=%f\n",edges_traversed/uusi::elapsed_time);
    printf("GTEPS=%f\n",edges_traversed/uusi::elapsed_time/1000000000);
    //end ref

    //data gather:
    std::cerr<<(atoi(argv[1]))<<","<<(atoi(argv[2]))<<","<<atoi(argv[3])<<","<<uusi::elapsed_time*1000<<",";
    fprintf(stderr,"%f\n",edges_traversed/uusi::elapsed_time);
    free(parentMap);


    return 0;
}



