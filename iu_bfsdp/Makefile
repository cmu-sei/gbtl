CXX = nvcc
CXXFLAGS = -std=c++11 -Xcompiler "-fopenmp -std=c++11 -Wall -Drestrict=__restrict__ -DNDEBUG -ffast-math -DGRAPH_GENERATOR_OMP -Wall" -lineinfo # -g -pg
LDFLAGS= -lcudadevrt 
LD=#.
INC= -I.
CFLAGS= -O3 -rdc=true -gencode arch=compute_35,code=sm_35 #-g -G
GENERATOR_SOURCES = generator/graph_generator.c generator/make_graph.c generator/splittable_mrg.c generator/utils.c xalloc.c

main: clean main.cu $(GENERATOR_SOURCES) bfsdp.cuh EdgeList.hpp graphDriver.cuh config.h config-user.h bitmap-common.cuh
	$(CXX) $(LDFLAGS) $(LD) $(INC) $(CXXFLAGS) $(CFLAGS) main.cu verify.c timer.c $(GENERATOR_SOURCES) -lm -o main

verify:
	gcc -c verify.c -o verify.o
clean:
	rm -f main *.o toDimacs *.out

TAGS: $(GENERATOR_SOURCES) main.cu  bfsdp.cuh EdgeList.hpp graphDriver.cuh
	etags -l c++ $(GENERATOR_SOURCES) main.cu  bfsdp.cuh EdgeList.hpp graphDriver.cuh

config-user.h:
	touch config-user.h

