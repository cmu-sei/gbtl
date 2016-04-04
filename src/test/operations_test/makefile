CC=nvcc -O3
CUDA_FLAGS=-Xcompiler -Wall -arch sm_35 -Xcompiler -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -DGB_USE_CUSP_GPU --expt-relaxed-constexpr
BOOST_FLAGS=#-lboost_unit_test_framework -L/usr/local/lib

FLAGS= -std=c++11  $(BOOST_FLAGS) $(CUDA_FLAGS)

INC=-I../.. -I../../../cusplibrary -I/usr/local/include

OBJS=extract buildmatrix assign extracttuples transpose apply ewise mxm bfs sssp vector transposeview utility
all: $(OBJS)

utility: utility_test.cu
	$(CC) $(FLAGS) $(INC) utility_test.cu -o utility

transposeview: transposeview_test.cu
	$(CC) $(FLAGS) $(INC) transposeview_test.cu -o transposeview

vector: vector_test.cu
	$(CC) $(FLAGS) $(INC) vector_test.cu -o vector

bfs: bfs_test.cu
	$(CC) $(FLAGS) $(INC) bfs_test.cu -o bfs

sssp: sssp_test.cu
	$(CC) $(FLAGS) $(INC) sssp_test.cu -o sssp #-g -G

extracttuples: extracttuples_test.cu
	$(CC) $(FLAGS) $(INC) extracttuples_test.cu -o extracttuples
buildmatrix: buildmatrix_test.cu
	$(CC) $(FLAGS) $(INC) buildmatrix_test.cu -o buildmatrix
transpose: transpose_test.cu
	$(CC) $(FLAGS) $(INC) transpose_test.cu -o transpose
assign: assign_test.cu
	$(CC) $(FLAGS) $(INC) assign_test.cu -o assign

apply: apply_test.cu
	$(CC) $(FLAGS) $(INC) apply_test.cu -o apply

ewise: ewise_test.cu
	$(CC) $(FLAGS) $(INC) ewise_test.cu -o ewise

extract: extract_test.cu
	$(CC) $(FLAGS) $(INC) extract_test.cu -o extract

mxm: mxm_test.cu
	$(CC) $(FLAGS) $(INC) mxm_test.cu -o mxm

clean:
	rm -rf *.a *.out *.o $(OBJS)
