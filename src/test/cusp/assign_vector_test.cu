#include <graphblas/cusp/utility.hpp>
#include <graphblas/cusp/ewiseapply.hpp>
#include <graphblas/cusp/operations.hpp>
#include <thrust/functional.h>
#include <cusp/coo_matrix.h>
#include <cstdlib>
#include <iostream>
#include <cuda.h>

int main() {
    using namespace cusp;
    //make matricies
    cusp::array1d<int,cusp::host_memory> A(4);
    cusp::array1d<int,cusp::host_memory> B(2);
    thrust::plus<float> binary_op;

    // initialize matrix entries on host
    A[0] = 10;
    A[1] = 20;
    A[2] = 15;
    A[3] = 40;

    B[0] = 0;
    B[1] = 2;

    cusp::array1d<int,cusp::device_memory> A_d(A);
    cusp::array1d<int,cusp::device_memory> B_d(B);
    cusp::array1d<int,cusp::device_memory> C_d(4);


    //add 3
    graphblas::cusp::assign(A_d, B_d, C_d, 3, binary_op);

    //write result to host mem
    cusp::array1d<int,cusp::host_memory> C(C_d);
    /*
    cusp::array1d<int,cusp::host_memory> C(4);
    graphblas::cusp::assign(A, B, C, 3, binary_op);
    */

    for (int i=0;i<C.size();i++) {
      std::cout<<C[i]<<std::endl;
    }

}
