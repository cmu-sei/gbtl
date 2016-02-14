#include <graphblas/system/cusp/utility.hpp>
#include <thrust/functional.h>
#include <cusp/coo_matrix.h>
#include <cstdlib>
#include <iostream>
#include <cuda.h>

int main() {
    using namespace cusp;
    //make matricies
    cusp::coo_matrix<int,float,cusp::host_memory> A(4,3,6);
    cusp::coo_matrix<int,float,cusp::host_memory> B(4,3,6);

    // initialize matrix entries on host
    B.row_indices[0] = 0; B.column_indices[0] = 0; B.values[0] = 10;
    B.row_indices[1] = 0; B.column_indices[1] = 2; B.values[1] = 20;
    B.row_indices[2] = 2; B.column_indices[2] = 2; B.values[2] = 30;
    B.row_indices[3] = 3; B.column_indices[3] = 0; B.values[3] = 40;
    B.row_indices[4] = 3; B.column_indices[4] = 1; B.values[4] = 50;
    B.row_indices[5] = 3; B.column_indices[5] = 2; B.values[5] = 60;

    A.row_indices[0] = 0; A.column_indices[0] = 0; A.values[0] = 10;
    A.row_indices[1] = 0; A.column_indices[1] = 2; A.values[1] = 20;
    A.row_indices[2] = 2; A.column_indices[2] = 2; A.values[2] = 30;
    A.row_indices[3] = 3; A.column_indices[3] = 0; A.values[3] = 40;
    A.row_indices[4] = 3; A.column_indices[4] = 1; A.values[4] = 50;
    A.row_indices[5] = 3; A.column_indices[5] = 2; A.values[5] = 60;

    cusp::coo_matrix<int,float,cusp::device_memory> A_d(A);
    cusp::coo_matrix<int,float,cusp::device_memory> B_d(B);
    thrust::plus<float> binary_op;
    thrust::multiplies<float> mult_op;

    graphblas::cusp::detail::merge(A_d,B_d, binary_op);

    graphblas::cusp::detail::merge(A_d,B_d, mult_op);
    cusp::coo_matrix<int,float,cusp::host_memory> C(B_d);

    for (int i=0;i<6;i++) {
      std::cout<<C.row_indices[i]<<" "<<C.column_indices[i]<<" "<<C.values[i]<<std::endl;
    }

}
