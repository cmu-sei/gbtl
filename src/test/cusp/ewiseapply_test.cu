#define GB_USE_CUSP

#include <cstdlib>
#include <iostream>
#include <cuda.h>

#include <cusp/coo_matrix.h>
#include <thrust/functional.h>

#include <graphblas/graphblas.hpp>

int main() {
    //make matricies
    cusp::coo_matrix<int, float, cusp::host_memory> A(4,3,6);
    cusp::coo_matrix<int, float, cusp::host_memory> B(4,3,6);

    // initialize matrix entries on host
    B.row_indices[0] = 0; B.column_indices[0] = 0; B.values[0] = 10;
    B.row_indices[1] = 0; B.column_indices[1] = 2; B.values[1] = 20;
    B.row_indices[2] = 1; B.column_indices[2] = 2; B.values[2] = 15;
    B.row_indices[3] = 3; B.column_indices[3] = 0; B.values[3] = 40;
    B.row_indices[4] = 3; B.column_indices[4] = 1; B.values[4] = 50;
    B.row_indices[5] = 3; B.column_indices[5] = 2; B.values[5] = 60;

    A.row_indices[0] = 0; A.column_indices[0] = 0; A.values[0] = 10;
    A.row_indices[1] = 0; A.column_indices[1] = 2; A.values[1] = 20;
    A.row_indices[2] = 2; A.column_indices[2] = 2; A.values[2] = 30;
    A.row_indices[3] = 3; A.column_indices[3] = 0; A.values[3] = 40;
    A.row_indices[4] = 3; A.column_indices[4] = 1; A.values[4] = 50;
    A.row_indices[5] = 3; A.column_indices[5] = 2; A.values[5] = 60;

    cusp::coo_matrix<int, float, cusp::device_memory> A_d(A);
    cusp::coo_matrix<int, float, cusp::device_memory> B_d(B);

    //result (takes b)
    cusp::coo_matrix<int, float, cusp::device_memory> C_d(B);
    thrust::plus<float> binary_op;
    thrust::multiplies<float> mult_op;

    /*
     * Create CooMatrix wrappers for the above matrices to pass into GraphBLAS
     * function calls.
     */

    std::cout << "HERE!!!\n";
    graphblas::CooMatrix<float, cusp::host_memory> A_wrapper(A);
    exit(0);
    graphblas::CooMatrix<float, cusp::host_memory> B_wrapper(B);
    graphblas::CooMatrix<float, cusp::device_memory> A_d_wrapper(A_d);
    graphblas::CooMatrix<float, cusp::device_memory> B_d_wrapper(B_d);
    graphblas::CooMatrix<float, cusp::device_memory> C_d_wrapper(C_d);

    graphblas::cusp::ewiseapply(A_d_wrapper, B_d_wrapper, C_d_wrapper, binary_op, mult_op);

    // TODO Conversions with wrappers (pass in host to device and vice versa)???
    //write result to host mem
    cusp::coo_matrix<int,float,cusp::host_memory> C(C_d_wrapper.m_matrix);
    graphblas::CooMatrix<float, cusp::host_memory> C_wrapper(C);

    std::cout << C_wrapper << std::endl;
}
