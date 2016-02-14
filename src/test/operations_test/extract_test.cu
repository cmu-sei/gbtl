#define GB_USE_CUSP

#include <cstdlib>
#include <iostream>

#include <graphblas/graphblas.hpp>
#include <cusp/print.h>
#include <vector>

int main() {
    using namespace graphblas;
    Matrix<int> a(4,4);
    //test on iterators:
    std::vector<IndexType> i = { 1, 2 , 2, 3, 3};
    std::vector<IndexType> j = { 0, 1 , 2, 1, 3};
    std::vector<int> v = { 4213, 234 , 242, 1123, 3342};
    graphblas::IndexType count = 5;
    buildmatrix(a, i.begin(), j.begin(), v.begin(), count );
    //output:
    Matrix<int> c(6,6);
    //4x4
    std::vector<IndexType> ia = {4, 3, 2, 0};
    std::vector<IndexType> ja = {1, 4, 3, 5};
    //call assign (matrix version)
    graphblas::assign(a,ia.begin(), ja.begin(), c);
    //using cusp func:
//::cusp::print(c);
    //new matrix b:
    Matrix <int> b(3,3);
    std::vector<IndexType> ic = {3,5,1};
    std::vector<IndexType> jc = {4,1,5};
    //extract from c to b:
    graphblas::extract(c, ic.begin(), jc.begin(), b);
 //   ::cusp::print(b);

    return 0;
}
