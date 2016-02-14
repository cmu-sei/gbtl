#include <cstdlib>
#include <iostream>
#include <algorithm>

#include <graphblas/graphblas.hpp>
#include <vector>

#include <cusp/print.h>

template <typename T>
struct multiply_operator{
    typedef T result_type;
    T operator() (T &a, T &b)
    {
        return a * b;
    }
};

int main(){
    using namespace graphblas;
    Matrix<int> a(4,4);
    Matrix<int> b(4,4);
    Matrix<int> c(4,4);
    //test on iterators:
    std::vector<IndexType> i = { 1, 2 , 2, 3, 3};
    std::vector<IndexType> j = { 0, 1 , 2, 1, 3};
    std::vector<int> v = { 4213, 234, 242, 1123, 3342};
    std::vector<int> v_b = { 2, 2, 2, 2, 2};
    graphblas::buildmatrix(a, i.begin(), j.begin(), v.begin(), i.size() );
    //ins new val:
    i.push_back(2);
    j.push_back(3);
    v_b.push_back(9);
    graphblas::buildmatrix(b, i.begin(), j.begin(), v_b.begin(), i.size() );
    cusp::print(a.getBackendMatrix());
    cusp::print(b.getBackendMatrix());

    //c<-a*b
    graphblas::ewisemult(a,b,c);
    cusp::print(c.getBackendMatrix());
}
