#include <graphblas/graphblas.hpp>
#include <cusp/print.h>

int main(){
    using namespace graphblas;
    Matrix<int> stuff(4,4);
    //test on iterators:
    std::vector<IndexType> i = { 1, 2 , 2, 3, 3};
    std::vector<IndexType> j = { 0, 1 , 2, 1, 3};
    std::vector<int> v = { 4213, 234 , 242, 1123, 3342};
    graphblas::IndexType count = 5;

    graphblas::buildmatrix(stuff, i.begin(), j.begin(), v.begin(), count);

    //transpose(stuff);

    //negateview:
    graphblas::negate(stuff);
};
