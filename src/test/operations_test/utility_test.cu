#include <graphblas/graphblas.hpp>
#include <graphblas/utility.hpp>

int main(){
    thrust::device_vector<int> a(32,1);
    thrust::device_vector<int> b(12,2);
    graphblas::filter(a, 32, b, 12);
};
