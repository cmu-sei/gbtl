#include <graphblas/graphblas.hpp>
#include <graphblas/system/cusp/Vector.hpp>


int main(){
    graphblas::backend::Vector<int> v(32);
    graphblas::backend::Matrix<int> m(32,32,32);
    graphblas::backend::vxm(v,m,v, graphblas::ArithmeticSemiring<int>(), graphblas::math::Assign<int>());
    graphblas::backend::mxv(m,v,v, graphblas::ArithmeticSemiring<int>(), graphblas::math::Assign<int>());
}
