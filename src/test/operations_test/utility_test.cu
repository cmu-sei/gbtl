#include <graphblas/graphblas.hpp>
#include <graphblas/algebra.hpp>
#include <graphblas/utility.hpp>

int main(){
    thrust::device_vector<int> a(32,1);
    thrust::device_vector<int> b(12,2);
    graphblas::filter(a, 32, b, 12);

    auto r=graphblas::backend::detail::make_col_index_iterator(4,4);
    auto c=graphblas::backend::detail::make_row_index_iterator(4,4);

    thrust::device_vector<int> e(12,1);
    thrust::device_vector<int> d(12,2);

    auto v=graphblas::backend::detail::make_val_iterator(e.begin(), d.begin(), 4,4, 16, 4, graphblas::ArithmeticSemiring<int>());

    cusp::coo_matrix_view <decltype(r), decltype(c), decltype(v)>
        vi(4,4,16,r,c,v);
};
