#pragma once

#include "../detail/detail.hpp"
#include "../util/util.hpp"
#include "../matrix.hpp"

namespace GRB_SPEC_NAMESPACE {

template <MatrixRange A,
          MatrixRange B,
          BinaryOperator<GRB_SPEC_NAMESPACE::matrix_scalar_t<A>, GRB_SPEC_NAMESPACE::matrix_scalar_t<B>> Combine = GRB_SPEC_NAMESPACE::multiplies<>,
          BinaryOperator<GRB_SPEC_NAMESPACE::elementwise_return_type_t<A, B, Combine>,
                         GRB_SPEC_NAMESPACE::elementwise_return_type_t<A, B, Combine>,
                         GRB_SPEC_NAMESPACE::elementwise_return_type_t<A, B, Combine>> Reduce = GRB_SPEC_NAMESPACE::plus<>,
          MaskMatrixRange M = GRB_SPEC_NAMESPACE::full_matrix_mask<>>
auto multiply(A&& a,
              B&& b,
              Reduce&& reduce = Reduce{},
              Combine&& combine = Combine{},
              M&& mask = GRB_SPEC_NAMESPACE::full_matrix_mask())
{
  using T = GRB_SPEC_NAMESPACE::elementwise_return_type_t<A, B, Combine>;
  matrix<T> c(a.shape()[0], b.shape()[1]);
  multiply(c, a, b, reduce, combine, mask);
}

} // end GRB_SPEC_NAMESPACE