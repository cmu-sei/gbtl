#pragma once

#include "../detail/detail.hpp"
#include "../util/util.hpp"
#include "../functional/functional.hpp"
#include "../matrix.hpp"
#include "../views/views.hpp"

namespace GRB_SPEC_NAMESPACE {

// NOTE: concepts are missing because `GRB_SPEC_NAMESPACE::matrix` does not
//       satisfy iteration yet.
template <typename A,
          typename B,
          typename Combine,
          typename C,
          typename M = GRB_SPEC_NAMESPACE::full_matrix_mask<>,
          typename Accumulate = GRB_SPEC_NAMESPACE::take_right<>
          >
void ewise_intersection(C&& c, A&& a, B&& b,
                        Combine&& combine,
                        M&& mask = M{},
                        Accumulate&& acc = Accumulate{},
                        bool merge = false)
{
    auto merge_enum = ((merge) ?
                       GBTL_NAMESPACE::OutputControlEnum::MERGE :
                       GBTL_NAMESPACE::OutputControlEnum::REPLACE);
    if constexpr(std::is_same_v<M, GRB_SPEC_NAMESPACE::full_matrix_mask<>>)
    {
        GBTL_NAMESPACE::eWiseMult(c.backend_,
                                  GBTL_NAMESPACE::NoMask(), acc,
                                  combine, a.backend_, b.backend_, merge_enum);
    } else
    {
        GBTL_NAMESPACE::eWiseMult(c.backend_,
                                  mask.backend_, acc,
                                  combine, a.backend_, b.backend_, merge_enum);
  }
}

} // GRB_SPEC_NAMESPACE
