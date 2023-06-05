#pragma once

#include "../detail/detail.hpp"

namespace GRB_SPEC_NAMESPACE {

struct sparse {};
struct dense {};
struct row {};
struct column {};

template <typename... Hints>
struct compose {};

} // end GRB_SPEC_NAMESPACE