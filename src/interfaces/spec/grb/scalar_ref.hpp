
#pragma once

#include <graphblas/graphblas.hpp>

#include "detail/detail.hpp"
#include "util/util.hpp"

#include <concepts>
#include <memory>

namespace GRB_SPEC_NAMESPACE {

template <typename T, std::integral I, typename Matrix>
class scalar_ref {
public:
  scalar_ref() = delete;
  ~scalar_ref() = default;
  scalar_ref(const scalar_ref &) = default;

  scalar_ref(index<I> index, Matrix* matrix) : index_(index), matrix_(matrix) {}

  operator T() const {
    return matrix_->extractElement(index_[0], index_[1]);
  }

  scalar_ref operator=(const T &value) const
    requires(!std::is_const_v<T>)
  {
    matrix_->setElement(index_[0], index_[1], value);
    return *this;
  }

  scalar_ref operator=(const scalar_ref &other) const {
    T value = other;
    *this = value;
    return *this;
  }

private:
  index<I> index_;
  Matrix* matrix_;
};

} // end GRB_SPEC_NAMESPACE
