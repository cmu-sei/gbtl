
#pragma once

#include <graphblas/graphblas.hpp>

#include "detail/detail.hpp"
#include "util/util.hpp"
#include "scalar_ref.hpp"

#include <concepts>
#include <memory>
#include <limits>

namespace GRB_SPEC_NAMESPACE {

template <typename T,
          std::integral I = std::size_t,
          typename Hint = GRB_SPEC_NAMESPACE::sparse,
          typename Allocator = std::allocator<T>>
class matrix {
public:
  /// Type of scalar elements stored in the matrix.
  using scalar_type = T;

  /// Type used to reference the indices of elements stored in the matrix.
  using index_type = I;

  /// A tuple-like type containing:
  /// 1. A tuple-like type with two `index_type` elements storing the row and column
  ///    index of the element.
  /// 2. An element of type `scalar_type`.
  using value_type = matrix_entry<T, I>;

  using key_type = index<I>;
  using map_type = T;

  /// Allocator type
  using allocator_type = Allocator;

  /// A large unsigned integral type
  using size_type = std::size_t;

  /// A large signed integral type
  using difference_type = std::ptrdiff_t;

  using hint_type = Hint;

  using backend_type = GBTL_NAMESPACE::Matrix<T>;

  using scalar_reference = scalar_ref<T, I, backend_type>;
  using const_scalar_reference = scalar_ref<const T, I, const backend_type>;

  /// Construct an empty matrix of dimension `shape[0]` x `shape[1]`
  matrix(index<I> shape) : backend_(shape[0], shape[1]) {}

  matrix(std::initializer_list<I> shape)
    : backend_(*shape.begin(), *(shape.begin()+1)) {}

  index<I> shape() const {
    return index<I>(backend_.nrows(), backend_.ncols());
  }

  size_type size() const {
    return backend_.nvals();
  }

  bool empty() const {
    return size() == 0;
  }

  std::size_t max_size() const {
    return std::numeric_limits<I>::max() - 1;
  }

  template <class M>
  void insert_or_assign(key_type k, M&& obj) {
    backend_.setElement(k[0], k[1], std::forward<M>(obj));
  }

  void insert(const value_type& value) {
    auto&& [k, v] = value;

    if (!backend_.hasElement(k[0], k[1])) {
      backend_.setElement(k[0], k[1], v);
    }
  }

  void insert(value_type&& value) {
    auto&& [k, v] = value;

    if (!backend_.hasElement(k[0], k[1])) {
      backend_.setElement(k[0], k[1], std::move(v));
    }
  }

  template <typename InputIt>
  void insert(InputIt first, InputIt last) {
    for (auto it = first; it != last; ++it) {
      auto&& [k, v] = *it;
      insert_or_assign(k, v);
    }
  }

  scalar_reference operator[](key_type index) {
    if (!backend_.hasElement(index[0], index[1])) {
      backend_.setElement(index[0], index[1], T());
    }
    return scalar_reference(index, &backend_);
  }

  std::size_t erase(key_type index) {
    std::size_t num_deleted = backend_.hasElement(index[0], index[1]);

    if (num_deleted) {
      backend_.removeElement(index[0], index[1]);
    }
    return num_deleted;
  }

  void clear() {
    backend_.clear();
  }

  // Missing

  // at()

  // Iteration
  // begin()
  // end()
  // cbegin()
  // cend()

  // find()

  // insert variants need to return iterators

  matrix() = default;
  matrix(const Allocator& allocator) {}

  matrix(const matrix&) = default;
  matrix(matrix&&) = default;
  matrix& operator=(const matrix&) = default;
  matrix& operator=(matrix&&) = default;

private:
  backend_type backend_;
};

};