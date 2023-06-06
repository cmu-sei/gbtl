#pragma once

#include <limits>
#include <type_traits>
#include <concepts>

#include "../detail/detail.hpp"
#include "../util/util.hpp"

namespace GRB_SPEC_NAMESPACE {

template <typename T, std::integral I>
class full_matrix_accessor {
public:
  using value_type = GRB_SPEC_NAMESPACE::matrix_entry<T, I>;
  using difference_type = std::ptrdiff_t;
  using iterator_accessor = full_matrix_accessor;
  using const_iterator_accessor = full_matrix_accessor;
  using nonconst_iterator_accessor = full_matrix_accessor;
  using reference = GRB_SPEC_NAMESPACE::matrix_entry<T, I>;
  using iterator_category = std::random_access_iterator_tag;

  constexpr full_matrix_accessor() noexcept = default;
  constexpr ~full_matrix_accessor() noexcept = default;
  constexpr full_matrix_accessor(const full_matrix_accessor&) noexcept = default;
  constexpr full_matrix_accessor& operator=(const full_matrix_accessor&) noexcept = default;

  constexpr full_matrix_accessor(I i, I j, I n, T value) noexcept : i_(i), j_(j), n_(n), value_(value) {}

  constexpr full_matrix_accessor& operator+=(difference_type offset) noexcept {
    auto new_offset = flat_offset() + offset;
    i_ = new_offset / n_;
    j_ = new_offset % n_;
    return *this;
  }

  constexpr difference_type operator-(const const_iterator_accessor& other) const noexcept {
    return difference_type(flat_offset()) - difference_type(other.flat_offset());
  }

  constexpr bool operator==(const const_iterator_accessor& other) const noexcept {
    return i_ == other.i_ && j_ == other.j_;
  }

  constexpr bool operator<(const const_iterator_accessor& other) const noexcept {
    return i_ < other.i_ || (i_ == other.i_ && j_ < other.j_);
  }

  constexpr reference operator*() const noexcept {
    return GRB_SPEC_NAMESPACE::matrix_entry<T, I>({i_, j_}, value_);
  }

private:

  std::size_t flat_offset() const noexcept {
    return std::size_t(i_)*n_ + j_;
  }

  I i_, j_;
  I n_;
  T value_;
};

template <typename T, std::integral I = std::size_t>
using full_matrix_iterator = GRB_SPEC_NAMESPACE::detail::iterator_adaptor<full_matrix_accessor<T, I>>;

template <typename T, typename I = std::size_t>
class full_matrix
{
public: 
  using scalar_type = T;

  using index_type = I;

  using value_type = GRB_SPEC_NAMESPACE::matrix_entry<T, I>;

  using key_type = GRB_SPEC_NAMESPACE::index<I>;
  using map_type = T;

  using size_type = std::size_t;

  using difference_type = std::ptrdiff_t;

  using scalar_reference = scalar_type;

  using iterator = full_matrix_iterator<T, I>;
  using const_iterator = iterator;


  full_matrix(GRB_SPEC_NAMESPACE::index<I> shape = {std::numeric_limits<I>::max(), std::numeric_limits<I>::max()},
              T value = T())
    : shape_(shape), value_(value)
  {}

  iterator begin() const noexcept {
    return iterator(0, 0, shape()[1], value_);
  }

  iterator end() const noexcept {
    return iterator(shape()[0], 0, shape()[1], value_);
  }

  GRB_SPEC_NAMESPACE::index<I> shape() const noexcept {
    return shape_;
  }

  size_type size() const noexcept {
    return shape()[0]*shape()[1];
  }

  scalar_reference operator[](GRB_SPEC_NAMESPACE::index<I> index) const noexcept {
    return value_;
  }

  iterator find(key_type key) const noexcept {
    if constexpr(std::is_signed_v<I>) {
      if (key[0] < 0 || key[1] < 0) {
        return end();
      }
    }

    if (key[0] < shape()[0] && key[1] < shape()[1]) {
      return iterator(key[0], key[1], shape()[1], value_);
    } else {
      return end();
    }
  }

private:
  T value_;
  GRB_SPEC_NAMESPACE::index<I> shape_;
};

template <typename I = std::size_t>
class full_matrix_mask : public full_matrix<bool, I>
{
public:

  full_matrix_mask(GRB_SPEC_NAMESPACE::index<I> shape = {std::numeric_limits<I>::max(), std::numeric_limits<I>::max()})
    : full_matrix<bool, I>(shape, true)
  {}
};


template <typename I = std::size_t>
class empty_matrix_mask : public full_matrix<bool, I>
{
public:

  empty_matrix_mask(GRB_SPEC_NAMESPACE::index<I> shape = {std::numeric_limits<I>::max(), std::numeric_limits<I>::max()})
    : full_matrix<bool, I>(shape, false)
  {}
};

} // end GRB_SPEC_NAMESPACE
