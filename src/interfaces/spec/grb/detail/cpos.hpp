#pragma once

#include <type_traits>
#include <concepts>
#include <utility>
#include <ranges>
#include <any>
#include "tag_invoke.hpp"
#include "matrix_traits.hpp"

namespace GRB_SPEC_NAMESPACE {

// Helper concepts for CPOs.

namespace {

template <typename T>
concept has_matrix_shape = requires(T t) { {t.shape()} -> std::same_as<typename std::tuple_element<0, std::ranges::range_value_t<T>>::type>; };

template <typename T>
concept has_vector_shape = requires(T t) { {t.shape()} -> std::same_as<std::ranges::range_value_t<T>>; };

template <typename T>
concept has_find_method = requires(T t) { {t.find(std::declval<typename container_traits<T>::key_type>())} -> std::same_as<typename container_traits<T>::iterator>; };

template <typename T>
concept has_insert_method = requires(T t) { {t.insert({std::declval<typename container_traits<T>::key_type>(), std::declval<container_scalar_t<T>>()})}; };

template <typename T, typename M>
concept has_insert_or_assign_method = requires(T t, M obj) { {t.insert_or_assign(std::declval<typename container_traits<T>::key_type>(), std::forward<M>(obj))}; };

} // end anonymous

inline constexpr struct shape_fn_ {
  template <typename T>
  auto operator()(T&& x) const
    requires(GRB_SPEC_NAMESPACE::is_tag_invocable_v<shape_fn_, T> ||
             has_matrix_shape<T> ||
             has_vector_shape<T>)
    {
    if constexpr(GRB_SPEC_NAMESPACE::is_tag_invocable_v<shape_fn_, T>) {
      return GRB_SPEC_NAMESPACE::tag_invoke(*this, std::forward<T>(x));
    } else if constexpr(has_matrix_shape<T>) {
      return std::forward<T>(x).shape();
    } else if constexpr(has_vector_shape<T>) {
      return std::forward<T>(x).shape();
    }
  }
} shape{};

inline constexpr struct size_fn_ {
  template <typename T>
  auto operator()(T&& x) const
  requires(GRB_SPEC_NAMESPACE::is_tag_invocable_v<size_fn_, T> ||
           std::ranges::sized_range<T>)
  {
    if constexpr(GRB_SPEC_NAMESPACE::is_tag_invocable_v<size_fn_, T>) {
      return GRB_SPEC_NAMESPACE::tag_invoke(*this, std::forward<T>(x));
    } else if constexpr(std::ranges::sized_range<T>) {
      return std::ranges::size(std::forward<T>(x));
    }
  }
} size{};

inline constexpr struct find_fn_ {
  template <typename T>
  auto operator()(T&& x, typename GRB_SPEC_NAMESPACE::container_traits<T>::key_type key) const
  requires(GRB_SPEC_NAMESPACE::is_tag_invocable_v<find_fn_, T, typename GRB_SPEC_NAMESPACE::container_traits<T>::key_type> ||
           has_find_method<T>)
  {
    if constexpr(GRB_SPEC_NAMESPACE::is_tag_invocable_v<find_fn_, T, typename GRB_SPEC_NAMESPACE::container_traits<T>::key_type>) {
      return GRB_SPEC_NAMESPACE::tag_invoke(*this, std::forward<T>(x), key);
    } else if constexpr(has_find_method<T>) {
      return std::forward<T>(x).find(key);
    }
  }
} find{};

inline constexpr struct insert_fn_ {
  template <typename T>
  auto operator()(T&& x, const container_value_t<T>& entry) const
  requires(GRB_SPEC_NAMESPACE::is_tag_invocable_v<insert_fn_, T, const container_value_t<T>&> ||
           has_insert_method<T>)
  {
    if constexpr(GRB_SPEC_NAMESPACE::is_tag_invocable_v<insert_fn_, T, const container_value_t<T>&>) {
      return GRB_SPEC_NAMESPACE::tag_invoke(*this, std::forward<T>(x), entry);
    } else if constexpr(has_insert_method<T>) {
      return std::forward<T>(x).insert(entry);
    }
  }
} insert{};

inline constexpr struct insert_or_assign_fn_ {
  template <typename T, typename M>
  auto operator()(T&& x, typename GRB_SPEC_NAMESPACE::container_traits<T>::key_type key, M&& obj) const
  requires(GRB_SPEC_NAMESPACE::is_tag_invocable_v<insert_or_assign_fn_, T, typename GRB_SPEC_NAMESPACE::container_traits<T>::key_type, M> ||
           has_insert_or_assign_method<T, M>)
  {
    if constexpr(GRB_SPEC_NAMESPACE::is_tag_invocable_v<insert_or_assign_fn_, T, typename GRB_SPEC_NAMESPACE::container_traits<T>::key_type, M>) {
      return GRB_SPEC_NAMESPACE::tag_invoke(*this, std::forward<T>(x), key, std::forward<M>(obj));
    } else if constexpr(has_insert_or_assign_method<T, M>) {
      return std::forward<T>(x).insert_or_assign(key, std::forward<M>(obj));
    }
  }
} insert_or_assign{};

} // end GRB_SPEC_NAMESPACE