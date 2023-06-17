#pragma once

#include <tuple>
#include <concepts>

#include "namespace_macros.hpp"

namespace GRB_SPEC_NAMESPACE {

template <typename T, std::size_t I>
concept method_gettable = requires(T tuple) {
  {tuple. template get<I>()} -> std::convertible_to<GRB_SPEC_NAMESPACE::any>;
};

template <typename T, std::size_t I>
concept adl_gettable = requires(T tuple) {
  {get<I>(tuple)} -> std::convertible_to<GRB_SPEC_NAMESPACE::any>;
};

template <typename T, std::size_t I>
concept std_gettable = requires(T tuple) {
  {std::get<I>(tuple)} -> std::convertible_to<GRB_SPEC_NAMESPACE::any>;
};

template <std::size_t I, typename T>
inline constexpr decltype(auto) get(T&& tuple)
requires(method_gettable<T, I>)
{
  return std::forward<T>(tuple). template get<I>();
}

template <std::size_t I, typename T>
inline constexpr decltype(auto) get(T&& tuple)
requires(!method_gettable<T, I> && std_gettable<T, I>)
{
  return std::get<I>(std::forward<T>(tuple));
}

} // end GRB_SPEC_NAMESPACE