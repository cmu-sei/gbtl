#pragma once

#include <type_traits>
#include <concepts>
#include <utility>
#include <ranges>
#include <tuple>
#include <any>

#include "tag_invoke.hpp"
#include "namespace_macros.hpp"

namespace GRB_SPEC_NAMESPACE {

using any = std::any;

namespace {

template <typename Container>
struct get_index_type {
private:
  using value_type = std::ranges::range_value_t<Container>;
  using key_type = typename std::tuple_element<0, value_type>::type;
public:
  using type = key_type;
};

template <typename Container>
requires requires { typename std::tuple_element<0, typename std::tuple_element<0, std::ranges::range_value_t<Container>>::type>::type; }
struct get_index_type<Container>
{
private:
  using value_type = std::ranges::range_value_t<Container>;
  using key_type = typename std::tuple_element<0, value_type>::type;
public:
  using type = typename std::tuple_element<0, key_type>::type;
};
}

template <typename Container>
struct container_traits {
  using size_type = std::ranges::range_size_t<Container>;
  using difference_type = std::ranges::range_difference_t<Container>;
  using iterator = std::ranges::iterator_t<Container>;
  using value_type = std::ranges::range_value_t<Container>;
  using key_type = std::remove_cvref_t<typename std::tuple_element<0, value_type>::type>;
  using map_type = std::remove_cvref_t<typename std::tuple_element<1, value_type>::type>;
  using scalar_type = std::remove_cvref_t<typename std::tuple_element<1, value_type>::type>;
  using index_type = std::remove_cvref_t<typename get_index_type<Container>::type>;
  using reference = std::ranges::range_reference_t<Container>;
};

template <typename Container>
struct matrix_traits {
  using size_type = std::ranges::range_size_t<Container>;
  using difference_type = std::ranges::range_difference_t<Container>;
  using iterator = std::ranges::iterator_t<Container>;
  using value_type = std::ranges::range_value_t<Container>;
  using key_type = std::remove_cvref_t<typename std::tuple_element<0, value_type>::type>;
  using map_type = std::remove_cvref_t<typename std::tuple_element<1, value_type>::type>;
  using scalar_type = std::remove_cvref_t<typename std::tuple_element<1, value_type>::type>;
  using index_type = std::remove_cvref_t<typename std::tuple_element<0, key_type>::type>;
  using reference = std::ranges::range_reference_t<Container>;
};

template <typename Container>
struct vector_traits {
  using size_type = std::ranges::range_size_t<Container>;
  using difference_type = std::ranges::range_difference_t<Container>;
  using iterator = std::ranges::iterator_t<Container>;
  using value_type = std::ranges::range_value_t<Container>;
  using key_type = std::remove_cvref_t<typename std::tuple_element<0, value_type>::type>;
  using map_type = std::remove_cvref_t<typename std::tuple_element<1, value_type>::type>;
  using scalar_type = std::remove_cvref_t<typename std::tuple_element<1, value_type>::type>;
  using index_type = key_type;
  using reference = std::ranges::range_reference_t<Container>;
};

template <typename T>
using container_scalar_t = typename container_traits<T>::scalar_type;

template <typename T>
using container_index_t = typename container_traits<T>::index_type;

template <typename T>
using container_value_t = typename container_traits<T>::value_type;

template <typename T>
using container_key_t = typename container_traits<T>::key_type;

template <typename T>
using matrix_scalar_t = typename container_traits<std::remove_cvref_t<T>>::scalar_type;

template <typename T>
using matrix_index_t = typename container_traits<std::remove_cvref_t<T>>::index_type;

template <typename T>
using vector_scalar_t = typename container_traits<std::remove_cvref_t<T>>::scalar_type;

template <typename T>
using vector_index_t = typename container_traits<std::remove_cvref_t<T>>::index_type;

template <typename A, typename B, typename Fn>
using elementwise_return_type_t = std::invoke_result_t<Fn, container_scalar_t<A>, container_scalar_t<B>>;

} // end GRB_SPEC_NAMESPACE