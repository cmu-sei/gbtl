#pragma once

#include <utility>
#include <limits>
#include <type_traits>
#include "monoid_traits.hpp"

namespace GRB_SPEC_NAMESPACE {

template <typename Fn, typename T = void>
class unary_op_impl_ {
public:
  constexpr T operator()(const T& value) const {
    return Fn{}(value);
  }
};

template <typename Fn>
class unary_op_impl_<Fn, void> {
public:
  template <typename T>
  constexpr auto operator()(T&& value) const {
    return Fn{}(std::forward<T>(value));
  }
};

template <typename Fn, typename T = void, typename U = T, typename V = void>
class binary_op_impl_ {
public:
  constexpr V operator()(const T& lhs, const U& rhs) const {
    return Fn{}(lhs, rhs);
  }

  template <typename X>
  static constexpr X identity()
  requires(has_identity_template_v<Fn, X>)
  {
    return Fn:: template identity<X>();
  }
};

template <typename Fn, typename T, typename U>
class binary_op_impl_<Fn, T, U, void> {
public:
  constexpr auto operator()(const T& lhs, const U& rhs) const {
    return Fn{}(lhs, rhs);
  }

  static constexpr T identity()
  requires(std::is_same_v<T, U> &&
           has_identity_template_v<Fn, T>)
  {
    return Fn:: template identity<T>();
  }
};

template <typename Fn>
class binary_op_impl_<Fn, void, void, void>
{
public:
  template <typename T, typename U>
  constexpr auto operator()(T&& lhs, U&& rhs) const
  {
    return Fn{}(lhs, rhs);
  }

  template <typename T>
  static constexpr T identity()
  requires(has_identity_template_v<Fn, T>)
  {
    return Fn:: template identity<T>();
  }
};

class plus_impl_ {
public:
  template <typename T, typename U>
  constexpr auto operator()(T&& lhs, U&& rhs) const {
    return std::forward<T>(lhs) + std::forward<U>(rhs);
  }

  template <typename T>
  static constexpr T identity() {
    return T(0);
  }
};

/// Binary Operator to perform subtraction. Uses the `-` operator.
struct minus_impl_ {
  template <typename T, typename U>
  constexpr auto operator()(const T& a, const U& b) const {
    return a - b;
  }
};

/// Binary Operator to perform multiplication. Uses the `*` operator.
struct multiplies_impl_ {
  template <typename T, typename U>
  constexpr auto operator()(const T& a, const U& b) const {
    return a * b;
  }

  template <typename T>
  static constexpr T identity() {
    return T(1);
  }
};

/// Binary Operator to perform division. Uses the `/` operator.
struct divides_impl_ {
  template <typename T, typename U>
  constexpr auto operator()(const T& a, const U& b) const {
    return a / b;
  }
};

// omit std::functional comparisons

/// Binary Operator to perform max, returning the greater of the two values,
/// or the first element if they are equal.  Uses the `<` operator.

template <typename T, typename U>
using larger_max_integral_t =
  std::conditional_t<std::cmp_less(std::numeric_limits<T>::max(),
                                   std::numeric_limits<U>::max()),
                     U, T>;

struct max_impl_ {
  template <std::integral T, std::integral U>
  constexpr auto operator()(const T& a, const U& b) const
    -> std::conditional_t<
        std::cmp_less(std::numeric_limits<T>::max(),
                      std::numeric_limits<U>::max()),
        U, T
       >
  {
    if (std::cmp_less(a, b)) {
      return b;
    } else {
      return a;
    }
  }

  template <typename T, typename U>
  constexpr auto operator()(const T& a, const U& b) const
    -> std::conditional_t<
        std::numeric_limits<T>::max() < std::numeric_limits<U>::max(),
        U, T
       >
  requires(!(std::is_integral_v<T> && std::is_integral_v<U>))
  {
    if (a < b) {
      return b;
    } else {
      return a;
    }
  }

  template <typename T>
  static constexpr T identity()
  requires(std::numeric_limits<T>::is_specialized())
  {
    return std::min(std::numeric_limits<T>::lowest(), -std::numeric_limits<T>::infinity());
  }
};


/// Binary Operator to perform min, returning the lesser of the two values,
/// or the first element if they are equal.  Uses the `<` operator.
struct min_impl_ {
  template <std::integral T, std::integral U>
  constexpr auto operator()(const T& a, const U& b) const
    -> std::conditional_t<
         std::cmp_less(std::numeric_limits<U>::lowest(),
                       std::numeric_limits<T>::lowest()),
         U, T
       >
  {
    if (std::cmp_less(b, a)) {
      return b;
    } else {
      return a;
    }
  }

  template <typename T, typename U>
  constexpr auto operator()(const T& a, const U& b) const
    -> std::conditional_t<
         std::numeric_limits<U>::lowest() < std::numeric_limits<T>::lowest(),
         U, T
       >
  requires(!(std::is_integral_v<T> && std::is_integral_v<U>))
  {
    if (b < a) {
      return b;
    } else {
      return a;
    }
  }

  template <typename T>
  static constexpr T identity()
  requires(std::numeric_limits<T>::is_specialized())
  {
    return std::max(std::numeric_limits<T>::max(), std::numeric_limits<T>::infinity());
  }
};

/// Binary Operator to perform modulus, uses the `%` operator.
struct modulus_impl_ {
  template <typename T, typename U>
  constexpr auto operator()(const T& a, const U& b) const {
    return a % b;
  }
};

/// The binary operator `grb::plus`, which forms a monoid
/// on integral types.
/*
template <typename T = void, typename U = T, typename V = void>
using plus = binary_op_impl_<plus_impl_, T, U, V>;
*/

template <typename T = void, typename U = T, typename V = void>
struct plus : public binary_op_impl_<plus_impl_, T, U, V> {};

template <typename T = void, typename U = T, typename V = void>
struct minus : public binary_op_impl_<minus_impl_, T, U, V> {};

template <typename T = void, typename U = T, typename V = void>
struct multiplies : public binary_op_impl_<multiplies_impl_, T, U, V> {};

template <typename T = void, typename U = T, typename V = void>
struct times : public multiplies<T, U, V> {};

template <typename T = void, typename U = T, typename V = void>
struct divides : public binary_op_impl_<divides_impl_, T, U, V> {};

template <typename T = void, typename U = T, typename V = void>
struct max : public binary_op_impl_<max_impl_, T, U, V> {};

template <typename T = void, typename U = T, typename V = void>
struct min : public binary_op_impl_<min_impl_, T, U, V> {};

template <typename T = void, typename U = T, typename V = void>
struct modulus : public binary_op_impl_<modulus_impl_, T, U, V> {};

// Unary operators

struct negate_impl_ {
  template <typename T>
  constexpr auto operator()(const T& a) const {
    return -a;
  }
};

struct logical_not_impl_ {
  template <typename T>
  constexpr auto operator()(const T& a) const {
    return !a;
  }
};

template <typename T = void>
struct negate : public unary_op_impl_<negate_impl_, T> {};

template <typename T = void>
struct logical_not : public unary_op_impl_<logical_not_impl_, T> {};

// Logical operators

struct logical_and_impl_ {
  template <typename T, typename U>
  constexpr auto operator()(const T& a, const U& b) const {
    return a && b;
  }

  template <typename T>
  static constexpr T identity() {
    return T(true);
  }
};

struct logical_or_impl_ {
  template <typename T, typename U>
  constexpr auto operator()(const T& a, const U& b) const {
    return a || b;
  }

  template <typename T>
  static constexpr T identity() {
    return T(false);
  }
};

struct logical_xor_impl_ {
  template <typename T, typename U>
  constexpr auto operator()(const T& a, const U& b) const {
    return (a || b) && !(a && b);
  }

  template <typename T>
  static constexpr T identity() {
    return T(false);
  }
};

struct logical_xnor_impl_ {
  template <typename T, typename U>
  constexpr auto operator()(const T& a, const U& b) const {
    return !((a || b) && !(a && b));
  }
};

template <typename T = void, typename U = T, typename V = void>
struct logical_and : binary_op_impl_<logical_and_impl_, T, U, V> {};

template <typename T = void, typename U = T, typename V = void>
struct logical_or : binary_op_impl_<logical_or_impl_, T, U, V> {};

template <typename T = void, typename U = T, typename V = void>
struct logical_xor : binary_op_impl_<logical_xor_impl_, T, U, V> {};

template <typename T = void, typename U = T, typename V = void>
struct logical_xnor : binary_op_impl_<logical_xnor_impl_, T, U, V> {};

template <typename T = void>
struct take_left {
  T operator()(const T& left, const T& right) {
    return left;
  }
};

template <>
struct take_left<void> {
  template <typename T, typename U>
  T operator()(const T& left, const U& right) const {
    return left;
  }
};

template <typename T = void>
struct take_right {
  T operator()(const T& left, const T& right) const {
    return right;
  }
};

template <>
struct take_right<void> {
  template <typename T, typename U>
  U operator()(const T& left, const U& right) const {
    return right;
  }
};

} // end GRB_SPEC_NAMESPACE
