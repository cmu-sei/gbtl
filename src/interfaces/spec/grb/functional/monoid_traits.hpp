
#pragma once

#include <type_traits>
#include <any>

namespace GRB_SPEC_NAMESPACE {

using any = std::any;

template <typename Fn, typename T, typename U = T, typename V = T>
inline constexpr bool is_binary_op_v = requires(Fn fn, T t, U u) {
                                         {fn(t, u)} -> std::convertible_to<V>;
                                       };

template <typename Fn, typename T>
inline constexpr bool has_identity_template_v = requires { {Fn:: template identity<T>()} -> std::same_as<T>; };

template <typename Fn, typename T>
inline constexpr bool has_identity_method_v = requires { {Fn::identity()} -> std::same_as<T>; };



template <typename Fn, typename T>
requires(
         is_binary_op_v<Fn, T, T, T> &&
         (has_identity_method_v<Fn, T> || has_identity_template_v<Fn, T>)
         )
class monoid_traits
{
public:
  static constexpr T identity() noexcept {
    if constexpr(has_identity_method_v<Fn, T>) {
      return Fn::identity();
    } else if constexpr(has_identity_template_v<Fn, T>) {
      return Fn:: template identity<T>();
    }
  }
};

template <typename T>
class monoid_traits<std::plus<T>, T>
{
public:
  static constexpr T identity() noexcept {
    return T(0);
  }
};

template <typename T>
class monoid_traits<std::plus<void>, T>
{
public:
  static constexpr T identity() noexcept {
    return T(0);
  }
};

template <typename Fn, typename T>
inline constexpr bool has_identity_v = requires { {GRB_SPEC_NAMESPACE::monoid_traits<Fn, T>::identity()} -> std::same_as<T>; };

template <typename Fn, typename T>
inline constexpr bool is_monoid_v = is_binary_op_v<Fn, T, T, T> &&
                                    has_identity_v<Fn, T>;

template <typename Fn, typename T, typename U = T, typename V = GRB_SPEC_NAMESPACE::any>
concept BinaryOperator = requires(Fn fn, T t, U u) {
                           {fn(t, u)} -> std::convertible_to<V>;
                         };

template <typename Fn, typename T>
concept Monoid = BinaryOperator<Fn, T, T, T> &&
                 requires { {GRB_SPEC_NAMESPACE::monoid_traits<Fn, T>::identity()} -> std::same_as<T>; };

} // end GRB_SPEC_NAMESPACE