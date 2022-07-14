/*
 * GraphBLAS Template Library (GBTL), Version 3.0
 *
 * Copyright 2020 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors.
 *
 * THIS MATERIAL WAS PREPARED AS AN ACCOUNT OF WORK SPONSORED BY AN AGENCY OF
 * THE UNITED STATES GOVERNMENT.  NEITHER THE UNITED STATES GOVERNMENT NOR THE
 * UNITED STATES DEPARTMENT OF ENERGY, NOR THE UNITED STATES DEPARTMENT OF
 * DEFENSE, NOR CARNEGIE MELLON UNIVERSITY, NOR BATTELLE, NOR ANY OF THEIR
 * EMPLOYEES, NOR ANY JURISDICTION OR ORGANIZATION THAT HAS COOPERATED IN THE
 * DEVELOPMENT OF THESE MATERIALS, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS,
 * OR USEFULNESS OR ANY INFORMATION, APPARATUS, PRODUCT, SOFTWARE, OR PROCESS
 * DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED
 * RIGHTS.
 *
 * Released under a BSD-style license, please see LICENSE file or contact
 * permission@sei.cmu.edu for full terms.
 *
 * [DISTRIBUTION STATEMENT A] This material has been approved for public release
 * and unlimited distribution.  Please see Copyright notice for non-US
 * Government use and distribution.
 *
 * DM20-0442
 */

#pragma once

#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <functional>
#include <limits>
#include <utility>

namespace grb
{
    namespace detail
    {
        // Overload Abs for different types (call the right library func).
        template<typename D2>
        inline D2 MyAbs(int8_t input)        { return abs(input); }
        template<typename D2>
        inline D2 MyAbs(int16_t input)       { return abs(input); }
        template<typename D2>
        inline D2 MyAbs(int32_t input)       { return abs(input); } // labs?
        template<typename D2>
        inline D2 MyAbs(int64_t input)       { return labs(input); } // llabs?

        template<typename D2>
        inline D2 MyAbs(float input)         { return fabsf(input); }

        template<typename D2>
        inline D2 MyAbs(double input)        { return fabs(input); }

        // all other types are unsigned; i.e., no op.
        template<typename D2>
        inline D2 MyAbs(bool input)          { return input; }
        template<typename D2>
        inline D2 MyAbs(uint8_t input)       { return input; }
        template<typename D2>
        inline D2 MyAbs(uint16_t input)      { return input; }
        template<typename D2>
        inline D2 MyAbs(uint32_t input)      { return input; }
        template<typename D2>
        inline D2 MyAbs(uint64_t input)      { return input; }

    } // namespace detail (within grb namespace

    //************************************************************************
    // The Unary Operators
    //************************************************************************
    // Following the removal of std::unary_function from C++17 and beyond,
    // these functors do not need to subclass from unary_function and don't
    // need to define:
    //   - argument_type
    //   - result_type

    // Also performs casting
    template <typename D1, typename D2 = D1>
    struct Identity
    {
        inline D2 operator()(D1 input) const { return input; }
    };

    template <typename D1, typename D2 = D1>
    struct Abs
    {
        inline D2 operator()(D1 input) const
        {
            return detail::MyAbs<D2>(input);
        }
    };


    template <typename D1, typename D2 = D1>
    struct AdditiveInverse
    {
        inline D2 operator()(D1 input) const { return -input; }
    };

    template <typename D1, typename D2 = D1>
    struct MultiplicativeInverse
    {
        inline D2 operator()(D1 input) const
        {
            return static_cast<D2>(1) / input;
        }
    };


    /// @todo should D2 default to bool
    template <typename D1 = bool, typename D2 = D1>
    struct LogicalNot
    {
        inline D2 operator()(D1 input) const { return !input; }
    };

    template <typename I1 = uint64_t, typename I2 = I1,
              typename std::enable_if_t<std::is_integral_v<I1> &&
                                        std::is_integral_v<I2>, int> = 0>
    struct BitwiseNot
    {
        inline I2 operator()(I1 input) const { return ~input; }
    };

    //************************************************************************
    // User std::bind to turn binary ops into unary ops
    //
    // Turn a binary op into a unary op by binding the 2nd term to a constant
    //
    //                     std::bind(grb::Minus<float>(),
    //                               std::placeholders::_1,
    //                               static_cast<float>(nsver)),
    //
    // Turn a binary op into a unary op by binding the 1st term to a constant
    //
    //                     std::bind(grb::Minus<float>(),
    //                               static_cast<float>(nsver),
    //                               std::placeholders::_1),
    //
    //************************************************************************
}

namespace grb
{
    //************************************************************************
    // The Binary Operators
    //************************************************************************
    // Following the removal of std::binary_function from C++17 and beyond,
    // these functors do not need to subclass from binary_function and don't
    // need to define:
    //   - first_argument_type
    //   - second_argument_type
    //   - result_type
    //
    // In lambda speak
    // [](auto x, auto y) → D3 { return x * y };
    // [](D1 x, D2 y) -> D3 { return x * y; }

    template <typename D1 = bool, typename D2 = D1, typename D3 = D1>
    struct LogicalOr
    {
        inline D3 operator()(D1 lhs, D2 rhs) const { return lhs || rhs; }
    };

    template <typename D1 = bool, typename D2 = D1, typename D3 = D1>
    struct LogicalAnd
    {
        inline D3 operator()(D1 lhs, D2 rhs) const { return lhs && rhs; }
    };

    template <typename D1 = bool, typename D2 = D1, typename D3 = D1>
    struct LogicalXor
    {
        // ((bool)lhs) != ((bool)rhs)
        // inline D3 operator()(D1 lhs, D2 rhs) const { return lhs ^ rhs; }
        inline D3 operator()(D1 lhs, D2 rhs) const
        {
            return ((lhs && !rhs) || (!lhs && rhs));
        }
    };

    template <typename D1 = bool, typename D2 = D1, typename D3 = D1>
    struct LogicalXnor
    {
        // ((bool)lhs) != ((bool)rhs)
        // inline D3 operator()(D1 lhs, D2 rhs) const { return !(lhs ^ rhs); }
        inline D3 operator()(D1 lhs, D2 rhs) const
        {
            return ((lhs && rhs) || (!lhs && !rhs));
        }
    };

    //-------------------------------------------------------------------------
    /// @todo Consider decltype for D3's default
    template <typename I1, typename I2 = I1, typename I3 = I1,
              typename std::enable_if_t<std::is_integral_v<I1> &&
                                        std::is_integral_v<I2> &&
                                        std::is_integral_v<I3>, int> = 0>
    struct BitwiseOr
    {
        inline I3 operator()(I1 lhs, I2 rhs) const { return lhs | rhs; }
    };

    template <typename I1, typename I2 = I1, typename I3 = I1,
              typename std::enable_if_t<std::is_integral_v<I1> &&
                                        std::is_integral_v<I2> &&
                                        std::is_integral_v<I3>, int> = 0>
    struct BitwiseAnd
    {
        inline I3 operator()(I1 lhs, I2 rhs) const { return lhs & rhs; }
    };

    template <typename I1, typename I2 = I1, typename I3 = I1,
              typename std::enable_if_t<std::is_integral_v<I1> &&
                                        std::is_integral_v<I2> &&
                                        std::is_integral_v<I3>, int> = 0>
    struct BitwiseXor
    {
        inline I3 operator()(I1 lhs, I2 rhs) const { return lhs ^ rhs; }
    };

    template <typename I1, typename I2 = I1, typename I3 = I1,
              typename std::enable_if_t<std::is_integral_v<I1> &&
                                        std::is_integral_v<I2> &&
                                        std::is_integral_v<I3>, int> = 0>
    struct BitwiseXnor
    {
        inline I3 operator()(I1 lhs, I2 rhs) const { return ~(lhs ^ rhs); }
    };

    //-------------------------------------------------------------------------

    template <typename D1, typename D2 = D1, typename D3 = bool>
    struct Equal
    {
        inline D3 operator()(D1 lhs, D2 rhs) const { return lhs == rhs; }
    };

    template <typename D1, typename D2 = D1, typename D3 = bool>
    struct NotEqual
    {
        inline D3 operator()(D1 lhs, D2 rhs) const { return lhs != rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = bool>
    struct GreaterThan
    {
        inline D3 operator()(D1 lhs, D2 rhs) const { return lhs > rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = bool>
    struct LessThan
    {
        inline D3 operator()(D1 lhs, D2 rhs) const { return lhs < rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = bool>
    struct GreaterEqual
    {
        inline D3 operator()(D1 lhs, D2 rhs) const { return lhs >= rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = bool>
    struct LessEqual
    {
        inline D3 operator()(D1 lhs, D2 rhs) const { return lhs <= rhs; }
    };

    //-------------------------------------------------------------------------

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct First
    {
        inline D3 operator()(D1 lhs, D2 rhs) const { return lhs; }
    };

    // Note output type, D3, defaults to D2
    template<typename D1, typename D2 = D1, typename D3 = D2>
    struct Second
    {
        inline D3 operator()(D1 lhs, D2 rhs) const { return rhs; }
    };

    //-------------------------------------------------------------------------
    /// @todo Consider decltype for D3's default
    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Min
    {
        inline D3 operator()(D1 lhs, D2 rhs) const { return lhs < rhs ? lhs : rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Max
    {
        inline D3 operator()(D1 lhs, D2 rhs) const { return lhs < rhs ? rhs : lhs; }
    };

    //-------------------------------------------------------------------------
    /// @todo Consider decltype for D3's default
    ///
    /// from std::plus<>::operator() (since C++14)
    ///
    /// template<typename T, typename U>
    /// constexpr auto operator()(T&& &lhs, U&& rhs) const
    ///  -> decltype(std::forward<T>(lhs) + std::forward<U>(rhs));

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Plus
    {
        inline D3 operator()(D1 lhs, D2 rhs) const { return lhs + rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Minus
    {
        inline D3 operator()(D1 lhs, D2 rhs) const { return lhs - rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Times
    {
        inline D3 operator()(D1 lhs, D2 rhs) const { return lhs * rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Div
    {
        inline D3 operator()(D1 lhs, D2 rhs) const { return lhs / rhs; }
    };

    // Note: not in GraphBLAS C API Specification
    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Power
    {
        inline D3 operator()(D1 lhs, D2 rhs) const { return std::pow(lhs, rhs); }
    };

} // namespace grb


//****************************************************************************
// Monoids
//****************************************************************************

//****************************************************************************
/**
 * The macro for building simple templated monoid classes
 *
 * @param[in]  M_NAME     The class name
 * @param[in]  BINARYOP   The binary op callable to turn into a monoid
 * @param[in]  IDENTITY   The multiplication binary function
 *
 * Note: the following only generates a template class where the identity is
 * the "same" (with casting) regardless of ScalarT
 *
 * @todo Explore if this can be done with a class template like with MaxMonoid
 */
#define GEN_GRAPHBLAS_MONOID(M_NAME, BINARYOP, IDENTITY)        \
    template <typename ScalarT>                                 \
    struct M_NAME                                               \
    {                                                           \
    public:                                                     \
        using result_type = ScalarT;                            \
                                                                \
        ScalarT identity() const                                \
        {                                                       \
            return static_cast<ScalarT>(IDENTITY);              \
        }                                                       \
                                                                \
        ScalarT operator()(ScalarT lhs, ScalarT rhs) const      \
        {                                                       \
            return BINARYOP<ScalarT>()(lhs, rhs);               \
        }                                                       \
    };

//****************************************************************************
namespace grb
{
    GEN_GRAPHBLAS_MONOID(PlusMonoid, Plus, 0)
    GEN_GRAPHBLAS_MONOID(TimesMonoid, Times, 1)

    /// @todo the following identity only works for boolean domain
    GEN_GRAPHBLAS_MONOID(LogicalOrMonoid,   LogicalOr,   false)
    GEN_GRAPHBLAS_MONOID(LogicalAndMonoid,  LogicalAnd,  true)
    GEN_GRAPHBLAS_MONOID(LogicalXorMonoid,  LogicalXor,  false)
    GEN_GRAPHBLAS_MONOID(LogicalXnorMonoid, LogicalXnor, true)

    // ***********************************************************************
    // MaxMonoid identity depends on the type requiring class templates and SFINAE
    // See below for explicit instantiations
    template <typename ScalarT, typename Enable = void>
    class MaxMonoid;

    // MaxMonoid for ints
    template <typename ScalarT>
    class MaxMonoid<ScalarT,
                    typename std::enable_if_t<std::is_integral_v<ScalarT> > >
    {
    public:
        using result_type = ScalarT;

        ScalarT identity() const
        {
            return static_cast<ScalarT>(std::numeric_limits<ScalarT>::min());
        }

        ScalarT operator()(ScalarT lhs, ScalarT rhs) const
        {
            return grb::Max<ScalarT>()(lhs, rhs);
        }
    };

    // MaxMonoid for floating point numbers
    template <typename ScalarT>
    class MaxMonoid<ScalarT,
                    typename std::enable_if_t<std::is_floating_point_v<ScalarT> > >
    {
    public:
        using result_type = ScalarT;

        ScalarT identity() const
        {
            return static_cast<ScalarT>(-std::numeric_limits<ScalarT>::infinity());
        }

        ScalarT operator()(ScalarT lhs, ScalarT rhs) const
        {
            return grb::Max<ScalarT>()(lhs, rhs);
        }
    };


    //GEN_GRAPHBLAS_MONOID(MinMonoid, Min, std::numeric_limits<ScalarT>::max())

    // ***********************************************************************
    // MinMonoid identity depends on the type requiring class templates and SFINAE
    // See below for explicit instantiations
    template <typename ScalarT, typename Enable = void>
    class MinMonoid;

    // MinMonoid for ints
    template <typename ScalarT>
    class MinMonoid<ScalarT,
                    typename std::enable_if_t<std::is_integral_v<ScalarT> > >
    {
    public:
        using result_type = ScalarT;

        ScalarT identity() const
        {
            return static_cast<ScalarT>(std::numeric_limits<ScalarT>::max());
        }

        ScalarT operator()(ScalarT lhs, ScalarT rhs) const
        {
            return grb::Min<ScalarT>()(lhs, rhs);
        }
    };

    // MinMonoid for floating point numbers
    template <typename ScalarT>
    class MinMonoid<ScalarT,
                    typename std::enable_if_t<std::is_floating_point_v<ScalarT> > >
    {
    public:
        using result_type = ScalarT;

        ScalarT identity() const
        {
            return static_cast<ScalarT>(std::numeric_limits<ScalarT>::infinity());
        }

        ScalarT operator()(ScalarT lhs, ScalarT rhs) const
        {
            return grb::Min<ScalarT>()(lhs, rhs);
        }
    };

} // grb

//****************************************************************************
// Semirings
//****************************************************************************

/**
 * The macro for building simple templated semiring classes
 *
 * @param[in]  SRNAME        The class name
 * @param[in]  ADD_MONOID    The addition monoid
 * @param[in]  MULT_BINARYOP The multiplication binary operator
 */
#define GEN_GRAPHBLAS_SEMIRING(SRNAME, ADD_MONOID, MULT_BINARYOP)       \
    template <typename D1, typename D2=D1, typename D3=D1>              \
    class SRNAME                                                        \
    {                                                                   \
    public:                                                             \
        using first_argument_type = D1;                                 \
        using second_argument_type = D2;                                \
        using result_type = D3;                                         \
                                                                        \
        D3 add(D3 a, D3 b) const                                        \
        { return ADD_MONOID<D3>()(a, b); }                              \
                                                                        \
        D3 mult(D1 a, D2 b) const                                       \
        { return MULT_BINARYOP<D1,D2,D3>()(a, b); }                     \
                                                                        \
        D3 zero() const                                                 \
        { return ADD_MONOID<D3>().identity(); }                         \
    };


namespace grb
{
    //************************************************************************
    // "true" and "useful" semirings
    //************************************************************************

    //************************************************************************
    // "true" Arithmetic Semiring aka PlusTimesSemiring
    GEN_GRAPHBLAS_SEMIRING(ArithmeticSemiring, PlusMonoid, Times)

    //************************************************************************
    /// @note the Plus operator would need to be "infinity aware" if the caller
    /// were to pass "infinity" sentinel as one of the arguments. But no
    /// GraphBLAS operations 'should' do that.
    GEN_GRAPHBLAS_SEMIRING(MinPlusSemiring, MinMonoid, Plus)

    //************************************************************************
    // MaxPlusSemiring is a true semiring signed integers and floating point
    // MaxPlusSemiring is a "useful" semiring for unsigned ints
    GEN_GRAPHBLAS_SEMIRING(MaxPlusSemiring, MaxMonoid, Plus)

    //************************************************************************
    // MinTimesSemiring is a true semiring signed integers and floating point
    // MinTimesSemiring is a "useful" semiring for unsigned ints
    GEN_GRAPHBLAS_SEMIRING(MinTimesSemiring, MinMonoid, Times)

    //************************************************************************
    // MaxTimesSemiring is a true semiring for unsigned ints
    // MaxTimesSemiring is a "useful" semiring signed integers and floating point
    GEN_GRAPHBLAS_SEMIRING(MaxTimesSemiring, MaxMonoid, Times)

    //************************************************************************
    // more "true" semirings
    GEN_GRAPHBLAS_SEMIRING(MinMaxSemiring, MinMonoid, Max)
    GEN_GRAPHBLAS_SEMIRING(MaxMinSemiring, MaxMonoid, Min)

    //************************************************************************
    // PlusMinSemiring is a true semiring for unsigned ints
    // PlusMinSemiring is a "useful" semiring signed integers and floating point
    GEN_GRAPHBLAS_SEMIRING(PlusMinSemiring, PlusMonoid, Min)

    //************************************************************************
    /// @todo restrict to boolean?
    GEN_GRAPHBLAS_SEMIRING(LogicalSemiring, LogicalOrMonoid,   LogicalAnd)
    GEN_GRAPHBLAS_SEMIRING(AndOrSemiring,   LogicalAndMonoid,  LogicalOr)
    GEN_GRAPHBLAS_SEMIRING(XorAndSemiring,  LogicalXorMonoid,  LogicalAnd)
    GEN_GRAPHBLAS_SEMIRING(XnorOrSemiring,  LogicalXnorMonoid, LogicalOr)

    //************************************************************************
    // "other useful" semirings
    //************************************************************************
    GEN_GRAPHBLAS_SEMIRING(MinFirstSemiring, MinMonoid, First)
    GEN_GRAPHBLAS_SEMIRING(MinSecondSemiring, MinMonoid, Second)

    GEN_GRAPHBLAS_SEMIRING(MaxFirstSemiring, MaxMonoid, First)
    GEN_GRAPHBLAS_SEMIRING(MaxSecondSemiring, MaxMonoid, Second)

} // namespace grb

//****************************************************************************
// Convert Semirings to BinaryOps
//****************************************************************************

namespace grb
{
    //************************************************************************
    template <typename SemiringT>
    struct MultiplicativeOpFromSemiring
    {
    public:
        //using result_type = typename SemiringT::result_type;

        MultiplicativeOpFromSemiring() = delete;
        MultiplicativeOpFromSemiring(SemiringT const &sr) : sr(sr) {}

        typename SemiringT::result_type operator() (
            typename SemiringT::first_argument_type  lhs,
            typename SemiringT::second_argument_type rhs) const
        {
            return sr.mult(lhs, rhs);
        }

    private:
        SemiringT sr;
    };

    //************************************************************************
    template <typename SemiringT>
    struct AdditiveMonoidFromSemiring
    {
    public:
        //using result_type = typename SemiringT::result_type;

        AdditiveMonoidFromSemiring() = delete;
        AdditiveMonoidFromSemiring(SemiringT const &sr) : sr(sr) {}

        typename SemiringT::result_type identity() const
        {
            return sr.zero();
        }

        typename SemiringT::result_type operator() (
            typename SemiringT::result_type lhs,
            typename SemiringT::result_type rhs) const
        {
            return sr.add(lhs, rhs);
        }

    private:
        SemiringT sr;
    };

    //************************************************************************
    template <typename SemiringT>
    MultiplicativeOpFromSemiring<SemiringT>
    multiply_op(SemiringT const &sr)
    {
        return MultiplicativeOpFromSemiring<SemiringT>(sr);
    }

    //************************************************************************
    template <typename SemiringT>
    AdditiveMonoidFromSemiring<SemiringT>
    add_monoid(SemiringT const &sr)
    {
        return AdditiveMonoidFromSemiring<SemiringT>(sr);
    }

} // namespace grb
