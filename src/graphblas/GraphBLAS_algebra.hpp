/*
 * Copyright (c) 2017 Carnegie Mellon University
 * All Rights Reserved.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS," WITH NO WARRANTIES WHATSOEVER. CARNEGIE
 * MELLON UNIVERSITY AND THE TRUSTEES OF INDIANA UNIVERSITY EXPRESSLY DISCLAIM
 * TO THE FULLEST EXTENT PERMITTED BY LAW ALL EXPRESS, IMPLIED, AND STATUTORY
 * WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT OF PROPRIETARY RIGHTS.
 *
 * This Program is distributed under a BSD license.  Please see LICENSE file or
 * permission@sei.cmu.edu for more information.  DM-0002659
 */

#ifndef GRAPHBLAS_ALGEBRA_HPP
#define GRAPHBLAS_ALGEBRA_HPP

#include <math.h>
#include <algorithm>
#include <functional>
#include <limits>
#include <utility>

namespace GraphBLAS
{
    //****************************************************************************
    // The Unary Operators
    //****************************************************************************

    // Also performs casting
    template <typename D1, typename D2 = D1>
    struct Identity
    {
        typedef D2 result_type;
        inline D2 operator()(D1 input) { return input; }
    };

    template <typename D1 = bool, typename D2 = D1>
    struct LogicalNot
    {
        typedef D2 result_type;
        inline D2 operator()(D1 input) { return !input; }
    };

    template <typename D1, typename D2 = D1>
    struct AdditiveInverse
    {
        typedef D2 result_type;
        inline D2 operator()(D1 input) { return -input; }
    };

    template <typename D1, typename D2 = D1>
    struct MultiplicativeInverse
    {
        typedef D2 result_type;
        inline D2 operator()(D1 input)
        {
            return static_cast<D2>(1) / input;
        }
    };

    //****************************************************************************
    // The Binary Operators
    //****************************************************************************

    template <typename D1 = bool, typename D2 = D1, typename D3 = D1>
    struct LogicalOr
    {
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs || rhs; }
    };

    template <typename D1 = bool, typename D2 = D1, typename D3 = D1>
    struct LogicalAnd
    {
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs && rhs; }
    };

    template <typename D1 = bool, typename D2 = D1, typename D3 = D1>
    struct LogicalXor
    {
        typedef D3 result_type;
        // ((bool)lhs) != ((bool)rhs)
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs ^ rhs; }
    };

    template <typename D1, typename D2 = D1, typename D3 = bool>
    struct Equal
    {
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs == rhs; }
    };

    template <typename D1, typename D2 = D1, typename D3 = bool>
    struct NotEqual
    {
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs != rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = bool>
    struct GreaterThan
    {
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs > rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = bool>
    struct LessThan
    {
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs < rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = bool>
    struct GreaterEqual
    {
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs >= rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = bool>
    struct LessEqual
    {
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs <= rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct First
    {
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Second
    {
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Min
    {
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs < rhs ? lhs : rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Max
    {
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs < rhs ? rhs : lhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Plus
    {
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs + rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Minus
    {
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs - rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Times
    {
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs * rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Div
    {
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs / rhs; }
    };

} // namespace GraphBLAS

//****************************************************************************
// Monoids
//****************************************************************************

#define GB_GEN_MONOID(M_NAME, BINARYOP, IDENTITY)               \
    template <typename ScalarT>                                 \
    struct M_NAME                                               \
    {                                                           \
    public:                                                     \
        typedef ScalarT ScalarType;                             \
        typedef ScalarT result_type;                            \
                                                                \
        ScalarT identity() const            \
        {                                                       \
            return static_cast<ScalarT>(IDENTITY);              \
        }                                                       \
                                                                \
        ScalarT operator()(ScalarT&& lhs, ScalarT&& rhs) const                 \
        {                                                       \
            return BINARYOP<ScalarT>()(lhs, rhs);                     \
        }                                                       \
    };

namespace GraphBLAS
{
    GB_GEN_MONOID(PlusMonoid, Plus, 0)
    GB_GEN_MONOID(TimesMonoid, Times, 1)
    GB_GEN_MONOID(MinMonoid, Min, std::numeric_limits<ScalarT>::max())
    GB_GEN_MONOID(MaxMonoid, Max, 0)
    GB_GEN_MONOID(LogicalOrMonoid, LogicalOr, false)
} // GraphBLAS

//****************************************************************************
// Semirings
//****************************************************************************

/**
 * The macro for building semi-ring objects
 *
 * @param[in]  SRNAME  The class name
 * @param[in]  SRADD   The addition monoid
 * @param[in]  SRMULT  The multiplication binary function
 */
#define GB_GEN_SEMIRING(SRNAME, ADD_MONOID, MULT_BINARYOP)              \
    template <typename D1, typename D2=D1, typename D3=D1>              \
    class SRNAME                                                        \
    {                                                                   \
    public:                                                             \
        typedef D3 ScalarType;                                          \
        typedef D3 result_type;                                         \
                                                                        \
        D3 add(D3&& a, D3&& b) const                \
        { return ADD_MONOID<D3>()(std::forward<D3>(a),                  \
                                  std::forward<D3>(b)); }               \
                                                                        \
        D3 mult(D1&& a, D2&& b) const               \
        { return MULT_BINARYOP<D1,D2,D3>()(std::forward<D1>(a),         \
                                           std::forward<D2>(b)); }      \
                                                                        \
        ScalarType zero() const                     \
        { return ADD_MONOID<D3>().identity(); }                         \
    };


namespace GraphBLAS
{
    GB_GEN_SEMIRING(ArithmeticSemiring, PlusMonoid, Times)

    GB_GEN_SEMIRING(LogicalSemiring, LogicalOrMonoid, LogicalAnd)

    GB_GEN_SEMIRING(MinPlusSemiring, MinMonoid, Plus)
    GB_GEN_SEMIRING(MaxTimesSemiring, MaxMonoid, Times)

    GB_GEN_SEMIRING(MinSelect2ndSemiring, MinMonoid, Second)
    GB_GEN_SEMIRING(MaxSelect2ndSemiring, MaxMonoid, Second)

    GB_GEN_SEMIRING(MinSelect1stSemiring, MinMonoid, First)
} // graphblas


#endif // GB_ALGEBRA_HPP
