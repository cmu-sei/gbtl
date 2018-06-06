/*
 * Copyright (c) 2015 Carnegie Mellon University and The Trustees of Indiana
 * University.
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

#ifndef GB_ALGEBRA_HPP
#define GB_ALGEBRA_HPP

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

    //************************************************************************
    // Turn a binary op into a unary op by binding the 2nd term to a constant
    //
    template <typename ConstT, typename BinaryOpT>
    struct BinaryOp_Bind2nd
    {
        ConstT n;  /// @todo consider defaulting ConstT to BinaryOpT::rhs_type
        BinaryOpT op;
        typedef typename BinaryOpT::result_type result_type;
        typedef typename BinaryOpT::lhs_type ValueType;

        BinaryOp_Bind2nd(ConstT const &value,
                         BinaryOpT     operation = BinaryOpT() ) :
            n(value),
            op(operation)
        {}

        result_type operator()(ValueType const &value)
        {
            return op(value, n);
        }
    };
}

namespace GraphBLAS
{
    //****************************************************************************
    // The Binary Operators
    //****************************************************************************

    template <typename D1 = bool, typename D2 = D1, typename D3 = D1>
    struct LogicalOr
    {
        typedef D1 lhs_type;
        typedef D2 rhs_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs || rhs; }
    };

    template <typename D1 = bool, typename D2 = D1, typename D3 = D1>
    struct LogicalAnd
    {
        typedef D1 lhs_type;
        typedef D2 rhs_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs && rhs; }
    };

    template <typename D1 = bool, typename D2 = D1, typename D3 = D1>
    struct LogicalXor
    {
        typedef D1 lhs_type;
        typedef D2 rhs_type;
        typedef D3 result_type;
        // ((bool)lhs) != ((bool)rhs)
        // inline D3 operator()(D1 lhs, D2 rhs) { return lhs ^ rhs; }
        inline D3 operator()(D1 lhs, D2 rhs)
        {
            return ((lhs && !rhs) || (!lhs && rhs));
        }
    };

    template <typename D1, typename D2 = D1, typename D3 = bool>
    struct Equal
    {
        typedef D1 lhs_type;
        typedef D2 rhs_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs == rhs; }
    };

    template <typename D1, typename D2 = D1, typename D3 = bool>
    struct NotEqual
    {
        typedef D1 lhs_type;
        typedef D2 rhs_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs != rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = bool>
    struct GreaterThan
    {
        typedef D1 lhs_type;
        typedef D2 rhs_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs > rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = bool>
    struct LessThan
    {
        typedef D1 lhs_type;
        typedef D2 rhs_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs < rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = bool>
    struct GreaterEqual
    {
        typedef D1 lhs_type;
        typedef D2 rhs_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs >= rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = bool>
    struct LessEqual
    {
        typedef D1 lhs_type;
        typedef D2 rhs_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs <= rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct First
    {
        typedef D1 lhs_type;
        typedef D2 rhs_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Second
    {
        typedef D1 lhs_type;
        typedef D2 rhs_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Min
    {
        typedef D1 lhs_type;
        typedef D2 rhs_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs < rhs ? lhs : rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Max
    {
        typedef D1 lhs_type;
        typedef D2 rhs_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs < rhs ? rhs : lhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Plus
    {
        typedef D1 lhs_type;
        typedef D2 rhs_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs + rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Minus
    {
        typedef D1 lhs_type;
        typedef D2 rhs_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs - rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Times
    {
        typedef D1 lhs_type;
        typedef D2 rhs_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs * rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Div
    {
        typedef D1 lhs_type;
        typedef D2 rhs_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs / rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Power
    {
        typedef D1 lhs_type;
        typedef D2 rhs_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return std::pow(lhs, rhs); }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Xor
    {
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return (lhs ^ rhs); }
    };

} // namespace GraphBLAS


typedef GraphBLAS::LogicalOr<bool>    GrB_LOR;
typedef GraphBLAS::LogicalAnd<bool>   GrB_LAND;
typedef GraphBLAS::LogicalXor<bool>   GrB_LXOR;

//****************************************************************************
// Monoids
//****************************************************************************

#define GEN_GRAPHBLAS_MONOID(M_NAME, BINARYOP, IDENTITY)               \
    template <typename ScalarT>                                 \
    struct M_NAME                                               \
    {                                                           \
    public:                                                     \
        typedef ScalarT ScalarType;                             \
        typedef ScalarT result_type;                            \
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

namespace GraphBLAS
{
    GEN_GRAPHBLAS_MONOID(PlusMonoid, Plus, 0)
    GEN_GRAPHBLAS_MONOID(TimesMonoid, Times, 1)
    GEN_GRAPHBLAS_MONOID(MinMonoid, Min, std::numeric_limits<ScalarT>::max())

    /// @todo The following identity only works for unsigned domains
    /// std::numerical_limits<>::min() does not work for floating point types
    GEN_GRAPHBLAS_MONOID(MaxMonoid, Max, 0)

    GEN_GRAPHBLAS_MONOID(LogicalOrMonoid, LogicalOr, false)
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
#define GEN_GRAPHBLAS_SEMIRING(SRNAME, ADD_MONOID, MULT_BINARYOP)       \
    template <typename D1, typename D2=D1, typename D3=D1>              \
    class SRNAME                                                        \
    {                                                                   \
    public:                                                             \
        typedef D3 ScalarType;                                          \
        typedef D3 result_type;                                         \
                                                                        \
        D3 add(D3 a, D3 b) const                                        \
        { return ADD_MONOID<D3>()(a, b); }                              \
                                                                        \
        D3 mult(D1 a, D2 b) const                                       \
        { return MULT_BINARYOP<D1,D2,D3>()(a, b); }                     \
                                                                        \
        ScalarType zero() const                                         \
        { return ADD_MONOID<D3>().identity(); }                         \
    };


namespace GraphBLAS
{
    GEN_GRAPHBLAS_SEMIRING(ArithmeticSemiring, PlusMonoid, Times)

    GEN_GRAPHBLAS_SEMIRING(LogicalSemiring, LogicalOrMonoid, LogicalAnd)

    /// @note the Plus operator would need to be "infinity aware" if the caller
    /// were to pass "infinity" sentinel as one of the arguments. But no GraphBLAS
    /// operations should do that.
    GEN_GRAPHBLAS_SEMIRING(MinPlusSemiring, MinMonoid, Plus)

    GEN_GRAPHBLAS_SEMIRING(MaxTimesSemiring, MaxMonoid, Times)

    GEN_GRAPHBLAS_SEMIRING(MinSelect2ndSemiring, MinMonoid, Second)
    GEN_GRAPHBLAS_SEMIRING(MaxSelect2ndSemiring, MaxMonoid, Second)

    GEN_GRAPHBLAS_SEMIRING(MinSelect1stSemiring, MinMonoid, First)
    GEN_GRAPHBLAS_SEMIRING(MaxSelect1stSemiring, MaxMonoid, First)
} // namespace GraphBLAS


#endif // GB_ALGEBRA_HPP
