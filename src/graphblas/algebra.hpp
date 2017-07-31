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
    /// @todo BinaryOps need typedefs for RHS arg so that ConstT
    /// can be defaulted (and ConstT is moved after BinaryOpT
    template <typename ConstT, typename BinaryOpT>
    struct BinaryOp_Bind2nd
    {
        ConstT n;
        BinaryOpT op;
        typedef typename BinaryOpT::result_type result_type;

        BinaryOp_Bind2nd(ConstT const &value,
                         BinaryOpT     operation = BinaryOpT() ) :
            n(value),
            op(operation)
        {}

        /// @todo value type should be BinaryOpT::D1
        /// @todo n should be cast to BinaryOpT::D2
        result_type operator()(result_type const &value)
        {
            return op(value, n);
        }
    };
}

/// @todo Remove or move to GraphBLAS namespace?
typedef GraphBLAS::Identity<bool>      GrB_IDENTITY_BOOL;
typedef GraphBLAS::Identity<int8_t>    GrB_IDENTITY_INT8;
typedef GraphBLAS::Identity<uint8_t>   GrB_IDENTITY_UINT8;
typedef GraphBLAS::Identity<int16_t>   GrB_IDENTITY_INT16;
typedef GraphBLAS::Identity<uint16_t>  GrB_IDENTITY_UINT16;
typedef GraphBLAS::Identity<int32_t>   GrB_IDENTITY_INT32;
typedef GraphBLAS::Identity<uint32_t>  GrB_IDENTITY_UINT32;
typedef GraphBLAS::Identity<int64_t>   GrB_IDENTITY_INT64;
typedef GraphBLAS::Identity<uint64_t>  GrB_IDENTITY_UINT64;
typedef GraphBLAS::Identity<float>     GrB_IDENTITY_FP32;
typedef GraphBLAS::Identity<double>    GrB_IDENTITY_FP64;

typedef GraphBLAS::AdditiveInverse<bool>      GrB_AINV_BOOL;
typedef GraphBLAS::AdditiveInverse<int8_t>    GrB_AINV_INT8;
typedef GraphBLAS::AdditiveInverse<uint8_t>   GrB_AINV_UINT8;
typedef GraphBLAS::AdditiveInverse<int16_t>   GrB_AINV_INT16;
typedef GraphBLAS::AdditiveInverse<uint16_t>  GrB_AINV_UINT16;
typedef GraphBLAS::AdditiveInverse<int32_t>   GrB_AINV_INT32;
typedef GraphBLAS::AdditiveInverse<uint32_t>  GrB_AINV_UINT32;
typedef GraphBLAS::AdditiveInverse<int64_t>   GrB_AINV_INT64;
typedef GraphBLAS::AdditiveInverse<uint64_t>  GrB_AINV_UINT64;
typedef GraphBLAS::AdditiveInverse<float>     GrB_AINV_FP32;
typedef GraphBLAS::AdditiveInverse<double>    GrB_AINV_FP64;

typedef GraphBLAS::MultiplicativeInverse<bool>      GrB_MINV_BOOL;
typedef GraphBLAS::MultiplicativeInverse<int8_t>    GrB_MINV_INT8;
typedef GraphBLAS::MultiplicativeInverse<uint8_t>   GrB_MINV_UINT8;
typedef GraphBLAS::MultiplicativeInverse<int16_t>   GrB_MINV_INT16;
typedef GraphBLAS::MultiplicativeInverse<uint16_t>  GrB_MINV_UINT16;
typedef GraphBLAS::MultiplicativeInverse<int32_t>   GrB_MINV_INT32;
typedef GraphBLAS::MultiplicativeInverse<uint32_t>  GrB_MINV_UINT32;
typedef GraphBLAS::MultiplicativeInverse<int64_t>   GrB_MINV_INT64;
typedef GraphBLAS::MultiplicativeInverse<uint64_t>  GrB_MINV_UINT64;
typedef GraphBLAS::MultiplicativeInverse<float>     GrB_MINV_FP32;
typedef GraphBLAS::MultiplicativeInverse<double>    GrB_MINV_FP64;

typedef GraphBLAS::LogicalNot<bool>    GrB_LNOT;

namespace GraphBLAS
{
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
        // inline D3 operator()(D1 lhs, D2 rhs) { return lhs ^ rhs; }
        inline D3 operator()(D1 lhs, D2 rhs)
        {
            return ((lhs && !rhs) || (!lhs && rhs));
        }
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

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Power
    {
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
