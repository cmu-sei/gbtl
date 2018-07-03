/*
 * GraphBLAS Template Library, Version 2.0
 *
 * Copyright 2018 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors. All Rights Reserved.
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
 * RIGHTS..
 *
 * Released under a BSD (SEI)-style license, please see license.txt or contact
 * permission@sei.cmu.edu for full terms.
 *
 * This release is an update of:
 *
 * 1. GraphBLAS Template Library (GBTL)
 * (https://github.com/cmu-sei/gbtl/blob/1.0.0/LICENSE) Copyright 2015 Carnegie
 * Mellon University and The Trustees of Indiana. DM17-0037, DM-0002659
 *
 * DM18-0559
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
    //************************************************************************
    // The Unary Operators
    //************************************************************************

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
    //************************************************************************
    // The Binary Operators
    //************************************************************************

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

#define GEN_GRAPHBLAS_MONOID(M_NAME, BINARYOP, IDENTITY)        \
    template <typename ScalarT>                                 \
    struct M_NAME                                               \
    {                                                           \
    public:                                                     \
        typedef ScalarT lhs_type;                               \
        typedef ScalarT rhs_type;                               \
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
 * @param[in]  SRNAME        The class name
 * @param[in]  ADD_MONOID    The addition monoid
 * @param[in]  MULT_BINARYOP The multiplication binary function
 */
#define GEN_GRAPHBLAS_SEMIRING(SRNAME, ADD_MONOID, MULT_BINARYOP)       \
    template <typename D1, typename D2=D1, typename D3=D1>              \
    class SRNAME                                                        \
    {                                                                   \
    public:                                                             \
        typedef D1 lhs_type;                                            \
        typedef D2 rhs_type;                                            \
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
    /// were to pass "infinity" sentinel as one of the arguments. But no
    /// GraphBLAS operations 'should' do that.
    GEN_GRAPHBLAS_SEMIRING(MinPlusSemiring, MinMonoid, Plus)

    GEN_GRAPHBLAS_SEMIRING(MaxTimesSemiring, MaxMonoid, Times)

    GEN_GRAPHBLAS_SEMIRING(MinSelect2ndSemiring, MinMonoid, Second)
    GEN_GRAPHBLAS_SEMIRING(MaxSelect2ndSemiring, MaxMonoid, Second)

    GEN_GRAPHBLAS_SEMIRING(MinSelect1stSemiring, MinMonoid, First)
    GEN_GRAPHBLAS_SEMIRING(MaxSelect1stSemiring, MaxMonoid, First)
} // namespace GraphBLAS

//****************************************************************************
// Convert Semirings to BinaryOps
//****************************************************************************

namespace GraphBLAS
{
    //************************************************************************
    template <typename SemiringT>
    struct MultiplicativeOpFromSemiring
    {
    public:
        typedef typename SemiringT::lhs_type lhs_type;
        typedef typename SemiringT::rhs_type rhs_type;
        typedef typename SemiringT::result_type result_type;
        typedef typename SemiringT::ScalarType ScalarType;

        MultiplicativeOpFromSemiring() = delete;
        MultiplicativeOpFromSemiring(SemiringT const &sr) : sr(sr) {}

        ScalarType operator() (lhs_type lhs, rhs_type rhs) const
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
        typedef typename SemiringT::result_type result_type;
        typedef typename SemiringT::ScalarType ScalarType;

        AdditiveMonoidFromSemiring() = delete;
        AdditiveMonoidFromSemiring(SemiringT const &sr) : sr(sr) {}

        ScalarType identity() const
        {
            return sr.zero();
        }

        ScalarType operator() (ScalarType lhs, ScalarType rhs) const
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

} // namespace GraphBLAS

#endif // GB_ALGEBRA_HPP
