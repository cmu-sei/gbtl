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

namespace graphblas
{
    /// @DEPRECATED

    // used by cluster.hpp and linalg_utils.hpp
    namespace math
    {
         /**
         * @brief Standard power function.
         *
         * @param[in] base     The base.
         * @param[in] exponent The exponent.
         *
         * @return  The base raised to the exponent.
         */
        template <typename T>
        __device__ __host__ inline T power(T base, T exponent)
        {
            return static_cast<T>(pow(static_cast<double>(base),
                                      static_cast<double>(exponent)));
        }
  } // math
} // graphblas


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

    typedef <typename D1 = bool, typename D2 = D1>
    struct LogicalNot
    {
        typedef D2 result_type;
        inline D2 operator()(D1 input) { return !input; }
    };

    typedef <typename D1, typename D2 = D1>
    struct AdditiveInverse
    {
        typedef D2 result_type;
        inline D2 operator()(D1 input) { return -input; }
    };

    typedef <typename D1, typename D2 = D1>
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

    typedef <typename D1 = bool, typename D2 = D1, typename D3 = D1>
    struct LogicalOr
    {
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs || rhs; }
    };

    typedef <typename D1 = bool, typename D2 = D1, typename D3 = D1>
    struct LogicalAnd
    {
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs && rhs; }
    };

    typedef <typename D1 = bool, typename D2 = D1, typename D3 = D1>
    struct LogicalXor
    {
        typedef D3 result_type;
        // ((bool)lhs) != ((bool)rhs)
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs ^ rhs; }
    };

    typedef <typename D1, typename D2 = D1, typename D3 = bool>
    struct Equal
    {
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) { return lhs == rhs; }
    };

    typedef <typename D1, typename D2 = D1, typename D3 = bool>
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
        typedef T result_type;
        inline T operator()(D1 lhs, D2 rhs) { return lhs; }
    };

    template<typename T>
    struct Second
    {
        typedef T result_type;
        inline T operator()(D1 lhs, D2 rhs) { return rhs; }
    };

    template<typename T>
    struct Min
    {
        typedef T result_type;
        inline T operator()(D1 lhs, D2 rhs) { return lhs < rhs ? lhs : rhs; }
    };

    template<typename T>
    struct Max
    {
        typedef T result_type;
        inline T operator()(D1 lhs, D2 rhs) { return lhs < rhs ? rhs : lhs; }
    };

    template<typename T>
    struct Plus
    {
        typedef T result_type;
        inline T operator()(D1 lhs, D2 rhs) { return lhs + rhs; }
    };

    template<typename T>
    struct Minus
    {
        typedef T result_type;
        inline T operator()(D1 lhs, D2 rhs) { return lhs - rhs; }
    };

    template<typename T>
    struct Times
    {
        typedef T result_type;
        inline T operator()(D1 lhs, D2 rhs) { return lhs * rhs; }
    };

    template<typename T>
    struct Div
    {
        typedef T result_type;
        inline T operator()(D1 lhs, D2 rhs) { return lhs / rhs; }
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
        __host__ __device__ ScalarT identity() const            \
        {                                                       \
            return static_cast<ScalarT>(IDENTITY);              \
        }                                                       \
                                                                \
        __host__ __device__ ScalarT operator()(ScalarT&& lhs,       \
                                               ScalarT&& rhs) const \
        {                                                       \
            return OP<ScalarT>()(lhs, rhs);                     \
        }                                                       \
    };

namespace GraphBLAS
{
    /// @todo Is zero the correct identity? negative inf? negative max?
    //GB_GEN_MONOID(RegmaxMonoid, graphblas::math::regmax, 0)

    GB_GEN_MONOID(PlusMonoid, Plus, 0)
    GB_GEN_MONOID(TimesMonoid, Times, 1)

    // Use Binary operator instead
    // The right-identity is 1, the left identity is 1/rhs
    //GB_GEN_MONOID(DivMonoid, graphblas::math::annihilator_div, 1)

    //GB_GEN_MONOID(MinMonoid,
    //              graphblas::math::arithmetic_min,
    //              std::numeric_limits<ScalarT>::max())

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
#define GB_GEN_SEMIRING(SRNAME, SRADD, SRMULT)                          \
    template <typename D1, typename D2=D1, typename D3=D1>              \
    class SRNAME                                                        \
    {                                                                   \
    public:                                                             \
        typedef D3 ScalarType;                                          \
        typedef D3 result_type;                                         \
                                                                        \
        __host__ __device__ D3 add(D3&& a, D3&& b) const                \
        { return SRADD<D3>()(std::forward<D3>(a),                       \
                             std::forward<D3>(b)); }                    \
                                                                        \
        __host__ __device__ D3 mult(D1&& a, D2&& b) const               \
        { return SRMULT<D1,D2,D3>()(std::forward<D1>(a),                \
                                    std::forward<D2>(b)); }             \
                                                                        \
        __host__ __device__ ScalarType zero() const                     \
        { return SRADD<D3>().identity(); }                              \
    };


namespace graphblas
{
    GB_GEN_SEMIRING(ArithmeticSemiring, math::plus, math::times, 0, 1)

    /** @note We are using annihilator_plus because arithmetic_plus only
     *        works when there is a 'real' infinity.  For scalars like
     *        integers max does not work without treating it specially to
     *        behave like infinity when using the plus function:
     *
     *        <max> + a == <max>
     */
    GB_GEN_SEMIRING(MinPlusSemiring,
                    math::arithmetic_min,
                    math::annihilator_plus,
                    std::numeric_limits<ScalarType>::max(), 0)

    /* logical ops using integer 0 and 1 */
    GB_GEN_SEMIRING(IntLogicalSemiring, math::or_fn, math::and_fn, 0, 1)

    /// @todo Is zero correct? negative inf? negative max?
    GB_GEN_SEMIRING(MaxTimesSemiring, math::regmax, math::times, 0, 1)

    /**
     * @note MinSelect2ndSemiring is not a real semiring!
     * @todo Verify that 0 is okay as the "one" (multiplicative identity)
     */
    GB_GEN_SEMIRING(MinSelect2ndSemiring,
                    math::annihilator_min,
                    math::select2nd,
                    std::numeric_limits<ScalarType>::max(),
                    0)


    /**
     * @note MaxSelect2ndSemiring is not a real semiring!
     * @note Zero max identity assume positive values
     */
    GB_GEN_SEMIRING(MaxSelect2ndSemiring,
                    math::annihilator_max,
                    math::select2ndZero,
                    0,
                    1)

    /**
     * @todo Can we remove this semiring and just use the MinSelect2ndSemiring
     *       above.
     * @todo This no longer works with the annihilator min that is sensitive to
     *       max() instead of 0.
     * @note In this case, the multiplicative identity is 1.
     */
    GB_GEN_SEMIRING(MinSelect1stSemiring,
                    math::arithmetic_min,
                    math::select1st,
                    std::numeric_limits<ScalarType>::max(),
                    1)

    //************************************************************************
    template <typename SemiringT>
    struct AdditiveMonoidFromSemiring
    {
    public:
        typedef typename SemiringT::ScalarType ScalarType;
        typedef typename SemiringT::ScalarType result_type;

        typedef typename SemiringT::ScalarType first_argument_type;
        typedef typename SemiringT::ScalarType second_argument_type;

        AdditiveMonoidFromSemiring() : sr() {}
        AdditiveMonoidFromSemiring(SemiringT const &sr) : sr(sr) {}

        __host__ __device__ ScalarType identity() const
        {
            return sr.zero();
        }

        template <typename LhsT, typename RhsT>
        __host__ __device__ ScalarType operator()(LhsT &&lhs, RhsT &&rhs) const
        {
            return sr.add(std::forward<LhsT>(lhs), std::forward<RhsT>(rhs));
        }

    private:
        SemiringT sr;
    };


    //************************************************************************
    template <typename SemiringT>
    struct MultiplicativeMonoidFromSemiring
    {
    public:
        typedef typename SemiringT::ScalarType ScalarType;
        typedef typename SemiringT::ScalarType result_type;

        typedef typename SemiringT::ScalarType first_argument_type;
        typedef typename SemiringT::ScalarType second_argument_type;

        MultiplicativeMonoidFromSemiring() : sr() {}
        MultiplicativeMonoidFromSemiring(SemiringT const &sr) : sr(sr) {}

        __host__ __device__ ScalarType identity() const
        {
            return sr.one();
        }

        template <typename LhsT, typename RhsT>
        __host__ __device__ ScalarType operator()(LhsT &&lhs, RhsT &&rhs) const
        {
            return sr.mult(std::forward<LhsT>(lhs), std::forward<RhsT>(rhs));
        }

    private:
        SemiringT sr;
    };

    //************************************************************************
    template <typename SemiringT>
    AdditiveMonoidFromSemiring<SemiringT>
    make_additive_monoid_from_semiring(SemiringT const &sr)
    {
        return AdditiveMonoidFromSemiring<SemiringT>(sr);
    }

    //************************************************************************
    template <typename SemiringT>
    MultiplicativeMonoidFromSemiring<SemiringT>
    make_multiplicative_monoid_from_semiring(SemiringT const &sr)
    {
        return MultiplicativeMonoidFromSemiring<SemiringT>(sr);
    }
} // graphblas


#endif // GB_ALGEBRA_HPP
