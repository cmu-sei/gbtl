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

/**
 * @todo  Sometimes, we return zeros here, but how do we know what "0" is?
 *        Should we set the semiring globally, so that we know what "0",
 *        "plus", and "times" are globally?
 */

namespace graphblas
{
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

        template <typename T>
        struct Power
        {
            typedef T result_type;
            __device__ __host__ inline T operator()(T base, T exponent)
            {
                power(base, exponent);
            }
        };

        /**
         * @brief Unary arithmetic negation.
         *
         * @param[in]  a  the number to negate.
         *
         * @return  Negative a.
         */
        template <typename T>
        __device__ __host__ inline T negate(T a) { return -a; }

        template <typename T>
        struct Negate
        {
            typedef T result_type;
            __device__ __host__ inline T operator()(T a) { return negate(a); }
        };

        /**
         * @brief Unary inverse.
         *
         * @param[in]  a  The number to take the inverse of.
         *
         * @return  The inverse of a if it is not zero; otherwise, zero.
         */
        template <typename T>
        __device__ __host__ inline T inverse(T a)
        {
            if (a == static_cast<T>(0))
            {
                return static_cast<T>(0);
            }
            else
            {
                return static_cast<T>(1) / a;
            }
        }

        template<typename T>
        struct Inverse
        {
            typedef T result_type;
            __device__ __host__ inline T operator()(T a) { return inverse(a); }
        };

        /**
         * @brief Standard arithmetic addition.
         *
         * @param[in]  a  An addend (or augend)
         * @param[in]  b  An addend
         *
         * @return  The sum of a and b.
         */
        template <typename T>
        __host__ __device__ T plus(T const &a, T const &b) { return a + b; }

        template <typename T>
        struct Plus
        {
            typedef T result_type;

            __host__ __device__ T identity() { return static_cast<T>(0); }

            __host__ __device__ T operator()(T const &a, T const &b) const
            {
                return plus(a, b);
            }
        };

        /**
         * @brief Standard arithmetic subtraction.
         *
         * @param[in]  a  The minuend (number to subtract b from).
         * @param[in]  b  The subtrahend (number to subtract from a).
         *
         * @return The difference a - b.
         */
        template <typename T>
        __device__ __host__  inline T sub(T a, T b) { return a - b; }

        template <typename T>
        struct Sub
        {
            typedef T result_type;
            __device__ __host__ inline T operator()(T a, T b)
            {
                return sub(a, b);
            }
        };

        /**
         * @brief Standard arithmetic multiplication.
         *
         * @param[in]  a  The multiplicand (number to multiply by b).
         * @param[in]  b  The multiplier (number to multiply a by).
         *
         * @return The product of a and b.
         */
        template <typename T>
        __host__ __device__ T times(T const &a, T const &b) { return a * b; }

        template <typename T>
        struct Times
        {
            typedef T result_type;
            __host__ __device__ T operator()(T const &a, T const &b) const
            {
                return times(a, b);
            }
        };

        /**
         * @brief Standard arithmetic division.
         *
         * @param[in]  a  The dividend
         * @param[in]  b  The divisor
         *
         * @return a divided by b.
         *
         * @note This does not check to see if the denominator is zero.
         */
        template <typename T>
        __device__ __host__ inline T div(T a, T b) { return a / b; }

        template<typename T>
        struct Div
        {
            typedef T result_type;
            __device__ __host__ inline T operator()(T a, T b)
            {
                return div(a, b);
            }
        };

        /**
         * @brief Standard arithmetic minimum.
         *
         * @param[in]  a  An element to compare
         * @param[in]  b  An element to compare
         *
         * @return The arithmetic minimum of a and b,
         *
         */
        template <typename T>
        __device__ __host__ inline T arithmetic_min(T a, T b)
        {
            //return std::min(a, b);
            return (a < b) ? a : b;
        }

        template <typename T>
        struct ArithmeticMin
        {
            typedef T result_type;
            __device__ __host__ inline T operator()(T a, T b)
            {
                return arithmetic_min(a, b);
            }
        };


        /**
         * @brief Maximum adjusted to take into account the
         *        structural zero (value 0)
         *
         * @param[in]  a  An element to compare
         * @param[in]  b  An element to compare
         *
         * @return If neither is zero, return the minimum of a and b;
         *         if one element is zero, return the other element;
         *         otherwise, return zero.
         *
         */
        template <typename T>
        __device__ __host__ inline T annihilator_max(T a, T b)
        {
            if (a == static_cast<T>(0))
            {
                return b;
            }
            else if (b == static_cast<T>(0))
            {
                return a;
            }
            else
            {
                return (a > b) ? a : b;
            }
        }

        /**
         * @brief Minimum adjusted to take into account the
         *        structural zero (for Min-Plus semirings).
         *
         * @param[in]  a  An element to compare
         * @param[in]  b  An element to compare
         *
         * @return If neither is zero, return the minimum of a and b;
         *         if one element is zero, return the other element;
         *         otherwise, return zero.
         *
         */
        template <typename T>
        __device__ __host__ inline T annihilator_min(T a, T b)
        {
            if (a == std::numeric_limits<T>::max())
            {
                return b;
            }
            else if (b == std::numeric_limits<T>::max())
            {
                return a;
            }
            else
            {
                return (a < b) ? a : b;
            }
        }

        template<typename T>
        struct AnnihilatorMin
        {
            typedef T result_type;
            __device__ __host__ T operator()(T a, T b)
            {
                return annihilator_min(a, b);
            }
        };

        /**
         * @brief Arithmetic addition that takes an annihilator into account
         *
         * @param[in]  a  An addend (or augend)
         * @param[in]  b  An addend
         *
         * @return  The sum of a and b.
         */
        template <typename T>
        __host__ __device__ T annihilator_plus(T const &a, T const &b)
        {
            if ((a == std::numeric_limits<T>::max()) ||
                (b == std::numeric_limits<T>::max()))
            {
                return std::numeric_limits<T>::max();
            }
            else
            {
                return a + b;
            }
        }

        template <typename SemiringT>
        struct AnnihilatorPlus
        {
            typedef typename SemiringT::ScalarType T;
            typedef T result_type;

            __host__ __device__ T identity() const
            {
                return SemiringT().zero();
            }

            __host__ __device__ T operator()(T const &a, T const &b) const
            {
                T zero = identity();
                if (a == zero)
                {
                    return b;
                }
                else if (b == zero)
                {
                    return a;
                }
                else
                {
                    return a + b;
                }
            }
        };

        /**
         * @brief Arithmetic multiplication that takes an annihilator into
         *       account.
         *
         * @param[in]  a  A factor
         * @param[in]  b  A factor
         *
         * @return  The product of a and b.
         */
        template <typename SemiringT>
        struct AnnihilatorTimes
        {
            typedef typename SemiringT::ScalarType T;
            typedef T result_type;

            __host__ __device__ T operator()(T const &a, T const &b) const
            {
                T zero = SemiringT().zero();
                if ((a == zero) || (b == zero))
                {
                    return zero;
                }
                else
                {
                    return a * b;
                }
            }
        };

        /**
         * @brief Standard arithmetic maximum.
         *
         * @param[in]  a  An element to compare
         * @param[in]  b  An element to compare
         *
         * @return The arithmetic maximum of a and b,
         */
        template <typename T>
        __device__ __host__ inline T regmax(T a, T b)
        {
            //return std::max(a, b);
            return (a > b) ? a : b;
        }

        template <typename T>
        struct Regmax
        {
            typedef T result_type;
            __device__ __host__ inline T operator()(T a, T b)
            {
                return regmax(a, b);
            }
        };

        /**
         * @brief Increment second by 1 if neither first nor second is zero.
         *
         * @param[in]  first   An element to compare
         * @param[in]  second  An element to compare
         *
         * @return b + 1 if a and b are not zero; otherwise zero.
         */
        template <typename T>
        __device__ __host__ inline T incr_second(T first, T second)
        {
            return ((first  != static_cast<T>(0)) &&
                    (second != static_cast<T>(0))) ? (second + 1) : 0;
        }

        template <typename T>
        struct IncrSecond
        {
            typedef T result_type;
            __device__ __host__ inline T operator()(T first, T second)
            {
                return incr_second(first, second);
            }
        };

        /**
         * @brief select1st operation for min/select1st semirings when
         *        adjacency matrix is being premultiplied by row wavefronts.
         *
         * @param[in]  first   The left hand side
         * @param[in]  second  The right hand side (from graph)
         *
         * @return second if first is not the semiring "zero";
         *         otherwise return the semiring "zero"
         *.
         * @todo Need to hook into MinMonoid for the "zero" value.  Or the
         *       Semiring in general.
         */
        template <typename T>
        __device__ __host__ inline
        T select1st(T first, T second)
        {
            return (second != std::numeric_limits<T>::max()) ? first : second;
        }

        template<typename T>
        struct Select1st
        {
            typedef T result_type;
            __device__ __host__ T operator()(T first, T second)
            {
                return select1st(first, second);
            }
        };

        /**
         * @brief select2nd operation for min/select2nd semirings when
         *        adjacency matrix is post multiplied by col wavefronts
         *
         * @param[in]  first   The left hand side (from graph)
         * @param[in]  second  The right hand side
         *
         * @return if first is not 0, return the second; otherwise return 0.
         * @todo Need to hook into MinMonoid for the "zero" value.
         */
        template <typename T>
        __device__ __host__ inline
        T select2nd(T first, T second)
        {
            return (first != std::numeric_limits<T>::max()) ? second : first;
        }

        template <typename T>
        __device__ __host__ inline
        T select2ndZero(T first, T second)
        {
            return (first != 0) ? second : first;
        }

        template<typename T>
        struct Select2nd
        {
            typedef T result_type;
            __device__ __host__ T operator()(T first, T second)
            {
                return select2nd(first, second);
            }
        };

        /**
         * @brief Determine whether or not two elements are equal
         *
         * @param[in]  a  An element to compare
         * @param[in]  b  An element to compare
         *
         * @return A boolean for a == b.
         */
        template <typename T>
        __device__ __host__ inline bool is_equal(T a, T b) { return a == b; }

        template<typename T>
        struct IsEqual
        {
            typedef T result_type;
            __device__ __host__ bool operator()(T a, T b)
            {
                return is_equal(a, b);
            }
        };

        /**
         * @brief Check for equality with zero.
         *
         * @param[in]  a  An element to compare
         *
         * @return A boolean for a == 0.
         */
        template <typename T>
        __device__ __host__ bool is_zero(T a)
        {
            return a == static_cast<T>(0);
        }

        template <typename T>
        struct IsZero
        {
            using result_type = bool;
            __device__ __host__ inline bool operator()(T a)
            {
                return is_zero(a);
            }
        };

        template <typename SemiringT>
        struct IsNotStructuralZero
        {
            typedef typename SemiringT::ScalarType T;
            using result_type = T;

            __device__ __host__ inline T operator()(T a)
            {
                return ((a == SemiringT().zero()) ?
                        SemiringT().zero() :
                        SemiringT().one());
            }
        };


        template <typename SemiringT>
        struct IsStructuralZero
        {
            typedef typename SemiringT::ScalarType T;
            using result_type = T;

            __device__ __host__ inline T operator()(T a)
            {
                return ((a == SemiringT().zero()) ?
                        SemiringT().one() :
                        SemiringT().zero());
            }
        };


        /**
         * @brief Determine whether or not an item should be removed because of
         *        repetition.
         *
         * @note This works with the Min-Plus semiring, meaning that
         *       true is 0, and false is infinity.  Used in MIS algorithm
         *
         * @param[in]  a  An element to compare
         * @param[in]  b  An element to compare
         *
         * @return 0 if b == 0 or b == 1, otherwise return a.
         *
         * @todo Improve the explanation of this function
         */
        template <typename T>
        __device__ __host__ inline T remove_item(T a, T b)
        {
            if ((b == 0) || (b == 1))
            {
                return static_cast<T>(0);
            }
            else
            {
                return a;
            }
        }

        template <typename T>
        struct RemoveItem
        {
            typedef T result_type;
            __device__ __host__ inline T operator()(T a, T b)
            {
                return remove_item(a, b);
            }
        };

        /**
         * @brief Standard logical xor.
         *
         * @param[in]  a  first operand
         * @param[in]  b  second operand
         *
         * @return the exclusive-or of a and b.
         */
        template <typename T>
        __device__ __host__ inline T xor_fn(T a, T b) { return a ^ b; }

        template <typename T>
        struct XorFn
        {
            typedef T result_type;
            __device__ __host__ inline T operator()(T a, T b)
            {
                return xor_fn(a, b);
            }
        };

        /**
         * @brief Standard logical or.
         *
         * @param[in]  a  first operand
         * @param[in]  b  second operand
         *
         * @return The logical-or of a and b.
         */
        template <typename T>
        __device__ __host__ inline bool or_fn(T a, T b) { return a || b; }

        template <typename T>
        struct OrFn
        {
            typedef T result_type;
            __device__ __host__ inline bool operator()(T a, T b)
            {
                return or_fn(a, b);
            }
        };

        /**
         * @brief Standard logical and.
         *
         * @param[in]  a  first operand
         * @param[in]  b  second operand
         *
         * @return The logical-and of a and b.
         */
        template <typename T>
        __device__ __host__ inline bool and_fn(T a, T b) { return a && b; }

        template <typename T>
        struct AndFn
        {
            typedef T result_type;
            __device__ __host__ inline bool operator()(T a, T b)
            {
                return and_fn(a, b);
            }
        };

        /**
         * @brief Standard logical not.
         *
         * @param[in]  a   The operand to logically invert.
         *
         * @return !a.
         */
        template <typename T>
        __device__ __host__ inline bool not_fn(T a) { return !a; }

        template <typename T>
        struct NotFn
        {
            typedef T result_type;
            __device__ __host__ inline bool operator()(T a)
            {
                return not_fn(a);
            }
        };
  } // math
} // graphblas


//****************************************************************************
// Monoids
//****************************************************************************

#define GB_GEN_MONOID(M_NAME, OP, IDENTITY)              \
    template <typename ScalarT>                          \
    struct M_NAME                                        \
    {                                                    \
    public:                                              \
        typedef ScalarT ScalarType;                      \
        typedef ScalarT result_type;                     \
                                                         \
        __host__ __device__ ScalarT identity() const     \
        {                                                \
            return static_cast<ScalarT>(IDENTITY);       \
        }                                                \
                                                         \
        template<typename LhsT, typename RhsT>           \
        __host__ __device__ ScalarT operator()(LhsT&& lhs,       \
                                               RhsT&& rhs) const \
        {                                                \
            return OP<ScalarT>(lhs, rhs);                \
        }                                                \
    };

namespace graphblas
{
    /// @todo Is zero the correct identity? negative inf? negative max?
    GB_GEN_MONOID(RegmaxMonoid, graphblas::math::regmax, 0)

    GB_GEN_MONOID(PlusMonoid, graphblas::math::plus, 0)
    GB_GEN_MONOID(TimesMonoid, graphblas::math::times, 1)

    GB_GEN_MONOID(MinMonoid,
                  graphblas::math::arithmetic_min,
                  std::numeric_limits<ScalarT>::max())

} // graphblas

//****************************************************************************
// Semirings
//****************************************************************************

/**
*   @todo We need to create semirings from two monoids:
*   additive and multiplicative.
*/

/**
 * The macro for building semi-ring objects
 *
 * @param[in]  SRNAME  The class name
 * @param[in]  SRADD   The addition binary function
 * @param[in]  SRMULT  The multiplication binary function
 * @param[in]  SRZERO  The addition funcion identity value
 * @param[in]  SRONE   The multiplication function identity value
 */
#define GB_GEN_SEMIRING(SRNAME, SRADD, SRMULT, SRZERO, SRONE)           \
    template <typename ScalarT>                                         \
    class SRNAME                                                        \
    {                                                                   \
    public:                                                             \
        typedef ScalarT ScalarType;                                     \
        typedef ScalarT result_type;                                    \
                                                                        \
        template<typename LhsT, typename RhsT>                          \
        __host__ __device__ ScalarType add(LhsT&& a, RhsT&& b) const    \
        { return SRADD<ScalarType>(std::forward<LhsT>(a),               \
                                   std::forward<RhsT>(b)); }            \
                                                                        \
        template<typename LhsT, typename RhsT>                          \
        __host__ __device__ ScalarType mult(LhsT&& a, RhsT&& b) const   \
        { return SRMULT<ScalarType>(std::forward<LhsT>(a),              \
                                    std::forward<RhsT>(b)); }           \
                                                                        \
        __host__ __device__ ScalarType zero() const                     \
        { return static_cast<ScalarType>(SRZERO); }                     \
                                                                        \
        __host__ __device__ ScalarType one() const                      \
        { return static_cast<ScalarType>(SRONE); }                      \
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
