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

#include <thrust/functional.h>
#include <thrust/transform.h>
#include <graphblas/system/cusp/detail/merge.inl>
#include <cusp/elementwise.h>

namespace graphblas
{
namespace backend
{
    namespace detail
    {
        template <typename UnaryFunctionT, typename T>
        struct u2bfunc : public thrust::binary_function<T, T, T>
        {
            u2bfunc(UnaryFunctionT f)
                : m_f(f) {}

            __device__ __host__
            T operator()(T x, T y) { return m_f(x); }

        private:
            UnaryFunctionT m_f;
        };
    }
    // TODO In the GraphBLAS math document, they say that the result of apply
    //      should be accumulated into c, however we have not been accumulating
    //      at all in the Python implementation, we have just been returning the
    //      result of the elementwise apply of func on a.

    template<typename AMatrixT,
             typename CMatrixT,
             typename UnaryFunctionT,
             typename AccumT>
    inline void apply(AMatrixT const &a,
                      CMatrixT       &c,
                      UnaryFunctionT  func,
                      AccumT          accum)
    {
        using namespace graphblas::detail;

        AMatrixT temp(a);
        thrust::transform(temp.values.begin(), temp.values.end(), temp.values.begin(), func);
        detail::merge(temp, c, accum);

        // using T = typename AMatrixT::ScalarType;
        // u2bfunc<UnaryFunctionT, T> f(func);
        // IndexType M, N;
        // a.get_shape(M, N);
        // AMatrixT zeros(M, N);
        // //call cusp elementwise:
        // ::cusp::elementwise(a, zeros, c, f);

        /*
         * FIXME Check to make sure accumulate works.
         * TODO Remove runtime check, and replace with a specialization of this
         *      template.
         */
        // if (typeid(AccumT) !=
        //     typeid(graphblas::math::Assign<typename AMatrixT::value_type>))
        // {
        //     // TODO: MZ: commented this line as for now as tmp is not defined.
        //     // ewiseapply<AMatrixT, CMatrixT, CMatrixT, AccumT>(tmp, c, c, accum);
        // }
        // else
        // {
            // TODO: MZ: commented this line as for now as tmp is not defined.
            // c = tmp;
    }
} // graphblas
}
