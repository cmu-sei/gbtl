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

    template<typename AMatrixT,
             typename CMatrixT,
             typename UnaryFunctionT,
             typename AccumT>
    inline void apply(AMatrixT const &a,
                      CMatrixT       &c,
                      UnaryFunctionT  func,
                      AccumT          accum)
    {
        AMatrixT temp(a);
        thrust::transform(temp.values, temp.values+temp.num_entries, temp.values, func);
    }
} // graphblas
}
