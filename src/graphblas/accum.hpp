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

#include <functional>
#include <utility>

#ifndef GB_ACCUM_HPP
#define GB_ACCUM_HPP

#include <graphblas/algebra.hpp>

namespace XXXgraphblas
{
    namespace math
    {
        /**
         * @brief  The functor to perform assignment into the destination
         *
         * @note  When using this functor the destination element is passed
         *        to the lhs parameter.
         */
        template <typename ScalarT>
        struct Assign
        {
            /// @todo We don't need both
            __host__ __device__ ScalarT accum(ScalarT const &lhs,
                                              ScalarT const &rhs)
            {
                return rhs;
            }

            __host__ __device__ ScalarT operator()(ScalarT const &lhs,
                                                   ScalarT const &rhs)
            {
                return accum(lhs, rhs);
            }
        };

        /**
         * @brief  The functor to perform accumulation with/into the destination
         *
         * @note  When using this functor the destination element is passed
         *        to the lhs parameter to be consistent with assign.
         */
        template<typename ScalarT,
                 typename MonoidT = Plus<ScalarT> >
        struct Accum
        {
            __host__ __device__ ScalarT accum(ScalarT const &lhs,
                                              ScalarT const &rhs,
                                              MonoidT a = MonoidT())
            {
                return a(lhs, rhs);
            }

            __host__ __device__ ScalarT operator()(ScalarT const &lhs,
                                                   ScalarT const &rhs)
            {
                return accum(lhs, rhs);
            }
        };
    } // math
} // graphblas

#endif // ACCUM_HPP
