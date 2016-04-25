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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/temporary_array.h>
#include <cusp/elementwise.h>
#include <cusp/print.h>

#include <graphblas/system/cusp/detail/merge.inl>
#include <graphblas/detail/config.hpp>

namespace graphblas
{
namespace backend
{
        template<typename AMatrixT,
                 typename BMatrixT,
                 typename CMatrixT,
                 typename MonoidT,
                 typename AccumT>
        inline void ewiseapply(const AMatrixT &a,
                               const BMatrixT &b,
                               CMatrixT       &c,
                               MonoidT         monoid,
                               AccumT          accum)
        {
          //as discussed last time, the shape checks should be done on the front
          //end, correct?

            if (a.num_rows != b.num_rows ||
                b.num_cols != a.num_cols)
            {
              return;
            }

            // Create a temporary matrix for return.
            CMatrixT temp_matrix(a.num_rows, a.num_cols, 0);
            temp_matrix.resize(a.num_rows, a.num_cols, a.num_entries + b.num_entries);

            //call cusp elementwise:
            ::cusp::elementwise(a, b, temp_matrix, monoid);

            //merging results:
            detail::merge(temp_matrix, c, accum);
        }

        // template<typename AMatrixT,
        //          typename BMatrixT,
        //          typename CMatrixT,
        //          typename MonoidT>
        // inline void ewiseapply(const AMatrixT &a,
        //                        const BMatrixT &b,
        //                        CMatrixT       &c,
        //                        MonoidT         monoid)
        // {
        //     //why do we have this function AND the function above?
        //     // Create a temporary matrix for return.
        //     if (a.num_rows != b.num_rows ||
        //             b.num_cols != a.num_cols)
        //     {
        //         return;
        //     }

        //     //just write to C, since we don't care anyway.
        //     c.resize(a.num_rows, a.num_cols, a.num_entries + b.num_entries);

        //     //call cusp elementwise:
        //     ::cusp::elementwise(a, b, c, monoid);
        // }

        /*
         * FIXME Is this the way it should be implemented based on the
         * GraphBLAS API discussions?
         */

        template<typename AMatrixT,
                 typename BMatrixT,
                 typename CMatrixT,
                 typename MonoidT,
                 typename AccumT>
        inline void ewiseadd(const AMatrixT &a,
                             const BMatrixT &b,
                             CMatrixT       &c,
                             MonoidT         m,
                             AccumT          accum)
        {
            // FIXME Implement ewiseapply optimized for addition here.
            //return ewiseapply<AMatrixT, BMatrixT, CMatrixT, MonoidT, AccumT>
            return ewiseapply
                (a, b, c, m, accum);
        }

        template<typename AMatrixT,
                 typename BMatrixT,
                 typename CMatrixT,
                 typename MonoidT,
                 typename AccumT>
        inline void ewisemult(const AMatrixT &a,
                              const BMatrixT &b,
                              CMatrixT       &c,
                              MonoidT         m,
                              AccumT          accum)
        {
            graphblas::IndexType num_rows=a.num_rows;
            graphblas::IndexType num_cols=a.num_cols;


            auto min_size = a.num_entries < b.num_entries ? a.num_entries : b.num_entries;
            cusp::array1d <IndexType, cusp::device_memory> a_indices(min_size), b_indices(min_size);
            auto intersection_end = thrust::set_intersection_by_key(
                    thrust::make_zip_iterator(thrust::make_tuple(a.row_indices.begin(), a.column_indices.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(a.row_indices.end(), a.column_indices.end())),
                    thrust::make_zip_iterator(thrust::make_tuple(b.row_indices.begin(), b.column_indices.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(b.row_indices.end(), b.column_indices.end())),
                    thrust::make_counting_iterator(0),
                    thrust::make_discard_iterator(),
                    a_indices.begin());

            auto intersection_size = thrust::distance(a_indices.begin(), intersection_end.second);

            thrust::set_intersection_by_key(
                    thrust::make_zip_iterator(thrust::make_tuple(b.row_indices.begin(), b.column_indices.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(b.row_indices.end(), b.column_indices.end())),
                    thrust::make_zip_iterator(thrust::make_tuple(a.row_indices.begin(), a.column_indices.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(a.row_indices.end(), a.column_indices.end())),
                    thrust::make_counting_iterator(0),
                    thrust::make_discard_iterator(),
                    b_indices.begin());
            //try transform:
            auto a_values_begin = thrust::make_permutation_iterator(
                    a.values.begin(),
                    a_indices.begin());

            auto a_values_end = thrust::make_permutation_iterator(
                    a.values.begin(),
                    a_indices.begin() + intersection_size);

            auto b_values_begin = thrust::make_permutation_iterator(
                    b.values.begin(),
                    b_indices.begin());

            CMatrixT temp(num_rows, num_cols, intersection_size);

            thrust::transform(
                    a_values_begin,
                    a_values_end,
                    b_values_begin,
                    temp.values.begin(),
                    m);
            //copy indices:
            thrust::copy_n(
                    thrust::make_permutation_iterator(
                        a.row_indices.begin(),
                        a_indices.begin()),
                    intersection_size,
                    temp.row_indices.begin());

            thrust::copy_n(
                    thrust::make_permutation_iterator(
                        a.column_indices.begin(),
                        a_indices.begin()),
                    intersection_size,
                    temp.column_indices.begin());
            c = temp;
        }
}
} // graphblas
