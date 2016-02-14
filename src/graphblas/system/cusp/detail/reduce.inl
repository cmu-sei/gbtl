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

#include <graphblas/system/cusp/detail/merge.inl>
#include <cusp/multiply.h>
#include <cusp/coo_matrix.h>

namespace graphblas
{
namespace backend
{
    /**
     * @brief Apply a reduction operation to each row of a matrix.
     *
     * @tparam <MatrixT>  Models the Matrix concept
     * @tparam <MonoidT>  Models a binary function
     * @tparam <AccumT>   Models a binary function
     *
     * @param[in]  a  The matrix to perform the row reduction on.
     * @param[out] c  The matrix (column vector) to assign the result to.
     * @param[in]  m  The monoid (binary op) used to reduce the row elements
     * @param[in]  accum   The function for how to assign to destination (e.g.
     *                     Accum to "add" to destination, or Assign to
     *                     "replace" the destimation.
     *
     * @throw DimensionException  If the rows a and c do not match, or the
     *                            the number of columns in c is not one.
     */
    template<typename AMatrixT,
             typename CMatrixT,
             typename MonoidT,
             typename AccumT>
    inline void row_reduce(AMatrixT const &a,
                           CMatrixT       &c, // vector?
                           MonoidT         m,
                           AccumT          accum)
    {

        //assuming cmatrix is of type coo matrix
        CMatrixT temp(a.num_rows, 1, a.num_rows);
        thrust::reduce_by_key(
            a.row_indices.begin(),
            a.row_indices.end(),
            a.values.begin(),
            temp.row_indices.begin(),
            temp.values.begin(),
            thrust::equal_to< IndexType >(),
            m);
        //write col indices:
        thrust::copy_n(
                thrust::make_constant_iterator(0),
                a.num_rows,
                temp.column_indices.begin());
        detail::merge(temp, c, accum);

    }

    /**
     * @brief Apply a reduction operation to each column of a matrix.
     *
     * @tparam <MatrixT>  Models the Matrix concept
     * @tparam <MonoidT>  Models a binary function
     * @tparam <AccumT>   Models a binary function
     *
     * @param[in]  a  The matrix to perform the column reduction on.
     * @param[out] c  The matrix (row vector) to assign the result to.
     * @param[in]  m  The monoid (binary op) used to reduce the column elements
     * @param[in]  accum   The function for how to assign to destination (e.g.
     *                     Accum to "add" to destination, or Assign to
     *                     "replace" the destimation.
     *
     * @throw DimensionException  If the columns a and c do not match, or the
     *                            the number of rows in c is not one.
     */
    template<typename AMatrixT,
             typename CMatrixT,
             typename MonoidT,
             typename AccumT>
    inline void col_reduce(AMatrixT const &a,
                           CMatrixT       &c, // vector?
                           MonoidT         m,
                           AccumT          accum)
    {
        AMatrixT temp1(a);
        CMatrixT temp2(1, a.num_cols, a.num_entries);
        //sort by column:
        thrust::sort_by_key(
                temp1.column_indices.begin(),
                temp1.column_indices.end(),
                thrust::make_zip_iterator(thrust::make_tuple(temp1.row_indices.begin(), temp1.values.begin())));

        thrust::reduce_by_key(
            temp1.column_indices.begin(),
            temp1.column_indices.end(),
            temp1.values.begin(),
            temp2.column_indices.begin(),
            temp2.values.begin(),
            thrust::equal_to< IndexType >(),
            m);
        //write row indices:
        thrust::copy_n(
                thrust::make_constant_iterator(0),
                a.num_cols,
                temp2.row_indices.begin());

        detail::merge(temp2, c, accum);
    }
}
} // graphblas
