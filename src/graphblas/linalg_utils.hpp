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

#ifndef GB_LINALG_UTILS_HPP
#define GB_LINALG_UTILS_HPP

#include <functional>
#include <vector>

#include <graphblas/operations.hpp>

namespace graphblas
{
    /**
     * @brief Split a matrix into its lower and upper triangular portions
     *
     * @param[in]  A  The matrix to split
     * @param[out] L  The lower triangular portion with the diagonal
     * @param[out] U  The upper triangular portion (no diagonal).
     */
    template<typename MatrixT>
    void split(MatrixT const &A, MatrixT &L, MatrixT &U)
    {
        graphblas::IndexType rows, cols;
        A.get_shape(rows,cols);

        for (graphblas::IndexType i = 0; i < rows; i++)
        {
            for (graphblas::IndexType j = 0; j < cols; j++)
            {
                if (i < j)
                {
                    U.set_value_at(i, j, A.get_value_at(i, j));
                }
                else
                {
                    L.set_value_at(i, j, A.get_value_at(i, j));
                }
            }
        }
    }

    /**
     * @brief Build a half-open range of indices of [start..stop)
     *
     * @param[in] start  The first value in the range
     * @param[in] stop   One past the end of the range
     *
     */
    graphblas::IndexArrayType range(graphblas::IndexType const &start,
                                    graphblas::IndexType const &stop)
    {
        graphblas::IndexArrayType rng;
        for (graphblas::IndexType i = start; i < stop; ++i)
        {
            rng.push_back(i);
        }
        return rng;
    }

    /**
     * @brief Build a half-open range of indices of [0..stop)
     *
     * @param[in] stop  One past the end of the range
     *
     */
    graphblas::IndexArrayType range(graphblas::IndexType const &stop)
    {
        return range(0, stop);
    }

    /**
     * @brief Build a conditional half-open range of indices from [start..stop)
     *
     * @param[in] start The beginning of the range
     * @param[in] stop  One past the end of the range
     * @param[in] cond  function object that tests whether or not each element
     *                  of the specified range should be a member of the
     *                  returned object.
     */
    graphblas::IndexArrayType range(
        graphblas::IndexType const                &start,
        graphblas::IndexType const                &stop,
        std::function<bool(graphblas::IndexType)>  cond)
    {
        graphblas::IndexArrayType rng;
        for (graphblas::IndexType i = start; i < stop; ++i)
        {
            if (cond(i))
            {
                rng.push_back(i);
            }
        }
        return rng;
    }

    /**
     * @brief Build a conditional half-open range of indices from [0..stop)
     *
     * @param[in] stop  One past the end of the range
     * @param[in] cond  function object that tests whether or not each element
     *                  of the specified range should be a member of the
     *                  returned object.
     */
    graphblas::IndexArrayType range(
        graphblas::IndexType const                &stop,
        std::function<bool(graphblas::IndexType)>  cond)
    {
        return range(0, stop, cond);
    }


    /**
     * @brief Calculate the determinant of a matrix.
     *
     * @param[in] m  The matrix to calculate the determinant of.
     */
    template<typename MatrixT>
    typename MatrixT::ScalarType det(MatrixT const &m)
    {
        using T = typename MatrixT::ScalarType;

        int rows, cols;
        m.get_shape(rows, cols);

        T d = m.zero();

        /// @todo Assert square...throw DimensionException
        if ((rows == 1) || (cols == 1))
        {
            d = m[0][0];
        }
        if ((rows == 2) || (cols == 2))
        {
            d = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        }
        else
        {
            MatrixT m_i(rows - 1, cols - 1);
            for (graphblas::IndexType i = 0; i < cols; ++i)
            {
                graphblas::IndexArrayType vect_I = range(1, rows);
                graphblas::IndexArrayType vect_J =
                    range(m, [i](graphblas::IndexType j){return j != i;});
                graphblas::extract(m, vect_I, vect_J, m_i);
                d += m[0][i] * det(m_i) *
                    graphblas::math::power<T>(
                        -1,
                        std::forward<graphblas::IndexType>(i));
            }
        }
        return d;
    }


    /**
     * @brief Calculate the cofactor of a matrix.
     *
     * @param[in]  m  The matrix to calculate the cofactor of.
     *
     * @todo It seems this could be implemented in parallel somehow...
     */
    template<typename MatrixT>
    MatrixT cofactor(MatrixT const &m)
    {
        using T = typename MatrixT::ScalarType;

        graphblas::IndexType num_rows, num_cols;
        m.get_shape(num_rows, num_cols);

        MatrixT C(num_rows, num_cols, m.get_zero());
        MatrixT C_ij(num_rows - 1, num_cols - 1);

        for (graphblas::IndexType i = 0; i < num_rows; ++i)
        {
            for (graphblas::IndexType j = 0; j < num_cols; ++j)
            {
                graphblas::IndexArrayType vect_I =
                    range(num_rows,
                          [i](graphblas::IndexType k){return k != i;});
                graphblas::IndexArrayType vect_J =
                    range(num_cols,
                          [j](graphblas::IndexType l){return l != j;});
                graphblas::extract(m, vect_I, vect_J, C_ij);
                C[i][j] = graphblas::math::power<T>(-1, i + j) *
                    det<MatrixT>(C_ij);
            }
        }

        return C;
    }

    /**
     * @brief Calculate the adjoint of a matrix.
     *
     * @param[in] m  The matrix to calculate the adjoint of.
     */
    template<typename MatrixT>
    MatrixT adjoint(MatrixT const &m)
    {
        using T = typename MatrixT::ScalarType;
        MatrixT C = cofactor<MatrixT>(m);

        graphblas::IndexType rows, cols;
        C.get_shape(rows, cols);
        MatrixT adj(rows, cols);

        graphblas::transpose(m, adj);
        return adj;
    }

    /**
     * @brief Calculate the inverse of a matrix.
     *
     * @param[in] m  The matrix to calculate the inverse of
     *
     * @todo Probably need to deal with sparse-to-dense issues.
     */
    template<typename MatrixT>
    MatrixT inverse(const MatrixT &m)
    {
        using T = typename MatrixT::ScalarType;

        graphblas::IndexType rows, cols;
        m.get_shape(rows, cols);

        MatrixT adj = adjoint(m);
        T det = det(m);

        MatrixT det_mat(rows, cols, det);  // zero := determinant?

        MatrixT m_inv(rows, cols);
        graphblas::ewisemult(adj, det_mat, m_inv, graphblas::math::div<T>);

        return m_inv;
    }
} // graphblas
#endif // GB_LINALG_UTILS_HPP
