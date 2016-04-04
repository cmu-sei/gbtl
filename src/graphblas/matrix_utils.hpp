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

#ifndef GB_MATRIX_UTILS_HPP
#define GB_MATRIX_UTILS_HPP

#include <functional>
#include <vector>

#include <graphblas/graphblas.hpp>

#define __GB_SYSTEM_UTILITY_HEADER <graphblas/system/__GB_SYSTEM_ROOT/utility.hpp>
#include __GB_SYSTEM_UTILITY_HEADER
#undef __GB_SYSTEM_UTILITY_HEADER

namespace graphblas
{

    /**
     * @brief Constuct and return a matrix with elements on the diagonal.
     *
     * @param[in] v    The elements to put on the diagonal of a matrix.
     * @param[in] zero The value of the structural zero.
     */

    //diag reimplementation:
    template<typename MatrixT,
             typename VectorT = std::vector<typename MatrixT::ScalarType> >
    MatrixT diag(VectorT const                 &v,
                 typename MatrixT::ScalarType  zero =
                     static_cast<typename MatrixT::ScalarType>(0))
    {
        MatrixT diag(v.size(), v.size(), zero);
        diag.set_zero(zero);
        //populate diagnals:
        std::vector<IndexType> indices;
        for (graphblas::IndexType ix = 0; ix < v.size(); ++ix) {
            indices.push_back(ix);
        }

        graphblas::buildmatrix(diag, indices.begin(), indices.begin(), v.begin(), v.size());
        return diag;
    }

    /**
     * @brief Construct and retrun an identity matrix of the given size.
     *
     * @param[in] mat_size The size of the identiy matrix to construct.
     * @param[in] zero     The value of the structural zero.
     * @param[in] one      The value to put on the diagonal.
     */
    template<typename MatrixT>
    MatrixT identity(graphblas::IndexType          mat_size,
                     typename MatrixT::ScalarType  zero =
                         static_cast<typename MatrixT::ScalarType>(0),
                     typename MatrixT::ScalarType  one =
                         static_cast<typename MatrixT::ScalarType>(1))
    {
        //init identity:
        MatrixT id(mat_size, mat_size);
        id.set_zero(zero);
        //populate diagnals:
        std::vector<IndexType> x_indices, y_indices;
        std::vector<typename MatrixT::ScalarType> v;

        for (graphblas::IndexType ix = 0; ix < mat_size; ++ix) {
            for (graphblas::IndexType j = 0; j < mat_size; ++j) {
                x_indices.push_back(ix);
                y_indices.push_back(j);
                if (ix==j){
                    v.push_back(one);
                }
                else {
                    v.push_back(zero);
                }
            }
        }

        graphblas::buildmatrix(id, x_indices.begin(),
                               y_indices.begin(), v.begin(),
                               mat_size*mat_size);
        return id;
    }

    /**
     * @brief Construct and retrun an identity matrix of the given size.
     * @TODO:
     * TEMPORARY FIX: sequential, which uses the function above, REQUIRES
     * dense matrix.
     *
     * This method is for a true sparse matrix impl.
     *
     * @param[in] mat_size The size of the identiy matrix to construct.
     * @param[in] zero     The value of the structural zero.
     * @param[in] one      The value to put on the diagonal.
     */
    template<typename MatrixT>
    MatrixT identity_sparse(graphblas::IndexType          mat_size,
                     typename MatrixT::ScalarType  zero =
                         static_cast<typename MatrixT::ScalarType>(0),
                     typename MatrixT::ScalarType  one =
                         static_cast<typename MatrixT::ScalarType>(1))
    {
        //init identity:
        MatrixT id(mat_size, mat_size);
        id.set_zero(zero);
        //populate diagnals:
        std::vector<IndexType> x_indices, y_indices;
        std::vector<typename MatrixT::ScalarType> v;

        for (graphblas::IndexType ix = 0; ix < mat_size; ++ix) {
            for (graphblas::IndexType j = 0; j < mat_size; ++j) {
                if (ix==j){
                    x_indices.push_back(ix);
                    y_indices.push_back(j);
                    v.push_back(one);
                }
            }
        }

        graphblas::buildmatrix(id, x_indices.begin(),
                               y_indices.begin(), v.begin(),
                               mat_size*mat_size);
        return id;
    }



    /**
     * @brief Construct and return a dense matrix with every element equal to
     *        the given value.
     *
     * @param[in] fill_value  The value to fill all entries of the matrix with.
     * @param[in] num_rows    The number of rows in the resulting matrix
     * @param[in] num_cols    The number of columns in the resulting matrix
     *
     * @note Use ConstantMatrix instead for a sparse representation.
     *
     * @todo change order of parameters and add the "zero"
     */
    template<typename MatrixT>
    MatrixT fill(typename MatrixT::ScalarType const &fill_value,
                 graphblas::IndexType                num_rows,
                 graphblas::IndexType                num_cols,
                 typename MatrixT::ScalarType        zero =
                     static_cast<typename MatrixT::ScalarType>(0))
    {
        MatrixT m(num_rows, num_cols, zero);

        for (graphblas::IndexType i = 0; i < num_rows; i++)
        {
            for (graphblas::IndexType j = 0; j < num_cols; j++)
            {
                m.set_value_at(i, j, fill_value);
            }
        }

        return m;
    }

    /**
     * @brief Sum the contents of the matrix
     *
     * @todo  This only works if the matrix "zero" is the "+" identity.
     * @todo  So far all found uses of this function have been replaced with
     *        m.get_nnz() as a loop exit condition.
     */
    template <typename MatrixT>
    typename MatrixT::ScalarType sum(MatrixT const &m)
    {
        using T = typename MatrixT::ScalarType;
        graphblas::IndexType rows, cols;
        m.get_shape(rows, cols);
        T sum = static_cast<T>(0);

        for (graphblas::IndexType i = 0; i < rows; ++i)
        {
            for (graphblas::IndexType j = 0; j < cols; ++j)
            {
                sum += m[i][j];
            }
        }

        return sum;
    }

    /**
     *  @brief Output the matrix in array form.  Mainly for debugging
     *         small matrices.
     *
     *  @param[in] ostr  The output stream to send the contents
     *  @param[in] mat   The matrix to output
     *  @param[in] label Optional label to output first.
     */
    template <typename MatrixT>
    void print_matrix(std::ostream &ostr, MatrixT const &mat,
                      std::string const &label = "")
    {
        ostr << label << ": zero = " << mat.m_mat.get_zero() << std::endl;
        backend::pretty_print_matrix(ostr, mat.m_mat);
    }

    /**
     * @deprecated method for compatibility
     */
    template <typename MatrixT>
    void pretty_print_matrix(std::ostream &ostr, MatrixT const &mat)
    {
        graphblas::print_matrix(ostr,mat, "");
    }
} // graphblas
#endif // GB_MATRIX_UTILS_HPP
