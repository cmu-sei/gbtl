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

#ifndef GB_SEQUENTIAL_UTILITY_HPP
#define GB_SEQUENTIAL_UTILITY_HPP

#include <limits>

namespace GraphBLAS
{
    namespace backend
    {
        /**
         *  @brief Output the matrix in array form.  Mainly for debugging
         *         small matrices.
         *
         *  @param[in] ostr  The output stream to send the contents
         *  @param[in] mat   The matrix to output
         *
         */
        template <typename MatrixT >
        void pretty_print_matrix(std::ostream &ostr, MatrixT const &mat)
        {
//          using scalar_type = typename MatrixT::ScalarT;
            typedef typename MatrixT::ScalarType ScalarT;
            typedef std::vector<std::tuple<GraphBLAS::IndexType, ScalarT> > RowType;

            IndexType num_rows = mat.nrows();
            IndexType num_cols = mat.ncols();

            for (IndexType row_idx = 0; row_idx < num_rows; ++row_idx)
            {
                ostr << ((row_idx == 0) ? "[[" : " [");

                RowType const &row(mat.getRow(row_idx));
                IndexType curr_idx = 0;

                if (row.empty())
                {
                    while (curr_idx < num_cols)
                    {
                        ostr << ((curr_idx == 0) ? " " : ",  " );
                        ++curr_idx;
                    }
                }
                else
                {
                    // Now walk the columns.  A sparse iter would be handy here...
                    IndexType col_idx;
                    ScalarT cell_val;

                    auto row_it = row.begin();
                    while (row_it != row.end())
                    {
                        std::tie(col_idx, cell_val) = *row_it;
                        while (curr_idx < col_idx)
                        {
                            ostr << ((curr_idx == 0) ? " " : ",  " );
                            ++curr_idx;
                        }

                        if (curr_idx != 0)
                            ostr << ", ";
                        ostr << cell_val;

                        ++row_it;
                        ++curr_idx;
                    }

                    // Fill in the rest to the end
                    while (curr_idx < num_cols)
                    {
                        ostr << ",  ";
                        ++curr_idx;
                    }

                }
                ostr << ((row_idx == num_rows - 1 ) ? "]]\n" : "]\n");
            }

            // IndexType rows, cols;
            // mat.get_shape(rows, cols);
            // typename MatrixT::ScalarType zero(mat.get_zero());
            // for (IndexType row = 0; row < rows; ++row)
            // {
            //     ostr << ((row == 0) ? "[[" : " [");
            //     if (cols > 0)
            //     {
            //         auto val = mat.extractElement(row, 0);
            //         if (val == zero)
            //             ostr << " ";
            //         else
            //             ostr << val;
            //     }

            //     for (IndexType col = 1; col < cols; ++col)
            //     {
            //         auto val = mat.extractElement(row, col);
            //         if (val == zero)
            //             ostr << ",  ";
            //         else
            //             ostr << ", " << val;
            //     }
            //     ostr << ((row == rows - 1) ? "]]\n" : "]\n");
            // }
        }

    } //backend
}

//****************************************************************************

namespace graphblas
{
    namespace backend_template_library = std;

    namespace backend
    {
    //************************************************************************
    template <typename MatrixA, typename MatrixB>
    void index_of(MatrixA const        &mat,
                  MatrixB              &indexed_of_mat,
                  graphblas::IndexType  base_index)
    {
        graphblas::IndexType rows, cols;
        mat.get_shape(rows, cols);

        using T = typename MatrixA::ScalarType;

        for (IndexType i = 0; i < rows; ++i)
        {
            for (IndexType j = 0; j < cols; ++j)
            {
                auto mat_ij = mat.extractElement(i, j);
                if (mat_ij > 0 || mat_ij == std::numeric_limits<T>::max())
                {
                    indexed_of_mat.setElement(i, j, i + base_index);
                }
                else
                {
                    // FIXME indexed_of_mat.get_zero()?
                    indexed_of_mat.setElement(i, j, mat.get_zero());
                }
            }
        }
    }

    //************************************************************************
    template<typename MatrixT>
    void col_index_of(MatrixT &mat)
    {
        graphblas::IndexType rows, cols;
        mat.get_shape(rows, cols);

        for (IndexType i = 0; i < rows; ++i)
        {
            for (IndexType j = 0; j < cols; ++j)
            {
                auto mat_ij = mat.extractElement(i, j);
                if (mat_ij != mat.get_zero())
                {
                    mat.setElement(i, j, j);
                }
            }
        }
    }

    //************************************************************************
    template<typename MatrixT>
    void row_index_of(MatrixT &mat)
    {
        graphblas::IndexType rows, cols;
        mat.get_shape(rows, cols);

        for (IndexType i = 0; i < rows; ++i)
        {
            for (IndexType j = 0; j < cols; ++j)
            {
                auto mat_ij = mat.extractElement(i, j);
                if (mat_ij != mat.get_zero())
                {
                    mat.setElement(i, j, i);
                }
            }
        }
    }

    /**
     *  @brief Output the matrix in array form.  Mainly for debugging
     *         small matrices.
     *
     *  @param[in] ostr  The output stream to send the contents
     *  @param[in] mat   The matrix to output
     *
     */
    template <typename MatrixT>
    void pretty_print_matrix(std::ostream &ostr, MatrixT const &mat)
    {
        IndexType rows, cols;
        mat.get_shape(rows, cols);
        typename MatrixT::ScalarType zero(mat.get_zero());

        for (IndexType row = 0; row < rows; ++row)
        {
            ostr << ((row == 0) ? "[[" : " [");
            if (cols > 0)
            {
                auto val = mat.extractElement(row, 0);
                if (val == zero)
                    ostr << " ";
                else
                    ostr << val;
            }

            for (IndexType col = 1; col < cols; ++col)
            {
                auto val = mat.extractElement(row, col);
                if (val == zero)
                    ostr << ",  ";
                else
                    ostr << ", " << val;
            }
            ostr << ((row == rows - 1) ? "]]\n" : "]\n");
        }
    }

} //backend
} // graphblas

#endif // GB_SEQUENTIAL_UTILITY_HPP
