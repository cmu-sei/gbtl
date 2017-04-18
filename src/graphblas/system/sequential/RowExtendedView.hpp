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

#ifndef GB_SEQUENTIAL_ROW_EXTENDED_VIEW_HPP
#define GB_SEQUENTIAL_ROW_EXTENDED_VIEW_HPP

#include <graphblas/exceptions.hpp>
#include <graphblas/system/sequential/Matrix.hpp>

namespace graphblas
{
    /**
     * @brief View of a singluar expansion of a column vector into a
     *        matrix.
     *
     * @tparam MatrixT   Implements a matrix
     *
     * @todo In the future this should implement a pure vector
     */
    template<typename MatrixT>
    class RowExtendedView
    {
    public:
        typedef typename MatrixT::ScalarType ScalarType;
        //WARNING: this breaks the sequential backend where this class is used.
        RowExtendedView<MatrixT> &m_mat;

        // CONSTRUCTORS
        /**
         *  @param[in] col_index Which column to extend
         *  @param[in] matrix    Matrix from which to pull a column
         *  @param[in] num_cols  Number of columns to repeat the view
         */
        RowExtendedView(IndexType  col_index,
                        MatrixT   &matrix,
                        IndexType  num_cols) :
            m_col_index(col_index),
            m_matrix(matrix),
            m_num_cols(num_cols),
            m_mat(*this)
        {
            IndexType nc;
            m_matrix.get_shape(m_num_rows, nc);
            if (col_index >= nc)
            {
                throw DimensionException();
            }
        }

        /**
         * Copy constructor.
         *
         * @param[in] rhs The row extended view to copy.
         *
         * @todo Is this const correct?
         */
        RowExtendedView(RowExtendedView<MatrixT> const &rhs)
            : m_col_index(rhs.m_col_index),
              m_matrix(rhs.m_matrix),
              m_num_rows(rhs.m_num_rows),
              m_num_cols(rhs.m_num_cols),
              m_mat(rhs.m_mat)
        {
        }

        ~RowExtendedView()
        {
        }

        /**
         * @brief Get the shape for this matrix.
         *
         * @return  A tuple containing the shape in the form (M, N),
         *          where M is the number of rows, and N is the number
         *          of columns.
         */
        void get_shape(IndexType &num_rows, IndexType &num_cols) const
        {
            num_cols = m_num_cols;
            num_rows = m_num_rows;
        }

        //version 2 of getshape:
        //returns a pair
        std::pair<IndexType, IndexType> get_shape() const
        {
            return std::make_pair(m_num_rows, m_num_cols);
        }


        /**
         * @brief Get the value of a structural zero element.
         *
         * @return  The structural zero value.
         */
        ScalarType get_zero() const
        {
            return m_matrix.get_zero();
        }


        /**
         * @brief Set the value of a structural zero element.
         *
         * @param[in] new_zero  The new zero value.
         *
         * @return The old zero element for this matrix.
         */
        ScalarType set_zero(ScalarType new_zero)
        {
            return m_matrix.set_zero(new_zero);
        }


        // EQUALITY OPERATORS
        /**
         * @brief Equality testing for matrix. (value equality?)
         *
         * @param[in] rhs  The right hand side of the equality
         *                 operation.
         *
         * @return true, if this matrix and rhs are identical.
         * @todo  Not sure we need this form.  Should we do equality
         *        with any matrix?
         */
        template <typename OtherMatrixT>
        bool operator==(OtherMatrixT const &rhs) const
        {
            IndexType rhs_nr, rhs_nc;
            rhs.get_shape(rhs_nr, rhs_nc);
            if ((m_num_rows != rhs_nr) || (m_num_cols != rhs_nc))
            {
                return false;
            }

            for (IndexType i = 0; i < m_num_rows; ++i)
            {
                for (IndexType j = 0; j < m_num_cols; ++j)
                {
                    if (extractElement(i, j) != rhs.extractElement(i, j))
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        /**
         * @brief Access the elements given row and column indexes.
         *
         * Function provided to access the elements given row and
         * column indices.  The functionality is the same as that
         * of the indexing function for a standard dense matrix.
         *
         * @param[in] row  The index of row to access.
         * @param[in] col  The index of column to access.
         *
         * @return The element at the given row and column of the
         *         extended view.
         */
        ScalarType extractElement(IndexType row, IndexType col) const
        {
            //assert(col < m_num_cols); /// @todo throw or ignore?
            return m_matrix.extractElement(row, m_col_index);
        }


        // Not certain about this implementation
        void setElement(IndexType         row,
                          IndexType         col,
                          ScalarType const &val)
        {
            //assert(col < m_num_cols); /// @todo throw or ignore?
            m_matrix.setElement(row, m_col_index, val);
        }

        friend std::ostream&
        operator<<(std::ostream                   &os,
                   RowExtendedView<MatrixT> const &mat)
        {
            IndexType num_rows, num_cols;
            mat.get_shape(num_rows, num_cols);
            for (IndexType row = 0; row < num_rows; ++row)
            {
                os << ((row == 0) ? "[[" : " [");
                if (num_cols > 0)
                {
                    os << mat.extractElement(row, 0);
                }

                for (IndexType col = 1; col < num_cols; ++col)
                {
                    os << ", " << mat.extractElement(row, col);
                }
                os << ((row == num_rows - 1) ? "]]" : "]\n");
            }
            return os;
        }

    private:

        /**
         * Copy assignment not implemented.
         *
         * @note  You cannot reassign a reference
         */
        RowExtendedView<MatrixT>&
        operator=(RowExtendedView<MatrixT> const &rhs);

    private:
        IndexType  m_col_index;
        MatrixT   &m_matrix;
        IndexType  m_num_rows;
        IndexType  m_num_cols;
    };

} // graphblas

#endif // GB_SEQUENTIAL_ROW_EXTENDED_VIEW_HPP
