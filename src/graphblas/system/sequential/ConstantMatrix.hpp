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

#ifndef GB_SEQUENTIAL_CONSTANTMATRIX_HPP
#define GB_SEQUENTIAL_CONSTANTMATRIX_HPP

#include <iostream>
#include <vector>

#include <graphblas/system/sequential/RowView.hpp>
#include <graphblas/system/sequential/Matrix.hpp>

namespace graphblas
{
    /// @todo Should we support assignment by row?

    /**
     * @brief Class representing a list of lists format sparse matrix.
     */
    template<typename ScalarT>
    class ConstantMatrix
    {
    public:
        typedef ScalarT ScalarType;
        //WARNING: this breaks the sequential backend where this class is used.
        ConstantMatrix<ScalarT> &m_mat;  //graphblas::backend::Matrix<ScalarT> m_mat;

        /**
         * @brief Construct a matrix whose entries are all same value
         *        with the given shape.
         *
         * @param[in] num_rows  Number of rows in the matrix
         * @param[in] num_cols  Number of cols in the matrix
         * @param[in] value     The value to "fill" the matrix with.
         * @param[in] zero      The "zero" value (not used).
         */
        ConstantMatrix(IndexType      num_rows,
                       IndexType      num_cols,
                       ScalarT const &value,
                       ScalarT const &zero = static_cast<ScalarT>(0))
            : m_num_rows(num_rows),
              m_num_cols(num_cols),
              m_value(value),
              m_zero(zero),
              m_mat(*this) //m_mat(num_rows, num_cols, zero)
        {
        }

        /**
         * @brief Copy constructor for ConstantMatrix.
         *
         * @param[in] rhs  The ConstantMatrix to copy construct this
         *                 ConstantMatrix from.
         */
        ConstantMatrix(ConstantMatrix<ScalarT> const &rhs)
            : m_mat(*this)
        {
            if (this != &rhs)
            {
                m_num_rows = rhs.m_num_rows;
                m_num_cols = rhs.m_num_cols;
                m_value = rhs.m_value;
                m_zero = rhs.m_zero;
                //m_mat = rhs.m_mat;
            }
        }

        /**
         * @brief Copy assignment.
         *
         * @param[in] rhs  The ConstantMatrix to assign to this one
         *
         * @return *this.
         */
        ConstantMatrix<ScalarT>& operator=(
            ConstantMatrix<ScalarT> const &rhs)
        {
            if (this != &rhs)
            {
                m_num_rows = rhs.m_num_rows;
                m_num_cols = rhs.m_num_cols;
                m_value = rhs.m_value;
                m_zero = rhs.m_zero;
                //m_mat = rhs.m_mat;
            }
            return *this;
        }

        ~ConstantMatrix()
        {
        }

        // FUNCTIONS

        /**
         * @brief Get the shape for this ConstantMatrix.
         * @return num_rows and num_cols.
         */
        void get_shape(IndexType &num_rows, IndexType &num_cols) const
        {
            num_rows = m_num_rows;
            num_cols = m_num_cols;
        }

        /**
         * @brief Get the zero element for this matrix.
         *
         * @return  The structural zero value.
         * @note This matrix does not have a structural zero really
         */
        ScalarT get_zero() const
        {
            return m_zero;
        }

        /**
         * @brief Set the value of a structural zero element.
         *
         * @param[in] new_zero  The new zero value.
         *
         * @return The old zero element for this matrix.
         * @note This matrix does not have a structural zero really
         */
        ScalarT set_zero(ScalarT new_zero)
        {
            ScalarT old_zero = m_zero;
            m_zero = new_zero;
            return old_zero;
        }

        /// @return the number of stored values in this
        IndexType get_nnz() const
        {
            IndexType count(0UL);
            if (m_value == m_zero)
            {
                count = m_num_rows * m_num_cols;
            }
            return count;
        }

        // EQUALITY OPERATORS
        /**
         * @brief Equality testing for ConstantMatrix.
         * @param rhs The right hand side of the equality operation.
         * @return If this ConstantMatrix and rhs are identical.
         */
        bool operator==(ConstantMatrix<ScalarT> const &rhs) const
        {
            /// @todo test m_zero?
            if ((m_num_rows != rhs.m_num_rows) ||
                (m_num_cols != rhs.m_num_cols) ||
                (m_value != rhs.m_value))
            {
                return false;
            }

            return true;
        }

        /**
         * @brief Inequality testing for ConstantMatrix.
         * @param rhs The right hand side of the inequality operation.
         * @return If this ConstantMatrix and rhs are not identical.
         */
        bool operator!=(ConstantMatrix<ScalarT> const &rhs) const
        {
            return !(*this == rhs);
        }

        /**
         * @brief Access the elements of this ConstantMatrix given
         *        row and column indexes.
         *
         * @param[in] row_index  The row to access.
         * @param[in] col_index  The column to access.
         *
         * @return The element of this ConstantMatrix at the given
         *         row and column.
         */
        ScalarT get_value_at(IndexType row_index,
                             IndexType col_index) const
        {
            /// @todo Assert indices within ranges?
            return m_value;
        }

        // Not certain about this implementation
        void set_value_at(IndexType      row,
                          IndexType      col,
                          ScalarT const &new_val)
        {
            /// @todo Assert indices within ranges?
            m_value = new_val;
        }

        /**
         * @brief Indexing function for accessing the rows of this
         *        ConstantMatrix.
         *
         * @param[in] row  The row to access.
         *
         * @return The row of this ConstantMatrix.
         */
        RowView<ConstantMatrix<ScalarT> const>
        get_row(IndexType row) const
        {
            return RowView<ConstantMatrix<ScalarT> const>(row, *this);
        }

        RowView<ConstantMatrix<ScalarT> >
        get_row(IndexType row)
        {
            return RowView<ConstantMatrix<ScalarT> >(row, *this);
        }

        /**
         * @brief Indexing operator for accessing the rows of this
         *        ConstantMatrix.
         *
         * @param[in] row  The row to access.
         * @return The row of this ConstantMatrix
         */
        RowView<ConstantMatrix<ScalarT> const>
        operator[](IndexType row) const
        {
            return RowView<ConstantMatrix<ScalarT> const>(row, *this);
        }

        // need a non-const version for mutation.
        RowView<ConstantMatrix<ScalarT> >
        operator[](IndexType row)
        {
            return RowView<ConstantMatrix<ScalarT> >(row, *this);
        }

        /// @todo output specific to the storage layout of this type
        friend std::ostream& operator<<(
            std::ostream             &os,
            ConstantMatrix<ScalarT> const &lil)
        {
            for (IndexType row = 0; row < lil.m_num_rows; ++row)
            {
                os << ((row == 0) ? "[[" : " [");
                if (lil.m_num_cols > 0)
                {
                    os << lil.get_value_at(row, 0);
                }

                for (IndexType col = 1; col < lil.m_num_cols; ++col)
                {
                    os << ", " << lil.get_value_at(row, col);
                }
                os << ((row == lil.m_num_rows - 1) ? "]]" : "]\n");
            }
            return os;
        }

    private:
        IndexType m_num_rows;
        IndexType m_num_cols;
        ScalarT   m_zero;
        ScalarT   m_value;
    };
} // graphblas

#endif // GB_SEQUENTIAL_CONSTANTMATRIX_HPP
