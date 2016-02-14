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

#ifndef GB_SEQUENTIAL_FUNCTION_VIEW_HPP
#define GB_SEQUENTIAL_FUNCTION_VIEW_HPP

#include <graphblas/system/sequential/RowView.hpp>

namespace graphblas
{
    /**
     * @brief View a matrix as filtered through a user-defined function.
     *
     * @tparam MatrixT         Implements the 2D matrix concept.
     * @tparam UnaryFunctionT  Implements the unary function concept
     *                         that takes the MatrixT::ScalarType and
     *                         returns the same.
     */
    template<typename MatrixT,
             typename UnaryFunctionT>
    class FunctionView
    {
    public:
        typedef typename MatrixT::ScalarType ScalarType;


        FunctionView(MatrixT &matrix, UnaryFunctionT func):
            m_matrix(matrix), m_func(func)
        { }

        FunctionView(FunctionView<MatrixT, UnaryFunctionT> const &rhs)
            : m_matrix(rhs.m_matrix),
              m_func(rhs.m_func)
        {
        }

        /**
         * Copy assignment.
         *
         * @param[in] rhs The function view to assign to this.
         *
         * @return *this.
         */
        FunctionView<MatrixT, UnaryFunctionT>&
        operator=(FunctionView<MatrixT, UnaryFunctionT> const &rhs)
        {
            if (this != &rhs)
            {
                m_matrix = rhs.m_matrix;
                m_func   = rhs.m_func;
            }
            return *this;
        }

        // DESTRUCTOR

        ~FunctionView()
        { }

        // FUNCTIONS

        /**
         * @brief Equality testing for matrix. (value equality?)
         *
         * @param[in] rhs  The right hand side of the equality
         *                 operation.
         *
         * @return True, if this matrix and rhs are identical.
         */
        bool
        operator==(
            FunctionView<MatrixT, UnaryFunctionT> const &rhs) const
        {
            return m_matrix == rhs.m_matrix;
        }


        /**
         * @brief Inequality testing for matrix.
         *
         * @param [in] rhs  The right hand side of the inequality
         *                  operation.
         *
         * @return true, if this matrix and rhs are not identical.
         */
        bool
        operator!=(
            FunctionView<MatrixT, UnaryFunctionT> const &rhs) const
        {
            return m_matrix != rhs.m_matrix;
        }


        /**
         * @brief Get the shape for this matrix.
         *
         * @return A tuple containing the shape in the form (M, N),
         *         where M is the number of rows, and N is the number
         *         of columns.
         */
        void get_shape(IndexType &nr, IndexType &nc) const
        {
            return m_matrix.get_shape(nr, nc);
        }

        /**
         * @brief Get the zero element for this matrix.
         *
         * @return The zero element for this matrix.
         */
        ScalarType get_zero() const
        {
            return m_matrix.get_zero();
        }

        /**
         * @brief Set the zero element for this matrix.
         *
         * @param[in] new_zero  The new zero element for this matrix.
         *
         * @return The old zero element for this matrix.
         */
        ScalarType set_zero(ScalarType new_zero)
        {
            return m_matrix.set_zero(new_zero);
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
         * @return The element at the given row and column.
         */
        ScalarType
        get_value_at(IndexType row, IndexType col) const
        {
            return m_func(m_matrix.get_value_at(row, col));
        }


        /**
         * @brief Indexing function for accessing the rows of this
         *        matrix.
         *
         * @param[in] row  The index of the row to access.
         *
         * @return The view of a row of this matrix.
         */
        RowView<FunctionView<MatrixT, UnaryFunctionT> >
        get_row(IndexType row) const
        {
            return RowView<FunctionView<MatrixT, UnaryFunctionT> >(
                row, *this);
        }

        /**
         * @brief Indexing operator for accessing the rows of this
         *        matrix.
         *
         * @param[in] row  The index of the row to access.
         *
         * @return The view of a row of this matrix.
         */
        RowView<FunctionView<MatrixT, UnaryFunctionT> >
        operator[](IndexType row) const
        {
            return RowView<FunctionView<MatrixT, UnaryFunctionT> >(
                row, *this);
        }

        friend std::ostream&
        operator<<(std::ostream                                &os,
                   FunctionView<MatrixT, UnaryFunctionT> const &fun)
        {
            IndexType M, N;
            fun.get_shape(M, N);
            os << "[";
            for (IndexType row = 0; row < M; ++row)
            {
                os << "[ ";
                for (IndexType col = 0; col < N - 1; ++col)
                {
                    os << fun[row][col] << ", ";
                }
                os << fun[row][N - 1] << " ]" << std::endl;
            }
            os << "]";
            return os;
        }

    private:
        MatrixT         &m_matrix;
        UnaryFunctionT   m_func;
    };
} // graphblas

#endif // GB_SEQUENTIAL_FUNCTION_VIEW_HPP
