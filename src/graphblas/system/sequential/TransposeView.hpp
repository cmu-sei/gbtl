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


#ifndef GB_SEQUENTIAL_TRANSPOSE_VIEW_HPP
#define GB_SEQUENTIAL_TRANSPOSE_VIEW_HPP

//#include <graphblas/system/sequential/ColumnView.hpp>
#include <graphblas/system/sequential/Matrix.hpp>

namespace GraphBLAS
{
namespace backend
{
    /**
     * @brief View a matrix as if it were transposed.
     *
     * @tparam MatrixT         Implements the 2D matrix concept.
     */
    template<typename MatrixT>
    class TransposeView
    {
    public:
        typedef typename MatrixT::ScalarType ScalarType;

        // CONSTRUCTORS

        TransposeView(MatrixT const &matrix):
            m_matrix(matrix)
        {
        }

        /**
         * Copy constructor.
         *
         * @param[in] rhs The transpose view to copy.
         *
         * @todo Is this const correct?
         */
        TransposeView(TransposeView<MatrixT> const &rhs)
            : m_matrix(rhs.m_matrix)
        {
        }

        ~TransposeView()
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
            m_matrix.get_shape(num_cols, num_rows);
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

        IndexType get_nnz() const { return m_matrix.get_nnz(); }

        // EQUALITY OPERATORS
        //bool
        //operator==(TransposeView<MatrixT> const &rhs) const
        //{
        //    return m_matrix == rhs.m_matrix;
        //}

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
            IndexType nr, nc, rhs_nr, rhs_nc;
            get_shape(nr, nc);
            rhs.get_shape(rhs_nr, rhs_nc);
            if ((nr != rhs_nr) || (nc != rhs_nc))
            {
                return false;
            }

            // Definitely a more efficient way than this.  Only compare
            // non-zero elements.  Then decide if compare zero's
            // explicitly
            for (IndexType i = 0; i < nr; ++i)
            {
                for (IndexType j = 0; j < nc; ++j)
                {
                    if (get_value_at(i, j) != rhs.get_value_at(i, j))
                    {
                        return false;
                    }
                }
            }

            return true;
        }


        /**
         * @brief Inequality testing for matrix. (value equality?)
         *
         * @param[in] rhs  The right hand side of the inequality
         *                 operation.
         *
         * @return true, if this matrix and rhs are not identical.
         */
        template <typename OtherMatrixT>
        bool operator!=(OtherMatrixT const &rhs) const
        {
            return !(*this == rhs);
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
         * @return The element at the given transposed row and column.
         */
        ScalarType get_value_at(IndexType row, IndexType col) const
        {
            return m_matrix.get_value_at(col, row);
        }


        // Not certain about this implementation
        //void set_value_at(IndexType         row,
        //                  IndexType         col,
        //                  ScalarType const &val)
        //{
        //    m_matrix.set_value_at(col, row, val);
        //}

        //other methods that may or may not belong here:
        //
        void print_info(std::ostream &os) const
        {
            os << "Backend TransposeView of:" << std::endl;
            m_matrix.print_info(os);
        }

        friend std::ostream&
        operator<<(std::ostream                 &os,
                   TransposeView<MatrixT> const &mat)
        {
            IndexType num_rows, num_cols;
            mat.get_shape(num_rows, num_cols);
            for (IndexType row = 0; row < num_rows; ++row)
            {
                os << ((row == 0) ? "[[" : " [");
                if (num_cols > 0)
                {
                    os << mat.get_value_at(row, 0);
                }

                for (IndexType col = 1; col < num_cols; ++col)
                {
                    os << ", " << mat.get_value_at(row, col);
                }
                os << ((row == num_rows - 1) ? "]]" : "]\n");
            }
            return os;
        }

    private:
        /**
         * Copy assignment not implemented.
         *
         * @param[in] rhs  The transpose view to assign to this.
         *
         * @todo Assignment should be disallowed as you cannot reassign a
         *       a reference.
         */
        TransposeView<MatrixT>&
        operator=(TransposeView<MatrixT> const &rhs);

    private:
        MatrixT const &m_matrix;
    };

} //backend
} // GraphBLAS

//****************************************************************************
/// @deprecated
//****************************************************************************


namespace graphblas
{
namespace backend
{
    /**
     * @brief View a matrix as if it were transposed.
     *
     * @tparam MatrixT         Implements the 2D matrix concept.
     */
    template<typename MatrixT>
    class TransposeView
    {
    public:
        typedef typename MatrixT::ScalarType ScalarType;

        // CONSTRUCTORS

        TransposeView(MatrixT const &matrix):
            m_matrix(matrix)
        {
        }

        /**
         * Copy constructor.
         *
         * @param[in] rhs The transpose view to copy.
         *
         * @todo Is this const correct?
         */
        TransposeView(TransposeView<MatrixT> const &rhs)
            : m_matrix(rhs.m_matrix)
        {
        }

        ~TransposeView()
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
            m_matrix.get_shape(num_cols, num_rows);
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

        IndexType get_nnz() const { return m_matrix.get_nnz(); }

        // EQUALITY OPERATORS
        //bool
        //operator==(TransposeView<MatrixT> const &rhs) const
        //{
        //    return m_matrix == rhs.m_matrix;
        //}

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
            IndexType nr, nc, rhs_nr, rhs_nc;
            get_shape(nr, nc);
            rhs.get_shape(rhs_nr, rhs_nc);
            if ((nr != rhs_nr) || (nc != rhs_nc))
            {
                return false;
            }

            // Definitely a more efficient way than this.  Only compare
            // non-zero elements.  Then decide if compare zero's
            // explicitly
            for (IndexType i = 0; i < nr; ++i)
            {
                for (IndexType j = 0; j < nc; ++j)
                {
                    if (get_value_at(i, j) != rhs.get_value_at(i, j))
                    {
                        return false;
                    }
                }
            }

            return true;
        }


        /**
         * @brief Inequality testing for matrix. (value equality?)
         *
         * @param[in] rhs  The right hand side of the inequality
         *                 operation.
         *
         * @return true, if this matrix and rhs are not identical.
         */
        template <typename OtherMatrixT>
        bool operator!=(OtherMatrixT const &rhs) const
        {
            return !(*this == rhs);
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
         * @return The element at the given transposed row and column.
         */
        ScalarType get_value_at(IndexType row, IndexType col) const
        {
            return m_matrix.get_value_at(col, row);
        }


        // Not certain about this implementation
        //void set_value_at(IndexType         row,
        //                  IndexType         col,
        //                  ScalarType const &val)
        //{
        //    m_matrix.set_value_at(col, row, val);
        //}

        //other methods that may or may not belong here:
        //
        void print_info(std::ostream &os) const
        {
            os << "Backend TransposeView of:" << std::endl;
            m_matrix.print_info(os);
        }

        friend std::ostream&
        operator<<(std::ostream                 &os,
                   TransposeView<MatrixT> const &mat)
        {
            IndexType num_rows, num_cols;
            mat.get_shape(num_rows, num_cols);
            for (IndexType row = 0; row < num_rows; ++row)
            {
                os << ((row == 0) ? "[[" : " [");
                if (num_cols > 0)
                {
                    os << mat.get_value_at(row, 0);
                }

                for (IndexType col = 1; col < num_cols; ++col)
                {
                    os << ", " << mat.get_value_at(row, col);
                }
                os << ((row == num_rows - 1) ? "]]" : "]\n");
            }
            return os;
        }

    private:
        /**
         * Copy assignment not implemented.
         *
         * @param[in] rhs  The transpose view to assign to this.
         *
         * @todo Assignment should be disallowed as you cannot reassign a
         *       a reference.
         */
        TransposeView<MatrixT>&
        operator=(TransposeView<MatrixT> const &rhs);

    private:
        MatrixT const &m_matrix;
    };

} //backend
} // graphblas


#endif // GB_SEQUENTIAL_TRANSPOSE_VIEW_HPP
