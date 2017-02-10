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

#ifndef GB_SEQUENTIAL_NEGATE_VIEW_HPP
#define GB_SEQUENTIAL_NEGATE_VIEW_HPP

//#include <graphblas/system/sequential/ColumnView.hpp>
#include <graphblas/system/sequential/Matrix.hpp>

namespace GraphBLAS
{
namespace backend
{
    //************************************************************************
    /**
     * @brief View a matrix as if it were complementd (stored values and
     *        structural zeroes are swapped).
     *
     * @tparam MatrixT     Implements the 2D matrix concept.
     * @tparam SemiringT   Used to define the behaviour of the complement
     */
    template<typename MatrixT>
    class ComplementView
    {
    public:
        typedef typename MatrixT::ScalarType ScalarType;

        // CONSTRUCTORS

        ComplementView(MatrixT const &matrix):
            m_matrix(matrix)
        {
            /// @todo assert that matrix and semiring zero() are the same?
        }

        /**
         * Copy constructor.
         *
         * @param[in] rhs The complement view to copy.
         *
         * @todo Is this const correct?
         */
        ComplementView(ComplementView<MatrixT> const &rhs)
            : m_matrix(rhs.m_matrix)
        {
        }

        ~ComplementView()
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
            m_matrix.get_shape(num_rows, num_cols);
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
        //ScalarType set_zero(ScalarType new_zero)
        //{
        //    return m_matrix.set_zero(new_zero);
        //}

        /**
         * @return  The number of stored elements in the view.
         */
        IndexType get_nnz() const
        {
            IndexType rows, cols;
            m_matrix.get_shape(rows, cols);
            return (rows*cols - m_matrix.get_nnz());
        }

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
         * @return The complemented element at the given row and column.
         *
         * @todo this needs to go away
         */
        ScalarType get_value_at(IndexType row, IndexType col) const
        {
            if (m_matrix.get_value_at(row, col) == get_zero())
                return !get_zero();
            else
                return get_zero();
        }

        // DONT ALLOW MODIFICATION OF MASKS.
        // Not certain about this implementation
        //void set_value_at(IndexType         row,
        //                  IndexType         col,
        //                  ScalarType const &val)
        //{
        //    m_matrix.set_value_at(row, col, math::NotFn<ScalarType>()(val));
        //}

        void print_info(std::ostream &os) const
        {
            os << "Backend ComplementView of:" << std::endl;
            m_matrix.print_info(os);
        }

        friend std::ostream&
        operator<<(std::ostream                  &os,
                   ComplementView<MatrixT> const &mat)
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
         * @param[in] rhs  The complement view to assign to this.
         *
         * @todo Assignment should be disallowed as you cannot reassign a
         *       a reference.
         */
        ComplementView<MatrixT> &
            operator=(ComplementView<MatrixT> const &rhs);

    private:
        MatrixT const &m_matrix;
    };

} // backend
} // GraphBLAS

//****************************************************************************
/// @deprecated
//****************************************************************************


namespace graphblas
{
namespace backend
{
    //************************************************************************
    // Generalized Negate/complement
    template <typename SemiringT>
    class SemiringNegate
    {
    public:
        typedef typename SemiringT::ScalarType ScalarType;
        ScalarType operator()(ScalarType const &value)
        {
            if (value == SemiringT().zero())
                return SemiringT().one();
            else
                return SemiringT().zero();
        }
    };

    //************************************************************************
    /**
     * @brief View a matrix as if it were negated (stored values and
     *        structural zeroes are swapped).
     *
     * @tparam MatrixT     Implements the 2D matrix concept.
     * @tparam SemiringT   Used to define the behaviour of the negate
     */
    template<typename MatrixT, typename SemiringT>
    class NegateView
    {
    public:
        typedef typename MatrixT::ScalarType ScalarType;

        // CONSTRUCTORS

        NegateView(MatrixT const &matrix):
            m_matrix(matrix)
        {
            /// @todo assert that matrix and semiring zero() are the same?
        }

        /**
         * Copy constructor.
         *
         * @param[in] rhs The negate view to copy.
         *
         * @todo Is this const correct?
         */
        NegateView(NegateView<MatrixT, SemiringT> const &rhs)
            : m_matrix(rhs.m_matrix)
        {
        }

        ~NegateView()
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
            m_matrix.get_shape(num_rows, num_cols);
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
        //ScalarType set_zero(ScalarType new_zero)
        //{
        //    return m_matrix.set_zero(new_zero);
        //}

        /**
         * @return  The number of stored elements in the view.
         */
        IndexType get_nnz() const
        {
            IndexType rows, cols;
            m_matrix.get_shape(rows, cols);
            return (rows*cols - m_matrix.get_nnz());
        }

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
         * @return The negated element at the given row and column.
         */
        ScalarType get_value_at(IndexType row, IndexType col) const
        {
            //return math::NotFn<ScalarType>()(m_matrix.get_value_at(row, col));
            return SemiringNegate<SemiringT>()(m_matrix.get_value_at(row, col));
        }


        // Not certain about this implementation
        //void set_value_at(IndexType         row,
        //                  IndexType         col,
        //                  ScalarType const &val)
        //{
        //    m_matrix.set_value_at(row, col, math::NotFn<ScalarType>()(val));
        //}

        void print_info(std::ostream &os) const
        {
            os << "Backend NegateView of:" << std::endl;
            m_matrix.print_info(os);
        }

        friend std::ostream&
        operator<<(std::ostream                         &os,
                   NegateView<MatrixT, SemiringT> const &mat)
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
         * @param[in] rhs  The negate view to assign to this.
         *
         * @todo Assignment should be disallowed as you cannot reassign a
         *       a reference.
         */
        NegateView<MatrixT, SemiringT> &
        operator=(NegateView<MatrixT, SemiringT> const &rhs);

    private:
        MatrixT const &m_matrix;
    };

} // backend
} // graphblas

#endif // GB_SEQUENTIAL_NEGATE_VIEW_HPP
