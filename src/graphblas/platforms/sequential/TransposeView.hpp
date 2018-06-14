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

#include <graphblas/platforms/sequential/Matrix.hpp>

//****************************************************************************

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
                if ((nrows() != rhs.nrows()) ||
                    (ncols() != rhs.ncols()) ||
                    (nvals() != rhs.nvals()))
                {
                    return false;
                }

                /// @todo Definitely a more efficient way than this.
                for (IndexType i = 0; i < nrows(); ++i)
                {
                    for (IndexType j = 0; j < ncols(); ++j)
                    {
                        bool has = hasElement(i, j);
                        bool rhas = rhs.hasElement(i, j);
                        if (has == rhas)
                        {
                            if (has)
                            {
                                if (extractElement(i, j) != rhs.extractElement(i, j))
                                {
                                    return false;
                                }
                            }
                        }
                        else
                        {
                            return false;
                        }
                    }
                }

                return true;
            }


            void clear()
            {
                m_matrix.clear();
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


            IndexType nrows() const { return m_matrix.ncols(); }
            IndexType ncols() const { return m_matrix.nrows(); }
            IndexType nvals() const { return m_matrix.nvals(); }

            bool hasElement(IndexType irow, IndexType icol) const
            {
                return m_matrix.hasElement(icol, irow);
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
            ScalarType extractElement(IndexType row, IndexType col) const
            {
                return m_matrix.extractElement(col, row);
            }


            // Not certain about this implementation.  Commented out because
            // currently TransposeViews are read-only.
            //void setElement(IndexType         row,
            //                IndexType         col,
            //                ScalarType const &val)
            //{
            //    m_matrix.setElement(col, row, val);
            //}

            // commented out because currently TransposeViews are read-only
            // template <typename BinaryOpT>
            // void setElement(IndexType irow, IndexType icol, ScalarType const &val,
            //                 BinaryOpT merge)
            // {
            //     m_matrix.setElement(icol, irow, val, merge);
            // }

            typedef typename MatrixT::ColType RowType;
            RowType getRow(IndexType row_index) const
            {
                return m_matrix.getCol(row_index);
            }

            // Not implemented
            //void setRow(IndexType row_index,
            //           std::vector<std::tuple<IndexType, ScalarType> > &row_data)

            typedef typename MatrixT::RowType ColType;
            ColType getCol(IndexType col_index) const
            {
                return m_matrix.getRow(col_index);
            }

            // Not implemented
            //void setCol(IndexType col_index,
            //            std::vector<std::tuple<IndexType, ScalarType> > &col_data)


            // Get column indices for a given row
            void getColumnIndices(IndexType irow, IndexArrayType &v) const
            {
                m_matrix.getRowIndices(irow, v);
            }

            // Get row indices for a given column
            void getRowIndices(IndexType icol, IndexArrayType &v) const
            {
                m_matrix.getColIndices(icol, v);
            }

            //other methods that may or may not belong here:
            //
            void printInfo(std::ostream &os) const
            {
                os << "Backend TransposeView of:" << std::endl;
                m_matrix.printInfo(os);
            }

            friend std::ostream&
            operator<<(std::ostream                 &os,
                       TransposeView<MatrixT> const &mat)
            {
                mat.printInfo(os);
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

#endif // GB_SEQUENTIAL_TRANSPOSE_VIEW_HPP
