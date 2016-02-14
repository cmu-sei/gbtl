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

#ifndef GB_CUSP_COOMATRIX_HPP
#define GB_CUSP_COOMATRIX_HPP

//#include <algorithm>
#include <iostream>
#include <vector>

#include <cusp/coo_matrix.h>

#include <graphblas/system/cusp/RowView.hpp>

namespace graphblas
{
    /**
     * @brief Class representing a coordinate format sparse matrix.
     */
    template<typename ScalarT, typename MemorySpaceT = ::cusp::device_memory>
    class CooMatrix : public ::cusp::coo_matrix<IndexType,
                                                ScalarT,
                                                MemorySpaceT>
    {
    public:
        typedef ScalarT ScalarType;
        typedef MemorySpaceT MemorySpaceType;
        typedef ::cusp::coo_matrix<IndexType,ScalarT,MemorySpaceT> BaseType;

        /**
         * @brief Construct an empty coordinate format sparse matrix
         *        with the given shape.
         *
         * @param[in] rows  The number of rows.
         * @param[in] cols  The number of columns.
         * @param[in] zero  The "zero" value, additive identity, and
         *                  the structural zero.
         */
        CooMatrix(IndexType      rows,
                  IndexType      cols,
                  ScalarT const &zero = static_cast<ScalarT>(0))
            : ::cusp::coo_matrix<IndexType,
                                 ScalarT,
                                 MemorySpaceT>(rows, cols, 0),
              m_num_rows(rows),
              m_num_cols(cols),
              m_nnz(0),
              m_zero(zero)
        {
        }

        /**
         * @brief Construct from a CUSP COO matrix
         */
        CooMatrix(::cusp::coo_matrix<IndexType,
                  ScalarT,
                  MemorySpaceT> const &matrix,
                  ScalarT              zero = static_cast<ScalarT>(0))
            : ::cusp::coo_matrix<IndexType, ScalarT, MemorySpaceT>(matrix),
              m_zero(zero)
        {
            m_num_rows = matrix.num_rows;
            m_num_cols = matrix.num_cols;
            m_nnz = matrix.row_indices.size();  // matrix.num_entries
        }

        /**
         * @brief Constuct a coordinate format sparse matrix from a
         *        given dense matrix.
         *
         * @param[in] values The dense matrix to construct a coordinate
         *                   sparse matrix from.
         * @param[in] zero   The "zero" value, additive identity, and
         *                   the structural zero.
         */
        CooMatrix(std::vector<std::vector<ScalarT> > const &values,
                  ScalarT const                            &zero =
                      static_cast<ScalarT>(0))
            : ::cusp::coo_matrix<IndexType, ScalarT, MemorySpaceT>(),
              m_zero(zero)
        {
            m_nnz = 0;

            m_num_rows = values.size();
            m_num_cols = values[0].size();

            for (IndexType i = 0; i < m_num_rows; ++i)
            {
                for (IndexType j = 0; j < m_num_cols; ++j)
                {
                    if (values[i][j] != m_zero)
                    {
                        ++m_nnz;
                    }
                }
            }

            BaseType::resize(m_num_rows, m_num_cols, m_nnz);

            IndexType idx = 0;
            for (IndexType i = 0; i < m_num_rows; ++i)
            {
                for (IndexType j = 0; j < m_num_cols; ++j)
                {
                    if (values[i][j] != m_zero)
                    {
                        BaseType::row_indices[idx] = i;
                        BaseType::column_indices[idx] = j;
                        BaseType::values[idx] = values[i][j];
                        ++idx;
                    }
                }
            }
        }

#if 0
        // /**
        // * @brief Construct a coordinate format sparse matrix from the
        // *        given row, column, and value vectors.
        // *
        // * @param[in] rows    The row indices of the non-zero elements.
        // * @param[in] columns The column indices of the non-zero elements.
        // * @param[in] values  The values of the non-zero elements.
        // */
        // CooMatrix(std::vector<IndexType> rows,
        //           std::vector<IndexType> columns,
        //           std::vector<ScalarT>   values,
        //           ScalarT         const &zero = static_cast<ScalarT>(0))
        //     : m_zero(zero)
        // {
        //     IndexType nnz = 0;

        //     m_num_rows = 0;
        //     m_num_cols = 0;

        //     for (IndexType i = 0; i < values.size(); ++i)
        //     {
        //         if (values[i] != m_zero)
        //         {
        //             m_num_rows = std::max(m_num_rows, rows[i]);
        //             m_num_cols = std::max(m_num_cols, columns[i]);
        //             ++nnz;
        //         }
        //     }

        //     resize(m_num_rows, m_num_cols, nnz);
        //     nnz = 0;

        //     for (IndexType i = 0; i < m_num_rows; ++i)
        //     {
        //         for (IndexType j = 0; j < m_num_cols; ++j)
        //         {
        //             if (values[i][j] != m_zero)
        //             {
        //                 this->row_indices[nnz] = i;
        //                 this->column_indices[nnz] = j;
        //                 this->values[nnz] = values[i][j];
        //                 ++nnz;
        //             }
        //         }
        //     }
        //     m_nnz = nnz;
        // }


        // /** @brief Copy constructor for CooMatrix.
        //  * @param[in]  rhs  The CooMatrix to copy construct this CooMatrix
        //  *                  from.
        //  */
        // CooMatrix(const CooMatrix<ScalarT, MemorySpaceT> &rhs)
        //     : ::cusp::coo_matrix<IndexType, ScalarT, MemorySpaceT>(rhs)
        // {
        //     if (this != &rhs)
        //     {
        //         m_num_rows = rhs.m_num_rows;
        //         m_num_cols = rhs.m_num_cols;
        //         m_nnz = rhs.m_nnz;
        //         m_zero = rhs.m_zero;
        //     }
        // }
#endif

        /**
         * Copy assignment.
         * @param[in]  rhs  The CooMatrix to assign to this CooMatrix.
         * @return *this.
         */
        CooMatrix<ScalarT, MemorySpaceT>& operator=(
            CooMatrix<ScalarT, MemorySpaceT> const &rhs)
        //: ::cusp::coo_matrix<IndexType, ScalarT, MemorySpaceT>(rhs)
        {
            if (this != &rhs)
            {
                /// assign the data in the base class.
                BaseType::resize(rhs.m_num_rows, rhs.m_num_cols, rhs.m_nnz);
                ::cusp::coo_matrix<IndexType,
                                   ScalarT,
                                   MemorySpaceT>::operator=(rhs);

                m_num_rows = rhs.m_num_rows;
                m_num_cols = rhs.m_num_cols;
                m_nnz = rhs.m_nnz;
                m_zero = rhs.m_zero;
            }
            return *this;
        }

        /**
         * @brief Assignment from a cusp matrix.
         * @param[in]  rhs  The matrix to assign to this CooMatrix.
         * @return *this.
         */
        CooMatrix<ScalarT, MemorySpaceT>& operator=(
            ::cusp::coo_matrix<IndexType, ScalarT, MemorySpaceT> rhs)
        {
            if (this != &rhs)
            {
                ::cusp::coo_matrix<IndexType,
                                   ScalarT,
                                   MemorySpaceT>::operator=(rhs);
                m_num_rows = rhs.num_rows;
                m_num_cols = rhs.num_cols;
                m_nnz = rhs.num_entries;

                /// @todo what is the new zero?
                //m_zero = ???
            }
            return *this;
        }

        /**
         * @brief Assignment from a dense matrix.
         * @param[in]  rhs  The dense matrix to assign to this CooMatrix.
         * @return *this.
         */
        CooMatrix<ScalarT, MemorySpaceT>& operator=(
            std::vector<std::vector<ScalarT> > const &rhs)
        {
            // THIS IS BAD (too much copy).
            CooMatrix<ScalarT, MemorySpaceT> rhs_coo(rhs);
            BaseType::operator=(rhs_coo);
            return *this;
        }


        /**
         * @brief Get the shape for this CooMatrix.
         * @return A tuple containing the shape in the form (M, N),
         *         where M is the number of rows, and N is the number
         *         of columns.
         */
        void get_shape(IndexType &rows, IndexType &cols) const
        {
            rows = m_num_rows;
            cols = m_num_cols;
        }

        /**
         * @brief Get the zero eleemnt for this CooMatrix.
         * @return The zero element for this CooMatrix.
         */
        ScalarT get_zero() const
        {
            return m_zero;
        }

        /**
         * @brief Get the number of nonzero elements for this matrix.
         *
         * @return The number of nonzero elements for this matrix.
         */
        IndexType get_nnz() const
        {
            return BaseType::num_entries;
        }

        /**
         * @brief Set the zero eleemnt for this CooMatrix.
         * @param new_zero The new zero element for this CooMatrix.
         * @return The old zero element for this CooMatrix.
         */
        ScalarT set_zero(ScalarT new_zero)
        {
            /**
             * @todo Should we go through and first convert all
             *       new_zeros to old_zero elements (recompute the
             *       sparsity?)?
             */
            ScalarT old_zero = m_zero;
            m_zero = new_zero;
            return old_zero;
        }

        /**
         * @brief Access the elements of this CooMatrix given row and
         *        column indexes.
         *
         * @param[in] i The row to access.
         * @param[in] j The column to access.
         *
         * @return The element of this CooMatrix at the given row and
         *         column.
         */
        ScalarT get_value_at(IndexType i, IndexType j) const
        {
            for (IndexType vec_index = 0;
                 vec_index < BaseType::values.size();
                 ++vec_index)
            {
                if ((BaseType::row_indices[vec_index] == i) &&
                    (BaseType::column_indices[vec_index] == j))
                {
                    // element found
                    return BaseType::values[vec_index];
                }
            }

            return m_zero;
        }


        void set_value_at(IndexType row_index, IndexType col_index,
                          ScalarT const &rhs)
        {
            /// @todo what if indices are out of bounds
            /// @todo what if rhs = m_zero
            for (IndexType vec_index = 0;
                 vec_index < BaseType::values.size();
                 ++vec_index)
            {
                if ((BaseType::row_indices[vec_index] == row_index) &&
                    (BaseType::column_indices[vec_index] == col_index))
                {
                    BaseType::values[vec_index] = rhs;
                    return;
                }
            }

            // Otherwise, append it to the end.
            if (row_index >= m_num_rows) m_num_rows = row_index + 1;
            if (col_index >= m_num_cols) m_num_cols = col_index + 1;
            IndexType idx = BaseType::num_entries;
            m_nnz = idx + 1;
            BaseType::resize(m_num_rows, m_num_cols, m_nnz);
            BaseType::row_indices[idx] = row_index;
            BaseType::column_indices[idx] = col_index;
            BaseType::values[idx] = rhs;
        }


        /**
         * @brief Indexing function for accessing the rows of this CooMatrix
         *
         * @param[in] row  The row to access.
         *
         * @return The row of this CooMatrix as a dense_vector.
         */
        RowView<CooMatrix<ScalarT, MemorySpaceT> const>
        get_row(IndexType row) const
        {
            return RowView<CooMatrix<ScalarT, MemorySpaceT> const>(row, *this);
        }

        RowView<CooMatrix<ScalarT, MemorySpaceT> > get_row(IndexType row)
        {
            return RowView<CooMatrix<ScalarT, MemorySpaceT> >(row, *this);
        }

        /**
         * @brief Indexing operator for accessing the rows of this
         *        CooMatrix.
         *
         * @param[in] row  The row to access.
         * @return The row of this CooMatrix as a dense_vector.
         */
        RowView<CooMatrix<ScalarT, MemorySpaceT> const>
        operator[](IndexType row) const
        {
            return RowView<CooMatrix<ScalarT, MemorySpaceT> const>(row, *this);
        }

        // need a non-const version for mutation.
        RowView<CooMatrix<ScalarT, MemorySpaceT> >
        operator[](IndexType row)
        {
            return RowView<CooMatrix<ScalarT, MemorySpaceT> >(row, *this);
        }

        // OPERATORS

        /**
         * @brief Equality testing for CooMatrix.
         * @param rhs The right hand side of the equality operation.
         * @return If this CooMatrix and rhs are identical.
         */
        bool operator==(const CooMatrix<ScalarT, MemorySpaceT> &rhs) const
        {
            if ((m_num_rows != rhs.m_num_rows) ||
                (m_num_cols != rhs.m_num_cols))
            {
                return false;
            }

            for (IndexType i = 0; i < m_num_rows; ++i)
            {
                for (IndexType j = 0; j < m_num_cols; ++j)
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
         * @brief Inequality testing for CooMatrix.
         * @param rhs The right hand side of the inequality operation.
         * @return If this CooMatrix and rhs are not identical.
         */
        bool operator!=(const CooMatrix<ScalarT, MemorySpaceT> &rhs) const
        {
            return !(*this == rhs);
        }

        friend std::ostream& operator<<(std::ostream        &os,
                                        const CooMatrix<ScalarT, MemorySpaceT> &coo)
        {
            for (IndexType row = 0; row < coo.m_num_rows; ++row)
            {
                os << ((row == 0) ? "[[" : " [");
                if (coo.m_num_cols > 0)
                {
                    os << coo.get_value_at(row, 0);
                }

                for (IndexType col = 1; col < coo.m_num_cols; ++col)
                {
                    os << ", " << coo.get_value_at(row, col);
                }
                os << ((row == coo.m_num_rows - 1) ? "]]" : "]\n");
            }
            return os;
        }

    private:
        IndexType m_num_rows;  // get from coo_matrix?
        IndexType m_num_cols;  // get from coo_matrix?
        IndexType m_nnz;       // get from coo_matrix?

        ScalarT m_zero;
    };
} // graphblas

#endif // GB_CUSP_COOMATRIX_HPP
