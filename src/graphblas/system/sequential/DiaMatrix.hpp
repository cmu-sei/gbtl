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

/**
 * @todo The way we currently deal with zero elements, we might have issues
 *       switching back and forth between zero elements.
 *
 * @todo Logical equality or value equality for different zero elements (to
 *       compare or not to compare zero elements in equality operators)?
 */

#ifndef GB_SEQUENTIAL_DIAMATRIX_HPP
#define GB_SEQUENTIAL_DIAMATRIX_HPP

#include <algorithm>
#include <iostream>
#include <vector>
#include <cstdlib>

#include <graphblas/system/sequential/RowView.hpp>

namespace graphblas
{
    /**
     * @brief Class representing a diagonal sparse matrix.
     */
    template<typename ScalarT>
    class DiaMatrix
    {
    public:
        typedef ScalarT ScalarType;

        /**
         * @brief Construct an empty diagonal sparse matrix with the
         *        given shape.
         *
         * @param[in] num_rows  The number of rows.
         * @param[in] num_cols  The number of columns.
         * @param[in] zero      The value of the structural zero.
         */
        DiaMatrix(IndexType num_rows,
                  IndexType num_cols,
                  ScalarT   zero = static_cast<ScalarT>(0))
            : m_num_rows(num_rows),
              m_num_cols(num_cols),
              m_zero(zero)
        { }

        /**
         * @todo Need to add a parameter to handle duplicate locations.
         *
         * @todo Don't handle assign to a non-empty matrix correctly.
         */
        template<typename RAIteratorI,
                 typename RAIteratorJ,
                 typename RAIteratorV,
                 typename AccumT = graphblas::math::Assign<ScalarType> >
        void buildmatrix(RAIteratorI  i_it,
                         RAIteratorJ  j_it,
                         RAIteratorV  v_it,
                         IndexType    n,
                         AccumT       accum = AccumT())
        {
            /// @todo the following implementation does not work; throw
            throw DimensionException();

            for (IndexType idx = 0; idx < n; ++idx)
            {
                IndexType i = i_it[idx];
                IndexType j = j_it[idx];
                if ((i >= m_num_rows) || (j >= m_num_cols))
                {
                    throw DimensionException();
                }

                set_value_at(i, j, accum(get_value_at(i, j), v_it[idx]));
            }
        }

        /**
         * @brief Constuct a diagonal sparse matrix from a given dense
         *        matrix.
         *
         * @param[in]  values  The dense matrix to construct a diagonal
         *                     sparse matrix form.
         * @param[in]  zero    The value of the structural zero.
         */
        DiaMatrix(std::vector<std::vector<ScalarT> > const &values,
                  ScalarT     zero = static_cast<ScalarT>(0))
            : m_zero(zero)
        {
            m_num_rows = values.size();
            m_num_cols = values[0].size();
            ssize_t N = static_cast<ssize_t>(m_num_cols);

            // Iterate along the diagonals.
            for (ssize_t offset = N - 1; offset > -N; --offset)
            {
                std::vector<ScalarT> row;

                bool should_append = false;
                ssize_t i = std::abs(offset);

                for (ssize_t j = 0;
                     j < static_cast<ssize_t>(m_num_rows) - offset;
                     ++j)
                {
                    ssize_t i1, j1;
                    ssize_t tmp;
                    if (offset < 0)
                    {
                        tmp = i;
                        i = j;
                        j = tmp;
                        i1 = i;
                        j1 = i + j;
                    }
                    else
                    {
                        i1 = i + j;
                        j1 = j;
                    }
                    if (i + j >= N)
                    {
                        continue;
                    }

                    // only append if "row" is not all zeros
                    should_append =
                        should_append || (values[i1][j1] != m_zero);
                    row.push_back(values[i1][j1]);
                    if (offset < 0)
                    {
                        tmp = i;
                        i = j;
                        j = tmp;
                    }
                }

                if (should_append)
                {
                    m_offsets.push_back(-1 * offset);
                    m_data.push_back(row);
                }
            }
        }

        /**
         * @brief Constuct a diagonal sparse matrix from a data set and
         *        offsets.
         * Constructor to construct a diagonal format sparse matrix from a
         * list of diagonals, with respective offsets found in offsets.
         *
         * @param[in]  data    The diagonals to use when constructing a
         *                     diagonal sparse matrix.
         * @param[in]  offsets The offsets of the diagonals found in data.
         * @param[in]  zero    The value of the structural zero.
         *
         * @todo This possibly allows extra values to be passed in in data,
         *        but based on the way indexing is done below, it should
         *        still work fine, but we still might want to fix this?
         */
        DiaMatrix(std::vector<std::vector<ScalarT> > const &data,
                  std::vector<ssize_t> const               &offsets,
                  ScalarT            zero = static_cast<ScalarT>(0))
            : m_data(data), m_offsets(offsets), m_zero(zero)
        {
            /// @todo sign problem
            ssize_t M = *std::max(offsets.begin(), offsets.end()) + 1;
            IndexType N = data[0].size();
            m_num_rows = M;
            m_num_cols = N;
        }

        /**
         * @brief Copy constructor for DiaMatrix.
         * @param other The DiaMatrix to copy construct this DiaMatrix from.
         */
        DiaMatrix(const DiaMatrix<ScalarT> &other)
        {
            if (this != &other)
            {
                m_num_rows = other.m_num_rows;
                m_num_cols = other.m_num_cols;
                m_offsets = other.m_offsets;
                m_data = other.m_data;
                /// @todo Copy zero?
                m_zero = other.m_zero;
            }
        }

        // DESTRUCTOR

        /**
         * @brief Destructor for DiaMatrix.
         */
        ~DiaMatrix()
        { }

        // FUNCTIONS

        /**
         * @brief Get the shape for this LilMatrix.
         * @return num_rows and num_cols.
         */
        void get_shape(IndexType &num_rows, IndexType &num_cols) const
        {
            num_rows = m_num_rows;
            num_cols = m_num_cols;
        }

        /**
         * @brief Get the zero eleemnt for this DiaMatrix.
         * @return The zero element for this DiaMatrix.
         */
        ScalarT get_zero() const
        { return m_zero; }

        /**
         * @brief Set the zero eleemnt for this DiaMatrix.
         * @param new_zero The new zero element for this DiaMatrix.
         * @return The old zero element for this DiaMatrix.
         */
        ScalarT set_zero(ScalarT new_zero)
        {
            /**
             * @todo Should we go through and first convert all new_zeros
             *       to old_zero elements (recompute the sparsity?)?
             */
            ScalarT old_zero = m_zero;
            m_zero = new_zero;
            return old_zero;
        }

        /// @return the number of stored values in this
        IndexType get_nnz() const
        {
            IndexType count = 0;
            for (auto it = m_data.begin(); it != m_data.end(); ++it)
            {
                count += it->size();
            }
            return count;
        }

        /**
         * @brief Access the elements of this DiaMatrix given row and
         *        column indexes.
         * Function provided to access the elements of this DiaMatrix
         * given row (i) and column (j) indices.  The functionality is
         * the same as that of the indexing function for a standard
         * dense matrix.
         *
         * @param[in] i The row to access.
         * @param[in] j The column to access.
         *
         * @return The element of this DiaMatrix at the given row and
         *         column.
         */
        ScalarT get_value_at(IndexType i, IndexType j) const
        {
            if (m_offsets.size() == 0)
            {
                return m_zero;
            }

            ssize_t offset = j - i;
            bool missing_element = false;
            ssize_t offset_index = 0;
            for (;
                 offset_index <= m_offsets.size();
                 ++offset_index)
            {
                if (offset_index == m_offsets.size())
                    //if (offset_index == 2 * m_num_cols - 1)
                {
                    missing_element = true;
                    break;
                }
                if (m_offsets[offset_index] == offset)
                {
                    break;
                }
            }

            if (offset_index > m_data.size())
            {
                missing_element = missing_element || true;
            }

            if (missing_element)
            {
                return m_zero;
            }
            else
            {
                if (offset > 0)
                {
                    // We have to "transpose".
                    j = i;
                }
                // j is now the index in the diagonal.
                return m_data[offset_index][j];
            }
        }

        void set_value_at(IndexType row, IndexType col, ScalarT const &rhs)
        {
            IndexType M = m_num_rows;
            IndexType N = m_num_cols;

            ssize_t offset = col - row;
            ssize_t offset_index = 0;

            IndexType data_index = 0;


            if (offset > 0)
            {
                data_index = row;
            }
            else
            {
                data_index = col;
            }

            if (m_offsets.size() != 0)
            {
                for (offset_index = 0;
                     offset_index < 2 * N;
                     ++offset_index)
                {
                    if (m_offsets[offset_index] == offset)
                    {
                        break;
                    }
                }
                if (offset_index == 2 * N - 1)
                {
                    // We don't have this one yet, so append it and break.
                    if (rhs != m_zero)
                    {
                        m_offsets.push_back(offset);
                        offset_index = m_offsets.size() - 1;

                        if (m_data.size() < m_offsets.size())
                        {
                            m_data.resize(m_offsets.size());
                        }
                    }
                }

                if (offset_index < 2 * N && rhs != m_zero)
                {
                    // Change our value.
                    if (m_data[offset_index].size() < data_index + 1)
                    {
                        m_data[offset_index].resize(data_index + 1);
                    }
                    m_data[offset_index][data_index] = rhs;

                    // Check if we can remove this row.
                    bool should_remove = true;
                    for (auto v : m_data[offset_index])
                    {
                        should_remove = should_remove && (v == m_zero);
                    }
                    if (should_remove)
                    {
                        m_offsets.erase(m_offsets.begin() + offset_index);
                        m_data.erase(m_data.begin() + offset_index);
                    }
                }
            }
            else
            {
                if (rhs != m_zero)
                {
                    m_offsets.push_back(offset);

                    if (m_data.size() == 0)
                    {
                        m_data.resize(1);
                    }
                    if (m_data[0].size() < data_index + 1)
                    {
                        m_data[0].resize(data_index + 1);
                    }
                    m_data[0][data_index] = rhs;
                }
            }
        }

        /**
         * @brief Indexing function for accessing the rows of this
         *        DiaMatrix.
         *
         * @param[in] row  The row to access.
         * @return The row of this DiaMatrix as a dense_vector.
         */
        RowView<DiaMatrix<ScalarT> const> get_row(IndexType row) const
        {
            return RowView<DiaMatrix<ScalarT> const>(row, *this);
        }

        RowView<DiaMatrix<ScalarT> > get_row(IndexType row)
        {
            return RowView<DiaMatrix<ScalarT> >(row, *this);
        }

        // OPERATORS

        /**
         * @brief Equality testing for DiaMatrix.
         * @param rhs The right hand side of the equality operation.
         * @return If this DiaMatrix and rhs are identical.
         */
        const bool operator==(const DiaMatrix<ScalarT> &rhs) const
        {
            IndexType M = m_num_rows;
            IndexType N = m_num_cols;


            bool equal = ((m_num_rows == rhs.m_num_rows) &&
                          (m_num_cols == rhs.m_num_cols));

            for (IndexType i = 0; i < M; ++i)
            {
                for (IndexType j = 0; j < N; ++j)
                {
                    equal = equal && this->operator[](i)[j] == rhs[i][j];
                }
            }

            return equal;
            /*return ((m_shape == rhs.m_shape)
              && (m_offsets == rhs.m_offsets)
              && (m_data == rhs.m_data));

              /// @todo Compare zeros?
              //&& (m_zero == rhs.m_zero));*/
        }

        /**
         * @brief Inequality testing for DiaMatrix.
         * @param rhs The right hand side of the inequality operation.
         * @return If this DiaMatrix and rhs are not identical.
         */
        const bool operator!=(const DiaMatrix<ScalarT> &rhs) const
        {
            return !(this->operator==(rhs));
        }

        /**
         * @brief Copy assignment.
         * @param rhs The DiaMatrix to assign to this DiaMatrix.
         * @return *this.
         */
        DiaMatrix<ScalarT>& operator=(const DiaMatrix<ScalarT> &rhs)
        {
            if (this != &rhs)
            {
                m_num_rows = rhs.m_num_rows;
                m_num_cols = rhs.m_num_cols;
                m_offsets = rhs.m_offsets;
                m_data = rhs.m_data;
                m_zero = rhs.m_zero;
            }
            return *this;
        }

        /**
         * @brief Assignment from a dense matrix.
         * @param rhs The dense matrix to assign to this DiaMatrix.
         * @return *this.
         */
        DiaMatrix<ScalarT>& operator=(const std::vector<std::vector<ScalarT> > &rhs)
        {
            /**
             * @todo How to handle zero in this case?  Should we just continue to
             *      use whatever the current zero element is?
             */
            DiaMatrix<ScalarT> rhs_dia(rhs);
            m_num_rows = rhs_dia.m_num_rows;
            m_num_cols = rhs_dia.m_num_cols;
            m_offsets = rhs_dia.m_offsets;
            m_data = rhs_dia.m_data;

            /// @todo Assign zero element?
            m_zero = rhs_dia.m_zero;
            return *this;
        }

        /**
         * @brief Indexing operator for accessing the rows of this
         *        DiaMatrix.
         *
         * @param[in] row  The row to access.
         * @return The row of this DiaMatrix as a dense_vector.
         */
        RowView<DiaMatrix<ScalarT> const> operator[](IndexType row) const
        {
            return RowView<DiaMatrix<ScalarT> const>(row, *this);
        }

        // need a non-const version for mutation.
        RowView<DiaMatrix<ScalarT> > operator[](IndexType row)
        {
            return RowView<DiaMatrix<ScalarT> >(row, *this);
        }

        friend std::ostream& operator<<(std::ostream        &os,
                                        const DiaMatrix<ScalarT> &dia)
        {
            IndexType M = dia.m_num_rows;
            IndexType N = dia.m_num_cols;

            /// @todo Add support for dense_vector construction like this.
            //dense_vector<T> row;

            /// @todo Minimum values for M and N?
            os << "[" << dia[0] << "," << std::endl;
            for (IndexType i = 1; i < M - 1; ++i)
            {
                os << " " << dia[i] << "," << std::endl;
            }
            os << " " << dia[M - 1] << "]";

            return os;
        }

    protected:
        /** The shape of this DiaMatrix. */
        IndexType  m_num_rows;
        IndexType  m_num_cols;
        ScalarT m_zero;

    private:
        /** The offsetts of the diagonals for this DiaMatrix. */
        std::vector<ssize_t> m_offsets;
        /** The diagonals of this DiaMatrix. */
        std::vector<std::vector<ScalarT> > m_data;
    };
} // graphblas

#endif // GB_SEQUENTIAL_DIAMATRIX_HPP
