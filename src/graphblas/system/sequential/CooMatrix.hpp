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


#ifndef GB_SEQUENTIAL_COO_HPP
#define GB_SEQUENTIAL_COO_HPP

#include <algorithm>
#include <iostream>
#include <vector>

#include <graphblas/system/sequential/RowView.hpp>

namespace graphblas
{
    /**
     * @brief Class representing a coordinate format sparse matrix.
     */
    template<typename ScalarT>
    class CooMatrix
    {
    public:
        typedef ScalarT ScalarType;

        /**
         * @brief Construct an empty coordinate format sparse matrix
         *        with the given shape.
         * @param[in] M The number of rows.
         * @param[in] N The number of columns.
         * @param[in]  zero    The value of the structural zero.
         */
        CooMatrix(IndexType M,
                  IndexType N,
                  ScalarT const &zero = static_cast<ScalarT>(0))
            : m_num_rows(M),
              m_num_cols(N),
              m_zero(zero)
        {
        }

        /**
         * @brief Constuct a coordinate format sparse matrix from a
         *        given dense matrix.
         *
         * @param[in] values The dense matrix to construct a coordinate
         *                   sparse matrix from.
         * @param[in]  zero    The value of the structural zero.
         */
        CooMatrix(std::vector<std::vector<ScalarT> > const &values,
                  ScalarT const &zero=static_cast<ScalarT>(0))
            : m_zero(zero)
        {
            m_num_rows = values.size();
            m_num_cols = values[0].size();

            for (IndexType i = 0; i < m_num_rows; i++)
            {
                for (IndexType j = 0; j < m_num_cols; j++)
                {
                    if (values[i][j] != m_zero)
                    {
                        m_rows.push_back(i);
                        m_columns.push_back(j);
                        m_values.push_back(values[i][j]);
                    }
                }
            }
        }


        /**
         * @brief Construct a coordinate format sparse matrix from the
         *        given row, column, and value vectors.
         *
         * @param[in] rows    The row indices of the non-zero elements.
         * @param[in] columns The column indices of the non-zero elements.
         * @param[in] values  The values of the non-zero elements.
         * @param[in]  zero    The value of the structural zero.
         */
        CooMatrix(std::vector<IndexType> rows,
                  std::vector<IndexType> columns,
                  std::vector<ScalarT>   values,
                  ScalarT         const &zero = static_cast<ScalarT>(0))
            : m_rows(rows),
              m_columns(columns),
              m_values(values),
              m_zero(zero)
        {
            m_num_rows =
                static_cast<IndexType>(
                    *std::max_element(rows.begin(),
                                      rows.end()));
            m_num_cols =
                static_cast<IndexType>(
                    *std::max_element(columns.begin(),
                                      columns.end()));
        }

        /**
         * @brief Copy constructor for CooMatrix.
         * @param other The CooMatrix to copy construct this CooMatrix from.
         */
        CooMatrix(const CooMatrix<ScalarT> &other)
        {
            if (this != &other)
            {
                m_num_rows = other.m_num_rows;
                m_num_cols = other.m_num_cols;
                m_rows = other.m_rows;
                m_columns = other.m_columns;
                m_values = other.m_values;
                m_zero = other.m_zero;
            }
        }

        // DESTRUCTOR

        /**
         * @brief Destructor for CooMatrix.
         */
        ~CooMatrix()
        { }


        /**
         * Copy assignment.
         * @param rhs The CooMatrix to assign to this CooMatrix.
         * @return *this.
         */
        CooMatrix<ScalarT>& operator=(const CooMatrix<ScalarT> &rhs)
        {
            if (this != &rhs)
            {
                m_num_rows = rhs.m_num_rows;
                m_num_cols = rhs.m_num_cols;
                m_rows = rhs.m_rows;
                m_columns = rhs.m_columns;
                m_values = rhs.m_values;
                m_zero = rhs.m_zero;
            }
            return *this;
        }

        /**
         * @brief Assignment from a dense matrix.
         * @param rhs The dense matrix to assign to this CooMatrix.
         * @return *this.
         */
        CooMatrix<ScalarT>& operator=(const std::vector<std::vector<ScalarT> > &rhs)
        {
            CooMatrix<ScalarT> rhs_coo(rhs);  // THIS IS BAD (to much copy).
            m_num_rows = rhs_coo.m_num_rows;
            m_num_cols = rhs_coo.m_num_cols;
            m_rows = rhs_coo.m_rows;
            m_columns = rhs_coo.m_columns;
            m_values = rhs_coo.m_values;
            m_zero = rhs_coo.m_zero;
            return *this;
        }


        /**
         * @brief Get the shape for this CooMatrix.
         * @return A tuple containing the shape in the form (M, N),
         *         where M is the number of rows, and N is the number
         *         of columns.
         */
        void get_shape(IndexType &num_rows, IndexType &num_cols) const
        {
            num_rows = m_num_rows;
            num_cols = m_num_cols;
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
         * @brief Set the zero eleemnt for this CooMatrix.
         * @param[in] new_zero The new zero element for this CooMatrix.
         * @return The old zero element for this CooMatrix.
         */
        ScalarT set_zero(ScalarT new_zero)
        {
            /*
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
            IndexType vec_index = 0;
            bool missing_element = false;

            /// @todo m_* all have same .shape(), use that here?
            if (m_rows.size() != 0 && m_columns.size() != 0
                && m_values.size() != 0)
            {
                for (vec_index = 0;
                     vec_index <= m_num_rows * m_num_cols;
                     ++vec_index)
                {
                    if (vec_index == m_num_rows * m_num_cols)
                    {
                        missing_element = true;
                        break;
                    }
                    if ((m_rows[vec_index] == i) &&
                        (m_columns[vec_index] == j))
                    {
                        break;
                    }
                }
            }
            else
            {
                missing_element = true;
            }

            if (missing_element)
            {
                return m_zero;
            }
            else
            {
                return m_values[vec_index];
            }
        }

        void set_value_at(IndexType row_index, IndexType elem_index,
                          ScalarT const &rhs)
        {
            IndexType vec_index = 0;

            /// @todo m_* all have same .shape(), use that here?
            if (m_rows.size() != 0 && m_columns.size() != 0
                && m_values.size() != 0)
            {
                for (vec_index = 0;
                     vec_index <= m_rows.size();
                     ++vec_index)
                {
                    if (vec_index == m_num_rows * m_num_cols)
                    {
                        break;
                    }
                    if ((m_rows[vec_index] == row_index) &&
                        (m_columns[vec_index] == elem_index))
                    {
                        break;
                    }
                }
            }


            if (rhs != m_zero)
            {
                // Insert the element.
                if (vec_index < m_rows.size())
                {
                    m_values[vec_index] = rhs;
                }
                // Otherwise, append it to the end.
                else
                {
                    m_rows.push_back(row_index);
                    m_columns.push_back(elem_index);
                    m_values.push_back(rhs);
                    ++m_num_rows;
                    ++m_num_cols;
                }
            }
            else
            {
                if (vec_index <= row_index * m_num_rows + elem_index)
                {
                    m_rows.erase(m_rows.begin() + vec_index);
                    m_columns.erase(m_columns.begin() + vec_index);
                    m_values.erase(m_values.begin() + vec_index);
                    --m_num_rows;
                    --m_num_cols;
                }
            }
        }

        /**
         * @brief Indexing function for accessing the rows of this
         *        CooMatrix.
         *
         * @param[in] row  The row to access.
         *
         * @return The row of this CooMatrix as a dense_vector.
         */
        RowView<CooMatrix<ScalarT> const> get_row(IndexType row) const
        {
            return RowView<CooMatrix<ScalarT> const>(row, *this);
        }

        RowView<CooMatrix<ScalarT> > get_row(IndexType row)
        {
            return RowView<CooMatrix<ScalarT> >(row, *this);
        }

        /**
         * @brief Indexing operator for accessing the rows of this
         *        CooMatrix.
         *
         * @param[in] row  The row to access.
         * @return The row of this CooMatrix as a dense_vector.
         */
        RowView<CooMatrix<ScalarT> const>
        operator[](IndexType row) const
        {
            return RowView<CooMatrix<ScalarT> const>(row, *this);
        }

        // need a non-const version for mutation.
        RowView<CooMatrix<ScalarT> >
        operator[](IndexType row)
        {
            return RowView<CooMatrix<ScalarT> >(row, *this);
        }

        // OPERATORS

        /**
         * @brief Equality testing for CooMatrix.
         * @param rhs The right hand side of the equality operation.
         * @return If this CooMatrix and rhs are identical.
         */
        bool operator==(const CooMatrix<ScalarT> &rhs) const
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
        bool operator!=(const CooMatrix<ScalarT> &rhs) const
        {
            return !(*this == rhs);
        }

        friend std::ostream& operator<<(std::ostream        &os,
                                        const CooMatrix<ScalarT> &coo)
        {
            IndexType M, N;
            coo.get_shape(M, N);
            /// @todo Minimum values for M and N?
            os << "[" << coo[0] << "," << std::endl;
            for (IndexType i = 1; i < M - 1; i++)
            {
                os << " " << coo[i] << "," << std::endl;
            }
            os << " " << coo[M - 1] << "]";
            return os;
        }

    protected:
        // DATA

        /** The shape of this CooMatrix. */
        IndexType m_num_rows;
        IndexType m_num_cols;
        /** The zero element for this CooMatrix. */
        ScalarT m_zero;

    private:
        /**
        * @todo We assume that these are "sorted" for efficiency (they are
        * added like in the constructor above).
        */

        /** The row indices for non-zero elements in this CooMatrix. */
        std::vector<IndexType> m_rows;

        /** The column indices for non-zero elements in this CooMatrix. */
        std::vector<IndexType> m_columns;

        /** The values in this CooMatrix. */
        std::vector<ScalarT> m_values;
    };
} // graphblas

#endif // GB_SEQUENTIAL_COO_HPP
