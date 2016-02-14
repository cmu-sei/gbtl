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

#ifndef GB_SEQUENTIAL_CSCMATRIX_HPP
#define GB_SEQUENTIAL_CSCMATRIX_HPP

#include <iostream>
#include <vector>

#include <graphblas/system/sequential/RowView.hpp>

namespace graphblas
{
    /**
     * @brief Class representing a Csc format matrix.
     */
    template<typename ScalarT>
    class CscMatrix
    {
    public:

        // TYPEDEFS
        typedef ScalarT ScalarType;

        // CONSTRUCTORS
        /**
         * @brief Construct an empty Csc matrix with
         *        the given shape.
         *
         * @param[in] num_rows
         * @param[in] num_cols
         * @param[in] zero      The "zero" value.
         */
        CscMatrix(IndexType num_rows,
                  IndexType num_cols,
                  ScalarT   zero = static_cast<ScalarT>(0))
            : m_num_rows(num_rows),
              m_num_cols(num_cols),
              m_zero(zero),
              m_col_ptr(num_cols+1, 0)
        {
        }

        /**
         * @todo need to add a parameter to handle duplicate locations.
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
         * @brief Construct a Csc matrix from a given dense
         *        matrix.
         *
         * @param[in] values  The dense matrix from which to construct
         *                    a Csc sparse matrix.
         * @param[in] zero    The "zero" value.
         */
        CscMatrix(std::vector<std::vector<ScalarT> >  const &values,
                  ScalarT  zero = static_cast<ScalarT>(0))
            : m_num_rows(values.size()),
              m_num_cols(values[0].size()),// warning assumes all same
              m_zero(zero)
        {
            for (IndexType col = 0; col < values[0].size(); col++)
            {
                // Have we seen a non-zero number in this column?
                bool is_first_nonzero = true;
                bool is_zero_col = true;
                for (IndexType row = 0; row < values.size(); row++)
                {
                    if (values[row][col] != zero)
                    {
                        is_zero_col = false;
                        m_val.push_back(values[row][col]);
                        m_row_ind.push_back(row);
                        if (is_first_nonzero)
                        {
                            m_col_ptr.push_back(m_val.size() -1);
                            is_first_nonzero = false;
                        }
                    }
                }
                if (is_zero_col)
                {
                    m_col_ptr.push_back(m_val.size());
                }
            }
            m_col_ptr.push_back(m_val.size());
        }

        /**
         * @brief Construct a Csc format matrix from a
         *        list of lists containing column indices and value
         *        tuples.
         *
         * @param[in] data  The Csc data to use when constructing.
         * @param[in] zero  The "zero" value.
         */
        CscMatrix(int,
            std::vector<std::vector<std::tuple<IndexType, ScalarT> > > const &data,
            ScalarT    zero = static_cast<ScalarT>(0))
            : m_num_rows(data.size()),
              m_num_cols(0),
              m_zero(zero)
        {
            // This is not the best way to do this, but it saves a
            // bit on performance
            /*
              Naive case MN + M^2 N
              ----
              Converting to CSR : MN
              Building CscMatrix: MN
              O(2MN)
            */
            CsrMatrix<ScalarT> values(0, data, zero);
            size_t M = 0;
            size_t N = 0;
            values.get_shape(M, N);
            m_num_cols = N;
            for (IndexType col = 0; col < N; ++col)
            {
                // Have we seen a non-zero number in this column?
                bool is_first_nonzero = true;
                bool is_zero_col = true;
                for (IndexType row = 0; row < M; ++row)
                {
                    if (values[row][col] != zero)
                    {
                        is_zero_col = false;
                        m_val.push_back(values[row][col]);
                        m_row_ind.push_back(row);
                        if (is_first_nonzero)
                        {
                            m_col_ptr.push_back(m_val.size() -1);
                            is_first_nonzero = false;
                        }
                    }
                }
                if (is_zero_col)
                {
                    m_col_ptr.push_back(m_val.size());
                }
            }
            m_col_ptr.push_back(m_val.size());
        }

        /**
         * @brief Copy constructor for CscMatrix.
         *
         * @param[in] rhs  The CscMatrix to copy construct this
         *                 CscMatrix from.
         */
        CscMatrix(CscMatrix<ScalarT> const &rhs)
        {
            if (this != &rhs)
            {
                m_num_rows = rhs.m_num_rows;
                m_num_cols = rhs.m_num_cols;
                m_zero = rhs.m_zero;
                m_val = rhs.m_val;
                m_row_ind = rhs.m_row_ind;
                m_col_ptr = rhs.m_col_ptr;
            }
        }

        /**
         * @brief Copy assignment.
         *
         * @param[in] rhs  The CscMatrix to assign to this CscMatrix.
         *
         * @return *this.
         */
        CscMatrix<ScalarT>& operator=(CscMatrix<ScalarT> const &rhs)
        {
            if (this != &rhs)
            {
                m_num_rows = rhs.m_num_rows;
                m_num_cols = rhs.m_num_cols;
                m_zero = rhs.m_zero;
                m_val = rhs.m_val;
                m_row_ind = rhs.m_row_ind;
                m_col_ptr = rhs.m_col_ptr;
            }
            return *this;
        }

        /**
         * @brief Assignment from a dense matrix.
         * @param rhs The dense matrix to assign to this CscMatrix.
         * @return *this.
         */
        CscMatrix<ScalarT>& operator=(
            std::vector<std::vector<ScalarT> > const &rhs)
        {
            CscMatrix<ScalarT> rhs_Csc(rhs);
            m_num_rows = rhs_Csc.m_num_rows;
            m_num_cols = rhs_Csc.m_num_cols;
            m_zero = rhs_Csc.m_zero;
            m_val = rhs_Csc.m_val;
            m_row_ind = rhs_Csc.m_row_ind;
            m_col_ptr = rhs_Csc.m_col_ptr;
            return *this;
        }

        ~CscMatrix()
        {
        }

        // FUNCTIONS

        /**
         * @brief Get the shape for this CscMatrix.
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

        /// @return the number of stored values in this
        IndexType get_nnz() const
        {
            return m_val.size();
        }

        // EQUALITY OPERATORS
        /**
         * @brief Equality testing for CscMatrix.
         * @param rhs The right hand side of the equality operation.
         * @return If this CscMatrix and rhs are identical.
         */
        bool operator==(CscMatrix<ScalarT> const &rhs) const
        {
            if ((m_num_rows != rhs.m_num_rows) ||
                (m_num_cols != rhs.m_num_cols))
            {
                return false;
            }

            // Definitely a more efficient way than this.  Only compare
            // non-zero elements.  Then decide if compare zero's explicitly
            for (IndexType i = 0; i < m_num_rows; i++)
            {
                for (IndexType j = 0; j < m_num_cols; j++)
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
         * @brief Inequality testing for CscMatrix.
         * @param rhs The right hand side of the inequality operation.
         * @return If this CscMatrix and rhs are not identical.
         */
        bool operator!=(CscMatrix<ScalarT> const &rhs) const
        {
            return !(*this == rhs);
        }

        /**
         * @brief Access the elements of this CscMatrix given row and
         *        column indexes.
         *
         * Function provided to access the elements of this CscMatrix
         * given row and column indices.  The functionality is the
         * same as that of the indexing function for a standard dense
         * matrix.
         *
         * @param[in] row_index  The row to access.
         * @param[in] col_index  The column to access.
         *
         * @return The element of this CscMatrix at the given row and
         *         column.
         */
        ScalarT get_value_at(IndexType row_index, IndexType col_index) const
        {
            IndexType start = m_col_ptr[col_index];
            IndexType end = m_col_ptr[col_index + 1];
            for (IndexType i = start; i < end; ++i)
            {
                if (m_row_ind[i] == row_index)
                {
                    return m_val[i];
                }
            }
            return m_zero;
        }


        void set_value_at(IndexType        row,
                          IndexType        col,
                          ScalarT const &new_val)
        {
            IndexType start = m_col_ptr[col];
            IndexType end = m_col_ptr[col + 1];

            for (IndexType i = start; i < end; ++i)
            {
                if (m_row_ind[i] == row)
                {
                    if (new_val == m_zero)
                    {
                        m_val.erase(m_val.begin() + i);
                        m_row_ind.erase(m_row_ind.begin() + i);
                        for (IndexType j = col + 1;
                             j < m_col_ptr.size();
                             ++j)
                        {
                            m_col_ptr[j]--;
                        }
                        return;
                    }
                    m_val[i] = new_val;
                    return;
                }
            }

            //If we didn't find an element to delete, but new_val is zero, ignore it
            if(new_val == m_zero)
                return;


            //Wasn't already in the array, need to make space for it
            m_val.insert(m_val.begin()+start, new_val);
            m_row_ind.insert(m_row_ind.begin()+ start, row);

            for (IndexType i=col+1; i<m_col_ptr.size();i++)
            {
                m_col_ptr[i]++;
            }
        }

        /**
         * @brief Indexing function for accessing the rows of this
         *        CscMatrix.
         *
         * @param[in] row  The row to access.
         *
         * @return The row of this CscMatrix as a dense_vector.
         */
        RowView<CscMatrix<ScalarT> const> get_row(IndexType row) const
        {
            return RowView<CscMatrix<ScalarT> const>(row, *this);
        }

        RowView<CscMatrix<ScalarT> > get_row(IndexType row)
        {
            return RowView<CscMatrix<ScalarT> >(row, *this);
        }

        /**
         * @brief Indexing operator for accessing the rows of this
         *        CscMatrix.
         *
         * @param[in] row  The row to access.
         * @return The row of this CscMatrix as a dense_vector.
         */
        RowView<CscMatrix<ScalarT> const> operator[](IndexType row) const
        {
            return RowView<CscMatrix<ScalarT> const>(row, *this);
        }

        // need a non-const version for mutation.
        RowView<CscMatrix<ScalarT> > operator[](IndexType row)
        {
            return RowView<CscMatrix<ScalarT> >(row, *this);
        }

        // output specific to the storage layout of this type of matrix
        friend std::ostream& operator<<(std::ostream             &os,
                                        CscMatrix<ScalarT> const &Csc)
        {
            for (IndexType row = 0; row < Csc.m_num_rows; ++row)
            {
                os << ((row == 0) ? "[[" : " [");
                if (Csc.m_num_cols > 0)
                {
                    os << Csc.get_value_at(row, 0);
                }

                for (IndexType col = 1; col < Csc.m_num_cols; ++col)
                {
                    os << ", " << Csc.get_value_at(row, col);
                }
                os << ((row == Csc.m_num_rows - 1) ? "]]" : "]\n");
            }
#ifdef GB_DEBUG
            // Friendly debugging
            os << "\nA: ";
            for (auto a: Csc.m_val) {
                os << a << " ";
            }
            os << std::endl;

            os << "IA: ";
            for (auto ia: Csc.m_col_ptr) {
                os << ia << " ";
            }
            os << std::endl;

            os << "JA: ";
            for (auto ja: Csc.m_row_ind) {
                os << ja << " ";
            }
            os << std::endl;
#endif
            return os;
        }

    protected:
        IndexType m_num_rows;
        IndexType m_num_cols;
        ScalarT   m_zero;

    private:
        IndexArrayType m_row_ind;
        IndexArrayType m_col_ptr;
        std::vector<ScalarT> m_val;
    };
} // graphblas



#endif // GB_SEQUENTIAL_CSCMATRIX_HPP
