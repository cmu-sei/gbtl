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

#ifndef GB_SEQUENTIAL_CSRMATRIX_HPP
#define GB_SEQUENTIAL_CSRMATRIX_HPP

#include <iostream>
#include <vector>

//#include <graphblas/system/sequential/RowView.hpp>

namespace graphblas
{
    /**
     * @brief Class representing a Csr format matrix.
     */
    template<typename ScalarT>
    class CsrMatrix
    {
    public:

        // TYPEDEFS
        typedef ScalarT ScalarType;

        // CONSTRUCTORS
        /**
         * @brief Construct an empty csr matrix with
         *        the given shape.
         *
         * @param[in] num_rows  The height of the matrix
         * @param[in] num_cols  The width of the matrix
         * @param[in] zero      The "structural zero" value.
         */
        CsrMatrix(IndexType num_rows,
                  IndexType num_cols,
                  ScalarT   zero = static_cast<ScalarT>(0))
            : m_num_rows(num_rows),
              m_num_cols(num_cols),
              m_zero(zero),
              m_row_ptr(num_rows+1, 0)
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
         * @brief Construct a csr matrix from a given dense
         *        matrix.
         *
         * @param[in] values  The dense matrix from which to construct
         *                    a csr sparse matrix.
         * @param[in] zero    The "zero" value.
         */
        CsrMatrix(std::vector<std::vector<ScalarT> > const &values,
                  ScalarT  zero = static_cast<ScalarT>(0))
            : m_num_rows(values.size()),
              m_num_cols(values[0].size()),  // warning assumes all same
              m_zero(zero)
        {
            for (IndexType i = 0; i < values.size(); i++)
            {
                bool is_first_nonzero = true; // Have we seen a non-zero number in this row?
                bool is_zero_row = true;
                for (IndexType j = 0; j < values[0].size(); j++)
                {
                    if (values[i][j] != zero)
                    {
                        is_zero_row = false;
                        m_val.push_back(values[i][j]);
                        m_col_ind.push_back(j);
                        if (is_first_nonzero)
                        {
                            m_row_ptr.push_back(m_val.size() -1);
                            is_first_nonzero = false;
                        }
                    }
                }
                if (is_zero_row)
                {
                    m_row_ptr.push_back(m_val.size());
                }
            }
            m_row_ptr.push_back(m_val.size());
        }

        /**
         * @brief Construct a csr format matrix from a
         *        list of lists containing column indices and value
         *        tuples.
         *
         * @param[in] data  The csr data to use when constructing.
         * @param[in] zero  The "zero" value.
         */
         CsrMatrix(int,
            std::vector<std::vector<std::tuple<IndexType, ScalarT> > > const &data,
            ScalarT    zero = static_cast<ScalarT>(0))
            : m_num_rows(data.size()),
              m_num_cols(0),
              m_zero(zero)
        {
            IndexType nnz = 0;
            for (auto row: data)
            {
                m_row_ptr.push_back(m_val.size());
                for (auto tup: row)
                {
                    IndexType col;
                    ScalarT val;
                    std::tie(col, val) = tup;
                    //This is an invalid assumption when final column(s) are
                    // all zeros
                    m_num_cols = std::max(m_num_cols, col+1);
                    if (val != zero)
                    {
                        m_col_ind.push_back(col);
                        m_val.push_back(val);
                        nnz++;
                    }
                }
            }

            m_row_ptr.push_back(nnz);
        }

        /**
         * @brief Copy constructor for CsrMatrix.
         *
         * @param[in] rhs  The CsrMatrix to copy construct this
         *                 CsrMatrix from.
         */
        CsrMatrix(CsrMatrix<ScalarT> const &rhs)
        {
            if (this != &rhs)
            {
                m_num_rows = rhs.m_num_rows;
                m_num_cols = rhs.m_num_cols;
                m_zero = rhs.m_zero;
                m_val = rhs.m_val;
                m_col_ind = rhs.m_col_ind;
                m_row_ptr = rhs.m_row_ptr;
            }
        }

        /**
         * @brief Copy assignment.
         *
         * @param[in] rhs  The CsrMatrix to assign to this CsrMatrix.
         *
         * @return *this.
         */
        CsrMatrix<ScalarT>& operator=(CsrMatrix<ScalarT> const &rhs)
        {
            if (this != &rhs)
            {
                m_num_rows = rhs.m_num_rows;
                m_num_cols = rhs.m_num_cols;
                m_zero = rhs.m_zero;
                m_val = rhs.m_val;
                m_col_ind = rhs.m_col_ind;
                m_row_ptr = rhs.m_row_ptr;
            }
            return *this;
        }

        /**
         * @brief Assignment from a dense matrix.
         * @param rhs The dense matrix to assign to this CsrMatrix.
         * @return *this.
         */
        CsrMatrix<ScalarT>& operator=(
            std::vector<std::vector<ScalarT> > const &rhs)
        {
            CsrMatrix<ScalarT> rhs_csr(rhs);
            m_num_rows = rhs_csr.m_num_rows;
            m_num_cols = rhs_csr.m_num_cols;
            m_zero = rhs_csr.m_zero;
            m_val = rhs_csr.m_val;
            m_col_ind = rhs_csr.m_col_ind;
            m_row_ptr = rhs_csr.m_row_ptr;
            return *this;
        }

        ~CsrMatrix()
        {
        }

        // FUNCTIONS

        /**
         * @brief Get the shape for this CsrMatrix.
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
            /*
             * FIXME Should we go through and first convert all
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
         * @brief Equality testing for CsrMatrix.
         * @param rhs The right hand side of the equality operation.
         * @return If this CsrMatrix and rhs are identical.
         */
        bool operator==(CsrMatrix<ScalarT> const &rhs) const
        {
            if ((m_num_rows != rhs.m_num_rows) ||
                (m_num_cols != rhs.m_num_cols))
            {
                return false;
            }

            // Definitely a more efficient way than this.  Only
            // compare non-zero elements.  Then decide if compare
            // zero's explicitly
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
         * @brief Inequality testing for CsrMatrix.
         * @param rhs The right hand side of the inequality operation.
         * @return If this CsrMatrix and rhs are not identical.
         */
        bool operator!=(CsrMatrix<ScalarT> const &rhs) const
        {
            return !(*this == rhs);
        }

        /**
         * @brief Access the elements of this CsrMatrix given row and
         *        column indexes.
         *
         * Function provided to access the elements of this CsrMatrix
         * given row and column indices.  The functionality is the
         * same as that of the indexing function for a standard dense
         * matrix.
         *
         * @param[in] row_index  The row to access.
         * @param[in] col_index  The column to access.
         *
         * @return The element of this CsrMatrix at the given row and
         *         column.
         */
        ScalarT get_value_at(IndexType row_index,
                             IndexType col_index) const
        {
            IndexType start = m_row_ptr[row_index];
            IndexType end = m_row_ptr[row_index + 1];
            for (IndexType i = start; i < end; ++i)
            {
                if (m_col_ind[i] == col_index)
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
            IndexType start = m_row_ptr[row];
            IndexType end = m_row_ptr[row + 1];

            for (IndexType i = start; i < end; ++i)
            {
                if (m_col_ind[i] == col)
                {
                    if (new_val == m_zero)
                    {
                        m_val.erase(m_val.begin()+i);
                        m_col_ind.erase(m_col_ind.begin()+ i);
                        for (IndexType j = row + 1;
                             j < m_row_ptr.size();
                             j++)
                        {
                            m_row_ptr[j]--;
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
            m_val.insert(m_val.begin() + start, new_val);
            m_col_ind.insert(m_col_ind.begin() + start, col);

            for (IndexType i = row + 1; i < m_row_ptr.size(); ++i)
            {
                m_row_ptr[i]++;
            }
        }

        /**
         * @brief Indexing function for accessing the rows of this
         *        CsrMatrix.
         *
         * @param[in] row  The row to access.
         *
         * @return The row of this CsrMatrix as a dense_vector.
         */
        // RowView<CsrMatrix<ScalarT> const> get_row(IndexType row) const
        // {
        //     return RowView<CsrMatrix<ScalarT> const>(row, *this);
        // }

        // RowView<CsrMatrix<ScalarT> > get_row(IndexType row)
        // {
        //     return RowView<CsrMatrix<ScalarT> >(row, *this);
        // }

        /**
         * @brief Indexing operator for accessing the rows of this
         *        CsrMatrix.
         *
         * @param[in] row  The row to access.
         * @return The row of this CsrMatrix as a dense_vector.
         */
        // RowView<CsrMatrix<ScalarT> const> operator[](IndexType row) const
        // {
        //     return RowView<CsrMatrix<ScalarT> const>(row, *this);
        // }

        // need a non-const version for mutation.
        // RowView<CsrMatrix<ScalarT> > operator[](IndexType row)
        // {
        //     return RowView<CsrMatrix<ScalarT> >(row, *this);
        // }

        // output specific to the storage layout of this type of matrix
        void print_info(std::ostream &os) const
        {
            os << "CsrMatrix<" << typeid(ScalarT).name() << ">" << std::endl;
            os << "dimension: " << m_num_rows << " x " << m_num_cols
               << std::endl;
            os << "num nonzeros = " << get_nnz() << std::endl;
            os << "structural zero value = " << get_zero() << std::endl;

            os << "col_ind: ";
            for (auto ja: m_col_ind) {
                os << ja << " ";
            }
            os << std::endl;

            os << "row_ptr: ";
            for (auto ia: m_row_ptr) {
                os << ia << " ";
            }
            os << std::endl;

            os << "val: ";
            for (auto a: m_val) {
                os << a << " ";
            }
            os << std::endl;
        }

        friend std::ostream& operator<<(std::ostream             &os,
                                        CsrMatrix<ScalarT> const &csr)
        {
            csr.print_info(os);
            return os;
        }

    private:
        IndexType m_num_rows;
        IndexType m_num_cols;
        ScalarT   m_zero;
        IndexArrayType m_col_ind;
        IndexArrayType m_row_ptr;
        std::vector<ScalarT> m_val;
    };
} // graphblas



#endif // GB_SEQUENTIAL_CSRMATRIX_HPP
