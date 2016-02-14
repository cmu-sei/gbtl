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

#ifndef GB_SEQUENTIAL_LILMATRIX_HPP
#define GB_SEQUENTIAL_LILMATRIX_HPP

#include <iostream>
#include <vector>
#include <typeinfo>

#include <graphblas/system/sequential/RowView.hpp>

namespace graphblas
{
    /**
     * @brief Class representing a list of lists format sparse matrix.
     * @todo Should we support assignment by row?
     */
    template<typename ScalarT, typename... TagsT>
    class LilMatrix
    {
    public:
        typedef ScalarT ScalarType;

        /**
         * @brief Construct an empty list of lists sparse matrix with
         *        the given shape.
         *
         * @param[in] num_rows  hieght of matrix
         * @param[in] num_cols  width of matrix
         * @param[in] zero      The "zero" value.
         */
        LilMatrix(IndexType   num_rows,
                  IndexType   num_cols,
                  ScalarT const &zero = static_cast<ScalarT>(0))
            : m_num_rows(num_rows),
              m_num_cols(num_cols),
              m_zero(zero)
        {
            m_data.resize(num_rows);
        }

        /**
         * @brief Construct a lil sparse matrix from a given dense
         *        matrix.
         *
         * @param[in] values  The dense matrix from which to construct
         *                    a lil sparse matrix.
         * @param[in] zero    The "zero" value.
         */
        LilMatrix(std::vector<std::vector<ScalarT> > const &values,
                  ScalarT const &zero = static_cast<ScalarT>(0))
            : m_num_rows(values.size()),
              m_num_cols(values[0].size()),  // warning assumes all same
              m_zero(zero)
        {
            m_data.resize(m_num_rows);
            for (IndexType i = 0; i < m_num_rows; ++i)
            {
                for (IndexType j = 0; j < m_num_cols; ++j)
                {
                    if (values[i][j] != m_zero)
                    {
                        m_data[i].push_back(
                            std::make_tuple(j, values[i][j]));
                    }
                }
            }
        }

        /**
         * @brief Construct a list of lists format sparse matrix from a
         *        list of lists containing column indices and value
         *        tuples.
         *
         * @param[in] <?>   Dummy parameter to prevent ambiguity
         * @param[in] data  The lil data to use when constructing.
         * @param[in] zero  The "zero" value.
         *
         * @todo  Someone please put this constructor out of its misery.
         */
         LilMatrix(int,
                   std::vector<std::vector<std::tuple<IndexType, ScalarT> > >
                       const &data,
                   ScalarT const &zero = static_cast<ScalarT>(0))
            : m_num_rows(data.size()),
              m_num_cols(0),
              m_zero(zero),
              m_data(data)
        {
            /// @todo There is probably a better way to do this.
            for (auto row : data) //foreach?
            {
                for (auto tup : row)
                {
                    IndexType idx = std::get<0>(tup);
                    if (idx + 1 > m_num_cols)
                    {
                        m_num_cols = idx + 1;
                    }
                }
            }
        }

        /**
         * @brief Copy constructor for LilMatrix.
         *
         * @param[in] rhs  The LilMatrix to copy construct this
         *                 LilMatrix from.
         */
        LilMatrix(LilMatrix<ScalarT> const &rhs)
        {
            if (this != &rhs)
            {
                m_num_rows = rhs.m_num_rows;
                m_num_cols = rhs.m_num_cols;
                // TODO Copy zero?
                m_zero = rhs.m_zero;
                m_data = rhs.m_data;
            }
        }

        ~LilMatrix() {}

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
         * @brief Copy assignment.
         *
         * @param[in] rhs  The LilMatrix to assign to this LilMatrix.
         *
         * @return *this.
         */
        LilMatrix<ScalarT>& operator=(LilMatrix<ScalarT> const &rhs)
        {
            if (this != &rhs)
            {
                m_num_rows = rhs.m_num_rows;
                m_num_cols = rhs.m_num_cols;
                m_zero = rhs.m_zero;
                m_data = rhs.m_data;
            }
            return *this;
        }

        /**
         * @brief Assignment from a dense matrix.
         *
         * @param[in]  rhs  The dense matrix to assign to this LilMatrix.
         *
         * @return *this.
         */
        LilMatrix<ScalarT>& operator=(
            std::vector<std::vector<ScalarT> > const &rhs)
        {
            // INEFFICIENT (share implementation with constructor)
            LilMatrix<ScalarT> rhs_lil(rhs);
            m_num_rows = rhs_lil.m_num_rows;
            m_num_cols = rhs_lil.m_num_cols;
            // FIXME Assign zero element?
            m_zero = rhs_lil.m_zero;
            m_data = rhs_lil.m_data;
            return *this;
        }

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
            IndexType count(0UL);
            for (auto it = m_data.begin(); it != m_data.end(); ++it)
            {
                count += (*it).size();
            }
            return count;
        }

        // EQUALITY OPERATORS
        /**
         * @brief Equality testing for LilMatrix.
         * @param rhs The right hand side of the equality operation.
         * @return If this LilMatrix and rhs are identical.
         */
        bool operator==(LilMatrix<ScalarT> const &rhs) const
        {
            if ((m_num_rows != rhs.m_num_rows) ||
                (m_num_cols != rhs.m_num_cols))
            {
                return false;
            }

            // Definitely a more efficient way than this.  Only compare
            // non-zero elements.  Then decide if compare zero's
            // explicitly
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
         * @brief Inequality testing for LilMatrix.
         * @param rhs The right hand side of the inequality operation.
         * @return If this LilMatrix and rhs are not identical.
         */
        bool operator!=(LilMatrix<ScalarT> const &rhs) const
        {
            return !(*this == rhs);
        }

        /**
         * @brief Access the elements of this LilMatrix given row and
         *        column indexes.
         *
         * Function provided to access the elements of this LilMatrix
         * given row and column indices.  The functionality is the
         * same as that of the indexing function for a standard dense
         * matrix.
         *
         * @param[in] row_index  The row to access.
         * @param[in] col_index  The column to access.
         *
         * @return The element of this LilMatrix at the given row and
         *         column.
         */
        ScalarT get_value_at(IndexType row_index,
                             IndexType col_index) const
        {
            /// @todo assert indices within ranges?
            std::vector<std::tuple<IndexType, ScalarT> > const &row =
                m_data[row_index];

            for (auto tupl : row)
            {
                IndexType ind;
                ScalarT val;
                std::tie(ind, val) = tupl;
                if (ind == col_index)
                {
                    return val;
                }
            }

            return m_zero;
        }

        // Not certain about this implementation
        void set_value_at(IndexType      row,
                          IndexType      col,
                          ScalarT const &new_val)
        {
            typename std::vector<std::tuple<
                IndexType,
                                     ScalarType> >::iterator tup_iter;

            for (tup_iter = m_data[row].begin();
                 tup_iter != m_data[row].end();
                 ++tup_iter)
            {
                if (std::get<0>(*tup_iter) == col)
                {
                    break;
                }
            }

            // Modify an existing element.
            if (tup_iter != m_data[row].end())
            {
                if (new_val != m_zero)
                {
                    std::get<1>(*tup_iter) = new_val;
                }
                else
                {
                    // Remove the element. Writing a zero is efficient
                    // Should we store zeros or turn them into
                    //   structural zeros
                    m_data[row].erase(tup_iter);
                }
            }
            // Otherwise, append it to the end if non-zero
            else if (new_val != m_zero)
            {
                // Does this list need to be in sorted order?
                m_data[row].push_back(std::make_tuple(col, new_val));
            }
            /// @todo should we store zero?? TBD.
        }

        /**
         * @brief Indexing function for accessing the rows of this
         *        LilMatrix.
         *
         * @param[in] row  The row to access.
         *
         * @return The row of this LilMatrix as a dense_vector.
         */
        RowView<LilMatrix<ScalarT> const> get_row(IndexType row) const
        {
            return RowView<LilMatrix<ScalarT> const>(row, *this);
        }

        RowView<LilMatrix<ScalarT> > get_row(IndexType row)
        {
            return RowView<LilMatrix<ScalarT> >(row, *this);
        }

        /**
         * @brief Indexing operator for accessing the rows of this
         *        LilMatrix.
         *
         * @param[in] row  The row to access.
         * @return The row of this LilMatrix as a dense_vector.
         */
        RowView<LilMatrix<ScalarT> const>
        operator[](IndexType row) const
        {
            return RowView<LilMatrix<ScalarT> const>(row, *this);
        }

        // need a non-const version for mutation.
        RowView<LilMatrix<ScalarT> >
        operator[](IndexType row)
        {
            return RowView<LilMatrix<ScalarT> >(row, *this);
        }

        // output specific to the storage layout of this type of matrix
        void print_info(std::ostream &os) const
        {
            os << "LilMatrix<" << typeid(ScalarT).name() << ">" << std::endl;
            os << "dimensions: " << m_num_rows << " x " << m_num_cols
               << std::endl;
            os << "num nonzeros = " << get_nnz() << std::endl;
            os << "structural zero value = " << get_zero() << std::endl;
            for (IndexType row = 0; row < m_data.size(); ++row)
            {
                os << row << " :";
                for (auto it = m_data[row].begin();
                     it != m_data[row].end();
                     ++it)
                {
                    os << " " << std::get<0>(*it)
                       << ":" << std::get<1>(*it);
                }
                os << std::endl;
            }
        }

        friend std::ostream &operator<<(std::ostream             &os,
                                        LilMatrix<ScalarT> const &mat)
        {
            mat.print_info(os);
            return os;
        }

    private:
        IndexType m_num_rows;
        IndexType m_num_cols;
        ScalarT   m_zero;

        std::vector<std::vector<std::tuple<IndexType, ScalarT> > > m_data;
    };

/*
    template<typename ScalarT, typename... TagsT>
    std::pair<IndexType, IndexType>
    get_shape(LilMatrix<ScalarT, TagsT...> const &m) {
        std::pair<IndexType, IndexType> ret;
        m.get_shape(ret.first, ret.second);
        return ret;
    }

    template<typename ScalarT, typename... TagsT>
    IndexType get_nnz(LilMatrix<ScalarT, TagsT...> const &m) {
        return m.get_nnz();
    }
*/
} // graphblas



#endif // GB_SEQUENTIAL_LILMATRIX_HPP
