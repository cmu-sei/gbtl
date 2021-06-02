/*
 * GraphBLAS Template Library (GBTL), Version 3.0
 *
 * Copyright 2020 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors.
 *
 * THIS MATERIAL WAS PREPARED AS AN ACCOUNT OF WORK SPONSORED BY AN AGENCY OF
 * THE UNITED STATES GOVERNMENT.  NEITHER THE UNITED STATES GOVERNMENT NOR THE
 * UNITED STATES DEPARTMENT OF ENERGY, NOR THE UNITED STATES DEPARTMENT OF
 * DEFENSE, NOR CARNEGIE MELLON UNIVERSITY, NOR BATTELLE, NOR ANY OF THEIR
 * EMPLOYEES, NOR ANY JURISDICTION OR ORGANIZATION THAT HAS COOPERATED IN THE
 * DEVELOPMENT OF THESE MATERIALS, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS,
 * OR USEFULNESS OR ANY INFORMATION, APPARATUS, PRODUCT, SOFTWARE, OR PROCESS
 * DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED
 * RIGHTS.
 *
 * Released under a BSD-style license, please see LICENSE file or contact
 * permission@sei.cmu.edu for full terms.
 *
 * [DISTRIBUTION STATEMENT A] This material has been approved for public release
 * and unlimited distribution.  Please see Copyright notice for non-US
 * Government use and distribution.
 *
 * DM20-0442
 */

#pragma once

#include <iostream>
#include <vector>
#include <typeinfo>
#include <stdexcept>
#include <algorithm>

#include <graphblas/graphblas.hpp>

//****************************************************************************

namespace grb
{
    namespace backend
    {

        template<typename ScalarT, typename... TagsT>
        class LilSparseMatrix
        {
        public:
            using ScalarType = ScalarT;
            using ElementType = std::tuple<IndexType, ScalarT>;
            using RowType = std::vector<ElementType>;

            // Constructor
            LilSparseMatrix(IndexType num_rows,
                            IndexType num_cols)
                : m_num_rows(num_rows),
                  m_num_cols(num_cols),
                  m_nvals(0)
            {
                m_data.resize(m_num_rows);
            }

            // Constructor - copy
            LilSparseMatrix(LilSparseMatrix<ScalarT> const &rhs)
                : m_num_rows(rhs.m_num_rows),
                  m_num_cols(rhs.m_num_cols),
                  m_nvals(rhs.m_nvals),
                  m_data(rhs.m_data)
            {
            }

            // Constructor - dense from dense matrix
            LilSparseMatrix(std::vector<std::vector<ScalarT>> const &val)
                : m_num_rows(val.size()),
                  m_num_cols(val[0].size())
            {
                m_data.resize(m_num_rows);
                m_nvals = 0;
                for (IndexType ii = 0; ii < m_num_rows; ii++)
                {
                    if (val[ii].size() != m_num_cols)
                    {
                        throw DimensionException("LilSparseMatix(dense ctor)");
                    }

                    for (IndexType jj = 0; jj < m_num_cols; jj++)
                    {
                        m_data[ii].emplace_back(jj, val[ii][jj]);
                        ++m_nvals;
                    }
                }
            }

            // Constructor - sparse from dense matrix, removing specifed implied zeros
            LilSparseMatrix(std::vector<std::vector<ScalarT>> const &val,
                            ScalarT zero)
                : m_num_rows(val.size()),
                  m_num_cols(val[0].size())
            {
                m_data.resize(m_num_rows);
                m_nvals = 0;
                for (IndexType ii = 0; ii < m_num_rows; ii++)
                {
                    if (val[ii].size() != m_num_cols)
                    {
                        throw DimensionException("LilSparseMatix(dense ctor)");
                    }

                    for (IndexType jj = 0; jj < m_num_cols; jj++)
                    {
                        if (val[ii][jj] != zero)
                        {
                            m_data[ii].emplace_back(jj, val[ii][jj]);
                            ++m_nvals;
                        }
                    }
                }
            }

            // Destructor
            ~LilSparseMatrix()
            {}

            // Assignment (currently restricted to same dimensions)
            LilSparseMatrix<ScalarT> &operator=(LilSparseMatrix<ScalarT> const &rhs)
            {
                if (this != &rhs)
                {
                    // push this check to frontend
                    if ((m_num_rows != rhs.m_num_rows) ||
                        (m_num_cols != rhs.m_num_cols))
                    {
                        throw DimensionException();
                    }

                    m_nvals = rhs.m_nvals;
                    m_data = rhs.m_data;
                }
                return *this;
            }

            // EQUALITY OPERATORS
            /**
             * @brief Equality testing for LilMatrix.
             * @param rhs The right hand side of the equality operation.
             * @return If this LilMatrix and rhs are identical.
             */
            bool operator==(LilSparseMatrix<ScalarT> const &rhs) const
            {
                return ((m_num_rows == rhs.m_num_rows) &&
                        (m_num_cols == rhs.m_num_cols) &&
                        (m_nvals == rhs.m_nvals) &&
                        (m_data == rhs.m_data));
            }

            /**
             * @brief Inequality testing for LilMatrix.
             * @param rhs The right hand side of the inequality operation.
             * @return If this LilMatrix and rhs are not identical.
             */
            bool operator!=(LilSparseMatrix<ScalarT> const &rhs) const
            {
                return !(*this == rhs);
            }

            template<typename RAIteratorI,
                     typename RAIteratorJ,
                     typename RAIteratorV,
                     typename DupT>
            void build(RAIteratorI  i_it,
                       RAIteratorJ  j_it,
                       RAIteratorV  v_it,
                       IndexType    n,
                       DupT         dup)
            {
                /// @todo should this function throw an error if matrix is not empty

                /// @todo should this function call clear?
                //clear();

                /// @todo DOING SOMETHING REALLY STUPID RIGHT NOW
                for (IndexType ix = 0; ix < n; ++ix)
                {
                    setElement(*i_it, *j_it, *v_it, dup);
                    ++i_it; ++j_it; ++v_it;
                }
            }

            void clear()
            {
                /// @todo make atomic? transactional?
                m_nvals = 0;
                for (IndexType row = 0; row < m_data.size(); ++row)
                {
                    m_data[row].clear();
                }
            }

            IndexType nrows() const { return m_num_rows; }
            IndexType ncols() const { return m_num_cols; }
            IndexType nvals() const { return m_nvals; }

            /**
             * @brief Resize the matrix dimensions (smaller or larger)
             *
             * @param[in]  new_num_rows  New number of rows (zero is invalid)
             * @param[in]  new_num_cols  New number of columns (zero is invalid)
             *
             */
            void resize(IndexType new_num_rows, IndexType new_num_cols)
            {
                // Invalid values check by frontend
                //if ((new_num_rows == 0) || (new_num_cols == 0))
                //    throw InvalidValueException();

                // *******************************************
                // Step 1: Deal with number of rows
                m_data.resize(new_num_rows);

                // Count how many elements are left when num_rows reduces
                if (new_num_rows < m_num_rows)
                {
                    m_nvals = 0UL;
                    for (auto const &row : m_data)
                        m_nvals += row.size();
                }
                m_num_rows = new_num_rows;

                // *******************************************
                // Step 2: Deal with number columns
                // Need to do nothing if size stays the same or increases
                if (new_num_cols < m_num_cols)
                {
                    // Need to eliminate any entries beyond new limit
                    // when decreasing
                    for (auto &row : m_data)
                    {
                        if (!row.empty())
                        {
                            auto it(row.begin());
                            for ( ; ((it != row.end()) &&
                                     (std::get<0>(*it) < new_num_cols)); ++it)
                            {
                            }

                            if (it != row.end())
                            {
                                IndexType nval(row.size());
                                row.erase(it, row.end());
                                m_nvals -= (nval - row.size()); // adjust nvals
                            }
                        }
                    }
                }
                m_num_cols = new_num_cols;
            }

            bool hasElement(IndexType irow, IndexType icol) const
            {
                if (irow >= m_num_rows || icol >= m_num_cols)
                {
                    throw IndexOutOfBoundsException(
                        "get_value_at: index out of bounds");
                }
                if (m_data.empty())
                {
                    return false;
                }
                if (m_data[irow].empty())
                {
                    return false;
                }

                for (auto tupl : m_data[irow])// Range-based loop, access by value
                {
                    if (std::get<0>(tupl) == icol)
                    {
                        return true;
                    }
                }
                return false;
            }

            // Get value at index
            ScalarT extractElement(IndexType irow, IndexType icol) const
            {
                if (irow >= m_num_rows || icol >= m_num_cols)
                {
                    throw IndexOutOfBoundsException(
                        "extractElement: index out of bounds");
                }
                if (m_data.empty())
                {
                    throw NoValueException("extractElement: no data");
                }
                if (m_data[irow].empty())
                {
                    throw NoValueException("extractElement: no data in row");
                }

                for (auto&& [idx, val] : m_data[irow])
                {
                    if (idx == icol)
                    {
                        return val;
                    }
                }
                throw NoValueException("extractElement: no entry at index");
            }

            // Set value at index
            void setElement(IndexType irow, IndexType icol, ScalarT const &val)
            {
                //m_data[irow].reserve(m_data[irow].capacity() + 10);
                if (irow >= m_num_rows || icol >= m_num_cols)
                {
                    throw IndexOutOfBoundsException("setElement: index out of bounds");
                }

                if (m_data[irow].empty())
                {
                    m_data[irow].emplace_back(icol, val);
                    ++m_nvals;
                }
                else
                {
                    for (auto it = m_data[irow].begin();
                         it != m_data[irow].end();
                         ++it)
                    {
                        if (std::get<0>(*it) == icol)
                        {
                            // overwrite existing stored value
                            std::get<1>(*it) = val;
                            return;
                        }
                        else if (std::get<0>(*it) > icol)
                        {
                            m_data[irow].emplace(it, icol, val);
                            ++m_nvals;
                            return;
                        }
                    }
                    m_data[irow].emplace_back(icol, val);
                    ++m_nvals;
                }
            }

            // Set value at index + 'merge' with any existing value
            // according to the BinaryOp passed.
            template <typename BinaryOpT>
            void setElement(IndexType irow, IndexType icol, ScalarT const &val,
                            BinaryOpT merge)
            {
                if (irow >= m_num_rows || icol >= m_num_cols)
                {
                    throw IndexOutOfBoundsException(
                        "setElement(merge): index out of bounds");
                }

                if (m_data[irow].empty())
                {
                    m_data[irow].emplace_back(icol, val);
                    ++m_nvals;
                }
                else
                {
                    for (auto it = m_data[irow].begin();
                         it != m_data[irow].end();
                         ++it)
                    {
                        if (std::get<0>(*it) == icol)
                        {
                            // merge with existing stored value
                            std::get<1>(*it) = merge(std::get<1>(*it), val);
                            return;
                        }
                        else if (std::get<0>(*it) > icol)
                        {
                            m_data[irow].emplace(it, icol, val);
                            ++m_nvals;
                            return;
                        }
                    }
                    m_data[irow].emplace_back(icol, val);
                    ++m_nvals;
                }
            }

            void removeElement(IndexType irow, IndexType icol)
            {
                if (irow >= m_num_rows || icol >= m_num_cols)
                {
                    throw IndexOutOfBoundsException("removeElement: index out of bounds");
                }

                /// @todo Replace with binary_search
                auto it = std::find_if(
                    m_data[irow].begin(), m_data[irow].end(),
                    [&icol](ElementType const &elt) { return icol == std::get<0>(elt); });

                if (it != m_data[irow].end())
                {
                    --m_nvals;
                    m_data[irow].erase(it);
                }
            }

            void recomputeNvals()
            {
                IndexType nvals(0);

                for (auto const &elt : m_data)
                {
                    nvals += elt.size();
                }
                m_nvals = nvals;
            }

            // TODO: add error checking on dimensions?
            void swap(LilSparseMatrix<ScalarT> &rhs)
            {
                for (IndexType idx = 0; idx < m_data.size(); ++idx)
                {
                    m_data[idx].swap(rhs.m_data[idx]);
                }
                m_nvals = rhs.m_nvals;
            }

            // Row access
            // Warning if you use this non-const row accessor then you should
            // call recomputeNvals() at some point to fix it
            RowType &operator[](IndexType row_index) { return m_data[row_index]; }

            RowType const &operator[](IndexType row_index) const
            {
                return m_data[row_index];
            }

            // RowType const &getRow(IndexType row_index) const
            // {
            //     return m_data[row_index];
            // }

            // Allow casting
            template <typename OtherScalarT>
            void setRow(
                IndexType row_index,
                std::vector<std::tuple<IndexType, OtherScalarT> > const &row_data)
            {
                IndexType old_nvals = m_data[row_index].size();
                IndexType new_nvals = row_data.size();

                m_nvals = m_nvals + new_nvals - old_nvals;
                //m_data[row_index] = row_data;   // swap here?
                m_data[row_index].clear();
                for (auto&& [idx, val] : row_data)
                {
                    m_data[row_index].emplace_back(idx, static_cast<ScalarT>(val));
                }
            }

            // When not casting vector swap used...should we use move semantics?
            void setRow(
                IndexType row_index,
                std::vector<std::tuple<IndexType, ScalarT> > &&row_data)
            {
                IndexType old_nvals = m_data[row_index].size();
                IndexType new_nvals = row_data.size();

                m_nvals = m_nvals + new_nvals - old_nvals;
                m_data[row_index].swap(row_data); // = row_data;
            }


            // Allow casting. TODO Do we need one that does not need casting?
            // mergeRow with no accumulator is same as setRow
            template <typename OtherScalarT, typename AccumT>
            void mergeRow(
                IndexType row_index,
                std::vector<std::tuple<IndexType, OtherScalarT> > &row_data,
                NoAccumulate const &op)
            {
                setRow(row_index, row_data);
            }


            // Allow casting. TODO Do we need one that does not need casting?
            template <typename OtherScalarT, typename AccumT>
            void mergeRow(
                IndexType row_index,
                std::vector<std::tuple<IndexType, OtherScalarT> > &row_data,
                AccumT const &op)
            {
                if (row_data.empty()) return;
                if (m_data[row_index].empty())
                {
                    setRow(row_index, row_data);
                    return;
                }

                std::vector<std::tuple<IndexType, ScalarT> > tmp;
                auto l_it(m_data[row_index].begin());
                auto r_it(row_data.begin());
                while ((l_it != m_data[row_index].end()) &&
                       (r_it != row_data.end()))
                {
                    IndexType li = std::get<0>(*l_it);
                    IndexType ri = std::get<0>(*r_it);
                    if (li < ri)
                    {
                        tmp.emplace_back(*l_it);
                        ++l_it;
                    }
                    else if (ri < li)
                    {
                        tmp.emplace_back(
                            ri, static_cast<ScalarT>(std::get<1>(*r_it)));
                        ++r_it;
                    }
                    else
                    {
                        tmp.emplace_back(
                            li, static_cast<ScalarT>(op(std::get<1>(*l_it),
                                                        std::get<1>(*r_it))));
                        ++l_it;
                        ++r_it;
                    }
                }

                while (l_it != m_data[row_index].end())
                {
                    tmp.emplace_back(*l_it);  ++l_it;
                }

                while (r_it != row_data.end())
                {
                    tmp.emplace_back(*r_it);  ++r_it;
                }

                setRow(row_index, tmp);
            }

            /// @deprecated Only needed for 4.3.7.3 assign: column variant"
            /// @todo need move semantics.
            using ColType = std::vector<std::tuple<IndexType, ScalarT> >;
            ColType getCol(IndexType col_index) const
            {
                std::vector<std::tuple<IndexType, ScalarT> > data;

                for (IndexType ii = 0; ii < m_num_rows; ii++)
                {
                    if (!m_data[ii].empty())
                    {
                        /// @todo replace with binary_search
                        for (auto&& [idx, val] : m_data[ii])
                        {
                            if (idx == col_index)
                            {
                                data.emplace_back(ii, val);
                            }
                        }
                    }
                }

                return data;  // hopefully compiles to a move
            }

            /// @deprecated Only needed for 4.3.7.3 assign: column variant"
            /// @note col_data must be in increasing index order
            /// @todo this could be vastly improved.
            template <typename OtherScalarT>
            void setCol(
                IndexType col_index,
                std::vector<std::tuple<IndexType, OtherScalarT> > const &col_data)
            {
                auto it = col_data.begin();
                for (IndexType row_index = 0; row_index < m_num_rows; row_index++)
                {
                    // Check for any values to clear: either there are column entries
                    // left to examine, or the index is less than the next one to
                    // insert

                    // No value to insert in this row.
                    if ((it == col_data.end()) || (row_index < std::get<0>(*it)))
                    {
                        for (auto row_it = m_data[row_index].begin();
                             row_it != m_data[row_index].end();
                             ++row_it)
                        {
                            if (std::get<0>(*row_it) == col_index)
                            {
                                //std::cerr << "Erasing row element" << std::endl;
                                m_data[row_index].erase(row_it);
                                --m_nvals;
                                break;
                            }
                        }
                    }
                    // replace existing or insert
                    else if (row_index == std::get<0>(*it))
                    {
                        //std::cerr << "Row index matches col_data row" << std::endl;
                        bool inserted=false;
                        for (auto row_it = m_data[row_index].begin();
                             row_it != m_data[row_index].end();
                             ++row_it)
                        {
                            if (std::get<0>(*row_it) == col_index)
                            {
                                //std::cerr << "Found row element to replace" << std::endl;
                                // replace
                                std::get<1>(*row_it) =
                                    static_cast<ScalarT>(std::get<1>(*it));
                                ++it;
                                inserted = true;
                                break;
                            }
                            else if (std::get<0>(*row_it) > col_index)
                            {
                                //std::cerr << "Inserting new row element" << std::endl;
                                m_data[row_index].emplace(
                                    row_it,
                                    col_index,
                                    static_cast<ScalarT>(std::get<1>(*it)));
                                ++m_nvals;
                                ++it;
                                inserted = true;
                                break;
                            }
                        }
                        if (!inserted)
                        {
                            //std::cerr << "Appending new row element" << std::endl;
                            m_data[row_index].emplace_back(
                                col_index,
                                static_cast<ScalarT>(std::get<1>(*it)));
                            ++m_nvals;
                            ++it;
                        }
                    }
                    else // row_index > next entry to insert
                    {
                        // This should not happen
                        throw grb::PanicException(
                            "LilSparseMatrix::setCol() INTERNAL ERROR");
                    }
                }

            }

            // Get column indices for a given row
            // void getColumnIndices(IndexType irow, IndexArrayType &v) const
            // {
            //     if (irow >= m_num_rows)
            //     {
            //         throw IndexOutOfBoundsException(
            //             "getColumnIndices: index out of bounds");
            //     }

            //     if (!m_data[irow].empty())
            //     {
            //         v.clear();

            //         for (auto&& [ind, val] : m_data[irow])
            //         {
            //             v.emplace_back(ind);
            //         }
            //     }
            // }

            // Get row indices for a given column
            // void getRowIndices(IndexType icol, IndexArrayType &v) const
            // {
            //     if (icol >= m_num_cols)
            //     {
            //         throw IndexOutOfBoundsException(
            //             "getRowIndices: index out of bounds");
            //     }

            //     v.clear();

            //     for (IndexType ii = 0; ii < m_num_rows; ii++)
            //     {
            //         if (!m_data[ii].empty())
            //         {
            //             /// @todo replace with binary_search
            //             for (auto&& [ind, val] : m_data[ii])
            //             {
            //                 if (ind == icol)
            //                 {
            //                     v.emplace_back(ii);
            //                     break;
            //                 }
            //                 if (ind > icol)
            //                 {
            //                     break;
            //                 }
            //             }
            //         }
            //     }
            // }

            template<typename RAIteratorIT,
                     typename RAIteratorJT,
                     typename RAIteratorVT>
            void extractTuples(RAIteratorIT        row_it,
                               RAIteratorJT        col_it,
                               RAIteratorVT        values) const
            {
                for (IndexType row = 0; row < m_data.size(); ++row)
                {
                    for (auto&& [col_idx, val] : m_data[row])
                    {
                        *row_it = row;     ++row_it;
                        *col_it = col_idx; ++col_it;
                        *values = val;     ++values;
                    }
                }
            }

            // output specific to the storage layout of this type of matrix
            void printInfo(std::ostream &os) const
            {
                os << "backend::LilSparseMatrix<" << typeid(ScalarT).name() << "> ";
                os << "(" << m_num_rows << " x " << m_num_cols << "), nvals = "
                   << nvals() << std::endl;

                // Used to print data in storage format instead of like a matrix
                #ifdef GRB_MATRIX_PRINT_RAW_STORAGE
                    for (IndexType row = 0; row < m_data.size(); ++row)
                    {
                        os << row << " :";
                        for (auto&& [idx, val] : m_data[row])
                        {
                            os << " " << idx << ":" << val;
                        }
                        os << std::endl;
                    }
                #else
                    for (IndexType row_idx = 0; row_idx < m_num_rows; ++row_idx)
                    {
                        // We like to start with a little whitespace indent
                        os << ((row_idx == 0) ? "  [[" : "   [");

                        RowType const &row(m_data[row_idx]);
                        IndexType curr_idx = 0;

                        if (row.empty())
                        {
                            while (curr_idx < m_num_cols)
                            {
                                os << ((curr_idx == 0) ? " " : ",  " );
                                ++curr_idx;
                            }
                        }
                        else
                        {
                            // Now walk the columns.  A sparse iter would be handy here...
                            auto row_it = row.begin();
                            while (row_it != row.end())
                            {
                                auto&& [col_idx, cell_val] = *row_it;
                                while (curr_idx < col_idx)
                                {
                                    os << ((curr_idx == 0) ? " " : ",  " );
                                    ++curr_idx;
                                }

                                if (curr_idx != 0)
                                    os << ", ";
                                os << cell_val;

                                ++row_it;
                                ++curr_idx;
                            }

                            // Fill in the rest to the end
                            while (curr_idx < m_num_cols)
                            {
                                os << ",  ";
                                ++curr_idx;
                            }
                        }
                        os << ((row_idx == m_num_rows - 1 ) ? "]]" : "]\n");
                    }
                #endif
            }

            friend std::ostream &operator<<(std::ostream             &os,
                                            LilSparseMatrix<ScalarT> const &mat)
            {
                mat.printInfo(os);
                return os;
            }

        private:
            IndexType m_num_rows;
            IndexType m_num_cols;
            IndexType m_nvals;

            // List-of-lists storage (LIL) really VOV
            std::vector<RowType> m_data;
        };

    } // namespace backend

} // namespace grb
