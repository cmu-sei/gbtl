/*
 * Copyright (c) 2017 Carnegie Mellon University.
 * All Rights Reserved.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS," WITH NO WARRANTIES WHATSOEVER. CARNEGIE
 * MELLON UNIVERSITY EXPRESSLY DISCLAIMS TO THE FULLEST EXTENT PERMITTED BY
 * LAW ALL EXPRESS, IMPLIED, AND STATUTORY WARRANTIES, INCLUDING, WITHOUT
 * LIMITATION, THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE, AND NON-INFRINGEMENT OF PROPRIETARY RIGHTS.
 *
 * This Program is distributed under a BSD license.  Please see LICENSE file or
 * permission@sei.cmu.edu for more information.  DM-0002659
 */

#ifndef GB_SEQUENTIAL_LILSPARSEMATRIX_HPP
#define GB_SEQUENTIAL_LILSPARSEMATRIX_HPP

#include <iostream>
#include <vector>
#include <typeinfo>
#include <stdexcept>

//****************************************************************************

namespace GraphBLAS
{
    namespace backend
    {

        template<typename ScalarT, typename... TagsT>
        class LilSparseMatrix
        {
        public:
            typedef ScalarT ScalarType;

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
                        m_data[ii].push_back(std::make_tuple(jj, val[ii][jj]));
                        m_nvals = m_nvals + 1;
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
                            m_data[ii].push_back(std::make_tuple(jj, val[ii][jj]));
                            m_nvals = m_nvals + 1;
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

            /// Version 1 of getshape that assigns to two passed parameters
            // void get_shape(IndexType &num_rows, IndexType &num_cols) const
            // {
            //     num_rows = m_num_rows;
            //     num_cols = m_num_cols;
            // }

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
                if (m_data.at(irow).empty())
                {
                    return false;
                }

                IndexType ind;
                ScalarT val;
                //for (auto tupl : m_data[irow])// Range-based loop, access by value
                for (auto tupl : m_data.at(irow))// Range-based loop, access by value
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
                        "get_value_at: index out of bounds");
                }
                if (m_data.empty())
                {
                    throw NoValueException("get_value_at: no entry at index");
                }
                if (m_data.at(irow).empty())
                {
                    throw NoValueException("get_value_at: no entry at index");
                }

                IndexType ind;
                ScalarT val;
                //for (auto tupl : m_data[irow])// Range-based loop, access by value
                for (auto tupl : m_data.at(irow))// Range-based loop, access by value
                {
                    //std::tie(ind, val) = tupl;
                    //if (ind == icol)
                    if (std::get<0>(tupl) == icol)
                    {
                        //return val;
                        return std::get<1>(tupl);
                    }
                }
                throw NoValueException("get_value_at: no entry at index");
            }

            // Set value at index
            void setElement(IndexType irow, IndexType icol, ScalarT const &val)
            {
                m_data[irow].reserve(m_data[irow].capacity() + 10);
                if (irow >= m_num_rows || icol >= m_num_cols)
                {
                    throw IndexOutOfBoundsException("setElement: index out of bounds");
                }

                if (m_data[irow].empty())
                {
                    m_data[irow].push_back(std::make_tuple(icol, val));
                    ++m_nvals;
                }
                if (std::get<0>(*m_data[irow].begin()) > icol)
                {
                    m_data[irow].insert(m_data[irow].begin(),
                                        std::make_tuple(icol, val));
                    ++m_nvals;
                }
                else
                {
                    typename std::vector<std::tuple<IndexType, ScalarT>>::iterator it;
                    for (it = m_data[irow].begin(); it != m_data[irow].end(); ++it)
                    {
                        if (std::get<0>(*it) == icol)
                        {
                            it = m_data[irow].erase(it);
                            m_data[irow].insert(it, std::make_tuple(icol, val));
                            return;
                        }
                        else if (std::get<0>(*it) > icol)
                        {
                            m_data[irow].insert(it, std::make_tuple(icol, val));
                            ++m_nvals;
                            return;
                        }
                    }
                    m_data[irow].push_back(std::make_tuple(icol, val));
                    m_nvals = m_nvals + 1;
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
                    m_data[irow].push_back(std::make_tuple(icol, val));
                    m_nvals = m_nvals + 1;
                }
                else
                {
                    typename std::vector<std::tuple<IndexType, ScalarT>>::iterator it;
                    for (it = m_data[irow].begin(); it != m_data[irow].end(); it++)
                    {
                        if (std::get<0>(*it) == icol)
                        {
                            // merge with existing stored value
                            std::get<1>(*it) = merge(std::get<1>(*it), val);
                            //it = m_data[irow].erase(it);
                            //m_data[irow].insert(it, std::make_tuple(icol, tmp));
                            return;
                        }
                        else if (std::get<0>(*it) > icol)
                        {
                            m_data[irow].insert(it, std::make_tuple(icol, val));
                            m_nvals = m_nvals + 1;
                            return;
                        }
                    }
                    m_data[irow].push_back(std::make_tuple(icol, val));
                    m_nvals = m_nvals + 1;
                }
            }

            typedef std::vector<std::tuple<IndexType, ScalarT>> const & RowType;
            RowType getRow(IndexType row_index) const
            {
                return m_data[row_index];
            }

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
                for (auto &tupl : row_data)
                {
                    m_data[row_index].push_back(
                        std::make_tuple(std::get<0>(tupl),
                                        static_cast<ScalarT>(std::get<1>(tupl))));
                }
            }

            // When not casting vector assignment used
            void setRow(
                IndexType row_index,
                std::vector<std::tuple<IndexType, ScalarT> > const &row_data)
            {
                IndexType old_nvals = m_data[row_index].size();
                IndexType new_nvals = row_data.size();

                m_nvals = m_nvals + new_nvals - old_nvals;
                m_data[row_index] = row_data;   // swap here?
            }

            /// @todo need move semantics.
            typedef std::vector<std::tuple<IndexType, ScalarT> > const ColType;
            ColType getCol(IndexType col_index) const
            {
                std::vector<std::tuple<IndexType, ScalarT> > data;

                for (IndexType ii = 0; ii < m_num_rows; ii++)
                {
                    if (!m_data[ii].empty())
                    {
                        for (auto tupl : m_data[ii])
                        {
                            if (std::get<0>(tupl) == col_index)
                            {
                                data.push_back(std::make_tuple(ii, std::get<1>(tupl)));
                            }
                        }
                    }
                }

                return data;
            }

            // col_data must be in increasing index order
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
                                m_data[row_index].insert(
                                    row_it,
                                    std::make_tuple(
                                        col_index,
                                        static_cast<ScalarT>(std::get<1>(*it))));
                                ++m_nvals;
                                ++it;
                                inserted = true;
                                break;
                            }
                        }
                        if (!inserted)
                        {
                            //std::cerr << "Appending new row element" << std::endl;
                            m_data[row_index].insert(
                                m_data[row_index].end(),
                                std::make_tuple(
                                    col_index,
                                    static_cast<ScalarT>(std::get<1>(*it))));
                            ++m_nvals;
                            ++it;
                        }
                    }
                    else // row_index > next entry to insert
                    {
                        // This should not happen
                        throw GraphBLAS::PanicException(
                            "LilSparseMatrix::setCol() INTERNAL ERROR");
                    }
                }

            }

            // Get column indices for a given row
            void getColumnIndices(IndexType irow, IndexArrayType &v) const
            {
                if (irow >= m_num_rows)
                {
                    throw IndexOutOfBoundsException(
                        "getColumnIndices: index out of bounds");
                }

                if (!m_data[irow].empty())
                {
                    IndexType ind;
                    ScalarT val;
                    v.resize(0);

                    for (auto tupl : m_data[irow])
                    {
                        std::tie(ind, val) = tupl;
                        v.push_back(ind);
                    }
                }
            }

            // Get row indices for a given column
            void getRowIndices(IndexType icol, IndexArrayType &v) const
            {
                if (icol >= m_num_cols)
                {
                    throw IndexOutOfBoundsException(
                        "getRowIndices: index out of bounds");
                }

                IndexType ind;
                ScalarT val;
                v.resize(0);

                for (IndexType ii = 0; ii < m_num_rows; ii++)
                {
                    if (!m_data[ii].empty())
                    {
                        for (auto tupl : m_data[ii])
                        {
                            std::tie(ind, val) = tupl;
                            if (ind == icol)
                            {
                                v.push_back(ii);
                                break;
                            }
                            if (ind > icol)
                            {
                                break;
                            }
                        }
                    }
                }
            }

            template<typename RAIteratorIT,
                     typename RAIteratorJT,
                     typename RAIteratorVT>
            ScalarT extractTuples(RAIteratorIT        row_it,
                                  RAIteratorJT        col_it,
                                  RAIteratorVT        values) const
            {
                for (IndexType row = 0; row < m_data.size(); ++row)
                {
                    for (auto it = m_data[row].begin();
                         it != m_data[row].end();
                         ++it)
                    {
                        *row_it = row;              ++row_it;
                        *col_it = std::get<0>(*it); ++col_it;
                        *values = std::get<1>(*it); ++values;
                    }
                }
            }

            // output specific to the storage layout of this type of matrix
            void printInfo(std::ostream &os) const
            {
                os << "LilSparseMatrix<" << typeid(ScalarT).name() << ">"
                   << std::endl;
                os << "dimensions: " << m_num_rows << " x " << m_num_cols
                   << std::endl;
                os << "num stored values = " << m_nvals << std::endl;
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
                                            LilSparseMatrix<ScalarT> const &mat)
            {
                mat.printInfo(os);
                return os;
            }

        private:
            IndexType m_num_rows;
            IndexType m_num_cols;
            IndexType m_nvals;

            // List-of-lists storage (LIL)
            std::vector<std::vector<std::tuple<IndexType, ScalarT>>> m_data;
        };

    } // namespace backend

} // namespace GraphBLAS

#endif // GB_SEQUENTIAL_LILSPARSEMATRIX_HPP
