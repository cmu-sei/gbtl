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
#include <graphblas/indices.hpp>

//****************************************************************************

namespace grb
{
    namespace backend
    {

        template<typename ScalarT, typename... TagsT>
        class GKCMatrix
        {
        public:
            using ScalarType = ScalarT;
            using WeightType = ScalarT;
            using ElementType = std::tuple<IndexType, ScalarT>;
            using RowType = std::vector<ElementType>;

            // Constructor with some default values
            GKCMatrix(IndexType num_rows,
                      IndexType num_cols, 
                      IndexType num_edges=0,
                      bool weighted = true)
                : m_num_rows(num_rows),
                  m_num_cols(num_cols),
                  m_num_edges(num_edges),
                  m_weighted(weighted)
            {
                // This is a CSR matrix, so the number of neighborhood
                // offsets is equal to 1 + the number of rows, not cols.
                m_offsets.resize(m_num_rows+1);
                // Allocate space for the initial number of edges if not zero.
                if (m_num_edges > 0){
                    m_neighbors.resize(m_num_edges);
                    // Similarly allocate space for edges if weighted
                    if (m_weighted)
                        m_weights.resize(m_num_edges);
                }
            }

            // Constructor - copy
            GKCMatrix(GKCMatrix<ScalarT> const &rhs)
                : m_num_rows(rhs.m_num_rows),
                  m_num_cols(rhs.m_num_cols),
                  m_num_edges(rhs.m_num_edges),
                  m_weighted(rhs.m_weighted),
                  m_offsets(rhs.m_offsets),
                  m_neighbors(rhs.m_neighbors),
                  m_weights(rhs.m_weights)
            {
            }

            // Constructor - dense from dense matrix
            GKCMatrix(std::vector<std::vector<ScalarT>> const &val)
            {
                /// @todo: Mark
                throw NotImplementedException("Matrix dense constructor not implemented");
            }

            // Constructor - sparse from dense matrix, removing specifed implied zeros
            GKCMatrix(std::vector<std::vector<ScalarT>> const &val,
                            ScalarT zero)
            {
                /// @todo: Mark
                throw NotImplementedException("Matrix dense constructor not implemented");
            }

            // Destructor
            ~GKCMatrix()
            {}

            // Assignment (currently restricted to same dimensions)
            GKCMatrix<ScalarT> &operator=(GKCMatrix<ScalarT> const &rhs)
            {
                /// @todo Mark
                throw NotImplementedException("Matrix assignment operator not implemented");
            }

            // EQUALITY OPERATORS
            /**
             * @brief Equality testing for GKC Matrix.
             * @param rhs The right hand side of the equality operation.
             * @return If this GKC Matrix and rhs are identical.
             */
            bool operator==(GKCMatrix<ScalarT> const &rhs) const
            {
                return ((m_num_rows == rhs.m_num_rows) &&
                        (m_num_cols == rhs.m_num_cols) &&
                        (m_num_edges == rhs.m_num_edges) &&
                        (m_weighted == rhs.m_weighted) &&
                        (m_offsets = rhs.m_offsets) &&
                        (m_neighbors == rhs.m_neighbors) && 
                        (m_weights == rhs.m_weights));
            }

            /**
             * @brief Inequality testing for GKC Matrix.
             * @param rhs The right hand side of the inequality operation.
             * @return If this GKC Matrix and rhs are not identical.
             */
            
            bool operator!=(GKCMatrix<ScalarT> const &rhs) const
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

                /// @todo mark build the matrix from empty, which means we can 
                // stream all the iterated elements in as long as they are sorted.
                // Are elements sorted? 
                /// @todo REALLY INEFFICIENT: Replace with ordered stream-in
                for (IndexType ix = 0; ix < n; ++ix)
                {
                    setElement(*i_it, *j_it, *v_it, dup);
                    ++i_it; ++j_it; ++v_it;
                } 
            }
            

            void clear()
            {
                /// @todo make atomic? transactional?
                m_num_edges = 0;
                m_num_rows = 0;
                m_num_cols = 0;
                /// @todo clear or resize to 0?
                m_offsets.clear();
                m_neighbors.clear();
                m_weights.clear();
            }

            IndexType nrows() const { return m_num_rows; }
            IndexType ncols() const { return m_num_cols; }
            IndexType nvals() const { return m_num_edges; }

            /**
             * @brief Resize the matrix dimensions (smaller or larger)
             *
             * @param[in]  new_num_rows  New number of rows (zero is invalid)
             * @param[in]  new_num_cols  New number of columns (zero is invalid)
             *
             */
            void resize(IndexType new_num_rows, IndexType new_num_cols)
            {
                throw NotImplementedException("Matrix resize not implemented");
                /*
                // Invalid values check by frontend

                // *******************************************
                // Step 1: Deal with number of rows

                // Count how many elements are left when num_rows reduces

                // *******************************************
                // Step 2: Deal with number columns
                // Need to do nothing if size stays the same or increases
                */
            }

            // This is basically column search, indexing a row.
            bool hasElement(IndexType irow, IndexType icol) const
            {
                if (irow >= m_num_rows || icol >= m_num_cols)
                {
                    throw IndexOutOfBoundsException(
                        "get_value_at: index out of bounds");
                }
                if (m_offsets.empty())
                {
                    return false;
                }
                if (m_neighbors.empty())
                {
                    return false;
                }
                if (m_offsets[irow+1] - m_offsets[irow] == 0)
                {
                    return false;
                }

                IndexType st = m_offsets[irow];
                IndexType nd = m_offsets[irow+1];

                for (auto nbr = m_neighbors.begin() + st; 
                nbr != m_neighbors.begin() + nd; nbr++)
                {
                    if (*nbr == icol)
                    {
                        return true;
                    }
                }
                return false;
            }

            // Get value at index (similar to has element, but then return it)
            ScalarT extractElement(IndexType irow, IndexType icol) const
            {
                if (irow >= m_num_vertices || icol >= m_num_vertices)
                {
                    throw IndexOutOfBoundsException(
                        "get_value_at: index out of bounds");
                }
                if (m_offsets.empty())
                {
                    throw NoValueException("extractElement: no data");
                }
                if (m_neighbors.empty())
                {
                    throw NoValueException("extractElement: no data");
                }
                if (m_offsets[irow+1] - m_offsets[irow] == 0)
                {
                    throw NoValueException("extractElement: no data in row");
                }

                IndexType st = m_offsets[irow];
                IndexType nd = m_offsets[irow+1];

                for (auto nbr_offset = st; nbr_offset != nd; nbr_offset++)
                {
                    if (m_neighbors[nbr_offset] == icol)
                    {
                        if (m_weighted)
                            return m_weights[nbr_offset];
                        else 
                            return 1;
                    }
                }
                throw NoValueException("extractElement: no entry at index");
            }

            // Set value at index (same as extract, but set it.)
            void setElement(IndexType irow, IndexType icol, ScalarT const &val)
            {
                // Step 1: figure out if element already exists.
                // Step 2: replace element that already exists or insert new element.
                throw NotImplementedException("Matrix set element not implemented");
                /// @todo make this work with extra lists, which need to be enabled for things like 
                /// extract and check if exists.
            }

            // Set value at index + 'merge' with any existing value
            // according to the BinaryOp passed.
            template <typename BinaryOpT>
            void setElement(IndexType irow, IndexType icol, ScalarT const &val,
                            BinaryOpT merge)
            {
                // Step 1: figure out if element already exists.
                // Step 2: merge with element that already exists or insert new element.
                throw NotImplementedException("Matrix set element not implemented");
                /// @todo make this work with extra lists, which need to be enabled for things like 
                /// extract and check if exists.
            }

            void removeElement(IndexType irow, IndexType icol)
            {
                // Step 1: figure out if element already exists.
                // Step 2: if exists, remove it (mark as deleted) 
                throw NotImplementedException("Matrix remove element not implemented");
                /// @todo make this work with extra lists, which need to be enabled for things like 
                /// extract and check if exists.
            }

            void recomputeNvals()
            {
                m_num_vertices = std::max(m_offsets.size() - 1, 0);
                /// @todo This will need to be updated to take into account the number
                /// of elements that are pending insertion but not in m_edges:
                m_num_edges = m_edges.size();
            }

/*
            // @todo: add error checking on dimensions?
            void swap(GKCMatrix<ScalarT> &rhs)
            {
                m_offsets.swap(rhs.m_offsets);
                m_neighbors.swap(rhs.m_neighbors);
                m_weights.swap(rhs.m_weights);

                std::swap(m_num_edges, rhs.m_num_edges);
                std::swap(m_num_vertices, rhs.m_num_vertices);
                std::swap(m_weighted, rhs.m_weighted);
                /// @todo Mark: let scott know that the original impl may be forgetting to swap 
                /// scalar vals into the rhs matrix.
            }
            */

/*
            // Row access
            // Warning if you use this non-const row accessor then you should
            // call recomputeNvals() at some point to fix it
            RowType &operator[](IndexType row_index) { return m_data[row_index]; }

            RowType const &operator[](IndexType row_index) const
            {
                /// @todo need a way to return a slice of the neighbor AND weights vectors.;
            }
            

            RowType const &getRow(IndexType row_index) const
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
                            "GKCMatrix::setCol() INTERNAL ERROR");
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
*/
            template<typename RAIteratorIT,
                     typename RAIteratorJT,
                     typename RAIteratorVT>
            void extractTuples(RAIteratorIT        row_it,
                               RAIteratorJT        col_it,
                               RAIteratorVT        values) const
            {
                for (IndexType row = 0; row < m_offsets.size()-1; ++row)
                {
                    st = m_offsets[row];
                    nd = m_offsets[row+1];
                    for (IndexType ii = st; ii < nd; ii++){                           
                        *row_it = row;              row_it++;
                        *col_it = m_neighbors[ii];  col_it++;
                        if (m_weighted) 
                            *values = m_weights[ii];    values++;
                        else 
                            *values = 1;                values++;
                    }
                }
            } 
            
            // output specific to the storage layout of this type of matrix
            void printInfo(std::ostream &os) const
            {
                os << "backend::GKCMatrix<" << typeid(ScalarT).name() << "> ";
                os << "(" << m_num_rows << " x " << m_num_cols << "), nvals = "
                   << nvals() << std::endl;

                // Used to print data in storage format instead of like a matrix
                #ifdef GRB_MATRIX_PRINT_RAW_STORAGE
                    for (IndexType row = 0; row < m_num_rows; ++row)
                    {
                        os << row << " :";
                        st = m_offsets[row];
                        nd = m_offsets[row+1];
                        for (IndexType ii = st; ii < nd; ii++){ 
                            if (m_weighted)                          
                                os << " " << m_neighbors[ii] << ":" << m_weights[ii];
                            else (m_weighted)
                                os << " " << m_neighbors[ii] << ":" << 1;

                        }
                        os << std::endl;
                    }
                #else
                    for (IndexType row_idx = 0; row_idx < m_num_rows; ++row_idx)
                    {
                        // We like to start with a little whitespace indent
                        os << ((row_idx == 0) ? "  [[" : "   [");

                        IndexType curr_idx = 0;
                        st = m_offsets[row];
                        nd = m_offsets[row+1];

                        if (nd - st == 0)
                        {
                            while (curr_idx < m_num_cols)
                            {
                                os << ((curr_idx == 0) ? " " : ",  " );
                                ++curr_idx;
                            }
                        }
                        else
                        {
                            // Walk the cols
                            for (IndexType ii = st; ii < nd; ii++){ 
                                auto col_idx =  m_neighbors[ii];
                                auto cell_val = m_weights[ii];
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
                                            GKCMatrix<ScalarT> const &mat)
            {
                mat.printInfo(os);
                return os;
            }
            
        private:
            IndexType m_num_rows;
            IndexType m_num_cols;
            IndexType m_num_edges;
            bool m_weighted;

            // Three array CSR matrix
            std::vector<IndexType> m_offsets;
            std::vector<IndexType> m_neighbors;
            std::vector<WeightType> m_weights;

        };

    } // namespace backend

} // namespace grb
