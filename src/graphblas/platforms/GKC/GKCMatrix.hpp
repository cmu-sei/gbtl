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
#include <numeric>
#include <map>
#include <cassert>

#include <graphblas/graphblas.hpp>
#include <graphblas/indices.hpp>

//****************************************************************************

namespace grb
{
    namespace backend
    {

        template<typename ScalarT>
        class GKCMatrix
        {
        public:
            using ScalarType = ScalarT;
            using WeightType = ScalarT;

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
                m_num_rows = val.size();
                // num cols is length of any row since it is dense.
                if (m_num_rows > 0){
                    m_num_cols = val[0].size();
                } 
                else 
                {
                    m_num_cols = 0;
                }
                // Dense has both structure and weights.
                m_weighted = true;
                m_num_edges = m_num_cols * m_num_rows;
                m_offsets.resize(m_num_rows+1);
                m_neighbors.resize(m_num_edges);
                m_weights.resize(m_num_edges);
                
                m_offsets[0] = 0;
                for (IndexType idx = 1; idx <= m_num_rows; idx++)
                {
                    m_offsets[idx] = idx * m_num_cols;
                }

                /// @todo: parallelize?
                for (auto row_idx = 0; row_idx < m_num_rows; row_idx++){
                    for (auto col_idx = 0; col_idx < m_num_cols; col_idx++) {
                        m_neighbors[row_idx * m_num_cols + col_idx] = col_idx;
                        m_weights[row_idx * m_num_cols + col_idx] = val[row_idx][col_idx];
                    }
                }
            }

            // Constructor - sparse from dense matrix, removing specifed implied zeros
            GKCMatrix(std::vector<std::vector<ScalarT>> const &val,
                            ScalarT zero)
            {
                m_num_rows = val.size();
                // num cols is length of any row since it is dense.
                if (m_num_rows > 0){
                    m_num_cols = val[0].size();
                } 
                else 
                {
                    m_num_cols = 0;
                }
                // Dense has both structure and weights.
                m_weighted = true;
                m_offsets.resize(m_num_rows+1);
                
                /// @todo: could preallocate a dense amount of space for
                /// "small" number of rows and cols, and otherwise 
                /// pre-scan the data to remove zeroes before allocating 
                /// weight and neighbor vectors.

                // For now, scan data to count number of non-zeroes
                // and then allocate the memory. Then rescan data to place into 
                // allocated space. 
                m_offsets[0] = 0;
                IndexType local_offset = 0;
                for (IndexType idx = 0; idx < m_num_rows; idx++)
                {
                    for (IndexType cdx = 0; cdx < m_num_cols; cdx++)
                    {
                        if (val[idx][cdx] != zero)
                        {
                            local_offset++;
                        }
                    }
                    // Record offset and record if num_cols increased.
                    m_offsets[idx+1] = local_offset;
                }
                m_num_edges = local_offset;

                // Allocate weights and neighbor lists:
                m_neighbors.resize(m_num_edges);
                m_weights.resize(m_num_edges);

                // Now copy edge data:
                /// @todo: parallelize?
                for (auto row_idx = 0; row_idx < m_num_rows; row_idx++)
                {
                    IndexType offset = m_offsets[row_idx];
                    for (auto col_idx = 0; col_idx < m_num_cols; col_idx++)
                    {
                        if (val[row_idx][col_idx] != zero){
                            m_neighbors[offset] = col_idx;
                            m_weights[offset] = val[row_idx][col_idx];
                            offset++;
                        }
                    }
                }
            }

            // Build Constructor - parse sparse coordinate data and construct matrix.
            // Similar to build method, but baked into the constructor.
            // Does not use addElement.
            template <typename RAIteratorI,
                      typename RAIteratorJ,
                      typename RAIteratorV,
                      typename DupT>
            GKCMatrix(RAIteratorI i_it,
                      RAIteratorJ j_it,
                      RAIteratorV v_it,
                      IndexType n,
                      DupT dup)
            {
                /// @todo require random access iterators
                // scan the data to determine num_edges, num_rows, num_cols
                // Note: assumes no duplicates:
                m_num_edges = n;
                m_num_rows = 0;
                m_num_cols = 0;
                for (IndexType idx = 0; idx < n; idx++)
                {
                    IndexType row_idx = *(i_it+idx);
                    IndexType col_idx = *(j_it+idx);
                    m_num_rows = std::max(m_num_rows, row_idx+1);
                    m_num_cols = std::max(m_num_cols, col_idx+1);
                }

                // allocate memory
                m_offsets.resize(m_num_rows+1);
                m_neighbors.resize(m_num_edges);
                /// @todo how to detect if graph is weighted?
                m_weighted = true;
                m_weights.resize(m_num_edges);
                std::fill(m_offsets.begin(), m_offsets.end(), (IndexType)0);

                /// compute neighborhood sizes and prefix sum
                /// @todo use map/counter to avoid repeated order M iteration? 
                for (IndexType idx = 0; idx < n; idx++)
                {
                    IndexType row_idx = *(i_it+idx);
                    m_offsets[row_idx+1]++;
                }
                /// @todo: parallel prefix sum
                for (IndexType idx = 0; idx < m_num_rows; idx++)
                {
                    m_offsets[idx+1] += m_offsets[idx];
                }

                /// @todo copy coordinates to appropriate locations // NOTE: need temporary size of each neighborhood till full.
                std::map<IndexType, IndexType> counters;
                for (IndexType idx = 0; idx < n; idx++)
                {
                    IndexType row_idx = *(i_it+idx);
                    IndexType col_idx = *(j_it+idx);
                    IndexType flat_idx = m_offsets[row_idx] + counters[row_idx];
                    m_neighbors[flat_idx] = col_idx;
                    if (m_weighted){
                        m_weights[flat_idx] = *(v_it + idx);
                    }
                    counters[row_idx]++;
                }

                // Sort the neighborhoods in-place
                /// @todo use a direct in-place sort for structure-only matrices. 
                /// The following permute-sort is designed to handle having weights.
                /// @todo use sort functions optimized for each neighborhood length
                /// @todo process neighborhoods in chunks so we don't need to allocate an 
                /// Order m vector for permutation.
                std::vector<size_t> permutation(m_num_edges);
                // Iota from 0 generates a sequence of indices.
                std::iota(permutation.begin(), permutation.end(), 0);
                for (size_t idx = 0; idx < m_num_rows; idx++)
                {
                    // The two input vectors are m_neighbors and m_weights. m_weights needs to be
                    // Permuted by the sorted order for m_neighbors.
                    // We need to sort the indices in m_neighbors in sections corresponding
                    // to each matrix row (each neighborhood)
                    auto st = m_offsets[idx];
                    auto nd = m_offsets[idx+1];
                    // Use a lambda to reorder part of permutation vector based on values in 
                    // m_neighbors.
                    if (nd > st)
                    {
                        // Note: only a portion of the vector is given for sort, but the indices
                        // i and j are based on permutation.begin(), not permutation.begin() + st.
                        std::sort(permutation.begin() + st, permutation.begin() + nd,
                            [&](const size_t i, const size_t j) { return (m_neighbors[i] < m_neighbors[j]); });
                    }
                }

                // Now we have a total permutation for the columns; apply it to m_neighbors and m_weights:
                /// @todo process neighborhoods in chunks to we don't need an O(m) bool vector, 
                /// and so that neighborhoods chunked together can be processed by diff threads.
                std::vector<bool> done(m_num_edges);
                for (IndexType i = 0; i < m_num_edges; i++)
                {
                    // If an original location i has not been permuted, 
                    // Find where it needs to go and continue swapping values until all are swapped.
                    if (!done[i])
                    {
                        done[i] = true;
                        size_t prev_j = i;
                        size_t j = permutation[i];
                        while (i!=j)
                        {
                            // Note that this swap loop will at most randomly access within 
                            // one row of the matrix.
                            std::swap(m_neighbors[prev_j], m_neighbors[j]);
                            if (m_weighted)
                                std::swap(m_weights[prev_j], m_weights[j]);
                            done[j] = true;
                            prev_j = j;
                            j = permutation[j];
                        }
                    }
                }
                /// @todo deduplicate edges using the dup operator:
                /// Use state bits on edges to 'delete' duplicate values
                for (IndexType row = 0; row < m_num_rows; row++){
                    auto st = m_offsets[row];
                    auto nd = m_offsets[row+1];
                    auto last_col = m_neighbors[st];
                    for (auto col_idx = st+1; col_idx < nd; col_idx++){
                        // Find and remove duplicate
                        if (m_neighbors[col_idx] == last_col){
                            // Set top bit to say that the entry is 'dead'
                            m_neighbors[col_idx] = (IndexType)-1;
                            /// @todo: change size of offsets
                            /// @todo: check for negative values in 
                            /// Parts of code that iterate over neighborhoods.
                            // Merge the two values:
                            /*if (m_weighted)
                            {
                             m_weights[first_occurrence_idx] = dup(
                                m_weights[first_occurrence_idx], m_weights[col_idx]);
                            } */
                        }
                        last_col = m_neighbors[col_idx];
                    }
                }
            }

            // Destructor
            ~GKCMatrix()
            {}

            // Assignment (currently restricted to same dimensions)
            GKCMatrix<ScalarT> &operator=(GKCMatrix<ScalarT> const &rhs)
            {
                // push this check to frontend
                if ((m_num_rows != rhs.m_num_rows) || 
                    (m_num_cols != rhs.m_num_cols))
                {
                    throw DimensionException("Dimensions of matrices do not match in rows and columns for assignment.");
                }    
                m_num_edges = rhs.num_edges;
                m_neighbors = rhs.m_neighbors;
                m_offsets = rhs.m_offsets;
                m_weights = rhs.m_weights;
                m_weighted = rhs.m_weighted;
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
                        (m_offsets == rhs.m_offsets) &&
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
                /// @todo require random access iterators
                // scan the data to determine num_edges, num_rows, num_cols
                // Note: assumes no duplicates:
                /*
                m_num_edges = n;
                m_num_rows = 0;
                m_num_cols = 0;
                for (IndexType idx = 0; idx < n; idx++)
                {
                    IndexType row_idx = *(i_it+idx);
                    IndexType col_idx = *(j_it+idx);
                    m_num_rows = std::max(m_num_rows, row_idx+1);
                    m_num_cols = std::max(m_num_cols, col_idx+1);
                }*/

                auto old_num_edges = m_num_edges;
                m_num_edges = n;

                // allocate memory
                m_offsets.resize(m_num_rows+1);
                m_neighbors.resize(m_num_edges);
                /// @todo how to detect if graph is weighted?
                m_weighted = true;
                m_weights.resize(m_num_edges);
                std::fill(m_offsets.begin(), m_offsets.end(), (IndexType)0);

                /// compute neighborhood sizes and prefix sum
                /// @todo use map/counter to avoid repeated order M iteration? 
                for (IndexType idx = 0; idx < n; idx++)
                {
                    IndexType row_idx = *(i_it+idx);
                    m_offsets[row_idx+1]++;
                }
                /// @todo: parallel prefix sum
                for (IndexType idx = 0; idx < m_num_rows; idx++)
                {
                    m_offsets[idx+1] += m_offsets[idx];
                }

                /// @todo copy coordinates to appropriate locations // NOTE: need temporary size of each neighborhood till full.
                std::map<IndexType, IndexType> counters;
                for (IndexType idx = 0; idx < n; idx++)
                {
                    IndexType row_idx = *(i_it+idx);
                    IndexType col_idx = *(j_it+idx);
                    IndexType flat_idx = m_offsets[row_idx] + counters[row_idx];
                    // assert(flat_idx < n);
                    m_neighbors[flat_idx] = col_idx;
                    if (m_weighted){
                        m_weights[flat_idx] = *(v_it + idx);
                    }
                    counters[row_idx]++;
                }

                // Sort the neighborhoods in-place
                /// @todo use a direct in-place sort for structure-only matrices. 
                /// The following permute-sort is designed to handle having weights.
                /// @todo use sort functions optimized for each neighborhood length
                /// @todo process neighborhoods in chunks so we don't need to allocate an 
                /// Order m vector for permutation.
                std::vector<size_t> permutation(m_num_edges);
                // Iota from 0 generates a sequence of indices.
                std::iota(permutation.begin(), permutation.end(), 0);
                for (size_t idx = 0; idx < m_num_rows; idx++)
                {
                    // The two input vectors are m_neighbors and m_weights. m_weights needs to be
                    // Permuted by the sorted order for m_neighbors.
                    // We need to sort the indices in m_neighbors in sections corresponding
                    // to each matrix row (each neighborhood)
                    auto st = m_offsets[idx];
                    auto nd = m_offsets[idx+1];
                    // Use a lambda to reorder part of permutation vector based on values in 
                    // m_neighbors.
                    if (nd > st)
                    {
                        // Note: only a portion of the vector is given for sort, but the indices
                        // i and j are based on permutation.begin(), not permutation.begin() + st.
                        std::sort(permutation.begin() + st, permutation.begin() + nd,
                            [&](const size_t i, const size_t j) { return (m_neighbors[i] < m_neighbors[j]); });
                    }
                }

                // Now we have a total permutation for the columns; apply it to m_neighbors and m_weights:
                /// @todo process neighborhoods in chunks to we don't need an O(m) bool vector, 
                /// and so that neighborhoods chunked together can be processed by diff threads.
                std::vector<bool> done(m_num_edges);
                for (IndexType i = 0; i < m_num_edges; i++)
                {
                    // If an original location i has not been permuted, 
                    // Find where it needs to go and continue swapping values until all are swapped.
                    if (!done[i])
                    {
                        done[i] = true;
                        size_t prev_j = i;
                        size_t j = permutation[i];
                        while (i!=j)
                        {
                            // Note that this swap loop will at most randomly access within 
                            // one row of the matrix.
                            std::swap(m_neighbors[prev_j], m_neighbors[j]);
                            if (m_weighted)
                                std::swap(m_weights[prev_j], m_weights[j]);
                            done[j] = true;
                            prev_j = j;
                            j = permutation[j];
                        }
                    }
                }
                /// @todo deduplicate edges using the dup operator:
                /// Use state bits on edges to 'delete' duplicate values
                /*
                for (IndexType row = 0; row < m_num_rows; row++){
                    auto st = m_offsets[row];
                    auto nd = m_offsets[row+1];
                    auto last_col = m_neighbors[st];
                    for (auto col_idx = st+1; col_idx < nd; col_idx++){
                        // Find and remove duplicate
                        if (m_neighbors[col_idx] == last_col){
                            // Set top bit to say that the entry is 'dead'
                            m_neighbors[col_idx] = (IndexType)-1;
                            throw NotImplementedException("Dedup is not implemented...");
                            /// @todo: change size of offsets
                            /// @todo: check for negative values in 
                            /// Parts of code that iterate over neighborhoods.
                            // Merge the two values:
                            //if (m_weighted)
                            //{
                            // m_weights[first_occurrence_idx] = dup(
                            //    m_weights[first_occurrence_idx], m_weights[col_idx]);
                            //} 
                        }
                        last_col = m_neighbors[col_idx];
                    }
                }
                */
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
                // Invalid values check by frontend
                if (new_num_rows == 0 || new_num_cols == 1)
                    throw DimensionException("Cannot resize matrix to have zero in either row or column dimension.");

                // *******************************************
                // Step 1: Deal with number of rows
                if (new_num_rows < m_num_rows)
                {
                    // truncate m_offsets
                    m_offsets.resize(new_num_rows + 1);
                    m_num_rows = new_num_rows;

                    // truncate m_neighbors, m_weights
                    /// @todo: if elements are deleted throughout, it 
                    /// is no longer correct to resize to number of elements
                    /// And also the last value in m_offsets will not reflect 
                    /// the actual number of edges in the graph.
                    auto new_num_vals = m_offsets[m_num_rows];
                    m_neighbors.resize(new_num_vals);
                    if (m_weighted)
                        m_weights.resize(new_num_vals);
                    m_num_edges = new_num_vals;
                } 
                else if (new_num_rows > m_num_rows)
                {
                    // Extend m_offsets with 0-size rows
                    m_offsets.resize(new_num_rows + 1);
                    std::fill(m_offsets.begin() + m_num_rows + 1, m_offsets.end(), m_num_edges); 
                }

                // *******************************************
                // Step 2: Deal with number columns
                // Need to do nothing if size stays the same or increases
                if (new_num_cols < m_num_cols)
                {
                    throw NotImplementedException("Matrix resize not implemented; need ability to mark elements as deleted.");
                }
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

                /// @todo: perform binary search but only for long rows.
                /// @note: the binary search will need to deal with invalidated entries.
                /// Either mask the upper bits for the original index or shift over by one.
                for (auto nbr = m_neighbors.begin() + st; 
                nbr != m_neighbors.begin() + nd; nbr++)
                {
                    if (*nbr == icol)
                    {
                        return true;
                    }
                }
                return false;
                /// @todo: search for element in insert lists, once those are added.
            }

            // Get value at index (similar to has element, but then return it)
            ScalarT extractElement(IndexType irow, IndexType icol) const
            {
                if (irow >= m_num_rows || icol >= m_num_cols)
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

                /// @todo: perform binary search but only for long rows.
                /// @note: the binary search will need to deal with invalidated entries.
                /// Either mask the upper bits for the original index or shift over by one.
                for (auto nbr_offset = st; nbr_offset != nd; nbr_offset++)
                {
                    if (m_neighbors[nbr_offset] == icol)
                    {
                        if (m_weighted)
                        {
                            return m_weights[nbr_offset];
                        }
                        else
                        {
                            return 1;
                        }
                    }
                }
                /// @todo: search for element in insert lists, once those are added.
                throw NoValueException("extractElement: no entry at index");
            }

            // Set value at index (same as extract, but set it.)
            void setElement(IndexType irow, IndexType icol, ScalarT const &val)
            {
                if (irow >= m_num_rows || icol >= m_num_cols)
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

                // Step 1: figure out if element already exists.
                IndexType st = m_offsets[irow];
                IndexType nd = m_offsets[irow+1];

                bool insert_success = false;
                for (auto nbr_offset = st; nbr_offset != nd && !insert_success; nbr_offset++)
                {
                    // Step 2: replace element that already exists or insert new element.
                    /// @todo: perform binary search but only for long rows.
                    /// @note: the binary search will need to deal with invalidated entries.
                    /// Either mask the upper bits for the original index or shift over by one.
                    if (m_neighbors[nbr_offset] == icol)
                    {
                        if (m_weighted)
                        {
                            m_weights[nbr_offset] = val;
                            insert_success = true;
                            break;
                        }
                        else
                        { 
                            throw InvalidValueException("Can't add weighted element to unweighted matrix.");
                        }
                    }
                    /// @todo can we assume that the neighborhoods are all sorted? 
                    /// What if some items are marked as deleted? 
                    else if (m_neighbors[nbr_offset] > icol) // Assume we passed it, so it's not there
                    {
                        throw NotImplementedException("Adding elements to matrix not yet implemented");
                        // break; and add the element to an insert list...
                    }
                }
                /// @todo make this work with extra lists, which need to be enabled for things like 
                /// extract and check if exists.
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

                // Step 1: figure out if element already exists.
                IndexType st = m_offsets[irow];
                IndexType nd = m_offsets[irow+1];

                bool insert_success = false;
                for (auto nbr_offset = st; nbr_offset != nd && !insert_success; nbr_offset++)
                {
                    // Step 2: replace element that already exists or insert new element.
                    /// @todo: perform binary search but only for long rows.
                    /// @note: the binary search will need to deal with invalidated entries.
                    /// Either mask the upper bits for the original index or shift over by one.
                    if (m_neighbors[nbr_offset] == icol)
                    {
                        if (m_weighted)
                        {
                            m_weights[nbr_offset] = merge(val, m_weights[nbr_offset]);
                            insert_success = true;
                            break;
                        }
                        else
                        {
                            throw InvalidValueException("Can't add weighted element to unweighted matrix.");
                        }
                    }
                    /// @todo can we assume that the neighborhoods are all sorted? 
                    /// What if some items are marked as deleted? 
                    else if (m_neighbors[nbr_offset] > icol) // Assume we passed it, so it's not there
                    {
                        throw NotImplementedException("Adding elements to matrix not yet implemented");
                        // break; and add the element to an insert list...
                    }
                }
                /// @todo make this work with extra lists, which need to be enabled for things like 
                /// extract and check if exists.
            }

            void removeElement(IndexType irow, IndexType icol)
            {
                if (irow >= m_num_rows || icol >= m_num_cols)
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

                // Step 1: figure out if element already exists.
                IndexType st = m_offsets[irow];
                IndexType nd = m_offsets[irow+1];

                bool insert_success = false;
                for (auto nbr_offset = st; nbr_offset != nd && !insert_success; nbr_offset++)
                {
                    // Step 2: replace element that already exists or insert new element.
                    /// @todo: perform binary search but only for long rows.
                    /// @note: the binary search will need to deal with invalidated entries.
                    /// Either mask the upper bits for the original index or shift over by one.
                    if (m_neighbors[nbr_offset] == icol)
                    {
                        /// @todo: set structural value (m_neighbors) as invalid (support zombies)
                        throw NotImplementedException("Matrix remove element not implemented");
                        break;
                    }
                }
                throw NoValueException("removeElement: no data to remove");
                /// @todo make this work with extra lists, which need to be enabled for things like 
                /// extract and check if exists.
            }

            size_t getRowDegree(IndexType index) const {
                if (index < m_num_rows){
                    return m_offsets[index + 1] - m_offsets[index];
                }
            }

            IndexType * getIA()
            {
                return m_offsets.data();
            }

            IndexType * getJA()
            {
                return m_neighbors.data();
            }

            ScalarT * getVA()
            {
                return m_weights.data();
            }

            using idx_iterator = typename std::vector<IndexType>::iterator;
            using wgt_iterator = typename std::vector<WeightType>::iterator;
            // Iterators for neighborhoods
            inline idx_iterator idxBegin(IndexType idx) {return m_neighbors.begin() + m_offsets[idx]; }
            inline wgt_iterator wgtBegin(IndexType idx) {return m_weights.begin()   + m_offsets[idx]; }
            inline idx_iterator idxEnd(IndexType idx)   {return m_neighbors.begin() + m_offsets[idx+1]; }
            inline wgt_iterator wgtEnd(IndexType idx)   {return m_weights.begin()   + m_offsets[idx+1]; }
            // Const versions 
            inline const idx_iterator idxBegin(IndexType idx) const {return m_neighbors.begin() + m_offsets[idx]; }
            inline const wgt_iterator wgtBegin(IndexType idx) const {return m_weights.begin()   + m_offsets[idx]; }
            inline const idx_iterator idxEnd(IndexType idx)   const {return m_neighbors.begin() + m_offsets[idx+1]; }
            inline const wgt_iterator wgtEnd(IndexType idx)   const {return m_weights.begin()   + m_offsets[idx+1]; }

            /// @todo: rewrite recompute Nvals for when we have insert lists and elements
            /// pending removal? 
/*
            void recomputeNvals()
            {
                m_num_rows = std::max((IndexType)(m_offsets.size() - 1), (IndexType)0);
                // Note: no recompute for numcols?
                m_num_cols = std::min(m_num_cols, m_num_rows);
                /// @todo This will need to be updated to take into account the number
                /// of elements that are pending insertion but not in m_neighbors:
                m_num_edges = m_neighbors.size();
            }
 */

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

            template<typename RAIteratorIT,
                     typename RAIteratorJT,
                     typename RAIteratorVT>
            void extractTuples(RAIteratorIT        row_it,
                               RAIteratorJT        col_it,
                               RAIteratorVT        values) const
            {
                for (IndexType row = 0; row < m_offsets.size()-1; ++row)
                {
                    auto st = m_offsets[row];
                    auto nd = m_offsets[row+1];
                    for (IndexType ii = st; ii < nd; ii++){                           
                        *row_it = row;              row_it++;
                        *col_it = m_neighbors[ii];  col_it++;
                        if (m_weighted)
                        { 
                            *values = m_weights[ii];    values++;
                        }
                        else 
                        { 
                            *values = 1;                values++;
                        }
                    }
                }
            } 
            
            // output specific to the storage layout of this type of matrix
            void printInfo(std::ostream &os) const
            {
                os << "GKC Backend: ";
                os << "backend::GKCMatrix<" << typeid(ScalarT).name() << "> ";
                os << "(" << m_num_rows << " x " << m_num_cols << "), nvals = "
                   << nvals() << std::endl;

                #if 1
                // Used to print data in storage format instead of like a matrix
                #ifdef GRB_MATRIX_PRINT_RAW_STORAGE
                    for (IndexType row = 0; row < m_num_rows; ++row)
                    {
                        os << row << " :";
                        auto st = m_offsets[row];
                        auto nd = m_offsets[row+1];
                        for (IndexType ii = st; ii < nd; ii++){ 
                            if (m_weighted){
                                os << " " << m_neighbors[ii] << ":" << m_weights[ii];
                            }
                            else
                            {
                                os << " " << m_neighbors[ii] << ":" << 1;
                            }
                        }
                        os << std::endl;
                    }
                #else
                    for (IndexType row_idx = 0; row_idx < m_num_rows; ++row_idx)
                    {
                        // We like to start with a little whitespace indent
                        os << ((row_idx == 0) ? "  [[" : "   [");

                        IndexType curr_idx = 0;
                        auto st = m_offsets[row_idx];
                        auto nd = m_offsets[row_idx+1];

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
            mutable std::vector<IndexType> m_offsets;
            mutable std::vector<IndexType> m_neighbors;
            mutable std::vector<WeightType> m_weights;

        };

    } // namespace backend

} // namespace grb