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

#include <graphblas/graphblas.hpp>
#include <graphblas/indices.hpp>

//****************************************************************************

namespace grb
{
    namespace backend
    {

        /**
         * @brief Class representing a sparse vector using an index vector
         * and a values vector. Based on frontier class in GKC 1.0
         */

        // Sort or no sort? 
        // When to keep things synchronized?
        // How to keep updates?

        template<typename ScalarT>
        class GKCSparseVector
        {
        public:
            using ScalarType = ScalarT;

            // Constructor with some default values
            /**
             * @brief construct an empty sparse vector with space for 
             * num_vals values.
             * The sparse vector can be structure only, with weights set
             * to false, so only vertex IDs are stored.
             */
            GKCSparseVector(IndexType num_vals,
                      bool weighted = true)
                : m_num_vals(num_vals),
                  m_num_stored_vals(0),
                  m_weighted(weighted)
            {
                if (m_num_vals <= 0){
                    throw InvalidValueException();
                }
                m_indices.resize(m_num_vals);
                if (m_weighted)
                    m_weights.resize(m_num_vals);
            }

            // Create GKC Sparse Vector with default weight/value
            /* THIS IS NOT A SENSIBLE USE OF THIS VECTOR. A DENSE VECTOR SHOULD INSTEAD BE USED. */
            GKCSparseVector(IndexType num_vals, ScalarT const &value)
                : m_num_vals(num_vals),
                  m_num_stored_vals(num_vals),
                  m_weighted(true),
                  m_weights(num_vals, value),
                  m_indices(num_vals)
            {
                if (m_num_vals <= 0){
                    throw InvalidValueException();
                }
                // Fill with ascending indices
                for (auto idx = 0; idx < m_num_vals; idx++){
                    m_indices[idx] = idx;
                }
                // Is iota as good (and parallelizable) as the above loop?
                //std::iota(m_indices.begin(), m_indices.end(), 0);
            }

            // Constructor - copy
            GKCSparseVector(GKCSparseVector<ScalarT> const &rhs)
                : m_num_vals(rhs.m_num_vals),
                  m_weighted(rhs.m_weighted),
                  m_indices(rhs.m_indices),
                  m_weights(rhs.m_weights),
                  m_num_stored_vals(rhs.m_num_stored_vals)
            {
            }

            // Should the non-filtering dense constructor be supplied?
            // It makes little sense to create a dense vector with a 
            // Sparse vector class.
            // Constructor - dense from dense vector
            GKCSparseVector(std::vector<ScalarT> const &val)
            {
                m_num_vals = val.size();
                // Dense has both structure and weights.
                m_weighted = true;
                m_indices.resize(m_num_vals);
                m_weights.resize(m_num_vals);
                m_num_stored_vals = 0;
                
                /// @todo: parallelize?
                // Copy values from dense vector into sparse
                for (auto idx = 0; idx < m_num_vals; idx++){
                    m_indices[m_num_stored_vals] = idx;
                    m_weights[m_num_stored_vals] = val[idx];
                    m_num_stored_vals ++;
                }
            }

            // Constructor - sparse from dense vector, removing specifed implied zeros
            GKCSparseVector(std::vector<ScalarT> const &val,
                            ScalarT zero)
            {
                m_num_vals = val.size();
                // Dense has both structure and weights.
                m_weighted = true;
                m_indices.resize(m_num_vals);
                m_weights.resize(m_num_vals);
                m_num_stored_vals = 0;
                
                /// @todo: parallelize?
                // Copy values from dense vector into sparse
                // While ignoring 'zero' elements
                for (auto idx = 0; idx < m_num_vals; idx++){
                    if (val[idx] != zero)
                    {
                        m_indices[m_num_stored_vals] = idx;
                        m_weights[m_num_stored_vals] = val[idx];
                        m_num_stored_vals ++;
                    }
                }
            }

            // Build Constructor - parse sparse coordinate data and construct vector.
            // Similar to build method, but baked into the constructor.
            // Does not use addElement.
            /// @todo: how do we limit the size of this vector? Right now it's 
            // implicit in the data provided by i_it.
            template <typename RAIteratorI,
                      typename RAIteratorV,
                      typename BinaryOpT = grb::Second<ScalarType> >
            GKCSparseVector(RAIteratorI i_it,
                            RAIteratorV v_it,
                            IndexType n,
                            BinaryOpT dup = BinaryOpT())
            {
                /// @todo require random access iterators
                // scan the data to determine num_vals
                /// @todo: OMP max reduction
                IndexType max_idx = 0;
                for (size_t i = 0; i < n; i++){
                    max_idx = std::max(*(i_it+i), max_idx);
                }

                /// @todo: should we always allocate to the max number of 
                /// vertices? Or in this case just to n that we know we may need?
                m_num_vals = max_idx+1;
                m_num_stored_vals = 0;

                // allocate memory
                m_indices.resize(m_num_vals);
                /// @todo how to detect if graph is weighted?
                m_weighted = true;
                m_weights.resize(m_num_vals);

                // Copy data from iterators into vector
                std::map<IndexType, IndexType> already_inserted;
                for (IndexType idx = 0; idx < n; idx++)
                {
                    IndexType vidx = *(i_it+idx);
                    auto found_itr = already_inserted.find(vidx);
                    if (found_itr == already_inserted.end())
                    {
                        // Index not recognized, add new entry
                        already_inserted[vidx] = m_num_stored_vals;
                        m_indices[m_num_stored_vals] = vidx;
                        m_weights[m_num_stored_vals] = *(v_it + idx);
                        m_num_stored_vals++;
                    }
                    else  
                    {
                        // Already have a value, so merge the weight.
                        auto old_idx = found_itr->second;
                        m_weights[old_idx] = dup(m_weights[old_idx], *(v_it + idx));
                    }
                }
                // Todo: sort here
            }

            // Destructor
            ~GKCSparseVector()
            {}

            // Copy assignment (currently restricted to same dimensions)
            GKCSparseVector<ScalarT> &operator=(GKCSparseVector<ScalarT> const &rhs)
            {
                if (this != &rhs) 
                {
                    if (m_num_vals != rhs.m_num_vals)
                    {
                        throw DimensionException("Dimensions of vectors do not match.");
                    }    
                    m_num_vals = rhs.m_num_vals;
                    m_num_stored_vals = rhs.m_num_stored_vals;
                    m_weighted = rhs.m_weighted;
                    m_indices = rhs.m_indices;
                    m_weights = rhs.m_weights;
                }
                return *this;
            }

            // EQUALITY OPERATORS
            /**
             * @brief Equality testing for GKC Vector.
             * @param rhs The right hand side of the equality operation.
             * @return If this GKC Vector and rhs are identical.
             * @todo: need create sets for the vectors to compare their contents.
             * @todo: lookup mutable keyword
             */
            bool operator==(GKCSparseVector<ScalarT> const &rhs) const
            {
                /// @todo need to check for ordering of indices/weights. 
                /// @todo need to check for only the part of each vector USED.
                //throw NotImplementedException(); 
                //if (this == &rhs) return true;
                bool test1 = (m_num_vals == rhs.m_num_vals);
                bool test2 = (m_weighted == rhs.m_weighted);
                bool test3 = (m_num_stored_vals == rhs.m_num_stored_vals);
                bool test4 = std::equal(m_indices.begin(), m_indices.begin() + m_num_stored_vals, rhs.m_indices.begin());
                bool test5 = std::equal(m_weights.begin(), m_weights.begin() + m_num_stored_vals, rhs.m_weights.begin());
                // std::cout << (test1 ? "true" : "false") << std::endl;
                // std::cout << (test2 ? "true" : "false") << std::endl;
                // std::cout << (test3 ? "true" : "false") << std::endl;
                // std::cout << (test4 ? "true" : "false") << std::endl;
                // std::cout << (test5 ? "true" : "false") << std::endl;
                // std::cout << (m_weighted ? "true" : "false") << std::endl;
                return ( test1 && test2 && test3 && test4 && ((m_weighted && test5) || !m_weighted) );
            }

            /**
             * @brief Inequality testing for GKC Vector.
             * @param rhs The right hand side of the inequality operation.
             * @return If this GKC Vector and rhs are not identical.
             */
            
            bool operator!=(GKCSparseVector<ScalarT> const &rhs) const
            {
                return !(*this == rhs);
            }
            
            template<typename RAIteratorIT,
                     typename RAIteratorVT,
                     typename BinaryOpT = grb::Second<ScalarType> >
            void build(RAIteratorIT  i_it,
                       RAIteratorVT  v_it,
                       IndexType     n,
                       BinaryOpT     dup = BinaryOpT())
            {
                /// @todo require random access iterators
                // scan the data to determine num_edges, num_rows, num_cols
                m_num_vals = n;
                m_num_stored_vals = 0;

                // allocate memory
                m_indices.resize(m_num_vals);
                /// @todo how to detect if graph is weighted?
                m_weighted = true;
                m_weights.resize(m_num_vals);

                // Copy data from iterators into vector
                std::map<IndexType, IndexType> already_inserted;
                for (IndexType idx = 0; idx < n; idx++)
                {
                    IndexType vidx = *(i_it+idx);
                    if (already_inserted[vidx] < idx + 1){
                        // Already have a value, so merge the weight.
                        auto old_idx = already_inserted[vidx] - 1;
                        m_weights[old_idx] = dup(m_weights[old_idx], *(v_it + idx));
                    }
                    else  // Index not recognized, add new entry
                    {
                        already_inserted[vidx] = idx + 1;
                        m_indices[idx] = vidx;
                        m_weights[idx] = *(v_it + idx);
                        m_num_stored_vals++;
                    }
                }
                // Todo, sort here

            }
            
            void clear()
            {
                /// @todo make atomic? transactional?
                m_num_stored_vals = 0;
            }

            IndexType size() const { return m_num_vals; }
            IndexType nvals() const { return m_num_stored_vals; }

            /**
             * @brief Resize the vector dimensions (smaller or larger)
             *
             * @param[in]  new_num_rows  New number of rows (zero is invalid)
             * @param[in]  new_num_cols  New number of columns (zero is invalid)
             *
             */
            void resize(IndexType new_size)
            {
                // Invalid values check by frontend
                if (new_size == 0)
                    throw DimensionException("Cannot resize vector to have zero dimension.");
                if (new_size < m_num_vals)
                {
                    m_num_vals = new_size;
                    // Linear scan and shift
                    IndexType new_num_stored = 0;
                    for (size_t i = 0; i < m_num_stored_vals; i++)
                    {
                        size_t idx = m_indices[i];
                        if (idx < new_size)
                        {
                            m_indices[new_num_stored] = idx;
                            if (m_weighted) {
                                m_weights[new_num_stored] = m_weights[i];
                            }
                            new_num_stored++;
                        }
                    }
                    m_num_stored_vals = new_num_stored;
                }
                else // Increase size of vectors
                {
                    m_num_vals = new_size;
                    m_indices.resize(new_size);
                    if (m_weighted) {
                        m_weights.resize(new_size);
                    }
                } 
            }

            bool hasElement(IndexType index) const
            {
                if (index >= m_num_vals)
                {
                    throw IndexOutOfBoundsException();
                }
                for (size_t idx = 0; idx < m_num_stored_vals; idx++)
                {
                    auto vidx = m_indices[idx];
                    if (vidx == index) return true;
                }
                return false;
            }

            ScalarT extractElement(IndexType index) const
            {
                /// @todo mark: if vector is sorted, 
                /// use binary search.
                /// Need to add a 'sorted' flag that is reset if 
                // vector is modified.
                if (index >= m_num_vals)
                {
                    throw IndexOutOfBoundsException();
                }
                for (size_t idx = 0; idx < m_num_stored_vals; idx++)
                {
                    auto vidx = m_indices[idx];
                    if (vidx == index)
                    {
                        if (m_weighted)
                        {
                            return m_weights[idx];
                        }
                        else // What to do if no weights?
                        {
                            return (ScalarT)1;
                        }
                    }
                }
                // No value found; throw error
                throw NoValueException();
            }

            void setElement(IndexType index, ScalarT const &new_val)
            {
                if (index >= m_num_vals)
                {
                    throw IndexOutOfBoundsException();
                }
                for (size_t idx = 0; idx < m_num_stored_vals; idx++)
                {
                    auto vidx = m_indices[idx];
                    if (vidx == index)
                    {
                        if (m_weighted){
                            m_weights[idx] = new_val;
                        }
                        return;
                    }
                }
                // No value found; insert it:
                m_indices[m_num_stored_vals] = index;
                if (m_weighted){
                    m_weights[m_num_stored_vals] = new_val;
                }
                m_num_stored_vals++;
                // Todo: add a sorted or not sorted flag
            }

            template <typename BinaryOpT, typename ZScalarT> 
            void mergeSetElement(IndexType index, ZScalarT const &new_val, BinaryOpT op)
            {
                if (index >= m_num_vals)
                {
                    throw IndexOutOfBoundsException();
                }
                if (index >= m_indices.size() || (index >= m_weights.size() && m_weighted))
                {
                    throw InvalidIndexException("Mismatch in internal vector size and size of vector.");
                }
                for (size_t idx = 0; idx < m_num_stored_vals; idx++)
                {
                    auto vidx = m_indices[idx];
                    if (vidx == index)
                    {
                        if (m_weighted){
                            m_weights[idx] = op(m_weights[idx], new_val);
                        }
                        return;
                    }
                }
                // No value found; insert it:
                m_indices[m_num_stored_vals] = index;
                if (m_weighted){
                    // Make sure to correctly cast for the output of the operation, 
                    // which in mxv is not the same as the reduction (additive) output.
                    using ZType = decltype(op(
                         std::declval<ScalarT>(),
                         std::declval<ZScalarT>()
                        ));
                    m_weights[m_num_stored_vals] = (ZType)new_val;
                }
                m_num_stored_vals++;
                // Todo: add a sorted or not sorted flag
            }

            void removeElement(IndexType index)
            {
                if (index >= m_num_vals)
                {
                    throw IndexOutOfBoundsException();
                }
                // Step 1: find element
                if (index > m_num_vals)
                {
                    throw IndexOutOfBoundsException();
                }
                for (size_t idx = 0; idx < m_num_stored_vals; idx++)
                {
                    auto vidx = m_indices[idx];
                    if (vidx == index)
                    {
                        // Step 2: vector doesn't need to remain sorted, 
                        // so just replace with last element.
                        // NOT THREAD SAFE!
                        if (idx < m_num_stored_vals - 1){
                            // Swap with last elem and decremement size
                            m_indices[idx] = m_indices[m_num_stored_vals - 1];
                            if (m_weighted)
                            {
                                m_weights[idx] = m_weights[m_num_stored_vals-1];
                            }
                            m_num_stored_vals--;
                            this->sortSelf();
                        }
                        else if (idx == m_num_stored_vals - 1)
                        { // Added to handle corner case when only one elem remains.
                            m_num_stored_vals = 0;
                        }
                        return;
                    }
                }
                // No value found; throw error
                throw NoValueException();
                // Todo: add a sorted or not sorted flag
            }

            template<typename RAIteratorIT,
                     typename RAIteratorVT>
            void extractTuples(RAIteratorIT        i_it,
                               RAIteratorVT        v_it) const
            {
                throw NotImplementedException();
            }

            // Note: this has to be const because changes to it could break 
            // the weights vector. 
            // further, using end() on the vector may iterate too far since
            // the vector is resized for the maximum number of vertices, 
            // not the amount currently stored.
            /// @todo mark: add iterator for GKC Sparse Vector
            const std::vector<IndexType>& getIndices() const
            {
                return m_indices;
            }

            const std::vector<ScalarT>& getWeights() const
            {
                if (m_weighted){
                    return m_weights;
                }
                throw NoValueException();
            }
            
            using idx_iterator = typename std::vector<IndexType>::iterator;
            using wgt_iterator = typename std::vector<ScalarType>::iterator;
            // Iterators for neighborhoods
            inline idx_iterator idxBegin() {return m_indices.begin(); }
            inline wgt_iterator wgtBegin() {return m_weights.begin(); }
            inline idx_iterator idxEnd()   {return m_indices.begin() + m_num_stored_vals; }
            inline wgt_iterator wgtEnd()   {return m_weights.begin() + m_num_stored_vals; }
            // Const versions 
            inline const idx_iterator idxBegin() const {return m_indices.begin(); }
            inline const wgt_iterator wgtBegin() const {return m_weights.begin(); }
            inline const idx_iterator idxEnd()   const {return m_indices.begin() + m_num_stored_vals; }
            inline const wgt_iterator wgtEnd()   const {return m_weights.begin() + m_num_stored_vals; }

            void sortSelf() const // note const because sorting doesn't semantically change the vector.
            {
                std::vector<size_t> permutation(m_num_stored_vals);
                // Iota from 0 generates a sequence of indices.
                std::iota(permutation.begin(), permutation.end(), 0);
                std::sort(permutation.begin(), permutation.end(), [&]
                (const size_t i, const size_t j){
                    return (m_indices[i] < m_indices[j]);
                });
                std::vector<bool> done(m_num_stored_vals);
                for (IndexType i = 0; i < m_num_stored_vals; i++)
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
                            std::swap(m_indices[prev_j], m_indices[j]);
                            if (m_weighted)
                                std::swap(m_weights[prev_j], m_weights[j]);
                            done[j] = true;
                            prev_j = j;
                            j = permutation[j];
                        }
                    }
                }
            }

            // output specific to the storage layout of this type of matrix
            void printInfo(std::ostream &os) const
            {
                os << "GKC Backend: ";
                os << "backend::GKCSparseVector<" << typeid(ScalarT).name() << ">";
                os << ", size  = " << m_num_vals;
                os << ", nvals = " << m_num_stored_vals << std::endl;
                os << "weighted: " << (m_weighted ? "true" : "false") << std::endl;
                os << "m_indices.size(): " << m_indices.size() << std::endl;
                os << "m_weights.size(): " << m_weights.size() << std::endl;

                os << "[";
                if (m_num_stored_vals > 0){
                    os << m_indices[0];
                    if (m_weighted){
                        os << ":" << m_weights[0];
                    } 
                }
                for (IndexType idx = 1; idx < m_num_stored_vals; ++idx)
                {
                    os << ", " << m_indices[idx];
                    if (m_weighted) {
                        os << ":" << m_weights[idx];
                    }
                }
                os << "]";
                os << std::endl;
            }

            friend std::ostream &operator<<(std::ostream             &os,
                                            GKCSparseVector<ScalarT> const &mat)
            {
                mat.printInfo(os);
                return os;
            }
            
        private:
            IndexType m_num_vals;
            IndexType m_num_stored_vals;
            bool m_weighted;

            // Two array compressed sparse vector
            mutable std::vector<IndexType> m_indices;
            mutable std::vector<ScalarType> m_weights;
        };

    } // namespace backend

} // namespace grb
