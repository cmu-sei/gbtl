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
         * @brief Class representing a sparse-valued vector using a dense
         * bitmap and a dense-coded values vector. 
         * Based on frontier class in GKC 1.0
         */

        template <typename ScalarT>
        class GKCDenseVector
        {
        public:
            using ScalarType = ScalarT;
            using BitT = char;

            // Constructor with some default values
            /**
             * @brief construct an empty dense-coded vector with space for 
             * num_vals values.
             * The vector can be structure only, with weights set
             * to false, so only vertex IDs are stored.
             */
            GKCDenseVector(IndexType num_vals,
                            bool weighted = true)
                : m_num_vals(num_vals),
                  m_num_stored_vals(0),
                  m_weighted(weighted)
            {
                if (m_num_vals <= 0)
                {
                    throw InvalidValueException();
                }
                m_bitmap.resize(m_num_vals);
                if (m_weighted)
                    m_weights.resize(m_num_vals);
            }

            // Create GKC Dense Vector with default weight/value
            GKCDenseVector(IndexType num_vals, ScalarT const &value)
                : m_num_vals(num_vals),
                  m_num_stored_vals(num_vals),
                  m_weighted(true),
                  m_weights(num_vals, value)
            {
                if (m_num_vals <= 0)
                {
                    throw InvalidValueException();
                }
                // Fill with ascending indices
                for (auto idx = 0; idx < m_num_vals; idx++)
                {
                    m_bitmap[idx] = true;
                    m_weights[idx] = value;
                }
            }

            // Constructor - copy
            GKCDenseVector(GKCDenseVector<ScalarT> const &rhs)
                : m_num_vals(rhs.m_num_vals),
                  m_weighted(rhs.m_weighted),
                  m_weights(rhs.m_weights),
                  m_num_stored_vals(rhs.m_num_stored_vals),
                  m_bitmap(rhs.m_bitmap)
            {
            }

            // Constructor - dense from dense vector
            GKCDenseVector(std::vector<ScalarT> const &val)
                : m_weighted(true)
            {
                m_num_vals = val.size();
                m_num_stored_vals = m_num_vals;
                // Dense has both structure and weights.
                m_weighted = true;
                m_weights.resize(m_num_vals);
                m_bitmap.resize(m_num_vals);

                // Copy values from dense vector into sparse
                for (auto idx = 0; idx < m_num_vals; idx++){
                    m_bitmap[idx] = true;
                    m_weights[idx] = val[idx];
                }
            }

            // Constructor - sparse from dense vector, removing specifed implied zeros
            GKCDenseVector(std::vector<ScalarT> const &val,
                            ScalarT zero)
                : m_weighted(true)
            {
                m_num_vals = val.size();
                // Dense has both structure and weights.
                m_weighted = true;
                m_weights.resize(m_num_vals);
                m_bitmap.resize(m_num_vals);
                m_num_stored_vals = 0;

                // Copy values from dense vector into sparse
                // While ignoring 'zero' elements
                for (auto idx = 0; idx < m_num_vals; idx++)
                {
                    if (val[idx] != zero)
                    {
                        m_bitmap[idx] = true;
                        m_weights[idx] = val[idx];
                        m_num_stored_vals++;
                    }
                }
            }

            // Build Constructor - parse sparse coordinate data and construct vector.
            // Similar to build method, but baked into the constructor.
            // Does not use addElement.
            template <typename RAIteratorI,
                      typename RAIteratorV,
                      typename BinaryOpT = grb::Second<ScalarType>>
            GKCDenseVector(RAIteratorI i_it,
                            RAIteratorV v_it,
                            IndexType n,
                            BinaryOpT dup = BinaryOpT(),
                            bool weighted = true)
            {
                /// @todo require random access iterators
                // scan the data to determine num_vals
                /// @todo: OMP max reduction
                IndexType max_idx = 0;
                for (size_t i = 0; i < n; i++)
                {
                    max_idx = std::max(*(i_it + i), max_idx);
                }

                /// Allocate to the max vertex ID
                m_num_vals = max_idx + 1;
                m_num_stored_vals = 0;

                // allocate memory
                m_weighted = weighted;
                if (m_weighted)
                    m_weights.resize(m_num_vals);
                m_bitmap.resize(m_num_vals);
                for (auto && itm : m_bitmap)
                    itm = false;

                // Copy data from iterators into vector
                #pragma omp parallel for
                for (IndexType idx = 0; idx < n; idx++)
                {
                    IndexType vidx = *(i_it + idx);
                    bool found = m_bitmap[vidx];
                    if (!found)
                    {
                        // Index not recognized, add new entry
                        m_bitmap[vidx] = true;
                        // #pragma omp atomic
                        m_num_stored_vals++;
                        if (m_weighted)
                            m_weights[vidx] = *(v_it + idx);
                    }
                    else if (m_weighted)
                    {
                        // Already have a value, so merge the weight.
                        auto old_idx = vidx;
                        m_weights[old_idx] = dup(m_weights[old_idx], *(v_it + idx));
                    }
                }
            }

            // Destructor
            ~GKCDenseVector()
            {
            }

            // Copy assignment (currently restricted to same dimensions)
            GKCDenseVector<ScalarT> &operator=(GKCDenseVector<ScalarT> const &rhs)
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
                    m_weights = rhs.m_weights;
                    m_bitmap = rhs.m_bitmap;
                }
                return *this;
            }

            // Copy assignment (for diff types, has cast)
            template <typename AltScalarT>
            GKCDenseVector<ScalarT> &operator=(GKCDenseVector<AltScalarT> const &rhs)
            {
                // if (this != &rhs)  // Assume they are different pointers since the
                // ScalarT to ScalarT assignment would otherwise have been used (above)
                if (m_num_vals != rhs.size())
                {
                    throw DimensionException("Dimensions of vectors do not match.");
                }
                m_num_vals = rhs.size();
                m_num_stored_vals = rhs.nvals();
                m_weighted = rhs.isWeighted();
                m_bitmap = rhs.getBitmap();

                if (m_weighted)
                {
                    auto w_ref = rhs.getWeights();
                    #pragma omp parallel for
                    for (size_t idx = 0; idx < m_num_vals; idx++)
                    {
                        m_weights[idx] = (ScalarT)(w_ref[idx]);
                    }
                }
                return *this;
            }

            // Move assignment
            GKCDenseVector<ScalarT> &operator=(GKCDenseVector<ScalarT> const &&rhs)
            {
                if (this != &rhs)
                {
                    if (m_num_vals != rhs.m_num_vals)
                    {
                        throw DimensionException("Dimensions of vectors do not match.");
                    }
                    m_num_vals = std::move(rhs.m_num_vals);
                    m_num_stored_vals = std::move(rhs.m_num_stored_vals);
                    m_weighted = std::move(rhs.m_weighted);
                    m_weights = std::move(rhs.m_weights);
                    m_bitmap = std::move(rhs.m_bitmap);
                }
                return *this;
            }

            void swap(GKCDenseVector<ScalarT> &rhs)
            {
                std::swap(m_num_vals, rhs.m_num_vals);
                std::swap(m_num_stored_vals, rhs.m_num_stored_vals);
                std::swap(m_weighted, rhs.m_weighted);
                std::swap(m_weights, rhs.m_weights);
                std::swap(m_bitmap, rhs.m_bitmap);
            }

            // EQUALITY OPERATORS
            /**
             * @brief Equality testing for GKC Vector.
             * @param rhs The right hand side of the equality operation.
             * @return If this GKC Vector and rhs are identical.
             * @todo: need create sets for the vectors to compare their contents.
             * @todo: lookup mutable keyword
             */
            bool operator==(GKCDenseVector<ScalarT> const &rhs) const
            {
                bool test1 = (m_num_vals == rhs.m_num_vals);
                bool test2 = (m_weighted == rhs.m_weighted);
                bool test3 = (m_num_stored_vals == rhs.m_num_stored_vals);
                bool test4 = (m_bitmap == rhs.m_bitmap);
                // Check only weights that have valid set bit
                bool test5 = m_weights.size() == rhs.m_weights.size();
                for (size_t idx = 0; idx < m_weights.size(); idx++)
                {
                    if (!test5 || !test4) break;
                    if (m_bitmap[idx])
                    {
                        test5 &= m_weights[idx] == rhs.m_weights[idx];
                    }
                }
                // std::cout << std::endl;
                // std::cout << (test1 ? "true" : "false") << std::endl;
                // std::cout << (test2 ? "true" : "false") << std::endl;
                // std::cout << (test3 ? "true" : "false") << std::endl;
                // std::cout << (test4 ? "true" : "false") << std::endl;
                // std::cout << (test5 ? "true" : "false") << std::endl;
                // std::cout << (m_weighted ? "true" : "false") << std::endl;
                return (test1 && test2 && test3 && test4 && ((m_weighted && test5) || !m_weighted));
            }

            /**
             * @brief Inequality testing for GKC Vector.
             * @param rhs The right hand side of the inequality operation.
             * @return If this GKC Vector and rhs are not identical.
             */

            bool operator!=(GKCDenseVector<ScalarT> const &rhs) const
            {
                return !(*this == rhs);
            }

            template <typename RAIteratorIT,
                      typename RAIteratorVT,
                      typename BinaryOpT = grb::Second<ScalarType>>
            void build(RAIteratorIT i_it,
                       RAIteratorVT v_it,
                       IndexType n,
                       BinaryOpT dup = BinaryOpT(),
                       bool weighted = true)
            {
                /// @todo require random access iterators
                // scan the data to determine num_vals
                /// @todo: OMP max reduction
                IndexType max_idx = 0;
                for (size_t i = 0; i < n; i++)
                {
                    max_idx = std::max(*(i_it + i), max_idx);
                }

                /// Allocate to the max vertex ID
                m_num_vals = max_idx + 1;
                m_num_stored_vals = 0;

                // allocate memory
                m_weighted = weighted;
                if (m_weighted)
                    m_weights.resize(m_num_vals);
                m_bitmap.resize(m_num_vals);
                for (auto && itm : m_bitmap)
                    itm = false;

                // Copy data from iterators into vector
                #pragma omp parallel for
                for (IndexType idx = 0; idx < n; idx++)
                {
                    IndexType vidx = *(i_it + idx);
                    bool found = m_bitmap[vidx];
                    if (!found)
                    {
                        // Index not recognized, add new entry
                        m_bitmap[vidx] = true;
                        // #pragma omp atomic
                        m_num_stored_vals++;
                        if (m_weighted)
                            m_weights[vidx] = *(v_it + idx);
                    }
                    else if (m_weighted)
                    {
                        // Already have a value, so merge the weight.
                        auto old_idx = vidx;
                        m_weights[old_idx] = dup(m_weights[old_idx], *(v_it + idx));
                    }
                }
            }

            void clear()
            {
                // / @todo make atomic? transactional?
                m_num_stored_vals = 0;
                #pragma omp parallel for
                for (int i = 0; i != size(); ++i)
                    m_bitmap[i] = false;
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
            /*
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
	  */
            bool hasElement(IndexType index) const
            {
                /*
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
	      */
                return m_bitmap[index];
            }

            ScalarT extractElement(IndexType index) const
            {
                if (index >= m_num_vals)
                {
                    throw IndexOutOfBoundsException();
                }

                if (!m_bitmap[index])
                    throw NoValueException();
                else if (m_weighted)
                    return m_weights[index];
                else
                    return (ScalarT)1; //this should throw unweighted exception.

                /*
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
		*/
            }

            bool boolExtractElement(IndexType index, ScalarT &val) const
            {
                if (index >= m_num_vals)
                {
                    throw IndexOutOfBoundsException();
                }

                /*
                for (size_t idx = 0; idx < m_num_stored_vals; idx++)
                {
                    auto vidx = m_indices[idx];
                    if (vidx == index)
                    {
                        if (m_weighted)
                        {
                            val =  m_weights[idx];
                        }
                        else // What to do if no weights?
                        {
                            val =  (ScalarT)1;
                        }
                        return true;
                    }
                }
                val = (ScalarT)1;
		*/

                if (m_bitmap[index] && m_weighted)
                    val = m_weights[index];
                else
                    val = (ScalarT)1 && (bool)m_bitmap[index];
                return m_bitmap[index]; //false;
            }

            void setElement(IndexType index, ScalarT const &new_val)
            {
                if (index >= m_num_vals)
                {
                    throw IndexOutOfBoundsException();
                }

                auto idx = index;

                if (!m_bitmap[idx])
                {
                    // #pragma omp atomic
                    m_num_stored_vals++;
                }

                m_weights[idx] = new_val;
                m_bitmap[idx] = true;

                /*
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
		*/
            }

            template <typename BinaryOpT, typename ZScalarT>
            void mergeSetElement(IndexType index, ZScalarT const &new_val, BinaryOpT op)
            {
                if (index >= m_num_vals)
                {
                    throw IndexOutOfBoundsException();
                }
                if (index >= m_bitmap.size() || (index >= m_weights.size() && m_weighted))
                {
                    throw InvalidIndexException("Mismatch in internal vector size and size of vector.");
                }

                if (m_bitmap[index])
                {
                    if (m_weighted)
                    {
                        m_weights[index] = op(m_weights[index], new_val);
                    }
                    else
                    {
                        //no weights reqiured
                    }
                }
                else
                {
                    m_bitmap[index] = true;
                    if (m_weighted)
                    {
                        // Make sure to correctly cast for the output of the operation,
                        // which in mxv is not the same as the reduction (additive) output.
                        using ZType = decltype(op(
                            std::declval<ScalarT>(),
                            std::declval<ZScalarT>()));
                        m_weights[index] = (ZType)new_val;
                    }
                    // #pragma omp atomic
                    m_num_stored_vals++;
                }
            }

            void removeElement(IndexType index)
            {
                if (index >= m_num_vals)
                {
                    throw IndexOutOfBoundsException();
                }
                // No value found; throw error
                if (!m_bitmap[index])
                    throw NoValueException();
                else
                {
                    m_bitmap[index] = false;
                    // #pragma omp atomic
                    m_num_stored_vals--;
                }
            }

            bool boolRemoveElement(IndexType index)
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

                bool was_set = m_bitmap[index];
                if (was_set)
                {
                    // #pragma omp atomic
                    m_num_stored_vals --;
                    m_bitmap[index] = false;
                }
                return was_set;
            }

/*
            void bulkRemoveElements(std::vector<IndexType> indices)
            {
                size_t rm_count = 0
                for (auto && idx : indices)
                {

                }
            }
            */

            bool isWeighted() const { return m_weighted; }

            // Note: this has to be const because changes to it could break
            // the weights vector.
            // further, using end() on the vector may iterate too far since
            // the vector is resized for the maximum number of vertices,
            // not the amount currently stored.
            /// @todo mark: add iterator for GKC Dense Vector

            const std::vector<ScalarT> &getWeights() const
            {
                if (m_weighted)
                {
                    return m_weights;
                }
                throw NoValueException();
            }

            std::vector<BitT> getBitmap() const {return m_bitmap;}

            using wgt_iterator = typename std::vector<ScalarType>::iterator;
            // Iterators for neighborhoods
            inline wgt_iterator wgtBegin() { return m_weights.begin(); }
            inline wgt_iterator wgtEnd() { return m_weights.end(); }
            // Const versions
            inline const wgt_iterator wgtBegin() const { return m_weights.begin(); }
            inline const wgt_iterator wgtEnd() const { return m_weights.end(); }

            ScalarT operator[](IndexType idx) const
            {
                //if (m_bitmap[*idx])
                //return m_weights[*idx];
                return extractElement(idx);
            }


            // output specific to the storage layout of this type of matrix
            void printInfo(std::ostream &os) const
            {
                os << "GKC Backend: ";
                os << "backend::GKCDenseVector<" << typeid(ScalarT).name() << ">";
                os << ", size  = " << m_num_vals;
                os << ", nvals = " << m_num_stored_vals << std::endl;
                os << "weighted: " << (m_weighted ? "true" : "false") << std::endl;
                os << "m_weights.size(): " << m_weights.size() << std::endl;

                os << "[";
                if (m_num_stored_vals > 0)
                {
                    if (m_bitmap[0])
                    {
                        os << 0;
                        if (m_weighted)
                        {
                            os << ":" << m_weights[0];
                        }
                    }
                }
                for (IndexType idx = 1; idx < m_num_vals; ++idx)
                {
                    if (m_bitmap[idx])
                    {
                        os << ", " << idx;
                        if (m_weighted)
                        {
                            os << ":" << m_weights[idx];
                        }
                    }
                }
                os << "]";
                os << std::endl;
            }

            friend std::ostream &operator<<(std::ostream &os,
                                            GKCDenseVector<ScalarT> const &mat)
            {
                mat.printInfo(os);
                return os;
            }

        private:
            IndexType m_num_vals;
            IndexType m_num_stored_vals;
            bool m_weighted;

            // Two array compressed sparse vector
            std::vector<ScalarType> m_weights;

            std::vector<BitT> m_bitmap;
        };

    } // namespace backend

} // namespace grb
