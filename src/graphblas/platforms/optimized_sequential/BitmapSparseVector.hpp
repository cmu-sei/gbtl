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
#include <numeric>

namespace grb
{
    namespace backend
    {
        /**
         * @brief Class representing a sparse vector by using a bitmap + dense vector
         */
        template<typename ScalarT>
        class BitmapSparseVector
        {
        public:
            using ScalarType = ScalarT;

            // Ambiguous with size constructor
            // template <typename OtherVectorT>
            // BitmapSparseVector(OtherVectorT const &rhs)
            //     : m_size(rhs.m_size),
            //       m_nvals(rhs.m_nvals),
            //       m_vals(rhs.m_vals.size()),
            //       m_bitmap(rhs.m_bitmap)
            // {
            //     for (size_t ix = 0; ix < rhs.m_vals.size(); ++ix)
            //     {
            //         m_vals[ix] =
            //             static_cast<ScalarType>(rhs.m_vals[ix]);
            //     }
            // }

            /**
             * @brief Construct an empty sparse vector with given size
             *
             * @param[in] nsize  Size of vector.
             */
            BitmapSparseVector(IndexType nsize)
                : m_size(nsize),
                  m_nvals(0),
                  m_vals(nsize),
                  m_bitmap(nsize, false)
            {
                if (nsize == 0)
                {
                    throw InvalidValueException();
                }
            }

            BitmapSparseVector(IndexType nsize, ScalarT const &value)
                : m_size(nsize),
                  m_nvals(0),
                  m_vals(nsize, value),
                  m_bitmap(nsize, true)
            {
            }

            /**
             * @brief Construct from a dense vector.
             *
             * @param[in]  rhs  The dense vector to assign to this BitmapSparseVector.
             *                  Size is implied by the vector.
             * @return *this.
             */
            BitmapSparseVector(std::vector<ScalarT> const &rhs)
                : m_size(rhs.size()),
                  m_nvals(rhs.size()),
                  m_vals(rhs),
                  m_bitmap(rhs.size(), true)
            {
                if (rhs.size() == 0)
                {
                    throw InvalidValueException();
                }
            }

            /**
             * @brief Construct a sparse vector from a dense array and zero val.
             *
             * @param[in]  rhs  The dense vector to assign to this BitmapSparseVector.
             *                  Size is implied by the vector.
             * @param[in]  zero An values in the rhs equal to this value will result
             *                  in an implied zero in the resulting sparse vector
             * @return *this.
             */
            BitmapSparseVector(std::vector<ScalarT> const &rhs,
                               ScalarT const              &zero)
                : m_size(rhs.size()),
                  m_nvals(0),
                  m_vals(rhs.size()),
                  m_bitmap(rhs.size(), false)
            {
                if (rhs.size() == 0)
                {
                    throw InvalidValueException();
                }

                for (IndexType idx = 0; idx < rhs.size(); ++idx)
                {
                    if (rhs[idx] != zero)
                    {
                        m_vals[idx] = rhs[idx];
                        m_bitmap[idx] = true;
                        ++m_nvals;
                    }
                }
            }

            /**
             * @brief Construct from index and value arrays.
             * @deprecated Use vectorBuild method
             */
            BitmapSparseVector(
                IndexType                     nsize,
                std::vector<IndexType> const &indices,
                std::vector<ScalarT>   const &values)
                : m_size(nsize),
                  m_nvals(0),
                  m_vals(nsize),
                  m_bitmap(nsize, false)
            {
                /// @todo check for same size indices and values
                for (IndexType idx = 0; idx < indices.size(); ++idx)
                {
                    IndexType i = indices[idx];
                    if (i >= m_size)
                    {
                        throw DimensionException();  // Should this be IndexOutOfBounds?
                    }

                    m_vals[i] = values[idx];
                    m_bitmap[i] = true;
                    ++m_nvals;
                }
            }

            /**
             * @brief Copy constructor for BitmapSparseVector.
             *
             * @param[in] rhs  The BitmapSparseVector to copy construct this
             *                 BitmapSparseVector from.
             */
            BitmapSparseVector(BitmapSparseVector<ScalarT> const &rhs)
                : m_size(rhs.m_size),
                  m_nvals(rhs.m_nvals),
                  m_vals(rhs.m_vals),
                  m_bitmap(rhs.m_bitmap)
            {
            }

            ~BitmapSparseVector() {}

            /**
             * @brief Copy assignment.
             *
             * @param[in] rhs  The BitmapSparseVector to assign to this
             *
             * @return *this.
             */
            BitmapSparseVector<ScalarT>& operator=(
                BitmapSparseVector<ScalarT> const &rhs)
            {
                if (this != &rhs)
                {
                    if (m_size != rhs.m_size)
                    {
                        throw DimensionException();
                    }

                    m_nvals = rhs.m_nvals;
                    m_vals = rhs.m_vals;
                    m_bitmap = rhs.m_bitmap;
                }
                return *this;
            }

            /**
             * @brief Assignment from a dense vector.
             *
             * @param[in]  rhs  The dense vector to assign to this BitmapSparseVector.
             *
             * @return *this.
             */
            BitmapSparseVector<ScalarT>& operator=(std::vector<ScalarT> const &rhs)
            {
                if (rhs.size() != m_size)
                {
                    throw DimensionException();
                }
                for (IndexType idx = 0; idx < rhs.size(); ++idx)
                {
                    m_vals[idx] = rhs[idx];
                    m_bitmap[idx] = true;
                }
                m_nvals = m_size;
                return *this;
            }

            // EQUALITY OPERATORS
            /**
             * @brief Equality testing for BitmapSparseVector.
             * @param rhs The right hand side of the equality operation.
             * @return If this BitmapSparseVector and rhs are identical.
             */
            bool operator==(BitmapSparseVector<ScalarT> const &rhs) const
            {
                if ((m_size != rhs.m_size) || (m_nvals != rhs.m_nvals))
                {
                    return false;
                }

                for (IndexType i = 0; i < m_size; ++i)
                {
                    if (m_bitmap[i] != rhs.m_bitmap[i])
                    {
                        return false;
                    }
                    if (m_bitmap[i])
                    {
                        if (m_vals[i] != rhs.m_vals[i])
                        {
                            return false;
                        }
                    }
                }

                return true;
            }

            /**
             * @brief Inequality testing for BitmapSparseVector.
             * @param rhs The right hand side of the inequality operation.
             * @return If this BitmapSparseVector and rhs are not identical.
             */
            bool operator!=(BitmapSparseVector<ScalarT> const &rhs) const
            {
                return !(*this == rhs);
            }

        public:
            // METHODS

            void clear()
            {
                m_nvals = 0;
                //m_vals.clear();
                m_bitmap.assign(m_size, false);
            }

            IndexType size() const { return m_size; }
            IndexType nvals() const { return m_nvals; }

            /**
             * @brief Resize the vector (smaller or larger)
             *
             * @param[in]  new_size  New number of elements (zero is invalid)
             *
             */
            void resize(IndexType new_size)
            {
                // Check in the frontend
                //if (nsize == 0)
                //   throw InvalidValueException();

                if (new_size < m_size)
                {
                    m_size = new_size;
                    // compute new m_nvals when shrinking
                    if (new_size < m_size/2)
                    {
                        // count remaining elements
                        IndexType new_nvals = 0UL;
                        new_nvals = std::reduce(m_bitmap.begin(),
                                                m_bitmap.begin() + new_size,
                                                new_nvals,
                                               std::plus<IndexType>());
                        m_nvals = new_nvals;
                    }
                    else
                    {
                        // count elements to be removed
                        IndexType num_vals = 0UL;
                        num_vals = std::reduce(m_bitmap.begin() + new_size,
                                               m_bitmap.end(),
                                               num_vals,
                                               std::plus<IndexType>());
                        m_nvals -= num_vals;
                    }

                    m_bitmap.resize(new_size);
                    m_vals.resize(new_size);
                }
                else if (new_size > m_size)
                {
                    m_vals.resize(new_size);
                    m_bitmap.resize(new_size, false);
                    m_size = new_size;
                }
            }

            /**
             *
             */
            template<typename RAIteratorIT,
                     typename RAIteratorVT,
                     typename BinaryOpT = grb::Second<ScalarType> >
            void build(RAIteratorIT  i_it,
                       RAIteratorVT  v_it,
                       IndexType     nvals,
                       BinaryOpT     dup = BinaryOpT())
            {
                std::vector<ScalarType> vals(m_size);
                std::vector<bool> bitmap(m_size);

                /// @todo check for same size indices and values
                for (IndexType idx = 0; idx < nvals; ++idx)
                {
                    IndexType i = i_it[idx];
                    if (i >= m_size)
                    {
                        throw IndexOutOfBoundsException();
                    }

                    if (bitmap[i] == true)
                    {
                        vals[i] = dup(vals[i], v_it[idx]);
                    }
                    else
                    {
                        vals[i] = v_it[idx];
                        bitmap[i] = true;
                    }
                }

                m_vals.swap(vals);
                m_bitmap.swap(bitmap);
                m_nvals = nvals;
            }

            bool hasElement(IndexType index) const
            {
                if (index >= m_size)
                {
                    throw IndexOutOfBoundsException();
                }

                return m_bitmap[index];
            }

            bool hasElementNoCheck(IndexType index) const
            {
                return m_bitmap[index];
            }

            /**
             * @brief Access the elements of this BitmapSparseVector given index.
             *
             * Function provided to access the elements of this BitmapSparseVector
             * given the index.
             *
             * @param[in] index  Position to access.
             *
             * @return The element of this BitmapSparseVector at the given row and
             *         column.
             */
            ScalarT extractElement(IndexType index) const
            {
                if (index >= m_size)
                {
                    throw IndexOutOfBoundsException();
                }

                if (m_bitmap[index] == false)
                {
                    throw NoValueException();
                }

                return m_vals[index];
            }

            ScalarT extractElementNoCheck(IndexType index) const
            {
                return m_vals[index];
            }

            /// @todo Not certain about this implementation
            void setElement(IndexType      index,
                            ScalarT const &new_val)
            {
                if (index >= m_size)
                {
                    throw IndexOutOfBoundsException();
                }
                m_vals[index] = new_val;
                if (m_bitmap[index] == false)
                {
                    ++m_nvals;
                    m_bitmap[index] = true;
                }
            }

            /// @todo Not certain about this implementation
            void setElementNoCheck(IndexType      index,
                                   ScalarT const &new_val)
            {
                m_vals[index] = new_val;
                if (m_bitmap[index] == false)
                {
                    ++m_nvals;
                    m_bitmap[index] = true;
                }
            }

            void removeElement(IndexType index)
            {
                if (index >= m_size)
                {
                    throw IndexOutOfBoundsException();
                }

                if (m_bitmap[index] == true)
                {
                    --m_nvals;
                    m_bitmap[index] = false;
                }
            }

            void removeElementNoCheck(IndexType index)
            {
                if (m_bitmap[index] == true)
                {
                    --m_nvals;
                    m_bitmap[index] = false;
                }
            }

            template<typename RAIteratorIT,
                     typename RAIteratorVT>
            void extractTuples(RAIteratorIT        i_it,
                               RAIteratorVT        v_it) const
            {
                for (IndexType idx = 0; idx < m_size; ++idx)
                {
                    if (m_bitmap[idx])
                    {
                        *i_it = idx;         ++i_it;
                        *v_it = m_vals[idx]; ++v_it;
                    }
                }
            }

            void extractTuples(IndexArrayType        &indices,
                               std::vector<ScalarT>  &values) const
            {
                extractTuples(indices.begin(), values.begin());
            }

            // output specific to the storage layout of this type of matrix
            void printInfo(std::ostream &os) const
            {
                os << "Optimized Sequential Backend: ";
                os << "backend::BitmapSparseVector<" << typeid(ScalarT).name() << ">";
                os << ", size  = " << m_size;
                os << ", nvals = " << m_nvals << std::endl;

                os << "[";
                if (m_bitmap[0]) os << m_vals[0]; else os << "-";
                for (IndexType idx = 1; idx < m_size; ++idx)
                {
                    if (m_bitmap[idx]) os << ", " << m_vals[idx]; else os << ", -";
                }
                os << "]";
            }

            friend std::ostream &operator<<(std::ostream             &os,
                                            BitmapSparseVector<ScalarT> const &mat)
            {
                mat.printInfo(os);
                return os;
            }
#if 0
            // =======================================
            // Iterator class
            // =======================================
            // a forward iterator only...for now.
            class iterator
            {
            public:
                iterator(BitmapSparseVector<ScalarT> &vec,
                         grb::IndexType               curr = 0) :
                    m_bsvec(vec), m_curr(curr)
                {
                    //std::cout << "Constructed: size = " << m_bsvec.m_size
                    //          << ", nvals = " << m_bsvec.m_nvals << std::endl;
                    if (!m_bsvec.m_bitmap[m_curr] && m_curr < m_bsvec.m_size)
                        advance();
                    //std::cout << "ctor DONE" << std::endl;
                }

                std::tuple<grb::IndexType, ScalarT&> operator*()
                {
                    return std::tuple<grb::IndexType, ScalarT&>(
                        m_curr, m_bsvec.m_vals[m_curr]);
                }

                iterator &operator++()
                {
                    //std::cout << "operator++" << std::endl;
                    advance();
                    return *this;
                }

                iterator &operator++(int)
                {
                    advance();
                    return iterator(m_bsvec, m_curr++);
                }

                bool operator==(iterator const &rhs)
                {
                    return (&m_bsvec == &rhs.m_bsvec) && (m_curr == rhs.m_curr);
                }

                bool operator!=(iterator const &rhs)
                {
                    return !operator==(rhs);
                }

            private:
                void advance()
                {
                    while (++m_curr < m_bsvec.m_size && !m_bsvec.m_bitmap[m_curr])
                    {
                        //std::cout << "advanced: " << m_curr << std::endl;
                    }
                    //std::cout << "advance DONE: " << m_curr << std::endl;
                }

            private:
                BitmapSparseVector<ScalarT> &m_bsvec;
                grb::IndexType               m_curr;  // current location
            };

            friend class iterator;

        public:
            iterator begin()
            {
                return iterator(*this);
            }

            const iterator begin() const
            {
                return const_iterator(*this);
            }

            iterator end()
            {
                return iterator(*this, m_size);
            }

            const iterator end() const
            {
                return iterator(*this, m_size);
            }
#endif
        public:
            std::vector<bool>    const &get_bitmap() const { return m_bitmap; }
            std::vector<ScalarT> const &get_vals() const   { return m_vals; }

            std::vector<std::tuple<IndexType,ScalarT> > getContents() const
            {
                std::vector<std::tuple<IndexType,ScalarT> > contents;
                contents.reserve(m_nvals);
                for (IndexType idx = 0; idx < m_size; ++idx)
                {
                    if (m_bitmap[idx])
                    {
                        contents.emplace_back(idx, m_vals[idx]);
                    }
                }
                return contents;
            }

            template <typename OtherScalarT>
            void setContents(
                std::vector<std::tuple<IndexType,OtherScalarT> > const &contents)
            {
                clear();
                for (auto&& [idx, val] : contents)
                {
                    m_bitmap[idx] = true;
                    m_vals[idx]   = static_cast<ScalarT>(val);
                    ++m_nvals;
                }
            }

        private:
            IndexType             m_size;
            IndexType             m_nvals;
            std::vector<ScalarT>  m_vals;
            std::vector<bool>     m_bitmap;
        };
    } // backend
} // grb
