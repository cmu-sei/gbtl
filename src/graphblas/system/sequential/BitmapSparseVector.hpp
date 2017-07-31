/*
 * Copyright (c) 2017 Carnegie Mellon University.
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

#ifndef GB_SEQUENTIAL_BITMAPSPARSEVECTOR_HPP
#define GB_SEQUENTIAL_BITMAPSPARSEVECTOR_HPP

#include <iostream>
#include <vector>
#include <typeinfo>

namespace GraphBLAS
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
            typedef ScalarT ScalarType;

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
             * @brief Construct an empty list of lists sparse matrix with
             *        the given shape.
             *
             * @param[in] num_vals  width of matrix
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

            BitmapSparseVector(IndexType const &nsize, ScalarT const &value)
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

            // FUNCTIONS

            /**
             *
             */
            template<typename RAIteratorIT,
                     typename RAIteratorVT,
                     typename BinaryOpT = GraphBLAS::Second<ScalarType> >
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

            void clear()
            {
                m_nvals = 0;
                //m_vals.clear();
                m_bitmap.assign(m_size, false);
            }

            IndexType size() const { return m_size; }
            IndexType nvals() const { return m_nvals; }

            bool hasElement(IndexType index) const
            {
                if (index >= m_size)
                {
                    throw IndexOutOfBoundsException();
                }

                return m_bitmap[index];
            }

            /**
             * @brief Access the elements of this BitmapSparseVector given row and
             *        column indexes.
             *
             * Function provided to access the elements of this BitmapSparseVector
             * given row and column indices.  The functionality is the
             * same as that of the indexing function for a standard dense
             * matrix.
             *
             * @param[in] row_index  The row to access.
             * @param[in] col_index  The column to access.
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

            // Not certain about this implementation
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

            template<typename RAIteratorIT,
                     typename RAIteratorVT>
            void extractTuples(RAIteratorIT        i_it,
                               RAIteratorVT        v_it) const
            {
                for (IndexType idx = 0; idx < m_size; ++idx)
                {
                    if (m_bitmap[idx])
                    {
                        *i_it = idx; ++i_it;
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
                os << "BitmapSparseVector<" << typeid(ScalarT).name() << ">" << std::endl;
                os << "size  = " << m_size << std::endl;
                os << "nvals = " << m_nvals << std::endl;
                os << "contents: [";
                if (m_bitmap[0]) os << m_vals[0]; else os << "-";
                for (IndexType idx = 1; idx < m_size; ++idx)
                {
                    if (m_bitmap[idx]) os << ", " << m_vals[idx]; else os << ", -";
                }
                os << "]" << std::endl;
            }

            friend std::ostream &operator<<(std::ostream             &os,
                                            BitmapSparseVector<ScalarT> const &mat)
            {
                mat.printInfo(os);
                return os;
            }

            std::vector<bool> const &get_bitmap() const { return m_bitmap; }
            std::vector<ScalarT> const &get_vals() const { return m_vals; }

            std::vector<std::tuple<IndexType,ScalarT> > getContents() const
            {
                std::vector<std::tuple<IndexType,ScalarT> > contents;
                contents.reserve(m_nvals);
                for (IndexType idx = 0; idx < m_size; ++idx)
                {
                    if (m_bitmap[idx])
                    {
                        contents.push_back(std::make_tuple(idx, m_vals[idx]));
                    }
                }
                return contents;
            }

            template <typename OtherScalarT>
            void setContents(
                std::vector<std::tuple<IndexType,OtherScalarT> > const &contents)
            {
                clear();
                for (auto tupl : contents)
                {
                    ++m_nvals;
                    m_bitmap[std::get<0>(tupl)] = true;
                    m_vals[std::get<0>(tupl)] = static_cast<ScalarT>(std::get<1>(tupl));
                }
            }

        private:
            IndexType const       m_size;   // immutable after construction
            IndexType             m_nvals;
            std::vector<ScalarT>  m_vals;
            std::vector<bool>     m_bitmap;
        };
    } // backend
} // GraphBLAS



#endif // GB_SEQUENTIAL_BITMAPSPARSEVECTOR_HPP
