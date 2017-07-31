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

#ifndef GB_SEQUENTIAL_NEW_COMPLEMENT_VIEW_HPP
#define GB_SEQUENTIAL_NEW_COMPLEMENT_VIEW_HPP

#include <graphblas/system/sequential/Matrix.hpp>

namespace GraphBLAS
{
    namespace backend
    {
        //************************************************************************
        /**
         * @brief View a matrix as if it were structurally complemented; that is,
         *        stored non-zero values become implied zeros, and
         *        implied zeros become 'true'.
         *
         * @note any stored value that evaluates to false will become true in the
         *       complement
         *
         * @tparam MatrixT     Implements the 2D matrix concept.
         */
        template<typename MatrixT>
        class MatrixComplementView
        {
        public:
            typedef bool ScalarType;
            typedef typename MatrixT::ScalarType InternalScalarType;

            // CONSTRUCTORS

            MatrixComplementView(MatrixT const &matrix):
                m_matrix(matrix)
            {
            }

            /**
             * Copy constructor.
             *
             * @param[in] rhs The complement view to copy.
             *
             * @todo Is this const correct?
             * @todo should it complement the complement?
             */
            MatrixComplementView(MatrixComplementView<MatrixT> const &rhs)
                : m_matrix(rhs.m_matrix)
            {
            }

            ~MatrixComplementView()
            {
            }

            IndexType nrows() const { return m_matrix.nrows(); }
            IndexType ncols() const { return m_matrix.ncols(); }
            IndexType nvals() const
            {
                IndexType num_vals = (m_matrix.nrows()*m_matrix.ncols() -
                                      m_matrix.nvals());

                // THIS IS COSTLY
                // Need to detect and deal with stored 'falses' in m_matrix.
                // They count for stored values in the structural complement
                for (IndexType ix = 0; ix < nrows(); ++ix)
                {
                    auto row = getRow(ix);
                    for (auto &ix : row)
                    {
                        if (false == static_cast<bool>(std::get<1>(ix)))
                        {
                            ++num_vals;
                        }
                    }
                }

                return num_vals;
            }

            /**
             * @brief Equality testing for matrix. (value equality?)
             *
             * @param[in] rhs  The right hand side of the equality
             *                 operation.
             *
             * @return true, if this matrix and rhs are identical.
             * @todo  Not sure we need this form.  Should we do equality
             *        with any matrix?
             */
            template <typename OtherMatrixT>
            bool operator==(OtherMatrixT const &rhs) const
            {
                if ((rhs.nrows() != nrows()) &&
                    (rhs.ncols() != ncols()) &&
                    (rhs.nvals() != nvals()))
                {
                    return false;
                }

                std::vector<IndexType> i(nvals());
                std::vector<IndexType> j(nvals());
                std::vector<typename OtherMatrixT::ScalarType> v(nvals());
                rhs.extractTuples(i.begin(), j.begin(), v.begin());

                for (IndexType ix = 0; ix < nvals(); ++ix)
                {
                    if (!hasElement(i[ix], j[ix]))
                    {
                        return false;
                    }
                    if (extractElement(i[ix], j[ix]) != v[ix])
                    {
                        return false;
                    }
                }

                return true;
            }


            /**
             * @brief Inequality testing for matrix. (value equality?)
             *
             * @param[in] rhs  The right hand side of the inequality
             *                 operation.
             *
             * @return true, if this matrix and rhs are not identical.
             */
            template <typename OtherMatrixT>
            bool operator!=(OtherMatrixT const &rhs) const
            {
                return !(*this == rhs);
            }


            bool hasElement(IndexType irow, IndexType icol) const
            {
                if (!m_matrix.hasElement(irow, icol))
                {
                    return true;
                }
                else if (false ==
                         static_cast<bool>(m_matrix.extractElement(irow, icol)))
                {
                    return true;
                }

                return false;
            }

            /**
             * @brief Access the elements given row and column indexes.
             *
             * Function provided to access the elements given row and
             * column indices.  The functionality is the same as that
             * of the indexing function for a standard dense matrix.
             *
             * @param[in] row  The index of row to access.
             * @param[in] col  The index of column to access.
             *
             * @return The complemented element at the given row and column.
             *
             * @todo this needs to go away
             */
            //InternalScalarType extractElement(IndexType row, IndexType col) const
            bool extractElement(IndexType row, IndexType col) const
            {
                if (!m_matrix.hasElement(row, col))
                {
                    return true;
                }
                else if (false ==
                         static_cast<bool>(m_matrix.extractElement(row, col)))
                {
                    return true;
                }

                throw GraphBLAS::NoValueException();
            }

            std::vector<std::tuple<IndexType, bool> > getRow(
                IndexType row) const
            {
                std::vector<std::tuple<IndexType, InternalScalarType> > row_tuples =
                    m_matrix.getRow(row);
                std::vector<std::tuple<IndexType, bool> > mask_tuples;
                auto it = row_tuples.begin();
                for (IndexType ix = 0; ix < ncols(); ++ix)
                {
                    if ((it == row_tuples.end()) || (ix < std::get<0>(*it)))
                    {
                        mask_tuples.push_back(std::make_tuple(ix, true));
                    }
                    else
                    {
                        if (static_cast<bool>(std::get<1>(*it)) == false)
                        {
                            mask_tuples.push_back(std::make_tuple(ix, true));
                        }
                        ++it;
                    }
                }

                return mask_tuples;

            }

            std::vector<std::tuple<IndexType, bool> > getCol(
                IndexType col) const
            {
                std::vector<std::tuple<IndexType, InternalScalarType> > col_tuples =
                    m_matrix.getCol(col);
                std::vector<std::tuple<IndexType, bool> > mask_tuples;
                auto it = col_tuples.begin();
                for (IndexType ix = 0; ix < nrows(); ++ix)
                {
                    if ((it == col_tuples.end()) || (ix < std::get<0>(*it)))
                    {
                        mask_tuples.push_back(std::make_tuple(ix, true));
                    }
                    else
                    {
                        if (static_cast<bool>(std::get<1>(*it)) == false)
                        {
                            mask_tuples.push_back(std::make_tuple(ix, true));
                        }
                        ++it;
                    }
                }

                return mask_tuples;
            }

            // Get column indices for a given row
            void getColumnIndices(IndexType irow, IndexArrayType &indices) const
            {
                auto row = getRow(irow);
                indices.clear();
                for (auto tuple : row)
                {
                    indices.push_back(std::get<0>(tuple));
                }
            }

            // Get row indices for a given column
            void getRowIndices(IndexType icol, IndexArrayType &indices) const
            {
                auto col = getCol(icol);
                indices.clear();
                for (auto tuple : col)
                {
                    indices.push_back(std::get<0>(tuple));
                }
            }

            void printInfo(std::ostream &os) const
            {
                os << "Backend MatrixComplementView of:" << std::endl;
                m_matrix.printInfo(os);

                for (IndexType row = 0; row < nrows(); ++row)
                {
                    os << ((row == 0) ? "[[" : " [");
                    if (ncols() > 0)
                    {
                        os << (hasElement(row, 0) ? "1" : "-");
                    }

                    for (IndexType col = 1; col < ncols(); ++col)
                    {
                        os << ", " << (hasElement(row, col) ? "1" : "-");
                    }
                    os << ((row == nrows() - 1) ? "]]" : "]\n");
                }
            }

            friend std::ostream&
            operator<<(std::ostream                  &os,
                       MatrixComplementView<MatrixT> const &mat)
            {
                mat.printInfo(os);
                return os;
            }

        private:
            /**
             * Copy assignment not implemented.
             *
             * @param[in] rhs  The complement view to assign to this.
             *
             * @todo Assignment should be disallowed as you cannot reassign a
             *       a reference.
             */
            MatrixComplementView<MatrixT> &
            operator=(MatrixComplementView<MatrixT> const &rhs) = delete;

        private:
            MatrixT const &m_matrix;
        };

//****************************************************************************
//****************************************************************************

        //************************************************************************
        /**
         * @brief View a vector as if it were structurally complemented; that is,
         *        stored non-zero values become implied zeros, and
         *        implied zeros become 'true'.
         *
         * @note any stored value that evaluates to false will become true in the
         *       complement
         *
         * @tparam VectorT     Implements the 2D vector concept.
         */
        template<typename VectorT>
        class VectorComplementView
        {
        public:
            typedef bool ScalarType;
            typedef typename VectorT::ScalarType InternalScalarType;

            // CONSTRUCTORS

            VectorComplementView(VectorT const &vector):
                m_vector(vector)
            {
            }

            /**
             * Copy constructor.
             *
             * @param[in] rhs The complement view to copy.
             *
             * @todo Is this const correct?
             * @todo should it complement the complement?
             */
            VectorComplementView(VectorComplementView<VectorT> const &rhs)
                : m_vector(rhs.m_vector)
            {
            }

            ~VectorComplementView()
            {
            }

            IndexType size() const  { return m_vector.size(); }
            IndexType nvals() const
            {
                // THIS IS COSTLY
                IndexType num_vals(0);
                auto bitmap(m_vector.get_bitmap());
                auto vals(m_vector.get_vals());

                for (IndexType idx = 0; idx < size(); ++idx)
                {
                    if (!bitmap[idx] || !vals[idx])
                    {
                        ++num_vals;
                    }
                }
                return num_vals;
            }

            /**
             * @brief Equality testing for vector. (value equality?)
             *
             * @param[in] rhs  The right hand side of the equality
             *                 operation.
             *
             * @return true, if this vector and rhs are identical.
             * @todo  Not sure we need this form.  Should we do equality
             *        with any vector?
             */
            template <typename OtherVectorT>
            bool operator==(OtherVectorT const &rhs) const
            {
                if (size() != rhs.size())
                    //    || (nvals() != rhs.nvals())) // Comparing nvals is tricky
                {
                    return false;
                }

                /// @todo stored zeros need to evaluate to true in the complement
                throw 1;  // Not implemented yet

                /// @todo Not implemented yet.

                return true;
            }


            /**
             * @brief Inequality testing for vector. (value equality?)
             *
             * @param[in] rhs  The right hand side of the inequality
             *                 operation.
             *
             * @return true, if this vector and rhs are not identical.
             */
            template <typename OtherVectorT>
            bool operator!=(OtherVectorT const &rhs) const
            {
                return !(*this == rhs);
            }


            bool hasElement(IndexType index) const
            {
                if (!m_vector.hasElement(index))
                {
                    return true;
                }
                else if (false == static_cast<bool>(m_vector.extractElement(index)))
                {
                    return true;
                }

                return false;
            }

            /**
             *
             */
            //InternalScalarType extractElement(IndexType row, IndexType col) const
            bool extractElement(IndexType index) const
            {
                if (!m_vector.hasElement(index))
                {
                    return true;
                }
                else if (false == static_cast<bool>(m_vector.extractElement(index)))
                {
                    return true;
                }

                throw GraphBLAS::NoValueException();
            }

            std::vector<std::tuple<IndexType, bool> > getContents() const
            {
                auto bitmap(m_vector.get_bitmap());
                auto vals(m_vector.get_vals());

                std::vector<std::tuple<IndexType, bool> > contents;
                //contents.reserve(nvals());
                for (IndexType idx = 0; idx < size(); ++idx)
                {
                    if (!bitmap[idx] || !vals[idx])
                    {
                        contents.push_back(std::make_tuple(idx, true));
                    }
                }
                return contents;
            }

            void printInfo(std::ostream &os) const
            {
                os << "Backend VectorComplementView of:" << std::endl;
                m_vector.printInfo(os);

                for (IndexType idx = 0; idx < size(); ++idx)
                {
                    os << ((idx == 0) ? "[[" : " [");
                    if (size() > 0)
                    {
                        os << (hasElement(idx) ? "1" : "-");
                    }
                    os << ((idx == size() - 1) ? "]]" : "]\n");
                }
            }

            friend std::ostream&
            operator<<(std::ostream                        &os,
                       VectorComplementView<VectorT> const &vec)
            {
                vec.printInfo(os);
                return os;
            }

        private:
            /**
             * Copy assignment not implemented.
             *
             * @param[in] rhs  The complement view to assign to this.
             *
             * @todo Assignment should be disallowed as you cannot reassign a
             *       a reference.
             */
            VectorComplementView<VectorT> &
            operator=(VectorComplementView<VectorT> const &rhs) = delete;

        private:
            VectorT const &m_vector;
        };


    } // backend
} // GraphBLAS

#endif // GB_SEQUENTIAL_NEw_COMPLEMENT_VIEW_HPP
