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


#ifndef GB_SEQUENTIAL_ROW_VIEW_HPP
#define GB_SEQUENTIAL_ROW_VIEW_HPP

#include <graphblas/system/sequential/ElementRef.hpp>

namespace graphblas
{
    /**
     * @brief View of a row of a matrix as a Vector.
     *
     */
    // Adhere to Vector Concept
    template<typename MatrixT>
    class RowView
    {
    public:
        typedef typename MatrixT::ScalarType ScalarType;

        // CONSTRUCTION
        RowView(IndexType  row_index,
                MatrixT   &parent_matrix)
            : m_row_index(row_index),
              m_parent_matrix(parent_matrix)
        {
        }

        // Not sure vector will have a 2D shape.
        void get_shape(IndexType &num_rows, IndexType &num_cols) const
        {
            IndexType dummy;
            num_rows = 1;
            m_parent_matrix.get_shape(dummy, num_cols);
        }

        IndexType get_size() const
        {
            IndexType num_rows, num_cols;
            m_parent_matrix.get_shape(num_rows, num_cols);
            return num_cols;
        }

        /// @todo Do we need to conform to the vector::size() interface?
        IndexType size() const
        {
            return get_size();
        }

        // Need const and non-const versions of the index operator?
        ElementRef<RowView<MatrixT> const> operator[](IndexType idx) const
        {
            return ElementRef<RowView<MatrixT> const>(idx, *this);
        }

        ElementRef<RowView<MatrixT> > operator[](IndexType idx)
        {
            return ElementRef<RowView<MatrixT> >(idx, *this);
        }

        // returns a copy of the value
        ScalarType get_value_at(IndexType index) const
        {
            return m_parent_matrix.get_value_at(m_row_index, index);
        }

        /**
        * @todo This is a dummy needed for use in RowExtendedView, since
        * RowExtendedView requires two arguments in get_value_at.
        */
        ScalarType get_value_at(IndexType row, IndexType col) const
        {
            // assert(row == 0);
            return get_value_at(col);
        }

        void set_value_at(IndexType index, ScalarType const &rhs)
        {
            m_parent_matrix.set_value_at(m_row_index, index, rhs);
        }

        void set_value_at(IndexType row,
                          IndexType col,
                          ScalarType const &rhs)
        {
            // assert(row == 0);
            set_value_at(col, rhs);
        }

        friend std::ostream& operator<<(std::ostream           &os,
                                        RowView<MatrixT> const &row)
        {
            IndexType M;
            IndexType N;
            row.m_parent_matrix.get_shape(M, N);

            os << "[";
            for (IndexType j = 0; j < N - 1; ++j)
            {
                os << row.m_parent_matrix.get_value_at(row.m_row_index, j)
                   << ", ";
            }
            os << row.m_parent_matrix.get_value_at(row.m_row_index, N - 1);
            os << "]";

            return os;
        }

    private:
        IndexType  m_row_index;
        MatrixT   &m_parent_matrix;
    };
} // graphblas

#endif // GB_SEQUENTIAL_ROW_VIEW_HPP
