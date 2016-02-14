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

#ifndef GB_CUSP_COLUMNVIEW_HPP
#define GB_CUSP_COLUMNVIEW_HPP

#include <graphblas/system/cusp/ElementRef.hpp>

namespace graphblas
{
    /**
     * @brief View of a Column of a (view of a) matrix.
     */
    template<typename MatrixT>    // Adhere to Vector concept
    class ColumnView
    {
    public:
        typedef typename MatrixT::ScalarType ScalarType;

        // CONSTRUCTION
        ColumnView(IndexType  col_index,
                   MatrixT   &parent_matrix)
            : m_col_index(col_index),
              m_parent_matrix(parent_matrix)
        {
        }

        //operator std::vector<ValueType>() const
        //{ return parent.get_row(m_index); }

        //operator dense_vector<ValueType>() const
        //{ return parent.get_row(m_index); }

        // Not sure vector will have a 2D shape.
        void get_shape(IndexType &num_rows, IndexType &num_cols) const
        {
            IndexType dummy;
            num_cols = 1;
            m_parent_matrix.get_shape(num_rows, dummy);
        }

        IndexType get_size() const
        {
            IndexType num_rows, num_cols;
            m_parent_matrix.get_shape(num_rows, num_cols);
            return num_rows;
        }

        /// @todo Do we need to conform to the vector::size() interface?
        IndexType size() const
        {
            return get_size();
        }

        // Need const and non-const versions of the index operator?
        ElementRef<ColumnView<MatrixT> const> operator[](IndexType idx) const
        {
            return ElementRef<ColumnView<MatrixT> const>(idx, *this);
        }

        ElementRef<ColumnView<MatrixT> > operator[](IndexType idx)
        {
            return ElementRef<ColumnView<MatrixT> >(idx, *this);
        }

        // returns a copy of the value
        ScalarType get_value_at(IndexType index) const
        {
            return m_parent_matrix.get_value_at(index, m_col_index);
        }

        /**
         * @todo This is a dummy needed for use in RowExtendedView, since
         * RowExtendedView requires two arguments in get_value_at.
         */
        ScalarType get_value_at(IndexType row, IndexType col) const
        {
            // assert(col == 0);
            return get_value_at(row);
        }

        void set_value_at(IndexType index, ScalarType const &rhs)
        {
            m_parent_matrix.set_value_at(index, m_col_index, rhs);
        }

        void set_value_at(IndexType row,
                          IndexType col,
                          ScalarType const &rhs)
        {
            // assert(col == 0);
            set_value_at(row, rhs);
        }

        friend std::ostream&
        operator<<(std::ostream              &os,
                   ColumnView<MatrixT> const &col)
        {
            IndexType M;
            IndexType N;
            col.m_parent_matrix.get_shape(M, N);

            os << "[";
            for (IndexType j = 0; j < M - 1; ++j)
            {
                os << col.m_parent_matrix.get_value_at(j, col.m_col_index)
                   << ", ";
            }
            os << col.m_parent_matrix.get_value_at(M - 1, col.m_col_index);
            os << "]'";

            return os;
        }

    private:
        IndexType  m_col_index;
        MatrixT   &m_parent_matrix;
    };
} // graphblas

#endif // GB_CUSP_COLUMNVIEW_HPP
