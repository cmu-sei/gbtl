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

#include <graphblas/detail/config.hpp>
#include <graphblas/system/cusp/detail/merge.inl>
#include <cusp/print.h>

namespace graphblas
{
namespace backend
{
    template<typename AMatrixT,
             typename CMatrixT,
             typename RAIteratorI,
             typename RAIteratorJ,
             typename AccumT>
    inline void assign(AMatrixT const    &a,
                       RAIteratorI        i,
                       RAIteratorJ        j,
                       CMatrixT          &c,
                       AccumT             accum)
    {
        if (a.num_entries == 0)
        {
            return;
        }

        typedef typename AMatrixT::value_type ValueType;
        typedef typename AMatrixT::index_type IndexType;
        typedef typename AMatrixT::memory_space MemorySpace;
        typedef cusp::array1d <IndexType, MemorySpace> ArrayType;

//matrix mult method (according to the math doc)
#if 0
        //copy i, j iterators
        auto i_size = a.num_rows;
        auto j_size = a.num_cols;

        ArrayType  i_d(i, i+a.num_rows);
        ArrayType  j_d(j, j+a.num_cols);

        CMatrixT temp1(c.num_rows, a.num_rows, i_size);
        CMatrixT temp2(c.num_cols, a.num_cols, j_size);

        thrust::sequence(temp1.column_indices.begin(), temp1.column_indices.begin()+i_size);
        temp1.row_indices = i_d;
        thrust::fill(temp1.values.begin(), temp1.values.begin()+i_size, 1);

        cusp::multiply(temp1, a, temp1);

        thrust::sequence(temp2.row_indices.begin(), temp2.row_indices.begin()+i_size);
        temp2.column_indices = j_d;
        thrust::fill(temp2.values.begin(), temp2.values.begin()+i_size, 1);

        cusp::multiply(temp1, temp2, temp2);


        //cusp::print(temp2);
        //merge:
        temp2.resize(c.num_rows, c.num_cols, temp2.num_entries);
        detail::merge(temp2, c, accum);
#endif

//raw thrust method:
#if 1 
        //temp storage:
        AMatrixT temp(c.num_rows, c.num_cols, a.num_entries);

        //copy i, j iterators in:
        ArrayType  i_d(i, i+a.num_rows);
        ArrayType  j_d(j, j+a.num_cols);

        //populate temp array:
        ::cusp::copy(a.values, temp.values);

        //pick out intersection of (i,j) with a:
        thrust::gather(
            a.row_indices.begin(),
            a.row_indices.end(),
            i_d.begin(),
            temp.row_indices.begin());

        thrust::gather(
            a.column_indices.begin(),
            a.column_indices.end(),
            j_d.begin(),
            temp.column_indices.begin());

        temp.sort_by_row_and_column();

        //merge:
        detail::merge(temp, c, accum);


        //manually remove assigned zeros:
        if (a.num_entries < a.num_rows*a.num_cols)
        {
            //test ``accum'' feature:
            if (c.get_zero() != accum(32, c.get_zero()))
            {
                return;
            }
            auto zero_size = a.num_rows*a.num_cols - a.num_entries;
            //figure out which values are zeros in a:
            auto sequence = thrust::make_counting_iterator(0);

            auto row_b = thrust::make_transform_iterator(sequence,
                    detail::row_index_transformer(a.num_cols));
            auto row_e = thrust::make_transform_iterator(sequence + (a.num_rows*a.num_cols),
                    detail::row_index_transformer(a.num_cols));

            auto col_b = thrust::make_transform_iterator(sequence,
                    detail::col_index_transformer(a.num_rows, a.num_cols));
            auto col_e = thrust::make_transform_iterator(sequence + (a.num_rows*a.num_cols),
                    detail::col_index_transformer(a.num_rows, a.num_cols));

            ArrayType zero_r(zero_size);
            ArrayType zero_c(zero_size);

            CMatrixT c_copy(c);

            thrust::set_difference(
                    thrust::make_zip_iterator(thrust::make_tuple(
                            row_b, col_b)),
                    thrust::make_zip_iterator(thrust::make_tuple(
                            row_e, col_e)),
                    thrust::make_zip_iterator(thrust::make_tuple(
                            a.row_indices.begin(), a.column_indices.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(
                            a.row_indices.end(), a.column_indices.end())),
                    thrust::make_zip_iterator(thrust::make_tuple(
                            zero_r.begin(), zero_c.begin())));

            //write to temp?
            temp.resize(c.num_rows, c.num_cols, zero_size);

            //pick out intersection of (i,j) with a:
            thrust::gather(
                zero_r.begin(),
                zero_r.end(),
                i_d.begin(),
                temp.row_indices.begin());

            thrust::gather(
                zero_c.begin(),
                zero_c.end(),
                j_d.begin(),
                temp.column_indices.begin());

            //remove entries from c:
            auto end_diff = thrust::set_difference_by_key(
                    thrust::make_zip_iterator(thrust::make_tuple(
                            c_copy.row_indices.begin(), c_copy.column_indices.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(
                            c_copy.row_indices.end(), c_copy.column_indices.end())),
                    thrust::make_zip_iterator(thrust::make_tuple(
                            temp.row_indices.begin(), temp.column_indices.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(
                            temp.row_indices.end(), temp.column_indices.end())),
                    c_copy.values.begin(),
                    thrust::make_constant_iterator(0),
                    thrust::make_zip_iterator(thrust::make_tuple(
                            c.row_indices.begin(), c.column_indices.begin())),
                    c.values.begin());

            c.num_entries = thrust::distance(c.values.begin(), end_diff.second);
        }

#endif 
    }

    template<typename AMatrixT,
             typename CMatrixT,
             typename AccumT>
    inline void assign(AMatrixT const        &a,
                       IndexArrayType const  &v_i,
                       IndexArrayType const  &v_j,
                       CMatrixT              &c,
                       AccumT                 accum)
    {
        IndexType m_A, n_A;
        a.get_shape(m_A, n_A);

        IndexType len_i = v_i.size();
        IndexType len_j = v_j.size();

        backend::assign(a, v_i.begin(), v_j.begin(), c, accum);
    }
}
}//end graphblas
