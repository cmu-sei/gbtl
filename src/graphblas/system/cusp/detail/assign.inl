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

namespace graphblas
{
namespace backend
{
    namespace detail
    {
        template <typename IndexType>
        struct RowTupleComparator : thrust::binary_function<thrust::tuple<IndexType, IndexType>, thrust::tuple<IndexType, IndexType>, bool >
        {
            __host__ __device__
            inline bool operator() (const thrust::tuple<IndexType, IndexType> &a,
                                    const thrust::tuple<IndexType, IndexType> &b)
            {
                return thrust::get<0>(a) < thrust::get<0>(b);
            }
        };
    }

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
        ::cusp::array1d<IndexType, MemorySpace>  selected_rows(a.num_entries);
        ::cusp::array1d<IndexType, MemorySpace>  selected_cols(a.num_entries);
        //make a copy of a.values:
        ::cusp::array1d<ValueType, MemorySpace> aValues(a.values);
        //temp storage:
        AMatrixT temp(c.num_rows, c.num_cols, a.num_entries);

        //copy i, j iterators in:
        ::cusp::array1d<IndexType, MemorySpace> i_d(a.num_rows);
        ::cusp::array1d<IndexType, MemorySpace> j_d(a.num_cols);
        thrust::copy_n(i, a.num_rows, i_d.begin());
        thrust::copy_n(j, a.num_cols, j_d.begin());

        //pick out intersection of (i,j) with a:
        thrust::gather(
            a.row_indices.begin(),
            a.row_indices.end(),
            i_d.begin(), //i is an iterator.
            selected_rows.begin());

        thrust::gather(
            a.column_indices.begin(),
            a.column_indices.end(),
            j_d.begin(),
            selected_cols.begin());

        //sort by key (don't care about stability):
        thrust::sort_by_key(
            thrust::make_zip_iterator(thrust::make_tuple(selected_rows.begin(), selected_cols.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(selected_rows.end(), selected_cols.end())),
            aValues.begin(),
            detail::RowTupleComparator<IndexType>());

        //populate temp array:
        ::cusp::copy(aValues, temp.values);
        ::cusp::copy(selected_rows, temp.row_indices);
        ::cusp::copy(selected_cols, temp.column_indices);

        //merge:
        detail::merge(temp, c, accum);
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
