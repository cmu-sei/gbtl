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
                //TODO: check if this works
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

        /// @todo
        // assert that the dimension assigning from are correct
        // if ((len_i != m_A) || (len_j != n_A))
        // {
        //     throw DimensionException();
        // }

        assign(a, v_i.begin(), v_j.begin(), c, accum);
    }


#if 0
        template<typename VectorI,
                 typename VectorJ,
                 typename IteratorType,
                 typename ValueType = typename VectorI::value_type,
                 typename Accum=graphblas::math::Assign<typename VectorI::value_type> >
        inline void assign_vector_helper(
            VectorI &v_src,
            VectorI &v_ind,
            IteratorType v_val_iter,
            VectorJ &v_out,
            Accum          accum=Accum())
        {
            v_out = VectorJ(v_src);
            typedef typename VectorJ::iterator OutputIterator;
            typedef typename VectorI::iterator IndexIterator;
            thrust::permutation_iterator<OutputIterator, IndexIterator> iter(v_out.begin(), v_ind.begin());
            thrust::transform(
                iter,
                iter+thrust::distance(v_ind.begin(), v_ind.end()),
                v_val_iter,
                iter,
                accum);
        }

        template<typename VectorI,
                 typename VectorJ,
                 typename ValueType,
                 typename Accum>
        inline void assign(
            VectorI &v_src,
            VectorI &v_ind,
            VectorI &v_val,
            VectorJ &v_out,
            Accum    accum)
        {
            if (v_val.size() != v_ind.size())
            {
                return;
            }
            assign_vector_helper(v_src, v_ind, v_val.begin(), v_out, accum);
        }

        template<typename VectorI,
                 typename VectorJ,
                 typename ValueType,
                 typename Accum>
        inline void assign(
            VectorI                           &v_src,
            VectorI                           &v_ind,
            const typename VectorI::value_type value,
            VectorJ                           &v_out,
            Accum                              accum)
        {
            thrust::constant_iterator<ValueType> tempValues(value);
            assign_vector_helper(v_src, v_ind, tempValues, v_out, accum);
        }
#endif
}
}//end graphblas
