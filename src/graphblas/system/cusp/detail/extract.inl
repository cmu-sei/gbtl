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

#include <tuple>
#include <thrust/device_vector.h>
#include <cusp/permutation_matrix.h>
#include <graphblas/detail/config.hpp>
#include <graphblas/system/cusp/detail/merge.inl>

namespace graphblas
{
namespace backend
{

    //extract: A->C
    template<typename AMatrixT,
             typename CMatrixT,
             typename RAIteratorI,
             typename RAIteratorJ,
             typename AccumT >
    inline void extract(AMatrixT const    &a,
                        RAIteratorI        i,
                        RAIteratorJ        j,
                        CMatrixT          &c,
                        AccumT             accum)
    {
        typedef typename CMatrixT::value_type ValueTypeC;
        typedef typename CMatrixT::index_type IndexTypeC;
        typedef typename CMatrixT::memory_space MemorySpaceC;
        typedef cusp::array1d <IndexTypeC, MemorySpaceC> ArrayType;

        IndexType i_size = c.num_rows;
        IndexType j_size = c.num_cols;

        ArrayType i_d(i, i+i_size);
        ArrayType j_d(j, j+j_size);


        CMatrixT temp2(i_size, j_size, a.num_entries);

        //build selection matrix for rows:
        CMatrixT temp(a.num_rows, a.num_rows, i_size);
        thrust::sequence(temp.row_indices.begin(), temp.row_indices.begin()+i_size);
        temp.column_indices = i_d;
        thrust::fill(temp.values.begin(), temp.values.begin()+i_size, 1);
        temp.num_entries = i_size;


        cusp::multiply(temp, a, temp2);

        temp.resize(a.num_cols, a.num_cols, j_size);

        //select columns:
        thrust::sequence(temp.column_indices.begin(), temp.column_indices.begin()+j_size);
        temp.row_indices = j_d;
        thrust::fill(temp.values.begin(), temp.values.begin()+j_size, 1);
        temp.num_entries = j_size;


        cusp::multiply(temp2, temp, temp2);

        temp2.resize(i_size, j_size, temp2.num_entries);

        detail::merge(temp2, c, accum);
    }

    template<typename AMatrixT,
             typename CMatrixT,
             typename AccumT>
    inline void extract(AMatrixT const       &a,
                        IndexArrayType const &v_i,
                        IndexArrayType const &v_j,
                        CMatrixT             &c,
                        AccumT                accum)
    {
        IndexType m_C, n_C;
        c.get_shape(m_C, n_C);

        IndexType len_i = v_i.size();
        IndexType len_j = v_j.size();

        // assert that assignment is in range of dimensions of C.
        // if ((len_i != m_C) || (len_j != n_C))
        // {
        //     throw DimensionException();
        // }

        backend::extract(a, v_i.begin(), v_j.begin(), c, accum);
    }
}
}//end graphblas
