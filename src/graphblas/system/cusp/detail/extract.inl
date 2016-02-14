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

//TODO: check that we have access to the sizes of the matrices
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
    inline void extract(//AMatrixT const    &a_o,
                        AMatrixT const    &a,
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

        ArrayType t_rows(a.num_entries);
        ArrayType t_cols(a.num_entries);
        cusp::array1d <ValueTypeC, MemorySpaceC> t_vals(a.num_entries);

        ArrayType i_d(i_size);
        ArrayType j_d(j_size);

        thrust::copy_n(i, i_size, i_d.begin());
        thrust::copy_n(j, j_size, j_d.begin());

        CMatrixT temp2(i_size, j_size, a.num_entries);
        CMatrixT temp3(i_size, j_size, a.num_entries);

        //build selection matrix for rows:
        CMatrixT temp(a.num_rows, a.num_rows, i_size);
        thrust::sequence(temp.row_indices.begin(), temp.row_indices.begin()+i_size);
        temp.column_indices = i_d;
        thrust::fill(temp.values.begin(), temp.values.begin()+i_size, 1);
        temp.num_entries = i_size;

        cusp::multiply(temp, a, temp2);

        //select columns:
        thrust::sequence(temp.column_indices.begin(), temp.column_indices.begin()+j_size);
        temp.column_indices = j_d;
        thrust::fill(temp.values.begin(), temp.values.begin()+j_size, 1);
        temp.num_entries = j_size;


        cusp::multiply(temp2, temp, temp3);


        temp3.resize(i_size, j_size, temp3.num_entries);

        //just swap for now, otherwise merge might mess up results.
        c.swap(temp3);
        //detail::merge(temp3, c, accum);
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

        extract(a, v_i.begin(), v_j.begin(), c, accum);
    }
}
}//end graphblas
