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

#include <thrust/transform.h>
#include <thrust/iterator/permutation_iterator.h>
#include "merge.inl"

#pragma once

namespace graphblas
{
namespace backend
{

    template<typename MatrixA, typename MatrixB>
    void index_of(const MatrixA &A, MatrixB &B, const IndexType base_index)
    {
        B.row_indices = A.row_indices;
        B.column_indices = A.column_indices;
        B.values = A.row_indices;
        B.resize(A.num_rows, A.num_cols, A.num_entries);

        return;
    }

    template <typename T>
    struct ReplaceNonZero{
        T zero;
        ReplaceNonZero(T z) : zero(z){}

        template <typename V>
        __device__ __host__ bool operator()(const V val){
            return fabs(val - zero) > 0.5;
        }
    };

    //col_index_of.
    //in-place implementation
    template<typename MatrixT>
    void col_index_of(MatrixT &mat)
    {
        //not sure why zero value is not passed in. manually hardcode.
        ReplaceNonZero <typename MatrixT::ScalarType > opr(std::numeric_limits<double>::max());
        std::cout<<"stored zeroval: "<<std::numeric_limits<double>::max()
            <<std::endl
            <<"numentries="<<mat.num_entries
            <<std::endl;
        //stencil cannot overlap with result...
        cusp::array1d<double, cusp::device_memory> temp(mat.values);
        thrust::copy_if(
                mat.column_indices.begin(),
                mat.column_indices.begin()+mat.num_entries,
                mat.values.begin(),
                temp.begin(),
                //mat.values.begin(),
                opr);
        thrust::copy(temp.begin(), temp.begin()+mat.num_entries, mat.values.begin());
    }

    template<typename MatrixT>
    void row_index_of(MatrixT &mat)
    {
        thrust::copy_n(mat.row_indices.begin(), mat.num_entries, mat.values.begin());
    }

    namespace detail
    {
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
    } //end detail


}
}
