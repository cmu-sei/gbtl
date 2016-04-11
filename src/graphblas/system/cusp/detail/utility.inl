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

    //template <typename T>
    //struct ReplaceNonZero{
    //    T zero;
    //    ReplaceNonZero(T z) : zero(z){}

    //    template <typename V>
    //    __device__ __host__ bool operator()(const V val){
    //        //return fabs(val - zero) > 0.5;
    //        //assume fp == is not overflown
    //        return val != zero;
    //    }
    //};

    //col_index_of.
    //in-place implementation
    template<typename MatrixT>
    void col_index_of(MatrixT &mat)
    {
        typedef typename MatrixT::ScalarType T;
        //not sure why zero value is not passed in. get from matrix?
        //ReplaceNonZero <T> opr(mat.get_zero());
        //
        //stencil cannot overlap with result...
        //cusp::array1d<T, cusp::device_memory> temp(mat.values);
        //thrust::copy_if(
        //        mat.column_indices.begin(),
        //        mat.column_indices.begin()+mat.num_entries,
        //        mat.values.begin(),
        //        temp.begin(),
        //        //mat.values.begin(),
        //        opr);
        //thrust::copy(temp.begin(), temp.begin()+mat.num_entries, mat.values.begin());

        //update 2016.04.11:
        //zero values are not supposed to be stored in this backend.
        //thus, just copy column indices.
        thrust::copy(mat.column_indices.begin(), mat.column_indices.begin()+mat.num_entries, mat.values.begin());
    }

    template<typename MatrixT>
    void row_index_of(MatrixT &mat)
    {
        thrust::copy_n(mat.row_indices.begin(), mat.num_entries, mat.values.begin());
    }
}
}
