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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/swap.h>
#include <thrust/tuple.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>

#include <cusp/coo_matrix.h>
#include <cusp/monitor.h>
#include <cusp/multiply.h>
#include <cusp/gallery/poisson.h>
#include <cusp/krylov/cg.h>
#include <cusp/detail/format.h>
#include <cusp/sort.h>
#include <cusp/verify.h>

namespace graphblas
{
namespace backend
{
    namespace detail
    {
        //implement a matrix index iterator (1111,2222,3333...), (123412341234...)

        struct row_index_transformer: public thrust::unary_function<IndexType,IndexType>{
            IndexType cols;

            __host__ __device__
            row_index_transformer(IndexType c) :cols(c) {}

            template <typename IntT>
            __host__ __device__
            IntT operator()(const IntT & sequence) {
                return (sequence / cols);
            }
        };

        struct col_index_transformer : public thrust::unary_function<IndexType,IndexType> {
            IndexType rows, cols;

            __host__ __device__
            col_index_transformer(IndexType r, IndexType c) : rows(r), cols(c) {}

            template <typename IntT>
            __host__ __device__
            IntT operator()(const IntT & sequence) {
                return sequence % cols;
            }
        };
        template <typename MatrixTypeSrc,
                  typename MatrixTypeDst,
                  typename MergeFunction >
        typename thrust::detail::enable_if<thrust::detail::is_same<typename MatrixTypeSrc::memory_space, typename MatrixTypeDst::memory_space >::value >::type
        merge(MatrixTypeSrc &A,
              MatrixTypeDst &B,
              MergeFunction mergeFunc,
              ::cusp::coo_format,
              ::cusp::coo_format);
        /**
         * merge coo matrices from source into destination
         * operating on different mem space
         */
        template <typename MatrixTypeSrc,
                  typename MatrixTypeDst,
                  typename MergeFunction >
        typename thrust::detail::disable_if<thrust::detail::is_same<typename MatrixTypeSrc::memory_space, typename MatrixTypeDst::memory_space >::value >::type
        merge(MatrixTypeSrc &A,
              MatrixTypeDst &B,
              MergeFunction mergeFunc,
              ::cusp::coo_format format1,
              ::cusp::coo_format format2)
        {
            typedef typename MatrixTypeDst::index_type   IndexType;
            typedef typename MatrixTypeDst::value_type   ValueType;
            typedef typename MatrixTypeDst::memory_space MemorySpaceDst;
            typedef typename MatrixTypeSrc::memory_space MemorySpaceSrc;

            //do copy:
            ::cusp::coo_matrix<IndexType, ValueType, MemorySpaceDst> temp_matrix(A);
            //call same memory space merge:
            merge(temp_matrix, B, mergeFunc, format1, format2);
        }

        /**
         * merge coo matrices from source into destination
         * operating on same mem space
         */
        template <typename MatrixTypeSrc,
                  typename MatrixTypeDst,
                  typename MergeFunction >
        typename thrust::detail::enable_if<thrust::detail::is_same<typename MatrixTypeSrc::memory_space, typename MatrixTypeDst::memory_space >::value >::type
        merge(MatrixTypeSrc &A,
              MatrixTypeDst &B,
              MergeFunction mergeFunc,
              ::cusp::coo_format,
              ::cusp::coo_format)
        {
            //using merging code sample given in cusp::elementwise:

            using namespace thrust::placeholders;
            using thrust::system::detail::generic::select_system;

            typedef typename MatrixTypeDst::index_type   IndexType;
            typedef typename MatrixTypeDst::value_type   ValueType;
            typedef typename MatrixTypeDst::memory_space MemorySpace;

            size_t A_nnz = A.num_entries;
            size_t B_nnz = B.num_entries;
            size_t num_entries = A_nnz + B_nnz;

            //temp storage for result:
            MatrixTypeDst temp(A.num_rows, A.num_cols, num_entries);

            //stop if A is empty
            if (A_nnz == 0)
            {
                return;
            }
            //swap a to b if b is empty
            if (B_nnz == 0) {
                A.swap(B);
                return;
            }

            //merging:
            //merge row, col and values of A and B into temp:
            thrust::merge_by_key(
                thrust::make_zip_iterator(thrust::make_tuple(B.row_indices.begin(), B.column_indices.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(B.row_indices.end(), B.column_indices.end())),
                thrust::make_zip_iterator(thrust::make_tuple(A.row_indices.begin(), A.column_indices.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(A.row_indices.end(), A.column_indices.end())),
                B.values.begin(),
                A.values.begin(),
                thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices.begin(), temp.column_indices.begin())),
                temp.values.begin());

            //calc unique number of keys:
            IndexType B_unique_nnz = thrust::inner_product(
                thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices.begin(), temp.column_indices.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices.end()-1, temp.column_indices.end()-1)),
                thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices.begin()+1, temp.column_indices.begin()+1)),
                IndexType(1),
                thrust::plus<IndexType>(),
                thrust::not_equal_to< thrust::tuple<IndexType,IndexType> >());

            //resize B:
            B.resize(A.num_rows, A.num_cols, B_unique_nnz);
        
            //reduce by key:
            thrust::reduce_by_key(
                thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices.begin(), temp.column_indices.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices.end(), temp.column_indices.end())),
                temp.values.begin(),
                thrust::make_zip_iterator(thrust::make_tuple(B.row_indices.begin(), B.column_indices.begin())),
                B.values.begin(),
                thrust::equal_to< thrust::tuple<IndexType,IndexType> >(),
                mergeFunc);
        }
        /**
         * merge source and destination into destination
         * using the given merge function
         *
         * should handle memory spaces in overloaded funcs.
         */
        template <typename MatrixTypeSrc,
                  typename MatrixTypeDst,
                  typename MergeFunction >
        void merge(MatrixTypeSrc &A,
                   MatrixTypeDst &B,
                   MergeFunction mergeFunc = MergeFunction() )
        {
            //should delegate:
            typedef typename MatrixTypeSrc::format Format1;
            typedef typename MatrixTypeDst::format Format2;
            Format1 format1;
            Format2 format2;

            merge(A, B, mergeFunc, format1, format2);
        }
    }//detail
}
} // end: graphblas
