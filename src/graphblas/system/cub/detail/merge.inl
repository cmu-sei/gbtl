/**
 * This is a reimplementation of the merge.inl file on the cusp side
 *
 */

#pragma once

#include "../header.hpp"
#include "../config.hpp"
#include "../utility.hpp"

#include <graphblas/detail/config.hpp>
#include <graphblas/exception.hpp>

#include <thrust/device_vector.h>
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

#include <cub/cub.cuh>

#include <cassert>

namespace graphblas
{
namespace backend
{
namespace detail
{

/**
 * merge source and destination into destination
 * using the given merge function
 *
 * this is the ewiseadd function in essence
 * CMatrix must have dimensions >= AMatrix
 */
template <typename AMatrixT,
          typename CMatrixT,
          typename AccumT >
void merge(AMatrixT const &A,
           CMatrixT       &C,
           AccumT accum = AccumT() )
{
    if (A.num_rows > C.num_rows ||
        A.num_cols > C.num_cols)
    {
        throw graphblas::DimensionException("backend::detail::merge: amatrix is bigger than cmatrix");
    }
    //build new matrix:
    CMatrixT temp(C.num_rows, C.num_cols, C.num_entries + A.num_entries);
    auto merge_result = thrust::merge_by_key(
            thrust::make_zip_iterator(thrust::make_tuple(C.row_indices, C.column_indices)),
            thrust::make_zip_iterator(thrust::make_tuple(C.row_indices+C.num_entries, C.column_indices+C.num_entries)),
            thrust::make_zip_iterator(thrust::make_tuple(A.row_indices, A.column_indices)),
            thrust::make_zip_iterator(thrust::make_tuple(A.row_indices+A.num_entries, A.column_indices+A.num_entries)),
            C.values,
            A.values,
            thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices, temp.column_indices)),
            temp.values);

    auto temp_size = thrust::distance(temp.values, merge_result.second);

    //calc unique number of keys:
    IndexType C_unique_nnz = thrust::inner_product(
            thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices, temp.column_indices)),
            thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices+temp_size-1, temp.column_indices+temp_size-1)),
            thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices+1, temp.column_indices+1)),
            IndexType(1),
            thrust::plus<IndexType>(),
            thrust::not_equal_to< thrust::tuple<IndexType,IndexType> >());

    //resize C:
    C.resize(A.num_rows, A.num_cols, C_unique_nnz);

    //reduce by key:
    thrust::reduce_by_key(
            thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices, temp.column_indices)),
            thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices+temp_size, temp.column_indices+temp_size)),
            temp.values,
            thrust::make_zip_iterator(thrust::make_tuple(C.row_indices, C.column_indices)),
            C.values,
            thrust::equal_to< thrust::tuple<IndexType,IndexType> >(),
            accum);
}

} //end detail
} //end backend
} //end graphblas
