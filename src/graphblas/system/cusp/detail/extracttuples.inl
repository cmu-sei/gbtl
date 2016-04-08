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
#include <graphblas/system/cusp/detail/merge.inl>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <tuple>
#include <thrust/copy.h>

namespace graphblas
{
namespace backend
{
    template<typename AMatrixT,
             typename RAIIteratorIT,
             typename RAIIteratorJT,
             typename RAIIteratorVT>
    inline void extracttuples(AMatrixT const &a,
                              RAIIteratorIT   i,
                              RAIIteratorJT   j,
                              RAIIteratorVT   v)
    {
        typedef typename AMatrixT::index_type IndexType;
        typedef typename AMatrixT::value_type ValueType;

        thrust::copy_n(a.row_indices.begin(), a.num_entries, i);
        thrust::copy_n(a.column_indices.begin(), a.num_entries, j);
        thrust::copy_n(a.values.begin(), a.num_entries, v);
    }

    template<typename AMatrixT>
    inline void extracttuples(AMatrixT const                             &a,
                              IndexArrayType                             &i,
                              IndexArrayType                             &j,
                              std::vector<typename AMatrixT::ScalarType> &v)
    {
        backend::extracttuples(a, i.begin(), j.begin(), v.begin());
    }
}
} // graphblas
