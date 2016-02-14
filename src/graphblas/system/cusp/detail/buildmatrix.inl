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

// specialized buildmatrix function for cusp-gpu backend
// currently supports coo matrix
// TODO: add more matrix formats.

#include <graphblas/system/cusp/detail/merge.inl>

namespace graphblas
{
namespace backend
{
    namespace detail
    {
        struct get_vector_size
        {
            template <typename VectorType >
            __host__ __device__
            IndexType operator()(const VectorType &x)
            {
                return x.size();
            }
        };
    }

    using namespace graphblas::detail;
    //buildmatrix with an iteraator of vector-like objects that
    //represents a dense matrix.
    //N: number of rows
    //the objects in iterator i must have the .size() method
    template<typename MatrixT,
             typename RAIteratorI,
             typename Accum>
    inline void buildmatrix(MatrixT     &m,
                            RAIteratorI  i,
                            IndexType    n,
                            Accum accum)
    {
        /*
          IndexType count = thrust::reduce(thrust::make_transform_iterator(i, get_vector_size()),
          thrust::make_transform_iterator(i+n, get_vector_size()));
        */
        //TODO: restore previous line
        IndexType count = n* (*i).size();
        //if iterator i is empty or if its elements' sizes are 0
        if (n == 0 || (n>0 && (*i).size() ==0)) {
            return;
        }

        if (count == n * (*i).size())
        {
            //opt: error message/assert?
            assert(-1);
            return;
        }

        //array2d:
        //this in theory should happen on the host
        MatrixT temp(n, (*i).size());
        //copy: we can just do this linearly
        IndexType counter = 0;
        for (RAIteratorI it = i; it < i + n; ++it)
        {
            thrust::copy_n((*it).begin(), (*it).size(), &(temp.values[(counter*(*it).size())]));
            ++counter;
        }
        MatrixT temp2(temp);
        if (m.num_entries == 0)
        {
            m.swap(temp2);
            return;
        }
        else if (m.num_entries < temp2.num_entries) {
            m.resize(temp2.num_entries);
        }
        detail::merge(temp2, m, accum);
    }

    template<typename MatrixT,
             typename RAIteratorI,
             typename RAIteratorJ,
             typename RAIteratorV,
             typename Accum>
    inline void buildmatrix(MatrixT     &m,
                            RAIteratorI  i,
                            RAIteratorJ  j,
                            RAIteratorV  v,
                            IndexType  n,
                            Accum accum)
    {
        typedef typename MatrixT::index_type IndexType;
        //hardcoded coo matrix:
        MatrixT temp(m.num_rows, m.num_cols, n);

        //TODO: decide what to do with duplicates
        //merge by key, maybe?
        //assuming thrust copy deals with memspaces
        thrust::copy_n(i, n, temp.row_indices.begin());
        thrust::copy_n(j, n, temp.column_indices.begin());
        thrust::copy_n(v, n, temp.values.begin());
        //sorting:
        temp.sort_by_row_and_column();

        if (m.num_entries == 0)
        {
            m.swap(temp);
            return;
        }
        else if (m.num_entries < temp.num_entries) {
            m.resize(temp.num_entries);
        }

        detail::merge(temp, m, accum);
    }

    template<typename MatrixT,
             typename AccumT>
    inline void buildmatrix(MatrixT              &m,
                            IndexArrayType const &i,
                            IndexArrayType const &j,
                            std::vector<typename MatrixT::ScalarType> const &v,
                            AccumT accum)
    {
        /// @todo Add dimension checks
        buildmatrix(m, i.begin(), j.begin(), v.begin(), i.size(), accum);
    }


    // namespace detail {
    //     struct FakeIterator
    //     {
    //         operator[](const IndexType n) { return n; }
    //     };
    // }

    // template<typename MatrixT,
    //          typename RAIteratorI,
    //          typename Accum>
    // inline void buildmatrix(MatrixT     &m,
    //                         RAIteratorI  i,
    //                         IndexType    n,
    //                         Accum        accum)
    // {
    //     buildmatrix(m, i, FakeIterator(), FakeIterator(), n, accum);
    // }
}
} // graphblas
