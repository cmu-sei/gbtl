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

        struct pick_min{
            template <typename T >
            __host__ __device__
            T operator()(const T &v1, const T &v2)
            {
                return v1<v2?v1:v2;
            }
        };
    }

    using namespace graphblas::detail;
    //buildmatrix with an iteraator of vector-like objects that
    //represents a dense matrix.
    //N: number of rows
    //the objects in iterator i must have the .size() method
    //
    //**this method should be deprecated since it supports a non-sparse construct.**
    template<typename MatrixT,
             typename RAIteratorI,
             typename Accum>
    inline void buildmatrix(MatrixT     &m,
                            RAIteratorI  i,
                            IndexType    n,
                            Accum accum)
    {
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



        //assuming thrust copy deals with memspaces
        thrust::copy_n(i, n, temp.row_indices.begin());
        thrust::copy_n(j, n, temp.column_indices.begin());
        thrust::copy_n(v, n, temp.values.begin());
        //sorting:
        temp.sort_by_row_and_column();

        m.resize(m.num_rows, m.num_cols, temp.num_entries);

        //filter out duplicates locally:
        auto end = thrust::reduce_by_key(
                thrust::make_zip_iterator(thrust::make_tuple(
                        temp.row_indices.begin(),
                        temp.column_indices.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(
                        temp.row_indices.begin()+n,
                        temp.column_indices.begin()+n)),
                temp.values.begin(),
                thrust::make_zip_iterator(thrust::make_tuple(
                        m.row_indices.begin(),
                        m.column_indices.begin())),
                m.values.begin(),
                thrust::equal_to< thrust::tuple<IndexType,IndexType> >(),
                detail::pick_min());

        auto filtered_size = thrust::distance(m.values.begin(), thrust::get<1>(end));

        m.resize(m.num_rows, m.num_cols, filtered_size);

        //do not merge. why would one want to build matrix on a matrix with valid data?
        //just do ewiseadd if that is needed.
        //thus, accum is useless.
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

}
} // graphblas
