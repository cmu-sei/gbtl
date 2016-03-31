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


#ifndef GB_CUSP_NEGATE_VIEW_HPP
#define GB_CUSP_NEGATE_VIEW_HPP

#include <graphblas/system/cusp/Matrix.hpp>
#include <thrust/iterator/iterator_adaptor.h>

namespace graphblas
{
namespace backend
{
    // Generalized Negate/complement
    template <typename SemiringT>
    class SemiringNegate
    {
    public:
        typedef typename SemiringT::ScalarType ScalarType;
        ScalarType operator()(ScalarType const &value)
        {
            if (value == SemiringT().zero())
                return SemiringT().one();
            else
                return SemiringT().zero();
        }
    };

    namespace detail{
    //implement a matrix index iterator (1111,2222,3333...), (123412341234...)

    struct row_index_transformer{
        IndexType cols;

        __host__ __device__
        row_index_transformer(IndexType c) :cols(c) {}

        __host__ __device__
        IndexType operator()(const IndexType & sequence) {
            return (sequence / cols);
        }
    };

    struct col_index_transformer{
        IndexType rows, cols;

        __host__ __device__
        col_index_transformer(IndexType r, IndexType c) : rows(r), cols(c) {}

        __host__ __device__
        IndexType operator()(const IndexType & sequence) {
            return sequence - ((sequence / cols) * rows);
        }
    };

    cusp::array1d_view<thrust::transform_iterator<row_index_transformer, thrust::counting_iterator<IndexType> > >
    make_row_index_iterator(IndexType rows, IndexType cols)
    {
        auto sequence = thrust::make_counting_iterator(rows*cols);
        auto begin_transformed = thrust::make_transform_iterator(sequence, row_index_transformer(cols));
        auto end_transformed = thrust::make_transform_iterator(sequence + (rows*cols), row_index_transformer(cols));
        return cusp::make_array1d_view(begin_transformed, end_transformed);
    }

    cusp::array1d_view<thrust::transform_iterator<col_index_transformer, thrust::counting_iterator<IndexType> > >
    make_col_index_iterator(IndexType rows, IndexType cols)
    {
        auto sequence = thrust::make_counting_iterator(rows*cols);
        auto begin_transformed = thrust::make_transform_iterator(sequence, col_index_transformer(rows, cols));
        auto end_transformed = thrust::make_transform_iterator(sequence + (rows*cols), col_index_transformer(rows, cols));
        return cusp::make_array1d_view(begin_transformed, end_transformed);
    }

    template <typename InputIt, typename OneT, typename ZeroT>
    struct find_entry_replace{
        InputIt a1, a2;
        OneT yks, r, c;
        ZeroT nol;

        typedef OneT type;

        __host__ __device__
        find_entry_replace(InputIt zip_begin, InputIt zip_end, OneT rows, OneT cols, OneT one, ZeroT zero)
            : a1(zip_begin), a2(zip_end), r(rows), c(cols), yks(one), nol(zero) {}

        __host__ __device__
        OneT operator()(const OneT& value)
        {
            //value to index:
            OneT row = value/c;
            OneT col = value - ((value/c)*r);
            return thrust::binary_search(a1, a2, thrust::make_tuple(row, col)) ? static_cast<OneT>(nol) : yks;
        }

    };

    template <typename IndexIterator, typename ValueType, typename SemiringT>
    cusp::array1d_view<thrust::transform_iterator<
        find_entry_replace<thrust::zip_iterator<thrust::tuple<IndexIterator, IndexIterator> >, ValueType, ValueType >,
        thrust::counting_iterator<ValueType> > >
    make_val_iterator(IndexIterator row_indices, IndexIterator col_indices, IndexType rows, IndexType cols, IndexType num_entries, ValueType, SemiringT)
    {
        auto sequence = thrust::make_counting_iterator(static_cast<ValueType>(rows*cols));
        auto zipped_indices_begin = thrust::make_zip_iterator(thrust::make_tuple(row_indices, col_indices));
        auto zipped_indices_end = thrust::make_zip_iterator(thrust::make_tuple(row_indices+num_entries, col_indices+num_entries));

        ValueType one = static_cast<ValueType>(SemiringT().one());
        ValueType zero = static_cast<ValueType>(SemiringT().zero());

        find_entry_replace<thrust::zip_iterator<thrust::tuple<IndexIterator, IndexIterator> >, ValueType, ValueType >
             op(zipped_indices_begin,
                zipped_indices_end,
                rows,
                cols,
                one,
                zero);

        auto begin_transformed = thrust::make_transform_iterator(sequence, op);
        auto end_transformed = thrust::make_transform_iterator(sequence + (rows*cols), op);

        return cusp::make_array1d_view(begin_transformed, end_transformed);

    }

    }//end detail

    //************************************************************************
    /**
     * @brief View a matrix as if it were negated (stored values and
     *        structural zeroes are swapped).
     *
     * @tparam MatrixT     Implements the backend matrix.
     * @tparam SemiringT   Used to define the behaviour of the negate
     */
    template<typename MatrixT, typename SemiringT>
    class NegateView : public cusp::coo_matrix_view<
            cusp::array1d_view<thrust::transform_iterator<detail::row_index_transformer, thrust::counting_iterator<IndexType> > >,
            cusp::array1d_view<thrust::transform_iterator<detail::col_index_transformer, thrust::counting_iterator<IndexType> > >,
            cusp::array1d_view<thrust::transform_iterator<
                detail::find_entry_replace<thrust::zip_iterator<thrust::tuple<
                    thrust::detail::normal_iterator<thrust::device_ptr<IndexType> >,
                    thrust::detail::normal_iterator<thrust::device_ptr<IndexType> >
                    > >, typename MatrixT::ScalarType, typename MatrixT::ScalarType >,
                thrust::counting_iterator<typename MatrixT::ScalarType> > > >
    {
    public:
        typedef typename MatrixT::ScalarType ScalarType;
        typedef cusp::coo_matrix_view<
            cusp::array1d_view<thrust::transform_iterator<detail::row_index_transformer, thrust::counting_iterator<IndexType> > >,
            cusp::array1d_view<thrust::transform_iterator<detail::col_index_transformer, thrust::counting_iterator<IndexType> > >,
            cusp::array1d_view<thrust::transform_iterator<
                detail::find_entry_replace<thrust::zip_iterator<thrust::tuple<
                    thrust::detail::normal_iterator<thrust::device_ptr<IndexType> >,
                    thrust::detail::normal_iterator<thrust::device_ptr<IndexType> >
                    > >, ScalarType, ScalarType >,
                thrust::counting_iterator<ScalarType> > > > ParentMatrixT;

        NegateView(MatrixT const &matrix)
        {

        }
    };



    template<typename MatrixT,
             typename SemiringT =
                 graphblas::ArithmeticSemiring<typename MatrixT::ScalarType> >
    inline NegateView<MatrixT, SemiringT> negate(
        MatrixT const   &a,
        SemiringT const &s = SemiringT())
    {
        return NegateView<MatrixT, SemiringT>(a);
    }

} // backend
} // graphblas

#endif
