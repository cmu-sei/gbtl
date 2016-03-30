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

        row_index_transformer(IndexType c) :cols(c) {}

        __host__ __device__
        inline IndexType operator()(const IndexType & sequence) {
            return (sequence / cols);
        }
    };

    struct col_index_transformer{
        IndexType rows, cols;

        col_index_transformer(IndexType r, IndexType c) : rows(r), cols(c) {}

        __host__ __device__
        inline IndexType operator()(const IndexType & sequence) {
            return sequence - ((sequence / cols) * rows);
        }
    };

    template <typename InputIt>
    struct find_entry{
        InputIt a_begin, a_end, b_begin, b_end;
        find_entry(InputIt a1, InputIt a2, InputIt b1, InputIt b2):
            a_begin(a1), a_end(a2), b_begin(b1), b_end(b2) {}
        template <typename T>
        __host__ __device__
        inline bool operator()(const T& first, const T& second)
        {
            return thrust::binary_search
        }

    };
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
    class NegateView : public graphblas::backend::Matrix<typename MatrixT::ScalarType>
    {
    public:
        typedef typename MatrixT::ScalarType ScalarType;
        typedef graphblas::backend::Matrix<typename MatrixT::ScalarType> ParentMatrixT;

        // CONSTRUCTORS

        NegateView(MatrixT const &matrix):
            ParentMatrix(matrix.num_rows, matrix.num_cols, 0)
        {
            /// @todo assert that matrix and semiring zero() are the same?
            //this is a problem, since matrices really shouldn't have zeroes


            //unfortunately will need to materialize the negation (not sparse in representation,
            //but still sparse in storage), the cost is search.
            auto rows = matrix.num_rows;
            auto cols = matrix.num_cols;
            this->num_entries = matrix.num_rows * matrix.num_cols - matrix.num_entries;

        }
    }



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
