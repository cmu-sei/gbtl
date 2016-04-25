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
#include <cusp/multiply.h>
#include <cusp/coo_matrix.h>
#include <cusp/print.h>

namespace graphblas
{
namespace backend
{
    template<typename AMatrixT,
             typename BMatrixT,
             typename CMatrixT,
             typename SemiringT,
             typename AccumT>
    inline void mxm(AMatrixT  const &a,
                    BMatrixT  const &b,
                    CMatrixT        &c,
                    SemiringT       s,
                    AccumT          accum)
    {
        typedef typename AMatrixT::ScalarType ScalarType;

        //::cusp::coo_matrix<IndexType, ScalarType, ::cusp::device_memory>
        CMatrixT tmp_matrix(a.num_rows, b.num_cols, 0);

        // multiply into a temporary
        ::cusp::multiply(a, b, tmp_matrix,
                         thrust::identity<ScalarType>(),
                         make_multiplicative_monoid_from_semiring(s),
                         make_additive_monoid_from_semiring(s));

        //c=tmp_matrix;
        c.swap(tmp_matrix);

        //do not merge for now, might cause issues with overwrites.
        // merge the results
        //detail::merge(tmp_matrix, c, accum);
    }

    namespace detail{
        struct SelectFirst{
            template <typename LhsT, typename RhsT>
            __host__ __device__
            LhsT operator()(const LhsT & lhs, const RhsT & rhs){
                return lhs;
            }
        };
    }

    template<typename AMatrixT,
             typename BMatrixT,
             typename CMatrixT,
             typename MMatrixT,
             typename SemiringT =
                    graphblas::ArithmeticSemiring<typename AMatrixT::ScalarType>,
             typename AccumT =
                    graphblas::math::Assign<typename AMatrixT::ScalarType> >
    inline void mxmMasked(AMatrixT const &a,
                          BMatrixT const &b,
                          CMatrixT       &c,
                          MMatrixT const &m,
                          SemiringT       s = SemiringT(),
                          AccumT          accum = AccumT())
    {
        //the mask **must** be sparse.
        backend::mxm(a,b,c,s,accum);
        //backend::ewisemult(c,m,c,graphblas::math::Times<typename AMatrixT::ScalarType>(),accum);
        backend::ewisemult(c,m,c,detail::SelectFirst(),accum);
    }

    template<typename AMatrixT,
             typename BMatrixT,
             typename CMatrixT,
             typename MMatrixT,
             typename SemiringT =
                 graphblas::ArithmeticSemiring<typename AMatrixT::ScalarType>,
             typename AccumT =
                 graphblas::math::Assign<typename AMatrixT::ScalarType> >
    inline void mxmMaskedV2(AMatrixT const &a,
                            BMatrixT const &b,
                            CMatrixT       &c,
                            MMatrixT       &m,
                            SemiringT       s = SemiringT(),
                            AccumT          accum = AccumT())
    {
        backend::mxmMasked(a,b,c,m,s,accum);
    }

    //if sparse vectors are needed, use mxm

    template<typename AVectorT,
             typename BMatrixT,
             typename CVectorT,
             typename SemiringT =
                 graphblas::ArithmeticSemiring<typename AVectorT::ScalarType>,
             typename AccumT =
                 graphblas::math::Assign<typename AVectorT::ScalarType> >
    inline void vxm(AVectorT const &a,
                    BMatrixT const &b,
                    CVectorT       &c,
                    SemiringT       s     = SemiringT(),
                    AccumT          accum = AccumT())
    {
        return backend::mxm(a,b,c,s,accum);
        //typedef typename AVectorT::ScalarType ScalarType;
        //cusp::constant_array<ScalarType> zeros(a.num_entries, 0);
        ////transpose B:
        //BMatrixT temp(b);
        //thrust::swap(temp.row_indices, temp.column_indices);
        //temp.sort_by_row_and_column();

        //::cusp::generalized_spmv(temp, a, zeros,
        //                 c,
        //                 make_multiplicative_monoid_from_semiring(s),
        //                 make_additive_monoid_from_semiring(s));
    }

    template<typename AMatrixT,
             typename BVectorT,
             typename CVectorT,
             typename SemiringT =
                 graphblas::ArithmeticSemiring<typename AMatrixT::ScalarType>,
             typename AccumT =
                 graphblas::math::Assign<typename AMatrixT::ScalarType> >
    inline void mxv(AMatrixT const &a,
                    BVectorT const &b,
                    CVectorT       &c,
                    SemiringT       s     = SemiringT(),
                    AccumT          accum = AccumT())
    {
        return backend::mxm(a,b,c,s,accum);
        //typedef typename AMatrixT::ScalarType ScalarType;
        //cusp::constant_array<ScalarType> zeros(b.size(), 0);

        //AMatrixT temp(a);

        //::cusp::generalized_spmv(temp, b, zeros,
        //                 c,
        //                 make_multiplicative_monoid_from_semiring(s),
        //                 make_additive_monoid_from_semiring(s));
    }
}
} // graphblas
