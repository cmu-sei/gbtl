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

/**
 * Implementations of all GraphBLAS functions optimized for the sequential
 * (CPU) backend.
 */

#ifndef GB_SEQUENTIAL_OPERATIONS_HPP
#define GB_SEQUENTIAL_OPERATIONS_HPP

#pragma once

#include <functional>
#include <utility>
#include <vector>
#include <iterator>

#include <graphblas/accum.hpp>
#include <graphblas/algebra.hpp>
#include <graphblas/system/sequential/TransposeView.hpp>
#include <graphblas/system/sequential/ComplementView.hpp>
#include <graphblas/system/sequential/NegateView.hpp>  // deprecated

// Add individual operation files here
#include <graphblas/system/sequential/sparse_apply.hpp>
#include <graphblas/system/sequential/sparse_assign.hpp>
#include <graphblas/system/sequential/sparse_mxm.hpp>
#include <graphblas/system/sequential/sparse_mxv.hpp>
#include <graphblas/system/sequential/sparse_vxm.hpp>
#include <graphblas/system/sequential/sparse_ewisemult.hpp>
#include <graphblas/system/sequential/sparse_ewiseadd.hpp>
#include <graphblas/system/sequential/sparse_extract.hpp>



namespace GraphBLAS
{
    namespace backend
    {
        /**
         *
         */
        // template<typename MatrixT>
        // inline ComplementView<MatrixT> complement(MatrixT const &A)
        // {
        //     return ComplementView<MatrixT>(A);
        // }

        template<typename MatrixT>
        inline MatrixComplementView<MatrixT> matrix_complement(MatrixT const &Mask)
        {
            return MatrixComplementView<MatrixT>(Mask);
        }

        template<typename VectorT>
        inline VectorComplementView<VectorT> vector_complement(VectorT const &mask)
        {
            return VectorComplementView<VectorT>(mask);
        }


        /**
         *
         */
        template<typename MatrixT>
        inline TransposeView<MatrixT> transpose(MatrixT const &A)
        {
            return TransposeView<MatrixT>(A);
        }

    } // backend
} // GraphBLAS

//****************************************************************************
/// @deprecated
//****************************************************************************

namespace graphblas
{
namespace backend{
    /**
     * @todo This is an internal function only. Use apply(),
     *       ewiseadd(), or ewisemult() instead
     */
    template<typename AMatrixT,
             typename BMatrixT,
             typename CMatrixT,
             typename MonoidT,
             typename AccumT =
                 graphblas::math::Assign<typename CMatrixT::ScalarType> >
    inline void _ewiseapply(AMatrixT const &a,
                            BMatrixT const &b,
                            CMatrixT       &c,
                            MonoidT         func  = MonoidT(),
                            AccumT          accum = AccumT())
    {
        IndexType a_num_rows, a_num_cols;
        IndexType b_num_rows, b_num_cols;
        IndexType c_num_rows, c_num_cols;

        a.get_shape(a_num_rows, a_num_cols);
        b.get_shape(b_num_rows, b_num_cols);
        c.get_shape(c_num_rows, c_num_cols);

        // Assert shapes are the same.
        if ((a_num_rows != b_num_rows) || (b_num_rows != c_num_rows) ||
            (a_num_cols != b_num_cols) || (b_num_cols != c_num_cols))
        {
            std::string err_msg =
                std::to_string(a_num_rows) + "," + std::to_string(b_num_rows) + "\n" +
                std::to_string(b_num_rows) + "," + std::to_string(c_num_rows) + "\n" +
                std::to_string(a_num_cols) + "," + std::to_string(b_num_cols) + "\n" +
                std::to_string(b_num_cols) + "," + std::to_string(c_num_cols);
            throw DimensionException(err_msg);
        }

        /// @todo Assert that all Matrixs have the same ScalarType.

        for (IndexType i = 0; i < a_num_rows; i++)
        {
            for (IndexType j = 0; j < a_num_cols; j++)
            {
                c.setElement(i, j,
                               accum(c.extractElement(i, j),
                                     func(a.extractElement(i,j),
                                          b.extractElement(i,j))));
            }
        }
    }

    /**
     *
     */
    template<typename AMatrixT,
             typename BMatrixT,
             typename CMatrixT,
             typename MonoidT,
             typename AccumT>
    inline void ewiseadd(AMatrixT const &a,
                         BMatrixT const &b,
                         CMatrixT       &c,
                         MonoidT         func,
                         AccumT          accum)
    {
        /// @todo optimize for addition (additive identity) here.
        return _ewiseapply(a, b, c, func, accum);
    }

    /**
     *
     */
    template<typename AMatrixT,
             typename BMatrixT,
             typename CMatrixT,
             typename MonoidT,
             typename AccumT>
    inline void ewisemult(AMatrixT const &a,
                          BMatrixT const &b,
                          CMatrixT       &c,
                          MonoidT         func,
                          AccumT          accum)
    {
        // @todo  optimize for multiplication (annihilator) here.
        return _ewiseapply(a, b, c, func, accum);
    }

    /**
     *
     */
    template<typename AMatrixT,
             typename BMatrixT,
             typename CMatrixT,
             typename MMatrixT,
             typename MonoidT,
             typename AccumT>
    inline void ewisemultMasked(AMatrixT const &a,
                                BMatrixT const &b,
                                CMatrixT       &c,
                                MMatrixT const &m,
                                bool            replace_flag,
                                MonoidT         func,
                                AccumT          accum)
    {

        IndexType num_rows, num_cols;

        a.get_shape(num_rows, num_cols);

        // Assert shapes are the same.
        // DONE IN frontend

        /// @todo Assert that all Matrixs have the same ScalarType.

        /// @todo can we assign directly to c?
        CMatrixT tmp(num_rows, num_cols, c.get_zero());

        for (IndexType i = 0; i < num_rows; i++)
        {
            for (IndexType j = 0; j < num_cols; j++)
            {
                if ((a.extractElement(i, j) != a.get_zero()) &&
                    (b.extractElement(i, j) != b.get_zero()))
                {
                    tmp.setElement(i, j,
                                     accum(c.extractElement(i, j),
                                           func(a.extractElement(i,j),
                                                b.extractElement(i,j))));
                }
            }
        }

        // Apply merge or replace semantics
        for (IndexType i = 0; i < num_rows; i++)
        {
            for (IndexType j = 0; j < num_cols; j++)
            {
                if (m.extractElement(i,j))
                {
                    c.setElement(i, j, tmp.extractElement(i, j));
                }
                else if (replace_flag)
                {
                    c.setElement(i, j, c.get_zero());
                }
            }
        }
    }

    /**
     *
     */
    template<typename AMatrixT,
             typename BMatrixT,
             typename CMatrixT,
             typename SemiringT,
             typename AccumT>
    inline void mxm(AMatrixT const &a,
                    BMatrixT const &b,
                    CMatrixT       &c,
                    SemiringT       s,
                    AccumT          accum)
    {
        IndexType a_num_rows, a_num_cols;
        IndexType b_num_rows, b_num_cols;
        IndexType c_num_rows, c_num_cols;

        a.get_shape(a_num_rows, a_num_cols);
        b.get_shape(b_num_rows, b_num_cols);
        c.get_shape(c_num_rows, c_num_cols);

        // Assert shapes are the compatible.
        if ((a_num_cols != b_num_rows) ||
            (a_num_rows != c_num_rows) ||
            (b_num_cols != c_num_cols))
        {
            throw DimensionException("mxm: shapes are incompatible");
        }

        /// @todo Assert that all Matrixs have the same ScalarType.

        /// @todo Can we just assign directly into c?
        CMatrixT tmp(c_num_rows, c_num_cols, c.get_zero());
        auto mult_res = s.zero();

        for (IndexType i = 0; i < a_num_rows; i++)
        {
            for (IndexType j = 0; j < b_num_cols; j++)
            {
                auto tmp_sum = s.zero();
                for (IndexType k = 0; k < b_num_rows; k++)
                {
                    mult_res = s.mult(a.extractElement(i, k),
                                      b.extractElement(k, j));
                    tmp_sum = s.add(tmp_sum, mult_res);
                }
                tmp.setElement(i, j, tmp_sum);
            }
        }

        // Accum or assign results
        for (IndexType i = 0; i < a_num_rows; i++)
        {
            for (IndexType j = 0; j < b_num_cols; j++)
            {
                c.setElement(i, j,
                               accum(c.extractElement(i, j),
                                     tmp.extractElement(i, j)));
            }
        }
    }

    /**
     *
     */
    template<typename AMatrixT,
             typename BMatrixT,
             typename CMatrixT,
             typename MMatrixT,
             typename SemiringT,
             typename AccumT>
    inline void mxmMasked(AMatrixT const  &a,
                          BMatrixT const  &b,
                          CMatrixT        &c,
                          MMatrixT const  &m,
                          SemiringT        s,
                          AccumT           accum)
    {
        IndexType a_num_rows, a_num_cols;
        IndexType b_num_rows, b_num_cols;
        IndexType c_num_rows, c_num_cols;
        IndexType m_num_rows, m_num_cols;

        a.get_shape(a_num_rows, a_num_cols);
        b.get_shape(b_num_rows, b_num_cols);
        c.get_shape(c_num_rows, c_num_cols);
        m.get_shape(m_num_rows, m_num_cols);

        // Assert shapes are the compatible.
        if ((a_num_cols != b_num_rows) ||
            (a_num_rows != c_num_rows) ||
            (b_num_cols != c_num_cols) ||
            (c_num_rows != m_num_rows) ||
            (c_num_cols != m_num_cols))
        {
            throw DimensionException("mxmMasked: shapes are incompatible");
        }

        CMatrixT tmp(c_num_rows, c_num_cols, c.get_zero());

        for (IndexType i = 0; i < c_num_rows; ++i)
        {
            for (IndexType j = 0; j < c_num_cols; ++j)
            {
                if (m.extractElement(i,j) != m.get_zero()) /// @todo Or s.zero()?
                {
                    auto tmp_sum = s.zero();
                    for (IndexType k = 0; k < b_num_rows; ++k)
                    {
                        auto mult_res = s.mult(a.extractElement(i, k),
                                               b.extractElement(k, j));
                        tmp_sum = s.add(tmp_sum, mult_res);
                    }
                    tmp.setElement(i, j, tmp_sum);
                }
            }
        }

         // Accum or assign results
        for (IndexType i = 0; i < a_num_rows; ++i)
        {
            for (IndexType j = 0; j < b_num_cols; ++j)
            {
                c.setElement(i, j,
                               accum(c.extractElement(i, j),
                                     tmp.extractElement(i, j)));
            }
        }
    }

    /**
     *
     */
    template<typename AMatrixT,
             typename BMatrixT,
             typename CMatrixT,
             typename MMatrixT,
             typename SemiringT,
             typename AccumT>
    inline void mxmMaskedV2(AMatrixT const &a,
                            BMatrixT const &b,
                            CMatrixT       &c,
                            MMatrixT       &m,
                            SemiringT       s,
                            AccumT          accum)
    {
        IndexType c_num_rows, c_num_cols;
        IndexType m_num_rows, m_num_cols;

        c.get_shape(c_num_rows, c_num_cols);
        m.get_shape(m_num_rows, m_num_cols);

        // Assert shapes are the compatible.
        if ((c_num_rows != m_num_rows) ||
            (c_num_cols != m_num_cols))
        {
            throw DimensionException();
        }

        backend::mxm(a, b, c, s, accum);

        for (IndexType ri = 0; ri < c_num_rows; ++ri)
        {
            for (IndexType ci = 0; ci < c_num_cols; ++ci)
            {
                if (m.extractElement(ri, ci) == m.get_zero()) /// @todo s.zero()?
                {
                    c.setElement(ri, ci, c.get_zero()); /// @todo s.zero()?
                }
            }
        }
    }

    /**
     *
     */
    template<typename AVectorT,
             typename BMatrixT,
             typename CVectorT,
             typename SemiringT,
             typename AccumT>
    inline void vxm(AVectorT const &a,
                    BMatrixT const &b,
                    CVectorT       &c,
                    SemiringT       s,
                    AccumT          accum)
    {
        IndexType a_num_rows, a_num_cols, c_num_rows, c_num_cols;

        a.get_shape(a_num_rows, a_num_cols);
        c.get_shape(c_num_rows, c_num_cols);

        // Assert the row vector shapes, mxm does the rest
        /// @todo Transposes not supported
        if ((a_num_rows != 1) || (c_num_rows != 1))
        {
            throw DimensionException("vxm: a and c must be vectors.");
        }

        backend::mxm(a, b, c, s, accum);
    }

    /**
     *
     */
    template<typename AMatrixT,
             typename BVectorT,
             typename CVectorT,
             typename SemiringT,
             typename AccumT>
    inline void mxv(AMatrixT const &a,
                    BVectorT const &b,
                    CVectorT       &c,
                    SemiringT       s,
                    AccumT          accum)
    {
        IndexType b_num_rows, b_num_cols, c_num_rows, c_num_cols;

        b.get_shape(b_num_rows, b_num_cols);
        c.get_shape(c_num_rows, c_num_cols);

        // Assert the column vector shapes, mxm does the rest
        /// @todo Transposes not supported
        if ((b_num_cols != 1) || (c_num_cols != 1))
        {
            throw DimensionException("mxv: b and c must be vectors.");
        }

        backend::mxm(a, b, c, s, accum);
    }

    /**
     *
     */
    template<typename AMatrixT,
             typename CMatrixT,
             typename RAIteratorI,
             typename RAIteratorJ,
             typename AccumT>
    inline void extract(AMatrixT       const &a,
                        RAIteratorI           i_it,
                        RAIteratorJ           j_it,
                        CMatrixT             &c,
                        AccumT                accum)
    {
        IndexType m_C, n_C;
        c.get_shape(m_C, n_C);

        /// @todo is there any way to assert this?
        // len(i_it) == m_C
        // len(j_it) == n_C

        /// @todo need to detect when assigning from out of range in A
        //IndexType m_A, n_A;
        //a.get_shape(m_A, n_A);

        /// %todo there is a better way that involves increment only
        for (IndexType i = 0; i < m_C; ++i)
        {
            for (IndexType j = 0; j < n_C; ++j)
            {
                /// @todo How do we detect and handle structural zeros
                ///       for accum or assign.
                c.setElement(i, j,
                               accum(c.extractElement(i, j),
                                     a.extractElement(i_it[i], j_it[j])));
            }
        }
    }

    /**
     *
     */
    template<typename AMatrixT,
             typename CMatrixT,
             typename AccumT>
    inline void extract(AMatrixT const       &a,
                        IndexArrayType const &v_i,
                        IndexArrayType const &v_j,
                        CMatrixT             &c,
                        AccumT                accum)
    {
        IndexType m_C, n_C;
        c.get_shape(m_C, n_C);

        IndexType len_i = v_i.size();
        IndexType len_j = v_j.size();

        // assert that assignment is in range of dimensions of C.
        if ((len_i != m_C) || (len_j != n_C))
        {
            throw DimensionException();
        }

        backend::extract(a, v_i.begin(), v_j.begin(), c, accum);
    }

    /**
     *
     */
    template<typename AMatrixT,
             typename CMatrixT,
             typename RAIteratorI,
             typename RAIteratorJ,
             typename AccumT>
    inline void assign(AMatrixT const    &a,
                       RAIteratorI        i_it,
                       RAIteratorJ        j_it,
                       CMatrixT          &c,
                       AccumT             accum)
    {
        IndexType m_A, n_A;
        a.get_shape(m_A, n_A);

        /// @todo any way to assert the following?
        /// len(i_it) == m_A
        /// len(j_it) == n_A

        /// @todo Need to check that dimension assigning to are correct?
        ///       Do that in the call to c.setElement()

        /// %todo there is a better way that involves increment only
        for (IndexType i = 0; i < m_A; ++i)
        {
            for (IndexType j = 0; j < n_A; ++j)
            {
                /// @todo What do we do with structural zeros on
                ///  the rhs or lhs of this operation?
                c.setElement(
                    *(i_it + i),
                    *(j_it + j),
                    accum(c.extractElement(i_it[i], j_it[j]),
                          a.extractElement(i, j)));
            }
        }
    }

    /**
     *
     */
    template<typename AMatrixT,
             typename CMatrixT,
             typename AccumT>
    inline void assign(AMatrixT const        &a,
                       IndexArrayType const  &v_i,
                       IndexArrayType const  &v_j,
                       CMatrixT              &c,
                       AccumT                 accum)
    {
        IndexType m_A, n_A;
        a.get_shape(m_A, n_A);

        IndexType len_i = v_i.size();
        IndexType len_j = v_j.size();

        // assert that the dimension assigning from are correct
        if ((len_i != m_A) || (len_j != n_A))
        {
            throw DimensionException();
        }

        backend::assign(a, v_i.begin(), v_j.begin(), c, accum);
    }

    /**
     *
     */
    template<typename AMatrixT,
             typename CMatrixT,
             typename UnaryFunctionT,
             typename AccumT>
    inline void apply(AMatrixT const &a,
                      CMatrixT       &c,
                      UnaryFunctionT  func,
                      AccumT          accum)
    {
        IndexType a_num_rows, a_num_cols, c_num_rows, c_num_cols;

        a.get_shape(a_num_rows, a_num_cols);
        c.get_shape(c_num_rows, c_num_cols);

        // Assert shapes are the same.
        if ((a_num_rows != c_num_rows) || (a_num_cols != c_num_cols))
        {
            std::string err_msg =
                std::to_string(a_num_rows) + "," + std::to_string(c_num_rows) + "\n" +
                std::to_string(a_num_cols) + "," + std::to_string(c_num_cols);
            throw DimensionException(err_msg);
        }

        /// @todo Assert that all Matrixs have the same ScalarType?

        /// @todo Can we just assign into c?
        for (IndexType i = 0; i < a_num_rows; i++)
        {
            for (IndexType j = 0; j < a_num_cols; j++)
            {
                c.setElement(i, j,
                               accum(c.extractElement(i, j),
                                     func(a.extractElement(i, j))));
            }
        }
    }

    /**
     *
     */
    template<typename AMatrixT,
             typename CMatrixT,
             typename MonoidT,
             typename AccumT>
    inline void row_reduce(AMatrixT const &a,
                           CMatrixT       &c,
                           MonoidT         sum,
                           AccumT          accum)
    {
        IndexType M, N;
        a.get_shape(M, N);
        IndexType m, n;
        c.get_shape(m, n);

        if ((m != M) || (n != 1))
        {
            throw DimensionException();
        }

        for (IndexType i = 0; i < M; ++i)
        {
            typename AMatrixT::ScalarType tmp_sum = sum.identity();
            for (IndexType j = 0; j < N; ++j)
            {
                tmp_sum = sum(tmp_sum, a.extractElement(i, j));
            }
            c.setElement(i, 0,
                           accum(c.extractElement(i, 0), tmp_sum));
        }
    }

    /**
     *
     */
    template<typename AMatrixT,
             typename CMatrixT,
             typename MonoidT,
             typename AccumT>
    inline void col_reduce(AMatrixT const &a,
                           CMatrixT       &c,
                           MonoidT         sum,
                           AccumT          accum)
    {
        IndexType M, N;
        a.get_shape(M, N);
        IndexType m, n;
        c.get_shape(m, n);

        if ((m != 1) || (n != N))
        {
            throw DimensionException();
        }

        for (IndexType j = 0; j < N; ++j)
        {
            typename AMatrixT::ScalarType tmp_sum = sum.identity();
            for (IndexType i = 0; i < M; ++i)
            {
                tmp_sum = sum(tmp_sum, a.extractElement(i, j));
            }
            c.setElement(0, j,
                           accum(c.extractElement(0, j), tmp_sum));
        }
    }

    /**
    *
    */
    template<typename AMatrixT,
             typename CMatrixT,
             typename MMatrixT,
             typename MonoidT =
                 graphblas::PlusMonoid<typename AMatrixT::ScalarType>,
             typename AccumT =
                 graphblas::math::Assign<typename CMatrixT::ScalarType> >
    inline void rowReduceMasked(AMatrixT const &a,
                           CMatrixT       &c,
                           MMatrixT       &mask,
                           MonoidT         sum     = MonoidT(),
                           AccumT          accum = AccumT())
    {
        IndexType M, N;
        a.get_shape(M, N);
        IndexType m, n;
        c.get_shape(m, n);

        if ((m != M) || (n != 1))
        {
            throw DimensionException();
        }

        for (IndexType i = 0; i < M; ++i)
        {
            typename AMatrixT::ScalarType tmp_sum = sum.identity();
            if(mask.extractElement(i,0) != mask.get_zero())
            {
                for (IndexType j = 0; j < N; ++j)
                {
                    tmp_sum = sum(tmp_sum, a.extractElement(i, j));
                }
                c.setElement(i, 0,
                               accum(c.extractElement(i, 0), tmp_sum));
            }
        }
    }

    /**
    *
    */
    template<typename AMatrixT,
             typename CMatrixT,
             typename MMatrixT,
             typename MonoidT =
                 graphblas::PlusMonoid<typename AMatrixT::ScalarType>,
             typename AccumT =
                 graphblas::math::Assign<typename CMatrixT::ScalarType> >
    inline void colReduceMasked(AMatrixT const &a,
                           CMatrixT       &c,
                           MMatrixT       &mask,
                           MonoidT         sum    = MonoidT(),
                           AccumT          accum = AccumT())
    {
        IndexType M, N;
        a.get_shape(M, N);
        IndexType m, n;
        c.get_shape(m, n);

        if ((m != 1) || (n != N))
        {
            throw DimensionException();
        }

        for (IndexType j = 0; j < N; ++j)
        {
            typename AMatrixT::ScalarType tmp_sum = sum.identity();
            if(mask.extractElement(0,j) != mask.get_zero())
            {
                for (IndexType i = 0; i < M; ++i)
                {
                    tmp_sum = sum(tmp_sum, a.extractElement(i, j));
                }
                c.setElement(0, j,
                               accum(c.extractElement(0, j), tmp_sum));
            }
        }
    }

    /**
    *
    */
    template<typename MatrixT, typename SemiringT>
    inline NegateView<MatrixT, SemiringT> negate(MatrixT const   &a,
                                                 SemiringT const &)
    {
        return NegateView<MatrixT, SemiringT>(a);
    }

    /**
     *
     */
    template<typename AMatrixT,
             typename CMatrixT>
    inline void transpose(AMatrixT const &a,
                          CMatrixT       &c)
    {
        IndexType a_num_rows, a_num_cols, c_num_rows, c_num_cols;
        a.get_shape(a_num_rows, a_num_cols);
        c.get_shape(c_num_rows, c_num_cols);

        // Assert shapes are compatible.
        if ((a_num_rows != c_num_cols) || (a_num_cols != c_num_rows))
        {
            throw DimensionException();
        }

        for (IndexType i = 0; i < c_num_rows; ++i)
        {
            for (IndexType j = 0; j < c_num_cols; ++j)
            {
                c.setElement(i, j, a.extractElement(j, i));
            }
        }
    }

    /**
    *
    */
    template<typename MatrixT>
    inline TransposeView<MatrixT> transpose(MatrixT const &a)
    {
        return TransposeView<MatrixT>(a);
    }

    /**
     *  The random access iterators must point to containers pre-sized to the
     *  minimum required length
     */
    template<typename AMatrixT,
             typename RAIteratorIT,
             typename RAIteratorJT,
             typename RAIteratorVT>
    inline void extracttuples(AMatrixT const &a,
                              RAIteratorIT    i,
                              RAIteratorJT    j,
                              RAIteratorVT    v)
    {
        IndexType num_rows, num_cols;
        a.get_shape(num_rows, num_cols);

        IndexType idx = 0;
        for (IndexType ii = 0; ii < num_rows; ii++)
        {
            for (IndexType jj = 0; jj < num_cols; jj++)
            {
                auto matrix_value = a.extractElement(ii, jj);
                if (matrix_value != a.get_zero())
                {
                    i[idx] = ii;
                    j[idx] = jj;
                    v[idx] = matrix_value;
                    ++idx;
                }
            }
        }
    }

    /**
     *
     */
    template<typename AMatrixT>
    inline void extracttuples(AMatrixT const &a,
                              IndexArrayType                             &v_i,
                              IndexArrayType                             &v_j,
                              std::vector<typename AMatrixT::ScalarType> &val)
    {
        IndexType nnz = a.get_nnz();

        /// @todo should we clear the destination or append?
        if ((v_i.size() < nnz) || (v_j.size() < nnz) || (val.size() < nnz))
        {
            throw DimensionException();
        }

        backend::extracttuples(a, v_i.begin(), v_j.begin(), val.begin());
    }

    /**
     *  @todo Need to add a parameter (functor?) to handle duplicate locations
     */
    template<typename MatrixT,
             typename RAIteratorI,
             typename RAIteratorJ,
             typename RAIteratorV,
             typename AccumT>
    inline void buildmatrix(MatrixT     &m,
                            RAIteratorI  i,
                            RAIteratorJ  j,
                            RAIteratorV  v,
                            IndexType    n,
                            AccumT       accum)
    {
        m.buildmatrix(i, j, v, n, accum);
    }

    /**
     *  @todo Need to add a parameter (functor?) to handle duplicate locations
     */
    template<typename MatrixT,
             typename AccumT>
    inline void buildmatrix(MatrixT              &m,
                            IndexArrayType const &i,
                            IndexArrayType const &j,
                            std::vector<typename MatrixT::ScalarType> const &v,
                            AccumT                accum)
    {
        /// @todo if accum==Assign, need to clear the matrix data first?
        if ((i.size() != j.size()) || (i.size() != v.size()))
        {
            throw DimensionException();
        }
        buildmatrix(m, i.begin(), j.begin(), v.begin(), i.size(), accum);
    }
}//backend
} // graphblas

#endif // GB_SEQUENTIAL_OPERATIONS_HPP
