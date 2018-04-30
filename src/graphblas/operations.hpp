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

#ifndef GB_OPERATIONS_HPP
#define GB_OPERATIONS_HPP

#pragma once

#include <cstddef>
#include <vector>

#include <graphblas/algebra.hpp>
#include <graphblas/TransposeView.hpp>
#include <graphblas/ComplementView.hpp>

#include <graphblas/Matrix.hpp>
#include <graphblas/Vector.hpp>
#include <graphblas/indices.hpp>

#include <graphblas/detail/logging.h>
#include <graphblas/detail/config.hpp>
#include <graphblas/detail/checks.hpp>

#define GB_INCLUDE_BACKEND_TRANSPOSE_VIEW 1
#define GB_INCLUDE_BACKEND_COMPLEMENT_VIEW 1
#define GB_INCLUDE_BACKEND_OPERATIONS 1
#include <graphblas/backend_include.hpp>

//****************************************************************************
// New signatures to conform to GraphBLAS Specification
//****************************************************************************

namespace GraphBLAS
{
    //************************************************************************
    // mxm, vxm, mxv
    //************************************************************************

    // 4.3.1: Matrix-matrix multiply
    template<typename CMatrixT,
             typename MaskT,
             typename AccumT,
             typename SemiringT,
             typename AMatrixT,
             typename BMatrixT>
    inline void mxm(CMatrixT         &C,
                    MaskT      const &Mask,
                    AccumT            accum,
                    SemiringT         op,
                    AMatrixT   const &A,
                    BMatrixT   const &B,
                    bool              replace_flag = false)
    {
        GRB_LOG_FN_BEGIN("mxm - 4.3.1 - matrix-matrix multiply");

        GRB_LOG_VERBOSE("C in :" << C.m_mat);
        GRB_LOG_VERBOSE("Mask in : " << Mask.m_mat);
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE_OP(op);
        GRB_LOG_VERBOSE("A in :" << A.m_mat);
        GRB_LOG_VERBOSE("B in :" << B.m_mat);
        GRB_LOG_VERBOSE_REPLACE(replace_flag);

        check_nrows_nrows(C, Mask, "mxm: C.nrows != Mask.nrows");
        check_ncols_ncols(C, Mask, "mxm: C.ncols != Mask.ncols");
        check_nrows_nrows(C, A, "mxm: C.nrows != A.nrows");
        check_ncols_ncols(C, B, "mxm: C.ncols != B.ncols");
        check_ncols_nrows(A, B, "mxm: A.ncols != B.nrows");

        backend::mxm(C.m_mat, Mask.m_mat, accum, op, A.m_mat, B.m_mat,
                     replace_flag);

        GRB_LOG_VERBOSE("C (Result): " << C.m_mat);
        GRB_LOG_FN_END("mxm - 4.3.1 - matrix-matrix multiply");
    }

    //************************************************************************

    // 4.3.2: Vector-matrix multiply
    template<typename WVectorT,
             typename MaskT,
             typename AccumT,
             typename SemiringT,
             typename UVectorT,
             typename AMatrixT>
    inline void vxm(WVectorT         &w,
                    MaskT      const &mask,
                    AccumT            accum,
                    SemiringT         op,
                    UVectorT   const &u,
                    AMatrixT   const &A,
                    bool              replace_flag = false)
    {
        check_size_size(w, mask, "vxm: w.size != mask.size");
        check_size_ncols(w, A, "vxm: w.size != A.ncols");
        check_size_nrows(u, A, "vxm: u.size != A.nrows");

        backend::vxm(w.m_vec, mask.m_vec, accum, op, u.m_vec, A.m_mat,
                     replace_flag);
    }

    //************************************************************************

    // 4.3.3: Matrix-vector multiply
    template<typename WVectorT,
             typename MaskT,
             typename AccumT,
             typename SemiringT,
             typename AMatrixT,
             typename UVectorT>
    inline void mxv(WVectorT        &w,
                    MaskT     const &mask,
                    AccumT           accum,
                    SemiringT        op,
                    AMatrixT  const &A,
                    UVectorT  const &u,
                    bool             replace_flag = false)
    {
        GRB_LOG_FN_BEGIN("mxv - 4.3.3 - matrix-vector multiply");

        GRB_LOG_VERBOSE("w in :" << w.m_vec);
        GRB_LOG_VERBOSE("Mask in : " << mask.m_vec);
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE_OP(op);
        GRB_LOG_VERBOSE("A in :" << A.m_mat);
        GRB_LOG_VERBOSE("u in :" << u.m_vec);
        GRB_LOG_VERBOSE_REPLACE(replace_flag);

        check_size_size(w, mask, "mxv: w.size != mask.size");
        check_size_nrows(w, A, "mxv: w.size != A.nrows");
        check_size_ncols(u, A, "mxv: u.size != A.ncols");

        backend::mxv(w.m_vec, mask.m_vec, accum, op, A.m_mat, u.m_vec,
                     replace_flag);
        GRB_LOG_FN_END("mxv - 4.3.3 - matrix-vector multiply");
    }


    //************************************************************************
    // eWiseAdd and eWiseMult
    //************************************************************************

    // 4.3.4.1: Element-wise multiplication - vector variant
    /**
     * @brief Perform an element wise binary operation that can be optimized
     *        for "multiply" semantics (AND short circuit logic).
     */
    template<typename WScalarT,
             typename MaskT,
             typename AccumT,
             typename BinaryOpT,  //can be BinaryOp, Monoid (not Semiring)
             typename UVectorT,
             typename VVectorT,
             typename ...WTagsT>
    inline void eWiseMult(Vector<WScalarT, WTagsT...> &w,
                          MaskT                 const &mask,
                          AccumT                       accum,
                          BinaryOpT                    op,
                          UVectorT              const &u,
                          VVectorT              const &v,
                          bool                         replace_flag = false)
    {
        check_size_size(w, mask, "eWiseMult(vec): w.size != mask.size");
        check_size_size(w, u, "eWiseMult(vec): w.size != u.size");
        check_size_size(u, v, "eWiseMult(vec): u.size != v.size");

        backend::eWiseMult(w.m_vec, mask.m_vec, accum, op, u.m_vec, v.m_vec,
                           replace_flag);
    }

    // 4.3.4.2: Element-wise multiplication - matrix variant
    template<typename CScalarT,
             typename MaskT,
             typename AccumT,
             typename BinaryOpT,  //can be BinaryOp, Monoid (not Semiring)
             typename AMatrixT,
             typename BMatrixT,
             typename ...CTagsT>
    inline void eWiseMult(Matrix<CScalarT, CTagsT...> &C,
                          MaskT                 const &Mask,
                          AccumT                       accum,
                          BinaryOpT                    op,
                          AMatrixT              const &A,
                          BMatrixT              const &B,
                          bool                         replace_flag = false)
    {
        check_ncols_ncols(C, Mask, "eWiseMult(mat): C.ncols != Mask.ncols");
        check_nrows_nrows(C, Mask, "eWiseMult(mat): C.nrows != Mask.nrows");
        check_ncols_ncols(C, A, "eWiseMult(mat): C.ncols != A.ncols");
        check_nrows_nrows(C, A, "eWiseMult(mat): C.nrows != A.nrows");
        check_ncols_ncols(A, B, "eWiseMult(mat): A.ncols != B.ncols");
        check_nrows_nrows(A, B, "eWiseMult(mat): A.nrows != B.nrows");

        backend::eWiseMult(C.m_mat, Mask.m_mat, accum, op, A.m_mat, B.m_mat,
                           replace_flag);
    }

    //************************************************************************

    // 4.3.5.1: Element-wise addition - vector variant
    /**
     * @brief Perform an element wise binary operation that can be optimized
     *        for "add" semantics (OR short circuit logic).
     */
    template<typename WScalarT,
             typename MaskT,
             typename AccumT,
             typename BinaryOpT,  //can be BinaryOp, Monoid (not Semiring)
             typename UVectorT,
             typename VVectorT,
             typename ...WTagsT>
    inline void eWiseAdd(Vector<WScalarT, WTagsT...> &w,
                         MaskT                 const &mask,
                         AccumT                       accum,
                         BinaryOpT                    op,
                         UVectorT              const &u,
                         VVectorT              const &v,
                         bool                         replace_flag = false)
    {
        check_size_size(w, mask, "eWiseAdd(vec): w.size != mask.size");
        check_size_size(w, u, "eWiseAdd(vec): w.size != u.size");
        check_size_size(u, v, "eWiseAdd(vec): u.size != v.size");

        backend::eWiseAdd(w.m_vec, mask.m_vec, accum, op, u.m_vec, v.m_vec,
                          replace_flag);
    }

    // 4.3.5.2: Element-wise addition - matrix variant
    template<typename CScalarT,
             typename MaskT,
             typename AccumT,
             typename BinaryOpT,  //can be BinaryOp, Monoid (not Semiring)
             typename AMatrixT,
             typename BMatrixT,
             typename ...CTagsT>
    inline void eWiseAdd(Matrix<CScalarT, CTagsT...> &C,
                         MaskT                 const &Mask,
                         AccumT                       accum,
                         BinaryOpT                    op,
                         AMatrixT              const &A,
                         BMatrixT              const &B,
                         bool                         replace_flag = false)
    {
        check_ncols_ncols(C, Mask, "eWiseAdd(mat): C.ncols != Mask.ncols");
        check_nrows_nrows(C, Mask, "eWiseAdd(mat): C.nrows != Mask.nrows");
        check_ncols_ncols(C, A, "eWiseAdd(mat): C.ncols != A.ncols");
        check_nrows_nrows(C, A, "eWiseAdd(mat): C.nrows != A.nrows");
        check_ncols_ncols(A, B, "eWiseAdd(mat): A.ncols != B.ncols");
        check_nrows_nrows(A, B, "eWiseAdd(mat): A.nrows != B.nrows");

        backend::eWiseAdd(C.m_mat, Mask.m_mat, accum, op, A.m_mat, B.m_mat,
                          replace_flag);
    }

    //************************************************************************
    // Extract
    //************************************************************************

    // 4.3.6.1 - extract: Standard vector variant
    template<typename WVectorT,
             typename MaskT,
             typename AccumT,
             typename UVectorT,
             typename SequenceT>
    inline void extract(WVectorT             &w,
                        MaskT          const &mask,
                        AccumT                accum,
                        UVectorT       const &u,
                        SequenceT      const &indices,
                        bool                  replace_flag = false)
    {
        GRB_LOG_FN_BEGIN("extract - 4.3.6.1 - standard vector variant");

        GRB_LOG_VERBOSE("w:    " << w.m_vec);
        GRB_LOG_VERBOSE("mask: " << mask.m_vec);
        GRB_LOG_VERBOSE("u:    " << u.m_vec);
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE("indices: " << indices);
        GRB_LOG_VERBOSE_REPLACE(replace_flag);

        check_size_size(w, mask, "extract(std vec): w.size != mask.size");
        check_size_nindices(w, indices,
                            "extract(std vec): w.size != indicies.size");

        backend::extract(w.m_vec, mask.m_vec, accum, u.m_vec,
                         indices, replace_flag);

        GRB_LOG_FN_END("extract - 4.3.6.1 - standard vector variant");
    }

    // 4.3.6.2 - extract: Standard matrix variant
    template<typename CScalarT,
             typename MaskT,
             typename AccumT,
             typename AMatrixT,
             typename RowSequenceT,
             typename ColSequenceT,
             typename ...CTags>
    inline void extract(Matrix<CScalarT, CTags...>   &C,
                        MaskT             const &Mask,
                        AccumT                   accum,
                        AMatrixT          const &A,
                        RowSequenceT      const &row_indices,
                        ColSequenceT      const &col_indices,
                        bool                     replace_flag = false)
    {
        GRB_LOG_FN_BEGIN("extract - 4.3.6.2 - standard matrix variant");

        GRB_LOG_VERBOSE("C: " << C.m_mat);
        GRB_LOG_VERBOSE("Mask: " << Mask.m_mat);
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE("A: " << A.m_mat);
        GRB_LOG_VERBOSE("row_indices: " << row_indices);
        GRB_LOG_VERBOSE("col_indices: " << col_indices);
        GRB_LOG_VERBOSE_REPLACE(replace_flag);

        check_nrows_nrows(C, Mask, "extract(std mat): C.nrows != Mask.nrows");
        check_ncols_ncols(C, Mask, "extract(std mat): C.ncols != Mask.ncols");
        check_nrows_nindices(C, row_indices,
                             "extract(std mat): C.nrows != row_indices");
        check_ncols_nindices(C, col_indices,
                             "extract(std mat): C.ncols != col_indices");

        backend::extract(C.m_mat, Mask.m_mat, accum, A.m_mat,
                         row_indices, col_indices, replace_flag);

        GRB_LOG_FN_END("SEQUENTIAL extract - 4.3.6.2 - standard matrix variant");
    }

    // 4.3.6.3 - extract: Column (and row) variant
    // Extract col (or row with transpose)
    template<typename WScalarT,
             typename MaskT,
             typename AccumT,
             typename AMatrixT,
             typename SequenceT,
             typename ...WTags>
    inline void extract(Vector<WScalarT, WTags...> &w,
                        MaskT                const &mask,
                        AccumT                      accum,
                        AMatrixT             const &A,
                        SequenceT            const &row_indices,
                        IndexType                   col_index,
                        bool                        replace_flag = false)
    {
        GRB_LOG_FN_BEGIN("extract - 4.3.6.3 - column (and row) variant");

        GRB_LOG_VERBOSE("w:    " << w.m_vec);
        GRB_LOG_VERBOSE("mask: " << mask.m_vec);
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE("A:    " << A.m_mat);
        GRB_LOG_VERBOSE("row_indices: " << row_indices);
        GRB_LOG_VERBOSE("col_index:   " << col_index);
        GRB_LOG_VERBOSE_REPLACE(replace_flag);

        check_size_size(w, mask, "extract(col): w.size != mask.size");
        check_size_nindices(w, row_indices,
                            "extract(col): w.size != row_indicies");
        check_index_within_ncols(col_index, A,
                                 "extract(col): col_index >= A.ncols");

        backend::extract(w.m_vec, mask.m_vec, accum, A.m_mat, row_indices,
                         col_index, replace_flag);
        GRB_LOG_FN_END("extract - 4.3.6.3 - column (and row) variant");
    }

    //************************************************************************
    // Assign
    //************************************************************************

    // 4.3.7.1: assign - standard vector variant
    template<typename WScalarT,
             typename MaskT,
             typename AccumT,
             typename UVectorT,
             typename SequenceT,
             typename std::enable_if<
                 std::is_same<vector_tag,
                              typename UVectorT::tag_type>::value,
                 int>::type = 0,
             typename ...WTags>
    inline void assign(Vector<WScalarT, WTags...>      &w,
                       MaskT                    const  &mask,
                       AccumT                           accum,
                       UVectorT                 const  &u,
                       SequenceT                const  &indices,
                       bool                             replace_flag = false)
    {
        GRB_LOG_FN_BEGIN("assign - 4.3.7.1 - standard vector variant");
        GRB_LOG_VERBOSE("w in: " << w.m_vec);
        GRB_LOG_VERBOSE("mask in: " << mask.m_vec);
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE("u in: " << u.m_vec);
        GRB_LOG_VERBOSE("indicies in: " << indices);
        GRB_LOG_VERBOSE_REPLACE(replace_flag);

        check_size_size(w, mask, "assign(std vec): w.size != mask.size");
        check_size_nindices(u, indices,
                            "assign(std vec): u.size != |indicies|");

        backend::assign(w.m_vec, mask.m_vec, accum, u.m_vec, indices,
                        replace_flag);

        GRB_LOG_VERBOSE("w out: " << w.m_vec);
        GRB_LOG_FN_END("assign - 4.3.7.1 - standard vector variant");
    }

    // 4.3.7.2: assign - standard matrix variant
    template<typename CMatrixT,
             typename MaskT,
             typename AccumT,
             typename AMatrixT,
             typename RowSequenceT,
             typename ColSequenceT,
             typename std::enable_if<
                 std::is_same<matrix_tag,
                              typename AMatrixT::tag_type>::value,
                 int>::type = 0>
    inline void assign(CMatrixT              &C,
                       MaskT           const &Mask,
                       AccumT                 accum,
                       AMatrixT        const &A,
                       RowSequenceT    const &row_indices,
                       ColSequenceT    const &col_indices,
                       bool                   replace_flag = false)
    {
        GRB_LOG_FN_BEGIN("assign - 4.3.7.2 - standard matrix variant");

        GRB_LOG_VERBOSE("C in: " << C.m_mat);
        GRB_LOG_VERBOSE("Mask in: " << Mask.m_mat);
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE("A in: " << A.m_mat);
        GRB_LOG_VERBOSE("row_indices in: " << row_indices);
        GRB_LOG_VERBOSE("col_indices in: " << col_indices);
        GRB_LOG_VERBOSE_REPLACE(replace_flag);

        check_nrows_nrows(C, Mask, "assign(std mat): C.nrows != Mask.nrows");
        check_ncols_ncols(C, Mask, "assign(std mat): C.ncols != Mask.ncols");
        check_nrows_nindices(A, row_indices,
                             "assign(std mat): A.nrows != |row_indices|");
        check_ncols_nindices(A, col_indices,
                             "assign(std mat): A.ncols != |col_indices|");

        backend::assign(C.m_mat, Mask.m_mat, accum, A.m_mat,
                        row_indices, col_indices, replace_flag);

        GRB_LOG_VERBOSE("C out: " << C.m_mat);
        GRB_LOG_FN_END("assign - 4.3.7.2 - standard matrix variant");
    }

    // 4.3.7.3: assign - column variant
    template<typename CScalarT,
             typename MaskT,
             typename AccumT,
             typename UVectorT,
             typename SequenceT,
             typename ...CTags>
    inline void assign(Matrix<CScalarT, CTags...>  &C,
                       MaskT                 const &mask,  // a vector
                       AccumT                       accum,
                       UVectorT              const &u,
                       SequenceT             const &row_indices,
                       IndexType                    col_index,
                       bool                         replace_flag = false)
    {
        GRB_LOG_FN_BEGIN("assign - 4.3.7.3 - column variant");

        GRB_LOG_VERBOSE("C in: " << C.m_mat);
        GRB_LOG_VERBOSE("mask in: " << mask.m_vec);
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE("u in: " << u.m_vec);
        GRB_LOG_VERBOSE("row_indices in:" << row_indices);
        GRB_LOG_VERBOSE("col_index in:" << col_index);
        GRB_LOG_VERBOSE_REPLACE(replace_flag);

        check_size_nrows(mask, C, "assign(col): C.nrows != mask.size");
        check_size_nindices(u, row_indices,
                            "assign(col): u.size != |row_indices|");
        check_index_within_ncols(col_index, C,
                                 "assign(col): col_index >= C.ncols");

        backend::assign(C.m_mat, mask.m_vec, accum, u.m_vec,
                        row_indices, col_index, replace_flag);

        GRB_LOG_VERBOSE("C out: " << C.m_mat);
        GRB_LOG_FN_END("assign - 4.3.7.3 - column variant");
    }

    // 4.3.7.4: assign - row variant
    template<typename CScalarT,
             typename MaskT,
             typename AccumT,
             typename UVectorT,
             typename SequenceT,
             typename ...CTags>
    inline void assign(Matrix<CScalarT, CTags...>  &C,
                       MaskT                 const &mask,  // a vector
                       AccumT                       accum,
                       UVectorT              const &u,
                       IndexType                    row_index,
                       SequenceT             const &col_indices,
                       bool                         replace_flag = false)
    {
        GRB_LOG_FN_BEGIN("assign - 4.3.7.4 - row variant");
        GRB_LOG_VERBOSE("C in: " << C.m_mat);
        GRB_LOG_VERBOSE("mask in: " << mask.m_vec);
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE("u in: " << u.m_vec);
        GRB_LOG_VERBOSE("row_index in:" << row_index);
        GRB_LOG_VERBOSE("col_indices in:" << col_indices);
        GRB_LOG_VERBOSE_REPLACE(replace_flag);

        check_size_ncols(mask, C, "assign(row): C.ncols != Mask.ncols");
        check_size_nindices(u, col_indices,
                            "assign(row): u.size != |col_indices|");
        check_index_within_nrows(row_index, C,
                                 "assign(col): row_index >= C.nrows");

        backend::assign(C.m_mat, mask.m_vec, accum, u.m_vec,
                        row_index, col_indices, replace_flag);

        GRB_LOG_VERBOSE("C out: " << C.m_mat);
        GRB_LOG_FN_END("assign - 4.3.7.4 - row variant");
    }

    // 4.3.7.5: assign: Constant vector variant
    template<typename WVectorT,
             typename MaskT,
             typename AccumT,
             typename ValueT,
             typename SequenceT,
             typename std::enable_if<
                 std::is_convertible<ValueT,
                                     typename WVectorT::ScalarType>::value,
                 int>::type = 0>
    inline void assign(WVectorT                     &w,
                       MaskT                const   &mask,
                       AccumT                        accum,
                       ValueT                        val,
                       SequenceT            const   &indices,
                       bool                          replace_flag = false)
    {
        GRB_LOG_FN_BEGIN("assign - 4.3.7.5 - constant vector variant");
        GRB_LOG_VERBOSE("w in: " << w.m_vec);
        GRB_LOG_VERBOSE("Mask in: " << mask.m_vec);
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE("val in: " << val);
        GRB_LOG_VERBOSE("indices in:" << indices);
        GRB_LOG_VERBOSE_REPLACE(replace_flag);

        check_size_size(w, mask, "assign(const vec): w.size != mask.size");
        check_nindices_within_size(indices, w,
                                   "assign(const vec): indicies.size !<= w.size");

        backend::assign_constant(w.m_vec, mask.m_vec, accum, val, indices,
                                 replace_flag);

        GRB_LOG_VERBOSE("w out: " << w.m_vec);
        GRB_LOG_FN_END("assign - 4.3.7.5 - constant vector variant");
    };

    // 4.3.7.6: assign: Constant Matrix Variant
    template<typename CMatrixT,
             typename MaskT,
             typename AccumT,
             typename ValueT,
             typename RowSequenceT,
             typename ColSequenceT,
             typename std::enable_if<
                 std::is_convertible<ValueT,
                                     typename CMatrixT::ScalarType>::value,
                 int>::type = 0>
    inline void assign(CMatrixT             &C,
                       MaskT          const &Mask,
                       AccumT                accum,
                       ValueT                val,
                       RowSequenceT   const &row_indices,
                       ColSequenceT   const &col_indices,
                       bool                  replace_flag = false)
    {
        GRB_LOG_FN_BEGIN("assign - 4.3.7.6 - constant matrix variant");
        GRB_LOG_VERBOSE("C in: " << C.m_mat);
        GRB_LOG_VERBOSE("Mask in: " << Mask.m_mat);
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE("val in: " << val);
        GRB_LOG_VERBOSE("row_indices in:" << row_indices);
        GRB_LOG_VERBOSE("col_indices in:" << col_indices);
        GRB_LOG_VERBOSE_REPLACE(replace_flag);

        check_nrows_nrows(C, Mask, "assign(const mat): C.nrows != Mask.nrows");
        check_ncols_ncols(C, Mask, "assign(const mat): C.ncols != Mask.ncols");

        check_nindices_within_nrows(
            row_indices, C,
            "assign(const mat): indicies.size !<= C.nrows");
        check_nindices_within_ncols(
            col_indices, C,
            "assign(const mat): indicies.size !<= C.ncols");
        backend::assign_constant(C.m_mat, Mask.m_mat, accum, val,
                                 row_indices, col_indices, replace_flag);

        GRB_LOG_VERBOSE("C out: " << C.m_mat);
        GRB_LOG_FN_END("assign - 4.3.7.6 - constant matrix variant");
    }

    //************************************************************************
    // Apply
    //************************************************************************

    // 4.3.8.1: vector variant
    template<typename WScalarT,
             typename MaskT,
             typename AccumT,
             typename UnaryFunctionT,
             typename UVectorT,
             typename ...WTagsT>
    inline void apply(Vector<WScalarT, WTagsT...> &w,
                      MaskT                 const &mask,
                      AccumT                       accum,
                      UnaryFunctionT               op,
                      UVectorT              const &u,
                      bool                         replace_flag = false)
    {
        check_size_size(w, mask, "apply(vec): w.size != mask.size");
        check_size_size(w, u, "apply(vec): w.size != u.size");

        backend::apply(w.m_vec, mask.m_vec, accum, op, u.m_vec, replace_flag);
    }

    // 4.3.8.2: matrix variant
    template<typename CScalarT,
             typename MaskT,
             typename AccumT,
             typename UnaryFunctionT,
             typename AMatrixT,
             typename ...ATagsT>
    inline void apply(Matrix<CScalarT, ATagsT...> &C,
                      MaskT                 const &Mask,
                      AccumT                       accum,
                      UnaryFunctionT               op,
                      AMatrixT              const &A,
                      bool                         replace_flag = false)
    {
        check_ncols_ncols(C, Mask, "apply(mat): C.ncols != Mask.ncols");
        check_nrows_nrows(C, Mask, "apply(mat): C.nrows != Mask.nrows");
        check_ncols_ncols(C, A, "apply(mat): C.ncols != A.ncols");
        check_nrows_nrows(C, A, "apply(mat): C.nrows != A.nrows");

        backend::apply(C.m_mat, Mask.m_mat, accum, op, A.m_mat, replace_flag);
    };

    //************************************************************************
    // reduce
    //************************************************************************

    // 4.3.9.1: reduce - Standard matrix to vector variant
    // matrix to column vector variant (row reduce, use transpose for col reduce)
    template<typename WVectorT,
             typename MaskT,
             typename AccumT,
             typename BinaryOpT,  // monoid or binary op only
             typename AMatrixT>
    inline void reduce(WVectorT        &w,
                       MaskT     const &mask,
                       AccumT           accum,
                       BinaryOpT        op,
                       AMatrixT  const &A,
                       bool             replace_flag = false)
    {
        check_size_size(w, mask, "reduce(mat2vec): w.size != mask.size");
        check_size_nrows(w, A, "reduce(mat2vec): w.size != A.nrows");

        backend::reduce(w.m_vec, mask.m_vec, accum, op, A.m_mat, replace_flag);
    }

    // 4.3.9.2: reduce - vector-scalar variant
    template<typename ValueT,
             typename AccumT,
             typename MonoidT, // monoid only
             typename UScalarT,
             typename ...UTagsT>
    inline void reduce(
            ValueT                            &val,
            AccumT                             accum,
            MonoidT                            op,
            Vector<UScalarT, UTagsT...> const &u)
    {
        backend::reduce_vector_to_scalar(val, accum, op, u.m_vec);
    }

    // 4.3.9.3: reduce - matrix-scalar variant
    /// @todo Won't support transpose of matrix here.
    template<typename ValueT,
             typename AccumT,
             typename MonoidT, // monoid only
             typename AScalarT,
             typename ...ATagsT>
    inline void reduce(
            ValueT                            &val,
            AccumT                             accum,
            MonoidT                            op,
            Matrix<AScalarT, ATagsT...> const &A)
    //  @TODO: We can probably support tranpose using the enable_if and tag
    //    typename std::enable_if<std::is_same<matrix_tag, typename AMatrixT::tag_type>::value, int>::type = 0>
    {
        backend::reduce_matrix_to_scalar(val, accum, op, A.m_mat);
    }

    //************************************************************************
    // Transpose
    //************************************************************************

    // 4.3.10: transpose
    template<typename CMatrixT,
             typename MaskT,
             typename AccumT,
             typename AMatrixT>
    inline void transpose(CMatrixT       &C,
                          MaskT    const &Mask,
                          AccumT          accum,
                          AMatrixT const &A,
                          bool            replace_flag = false)
    {
        check_nrows_nrows(C, Mask, "transpose: C.nrows != Mask.nrows");
        check_ncols_ncols(C, Mask, "transpose: C.ncols != Mask.ncols");
        check_ncols_nrows(C, A, "transpose: C.ncols != A.nrows");
        check_ncols_nrows(A, C, "transpose: A.ncols != C.nrows");

        backend::transpose(C.m_mat, Mask.m_mat, accum, A.m_mat, replace_flag);
    }

    //************************************************************************
    // Views
    //************************************************************************

    /**
     * @brief  "Flip" the rows and columns of a matrix
     * @param[in]  a  The matrix to transpose
     *
     */
    template<typename MatrixT>
    inline TransposeView<MatrixT> transpose(MatrixT const &A)
    {
        return TransposeView<MatrixT>(backend::transpose(A.m_mat));
    }

    /**
     * @brief  Return a view that complements the structure of a matrix.
     * @param[in]  a  The matrix to complement
     *
     */
    template<typename ScalarT, typename... TagsT>
    inline MatrixComplementView<Matrix<ScalarT, TagsT...>> complement(
        Matrix<ScalarT, TagsT...> const &Mask)
    {
        return MatrixComplementView<Matrix<ScalarT, TagsT...>>(
            backend::matrix_complement(Mask.m_mat));
    }

    template<typename ScalarT, typename... TagsT>
    inline VectorComplementView<Vector<ScalarT, TagsT...>> complement(
        Vector<ScalarT, TagsT...> const &mask)
    {
        return VectorComplementView<Vector<ScalarT, TagsT...>>(
            backend::vector_complement(mask.m_vec));
    }

} // GraphBLAS


#include <graphblas/detail/config.hpp>
#endif
