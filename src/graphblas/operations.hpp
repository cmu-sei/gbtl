/*
 * GraphBLAS Template Library (GBTL), Version 3.0
 *
 * Copyright 2020 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors.
 *
 * THIS MATERIAL WAS PREPARED AS AN ACCOUNT OF WORK SPONSORED BY AN AGENCY OF
 * THE UNITED STATES GOVERNMENT.  NEITHER THE UNITED STATES GOVERNMENT NOR THE
 * UNITED STATES DEPARTMENT OF ENERGY, NOR THE UNITED STATES DEPARTMENT OF
 * DEFENSE, NOR CARNEGIE MELLON UNIVERSITY, NOR BATTELLE, NOR ANY OF THEIR
 * EMPLOYEES, NOR ANY JURISDICTION OR ORGANIZATION THAT HAS COOPERATED IN THE
 * DEVELOPMENT OF THESE MATERIALS, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS,
 * OR USEFULNESS OR ANY INFORMATION, APPARATUS, PRODUCT, SOFTWARE, OR PROCESS
 * DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED
 * RIGHTS.
 *
 * Released under a BSD-style license, please see LICENSE file or contact
 * permission@sei.cmu.edu for full terms.
 *
 * [DISTRIBUTION STATEMENT A] This material has been approved for public release
 * and unlimited distribution.  Please see Copyright notice for non-US
 * Government use and distribution.
 *
 * DM20-0442
 */

#pragma once

#include <cstddef>
#include <vector>

#include <graphblas/types.hpp>
#include <graphblas/algebra.hpp>
#include <graphblas/TransposeView.hpp>
#include <graphblas/StructureView.hpp>
#include <graphblas/ComplementView.hpp>
#include <graphblas/StructuralComplementView.hpp>

#include <graphblas/Matrix.hpp>
#include <graphblas/Vector.hpp>
#include <graphblas/indices.hpp>

#include <graphblas/detail/logging.h>
#include <graphblas/detail/config.hpp>
#include <graphblas/detail/checks.hpp>

#define GB_INCLUDE_BACKEND_TRANSPOSE_VIEW 1
#define GB_INCLUDE_BACKEND_COMPLEMENT_VIEW 1
#define GB_INCLUDE_BACKEND_OPERATIONS 1
#include <backend_include.hpp>

//****************************************************************************
// New signatures to conform to GraphBLAS Specification
//****************************************************************************

namespace grb
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
                    AccumT     const &accum,
                    SemiringT         op,
                    AMatrixT   const &A,
                    BMatrixT   const &B,
                    OutputControlEnum outp = MERGE)
    {
        GRB_LOG_FN_BEGIN("mxm - 4.3.1 - matrix-matrix multiply");
        GRB_LOG_VERBOSE("C in :" << get_internal_matrix(C));
        GRB_LOG_VERBOSE("Mask in : " << get_internal_matrix(Mask));
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE_OP(op);
        GRB_LOG_VERBOSE("A in :" << get_internal_matrix(A));
        GRB_LOG_VERBOSE("B in :" << get_internal_matrix(B));
        GRB_LOG_VERBOSE_OUTP(outp);

        check_nrows_nrows(C, Mask, "mxm: C.nrows != Mask.nrows");
        check_ncols_ncols(C, Mask, "mxm: C.ncols != Mask.ncols");
        check_nrows_nrows(C, A, "mxm: C.nrows != A.nrows");
        check_ncols_ncols(C, B, "mxm: C.ncols != B.ncols");
        check_ncols_nrows(A, B, "mxm: A.ncols != B.nrows");

        backend::mxm(get_internal_matrix(C),
                     get_internal_matrix(Mask),
                     accum, op,
                     get_internal_matrix(A),
                     get_internal_matrix(B),
                     outp);

        GRB_LOG_VERBOSE("C (Result): " << get_internal_matrix(C));
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
                    AccumT     const &accum,
                    SemiringT         op,
                    UVectorT   const &u,
                    AMatrixT   const &A,
                    OutputControlEnum outp = MERGE)
    {
        GRB_LOG_FN_BEGIN("mxv - 4.3.2 - vector-matrix multiply");
        GRB_LOG_VERBOSE("w in :" << get_internal_vector(w));
        GRB_LOG_VERBOSE("mask in : " << get_internal_vector(mask));
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE_OP(op);
        GRB_LOG_VERBOSE("u in :" << get_internal_vector(u));
        GRB_LOG_VERBOSE("A in :" << get_internal_matrix(A));
        GRB_LOG_VERBOSE_OUTP(outp);

        check_size_size(w, mask, "vxm: w.size != mask.size");
        check_size_ncols(w, A, "vxm: w.size != A.ncols");
        check_size_nrows(u, A, "vxm: u.size != A.nrows");

        backend::vxm(get_internal_vector(w), get_internal_vector(mask), accum, op, get_internal_vector(u), get_internal_matrix(A), outp);

        GRB_LOG_VERBOSE("w out :" << get_internal_vector(w));
        GRB_LOG_FN_END("mxm - 4.3.2 - vector-matrix multiply");
    }

    //************************************************************************

    // 4.3.3: Matrix-vector multiply
    template<typename WVectorT,
             typename MaskT,
             typename AccumT,
             typename SemiringT,
             typename AMatrixT,
             typename UVectorT>
    inline void mxv(WVectorT          &w,
                    MaskT       const &mask,
                    AccumT      const &accum,
                    SemiringT          op,
                    AMatrixT    const &A,
                    UVectorT    const &u,
                    OutputControlEnum  outp = MERGE)
    {
        GRB_LOG_FN_BEGIN("mxv - 4.3.3 - matrix-vector multiply");
        GRB_LOG_VERBOSE("w in :" << get_internal_vector(w));
        GRB_LOG_VERBOSE("Mask in : " << get_internal_vector(mask));
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE_OP(op);
        GRB_LOG_VERBOSE("A in :" << get_internal_matrix(A));
        GRB_LOG_VERBOSE("u in :" << get_internal_vector(u));
        GRB_LOG_VERBOSE_OUTP(outp);

        check_size_size(w, mask, "mxv: w.size != mask.size");
        check_size_nrows(w, A, "mxv: w.size != A.nrows");
        check_size_ncols(u, A, "mxv: u.size != A.ncols");

        backend::mxv(get_internal_vector(w), get_internal_vector(mask), accum, op, get_internal_matrix(A), get_internal_vector(u), outp);
        GRB_LOG_VERBOSE("w out :" << get_internal_vector(w));
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
                          AccumT                const &accum,
                          BinaryOpT                    op,
                          UVectorT              const &u,
                          VVectorT              const &v,
                          OutputControlEnum            outp = MERGE)
    {
        GRB_LOG_FN_BEGIN("eWiseMult - 4.3.4.1 - element-wise vector multiply");
        GRB_LOG_VERBOSE("w in :" << get_internal_vector(w));
        GRB_LOG_VERBOSE("Mask in : " << get_internal_vector(mask));
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE_OP(op);
        GRB_LOG_VERBOSE("u in :" << get_internal_vector(u));
        GRB_LOG_VERBOSE("v in :" << get_internal_vector(v));
        GRB_LOG_VERBOSE_OUTP(outp);

        check_size_size(w, mask, "eWiseMult(vec): w.size != mask.size");
        check_size_size(w, u, "eWiseMult(vec): w.size != u.size");
        check_size_size(u, v, "eWiseMult(vec): u.size != v.size");

        backend::eWiseMult(get_internal_vector(w),
                           get_internal_vector(mask), accum, op, get_internal_vector(u), get_internal_vector(v),
                           outp);

        GRB_LOG_VERBOSE("w out :" << get_internal_vector(w));
        GRB_LOG_FN_END("eWiseMult - 4.3.4.1 - element-wise vector multiply");
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
                          AccumT                const &accum,
                          BinaryOpT                    op,
                          AMatrixT              const &A,
                          BMatrixT              const &B,
                          OutputControlEnum            outp = MERGE)
    {
        GRB_LOG_FN_BEGIN("eWiseMult - 4.3.4.2 - element-wise matrix multiply");
        GRB_LOG_VERBOSE("C in :" << get_internal_matrix(C));
        GRB_LOG_VERBOSE("Mask in : " << get_internal_matrix(Mask));
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE_OP(op);
        GRB_LOG_VERBOSE("A in :" << get_internal_matrix(A));
        GRB_LOG_VERBOSE("B in :" << get_internal_matrix(B));
        GRB_LOG_VERBOSE_OUTP(outp);

        check_ncols_ncols(C, Mask, "eWiseMult(mat): C.ncols != Mask.ncols");
        check_nrows_nrows(C, Mask, "eWiseMult(mat): C.nrows != Mask.nrows");
        check_ncols_ncols(C, A, "eWiseMult(mat): C.ncols != A.ncols");
        check_nrows_nrows(C, A, "eWiseMult(mat): C.nrows != A.nrows");
        check_ncols_ncols(A, B, "eWiseMult(mat): A.ncols != B.ncols");
        check_nrows_nrows(A, B, "eWiseMult(mat): A.nrows != B.nrows");

        backend::eWiseMult(get_internal_matrix(C),
                           get_internal_matrix(Mask),
                           accum, op,
                           get_internal_matrix(A),
                           get_internal_matrix(B),
                           outp);

        GRB_LOG_VERBOSE("C out :" << get_internal_matrix(C));
        GRB_LOG_FN_END("eWiseMult - 4.3.4.2 - element-wise matrix multiply");
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
                         AccumT                const &accum,
                         BinaryOpT                    op,
                         UVectorT              const &u,
                         VVectorT              const &v,
                         OutputControlEnum            outp = MERGE)
    {
        GRB_LOG_FN_BEGIN("eWiseAdd - 4.3.5.1 - element-wise vector addition");
        GRB_LOG_VERBOSE("w in :" << get_internal_vector(w));
        GRB_LOG_VERBOSE("Mask in : " << get_internal_vector(mask));
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE_OP(op);
        GRB_LOG_VERBOSE("u in :" << get_internal_vector(u));
        GRB_LOG_VERBOSE("v in :" << get_internal_vector(v));
        GRB_LOG_VERBOSE_OUTP(outp);

        check_size_size(w, mask, "eWiseAdd(vec): w.size != mask.size");
        check_size_size(w, u, "eWiseAdd(vec): w.size != u.size");
        check_size_size(u, v, "eWiseAdd(vec): u.size != v.size");

        backend::eWiseAdd(get_internal_vector(w),
                          get_internal_vector(mask),
                          accum, op,
                          get_internal_vector(u),
                          get_internal_vector(v),
                          outp);

        GRB_LOG_VERBOSE("w out :" << get_internal_vector(w));
        GRB_LOG_FN_END("eWiseAdd - 4.3.5.1 - element-wise vector addition");
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
                         AccumT                const &accum,
                         BinaryOpT                    op,
                         AMatrixT              const &A,
                         BMatrixT              const &B,
                         OutputControlEnum            outp = MERGE)
    {
        GRB_LOG_FN_BEGIN("eWiseAdd - 4.3.5.2 - element-wise matrix addition");
        GRB_LOG_VERBOSE("C in :" << get_internal_matrix(C));
        GRB_LOG_VERBOSE("Mask in : " << get_internal_matrix(Mask));
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE_OP(op);
        GRB_LOG_VERBOSE("A in :" << get_internal_matrix(A));
        GRB_LOG_VERBOSE("B in :" << get_internal_matrix(B));
        GRB_LOG_VERBOSE_OUTP(outp);

        check_ncols_ncols(C, Mask, "eWiseAdd(mat): C.ncols != Mask.ncols");
        check_nrows_nrows(C, Mask, "eWiseAdd(mat): C.nrows != Mask.nrows");
        check_ncols_ncols(C, A, "eWiseAdd(mat): C.ncols != A.ncols");
        check_nrows_nrows(C, A, "eWiseAdd(mat): C.nrows != A.nrows");
        check_ncols_ncols(A, B, "eWiseAdd(mat): A.ncols != B.ncols");
        check_nrows_nrows(A, B, "eWiseAdd(mat): A.nrows != B.nrows");

        backend::eWiseAdd(get_internal_matrix(C),
                          get_internal_matrix(Mask)
                          , accum, op,
                          get_internal_matrix(A),
                          get_internal_matrix(B),
                          outp);

        GRB_LOG_VERBOSE("C out :" << get_internal_matrix(C));
        GRB_LOG_FN_END("eWiseAdd - 4.3.5.2 - element-wise matrix addition");
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
                        AccumT         const &accum,
                        UVectorT       const &u,
                        SequenceT      const &indices,
                        OutputControlEnum     outp = MERGE)
    {
        GRB_LOG_FN_BEGIN("extract - 4.3.6.1 - standard vector variant");

        GRB_LOG_VERBOSE("w:    " << get_internal_vector(w));
        GRB_LOG_VERBOSE("mask: " << get_internal_vector(mask));
        GRB_LOG_VERBOSE("u:    " << get_internal_vector(u));
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE("indices: " << indices);
        GRB_LOG_VERBOSE_OUTP(outp);

        check_size_size(w, mask, "extract(std vec): w.size != mask.size");
        check_size_nindices(w, indices,
                            "extract(std vec): w.size != indicies.size");

        backend::extract(get_internal_vector(w),
                         get_internal_vector(mask),
                         accum,
                         get_internal_vector(u),
                         indices, outp);

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
                        AccumT            const &accum,
                        AMatrixT          const &A,
                        RowSequenceT      const &row_indices,
                        ColSequenceT      const &col_indices,
                        OutputControlEnum        outp = MERGE)
    {
        GRB_LOG_FN_BEGIN("extract - 4.3.6.2 - standard matrix variant");

        GRB_LOG_VERBOSE("C: " << get_internal_matrix(C));
        GRB_LOG_VERBOSE("Mask: " << get_internal_matrix(Mask));
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE("A: " << get_internal_matrix(A));
        GRB_LOG_VERBOSE("row_indices: " << row_indices);
        GRB_LOG_VERBOSE("col_indices: " << col_indices);
        GRB_LOG_VERBOSE_OUTP(outp);

        check_nrows_nrows(C, Mask, "extract(std mat): C.nrows != Mask.nrows");
        check_ncols_ncols(C, Mask, "extract(std mat): C.ncols != Mask.ncols");
        check_nrows_nindices(C, row_indices,
                             "extract(std mat): C.nrows != row_indices");
        check_ncols_nindices(C, col_indices,
                             "extract(std mat): C.ncols != col_indices");

        backend::extract(get_internal_matrix(C),
                         get_internal_matrix(Mask),
                         accum,
                         get_internal_matrix(A),
                         row_indices, col_indices, outp);

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
                        AccumT               const &accum,
                        AMatrixT             const &A,
                        SequenceT            const &row_indices,
                        IndexType                   col_index,
                        OutputControlEnum           outp = MERGE)
    {
        GRB_LOG_FN_BEGIN("extract - 4.3.6.3 - column (and row) variant");

        GRB_LOG_VERBOSE("w:    " << get_internal_vector(w));
        GRB_LOG_VERBOSE("mask: " << get_internal_vector(mask));
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE("A:    " << get_internal_matrix(A));
        GRB_LOG_VERBOSE("row_indices: " << row_indices);
        GRB_LOG_VERBOSE("col_index:   " << col_index);
        GRB_LOG_VERBOSE_OUTP(outp);

        check_size_size(w, mask, "extract(col): w.size != mask.size");
        check_size_nindices(w, row_indices,
                            "extract(col): w.size != row_indicies");
        check_index_within_ncols(col_index, A,
                                 "extract(col): col_index >= A.ncols");

        backend::extract(get_internal_vector(w),
                         get_internal_vector(mask),
                         accum,
                         get_internal_matrix(A),
                         row_indices,
                         col_index, outp);
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
             typename std::enable_if_t<is_vector_v<UVectorT>, int> = 0,
             typename ...WTags>
    inline void assign(Vector<WScalarT, WTags...>      &w,
                       MaskT                    const  &mask,
                       AccumT                   const  &accum,
                       UVectorT                 const  &u,
                       SequenceT                const  &indices,
                       OutputControlEnum                outp = MERGE)
    {
        GRB_LOG_FN_BEGIN("assign - 4.3.7.1 - standard vector variant");
        GRB_LOG_VERBOSE("w in: " << get_internal_vector(w));
        GRB_LOG_VERBOSE("mask in: " << get_internal_vector(mask));
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE("u in: " << get_internal_vector(u));
        GRB_LOG_VERBOSE("indicies in: " << indices);
        GRB_LOG_VERBOSE_OUTP(outp);

        check_size_size(w, mask, "assign(std vec): w.size != mask.size");
        check_size_nindices(u, indices,
                            "assign(std vec): u.size != |indicies|");

        backend::assign(get_internal_vector(w),
                        get_internal_vector(mask),
                        accum,
                        get_internal_vector(u),
                        indices, outp);

        GRB_LOG_VERBOSE("w out: " << get_internal_vector(w));
        GRB_LOG_FN_END("assign - 4.3.7.1 - standard vector variant");
    }

    // 4.3.7.2: assign - standard matrix variant
    template<typename CMatrixT,
             typename MaskT,
             typename AccumT,
             typename AMatrixT,
             typename RowSequenceT,
             typename ColSequenceT,
             typename std::enable_if_t<is_matrix_v<AMatrixT>, int> = 0>
    inline void assign(CMatrixT              &C,
                       MaskT           const &Mask,
                       AccumT          const &accum,
                       AMatrixT        const &A,
                       RowSequenceT    const &row_indices,
                       ColSequenceT    const &col_indices,
                       OutputControlEnum      outp = MERGE)
    {
        GRB_LOG_FN_BEGIN("assign - 4.3.7.2 - standard matrix variant");

        GRB_LOG_VERBOSE("C in: " << get_internal_matrix(C));
        GRB_LOG_VERBOSE("Mask in: " << get_internal_matrix(Mask));
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE("A in: " << get_internal_matrix(A));
        GRB_LOG_VERBOSE("row_indices in: " << row_indices);
        GRB_LOG_VERBOSE("col_indices in: " << col_indices);
        GRB_LOG_VERBOSE_OUTP(outp);

        check_nrows_nrows(C, Mask, "assign(std mat): C.nrows != Mask.nrows");
        check_ncols_ncols(C, Mask, "assign(std mat): C.ncols != Mask.ncols");
        check_nrows_nindices(A, row_indices,
                             "assign(std mat): A.nrows != |row_indices|");
        check_ncols_nindices(A, col_indices,
                             "assign(std mat): A.ncols != |col_indices|");

        backend::assign(get_internal_matrix(C),
                        get_internal_matrix(Mask),
                        accum,
                        get_internal_matrix(A),
                        row_indices, col_indices, outp);

        GRB_LOG_VERBOSE("C out: " << get_internal_matrix(C));
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
                       AccumT                const &accum,
                       UVectorT              const &u,
                       SequenceT             const &row_indices,
                       IndexType                    col_index,
                       OutputControlEnum            outp = MERGE)
    {
        GRB_LOG_FN_BEGIN("assign - 4.3.7.3 - column variant");

        GRB_LOG_VERBOSE("C in: " << get_internal_matrix(C));
        GRB_LOG_VERBOSE("mask in: " << get_internal_vector(mask));
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE("u in: " << get_internal_vector(u));
        GRB_LOG_VERBOSE("row_indices in:" << row_indices);
        GRB_LOG_VERBOSE("col_index in:" << col_index);
        GRB_LOG_VERBOSE_OUTP(outp);

        check_size_nrows(mask, C, "assign(col): C.nrows != mask.size");
        check_size_nindices(u, row_indices,
                            "assign(col): u.size != |row_indices|");
        check_index_within_ncols(col_index, C,
                                 "assign(col): col_index >= C.ncols");

        backend::assign(get_internal_matrix(C),
                        get_internal_vector(mask),
                        accum,
                        get_internal_vector(u),
                        row_indices, col_index, outp);

        GRB_LOG_VERBOSE("C out: " << get_internal_matrix(C));
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
                       AccumT                const &accum,
                       UVectorT              const &u,
                       IndexType                    row_index,
                       SequenceT             const &col_indices,
                       OutputControlEnum            outp = MERGE)
    {
        GRB_LOG_FN_BEGIN("assign - 4.3.7.4 - row variant");
        GRB_LOG_VERBOSE("C in: " << get_internal_matrix(C));
        GRB_LOG_VERBOSE("mask in: " << get_internal_vector(mask));
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE("u in: " << get_internal_vector(u));
        GRB_LOG_VERBOSE("row_index in:" << row_index);
        GRB_LOG_VERBOSE("col_indices in:" << col_indices);
        GRB_LOG_VERBOSE_OUTP(outp);

        check_size_ncols(mask, C, "assign(row): C.ncols != Mask.ncols");
        check_size_nindices(u, col_indices,
                            "assign(row): u.size != |col_indices|");
        check_index_within_nrows(row_index, C,
                                 "assign(col): row_index >= C.nrows");

        backend::assign(get_internal_matrix(C),
                        get_internal_vector(mask),
                        accum,
                        get_internal_vector(u),
                        row_index, col_indices, outp);

        GRB_LOG_VERBOSE("C out: " << get_internal_matrix(C));
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
                       AccumT               const   &accum,
                       ValueT                        val,
                       SequenceT            const   &indices,
                       OutputControlEnum             outp = MERGE)
    {
        GRB_LOG_FN_BEGIN("assign - 4.3.7.5 - constant vector variant");
        GRB_LOG_VERBOSE("w in: " << get_internal_vector(w));
        GRB_LOG_VERBOSE("Mask in: " << get_internal_vector(mask));
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE("val in: " << val);
        GRB_LOG_VERBOSE("indices in:" << indices);
        GRB_LOG_VERBOSE_OUTP(outp);

        check_size_size(w, mask, "assign(const vec): w.size != mask.size");
        check_nindices_within_size(indices, w,
                                   "assign(const vec): indicies.size !<= w.size");

        backend::assign_constant(get_internal_vector(w),
                                 get_internal_vector(mask),
                                 accum, val,
                                 indices, outp);

        GRB_LOG_VERBOSE("w out: " << get_internal_vector(w));
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
                       AccumT         const &accum,
                       ValueT                val,
                       RowSequenceT   const &row_indices,
                       ColSequenceT   const &col_indices,
                       OutputControlEnum     outp = MERGE)
    {
        GRB_LOG_FN_BEGIN("assign - 4.3.7.6 - constant matrix variant");
        GRB_LOG_VERBOSE("C in: " << get_internal_matrix(C));
        GRB_LOG_VERBOSE("Mask in: " << get_internal_matrix(Mask));
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE("val in: " << val);
        GRB_LOG_VERBOSE("row_indices in:" << row_indices);
        GRB_LOG_VERBOSE("col_indices in:" << col_indices);
        GRB_LOG_VERBOSE_OUTP(outp);

        check_nrows_nrows(C, Mask, "assign(const mat): C.nrows != Mask.nrows");
        check_ncols_ncols(C, Mask, "assign(const mat): C.ncols != Mask.ncols");

        check_nindices_within_nrows(
            row_indices, C,
            "assign(const mat): indicies.size !<= C.nrows");
        check_nindices_within_ncols(
            col_indices, C,
            "assign(const mat): indicies.size !<= C.ncols");
        backend::assign_constant(get_internal_matrix(C),
                                 get_internal_matrix(Mask),
                                 accum, val,
                                 row_indices, col_indices,
                                 outp);

        GRB_LOG_VERBOSE("C out: " << get_internal_matrix(C));
        GRB_LOG_FN_END("assign - 4.3.7.6 - constant matrix variant");
    }

    //************************************************************************
    // Apply
    //************************************************************************

    // 4.3.8.1: vector variant
    template<typename WScalarT,
             typename MaskT,
             typename AccumT,
             typename UnaryOpT,
             typename UVectorT,
             typename ...WTagsT>
    inline void apply(Vector<WScalarT, WTagsT...> &w,
                      MaskT                 const &mask,
                      AccumT                const &accum,
                      UnaryOpT                     op,
                      UVectorT              const &u,
                      OutputControlEnum            outp = MERGE)
    {
        GRB_LOG_FN_BEGIN("apply - 4.3.8.1 - vector variant");
        GRB_LOG_VERBOSE("w in: " << get_internal_vector(w));
        GRB_LOG_VERBOSE("mask in: " << get_internal_vector(mask));
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE_OP(op);
        GRB_LOG_VERBOSE("u in: " << get_internal_vector(u));
        GRB_LOG_VERBOSE_OUTP(outp);

        check_size_size(w, mask, "apply(vec): w.size != mask.size");
        check_size_size(w, u, "apply(vec): w.size != u.size");

        backend::apply(get_internal_vector(w),
                       get_internal_vector(mask),
                       accum, op,
                       get_internal_vector(u),
                       outp);

        GRB_LOG_VERBOSE("w out: " << get_internal_vector(w));
        GRB_LOG_FN_END("apply - 4.3.8.1 - vector variant");
    }

    // 4.3.8.2: matrix variant
    template<typename CScalarT,
             typename MaskT,
             typename AccumT,
             typename UnaryOpT,
             typename AMatrixT,
             typename ...CTagsT>
    inline void apply(Matrix<CScalarT, CTagsT...> &C,
                      MaskT                 const &Mask,
                      AccumT                const &accum,
                      UnaryOpT                     op,
                      AMatrixT              const &A,
                      OutputControlEnum            outp = MERGE)
    {

        GRB_LOG_FN_BEGIN("apply - 4.3.8.2 - matrix variant");
        GRB_LOG_VERBOSE("C in: " << get_internal_matrix(C));
        GRB_LOG_VERBOSE("Mask in: " << get_internal_matrix(Mask));
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE_OP(op);
        GRB_LOG_VERBOSE("A in: " << A);
        GRB_LOG_VERBOSE_OUTP(outp);

        check_ncols_ncols(C, Mask, "apply(mat): C.ncols != Mask.ncols");
        check_nrows_nrows(C, Mask, "apply(mat): C.nrows != Mask.nrows");
        check_ncols_ncols(C, A, "apply(mat): C.ncols != A.ncols");
        check_nrows_nrows(C, A, "apply(mat): C.nrows != A.nrows");

        backend::apply(get_internal_matrix(C),
                       get_internal_matrix(Mask),
                       accum, op,
                       get_internal_matrix(A),
                       outp);

        GRB_LOG_VERBOSE("C out: " << get_internal_matrix(C));
        GRB_LOG_FN_END("apply - 4.3.8.2 - matrix variant");
    }

    // 4.3.8.3: vector binaryop variants
    template<typename WScalarT,
             typename MaskT,
             typename AccumT,
             typename BinaryOpT,
             typename FirstT,
             typename SecondT,
             typename ...WTagsT>
    inline void apply(
        Vector<WScalarT, WTagsT...> &w,
        MaskT                 const &mask,
        AccumT                const &accum,
        BinaryOpT                    op,
        FirstT                const &lhs,
        SecondT               const &rhs,
        OutputControlEnum            outp = MERGE)
    {
        // figure out if the user wants bind1st or bind2nd based on the arg types
        constexpr bool is_bind1st = is_vector_v<SecondT>;
        constexpr bool is_bind2nd = is_vector_v<FirstT>;

        // make sure only one of the types matches
        static_assert(is_bind1st ^ is_bind2nd, "apply isn't going to work");

        if constexpr(is_bind1st) {
            GRB_LOG_FN_BEGIN("apply - 4.3.8.3 - vector binaryop bind1st variant");
            GRB_LOG_VERBOSE("w in: " << get_internal_vector(w));
            GRB_LOG_VERBOSE("mask in: " << get_internal_vector(mask));
            GRB_LOG_VERBOSE_ACCUM(accum);
            GRB_LOG_VERBOSE_OP(op);
            GRB_LOG_VERBOSE("val in: " << lhs);
            GRB_LOG_VERBOSE("u in: " << get_internal_vector(rhs));
            GRB_LOG_VERBOSE_OUTP(outp);

            check_size_size(w, mask, "apply(vec,binop): w.size != mask.size");
            check_size_size(w, rhs, "apply(vec,binop): w.size != u.size");

            backend::apply_binop_1st(get_internal_vector(w),
                                     get_internal_vector(mask),
                                     accum, op,
                                     lhs,
                                     get_internal_vector(rhs),
                                     outp);

            GRB_LOG_VERBOSE("w out: " << get_internal_vector(w));
            GRB_LOG_FN_END("apply - 4.3.8.3 - vector binaryop bind1st variant");
        }
        else {
            GRB_LOG_FN_BEGIN("apply - 4.3.8.3 - vector binaryop bind2nd variant");
            GRB_LOG_VERBOSE("w in: " << get_internal_vector(w));
            GRB_LOG_VERBOSE("mask in: " << get_internal_vector(mask));
            GRB_LOG_VERBOSE_ACCUM(accum);
            GRB_LOG_VERBOSE_OP(op);
            GRB_LOG_VERBOSE("u in: " << get_internal_vector(lhs));
            GRB_LOG_VERBOSE("val in: " << rhs);
            GRB_LOG_VERBOSE_OUTP(outp);

            check_size_size(w, mask, "apply(vec,binop): w.size != mask.size");
            check_size_size(w, lhs, "apply(vec,binop): w.size != u.size");

            backend::apply_binop_2nd(get_internal_vector(w),
                                     get_internal_vector(mask),
                                     accum, op,
                                     get_internal_vector(lhs),
                                     rhs,
                                     outp);

            GRB_LOG_VERBOSE("w out: " << get_internal_vector(w));
            GRB_LOG_FN_END("apply - 4.3.8.3 - vector binaryop bind2nd variant");
        }
    }

    // 4.3.8.4: matrix binaryop variants
    template<typename CScalarT,
             typename MaskT,
             typename AccumT,
             typename BinaryOpT,
             typename FirstT,
             typename SecondT,
             typename ...CTagsT>
    inline void apply(
        Matrix<CScalarT, CTagsT...> &C,
        MaskT                 const &Mask,
        AccumT                const &accum,
        BinaryOpT                    op,
        FirstT                const &lhs,
        SecondT               const &rhs,
        OutputControlEnum            outp = MERGE)
    {
        // figure out if the user wants bind1st or bind2nd based on the argument types
        constexpr bool is_bind1st = is_matrix_v<SecondT>;
        constexpr bool is_bind2nd = is_matrix_v<FirstT>;

        // make sure only one of the types matches
        static_assert(is_bind1st ^ is_bind2nd, "apply isn't going to work");

        if constexpr(is_bind1st) {
            GRB_LOG_FN_BEGIN("apply - 4.3.8.4 - matrix binaryop bind1st variant");
            GRB_LOG_VERBOSE("C in: " << get_internal_matrix(C));
            GRB_LOG_VERBOSE("Mask in: " << get_internal_matrix(Mask));
            GRB_LOG_VERBOSE_ACCUM(accum);
            GRB_LOG_VERBOSE_OP(op);
            GRB_LOG_VERBOSE("val in: " << lhs);
            GRB_LOG_VERBOSE("A in: " << get_internal_matrix(rhs));
            GRB_LOG_VERBOSE_OUTP(outp);

            check_ncols_ncols(C, Mask, "apply(mat,binop): C.ncols != Mask.ncols");
            check_nrows_nrows(C, Mask, "apply(mat,binop): C.nrows != Mask.nrows");
            check_ncols_ncols(C, rhs, "apply(mat,binop): C.ncols != A.ncols");
            check_nrows_nrows(C, rhs, "apply(mat,binop): C.nrows != A.nrows");

            backend::apply_binop_1st(get_internal_matrix(C),
                                     get_internal_matrix(Mask),
                                     accum, op,
                                     lhs,
                                     get_internal_matrix(rhs),
                                     outp);

            GRB_LOG_VERBOSE("C out: " << get_internal_matrix(C));
            GRB_LOG_FN_END("apply - 4.3.8.4 - matrix binaryop bind1st variant");
        }
        else
        {
            GRB_LOG_FN_BEGIN("apply - 4.3.8.4 - matrix binaryop bind2nd variant");
            GRB_LOG_VERBOSE("C in: " << get_internal_matrix(C));
            GRB_LOG_VERBOSE("Mask in: " << get_internal_matrix(Mask));
            GRB_LOG_VERBOSE_ACCUM(accum);
            GRB_LOG_VERBOSE_OP(op);
            GRB_LOG_VERBOSE("A in: " << get_internal_matrix(lhs));
            GRB_LOG_VERBOSE("val in: " << rhs);
            GRB_LOG_VERBOSE_OUTP(outp);

            check_ncols_ncols(C, Mask, "apply(mat,binop): C.ncols != Mask.ncols");
            check_nrows_nrows(C, Mask, "apply(mat,binop): C.nrows != Mask.nrows");
            check_ncols_ncols(C, lhs, "apply(mat,binop): C.ncols != A.ncols");
            check_nrows_nrows(C, lhs, "apply(mat,binop): C.nrows != A.nrows");

            backend::apply_binop_2nd(get_internal_matrix(C),
                                     get_internal_matrix(Mask),
                                     accum, op,
                                     get_internal_matrix(lhs),
                                     rhs,
                                     outp);

            GRB_LOG_VERBOSE("C out: " << get_internal_matrix(C));
            GRB_LOG_FN_END("apply - 4.3.8.4 - matrix binaryop bind2nd variant");
        }
    }

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
    inline void reduce(WVectorT          &w,
                       MaskT       const &mask,
                       AccumT      const &accum,
                       BinaryOpT          op,
                       AMatrixT    const &A,
                       OutputControlEnum  outp = MERGE)
    {
        GRB_LOG_FN_BEGIN("reduce - 4.3.9.1 - matrix to vector variant");
        GRB_LOG_VERBOSE("w in: " << get_internal_vector(w));
        GRB_LOG_VERBOSE("mask in: " << get_internal_vector(mask));
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE_OP(op);
        GRB_LOG_VERBOSE("A in: " << get_internal_matrix(A));
        GRB_LOG_VERBOSE_OUTP(outp);

        check_size_size(w, mask, "reduce(mat2vec): w.size != mask.size");
        check_size_nrows(w, A, "reduce(mat2vec): w.size != A.nrows");

        backend::reduce(get_internal_vector(w),
                        get_internal_vector(mask),
                        accum, op,
                        get_internal_matrix(A),
                        outp);

        GRB_LOG_VERBOSE("w out: " << get_internal_vector(w));
        GRB_LOG_FN_END("reduce - 4.3.9.1 - matrix to vector variant");
    }

    // 4.3.9.2: reduce - vector-scalar variant
    template<typename ValueT,
             typename AccumT,
             typename MonoidT, // monoid only
             typename UScalarT,
             typename ...UTagsT>
    inline void reduce(
            ValueT                            &val,
            AccumT                      const &accum,
            MonoidT                            op,
            Vector<UScalarT, UTagsT...> const &u)
    {
        GRB_LOG_FN_BEGIN("reduce - 4.3.9.2 - vector to scalar variant");
        GRB_LOG_VERBOSE("val in: " << val);
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE_OP(op);
        GRB_LOG_VERBOSE("u in: " << get_internal_vector(u));

        backend::reduce_vector_to_scalar(val,
                                         accum, op,
                                         get_internal_vector(u));

        GRB_LOG_VERBOSE("val out: " << val);
        GRB_LOG_FN_END("reduce - 4.3.9.2 - vector to scalar variant");
    }

    // 4.3.9.3: reduce - matrix-scalar variant
    /// @note We aren't supporting transpose of matrix here. The spec does not
    /// require support.
    template<typename ValueT,
             typename AccumT,
             typename MonoidT, // monoid only
             typename AScalarT,
             typename ...ATagsT>
    inline void reduce(
            ValueT                            &val,
            AccumT                      const &accum,
            MonoidT                            op,
            Matrix<AScalarT, ATagsT...> const &A)
    {
        GRB_LOG_FN_BEGIN("reduce - 4.3.9.3 - matrix to scalar variant");
        GRB_LOG_VERBOSE("val in: " << val);
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE_OP(op);
        GRB_LOG_VERBOSE("A in: " << get_internal_matrix(A));

        backend::reduce_matrix_to_scalar(val,
                                         accum, op,
                                         get_internal_matrix(A));

        GRB_LOG_VERBOSE("val out: " << val);
        GRB_LOG_FN_END("reduce - 4.3.9.3 - matrix to scalar variant");
    }

    //************************************************************************
    // Transpose
    //************************************************************************

    // 4.3.10: transpose
    template<typename CMatrixT,
             typename MaskT,
             typename AccumT,
             typename AMatrixT>
    inline void transpose(CMatrixT          &C,
                          MaskT       const &Mask,
                          AccumT      const &accum,
                          AMatrixT    const &A,
                          OutputControlEnum  outp = MERGE)
    {
        GRB_LOG_FN_BEGIN("transpose - 4.3.10");
        GRB_LOG_VERBOSE("C in: " << get_internal_matrix(C));
        GRB_LOG_VERBOSE("Mask in: " << get_internal_matrix(Mask));
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE("A in: " << get_internal_matrix(A));
        GRB_LOG_VERBOSE_OUTP(outp);

        check_nrows_nrows(C, Mask, "transpose: C.nrows != Mask.nrows");
        check_ncols_ncols(C, Mask, "transpose: C.ncols != Mask.ncols");
        check_ncols_nrows(C, A, "transpose: C.ncols != A.nrows");
        check_ncols_nrows(A, C, "transpose: A.ncols != C.nrows");

        backend::transpose(get_internal_matrix(C), get_internal_matrix(Mask), accum, get_internal_matrix(A), outp);

        GRB_LOG_VERBOSE("C out: " << get_internal_matrix(C));
        GRB_LOG_FN_END("transpose - 4.3.10");
    }

    //************************************************************************
    // Kronecker product
    //************************************************************************

    // 4.3.11: Kronecker product
    template<typename CMatrixT,
             typename MaskT,
             typename AccumT,
             typename BinaryOpT,
             typename AMatrixT,
             typename BMatrixT>
    inline void kronecker(CMatrixT          &C,
                          MaskT       const &Mask,
                          AccumT      const &accum,
                          BinaryOpT          op,
                          AMatrixT    const &A,
                          BMatrixT    const &B,
                          OutputControlEnum  outp = MERGE)
    {
        GRB_LOG_FN_BEGIN("kronecker - 4.3.11");
        GRB_LOG_VERBOSE("C in: " << get_internal_matrix(C));
        GRB_LOG_VERBOSE("Mask in: " << get_internal_matrix(Mask));
        GRB_LOG_VERBOSE_ACCUM(accum);
        GRB_LOG_VERBOSE_OP(op);
        GRB_LOG_VERBOSE("A in: " << get_internal_matrix(A));
        GRB_LOG_VERBOSE("B in: " << get_internal_matrix(B));
        GRB_LOG_VERBOSE_OUTP(outp);

        check_nrows_nrows(C, Mask, "kronecker: C.nrows != Mask.nrows");
        check_ncols_ncols(C, Mask, "kronecker: C.ncols != Mask.ncols");
        check_ncols_ncolsxncols(C, A, B,
                                "kronecker: C.ncols != A.ncols*B.ncols");
        check_nrows_nrowsxnrows(C, A, B,
                                "kronecker: C.nrows != A.nrows*B.nrows");

        backend::kronecker(get_internal_matrix(C),
                           get_internal_matrix(Mask),
                           accum, op,
                           get_internal_matrix(A),
                           get_internal_matrix(B),
                           outp);

        GRB_LOG_VERBOSE("C out: " << get_internal_matrix(C));
        GRB_LOG_FN_END("kronecker - 4.3.11");
    }

    //************************************************************************
    // Context etc.
    //************************************************************************
    void init() {}

    template <typename T>
    void wait(T&& obj) {}

    //************************************************************************
    // Views
    //************************************************************************

    // /**
    //  * @brief  "Flip" the rows and columns of a matrix
    //  * @param[in]  A  The matrix to transpose
    //  *
    //  */
    // template<typename MatrixT>
    // inline TransposeView<MatrixT> transpose(MatrixT const &A)
    // {
    //     return TransposeView<MatrixT>(backend::transpose(get_internal_matrix(A)));
    // }

    //************************************************************************
    /**
     * @brief  "Flip" the rows and columns of a matrix
     * @param[in]  A  The matrix to transpose
     *
     */
    template<typename MatrixT,
             typename std::enable_if_t<is_matrix_v<MatrixT>, int> = 0>
    inline TransposeView<MatrixT> transpose(MatrixT const &A)
    {
        return TransposeView<MatrixT>(A);
    }

    //************************************************************************
    /**
     * @brief  Return a view that uses only the structure of a matrix mask.
     * @param[in]  Mask  The matrix to take the structure of
     *
     */
    template<typename MatrixT,
             typename std::enable_if_t<is_matrix_v<MatrixT>, int> = 0>
    MatrixStructureView<MatrixT> structure(MatrixT const &Mask)
    {
        return MatrixStructureView<MatrixT>(Mask);
    }

    /**
     * @brief  Return a view that complements a matrix mask.
     * @param[in]  Mask  The matrix to complement
     *
     */
    template<typename MatrixT,
             typename std::enable_if_t<is_matrix_v<MatrixT>, int> = 0>
    MatrixComplementView<MatrixT> complement(MatrixT const &Mask)
    {
        return MatrixComplementView<MatrixT>(Mask);
    }

    /**
     * @brief  Return a view that complements the structure of a matrix mask.
     * @param[in]  structure_view  The matrix structure view to complement
     *
     */
    template<class MatrixT,
             typename std::enable_if_t<is_matrix_v<MatrixT>, int> = 0>
    MatrixStructuralComplementView<MatrixT> complement(
        MatrixStructureView<MatrixT> const &structure_view)
    {
        return MatrixStructuralComplementView<MatrixT>(
            structure_view.m_mat);
    }


    //************************************************************************
    /**
     * @brief  Return a view that uses only the structure of a vector mask mask.
     * @param[in]  mask  The vector to take the structure of
     *
     */
    template<class VectorT,
             typename std::enable_if_t<is_vector_v<VectorT>, int> = 0>
    VectorStructureView<VectorT> structure(VectorT const &mask)
    {
        return VectorStructureView<VectorT>(mask);
    }

    /**
     * @brief  Return a view that complements a vector mask.
     * @param[in]  mask  The vector to take the complement of
     *
     */
    template<class VectorT,
             typename std::enable_if_t<is_vector_v<VectorT>, int> = 0>
    VectorComplementView<VectorT> complement(VectorT const &mask)
    {
        return VectorComplementView<VectorT>(mask);
    }

    /**
     * @brief  Return a view that complements the structure of a vector mask.
     * @param[in]  structure_view  The vector structure view to complement
     *
     */
    template<class VectorT,
             typename std::enable_if_t<is_vector_v<VectorT>, int> = 0>
    VectorStructuralComplementView<VectorT> complement(
        VectorStructureView<VectorT> const &structure_view)
    {
        return VectorStructuralComplementView<VectorT>(
            structure_view.m_vec);
    }

} // grb


//****************************************************************************
#include <graphblas/detail/config.hpp>
//****************************************************************************
