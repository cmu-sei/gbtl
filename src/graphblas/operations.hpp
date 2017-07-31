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

#include <graphblas/detail/config.hpp>

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
        GraphBLAS::check_nrows_nrows(C, A);
        GraphBLAS::check_ncols_nrows(A, B);
        GraphBLAS::check_ncols_ncols(C, B);
        GraphBLAS::check_nrows_nrows(C, Mask);
        GraphBLAS::check_ncols_ncols(C, Mask);

        backend::mxm(C.m_mat, Mask.m_mat, accum, op, A.m_mat, B.m_mat,
                     replace_flag);
    }

    //************************************************************************

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
        backend::mxv(w.m_vec, mask.m_vec, accum, op, A.m_mat, u.m_vec,
                     replace_flag);
    }

    //************************************************************************

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
        backend::vxm(w.m_vec, mask.m_vec, accum, op, u.m_vec, A.m_mat,
                     replace_flag);
    }

    //************************************************************************
    // eWiseAdd and eWiseMult
    //************************************************************************
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
    inline void eWiseMult(
        GraphBLAS::Vector<WScalarT, WTagsT...> &w,
        MaskT                            const &mask,
        AccumT                                  accum,
        BinaryOpT                               op,
        UVectorT                         const &u,
        VVectorT                         const &v,
        bool                                    replace_flag = false)
    {
        backend::eWiseMult(w.m_vec, mask.m_vec, accum, op, u.m_vec, v.m_vec,
                           replace_flag);
    }

    template<typename CScalarT,
             typename MaskT,
             typename AccumT,
             typename BinaryOpT,  //can be BinaryOp, Monoid (not Semiring)
             typename AMatrixT,
             typename BMatrixT,
             typename ...CTagsT>
    inline void eWiseMult(
        GraphBLAS::Matrix<CScalarT, CTagsT...> &C,
        MaskT                            const &Mask,
        AccumT                                  accum,
        BinaryOpT                               op,
        AMatrixT                         const &A,
        BMatrixT                         const &B,
        bool                                    replace_flag = false)
    {
        backend::eWiseMult(C.m_mat, Mask.m_mat, accum, op, A.m_mat, B.m_mat,
                           replace_flag);
    }

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
    inline void eWiseAdd(
        GraphBLAS::Vector<WScalarT, WTagsT...> &w,
        MaskT                            const &mask,
        AccumT                                  accum,
        BinaryOpT                               op,
        UVectorT                         const &u,
        VVectorT                         const &v,
        bool                                    replace_flag = false)
    {
        backend::eWiseAdd(w.m_vec, mask.m_vec, accum, op, u.m_vec, v.m_vec,
                          replace_flag);
    }

    template<typename CScalarT,
             typename MaskT,
             typename AccumT,
             typename BinaryOpT,  //can be BinaryOp, Monoid (not Semiring)
             typename AMatrixT,
             typename BMatrixT,
             typename ...CTagsT>
    inline void eWiseAdd(
        GraphBLAS::Matrix<CScalarT, CTagsT...> &C,
        MaskT                            const &Mask,
        AccumT                                  accum,
        BinaryOpT                               op,
        AMatrixT                         const &A,
        BMatrixT                         const &B,
        bool                                    replace_flag = false)
    {
        backend::eWiseAdd(C.m_mat, Mask.m_mat, accum, op, A.m_mat, B.m_mat,
                          replace_flag);
    }

    //************************************************************************
    // Extract
    //************************************************************************

    /// Extract: Standard vector variant
//    template<typename WVectorT,
//             typename MaskT,
//             typename AccumT,
//             typename UVectorT,
//             typename RAIteratorT>
//    inline void extract(WVectorT           &w,
//                        MaskT        const &mask,
//                        AccumT             accum,
//                        UVectorT    const &u,
//                        RAIteratorT        indices,
//                        IndexType          num_indices,
//                        bool               replace_flag = false);

    /// Standard Matrix version
//    template<typename CMatrixT,
//             typename MaskT,
//             typename AccumT,
//             typename AMatrixT,
//             typename RAIteratorI,
//             typename RAIteratorJ>
//    inline void extract(CMatrixT             &C,
//                        MaskT          const &Mask,
//                        AccumT                accum,
//                        AMatrixT       const &A,
//                        RAIteratorI           row_indices,
//                        IndexType             nrows,
//                        RAIteratorJ           col_indices,
//                        IndexType             ncols,
//                        bool                  replace_flag = false);

    // 4.3.6.1 - extract: Standard vector variant
    template<typename WVectorT,
             typename MaskT,
             typename AccumT,
             typename UVectorT>
    inline void extract(WVectorT             &w,
                        MaskT          const &mask,
                        AccumT                accum,
                        UVectorT       const &u,
                        IndexArrayType const &indices,
                        bool                  replace_flag = false)
    {
        backend::extract(w.m_vec, mask.m_vec, accum, u.m_vec,
                         indices, replace_flag);
    }

    // 4.3.6.2 - extract: Standard matrix variant
    template<typename CMatrixT,
             typename MaskT,
             typename AccumT,
             typename AMatrixT>
    inline void extract(CMatrixT             &C,
                        MaskT          const &Mask,
                        AccumT                accum,
                        AMatrixT       const &A,
                        IndexArrayType const &row_indices,
                        IndexArrayType const &col_indices,
                        bool                  replace_flag = false)
    {
        backend::extract(C.m_mat, Mask.m_mat, accum, A.m_mat,
                         row_indices, col_indices, replace_flag);
    }

    // 4.3.6.3 - extract: Column (and row) variant
    // Extract col (or row with transpose)
    template<typename WVectorT,
             typename MaskT,
             typename AccumT,
             typename AMatrixT>
    inline void extract(WVectorT             &w,
                        MaskT          const &mask,
                        AccumT                accum,
                        AMatrixT       const &A,
                        IndexArrayType const &row_indices,
                        IndexType             col_index,
                        bool                  replace_flag = false)
    {
        backend::extract(w.m_vec, mask.m_vec, accum, A.m_mat, row_indices,
                         col_index, replace_flag);
    }

    //************************************************************************
    // Assign
    //************************************************************************
//
//    // Standard Vector Variant
//    template<typename WVectorT,
//             typename MaskT,
//             typename AccumT,
//             typename UVectorT,
//             typename RAIteratorT>
//    inline void assign(WVectorT          &w,
//                       MaskT       const &mask,
//                       AccumT             accum,
//                       UVectorT    const &u,
//                       RAIteratorT const &indices,
//                       IndexType          num_indices,
//                       bool               replace_flag = false);
//
//    template<typename WVectorT,
//             typename MaskT,
//             typename AccumT,
//             typename UVectorT>
//    inline void assign(WVectorT             &w,
//                       MaskT          const &mask,
//                       AccumT                accum,
//                       UVectorT       const &u,
//                       IndexArrayType const &indices,
//                       bool                  replace_flag = false);
//

    // 4.3.7.2 assign: Standard matrix variant
    template<typename CMatrixT,
             typename MaskT,
             typename AccumT,
             typename AMatrixT>
    inline void assign(CMatrixT             &C,
                       MaskT          const &Mask,
                       AccumT                accum,
                       AMatrixT       const &A,
                       IndexArrayType const &row_indices,
                       IndexArrayType const &col_indices,
                       bool                  replace_flag = false)
    {
        backend::assign(C.m_mat, Mask.m_mat, accum, A.m_mat,
                        row_indices, col_indices, replace_flag);
    };

//
//    // Assign Column variant
//    template<typename CMatrixT,
//             typename MaskT,
//             typename AccumT,
//             typename UVectorT,
//             typename RAIteratorT>
//    inline void assign(CMatrixT          &C,
//                       MaskT       const &mask,  // a vector
//                       AccumT             accum,
//                       UVectorT    const &u,
//                       RAIteratorT        row_indices,
//                       IndexType          num_rows,
//                       IndexType          col_index,
//                       bool               replace_flag = false);
//
//    template<typename CMatrixT,
//             typename MaskT,
//             typename AccumT,
//             typename UVectorT>
//    inline void assign(CMatrixT             &C,
//                       MaskT          const &mask,  // a vector
//                       AccumT                accum,
//                       UVectorT       const &u,
//                       IndexArrayType const &row_indices,
//                       IndexType             col_index,
//                       bool                  replace_flag = false);
//
//    // Assign row variant
//    template<typename CMatrixT,
//             typename MaskT,
//             typename AccumT,
//             typename UVectorT,
//             typename RAIteratorT>
//    inline void assign(CMatrixT          &C,
//                       MaskT       const &mask,  // a vector
//                       AccumT             accum,
//                       UVectorT    const &u,
//                       IndexType          row_index,
//                       RAIteratorT        col_indices,
//                       IndexType          num_cols,
//                       bool               replace_flag = false);
//
//    template<typename CMatrixT,
//             typename MaskT,
//             typename AccumT,
//             typename UVectorT>
//    inline void assign(CMatrixT             &C,
//                       MaskT          const &mask,  // a vector
//                       AccumT                accum,
//                       UVectorT       const &u,
//                       IndexType             row_index,
//                       IndexArrayType const &col_indices,
//                       bool                  replace_flag = false);
//
//    // Single value variants (vector and matrix)
//    template<typename WVectorT,
//             typename AccumT,
//             typename ValueT>
//    inline void assign(WVectorT             &w,
//                       AccumT                accum,
//                       ValueT                src,
//                       IndexType             index,
//                       bool                  replace_flag = false);
//
//    template<typename CMatrixT,
//             typename AccumT,
//             typename ValueT>
//    inline void assign(CMatrixT             &C,
//                       AccumT                accum,
//                       ValueT                src,
//                       IndexType             row_index,
//                       IndexType             col_index,
//                       bool                  replace_flag = false);
//
//
//    // Vector constant variant
//    template<typename WVectorT,
//             typename MaskT,
//             typename AccumT,
//             typename ValueT,
//             typename RAIteratorT>
//    inline void assign(WVectorT          &w,
//                       MaskT       const &mask,
//                       AccumT             accum,
//                       ValueT             val,
//                       RAIteratorT const &indices,
//                       IndexType          num_indices,
//                       bool               replace_flag = false);
//

    // 4.3.7.5: assign: Constant vector variant
    template<typename WVectorT,
             typename MaskT,
             typename AccumT,
             typename ValueT>
    inline void assign_constant(WVectorT             &w,
                                MaskT          const &mask,
                                AccumT                accum,
                                ValueT                val,
                                IndexArrayType const &indices,
                                bool                  replace_flag = false)
    {
        backend::assign_constant(w.m_vec, mask.m_vec, accum, val, indices,
                                 replace_flag);
    };

    // 4.3.7.6: assign: Constant Matrix Variant
//    template<typename CMatrixT,
//             typename MaskT,
//             typename AccumT,
//             typename ValueT,
//             typename RAIteratorIT,
//             typename RAIteratorJT>
//    inline void assign(CMatrixT           &C,
//                       MaskT        const &Mask,
//                       AccumT              accum,
//                       ValueT              val,
//                       RAIteratorIT const &row_indices,
//                       IndexType           num_rows,
//                       RAIteratorJT const &col_indices,
//                       IndexType           num_cols,
//                       bool                replace_flag = false);


    // 4.3.7.6: assign: Constant Matrix Variant
    // @TODO;  Deal with overloads
    template<typename CMatrixT,
             typename MaskT,
             typename AccumT,
             typename ValueT>
    inline void assign_constant(CMatrixT             &C,
                                MaskT          const &Mask,
                                AccumT                accum,
                                ValueT                val,
                                IndexArrayType const &row_indices,
                                IndexArrayType const &col_indices,
                                bool                  replace_flag = false)
    {
        backend::assign_constant(C.m_mat, Mask.m_mat, accum, val,
                                 row_indices, col_indices, replace_flag);
    }

    //************************************************************************
    // Apply
    //************************************************************************

    // vector variant
    template<typename WScalarT,
             typename MaskT,
             typename AccumT,
             typename UnaryFunctionT,
             typename UVectorT,
             typename ...WTagsT>
    inline void apply(GraphBLAS::Vector<WScalarT, WTagsT...> &w,
                      MaskT                            const &mask,
                      AccumT                                  accum,
                      UnaryFunctionT                          op,
                      UVectorT                         const &u,
                      bool                                    replace_flag = false)
    {
        backend::apply(w.m_vec, mask.m_vec, accum, op, u.m_vec, replace_flag);
    }

    // matrix variant
    template<typename CScalarT,
             typename MaskT,
             typename AccumT,
             typename UnaryFunctionT,
             typename AMatrixT,
             typename ...ATagsT>
    inline void apply(GraphBLAS::Matrix<CScalarT, ATagsT...> &C,
                      MaskT                            const &Mask,
                      AccumT                                  accum,
                      UnaryFunctionT                          op,
                      AMatrixT                         const &A,
                      bool                                    replace_flag = false)
    {
        backend::apply(C.m_mat, Mask.m_mat, accum, op, A.m_mat, replace_flag);
    };

    //************************************************************************
    // reduce
    //************************************************************************

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
        backend::reduce(w.m_vec, mask.m_vec, accum, op, A.m_mat, replace_flag);
    }

    // vector-scalar variant
    template<typename ValueT,
             typename AccumT,
             typename MonoidT, // monoid only
             typename UScalarT,
             typename ...UTagsT>
    inline void reduce(
            ValueT                                       &val,
            AccumT                                        accum,
            MonoidT                                       op,
            GraphBLAS::Vector<UScalarT, UTagsT...> const &u)
    {
        backend::reduce_vector_to_scalar(val, accum, op, u.m_vec);
    }

    /// @todo Won't support transpose of matrix here.
    // matrix-scalar variant
    template<typename ValueT,
             typename AccumT,
             typename MonoidT, // monoid only
             typename AScalarT,
             typename ...ATagsT>
    inline void reduce(
            ValueT                                       &val,
            AccumT                                        accum,
            MonoidT                                       op,
            GraphBLAS::Matrix<AScalarT, ATagsT...> const &A)
    {
        backend::reduce_matrix_to_scalar(val, accum, op, A.m_mat);
    }

    //************************************************************************
    // Transpose
    //************************************************************************

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
