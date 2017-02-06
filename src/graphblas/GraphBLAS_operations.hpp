/*
 * Copyright (c) 2017 Carnegie Mellon University
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

#ifndef GRAPHBLAS_OPERATIONS_HPP
#define GRAPHBLAS_OPERATIONS_HPP

#pragma once

#include <cstddef>
#include <vector>

#include <graphblas/accum.hpp>
#include <graphblas/algebra.hpp>
#include <graphblas/View.hpp>
#include <graphblas/utility.hpp>

#include <graphblas/detail/config.hpp>

#define __GB_SYSTEM_HEADER <graphblas/system/__GB_SYSTEM_ROOT/TransposeView.hpp>
#include __GB_SYSTEM_HEADER
#undef __GB_SYSTEM_HEADER
#define __GB_SYSTEM_HEADER <graphblas/system/__GB_SYSTEM_ROOT/NegateView.hpp>
#include __GB_SYSTEM_HEADER
#undef __GB_SYSTEM_HEADER

#define __GB_SYSTEM_HEADER <graphblas/system/__GB_SYSTEM_ROOT/operations.hpp>
#include __GB_SYSTEM_HEADER
#undef __GB_SYSTEM_HEADER

namespace GraphBLAS
{
    /**
     * @brief Populate a Matrix with stored values at specified locations from a
     *        collection of tuples
     */
    template<typename CMatrixT,
             typename MaskT,
             typename AccumT,
             typename RAIteratorIT,
             typename RAIteratorJT,
             typename RAIteratorVT,
             typename BinaryOpT>
    inline void matrixBuild(CMatrixT           &C,
                            MaskT        const &Mask,
                            AccumT              accum,
                            RAIteratorIT        row_indices,
                            RAIteratorJT        col_indices,
                            RAIteratorVT        values,
                            IndexType           num_vals,
                            BinaryOpT           dup,
                            bool                replace_flag = false);


    template<typename CMatrixT,
             typename MaskT,
             typename AccumT,
             typename ValueT,
             typename BinaryOpT> >
    inline void matrixBuild(CMatrixT                   &C,
                            MaskT                const &Mask,
                            AccumT                      accum,
                            IndexArrayType       const &row_indices,
                            IndexArrayType       const &col_indices,
                            std::vector<ValueT>  const &values,
                            BinaryOpT                   dup,
                            bool                        replace_flag = false);

    /**
     * @brief Populate a Vector with stored values at specified locations from a
     *        collection of tuples
     */
    template<typename WVectorT,
             typename MaskT,
             typename AccumT,
             typename RAIteratorIT,
             typename RAIteratorVT,
             typename BinaryOpT>
    inline void vectorBuild(WVectorT           &w,
                            MaskT        const &mask,
                            AccumT              accum,
                            RAIteratorIT        indices,
                            RAIteratorVT        values,
                            IndexType           numVals,
                            BinaryOpT           dup,
                            bool                replace_flag = false);

    template<typename WVectorT,
             typename MaskT,
             typename AccumT,
             typename ValueT,
             typename BinaryOpT> >
    inline void vectorBuild(WVectorT                   &w,
                            MaskT                const &mask,
                            AccumT                      accum = AccumT(),
                            IndexArrayType       const &indices,
                            std::vector<ValueT>  const &values,
                            BinaryOpT                   dup,
                            bool                        replace_flag = false);

    /**
     * @brief Output (row, col, value) tuples from Matrix as three vectors.
     */
    template<typename RAIteratorIT,
             typename RAIteratorJT,
             typename RAIteratorVT,
             typename AMatrixT,
             typename MaskT>
    inline void matrixExtract(RAIteratorIT        row_indices,
                              RAIteratorJT        col_indices,
                              RAIteratorVT        values,
                              AMatrixT     const &A,
                              MaskT        const &Mask,
                              std::string        &err);

    template<typename ValueT,
             typename AMatrixT,
             typename MaskT>
    inline void matrixExtract(IndexArrayType            &row_indices,
                              IndexArrayType            &col_indices,
                              std::vector<ValueT>       &values,
                              AMatrixT            const &A,
                              MaskT               const &Mask,
                              std::string               &err);

    /**
     * @brief Output (index, value) tuples from Vector as three vectors.
     */
    template<typename RAIteratorIT,
             typename RAIteratorVT,
             typename WVectorT,
             typename MaskT>
    inline void vectorExtract(RAIteratorIT        indices,
                              RAIteratorVT        values,
                              WVectorT     const &w,
                              MaskT        const &mask,
                              std::string        &err);

    template<typename ValueT,
             typename WVectorT,
             typename MaskT>
    inline void vectorExtract(IndexArrayType            &indices,
                              std::vector<ValueT>       &values,
                              WVectorT            const &w,
                              MaskT               cons  &mask,
                              std::string               &err);

    //****************************************************************************
    // mxm, vxm, mxv
    //****************************************************************************

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
                    bool              replace_flag);

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
                    bool              replace_flag);

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
                    bool             replace_flag);

    //****************************************************************************
    // eWiseAdd and eWiseMult
    //****************************************************************************
    template<typename WVectorT,
             typename MaskT,
             typename AccumT,
             typename BinaryOpT,  //can be monoid or binaryop (Semiring not supported)
             typename UVectorT,
             typename VVectorT>
    inline void eWiseMult(WVectorT        &w,
                          MaskT     const &mask
                          AccumT           accum,
                          BinaryOpT        op,
                          UVectorT  const &u,
                          VVectorT  const &v,
                          bool             replace_flag);

    /// @todo no way to distinguish between vector and matrix variants
    template<typename CMatrixT,
             typename MaskT,
             typename AccumT,
             typename BinaryOpT,  //can be monoid or binaryop (Semiring not supported)
             typename AMatrixT,
             typename BMatrixT>
    inline void eWiseMult(CMatrixT         &C,
                          MaskT      const &Mask
                          AccumT            accum,
                          BinaryOpT         op,
                          AMatrixT   const &A,
                          BMatrixT   const &B,
                          bool              replace_flag);

    /**
     * @brief Perform an element wise binary operation that can be optimized
     *        for "add" semantics (OR short circuit logic).
     */
    template<typename WVectorT,
             typename MaskT,
             typename AccumT,
             typename BinaryOpT,  //can be monoid or binaryop (Semiring not supported)
             typename UVectorT,
             typename VVectorT>
    inline void eWiseAdd(WVectorT         &w,
                         MaskT      const &mask
                         AccumT            accum,
                         BinaryOpT         op,
                         UVectorT   const &u,
                         VVectorT   const &v,
                         bool              replace_flag);

    /// @todo no way to distinguish between vector and matrix variants
    template<typename CMatrixT,
             typename MaskT,
             typename AccumT,
             typename BinaryOpT,
             typename AMatrixT,
             typename BMatrixT >
    inline void eWiseAdd(CMatrixT         &C,
                         MaskT      const &Mask,
                         AccumT            accum,
                         BinaryOpT         op,
                         AMatrixT   const &A,
                         BMatrixT   const &B,
                         bool              replace_flag);

    //****************************************************************************
    // Extract
    //****************************************************************************

    /// Extract: Standard vector variant
    template<typename WVectorT,
             typename MaskT,
             typename AccumT,
             typename UVectorT,
             typename RAIteratorT>
    inline void extract(WVectorT           &w,
                        MaskT        const &mask,
                        AccumT             accum,
                        UVectorT    const &u,
                        RAIteratorT        indices,
                        IndexType          num_indices,
                        bool               replace_flag = false);

    template<typename WVectorT,
             typename MaskT,
             typename AccumT,
             typename UVectorT>
    inline void extract(WVectorT             &w,
                        MaskT          const &mask,
                        AccumT                accum,
                        UVectorT       const &u,
                        IndexArrayType const &indices,
                        bool                  replace_flag = false);

    /// Standard Matrix version
    template<typename CMatrixT,
             typename MaskT,
             typename AccumT,
             typename AMatrixT,
             typename RAIteratorI,
             typename RAIteratorJ>
    inline void extract(CMatrixT             &C,
                        MaskT          const &Mask,
                        AccumT                accum,
                        AMatrixT       const &A,
                        RAIteratorI           row_indices,
                        IndexType             nrows,
                        RAIteratorJ           col_indices,
                        IndexType             ncols,
                        bool                  replace_flag = false);

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
                        bool                  replace_flag = false);

    // Extract col (or row with transpose)
    template<typename WVectorT,
             typename MaskT,
             typename AccumT,
             typename AMatrixT,
             typename RAIteratorI>
    inline void extract(WVectorT             &w,
                        MaskT          const &mask,
                        AccumT                accum,
                        AMatrixT       const &A,
                        RAIteratorI           row_indices,
                        IndexType             nrows,
                        IndexType             col_index,
                        bool                  replace_flag = false);

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
                        bool                  replace_flag = false);

    // Extract single element (vector and matrix variants
    template <typename ValueT,
              typename AccumT,
              typename UVectorT>
    inline void extract(ValueT            &dst,
                        AccumT             accum,
                        UVectorT    const &u,
                        IndexType          index,
                        std::string       &err);

    template <typename ValueT,
              typename AccumT,
              typename AMatrixT>
    inline void extract(ValueT            &dst,
                        AccumT             accum,
                        AMatrixT    const &A,
                        IndexType          row_index,
                        IndexType          col_index,
                        std::string       &err);

    //****************************************************************************
    // Assign
    //****************************************************************************

    // Standard Vector Variant
    template<typename WVectorT,
             typename MaskT,
             typename AccumT,
             typename UVectorT,
             typename RAIteratorT>
    inline void assign(WVectorT          &w,
                       MaskT       const &mask,
                       AccumT             accum,
                       UVectorT    const &u,
                       RAIteratorT const &indices,
                       IndexType          num_indices,
                       bool               replace_flag = false);

    template<typename WVectorT,
             typename MaskT,
             typename AccumT,
             typename UVectorT>
    inline void assign(WVectorT             &w,
                       MaskT          const &mask,
                       AccumT                accum,
                       UVectorT       const &u,
                       IndexArrayType const &indices,
                       bool                  replace_flag = false);

    // Standard Matrix variant
    template<typename CMatrixT,
             typename MaskT,
             typename AccumT,
             typename AMatrixT,
             typename RAIteratorI,
             typename RAIteratorJ>
    inline void assign(CMatrixT          &C,
                       MaskT       const &Mask,
                       AccumT             accum,
                       AMatrixT    const &A,
                       RAIteratorI        row_indices,
                       IndexType          num_rows,
                       RAIteratorJ        col_indices,
                       IndexType          num_cols,
                       bool               replace_flag = false);

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
                       bool                  replace_flag = false);

    // Assign Column variant
    template<typename CMatrixT,
             typename MaskT,
             typename AccumT,
             typename UVectorT,
             typename RAIteratorT>
    inline void assign(CMatrixT          &C,
                       MaskT       const &mask,  // a vector
                       AccumT             accum,
                       UVectorT    const &u,
                       RAIteratorT        row_indices,
                       IndexType          num_rows,
                       IndexType          col_index,
                       bool               replace_flag = false);

    template<typename CMatrixT,
             typename MaskT,
             typename AccumT,
             typename UVectorT>
    inline void assign(CMatrixT             &C,
                       MaskT          const &mask,  // a vector
                       AccumT                accum,
                       UVectorT       const &u,
                       IndexArrayType const &row_indices,
                       IndexType             col_index,
                       bool                  replace_flag = false);

    // Assign row variant
    template<typename CMatrixT,
             typename MaskT,
             typename AccumT,
             typename UVectorT,
             typename RAIteratorT>
    inline void assign(CMatrixT          &C,
                       MaskT       const &mask,  // a vector
                       AccumT             accum,
                       UVectorT    const &u,
                       IndexType          row_index,
                       RAIteratorT        col_indices,
                       IndexType          num_cols,
                       bool               replace_flag = false);

    template<typename CMatrixT,
             typename MaskT,
             typename AccumT,
             typename UVectorT>
    inline void assign(CMatrixT             &C,
                       MaskT          const &mask,  // a vector
                       AccumT                accum,
                       UVectorT       const &u,
                       IndexType             row_index,
                       IndexArrayType const &col_indices,
                       bool                  replace_flag = false);

    // Single value variants (vector and matrix)
    template<typename WVectorT,
             typename AccumT,
             typename ValueT>
    inline void assign(WVectorT             &w,
                       AccumT                accum,
                       ValueT                src,
                       IndexType             index,
                       bool                  replace_flag = false);

    template<typename CMatrixT,
             typename AccumT,
             typename ValueT>
    inline void assign(CMatrixT             &C,
                       AccumT                accum,
                       ValueT                src,
                       IndexType             row_index,
                       IndexType             col_index,
                       bool                  replace_flag = false);


    // Vector constant variant
    template<typename WVectorT,
             typename MaskT,
             typename AccumT,
             typename ValueT,
             typename RAIteratorT>
    inline void assign(WVectorT          &w,
                       MaskT       const &mask,
                       AccumT             accum,
                       ValueT             val,
                       RAIteratorT const &indices,
                       IndexType          num_indices,
                       bool               replace_flag = false);

    template<typename WVectorT,
             typename MaskT,
             typename AccumT,
             typename ValueT>
    inline void assign(WVectorT             &w,
                       MaskT          const &mask,
                       AccumT                accum,
                       ValueT                val,
                       IndexArrayType const &indices,
                       bool                  replace_flag = false);

    // Matrix constant variant
    template<typename CMatrixT,
             typename MaskT,
             typename AccumT,
             typename ValueT,
             typename RAIteratorIT,
             typename RAIteratorJT>
    inline void assign(CMatrixT           &C,
                       MaskT        const &Mask,
                       AccumT              accum,
                       ValueT              val,
                       RAIteratorIT const &row_indices,
                       IndexType           num_rows,
                       RAIteratorJT const &col_indices,
                       IndexType           num_cols,
                       bool                replace_flag = false);


    template<typename CMatrixT,
             typename MaskT,
             typename AccumT,
             typename ValueT>
    inline void assign(CMatrixT             &C,
                       MaskT          const &Mask,
                       AccumT                accum,
                       ValueT                val,
                       IndexArrayType const &row_indices,
                       IndexArrayType const &col_indices,
                       bool                  replace_flag = false);

    //****************************************************************************
    // Apply
    //****************************************************************************

    // vector variant
    template<typename WVectorT,
             typename MaskT,
             typename AccumT,
             typename UnaryFunctionT,
             typename UVectorT>
    inline void apply(WVectorT             &w,
                      MaskT          const &mask,
                      AccumT                accum,
                      UnaryFunctionT        op,
                      UVectorT       const &u,
                      bool                  replace_flag = false);

    // matrix variant
    template<typename CMatrixT,
             typename MaskT,
             typename AccumT,
             typename UnaryFunctionT,
             typename AMatrixT>
    inline void apply(CMatrixT             &C,
                      MaskT          const &Mask,
                      AccumT                accum,
                      UnaryFunctionT        op,
                      AMatrixT       const &A,
                      bool                  replace_flag = false);

    //****************************************************************************
    // reduce
    //****************************************************************************

    // matrix to column vector variant (row reduce, use transpose for col reduce)
    template<typename WVectorT,
             typename MaskT,
             typename AccumT,
             typename BinaryFunctionT,  // monoid or binary op only
             typename AMatrixT>
    inline void reduce(WVectorT              &u,
                       MaskT           const &mask,
                       AccumT                 accum,
                       BinaryFunctionT        op,
                       AMatrixT        const &A,
                       bool                   replace_flag = false);

    // vector-scalar variant
    template<typename ValueT,
             typename AccumT,
             typename MonoidT, // monoid only
             typename UVectorT>
    inline void reduce(ValueT                &dst,
                       AccumT                 accum,
                       BinaryFunctionT        op,
                       UVectorT        const &u,
                       bool                   replace_flag = false);

    // matrix-scalar variant
    template<typename ValueT,
             typename AccumT,
             typename MonoidT, // monoid only
             typename AMatrixT>
    inline void reduce(ValueT                &dst,
                       AccumT                 accum,
                       BinaryFunctionT        op,
                       AMatrixT        const &A,
                       bool                   replace_flag = false);

    //****************************************************************************
    // Transpose
    //****************************************************************************

    template<typename CMatrixT,
             typename MaskT,
             typename AccumT,
             typename AMatrixT>
    inline void transpose(CMatrixT       &C,
                          MaskT    const &Mask,
                          AccumT          accum
                          AMatrixT const &A,
                          bool            replace_flag);


    //****************************************************************************
    //****************************************************************************


    /**
     * @brief  Return a view that structurally negates the elements of a matrix.
     * @param[in]  a  The matrix to negate
     */
    template<typename MatrixT,
             typename SemiringT =
                 graphblas::ArithmeticSemiring<typename MatrixT::ScalarType> >
    inline NegateView<MatrixT, SemiringT> negate(
        MatrixT const   &a,
        SemiringT const &s = SemiringT())
    {
        return NegateView<MatrixT, SemiringT>(backend::negate(a.m_mat, s));
    }

    /**
     * @brief  "Flip" the rows and columns of a matrix
     * @param[in]  a  The matrix to flip
    */
    template<typename AMatrixT>
    inline TransposeView<AMatrixT> transpose(AMatrixT const &a)
    {
        return TransposeView<AMatrixT>(backend::transpose(a.m_mat));
    }

} // graphblas

#include <graphblas/detail/config.hpp>
//#define __GB_SYSTEM_OPERATIONS_HEADER <graphblas/system/__GB_SYSTEM_ROOT/operations.hpp>
//#include __GB_SYSTEM_OPERATIONS_HEADER
//#undef __GB_SYSTEM_OPERATIONS_HEADER

#endif
