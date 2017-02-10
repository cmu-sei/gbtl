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

//****************************************************************************
// New signatures to conform to GraphBLAS Specification
//****************************************************************************

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
             typename BinaryOpT>
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
             typename BinaryOpT>
    inline void vectorBuild(WVectorT                   &w,
                            MaskT                const &mask,
                            AccumT                      accum,
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
                              MaskT               const &mask,
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
                          MaskT     const &mask,
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
                          MaskT      const &Mask,
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
                         MaskT      const &mask,
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
    inline void reduce(ValueT           &dst,
                       AccumT            accum,
                       MonoidT           op,
                       UVectorT   const &u,
                       bool              replace_flag = false);

    // matrix-scalar variant
    template<typename ValueT,
             typename AccumT,
             typename MonoidT, // monoid only
             typename AMatrixT>
    inline void reduce(ValueT           &dst,
                       AccumT            accum,
                       MonoidT           op,
                       AMatrixT   const &A,
                       bool              replace_flag = false);

    //****************************************************************************
    // Transpose
    //****************************************************************************

    template<typename CMatrixT,
             typename MaskT,
             typename AccumT,
             typename AMatrixT>
    inline void transpose(CMatrixT       &C,
                          MaskT    const &Mask,
                          AccumT          accum,
                          AMatrixT const &A,
                          bool            replace_flag);


    //****************************************************************************
    //****************************************************************************


    /**
     * @brief  Return a view that structurally negates the elements of a matrix.
     * @param[in]  a  The matrix to negate
     *
     * @todo MOVE NegateView to GraphBLAS namespace
     */
    template<typename MatrixT,
             typename SemiringT =
                 graphblas::ArithmeticSemiring<typename MatrixT::ScalarType> >
    inline graphblas::backend::NegateView<MatrixT, SemiringT> negate(
        MatrixT const   &a,
        SemiringT const &s = SemiringT())
    {
        return graphblas::backend::NegateView<MatrixT, SemiringT>(graphblas::backend::negate(a.m_mat, s));
    }

    /**
     * @brief  "Flip" the rows and columns of a matrix
     * @param[in]  a  The matrix to flip
     *
     * @todo MOVE TransposeView to GraphBLAS namespace
    */
    template<typename AMatrixT>
    inline graphblas::backend::TransposeView<AMatrixT> transpose(AMatrixT const &a)
    {
        return graphblas::backend::TransposeView<AMatrixT>(graphblas::backend::transpose(a.m_mat));
    }

} // GraphBLAS


//****************************************************************************
// The following signatures (and namespace) are deprecated
//****************************************************************************

/// @deprecated
namespace graphblas
{
    /**
     * @brief Perform an element wise binary operation that can be optimized
     *        for "add" semantics (OR short circuit logic).
     *
     * If one element of the Monoid (binary) operation is a structural zero
     * then the result is the other element.  If they are both stored values
     * then the result is the result of the specified Monoid.
     *
     * @tparam <MatrixT>  Models the Matrix concept
     * @tparam <MonoidT>  Models a Monoid (zero and binary function).
     * @tparam <AccumT>   Models a binary function
     *
     * @param[in]  a       The addend (left-hand) operand
     * @param[in]  b       The addend (right-hand) operand
     * @param[out] c       The sum (destination) operand
     * @param[in]  monoid  The element-wise operation to combine the operands
     * @param[in]  accum   The function for how to assign to destination (e.g.
     *                     Accum to "add" to destination, or Assign to
     *                     "replace" the destimation.
     *
     * @throw DimensionException  If a, b, and c do not have the same shape
     */
    template<typename AMatrixT,
             typename BMatrixT,
             typename CMatrixT,
             typename MonoidT =
                 graphblas::math::Plus<typename AMatrixT::ScalarType>,
             typename AccumT =
                 graphblas::math::Assign<typename CMatrixT::ScalarType> >
    inline void ewiseadd(AMatrixT const &a,
                         BMatrixT const &b,
                         CMatrixT       &c,
                         MonoidT         monoid = MonoidT(),
                         AccumT          accum = AccumT())
    {
        same_dimension_check(a,b,std::string("ewiseadd"));
        backend::ewiseadd(a.m_mat, b.m_mat, c.m_mat, monoid, accum);
    }

    /**
     * @brief Perform an element wise binary operation that can be optimized
     *        for "multiply" semantics (AND short circuit logic).
     *
     * If either element of the Monoid (binary) operation is a structural zero
     * then the result is the structural zero.  If they are both stored values
     * then the result is the result of the specified Monoid.
     *
     * @tparam <MatrixT>  Models the Matrix concept
     * @tparam <MonoidT>  Models a Monoid (zero and binary function).
     * @tparam <AccumT>   Models a binary function
     *
     * @param[in]  a       The multiplicand (left-hand) operand
     * @param[in]  b       The multiplier (right-hand) operand
     * @param[out] c       The product (destination) operand
     * @param[in]  monoid  The element-wise operation to combine the operands
     * @param[in]  accum   The function for how to assign to destination (e.g.
     *                     Accum to "add" to destination, or Assign to
     *                     "replace" the destimation.
     *
     * @throw DimensionException  If a, b, and c do not have the same shape
     */
    template<typename AMatrixT,
             typename BMatrixT,
             typename CMatrixT,
             typename MonoidT =
                 graphblas::math::Times<typename AMatrixT::ScalarType>,
             typename AccumT =
                 graphblas::math::Assign<typename CMatrixT::ScalarType> >
    inline void ewisemult(AMatrixT const &a,
                          BMatrixT const &b,
                          CMatrixT       &c,
                          MonoidT         monoid = MonoidT(),
                          AccumT          accum = AccumT())
    {
        same_dimension_check(a,b,std::string("ewisemult"));
        backend::ewisemult(a.m_mat, b.m_mat, c.m_mat, monoid, accum);
    }

    /**
     * @brief Perform an element wise binary operation that can be optimized
     *        for "multiply" semantics (AND short circuit logic) with output mask
     */
    template<typename AMatrixT,
             typename BMatrixT,
             typename CMatrixT,
             typename MMatrixT,
             typename MonoidT =
                 graphblas::math::Times<typename AMatrixT::ScalarType>,
             typename AccumT =
                 graphblas::math::Assign<typename CMatrixT::ScalarType> >
    inline void ewisemultMasked(AMatrixT const &a,
                                BMatrixT const &b,
                                CMatrixT       &c,
                                MMatrixT const &m,
                                bool            replace_flag,
                                MonoidT         monoid = MonoidT(),
                                AccumT          accum = AccumT())
    {
        same_dimension_check(a,b,std::string("ewisemultMasked"));
        same_dimension_check(a,c,std::string("ewisemultMasked"));
        same_dimension_check(c,m,std::string("ewisemultMasked"));
        backend::ewisemultMasked(a.m_mat, b.m_mat, c.m_mat,
                                 m.m_mat, replace_flag,
                                 monoid, accum);
    }

    /**
     * @brief Perform matrix-matrix multiply
     *
     * @tparam <MatrixT>   Models the Matrix concept
     * @tparam <SemiringT> Models a Semiring concept (zero add and mult).
     * @tparam <AccumT>    Models a binary function
     *
     * @param[in]  a  The left-hand matrix
     * @param[in]  b  The right-hand matrix
     * @param[out] c  The result matrix.
     * @param[in]  s  The Semiring to use for the matrix multiplication
     * @param[in]  accum   The function for how to assign to destination (e.g.
     *                     Accum to "add" to destination, or Assign to
     *                     "replace" the destination.
     * @param[in]  m  The mask
     *
     * @throw DimensionException  If the matrix dimensions for a, b, and c
     *                            are not consistent for matrix multiply
     */
    template<typename AMatrixT,
             typename BMatrixT,
             typename CMatrixT,
             typename SemiringT =
                 graphblas::ArithmeticSemiring<typename AMatrixT::ScalarType>,
             typename AccumT =
                 graphblas::math::Assign<typename CMatrixT::ScalarType> >
    inline void mxm(AMatrixT const &a,
                    BMatrixT const &b,
                    CMatrixT       &c,
                    SemiringT       s     = SemiringT(),
                    AccumT          accum = AccumT())
    {
        multiply_dimension_check(a,b,std::string("mxm"));
        backend::mxm(a.m_mat, b.m_mat, c.m_mat, s, accum);
    }

    /**
     *  Masked matrix multiply where the mask is applied to the result of
     *  A * B *before* it is accumulated with C.
     *
     * @param[in]  a  The left-hand matrix
     * @param[in]  b  The right-hand matrix
     * @param[out] c  The result matrix.
     * @param[in]  m  The mask
     * @param[in]  s  The Semiring to use for the matrix multiplication
     * @param[in]  accum   The function for how to assign to destination (e.g.
     *                     Accum to "add" to destination, or Assign to
     *                     "replace" the destination.
     *
     */
    template<typename AMatrixT,
             typename BMatrixT,
             typename CMatrixT,
             typename MMatrixT,
             typename SemiringT =
                    graphblas::ArithmeticSemiring<typename AMatrixT::ScalarType>,
             typename AccumT =
                    graphblas::math::Assign<typename CMatrixT::ScalarType> >
    inline void mxmMasked(AMatrixT const &a,
                          BMatrixT const &b,
                          CMatrixT       &c,
                          MMatrixT const &m,
                          SemiringT       s = SemiringT(),
                          AccumT          accum = AccumT())
    {
        multiply_dimension_check(a,b,std::string("mxmMasked"));
        backend::mxmMasked(a.m_mat, b.m_mat, c.m_mat, m.m_mat, s, accum);
    }

    /**
     *  Masked matrix multiply (a different semantic on the mask).  This applies
     *  the mask to the result matrix after accumulation, and will remove
     *  stored elements anywhere the mask is not set.
     *
     * @param[in]  a  The left-hand matrix
     * @param[in]  b  The right-hand matrix
     * @param[out] c  The result matrix.
     * @param[in]  m  The mask
     * @param[in]  s  The Semiring to use for the matrix multiplication
     * @param[in]  accum   The function for how to assign to destination (e.g.
     *                     Accum to "add" to destination, or Assign to
     *                     "replace" the destination.
     *
     */
    template<typename AMatrixT,
             typename BMatrixT,
             typename CMatrixT,
             typename MMatrixT,
             typename SemiringT =
                 graphblas::ArithmeticSemiring<typename AMatrixT::ScalarType>,
             typename AccumT =
                 graphblas::math::Assign<typename CMatrixT::ScalarType> >
    inline void mxmMaskedV2(AMatrixT const &a,
                            BMatrixT const &b,
                            CMatrixT       &c,
                            MMatrixT       &m,
                            SemiringT       s = SemiringT(),
                            AccumT          accum = AccumT())
    {
        multiply_dimension_check(a,b,std::string("mxmMaskedV2"));
        backend::mxmMasked(a.m_mat, b.m_mat, c.m_mat, m.m_mat, s, accum);
    }

    /**
     * @brief Perform row vector-matrix multiply.
     *
     * @tparam <VectorT>   Models the Matrix concept (1 x N)
     * @tparam <MatrixT>   Models the Matrix concept
     * @tparam <SemiringT> Models a Semiring concept (zero add and mult).
     * @tparam <AccumT>    Models a binary function
     *
     * @param[in]  a  The left-hand row vector
     * @param[in]  b  The right-hand matrix
     * @param[out] c  The result row vector.
     * @param[in]  s  The Semiring to use for the matrix multiplication
     * @param[in]  accum   The function for how to assign to destination (e.g.
     *                     Accum to "add" to destination, or Assign to
     *                     "replace" the destimation.
     *
     * @throw DimensionException  If a or c do not have row vector dimension.
     * @throw DimensionException  If the matrix dimensions for a, b, and c
     *                            are not consistent for matrix multiply
     */
    template<typename AVectorT,
             typename BMatrixT,
             typename CVectorT,
             typename SemiringT =
                 graphblas::ArithmeticSemiring<typename AVectorT::ScalarType>,
             typename AccumT =
                 graphblas::math::Assign<typename CVectorT::ScalarType> >
    inline void vxm(AVectorT const &a,
                    BMatrixT const &b,
                    CVectorT       &c,
                    SemiringT       s     = SemiringT(),
                    AccumT          accum = AccumT())
    {
        //vector_multiply_dimension_check(a, b.get_shape().first);
        //backend::vxm(a.m_vec, b.m_mat, c.m_vec, s, accum);
        backend::vxm(a.m_mat, b.m_mat, c.m_mat, s, accum);
    }

    /**
     * @brief Perform matrix-column vector multiply.
     *
     * @tparam <MatrixT>   Models the Matrix concept
     * @tparam <VectorT>   Models the Matrix concept (M x 1)
     * @tparam <SemiringT> Models a Semiring concept (zero add and mult).
     * @tparam <AccumT>    Models a binary function
     *
     * @param[in]  a  The left-hand matrix
     * @param[in]  b  The right-hand column vector
     * @param[out] c  The result column vector.
     * @param[in]  s  The Semiring to use for the matrix multiplication
     * @param[in]  accum   The function for how to assign to destination (e.g.
     *                     Accum to "add" to destination, or Assign to
     *                     "replace" the destimation.
     *
     * @throw DimensionException  If a or c do not have column vector dimension.
     * @throw DimensionException  If the matrix dimensions for a, b, and c
     *                            are not consistent for matrix multiply
     */
    template<typename AMatrixT,
             typename BVectorT,
             typename CVectorT,
             typename SemiringT =
                 graphblas::ArithmeticSemiring<typename AMatrixT::ScalarType>,
             typename AccumT =
                 graphblas::math::Assign<typename CVectorT::ScalarType> >
    inline void mxv(AMatrixT const &a,
                    BVectorT const &b,
                    CVectorT       &c,
                    SemiringT       s     = SemiringT(),
                    AccumT          accum = AccumT())
    {
        //vector_multiply_dimension_check(b, a.get_shape().second);
        //backend::mxv(a.m_mat, b.m_vec, c.m_vec, s, accum);
        backend::mxv(a.m_mat, b.m_mat, c.m_mat, s, accum);
    }

    /**
     * @brief Extract a sub-matrix (sub-graph) from a larger matrix (graph).
     *
     * @tparam <MatrixT>     Models the Matrix concept
     * @tparam <RAIterator>  Models a random access iterator
     * @tparam <AccumT>      Models a binary function
     *
     * @param[in]  a  The matrix to extract from
     * @param[in]  i  Iterator into the ordered set of rows to select from a
     * @param[in]  j  Iterator into the ordered set of columns to select from a
     * @param[out] c  Holds the resulting submatrix
     * @param[in]  accum   The function for how to assign to destination (e.g.
     *                     Accum to "add" to destination, or Assign to
     *                     "replace" the destimation.
     *
     * @note The number of rows in c is taken to equal the number of values
     *       iterated over by i.  The number of columns in c is taken to equal
     *       the number values iterated over by j.
     *
     * @todo Need to throw Dimension exception if attempt to access element
     *       outside the dimensions of a.
     */
    template<typename AMatrixT,
             typename CMatrixT,
             typename RAIteratorI,
             typename RAIteratorJ,
             typename AccumT =
                 graphblas::math::Assign<typename CMatrixT::ScalarType> >
    inline void extract(AMatrixT       const &a,
                        RAIteratorI           i,
                        RAIteratorJ           j,
                        CMatrixT             &c,
                        AccumT                accum = AccumT())
    {
        assign_extract_dimension_check(a,c,i,j);
        backend::extract(a.m_mat, i, j, c.m_mat, accum);
    }

    /**
     * @brief Extract a sub-matrix (sub-graph) from a larger matrix (graph).
     * This wrapper is provided for converting IndexArrayType to iterators
     *
     * @tparam <MatrixT>     Models the Matrix concept
     * @tparam <AccumT>      Models a binary function
     *
     * @param[in]  a  The matrix to extract from
     * @param[in]  i  The ordered set of rows to select from a
     * @param[in]  j  The ordered set of columns to select from a
     * @param[out] c  Holds the resulting submatrix
     * @param[in]  accum   The function for how to assign to destination (e.g.
     *                     Accum to "add" to destination, or Assign to
     *                     "replace" the destimation.
     *
     * @throw DimensionException  If the number of rows in c does not equal the
     *                            number of values in i.
     * @throw DimensionException  If the number of columns in c doesn't equal
     *                            the number of values in j.
     *
     * @todo Need to throw Dimension exception if attempt to access element
     *       outside the dimensions of a.
     */
    template<typename AMatrixT,
             typename CMatrixT,
             typename AccumT =
                 graphblas::math::Assign<typename CMatrixT::ScalarType> >
    inline void extract(AMatrixT       const &a,
                        IndexArrayType const &i,
                        IndexArrayType const &j,
                        CMatrixT             &c,
                        AccumT                accum = AccumT())
    {
        /// @todo this does not check the dimension of C against i and j
        assign_extract_dimension_check(a,c,i.begin(),j.begin());
        backend::extract(a.m_mat, i, j, c.m_mat, accum);
    }

    /**
     * @brief Assign a matrix to a set of indices in a larger matrix.
     *
     * @tparam <MatrixT>     Models the Matrix concept
     * @tparam <RAIterator>  Models a random access iterator
     * @tparam <AccumT>      Models a binary function
     *
     * @param[in]  a  The matrix to assign from
     * @param[in]  i  Iterator into the ordered set of rows to assign in c
     * @param[in]  j  Iterator into the ordered set of columns to assign in c
     * @param[out] c  The matrix to assign into a subset of
     * @param[in]  accum   The function for how to assign to destination (e.g.
     *                     Accum to "add" to destination, or Assign to
     *                     "replace" the destimation.
     *
     * @note The number of rows in a is taken to equal the number of values
     *       iterated over by i.  The number of columns in a is taken to equal
     *       the number values iterated over by j.
     *
     * @todo Need to throw Dimension exception if attempt to assign to elements
     *       outside the dimensions of c.
     */
    template<typename AMatrixT,
             typename CMatrixT,
             typename RAIteratorI,
             typename RAIteratorJ,
             typename AccumT =
                 graphblas::math::Assign<typename CMatrixT::ScalarType> >
    inline void assign(AMatrixT const    &a,
                       RAIteratorI        i,
                       RAIteratorJ        j,
                       CMatrixT          &c,
                       AccumT             accum = AccumT())
    {
        assign_extract_dimension_check(c,a,i,j);
        backend::assign(a.m_mat, i, j, c.m_mat, accum);
    }

    /**
     * @brief Assign a matrix to a set of indices in a larger matrix.
     * This wrapper is provided for IndexArrayType to iterators.
     *
     * @tparam <MatrixT>     Models the Matrix concept
     * @tparam <AccumT>      Models a binary function
     *
     * @param[in]  a  The matrix to assign from
     * @param[in]  i  The ordered set of rows to assign in c
     * @param[in]  j  The ordered set of columns to assign in c
     * @param[out] c  The matrix to assign into a subset of
     * @param[in]  accum   The function for how to assign to destination (e.g.
     *                     Accum to "add" to destination, or Assign to
     *                     "replace" the destimation.
     *
     * @throw DimensionException  If the number of rows in a does not equal the
     *                            number of values in i.
     * @throw DimensionException  If the number of columns in a doesn't equal
     *                            the number of values in j.
     *
     * @todo Need to throw Dimension exception if attempt to assign to elements
     *       outside the dimensions of c.
     */
    template<typename AMatrixT,
             typename CMatrixT,
             typename AccumT =
                 graphblas::math::Assign<typename CMatrixT::ScalarType> >
    inline void assign(AMatrixT const       &a,
                       IndexArrayType const &i,
                       IndexArrayType const &j,
                       CMatrixT             &c,
                       AccumT                accum = AccumT())
    {
        assign_extract_dimension_check(c, a, i.begin(), j.begin());
        backend::assign(a.m_mat, i, j, c.m_mat, accum);
    }

    /**
     * @brief Apply a unary function to all elements of a matrix.
     *
     * @tparam <MatrixT>         Models the Matrix concept
     * @tparam <UnaryFunctionT>  Models a unary function
     * @tparam <AccumT>          Models a binary function
     *
     * @param[in]  a  The matrix to access from
     * @param[out] c  The matrix to assign into
     * @param[in]  f  The function to apply the result to
     * @param[in]  accum   The function for how to assign to destination (e.g.
     *                     Accum to "add" to destination, or Assign to
     *                     "replace" the destimation.
     *
     * @throw DimensionException  If the sizes of a and c do not match
     */
    template<typename AMatrixT,
             typename CMatrixT,
             typename UnaryFunctionT,
             typename AccumT=
                 graphblas::math::Assign<typename CMatrixT::ScalarType> >
    inline void apply(AMatrixT const &a,
                      CMatrixT       &c,
                      UnaryFunctionT  f,
                      AccumT          accum = AccumT())
    {
        same_dimension_check(a,c,std::string("apply"));
        backend::apply(a.m_mat, c.m_mat, f, accum);
    }

    /**
     * @brief Apply a reduction operation to each row of a matrix.
     *
     * @tparam <MatrixT>  Models the Matrix concept
     * @tparam <MonoidT>  Models a binary function
     * @tparam <AccumT>   Models a binary function
     *
     * @param[in]  a  The matrix to perform the row reduction on.
     * @param[out] c  The matrix (column vector) to assign the result to.
     * @param[in]  m  The monoid (binary op) used to reduce the row elements
     * @param[in]  accum   The function for how to assign to destination (e.g.
     *                     Accum to "add" to destination, or Assign to
     *                     "replace" the destimation.
     *
     * @throw DimensionException  If the rows a and c do not match, or the
     *                            the number of columns in c is not one.
     */
    template<typename AMatrixT,
             typename CMatrixT,
             typename MonoidT =
                 graphblas::PlusMonoid<typename AMatrixT::ScalarType>,
             typename AccumT =
                 graphblas::math::Assign<typename CMatrixT::ScalarType> >
    inline void row_reduce(AMatrixT const &a,
                           CMatrixT       &c, // vector?
                           MonoidT         m     = MonoidT(),
                           AccumT          accum = AccumT())
    {
        if (a.get_shape().first != c.get_shape().first || c.get_shape().second != 1){
            throw graphblas::DimensionException("row reduce dimension error");
        }
        backend::row_reduce(a.m_mat, c.m_mat, m, accum);
    }

    /**
     * @brief Apply a reduction operation to each column of a matrix.
     *
     * @tparam <MatrixT>  Models the Matrix concept
     * @tparam <MonoidT>  Models a binary function
     * @tparam <AccumT>   Models a binary function
     *
     * @param[in]  a  The matrix to perform the column reduction on.
     * @param[out] c  The matrix (row vector) to assign the result to.
     * @param[in]  m  The monoid (binary op) used to reduce the column elements
     * @param[in]  accum   The function for how to assign to destination (e.g.
     *                     Accum to "add" to destination, or Assign to
     *                     "replace" the destimation.
     *
     * @throw DimensionException  If the columns a and c do not match, or the
     *                            the number of rows in c is not one.
     */
    template<typename AMatrixT,
             typename CMatrixT,
             typename MonoidT =
                 graphblas::PlusMonoid<typename AMatrixT::ScalarType>,
             typename AccumT =
                 graphblas::math::Assign<typename CMatrixT::ScalarType> >
    inline void col_reduce(AMatrixT const &a,
                           CMatrixT       &c, // vector?
                           MonoidT         m     = MonoidT(),
                           AccumT          accum = AccumT())
    {
        if (a.get_shape().second != c.get_shape().second || c.get_shape().first != 1){
            throw graphblas::DimensionException("col reduce dimension error, arg1.cols="+std::to_string(a.get_shape().second)
                    +" ,arg2.cols="+std::to_string(c.get_shape().second));
        }
        backend::col_reduce(a.m_mat, c.m_mat, m, accum);
    }


    /**
     * @brief Apply a masked reduction operation to each row of a matrix.
     *
     * @tparam <MatrixT>  Models the Matrix concept
     * @tparam <MonoidT>  Models a binary function
     * @tparam <AccumT>   Models a binary function
     *
     * @param[in]  a  The matrix to perform the row reduction on.
     * @param[out] c  The matrix (column vector) to assign the result to.
     * @param[in]  mask The matrix containing the mask.
     * @param[in]  m  The monoid (binary op) used to reduce the row elements
     * @param[in]  accum   The function for how to assign to destination (e.g.
     *                     Accum to "add" to destination, or Assign to
     *                     "replace" the destimation.
     *
     * @throw DimensionException  If the rows a and c do not match, or the
     *                            the number of columns in c is not one.
     */
    template<typename AMatrixT,
             typename CMatrixT,
             typename MMatrixT,
             typename MonoidT =
                 graphblas::PlusMonoid<typename AMatrixT::ScalarType>,
             typename AccumT =
                 graphblas::math::Assign<typename CMatrixT::ScalarType> >
    inline void rowReduceMasked(AMatrixT const &a,
                           CMatrixT       &c, // vector?
                           MMatrixT       &mask,
                           MonoidT         sum     = MonoidT(),
                           AccumT          accum = AccumT())
    {
        if (a.get_shape().first != c.get_shape().first || c.get_shape().second != 1){
            throw graphblas::DimensionException("row reduce masked dimension error");
        }
        backend::rowReduceMasked(a.m_mat, c.m_mat, mask.m_mat, sum, accum);
    }

    /**
     * @brief Apply a masked reduction operation to each column of a matrix.
     *
     * @tparam <MatrixT>  Models the Matrix concept
     * @tparam <MonoidT>  Models a binary function
     * @tparam <AccumT>   Models a binary function
     *
     * @param[in]  a  The matrix to perform the column reduction on.
     * @param[out] c  The matrix (row vector) to assign the result to.
     * @param[in]  mask The matrix containing the mask.
     * @param[in]  m  The monoid (binary op) used to reduce the column elements
     * @param[in]  accum   The function for how to assign to destination (e.g.
     *                     Accum to "add" to destination, or Assign to
     *                     "replace" the destimation.
     *
     * @throw DimensionException  If the columns a and c do not match, or the
     *                            the number of rows in c is not one.
     */
    template<typename AMatrixT,
             typename CMatrixT,
             typename MMatrixT,
             typename MonoidT =
                 graphblas::PlusMonoid<typename AMatrixT::ScalarType>,
             typename AccumT =
                 graphblas::math::Assign<typename CMatrixT::ScalarType> >
    inline void colReduceMasked(AMatrixT const &a,
                           CMatrixT       &c, // vector?
                           MMatrixT       &mask,
                           MonoidT         sum     = MonoidT(),
                           AccumT          accum = AccumT())
    {
        if (a.get_shape().second != c.get_shape().second || c.get_shape().first != 1){
            throw graphblas::DimensionException("col reduce dimension error, arg1.cols="+std::to_string(a.get_shape().second)
                    +" ,arg2.cols="+std::to_string(c.get_shape().second));
        }
        backend::colReduceMasked(a.m_mat, c.m_mat, mask.m_mat, sum, accum);
    }

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
     *
     * @tparam <MatrixT>  Models the Matrix concept
     *
     * @param[in]  a  The matrix to flip
     * @param[out] c  The matrix to assign the result to.
     *
     * @brief This does not work if a and c are the same matrix (and if it
     *        did, the matrix would have to be square due to the immutability
     *       of matrix dimensions)
     * @todo Should this support the Accum operations
     *
     * @throw DimensionException  If the columns of a != rows of c, or
     *                            if the rows of a != columns of c.
     */
    template<typename AMatrixT,
             typename CMatrixT>
    inline void transpose(AMatrixT const &a,
                          CMatrixT       &c)
    {
        multiply_dimension_check(a,c, std::string("transpose"));
        multiply_dimension_check(c,a, std::string("transpose"));
        backend::transpose(a.m_mat, c.m_mat);
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

    /**
     * @brief Output (row, col, value) tuples from Matrix as three vectors.
     *
     * @note In a departure from the STL standards, we require the
     *       output iterators to be random access (RA) iterators,
     *       although we never read from these iterators, only write.
     *       This allows potentially parallel implementations that
     *       will not be linear in the number of tuples.  The RA
     *       iterators must either be prepared to receive the number
     *       of elements equal to get_nnz() or they must include
     *       all necessary memory allocation in their random access logic.
     *
     * @tparam <MatrixT>    model GraphBLAS matrix concept
     * @tparam <RAIterator> model a random access iterator
     *
     * @param[in]  a The matrix with the values to be output
     * @param[out] i The row indices random access iterator (accepts
     *               IndexType values)
     * @param[out] j The column indices random access iterator (accepts
     *               IndexType values)
     * @param[out] v The values random access iterator (accepts
     *               AMatrixT::ScalarType )
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
        backend::extracttuples(a.m_mat, i, j, v);
    }
    /**
     * @brief Output (row, col, value) tuples from a Matrix as three vectors.
     * This wrapper is provided for IndexArrayType to iterators.
     *
     * @note This version takes specific container types for outputs.
     *       It is provided to (hopefully) mirror the current
     *       GraphBLAS design discussed by the GraphBLAS committee.
     *       The iterator version subsumes this version.
     *
     * @tparam <MatrixT>    model GraphBLAS matrix concept
     *
     * @param[in]  a The matrix with the values to be output
     * @param[out] i The row indices index array
     * @param[out] j The column indices index array
     * @param[out] v The values vector
     */
    template<typename AMatrixT>
    inline void extracttuples(AMatrixT const                             &a,
                              IndexArrayType                             &i,
                              IndexArrayType                             &j,
                              std::vector<typename AMatrixT::ScalarType> &v)
    {
        backend::extracttuples(a.m_mat, i, j, v);
    }

    /**
     * @brief Populate a Matrix with stored values at specified locations
     *
     * @tparam <MatrixT>      Models the Matrix concept
     * @tparam <RAIteratorT>  Models a random access iterator
     * @tparam <AccumT>       Models a binary function
     *
     * @param[out] m The matrix to assign/accum values to
     * @param[in]  i The iterator over the row indices
     * @param[in]  j The iterator over the col indices
     * @param[in]  v The iterator over the values to store
     * @param[in]  n The number of values to assign.
     * @param[in]  accum   The function for how to assign to destination (e.g.
     *                     Accum to "add" to destination, or Assign to
     *                     "replace" the destimation.
     *
     * @brief Need to add a parameter (functor?) to handle duplicate locations
     * (handled in cusp backend)
     *
     *
     * @throw DimensionException  If an element of i or j index outisde the
     *                            size of m.
     */
    template<typename MatrixT,
             typename RAIteratorI,
             typename RAIteratorJ,
             typename RAIteratorV,
             typename AccumT =
                 graphblas::math::Assign<typename MatrixT::ScalarType> >
    inline void buildmatrix(MatrixT     &m,
                            RAIteratorI  i,
                            RAIteratorJ  j,
                            RAIteratorV  v,
                            IndexType    n,
                            AccumT       accum = AccumT())
    {
        //if (*(i+n) == 0 || *(j+n) == 0 || *(v+n) == 0){
        //    throw graphblas::DimensionException();
        //}
        backend::buildmatrix(m.m_mat, i, j, v, n, accum);
    }
    /**
     * @brief Populate a Matrix with stored values at specified locations
     * This wrapper is provided for IndexArrayType to iterators.
     *
     * @tparam <MatrixT>      Models the Matrix concept
     * @tparam <AccumT>       Models a binary function
     *
     * @param[out] m The matrix to assign/accum values to
     * @param[in]  i The array of row indices
     * @param[in]  j The array of column indices
     * @param[in]  v The array of values to store
     * @param[in]  accum   The function for how to assign to destination (e.g.
     *                     Accum to "add" to destination, or Assign to
     *                     "replace" the destimation.
     *
     * @todo Need to add a parameter (functor?) to handle duplicate locations
     *
     * @throw DimensionException  If the sizes of i, j, and v are not same.
     * @throw DimensionException  If an element of i or j index outisde the
     *                            size of m.
     */
    template<typename MatrixT,
             typename AccumT =
                 graphblas::math::Assign<typename MatrixT::ScalarType> >
    inline void buildmatrix(MatrixT              &m,
                            IndexArrayType const &i,
                            IndexArrayType const &j,
                            std::vector<typename MatrixT::ScalarType> const &v,
                            AccumT                accum = AccumT())
    {
        backend::buildmatrix(m.m_mat, i.begin(), j.begin(), v.begin(), v.size(), accum);
    }
} // graphblas

#include <graphblas/detail/config.hpp>
//#define __GB_SYSTEM_OPERATIONS_HEADER <graphblas/system/__GB_SYSTEM_ROOT/operations.hpp>
//#include __GB_SYSTEM_OPERATIONS_HEADER
//#undef __GB_SYSTEM_OPERATIONS_HEADER

#endif
