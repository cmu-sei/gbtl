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

#include <cstddef>
#include <graphblas/detail/config.hpp>
#include <graphblas/operations.hpp>
#include <graphblas/utility.hpp>
#include <graphblas/View.hpp>

// Include vector definitions from the appropriate backend.
#define __GB_SYSTEM_VECTOR_HEADER <graphblas/system/__GB_SYSTEM_ROOT/Vector.hpp>
#include __GB_SYSTEM_VECTOR_HEADER
#undef __GB_SYSTEM_VECTOR_HEADER

namespace GraphBLAS
{

    //**************************************************************************
    template<typename ScalarT, typename... TagsT>
    class Vector
    {
    public:
        typedef ScalarT ScalarType;
        typedef GraphBLAS::backend::Vector<ScalarT> BackendType;

        Vector() = delete;

        /**
         * @brief Construct a copy of a vector
         *
         * @note Calls backend constructor.
         *
         * @param[in]  vec  The vector to copy.
         */
        template <typename OtherVectorT>
        Vector(OtherVectorT const &vec)
            : m_vec(vec)
        {
        }

        /**
         * @brief Construct a dense vector with 'count' copies of 'value'
         *
         * @note Calls backend constructor.
         *
         * @param[in]  count  Number of elements in the vector.
         * @param[in]  value  The scalar value to store in each element
         */
        template <typename SizeT>
        Vector(SizeT const &count, ScalarT const &value)
            : m_vec(count, value)
        {
        }

        /// Destructor
        ~Vector() { }

        /**
         * @brief Assignment from another vector
         *
         * @param[in]  rhs  The vector to copy from.
         *
         * @todo Should assignment work only if dimensions are same?
         * @note This clears any previous information
         */
        Vector<ScalarT, TagsT...>
        operator=(Vector<ScalarT, TagsT...> const &rhs)
        {
            if (this != &rhs)
            {
                m_vec = rhs.m_vec;
            }
            return *this;
        }

        /**
         * @brief Assignment from dense data
         *
         * @param[in]  rhs  The C++ vector of vectors to copy from.
         *
         * @todo revisit vector of vectors?
         * @todo This ignores the structural zero value.
         */
        Vector<ScalarT, TagsT...>& operator=(std::vector<ScalarT> const &rhs)
        {
            m_vec = rhs;
            return *this;
        }

        /// @todo need to change to mix and match internal types
        bool operator==(Vector<ScalarT, TagsT...> const &rhs) const
        {
            return (m_vec == rhs.m_vec);
        }

        bool operator!=(Vector<ScalarT, TagsT...> const &rhs) const
        {
            return !(*this == rhs);
        }

    private:
        template<typename WVectorT,
                 typename RAIteratorIT,
                 typename RAIteratorVT,
                 typename BinaryOpT>
        friend inline void vectorBuild(WVectorT           &w,
                                       RAIteratorIT        indices,
                                       RAIteratorVT        values,
                                       IndexType           numVals,
                                       BinaryOpT           dup);

        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename ValueT,
                 typename BinaryOpT>
        friend inline void vectorBuild(WVectorT                   &w,
                                       IndexArrayType       const &indices,
                                       std::vector<ValueT>  const &values,
                                       BinaryOpT                   dup);

        template<typename RAIteratorIT,
                 typename RAIteratorVT,
                 typename WVectorT>
        friend inline void vectorExtract(RAIteratorIT        indices,
                                         RAIteratorVT        values,
                                         WVectorT     const &w,
                                         std::string        &err);

        template<typename ValueT,
                 typename WVectorT>
        friend inline void vectorExtract(IndexArrayType            &indices,
                                         std::vector<ValueT>       &values,
                                         WVectorT            const &w,
                                         std::string               &err);

        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename SemiringT,
                 typename UVectorT,
                 typename AMatrixT>
        friend inline void vxm(WVectorT         &w,
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
        friend inline void mxv(WVectorT        &w,
                               MaskT     const &mask,
                               AccumT           accum,
                               SemiringT        op,
                               AMatrixT  const &A,
                               UVectorT  const &u,
                               bool             replace_flag);

        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  //can be monoid or binaryop (not Semiring)
                 typename UVectorT,
                 typename VVectorT>
        friend inline void eWiseMult(WVectorT        &w,
                                     MaskT     const &mask,
                                     AccumT           accum,
                                     BinaryOpT        op,
                                     UVectorT  const &u,
                                     VVectorT  const &v,
                                     bool             replace_flag);

        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  //can be monoid or binaryop (not Semiring)
                 typename UVectorT,
                 typename VVectorT>
        friend inline void eWiseAdd(WVectorT         &w,
                                    MaskT      const &mask,
                                    AccumT            accum,
                                    BinaryOpT         op,
                                    UVectorT   const &u,
                                    VVectorT   const &v,
                                    bool              replace_flag);

        /// Extract: Standard vector variant
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename UVectorT,
                 typename RAIteratorT>
        friend inline void extract(WVectorT           &w,
                                   MaskT        const &mask,
                                   AccumT             accum,
                                   UVectorT    const &u,
                                   RAIteratorT        indices,
                                   IndexType          num_indices,
                                   bool               replace_flag);

        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename UVectorT>
        friend inline void extract(WVectorT             &w,
                                   MaskT          const &mask,
                                   AccumT                accum,
                                   UVectorT       const &u,
                                   IndexArrayType const &indices,
                                   bool                  replace_flag);

        // Extract col (or row with transpose)
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename AMatrixT,
                 typename RAIteratorI>
        friend inline void extract(WVectorT             &w,
                                   MaskT          const &mask,
                                   AccumT                accum,
                                   AMatrixT       const &A,
                                   RAIteratorI           row_indices,
                                   IndexType             nrows,
                                   IndexType             col_index,
                                   bool                  replace_flag);

        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename AMatrixT>
        friend inline void extract(WVectorT             &w,
                                   MaskT          const &mask,
                                   AccumT                accum,
                                   AMatrixT       const &A,
                                   IndexArrayType const &row_indices,
                                   IndexType             col_index,
                                   bool                  replace_flag);

        // Extract single element (vector)
        template <typename ValueT,
                  typename AccumT,
                  typename UVectorT>
        friend inline void extract(ValueT            &dst,
                                   AccumT             accum,
                                   UVectorT    const &u,
                                   IndexType          index,
                                   std::string       &err);

        // Standard Vector Variant
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename UVectorT,
                 typename RAIteratorT>
        friend inline void assign(WVectorT          &w,
                                  MaskT       const &mask,
                                  AccumT             accum,
                                  UVectorT    const &u,
                                  RAIteratorT const &indices,
                                  IndexType          num_indices,
                                  bool               replace_flag);

        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename UVectorT>
        friend inline void assign(WVectorT             &w,
                                  MaskT          const &mask,
                                  AccumT                accum,
                                  UVectorT       const &u,
                                  IndexArrayType const &indices,
                                  bool                  replace_flag);

        // Assign Column variant
        template<typename CMatrixT,
                 typename MaskT,
                 typename AccumT,
                 typename UVectorT,
                 typename RAIteratorT>
        friend inline void assign(CMatrixT          &C,
                                  MaskT       const &mask,  // a vector
                                  AccumT             accum,
                                  UVectorT    const &u,
                                  RAIteratorT        row_indices,
                                  IndexType          num_rows,
                                  IndexType          col_index,
                                  bool               replace_flag);

        template<typename CMatrixT,
                 typename MaskT,
                 typename AccumT,
                 typename UVectorT>
        friend inline void assign(CMatrixT             &C,
                                  MaskT          const &mask,  // a vector
                                  AccumT                accum,
                                  UVectorT       const &u,
                                  IndexArrayType const &row_indices,
                                  IndexType             col_index,
                                  bool                  replace_flag);

        // Assign row variant
        template<typename CMatrixT,
                 typename MaskT,
                 typename AccumT,
                 typename UVectorT,
                 typename RAIteratorT>
        friend inline void assign(CMatrixT          &C,
                                  MaskT       const &mask,  // a vector
                                  AccumT             accum,
                                  UVectorT    const &u,
                                  IndexType          row_index,
                                  RAIteratorT        col_indices,
                                  IndexType          num_cols,
                                  bool               replace_flag);

        template<typename CMatrixT,
                 typename MaskT,
                 typename AccumT,
                 typename UVectorT>
        friend inline void assign(CMatrixT             &C,
                                  MaskT          const &mask,  // a vector
                                  AccumT                accum,
                                  UVectorT       const &u,
                                  IndexType             row_index,
                                  IndexArrayType const &col_indices,
                                  bool                  replace_flag);

        // Single value variant (vector)
        template<typename WVectorT,
                 typename AccumT,
                 typename ValueT>
        friend inline void assign(WVectorT             &w,
                                  AccumT                accum,
                                  ValueT                src,
                                  IndexType             index,
                                  bool                  replace_flag);

        // Vector constant variant
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename ValueT,
                 typename RAIteratorT>
        friend inline void assign(WVectorT          &w,
                                  MaskT       const &mask,
                                  AccumT             accum,
                                  ValueT             val,
                                  RAIteratorT const &indices,
                                  IndexType          num_indices,
                                  bool               replace_flag);

        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename ValueT>
        friend inline void assign(WVectorT             &w,
                                  MaskT          const &mask,
                                  AccumT                accum,
                                  ValueT                val,
                                  IndexArrayType const &indices,
                                  bool                  replace_flag);

        // vector variant
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename UnaryFunctionT,
                 typename UVectorT>
        friend inline void apply(WVectorT             &w,
                                 MaskT          const &mask,
                                 AccumT                accum,
                                 UnaryFunctionT        op,
                                 UVectorT       const &u,
                                 bool                  replace_flag);

        // row reduce matrix to column vector (use transpose for col reduce)
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryFunctionT,  // monoid or binary op only
                 typename AMatrixT>
        friend inline void reduce(WVectorT              &u,
                                  MaskT           const &mask,
                                  AccumT                 accum,
                                  BinaryFunctionT        op,
                                  AMatrixT        const &A,
                                  bool                   replace_flag);

        // vector-scalar variant
        template<typename ValueT,
                 typename AccumT,
                 typename MonoidT, // monoid only
                 typename UVectorT>
        friend inline void reduce(ValueT           &dst,
                                  AccumT            accum,
                                  MonoidT           op,
                                  UVectorT   const &u,
                                  bool              replace_flag);

        template<typename VectorT>
        friend inline backend::ComplementView<VectorT> complement(
            VectorT const &w);

    private:
        BackendType m_vec;
    };
} // end namespace graphblas
