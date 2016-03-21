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

namespace graphblas
{

    //************************************************************************
    template<typename ScalarT, typename... TagsT>
    class Vector
    {
    public:
        typedef ScalarT ScalarType;
        typedef graphblas::backend::Vector<ScalarT> BackendType;

        //@brief calls backend constructor
        template <typename T>
        Vector(T & t)
            : m_vec(t)
        {
        }

        //@brief calls backend constructor
        template <typename T1, typename T2>
        Vector(const T1 &t1, const T2 &t2)
            : m_vec(t1, t2)
        {
        }

        ~Vector() { }


        /// @todo Should assignment work only if dimensions are same?
        Vector<ScalarT, TagsT...>
        operator=(Vector<ScalarT, TagsT...> const &rhs)
        {
            if (this != &rhs)
            {
                m_vec = rhs.m_vec;
            }
            return *this;
        }


        /// Assignment from dense data
        /// @todo This ignores the structural zero value.
        Vector<ScalarT, TagsT...>& operator=(
            std::vector<std::vector<ScalarT> > const &rhs)
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
        BackendType m_vec;

        template<typename AMatrixT,
                 typename BMatrixT,
                 typename CMatrixT,
                 typename MonoidT,
                 typename AccumT >
        friend inline void ewiseadd(AMatrixT const &a,
                                    BMatrixT const &b,
                                    CMatrixT       &c,
                                    MonoidT         monoid,
                                    AccumT          accum);
        template<typename AMatrixT,
                 typename BMatrixT,
                 typename CMatrixT,
                 typename MonoidT,
                 typename AccumT >
        friend inline void ewisemult(AMatrixT const &a,
                              BMatrixT const &b,
                              CMatrixT       &c,
                              MonoidT         monoid,
                              AccumT          accum);

        template<typename AMatrixT,
                 typename BMatrixT,
                 typename CMatrixT,
                 typename SemiringT,
                 typename AccumT >
        friend inline void mxm(AMatrixT const &a,
                        BMatrixT const &b,
                        CMatrixT       &c,
                        SemiringT       s,
                        AccumT          accum);

        template<typename AMatrixT,
                 typename BMatrixT,
                 typename CMatrixT,
                 typename SemiringT,
                 typename AccumT >
        friend inline void mxv(AMatrixT const &a,
                        BMatrixT const &b,
                        CMatrixT       &c,
                        SemiringT       s,
                        AccumT          accum);


        template<typename AMatrixT,
                 typename BMatrixT,
                 typename CMatrixT,
                 typename MMatrixT,
                 typename SemiringT,
                 typename AccumT >
        friend inline void mxmMasked(AMatrixT const &a,
                              BMatrixT const &b,
                              CMatrixT       &c,
                              MMatrixT const &m,
                              SemiringT       s,
                              AccumT          accum);


        template<typename AMatrixT,
                 typename BMatrixT,
                 typename CMatrixT,
                 typename MMatrixT,
                 typename SemiringT,
                 typename AccumT >
        friend inline void mxmMaskedV2(AMatrixT const &a,
                                BMatrixT const &b,
                                CMatrixT       &c,
                                MMatrixT       &m,
                                SemiringT       s,
                                AccumT          accum);

        template<typename AVectorT,
                 typename BMatrixT,
                 typename CVectorT,
                 typename SemiringT,
                 typename AccumT >
        friend inline void vxm(AVectorT const &a,
                        BMatrixT const &b,
                        CVectorT       &c,
                        SemiringT       s,
                        AccumT          accum);


        template<typename AMatrixT,
                 typename BVectorT,
                 typename CVectorT,
                 typename SemiringT,
                 typename AccumT >
        friend inline void mxv(AMatrixT const &a,
                        BVectorT const &b,
                        CVectorT       &c,
                        SemiringT       s,
                        AccumT          accum);

        template<typename AMatrixT,
                 typename CMatrixT,
                 typename RAIteratorI,
                 typename RAIteratorJ,
                 typename AccumT >
        friend inline void extract(AMatrixT       const &a,
                            RAIteratorI           i,
                            RAIteratorJ           j,
                            CMatrixT             &c,
                            AccumT                accum);

        template<typename AMatrixT,
                 typename CMatrixT,
                 typename AccumT >
        friend inline void extract(AMatrixT       const &a,
                            IndexArrayType const &i,
                            IndexArrayType const &j,
                            CMatrixT             &c,
                            AccumT                accum);

        template<typename AMatrixT,
                 typename CMatrixT,
                 typename RAIteratorI,
                 typename RAIteratorJ,
                 typename AccumT  >
        friend inline void assign(AMatrixT const    &a,
                           RAIteratorI        i,
                           RAIteratorJ        j,
                           CMatrixT          &c,
                           AccumT             accum);


        template<typename AMatrixT,
                 typename CMatrixT,
                 typename AccumT  >
        friend inline void assign(AMatrixT const       &a,
                           IndexArrayType const &i,
                           IndexArrayType const &j,
                           CMatrixT             &c,
                           AccumT                accum);


        template<typename AMatrixT,
                 typename CMatrixT,
                 typename UnaryFunctionT,
                 typename AccumT >
        friend inline void apply(AMatrixT const &a,
                          CMatrixT       &c,
                          UnaryFunctionT  f,
                          AccumT          accum);

        template<typename AMatrixT,
                 typename CMatrixT,
                 typename MonoidT,
                 typename AccumT >
        friend inline void row_reduce(AMatrixT const &a,
                               CMatrixT       &c, // vector?
                               MonoidT         m,
                               AccumT          accum);


        template<typename AMatrixT,
                 typename CMatrixT,
                 typename MonoidT ,
                 typename AccumT  >
        friend inline void col_reduce(AMatrixT const &a,
                               CMatrixT       &c, // vector?
                               MonoidT         m,
                               AccumT          accum);

        template<typename MatrixT,
                 typename SemiringT  >
        friend inline NegateView<MatrixT, SemiringT> negate(
            MatrixT const   &a,
            SemiringT const &s);


        template<typename AMatrixT,
                 typename CMatrixT>
        friend inline void transpose(AMatrixT const &a,
                              CMatrixT       &c);

        template<typename AMatrixT>
        friend inline TransposeView<AMatrixT> transpose(AMatrixT const &a);


        template<typename AMatrixT,
                 typename RAIteratorIT,
                 typename RAIteratorJT,
                 typename RAIteratorVT>
        friend inline void extracttuples(AMatrixT const &a,
                                  RAIteratorIT    i,
                                  RAIteratorJT    j,
                                  RAIteratorVT    v);

        template<typename AMatrixT>
        friend inline void extracttuples(AMatrixT const                             &a,
                                  IndexArrayType                             &i,
                                  IndexArrayType                             &j,
                                  std::vector<typename AMatrixT::ScalarType> &v);


        template<typename MatrixT,
                 typename RAIteratorI,
                 typename RAIteratorJ,
                 typename RAIteratorV,
                 typename AccumT >
        friend inline void buildmatrix(MatrixT     &m,
                                RAIteratorI  i,
                                RAIteratorJ  j,
                                RAIteratorV  v,
                                IndexType    n,
                                AccumT       accum);

    template<typename MatrixT,
             typename AccumT >
    friend inline void buildmatrix(MatrixT              &m,
                            IndexArrayType const &i,
                            IndexArrayType const &j,
                            std::vector<typename MatrixT::ScalarType> const &v,
                            AccumT                accum );


        template<typename AMatrixT, typename BMatrixT>
        friend void index_of(AMatrixT const  &A,
                      BMatrixT        &B,
                      IndexType const  base_index);

        template<typename MatrixT>
        friend void col_index_of(MatrixT &mat);

        template<typename MatrixT>
        friend void row_index_of(MatrixT &mat);

    template<typename AMatrixT,
             typename CMatrixT,
             typename MMatrixT,
             typename MonoidT,
             typename AccumT >
    friend inline void rowReduceMasked(AMatrixT const &a,
                           CMatrixT       &c, // vector?
                           MMatrixT       &mask,
                           MonoidT         sum,
                           AccumT          accum);

    template<typename AMatrixT,
             typename CMatrixT,
             typename MMatrixT,
             typename MonoidT,
             typename AccumT >
    friend inline void colReduceMasked(AMatrixT const &a,
                           CMatrixT       &c, // vector?
                           MMatrixT       &mask,
                           MonoidT         sum,
                           AccumT          accum);

    };
} // end namespace graphblas
