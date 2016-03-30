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
#include <graphblas/detail/param_unpack.hpp>
#include <graphblas/operations.hpp>
#include <graphblas/utility.hpp>
#include <graphblas/Matrix.hpp>

// Include matrix definitions from the appropriate backend.
#define __GB_SYSTEM_NEGATEVIEW_HEADER <graphblas/system/__GB_SYSTEM_ROOT/NegateView.hpp>
#include __GB_SYSTEM_NEGATEVIEW_HEADER
#undef __GB_SYSTEM_NEGATEVIEW_HEADER

#define __GB_SYSTEM_TRANSPOSEVIEW_HEADER <graphblas/system/__GB_SYSTEM_ROOT/TransposeView.hpp>
#include __GB_SYSTEM_TRANSPOSEVIEW_HEADER
#undef __GB_SYSTEM_TRANSPOSEVIEW_HEADER

namespace graphblas
{
    template<typename MatrixT>
    class TransposeView;

    //************************************************************************
    template<typename MatrixT, typename SemiringT>
    class NegateView
    {
    public:
        typedef typename backend::NegateView<typename MatrixT::BackendType,
                                             SemiringT> BackendType;
        typedef typename MatrixT::ScalarType ScalarType;

        /**
         * @brief Construct an empty matrix with the specified shape.
         *
         * @param[in] num_rows  Number of rows in the matrix
         * @param[in] num_cols  Number of columns in the matrix
         * @param[in] zero      The "zero" value, additive identity, and
         *                      the structural zero.
         */

        //note:
        //the backend should be able to decide when to ignore any of the
        //tags and/or arguments
        NegateView(BackendType backend_view)
            : m_mat(backend_view)
        {
        }

        /**
         * @brief Copy constructor.
         *
         * @param[in] rhs   The matrix to copy.
         */
        NegateView(NegateView<MatrixT, SemiringT> const &rhs)
            : m_mat(rhs.m_mat)
        {
        }

        ~NegateView() { }

        void get_shape(IndexType &num_rows, IndexType &num_cols) const
        {
            m_mat.get_shape(num_rows, num_cols);
        }

        std::pair<IndexType, IndexType> get_shape() const
        {
            IndexType num_rows, num_cols;
            m_mat.get_shape(num_rows, num_cols);
            return std::make_pair(num_rows, num_cols);
        }

        ScalarType get_zero() const
        {
            return m_mat.get_zero();
        }

        IndexType get_nnz() const
        {
            return m_mat.get_nnz();
        }

        ScalarType get_value_at(IndexType row, IndexType col) const
        {
            return m_mat.get_value_at(row, col);
        }

        //other methods that may or may not belong here:
        //
        void print_info(std::ostream &os) const
        {
            os << "Frontend NegateView of:";
            m_mat.print_info(os);
        }

        /// @todo This does not need to be a friend
        friend std::ostream &operator<<(std::ostream &os, NegateView const &mat)
        {
            mat.print_info(os);
            return os;
        }

        /// @todo need to change to mix and match internal types
        template <typename OtherMatrixT>
        bool operator==(OtherMatrixT const &rhs) const
        {
            return (m_mat.operator==(rhs));
        }

        template <typename OtherMatrixT>
        bool operator!=(OtherMatrixT const &rhs) const
        {
            return !(*this == rhs);
        }
        //end other methods

    private:
        BackendType m_mat;

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
                 typename SringT,
                 typename AccumT >
        friend inline void mxm(AMatrixT const &a,
                               BMatrixT const &b,
                               CMatrixT       &c,
                               SringT          s,
                               AccumT          accum);

        template<typename AMatrixT,
                 typename BMatrixT,
                 typename CMatrixT,
                 typename SringT,
                 typename AccumT >
        friend inline void mxv(AMatrixT const &a,
                               BMatrixT const &b,
                               CMatrixT       &c,
                               SringT          s,
                               AccumT          accum);


        template<typename AMatrixT,
                 typename BMatrixT,
                 typename CMatrixT,
                 typename MMatrixT,
                 typename SringT,
                 typename AccumT >
        friend inline void mxmMasked(AMatrixT const &a,
                                     BMatrixT const &b,
                                     CMatrixT       &c,
                                     MMatrixT const &m,
                                     SringT          s,
                                     AccumT          accum);


        template<typename AMatrixT,
                 typename BMatrixT,
                 typename CMatrixT,
                 typename MMatrixT,
                 typename SringT,
                 typename AccumT >
        friend inline void mxmMaskedV2(AMatrixT const &a,
                                       BMatrixT const &b,
                                       CMatrixT       &c,
                                       MMatrixT       &m,
                                       SringT          s,
                                       AccumT          accum);

        template<typename AVectorT,
                 typename BMatrixT,
                 typename CVectorT,
                 typename SringT,
                 typename AccumT >
        friend inline void vxm(AVectorT const &a,
                               BMatrixT const &b,
                               CVectorT       &c,
                               SringT          s,
                               AccumT          accum);


        template<typename AMatrixT,
                 typename BVectorT,
                 typename CVectorT,
                 typename SringT,
                 typename AccumT >
        friend inline void mxv(AMatrixT const &a,
                               BVectorT const &b,
                               CVectorT       &c,
                               SringT          s,
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

        template<typename AMatrixT,
                 typename SringT  >
        friend inline NegateView<AMatrixT, SringT> negate(
            AMatrixT const  &a,
            SringT const    &s);


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
        friend inline void extracttuples(
            AMatrixT const                             &a,
            IndexArrayType                             &i,
            IndexArrayType                             &j,
            std::vector<typename AMatrixT::ScalarType> &v);


        template<typename AMatrixT,
                 typename RAIteratorI,
                 typename RAIteratorJ,
                 typename RAIteratorV,
                 typename AccumT >
        friend inline void buildmatrix(AMatrixT    &m,
                                       RAIteratorI  i,
                                       RAIteratorJ  j,
                                       RAIteratorV  v,
                                       IndexType    n,
                                       AccumT       accum);

        template<typename AMatrixT,
                 typename AccumT >
        friend inline void buildmatrix(
            AMatrixT             &m,
            IndexArrayType const &i,
            IndexArrayType const &j,
            std::vector<typename MatrixT::ScalarType> const &v,
            AccumT                accum );

        template<typename AMatrixT, typename BMatrixT>
        friend void index_of(AMatrixT const  &A,
                             BMatrixT        &B,
                             IndexType const  base_index);

        template<typename AMatrixT>
        friend void col_index_of(AMatrixT &mat);

        template<typename AMatrixT>
        friend void row_index_of(AMatrixT &mat);

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

    //************************************************************************
    template<typename MatrixT>
    class TransposeView
    {
    public:
        typedef typename backend::TransposeView<
            typename MatrixT::BackendType> BackendType;
        typedef typename MatrixT::ScalarType ScalarType;

        /**
         * @brief Construct an empty matrix with the specified shape.
         *
         * @param[in] num_rows  Number of rows in the matrix
         * @param[in] num_cols  Number of columns in the matrix
         * @param[in] zero      The "zero" value, additive identity, and
         *                      the structural zero.
         */

        //note:
        //the backend should be able to decide when to ignore any of the
        //tags and/or arguments
        TransposeView(BackendType backend_view)
            : m_mat(backend_view)
        {
        }

        /**
         * @brief Copy constructor.
         *
         * @param[in] rhs   The matrix to copy.
         */
        TransposeView(TransposeView<MatrixT> const &rhs)
            : m_mat(rhs.m_mat)
        {
        }


        ~TransposeView() { }

        void get_shape(IndexType &num_rows, IndexType &num_cols) const
        {
            m_mat.get_shape(num_rows, num_cols);
        }

        std::pair<IndexType, IndexType> get_shape() const
        {
            IndexType num_rows, num_cols;
            m_mat.get_shape(num_rows, num_cols);
            return std::make_pair(num_rows, num_cols);
        }

        ScalarType get_zero() const
        {
            return m_mat.get_zero();
        }

        IndexType get_nnz() const
        {
            return m_mat.get_nnz();
        }

        ScalarType get_value_at(IndexType row, IndexType col) const
        {
            return m_mat.get_value_at(row, col);
        }

        //other methods that may or may not belong here:
        //
        void print_info(std::ostream &os) const
        {
            os << "Frontend TransposeView of:" << std::endl;
            m_mat.print_info(os);
        }

        /// @todo This does not need to be a friend
        friend std::ostream &operator<<(std::ostream &os, TransposeView const &mat)
        {
            mat.print_info(os);
            return os;
        }

        /// @todo need to change to mix and match internal types
        template <typename OtherMatrixT>
        bool operator==(OtherMatrixT const &rhs) const
        {
            return (m_mat.operator==(rhs));
        }

        template <typename OtherMatrixT>
        bool operator!=(OtherMatrixT const &rhs) const
        {
            return !(*this == rhs);
        }
        //end other methods


    private:
        BackendType m_mat;

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
                 typename SringT,
                 typename AccumT >
        friend inline void mxm(AMatrixT const &a,
                               BMatrixT const &b,
                               CMatrixT       &c,
                               SringT       s,
                               AccumT          accum);

        template<typename AMatrixT,
                 typename BMatrixT,
                 typename CMatrixT,
                 typename SringT,
                 typename AccumT >
        friend inline void mxv(AMatrixT const &a,
                               BMatrixT const &b,
                               CMatrixT       &c,
                               SringT          s,
                               AccumT          accum);


        template<typename AMatrixT,
                 typename BMatrixT,
                 typename CMatrixT,
                 typename MMatrixT,
                 typename SringT,
                 typename AccumT >
        friend inline void mxmMasked(AMatrixT const &a,
                                     BMatrixT const &b,
                                     CMatrixT       &c,
                                     MMatrixT const &m,
                                     SringT          s,
                                     AccumT          accum);


        template<typename AMatrixT,
                 typename BMatrixT,
                 typename CMatrixT,
                 typename MMatrixT,
                 typename SringT,
                 typename AccumT >
        friend inline void mxmMaskedV2(AMatrixT const &a,
                                       BMatrixT const &b,
                                       CMatrixT       &c,
                                       MMatrixT       &m,
                                       SringT          s,
                                       AccumT          accum);

        template<typename AVectorT,
                 typename BMatrixT,
                 typename CVectorT,
                 typename SringT,
                 typename AccumT >
        friend inline void vxm(AVectorT const &a,
                               BMatrixT const &b,
                               CVectorT       &c,
                               SringT          s,
                               AccumT          accum);


        template<typename AMatrixT,
                 typename BVectorT,
                 typename CVectorT,
                 typename SringT,
                 typename AccumT >
        friend inline void mxv(AMatrixT const &a,
                               BVectorT const &b,
                               CVectorT       &c,
                               SringT       s,
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

        template<typename AMatrixT,
                 typename SringT  >
        friend inline NegateView<AMatrixT, SringT> negate(
            AMatrixT const   &a,
            SringT const &s);


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
        friend inline void extracttuples(
            AMatrixT const                             &a,
            IndexArrayType                             &i,
            IndexArrayType                             &j,
            std::vector<typename AMatrixT::ScalarType> &v);

        template<typename AMatrixT,
                 typename RAIteratorI,
                 typename RAIteratorJ,
                 typename RAIteratorV,
                 typename AccumT >
        friend inline void buildmatrix(AMatrixT    &m,
                                       RAIteratorI  i,
                                       RAIteratorJ  j,
                                       RAIteratorV  v,
                                       IndexType    n,
                                       AccumT       accum);

        template<typename AMatrixT,
                 typename AccumT >
        friend inline void buildmatrix(
            AMatrixT             &m,
            IndexArrayType const &i,
            IndexArrayType const &j,
            std::vector<typename MatrixT::ScalarType> const &v,
            AccumT                accum );

        template<typename AMatrixT, typename BMatrixT>
        friend void index_of(AMatrixT const  &A,
                             BMatrixT        &B,
                             IndexType const  base_index);

        template<typename AMatrixT>
        friend void col_index_of(AMatrixT &mat);

        template<typename AMatrixT>
        friend void row_index_of(AMatrixT &mat);

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
