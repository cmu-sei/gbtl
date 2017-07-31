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
#include <graphblas/Matrix.hpp>

#define GB_INCLUDE_BACKEND_COMPLEMENT_VIEW 1
#include <graphblas/backend_include.hpp>

//****************************************************************************
//****************************************************************************

namespace GraphBLAS
{
    //************************************************************************
    template<typename MatrixT>
    class MatrixComplementView
    {
    public:
        typedef typename backend::MatrixComplementView<
            typename MatrixT::BackendType> BackendType;
        typedef typename MatrixT::ScalarType ScalarType;

        //note:
        //the backend should be able to decide when to ignore any of the
        //tags and/or arguments
        MatrixComplementView(BackendType backend_view)
            : m_mat(backend_view)
        {
        }

        /**
         * @brief Copy constructor.
         *
         * @param[in] rhs   The matrix to copy.
         */
        MatrixComplementView(MatrixComplementView<MatrixT> const &rhs)
            : m_mat(rhs.m_mat)
        {
        }

        ~MatrixComplementView() { }

        /// @todo need to change to mix and match internal types
        template <typename OtherMatrixT>
        bool operator==(OtherMatrixT const &rhs) const
        {
            return (m_mat == rhs);
        }

        template <typename OtherMatrixT>
        bool operator!=(OtherMatrixT const &rhs) const
        {
            return !(*this == rhs);
        }

        IndexType nrows() const { return m_mat.nrows(); }
        IndexType ncols() const { return m_mat.ncols(); }
        IndexType nvals() const { return m_mat.nvals(); }

        bool hasElement(IndexType row, IndexType col) const
        {
            return (m_mat.hasElement(row,col));
        }

        ScalarType extractElement(IndexType row, IndexType col) const
        {
            return m_mat.extractElement(row, col);
        }

        template<typename RAIteratorIT,
                 typename RAIteratorJT,
                 typename RAIteratorVT,
                 typename AMatrixT>
        inline void extractTuples(RAIteratorIT        row_it,
                                  RAIteratorJT        col_it,
                                  RAIteratorVT        values)
        {
            m_mat.extractTuples(row_it, col_it, values);
        }

        template<typename ValueT,
                 typename AMatrixT>
        inline void extractTuples(IndexArrayType            &row_indices,
                                  IndexArrayType            &col_indices,
                                  std::vector<ValueT>       &values)
        {
            m_mat.extractTuples(row_indices, col_indices, values);
        }

        //other methods that may or may not belong here:
        //
        void printInfo(std::ostream &os) const
        {
            os << "Frontend MatrixComplementView of:";
            m_mat.printInfo(os);
        }

        /// @todo This does not need to be a friend
        friend std::ostream &operator<<(std::ostream               &os,
                                        MatrixComplementView const &mat)
        {
            mat.printInfo(os);
            return os;
        }

    private:
        BackendType m_mat;

        // PUT ALL FRIEND DECLARATIONS (that use matrix masks) HERE

        template<typename CMatrixT,
                 typename MaskT,
                 typename AccumT,
                 typename SemiringT,
                 typename AMatrixT,
                 typename BMatrixT>
        friend inline void mxm(CMatrixT         &C,
                               MaskT      const &Mask,
                               AccumT            accum,
                               SemiringT         op,
                               AMatrixT   const &A,
                               BMatrixT   const &B,
                               bool              replace_flag);

        //--------------------------------------------------------------------

        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  //can be BinaryOp, Monoid (not Semiring)
                 typename AMatrixT,
                 typename BMatrixT,
                 typename... CTagsT>
        friend inline void eWiseMult(
            GraphBLAS::Matrix<CScalarT, CTagsT...> &C,
            MaskT                            const &Mask,
            AccumT                                  accum,
            BinaryOpT                               op,
            AMatrixT                         const &A,
            BMatrixT                         const &B,
            bool                                    replace_flag);

        //--------------------------------------------------------------------

        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  //can be BinaryOp, Monoid (not Semiring)
                 typename AMatrixT,
                 typename BMatrixT,
                 typename... CTagsT>
        friend inline void eWiseAdd(
            GraphBLAS::Matrix<CScalarT, CTagsT...> &C,
            MaskT                            const &Mask,
            AccumT                                  accum,
            BinaryOpT                               op,
            AMatrixT                         const &A,
            BMatrixT                         const &B,
            bool                                    replace_flag);

        //--------------------------------------------------------------------

        template<typename CMatrixT,
                 typename MMatrixT,
                 typename AccumT,
                 typename AMatrixT >
        friend inline void extract(CMatrixT &C,
                                   MMatrixT const &mask,
                                   AccumT accum,
                                   AMatrixT const &A,
                                   IndexArrayType const &row_indicies,
                                   IndexArrayType const &col_indicies,
                                   bool replace);

        //--------------------------------------------------------------------

        template<typename CMatrixT,
                 typename MaskT,
                 typename AccumT,
                 typename AMatrixT>
        friend inline void assign(CMatrixT             &C,
                                  MaskT          const &Mask,
                                  AccumT                accum,
                                  AMatrixT       const &A,
                                  IndexArrayType const &row_indices,
                                  IndexArrayType const &col_indices,
                                  bool                  replace_flag);

        template<typename CMatrixT,
                 typename MaskT,
                 typename AccumT,
                 typename ValueT>
        friend inline void assign_constant(CMatrixT             &C,
                                           MaskT          const &Mask,
                                           AccumT                accum,
                                           ValueT                val,
                                           IndexArrayType const &row_indices,
                                           IndexArrayType const &col_indices,
                                           bool                  replace_flag);

        //--------------------------------------------------------------------

        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename UnaryFunctionT,
                 typename AMatrixT,
                 typename ...ATagsT>
        friend inline void apply(
            GraphBLAS::Matrix<CScalarT, ATagsT...>           &C,
            MaskT                                     const  &Mask,
            AccumT                                            accum,
            UnaryFunctionT                                    op,
            AMatrixT                                  const  &A,
            bool                                              replace_flag);

        //--------------------------------------------------------------------

    };

    //************************************************************************
    //************************************************************************
    template<typename VectorT>
    class VectorComplementView
    {
    public:
        typedef typename backend::VectorComplementView<
            typename VectorT::BackendType> BackendType;
        typedef typename VectorT::ScalarType ScalarType;

        //note:
        //the backend should be able to decide when to ignore any of the
        //tags and/or arguments
        VectorComplementView(BackendType backend_view)
            : m_vec(backend_view)
        {
        }

        /**
         * @brief Copy constructor.
         *
         * @param[in] rhs   The vector to copy.
         */
        VectorComplementView(VectorComplementView<VectorT> const &rhs)
            : m_vec(rhs.m_vec)
        {
        }

        ~VectorComplementView() { }

        /// @todo need to change to mix and match internal types
        template <typename OtherVectorT>
        bool operator==(OtherVectorT const &rhs) const
        {
            //return (m_vec.operator==(rhs));
            throw 1;
            ///@todo Not implemented yet
            //return vector_equal_helper(*this, rhs);
        }

        template <typename OtherVectorT>
        bool operator!=(OtherVectorT const &rhs) const
        {
            return !(*this == rhs);
        }

        IndexType size() const  { return m_vec.size(); }
        IndexType nvals() const { return m_vec.nvals(); }

        bool hasElement(IndexType index) const
        {
            return m_vec.hasElement(index);
        }

        ScalarType extractElement(IndexType index) const
        {
            return m_vec.extractElement(index);
        }

        template<typename RAIteratorIT,
                 typename RAIteratorVT>
        inline void extractTuples(RAIteratorIT        i_it,
                                  RAIteratorVT        v_it)
        {
            m_vec.extractTuples(i_it, v_it);
        }

        template<typename ValueT>
        inline void extractTuples(IndexArrayType            &indices,
                                  std::vector<ValueT>       &values)
        {
            m_vec.extractTuples(indices, values);
        }

        //other methods that may or may not belong here:
        //
        void printInfo(std::ostream &os) const
        {
            os << "Frontend VectorComplementView of:";
            m_vec.printInfo(os);
        }

        /// @todo This does not need to be a friend
        friend std::ostream &operator<<(std::ostream               &os,
                                        VectorComplementView const &vec)
        {
            vec.printInfo(os);
            return os;
        }

    private:
        BackendType m_vec;

        // PUT ALL FRIEND DECLARATIONS (that use vector masks) HERE

        //--------------------------------------------------------------------
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

        //--------------------------------------------------------------------

        template<typename WScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  //can be BinaryOp, Monoid (not Semiring)
                 typename UVectorT,
                 typename VVectorT,
                 typename ...WTagsT>
        friend inline void eWiseMult(
            GraphBLAS::Vector<WScalarT, WTagsT...> &w,
            MaskT                            const &mask,
            AccumT                                  accum,
            BinaryOpT                               op,
            UVectorT                         const &u,
            VVectorT                         const &v,
            bool                                    replace_flag);

        template<typename WScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  //can be BinaryOp, Monoid (not Semiring)
                 typename UVectorT,
                 typename VVectorT,
                 typename ...WTagsT>
        friend inline void eWiseAdd(
            GraphBLAS::Vector<WScalarT, WTagsT...> &w,
            MaskT                            const &mask,
            AccumT                                  accum,
            BinaryOpT                               op,
            UVectorT                         const &u,
            VVectorT                         const &v,
            bool                                    replace_flag);

        //--------------------------------------------------------------------

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

        //--------------------------------------------------------------------

        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename ValueT>
        friend inline void assign_constant(WVectorT             &w,
                                           MaskT          const &mask,
                                           AccumT                accum,
                                           ValueT                val,
                                           IndexArrayType const &indices,
                                           bool                  replace_flag);

        //--------------------------------------------------------------------
        template<typename WScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename UnaryFunctionT,
                 typename UVectorT,
                 typename ...WTagsT>
        friend inline void apply(GraphBLAS::Vector<WScalarT, WTagsT...> &w,
                                 MaskT                            const &mask,
                                 AccumT                                  accum,
                                 UnaryFunctionT                          op,
                                 UVectorT                         const &u,
                                 bool                                    replace_flag);

        //--------------------------------------------------------------------

        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  // monoid or binary op only
                 typename AMatrixT>
        friend inline void reduce(WVectorT        &w,
                                  MaskT     const &mask,
                                  AccumT           accum,
                                  BinaryOpT        op,
                                  AMatrixT  const &A,
                                  bool             replace_flag);
    };

} // end namespace GraphBLAS
