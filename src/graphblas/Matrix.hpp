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
#include <graphblas/matrix_utils.hpp>
#include <graphblas/View.hpp>

// Include matrix definitions from the appropriate backend.
#define __GB_SYSTEM_MATRIX_HEADER <graphblas/system/__GB_SYSTEM_ROOT/Matrix.hpp>
#include __GB_SYSTEM_MATRIX_HEADER
#undef __GB_SYSTEM_MATRIX_HEADER

//****************************************************************************
// The new namespace
//****************************************************************************

namespace GraphBLAS
{
    //************************************************************************
    template<typename ScalarT, typename... TagsT>
    class Matrix
    {
    public:
        typedef ScalarT ScalarType;
        typedef typename detail::matrix_generator::result<
            ScalarT,
            detail::SparsenessCategoryTag,
            detail::DirectednessCategoryTag,
            TagsT... ,
            detail::NullTag,
            detail::NullTag >::type BackendType;

        /**
         * @brief Construct an empty matrix with the specified shape.
         *
         * @note The backend should be able to decide when to ignore any of the
         *       tags and/or arguments.
         *
         * @param[in] num_rows  Number of rows in the matrix
         * @param[in] num_cols  Number of columns in the matrix
         * @param[in] zero      The "zero" value, additive identity, and
         *                      the structural zero.
         */
        Matrix(IndexType num_rows, IndexType num_cols)
            : m_mat(num_rows, num_cols)
        {
        }

        /**
         * @brief Copy constructor.
         *
         * @param[in] rhs   The matrix to copy.
         */
        Matrix(Matrix<ScalarT, TagsT...> const &rhs)
            : m_mat(rhs.m_mat)
        {
        }

        /**
         * @brief Construct a dense matrix from dense data
         *
         * @param[in] values The dense matrix from which to construct a
         *                   sparse matrix from.
         *
         * @todo Should we really support this interface?
         */
        Matrix(std::vector<std::vector<ScalarT> > const &values)
            : m_mat(values)
        {
        }

        /**
         * @brief Construct a sparse matrix from dense data.
         *
         * @param[in] values The dense matrix from which to construct a
         *                   sparse matrix from.
         * @param[in] zero   The "zero" value.
         *
         * @todo Should we really support this interface?
         */
        Matrix(std::vector<std::vector<ScalarT> > const &values, ScalarT zero)
            : m_mat(values, zero)
        {
        }

        ~Matrix() { }

        /// Version 1 of getshape that assigns to two passed parameters
        void get_shape(IndexType &num_rows, IndexType &num_cols) const
        {
            m_mat.get_shape(num_rows, num_cols);
        }

        /// Version 2 of getshape that returns a std::pair = [rows, cols]
        std::pair<IndexType, IndexType> get_shape() const
        {
            IndexType num_rows, num_cols;
            m_mat.get_shape(num_rows, num_cols);
            return std::make_pair(num_rows, num_cols);
        }

        /**
         * Populate the matrix with stored values (using iterators).
         *
         * @param[in]  i_it  Row index iterator
         * @param[in]  j_it  Column index iterator
         * @param[in]  v_it  Value (scalar) iterator
         * @param[in]  n     Number of elements to store
         * @param[in]  accum binary function to call when computing value to
         *                   store. Takes current value and incoming value
         *                   as input.
         *
         * @todo need to add a parameter to handle duplicate locations.
         * @todo Should this clear out all previous storage if accum is Assign?
         */
        template<typename RAIteratorI,
                 typename RAIteratorJ,
                 typename RAIteratorV,
                 typename DupT = GraphBLAS::Second<ScalarType> >
        void build(RAIteratorI  i_it,
                   RAIteratorJ  j_it,
                   RAIteratorV  v_it,
                   IndexType    n,
                   DupT         dup = DupT())
        {
            m_mat.build(i_it, j_it, v_it, n, dup);
        }

        /// @todo Should assignment work only if dimensions are same?
        Matrix<ScalarT, TagsT...> &
        operator=(Matrix<ScalarT, TagsT...> const &rhs)
        {
            if (this != &rhs)
            {
                // backend currently doing dimension check.
                m_mat = rhs.m_mat;
            }
            return *this;
        }


        /// Assignment from dense data
        /// @todo This ignores the structural zero value.
        //Matrix<ScalarT, TagsT...>& operator=(
        //    std::vector<std::vector<ScalarT> > const &rhs)
        //{
        //    m_mat = rhs;
        //    return *this;
        //}

        void clear() { m_mat.clear(); }

        IndexType nrows() const  { return m_mat.nrows(); }
        IndexType ncols() const  { return m_mat.ncols(); }
        IndexType nvals() const  { return m_mat.nvals(); }

        /// @todo need to change to mix and match internal types
        bool operator==(Matrix<ScalarT, TagsT...> const &rhs) const
        {
            return (m_mat == rhs.m_mat);
        }

        bool operator!=(Matrix<ScalarT, TagsT...> const &rhs) const
        {
            //return !(m_mat == rhs.m_mat);
            return !(*this == rhs);
        }

        /// @todo I don't think this is a valid interface for sparse
        ScalarT extractElement(IndexType row, IndexType col) const
        {
            return m_mat.extractElement(row, col);
        }

        /// @todo I don't think this is a valid interface for sparse
        void assignElement(IndexType row, IndexType col, ScalarT const &val)
        {
            m_mat.setElement(row, col, val);
        }

        /// This replaces operator<< and outputs implementation specific
        /// information.
        void print_info(std::ostream &os) const
        {
            m_mat.print_info(os);
        }

        /// @todo This does not need to be a friend
        friend std::ostream &operator<<(std::ostream &os, Matrix const &mat)
        {
            mat.print_info(os);
            return os;
        }

    private:
        template<typename CMatrixT,
                 typename RAIteratorIT,
                 typename RAIteratorJT,
                 typename RAIteratorVT,
                 typename BinaryOpT>
        friend inline void matrixBuild(CMatrixT           &C,
                                       RAIteratorIT        row_indices,
                                       RAIteratorJT        col_indices,
                                       RAIteratorVT        values,
                                       IndexType           num_vals,
                                       BinaryOpT           dup);


        template<typename CMatrixT,
                 typename ValueT,
                 typename BinaryOpT>
        friend inline void matrixBuild(CMatrixT                   &C,
                                       IndexArrayType       const &row_indices,
                                       IndexArrayType       const &col_indices,
                                       std::vector<ValueT>  const &values,
                                       BinaryOpT                   dup);

        template<typename CMatrixT,
                 typename AccumT,
                 typename SemiringT,
                 typename AMatrixT,
                 typename BMatrixT>
        friend inline void mxm(CMatrixT         &C,
                               AccumT            accum,
                               SemiringT         op,
                               AMatrixT   const &A,
                               BMatrixT   const &B);

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

        // .... ADD OTHER OPERATIONS AS FRIENDS AS THEY ARE IMPLEMENTED .....

        template <typename MatrixT>
        friend void print_matrix(std::ostream      &ostr,
                                 MatrixT const     &mat,
                                 std::string const &label);

    private:
        BackendType m_mat;
    };

} // end namespace GraphBLAS

//****************************************************************************
// The deprecated namespace -- scroll down for the new namespace object
//****************************************************************************

namespace graphblas
{

    //************************************************************************
    template<typename ScalarT, typename... TagsT>
    class Matrix
    {
    public:
        typedef ScalarT ScalarType;
        typedef typename detail::matrix_generator::result<
            ScalarT,
            detail::SparsenessCategoryTag,
            detail::DirectednessCategoryTag,
            TagsT... ,
            detail::NullTag,
            detail::NullTag >::type BackendType;

        /**
         * @brief Construct an empty matrix with the specified shape.
         *
         * @note The backend should be able to decide when to ignore any of the
         *       tags and/or arguments.
         *
         * @param[in] num_rows  Number of rows in the matrix
         * @param[in] num_cols  Number of columns in the matrix
         * @param[in] zero      The "zero" value, additive identity, and
         *                      the structural zero.
         */
        Matrix(IndexType num_rows,
               IndexType num_cols,
               ScalarT   zero = static_cast<ScalarT>(0))
            : m_mat(num_rows, num_cols, zero)
        {
            m_mat.set_zero(zero);
        }

        /**
         * @brief Construct a matrix from a given dense matrix.
         *
         * @param[in] values The dense matrix from which to construct a
         *                   sparse matrix from.
         * @param[in] zero   The "zero" value.
         *
         * @todo Should we really support this interface?
         */
        Matrix(std::vector<std::vector<ScalarT> > const &values,
               ScalarT    zero = static_cast<ScalarT>(0))
            : m_mat(values, zero)
        {
            m_mat.set_zero(zero);
        }

        /**
         * @brief Copy constructor.
         *
         * @param[in] rhs   The matrix to copy.
         */
        Matrix(Matrix<ScalarT, TagsT...> const &rhs)
            : m_mat(rhs.m_mat)
        {
        }

        ~Matrix() { }

        /**
         * Populate the matrix with stored values (using iterators).
         *
         * @param[in]  i_it  Row index iterator
         * @param[in]  j_it  Column index iterator
         * @param[in]  v_it  Value (scalar) iterator
         * @param[in]  n     Number of elements to store
         * @param[in]  accum binary function to call when computing value to
         *                   store. Takes current value and incoming value
         *                   as input.
         *
         * @todo need to add a parameter to handle duplicate locations.
         * @todo Should this clear out all previous storage if accum is Assign?
         */
        template<typename RAIteratorI,
                 typename RAIteratorJ,
                 typename RAIteratorV,
                 typename AccumT = graphblas::math::Assign<ScalarType> >
        void buildmatrix(RAIteratorI  i_it,
                         RAIteratorJ  j_it,
                         RAIteratorV  v_it,
                         IndexType    n,
                         AccumT       accum = AccumT())
        {
            m_mat.buildmatrix(i_it, j_it, v_it, n, accum);
        }

        /// @todo Should assignment work only if dimensions are same?
        Matrix<ScalarT, TagsT...>
        operator=(Matrix<ScalarT, TagsT...> const &rhs)
        {
            if (this != &rhs)
            {
                m_mat = rhs.m_mat;
            }
            return *this;
        }


        /// Assignment from dense data
        /// @todo This ignores the structural zero value.
        Matrix<ScalarT, TagsT...>& operator=(
            std::vector<std::vector<ScalarT> > const &rhs)
        {
            m_mat = rhs;
            return *this;
        }

        /// Version 1 of getshape that assigns to two passed parameters
        void get_shape(IndexType &num_rows, IndexType &num_cols) const
        {
            m_mat.get_shape(num_rows, num_cols);
        }

        /// Version 2 of getshape that returns a std::pair = [rows, cols]
        std::pair<IndexType, IndexType> get_shape() const
        {
            IndexType num_rows, num_cols;
            m_mat.get_shape(num_rows, num_cols);
            return std::make_pair(num_rows, num_cols);
        }

        ScalarT get_zero() const { return m_mat.get_zero(); }
        void set_zero(ScalarT new_zero) { m_mat.set_zero(new_zero); }

        /// Get the number of stored values (including stored zeros).
        IndexType get_nnz() const { return m_mat.get_nnz(); }

        /// @todo need to change to mix and match internal types
        bool operator==(Matrix<ScalarT, TagsT...> const &rhs) const
        {
            //return (m_mat == rhs.m_mat);
            return matrix_equal_helper(*this, rhs);
        }

        bool operator!=(Matrix<ScalarT, TagsT...> const &rhs) const
        {
            //return !(m_mat == rhs.m_mat);
            return !(*this == rhs);
        }

        /// @todo I don't think this is a valid interface for sparse
        ScalarT extractElement(IndexType row, IndexType col) const
        {
            return m_mat.extractElement(row, col);
        }

        /// @todo I don't think this is a valid interface for sparse
        void setElement(IndexType row, IndexType col, ScalarT const &val)
        {
            m_mat.setElement(row, col, val);
        }

        /// This replaces operator<< and outputs implementation specific
        /// information.
        void print_info(std::ostream &os) const
        {
            m_mat.print_info(os);
        }

        /// @todo This does not need to be a friend
        friend std::ostream &operator<<(std::ostream &os, Matrix const &mat)
        {
            mat.print_info(os);
            return os;
        }

    private:
        BackendType m_mat;

        template <typename AMatrixT, typename BMatrixT>
        friend bool matrix_equal_helper(
            const AMatrixT& a,
            const BMatrixT& b);

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
                 typename MMatrixT,
                 typename MonoidT,
                 typename AccumT >
        friend inline void ewisemultMasked(AMatrixT const &a,
                                           BMatrixT const &b,
                                           CMatrixT       &c,
                                           MMatrixT const &m,
                                           bool            replace_flag,
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
        friend inline void extracttuples(
            AMatrixT const                             &a,
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
        friend inline void buildmatrix(
            MatrixT              &m,
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

        template <typename MatrixT>
        friend void print_matrix(std::ostream      &ostr,
                                 MatrixT const     &mat,
                                 std::string const &label);

    };

} // end namespace graphblas

