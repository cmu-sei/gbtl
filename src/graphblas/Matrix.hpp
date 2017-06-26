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
#include <graphblas/utility.hpp>
//#include <graphblas/TransposeView.hpp>
//#include <graphblas/ComplementView.hpp>
#include <graphblas/View.hpp> // deprecated

// Include matrix definitions from the appropriate backend.
#define __GB_SYSTEM_MATRIX_HEADER <graphblas/system/__GB_SYSTEM_ROOT/Matrix.hpp>
#include __GB_SYSTEM_MATRIX_HEADER
#undef __GB_SYSTEM_MATRIX_HEADER

//****************************************************************************
// The new namespace
//****************************************************************************

namespace GraphBLAS
{
    template<typename ScalarT, typename... TagsT>
    class Matrix;

    template<typename MatrixT>
    class TransposeView;

    template<typename MatriT>
    class MatrixComplementView;

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
         * @brief Construct a sparse matrix from dense data and a sentinel zero value.
         *
         * @param[in] values The dense matrix from which to construct a
         *                   sparse matrix from.
         * @param[in] zero   The "zero" value used to determine implied
         *                   zeroes (no stored value) in the sparse structure
         *
         * @todo Should we really support this interface?
         */
        Matrix(std::vector<std::vector<ScalarT> > const &values, ScalarT zero)
            : m_mat(values, zero)
        {
        }

        ~Matrix() { }

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

        /// Version 1 of getshape that assigns to two passed parameters
        //void get_shape(IndexType &num_rows, IndexType &num_cols) const
        //{
        //    m_mat.get_shape(num_rows, num_cols);
        //}

        /// Version 2 of getshape that returns a std::pair = [rows, cols]
        //std::pair<IndexType, IndexType> get_shape() const
        //{
        //    IndexType num_rows, num_cols;
        //    m_mat.get_shape(num_rows, num_cols);
        //    return std::make_pair(num_rows, num_cols);
        //}

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

        /**
         * Populate the matrix with stored values (using iterators).
         *
         * @param[in]  i_it      Row index iterator
         * @param[in]  j_it      Column index iterator
         * @param[in]  v_it      Value (scalar) iterator
         * @param[in]  num_vals  Number of elements to store
         * @param[in]  dup       Binary function to call when value is being stored
         *                       in a location that already has a stored value.
         *                       stored_val = dup(stored_val, *v_it)
         *
         * @todo The C spec says it is an error to call build on a non-empty
         *       matrix.  Unclear if the C++ should.
         */
        template<typename RAIteratorI,
                 typename RAIteratorJ,
                 typename RAIteratorV,
                 typename BinaryOpT = GraphBLAS::Second<ScalarType> >
        void build(RAIteratorI  i_it,
                   RAIteratorJ  j_it,
                   RAIteratorV  v_it,
                   IndexType    num_vals,
                   BinaryOpT    dup = BinaryOpT())
        {
            m_mat.build(i_it, j_it, v_it, num_vals, dup);
        }

        /**
         * Populate the matrix with stored values (using iterators).
         *
         * @param[in]  row_indices  Array of row indices
         * @param[in]  col_indices  Array of column indices
         * @param[in]  values       Array of values
         * @param[in]  dup          binary function to call when value is being stored
         *                          in a location that already has a stored value.
         *                          stored_val = dup(stored_val, *v_it)
         *
         * @todo The C spec says it is an error to call build on a non-empty
         *       matrix.  Unclear if the C++ should.
         */
        template<typename ValueT,
                 typename BinaryOpT = GraphBLAS::Second<ScalarType> >
        inline void build(IndexArrayType       const &row_indices,
                          IndexArrayType       const &col_indices,
                          std::vector<ValueT>  const &values,
                          BinaryOpT                   dup = BinaryOpT())
        {
            if ((row_indices.size() != col_indices.size()) ||
                (row_indices.size() != values.size()))
            {
                throw DimensionException("Matrix::build");
            }

            m_mat.build(row_indices.begin(), col_indices.begin(),
                        values.begin(), values.size(), dup);
        }

        void clear() { m_mat.clear(); }

        IndexType nrows() const  { return m_mat.nrows(); }
        IndexType ncols() const  { return m_mat.ncols(); }
        IndexType nvals() const  { return m_mat.nvals(); }

        bool hasElement(IndexType row, IndexType col) const
        {
            return m_mat.hasElement(row, col);
        }

        /// @todo I don't think this is a valid interface for sparse
        void setElement(IndexType row, IndexType col, ScalarT const &val)
        {
            m_mat.setElement(row, col, val);
        }

        /// @throw NoValueException if there is no value stored at (row,col)
        ScalarT extractElement(IndexType row, IndexType col) const
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
            /// @todo
        }

        template<typename ValueT,
                 typename AMatrixT>
        inline void extractTuples(IndexArrayType            &row_indices,
                                  IndexArrayType            &col_indices,
                                  std::vector<ValueT>       &values)
        {
            /// @todo
        }

        /// This replaces operator<< and outputs implementation specific
        /// information.
        void printInfo(std::ostream &os) const
        {
            m_mat.printInfo(os);
        }

        /// @todo This does not need to be a friend
        friend std::ostream &operator<<(std::ostream &os, Matrix const &mat)
        {
            mat.printInfo(os);
            return os;
        }

    private:

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

        template<typename WVectorT,
                 typename AccumT,
                 typename SemiringT,
                 typename UVectorT,
                 typename AMatrixT>
        friend inline void vxm(WVectorT         &w,
                               AccumT            accum,
                               SemiringT         op,
                               UVectorT   const &u,
                               AMatrixT   const &A);

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

        //--------------------------------------------------------------------

        template<typename WVectorT,
                 typename AccumT,
                 typename SemiringT,
                 typename AMatrixT,
                 typename UVectorT>
        friend inline void mxv(WVectorT        &w,
                               AccumT           accum,
                               SemiringT        op,
                               AMatrixT  const &A,
                               UVectorT  const &u);

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
                 typename... ATagsT>
        friend inline void apply(
                GraphBLAS::Matrix<CScalarT, ATagsT...>          &C,
                MaskT                                   const   &Mask,
                AccumT                                           accum,
                UnaryFunctionT                                   op,
                AMatrixT                                const   &A,
                bool                                             replace_flag);

        //--------------------------------------------------------------------

        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  // monoid or binary op only
                 typename AMatrixT>
        friend inline void reduce(WVectorT        &u,
                                  MaskT     const &mask,
                                  AccumT           accum,
                                  BinaryOpT        op,
                                  AMatrixT  const &A,
                                  bool             replace_flag);

        template<typename ValueT,
                 typename AccumT,
                 typename MonoidT, // monoid only
                 typename AScalarT,
                 typename... ATagsT>
        friend inline void reduce(
            ValueT                                       &dst,
            AccumT                                        accum,
            MonoidT                                       op,
            GraphBLAS::Matrix<AScalarT, ATagsT...> const &A);

        //--------------------------------------------------------------------

        template<typename MatrixT>
        friend inline TransposeView<MatrixT> transpose(MatrixT const &A);

        template<typename OtherScalarT, typename... OtherTagsT>
        friend inline MatrixComplementView<Matrix<OtherScalarT, OtherTagsT...> >
            complement(Matrix<OtherScalarT, OtherTagsT...> const &Mask);

        // .... ADD OTHER OPERATIONS AS FRIENDS AS THEY ARE IMPLEMENTED .....

        template <typename MatrixT>
        friend void print_matrix(std::ostream      &ostr,
                                 MatrixT const     &mat,
                                 std::string const &label);

    private:
        BackendType m_mat;
    };

    //**************************************************************************
    // @todo We should move this to a different file (types.hpp or graphblas.hpp?)
    class NoMask
    {
    public:
        typedef bool ScalarType; // not necessary?
        typedef backend::NoMask BackendType; // not necessary?

        backend::NoMask m_mat;  // can be const?
        backend::NoMask m_vec;
    };


    //**************************************************************************
    // Currently these won't work because of include order.
    /// @todo move all these to backend::sparse_helpers

    template <typename M1, typename M2>
    void check_nrows_nrows(M1 m1, M2 m2)
    {
        if (m1.nrows() != m2.nrows())
            throw DimensionException("nrows doesn't match nrows");
    }

    template <typename M1>
    void check_nrows_nrows(M1 m1, NoMask mask)
    {
        // No op
    }

    template <typename M1, typename M2>
    void check_ncols_ncols(M1 m1, M2 m2)
    {
        if (m1.ncols() != m2.ncols())
            throw DimensionException("ncols doesn't match ncols");
    }

    template <typename M1>
    void check_ncols_ncols(M1 m1, NoMask mask)
    {
        // No op
    }

    template <typename M1, typename M2>
    void check_ncols_nrows(M1 m1, M2 m2)
    {
        if (m1.ncols() != m2.nrows())
            throw DimensionException("ncols doesn't match nrows");
    };

    template <typename M>
    void check_ncols_nrows(M m1, NoMask mask)
    {
        // No op
    };

    template <typename M>
    void check_ncols_nrows(NoMask m1, M mask)
    {
        // No op
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
