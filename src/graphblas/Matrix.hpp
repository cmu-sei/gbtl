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

#define GB_INCLUDE_BACKEND_MATRIX 1
#include <graphblas/backend_include.hpp>

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
                 typename RAIteratorVT>
        inline void extractTuples(RAIteratorIT        row_it,
                                  RAIteratorJT        col_it,
                                  RAIteratorVT        values) const
        {
            m_mat.extractTuples(row_it, col_it, values);
        }

        inline void extractTuples(IndexArrayType            &row_indices,
                                  IndexArrayType            &col_indices,
                                  std::vector<ScalarT>      &values) const
        {
            m_mat.extractTuples(row_indices.begin(),
                                col_indices.begin(),
                                values.begin());
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

        template<typename CMatrixT,
                 typename MaskT,
                 typename AccumT,
                 typename AMatrixT>
        friend inline void transpose(CMatrixT       &C,
                                     MaskT    const &Mask,
                                     AccumT          accum,
                                     AMatrixT const &A,
                                     bool            replace_flag);

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
    // GrB_NULL mask: should be GrB_FULL
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

    /**
     *  @brief Output the matrix in array form.  Mainly for debugging
     *         small matrices.
     *
     *  @param[in] ostr  The output stream to send the contents
     *  @param[in] mat   The matrix to output
     *  @param[in] label Optional label to output first.
     */
    template <typename MatrixT>
    void print_matrix(std::ostream      &ostr,
                      MatrixT const     &mat,
                      std::string const &label = "")
    {
        // The new backend doesn't have get_zero.   Should we have it???
        // ostr << label << ": zero = " << mat.m_mat.get_zero() << std::endl;
        ostr << label << " (" << mat.nrows() << "x" << mat.ncols() << ")"
             << std::endl;
        backend::pretty_print_matrix(ostr, mat.m_mat);
    }
} // end namespace GraphBLAS
