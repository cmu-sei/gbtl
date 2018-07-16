/*
 * GraphBLAS Template Library, Version 2.0
 *
 * Copyright 2018 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors. All Rights Reserved.
 *
 * THIS MATERIAL WAS PREPARED AS AN ACCOUNT OF WORK SPONSORED BY AN AGENCY OF
 * THE UNITED STATES GOVERNMENT.  NEITHER THE UNITED STATES GOVERNMENT NOR THE
 * UNITED STATES DEPARTMENT OF ENERGY, NOR THE UNITED STATES DEPARTMENT OF
 * DEFENSE, NOR CARNEGIE MELLON UNIVERSITY, NOR BATTELLE, NOR ANY OF THEIR
 * EMPLOYEES, NOR ANY JURISDICTION OR ORGANIZATION THAT HAS COOPERATED IN THE
 * DEVELOPMENT OF THESE MATERIALS, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS,
 * OR USEFULNESS OR ANY INFORMATION, APPARATUS, PRODUCT, SOFTWARE, OR PROCESS
 * DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED
 * RIGHTS..
 *
 * Released under a BSD (SEI)-style license, please see license.txt or contact
 * permission@sei.cmu.edu for full terms.
 *
 * This release is an update of:
 *
 * 1. GraphBLAS Template Library (GBTL)
 * (https://github.com/cmu-sei/gbtl/blob/1.0.0/LICENSE) Copyright 2015 Carnegie
 * Mellon University and The Trustees of Indiana. DM17-0037, DM-0002659
 *
 * DM18-0559
 */

#pragma once

#include <cstddef>
#include <type_traits>
#include <graphblas/detail/config.hpp>
#include <graphblas/detail/param_unpack.hpp>

#define GB_INCLUDE_BACKEND_MATRIX 1
#include <backend_include.hpp>


//****************************************************************************
// The new namespace
//****************************************************************************

namespace GraphBLAS
{
    // We need to declare class so we can include later.
    template<typename ScalarT, typename... TagsT>
    class Vector;

    template<typename ScalarT, typename... TagsT>
    class Matrix;

    template<typename MatrixT>
    class TransposeView;

    template<typename MatrixT>
    class MatrixComplementView;

    //************************************************************************

    /**
     * @brief Frontend Matrix class. Performs API checks and forwards to
     *        backend code.
     *
     * @note The backend should be able to decide when to ignore any of the
     *       template tags and/or arguments.
     *
     */
    template<typename ScalarT, typename... TagsT>
    class Matrix
    {
    public:
        typedef matrix_tag  tag_type;

        typedef ScalarT     ScalarType;
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
        inline Info build(IndexArrayType       const &row_indices,
                          IndexArrayType       const &col_indices,
                          std::vector<ValueT>  const &values,
                          BinaryOpT                   dup = BinaryOpT())
        {
            if ((row_indices.size() != col_indices.size()) ||
                (row_indices.size() != values.size()))
            {
                //throw DimensionException("Matrix::build");
                return DIMENSION_MISMATCH;
            }

            return m_mat.build(row_indices.begin(), col_indices.begin(),
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

        template <typename RowSequenceT,
                  typename ColSequenceT>
        inline void extractTuples(RowSequenceT            &row_indices,
                                  ColSequenceT            &col_indices,
                                  std::vector<ScalarT>    &values) const
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

    private:

        // 4.3.1:
        template<typename CMatrixT,
                 typename MaskT,
                 typename AccumT,
                 typename SemiringT,
                 typename AMatrixT,
                 typename BMatrixT>
        friend inline Info mxm(CMatrixT         &C,
                               MaskT      const &Mask,
                               AccumT            accum,
                               SemiringT         op,
                               AMatrixT   const &A,
                               BMatrixT   const &B,
                               bool              replace_flag);

        //--------------------------------------------------------------------

        // 4.3.2:
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename SemiringT,
                 typename UVectorT,
                 typename AMatrixT>
        friend inline Info vxm(WVectorT         &w,
                               MaskT      const &mask,
                               AccumT            accum,
                               SemiringT         op,
                               UVectorT   const &u,
                               AMatrixT   const &A,
                               bool              replace_flag);

        //--------------------------------------------------------------------

        // 4.3.3
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename SemiringT,
                 typename AMatrixT,
                 typename UVectorT>
        friend inline Info mxv(WVectorT        &w,
                               MaskT     const &mask,
                               AccumT           accum,
                               SemiringT        op,
                               AMatrixT  const &A,
                               UVectorT  const &u,
                               bool             replace_flag);

        //--------------------------------------------------------------------

        // 4.3.4.2:
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  //can be BinaryOp, Monoid (not Semiring)
                 typename AMatrixT,
                 typename BMatrixT,
                 typename... CTagsT>
        friend inline Info eWiseMult(
            GraphBLAS::Matrix<CScalarT, CTagsT...> &C,
            MaskT                            const &Mask,
            AccumT                                  accum,
            BinaryOpT                               op,
            AMatrixT                         const &A,
            BMatrixT                         const &B,
            bool                                    replace_flag);

        //--------------------------------------------------------------------

        // 4.3.5.2
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  //can be BinaryOp, Monoid (not Semiring)
                 typename AMatrixT,
                 typename BMatrixT,
                 typename... CTagsT>
        friend inline Info eWiseAdd(
            GraphBLAS::Matrix<CScalarT, CTagsT...> &C,
            MaskT                            const &Mask,
            AccumT                                  accum,
            BinaryOpT                               op,
            AMatrixT                         const &A,
            BMatrixT                         const &B,
            bool                                    replace_flag);

        //--------------------------------------------------------------------

        // 4.3.6.2
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename AMatrixT,
                 typename RowSequenceT,
                 typename ColSequenceT,
                 typename ...CTags>
        friend inline Info extract(
                GraphBLAS::Matrix<CScalarT, CTags...>   &C,
                MaskT               const   &Mask,
                AccumT                       accum,
                AMatrixT            const   &A,
                RowSequenceT        const   &row_indices,
                ColSequenceT        const   &col_indices,
                bool                         replace_flag);

        // 4.3.6.3
        template<typename WScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename AMatrixT,
                 typename SequenceT,
                 typename ...WTags>
        friend inline Info extract(
                GraphBLAS::Vector<WScalarT, WTags...> &w,
                MaskT          const &mask,
                AccumT                accum,
                AMatrixT       const &A,
                SequenceT      const &row_indices,
                IndexType             col_index,
                bool                  replace_flag);

        //--------------------------------------------------------------------
        // 4.3.7.2
        template<typename CMatrixT,
                 typename MaskT,
                 typename AccumT,
                 typename AMatrixT,
                 typename RowSequenceT,
                 typename ColSequenceT,
                 typename std::enable_if<
                     std::is_same<matrix_tag,
                                  typename AMatrixT::tag_type>::value,
                     int>::type>
        friend inline Info assign(CMatrixT              &C,
                                  MaskT           const &Mask,
                                  AccumT                 accum,
                                  AMatrixT        const &A,
                                  RowSequenceT    const &row_indices,
                                  ColSequenceT    const &col_indices,
                                  bool                   replace_flag);

        // 4.3.7.3:
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename UVectorT,
                 typename SequenceT,
                 typename ...CTags>
        friend inline Info assign(Matrix<CScalarT, CTags...>  &C,
                                  MaskT                 const &mask,  // a vector
                                  AccumT                       accum,
                                  UVectorT              const &u,
                                  SequenceT             const &row_indices,
                                  IndexType                    col_index,
                                  bool                         replace_flag);

        // 4.3.7.4:
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename UVectorT,
                 typename SequenceT,
                 typename ...CTags>
        friend inline Info assign(Matrix<CScalarT, CTags...>  &C,
                                  MaskT                 const &mask,  // a vector
                                  AccumT                       accum,
                                  UVectorT              const &u,
                                  IndexType                    row_index,
                                  SequenceT             const &col_indices,
                                  bool                         replace_flag);

        // 4.3.7.6
        template<typename CMatrixT,
                 typename MaskT,
                 typename AccumT,
                 typename ValueT,
                 typename RowSequenceT,
                 typename ColSequenceT,
                 typename std::enable_if<
                     std::is_convertible<ValueT,
                                         typename CMatrixT::ScalarType>::value,
                     int>::type>
        friend inline Info assign(CMatrixT             &C,
                                  MaskT          const &Mask,
                                  AccumT                accum,
                                  ValueT                val,
                                  RowSequenceT   const &row_indices,
                                  ColSequenceT   const &col_indices,
                                  bool                  replace_flag);

        //--------------------------------------------------------------------

        // 4.3.8.2:
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename UnaryFunctionT,
                 typename AMatrixT,
                 typename... ATagsT>
        friend inline Info apply(
                GraphBLAS::Matrix<CScalarT, ATagsT...>          &C,
                MaskT                                   const   &Mask,
                AccumT                                           accum,
                UnaryFunctionT                                   op,
                AMatrixT                                const   &A,
                bool                                             replace_flag);

        //--------------------------------------------------------------------

        // 4.3.9.1
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  // monoid or binary op only
                 typename AMatrixT>
        friend inline Info reduce(WVectorT        &u,
                                  MaskT     const &mask,
                                  AccumT           accum,
                                  BinaryOpT        op,
                                  AMatrixT  const &A,
                                  bool             replace_flag);

        // 4.3.9.3
        template<typename ValueT,
                 typename AccumT,
                 typename MonoidT, // monoid only
                 typename AScalarT,
                 typename... ATagsT>
        friend inline Info reduce(
            ValueT                                       &dst,
            AccumT                                        accum,
            MonoidT                                       op,
            GraphBLAS::Matrix<AScalarT, ATagsT...> const &A);

        //--------------------------------------------------------------------

        // 4.3.10
        template<typename CMatrixT,
                 typename MaskT,
                 typename AccumT,
                 typename AMatrixT>
        friend inline Info transpose(CMatrixT       &C,
                                     MaskT    const &Mask,
                                     AccumT          accum,
                                     AMatrixT const &A,
                                     bool            replace_flag);

        //--------------------------------------------------------------------

        template<typename MatrixT>
        friend inline GraphBLAS::TransposeView<MatrixT> transpose(MatrixT const &A);


        //--------------------------------------------------------------------

        template<typename OtherScalarT, typename... OtherTagsT>
        friend inline MatrixComplementView<Matrix<OtherScalarT, OtherTagsT...> >
            complement(Matrix<OtherScalarT, OtherTagsT...> const &Mask);

        //--------------------------------------------------------------------

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



    // ================================================

    /**
     *  @brief Output the matrix in array form.  Mainly for debugging
     *         small matrices.
     *
     *  @param[in] ostr  The output stream to send the contents
     *  @param[in] mat   The matrix to output
     *  @param[in] label Optional label to output first.
     *
     *  @deprecated - use ostream inserter
     */
    template <typename MatrixT>
    void print_matrix(std::ostream      &ostr,
                      MatrixT const     &mat,
                      std::string const &label = "")
    {
        // ostr << label << ": zero = " << mat.m_mat.get_zero() << std::endl;
        ostr << label << " (" << mat.nrows() << "x" << mat.ncols() << ")"
             << std::endl;
        backend::pretty_print_matrix(ostr, mat.m_mat);
    }


    template<typename ScalarT, typename... TagsT>
    std::ostream &operator<<(std::ostream &os, const Matrix<ScalarT, TagsT...> &mat)
    {
        mat.printInfo(os);
        return os;
    }

} // end namespace GraphBLAS
