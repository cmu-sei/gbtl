/*
 * GraphBLAS Template Library (GBTL), Version 3.0
 *
 * Copyright 2020 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors.
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
 * RIGHTS.
 *
 * Released under a BSD-style license, please see LICENSE file or contact
 * permission@sei.cmu.edu for full terms.
 *
 * [DISTRIBUTION STATEMENT A] This material has been approved for public release
 * and unlimited distribution.  Please see Copyright notice for non-US
 * Government use and distribution.
 *
 * DM20-0442
 */

#pragma once

#include <cstddef>
#include <type_traits>
#include <graphblas/detail/config.hpp>
#include <graphblas/detail/param_unpack.hpp>
#include <graphblas/types.hpp>

#define GB_INCLUDE_BACKEND_MATRIX 1
#include <backend_include.hpp>

namespace grb
{
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
        using ScalarType = ScalarT;
        using BackendType = typename detail::matrix_generator::result<
            ScalarT,
            detail::SparsenessCategoryTag,
            detail::DirectednessCategoryTag,
            TagsT... ,
            detail::NullTag,
            detail::NullTag >::type;

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
                 typename BinaryOpT = grb::Second<ScalarType> >
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
                 typename BinaryOpT = grb::Second<ScalarType> >
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

        /**
         * @brief Resize the matrix dimensions (smaller or larger)
         *
         * @param[in]  new_num_rows  New number of rows (zero is invalid)
         * @param[in]  new_num_cols  New number of columns (zero is invalid)
         *
         */
        void resize(IndexType new_num_rows, IndexType new_num_cols)
        {
            if ((new_num_rows == 0) || (new_num_cols == 0))
                throw InvalidValueException();

            m_mat.resize(new_num_rows, new_num_cols);
        }

        bool hasElement(IndexType row, IndexType col) const
        {
            return m_mat.hasElement(row, col);
        }

        void setElement(IndexType row, IndexType col, ScalarT const &val)
        {
            m_mat.setElement(row, col, val);
        }

        void removeElement(IndexType row, IndexType col)
        {
            m_mat.removeElement(row, col);
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

        // ================================================
        void printInfo(std::ostream &ostr) const
        {
            ostr << "grb::Matrix: ";
            m_mat.printInfo(ostr);
        }

        friend std::ostream &operator<<(std::ostream &ostr, Matrix const &mat)
        {
            mat.printInfo(ostr);
            return ostr;
        }

    private:
        BackendType m_mat;

        // FRIEND FUNCTIONS

        friend inline BackendType &get_internal_matrix(Matrix &matrix)
        {
            return matrix.m_mat;
        }

        friend inline BackendType const &get_internal_matrix(Matrix const &matrix)
        {
            return matrix.m_mat;
        }

    };

    /// @deprecated
    template<typename ScalarT, typename... TagsT>
    void print_matrix(std::ostream                     &ostr,
                      Matrix<ScalarT, TagsT...> const  &mat,
                      std::string const                &label = "")
    {
        ostr << label << ":" << std::endl;
        ostr << mat;
        ostr << std::endl;
    }

} // end namespace grb
