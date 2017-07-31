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

#define GB_INCLUDE_BACKEND_VECTOR 1
#include <graphblas/backend_include.hpp>

namespace GraphBLAS
{
    template<typename ScalarT, typename... TagsT>
    class Vector;

    template<typename VectorT>
    class VectorComplementView;

    //**************************************************************************
    template<typename ScalarT, typename... TagsT>
    class Vector
    {
    public:
        typedef ScalarT ScalarType;
        typedef typename detail::vector_generator::result<
            ScalarT,
            detail::SparsenessCategoryTag,
            TagsT... ,
            detail::NullTag >::type BackendType;
        //typedef GraphBLAS::backend::Vector<ScalarT> BackendType;

        Vector() = delete;

        Vector(IndexType const & nsize) : m_vec(nsize) {}

        /**
         * @brief Construct a dense vector with 'count' copies of 'value'
         *
         * @note Calls backend constructor.
         *
         * @param[in]  count  Number of elements in the vector.
         * @param[in]  value  The scalar value to store in each element
         */
        Vector(IndexType const &count, ScalarT const &value)
            : m_vec(count, value)
        {
        }

        /**
         * @brief Construct a dense vector from dense data
         *
         * @param[in] values The dense vector from which to construct a
         *                   sparse vector from.
         *
         * @todo Should we really support this interface?
         */
        Vector(std::vector<ScalarT> const &values)
            : m_vec(values)
        {
        }

        /**
         * @brief Construct a sparse vector from dense data and a sentinel zero value.
         *
         * @param[in] values The dense vector from which to construct a
         *                   sparse vector from.
         * @param[in] zero   The "zero" value used to determine implied
         *                   zeroes (no stored value) in the sparse structure
         *
         * @todo Should we really support this interface?
         */
        Vector(std::vector<ScalarT> const &values, ScalarT zero)
            : m_vec(values, zero)
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

        /**
         * Populate the vector with stored values (using iterators).
         *
         * @param[in]  i_it      index iterator
         * @param[in]  v_it      Value (scalar) iterator
         * @param[in]  num_vals  Number of elements to store
         * @param[in]  dup       Binary function to call when value is being stored
         *                       in a location that already has a stored value.
         *                       stored_val = dup(stored_val, *v_it)
         *
         * @todo The C spec says it is an error to call build on a non-empty
         *       vector.  Unclear if the C++ should.
         */
        template<typename RAIteratorI,
                 typename RAIteratorV,
                 typename BinaryOpT = GraphBLAS::Second<ScalarType> >
        void build(RAIteratorI  i_it,
                   RAIteratorV  v_it,
                   IndexType    num_vals,
                   BinaryOpT    dup = BinaryOpT())
        {
            m_vec.build(i_it, v_it, num_vals, dup);
        }

        /**
         * Populate the vector with stored values (using iterators).
         *
         * @param[in]  indices   Array of indices
         * @param[in]  values    Array of values
         * @param[in]  dup       binary function to call when value is being stored
         *                       in a location that already has a stored value.
         *                       stored_val = dup(stored_val, *v_it)
         *
         * @todo The C spec says it is an error to call build on a non-empty
         *       vector.  Unclear if the C++ should.
         */
        template<typename ValueT,
                 typename BinaryOpT = GraphBLAS::Second<ScalarType> >
        inline void build(IndexArrayType       const &indices,
                          std::vector<ValueT>  const &values,
                          BinaryOpT                   dup = BinaryOpT())
        {
            if (indices.size() != values.size())
            {
                throw DimensionException("Vector::build");
            }
            m_vec.build(indices.begin(), values.begin(), values.size(), dup);
        }

        void clear() { m_vec.clear(); }

        IndexType size() const   { return m_vec.size(); }
        IndexType nvals() const  { return m_vec.nvals(); }

        bool hasElement(IndexType index) const
        {
            return m_vec.hasElement(index);
        }

        void setElement(IndexType      index,
                        ScalarT const &new_val)
        {
            m_vec.setElement(index, new_val);
        }

        /// @throw NoValueException if there is no value stored at (row,col)
        ScalarT extractElement(IndexType index) const
        {
            return m_vec.extractElement(index);
        }

        template<typename RAIteratorIT,
                 typename RAIteratorVT>
        void extractTuples(RAIteratorIT        i_it,
                           RAIteratorVT        v_it) const
        {
            m_vec.extractTuples(i_it, v_it);
        }

        void extractTuples(IndexArrayType        &indices,
                           std::vector<ScalarT>  &values) const
        {
            m_vec.extractTuples(indices, values);
        }

        /// This replaces operator<< and outputs implementation specific
        /// information.
        void printInfo(std::ostream &os) const
        {
            m_vec.printInfo(os);
        }

        /// @todo This does not need to be a friend
        friend std::ostream &operator<<(std::ostream &os, Vector const &vec)
        {
            vec.printInfo(os);
            return os;
        }

    private:
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
        friend inline void assign_constant(WVectorT             &w,
                                           MaskT          const &mask,
                                           AccumT                accum,
                                           ValueT                val,
                                           IndexArrayType const &indices,
                                           bool                  replace_flag);

        // vector variant
        template<typename WScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename UnaryFunctionT,
                 typename UVectorT,
                 typename ...WTagsT>
        friend inline void apply(
            GraphBLAS::Vector<WScalarT, WTagsT...> &w,
            MaskT                            const &mask,
            AccumT                                  accum,
            UnaryFunctionT                          op,
            UVectorT                         const &u,
            bool                                    replace_flag);

        // row reduce matrix to column vector (use transpose for col reduce)
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

        // vector-scalar variant
        template<typename ValueT,
                 typename AccumT,
                 typename MonoidT, // monoid only
                 typename UScalarT,
                 typename ...UTagsT>
        friend inline void reduce(
            ValueT                                       &dst,
            AccumT                                        accum,
            MonoidT                                       op,
            GraphBLAS::Vector<UScalarT, UTagsT...> const &u);

        template<typename OtherScalarT, typename... OtherTagsT>
        friend inline VectorComplementView<Vector<OtherScalarT,
                                                  OtherTagsT...>> complement(
            Vector<OtherScalarT, OtherTagsT...> const &mask);

        // .... ADD OTHER OPERATIONS AS FRIENDS AS THEY ARE IMPLEMENTED .....

    private:
        BackendType m_vec;
    };

    /**
     *  @brief Output the vector in array form.  Mainly for debugging
     *         small vectors.
     *
     *  @param[in] ostr  The output stream to send the contents
     *  @param[in] vec   The vector to output
     *  @param[in] label Optional label to output first.
     */
    template <typename VectorT>
    void print_vector(std::ostream      &ostr,
                      VectorT const     &vec,
                      std::string const &label = "")
    {
        // The new backend doesn't have get_zero.   Should we have it???
        // ostr << label << ": zero = " << vec.m_vec.get_zero() << std::endl;
        ostr << label << ":" << std::endl;
        vec.printInfo(ostr);
    }
} // end namespace GraphBLAS
