/*
 * GraphBLAS Template Library, Version 2.1
 *
 * Copyright 2019 Carnegie Mellon University, Battelle Memorial Institute, and
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
#include <graphblas/types.hpp>

#define GB_INCLUDE_BACKEND_VECTOR 1
#include <backend_include.hpp>

namespace GraphBLAS
{
    //**************************************************************************
    template<typename ScalarT, typename... TagsT>
    class Vector
    {
    public:
        using ScalarType = ScalarT;
        using BackendType = typename detail::vector_generator::result<
            ScalarT,
            detail::SparsenessCategoryTag,
            TagsT... ,
            detail::NullTag >::type;

        // current equivalent:
        //using BackendType = GraphBLAS::backend::Vector<ScalarT>;

        Vector() = delete;

        Vector(IndexType nsize) : m_vec(nsize) {}

        /**
         * @brief Construct a dense vector with 'count' copies of 'value'
         *
         * @note Calls backend constructor.
         *
         * @param[in]  count  Number of elements in the vector.
         * @param[in]  value  The scalar value to store in each element
         */
        Vector(IndexType count, ScalarT const &value)
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

        /**
         * @brief Resize the vector (smaller or larger)
         *
         * @param[in]  new_size  New size of the dimension (zero is invalid)
         *
         */
        void resize(IndexType new_size)
        {
            if (new_size == 0)
                throw InvalidValueException();

            m_vec.resize(new_size);
        }

        bool hasElement(IndexType index) const
        {
            return m_vec.hasElement(index);
        }

        void setElement(IndexType index, ScalarT const &new_val)
        {
            m_vec.setElement(index, new_val);
        }

        void removeElement(IndexType index)
        {
            m_vec.removeElement(index);
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

        // ================================================
        void printInfo(std::ostream &ostr) const
        {
            ostr << "GraphBLAS::Vector: ";
            m_vec.printInfo(ostr);
        }

        friend std::ostream &operator<<(std::ostream &ostr, Vector const &vec)
        {
            vec.printInfo(ostr);
            return ostr;
        }

    private:
        BackendType m_vec;

        // FRIEND FUNCTIONS

        friend inline BackendType &get_internal_vector(Vector &vector)
        {
            return vector.m_vec;
        }

        friend inline BackendType const &get_internal_vector(Vector const &vector)
        {
            return vector.m_vec;
        }
    };

    /// @deprecated
    template<typename ScalarT, typename... TagsT>
    void print_vector(std::ostream                    &ostr,
                      Vector<ScalarT, TagsT...> const &vec,
                      std::string const               &label = "")
        {
            ostr << label << ":" << std::endl;
            ostr << vec;
            ostr << std::endl;
        }

} // end namespace GraphBLAS
