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

#ifndef GB_CUSP_ELEMENTREF_HPP
#define GB_CUSP_ELEMENTREF_HPP

#include <iosfwd>

namespace graphblas
{
    template<typename VectorT,
             typename ScalarT = typename VectorT::ScalarType>
    class ElementRef
    {
    public:
        typedef ScalarT ScalarType;

        ElementRef(IndexType element_index,
                   VectorT  &parent_vec)
            : m_element_index(element_index),
              m_parent_vec(parent_vec),
              m_value(parent_vec.get_value_at(m_element_index))
        {
        }

        // return by value?
        // This performs the cast necessary to turn an
        // ElementRef<ScalarT> into a ScalarType value
        operator ScalarType() const
        {
            return m_value;
        }

        // Any way to dispatch to the Row and then Matrix
        // class in turn, to avoid reaching inside other
        // classes?
        ElementRef<VectorT, ScalarT>& operator=(ScalarT const &rhs)
        {
            m_parent_vec.set_value_at(m_element_index, rhs);
            m_value = rhs;
            return *this;
        }

        ElementRef<VectorT, ScalarT>& operator=(ElementRef &rhs)
        {
            m_parent_vec.set_value_at(m_element_index, rhs.m_value);
            m_value = rhs.m_value;
            return *this;
        }

        bool operator==(ElementRef<VectorT, ScalarT> const &rhs) const
        {
            return m_value == rhs.m_value;
        }

        friend std::ostream& operator<<(
            std::ostream              &os,
            ElementRef<VectorT, ScalarT> const &elem)
        {
            os << elem.m_value;
            return os;
        }

    private:
        IndexType      m_element_index;
        VectorT       &m_parent_vec;
        ScalarType     m_value;
    };
} // graphblas

#endif // GB_CUSP_ELEMENTREF_HPP
