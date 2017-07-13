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
/**
 * @brief Some basic typedefs.
 */

#ifndef GB_TYPES_HPP
#define GB_TYPES_HPP

#include <cstdint>
#include <exception>
#include <vector>

namespace GraphBLAS
{
    typedef uint64_t IndexType;
    typedef std::vector<IndexType> IndexArrayType;

    //**************************************************************************
    // GrB_ALL: We just keep an empty one.
    extern const IndexArrayType GrB_ALL;

    //**************************************************************************
    // GrB_NULL accum
    struct NoAccumulate
    {
        // It doesn't really matter what the type is, it never gets executed.
        typedef bool result_type;
        inline bool operator()(bool lhs, bool rhs) { return true; }
    };

    //**************************************************************************
    // Used by GrB_ALL
    // this should go somewhere else.  C++ should have something like this...
    class index_iterator
    {
    public:
        index_iterator(IndexType value) : m_value(value) {}

        index_iterator(const index_iterator &other) : m_value(other.m_value) {}

        IndexType operator*() const { return m_value; }

        const index_iterator &operator++()
        {
            ++m_value;
            return *this;
        }

        index_iterator operator++(int)
        {
            index_iterator copy(*this);
            ++(*this);
            return copy;
        }

        bool operator==(const index_iterator &other) const
        {
            return m_value == other.m_value;
        }

        bool operator!=(const index_iterator &other) const
        {
            return m_value != other.m_value;
        }

        index_iterator &operator=(const index_iterator &other)
        {
            m_value = other.m_value;
            return *this;
        }

    private:
        IndexType m_value;
    };
}

#endif // GB_TYPES_HPP
