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

#ifndef SPIRALGRAPH_ITERATOR_SUPPORT_HPP
#define SPIRALGRAPH_ITERATOR_SUPPORT_HPP

#include <type_traits>

#include <graphblas/types.hpp>

namespace GraphBLAS
{
    // WE ARE NOT A RANDOM_ACCESS_ITERATOR
    // This is an iterator that returns numbers a range without fully realizing
    // that range in memory. It is not mutable.
    class index_generator
    {
        typedef int64_t difference_type;

    public:
        index_generator(IndexType value) : m_value(value)
        {

        }

        index_generator(const index_generator &other)
                : m_value(other.m_value)
        {
        }

        index_generator()
                : m_value(0)
        {
        }

        // reference operator*() const
        // const IndexType &operator*() const
        IndexType operator*() const
        {
            return m_value;
        }

        const index_generator &operator++()
        {
            ++m_value;
            return *this;
        }

        index_generator operator++(int)
        {
            index_generator copy(*this);
            ++(*this);
            return copy;
        }

        const index_generator &operator--()
        {
            --m_value;
            return *this;
        }

        index_generator operator--(int)
        {
            index_generator copy(*this);
            --(*this);
            return copy;
        }

        bool operator==(const index_generator &other) const
        {
            return m_value == other.m_value;
        }

        bool operator!=(const index_generator &other) const
        {
            return m_value != other.m_value;
        }

        index_generator &operator=(const index_generator &other)
        {
            m_value = other.m_value;
            return *this;
        }

        //  We don't need these based on the concept in the front end.
//        index_generator &operator+=(difference_type n)
//        {
//            m_value += n;
//            return *this;
//        }
//
//        index_generator operator+(difference_type n)
//        {
//            return index_generator(m_value + n);
//        }
//
//        index_generator &operator-=(difference_type n)
//        {
//            m_value -= n;
//            return *this;
//        }
//
//        index_generator operator-(difference_type n)
//        {
//            return index_generator(m_value - n);
//        }
//
//        difference_type operator-(index_generator &rhs)
//        {
//            return m_value - rhs.m_value;
//        }

        // reference operator[](difference_type n)
        // const IndexType &operator[](difference_type n)
        IndexType operator[](difference_type n)
        {
            return m_value + n;
        }

//        bool operator<(const index_generator &rhs) const
//        {
//            return m_value < rhs.m_value;
//        }
//
//        bool operator<=(const index_generator &rhs) const
//        {
//            return m_value <= rhs.m_value;
//        }
//
//        bool operator>(const index_generator &rhs)
//        {
//            return m_value > rhs.m_value;
//        }
//
//        bool operator>=(const index_generator &rhs)
//        {
//            return m_value >= rhs.m_value;
//        }

    private:
        IndexType m_value;

    };

    // ========================================================================

    // This class is used to provide a container for iterators that work
    // over a range. So, this isn't a real container, but something that
    // appears to be a container so we can pass around begin and end.
    class IndexSequenceRange
    {
    public:
        IndexSequenceRange(IndexType begin, IndexType end)
                : m_begin(begin), m_end(end)
        {
            // @TODO: Check begin < end
        }

        bool empty() const                          { return m_begin == m_end; }

        index_generator begin() const               { return index_generator(m_begin); }
        index_generator end() const                 { return index_generator(m_end); }

        IndexType size() const                      { return m_end - m_begin; }

        IndexType operator[](IndexType n) const     { return m_begin + n; }

        const IndexType m_begin;
        const IndexType m_end;
    };

    // We can sse these as type traits if we need to switch off all sequence
    template<typename>
    struct is_all_sequence
        : public std::false_type {};

    template<>
    struct is_all_sequence<AllIndices>
        : public std::true_type {};

    // Functions to dynamically see if this is a all sequence.

    template <typename T>
    bool IsAllSequence(T seq)
    {
        return false;
    };

    template <>
    bool IsAllSequence(AllIndices seq)
    {
        return true;
    };


} // namespace GraphBLAS

#endif //SPIRALGRAPH_ITERATOR_SUPPORT_HPP
