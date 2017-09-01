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

#ifndef GRAPHBLAS_ITERATOR_SUPPORT_HPP
#define GRAPHBLAS_ITERATOR_SUPPORT_HPP

#include <type_traits>

#include <graphblas/types.hpp>

namespace GraphBLAS
{
    //**************************************************************************
    // @TODO: How do we get this into comments?
//    // Sequence::iterator concept
//    class SequenceIteratorConcept
//    {
//    public:
//        typedef IndexType difference_type;
//
//        IndexType operator*() const { return 0; }
//
//        const SequenceIteratorConcept &operator++() { return *this; }
//        SequenceIteratorConcept operator++(int)     { return *this; }
//
//        const SequenceIteratorConcept &operator--() { return *this; }
//        SequenceIteratorConcept operator--(int)     { return *this; }
//
//        bool operator==(const SequenceIteratorConcept &other) const { return false; }
//        bool operator!=(const SequenceIteratorConcept &other) const { return false; }
//
//        SequenceIteratorConcept &operator=(const SequenceIteratorConcept &other) { return * this; }
//
//        // If we support some of these, we can get slices like
//        // begin() + size()/4
//        IndexType operator[](difference_type n) {};
//    };
//
//    // Sequence concept
//    class SequenceConcept
//    {
//    public:
//        typedef IndexType difference_type;
//
//        bool empty() const { return false; }
//
//        SequenceIteratorConcept begin() const { return SequenceIteratorConcept(); }
//        SequenceIteratorConcept end() const { return SequenceIteratorConcept(); }
//
//        IndexType size() const { return 0; }
//
//        IndexType operator[](difference_type n) const { return 0; }
//
//        const IndexType m_begin;
//        const IndexType m_end;
//    };

    //**************************************************************************
    //**************************************************************************

    // WE ARE NOT A RANDOM_ACCESS_ITERATOR
    // This is an iterator that returns numbers a range without fully realizing
    // that range in memory. It is not mutable.
    class IndexGenerator
    {
        typedef int64_t difference_type;

    public:
        IndexGenerator(IndexType value) : m_value(value)
        {

        }

        IndexGenerator(const IndexGenerator &other)
                : m_value(other.m_value)
        {
        }

        IndexGenerator()
                : m_value(0)
        {
        }

        // reference operator*() const
        // const IndexType &operator*() const
        IndexType operator*() const
        {
            return m_value;
        }

        const IndexGenerator &operator++()
        {
            ++m_value;
            return *this;
        }

        IndexGenerator operator++(int)
        {
            IndexGenerator copy(*this);
            ++(*this);
            return copy;
        }

        const IndexGenerator &operator--()
        {
            --m_value;
            return *this;
        }

        IndexGenerator operator--(int)
        {
            IndexGenerator copy(*this);
            --(*this);
            return copy;
        }

        bool operator==(const IndexGenerator &other) const
        {
            return m_value == other.m_value;
        }

        bool operator!=(const IndexGenerator &other) const
        {
            return m_value != other.m_value;
        }

        IndexGenerator &operator=(const IndexGenerator &other)
        {
            m_value = other.m_value;
            return *this;
        }

        IndexType operator[](difference_type n)
        {
            return m_value + n;
        }

    private:
        IndexType m_value;

    };

    //**************************************************************************

    /** 
     * This class is used to provide a "container" for iterators that work
     * over a range without realizing the backing store.
     */
    class IndexSequenceRange
    {
    public:
        typedef IndexGenerator                      iterator;

        IndexSequenceRange(IndexType begin, IndexType end)
                : m_begin(begin), m_end(end)
        {
            // @TODO: Check begin < end
        }

        bool empty() const                  { return m_begin == m_end; }

        IndexGenerator begin() const        { return IndexGenerator(m_begin); }
        IndexGenerator end() const          { return IndexGenerator(m_end); }

        IndexType size() const              { return m_end - m_begin; }

        IndexType operator[](IndexType n) const     { return m_begin + n; }

        const IndexType m_begin;
        const IndexType m_end;
    };

    //**************************************************************************
    //**************************************************************************

    // Special marker classes to support AllIndices iteration.
    // NOTE: This should never be used, as we the AllIndicies never gets invoked.
    class AllIndicesIterator
    {
    public:
        IndexType operator*() const                            { return 0; }

        const AllIndicesIterator &operator++()                 { return *this; }
        AllIndicesIterator operator++(int)                     { return *this; }

        const AllIndicesIterator &operator--()                 { return *this; }
        AllIndicesIterator operator--(int)                     { return *this; }

        bool operator==(const AllIndicesIterator &other) const { return false; }
        bool operator!=(const AllIndicesIterator &other) const { return false; }

        AllIndicesIterator &operator=(const AllIndicesIterator &other)
        {
            return *this;
        }
    };

    //**************************************************************************

    /**
     * Special marker class that can be used to trigger the implementation
     * to use all the indices in source or dest (as is appropriate for the
     * API) when iterating.  For example, when doing an extract, AllIndicies()
     * can be provided instead of explicitly providing a range from 0..N of
     * the respective component.
     */
    class AllIndices
    {
    public:

        typedef AllIndicesIterator              iterator;

        AllIndices() {}

        bool empty() const                      { return false; }

        AllIndicesIterator begin() const        { return AllIndicesIterator(); }
        AllIndicesIterator end() const          { return AllIndicesIterator(); }

        IndexType size() const                  { return 0; }

        IndexType operator[](IndexType n) const { return 0; }
    };

    //**************************************************************************

    // For logging
    std::ostream &operator<<(std::ostream &os, const AllIndices &allIndices)
    {
        os << "AllIndices";
        return os;
    }

    //**************************************************************************
    //**************************************************************************

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

#endif //GRAPHBLAS_ITERATOR_SUPPORT_HPP
