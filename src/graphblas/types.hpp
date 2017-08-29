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
#include <iostream>

namespace GraphBLAS
{
    typedef uint64_t IndexType;
    typedef std::vector<IndexType> IndexArrayType;

    //**************************************************************************
    struct NoAccumulate
    {
        // It doesn't really matter what the type is, it never gets executed.
        typedef bool result_type;
        inline bool operator()(bool lhs, bool rhs) { return true; }
    };

    //**************************************************************************
    // Sequence::iterator concept
    class SequenceIteratorConcept
    {
    public:
        typedef IndexType difference_type;

        IndexType operator*() const { return 0; }

        const SequenceIteratorConcept &operator++() { return *this; }
        SequenceIteratorConcept operator++(int)     { return *this; }

        const SequenceIteratorConcept &operator--() { return *this; }
        SequenceIteratorConcept operator--(int)     { return *this; }

        bool operator==(const SequenceIteratorConcept &other) const { return false; }
        bool operator!=(const SequenceIteratorConcept &other) const { return false; }

        SequenceIteratorConcept &operator=(const SequenceIteratorConcept &other) { return * this; }

        // If we support some of these, we can get slices like
        // begin() + size()/4
//        SequenceIteratorConcept &operator+=(difference_type n)
//        SequenceIteratorConcept operator+(difference_type n)
//
//        SequenceIteratorConcept &operator-=(difference_type n)
//        SequenceIteratorConcept operator-(difference_type n)
//
//        difference_type operator-(SequenceIteratorConcept &rhs)
//
//        IndexType operator[](difference_type n)
    };

    // Sequence concept
    class SequenceConcept
    {
    public:
        typedef IndexType difference_type;

        bool empty() const { return false; }

        SequenceIteratorConcept begin() const { return SequenceIteratorConcept(); }
        SequenceIteratorConcept end() const { return SequenceIteratorConcept(); }

        IndexType size() const { return 0; }

        IndexType operator[](difference_type n) const { return 0; }

        const IndexType m_begin;
        const IndexType m_end;
    };

    //**************************************************************************

    // Special marker classes to support AllIndices iteration.
    class AllIndicesIterator
    {
    public:
        typedef IndexType difference_type;

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

    class AllIndices
    {
    public:

        AllIndices() {}

        bool empty() const                      { return false; }

        AllIndicesIterator begin() const        { return AllIndicesIterator(); }
        AllIndicesIterator end() const          { return AllIndicesIterator(); }

        IndexType size() const                  { return 0; }

        IndexType operator[](IndexType n) const { return 0; }
    };


    // This is the "Matrix" class for this example
    struct matrix_tag {};

    // This is the "Vector" class for this example
    struct vector_tag {};

    // For logging
    std::ostream &operator<<(std::ostream &os, const AllIndices &allIndices)
    {
        os << "AllIndices";
        return os;
    }




//    template <>
//    struct is_matrix<GraphBLAS::Matrix>
//            : public std::true_type {} ;

//    template <typename T>
//    struct is_matrix
//    : public std::false_type {} ;
//

} // namespace GraphBLAS

namespace std
{

// @TODO; It seems that unit tests can't find this!
    inline std::ostream &
    operator<<(std::ostream &os, const std::vector<long unsigned int> vec)
    {
        bool first = true;
        for (auto it = vec.begin(); it != vec.end(); ++it)
        {
            os << (first ? "" : ",") << *it;
            first = false;
        }
        return os;
    }

} // namespace std

#endif // GB_TYPES_HPP
