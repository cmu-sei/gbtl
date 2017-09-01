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

#ifndef GRAPHBLAS_INDEX_SUPPORT_HPP
#define GRAPHBLAS_INDEX_SUPPORT_HPP

#include <graphblas/types.hpp>
#include <capi/graphblas.h>

namespace GraphBLAS
{
    template <typename ContainerT>
    class IndicesSequenceProxy
    {
    public:
        typedef typename ContainerT::iterator       iterator;
        typedef IndexType difference_type;

        IndicesSequenceProxy(const ContainerT &container);

        bool empty() const { return false; }

        iterator begin() const { return SequenceIteratorConcept(); }
        iterator end() const { return SequenceIteratorConcept(); }

        IndexType size() const { return 0; }

        IndexType operator[](difference_type n) const { return 0; }

        const ContainerT &m_begin;
    };

    //@TODO: Figure out how to move this inside something appropriate
    enum IteratorStyle { VEC_IT, NUM_IT };

    union iterator_storage
    {
        std::vector<IndexType>::const_iterator    vec;
//        std::vector<IndexType>::iterator    vec;
        GraphBLAS::index_generator                           num;

        iterator_storage() { vec = std::vector<IndexType>::iterator(); num = index_generator(0); }
    };


    class IndexSequence
    {
    public:

        IndexSequence(const IndexSequence &other)
                : m_range(other.m_range), m_begin(other.m_begin), m_end(other.m_end), m_all(other.m_all)
        {
        }

        IndexSequence(const std::vector<IndexType> &range)
                : m_range(&range), m_begin(0), m_end(0), m_all(false)
        {
        }

        IndexSequence(const GrB_Index *indicies, const GrB_Index count)
                : m_range(NULL), m_begin(start), m_end(end), m_all(false)
        {
        }

        // Typedefs...
        typedef IndexType difference_type;

        // ===========   Concept methods

        // === Tier 1

        // This is the first one.  This isn't all, it has to support tier 2
        bool all() const
        {
            return m_all;
        }

        bool empty() const
        {
            if (m_range != NULL)
                return m_range->empty();
            else
                return m_begin == m_end;
        }

        // fwd decl so I can put all the methods first
        class iterator;

        // === Tier 2

        iterator begin() const
        {
            if (m_range != NULL)
                return iterator(m_range->begin());
            else
                return iterator(index_generator(m_begin));
        }

        iterator end() const
        {
            if (m_range != NULL)
                return iterator(m_range->end());
            else
                return iterator(index_generator(m_end));
        }

        IndexType size() const
        {
            if (m_range != NULL)
                return m_range->size();
            else
                return m_end - m_begin;
        }

        IndexType operator[](difference_type n) const
        {
            if (m_range != NULL)
                return (*m_range)[n];
            else
                return m_begin + n;
        }

        // =======================

        class iterator
        {
        public:
            typedef int64_t difference_type;

            iterator(std::vector<IndexType>::const_iterator vec_it)
//            iterator(std::vector<IndexType>::iterator vec_it)
                    : m_style(VEC_IT)
            {
                m_it.vec = vec_it;
            }

            //iterator(index_generator &num_it)
            iterator(index_generator num_it)
                    : m_style(NUM_IT)
            {
                m_it.num = num_it;
            }

            iterator(const iterator& other)
                    : m_style(other.m_style)
            {
                switch (m_style)
                {
                    case VEC_IT: m_it.vec = other.m_it.vec; break;
                    case NUM_IT: m_it.num = other.m_it.num; break;
                }
            }

//            bool empty()
//            {
////                switch (m_style)
////                {
////                    case VEC_IT: return m_it.vec->empty();
////                    case NUM_IT: return *m_it.num;
////                }
//                return false;
//            }

            IndexType operator*() const
            {
                switch (m_style)
                {
                    case VEC_IT: return *m_it.vec;
                    case NUM_IT: return *m_it.num;
                }
                return 0;
            }

            const iterator &operator++()
            {
                switch (m_style)
                {
                    case VEC_IT: ++m_it.vec; break;
                    case NUM_IT: ++m_it.num; break;
                }
                return *this;
            }

            iterator operator++(int)
            {
                iterator copy(*this);
                ++(*this);
                return copy;
            }

            const iterator &operator--()
            {
                switch (m_style)
                {
                    case VEC_IT: --m_it.vec; break;
                    case NUM_IT: --m_it.num; break;
                }
                return *this;
            }

            iterator operator--(int)
            {
                iterator copy(*this);
                --(*this);
                return copy;
            }

            bool operator==(const iterator &other) const
            {
                if (m_style != other.m_style)
                    return false;

                switch (m_style)
                {
                    case VEC_IT: return m_it.vec == other.m_it.vec;
                    case NUM_IT: return m_it.num == other.m_it.num;
                }

                return false;
            }

            bool operator!=(const iterator &other) const
            {
                return ! (*this == other);
            }

            iterator &operator=(const iterator &other)
            {
                m_style = other.m_style;
                switch (m_style)
                {
                    case VEC_IT: m_it.vec = other.m_it.vec; break;
                    case NUM_IT: m_it.num = other.m_it.num; break;
                }

                return *this;
            }

            iterator &operator+=(difference_type n)
            {
                switch (m_style)
                {
                    case VEC_IT: m_it.vec = m_it.vec += n; break;
                    case NUM_IT: m_it.num = m_it.num += n; break;
                }

                return *this;
            }

            iterator operator+(difference_type n)
            {
                switch (m_style)
                {
                    case VEC_IT: return iterator(m_it.vec + n);
                    case NUM_IT: return iterator(m_it.num + n);
                }

                return *this;
            }

            iterator &operator-=(difference_type n)
            {
                switch (m_style)
                {
                    case VEC_IT: m_it.vec = m_it.vec -= n; break;
                    case NUM_IT: m_it.num = m_it.num -= n; break;
                }

                return *this;
            }

            iterator operator-(difference_type n)
            {
                switch (m_style)
                {
                    case VEC_IT: return m_it.vec - n;
                    case NUM_IT: return m_it.num - n;
                }

                return *this;
            }

            difference_type operator-(iterator &rhs)
            {
                switch (m_style)
                {
                    case VEC_IT: return m_it.vec - rhs.m_it.vec;
                    case NUM_IT: return m_it.num - rhs.m_it.num;
                }

                return 0;
            }

//        reference operator[](difference_type n)
            IndexType operator[](difference_type n)
            {
                switch (m_style)
                {
                    case VEC_IT: return m_it.vec[n];
                    case NUM_IT: return m_it.num[n];
                }

                return 0;
            }

            bool operator<(const iterator &rhs) const
            {
                switch (m_style)
                {
                    case VEC_IT: return m_it.vec < rhs.m_it.vec;
                    case NUM_IT: return m_it.num < rhs.m_it.num;
                }

                return false;
            }

            bool operator<=(const iterator &rhs) const
            {
                switch (m_style)
                {
                    case VEC_IT: return m_it.vec <= rhs.m_it.vec;
                    case NUM_IT: return m_it.num <= rhs.m_it.num;
                }
                return false;
            }

            bool operator>(const iterator &rhs)
            {
                switch (m_style)
                {
                    case VEC_IT: return m_it.vec > rhs.m_it.vec;
                    case NUM_IT: return m_it.num > rhs.m_it.num;
                }
                return false;
            }

            bool operator>=(const iterator &rhs)
            {
                switch (m_style)
                {
                    case VEC_IT: return m_it.vec >= rhs.m_it.vec;
                    case NUM_IT: return m_it.num >= rhs.m_it.num;
                }
                return false;
            }

            std::ostream &print_info(std::ostream &os) const
            {
                // @TODO: Make internal print call and remove the need for friend-ness
                switch (m_style)
                {
                    case IteratorStyle::VEC_IT: os << *m_it.vec; break;
                    case IteratorStyle::NUM_IT: os << *m_it.num; break;
                }
                return os;
            }

        private:
            IteratorStyle                           m_style;
            iterator_storage m_it;
        }; // iterator

    private:

        const std::vector<IndexType> *const m_range;
        const IndexType m_begin;
        const IndexType m_end;
        const bool      m_all;


        // This is all so we can get the constructor for AllIndicies()
        IndexSequence(bool all)
                : m_range(NULL), m_begin(0), m_end(0), m_all(true)
        {

        }

        friend IndexSequence AllIndicies();

    };

} // end namespace GraphBLAS


#endif //GRAPHBLAS_INDEX_SUPPORT_HPP
