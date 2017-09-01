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
#include <graphblas/indices.hpp>
#include <capi/graphblas.h>

namespace GraphBLAS
{
    /**
     * Set of internal classes to support wrapping the supplied index vector
     * or providing a range from 0 to ncols.
     */

    //@TODO: Figure out how to move this inside something appropriate
    enum IteratorStyle { PTR_IT, NUM_IT };

    union iterator_storage
    {
        const GrB_Index                         *ptr;
        GraphBLAS::IndexGenerator                num;

        iterator_storage()
        {
            ptr = NULL;
            num = GraphBLAS::IndexGenerator(0);
        }
    };

    class IndexProxy
    {
    public:

        IndexProxy(const GrB_Index *indicies, const GrB_Index size)
                : m_indicies(indicies), m_size(size)
        {
        }

        // Typedefs...
        typedef IndexType difference_type;


        // fwd decl so I can put all the methods first
        class iterator;

        // ===========   Concept methods

        bool empty() const
        {
            return m_indicies == NULL && m_size == 0;
        }

        iterator begin() const
        {
            if (m_indicies == NULL)
                return iterator(GraphBLAS::IndexGenerator(0));

            return iterator(m_indicies);
        }

        iterator end() const
        {
            if (m_indicies == NULL)
                return iterator(GraphBLAS::IndexGenerator(m_size));

            return iterator(m_indicies + m_size);
        }

        IndexType size() const
        {
            return m_size;
        }

        IndexType operator[](difference_type n) const
        {
            if (m_indicies == NULL)
                return n;

            return m_indicies[n];
        }

        // =======================

        class iterator
        {
        public:
            typedef int64_t difference_type;

            iterator(const GrB_Index *vec_it)
                    : m_style(PTR_IT)
            {
                m_it.ptr = vec_it;
            }

            iterator(GraphBLAS::IndexGenerator num_it)
                    : m_style(NUM_IT)
            {
                m_it.num = num_it;
            }

            iterator(const iterator& other)
                    : m_style(other.m_style)
            {
                switch (m_style)
                {
                    case PTR_IT: m_it.ptr = other.m_it.ptr; break;
                    case NUM_IT: m_it.num = other.m_it.num; break;
                }
            }

            IndexType operator*() const
            {
                switch (m_style)
                {
                    case PTR_IT: return *m_it.ptr;
                    case NUM_IT: return *m_it.num;
                }
                return 0;
            }

            const iterator &operator++()
            {
                switch (m_style)
                {
                    case PTR_IT: ++m_it.ptr; break;
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
                    case PTR_IT: --m_it.ptr; break;
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
                    case PTR_IT: return m_it.ptr == other.m_it.ptr;
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
                    case PTR_IT: m_it.ptr = other.m_it.ptr; break;
                    case NUM_IT: m_it.num = other.m_it.num; break;
                }

                return *this;
            }

            IndexType operator[](difference_type n)
            {
                switch (m_style)
                {
                    case PTR_IT: return m_it.ptr[n];
                    case NUM_IT: return m_it.num[n];
                }

                return 0;
            }

            std::ostream &print_info(std::ostream &os) const
            {
                // @TODO: Make internal print call and remove the need for friend-ness
                switch (m_style)
                {
                    case IteratorStyle::PTR_IT: os << *m_it.ptr; break;
                    case IteratorStyle::NUM_IT: os << *m_it.num; break;
                }
                return os;
            }

        private:
            IteratorStyle               m_style;
            iterator_storage            m_it;
        }; // iterator

    private:

        const GrB_Index *m_indicies;
        const GrB_Index m_size;
    };

} // end namespace GraphBLAS


#endif //GRAPHBLAS_INDEX_SUPPORT_HPP
