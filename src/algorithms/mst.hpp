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

#ifndef MST_HPP
#define MST_HPP

#include <utility>
#include <vector>
#include <iostream>

#include <graphblas/graphblas.hpp>

//****************************************************************************
namespace algorithms
{
    //************************************************************************
    template <typename T>
    using MSTType = std::pair<GraphBLAS::IndexType, T>;

    template <typename T>
    std::ostream &operator<<(std::ostream &ostr, MSTType<T> const &mst)
    {
        ostr << "(" << mst.first << "," << mst.second << ")";
        return ostr;
    }

    //************************************************************************
    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct MSTPlus
    {
        typedef D3 result_type;
        inline D3 operator()(D1 const &lhs, MSTType<D2> const &rhs)
        {
            return lhs + rhs.second;
        }
    };

    //************************************************************************
    template<typename D1>
    struct MSTMin
    {
        typedef MSTType<D1> result_type;
        inline MSTType<D1> operator()(MSTType<D1> const &lhs,
                                      MSTType<D1> const &rhs)
        {
            return (rhs.second > lhs.second) ? lhs : rhs;
        }
    };


    //************************************************************************
    /// position of a single minimum element in vector
    /// @todo Need to find a more GraphBLAS-like way to do this
    template <typename T>
    GraphBLAS::IndexType argmin(GraphBLAS::Vector<T> const &v)
    {
        if (v.nvals() == 0)
        {
            throw GraphBLAS::PanicException();
        }

        std::vector<T> vals(v.nvals());
        GraphBLAS::IndexArrayType idx(v.nvals());
        v.extractTuples(idx.begin(), vals.begin());

        GraphBLAS::IndexType idx_min = idx[0];
        T val_min = vals[0];

        for (GraphBLAS::IndexType i = 1; i < vals.size(); ++i)
        {
            if (vals[i] < val_min)
            {
                val_min = vals[i];
                idx_min = idx[i];
            }
        }
        return idx_min;
    }

    /**
     * @brief Compute the weight of the minimal spanning tree for the given
     *        graph.
     *
     * This function takes in the adjacency matrix representation of a
     * graph, and computes the weight of the minimal spanning tree of
     * the graph.  This is done using Prim's algorithm.
     *
     * Prim's algorithm works by growing a single set of vertices
     * (call it S) which belong to the spanning tree.  On each iteration,
     * we add the closest vertex to S not already in S, and add the
     * weight of the edge between that vertex and its parent to the weight
     * of the minimal spanning tree.
     *
     * @param[in]  graph        The graph to perform the computation on.
     * @param[out] mst_parents  Parent list
     *
     * @return The weight of the minimum spanning tree of the graph,
     *         followed by a parent list containing the edges in the
     *         minimum spanning tree.
     */
    template<typename MatrixT>
    typename MatrixT::ScalarType mst(
        MatrixT const                           &graph,
        GraphBLAS::Vector<GraphBLAS::IndexType> &mst_parents)
    {
        using T = typename MatrixT::ScalarType;

        GraphBLAS::IndexType rows(graph.nrows());
        GraphBLAS::IndexType cols(graph.ncols());

        if ((rows != cols) || (rows != mst_parents.size()))
        {
            throw GraphBLAS::DimensionException();
        }
        //GraphBLAS::print_matrix(std::cout, graph, "GRAPH***");

        // Create an adjacency matrix of the correct type (combining
        // source vertex ID with edge weight
        GraphBLAS::IndexArrayType i(graph.nvals());
        GraphBLAS::IndexArrayType j(graph.nvals());
        std::vector<T> vals(graph.nvals());
        graph.extractTuples(i.begin(), j.begin(), vals.begin());
        std::vector<MSTType<T>> new_vals;
        for (GraphBLAS::IndexType ix = 0; ix < i.size(); ++ix)
        {
            new_vals.push_back(std::make_pair(i[ix], vals[ix]));
        }
        GraphBLAS::Matrix<MSTType<T>> A(rows, cols);
        A.build(i, j, new_vals);
        //GraphBLAS::print_matrix(std::cout, A, "Hybrid A matrix");

        // chose some arbitrary vertex to start
        GraphBLAS::IndexType const START_NODE = 0;

        T weight = static_cast<T>(0);

        GraphBLAS::Vector<MSTType<T>> d(rows);
        GraphBLAS::Vector<bool> mask(rows);
        mask.setElement(START_NODE, true);
        //GraphBLAS::print_vector(std::cout, mask, "Initial mask");

        // Using complement of mask so we don't deal with max()
        GraphBLAS::Vector<T> s(rows);
        GraphBLAS::assign_constant(s,
                                   GraphBLAS::complement(mask),
                                   GraphBLAS::NoAccumulate(),
                                   0, GraphBLAS::GrB_ALL, true);
        //GraphBLAS::print_vector(std::cout, s, "Initial s");

        mst_parents.clear();

        // Get the START NODE's neighbors
        GraphBLAS::extract(d,
                           GraphBLAS::NoMask(), // complement(mask)?
                           GraphBLAS::NoAccumulate(),
                           GraphBLAS::transpose(A),
                           GraphBLAS::GrB_ALL, START_NODE);
        //GraphBLAS::print_vector(std::cout, d, "Initial d");

        while (mask.nvals() < rows)
        {
            //std::cout << "===================== ITERATION === s.nvals() = "
            //          << s.nvals() << std::endl;
            GraphBLAS::Vector<T> temp(rows);

            // Note that eWiseMult is used with Plus b/c implied value is infinity
            GraphBLAS::eWiseMult(temp,
                                 GraphBLAS::NoMask(),
                                 GraphBLAS::NoAccumulate(),
                                 MSTPlus<T>(),
                                 s, d);

            GraphBLAS::IndexType u = argmin(temp);
            //GraphBLAS::print_vector(std::cout, temp, "-------- s + d.second");
            //std::cout << "argmin(s + d.second) = " << u << std::endl;

            mask.setElement(u, true);
            //GraphBLAS::print_vector(std::cout, mask, "-------- mask");
            GraphBLAS::apply(s,
                             GraphBLAS::complement(mask),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Identity<T>(),
                             s, true);
            //GraphBLAS::print_vector(std::cout, s, "-------- Seen vector");
            auto idx_weight = d.extractElement(u);
            weight += idx_weight.second;
            mst_parents.setElement(u, idx_weight.first);

            //GraphBLAS::print_vector(std::cout, mst_parents, "-------- Parent list");
            GraphBLAS::Vector<MSTType<T>> Arow(cols);
            GraphBLAS::extract(Arow,
                               GraphBLAS::NoMask(), // complement(mask)?
                               GraphBLAS::NoAccumulate(),
                               GraphBLAS::transpose(A),
                               GraphBLAS::GrB_ALL, u);
            //GraphBLAS::print_vector(std::cout, Arow, "-------- A(u,:)");

            GraphBLAS::eWiseAdd(d,
                                GraphBLAS::NoMask(), //mask,?
                                GraphBLAS::NoAccumulate(),
                                MSTMin<T>(),
                                d, Arow);
            //GraphBLAS::print_vector(std::cout, d, "-------- d = d .min A(u,:)");
        }

        return weight;
    }

} // algorithms

#endif // MST_HPP
