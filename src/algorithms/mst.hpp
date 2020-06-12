/*
 * GraphBLAS Template Library (GBTL), Version 3.0
 *
 * Copyright 2020 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors.
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
 * RIGHTS.
 *
 * Released under a BSD-style license, please see LICENSE file or contact
 * permission@sei.cmu.edu for full terms.
 *
 * [DISTRIBUTION STATEMENT A] This material has been approved for public release
 * and unlimited distribution.  Please see Copyright notice for non-US
 * Government use and distribution.
 *
 * DM20-0442
 */

#pragma once

#include <utility>
#include <vector>
#include <iostream>

#include <graphblas/graphblas.hpp>

//****************************************************************************
namespace algorithms
{
    //************************************************************************
    template <typename T>
    using MSTType = std::pair<grb::IndexType, T>;

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
        inline D3 operator()(D1 const &lhs, MSTType<D2> const &rhs)
        {
            return lhs + rhs.second;
        }
    };

    //************************************************************************
    template<typename D1>
    struct MSTMin
    {
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
    grb::IndexType argmin(grb::Vector<T> const &v)
    {
        if (v.nvals() == 0)
        {
            throw grb::PanicException();
        }

        std::vector<T> vals(v.nvals());
        grb::IndexArrayType idx(v.nvals());
        v.extractTuples(idx.begin(), vals.begin());

        grb::IndexType idx_min = idx[0];
        T val_min = vals[0];

        for (grb::IndexType i = 1; i < vals.size(); ++i)
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
        MatrixT const               &graph,
        grb::Vector<grb::IndexType> &mst_parents)
    {
        using T = typename MatrixT::ScalarType;

        grb::IndexType rows(graph.nrows());
        grb::IndexType cols(graph.ncols());

        if ((rows != cols) || (rows != mst_parents.size()))
        {
            throw grb::DimensionException();
        }
        //grb::print_matrix(std::cout, graph, "GRAPH***");

        // Create an adjacency matrix of the correct type (combining
        // source vertex ID with edge weight
        grb::IndexArrayType i(graph.nvals());
        grb::IndexArrayType j(graph.nvals());
        std::vector<T> vals(graph.nvals());
        graph.extractTuples(i.begin(), j.begin(), vals.begin());
        std::vector<MSTType<T>> new_vals;
        for (grb::IndexType ix = 0; ix < i.size(); ++ix)
        {
            new_vals.push_back(std::make_pair(i[ix], vals[ix]));
        }
        grb::Matrix<MSTType<T>> A(rows, cols);
        A.build(i, j, new_vals);
        //grb::print_matrix(std::cout, A, "Hybrid A matrix");

        // chose some arbitrary vertex to start
        grb::IndexType const START_NODE = 0;

        T weight = static_cast<T>(0);

        grb::Vector<MSTType<T>> d(rows);
        grb::Vector<bool> mask(rows);
        mask.setElement(START_NODE, true);
        //grb::print_vector(std::cout, mask, "Initial mask");

        // Using complement of mask so we don't deal with max()
        grb::Vector<T> s(rows);
        grb::assign(s,
                    grb::complement(mask),
                    grb::NoAccumulate(),
                    0, grb::AllIndices(), grb::REPLACE);
        //grb::print_vector(std::cout, s, "Initial s");

        mst_parents.clear();

        // Get the START NODE's neighbors
        grb::extract(d,
                     grb::NoMask(), // complement(mask)?
                     grb::NoAccumulate(),
                     grb::transpose(A),
                     grb::AllIndices(), START_NODE);
        //grb::print_vector(std::cout, d, "Initial d");

        while (mask.nvals() < rows)
        {
            //std::cout << "===================== ITERATION === s.nvals() = "
            //          << s.nvals() << std::endl;
            grb::Vector<T> temp(rows);

            // Note that eWiseMult is used with Plus b/c implied value is infinity
            grb::eWiseMult(temp,
                           grb::NoMask(),
                           grb::NoAccumulate(),
                           MSTPlus<T>(),
                           s, d);

            grb::IndexType u = argmin(temp);
            //grb::print_vector(std::cout, temp, "-------- s + d.second");
            //std::cout << "argmin(s + d.second) = " << u << std::endl;

            mask.setElement(u, true);
            //grb::print_vector(std::cout, mask, "-------- mask");
            grb::apply(s,
                       grb::complement(mask),
                       grb::NoAccumulate(),
                       grb::Identity<T>(),
                       s, grb::REPLACE);
            //grb::print_vector(std::cout, s, "-------- Seen vector");
            auto idx_weight = d.extractElement(u);
            weight += idx_weight.second;
            mst_parents.setElement(u, idx_weight.first);

            //grb::print_vector(std::cout, mst_parents, "-------- Parent list");
            grb::Vector<MSTType<T>> Arow(cols);
            grb::extract(Arow,
                         grb::NoMask(), // complement(mask)?
                         grb::NoAccumulate(),
                         grb::transpose(A),
                         grb::AllIndices(), u);
            //grb::print_vector(std::cout, Arow, "-------- A(u,:)");

            grb::eWiseAdd(d,
                          grb::NoMask(), //mask,?
                          grb::NoAccumulate(),
                          MSTMin<T>(),
                          d, Arow);
            //grb::print_vector(std::cout, d, "-------- d = d .min A(u,:)");
        }

        return weight;
    }

} // algorithms
