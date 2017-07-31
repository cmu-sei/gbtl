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

#ifndef GBTL_APSP_HPP
#define GBTL_APSP_HPP

#include <functional>
#include <limits>
#include <tuple>

#include <graphblas/graphblas.hpp>

namespace algorithms
{
    /**
     * @brief Given a graph, G = (V,E), with edge weights w and no negative
     *        weight cycles, this algorithm returns the shortest path
     *        distances for all vertex pairs (in a dense NxN matrix).
     *        This implements Floyd-Warshall.
     *
     * @param[in]  graph     The adjacency matrix of the graph for which APSP
     *                       will be computed.  Entries are edge weights.
     * @return               A dense matrix where entry u,v contains
     *                       the distance (edge weight sum) of shortest path
     *                       from vertex u to vertex v.
     */
    template<typename MatrixT>
    GraphBLAS::Matrix<typename MatrixT::ScalarType> apsp(MatrixT const &graph)
    {
        using T = typename MatrixT::ScalarType;

        GraphBLAS::IndexType num_vertices(graph.nrows());
        if (num_vertices != graph.ncols())
        {
            throw GraphBLAS::DimensionException();
        }

        auto Distances(
            GraphBLAS::scaled_identity<GraphBLAS::Matrix<T>>(num_vertices, 0));
        GraphBLAS::eWiseAdd(Distances,
                            GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                            GraphBLAS::Plus<T>(),
                            Distances,
                            graph);
        GraphBLAS::Matrix<T> row_k(1, num_vertices);
        GraphBLAS::Matrix<T> col_k(num_vertices, 1);
        GraphBLAS::IndexArrayType row_indices(1);

        for (GraphBLAS::IndexType k = 0; k < num_vertices; ++k)
        {
            row_indices[0] = k;
            GraphBLAS::extract(row_k,
                               GraphBLAS::NoMask(),
                               GraphBLAS::NoAccumulate(),
                               Distances,
                               row_indices, GraphBLAS::GrB_ALL);

            GraphBLAS::extract(col_k,
                               GraphBLAS::NoMask(),
                               GraphBLAS::NoAccumulate(),
                               Distances,
                               GraphBLAS::GrB_ALL, row_indices);

            GraphBLAS::mxm(Distances,
                           GraphBLAS::NoMask(),
                           GraphBLAS::Min<T>(),
                           GraphBLAS::MinPlusSemiring<T>(),
                           col_k, row_k);

            //std::cout << "============== Distances for Iteration " << k
            //          << std::endl;
            //GraphBLAS::print_matrix(std::cout, Distances, "Distances");
        }
        return Distances;
    }
} // algorithms

#endif // GBTL_SSSP_HPP
