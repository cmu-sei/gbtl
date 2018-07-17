/*
 * GraphBLAS Template Library, Version 2.0
 *
 * Copyright 2018 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors. All Rights Reserved.
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
 * RIGHTS..
 *
 * Released under a BSD (SEI)-style license, please see license.txt or contact
 * permission@sei.cmu.edu for full terms.
 *
 * This release is an update of:
 *
 * 1. GraphBLAS Template Library (GBTL)
 * (https://github.com/cmu-sei/gbtl/blob/master/LICENSE) Copyright 2015 Carnegie
 * Mellon University and The Trustees of Indiana. DM17-0037, DM-0002659
 *
 * DM18-0559
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
            //throw GraphBLAS::DimensionException();
            return GraphBLAS::Matrix<T>(num_vertices, num_vertices);
        }

        auto Distances(
            GraphBLAS::scaled_identity<GraphBLAS::Matrix<T>>(num_vertices, 0));
        GraphBLAS::eWiseAdd(Distances,
                            GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                            GraphBLAS::Plus<T>(),
                            Distances,
                            graph);
        GraphBLAS::Matrix<T> col_k(num_vertices, 1);
        GraphBLAS::IndexArrayType row_indices(1);
        GraphBLAS::Matrix<T> row_k(1, num_vertices);

        for (GraphBLAS::IndexType k = 0; k < num_vertices; ++k)
        {
            row_indices[0] = k;
            GraphBLAS::extract(row_k,
                               GraphBLAS::NoMask(),
                               GraphBLAS::NoAccumulate(),
                               Distances,
                               row_indices, GraphBLAS::AllIndices());

            GraphBLAS::extract(col_k,
                               GraphBLAS::NoMask(),
                               GraphBLAS::NoAccumulate(),
                               Distances,
                               GraphBLAS::AllIndices(), row_indices);

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
