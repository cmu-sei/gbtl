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
    grb::Matrix<typename MatrixT::ScalarType> apsp(MatrixT const &graph)
    {
        using T = typename MatrixT::ScalarType;

        grb::IndexType num_vertices(graph.nrows());
        if (num_vertices != graph.ncols())
        {
            throw grb::DimensionException();
        }

        auto Distances(grb::scaled_identity<grb::Matrix<T>>(num_vertices, 0));
        grb::eWiseAdd(Distances,
                      grb::NoMask(), grb::NoAccumulate(),
                      grb::Plus<T>(),
                      Distances,
                      graph);
        grb::Matrix<T> col_k(num_vertices, 1);
        grb::IndexArrayType row_indices(1);
        grb::Matrix<T> row_k(1, num_vertices);

        for (grb::IndexType k = 0; k < num_vertices; ++k)
        {
            row_indices[0] = k;
            grb::extract(row_k,
                         grb::NoMask(),
                         grb::NoAccumulate(),
                         Distances,
                         row_indices, grb::AllIndices());

            grb::extract(col_k,
                         grb::NoMask(),
                         grb::NoAccumulate(),
                         Distances,
                         grb::AllIndices(), row_indices);

            grb::mxm(Distances,
                     grb::NoMask(),
                     grb::Min<T>(),
                     grb::MinPlusSemiring<T>(),
                     col_k, row_k);

            //std::cout << "============== Distances for Iteration " << k
            //          << std::endl;
            //grb::print_matrix(std::cout, Distances, "Distances");
        }
        return Distances;
    }
} // algorithms
