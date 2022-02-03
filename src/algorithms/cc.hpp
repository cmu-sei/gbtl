/*
 * GraphBLAS Template Library (GBTL), Version 3.0
 *
 * Copyright 2022 Carnegie Mellon University, Battelle Memorial Institute, and
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

#include <limits>
#include <tuple>

#include <graphblas/graphblas.hpp>

//****************************************************************************
namespace algorithms
{
    template <typename MatrixT>
    void cc_bfs(MatrixT const     &graph,
                grb::Vector<char> &frontier,
                grb::Vector<char> &visited)
    {
        visited.clear();
        while (frontier.nvals())
        {
            //std::cerr << "frontier size = " << frontier.nvals() << std::endl;
            grb::assign(visited, frontier, grb::NoAccumulate(),
                        (char)1, grb::AllIndices());
            grb::vxm(frontier, grb::complement(grb::structure(visited)),
                     grb::NoAccumulate(),
                     grb::LogicalSemiring<char>(), frontier, graph,
                     grb::REPLACE);
        }
    }

    //************************************************************************
    /**
     * @brief Identify connected components of the given graph using iterative
     *        BFS.
     *
     * @param[in]  graph        N x N adjacency matrix of the graph on which to
     *                          perform connected components algo.  The value
     *                          1 should indicate an edge.
     * @param[out] components   Vertex vector with component IDs.
     *
     * @retval The number of components found.
     */
    template <typename MatrixT,
              typename CompVectorT>
    grb::IndexType cc(MatrixT const     &graph,
                      CompVectorT       &component_ids)
    {
        grb::IndexType const N(graph.nrows());
        if (graph.nrows() != graph.ncols())
        {
            throw grb::DimensionException("cc error: graph dimension issue.");
        }

        // assert component_ids is N-vector
        if (component_ids.size() != N)
        {
            throw grb::DimensionException("cc error: component_ids dimension issue.");
        }

        // Determine size if integer to use in operations
        // assert components_ids ScalarType is integral and large enough
        using UInt = typename CompVectorT::ScalarType;
        if (!std::is_integral_v<UInt> ||
            (N > std::numeric_limits<UInt>::max()))
        {
            std::cerr << "bad scalar type: " << std::is_integral_v<UInt>
                      << std::is_unsigned_v<UInt> << "," << N << "?>"
                      << std::numeric_limits<UInt>::max() << std::endl;
            throw grb::PanicException(
                "cc error: bad scalar type for component IDs.");
        }

        // set root parent to self;
        component_ids.clear();
        grb::IndexType num_ccs = 0UL;

        // initialize frontier to source node.
        grb::Vector<char> visited(N);
        grb::Vector<char> frontier(N);

        //---------------------------------------------------------------------
        // BFS traversal and label the nodes
        //---------------------------------------------------------------------
        for (grb::IndexType idx = 0; idx < N; ++idx)
        {
            if (!component_ids.hasElement(idx))
            {
                frontier.clear();
                frontier.setElement(idx, (char)1);

                cc_bfs(graph, frontier, visited);

                grb::assign(component_ids, visited, grb::NoAccumulate(),
                            num_ccs, grb::AllIndices(), grb::MERGE);

                //std::cerr << num_ccs << ": processed node: " << idx
                //          << ", component size = " << visited.nvals()
                //          << std::endl;
                ++num_ccs;
            }
        }
        return num_ccs;
    }

}
