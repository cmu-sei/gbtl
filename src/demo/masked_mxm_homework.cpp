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

#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <vector>

#include <graphblas/graphblas.hpp>

using namespace grb;

//****************************************************************************
namespace
{
    //************************************************************************
    /**
     * @brief Perform a breadth first search (BFS) on the given graph.
     *
     * @param[in]  graph      NxN adjacency matrix of the graph on which to
     *                        perform a BFS (not the transpose).  A value of
     *                        1 indicates an edge (structural zero = 0).
     * @param[in]  wavefront  RxN initial wavefronts to use in the calculation
     *                        of R simultaneous traversals.  A value of 1 in a
     *                        given row indicates a root for the corresponding
     *                        traversal. (structural zero = 0).
     * @param[out] levels     The level (distance in unweighted graphs) from
     *                        the corresponding root of that BFS.  Roots are
     *                        assigned a value of 1. (a value of 0 implies not
     *                        reachable.
     */
    template <typename MatrixT,
              typename WavefrontMatrixT>
    void bfs_level_masked(MatrixT const     &graph,
                          WavefrontMatrixT   wavefront, //row vector, copy made
                          Matrix<IndexType> &levels)
    {
        /// Assert graph is square/have a compatible shape with wavefront
        IndexType grows(graph.nrows());
        IndexType gcols(graph.ncols());
        IndexType cols(wavefront.ncols());

        if ((grows != gcols) || (cols != grows))
        {
            throw DimensionException("grows="+std::to_string(grows)
                                     + ", gcols"+std::to_string(gcols)
                                     + "\ncols="+std::to_string(cols)
                                     + ", grows="+std::to_string(grows));
        }

        IndexType depth = 0;
        while (wavefront.nvals() > 0)
        {
            // Increment the level
            ++depth;

            // Apply the level to all newly visited nodes
            apply(levels, NoMask(), Plus<unsigned int>(),
                  std::bind(Times<IndexType>(),
                            std::placeholders::_1,
                            depth),
                  wavefront);

            // Advance the wavefront and mask out nodes already assigned levels
            mxm(wavefront,
                complement(levels), NoAccumulate(),
                LogicalSemiring<IndexType>(),
                wavefront, graph, REPLACE);
        }
    }
}


//****************************************************************************
int main(int, char**)
{
    // syntatic sugar
    using ScalarType = IndexType;

    // Create an adjacency matrix for "Gilbert's" directed graph
    IndexType const NUM_NODES(7);
    IndexArrayType i = {0, 0, 1, 1, 2, 3, 3, 3, 4, 5, 6, 6, 6};
    IndexArrayType j = {1, 3, 4, 6, 5, 0, 2, 6, 5, 2, 2, 3, 4};
    std::vector<ScalarType>   v(i.size(), 1.0);

    Matrix<ScalarType> graph(NUM_NODES, NUM_NODES);
    graph.build(i.begin(), j.begin(), v.begin(), i.size());

    print_matrix(std::cout, graph, "Test Graph:");
    // Outputs:
    // Test Graph:: structural zero = 0
    // [[ , 1,  , 1,  ,  ,  ]
    //  [ ,  ,  ,  , 1,  , 1]
    //  [ ,  ,  ,  ,  , 1,  ]
    //  [1,  , 1,  ,  ,  , 1]
    //  [ ,  ,  ,  ,  , 1,  ]
    //  [ ,  , 1,  ,  ,  ,  ]
    //  [ ,  , 1, 1, 1,  ,  ]]


    // Initialize NUM_NODES wavefronts starting from each vertex in the system
    Matrix<ScalarType> roots(NUM_NODES, NUM_NODES);
    std::vector<IndexType> indices;
    for (IndexType idx = 0; idx < NUM_NODES; ++idx)
    {
        indices.push_back(idx);
    }
    roots.build(indices, indices, std::vector<ScalarType>(NUM_NODES, 1));

    // Perform NUM_NODES traversals simultaneously
    Matrix<IndexType> levels(NUM_NODES, NUM_NODES);
    bfs_level_masked(graph, roots, levels);

    print_matrix(std::cout, levels, "Levels");
    // Outputs:
    // Levels: structural zero = 0
    // [[1, 2, 3, 2, 3, 4, 3]
    //  [4, 1, 3, 3, 2, 3, 2]
    //  [ ,  , 1,  ,  , 2,  ]
    //  [2, 3, 2, 1, 3, 3, 2]
    //  [ ,  , 3,  , 1, 2,  ]
    //  [ ,  , 2,  ,  , 1,  ]
    //  [3, 4, 2, 2, 2, 3, 1]]

    return 0;
}
