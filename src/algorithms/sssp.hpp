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

#ifndef GBTL_SSSP_HPP
#define GBTL_SSSP_HPP

#include <functional>
#include <limits>
#include <tuple>

#include <graphblas/graphblas.hpp>

namespace algorithms
{
    /**
     * @brief Compute the lenghts of the single source shortest path(s) of
     *        one specified starting vertex in the given graph.
     *
     * The algorithm used in the implementation is the Floyd-Warshall
     * algorithm.  It is a dynamic programming solution (similar to the
     * Bellman-Ford algorithm).  The algorithm (like Bellman-Ford) is
     * based off of relaxation.  At each iteration in the algorithm, the
     * approximate lengths of the paths are always overestimates.  The
     * edges are relaxed at each iteration, until the lenghts are no
     * longer overestimates, at which point the algorithm terminates and
     * the path lengths are returned.
     *
     * The start can be specified multiple ways.  It can be in one
     * of the following three formats:
     *
     * <ol>
     * <li>The vertex itself (to specify only one vertex).</li>
     * <li>As a \f$N\f$ by \f$1\f$ column vector, where if the \f$i^{th}\f$
     * index is \f$1\f$, include that vertex in the set of vertices to
     * compute single source shortest paths for.</li>
     * <li>As a \f$N\f$ by \f$N\f$ matrix (say, \f$M\f$), where if
     * \f$M_{ii} = 1\f$, include vertex \f$i\f$ in the set of vertices to
     * compute single source shortest paths for.</li>
     * </ol>
     *
     * @param[in]  graph     The adjacency matrix of the graph for which SSSP
     *                       will be computed.  The implied zero will be
     *                       infinity.
     * @param[in,out] path   On input, the root vertex of the SSSP is set to zero.
     *                       On output, contains the shortest path for this
     *                       specified starting vertex, specified in terms of
     *                       length from the source vertex.
     */
    template<typename MatrixT,
             typename PathVectorT>
    void sssp(MatrixT const     &graph,
              PathVectorT       &path)
    {
        using T = typename MatrixT::ScalarType;

        if ((graph.nrows() != path.size()) ||
            (graph.ncols() != path.size()))
        {
            throw GraphBLAS::DimensionException();
        }

        /// @todo why num_rows iterations? (REALLY should be bounded by diameter)
        /// or detection of no change in paths matrix
        for (GraphBLAS::IndexType k = 0; k < graph.nrows(); ++k)
        {
            GraphBLAS::vxm(path,
                           GraphBLAS::NoMask(), GraphBLAS::Min<T>(),
                           GraphBLAS::MinPlusSemiring<T>(),
                           path, graph);

            //std::cout << "Iteration " << k << std::endl;
            //GraphBLAS::print_vector(std::cout, path, "Path");
            //std::cout << std::endl;
        }
        // paths holds return value
    }

    //****************************************************************************

    /**
     * @brief Compute the lenghts of the single source shortest path(s) of
     *        the specified starting vertex (vertices) in the given graph.
     *
     * The algorithm used in the implementation is the Floyd-Warshall
     * algorithm.  It is a dynamic programming solution (similar to the
     * Bellman-Ford algorithm).  The algorithm (like Bellman-Ford) is
     * based off of relaxation.  At each iteration in the algorithm, the
     * approximate lengths of the paths are always overestimates.  The
     * edges are relaxed at each iteration, until the lenghts are no
     * longer overestimates, at which point the algorithm terminates and
     * the path lengths are returned.
     *
     * The start can be specified multiple ways.  It can be in one
     * of the following three formats:
     *
     * <ol>
     * <li>The vertex itself (to specify only one vertex).</li>
     * <li>As a \f$N\f$ by \f$1\f$ column vector, where if the \f$i^{th}\f$
     * index is \f$1\f$, include that vertex in the set of vertices to
     * compute single source shortest paths for.</li>
     * <li>As a \f$N\f$ by \f$N\f$ matrix (say, \f$M\f$), where if
     * \f$M_{ii} = 1\f$, include vertex \f$i\f$ in the set of vertices to
     * compute single source shortest paths for.</li>
     * </ol>
     *
     * @param[in]  graph     The adjacency matrix of the graph for which SSSP
     *                       will be computed.  The implied zero will be
     *                       infinity.
     * @param[in,out] paths  Each row contains the lengths of single source
     *                       shortest path(s) for each the specified starting
     *                       vertices. These roots are specified, in this matrix
     *                       on input, by setting the source vertex to zero in the
     *                       corresponding row.
     */
    template<typename MatrixT,
             typename PathMatrixT>
    void batch_sssp(MatrixT const     &graph,
                    PathMatrixT       &paths)  // paths are initialized to start
    {
        using T = typename MatrixT::ScalarType;

        if ((graph.nrows() != paths.ncols()) ||
            (graph.ncols() != paths.ncols()))
        {
            throw GraphBLAS::DimensionException();
        }

        /// @todo why num_rows iterations?  Should be the diameter or terminate
        /// when there are no changes?
        for (GraphBLAS::IndexType k = 0; k < graph.nrows(); ++k)
        {
            GraphBLAS::mxm(paths,
                           GraphBLAS::NoMask(),
                           GraphBLAS::Min<T>(),
                           GraphBLAS::MinPlusSemiring<T>(),
                           paths, graph);

            //std::cout << "Iteration " << k << std::endl;
            //paths.print_info(std::cout);
            //std::cout << std::endl;
        }
        // paths holds return value
    }
} // algorithms

#endif // GBTL_SSSP_HPP
