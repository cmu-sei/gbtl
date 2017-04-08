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
     * @param[in]  graph  The adjacency matrix of the graph for which SSSP
     *                    will be computed.  The structural zero will be
     *                    infinity.
     * @param[in]  start  A P x M matrix (for computing P different sources)
     *                    where each row of P has a single stored 0 value
     *                    indicating the source of one SSSP and the rest of
     *                    would contain the structural zero (i.e. infinity)
     * @param[out] paths  Each row contains the lengths of single source
     *                    shortest path(s) for the specified starting vertex
     *                    specified in the start matrix for the same row.
     */
    template<typename MatrixT,
             typename PathMatrixT>
    void sssp(MatrixT const     &graph,
              PathMatrixT const &start,
              PathMatrixT       &paths)
    {
        using T = typename MatrixT::ScalarType;
        using MinAccum =
            graphblas::math::Accum<T, graphblas::math::ArithmeticMin<T> >;

        paths = start;

        graphblas::IndexType rows, cols, prows, pcols, rrows, rcols;
        graph.get_shape(rows, cols);
        start.get_shape(prows, pcols);
        paths.get_shape(rrows, rcols);

        if ((rows != pcols) || (prows != rrows) || (pcols != rcols))
        {
            throw graphblas::DimensionException();
        }

        /// @todo why num_rows iterations?
        for (graphblas::IndexType k = 0; k < rows; ++k)
        {
            graphblas::mxm<MatrixT, PathMatrixT, PathMatrixT,
                           graphblas::MinPlusSemiring<T>,
                           MinAccum>(paths, graph, paths);
            //std::cout << "Iteration " << k << std::endl;
            //graphblas::backend::pretty_print_matrix(std::cout, paths);
            //std::cout << std::endl;
        }
        // paths holds return value
    }

    //****************************************************************************
    template<typename MatrixT,
             typename PathMatrixT>
    void GrB_sssp(MatrixT const     &graph,
                  PathMatrixT       &paths)  // paths are initialized to start
    {
        using T = typename MatrixT::ScalarType;

        if ((graph.get_nrows() != paths.get_ncols()) ||
            (graph.get_ncols() != paths.get_ncols()))
        {
            throw graphblas::DimensionException();
        }

        /// @todo why num_rows iterations?  Should be the diameter or terminate
        /// when there are no changes?
        for (graphblas::IndexType k = 0; k < graph.get_nrows(); ++k)
        {
            GraphBLAS::mxm(paths,
                           GraphBLAS::Min<T>(), GraphBLAS::MinPlusSemiring<T>(),
                           paths, graph);

            //std::cout << "Iteration " << k << std::endl;
            //paths.print_info(std::cout);
            //std::cout << std::endl;
        }
        // paths holds return value
    }
} // algorithms

#endif // GBTL_SSSP_HPP
