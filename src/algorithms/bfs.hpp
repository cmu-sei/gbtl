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

/**
 * @file bfs.hpp
 *
 * @brief Standard breadth first search implementation.
 *
 * <p>Breadth first search implementation, verification function, and human
 * readable error code conversion.</p>
 */

#include <limits>
#include <tuple>

#include <graphblas/graphblas.hpp>
#include <graphblas/Matrix.hpp>
#include <graphblas/utility.hpp>

namespace algorithms
{
    /**
     * @brief Perform a breadth first search (BFS) traversal on the given graph.
     *
     * @param[in]  graph        N x N adjacency matrix of the graph on which to
     *                          perform a BFS. (NOT the transpose).  The value
     *                          1 should indicate an edge, and max<T> is the
     *                          structural zero.
     * @param[in]  wavefronts   R x N, initial wavefront(s)/root(s) to use in the
     *                          calculation. Each row (1) corresponds to a
     *                          different BFS traversal and (2) should have a
     *                          single value set to '1' corresponding to the
     *                          root. The structural zero is <T>::max().
     * @param[out] parent_list  The list of parents for each traversal (row)
     *                          specified in the roots array.
     */
    template <typename MatrixT,
              typename WavefrontMatrixT,
              typename ParentListMatrixT>
    void bfs(MatrixT const          &graph,
             WavefrontMatrixT        wavefronts,   // copy is intentional
             ParentListMatrixT      &parent_list)
    {
        using T = typename MatrixT::ScalarType;

        graphblas::IndexType rows, cols;
        wavefronts.get_shape(rows, cols);

        auto wf_zero = wavefronts.get_zero();
        auto pl_zero = parent_list.get_zero();

        WavefrontMatrixT next_wavefronts(rows, cols, wf_zero);
        WavefrontMatrixT next_wavefronts_not_zeros(rows, cols, wf_zero);

        ParentListMatrixT parent_list_zeros(rows, cols, pl_zero);

        // Set the roots parents to themselves.
        parent_list = wavefronts;
        graphblas::col_index_of(parent_list);
        while (wavefronts.get_nnz() > 0)
        {
            // convert all stored values to their column index
            graphblas::col_index_of(wavefronts);

            // Select1st because we are left multiplying wavefront rows
            graphblas::mxm(wavefronts, graph, next_wavefronts,
                           graphblas::MinSelect1stSemiring<T>());

            // next_wavefronts_not_zeros = negate(negate(next_wavefronts)
            graphblas::apply(
                next_wavefronts,
                next_wavefronts_not_zeros,
                graphblas::math::IsNotStructuralZero<
                    graphblas::MinSelect1stSemiring<T> >());

            // parent_list_zeros = negate(parent_list)
            // We should use Negate of parent_list so that we do not have
            // to materialize the parent_list_zeros matrix (dense at first)
            graphblas::apply(
                parent_list,
                parent_list_zeros,
                graphblas::math::IsStructuralZero<
                    graphblas::MinSelect1stSemiring<T> >());

            // wavefronts = next_wavefronts_not_zeros .* parent_list_zeros
            graphblas::ewisemult(
                next_wavefronts_not_zeros,
                parent_list_zeros,
                wavefronts,
                graphblas::math::AnnihilatorTimes<
                    graphblas::MinSelect1stSemiring<T> >());

            // parent_list += wavefronts .* next_wavefronts
            graphblas::ewisemult(
                wavefronts,
                next_wavefronts,
                parent_list,
                graphblas::math::AnnihilatorTimes<
                    graphblas::MinSelect1stSemiring<T> >(),
                graphblas::math::Accum<
                    T,
                    graphblas::math::AnnihilatorPlus<
                        graphblas::MinSelect1stSemiring<T> > >());
        }
    }

    /**
     * @brief Perform a breadth first search (BFS) on the given graph.
     *
     * @param[in]  graph        The graph to perform a BFS on.  NOT built from
     *                          the transpose of the adjacency matrix.
     *                          (1 indicates edge, structural zero = 0).
     * @param[in]  wavefront    The initial wavefront to use in the calculation.
     *                          (1 indicates root, structural zero = 0).
     * @param[out] levels       The level (distance in unweighted graphs) from
     *                          the corresponding root of that BFS
     */
    template <typename MatrixT,
              typename WavefrontMatrixT,
              typename LevelListMatrixT>
    void bfs_level(MatrixT const     &graph,
                   WavefrontMatrixT   wavefront, //row vector, copy made
                   LevelListMatrixT  &levels)
    {
        using T = typename MatrixT::ScalarType;

        /// @todo Assert graph is square/have a compatible shape with wavefront?
        // graph.get_shape(rows, cols);

        graphblas::IndexType rows, cols;
        wavefront.get_shape(rows, cols);

        WavefrontMatrixT not_visited(rows, cols);

        unsigned int depth = 0;

        while (wavefront.get_nnz() > 0)
        {
            // Increment the level
            ++depth;
            graphblas::ConstantMatrix<unsigned int> depth_mat(rows, cols, depth);

            // Apply the level to all newly visited nodes
            graphblas::ewisemult(wavefront, depth_mat, levels,
                                 graphblas::math::Times<unsigned int>(),
                                 graphblas::math::Accum<unsigned int>());

            graphblas::mxm(wavefront, graph, wavefront,
                           graphblas::IntLogicalSemiring<unsigned int>());

            // Cull previously visited nodes from the wavefront
            // Replace these lines with the negate(levels) mask
            graphblas::apply(levels, not_visited,
                             graphblas::math::IsZero<unsigned int>());
            graphblas::ewisemult(not_visited, wavefront, wavefront,
                                 graphblas::math::AndFn<unsigned int>());
        }
    }

    //************************************************************************
    /**
     * @brief Perform a breadth first search (BFS) on the given graph.
     *
     * @param[in]  graph      NxN adjacency matrix of the graph on which to
     *                        perform a BFS (not the transpose).  A value of
     *                        1 indicates an edge (structural zero = 0).
     * @param[in]  wavefront  RxN initial wavefronts to use in the calculation of
     *                        R simultaneous traversals.  A value of 1 in a given
     *                        row indicates a root for the corresponding traversal.
     *                        (structural zero = 0).
     * @param[out] levels     The level (distance in unweighted graphs) from
     *                        the corresponding root of that BFS.  Roots are
     *                        assigned a value of 1. (a value of 0 implies not
     *                        reachable.
     */
    template <typename MatrixT,
              typename WavefrontMatrixT,
              typename LevelListMatrixT>
    void bfs_level_masked(MatrixT const     &graph,
                          WavefrontMatrixT   wavefront, //row vector, copy made
                          LevelListMatrixT  &levels)
    {
        using T = typename MatrixT::ScalarType;

        /// Assert graph is square/have a compatible shape with wavefront
        graphblas::IndexType grows, gcols, rows, cols;
        graph.get_shape(grows, gcols);
        wavefront.get_shape(rows, cols);
        if ((grows != gcols) || (cols != grows))
        {
            throw graphblas::DimensionException();
        }

        graphblas::IndexType depth = 0;
        while (wavefront.get_nnz() > 0)
        {
            // Increment the level
            ++depth;
            graphblas::ConstantMatrix<graphblas::IndexType>
                depth_mat(rows, cols, depth);

            // Apply the level to all newly visited nodes
            graphblas::ewisemult(
                wavefront, depth_mat, levels,
                graphblas::math::Times<graphblas::IndexType>(),
                graphblas::math::Accum<graphblas::IndexType>());

            // Advance the wavefront and mask out nodes already assigned levels
            graphblas::mxmMasked(
                wavefront, graph, wavefront,
                graphblas::negate(
                    levels,
                    graphblas::ArithmeticSemiring<graphblas::IndexType>()),
                graphblas::IntLogicalSemiring<graphblas::IndexType>());
        }
    }
}
