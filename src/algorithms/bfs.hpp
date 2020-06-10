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

#include <limits>
#include <tuple>

#include <graphblas/graphblas.hpp>

//****************************************************************************
namespace algorithms
{
    //************************************************************************
    /**
     * @brief Perform a single "parent" breadth first search (BFS) traversal
     *        on the given graph.
     *
     * @param[in]  graph        N x N adjacency matrix of the graph on which to
     *                          perform a BFS. (NOT the transpose).  The value
     *                          1 should indicate an edge.
     * @param[in]  source       Index of the root vertex to use in the
     *                          calculation.
     * @param[out] parent_list  The list of parents for each traversal (row)
     *                          specified in the roots array.
     */
    template <typename MatrixT,
              typename ParentListVectorT>
    void bfs(MatrixT const          &graph,
             GraphBLAS::IndexType    source,
             ParentListVectorT      &parent_list)
    {
        GraphBLAS::IndexType const N(graph.nrows());

        // assert parent_list is N-vector
        // assert source is in proper range
        // assert parent_list ScalarType is GraphBLAS::IndexType

        // create index ramp for index_of() functionality
        GraphBLAS::Vector<GraphBLAS::IndexType> index_ramp(N);
        {
            std::vector<GraphBLAS::IndexType> idx(N);
            for (GraphBLAS::IndexType i = 0; i < N; ++i)
            {
                idx[i] = i;
            }

            index_ramp.build(idx.begin(), idx.begin(), N);
        }

        // initialize wavefront to source node.
        GraphBLAS::Vector<GraphBLAS::IndexType> wavefront(N);
        wavefront.setElement(source, 1UL);

        // set root parent to self;
        parent_list.clear();
        parent_list.setElement(source, source);

        while (wavefront.nvals() > 0)
        {
            // convert all stored values to their column index
            GraphBLAS::eWiseMult(wavefront,
                                 GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                 GraphBLAS::First<GraphBLAS::IndexType>(),
                                 index_ramp, wavefront);

            // First because we are left multiplying wavefront rows
            // Masking out the parent list ensures wavefront values do not
            // overlap values already stored in the parent list
            GraphBLAS::vxm(wavefront,
                           GraphBLAS::complement(GraphBLAS::structure(parent_list)),
                           GraphBLAS::NoAccumulate(),
                           GraphBLAS::MinFirstSemiring<GraphBLAS::IndexType>(),
                           wavefront, graph, GraphBLAS::REPLACE);

            // We don't need to mask here since we did it in mxm.
            // Merges new parents in current wavefront with existing parents
            // parent_list<!parent_list,merge> += wavefront
            GraphBLAS::apply(parent_list,
                             GraphBLAS::NoMask(),
                             GraphBLAS::Plus<GraphBLAS::IndexType>(),
                             GraphBLAS::Identity<GraphBLAS::IndexType>(),
                             wavefront,
                             GraphBLAS::MERGE);
        }
    }

    //************************************************************************
    /**
     * @brief Perform a single "parent" breadth first search (BFS) traversal
     *        on the given graph.
     *
     * @param[in]  graph        N x N adjacency matrix of the graph on which to
     *                          perform a BFS. (NOT the transpose).  The value
     *                          1 should indicate an edge.
     * @param[in]  wavefront    N-vector, initial wavefront/root to use in the
     *                          calculation. It should have a
     *                          single value set to '1' corresponding to the
     *                          root.
     * @param[out] parent_list  The list of parents for each traversal (row)
     *                          specified in the roots array.
     */
    template <typename MatrixT,
              typename WavefrontVectorT,
              typename ParentListVectorT>
    void bfs(MatrixT const          &graph,
             WavefrontVectorT        wavefront,   // copy is intentional
             ParentListVectorT      &parent_list)
    {
        using T = typename MatrixT::ScalarType;
        GraphBLAS::IndexType const N(graph.nrows());

        // assert parent_list is N-vector
        // assert wavefront is N-vector
        // assert parent_list ScalarType is GraphBLAS::IndexType

        // create index ramp for index_of() functionality
        GraphBLAS::Vector<GraphBLAS::IndexType> index_ramp(N);
        {
            std::vector<GraphBLAS::IndexType> idx(N);
            for (GraphBLAS::IndexType i = 0; i < N; ++i)
            {
                idx[i] = i;
            }

            index_ramp.build(idx.begin(), idx.begin(), N);
        }

        // Set the roots parents to themselves using indices
        GraphBLAS::eWiseMult(parent_list,
                             GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                             GraphBLAS::First<GraphBLAS::IndexType>(),
                             index_ramp, wavefront);

        // convert all stored values to their column index
        GraphBLAS::eWiseMult(parent_list,
                             GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                             GraphBLAS::First<GraphBLAS::IndexType>(),
                             index_ramp, parent_list);

        while (wavefront.nvals() > 0)
        {
            // convert all stored values to their column index
            GraphBLAS::eWiseMult(wavefront,
                                 GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                 GraphBLAS::First<GraphBLAS::IndexType>(),
                                 index_ramp, wavefront);

            // First because we are left multiplying wavefront rows
            // Masking out the parent list ensures wavefront values do not
            // overlap values already stored in the parent list
            GraphBLAS::vxm(wavefront,
                           GraphBLAS::complement(GraphBLAS::structure(parent_list)),
                           GraphBLAS::NoAccumulate(),
                           GraphBLAS::MinFirstSemiring<T>(),
                           wavefront, graph, GraphBLAS::REPLACE);

            // We don't need to mask here since we did it in mxm.
            // Merges new parents in current wavefront with existing parents
            // parent_list<!parent_list,merge> += wavefront
            GraphBLAS::apply(parent_list,
                             GraphBLAS::NoMask(),
                             GraphBLAS::Plus<T>(),
                             GraphBLAS::Identity<T>(),
                             wavefront,
                             GraphBLAS::MERGE);
        }
    }

    //************************************************************************
    /**
     * @brief Perform a set of breadth first search (BFS) traversals on the
     *        given graph.
     *
     * @param[in]  graph        N x N adjacency matrix of the graph on which to
     *                          perform a BFS. (NOT the transpose).  The value
     *                          1 should indicate an edge,
     * @param[in]  wavefronts   R x N, initial wavefront(s)/root(s) to use in the
     *                          calculation. Each row (1) corresponds to a
     *                          different BFS traversal and (2) should have a
     *                          single value set to '1' corresponding to the
     *                          root.
     * @param[out] parent_list  The list of parents for each traversal (row)
     *                          specified in the roots array.
     */
    template <typename MatrixT,
              typename WavefrontMatrixT,
              typename ParentListMatrixT>
    void bfs_batch(MatrixT const          &graph,
                   WavefrontMatrixT        wavefronts,   // copy is intentional
                   ParentListMatrixT      &parent_list)
    {
        using T = typename MatrixT::ScalarType;
        GraphBLAS::IndexType const N(graph.nrows());

        // assert parent_list is RxN
        // assert wavefront is RxN
        // assert parent_list ScalarType is GraphBLAS::IndexType

        // create index ramp for index_of() functionality
        GraphBLAS::Matrix<GraphBLAS::IndexType> index_ramp(N, N);
        {
            GraphBLAS::IndexArrayType idx;
            for (GraphBLAS::IndexType i = 0; i < N; ++i)
            {
                idx.push_back(i);
            }

            index_ramp.build(idx, idx, idx);
        }

        // Set the roots parents to themselves.
        GraphBLAS::mxm(parent_list,
                       GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                       GraphBLAS::MinSecondSemiring<T>(),
                       wavefronts, index_ramp);

        while (wavefronts.nvals() > 0)
        {
            // convert all stored values to their column index
            GraphBLAS::mxm(wavefronts,
                           GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                           GraphBLAS::MinSecondSemiring<T>(),
                           wavefronts, index_ramp);

            // First because we are left multiplying wavefront rows
            // Masking out the parent list ensures wavefronts values do not
            // overlap values already stored in the parent list
            GraphBLAS::mxm(wavefronts,
                           GraphBLAS::complement(GraphBLAS::structure(parent_list)),
                           GraphBLAS::NoAccumulate(),
                           GraphBLAS::MinFirstSemiring<T>(),
                           wavefronts, graph, GraphBLAS::REPLACE);

            // We don't need to mask here since we did it in mxm.
            // Merges new parents in current wavefront with existing parents
            // parent_list<!parent_list,merge> += wavefronts
            GraphBLAS::apply(parent_list,
                             GraphBLAS::NoMask(),
                             GraphBLAS::Plus<T>(),
                             GraphBLAS::Identity<T>(),
                             wavefronts,
                             GraphBLAS::MERGE);
        }
    }

    //************************************************************************
    /**
     * @brief Perform a single "level" breadth first search (BFS) traversal
     *        on the given graph.
     *
     * @param[in]  graph        N x N adjacency matrix of the graph on which to
     *                          perform a BFS. (NOT the transpose).  The value
     *                          1 should indicate an edge.
     * @param[in]  source       Index of the root vertex to use in the
     *                          calculation.
     * @param[out] levels       The level (distance in unweighted graphs) from
     *                          the source (root) of the BFS.
     */
    template <typename MatrixT,
              typename LevelsVectorT>
    void bfs_level(MatrixT const          &graph,
                   GraphBLAS::IndexType    source,
                   LevelsVectorT          &levels)
    {
        GraphBLAS::IndexType const N(graph.nrows());

        /// @todo Assert graph is square

        GraphBLAS::Vector<bool> wavefront(N);
        wavefront.setElement(source, true);

        GraphBLAS::IndexType depth(0);
        while (wavefront.nvals() > 0)
        {
            // Increment the level
            ++depth;

            // Apply the level to all newly visited nodes
            GraphBLAS::apply(levels, GraphBLAS::NoMask(),
                             GraphBLAS::Plus<GraphBLAS::IndexType>(),
                             //[depth](auto arg) { return arg * depth; },
                             std::bind(GraphBLAS::Times<GraphBLAS::IndexType>(),
                                       depth,
                                       std::placeholders::_1),
                             wavefront);

            GraphBLAS::mxv(wavefront, complement(levels),
                           GraphBLAS::NoAccumulate(),
                           GraphBLAS::LogicalSemiring<bool>(),
                           transpose(graph), wavefront, GraphBLAS::REPLACE);
        }
    }

    //************************************************************************
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
                   WavefrontMatrixT   wavefront, //row vectors, copy made
                   LevelListMatrixT  &levels)
    {
        /// @todo Assert graph is square
        /// @todo Assert graph has a compatible shape with wavefront?

        unsigned int depth = 0;

        while (wavefront.nvals() > 0)
        {
            // Increment the level
            ++depth;

            // Apply the level to all newly visited nodes
            GraphBLAS::apply(levels,
                             GraphBLAS::NoMask(),
                             GraphBLAS::Plus<unsigned int>(),
                             std::bind(GraphBLAS::Times<unsigned int>(),
                                       depth,
                                       std::placeholders::_1),
                             wavefront,
                             GraphBLAS::REPLACE);

            GraphBLAS::mxm(wavefront,
                           GraphBLAS::NoMask(),
                           GraphBLAS::NoAccumulate(),
                           GraphBLAS::LogicalSemiring<unsigned int>(),
                           wavefront, graph,
                           GraphBLAS::REPLACE);

            // Cull previously visited nodes from the wavefront
            // Replace these lines with the negate(levels) mask
            //GraphBLAS::apply(levels, not_visited,
            //                 GraphBLAS::math::IsZero<unsigned int>());
            //GraphBLAS::ewisemult(not_visited, wavefront, wavefront,
            //                     GraphBLAS::math::AndFn<unsigned int>());
            GraphBLAS::apply(
                wavefront,
                GraphBLAS::complement(levels),
                GraphBLAS::NoAccumulate(),
                GraphBLAS::Identity<typename WavefrontMatrixT::ScalarType>(),
                wavefront,
                GraphBLAS::REPLACE);
        }
    }

    //************************************************************************
    /**
     * @brief Perform a single breadth first searches (BFS) on the given graph.
     *
     * @param[in]  graph      NxN adjacency matrix of the graph on which to
     *                        perform a BFS (not the transpose).  A value of
     *                        1 indicates an edge (structural zero = 0).
     * @param[in]  wavefront  N-vector initial wavefront to use in the calculation
     *                        of R simultaneous traversals.  A value of 1 in a
     *                        given position indicates a root for the
     *                        traversal..
     * @param[out] levels     The level (distance in unweighted graphs) from
     *                        the corresponding root of that BFS.  Roots are
     *                        assigned a value of 1. (a value of 0 implies not
     *                        reachable.
     */
    template <typename MatrixT,
              typename WavefrontT,
              typename LevelListT>
    void bfs_level_masked(MatrixT const  &graph,
                          WavefrontT      wavefront, //row vector, copy made
                          LevelListT     &levels)
    {
        /// Assert graph is square/have a compatible shape with wavefront
        GraphBLAS::IndexType grows(graph.nrows());
        GraphBLAS::IndexType gcols(graph.ncols());

        GraphBLAS::IndexType wsize(wavefront.size());

        if ((grows != gcols) || (wsize != grows))
        {
            throw GraphBLAS::DimensionException();
        }

        GraphBLAS::IndexType depth = 0;
        while (wavefront.nvals() > 0)
        {
            // Increment the level
            ++depth;

            // Apply the level to all newly visited nodes
            GraphBLAS::apply(levels,
                             GraphBLAS::NoMask(),
                             GraphBLAS::Plus<unsigned int>(),
                             std::bind(GraphBLAS::Times<GraphBLAS::IndexType>(),
                                       depth,
                                       std::placeholders::_1),
                             wavefront,
                             GraphBLAS::REPLACE);

            // Advance the wavefront and mask out nodes already assigned levels
            GraphBLAS::vxm(
                wavefront,
                GraphBLAS::complement(levels),
                GraphBLAS::NoAccumulate(),
                GraphBLAS::LogicalSemiring<GraphBLAS::IndexType>(),
                wavefront, graph,
                GraphBLAS::REPLACE);
        }
    }

    //************************************************************************
    /**
     * @brief Perform multiple breadth first searches (BFS) on the given graph.
     *
     * @param[in]  graph      NxN adjacency matrix of the graph on which to
     *                        perform a BFS (not the transpose).  A value of
     *                        1 indicates an edge (structural zero = 0).
     * @param[in]  wavefronts RxN initial wavefronts to use in the calculation
     *                        of R simultaneous traversals.  A value of 1 in a
     *                        given row indicates a root for the corresponding
     *                        traversal. (structural zero = 0).
     * @param[out] levels     The level (distance in unweighted graphs) from
     *                        the corresponding root of that BFS.  Roots are
     *                        assigned a value of 1. (a value of 0 implies not
     *                        reachable.
     */
    template <typename MatrixT,
              typename WavefrontsMatrixT,
              typename LevelListMatrixT>
    void batch_bfs_level_masked(MatrixT const     &graph,
                                WavefrontsMatrixT  wavefronts, //row vectors, copy
                                LevelListMatrixT  &levels)
    {
        /// Assert graph is square/have a compatible shape with wavefronts
        GraphBLAS::IndexType grows(graph.nrows());
        GraphBLAS::IndexType gcols(graph.ncols());

        GraphBLAS::IndexType cols(wavefronts.ncols());

        if ((grows != gcols) || (cols != grows))
        {
            throw GraphBLAS::DimensionException();
        }

        GraphBLAS::IndexType depth = 0;
        while (wavefronts.nvals() > 0)
        {
            // Increment the level
            ++depth;

            // Apply the level to all newly visited nodes
            GraphBLAS::apply(levels,
                             GraphBLAS::NoMask(),
                             GraphBLAS::Plus<unsigned int>(),
                             std::bind(GraphBLAS::Times<GraphBLAS::IndexType>(),
                                       depth,
                                       std::placeholders::_1),
                             wavefronts,
                             GraphBLAS::REPLACE);

            // Advance the wavefronts and mask out nodes already assigned levels
            GraphBLAS::mxm(
                wavefronts,
                GraphBLAS::complement(levels),
                GraphBLAS::NoAccumulate(),
                GraphBLAS::LogicalSemiring<GraphBLAS::IndexType>(),
                wavefronts, graph,
                GraphBLAS::REPLACE);
        }
    }


    //************************************************************************
    /**
     * @brief Perform a single breadth first searches (BFS) on the given graph.
     *
     * @param[in]  graph      NxN adjacency matrix of the graph on which to
     *                        perform a BFS (not the transpose).  A value of
     *                        1 indicates an edge (structural zero = 0).
     * @param[in]  wavefront  N-vector initial wavefront to use in the calculation
     *                        of R simultaneous traversals.  A value of 1 in a
     *                        given position indicates a root for the
     *                        traversal..
     * @param[out] levels     The level (distance in unweighted graphs) from
     *                        the corresponding root of that BFS.  Roots are
     *                        assigned a value of 1. (a value of 0 implies not
     *                        reachable.
     */
    template <typename MatrixT,
              typename WavefrontT,
              typename LevelListT>
    void bfs_level_masked_v2(MatrixT const  &graph,
                             WavefrontT      wavefront, //row vector, copy made
                             LevelListT     &levels)
    {
        /// Assert graph is square/have a compatible shape with wavefront
        GraphBLAS::IndexType grows(graph.nrows());
        GraphBLAS::IndexType gcols(graph.ncols());

        GraphBLAS::IndexType wsize(wavefront.size());

        if ((grows != gcols) || (wsize != grows))
        {
            throw GraphBLAS::DimensionException();
        }

        GraphBLAS::IndexType depth = 0;
        while (wavefront.nvals() > 0)
        {
            // Increment the level
            ++depth;

            GraphBLAS::assign(levels,
                              wavefront,
                              GraphBLAS::NoAccumulate(),
                              depth,
                              GraphBLAS::AllIndices(),
                              GraphBLAS::MERGE);

            // Advance the wavefront and mask out nodes already assigned levels
            GraphBLAS::vxm(
                wavefront,
                GraphBLAS::complement(levels),
                GraphBLAS::NoAccumulate(),
                GraphBLAS::LogicalSemiring<GraphBLAS::IndexType>(),
                wavefront, graph,
                GraphBLAS::REPLACE);
        }
    }

}
