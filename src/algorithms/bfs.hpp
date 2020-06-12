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
    void bfs(MatrixT const     &graph,
             grb::IndexType     source,
             ParentListVectorT &parent_list)
    {
        grb::IndexType const N(graph.nrows());

        // assert parent_list is N-vector
        // assert source is in proper range
        // assert parent_list ScalarType is grb::IndexType

        // create index ramp for index_of() functionality
        grb::Vector<grb::IndexType> index_ramp(N);
        for (grb::IndexType i = 0; i < N; ++i)
        {
            index_ramp.setElement(i, i);
        }

        // initialize wavefront to source node.
        grb::Vector<grb::IndexType> wavefront(N);
        wavefront.setElement(source, 1UL);

        // set root parent to self;
        parent_list.clear();
        parent_list.setElement(source, source);

        while (wavefront.nvals() > 0)
        {
            // convert all stored values to their column index
            grb::eWiseMult(wavefront,
                           grb::NoMask(), grb::NoAccumulate(),
                           grb::First<grb::IndexType>(),
                           index_ramp, wavefront);

            // First because we are left multiplying wavefront rows
            // Masking out the parent list ensures wavefront values do not
            // overlap values already stored in the parent list
            grb::vxm(wavefront,
                     grb::complement(grb::structure(parent_list)),
                     grb::NoAccumulate(),
                     grb::MinFirstSemiring<grb::IndexType>(),
                     wavefront, graph, grb::REPLACE);

            // We don't need to mask here since we did it in mxm.
            // Merges new parents in current wavefront with existing parents
            // parent_list<!parent_list,merge> += wavefront
            grb::apply(parent_list,
                       grb::NoMask(),
                       grb::Plus<grb::IndexType>(),
                       grb::Identity<grb::IndexType>(),
                       wavefront,
                       grb::MERGE);
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
        grb::IndexType const N(graph.nrows());

        // assert parent_list is N-vector
        // assert wavefront is N-vector
        // assert parent_list ScalarType is grb::IndexType

        // create index ramp for index_of() functionality
        grb::Vector<grb::IndexType> index_ramp(N);
        for (grb::IndexType i = 0; i < N; ++i)
        {
            index_ramp.setElement(i, i);
        }

        // Set the roots parents to themselves using indices
        grb::eWiseMult(parent_list,
                       grb::NoMask(), grb::NoAccumulate(),
                       grb::First<grb::IndexType>(),
                       index_ramp, wavefront);

        // convert all stored values to their column index
        grb::eWiseMult(parent_list,
                       grb::NoMask(), grb::NoAccumulate(),
                       grb::First<grb::IndexType>(),
                       index_ramp, parent_list);

        while (wavefront.nvals() > 0)
        {
            // convert all stored values to their column index
            grb::eWiseMult(wavefront,
                           grb::NoMask(), grb::NoAccumulate(),
                           grb::First<grb::IndexType>(),
                           index_ramp, wavefront);

            // First because we are left multiplying wavefront rows
            // Masking out the parent list ensures wavefront values do not
            // overlap values already stored in the parent list
            grb::vxm(wavefront,
                     grb::complement(grb::structure(parent_list)),
                     grb::NoAccumulate(),
                     grb::MinFirstSemiring<T>(),
                     wavefront, graph, grb::REPLACE);

            // We don't need to mask here since we did it in mxm.
            // Merges new parents in current wavefront with existing parents
            // parent_list<!parent_list,merge> += wavefront
            grb::apply(parent_list,
                       grb::NoMask(),
                       grb::Plus<T>(),
                       grb::Identity<T>(),
                       wavefront,
                       grb::MERGE);
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
        grb::IndexType const N(graph.nrows());

        // assert parent_list is RxN
        // assert wavefront is RxN
        // assert parent_list ScalarType is grb::IndexType

        // create index ramp for index_of() functionality
        grb::Matrix<grb::IndexType> index_ramp(N, N);
        for (grb::IndexType idx = 0; idx < N; ++idx)
        {
            index_ramp.setElement(idx, idx, idx);
        }

        // Set the roots parents to themselves.
        grb::mxm(parent_list,
                 grb::NoMask(), grb::NoAccumulate(),
                 grb::MinSecondSemiring<T>(),
                 wavefronts, index_ramp);

        while (wavefronts.nvals() > 0)
        {
            // convert all stored values to their column index
            grb::mxm(wavefronts,
                     grb::NoMask(), grb::NoAccumulate(),
                     grb::MinSecondSemiring<T>(),
                     wavefronts, index_ramp);

            // First because we are left multiplying wavefront rows
            // Masking out the parent list ensures wavefronts values do not
            // overlap values already stored in the parent list
            grb::mxm(wavefronts,
                     grb::complement(grb::structure(parent_list)),
                     grb::NoAccumulate(),
                     grb::MinFirstSemiring<T>(),
                     wavefronts, graph, grb::REPLACE);

            // We don't need to mask here since we did it in mxm.
            // Merges new parents in current wavefront with existing parents
            // parent_list<!parent_list,merge> += wavefronts
            grb::apply(parent_list,
                       grb::NoMask(),
                       grb::Plus<T>(),
                       grb::Identity<T>(),
                       wavefronts,
                       grb::MERGE);
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
                   grb::IndexType    source,
                   LevelsVectorT          &levels)
    {
        grb::IndexType const N(graph.nrows());

        /// @todo Assert graph is square

        grb::Vector<bool> wavefront(N);
        wavefront.setElement(source, true);

        grb::IndexType depth(0);
        while (wavefront.nvals() > 0)
        {
            // Increment the level
            ++depth;

            // Apply the level to all newly visited nodes
            grb::apply(levels, grb::NoMask(),
                       grb::Plus<grb::IndexType>(),
                       //[depth](auto arg) { return arg * depth; },
                       std::bind(grb::Times<grb::IndexType>(),
                                 depth,
                                 std::placeholders::_1),
                       wavefront);

            grb::mxv(wavefront, complement(levels),
                     grb::NoAccumulate(),
                     grb::LogicalSemiring<bool>(),
                     transpose(graph), wavefront, grb::REPLACE);
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
     *
     * @deprecated  Use batched_bfs_level_masked
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
            grb::apply(levels,
                       grb::NoMask(),
                       grb::Plus<unsigned int>(),
                       std::bind(grb::Times<unsigned int>(),
                                 depth,
                                 std::placeholders::_1),
                       wavefront,
                       grb::REPLACE);

            grb::mxm(wavefront,
                     grb::NoMask(),
                     grb::NoAccumulate(),
                     grb::LogicalSemiring<unsigned int>(),
                     wavefront, graph,
                     grb::REPLACE);

            // Cull previously visited nodes from the wavefront
            grb::apply(
                wavefront,
                grb::complement(levels),
                grb::NoAccumulate(),
                grb::Identity<typename WavefrontMatrixT::ScalarType>(),
                wavefront,
                grb::REPLACE);
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
        grb::IndexType grows(graph.nrows());
        grb::IndexType gcols(graph.ncols());

        grb::IndexType wsize(wavefront.size());

        if ((grows != gcols) || (wsize != grows))
        {
            throw grb::DimensionException();
        }

        grb::IndexType depth = 0;
        while (wavefront.nvals() > 0)
        {
            // Increment the level
            ++depth;

            // Apply the level to all newly visited nodes
            grb::apply(levels,
                       grb::NoMask(),
                       grb::Plus<unsigned int>(),
                       std::bind(grb::Times<grb::IndexType>(),
                                 depth,
                                 std::placeholders::_1),
                       wavefront,
                       grb::REPLACE);

            // Advance the wavefront and mask out nodes already assigned levels
            grb::vxm(wavefront,
                     grb::complement(levels),
                     grb::NoAccumulate(),
                     grb::LogicalSemiring<grb::IndexType>(),
                     wavefront, graph,
                     grb::REPLACE);
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
        grb::IndexType grows(graph.nrows());
        grb::IndexType gcols(graph.ncols());

        grb::IndexType cols(wavefronts.ncols());

        if ((grows != gcols) || (cols != grows))
        {
            throw grb::DimensionException();
        }

        grb::IndexType depth = 0;
        while (wavefronts.nvals() > 0)
        {
            // Increment the level
            ++depth;

            // Apply the level to all newly visited nodes
            grb::apply(levels,
                       grb::NoMask(),
                       grb::Plus<unsigned int>(),
                       std::bind(grb::Times<grb::IndexType>(),
                                 depth,
                                 std::placeholders::_1),
                       wavefronts,
                       grb::REPLACE);

            // Advance the wavefronts and mask out nodes already assigned levels
            grb::mxm(wavefronts,
                     grb::complement(levels),
                     grb::NoAccumulate(),
                     grb::LogicalSemiring<grb::IndexType>(),
                     wavefronts, graph,
                     grb::REPLACE);
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
        grb::IndexType grows(graph.nrows());
        grb::IndexType gcols(graph.ncols());

        grb::IndexType wsize(wavefront.size());

        if ((grows != gcols) || (wsize != grows))
        {
            throw grb::DimensionException();
        }

        grb::IndexType depth = 0;
        while (wavefront.nvals() > 0)
        {
            // Increment the level
            ++depth;

            grb::assign(levels,
                        wavefront,
                        grb::NoAccumulate(),
                        depth,
                        grb::AllIndices(),
                        grb::MERGE);

            // Advance the wavefront and mask out nodes already assigned levels
            grb::vxm(wavefront,
                     grb::complement(levels),
                     grb::NoAccumulate(),
                     grb::LogicalSemiring<grb::IndexType>(),
                     wavefront, graph,
                     grb::REPLACE);
        }
    }
}
