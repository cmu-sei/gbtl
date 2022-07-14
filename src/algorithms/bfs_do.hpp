/*
 * GraphBLAS Template Library (GBTL), Version 3.0
 *
 * Copyright 2021 Carnegie Mellon University, Battelle Memorial Institute, and
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
namespace
{
    template <typename D1, typename D2=D1, typename D3=D1>
    class PlusOneSemiring
    {
    public:
        using first_argument_type = D1;
        using second_argument_type = D2;
        using result_type = D3;

        D3 add(D3 a, D3 b) const
        { return a + b; }

        D3 mult(D1 a, D2 b) const
        { return static_cast<D3>(1); }

        D3 zero() const
        { return static_cast<D3>(0); }
    };
}

//****************************************************************************
namespace algorithms
{
    //************************************************************************
    /**
     * @brief Perform a single "parent" breadth first search (BFS) traversal
     *        on the given graph
     *
     * @param[in]  graph        N x N adjacency matrix of the graph on which to
     *                          perform a BFS. (NOT the transpose).  The value
     *                          1 should indicate an edge.
     * @param[in]  graphT       The transpose of 'graph' parameter.  If graph is
     *                          is symmetric these could be the same.
     * @param[in]  source       Index of the root vertex to use in the
     *                          calculation.
     * @param[out] parents      The list of parents for each traversal (row)
     *                          specified in the roots array.  User can specify
     *                          smaller unsigned ints if number of vertices in
     *                          the graph is lower.
     *
     * @todo Tuning and optimization of this algorithm still needs to be performed.
     *       It is running slower than other BFS algos.
     */
    template <typename MatrixT,
              typename ParentsVectorT>
    void bfs_do_parents(MatrixT const     &graph,
                        MatrixT const     &graphT,
                        grb::IndexType     source,
                        ParentsVectorT    &parents)
    {
        grb::IndexType const N(graph.nrows());
        if (graph.nrows() != graph.ncols() ||
            graph.nrows() != graphT.ncols() ||
            graph.ncols() != graphT.nrows())
        {
            throw grb::DimensionException("bfs_do error: graph dimension issue.");
        }

        // assert source is in proper range
        if (source >= N)
        {
            throw grb::IndexOutOfBoundsException("bfs_do error: invalid source.");
        }

        // assert parent_list is N-vector
        if (parents.size() != N)
        {
            throw grb::DimensionException("bfs_do error: parents dimension issue.");
        }

        // Determine size if integer to use in operations
        // assert parent_list ScalarType is integral and large enough
        using UInt = typename ParentsVectorT::ScalarType;
        if (!std::is_integral_v<UInt> ||
            (N > std::numeric_limits<UInt>::max()))
        {
            std::cerr << "bad scalar type: " << std::is_integral_v<UInt>
                      << "," << N << "?>"
                      << std::numeric_limits<UInt>::max() << std::endl;
            throw grb::PanicException(
                "bfs_do error: bad scalar type for parents.");
        }

        // semiring: MinFirstSemiring<UInt>
        // second_op: Second<UInt>
        // ramp_op: RowIndex<UInt, UInt>

        // Compute row degree (number of out neighbors for each vertex)
        /// @todo Consider moving this outside the function
        grb::Vector<UInt> row_degree(N);
        {
            /// @todo improve with iso valued vector
            grb::Vector<int8_t> ones(N);
            grb::assign(ones, grb::NoMask(), grb::NoAccumulate(),
                        1, grb::AllIndices());
            grb::mxv(row_degree, grb::NoMask(), grb::NoAccumulate(),
                     PlusOneSemiring<UInt>(), graph, ones);
            //grb::print_vector(std::cerr, row_degree, "Row degree");
            //std::cerr << "Finished row degree\n";
        }

        // set root parent to self;
        parents.clear();
        parents.setElement(source, static_cast<UInt>(source));

        // initialize frontier to source node.
        grb::Vector<UInt> frontier(N);
        frontier.setElement(source, static_cast<UInt>(source));

        // workspace for computing work remaining
        grb::Vector<int64_t> w(N);

        //---------------------------------------------------------------------
        // Setup parameters used to switch push <--> pull
        //---------------------------------------------------------------------
        grb::IndexType nq = 1;   // num nodes in frontier
        grb::IndexType last_nq = 0;
        grb::IndexType current_level = 0;
        grb::IndexType nvals = 1;

        // Magic numbers, needs to be tuned
        double alpha = 8.;
        double beta1 = 8.;
        double beta2 = 512.;
        int64_t n_over_beta1 = (int64_t)(((double)N)/beta1);
        int64_t n_over_beta2 = (int64_t)(((double)N)/beta2);
        int64_t edges_unexplored = graph.nvals();
        bool any_pull = false;  // true if any pull is done
        bool do_push = true; // start with push

        //---------------------------------------------------------------------
        // BFS traversal and label the nodes
        //---------------------------------------------------------------------
        for (int64_t nvisited = 1, k = 1;
             (nvisited < N);
             nvisited += nq, k++)
        {
            if (do_push)
            {
                /// @todo Switching logic from LAGraph.  Needs to be adjusted

                // Check for switch to pull
                bool growing = nq > last_nq;
                bool switch_to_pull = false;
                if (edges_unexplored < N)
                {
                    do_push = true;  //nop
                }
                else if (any_pull)
                {
                    // once any pull phase has been done, the #edges
                    // in the frontier has no longer been tracked.
                    // Now that it is back to push, check for another
                    // switch to pull.  This switch is unlikely, so
                    // just keep track of the size of the frontier,
                    // and switch if it starts growing again and is
                    // getting big.
                    switch_to_pull = (growing && nq > n_over_beta1);
                }
                else
                {
                    // update number of unexplored edges
                    // w<frontier> = Degree, w[i] = outdegree of i if in frontier
                    grb::assign(w, grb::structure(frontier), grb::NoAccumulate(),
                                row_degree, grb::AllIndices(), grb::REPLACE);
                    int64_t edges_in_frontier = 0;
                    grb::reduce(edges_in_frontier, grb::NoAccumulate(),
                                grb::PlusMonoid<int64_t>(), w);
                    edges_unexplored -= edges_in_frontier;
                    switch_to_pull =
                        growing &&
                        (edges_in_frontier > (edges_unexplored / alpha));
                }

                if (switch_to_pull)
                {
                    do_push = false;
                }
            }
            else  // do pull
            {
                // check for switch back to push
                if ((nq < last_nq) && (nq <= n_over_beta2))
                {
                    do_push = true;
                }
            }

            any_pull = any_pull || !do_push;

            //-----------------------------------------------------------------
            // frontier = kth level of the BFS
            //-----------------------------------------------------------------
            /// @todo if do_push switch q to a sparse format
            grb::apply(frontier, grb::NoMask(), grb::NoAccumulate(),
                       grb::RowIndex<UInt, UInt>(), frontier, 0);

            if (do_push)
            {
                //std::cerr << k << ", " << frontier.nvals() << ", " << parents.nvals() << ": doing push\n";
                grb::vxm(frontier, grb::complement(grb::structure(parents)),
                         grb::NoAccumulate(),
                         grb::MinFirstSemiring<UInt>(), frontier, graph,
                         grb::REPLACE);
            }
            else
            {
                //std::cerr << k << ", " << frontier.nvals() << ", " << parents.nvals() << ": doing pull\n";
                grb::mxv(frontier, grb::complement(grb::structure(parents)),
                         grb::NoAccumulate(),
                         grb::MinSecondSemiring<UInt>(), graphT, frontier,
                         grb::REPLACE);
            }

            if (frontier.nvals() == 0)
            {
                break;
            }

            // We don't need to mask here since we did it in mxm.
            // Merges new parents in current frontier with existing parents
            // parents<!parents,merge> += frontier
            grb::apply(parents, grb::NoMask(), grb::Plus<UInt>(),
                       grb::Identity<UInt>(), frontier, grb::MERGE);
            //
            // ...OR...
            //
            // grb::assign(parents, grb::structure(frontier), grb::NoAccumulate(),
            //             frontier, grb::AllIndices());
        }
    }

}
