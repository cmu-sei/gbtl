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

#include <graphblas/graphblas.hpp>

namespace
{
    GEN_GRAPHBLAS_SEMIRING(PlusSecondSemiring, grb::PlusMonoid, grb::Second)
}

//****************************************************************************
namespace algorithms
{
    /**
     * @brief Compute the page rank for each node in a graph.
     *
     * Need more documentation
     *
     * @note Because of the random component of this algorithm, the MIS
     *       calculated across various calls to <code>mis</code> may vary.
     *
     * @note This only works with floating point scalars.
     *
     * @param[in]  graph            NxN adjacency matrix of graph to compute
     *                              the page rank.  The structural zero needs
     *                              to be '0' and edges are indicated by '1'
     *                              to support use of the Arithmetic semiring.
     * @param[out] page_rank        N vector of page ranks
     * @param[in]  damping_factor   The constant to ensure stability in cyclic
     *                              graphs
     * @param[in]  threshold        The sum of squared errors termination
     *                              threshold.
     * @param[in]  max_iters        The maximum number of iterations to perform
     *                              in case threshold is not met prior.
     *
     */
    template<typename MatrixT, typename RealT>
    void page_rank(
        MatrixT const       &graph,
        grb::Vector<RealT>  &page_rank,
        RealT                damping_factor = 0.85,
        RealT                threshold = 1.e-5,
        unsigned int         max_iters = std::numeric_limits<unsigned int>::max())
    {
        using T = typename MatrixT::ScalarType;

        grb::IndexType rows(graph.nrows());
        grb::IndexType cols(graph.ncols());

        if ((rows != cols) || (page_rank.size() != rows))
        {
            throw grb::DimensionException();
        }

        // Compute the scaled graph matrix
        grb::Matrix<RealT> m(rows, cols);

        // cast graph scalar type to RealT
        grb::apply(m,
                   grb::NoMask(), grb::NoAccumulate(),
                   grb::Identity<T,RealT>(),
                   graph);

        // Normalize the edge weights of the graph by the vertices out-degree
        grb::normalize_rows(m);
        //grb::print_matrix(std::cout, m, "Normalized Graph");

        // scale the normalized edge weights by the damping factor
        grb::apply(m,
                   grb::NoMask(), grb::NoAccumulate(),
                   std::bind(grb::Times<RealT>(),
                             std::placeholders::_1,
                             damping_factor),
                   m);
        //grb::print_matrix(std::cout, m, "Scaled Graph");

        auto add_scaled_teleport =
            std::bind(grb::Plus<RealT>(),
                      std::placeholders::_1,
                      (1.0 - damping_factor)/static_cast<T>(rows));

        grb::assign(page_rank,
                    grb::NoMask(),
                    grb::NoAccumulate(),
                    1.0 / static_cast<RealT>(rows),
                    grb::AllIndices());


        grb::Vector<RealT> new_rank(rows);
        grb::Vector<RealT> delta(rows);
        for (grb::IndexType i = 0; i < max_iters; ++i)
        {
            //std::cout << "============= ITERATION " << i << " ============\n";
            //print_vector(std::cout, page_rank, "rank");

            // Compute the new rank: [1 x M][M x N] = [1 x N]
            grb::vxm(new_rank,
                     grb::NoMask(),
                     grb::Second<RealT>(),
                     grb::ArithmeticSemiring<RealT>(),
                     page_rank, m);
            //print_vector(std::cout, new_rank, "step 1:");

            // [1 x M][M x 1] = [1 x 1] = always (1 - damping_factor)
            // rank*(m + scaling_mat*teleport): [1 x 1][1 x M] + [1 x N] = [1 x M]
            //use apply:
            grb::apply(new_rank,
                       grb::NoMask(),
                       grb::NoAccumulate(),
                       add_scaled_teleport,
                       new_rank);
            //grb::print_vector(std::cout, new_rank, "new_rank");

            // Test for convergence - compute squared error
            /// @todo should be mean squared error. (divide r2/N)
            /// @todo parameterize the algorithm on the convergence test.
            RealT squared_error(0);
            grb::eWiseAdd(delta,
                          grb::NoMask(),
                          grb::NoAccumulate(),
                          grb::Minus<RealT>(),
                          page_rank, new_rank);
            grb::eWiseMult(delta,
                           grb::NoMask(),
                           grb::NoAccumulate(),
                           grb::Times<RealT>(),
                           delta, delta);
            grb::reduce(squared_error,
                        grb::NoAccumulate(),
                        grb::PlusMonoid<RealT>(),
                        delta);

            //std::cout << "Squared error = " << r2 << std::endl;

            page_rank = new_rank;
            // check mean-squared error
            if (squared_error/((RealT)rows) < threshold)
            {
                break;
            }
        }

        // for any elements missing from page rank vector we need to set
        // to scaled teleport.
        grb::assign(new_rank,
                    grb::NoMask(),
                    grb::NoAccumulate(),
                    (1.0 - damping_factor) / static_cast<T>(rows),
                    grb::AllIndices());
        grb::eWiseAdd(page_rank,
                      grb::complement(page_rank),
                      grb::NoAccumulate(),
                      grb::Plus<RealT>(),
                      page_rank,
                      new_rank);
    }

    //************************************************************************
    // Translated/adapted from LAGraph: LAGr_PageRankGAP.c
    //
    template<typename MatrixT,
             typename RealT,
             std::enable_if_t<std::is_floating_point<RealT>::value, bool> = true>
    void pagerank_gap(
        MatrixT const       &graph,
        grb::Vector<RealT>  &page_rank,
        unsigned int        &num_iters,
        RealT                damping = 0.85,
        RealT                tolerance = 1.e-5,
        unsigned int         max_iters = std::numeric_limits<unsigned int>::max())
    {
        //using T = typename MatrixT::ScalarType;

        grb::IndexType n(graph.nrows());

        if ((n != graph.ncols()) || (page_rank.size() != n))
        {
            throw grb::DimensionException();
        }

        // Compute row (out) degrees
        grb::Vector<RealT> d_out(n);
        grb::reduce(d_out,
                    grb::NoMask(), grb::NoAccumulate(),
                    grb::Plus<RealT>(),
                    graph);

        // RealT const scaled_damping = (1 - damping)/n;
        RealT const teleport = (1. - damping)/(RealT)n;
        RealT rdiff = 1 + tolerance;  // make sure it iterates at least once

        grb::Vector<RealT> t(n);
        grb::Vector<RealT> temp(n);
        grb::Vector<RealT> r(n);
        grb::Vector<RealT> w(n);

        // r = 1 / n
        grb::assign(r,
                    grb::NoMask(),
                    grb::NoAccumulate(),
                    1.0 / static_cast<RealT>(n),
                    grb::AllIndices());

        // prescale with damping factor so it is not done every iteration
        // d = d_out / damping ;
        grb::Vector<RealT> d(n);
        grb::apply(d,
                   grb::NoMask(),
                   grb::NoAccumulate(),
                   std::bind(grb::Times<RealT>(),   /// @todo Replace with lambda
                             std::placeholders::_1,
                             1./damping),
                   d_out);

        {
            // Temporary
            grb::Vector<RealT> d1(n);

            // d1 = 1 / damping
            grb::assign(d1,
                        grb::NoMask(),
                        grb::NoAccumulate(),
                        (1./damping),
                        grb::AllIndices());

            // d = max(d1, d)
            grb::eWiseAdd(d,
                          grb::NoMask(),
                          grb::NoAccumulate(),
                          grb::Max<RealT>(),
                          d1,
                          d);
        }

        //--------------------------------------------------------------------
        // pagerank iterations
        //--------------------------------------------------------------------

        for (num_iters = 0; num_iters < max_iters && rdiff > tolerance; ++num_iters)
        {
            //std::cout << "============= ITERATION " << num_iters << " ============\n";
            //print_vector(std::cout, page_rank, "rank");

            // swap t and r ; now t is the old pagerank score
            /// @todo make this cheap (use move? add *::swap?)
            temp = t;
            t = r;
            r = temp;

            // w = t ./ d
            grb::eWiseMult(w,
                           grb::NoMask(),
                           grb::NoAccumulate(),
                           grb::Div<RealT>(),
                           t, d);

            // r = ones()*teleport
            grb::assign(r,
                        grb::NoMask(),
                        grb::NoAccumulate(),
                        teleport,
                        grb::AllIndices());

            // r += A'*w
            grb::mxv(r,
                     grb::NoMask(),
                     grb::Plus<RealT>(),
                     PlusSecondSemiring<RealT>(),
                     grb::transpose(graph),  /// @todo precompute row-major transpose
                     w);

            // Test for convergence - compute absolute difference
            // t -= r
            grb::assign(t,
                        grb::NoMask(),
                        grb::Minus<RealT>(),
                        r,
                        grb::AllIndices());
            // t = abs (t)
            grb::apply(t,
                       grb::NoMask(),
                       grb::NoAccumulate(),
                       grb::Abs<RealT>(),
                       t);
            // rdiff = sum (t)
            grb::reduce(rdiff,
                        grb::NoAccumulate(),
                        grb::PlusMonoid<RealT>(),
                        t);
        }

        page_rank = r;  /// @todo std::move(r)
    }
} // algorithms
