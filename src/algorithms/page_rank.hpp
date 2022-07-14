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
    template<typename MatrixT, typename RealT = double>
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
            //std::cout << "============= ITERATION " << i << " ============"
            //          << std::endl;
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
} // algorithms
