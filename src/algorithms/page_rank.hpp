/*
 * GraphBLAS Template Library, Version 2.0
 *
 * Copyright 2018 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors. All Rights Reserved.
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
 * RIGHTS..
 *
 * Released under a BSD (SEI)-style license, please see license.txt or contact
 * permission@sei.cmu.edu for full terms.
 *
 * This release is an update of:
 *
 * 1. GraphBLAS Template Library (GBTL)
 * (https://github.com/cmu-sei/gbtl/blob/1.0.0/LICENSE) Copyright 2015 Carnegie
 * Mellon University and The Trustees of Indiana. DM17-0037, DM-0002659
 *
 * DM18-0559
 */

/**
 * @file page_rank.hpp
 *
 * @brief A PageRank implementation using GraphBLAS.
 */

#ifndef PAGE_RANK_HPP
#define PAGE_RANK_HPP

#include <graphblas/graphblas.hpp>

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
        MatrixT const             &graph,
        GraphBLAS::Vector<RealT>  &page_rank,
        RealT                      damping_factor = 0.85,
        RealT                      threshold = 1.e-5,
        unsigned int max_iters = std::numeric_limits<unsigned int>::max())
    {
        using T = typename MatrixT::ScalarType;

        GraphBLAS::IndexType rows(graph.nrows());
        GraphBLAS::IndexType cols(graph.ncols());

        if ((rows != cols) || (page_rank.size() != rows))
        {
            //throw GraphBLAS::DimensionException();
            return;
        }

        // Compute the scaled graph matrix
        GraphBLAS::Matrix<RealT> m(rows, cols);

        // cast graph scalar type to RealT
        GraphBLAS::apply(m,
                         GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                         GraphBLAS::Identity<T,RealT>(),
                         graph);

        // Normalize the edge weights of the graph by the vertices out-degree
        GraphBLAS::normalize_rows(m);
        //GraphBLAS::print_matrix(std::cout, m, "Normalized Graph");

        // scale the normalized edge weights by the damping factor
        GraphBLAS::apply(
            m,
            GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
            GraphBLAS::BinaryOp_Bind2nd<RealT,
                                        GraphBLAS::Times<RealT>>(damping_factor),
            m);
        //GraphBLAS::print_matrix(std::cout, m, "Scaled Graph");

        GraphBLAS::BinaryOp_Bind2nd<RealT, GraphBLAS::Plus<RealT> >
            add_scaled_teleport((1.0 - damping_factor)/
                                static_cast<T>(rows));

        GraphBLAS::assign(page_rank,
                          GraphBLAS::NoMask(),
                          GraphBLAS::NoAccumulate(),
                          1.0 / static_cast<RealT>(rows),
                          GraphBLAS::AllIndices());


        GraphBLAS::Vector<RealT> new_rank(rows);
        GraphBLAS::Vector<RealT> delta(rows);
        for (GraphBLAS::IndexType i = 0; i < max_iters; ++i)
        {
            //std::cout << "============= ITERATION " << i << " ============"
            //          << std::endl;
            //print_vector(std::cout, page_rank, "rank");

            // Compute the new rank: [1 x M][M x N] = [1 x N]
            GraphBLAS::vxm(new_rank,
                           GraphBLAS::NoMask(),
                           GraphBLAS::Second<RealT>(),
                           GraphBLAS::ArithmeticSemiring<RealT>(),
                           page_rank, m);
            //print_vector(std::cout, new_rank, "step 1:");

            // [1 x M][M x 1] = [1 x 1] = always (1 - damping_factor)
            // rank*(m + scaling_mat*teleport): [1 x 1][1 x M] + [1 x N] = [1 x M]
            //use apply:
            GraphBLAS::apply(new_rank,
                             GraphBLAS::NoMask(),
                             GraphBLAS::NoAccumulate(),
                             add_scaled_teleport,
                             new_rank);
            //GraphBLAS::print_vector(std::cout, new_rank, "new_rank");

            // Test for convergence - compute squared error
            /// @todo should be mean squared error. (divide r2/N)
            RealT squared_error(0);
            GraphBLAS::eWiseAdd(delta,
                                GraphBLAS::NoMask(),
                                GraphBLAS::NoAccumulate(),
                                GraphBLAS::Minus<RealT>(),
                                page_rank, new_rank);
            GraphBLAS::eWiseMult(delta,
                                 GraphBLAS::NoMask(),
                                 GraphBLAS::NoAccumulate(),
                                 GraphBLAS::Times<RealT>(),
                                 delta, delta);
            GraphBLAS::reduce(squared_error,
                              GraphBLAS::NoAccumulate(),
                              GraphBLAS::PlusMonoid<RealT>(),
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
        GraphBLAS::assign(new_rank,
                          GraphBLAS::NoMask(),
                          GraphBLAS::NoAccumulate(),
                          (1.0 - damping_factor) / static_cast<T>(rows),
                          GraphBLAS::AllIndices());
        GraphBLAS::eWiseAdd(page_rank,
                            GraphBLAS::complement(page_rank),
                            GraphBLAS::NoAccumulate(),
                            GraphBLAS::Plus<RealT>(),
                            page_rank,
                            new_rank);
    }
} // algorithms

#endif // PAGE_RANK_HPP
