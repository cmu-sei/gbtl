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
     * @param[out] result           N vector of page ranks
     * @param[in]  damping_factor   The constant to ensure stability in cyclic
     *                              graphs
     * @param[in]  threshold        The sum of squared errors termination
     *                              threshold.
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
            throw GraphBLAS::DimensionException();
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

        GraphBLAS::assign_constant(page_rank,
                                   GraphBLAS::NoMask(),
                                   GraphBLAS::NoAccumulate(),
                                   1.0/static_cast<RealT>(rows),
                                   GraphBLAS::GrB_ALL);


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
        GraphBLAS::assign_constant(new_rank,
                                   GraphBLAS::NoMask(),
                                   GraphBLAS::NoAccumulate(),
                                   (1.0 - damping_factor)/static_cast<T>(rows),
                                   GraphBLAS::GrB_ALL);
        GraphBLAS::eWiseAdd(page_rank,
                            GraphBLAS::complement(page_rank),
                            GraphBLAS::NoAccumulate(),
                            GraphBLAS::Plus<RealT>(),
                            page_rank,
                            new_rank);
    }
} // algorithms

#endif // PAGE_RANK_HPP
