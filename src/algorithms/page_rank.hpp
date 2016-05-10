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
     * @param[out] result           1xN vector of page ranks
     * @param[in]  damping_factor   The constant to ensure stability in cyclic
     *                              graphs
     * @param[in]  threshold        The sum of squared errors termination
     *                              threshold.
     *
     */
    template<typename MatrixT, typename PRMatrixT>
    void page_rank(MatrixT const  &graph,
                   PRMatrixT      &page_rank,
                   double          damping_factor = 0.85,
                   double          threshold = 1.e-4)
    {
        using T = typename PRMatrixT::ScalarType;
        graphblas::IndexType rows, cols, pr_rows, pr_cols;
        graph.get_shape(rows, cols);
        page_rank.get_shape(pr_rows, pr_cols);

        if ((rows != cols) || (pr_rows != 1) || (pr_cols != rows))
        {
            throw graphblas::DimensionException();
        }

        // Compute the scaled graph matrix
        graphblas::Matrix<T> m(rows, cols);
        {
            // Normalize the edge weights of the graph by the vertices out-degree
            graphblas::Matrix<T> out_degree(rows, 1);
            graphblas::Matrix<T> norm_graph(rows, cols);
            graphblas::Matrix<T> scaling_mat(rows, cols);

            graphblas::row_reduce(graph, out_degree);
            graphblas::IndexType nnz = out_degree.get_nnz();
            graphblas::apply(out_degree, out_degree,
                             graphblas::math::Inverse<T>());

            graphblas::IndexArrayType i(nnz), j(nnz);
            std::vector<T> v(nnz);
            graphblas::extracttuples(out_degree, i, j, v);
            graphblas::buildmatrix(scaling_mat, i, i, v);
            graphblas::mxm(scaling_mat, graph, norm_graph);
            //graphblas::print_matrix(std::cout, norm_graph, "Normalized Graph");

            // scale the normalized edge weights by the damping factor
            i.clear();
            for (graphblas::IndexType ix = 0; ix < rows; ++ix)
            {
                i.push_back(ix);
            }
            graphblas::Matrix<T> damping_mat(rows, rows);
            std::vector<T> val(rows, damping_factor);
            graphblas::buildmatrix(damping_mat, i, i, val);
            graphblas::mxm(norm_graph, damping_mat, m);
            //graphblas::print_matrix(std::cout, m, "Scaled Graph");
        }

        // Scaling matrix
        //graphblas::ConstantMatrix<T> scaling_matrix(rows, 1,
        //                                            (1.0 - damping_factor),
        //                                            0.0);
        //graphblas::print_matrix(std::cout, scaling_matrix, "scaling_matrix");

        // teleporation probability is uniform across all vertices
        //graphblas::ConstantMatrix<T> teleportation(1, rows, 1.0/T(rows), 0.0);
        //graphblas::print_matrix(std::cout, teleportation, "teleportation");

        //graphblas::ConstantMatrix<T> scaled_teleport(
        //    1, rows, (1.0 - damping_factor)/static_cast<T>(rows), 0.0);
        //
        graphblas::arithmetic_n <T, graphblas::math::Plus<T> > scaled_teleport(
            (1.0 - damping_factor)/static_cast<T>(rows));

        //graphblas::print_matrix(std::cout, scaled_teleport,
        //                        "scaled_teleportation");

        page_rank = (graphblas::fill<graphblas::Matrix<T> >(
                         (1.0/T(rows)), 1, rows));

        graphblas::Matrix<T> new_rank(1, rows);
        graphblas::Matrix<T> delta(1, rows);
        graphblas::Matrix<T> r2(1, 1);
        for (graphblas::IndexType i = 0; i < cols; ++i)
        {
            //std::cout << "============= ITERATION " << i << " ============"
            //          << std::endl;
            //print_matrix(std::cout, page_rank, "rank");

            // Compute the new rank: [1 x M][M x N] = [1 x N]
            graphblas::mxm(page_rank, m, new_rank);
            //print_matrix(std::cout, new_rank, "step 1:");

            // [1 x M][M x 1] = [1 x 1] = always (1 - damping_factor)
            //graphblas::mxm(page_rank, scaling_matrix, r2);
            //print_matrix(std::cout, r2, "step 2:");

            // rank*(m + scaling_mat*teleport): [1 x 1][1 x M] + [1 x N] = [1 x M]
            //graphblas::mxm(r2, teleportation, new_rank,
            //               graphblas::ArithmeticSemiring<T>(),
            //               graphblas::math::Accum<T>());
            //graphblas::ewiseadd(scaled_teleport, new_rank, new_rank);
            //use apply:
            graphblas::apply(new_rank, new_rank, scaled_teleport);
            //graphblas::print_matrix(std::cout, new_rank, "new_rank");

            // Test for convergence - we really need dot-product here so that
            // we don't need to use get_value_at(0,0) to get the result of
            // delta * delta'.
            graphblas::ewiseadd(page_rank, new_rank, delta,
                                graphblas::math::Sub<T>());
            graphblas::mxm(delta, graphblas::transpose(delta), r2);
            //graphblas::print_matrix(std::cout, r2, "Squared error");

            page_rank = new_rank;
            if (r2.get_value_at(0, 0) < threshold)
            {
                break;
            }
        }
    }
} // algorithms

#endif // PAGE_RANK_HPP
