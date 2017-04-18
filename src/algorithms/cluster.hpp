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

#ifndef ALGORITHMS_CLUSTER_HPP
#define ALGORITHMS_CLUSTER_HPP

#include <vector>
#include <math.h>
#include <graphblas/graphblas.hpp>

namespace algorithms
{
    /**
     * @brief Assign a zero-base cluster ID to each vertex based on the
     *        cluster matrix that was output by the clustering algorithm.
     *
     * @param[in]  cluster_matrix  Matrix output from one of the clustering
     *                             algorithms.  Each column corresponds to a
     *                             vertex and each row corresponds to a cluster.
     *
     * @return An array with each vertex's cluster assignment (the row index for
     *         the maximum value).  MAX_UINT is returned if no cluster was
     *         assigned.
     */
    template <typename MatrixT>
    graphblas::IndexArrayType get_cluster_assignments_v2(
        MatrixT const &cluster_matrix)
    {
        graphblas::IndexType num_clusters, num_nodes;
        cluster_matrix.get_shape(num_clusters, num_nodes);
        graphblas::IndexArrayType clusters(
            num_nodes,
            std::numeric_limits<graphblas::IndexType>::max());

        /// @todo this can be replaced with MaxSelect1st (transposed)
        for (graphblas::IndexType v = 0; v < num_nodes; ++v)
        {
            double max_val = 0.0;
            for (graphblas::IndexType c = 0; c < num_clusters; ++c)
            {
                if (cluster_matrix.extractElement(c, v) > max_val)
                {
                    max_val = cluster_matrix.extractElement(c, v);
                    clusters[v] = c;
                }
            }
        }

        return clusters;
    }

    /**
     * @brief Assign a zero-base cluster ID to each vertex based on the
     *        cluster matrix that was output by the clustering algorithm.
     *
     * @param[in]  cluster_matrix  Matrix output from one of the clustering
     *                             algorithms.  Each column corresponds to a
     *                             vertex and each row corresponds to a cluster.
     *
     * @return An array with each vertex's cluster assignment (the row index for
     *         the maximum value).  MAX_UINT is returned if no cluster was
     *         assigned.
     */
    template <typename MatrixT>
    graphblas::IndexArrayType get_cluster_assignments(MatrixT cluster_matrix)
    {
        graphblas::IndexType      num_nodes, num_clusters;
        cluster_matrix.get_shape(num_clusters, num_nodes);

        graphblas::IndexType nnz = cluster_matrix.get_nnz();
        graphblas::IndexArrayType cluster_ids(nnz), vertex_ids(nnz);
        std::vector<double> vals(nnz);
        extracttuples(cluster_matrix, cluster_ids, vertex_ids, vals);

        std::vector<double> max_vals(num_nodes, -1.0);
        graphblas::IndexArrayType cluster_assignments(
            num_nodes,
            std::numeric_limits<graphblas::IndexType>::max());

        for (graphblas::IndexType idx = 0; idx < vals.size(); ++idx)
        {
            if (max_vals[vertex_ids[idx]] < vals[idx])
            {
                max_vals[vertex_ids[idx]] = vals[idx];
                cluster_assignments[vertex_ids[idx]] = cluster_ids[idx];
            }
        }
        return cluster_assignments;
    }

    /**
     * @brief Compute the clusters in the given graph using a peer-pressure
     *        clustering implementation.
     *
     * Peer-pressure clustering works as follows:
     * <ul>
     * <li>Each vertex in the graph votes for it’s neighbors to be in that
     * vertex’s current cluster.</p>
     * <li>These votes are tallied, and a new cluster approximation is formed,
     * with vertices moved into the cluster for which they obtained the most
     * votes.</li>
     * <li>In order to assure that all of the vertices have equal votes, the
     * votes of each vertex are normalized to 1 (This is done by dividing the
     * weights by the out-degree of that vertex).</li>
     * <li>When enough cluster approximations are identical (usually one or
     * two), the algorithm stops and returns that cluster approximation.</li>
     * </ul>
     *
     * @param[in] graph          NxN adjacency matrix of the graph to compute
     *                           clusters for.
     * @param[in] cluster_approx The initial cluster approximation to use.
     * @param[in] max_iter       The maximum number of iterations to run if
     *                           convergence doesn't occur first.
     *
     * @return The resulting cluster matrix.  If <code>C_f[i][j] == 1</code>,
     *         vertex <code>j</code> belongs to cluster <code>i</code>.
     */
    template<typename MatrixT,
             typename ClusterMatrixT>
    ClusterMatrixT peer_pressure_cluster(
        MatrixT const  &graph,
        ClusterMatrixT  cluster_approx,
        unsigned int    max_iters = std::numeric_limits<unsigned int>::max())
    {
        graphblas::IndexType rows, cols;
        graph.get_shape(rows, cols);
        if (rows != cols)
        {
            throw graphblas::DimensionException();
        }

        ssize_t n_vertices = cols;  // usigned to signed
        MatrixT a(graph);
        MatrixT w(n_vertices,1);

        row_reduce(graph, w);

        /// @todo Replace with singleton expansion of w and then ewise ops.
        for (graphblas::IndexType i = 0; i < n_vertices; ++i)
        {
            for (graphblas::IndexType j = 0; j < n_vertices; ++j)
            {
                auto w_i0 = w.extractElement(i, 0);
                if (w_i0 > 0)
                {
                    a.setElement(i, j, (a.extractElement(i, j) / w_i0));
                }
            }
        }

        MatrixT c_i = cluster_approx;

        for (unsigned int iters = 0; iters < max_iters; ++iters)
        {
            MatrixT tally(n_vertices, n_vertices);
            graphblas::mxm(c_i, a, tally);

            std::vector<graphblas::IndexType> m;

            for (graphblas::IndexType vertex = 0; vertex < n_vertices; ++vertex)
            {
                graphblas::IndexType maxpos = 0;
                for (graphblas::IndexType cluster = 1;
                     cluster < n_vertices;
                     ++cluster)
                {
                    if (tally.extractElement(maxpos, vertex) <
                        tally.extractElement(cluster, vertex))
                    {
                        maxpos = cluster;
                    }
                }
                m.push_back(maxpos);
            }

            MatrixT c_f(n_vertices, n_vertices);

            for (graphblas::IndexType vertex = 0; vertex < n_vertices; ++vertex)
            {
                c_f.setElement(m[vertex], vertex, 1);
            }

            if (c_f == c_i)  // costly comparison?
            {
                return c_f;
            }
            else
            {
                c_i = c_f;
            }
        }

        return c_i;
    }

    /**
     * @brief Compute the clusters in the given graph using a peer-pressure
     *        clustering implementation.
     *
     * Peer-pressure clustering works as follows:
     * <ul>
     * <li>Each vertex in the graph votes for it’s neighbors to be in that
     * vertex’s current cluster.</p>
     * <li>These votes are tallied, and a new cluster approximation is formed,
     * with vertices moved into the cluster for which they obtained the most
     * votes.</li>
     * <li>In order to assure that all of the vertices have equal votes, the
     * votes of each vertex are normalized to 1 (This is done by dividing the
     * weights by the out-degree of that vertex).</li>
     * <li>When enough cluster approximations are identical (usually one or
     * two), the algorithm stops and returns that cluster approximation.</li>
     * </ul>
     *
     * @param[in] graph      The graph to compute the clusters of.
     * @param[in] max_iters  The maximum number of iterations to run if
     *                       convergence doesn't occur first.
     *
     * @return The resulting cluster matrix.  If <code>C_f[i][j] == 1</code>,
     *         vertex <code>j</code> belongs to cluster <code>i</code>.
     */
    template<typename MatrixT>
    MatrixT peer_pressure_cluster(
        MatrixT const &graph,
        unsigned int   max_iters = std::numeric_limits<unsigned int>::max())
    {
        graphblas::IndexType rows, cols;
        graph.get_shape(rows, cols);
        MatrixT cluster_approx = graphblas::identity<MatrixT>(rows);
        return peer_pressure_cluster<MatrixT, MatrixT>(graph,
                                                       cluster_approx,
                                                       max_iters);
    }

    /**
     * @brief Compute the clusters in the given graph using a varient of the
     *        peer-pressure clustering implementation.
     *
     * @note This variant does not normalize the edge weight by vertex degree
     *
     * @param[in] graph          The graph to compute the clusters of.
     * @param[in] cluster_approx The initial cluster approximation to use.
     * @param[in] max_iters      The maximum number of iterations to run if
     *                           convergence doesn't occur first.
     *
     * @return The resulting cluster matrix.  If <code>C_f[i][j] == 1</code>,
     *         vertex <code>j</code> belongs to cluster <code>i</code>.
     *
     * @todo This should be merged with the other version and a configuration
     *       parameter should be added for each difference.
     */
    template<typename MatrixT,
             typename ClusterMatrixT>
    ClusterMatrixT peer_pressure_cluster_v2(MatrixT const  &graph,
                                            ClusterMatrixT  cluster_approx,
                                            size_t          max_iters)
    {
        size_t iter_num(0);
        graphblas::IndexType num_rows, num_cols;
        graph.get_shape(num_rows, num_cols);
        if (num_rows != num_cols)
        {
            throw graphblas::DimensionException();
        }

        graphblas::IndexType num_vertices = num_cols;
        MatrixT a(num_vertices, num_vertices);
        MatrixT w(num_vertices, 1);
        MatrixT w_inv(num_vertices, 1);

        graphblas::row_reduce(graph, w);
        graphblas::apply(w, w_inv, graphblas::math::Inverse<double>());
        graphblas::ewisemult(
            graph,
            graphblas::RowExtendedView<MatrixT>(0, w_inv, num_vertices),
            a);

        // throwing away normalization across rows
        a = graph;

        //for (IndexType i = 0; i < num_vertices; i++)
        //{
        //    for (IndexType j = 0; j < num_vertices; j++ )
        //    {
        //        if (w[i][0] > 0)
        //        {
        //            a[i][j] = a[i][j] / w[i][0];
        //        }
        //    }
        //}

        MatrixT c_i = cluster_approx;
        ClusterMatrixT c_f(num_vertices, num_vertices);
        MatrixT tally(num_vertices, num_vertices);
        do
        {
            tally = MatrixT(num_vertices, num_vertices);
            graphblas::mxm(c_i, a, tally);

            MatrixT col_max = MatrixT(1, num_vertices);
            graphblas::col_reduce(tally,
                                  col_max,
                                  graphblas::RegmaxMonoid<double>());
            graphblas::ewisemult(
                tally,
                graphblas::ColumnExtendedView<MatrixT>(0,
                                                       col_max,
                                                       num_vertices),
                c_f,
                graphblas::math::IsEqual<double>());

            std::cout << "Iteration: " << iter_num << std::endl;
            for (graphblas::IndexType vertex = 0;
                 vertex < num_vertices;
                 ++vertex)
            {
                std::cout << "(" << vertex << ":";
                for (graphblas::IndexType cluster = 0;
                     cluster < num_vertices;
                     ++cluster)
                {
                    if (c_f.extractElement(cluster, vertex) != 0)
                    {
                        std::cout << " " << cluster;
                    }
                }
                std::cout << "),";
            }
            std::cout << std::endl;

            if (c_f == c_i)
            {
                graphblas::print_matrix(std::cout, tally, "Tally:");
                return c_f;
            }
            else
            {
                c_i = c_f;
            }
            ++iter_num;
        } while (iter_num < max_iters);

        return c_f;
    }

    /**
     * @brief Compute the clusters in the given graph using markov clustering.
     *
     * @param[in] graph     The graph to compute the clusters of.
     * @param[in] i         The inflation parameter (default 2).
     * @param[in] e         The expansion parameter (default 2).
     * @param[in] max_iters The maximum number of iterations to run if
     *                      convergence doesn't occur first.
     *
     * @return A matrix with rows with the same value belong to the same
     *         cluster.
     */
    template<typename MatrixT>
    MatrixT markov_cluster(
        const MatrixT &graph,
        double           e=2.0,
        double           r=2.0,
        unsigned int max_iters = std::numeric_limits<unsigned int>::max())
    {
        using T = typename MatrixT::ScalarType;

        graphblas::IndexType rows, cols;
        graph.get_shape(rows, cols);
        if (rows != cols)
        {
            throw graphblas::DimensionException();
        }

        ssize_t n_vertices = cols;
        MatrixT a(graph);
        MatrixT w(1,n_vertices);

        col_reduce(graph, w);

        for (graphblas::IndexType i = 0; i < n_vertices; i++)
        {
            for (graphblas::IndexType j = 0; j < n_vertices; j++ )
            {
                auto w_0i = w.extractElement(0, i);
                if (w_0i > 0)
                {
                    a.setElement(j, i, (a.extractElement(j, i) / w_0i));
                }
            }
        }

        MatrixT c_i = a;
        MatrixT c_f = a;
        unsigned int iters = 0;
        do
        {
            // Expand.
            for(graphblas::IndexType k = 0; k < (e-1); k++)
            {
                graphblas::mxm(c_i, c_i, c_i);
            }

            // Inflate.
            MatrixT exponent(rows,cols);
            exponent.set_zero(r);

            graphblas::ewisemult(c_i, exponent, c_i,
                      graphblas::math::power<T>,
                      graphblas::math::Assign<T>());

            col_reduce(c_i, w);

            for (graphblas::IndexType i = 0; i < n_vertices; i++)
            {
                for (graphblas::IndexType j = 0; j < n_vertices; j++ )
                {
                    auto w_0i = w.extractElement(0, i);
                    if (w_0i > 0)
                    {
                        c_i.setElement(j, i, (c_i.extractElement(j, i) / w_0i));
                    }
                }
            }

            if(c_f == c_i)
            {
                return c_i;
            }
            else
            {
                c_f = c_i;
            }
            iters++;
        } while(iters < max_iters);
        return c_i;
    }
} // algorithms

#endif // CLUSTER_HPP
