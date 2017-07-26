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

//****************************************************************************
namespace algorithms
{
    GEN_GRAPHBLAS_SEMIRING(OrEqualSemiring,
                           GraphBLAS::LogicalOrMonoid,
                           GraphBLAS::Equal)

    GEN_GRAPHBLAS_SEMIRING(PlusEqualSemiring,
                           GraphBLAS::PlusMonoid,
                           GraphBLAS::Equal)

    //************************************************************************
    /**
     * @brief Assign a zero-based cluster ID to each vertex based on the
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
    GraphBLAS::Vector<GraphBLAS::IndexType> get_cluster_assignments_v2(
        MatrixT const &cluster_matrix)
    {
        GraphBLAS::IndexType num_clusters(cluster_matrix.nrows());
        GraphBLAS::IndexType num_nodes(cluster_matrix.ncols());

        // GraphBLAS::IndexArrayType clusters(
        //     num_nodes,
        //     std::numeric_limits<GraphBLAS::IndexType>::max());

        // /// @todo this can be replaced with MaxSelect1st (transposed)
        // for (GraphBLAS::IndexType v = 0; v < num_nodes; ++v)
        // {
        //     double max_val = 0.0;
        //     for (GraphBLAS::IndexType c = 0; c < num_clusters; ++c)
        //     {
        //         if (cluster_matrix.extractElement(c, v) > max_val)
        //         {
        //             max_val = cluster_matrix.extractElement(c, v);
        //             clusters[v] = c;
        //         }
        //     }
        // }

        GraphBLAS::Vector<GraphBLAS::IndexType> clusters(num_nodes);
        GraphBLAS::Vector<GraphBLAS::IndexType> index_of_vec(num_clusters);
        std::vector<GraphBLAS::IndexType> indices;
        for (GraphBLAS::IndexType ix=0; ix<num_clusters; ++ix)
        {
            indices.push_back(ix);
        }
        index_of_vec.build(indices, indices);

        // return a GraphBLAS::Vector with cluster assignments
        GraphBLAS::vxm(clusters,
                       GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                       GraphBLAS::MaxSelect1stSemiring<GraphBLAS::IndexType>(),
                       index_of_vec, cluster_matrix);


        return clusters;
    }

    //************************************************************************
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
    GraphBLAS::IndexArrayType get_cluster_assignments(MatrixT cluster_matrix)
    {
        using T = typename MatrixT::ScalarType;

        GraphBLAS::IndexType num_nodes(cluster_matrix.nrows());
        GraphBLAS::IndexType num_clusters(cluster_matrix.ncols());

        GraphBLAS::IndexType nnz = cluster_matrix.nvals();
        GraphBLAS::IndexArrayType cluster_ids(nnz), vertex_ids(nnz);
        std::vector<T> vals(nnz);
        cluster_matrix.extractTuples(cluster_ids.begin(), vertex_ids.begin(), vals.begin());

        std::vector<T> max_vals(num_nodes, -1.0);
        GraphBLAS::IndexArrayType cluster_assignments(
            num_nodes,
            std::numeric_limits<GraphBLAS::IndexType>::max());

        for (GraphBLAS::IndexType idx = 0; idx < vals.size(); ++idx)
        {
            if (max_vals[vertex_ids[idx]] < vals[idx])
            {
                max_vals[vertex_ids[idx]] = vals[idx];
                cluster_assignments[vertex_ids[idx]] = cluster_ids[idx];
            }
        }
        return cluster_assignments;
    }
#if 0
    //************************************************************************
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
    template<typename MatrixT>
    GraphBLAS::Matrix<bool> peer_pressure_cluster(
        MatrixT const            &graph,
        GraphBLAS::Matrix<bool>  &cluster_approx,
        unsigned int              max_iters = std::numeric_limits<unsigned int>::max())
    {
        using T = typename MatrixT::ScalarType;
        using VectorT = GraphBLAS::Vector<T>;

        size_t iter_num(0);
        GraphBLAS::IndexType rows(graph.nrows());
        GraphBLAS::IndexType n_vertices(graph.ncols());

        if (rows != n_vertices)
        {
            throw GraphBLAS::DimensionException();
        }

        // assert cluster_approx dimensions should be same as graph.
        if (rows != cluster_approx.nrows() ||
            n_vertices != cluster_approx.ncols())
        {
            throw GraphBLAS::DimensionException();
        }

        GraphBLAS::print_matrix(std::cerr, graph, "GRAPH");

        GraphBLAS::Matrix<GraphBLAS::IndexType> index_of_mat(n_vertices,
                                                             n_vertices);
        std::vector<GraphBLAS::IndexType> indices;
        for (GraphBLAS::IndexType ix=0; ix<n_vertices; ++ix)
        {
            indices.push_back(ix);
        }
        index_of_mat.build(indices, indices, indices);
        GraphBLAS::print_matrix(std::cerr, index_of_mat, "INDEX_OF(mat)");

        // Noramlize the rows of graph and assign to a
        GraphBLAS::Vector<float> w(n_vertices);

        //row_reduce(graph, w);
        GraphBLAS::reduce(w,
                          GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                          GraphBLAS::Plus<float>(),
                          graph);

        GraphBLAS::print_vector(std::cerr, w, "Node out 'degree'");

        /// @todo Replace with singleton expansion of w and then ewise ops.
        //for (GraphBLAS::IndexType i = 0; i < n_vertices; ++i)
        //{
        //    for (GraphBLAS::IndexType j = 0; j < n_vertices; ++j)
        //    {
        //        auto w_i0 = w.extractElement(i, 0);
        //        if (w_i0 > 0)
        //        {
        //            a.setElement(i, j, (a.extractElement(i, j) / w_i0));
        //        }
        //    }
        //}

        GraphBLAS::Vector<float> w_inv(n_vertices);
        GraphBLAS::apply(w_inv,
                         GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                         GraphBLAS::MultiplicativeInverse<float>(),
                         w);
        auto winv_mat(GraphBLAS::diag<MatrixT, VectorT>(w_inv));

        GraphBLAS::print_matrix(std::cerr, winv_mat, "Node out 'degree' inverse");

        GraphBLAS::Matrix<float> a(n_vertices, n_vertices);
        GraphBLAS::mxm(a,
                       GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                       GraphBLAS::ArithmeticSemiring<T>(),
                       winv_mat, graph);

        GraphBLAS::print_matrix(std::cerr, a, "winv_mat * graph");

        GraphBLAS::Matrix<bool> c_i(cluster_approx);

        GraphBLAS::print_matrix(std::cerr, cluster_approx, "cluster_approx");

        for (unsigned int iters = 0; iters < max_iters; ++iters)
        {
            std::cerr << "===================== Iteration " << iters << std::endl;
            GraphBLAS::Matrix<float> tally(n_vertices, n_vertices);
            //GraphBLAS::mxm(c_i, a, tally);
            GraphBLAS::mxm(tally,
                           GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                           GraphBLAS::ArithmeticSemiring<T>(),
                           c_i, a);

            GraphBLAS::print_matrix(std::cerr, tally, "Tally");

            // Find the largest element in each column
            // std::vector<GraphBLAS::IndexType> m;
            // for (GraphBLAS::IndexType vertex = 0; vertex < n_vertices; ++vertex)
            // {
            //     GraphBLAS::IndexType maxpos = 0;
            //     for (GraphBLAS::IndexType cluster = 1;
            //          cluster < n_vertices;
            //          ++cluster)
            //     {
            //         if (tally.extractElement(maxpos, vertex) <
            //             tally.extractElement(cluster, vertex))
            //         {
            //             maxpos = cluster;
            //         }
            //     }
            //     m.push_back(maxpos);
            // }

            GraphBLAS::Vector<float> m(n_vertices);
            GraphBLAS::reduce(m,
                              GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                              GraphBLAS::Max<float>(),
                              GraphBLAS::transpose(tally));

            GraphBLAS::print_vector(std::cerr, m, "col_max(Tally)");

            auto m_mat(GraphBLAS::diag<GraphBLAS::Matrix<float>>(m));

            //MatrixT c_f(n_vertices, n_vertices);
            GraphBLAS::Matrix<bool> c_f(n_vertices, n_vertices);
            GraphBLAS::mxm(c_f,
                           GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                           PlusEqualSemiring<float, float, bool>(),
                           m_mat, tally);

            GraphBLAS::print_matrix(std::cerr, c_f, "Next cluster mat");

            return c_f;
            // for (GraphBLAS::IndexType vertex = 0; vertex < n_vertices; ++vertex)
            // {
            //     c_f.setElement(m[vertex], vertex, 1);
            // }

            // if (c_f == c_i)  // costly comparison?
            // {
            //     return c_f;
            // }
            // else
            // {
            //     c_i = c_f;
            // }
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
    template <typename MatrixT>
    GraphBLAS::Matrix<bool> peer_pressure_cluster(
        MatrixT const &graph,
        unsigned int   max_iters = std::numeric_limits<unsigned int>::max())
    {
        // assign each vertex to its own cluster (boolean matrix)
        auto cluster_approx(
            GraphBLAS::scaled_identity<GraphBLAS::Matrix<bool>>(graph.nrows()));
        return peer_pressure_cluster(graph, cluster_approx, max_iters);
    }

    /**
     * @brief Compute the clusters in the given graph using a varient of the
     *        peer-pressure clustering implementation.
     *
     * @note This variant does not normalize the edge weight by vertex degree
     *
     * @param[in] graph          The graph to compute the clusters of (must be a
     *                           floating point type)
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
        using T = typename MatrixT::ScalarType;
        using VectorT = GraphBLAS::Vector<T>;

        size_t iter_num(0);
        GraphBLAS::IndexType num_rows(graph.nrows());
        GraphBLAS::IndexType num_cols(graph.ncols());

        if (num_rows != num_cols)
        {
            throw GraphBLAS::DimensionException();
        }

        // assert cluster_approx dimensions should be same as graph.
        if (num_rows != cluster_approx.nrows() ||
            num_cols != cluster_approx.ncols())
        {
            throw GraphBLAS::DimensionException();
        }

        GraphBLAS::IndexType num_vertices = num_cols;

        // Option: normalize the rows of G

        //for (IndexType i = 0; i < num_vertices; i++)
        //{
        //    for (IndexType j = 0; j < num_vertices; j++ )
        //    {
        //        if (w[i] > 0)
        //        {
        //            a[i][j] = a[i][j] / w[i];
        //        }
        //    }
        //}

        MatrixT A(num_vertices, num_vertices);
        VectorT w(num_vertices);
        VectorT w_inv(num_vertices);

        // GraphBLAS::row_reduce(graph, w);
        // GraphBLAS::apply(w, w_inv, GraphBLAS::math::Inverse<double>());
        // GraphBLAS::ewisemult(
        //     graph,
        //     GraphBLAS::RowExtendedView<MatrixT>(0, w_inv, num_vertices),
        //     a);

        GraphBLAS::reduce(w,
                          GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                          GraphBLAS::Plus<T>(),
                          graph);
        GraphBLAS::apply(w_inv,
                         GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                         GraphBLAS::MultiplicativeInverse<T>(),
                         w);
        auto winv_mat(GraphBLAS::diag<MatrixT, VectorT>(w_inv));
        GraphBLAS::mxm(A,
                       GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                       GraphBLAS::ArithmeticSemiring<T>(),
                       winv_mat, graph);

        GraphBLAS::print_matrix(std::cout, A, "Row normalized graph");

        // throwing away normalization across rows
        MatrixT a(graph);

        ClusterMatrixT c_i(cluster_approx);
        ClusterMatrixT c_f(num_vertices, num_vertices);

        do
        {
            MatrixT tally(num_vertices, num_vertices);
            //GraphBLAS::mxm(c_i, a, tally);
            GraphBLAS::mxm(tally,
                           GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                           GraphBLAS::ArithmeticSemiring<T>(),
                           c_i, a);

            //MatrixT col_max = MatrixT(1, num_vertices);
            //GraphBLAS::col_reduce(tally,
            //                      col_max,
            //                      GraphBLAS::RegmaxMonoid<double>());
            VectorT col_max(num_vertices);
            GraphBLAS::reduce(col_max,
                              GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                              GraphBLAS::MaxMonoid<T>(),
                              tally);

            // GraphBLAS::ewisemult(
            //     tally,
            //     GraphBLAS::ColumnExtendedView<MatrixT>(0,
            //                                            col_max,
            //                                            num_vertices),
            //     c_f,
            //     GraphBLAS::math::IsEqual<double>());
            auto tally_mat(GraphBLAS::diag<MatrixT, VectorT>(col_max));

            std::cout << "Iteration: " << iter_num << std::endl;
            for (GraphBLAS::IndexType vertex = 0;
                 vertex < num_vertices;
                 ++vertex)
            {
                std::cout << "(" << vertex << ":";
                for (GraphBLAS::IndexType cluster = 0;
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
                GraphBLAS::print_matrix(std::cout, tally, "Tally:");
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

    //************************************************************************
    /**
     * @brief Compute the clusters in the given graph using markov clustering.
     *
     * @param[in] graph     The graph to compute the clusters of.  It is
     *                      recommended that graph has self loops added.
     * @param[in] e         The power parameter (default 2).
     * @param[in] i         The inflation parameter (default 2).
     * @param[in] max_iters The maximum number of iterations to run if
     *                      convergence doesn't occur first (oscillation common)
     *
     * @return A matrix where vertices in rows with the same value belong to the
     *         same cluster.
     */
    template<typename MatrixT>
    MatrixT markov_cluster_old(
        MatrixT const        &graph,
        GraphBLAS::IndexType  e = 2,
        GraphBLAS::IndexType  i = 2,
        unsigned int          max_iters = std::numeric_limits<unsigned int>::max())
    {
        using T = typename MatrixT::ScalarType;

        GraphBLAS::IndexType rows(graph.nrows());
        GraphBLAS::IndexType cols(graph.ncols());
        if (rows != cols)
        {
            throw GraphBLAS::DimensionException();
        }

        MatrixT a(graph);

        //col_reduce(graph, w);
        GraphBLAS::Vector<T> w(cols);
        GraphBLAS::reduce(w,
                          GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                          GraphBLAS::Plus<T>(),
                          GraphBLAS::transpose(graph));

        ssize_t n_vertices = cols;
        for (GraphBLAS::IndexType i = 0; i < n_vertices; i++)
        {
            for (GraphBLAS::IndexType j = 0; j < n_vertices; j++ )
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
            for(GraphBLAS::IndexType k = 0; k < (e-1); k++)
            {
                GraphBLAS::mxm(c_i, c_i, c_i);
            }

            // Inflate.
            MatrixT exponent(rows,cols);
            exponent.set_zero(r);

            GraphBLAS::ewisemult(c_i, exponent, c_i,
                      GraphBLAS::math::power<T>,
                      GraphBLAS::math::Assign<T>());

            col_reduce(c_i, w);

            for (GraphBLAS::IndexType i = 0; i < n_vertices; i++)
            {
                for (GraphBLAS::IndexType j = 0; j < n_vertices; j++ )
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
#endif

    //************************************************************************
    /**
     * @brief Compute the clusters in the given graph using markov clustering.
     *
     * @param[in] graph     The graph to compute the clusters of.  It is
     *                      recommended that graph has self loops added.
     * @param[in] e         The power parameter (default 2).
     * @param[in] r         The inflation parameter (default 2).
     * @param[in] max_iters The maximum number of iterations to run if
     *                      convergence doesn't occur first (oscillation common)
     *
     * @return A matrix where vertices in rows with the same value belong to the
     *         same cluster.
     */
    template<typename MatrixT, typename RealT=double>
    GraphBLAS::Matrix<RealT> markov_cluster(
        MatrixT const        &graph,
        GraphBLAS::IndexType  e = 2,
        GraphBLAS::IndexType  r = 2,
        unsigned int          max_iters = std::numeric_limits<unsigned int>::max(),
        double                convergence_threshold = 1.0e-16)
    {
        using T = typename MatrixT::ScalarType;
        using RealMatrixT = GraphBLAS::Matrix<RealT>;

        GraphBLAS::IndexType rows(graph.nrows());
        GraphBLAS::IndexType cols(graph.ncols());
        if (rows != cols)
        {
            throw GraphBLAS::DimensionException();
        }

        // A = (RealT)graph
        RealMatrixT Anorm(rows, cols);
        GraphBLAS::apply(Anorm,
                         GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                         GraphBLAS::Identity<RealT>(),
                         graph);
        GraphBLAS::normalize_cols(Anorm);
        GraphBLAS::print_matrix(std::cout, Anorm, "Anorm");

        for (unsigned int iter = 0; iter < max_iters; ++iter)
        {
            std::cout << "================= ITERATION " << iter << "============="
                      << std::endl;
            GraphBLAS::print_matrix(std::cout, Anorm, "A(col norm)");

            // Power Expansion: compute Anorm^e.
            RealMatrixT Apower(Anorm);
            for (GraphBLAS::IndexType k = 0; k < (e - 1); ++k)
            {
                GraphBLAS::mxm(Apower,
                               Apower, GraphBLAS::NoAccumulate(),
                               GraphBLAS::ArithmeticSemiring<RealT>(),
                               Apower, Anorm, true);
            }

            GraphBLAS::print_matrix(std::cout, Apower, "Apower = Anorm^e");

            // Inflate. element wise raise to power r, normalize columns
            RealMatrixT Ainfl(Apower);
            for (GraphBLAS::IndexType k = 0; k < (r - 1); ++k)
            {
                GraphBLAS::eWiseMult(Ainfl,
                                     Ainfl,
                                     GraphBLAS::NoAccumulate(),
                                     GraphBLAS::Times<RealT>(),
                                     Ainfl, Apower, true);
            }
            GraphBLAS::print_matrix(std::cout, Ainfl, "Ainfl = Apower .^ r");
            GraphBLAS::normalize_cols(Ainfl);
            GraphBLAS::print_matrix(std::cout, Ainfl, "*** new Anorm ***");

            // Compute mean squared error
            RealMatrixT E(rows, cols);
            GraphBLAS::eWiseAdd(E,
                                GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                GraphBLAS::Minus<RealT>(),
                                Anorm, Ainfl);
            GraphBLAS::eWiseMult(E,
                                 GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                 GraphBLAS::Times<RealT>(),
                                 E, E);
            RealT mean_squared_error(0.);
            GraphBLAS::reduce(mean_squared_error,
                              GraphBLAS::NoAccumulate(),
                              GraphBLAS::PlusMonoid<RealT>(),
                              E);
            mean_squared_error /= ((double)rows*(double)rows);
            std::cout << "****|error|^2/N^2 = " << mean_squared_error << std::endl;

            if (mean_squared_error < convergence_threshold) //(Ainfl == Anorm)
            {
                std::cout << "****CONVERGED****" << std::endl;
                return Ainfl;
            }

            Anorm = Ainfl;
        }

        std::cout << "Exceeded max interations." << std::endl;
        return Anorm;
    }

} // algorithms

#endif // CLUSTER_HPP
