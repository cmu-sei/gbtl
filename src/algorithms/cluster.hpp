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
    GraphBLAS::IndexArrayType get_cluster_assignments(MatrixT cluster_matrix)
    {
        using T = typename MatrixT::ScalarType;

        GraphBLAS::IndexType num_nodes(cluster_matrix.nrows());
        GraphBLAS::IndexType num_clusters(cluster_matrix.ncols());

        GraphBLAS::IndexType nnz = cluster_matrix.nvals();
        GraphBLAS::IndexArrayType cluster_ids(nnz), vertex_ids(nnz);
        std::vector<T> vals(nnz);
        cluster_matrix.extractTuples(cluster_ids.begin(),
                                     vertex_ids.begin(),
                                     vals.begin());

        std::vector<double> max_vals(num_nodes, -1.0);
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
     * @param[in]     graph      NxN adjacency matrix of the graph to compute
     *                           clusters for.
     * @param[in,out] C          On input containes the initial cluster
     *                           approximation to use (identity is good enough).
     *                           On output contains the cluster assignments.
     *                           If element (i,j) == 1, then vertex j belongs
     *                           to cluster i.
     * @param[in]     max_iters  The maximum number of iterations to run if
     *                           convergence doesn't occur first.
     *
     */
    template<typename MatrixT, typename RealT = double>
    void peer_pressure_cluster(
        MatrixT const            &graph,
        GraphBLAS::Matrix<bool>  &C,
        unsigned int  max_iters = std::numeric_limits<unsigned int>::max())
    {
        using T = typename MatrixT::ScalarType;

        size_t iter_num(0);
        GraphBLAS::IndexType num_rows(graph.nrows());
        GraphBLAS::IndexType num_cols(graph.ncols());

        if (num_rows != num_cols)
        {
            throw GraphBLAS::DimensionException();
        }

        // assert C dimensions should be same as graph.
        if (num_rows != C.nrows() ||
            num_cols != C.ncols())
        {
            throw GraphBLAS::DimensionException();
        }

        GraphBLAS::IndexType num_vertices = num_cols;

        //GraphBLAS::print_matrix(std::cerr, graph, "GRAPH");

        // used for breaking ties.
        GraphBLAS::Matrix<GraphBLAS::IndexType> cluster_num_mat(num_vertices,
                                                                num_vertices);
        std::vector<GraphBLAS::IndexType> indices;
        std::vector<GraphBLAS::IndexType> cluster_num; // 1-based
        for (GraphBLAS::IndexType ix=0; ix<num_vertices; ++ix)
        {
            indices.push_back(ix);
            cluster_num.push_back(ix + 1);
        }
        cluster_num_mat.build(indices, indices, cluster_num);
        //GraphBLAS::print_matrix(std::cerr, cluster_num_mat, "cluster_num_mat");

        GraphBLAS::Matrix<RealT> A(num_vertices, num_vertices);
        GraphBLAS::apply(A,
                         GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                         GraphBLAS::Identity<T,RealT>(),
                         graph);
        GraphBLAS::normalize_rows(A);
        //GraphBLAS::print_matrix(std::cerr, A, "row normalized graph");

        //GraphBLAS::print_matrix(std::cerr, C, "cluster_approx");

        GraphBLAS::Vector<RealT> m(num_vertices);
        GraphBLAS::Matrix<bool>  Cf(num_vertices, num_vertices);
        GraphBLAS::Matrix<RealT> Tally(num_vertices, num_vertices);

        for (unsigned int iters = 0; iters < max_iters; ++iters)
        {
            //std::cerr << "===================== Iteration " << iters << std::endl;
            GraphBLAS::mxm(Tally,
                           GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                           GraphBLAS::ArithmeticSemiring<RealT>(),
                           C, A);
            //GraphBLAS::print_matrix(std::cerr, Tally, "Tally");

            // Find the largest element (max vote) in each column
            GraphBLAS::reduce(m,
                              GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                              GraphBLAS::Max<RealT>(),
                              GraphBLAS::transpose(Tally)); //col reduce
            //GraphBLAS::print_vector(std::cerr, m, "col_max(Tally)");

            auto m_mat(GraphBLAS::diag<GraphBLAS::Matrix<RealT>>(m));
            //print_matrix(std::cout, m_mat, "diag(col_max)");

            GraphBLAS::mxm(Cf,
                           GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                           PlusEqualSemiring<RealT, RealT, bool>(),
                           Tally, m_mat);
            //GraphBLAS::print_matrix(std::cerr, Cf, "Next cluster mat");

            // -------------------------------------------------------------
            // Need to pick one element per column (break any ties by picking
            // highest cluster number per column)
            GraphBLAS::mxm(Tally,
                           Cf, GraphBLAS::NoAccumulate(),
                           GraphBLAS::ArithmeticSemiring<RealT>(),
                           cluster_num_mat, Cf, true);
            //GraphBLAS::print_matrix(std::cerr, Tally,
            //                        "Next cluster mat x clusternum");
            GraphBLAS::reduce(m,
                              GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                              GraphBLAS::Max<RealT>(),
                              GraphBLAS::transpose(Tally)); //col reduce
            //print_vector(std::cout, m, "col max(Tally)");
            m_mat = GraphBLAS::diag<GraphBLAS::Matrix<RealT>>(m);
            GraphBLAS::mxm(Cf,
                           GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                           PlusEqualSemiring<RealT, RealT, bool>(),
                           Tally, m_mat);
            //GraphBLAS::print_matrix(std::cerr, Cf, "Next cluster mat, no ties");

            // mask out unselected ties (annihilate).
            GraphBLAS::apply(Cf,
                             Cf, GraphBLAS::NoAccumulate(),
                             GraphBLAS::Identity<bool>(),
                             Cf, true);
            //GraphBLAS::print_matrix(std::cerr, Cf, "Next cluster mat, masked");

            if (Cf == C)
            {
                break;
            }

            C = Cf;
        }
        // MAX_ITERS EXCEEDED
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
        auto C(
            GraphBLAS::scaled_identity<GraphBLAS::Matrix<bool>>(graph.nrows()));
        peer_pressure_cluster(graph, C, max_iters);
        return C;
    }

    //************************************************************************
    /**
     * @brief Compute the clusters in the given graph using a variant of the
     *        peer-pressure clustering implementation.
     *
     * @note This variant does not normalize the edge weight by vertex degree
     *
     * @param[in]     graph      NxN adjacency matrix of the graph to compute
     *                           clusters for.
     * @param[in,out] C          On input containes the initial cluster
     *                           approximation to use (identity is good enough).
     *                           On output contains the cluster assignments.
     *                           If element (i,j) == 1, then vertex j belongs
     *                           to cluster i.
     * @param[in]     max_iter   The maximum number of iterations to run if
     *                           convergence doesn't occur first.
     *
     * @todo This should be merged with the other version and a configuration
     *       parameter should be added for each difference.
     */
    template<typename MatrixT, typename RealT = double>
    void peer_pressure_cluster_v2(
        MatrixT const            &graph,
        GraphBLAS::Matrix<bool>  &C,
        unsigned int  max_iters = std::numeric_limits<unsigned int>::max())
    {
        using T = typename MatrixT::ScalarType;

        size_t iter_num(0);
        GraphBLAS::IndexType num_rows(graph.nrows());
        GraphBLAS::IndexType num_cols(graph.ncols());

        if (num_rows != num_cols)
        {
            throw GraphBLAS::DimensionException();
        }

        // assert C dimensions should be same as graph.
        if (num_rows != C.nrows() ||
            num_cols != C.ncols())
        {
            throw GraphBLAS::DimensionException();
        }

        GraphBLAS::IndexType num_vertices = num_cols;

        // used for breaking ties.
        GraphBLAS::Matrix<GraphBLAS::IndexType> cluster_num_mat(num_vertices,
                                                                num_vertices);
        std::vector<GraphBLAS::IndexType> indices;
        std::vector<GraphBLAS::IndexType> cluster_num;
        for (GraphBLAS::IndexType ix=0; ix<num_vertices; ++ix)
        {
            indices.push_back(ix);
            cluster_num.push_back(ix + 1);
        }
        cluster_num_mat.build(indices, indices, cluster_num);

        // Option: normalize the rows of G (using double scalar type)
        MatrixT A(num_vertices, num_vertices);
        GraphBLAS::apply(A,
                         GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                         GraphBLAS::Identity<T,RealT>(),
                         graph);
        // Disabling normalization across rows
        //GraphBLAS::normalize_rows(A);
        //GraphBLAS::print_matrix(std::cout, A, "Row normalized graph");

        GraphBLAS::Vector<RealT> m(num_vertices);
        GraphBLAS::Matrix<bool>  Cf(num_vertices, num_vertices);
        GraphBLAS::Matrix<RealT> Tally(num_vertices, num_vertices);

        for (unsigned int iters = 0; iters < max_iters; ++iters)
        {
            GraphBLAS::mxm(Tally,
                           GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                           GraphBLAS::ArithmeticSemiring<RealT>(),
                           C, A);

            // Find the largest element (max vote) in each column
            GraphBLAS::reduce(m,
                              GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                              GraphBLAS::Max<RealT>(),
                              GraphBLAS::transpose(Tally)); //col reduce

            auto m_mat(GraphBLAS::diag<GraphBLAS::Matrix<RealT>>(m));

            GraphBLAS::mxm(Cf,
                           GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                           PlusEqualSemiring<RealT, RealT, bool>(),
                           Tally, m_mat);

            // -------------------------------------------------------------
            // Need to pick one element per column (break any ties by picking
            // highest cluster number per column)
            GraphBLAS::mxm(Tally,
                           Cf, GraphBLAS::NoAccumulate(),
                           GraphBLAS::ArithmeticSemiring<RealT>(),
                           cluster_num_mat, Cf, true);

            GraphBLAS::reduce(m,
                              GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                              GraphBLAS::Max<RealT>(),
                              GraphBLAS::transpose(Tally)); //col reduce

            m_mat = GraphBLAS::diag<GraphBLAS::Matrix<RealT>>(m);
            GraphBLAS::mxm(Cf,
                           GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                           PlusEqualSemiring<RealT, RealT, bool>(),
                           Tally, m_mat);

            // mask out unselected ties (annihilate).
            GraphBLAS::apply(Cf,
                             Cf, GraphBLAS::NoAccumulate(),
                             GraphBLAS::Identity<bool>(),
                             Cf, true);

            if (Cf == C)
            {
                break;
            }

            C = Cf;
        }

        // MAX_ITERS EXCEEDED
    }

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
     * @param[in] convergence_threshold  If the mean-squared difference in the
     *                      cluster_matrix of two successive iterations falls
     *                      below this threshold the algorithm will return the
     *                      last computed matrix.
     *
     * @return A matrix whose columns correspond to the vertices, and vertices
     *         with the same (max) value in a given row belong to the
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
        //GraphBLAS::print_matrix(std::cout, Anorm, "Anorm");

        for (unsigned int iter = 0; iter < max_iters; ++iter)
        {
            //std::cout << "================= ITERATION " << iter << "============="
            //          << std::endl;
            //GraphBLAS::print_matrix(std::cout, Anorm, "A(col norm)");

            // Power Expansion: compute Anorm^e.
            RealMatrixT Apower(Anorm);
            for (GraphBLAS::IndexType k = 0; k < (e - 1); ++k)
            {
                GraphBLAS::mxm(Apower,
                               Apower, GraphBLAS::NoAccumulate(),
                               GraphBLAS::ArithmeticSemiring<RealT>(),
                               Apower, Anorm, true);
            }

            //GraphBLAS::print_matrix(std::cout, Apower, "Apower = Anorm^e");

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
            //GraphBLAS::print_matrix(std::cout, Ainfl, "Ainfl = Apower .^ r");
            GraphBLAS::normalize_cols(Ainfl);
            //GraphBLAS::print_matrix(std::cout, Ainfl, "*** new Anorm ***");

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
            //std::cout << "****|error|^2/N^2 = " << mean_squared_error << std::endl;

            if (mean_squared_error < convergence_threshold) //(Ainfl == Anorm)
            {
                //std::cout << "****CONVERGED****" << std::endl;
                return Ainfl;
            }

            Anorm = Ainfl;
        }

        //std::cout << "Exceeded max interations." << std::endl;
        return Anorm;
    }

} // algorithms

#endif // CLUSTER_HPP
