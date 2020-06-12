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

#include <vector>
#include <math.h>
#include <graphblas/graphblas.hpp>

//****************************************************************************
namespace algorithms
{
    GEN_GRAPHBLAS_SEMIRING(PlusEqualSemiring,
                           grb::PlusMonoid,
                           grb::Equal)
}

namespace algorithms
{
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
    grb::IndexArrayType get_cluster_assignments(MatrixT cluster_matrix)
    {
        using T = typename MatrixT::ScalarType;

        grb::IndexType num_nodes(cluster_matrix.nrows());

        grb::IndexType nnz = cluster_matrix.nvals();
        grb::IndexArrayType cluster_ids(nnz), vertex_ids(nnz);
        std::vector<T> vals(nnz);
        cluster_matrix.extractTuples(cluster_ids.begin(),
                                     vertex_ids.begin(),
                                     vals.begin());

        std::vector<double> max_vals(num_nodes, -1.0);
        grb::IndexArrayType cluster_assignments(
            num_nodes,
            std::numeric_limits<grb::IndexType>::max());

        for (grb::IndexType idx = 0; idx < vals.size(); ++idx)
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
    grb::Vector<grb::IndexType> get_cluster_assignments_v2(
        MatrixT const &cluster_matrix)
    {
        grb::IndexType num_clusters(cluster_matrix.nrows());
        grb::IndexType num_nodes(cluster_matrix.ncols());

        grb::Vector<grb::IndexType> clusters(num_nodes);
        grb::Vector<grb::IndexType> index_of_vec(num_clusters);
        for (grb::IndexType ix=0; ix<num_clusters; ++ix)
        {
            index_of_vec.setElement(ix, ix);
        }

        // return a grb::Vector with cluster assignments
        grb::vxm(clusters,
                 grb::NoMask(), grb::NoAccumulate(),
                 grb::MaxFirstSemiring<grb::IndexType>(),
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
        MatrixT const     &graph,
        grb::Matrix<bool> &C,
        unsigned int       max_iters = std::numeric_limits<unsigned int>::max())
    {
        using T = typename MatrixT::ScalarType;

        grb::IndexType num_rows(graph.nrows());
        grb::IndexType num_cols(graph.ncols());

        if (num_rows != num_cols)
        {
            throw grb::DimensionException();
        }

        // assert C dimensions should be same as graph.
        if (num_rows != C.nrows() ||
            num_cols != C.ncols())
        {
            throw grb::DimensionException();
        }

        grb::IndexType num_vertices = num_cols;

        //grb::print_matrix(std::cerr, graph, "GRAPH");

        // used for breaking ties.
        /// @todo figure out how to remove "+ 1"
        grb::Matrix<grb::IndexType> cluster_num_mat(num_vertices,
                                                    num_vertices);
        for (grb::IndexType ix=0; ix<num_vertices; ++ix)
        {
            cluster_num_mat.setElement(ix, ix, ix + 1);
        }
        //grb::print_matrix(std::cerr, cluster_num_mat, "cluster_num_mat");

        grb::Matrix<RealT> A(num_vertices, num_vertices);
        grb::apply(A,
                   grb::NoMask(), grb::NoAccumulate(),
                   grb::Identity<T,RealT>(),
                   graph);
        grb::normalize_rows(A);
        //grb::print_matrix(std::cerr, A, "row normalized graph");

        //grb::print_matrix(std::cerr, C, "cluster_approx");

        grb::Vector<RealT> m(num_vertices);
        grb::Matrix<bool>  Cf(num_vertices, num_vertices);
        grb::Matrix<RealT> Tally(num_vertices, num_vertices);

        for (unsigned int iters = 0; iters < max_iters; ++iters)
        {
            //std::cerr << "===================== Iteration " << iters << std::endl;
            grb::mxm(Tally,
                     grb::NoMask(), grb::NoAccumulate(),
                     grb::ArithmeticSemiring<RealT>(),
                     C, A);
            //grb::print_matrix(std::cerr, Tally, "Tally");

            // Find the largest element (max vote) in each column
            grb::reduce(m,
                        grb::NoMask(), grb::NoAccumulate(),
                        grb::Max<RealT>(),
                        grb::transpose(Tally)); //col reduce
            //grb::print_vector(std::cerr, m, "col_max(Tally)");

            auto m_mat(grb::diag<grb::Matrix<RealT>>(m));
            //print_matrix(std::cout, m_mat, "diag(col_max)");

            grb::mxm(Cf,
                     grb::NoMask(), grb::NoAccumulate(),
                     PlusEqualSemiring<RealT, RealT, bool>(),
                     Tally, m_mat);
            //grb::print_matrix(std::cerr, Cf, "Next cluster mat");

            // -------------------------------------------------------------
            // Need to pick one element per column (break any ties by picking
            // highest cluster number per column)
            grb::mxm(Tally,
                     Cf, grb::NoAccumulate(),
                     grb::ArithmeticSemiring<RealT>(),
                     cluster_num_mat, Cf, grb::REPLACE);
            //grb::print_matrix(std::cerr, Tally,
            //                        "Next cluster mat x clusternum");
            grb::reduce(m,
                        grb::NoMask(), grb::NoAccumulate(),
                        grb::Max<RealT>(),
                        grb::transpose(Tally)); //col reduce
            //print_vector(std::cout, m, "col max(Tally)");
            m_mat = grb::diag<grb::Matrix<RealT>>(m);
            grb::mxm(Cf,
                     grb::NoMask(), grb::NoAccumulate(),
                     PlusEqualSemiring<RealT, RealT, bool>(),
                     Tally, m_mat);
            //grb::print_matrix(std::cerr, Cf, "Next cluster mat, no ties");

            // mask out unselected ties (annihilate).
            grb::apply(Cf,
                       Cf, grb::NoAccumulate(),
                       grb::Identity<bool>(),
                       Cf, grb::REPLACE);
            //grb::print_matrix(std::cerr, Cf, "Next cluster mat, masked");

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
    grb::Matrix<bool> peer_pressure_cluster(
        MatrixT const  &graph,
        unsigned int    max_iters = std::numeric_limits<unsigned int>::max())
    {
        // assign each vertex to its own cluster (boolean matrix)
        auto C(grb::scaled_identity<grb::Matrix<bool>>(graph.nrows()));
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
     * @param[in]     max_iters  The maximum number of iterations to run if
     *                           convergence doesn't occur first.
     *
     * @todo This should be merged with the other version and a configuration
     *       parameter should be added for each difference.
     */
    template<typename MatrixT, typename RealT = double>
    void peer_pressure_cluster_v2(
        MatrixT const      &graph,
        grb::Matrix<bool>  &C,
        unsigned int        max_iters = std::numeric_limits<unsigned int>::max())
    {
        using T = typename MatrixT::ScalarType;

        grb::IndexType num_rows(graph.nrows());
        grb::IndexType num_cols(graph.ncols());

        if (num_rows != num_cols)
        {
            throw grb::DimensionException();
        }

        // assert C dimensions should be same as graph.
        if (num_rows != C.nrows() ||
            num_cols != C.ncols())
        {
            throw grb::DimensionException();
        }

        grb::IndexType num_vertices = num_cols;

        // used for breaking ties.
        /// @todo Figure out way to not add 1.
        grb::Matrix<grb::IndexType> cluster_num_mat(num_vertices,
                                                    num_vertices);
        for (grb::IndexType ix=0; ix<num_vertices; ++ix)
        {
            cluster_num_mat.setElement(ix, ix, ix + 1);
        }

        // Option: normalize the rows of G (using double scalar type)
        MatrixT A(num_vertices, num_vertices);
        grb::apply(A,
                   grb::NoMask(), grb::NoAccumulate(),
                   grb::Identity<T,RealT>(),
                   graph);
        // Disabling normalization across rows
        //grb::normalize_rows(A);
        //grb::print_matrix(std::cout, A, "Row normalized graph");

        grb::Vector<RealT> m(num_vertices);
        grb::Matrix<bool>  Cf(num_vertices, num_vertices);
        grb::Matrix<RealT> Tally(num_vertices, num_vertices);

        for (unsigned int iters = 0; iters < max_iters; ++iters)
        {
            grb::mxm(Tally,
                     grb::NoMask(), grb::NoAccumulate(),
                     grb::ArithmeticSemiring<RealT>(),
                     C, A);

            // Find the largest element (max vote) in each column
            grb::reduce(m,
                        grb::NoMask(), grb::NoAccumulate(),
                        grb::Max<RealT>(),
                        grb::transpose(Tally)); //col reduce

            auto m_mat(grb::diag<grb::Matrix<RealT>>(m));

            grb::mxm(Cf,
                     grb::NoMask(), grb::NoAccumulate(),
                     PlusEqualSemiring<RealT, RealT, bool>(),
                     Tally, m_mat);

            // -------------------------------------------------------------
            // Need to pick one element per column (break any ties by picking
            // highest cluster number per column)
            grb::mxm(Tally,
                     Cf, grb::NoAccumulate(),
                     grb::ArithmeticSemiring<RealT>(),
                     cluster_num_mat, Cf, grb::REPLACE);

            grb::reduce(m,
                        grb::NoMask(), grb::NoAccumulate(),
                        grb::Max<RealT>(),
                        grb::transpose(Tally)); //col reduce

            m_mat = grb::diag<grb::Matrix<RealT>>(m);
            grb::mxm(Cf,
                     grb::NoMask(), grb::NoAccumulate(),
                     PlusEqualSemiring<RealT, RealT, bool>(),
                     Tally, m_mat);

            // mask out unselected ties (annihilate).
            grb::apply(Cf,
                       Cf, grb::NoAccumulate(),
                       grb::Identity<bool>(),
                       Cf, grb::REPLACE);

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
     * @param[in] convergence_threshold  If the mean-squared difference in the
     *                      cluster_matrix of two successive iterations falls
     *                      below this threshold the algorithm will return the
     *                      last computed matrix.
     * @param[in] max_iters The maximum number of iterations to run if
     *                      convergence doesn't occur first (oscillation common)
     *
     * @return A matrix whose columns correspond to the vertices, and vertices
     *         with the same (max) value in a given row belong to the
     *         same cluster.
     */
    template<typename MatrixT, typename RealT=double>
    grb::Matrix<RealT> markov_cluster(
        MatrixT const  &graph,
        grb::IndexType  e = 2,
        grb::IndexType  r = 2,
        double          convergence_threshold = 1.0e-16,
        unsigned int    max_iters = std::numeric_limits<unsigned int>::max())
    {
        using RealMatrixT = grb::Matrix<RealT>;

        grb::IndexType rows(graph.nrows());
        grb::IndexType cols(graph.ncols());
        if (rows != cols)
        {
            throw grb::DimensionException();
        }

        // A = (RealT)graph
        RealMatrixT Anorm(rows, cols);
        grb::apply(Anorm,
                   grb::NoMask(), grb::NoAccumulate(),
                   grb::Identity<RealT>(),
                   graph);
        grb::normalize_cols(Anorm);
        //grb::print_matrix(std::cout, Anorm, "Anorm");

        for (unsigned int iter = 0; iter < max_iters; ++iter)
        {
            //std::cout << "================= ITERATION " << iter << "============="
            //          << std::endl;
            //grb::print_matrix(std::cout, Anorm, "A(col norm)");

            // Power Expansion: compute Anorm^e.
            RealMatrixT Apower(Anorm);
            for (grb::IndexType k = 0; k < (e - 1); ++k)
            {
                grb::mxm(Apower,
                         Apower, grb::NoAccumulate(),
                         grb::ArithmeticSemiring<RealT>(),
                         Apower, Anorm, grb::REPLACE);
            }

            //grb::print_matrix(std::cout, Apower, "Apower = Anorm^e");

            // Inflate. element wise raise to power r, normalize columns
            RealMatrixT Ainfl(Apower);
            for (grb::IndexType k = 0; k < (r - 1); ++k)
            {
                grb::eWiseMult(Ainfl,
                               Ainfl,
                               grb::NoAccumulate(),
                               grb::Times<RealT>(),
                               Ainfl, Apower, grb::REPLACE);
            }
            //grb::print_matrix(std::cout, Ainfl, "Ainfl = Apower .^ r");
            grb::normalize_cols(Ainfl);
            //grb::print_matrix(std::cout, Ainfl, "*** new Anorm ***");

            // Compute mean squared error
            RealMatrixT E(rows, cols);
            grb::eWiseAdd(E,
                          grb::NoMask(), grb::NoAccumulate(),
                          grb::Minus<RealT>(),
                          Anorm, Ainfl);
            grb::eWiseMult(E,
                           grb::NoMask(), grb::NoAccumulate(),
                           grb::Times<RealT>(),
                           E, E);
            RealT mean_squared_error(0.);
            grb::reduce(mean_squared_error,
                        grb::NoAccumulate(),
                        grb::PlusMonoid<RealT>(),
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
