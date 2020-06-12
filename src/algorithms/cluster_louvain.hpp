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
#include <random>
#include <graphblas/graphblas.hpp>
#include <graphblas/matrix_utils.hpp>

//****************************************************************************
namespace algorithms
{
    //************************************************************************
    /**
     * @brief Assign a zero-based cluster ID to each vertex based on the
     *        cluster matrix that was output by the louvain clustering algorithm.
     *
     * @param[in]  cluster_matrix  Matrix output from louvain clustering
     *                             algorithms.  Each row corresponds to a
     *                             vertex and each column corresponds to a cluster.
     *
     * @return An array with each vertex's cluster assignment (the row index for
     *         the maximum value).  MAX_UINT is returned if no cluster was
     *         assigned.
     */
    template <typename MatrixT>
    grb::Vector<grb::IndexType> get_louvain_cluster_assignments(
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
        grb::mxv(clusters,
                 grb::NoMask(), grb::NoAccumulate(),
                 grb::MaxSecondSemiring<grb::IndexType>(),
                 cluster_matrix, index_of_vec);

        return clusters;
    }

    //************************************************************************
    /**
     * @brief Compute the clusters in the given graph using louvain clustering.
     *
     * @param[in]  graph    The graph to compute the clusters on.  Can be a
     *                         weighted graph.
     * @param[in]  random_seed The seed for the RNG for tie breaking
     * @param[in]  max_iters   The maximum number of iterations to run if
     *                         convergence doesn't occur first (uncommon)
     *
     * @return A matrix whose columns correspond to the vertices, and vertices
     *         with the same (max) value in a given row belong to the
     *         same cluster.
     */
    template<typename MatrixT, typename RealT=double>
    grb::Matrix<bool> louvain_cluster(
        MatrixT const &graph,
        double         random_seed = 11.0, // arbitrary
        unsigned int   max_iters = std::numeric_limits<unsigned int>::max())
    {
        using T = typename MatrixT::ScalarType;
        unsigned int iters = 0;

        grb::IndexType rows(graph.nrows());
        grb::IndexType cols(graph.ncols());
        if (rows != cols)
        {
            throw grb::DimensionException();
        }

        //grb::print_matrix(std::cout, graph, "*** graph ***");

        // precompute A + A'
        grb::Matrix<T> ApAT(graph);
        grb::transpose(ApAT, grb::NoMask(),
                       grb::Plus<T>(), graph);

        //grb::print_matrix(std::cout, ApAT, "*** A+A' ***");

        // k = A * vec(1)  (arithmetric row reduce of adj. matrix)
        grb::Vector<RealT> k(rows);
        grb::reduce(k, grb::NoMask(), grb::NoAccumulate(),
                    grb::Plus<RealT>(), graph);

        // m = 0.5*k'*vec(1) (reduce k to scalar)
        RealT m(0);
        grb::reduce(m, grb::NoAccumulate(),
                    grb::PlusMonoid<RealT>(), k);
        m = m/2.0;

        // Initialize S to identity?
        auto S(grb::scaled_identity<grb::Matrix<bool>>(rows));

        grb::Vector<bool> S_row(rows);

        // create a dense mask vector of all trues
        grb::Vector<bool> mask(rows);
        grb::assign(mask, grb::NoMask(), grb::NoAccumulate(),
                    true, grb::AllIndices());

        //SetRandom<RealT> set_random(random_seed);
        std::default_random_engine             generator;
        std::uniform_real_distribution<double> distribution;
        generator.seed(random_seed);

        bool vertices_changed(true);

        // repeat while modularity is increasing
        do
        {
            //std::cout << "====== Start of new outer iteration1 ======" << std::endl;

            vertices_changed = false;

            for (grb::IndexType i = 0; i < rows; ++i)
            {
                // only perform the iteration if the ith vertex is not isolated
                if (k.hasElement(i))
                {
                    //std::cout << "====== Start of vertex " << i << std::endl;

                    // s_i = S[i,:]
                    grb::extract(S_row, grb::NoMask(),
                                 grb::NoAccumulate(),
                                 grb::transpose(S),
                                 grb::AllIndices(), i);

                    // S := (I - e_i*e_i')S means clear i-th row of S (using a mask)
                    grb::Matrix<bool> Mask(rows, rows);
                    grb::assign(Mask, grb::NoMask(),
                                grb::NoAccumulate(),
                                mask, i, grb::AllIndices());

                    grb::apply(S, grb::complement(Mask),
                               grb::NoAccumulate(),
                               grb::Identity<bool>(),
                               S, grb::REPLACE);

                    // v' = e_i' * (A + A') == extract row i of (A + A')
                    grb::Vector<RealT> v(rows);
                    grb::extract(v, grb::NoMask(),
                                 grb::NoAccumulate(),
                                 ApAT,
                                 grb::AllIndices(), i);

                    // v' += (-k_i/m)*k'
                    grb::apply(
                        v, grb::NoMask(),
                        grb::Plus<RealT>(),
                        std::bind(grb::Times<RealT>(),
                                  std::placeholders::_1,
                                  static_cast<RealT>(-k.extractElement(i)/m)),
                        k);


                    // q' = v' * S = [e_i' * (A + A') + (-k_i/m)*k'] * S
                    grb::Vector<RealT> q(rows);
                    grb::vxm(q, grb::NoMask(),
                             grb::NoAccumulate(),
                             grb::ArithmeticSemiring<RealT>(),
                             v, S);
                    //grb::print_vector(
                    //    std::cout, q, "q' = [e_i' * (A + A') + (-k_i/m)*k'] * S");

                    // kappa = max(q)
                    RealT kappa(0);
                    grb::reduce(kappa, grb::NoAccumulate(),
                                grb::MaxMonoid<RealT>(), q);
                    //std::cout << "kappa = " << kappa << std::endl;

                    // t = (q == kappa)
                    grb::Vector<bool> t(rows);
                    grb::apply(t, grb::NoMask(),
                               grb::NoAccumulate(),
                               std::bind(grb::Equal<RealT>(),
                                         std::placeholders::_1,
                                         kappa),
                               q);

                    // remove all stored falses (TODO: Replace with GxB_select?)
                    grb::apply(t, t, grb::NoAccumulate(),
                               grb::Identity<T>(), t, grb::REPLACE);
                    //grb::print_vector(
                    //    std::cout, t, "t = (q == kappa), with zeros removed");

                    // break ties if necessary
                    while (t.nvals() != 1)
                    {
                        //std::cout << "breaking ties: t.nvals() = "
                        //          << t.nvals() << std::endl;

                        // Assign a random number to each possible cluster
                        grb::Vector<RealT> p(rows);
                        grb::apply(p, grb::NoMask(),
                                   grb::NoAccumulate(),
                                   [&](bool tval)      //set_random,
                                   {
                                       return static_cast<RealT>(
                                           tval*distribution(generator) + 0.0001);
                                   },
                                   t);
                        // max_p = max(p)
                        RealT max_p(0);
                        grb::reduce(max_p, grb::NoAccumulate(),
                                    grb::MaxMonoid<RealT>(), p);

                        // t = (q == kappa)
                        grb::apply(t, grb::NoMask(),
                                   grb::NoAccumulate(),
                                   std::bind(grb::Equal<RealT>(),
                                             std::placeholders::_1,
                                             max_p),
                                   p);

                        // remove all stored falses (TODO: Replace with GxB_select?)
                        grb::apply(t, t, grb::NoAccumulate(),
                                   grb::Identity<T>(), t,
                                   grb::REPLACE);

                        //grb::print_vector(
                        //    std::cout, t, "new t after breaking ties");
                    }

                    // Replace row i of S with t
                    grb::assign(S, grb::NoMask(),
                                grb::NoAccumulate(),
                                t, i, grb::AllIndices());

                    // Compare new community w/ previous community to
                    // see if it has changed
                    if (t != S_row) // vertex changed communities
                    {
                        vertices_changed = true;
                        //grb::print_matrix(
                        //    std::cout, S, "*** changed cluster assignments, S ***");
                    }
                    else
                    {
                        //std::cout<< "No change. "<< std::endl;
                    }
                }
                //std::cout << "====== End   of vertex " << i << std::endl;
            }

            //std::cout << "====== End   of     outer iteration ======" << std::endl;
            //auto cluster_assignments =
            //    algorithms::get_louvain_cluster_assignments(S);
            //print_vector(std::cout, cluster_assignments, "cluster assignments");

            ++iters;
        } while (vertices_changed && iters < max_iters);

        if (vertices_changed)
            throw iters;

        //grb::print_matrix(
        //    std::cout, S, "Final cluster assignments, S");
        return S;
    }

    //************************************************************************
    /**
     * @brief Compute the clusters in the given graph using louvain clustering
     *        and applying an mask to the clusters considered.
     *
     * @param[in]  graph    The graph to compute the clusters on.  Can be a
     *                         weighted graph.
     * @param[in]  random_seed The seed for the RNG for tie breaking
     * @param[in]  max_iters   The maximum number of iterations to run if
     *                         convergence doesn't occur first (uncommon)
     *
     * @return A matrix whose columns correspond to the vertices, and vertices
     *         with the same (max) value in a given row belong to the
     *         same cluster.
     */
    template<typename MatrixT, typename RealT=double>
    grb::Matrix<bool> louvain_cluster_masked(
        MatrixT const &graph,
        double         random_seed = 11.0, // arbitrary
        unsigned int   max_iters = std::numeric_limits<unsigned int>::max())
    {
        using T = typename MatrixT::ScalarType;
        unsigned int iters = 0;

        grb::IndexType rows(graph.nrows());
        grb::IndexType cols(graph.ncols());
        if (rows != cols)
        {
            throw grb::DimensionException();
        }

        //grb::print_matrix(std::cout, graph, "*** graph ***");

        // precompute A + A'
        grb::Matrix<T> ApAT(graph);
        grb::transpose(ApAT, grb::NoMask(),
                       grb::Plus<T>(), graph);

        //grb::print_matrix(std::cout, ApAT, "*** A+A' ***");

        // k = A * vec(1)  (arithmetric row reduce of adj. matrix)
        grb::Vector<RealT> k(rows);
        grb::reduce(k, grb::NoMask(), grb::NoAccumulate(),
                    grb::Plus<RealT>(), graph);

        // m = 0.5*k'*vec(1) (reduce k to scalar)
        RealT m(0);
        grb::reduce(m, grb::NoAccumulate(),
                    grb::PlusMonoid<RealT>(), k);
        m = m/2.0;

        // Initialize S to identity?
        auto S(grb::scaled_identity<grb::Matrix<bool>>(rows));

        grb::Vector<bool> S_row(rows);

        // create a dense mask vector of all trues
        grb::Vector<bool> mask(rows);
        grb::assign(mask, grb::NoMask(), grb::NoAccumulate(),
                    true, grb::AllIndices());

        //SetRandom<RealT> set_random(random_seed);
        std::default_random_engine             generator;
        std::uniform_real_distribution<double> distribution;
        generator.seed(random_seed);

        bool vertices_changed(true);

        // allocate once
        grb::Vector<RealT> v(rows);
        grb::Vector<bool> q_mask(rows);
        grb::Vector<RealT> q(rows);
        grb::Vector<bool> t(rows);

        // repeat while modularity is increasing
        do
        {
            //std::cout << "====== Start of new outer iteration ======" << std::endl;

            vertices_changed = false;

            for (grb::IndexType i = 0; i < rows; ++i)
            {
                // only perform the iteration if the ith vertex is not isolated
                if (k.hasElement(i))
                {
                    //std::cout << "====== Start of vertex " << i << std::endl;

                    // s_i = S[i,:]
                    grb::extract(S_row, grb::NoMask(),
                                 grb::NoAccumulate(),
                                 grb::transpose(S),
                                 grb::AllIndices(), i);

                    // S := (I - e_i*e_i')S means clear i-th row of S (using a mask)
                    grb::Matrix<bool> Mask(rows, rows);
                    grb::assign(Mask, grb::NoMask(),
                                grb::NoAccumulate(),
                                mask, i, grb::AllIndices());

                    grb::apply(S, grb::complement(Mask),
                               grb::NoAccumulate(),
                               grb::Identity<bool>(),
                               S, grb::REPLACE);

                    // v' = e_i' * (A + A') == extract row i of (A + A')
                    grb::extract(v, grb::NoMask(),
                                 grb::NoAccumulate(),
                                 grb::transpose(ApAT),
                                 grb::AllIndices(), i);

                    // v' += (-k_i/m)*k'
                    grb::apply(
                        v, grb::NoMask(),
                        grb::Plus<RealT>(),
                        std::bind(grb::Times<RealT>(),
                                  std::placeholders::_1,
                                  static_cast<RealT>(-k.extractElement(i)/m)),
                        k);

                    // ============= MASKING =============
                    // find mask of communities that are neighbors of vertex i

                    // Extract the neighbors of vertex i as bools
                    /// @todo figure out if extract can be done once (note different domain)
                    grb::extract(q_mask, grb::NoMask(),
                                 grb::NoAccumulate(),
                                 grb::transpose(ApAT),  // transpose optional
                                 grb::AllIndices(), i);

                    // Compute q_mask' = (row i of A+A') * S
                    grb::vxm(q_mask, grb::NoMask(), grb::NoAccumulate(),
                             grb::LogicalSemiring<bool>(), q_mask, S);

                    //grb::print_matrix(std::cout, Sprime, "filtered community matrix");
                    grb::vxm(q, q_mask, //grb::NoMask(),
                             grb::NoAccumulate(),
                             grb::ArithmeticSemiring<RealT>(),
                             v, S, grb::REPLACE);
                    //v, Sprime);
                    // ============= MASKING =============

                    //grb::print_vector(
                    //    std::cout, q, "q'<mask> = [e_i' * (A + A') + (-k_i/m)*k'] * S");

                    // kappa = max(q)
                    RealT kappa(0);
                    grb::reduce(kappa, grb::NoAccumulate(),
                                grb::MaxMonoid<RealT>(), q);
                    //std::cout << "kappa = " << kappa << std::endl;

                    // t = (q == kappa)
                    grb::apply(t, grb::NoMask(),
                               grb::NoAccumulate(),
                               std::bind(grb::Equal<RealT>(),
                                         std::placeholders::_1,
                                         kappa),
                               q);

                    // remove all stored falses (TODO: Replace with GxB_select?)
                    grb::apply(t, t, grb::NoAccumulate(),
                               grb::Identity<T>(), t, grb::REPLACE);
                    //grb::print_vector(
                    //    std::cout, t, "t = (q == kappa), with zeros removed");

                    // break ties if necessary
                    while (t.nvals() != 1)
                    {
                        //std::cout << "breaking ties: t.nvals() = "
                        //          << t.nvals() << std::endl;

                        // Assign a random number to each possible cluster
                        grb::Vector<RealT> p(rows);
                        grb::apply(p, grb::NoMask(),
                                   grb::NoAccumulate(),
                                   [&](bool tval)      //set_random,
                                   {
                                       return static_cast<RealT>(
                                           tval*distribution(generator) + 0.0001);
                                   },
                                   t);
                        // max_p = max(p)
                        RealT max_p(0);
                        grb::reduce(max_p, grb::NoAccumulate(),
                                    grb::MaxMonoid<RealT>(), p);

                        // t = (q == kappa)
                        grb::apply(t, grb::NoMask(),
                                   grb::NoAccumulate(),
                                   std::bind(grb::Equal<RealT>(),
                                             std::placeholders::_1,
                                             max_p),
                                   p);
                        // remove all stored falses (TODO: Replace with GxB_select?)
                        grb::apply(t, t, grb::NoAccumulate(),
                                   grb::Identity<T>(), t,
                                   grb::REPLACE);

                        //grb::print_vector(
                        //    std::cout, t, "new t after breaking ties");
                    }

                    // Replace row i of S with t
                    grb::assign(S, grb::NoMask(),
                                grb::NoAccumulate(),
                                t, i, grb::AllIndices());

                    // Compare new community w/ previous community to
                    // see if it has changed
                    if (t != S_row) // vertex changed communities
                    {
                        vertices_changed = true;
                        //grb::print_matrix(
                        //    std::cout, S, "*** changed cluster assignments, S ***");
                    }
                    else
                    {
                        //std::cout<< "No change. "<< std::endl;
                    }
                }
                //std::cout << "====== End   of vertex " << i << std::endl;
            }

            //std::cout << "====== End   of     outer iteration ======" << std::endl;
            //auto cluster_assignments =
            //    algorithms::get_louvain_cluster_assignments(S);
            //print_vector(std::cout, cluster_assignments, "cluster assignments");

            ++iters;
        } while (vertices_changed && iters < max_iters);

        if (vertices_changed)
            throw iters;

        //grb::print_matrix(
        //    std::cout, S, "Final cluster assignments, S");
        return S;
    }

} // algorithms
