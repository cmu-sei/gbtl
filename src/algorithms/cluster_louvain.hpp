/*
 * GraphBLAS Template Library, Version 2.0
 *
 * Copyright 2019 Carnegie Mellon University, Battelle Memorial Institute, and
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

#ifndef ALGORITHMS_CLUSTER_LOUVAIN_HPP
#define ALGORITHMS_CLUSTER_LOUVAIN_HPP

#include <vector>
#include <random>
//#include <math.h>
#include <graphblas/graphblas.hpp>
#include <graphblas/matrix_utils.hpp>

//****************************************************************************
namespace
{
    //************************************************************************
    // Return a random value that is scaled by the value passed in
    /// @warning this operator has state
    template <typename T=float>
    class SetRandom
    {
    public:
        SetRandom(double seed = 0.) { m_generator.seed(seed); }

        inline T operator()(T val)
        {
            return static_cast<T>(val*m_distribution(m_generator) + 0.0001);
        }

    private:
        std::default_random_engine             m_generator;
        std::uniform_real_distribution<double> m_distribution;
    };
}

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
    GraphBLAS::Vector<GraphBLAS::IndexType> get_louvain_cluster_assignments(
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
        GraphBLAS::mxv(clusters,
                       GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                       GraphBLAS::MaxSecondSemiring<GraphBLAS::IndexType>(),
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
    GraphBLAS::Matrix<bool> louvain_cluster(
        MatrixT const &graph,
        double         random_seed = 11.0, // arbitrary
        unsigned int   max_iters = std::numeric_limits<unsigned int>::max())
    {
        using T = typename MatrixT::ScalarType;
        unsigned int iters = 0;

        GraphBLAS::IndexType rows(graph.nrows());
        GraphBLAS::IndexType cols(graph.ncols());
        if (rows != cols)
        {
            throw GraphBLAS::DimensionException();
        }

        //GraphBLAS::print_matrix(std::cout, graph, "*** graph ***");

        // precompute A + A'
        GraphBLAS::Matrix<T> ApAT(graph);
        GraphBLAS::transpose(ApAT, GraphBLAS::NoMask(),
                             GraphBLAS::Plus<T>(), graph);

        //GraphBLAS::print_matrix(std::cout, ApAT, "*** A+A' ***");

        // k = A * vec(1)  (arithmetric row reduce of adj. matrix)
        GraphBLAS::Vector<RealT> k(rows);
        GraphBLAS::reduce(k, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                          GraphBLAS::Plus<RealT>(), graph);

        // m = 0.5*k'*vec(1) (reduce k to scalar)
        RealT m(0);
        GraphBLAS::reduce(m, GraphBLAS::NoAccumulate(),
                          GraphBLAS::PlusMonoid<RealT>(), k);
        m = m/2.0;

        // Initialize S to identity?
        auto S(GraphBLAS::scaled_identity<GraphBLAS::Matrix<bool>>(rows));

        GraphBLAS::Vector<bool> S_row(rows);

        // create a dense mask vector of all trues
        GraphBLAS::Vector<bool> mask(rows);
        GraphBLAS::assign(mask, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                          true, GraphBLAS::AllIndices());

        SetRandom<RealT> set_random(random_seed);
        bool vertices_changed(true);

        // repeat while modularity is increasing
        do
        {
            //std::cout << "====== Start of new outer iteration1 ======" << std::endl;

            vertices_changed = false;

            for (GraphBLAS::IndexType i = 0; i < rows; ++i)
            {
                // only perform the iteration if the ith vertex is not isolated
                if (k.hasElement(i))
                {
                    //std::cout << "====== Start of vertex " << i << std::endl;

                    // s_i = S[i,:]
                    GraphBLAS::extract(S_row, GraphBLAS::NoMask(),
                                       GraphBLAS::NoAccumulate(),
                                       GraphBLAS::transpose(S),
                                       GraphBLAS::AllIndices(), i);

                    // S := (I - e_i*e_i')S means clear i-th row of S (using a mask)
                    GraphBLAS::Matrix<bool> Mask(rows, rows);
                    GraphBLAS::assign(Mask, GraphBLAS::NoMask(),
                                      GraphBLAS::NoAccumulate(),
                                      mask, i, GraphBLAS::AllIndices());

                    GraphBLAS::apply(S, GraphBLAS::complement(Mask),
                                     GraphBLAS::NoAccumulate(),
                                     GraphBLAS::Identity<bool>(),
                                     S, GraphBLAS::REPLACE);

                    // v' = e_i' * (A + A') == extract row i of (A + A')
                    GraphBLAS::Vector<RealT> v(rows);
                    GraphBLAS::extract(v, GraphBLAS::NoMask(),
                                       GraphBLAS::NoAccumulate(),
                                       ApAT,
                                       GraphBLAS::AllIndices(), i);

                    // v' += (-k_i/m)*k'
                    GraphBLAS::apply(
                        v, GraphBLAS::NoMask(),
                        GraphBLAS::Plus<RealT>(),
                        std::bind(GraphBLAS::Times<RealT>(),
                                  std::placeholders::_1,
                                  static_cast<RealT>(-k.extractElement(i)/m)),
                        k);


                    // q' = v' * S = [e_i' * (A + A') + (-k_i/m)*k'] * S
                    GraphBLAS::Vector<RealT> q(rows);
                    GraphBLAS::vxm(q, GraphBLAS::NoMask(),
                                   GraphBLAS::NoAccumulate(),
                                   GraphBLAS::ArithmeticSemiring<RealT>(),
                                   v, S);
                    //GraphBLAS::print_vector(
                    //    std::cout, q, "q' = [e_i' * (A + A') + (-k_i/m)*k'] * S");

                    // kappa = max(q)
                    RealT kappa(0);
                    GraphBLAS::reduce(kappa, GraphBLAS::NoAccumulate(),
                                      GraphBLAS::MaxMonoid<RealT>(), q);
                    //std::cout << "kappa = " << kappa << std::endl;

                    // t = (q == kappa)
                    GraphBLAS::Vector<bool> t(rows);
                    GraphBLAS::apply(t, GraphBLAS::NoMask(),
                                     GraphBLAS::NoAccumulate(),
                                     std::bind(GraphBLAS::Equal<RealT>(),
                                               std::placeholders::_1,
                                               kappa),
                                     q);

                    // remove all stored falses (TODO: Replace with GxB_select?)
                    GraphBLAS::apply(t, t, GraphBLAS::NoAccumulate(),
                                     GraphBLAS::Identity<T>(), t, GraphBLAS::REPLACE);
                    //GraphBLAS::print_vector(
                    //    std::cout, t, "t = (q == kappa), with zeros removed");

                    // break ties if necessary
                    while (t.nvals() != 1)
                    {
                        //std::cout << "breaking ties: t.nvals() = "
                        //          << t.nvals() << std::endl;

                        // Assign a random number to each possible cluster
                        GraphBLAS::Vector<RealT> p(rows);
                        GraphBLAS::apply(p, GraphBLAS::NoMask(),
                                         GraphBLAS::NoAccumulate(),
                                         set_random, t);
                        // max_p = max(p)
                        RealT max_p(0);
                        GraphBLAS::reduce(max_p, GraphBLAS::NoAccumulate(),
                                          GraphBLAS::MaxMonoid<RealT>(), p);

                        // t = (q == kappa)
                        GraphBLAS::apply(t, GraphBLAS::NoMask(),
                                         GraphBLAS::NoAccumulate(),
                                         std::bind(GraphBLAS::Equal<RealT>(),
                                                   std::placeholders::_1,
                                                   max_p),
                                         p);

                        // remove all stored falses (TODO: Replace with GxB_select?)
                        GraphBLAS::apply(t, t, GraphBLAS::NoAccumulate(),
                                         GraphBLAS::Identity<T>(), t,
                                         GraphBLAS::REPLACE);

                        //GraphBLAS::print_vector(
                        //    std::cout, t, "new t after breaking ties");
                    }

                    // Replace row i of S with t
                    GraphBLAS::assign(S, GraphBLAS::NoMask(),
                                      GraphBLAS::NoAccumulate(),
                                      t, i, GraphBLAS::AllIndices());

                    // Compare new community w/ previous community to
                    // see if it has changed
                    if (t != S_row) // vertex changed communities
                    {
                        vertices_changed = true;
                        //GraphBLAS::print_matrix(
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

        //GraphBLAS::print_matrix(
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
    GraphBLAS::Matrix<bool> louvain_cluster_masked(
        MatrixT const &graph,
        double         random_seed = 11.0, // arbitrary
        unsigned int   max_iters = std::numeric_limits<unsigned int>::max())
    {
        using T = typename MatrixT::ScalarType;
        unsigned int iters = 0;

        GraphBLAS::IndexType rows(graph.nrows());
        GraphBLAS::IndexType cols(graph.ncols());
        if (rows != cols)
        {
            throw GraphBLAS::DimensionException();
        }

        //GraphBLAS::print_matrix(std::cout, graph, "*** graph ***");

        // precompute A + A'
        GraphBLAS::Matrix<T> ApAT(graph);
        GraphBLAS::transpose(ApAT, GraphBLAS::NoMask(),
                             GraphBLAS::Plus<T>(), graph);

        //GraphBLAS::print_matrix(std::cout, ApAT, "*** A+A' ***");

        // k = A * vec(1)  (arithmetric row reduce of adj. matrix)
        GraphBLAS::Vector<RealT> k(rows);
        GraphBLAS::reduce(k, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                          GraphBLAS::Plus<RealT>(), graph);

        // m = 0.5*k'*vec(1) (reduce k to scalar)
        RealT m(0);
        GraphBLAS::reduce(m, GraphBLAS::NoAccumulate(),
                          GraphBLAS::PlusMonoid<RealT>(), k);
        m = m/2.0;

        // Initialize S to identity?
        auto S(GraphBLAS::scaled_identity<GraphBLAS::Matrix<bool>>(rows));

        GraphBLAS::Vector<bool> S_row(rows);

        // create a dense mask vector of all trues
        GraphBLAS::Vector<bool> mask(rows);
        GraphBLAS::assign(mask, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                          true, GraphBLAS::AllIndices());

        SetRandom<RealT> set_random(random_seed);
        bool vertices_changed(true);

        // allocate once
        GraphBLAS::Vector<RealT> v(rows);
        GraphBLAS::Vector<bool> q_mask(rows);
        GraphBLAS::Vector<RealT> q(rows);
        GraphBLAS::Vector<bool> t(rows);

        // repeat while modularity is increasing
        do
        {
            //std::cout << "====== Start of new outer iteration ======" << std::endl;

            vertices_changed = false;

            for (GraphBLAS::IndexType i = 0; i < rows; ++i)
            {
                // only perform the iteration if the ith vertex is not isolated
                if (k.hasElement(i))
                {
                    //std::cout << "====== Start of vertex " << i << std::endl;

                    // s_i = S[i,:]
                    GraphBLAS::extract(S_row, GraphBLAS::NoMask(),
                                       GraphBLAS::NoAccumulate(),
                                       GraphBLAS::transpose(S),
                                       GraphBLAS::AllIndices(), i);

                    // S := (I - e_i*e_i')S means clear i-th row of S (using a mask)
                    GraphBLAS::Matrix<bool> Mask(rows, rows);
                    GraphBLAS::assign(Mask, GraphBLAS::NoMask(),
                                      GraphBLAS::NoAccumulate(),
                                      mask, i, GraphBLAS::AllIndices());

                    GraphBLAS::apply(S, GraphBLAS::complement(Mask),
                                     GraphBLAS::NoAccumulate(),
                                     GraphBLAS::Identity<bool>(),
                                     S, GraphBLAS::REPLACE);

                    // v' = e_i' * (A + A') == extract row i of (A + A')
                    GraphBLAS::extract(v, GraphBLAS::NoMask(),
                                       GraphBLAS::NoAccumulate(),
                                       GraphBLAS::transpose(ApAT),
                                       GraphBLAS::AllIndices(), i);

                    // v' += (-k_i/m)*k'
                    GraphBLAS::apply(
                        v, GraphBLAS::NoMask(),
                        GraphBLAS::Plus<RealT>(),
                        std::bind(GraphBLAS::Times<RealT>(),
                                  std::placeholders::_1,
                                  static_cast<RealT>(-k.extractElement(i)/m)),
                        k);

                    // ============= MASKING =============
                    // find mask of communities that are neighbors of vertex i

                    // Extract the neighbors of vertex i as bools
                    /// @todo figure out if extract can be done once (note different domain)
                    GraphBLAS::extract(q_mask, GraphBLAS::NoMask(),
                                       GraphBLAS::NoAccumulate(),
                                       GraphBLAS::transpose(ApAT),  // transpose optional
                                       GraphBLAS::AllIndices(), i);

                    // Compute q_mask' = (row i of A+A') * S
                    GraphBLAS::vxm(q_mask, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                   GraphBLAS::LogicalSemiring<bool>(), q_mask, S);

                    //GraphBLAS::print_matrix(std::cout, Sprime, "filtered community matrix");
                    GraphBLAS::vxm(q, q_mask, //GraphBLAS::NoMask(),
                                   GraphBLAS::NoAccumulate(),
                                   GraphBLAS::ArithmeticSemiring<RealT>(),
                                   v, S, GraphBLAS::REPLACE);
                                   //v, Sprime);
                    // ============= MASKING =============

                    //GraphBLAS::print_vector(
                    //    std::cout, q, "q'<mask> = [e_i' * (A + A') + (-k_i/m)*k'] * S");

                    // kappa = max(q)
                    RealT kappa(0);
                    GraphBLAS::reduce(kappa, GraphBLAS::NoAccumulate(),
                                      GraphBLAS::MaxMonoid<RealT>(), q);
                    //std::cout << "kappa = " << kappa << std::endl;

                    // t = (q == kappa)
                    GraphBLAS::apply(t, GraphBLAS::NoMask(),
                                     GraphBLAS::NoAccumulate(),
                                     std::bind(GraphBLAS::Equal<RealT>(),
                                               std::placeholders::_1,
                                               kappa),
                                     q);

                    // remove all stored falses (TODO: Replace with GxB_select?)
                    GraphBLAS::apply(t, t, GraphBLAS::NoAccumulate(),
                                     GraphBLAS::Identity<T>(), t, GraphBLAS::REPLACE);
                    //GraphBLAS::print_vector(
                    //    std::cout, t, "t = (q == kappa), with zeros removed");

                    // break ties if necessary
                    while (t.nvals() != 1)
                    {
                        //std::cout << "breaking ties: t.nvals() = "
                        //          << t.nvals() << std::endl;

                        // Assign a random number to each possible cluster
                        GraphBLAS::Vector<RealT> p(rows);
                        GraphBLAS::apply(p, GraphBLAS::NoMask(),
                                         GraphBLAS::NoAccumulate(),
                                         set_random, t);
                        // max_p = max(p)
                        RealT max_p(0);
                        GraphBLAS::reduce(max_p, GraphBLAS::NoAccumulate(),
                                          GraphBLAS::MaxMonoid<RealT>(), p);

                        // t = (q == kappa)
                        GraphBLAS::apply(t, GraphBLAS::NoMask(),
                                         GraphBLAS::NoAccumulate(),
                                         std::bind(GraphBLAS::Equal<RealT>(),
                                                   std::placeholders::_1,
                                                   max_p),
                                         p);
                        // remove all stored falses (TODO: Replace with GxB_select?)
                        GraphBLAS::apply(t, t, GraphBLAS::NoAccumulate(),
                                         GraphBLAS::Identity<T>(), t,
                                         GraphBLAS::REPLACE);

                        //GraphBLAS::print_vector(
                        //    std::cout, t, "new t after breaking ties");
                    }

                    // Replace row i of S with t
                    GraphBLAS::assign(S, GraphBLAS::NoMask(),
                                      GraphBLAS::NoAccumulate(),
                                      t, i, GraphBLAS::AllIndices());

                    // Compare new community w/ previous community to
                    // see if it has changed
                    if (t != S_row) // vertex changed communities
                    {
                        vertices_changed = true;
                        //GraphBLAS::print_matrix(
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

        //GraphBLAS::print_matrix(
        //    std::cout, S, "Final cluster assignments, S");
        return S;
    }

} // algorithms

#endif // ALGORITHMS_CLUSTER_LOUVAIN_HPP
