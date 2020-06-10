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

namespace GrB = GraphBLAS;

namespace algorithms
{

    //************************************************************************
    /**
     * @brief Perform a single "level" breadth first search (BFS) traversal
     *        on the given graph.
     *
     * @param[out] v    The level (distance in unweighted graphs) from
     *                  the source (root) of the BFS.
     * @param[in]  A    N x N adjacency matrix of the graph on which to
     *                  perform a BFS. (NOT the transpose).  The value
     *                  1 should indicate an edge.
     * @param[in]  src  Index of the root vertex to use in the
     *                  calculation.
     */
    template <typename LevelsVectorT, typename MatrixT>
    void bfs_level_appendixB1(LevelsVectorT   &v,
                              MatrixT const   &A,
                              GrB::IndexType   src)
    {

        GrB::IndexType const n(A.nrows());

        /// @todo Assert dimensions and throw DimensionException

        GrB::Vector<bool> q(n);      // wavefront
        q.setElement(src, true);

        // BFS traversal and label the vertices
        GrB::IndexType d(0);

        do {
            ++d;
            GrB::assign(v, q, GrB::NoAccumulate(),
                        d, GrB::AllIndices());
            GrB::vxm(q, GrB::complement(v), GrB::NoAccumulate(),
                     GrB::LogicalSemiring<bool>(), q, A, GrB::REPLACE);

        } while (q.nvals() > 0);
    }


    //************************************************************************
    /**
     * @brief Perform a single "level" breadth first search (BFS) traversal
     *        on the given graph.
     *
     * @param[out] v    The level (distance in unweighted graphs) from
     *                  the source (root) of the BFS.
     * @param[in]  A    N x N adjacency matrix of the graph on which to
     *                  perform a BFS. (NOT the transpose).  The value
     *                  1 should indicate an edge.
     * @param[in]  src  Index of the root vertex to use in the
     *                  calculation.
     */
    template <typename LevelsVectorT, typename MatrixT>
    void bfs_level_appendixB2(LevelsVectorT  &v,
                              MatrixT const  &A,
                              GrB::IndexType  src)
    {
        GrB::IndexType const n(A.nrows());

        /// @todo Assert dimensions and throw DimensionException

        GrB::Vector<bool> q(n);  //wavefront
        q.setElement(src, true);

        // BFS traversal and label the vertices
        GrB::IndexType level(0);
        do {
            ++level;
            GrB::apply(v, GrB::NoMask(), GrB::Plus<GrB::IndexType>(),
                       GrB::Times<GrB::IndexType>(), q, level, GrB::REPLACE);
            GrB::vxm(q, GrB::complement(v), GrB::NoAccumulate(),
                     GrB::LogicalSemiring<bool>(), q, A,
                     GrB::REPLACE);

        } while (q.nvals() > 0);
    }


    //************************************************************************
    /**
     * @brief Perform a single "parent" breadth first search (BFS) traversal
     *        on the given graph.
     *
     * @param[out] parents   A valid parent ID for each visited node in the
     *                       traversal from the source (root) of the BFS.
     *                       An N vector of GraphBLAS::IndexType's.
     * @param[in]  A         N x N binary adjacency matrix of the graph on
     *                       which to perform a BFS. (NOT the transpose).
     *                       A stored value of 1/true indicates an edge.
     * @param[in]  src       Index of the root vertex to use in the
     *                       calculation.
     */
    template <typename MatrixT,
              typename ParentsVectorT>
    void bfs_parent_appendixB3(ParentsVectorT &parents,
                               MatrixT const  &A,
                               GrB::IndexType  src)
    {
        GrB::IndexType const N(A.nrows());

        // create index ramp for index_of() functionality
        GrB::Vector<GrB::IndexType> index_ramp(N);
        for (GrB::IndexType i = 0; i < N; ++i)
            index_ramp.setElement(i, i);

        parents.clear();
        parents.setElement(src, src);

        GrB::Vector<GrB::IndexType> q(N);
        q.setElement(src, 1UL);

        // BFS traversal and label the vertices.
        while (q.nvals() > 0) {
            GrB::eWiseMult(q, GrB::NoMask(), GrB::NoAccumulate(),
                           GrB::First<GrB::IndexType>(), index_ramp, q);
            GrB::vxm(q,
                     GrB::complement(GrB::structure(parents)),
                     GrB::NoAccumulate(),
                     GrB::MinFirstSemiring<GrB::IndexType>(),
                     q, A,
                     GrB::REPLACE);
            GrB::apply(parents, GrB::NoMask(), GrB::Plus<GrB::IndexType>(),
                       GrB::Identity<GrB::IndexType>(), q);
        }
    }

    //************************************************************************
    /**
     * @brief Compute the betweenness centrality contribution of all
     *        vertices in the graph for a given source node and add it to
     *        an existing BC vector
     *
     * @param[in,out] delta  On output, updated BC for the specifed source
     * @param[in]     A      The graph to compute the betweenness centrality of.
     * @param[in]     src    The source node for the BC computation.
     *
     * @return The betweenness centrality of all vertices in the graph.
     */
    template<typename BCVectorT, typename MatrixT>
    void BC_appendixB4(BCVectorT      &delta,
                       MatrixT const  &A,
                       GrB::IndexType  src)
    {
        GrB::IndexType const n(A.nrows());
        GrB::Matrix<int32_t> sigma(n, n);

        GrB::Vector<int32_t> q(n);  // wavefront/frontier
        q.setElement(src, 1);

        GrB::Vector<int32_t> p(q);  // path

        GrB::vxm(q, GrB::complement(p), GrB::NoAccumulate(),
                 GrB::ArithmeticSemiring<int32_t>(), q, A, GrB::REPLACE);

        // BFS phase
        GrB::IndexType d(0);        // depth

        do {
            GrB::assign(sigma, GrB::NoMask(), GrB::NoAccumulate(),
                        q, d, GrB::AllIndices());
            GrB::eWiseAdd(p, GrB::NoMask(), GrB::NoAccumulate(),
                          GrB::Plus<int32_t>(), p, q);
            GrB::vxm(q, GrB::complement(p), GrB::NoAccumulate(),
                     GrB::ArithmeticSemiring<int32_t>(), q, A, GrB::REPLACE);


            ++d;
        } while (q.nvals() > 0);

        // BC computation phase
        GrB::Vector<float> t1(n), t2(n), t3(n), t4(n);




        for (int i = d-1; i > 0; --i) {
            GrB::assign(t1, GrB::NoMask(), GrB::NoAccumulate(),
                        1.f, GrB::AllIndices());
            GrB::eWiseAdd(t1, GrB::NoMask(), GrB::NoAccumulate(),
                          GrB::Plus<float>(), t1, delta);
            GrB::extract(t2, GrB::NoMask(), GrB::NoAccumulate(),
                         GrB::transpose(sigma), GrB::AllIndices(), i);
            GrB::eWiseMult(t2, GrB::NoMask(), GrB::NoAccumulate(),
                           GrB::Div<float>(), t1, t2);
            GrB::mxv(t3, GrB::NoMask(), GrB::NoAccumulate(),
                     GrB::ArithmeticSemiring<float>(), A, t2);
            GrB::extract(t4, GrB::NoMask(), GrB::NoAccumulate(),
                         GrB::transpose(sigma), GrB::AllIndices(), i-1);
            GrB::eWiseMult(t4, GrB::NoMask(), GrB::NoAccumulate(),
                           GrB::Times<float>(), t4, t3);
            GrB::eWiseAdd(delta, GrB::NoMask(), GrB::NoAccumulate(),
                          GrB::Plus<float>(), delta, t4);
        }
    }

    //************************************************************************
    /**
     * @brief Compute the vertex betweenness centrality of all vertices in the
     *        given graph from a batch of src vertices.
     *
     * @param[out] delta The BC update for a set of source nodes (not accum-ed)
     *                   Should be float.
     * @param[in]  A     The graph to compute the betweenness centrality of.
     * @param[in]  src   The set of source vertex indices from which to compute
     *                   BC contributions
     *
     */
    // vertex_betweenness_centrality_batch_alt() from bc.hpp
    template<typename BCVectorT, typename MatrixT>
    void BC_update_appendixB5(
        BCVectorT                         &delta,
        MatrixT const                     &A,
        std::vector<GrB::IndexType> const &src)  // aka GrB::IndexArrayType
    {

        GrB::IndexType nsver(src.size());
        GrB::IndexType n(A.nrows());
        /// @todo assert proper dimensions and nsver > 0

        // index and value arrays needed to build NumSP
        GrB::IndexArrayType GrB_ALL_nsver;     // fill with sequence
        GrB_ALL_nsver.reserve(nsver);
        for (GrB::IndexType idx = 0; idx < nsver; ++idx) {
            GrB_ALL_nsver.push_back(idx);
        }


        // NumSP holds # of shortest paths to each vertex from each src.
        // Initialized to source vertices
        // NumSP[s[i],i] = 1 where 0 <= i < nsver; implied zero elsewhere
        GrB::Matrix<int32_t> NumSP(n, nsver);
        NumSP.build(src, GrB_ALL_nsver, std::vector<int32_t>(nsver, 1));


        // The current frontier for all BFS's (from all roots)
        // Initialized to out neighbors of each source node in src
        GrB::Matrix<int32_t> Frontier(n, nsver);

        GrB::extract(Frontier, GrB::complement(NumSP), GrB::NoAccumulate(),
                     GrB::transpose(A), GrB::AllIndices(), src, GrB::REPLACE);
        //GrB::extract(Frontier, GrB::NoMask(), GrB::NoAccumulate(),
        //        GrB::transpose(A), GrB::AllIndices(), src);

        // std::vector manages allocation
        std::vector<GrB::Matrix<bool>*> Sigmas;

        int32_t d = 0;


        // ==================== BFS phase ====================
        while (Frontier.nvals() > 0) {
            Sigmas.push_back(new GrB::Matrix<bool>(n, nsver));
            GrB::apply(*(Sigmas[d]), GrB::NoMask(), GrB::NoAccumulate(),
                       GrB::Identity<int32_t, bool>(), Frontier);

            GrB::eWiseAdd(NumSP, GrB::NoMask(), GrB::NoAccumulate(),
                          GrB::Plus<int32_t>(), NumSP, Frontier);
            GrB::mxm(Frontier, GrB::complement(NumSP), GrB::NoAccumulate(),
                     GrB::ArithmeticSemiring<int32_t>(),
                     GrB::transpose(A), Frontier, GrB::REPLACE);
            ++d;

        }

        GrB::Matrix<float> NspInv(n, nsver);

        GrB::apply(NspInv, GrB::NoMask(), GrB::NoAccumulate(),
                   GrB::MultiplicativeInverse<float>(), NumSP);

        GrB::Matrix<float> BCu(n, nsver);
        GrB::assign(BCu, GrB::NoMask(), GrB::NoAccumulate(),
                    1.0f, GrB::AllIndices(), GrB::AllIndices());

        GrB::Matrix<float> W(n, nsver);

        // ================== backprop phase ==================
        for (int32_t i = d-1; i > 0; --i) {
            GrB::eWiseMult(W, *Sigmas[i], GrB::NoAccumulate(),
                           GrB::Times<float>(), BCu, NspInv, GrB::REPLACE);

            GrB::mxm(W, *Sigmas[i-1], GrB::NoAccumulate(),
                     GrB::ArithmeticSemiring<float>(), A, W, GrB::REPLACE);
            GrB::eWiseMult(BCu, GrB::NoMask(), GrB::Plus<float>(),
                           GrB::Times<float>(), W, NumSP);
        }

        // adjust result
        GrB::assign(delta, GrB::NoMask(), GrB::NoAccumulate(),
                    -1.f*nsver, GrB::AllIndices());
        GrB::reduce(delta, GrB::NoMask(), GrB::Plus<float>(),
                    GrB::Plus<float>(), BCu);

        // Release resources
        for (auto it = Sigmas.begin(); it != Sigmas.end(); ++it)
            delete *it;

    }

    /**
     * @brief Compute the Maximal Independent Set for a given graph.
     *
     * A variant of Luby's randomized algorithm (Luby 1985) is used:
     *  - A random number scaled by the inverse of the node's degree is assigned
     *    to each candidate node in the graph (algorithm starts with all nodes
     *    in the candidate set).
     *  - Each node's neighbor with the maximum random value is determined.
     *  - If the node has a larger random number than any of its neighbors,
     *    it is added to the independent set (IS).  This has the tendency to
     *    prefer selecting lower degree nodes.
     *  - Each selected node and all of their neighbors are removed from
     *    candidate set
     *  - The process is repeated
     *
     * @param[out] iset    N-vector of flags, 'true' indicates vertex
     *                     is selected.  Must be empty on call.
     * @param[in]  A       NxN adjacency matrix of graph to compute
     *                     the maximal independent set on. This
     *                     must be an unweighted and undirected graph
     *                     (meaning the matrix is symmetric).  The
     *                     structural zero needs to be '0' and edges
     *                     are indicated by '1' to support use of the
     *                     Arithmetic semiring.
     * @param[in]  seed    The seed for the random number generator
     *
     */
    template <typename MatrixT>
    void mis_appendixB6(GrB::Vector<bool> &iset, MatrixT const &A,
                        double seed = 0.)
    {
        std::default_random_engine             generator;
        std::uniform_real_distribution<double> distribution;
        generator.seed(seed);

        GrB::IndexType n(A.nrows());

        GrB::Vector<float> prob(n);
        GrB::Vector<float> neighbor_max(n);
        GrB::Vector<bool>  new_members(n);
        GrB::Vector<bool>  new_neighbors(n);
        GrB::Vector<bool>  candidates(n);

        GrB::Vector<double> degrees(n);
        GrB::reduce(degrees, GrB::NoMask(), GrB::NoAccumulate(), GrB::Plus<double>(), A);

        // Remove isolated vertices
        GrB::assign(candidates, degrees, GrB::NoAccumulate(), true, GrB::AllIndices());

        // Add all singletons to iset
        GrB::assign(iset, GrB::complement(degrees), GrB::NoAccumulate(),
                    true, GrB::AllIndices(), GrB::REPLACE);

        while (candidates.nvals() > 0) {
            // compute a random probability of each candidate scaled by inverse of degree.
            GrB::eWiseMult(prob, GrB::NoMask(), GrB::NoAccumulate(),
                           [&](bool candidate, float const &degree)
                           { return static_cast<float>(
                                   0.0001 + distribution(generator)/(1. + 2.*degree)); },
                           candidates, degrees);

            // find the max probability of all neighbors
            GrB::mxv(neighbor_max, candidates, GrB::NoAccumulate(),
                     GrB::MaxSecondSemiring<float>(), A, prob, GrB::REPLACE);

            // Select source node if its probability is > neighbor_max
            GrB::eWiseAdd(new_members, GrB::NoMask(), GrB::NoAccumulate(),
                          GrB::GreaterThan<double>(), prob, neighbor_max);
            GrB::apply(new_members, new_members, GrB::NoAccumulate(),
                       GrB::Identity<bool>(), new_members, GrB::REPLACE);

            // Add new members to independent set.
            GrB::eWiseAdd(iset, GrB::NoMask(), GrB::NoAccumulate(),
                          GrB::LogicalOr<bool>(), iset, new_members);

            // Remove new_members from set of candidates
            GrB::eWiseMult(candidates, GrB::complement(new_members), GrB::NoAccumulate(),
                           GrB::LogicalAnd<bool>(), candidates, candidates, GrB::REPLACE);


            if (candidates.nvals() == 0) break;

            // Neighbors of new members can also be removed
            GrB::mxv(new_neighbors, candidates, GrB::NoAccumulate(),
                     GrB::LogicalSemiring<bool>(), A, new_members);
            GrB::eWiseMult(candidates, GrB::complement(new_neighbors), GrB::NoAccumulate(),
                           GrB::LogicalAnd<bool>(), candidates, candidates, GrB::REPLACE);
        }
    }

    //************************************************************************
    /**
     * @brief Compute the number of triangles in a graph
     *
     * @param[in] L  Strictly lower triangular NxN matrix from the graph's
     *               adjacency matrix.  Graph is assumed to be undirected and
     *               unweighted.
     */
    template<typename MatrixT>
    uint64_t triangle_count_appendixB7(MatrixT const &L)
    {
        GrB::IndexType n(L.nrows());
        GrB::Matrix<uint64_t> C(n, n);

        GrB::mxm(C, L, GrB::NoAccumulate(),
                 GrB::ArithmeticSemiring<uint64_t>(), L, GrB::transpose(L));

        uint64_t count(0);
        GrB::reduce(count, GrB::NoAccumulate(),
                    GrB::PlusMonoid<uint64_t>(), C);

        return count;
    }

} // end namespace algorithms
