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
#include <memory>
#include <graphblas/graphblas.hpp>

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
                              grb::IndexType   src)
    {

        grb::IndexType const n(A.nrows());

        /// @todo Assert dimensions and throw DimensionException

        grb::Vector<bool> q(n);      // wavefront
        q.setElement(src, true);

        // BFS traversal and label the vertices
        grb::IndexType d(0);

        do {
            ++d;
            grb::assign(v, q, grb::NoAccumulate(),
                        d, grb::AllIndices());
            grb::vxm(q, grb::complement(v), grb::NoAccumulate(),
                     grb::LogicalSemiring<bool>(), q, A, grb::REPLACE);

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
                              grb::IndexType  src)
    {
        grb::IndexType const n(A.nrows());

        /// @todo Assert dimensions and throw DimensionException

        grb::Vector<bool> q(n);  //wavefront
        q.setElement(src, true);

        // BFS traversal and label the vertices
        grb::IndexType level(0);
        do {
            ++level;
            grb::apply(v, grb::NoMask(), grb::Plus<grb::IndexType>(),
                       grb::Times<grb::IndexType>(), q, level, grb::REPLACE);
            grb::vxm(q, grb::complement(v), grb::NoAccumulate(),
                     grb::LogicalSemiring<bool>(), q, A,
                     grb::REPLACE);

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
                               grb::IndexType  src)
    {
        grb::IndexType const N(A.nrows());

        // create index ramp for index_of() functionality
        grb::Vector<grb::IndexType> index_ramp(N);
        for (grb::IndexType i = 0; i < N; ++i)
            index_ramp.setElement(i, i);

        parents.clear();
        parents.setElement(src, src);

        grb::Vector<grb::IndexType> q(N);
        q.setElement(src, 1UL);

        // BFS traversal and label the vertices.
        while (q.nvals() > 0) {
            grb::eWiseMult(q, grb::NoMask(), grb::NoAccumulate(),
                           grb::First<grb::IndexType>(), index_ramp, q);
            grb::vxm(q,
                     grb::complement(grb::structure(parents)),
                     grb::NoAccumulate(),
                     grb::MinFirstSemiring<grb::IndexType>(),
                     q, A,
                     grb::REPLACE);
            grb::apply(parents, grb::NoMask(), grb::Plus<grb::IndexType>(),
                       grb::Identity<grb::IndexType>(), q);
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
                       grb::IndexType  src)
    {
        grb::IndexType const n(A.nrows());
        grb::Matrix<int32_t> sigma(n, n);

        grb::Vector<int32_t> q(n);  // wavefront/frontier
        q.setElement(src, 1);

        grb::Vector<int32_t> p(q);  // path

        grb::vxm(q, grb::complement(p), grb::NoAccumulate(),
                 grb::ArithmeticSemiring<int32_t>(), q, A, grb::REPLACE);

        // BFS phase
        grb::IndexType d(0);        // depth

        do {
            grb::assign(sigma, grb::NoMask(), grb::NoAccumulate(),
                        q, d, grb::AllIndices());
            grb::eWiseAdd(p, grb::NoMask(), grb::NoAccumulate(),
                          grb::Plus<int32_t>(), p, q);
            grb::vxm(q, grb::complement(p), grb::NoAccumulate(),
                     grb::ArithmeticSemiring<int32_t>(), q, A, grb::REPLACE);


            ++d;
        } while (q.nvals() > 0);

        // BC computation phase
        grb::Vector<float> t1(n), t2(n), t3(n), t4(n);

        for (int i = d-1; i > 0; --i) {
            grb::assign(t1, grb::NoMask(), grb::NoAccumulate(),
                        1.f, grb::AllIndices());
            grb::eWiseAdd(t1, grb::NoMask(), grb::NoAccumulate(),
                          grb::Plus<float>(), t1, delta);
            grb::extract(t2, grb::NoMask(), grb::NoAccumulate(),
                         grb::transpose(sigma), grb::AllIndices(), i);
            grb::eWiseMult(t2, grb::NoMask(), grb::NoAccumulate(),
                           grb::Div<float>(), t1, t2);
            grb::mxv(t3, grb::NoMask(), grb::NoAccumulate(),
                     grb::ArithmeticSemiring<float>(), A, t2);
            grb::extract(t4, grb::NoMask(), grb::NoAccumulate(),
                         grb::transpose(sigma), grb::AllIndices(), i-1);
            grb::eWiseMult(t4, grb::NoMask(), grb::NoAccumulate(),
                           grb::Times<float>(), t4, t3);
            grb::eWiseAdd(delta, grb::NoMask(), grb::NoAccumulate(),
                          grb::Plus<float>(), delta, t4);
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
        std::vector<grb::IndexType> const &src)  // aka grb::IndexArrayType
    {

        grb::IndexType nsver(src.size());
        grb::IndexType n(A.nrows());
        /// @todo assert proper dimensions and nsver > 0

        // index and value arrays needed to build NumSP
        grb::IndexArrayType GrB_ALL_nsver;     // fill with sequence
        GrB_ALL_nsver.reserve(nsver);
        for (grb::IndexType idx = 0; idx < nsver; ++idx) {
            GrB_ALL_nsver.push_back(idx);
        }

        // NumSP holds # of shortest paths to each vertex from each src.
        // Initialized to source vertices
        // NumSP[s[i],i] = 1 where 0 <= i < nsver; implied zero elsewhere
        grb::Matrix<int32_t> NumSP(n, nsver);
        NumSP.build(src, GrB_ALL_nsver, std::vector<int32_t>(nsver, 1));


        // The current frontier for all BFS's (from all roots)
        // Initialized to out neighbors of each source node in src
        grb::Matrix<int32_t> Frontier(n, nsver);

        grb::extract(Frontier, grb::complement(NumSP), grb::NoAccumulate(),
                     grb::transpose(A), grb::AllIndices(), src, grb::REPLACE);
        //grb::extract(Frontier, grb::NoMask(), grb::NoAccumulate(),
        //        grb::transpose(A), grb::AllIndices(), src);

        // std::vector manages allocation
        std::vector<std::unique_ptr<grb::Matrix<bool>>> Sigmas;

        int32_t d = 0;

        // ==================== BFS phase ====================
        while (Frontier.nvals() > 0) {
            Sigmas.emplace_back(std::make_unique<grb::Matrix<bool>>(nsver, n));
            grb::apply(*(Sigmas[d]), grb::NoMask(), grb::NoAccumulate(),
                       grb::Identity<int32_t, bool>(), Frontier);

            grb::eWiseAdd(NumSP, grb::NoMask(), grb::NoAccumulate(),
                          grb::Plus<int32_t>(), NumSP, Frontier);
            grb::mxm(Frontier, grb::complement(NumSP), grb::NoAccumulate(),
                     grb::ArithmeticSemiring<int32_t>(),
                     grb::transpose(A), Frontier, grb::REPLACE);
            ++d;
        }

        grb::Matrix<float> NspInv(n, nsver);

        grb::apply(NspInv, grb::NoMask(), grb::NoAccumulate(),
                   grb::MultiplicativeInverse<float>(), NumSP);

        grb::Matrix<float> BCu(n, nsver);
        grb::assign(BCu, grb::NoMask(), grb::NoAccumulate(),
                    1.0f, grb::AllIndices(), grb::AllIndices());

        grb::Matrix<float> W(n, nsver);

        // ================== backprop phase ==================
        for (int32_t i = d-1; i > 0; --i) {
            grb::eWiseMult(W, *Sigmas[i], grb::NoAccumulate(),
                           grb::Times<float>(), BCu, NspInv, grb::REPLACE);

            grb::mxm(W, *Sigmas[i-1], grb::NoAccumulate(),
                     grb::ArithmeticSemiring<float>(), A, W, grb::REPLACE);
            grb::eWiseMult(BCu, grb::NoMask(), grb::Plus<float>(),
                           grb::Times<float>(), W, NumSP);
        }

        // adjust result
        grb::assign(delta, grb::NoMask(), grb::NoAccumulate(),
                    -1.f*nsver, grb::AllIndices());
        grb::reduce(delta, grb::NoMask(), grb::Plus<float>(),
                    grb::Plus<float>(), BCu);
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
    void mis_appendixB6(grb::Vector<bool> &iset, MatrixT const &A,
                        double seed = 0.)
    {
        std::default_random_engine             generator;
        std::uniform_real_distribution<double> distribution;
        generator.seed(seed);

        grb::IndexType n(A.nrows());

        grb::Vector<float> prob(n);
        grb::Vector<float> neighbor_max(n);
        grb::Vector<bool>  new_members(n);
        grb::Vector<bool>  new_neighbors(n);
        grb::Vector<bool>  candidates(n);

        grb::Vector<double> degrees(n);
        grb::reduce(degrees, grb::NoMask(), grb::NoAccumulate(), grb::Plus<double>(), A);

        // Remove isolated vertices
        grb::assign(candidates, degrees, grb::NoAccumulate(), true, grb::AllIndices());

        // Add all singletons to iset
        grb::assign(iset, grb::complement(degrees), grb::NoAccumulate(),
                    true, grb::AllIndices(), grb::REPLACE);

        while (candidates.nvals() > 0) {
            // compute a random probability of each candidate scaled by inverse of degree.
            grb::eWiseMult(prob, grb::NoMask(), grb::NoAccumulate(),
                           [&](bool candidate, float const &degree)
                           { return static_cast<float>(
                                   0.0001 + distribution(generator)/(1. + 2.*degree)); },
                           candidates, degrees);

            // find the max probability of all neighbors
            grb::mxv(neighbor_max, candidates, grb::NoAccumulate(),
                     grb::MaxSecondSemiring<float>(), A, prob, grb::REPLACE);

            // Select source node if its probability is > neighbor_max
            grb::eWiseAdd(new_members, grb::NoMask(), grb::NoAccumulate(),
                          grb::GreaterThan<double>(), prob, neighbor_max);
            grb::apply(new_members, new_members, grb::NoAccumulate(),
                       grb::Identity<bool>(), new_members, grb::REPLACE);

            // Add new members to independent set.
            grb::eWiseAdd(iset, grb::NoMask(), grb::NoAccumulate(),
                          grb::LogicalOr<bool>(), iset, new_members);

            // Remove new_members from set of candidates
            grb::eWiseMult(candidates, grb::complement(new_members), grb::NoAccumulate(),
                           grb::LogicalAnd<bool>(), candidates, candidates, grb::REPLACE);


            if (candidates.nvals() == 0) break;

            // Neighbors of new members can also be removed
            grb::mxv(new_neighbors, candidates, grb::NoAccumulate(),
                     grb::LogicalSemiring<bool>(), A, new_members);
            grb::eWiseMult(candidates, grb::complement(new_neighbors), grb::NoAccumulate(),
                           grb::LogicalAnd<bool>(), candidates, candidates, grb::REPLACE);
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
        grb::IndexType n(L.nrows());
        grb::Matrix<uint64_t> C(n, n);

        grb::mxm(C, L, grb::NoAccumulate(),
                 grb::ArithmeticSemiring<uint64_t>(), L, grb::transpose(L));

        uint64_t count(0);
        grb::reduce(count, grb::NoAccumulate(), grb::PlusMonoid<uint64_t>(), C);

        return count;
    }

} // end namespace algorithms
