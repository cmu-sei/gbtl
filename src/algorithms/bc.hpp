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

#include <iostream>
#include <functional>

#include <graphblas/graphblas.hpp>

//****************************************************************************


// Logging macros for the BC algorithms.
#if GRAPHBLAS_BC_DEBUG
    #define GRB_BC_LOG(x) do { std::cout << x << std::endl; } while(0)
#else
    #define GRB_BC_LOG(x)
#endif

namespace algorithms
{
    //************************************************************************
    /**
     * @brief Compute the vertex betweenness centrality of all vertices in the
     *        given graph from a batch of src vertices.
     *
     * @todo This version extracts source neighbors for the first frontier
     *       It precomputes 1 ./ Nsp
     *       It DOES NOT use transpose(A) in the BFS phase
     *       *It LIMITS the number of BFS traversals to 10 iterations
     *
     * The betweenness centrality of a vertex measures the number of
     * times a vertex acts as a bridge along the shortest path between two
     * vertices.
     *
     * \f$C_b(v) = \sum\limits_{s \\neq v \in V}
     *             \sum\limits_{t \\neq v \in V}\\frac{\sigma_{st}(v)}{\sigma_{st}}\f$
     *
     * @param[in]  A     The graph to compute the betweenness centrality of.
     * @param[in]  s     The set of source vertex indices from which to compute
     *                   BC contributions
     *
     * @return The betweenness centrality of all vertices in the graph relative to
     *         the specified source vertices.
     */
    template<typename MatrixT>
    std::vector<float>
    vertex_betweenness_centrality_batch_alt_trans_v2(
        MatrixT const                   &A,
        GraphBLAS::IndexArrayType const &s)
    {
        //GRB_BC_LOG(GraphBLAS::print_matrix(std::cerr, A, "Graph"));
        GRB_BC_LOG("vertex_betweenness_centrality_batch_alt_trans_v2 Graph " << A);

        // nsver = |s| (partition size)
        GraphBLAS::IndexType nsver(s.size());
        if (nsver == 0)
        {
            throw GraphBLAS::DimensionException();
        }

        GRB_BC_LOG("batch size (p): " << nsver);

        GraphBLAS::IndexType m = A.nrows();
        GraphBLAS::IndexType n = A.ncols();
        if (m != n)
        {
            throw GraphBLAS::DimensionException();
        }

        GRB_BC_LOG( "num nodes (n): " << n );

        // The current frontier for all BFS's (from all roots)
        // It is initialized to the out neighbors of the specified roots
        GraphBLAS::Matrix<int32_t> Frontier(nsver, n);     // F is nsver x n (rows)
        GraphBLAS::extract(Frontier,
                           GraphBLAS::NoMask(),
                           GraphBLAS::NoAccumulate(),
                           A,
                           s,
                           GraphBLAS::AllIndices());
        GRB_BC_LOG("initial frontier " << Frontier);

        // NumSP holds number of shortest paths to a vertex from a given root
        // NumSP is initialized with the each starting root in 's':
        // NumSP[i,s[i]] = 1 where 0 <= i < nsver; implied zero elsewhere
        GraphBLAS::Matrix<int32_t> NumSP(nsver, n);
        GraphBLAS::IndexArrayType GrB_ALL_nsver;     // fill with sequence
        GrB_ALL_nsver.reserve(nsver);
        for (GraphBLAS::IndexType idx = 0; idx < nsver; ++idx)
        {
            GrB_ALL_nsver.push_back(idx); // tilln
        }
        NumSP.build(GrB_ALL_nsver, s, std::vector<int32_t>(nsver, 1));

        // ==================== BFS phase ====================
        GRB_BC_LOG("======= START BFS phase ======");
        std::vector<GraphBLAS::Matrix<bool>* > Sigmas;
        int32_t d = 0;

        // For testing purpose we only allow 10 iterations so it doesn't
        // get into an infinite loop
        while (Frontier.nvals() > 0 && d < 10)
        {
            GRB_BC_LOG("------- BFS iteration " << d << " --------");

            // Sigma[d] = (bool)Frontier
            Sigmas.push_back(new GraphBLAS::Matrix<bool>(nsver, n));
            GraphBLAS::apply(*(Sigmas[d]),
                             GraphBLAS::NoMask(),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Identity<bool>(),
                             Frontier);
            GRB_BC_LOG("Sigma[d] = (bool)Frontier " << *Sigmas[d]);

            // P = F + P
            GraphBLAS::eWiseAdd(NumSP,
                                GraphBLAS::NoMask(),
                                GraphBLAS::NoAccumulate(),
                                GraphBLAS::Plus<int32_t>(),
                                NumSP,
                                Frontier);
            GRB_BC_LOG("NumSP " << NumSP);

            // F<!P> = F +.* A
            GraphBLAS::mxm(Frontier,
                           GraphBLAS::complement(NumSP),
                           GraphBLAS::NoAccumulate(),
                           GraphBLAS::ArithmeticSemiring<int32_t>(),
                           Frontier,
                           A,
                           GraphBLAS::REPLACE);
            GRB_BC_LOG("New frontier " << Frontier);

            ++d;
        }
        GRB_BC_LOG("======= END BFS phase =======");

        // ================== backprop phase ==================
        GraphBLAS::Matrix<float> NspInv(nsver, n);
        GraphBLAS::apply(NspInv,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::MultiplicativeInverse<float>(),
                         NumSP);
        GRB_BC_LOG("(1 ./ P) " << NspInv);

        GraphBLAS::Matrix<float> BCu(nsver, n);
//        GraphBLAS::assign_constant(BCu,
        GraphBLAS::assign(BCu,
                          GraphBLAS::NoMask(),
                          GraphBLAS::NoAccumulate(),
                          1.0f,
                          GraphBLAS::AllIndices(),
                          GraphBLAS::AllIndices());

        GraphBLAS::Matrix<float> W(nsver, n);
        GRB_BC_LOG("U " << BCu);

        GRB_BC_LOG("======= BEGIN BACKPROP phase =======");
        for (int32_t i = d - 1; i > 0; --i)
        {
            GRB_BC_LOG("------- BFS iteration " << d << " --------");

            // W<Sigma[i]> = (1 ./ P) .* U
            GraphBLAS::eWiseMult(W,
                                 *Sigmas[i],
                                 GraphBLAS::NoAccumulate(),
                                 GraphBLAS::Times<float>(),
                                 NspInv, BCu,
                                 GraphBLAS::REPLACE);
            GRB_BC_LOG("W<Sigma[i]> = (1 ./ P) .* U " << W);

            // W<Sigma[i-1]> = A +.* W
            GraphBLAS::mxm(W,
                           *Sigmas[i-1],
                           GraphBLAS::NoAccumulate(),
                           GraphBLAS::ArithmeticSemiring<float>(),
                           W,
                           GraphBLAS::transpose(A),
                           GraphBLAS::REPLACE);
            GRB_BC_LOG("W<Sigma[i-1]> = A +.* W " << W);

            // U += W .* P
            GraphBLAS::eWiseMult(BCu,
                                 GraphBLAS::NoMask(),
                                 GraphBLAS::Plus<float>(),
                                 GraphBLAS::Times<float>(),
                                 W,
                                 NumSP);
            GRB_BC_LOG("U += W .* P " << BCu);

            --d;
        }
        for (auto it = Sigmas.begin(); it != Sigmas.end(); ++it)
        {
            delete *it;
        }

        GRB_BC_LOG("======= END BACKPROP phase =======");
        GRB_BC_LOG("BC Updates " << BCu);

        GraphBLAS::Vector<float, GraphBLAS::SparseTag> result(n);

        // There are a couple of different ways to subtract nsver from
        // each value in result (which is the column reduction of BCu)

#if 0
        // Method 1: Make a dense vector filled with -nsver and accumulate
        // the column reduction into the dense vector (with Plus).
        // Disadvantage: requires an additional dense vector
        GraphBLAS::assign(result,
                          GraphBLAS::NoMask(),
                          GraphBLAS::NoAccumulate(),
                          nsver * -1.0f,
                          GraphBLAS::AllIndices());
        GraphBLAS::reduce(result,                          // W
                          GraphBLAS::NoMask(),             // Mask
                          GraphBLAS::Plus<float>(),        // Accum
                          GraphBLAS::Plus<float>(),        // Op
                          GraphBLAS::transpose(BCu),       // A, transpose make col reduce
                          GraphBLAS::REPLACE);             // replace
#else
        // Method 2: column reduce BCu into result first and use apply to
        // subtract a constant.
        GraphBLAS::reduce(result,                          // W
                          GraphBLAS::NoMask(),             // Mask
                          GraphBLAS::NoAccumulate(),       // Accum
                          GraphBLAS::Plus<float>(),        // Op
                          GraphBLAS::transpose(BCu),       // A, transpose make col reduce
                          GraphBLAS::REPLACE);             // replace

        GraphBLAS::apply(result,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         std::bind(GraphBLAS::Minus<float>(),
                                   std::placeholders::_1,
                                   static_cast<float>(nsver)),
                         result,
                         GraphBLAS::REPLACE);

        // The following commented out function is a C API 1.3 version of
        // apply that takes a binaryop and a constant, that probably won't
        // be supported in the C++ API in favor of using lambdas or the
        // method above using std::bind

        // GraphBLAS::apply(result,
        //                  GraphBLAS::NoMask(),
        //                  GraphBLAS::NoAccumulate(),
        //                  GraphBLAS::Minus<float>(),
        //                  result,
        //                  static_cast<float>(nsver),
        //                  GraphBLAS::REPLACE);
#endif
        GRB_BC_LOG("RESULT: " << result);

        std::vector<float> betweenness_centrality(n, 0.f);
        for (GraphBLAS::IndexType k = 0; k < n; k++)
        {
            if (result.hasElement(k))
            {
                betweenness_centrality[k] = result.extractElement(k);
            }
        }

        return betweenness_centrality;
    }


    //************************************************************************
    /**
     * @brief Compute the vertex betweenness centrality of all vertices in the
     *        given graph from a batch of src vertices.
     *
     * @todo This version extracts source neighbors for the first frontier
     *       It precomputes 1 ./ Nsp
     *       It DOES NOT use transpose(A) in the BFS phase
     *
     * The betweenness centrality of a vertex measures the number of
     * times a vertex acts as a bridge along the shortest path between two
     * vertices.
     *
     * \f$C_b(v)=\sum\limits_{s \\neq v \in V}
     *           \sum\limits_{t \\neq v \in V}\\frac{\sigma_{st}(v)}{\sigma_{st}}\f$
     *
     * @param[in]  A         The graph to compute the betweenness centrality of.
     * @param[in]  s         The set of source vertex indices from which to compute
     *                       BC contributions
     *
     * @return The betweenness centrality of all vertices in the graph relative to
     *         the specified source vertices.
     */
    template<typename MatrixT>
    std::vector<float>
    vertex_betweenness_centrality_batch_alt_trans(
        MatrixT const                   &A,
        GraphBLAS::IndexArrayType const &s)
    {
        // nsver = |s| (partition size)
        GraphBLAS::IndexType nsver(s.size());
        if (nsver == 0)
        {
            throw GraphBLAS::DimensionException();
        }

        GraphBLAS::IndexType m = A.nrows();
        GraphBLAS::IndexType n = A.ncols();
        if (m != n)
        {
            throw GraphBLAS::DimensionException();
        }

        // The current frontier for all BFS's (from all roots)
        // It is initialized to the out neighbors of the specified roots
        GraphBLAS::Matrix<int32_t> Frontier(nsver, n);     // F is nsver x n (rows)
        GraphBLAS::extract(Frontier,
                           GraphBLAS::NoMask(),
                           GraphBLAS::NoAccumulate(),
                           A,
                           s,
                           GraphBLAS::AllIndices());

        // NumSP holds number of shortest paths to a vertex from a given root
        // NumSP is initialized with the each starting root in 's':
        // NumSP[i,s[i]] = 1 where 0 <= i < nsver; implied zero elsewhere
        GraphBLAS::Matrix<int32_t> NumSP(nsver, n);
        GraphBLAS::IndexArrayType GrB_ALL_nsver;     // fill with sequence
        GrB_ALL_nsver.reserve(nsver);
        for (GraphBLAS::IndexType idx = 0; idx < nsver; ++idx)
        {
            GrB_ALL_nsver.push_back(idx);
        }
        NumSP.build(GrB_ALL_nsver, s, std::vector<int32_t>(nsver, 1));

        // ==================== BFS phase ====================
        std::vector<GraphBLAS::Matrix<bool>* > Sigmas;
        int32_t d = 0;
        while (Frontier.nvals() > 0)
        {
            // Sigma[d] = (bool)Frontier
            Sigmas.push_back(new GraphBLAS::Matrix<bool>(nsver, n));
            GraphBLAS::apply(*(Sigmas[d]),
                             GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                             GraphBLAS::Identity<int32_t, bool>(),
                             Frontier);

            // P = F + P
            GraphBLAS::eWiseAdd(NumSP,
                                GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                GraphBLAS::Plus<int32_t>(),
                                NumSP, Frontier);

            // F<!P> = F +.* A
            GraphBLAS::mxm(Frontier,
                           GraphBLAS::complement(NumSP),
                           GraphBLAS::NoAccumulate(),
                           GraphBLAS::ArithmeticSemiring<int32_t>(),
                           Frontier, A, GraphBLAS::REPLACE);

            ++d;
        }
        //std::cerr << "======= END BFS phase =======" << std::endl;

        // ================== backprop phase ==================
        GraphBLAS::Matrix<float> NspInv(nsver, n);
        GraphBLAS::apply(NspInv,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::MultiplicativeInverse<float>(),
                         NumSP,
                         GraphBLAS::REPLACE);

        GraphBLAS::Matrix<float> BCu(nsver, n);
//        GraphBLAS::assign_constant(BCu,
        GraphBLAS::assign(BCu,
                                   GraphBLAS::NoMask(),
                                   GraphBLAS::NoAccumulate(),
                                   1.0f,
                                   GraphBLAS::AllIndices(),
                                   GraphBLAS::AllIndices());

        GraphBLAS::Matrix<float> W(nsver, n);

        //std::cerr << "======= BEGIN BACKPROP phase =======" << std::endl;
        for (int32_t i = d - 1; i > 0; --i)
        {
            // W<Sigma[i]> = (1 ./ P) .* U
            GraphBLAS::eWiseMult(W,
                                 *Sigmas[i], GraphBLAS::NoAccumulate(),
                                 GraphBLAS::Times<float>(),
                                 NspInv, BCu,
                                 GraphBLAS::REPLACE);

            // W<Sigma[i-1]> = A +.* W
            GraphBLAS::mxm(W,
                           *Sigmas[i-1], GraphBLAS::NoAccumulate(),
                           GraphBLAS::ArithmeticSemiring<float>(),
                           W, GraphBLAS::transpose(A),
                           GraphBLAS::REPLACE);

            // U += W .* P
            GraphBLAS::eWiseMult(BCu,
                                 GraphBLAS::NoMask(), GraphBLAS::Plus<float>(),
                                 GraphBLAS::Times<float>(),
                                 W, NumSP);

            --d;
        }

        for (auto it = Sigmas.begin(); it != Sigmas.end(); ++it)
        {
            delete *it;
        }
        //std::cerr << "======= END BACKPROP phase =======" << std::endl;

        GraphBLAS::Vector<float> result(n);
        GraphBLAS::assign(result,
                          GraphBLAS::NoMask(),
                          GraphBLAS::NoAccumulate(),
                          -1.0f * nsver,
                          GraphBLAS::AllIndices());


        // column-wise reduction
        GraphBLAS::reduce(result,
                          GraphBLAS::NoMask(),
                          GraphBLAS::Plus<float>(),
                          GraphBLAS::Plus<float>(),
                          GraphBLAS::transpose(BCu));

        std::vector<float> betweenness_centrality(n, 0.f);
        for (GraphBLAS::IndexType k = 0; k < n;k++)
        {
            if (result.hasElement(k))
            {
                betweenness_centrality[k] = result.extractElement(k);
            }
        }

        return betweenness_centrality;
    }

    //************************************************************************
    /**
     * @brief Compute the vertex betweenness centrality of all vertices in the
     *        given graph from a batch of src vertices.
     *
     * @todo This version extracts source neighbors for the first frontier
     *       It precomputes 1 ./ Nsp
     *       It uses transpose(A) in the BFS phase
     *
     * The betweenness centrality of a vertex measures the number of
     * times a vertex acts as a bridge along the shortest path between two
     * vertices.
     *
     * \f$C_b(v)=\sum\limits_{s \\neq v \in V}
     *           \sum\limits_{t \\neq v \in V}\\frac{\sigma_{st}(v)}{\sigma_{st}}\f$
     *
     * @param[in]  A         The graph to compute the betweenness centrality of.
     * @param[in]  s         The set of source vertex indices from which to compute
     *                       BC contributions
     *
     * @return The betweenness centrality of all vertices in the graph relative to
     *         the specified source vertices.
     */
    template<typename MatrixT>
    std::vector<float>
    vertex_betweenness_centrality_batch_alt(
        MatrixT const                   &A,
        GraphBLAS::IndexArrayType const &s)
    {
        // nsver = |s| (partition size)
        GraphBLAS::IndexType nsver(s.size());
        if (nsver == 0)
        {
            throw GraphBLAS::DimensionException();
        }

        GraphBLAS::IndexType m(A.nrows());
        GraphBLAS::IndexType n(A.ncols());
        if (m != n)
        {
            throw GraphBLAS::DimensionException();
        }

        // The current frontier for all BFS's (from all roots)
        // It is initialized to the out neighbors of the specified roots
        GraphBLAS::Matrix<int32_t> Frontier(n, nsver);         // F is n x nsver
        GraphBLAS::IndexArrayType GrB_ALL_nsver;     // fill with sequence
        GrB_ALL_nsver.reserve(nsver);
        for (GraphBLAS::IndexType idx = 0; idx < nsver; ++idx)
        {
            GrB_ALL_nsver.push_back(idx); // tilln
        }
        GraphBLAS::extract(Frontier,
                           GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                           GraphBLAS::transpose(A),
                           GraphBLAS::AllIndices(),
                           s);

        // NumSP holds number of shortest paths to a vertex from a given root
        // NumSP is initialized with the each starting root in 's':
        // NumSP[s[i],i] = 1 where 0 <= i < nsver; implied zero elsewhere
        GraphBLAS::Matrix<int32_t> NumSP(n, nsver);
        NumSP.build(s, GrB_ALL_nsver, std::vector<int32_t>(nsver, 1));

        // ==================== BFS phase ====================
        std::vector<GraphBLAS::Matrix<bool>* > Sigmas;
        int32_t d = 0;
        while (Frontier.nvals() > 0)
        {
            Sigmas.push_back(new GraphBLAS::Matrix<bool>(n, nsver));
            GraphBLAS::apply(*(Sigmas[d]),
                             GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                             GraphBLAS::Identity<int32_t, bool>(),
                             Frontier);

            // P = F + P
            GraphBLAS::eWiseAdd(NumSP,
                                GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                GraphBLAS::Plus<int32_t>(),
                                NumSP, Frontier);

            // F<!P> = A' +.* F
            GraphBLAS::mxm(Frontier,
                           GraphBLAS::complement(NumSP),
                           GraphBLAS::NoAccumulate(),
                           GraphBLAS::ArithmeticSemiring<int32_t>(),
                           GraphBLAS::transpose(A),
                           Frontier,
                           GraphBLAS::REPLACE); // Replace?

            ++d;
        }
        //std::cerr << "======= END BFS phase =======" << std::endl;

        // ================== backprop phase ==================
        GraphBLAS::Matrix<float> NspInv(n, nsver);
        GraphBLAS::apply(NspInv,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::MultiplicativeInverse<float>(),
                         NumSP,
                         GraphBLAS::REPLACE);

        GraphBLAS::Matrix<float> BCu(n, nsver);
//        GraphBLAS::assign_constant(BCu,
        GraphBLAS::assign(BCu,
                                   GraphBLAS::NoMask(),
                                   GraphBLAS::NoAccumulate(),
                                   1.0f,
                                   GraphBLAS::AllIndices(),
                                   GraphBLAS::AllIndices(),
                                   GraphBLAS::REPLACE);

        GraphBLAS::Matrix<float> W(n, nsver);

        //std::cerr << "======= BEGIN BACKPROP phase =======" << std::endl;
        for (int32_t i = d - 1; i > 0; --i)
        {
            // W<Sigma[i]> = (1 ./ P) .* U
            GraphBLAS::eWiseMult(W,
                                 *Sigmas[i],
                                 GraphBLAS::NoAccumulate(),
                                 GraphBLAS::Times<float>(),
                                 NspInv,
                                 BCu,
                                 GraphBLAS::REPLACE);

            // W<Sigma[i-1]> = A +.* W
            GraphBLAS::mxm(W,
                           *Sigmas[i-1], GraphBLAS::NoAccumulate(),
                           GraphBLAS::ArithmeticSemiring<float>(),
                           A, W,
                           GraphBLAS::REPLACE);

            // U += W .* P
            GraphBLAS::eWiseMult(BCu,
                                 GraphBLAS::NoMask(),
                                 GraphBLAS::Plus<float>(),
                                 GraphBLAS::Times<float>(),
                                 W,
                                 NumSP); // Replace == false?
            --d;
        }

        for (auto it = Sigmas.begin(); it != Sigmas.end(); ++it)
        {
            delete *it;
        }
        //std::cerr << "======= END BACKPROP phase =======" << std::endl;

        GraphBLAS::Vector<float> result(n);
        GraphBLAS::assign(result,
                          GraphBLAS::NoMask(),
                          GraphBLAS::NoAccumulate(),
                          -1.0f * nsver,
                          GraphBLAS::AllIndices());

        GraphBLAS::reduce(result,
                          GraphBLAS::NoMask(),
                          GraphBLAS::Plus<float>(),
                          GraphBLAS::Plus<float>(),
                          BCu); // replace = dont care

        std::vector<float> betweenness_centrality(n, 0.f);
        for (GraphBLAS::IndexType k = 0; k < n;k++)
        {
            if (result.hasElement(k))
            {
                betweenness_centrality[k] = result.extractElement(k);
            }
        }

        return betweenness_centrality;
    }

    //************************************************************************
    /**
     * @brief Compute the vertex betweenness centrality of all vertices in the
     *        given graph from a batch of src vertices.
     *
     * @todo This version sets the first frontier to the source vertices,
     *       It precomputes 1 ./ Nsp
     *       It uses transpose(A) in the BFS phase
     *
     * The betweenness centrality of a vertex measures the number of
     * times a vertex acts as a bridge along the shortest path between two
     * vertices.
     *
     * \f$C_b(v)=\sum\limits_{s \\neq v \in V}
     *           \sum\limits_{t \\neq v \in V}\\frac{\sigma_{st}(v)}{\sigma_{st}}\f$
     *
     * @param[in]  A         The graph to compute the betweenness centrality of.
     * @param[in]  s         The set of source vertex indices from which to compute
     *                       BC contributions
     *
     * @return The betweenness centrality of all vertices in the graph relative to
     *         the specified source vertices.
     */
    template<typename MatrixT>
    std::vector<float>
    vertex_betweenness_centrality_batch(
        MatrixT const                   &A,
        GraphBLAS::IndexArrayType const &s)
    {
        // nsver = |s| (partition size)
        GraphBLAS::IndexType nsver(s.size());
        if (nsver == 0)
        {
            throw GraphBLAS::DimensionException();
        }

        GraphBLAS::IndexType m(A.nrows());
        GraphBLAS::IndexType n(A.ncols());
        if (m != n)
        {
            throw GraphBLAS::DimensionException();
        }

        // The current frontier for all BFS's (from all roots)
        // It is initialized with the each starting root in 's':
        // F[s[i],i] = 1 where 0 <= i < nsver; implied zero elsewhere
        GraphBLAS::Matrix<int32_t> Frontier(n, nsver);         // F is n x nsver
        GraphBLAS::IndexArrayType GrB_ALL_nsver;     // fill with sequence
        GrB_ALL_nsver.reserve(nsver);
        for (GraphBLAS::IndexType idx = 0; idx < nsver; ++idx)
        {
            GrB_ALL_nsver.push_back(idx);
        }

        Frontier.build(s, GrB_ALL_nsver, std::vector<int32_t>(nsver, 1));

        // NumSP holds number of shortest paths to a vertex from a given root
        // NumSP is initialized to a copy of Frontier. NumSP is n x nsver
        GraphBLAS::Matrix<int32_t> NumSP(Frontier);

        // ==================== BFS phase ====================
        std::vector<GraphBLAS::Matrix<bool>* > Sigmas;
        int32_t d = 0;
        while (Frontier.nvals() > 0)
        {
            // F<!P> = A' +.* F
            GraphBLAS::mxm(Frontier,
                           GraphBLAS::complement(NumSP),
                           GraphBLAS::NoAccumulate(),
                           GraphBLAS::ArithmeticSemiring<int32_t>(),
                           GraphBLAS::transpose(A),
                           Frontier,
                           GraphBLAS::REPLACE);

            // Sigma[d] = (bool)F
            Sigmas.push_back(new GraphBLAS::Matrix<bool>(n, nsver));
            GraphBLAS::apply(*(Sigmas[d]),
                             GraphBLAS::NoMask(),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Identity<int32_t, bool>(),
                             Frontier);

            // P = F + P
            GraphBLAS::eWiseAdd(NumSP,
                                GraphBLAS::NoMask(),
                                GraphBLAS::NoAccumulate(),
                                GraphBLAS::Plus<int32_t>(),
                                NumSP, Frontier);

            ++d;
        }
        // ======= END BFS phase =======

        GraphBLAS::Matrix<float> NspInv(n, nsver);
        GraphBLAS::apply(NspInv,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::MultiplicativeInverse<float>(),
                         NumSP,
                         GraphBLAS::REPLACE);

        GraphBLAS::Matrix<float> BCu(n, nsver);
//        GraphBLAS::assign_constant(BCu,
        GraphBLAS::assign(BCu,
                                   GraphBLAS::NoMask(),
                                   GraphBLAS::NoAccumulate(),
                                   1.0f,
                                   GraphBLAS::AllIndices(),
                                   GraphBLAS::AllIndices(),
                                   GraphBLAS::REPLACE);

        GraphBLAS::Matrix<float> W(n, nsver);

        // ======= BEGIN BACKPROP phase =======
        for (int32_t i = d - 1; i > 0; --i)
        {
            // W<Sigma[i]> = (1 ./ P) .* U
            GraphBLAS::eWiseMult(W,
                                 *Sigmas[i],
                                 GraphBLAS::NoAccumulate(),
                                 GraphBLAS::Times<float>(),
                                 NspInv,
                                 BCu,
                                 GraphBLAS::REPLACE);

            // W<Sigma[i-1]> = A +.* W
            GraphBLAS::mxm(W,
                           *Sigmas[i-1],
                           GraphBLAS::NoAccumulate(),
                           GraphBLAS::ArithmeticSemiring<float>(),
                           A,
                           W,
                           GraphBLAS::REPLACE);

            // U += W .* P
            GraphBLAS::eWiseMult(BCu,
                                 GraphBLAS::NoMask(),
                                 GraphBLAS::Plus<float>(),
                                 GraphBLAS::Times<float>(),
                                 W,
                                 NumSP); // Replace == false?

            --d;
        }
        // ======= END BACKPROP phase =======

        for (auto it = Sigmas.begin(); it != Sigmas.end(); ++it)
        {
            delete *it;
        }

        GraphBLAS::Vector<float> result(n);
        GraphBLAS::assign(result,
                          GraphBLAS::NoMask(),
                          GraphBLAS::NoAccumulate(),
                          -1.0f * nsver,
                          GraphBLAS::AllIndices());

        // row wise reduction
        GraphBLAS::reduce(result,
                          GraphBLAS::NoMask(),
                          GraphBLAS::Plus<float>(),
                          GraphBLAS::Plus<float>(),
                          BCu); // replace = dont care

        std::vector<float> betweenness_centrality(n, 0.f);
        for (GraphBLAS::IndexType k = 0; k < n;k++)
        {
            if (result.hasElement(k))
            {
                betweenness_centrality[k] = result.extractElement(k);
            }
        }

        return betweenness_centrality;
    }


    //************************************************************************
    /**
     * @brief Compute the vertex betweenness centrality of all vertices in the
     *        given graph from a batch of src vertices.
     *
     * @todo this version extracts the first frontier, and does not precompute 1 ./ Nsp
     *
     * @todo This version extracts the source neighbors for the first frontier
     *       It does not precompute 1 ./ Nsp
     *       It does not use transpose(A) in the BFS phase
     *
     * The betweenness centrality of a vertex measures the number of
     * times a vertex acts as a bridge along the shortest path between two
     * vertices.
     *
     * \f$C_b(v)=\sum\limits_{s \\neq v \in V}
     *           \sum\limits_{t \\neq v \in V}\\frac{\sigma_{st}(v)}{\sigma_{st}}\f$
     *
     * @param[in]  graph     The graph to compute the betweenness centrality of.
     * @param[in]  src_nodes The set of source vertex indices from which to compute
     *                       BC contributions
     *
     * @return The betweenness centrality of all vertices in the graph relative to
     *         the specified source vertices.
     */
    template<typename MatrixT>
    std::vector<double>
    vertex_betweenness_centrality_batch_old(
        MatrixT const                   &graph,
        GraphBLAS::IndexArrayType const &src_nodes)
    {
        //GraphBLAS::print_matrix(std::cerr, graph, "Graph");

        // p = nsver = |src_nodes| (partition size)
        GraphBLAS::IndexType p(src_nodes.size());
        if (p == 0)
        {
            throw GraphBLAS::DimensionException();
        }

        GraphBLAS::IndexType N(graph.nrows());
        GraphBLAS::IndexType num_cols(graph.ncols());

        if (N != num_cols)
        {
            throw GraphBLAS::DimensionException();
        }

        // P holds number of shortest paths to a vertex from a given root
        // P[i,src_nodes[i]] = 1 where 0 <= i < nsver; implied zero elsewher
        GraphBLAS::Matrix<int32_t> P(p, N);         // P is pxN
        GraphBLAS::IndexArrayType GrB_ALL_p;     // fill with sequence
        GrB_ALL_p.reserve(p);
        for (GraphBLAS::IndexType idx = 0; idx < p; ++idx)
        {
            GrB_ALL_p.push_back(idx);
        }

        P.build(GrB_ALL_p, src_nodes, std::vector<int32_t>(p, 1));

        // F is the current frontier for all BFS's (from all roots)
        // It is initialized with the out neighbors for each root
        GraphBLAS::Matrix<int32_t> F(p, N);
        GraphBLAS::extract(F,
                           GraphBLAS::NoMask(),
                           GraphBLAS::NoAccumulate(),
                           graph,
                           src_nodes,
                           GraphBLAS::AllIndices());

        // ==================== BFS phase ====================
        std::vector<GraphBLAS::Matrix<int32_t>* > Sigmas;
        int32_t d = 0;
        while (F.nvals() > 0)
        {
            Sigmas.push_back(new GraphBLAS::Matrix<int32_t>(F));  // Sigma[d] = F

            // P = F + P
            GraphBLAS::eWiseAdd(P,
                                GraphBLAS::NoMask(),
                                GraphBLAS::NoAccumulate(),
                                GraphBLAS::Plus<int32_t>(),
                                F, P);

            // F<!P> = F +.* A
            GraphBLAS::mxm(F,                                    // C
                           GraphBLAS::complement(P),             // M
                           GraphBLAS::NoAccumulate(),            // accum
                           GraphBLAS::ArithmeticSemiring<int32_t>(), // op
                           F,                                    // A
                           graph,                                // B
                           GraphBLAS::REPLACE);                                // replace
            ++d;
        }

        // ================== backprop phase ==================
        GraphBLAS::Matrix<double> U(p, N);        // U is pxN
//        GraphBLAS::assign_constant(U,
        GraphBLAS::assign(U,
                                   GraphBLAS::NoMask(),
                                   GraphBLAS::NoAccumulate(),
                                   1.0f,
                                   GraphBLAS::AllIndices(),
                                   GraphBLAS::AllIndices());

        GraphBLAS::Matrix<double> W(p, N);

        // ======= BEGIN BACKPROP phase =======
        while (d >= 2)
        {
            // W<Sigma[d-1]> = (1 + U)./P
            GraphBLAS::eWiseMult(W,                            // C
                                 *Sigmas[d-1],                 // Mask
                                 GraphBLAS::NoAccumulate(),    // accum
                                 GraphBLAS::Div<double>(),     // op
                                 U,                            // A
                                 P,                            // B
                                 GraphBLAS::REPLACE);                        // replace

            // W = (A +.* W')' = W +.* A'
            GraphBLAS::mxm(W,                                        // C
                           GraphBLAS::NoMask(),                      // M
                           GraphBLAS::NoAccumulate(),                // accum
                           GraphBLAS::ArithmeticSemiring<double>(),  // op
                           W,                                        // A
                           GraphBLAS::transpose(graph),              // B
                           GraphBLAS::REPLACE);                                    // replace

            // W<Sigma[d-2]> = W .* P
            GraphBLAS::eWiseMult(W,                            // C
                                 *Sigmas[d-2],                 // Mask
                                 GraphBLAS::NoAccumulate(),    // accum
                                 GraphBLAS::Times<double>(),   // op
                                 W,                            // A
                                 P,                            // B
                                 GraphBLAS::REPLACE);                        // replace

            // U = U + W
            GraphBLAS::eWiseAdd(U,                            // C
                                GraphBLAS::NoMask(),          // Mask
                                GraphBLAS::NoAccumulate(),    // accum
                                GraphBLAS::Plus<double>(),    // op
                                U,                            // A
                                W,                            // B
                                GraphBLAS::REPLACE);                        // replace

            --d;
        }

        for (auto it = Sigmas.begin(); it != Sigmas.end(); ++it)
        {
            delete *it;
        }

        // ======= END BACKPROP phase =======

        // result = col_reduce(U) - p = reduce(transpose(U)) - p
        GraphBLAS::Vector<double> result(N);
        GraphBLAS::assign(result,
                          GraphBLAS::NoMask(),
                          GraphBLAS::NoAccumulate(),
                          -1. * p,
                          GraphBLAS::AllIndices());
        GraphBLAS::reduce(result,                          // W
                          GraphBLAS::NoMask(),             // Mask
                          GraphBLAS::Plus<double>(),       // Accum
                          GraphBLAS::Plus<double>(),       // Op
                          GraphBLAS::transpose(U),         // A, transpose make col reduce
                          GraphBLAS::REPLACE);                           // replace

        std::vector<double> betweenness_centrality(N, 0.);
        for (GraphBLAS::IndexType k = 0; k < N;k++)
        {
            if (result.hasElement(k))
            {
                betweenness_centrality[k] = result.extractElement(k);
            }
        }

        return betweenness_centrality;
    }

    //************************************************************************
    /**
     * @brief Compute the vertex betweenness centrality of all vertices in the
     *        given graph ("for all" one at a time).
     *
     * The betweenness centrality of a vertex measures the number of
     * times a vertex acts as a bridge along the shortest path between two
     * vertices.
     *
     * \f$C_b(v) = \sum\limits_{s \\neq v \in V}
     *             \sum\limits_{t \\neq v \in V}\\frac{\sigma_{st}(v)}{\sigma_{st}}\f$
     *
     * @param[in]  graph  The graph to compute the betweenness centrality of.
     *
     * @return The betweenness centrality of all vertices in the graph.
     */
    template<typename MatrixT>
    std::vector<typename MatrixT::ScalarType>
    vertex_betweenness_centrality(MatrixT const &graph)
    {
        using T = typename MatrixT::ScalarType;
        using VectorT = GraphBLAS::Vector<T>;

        GraphBLAS::IndexType num_nodes(graph.nrows());
        GraphBLAS::IndexType cols(graph.ncols());

        if (num_nodes != cols)
        {
            throw GraphBLAS::DimensionException();
        }

        VectorT bc_vec(num_nodes);

        // For each vertex in graph compute the BC updates (accumulate in bc_vec)
        for (GraphBLAS::IndexType i = 0; i < num_nodes; ++i)
        {
            std::vector<GraphBLAS::Vector<bool>*> search;
            VectorT frontier(num_nodes);

            GraphBLAS::extract(frontier,
                               GraphBLAS::NoMask(),
                               GraphBLAS::NoAccumulate(),
                               GraphBLAS::transpose(graph),
                               // AOM - This might be wrong. Maybe, it should be col indices
                               GraphBLAS::AllIndices(),
                               i,
                               GraphBLAS::REPLACE);

            VectorT n_shortest_paths(num_nodes);
            n_shortest_paths.setElement(i, static_cast<T>(1));

            GraphBLAS::IndexType depth = 0;
            while (frontier.nvals() > 0)
            {
                search.push_back(new GraphBLAS::Vector<bool>(num_nodes));

                GraphBLAS::apply(*(search[depth]),
                                 GraphBLAS::NoMask(),
                                 GraphBLAS::NoAccumulate(),
                                 GraphBLAS::Identity<T, bool>(),
                                 frontier);

                GraphBLAS::eWiseAdd(n_shortest_paths,
                                    GraphBLAS::NoMask(),
                                    GraphBLAS::NoAccumulate(),
                                    GraphBLAS::Plus<T>(),
                                    n_shortest_paths,
                                    frontier);

                GraphBLAS::vxm(frontier,
                               GraphBLAS::complement(n_shortest_paths),
                               GraphBLAS::NoAccumulate(),
                               GraphBLAS::ArithmeticSemiring<T>(),
                               frontier,
                               graph,
                               GraphBLAS::REPLACE);

                ++depth;
            }

            VectorT n_shortest_paths_inv(num_nodes);
            GraphBLAS::apply(n_shortest_paths_inv,
                             GraphBLAS::NoMask(),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::MultiplicativeInverse<T>(),
                             n_shortest_paths);

            VectorT update(num_nodes);
            GraphBLAS::assign(update,
                              GraphBLAS::NoMask(),
                              GraphBLAS::NoAccumulate(),
                              1,
                              GraphBLAS::AllIndices());

            VectorT weights(num_nodes);
            for (int32_t idx = depth - 1; idx > 0; --idx)
            {
                --depth;

                GraphBLAS::eWiseMult(weights,
                                     *(search[idx]),
                                     GraphBLAS::NoAccumulate(),
                                     GraphBLAS::Times<T>(),
                                     n_shortest_paths_inv,
                                     update,
                                     GraphBLAS::REPLACE);

                GraphBLAS::mxv(weights,
                               *(search[idx - 1]),
                               GraphBLAS::NoAccumulate(),
                               GraphBLAS::ArithmeticSemiring<T>(),
                               graph,
                               weights,
                               GraphBLAS::REPLACE);

                GraphBLAS::eWiseMult(update,
                                     GraphBLAS::NoMask(),
                                     GraphBLAS::Plus<T>(),
                                     GraphBLAS::Times<T>(),
                                     weights,
                                     n_shortest_paths,
                                     GraphBLAS::REPLACE);
            }

            for (auto it = search.begin(); it != search.end(); ++it)
            {
                delete *it;
            }

            GraphBLAS::eWiseAdd(bc_vec,
                                GraphBLAS::NoMask(),
                                GraphBLAS::NoAccumulate(),
                                GraphBLAS::Plus<T>(),
                                bc_vec,
                                update);

        }

        GraphBLAS::Vector<T> bias(num_nodes);
        GraphBLAS::assign(bias,
                          GraphBLAS::NoMask(),
                          GraphBLAS::NoAccumulate(),
                          static_cast<T>(num_nodes),
                          GraphBLAS::AllIndices());
        GraphBLAS::eWiseAdd(bc_vec,
                            GraphBLAS::NoMask(),
                            GraphBLAS::NoAccumulate(),
                            GraphBLAS::Minus<T>(),
                            bc_vec,
                            bias,
                            GraphBLAS::REPLACE);

        std::vector <typename MatrixT::ScalarType> betweenness_centrality(num_nodes, 0);
        for (GraphBLAS::IndexType k = 0; k<num_nodes;k++)
        {
            if (bc_vec.hasElement(k))
            {
                betweenness_centrality[k] = bc_vec.extractElement(k);
            }
        }

        return betweenness_centrality;
    }

    //************************************************************************
    /**
     * @brief Compute the edge betweenness centrality of all vertices in the
     *        given graph.
     *
     * The betweenness centrality of a vertex measures the number of
     * times an edge acts as a bridge along the shortest path between two
     * vertices.  Formally stated:
     *
     * @param[in] graph The graph to compute the edge betweenness centrality
     *
     * @return The betweenness centrality of all edges in the graph
     */
    template<typename MatrixT>
    MatrixT edge_betweenness_centrality(MatrixT const &graph)
    {
        using T = typename MatrixT::ScalarType;
        using VectorT = GraphBLAS::Vector<T>;

        GraphBLAS::IndexType num_nodes(graph.nrows());
        GraphBLAS::IndexType cols(graph.ncols());

        if (num_nodes != cols)
        {
            throw GraphBLAS::DimensionException();
        }

        MatrixT score(num_nodes, num_nodes);

        GraphBLAS::IndexType depth;
        for (GraphBLAS::IndexType root = 0; root < num_nodes; root++)
        {
            VectorT frontier(num_nodes);
            GraphBLAS::extract(frontier,
                               GraphBLAS::NoMask(),
                               GraphBLAS::NoAccumulate(),
                               GraphBLAS::transpose(graph),
                               GraphBLAS::AllIndices(),
                               root);

            VectorT n_shortest_paths(num_nodes);
            n_shortest_paths.setElement(root, 1);

            MatrixT update(num_nodes, num_nodes);
            VectorT flow(num_nodes);

            depth = 0;

            std::vector<GraphBLAS::Vector<bool>*> search;
            search.push_back(new GraphBLAS::Vector<bool>(num_nodes));
            search[depth]->setElement(root, 1);

            while (frontier.nvals() != 0)
            {
                ++depth;
                search.push_back(new GraphBLAS::Vector<bool>(num_nodes));

                // search[depth] = (bool)frontier
                GraphBLAS::apply(*(search[depth]),
                                 GraphBLAS::NoMask(),
                                 GraphBLAS::NoAccumulate(),
                                 GraphBLAS::Identity<int32_t, bool>(),
                                 frontier);

                // n_shortest_paths += frontier
                GraphBLAS::eWiseAdd(n_shortest_paths,
                                    GraphBLAS::NoMask(),
                                    GraphBLAS::NoAccumulate(),
                                    GraphBLAS::Plus<T>(),
                                    n_shortest_paths,
                                    frontier,
                                    GraphBLAS::REPLACE);

                // frontier<!n_shortest_paths>{z} = frontier *.+ graph
                GraphBLAS::vxm(frontier,
                               GraphBLAS::complement(n_shortest_paths),
                               GraphBLAS::NoAccumulate(),
                               GraphBLAS::ArithmeticSemiring<T>(),
                               frontier,
                               graph,
                               GraphBLAS::REPLACE);
            }

            while (depth >= 1)
            {
                VectorT weights(num_nodes);

                // weights = <search[depth]>(flow ./ n_shortest_paths) + search[depth]
                GraphBLAS::eWiseMult(weights,
                                     *(search[depth]),
                                     GraphBLAS::NoAccumulate(),
                                     GraphBLAS::Div<T>(),
                                     flow,
                                     n_shortest_paths);
                GraphBLAS::eWiseAdd(weights,
                                    GraphBLAS::NoMask(),
                                    GraphBLAS::NoAccumulate(),
                                    GraphBLAS::Plus<T>(),
                                    weights, *(search[depth]));

                {
                    // update = A *.+ diag(weights)
                    std::vector<T> w(weights.nvals());
                    GraphBLAS::IndexArrayType widx(weights.nvals());
                    weights.extractTuples(widx, w);
                    MatrixT wmat(num_nodes, num_nodes);
                    wmat.build(widx, widx, w);
                    GraphBLAS::mxm(update,
                                   GraphBLAS::NoMask(),
                                   GraphBLAS::NoAccumulate(),
                                   GraphBLAS::ArithmeticSemiring<T>(),
                                   graph, wmat);
                }

                // weights<search[depth-1]>{z} = n_shortest_paths
                GraphBLAS::apply(weights,
                                 *search[depth-1],
                                 GraphBLAS::NoAccumulate(),
                                 GraphBLAS::Identity<T>(),
                                 n_shortest_paths, GraphBLAS::REPLACE);

                {
                    // update = diag(weights) *.+ update
                    std::vector<T> w(weights.nvals());
                    GraphBLAS::IndexArrayType widx(weights.nvals());
                    weights.extractTuples(widx, w);
                    MatrixT wmat(num_nodes, num_nodes);
                    wmat.build(widx, widx, w);
                    GraphBLAS::mxm(update,
                                   GraphBLAS::NoMask(),
                                   GraphBLAS::NoAccumulate(),
                                   GraphBLAS::ArithmeticSemiring<T>(),
                                   wmat, update);
                }

                // score += update
                GraphBLAS::eWiseAdd(score,
                                    GraphBLAS::NoMask(),
                                    GraphBLAS::NoAccumulate(),
                                    GraphBLAS::Plus<T>(),
                                    score, update);

                // flow = update +.
                GraphBLAS::reduce(flow,
                                  GraphBLAS::NoMask(),
                                  GraphBLAS::NoAccumulate(),
                                  GraphBLAS::Plus<T>(),
                                  update);

                depth = depth - 1;
            }

            // clean up
            for (auto it = search.begin(); it != search.end(); ++it)
            {
                delete *it;
            }
        }
        return score;
    }

} // algorithms
