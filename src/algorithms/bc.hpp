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
#include <memory>

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
        grb::IndexArrayType const &s)
    {
        //GRB_BC_LOG(grb::print_matrix(std::cerr, A, "Graph"));
        GRB_BC_LOG("vertex_betweenness_centrality_batch_alt_trans_v2 Graph " << A);

        // nsver = |s| (partition size)
        grb::IndexType nsver(s.size());
        if (nsver == 0)
        {
            throw grb::DimensionException();
        }

        GRB_BC_LOG("batch size (p): " << nsver);

        grb::IndexType m = A.nrows();
        grb::IndexType n = A.ncols();
        if (m != n)
        {
            throw grb::DimensionException();
        }

        GRB_BC_LOG( "num nodes (n): " << n );

        // The current frontier for all BFS's (from all roots)
        // It is initialized to the out neighbors of the specified roots
        grb::Matrix<int32_t> Frontier(nsver, n);     // F is nsver x n (rows)
        grb::extract(Frontier,
                     grb::NoMask(),
                     grb::NoAccumulate(),
                     A,
                     s,
                     grb::AllIndices());
        GRB_BC_LOG("initial frontier " << Frontier);

        // NumSP holds number of shortest paths to a vertex from a given root
        // NumSP is initialized with the each starting root in 's':
        // NumSP[i,s[i]] = 1 where 0 <= i < nsver; implied zero elsewhere
        grb::Matrix<int32_t> NumSP(nsver, n);
        grb::IndexArrayType GrB_ALL_nsver;     // fill with sequence
        GrB_ALL_nsver.reserve(nsver);
        for (grb::IndexType idx = 0; idx < nsver; ++idx)
        {
            GrB_ALL_nsver.push_back(idx); // tilln
        }
        NumSP.build(GrB_ALL_nsver, s, std::vector<int32_t>(nsver, 1));

        // ==================== BFS phase ====================
        GRB_BC_LOG("======= START BFS phase ======");
        std::vector<std::unique_ptr<grb::Matrix<bool>>> Sigmas;
        int32_t d = 0;

        // For testing purpose we only allow 10 iterations so it doesn't
        // get into an infinite loop
        while (Frontier.nvals() > 0 && d < 10)
        {
            GRB_BC_LOG("------- BFS iteration " << d << " --------");

            // Sigma[d] = (bool)Frontier
            Sigmas.emplace_back(std::make_unique<grb::Matrix<bool>>(nsver, n));
            grb::apply(*(Sigmas[d]),
                       grb::NoMask(),
                       grb::NoAccumulate(),
                       grb::Identity<bool>(),
                       Frontier);
            GRB_BC_LOG("Sigma[d] = (bool)Frontier " << *Sigmas[d]);

            // P = F + P
            grb::eWiseAdd(NumSP,
                          grb::NoMask(),
                          grb::NoAccumulate(),
                          grb::Plus<int32_t>(),
                          NumSP,
                          Frontier);
            GRB_BC_LOG("NumSP " << NumSP);

            // F<!P> = F +.* A
            grb::mxm(Frontier,
                     grb::complement(NumSP),
                     grb::NoAccumulate(),
                     grb::ArithmeticSemiring<int32_t>(),
                     Frontier,
                     A,
                     grb::REPLACE);
            GRB_BC_LOG("New frontier " << Frontier);

            ++d;
        }
        GRB_BC_LOG("======= END BFS phase =======");

        // ================== backprop phase ==================
        grb::Matrix<float> NspInv(nsver, n);
        grb::apply(NspInv,
                   grb::NoMask(),
                   grb::NoAccumulate(),
                   grb::MultiplicativeInverse<float>(),
                   NumSP);
        GRB_BC_LOG("(1 ./ P) " << NspInv);

        grb::Matrix<float> BCu(nsver, n);
//        grb::assign_constant(BCu,
        grb::assign(BCu,
                    grb::NoMask(),
                    grb::NoAccumulate(),
                    1.0f,
                    grb::AllIndices(),
                    grb::AllIndices());

        grb::Matrix<float> W(nsver, n);
        GRB_BC_LOG("U " << BCu);

        GRB_BC_LOG("======= BEGIN BACKPROP phase =======");
        for (int32_t i = d - 1; i > 0; --i)
        {
            GRB_BC_LOG("------- BFS iteration " << d << " --------");

            // W<Sigma[i]> = (1 ./ P) .* U
            grb::eWiseMult(W,
                           *Sigmas[i],
                           grb::NoAccumulate(),
                           grb::Times<float>(),
                           NspInv, BCu,
                           grb::REPLACE);
            GRB_BC_LOG("W<Sigma[i]> = (1 ./ P) .* U " << W);

            // W<Sigma[i-1]> = A +.* W
            grb::mxm(W,
                     *Sigmas[i-1],
                     grb::NoAccumulate(),
                     grb::ArithmeticSemiring<float>(),
                     W,
                     grb::transpose(A),
                     grb::REPLACE);
            GRB_BC_LOG("W<Sigma[i-1]> = A +.* W " << W);

            // U += W .* P
            grb::eWiseMult(BCu,
                           grb::NoMask(),
                           grb::Plus<float>(),
                           grb::Times<float>(),
                           W,
                           NumSP);
            GRB_BC_LOG("U += W .* P " << BCu);

            --d;
        }

        GRB_BC_LOG("======= END BACKPROP phase =======");
        GRB_BC_LOG("BC Updates " << BCu);

        grb::Vector<float, grb::SparseTag> result(n);

        // There are a couple of different ways to subtract nsver from
        // each value in result (which is the column reduction of BCu)

#if 0
        // Method 1: Make a dense vector filled with -nsver and accumulate
        // the column reduction into the dense vector (with Plus).
        // Disadvantage: requires an additional dense vector
        grb::assign(result,
                    grb::NoMask(),
                    grb::NoAccumulate(),
                    nsver * -1.0f,
                    grb::AllIndices());
        grb::reduce(result,                          // W
                    grb::NoMask(),             // Mask
                    grb::Plus<float>(),        // Accum
                    grb::Plus<float>(),        // Op
                    grb::transpose(BCu),       // A, transpose make col reduce
                    grb::REPLACE);             // replace
#else
        // Method 2: column reduce BCu into result first and use apply to
        // subtract a constant.
        grb::reduce(result,                          // W
                    grb::NoMask(),             // Mask
                    grb::NoAccumulate(),       // Accum
                    grb::Plus<float>(),        // Op
                    grb::transpose(BCu),       // A, transpose make col reduce
                    grb::REPLACE);             // replace

        grb::apply(result,
                   grb::NoMask(),
                   grb::NoAccumulate(),
                   std::bind(grb::Minus<float>(),
                             std::placeholders::_1,
                             static_cast<float>(nsver)),
                   result,
                   grb::REPLACE);

        // The following commented out function is a C API 1.3 version of
        // apply that takes a binaryop and a constant, that probably won't
        // be supported in the C++ API in favor of using lambdas or the
        // method above using std::bind

        // grb::apply(result,
        //                  grb::NoMask(),
        //                  grb::NoAccumulate(),
        //                  grb::Minus<float>(),
        //                  result,
        //                  static_cast<float>(nsver),
        //                  grb::REPLACE);
#endif
        GRB_BC_LOG("RESULT: " << result);

        std::vector<float> betweenness_centrality(n, 0.f);
        for (grb::IndexType k = 0; k < n; k++)
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
        grb::IndexArrayType const &s)
    {
        // nsver = |s| (partition size)
        grb::IndexType nsver(s.size());
        if (nsver == 0)
        {
            throw grb::DimensionException();
        }

        grb::IndexType m = A.nrows();
        grb::IndexType n = A.ncols();
        if (m != n)
        {
            throw grb::DimensionException();
        }

        // The current frontier for all BFS's (from all roots)
        // It is initialized to the out neighbors of the specified roots
        grb::Matrix<int32_t> Frontier(nsver, n);     // F is nsver x n (rows)
        grb::extract(Frontier,
                     grb::NoMask(),
                     grb::NoAccumulate(),
                     A,
                     s,
                     grb::AllIndices());

        // NumSP holds number of shortest paths to a vertex from a given root
        // NumSP is initialized with the each starting root in 's':
        // NumSP[i,s[i]] = 1 where 0 <= i < nsver; implied zero elsewhere
        grb::Matrix<int32_t> NumSP(nsver, n);
        grb::IndexArrayType GrB_ALL_nsver;     // fill with sequence
        GrB_ALL_nsver.reserve(nsver);
        for (grb::IndexType idx = 0; idx < nsver; ++idx)
        {
            GrB_ALL_nsver.push_back(idx);
        }
        NumSP.build(GrB_ALL_nsver, s, std::vector<int32_t>(nsver, 1));

        // ==================== BFS phase ====================
        std::vector<std::unique_ptr<grb::Matrix<bool>>> Sigmas;
        int32_t d = 0;
        while (Frontier.nvals() > 0)
        {
            // Sigma[d] = (bool)Frontier
            Sigmas.emplace_back(std::make_unique<grb::Matrix<bool>>(nsver, n));
            grb::apply(*(Sigmas[d]),
                       grb::NoMask(), grb::NoAccumulate(),
                       grb::Identity<int32_t, bool>(),
                       Frontier);

            // P = F + P
            grb::eWiseAdd(NumSP,
                          grb::NoMask(), grb::NoAccumulate(),
                          grb::Plus<int32_t>(),
                          NumSP, Frontier);

            // F<!P> = F +.* A
            grb::mxm(Frontier,
                     grb::complement(NumSP),
                     grb::NoAccumulate(),
                     grb::ArithmeticSemiring<int32_t>(),
                     Frontier, A, grb::REPLACE);

            ++d;
        }
        //std::cerr << "======= END BFS phase =======" << std::endl;

        // ================== backprop phase ==================
        grb::Matrix<float> NspInv(nsver, n);
        grb::apply(NspInv,
                   grb::NoMask(),
                   grb::NoAccumulate(),
                   grb::MultiplicativeInverse<float>(),
                   NumSP,
                   grb::REPLACE);

        grb::Matrix<float> BCu(nsver, n);
//        grb::assign_constant(BCu,
        grb::assign(BCu,
                    grb::NoMask(),
                    grb::NoAccumulate(),
                    1.0f,
                    grb::AllIndices(),
                    grb::AllIndices());

        grb::Matrix<float> W(nsver, n);

        //std::cerr << "======= BEGIN BACKPROP phase =======" << std::endl;
        for (int32_t i = d - 1; i > 0; --i)
        {
            // W<Sigma[i]> = (1 ./ P) .* U
            grb::eWiseMult(W,
                           *Sigmas[i], grb::NoAccumulate(),
                           grb::Times<float>(),
                           NspInv, BCu,
                           grb::REPLACE);

            // W<Sigma[i-1]> = A +.* W
            grb::mxm(W,
                     *Sigmas[i-1], grb::NoAccumulate(),
                     grb::ArithmeticSemiring<float>(),
                     W, grb::transpose(A),
                     grb::REPLACE);

            // U += W .* P
            grb::eWiseMult(BCu,
                           grb::NoMask(), grb::Plus<float>(),
                           grb::Times<float>(),
                           W, NumSP);

            --d;
        }

        //std::cerr << "======= END BACKPROP phase =======" << std::endl;

        grb::Vector<float> result(n);
        grb::assign(result,
                    grb::NoMask(),
                    grb::NoAccumulate(),
                    -1.0f * nsver,
                    grb::AllIndices());


        // column-wise reduction
        grb::reduce(result,
                    grb::NoMask(),
                    grb::Plus<float>(),
                    grb::Plus<float>(),
                    grb::transpose(BCu));

        std::vector<float> betweenness_centrality(n, 0.f);
        for (grb::IndexType k = 0; k < n;k++)
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
        grb::IndexArrayType const &s)
    {
        // nsver = |s| (partition size)
        grb::IndexType nsver(s.size());
        if (nsver == 0)
        {
            throw grb::DimensionException();
        }

        grb::IndexType m(A.nrows());
        grb::IndexType n(A.ncols());
        if (m != n)
        {
            throw grb::DimensionException();
        }

        // The current frontier for all BFS's (from all roots)
        // It is initialized to the out neighbors of the specified roots
        grb::Matrix<int32_t> Frontier(n, nsver);         // F is n x nsver
        grb::IndexArrayType GrB_ALL_nsver;     // fill with sequence
        GrB_ALL_nsver.reserve(nsver);
        for (grb::IndexType idx = 0; idx < nsver; ++idx)
        {
            GrB_ALL_nsver.push_back(idx); // tilln
        }
        grb::extract(Frontier,
                     grb::NoMask(), grb::NoAccumulate(),
                     grb::transpose(A),
                     grb::AllIndices(),
                     s);

        // NumSP holds number of shortest paths to a vertex from a given root
        // NumSP is initialized with the each starting root in 's':
        // NumSP[s[i],i] = 1 where 0 <= i < nsver; implied zero elsewhere
        grb::Matrix<int32_t> NumSP(n, nsver);
        NumSP.build(s, GrB_ALL_nsver, std::vector<int32_t>(nsver, 1));

        // ==================== BFS phase ====================
        std::vector<grb::Matrix<bool>* > Sigmas;
        int32_t d = 0;
        while (Frontier.nvals() > 0)
        {
            Sigmas.push_back(new grb::Matrix<bool>(n, nsver));
            grb::apply(*(Sigmas[d]),
                       grb::NoMask(), grb::NoAccumulate(),
                       grb::Identity<int32_t, bool>(),
                       Frontier);

            // P = F + P
            grb::eWiseAdd(NumSP,
                          grb::NoMask(), grb::NoAccumulate(),
                          grb::Plus<int32_t>(),
                          NumSP, Frontier);

            // F<!P> = A' +.* F
            grb::mxm(Frontier,
                     grb::complement(NumSP),
                     grb::NoAccumulate(),
                     grb::ArithmeticSemiring<int32_t>(),
                     grb::transpose(A),
                     Frontier,
                     grb::REPLACE); // Replace?

            ++d;
        }
        //std::cerr << "======= END BFS phase =======" << std::endl;

        // ================== backprop phase ==================
        grb::Matrix<float> NspInv(n, nsver);
        grb::apply(NspInv,
                   grb::NoMask(),
                   grb::NoAccumulate(),
                   grb::MultiplicativeInverse<float>(),
                   NumSP,
                   grb::REPLACE);

        grb::Matrix<float> BCu(n, nsver);
//        grb::assign_constant(BCu,
        grb::assign(BCu,
                    grb::NoMask(),
                    grb::NoAccumulate(),
                    1.0f,
                    grb::AllIndices(),
                    grb::AllIndices(),
                    grb::REPLACE);

        grb::Matrix<float> W(n, nsver);

        //std::cerr << "======= BEGIN BACKPROP phase =======" << std::endl;
        for (int32_t i = d - 1; i > 0; --i)
        {
            // W<Sigma[i]> = (1 ./ P) .* U
            grb::eWiseMult(W,
                           *Sigmas[i],
                           grb::NoAccumulate(),
                           grb::Times<float>(),
                           NspInv,
                           BCu,
                           grb::REPLACE);

            // W<Sigma[i-1]> = A +.* W
            grb::mxm(W,
                     *Sigmas[i-1], grb::NoAccumulate(),
                     grb::ArithmeticSemiring<float>(),
                     A, W,
                     grb::REPLACE);

            // U += W .* P
            grb::eWiseMult(BCu,
                           grb::NoMask(),
                           grb::Plus<float>(),
                           grb::Times<float>(),
                           W,
                           NumSP); // Replace == false?
            --d;
        }

        for (auto it = Sigmas.begin(); it != Sigmas.end(); ++it)
        {
            delete *it;
        }
        //std::cerr << "======= END BACKPROP phase =======" << std::endl;

        grb::Vector<float> result(n);
        grb::assign(result,
                    grb::NoMask(),
                    grb::NoAccumulate(),
                    -1.0f * nsver,
                    grb::AllIndices());

        grb::reduce(result,
                    grb::NoMask(),
                    grb::Plus<float>(),
                    grb::Plus<float>(),
                    BCu); // replace = dont care

        std::vector<float> betweenness_centrality(n, 0.f);
        for (grb::IndexType k = 0; k < n;k++)
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
        grb::IndexArrayType const &s)
    {
        // nsver = |s| (partition size)
        grb::IndexType nsver(s.size());
        if (nsver == 0)
        {
            throw grb::DimensionException();
        }

        grb::IndexType m(A.nrows());
        grb::IndexType n(A.ncols());
        if (m != n)
        {
            throw grb::DimensionException();
        }

        // The current frontier for all BFS's (from all roots)
        // It is initialized with the each starting root in 's':
        // F[s[i],i] = 1 where 0 <= i < nsver; implied zero elsewhere
        grb::Matrix<int32_t> Frontier(n, nsver);         // F is n x nsver
        grb::IndexArrayType GrB_ALL_nsver;     // fill with sequence
        GrB_ALL_nsver.reserve(nsver);
        for (grb::IndexType idx = 0; idx < nsver; ++idx)
        {
            GrB_ALL_nsver.push_back(idx);
        }

        Frontier.build(s, GrB_ALL_nsver, std::vector<int32_t>(nsver, 1));

        // NumSP holds number of shortest paths to a vertex from a given root
        // NumSP is initialized to a copy of Frontier. NumSP is n x nsver
        grb::Matrix<int32_t> NumSP(Frontier);

        // ==================== BFS phase ====================
        std::vector<grb::Matrix<bool>* > Sigmas;
        int32_t d = 0;
        while (Frontier.nvals() > 0)
        {
            // F<!P> = A' +.* F
            grb::mxm(Frontier,
                     grb::complement(NumSP),
                     grb::NoAccumulate(),
                     grb::ArithmeticSemiring<int32_t>(),
                     grb::transpose(A),
                     Frontier,
                     grb::REPLACE);

            // Sigma[d] = (bool)F
            Sigmas.push_back(new grb::Matrix<bool>(n, nsver));
            grb::apply(*(Sigmas[d]),
                       grb::NoMask(),
                       grb::NoAccumulate(),
                       grb::Identity<int32_t, bool>(),
                       Frontier);

            // P = F + P
            grb::eWiseAdd(NumSP,
                          grb::NoMask(),
                          grb::NoAccumulate(),
                          grb::Plus<int32_t>(),
                          NumSP, Frontier);

            ++d;
        }
        // ======= END BFS phase =======

        grb::Matrix<float> NspInv(n, nsver);
        grb::apply(NspInv,
                   grb::NoMask(),
                   grb::NoAccumulate(),
                   grb::MultiplicativeInverse<float>(),
                   NumSP,
                   grb::REPLACE);

        grb::Matrix<float> BCu(n, nsver);
//        grb::assign_constant(BCu,
        grb::assign(BCu,
                    grb::NoMask(),
                    grb::NoAccumulate(),
                    1.0f,
                    grb::AllIndices(),
                    grb::AllIndices(),
                    grb::REPLACE);

        grb::Matrix<float> W(n, nsver);

        // ======= BEGIN BACKPROP phase =======
        for (int32_t i = d - 1; i > 0; --i)
        {
            // W<Sigma[i]> = (1 ./ P) .* U
            grb::eWiseMult(W,
                           *Sigmas[i],
                           grb::NoAccumulate(),
                           grb::Times<float>(),
                           NspInv,
                           BCu,
                           grb::REPLACE);

            // W<Sigma[i-1]> = A +.* W
            grb::mxm(W,
                     *Sigmas[i-1],
                     grb::NoAccumulate(),
                     grb::ArithmeticSemiring<float>(),
                     A,
                     W,
                     grb::REPLACE);

            // U += W .* P
            grb::eWiseMult(BCu,
                           grb::NoMask(),
                           grb::Plus<float>(),
                           grb::Times<float>(),
                           W,
                           NumSP); // Replace == false?

            --d;
        }
        // ======= END BACKPROP phase =======

        grb::Vector<float> result(n);
        grb::assign(result,
                    grb::NoMask(),
                    grb::NoAccumulate(),
                    -1.0f * nsver,
                    grb::AllIndices());

        // row wise reduction
        grb::reduce(result,
                    grb::NoMask(),
                    grb::Plus<float>(),
                    grb::Plus<float>(),
                    BCu); // replace = dont care

        std::vector<float> betweenness_centrality(n, 0.f);
        for (grb::IndexType k = 0; k < n;k++)
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
        grb::IndexArrayType const &src_nodes)
    {
        //grb::print_matrix(std::cerr, graph, "Graph");

        // p = nsver = |src_nodes| (partition size)
        grb::IndexType p(src_nodes.size());
        if (p == 0)
        {
            throw grb::DimensionException();
        }

        grb::IndexType N(graph.nrows());
        grb::IndexType num_cols(graph.ncols());

        if (N != num_cols)
        {
            throw grb::DimensionException();
        }

        // P holds number of shortest paths to a vertex from a given root
        // P[i,src_nodes[i]] = 1 where 0 <= i < nsver; implied zero elsewher
        grb::Matrix<int32_t> P(p, N);         // P is pxN
        grb::IndexArrayType GrB_ALL_p;     // fill with sequence
        GrB_ALL_p.reserve(p);
        for (grb::IndexType idx = 0; idx < p; ++idx)
        {
            GrB_ALL_p.push_back(idx);
        }

        P.build(GrB_ALL_p, src_nodes, std::vector<int32_t>(p, 1));

        // F is the current frontier for all BFS's (from all roots)
        // It is initialized with the out neighbors for each root
        grb::Matrix<int32_t> F(p, N);
        grb::extract(F,
                     grb::NoMask(),
                     grb::NoAccumulate(),
                     graph,
                     src_nodes,
                     grb::AllIndices());

        // ==================== BFS phase ====================
        std::vector<std::unique_ptr<grb::Matrix<int32_t>>> Sigmas;
        int32_t d = 0;
        while (F.nvals() > 0)
        {
            // Sigma[d] = F
            Sigmas.emplace_back(std::make_unique<grb::Matrix<int32_t>>(F));

            // P = F + P
            grb::eWiseAdd(P,
                          grb::NoMask(),
                          grb::NoAccumulate(),
                          grb::Plus<int32_t>(),
                          F, P);

            // F<!P> = F +.* A
            grb::mxm(F,                                    // C
                     grb::complement(P),             // M
                     grb::NoAccumulate(),            // accum
                     grb::ArithmeticSemiring<int32_t>(), // op
                     F,                                    // A
                     graph,                                // B
                     grb::REPLACE);                                // replace
            ++d;
        }

        // ================== backprop phase ==================
        grb::Matrix<double> U(p, N);        // U is pxN
        grb::assign(U,
                    grb::NoMask(),
                    grb::NoAccumulate(),
                    1.0f,
                    grb::AllIndices(),
                    grb::AllIndices());

        grb::Matrix<double> W(p, N);

        // ======= BEGIN BACKPROP phase =======
        while (d >= 2)
        {
            // W<Sigma[d-1]> = (1 + U)./P
            grb::eWiseMult(W,                            // C
                           *Sigmas[d-1],                 // Mask
                           grb::NoAccumulate(),    // accum
                           grb::Div<double>(),     // op
                           U,                            // A
                           P,                            // B
                           grb::REPLACE);                        // replace

            // W = (A +.* W')' = W +.* A'
            grb::mxm(W,                                        // C
                     grb::NoMask(),                      // M
                     grb::NoAccumulate(),                // accum
                     grb::ArithmeticSemiring<double>(),  // op
                     W,                                        // A
                     grb::transpose(graph),              // B
                     grb::REPLACE);                                    // replace

            // W<Sigma[d-2]> = W .* P
            grb::eWiseMult(W,                            // C
                           *Sigmas[d-2],                 // Mask
                           grb::NoAccumulate(),    // accum
                           grb::Times<double>(),   // op
                           W,                            // A
                           P,                            // B
                           grb::REPLACE);                        // replace

            // U = U + W
            grb::eWiseAdd(U,                            // C
                          grb::NoMask(),          // Mask
                          grb::NoAccumulate(),    // accum
                          grb::Plus<double>(),    // op
                          U,                            // A
                          W,                            // B
                          grb::REPLACE);                        // replace

            --d;
        }

        // ======= END BACKPROP phase =======

        // result = col_reduce(U) - p = reduce(transpose(U)) - p
        grb::Vector<double> result(N);
        grb::assign(result,
                    grb::NoMask(),
                    grb::NoAccumulate(),
                    -1. * p,
                    grb::AllIndices());
        grb::reduce(result,                    // W
                    grb::NoMask(),
                    grb::Plus<double>(),       // Accum
                    grb::Plus<double>(),       // Op
                    grb::transpose(U),         // transpose make col reduce
                    grb::REPLACE);

        std::vector<double> betweenness_centrality(N, 0.);
        for (grb::IndexType k = 0; k < N;k++)
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
        using VectorT = grb::Vector<T>;

        grb::IndexType num_nodes(graph.nrows());
        grb::IndexType cols(graph.ncols());

        if (num_nodes != cols)
        {
            throw grb::DimensionException();
        }

        VectorT bc_vec(num_nodes);

        // For each vertex in graph compute the BC updates (accumulate in bc_vec)
        for (grb::IndexType i = 0; i < num_nodes; ++i)
        {
            std::vector<std::unique_ptr<grb::Vector<bool>>> search;
            VectorT frontier(num_nodes);

            grb::extract(frontier,
                         grb::NoMask(),
                         grb::NoAccumulate(),
                         grb::transpose(graph),
                         // AOM - This might be wrong. Maybe, it should be col indices
                         grb::AllIndices(),
                         i,
                         grb::REPLACE);

            VectorT n_shortest_paths(num_nodes);
            n_shortest_paths.setElement(i, static_cast<T>(1));

            grb::IndexType depth = 0;
            while (frontier.nvals() > 0)
            {
                search.emplace_back(std::make_unique<grb::Vector<bool>>(num_nodes));

                grb::apply(*(search[depth]),
                           grb::NoMask(),
                           grb::NoAccumulate(),
                           grb::Identity<T, bool>(),
                           frontier);

                grb::eWiseAdd(n_shortest_paths,
                              grb::NoMask(),
                              grb::NoAccumulate(),
                              grb::Plus<T>(),
                              n_shortest_paths,
                              frontier);

                grb::vxm(frontier,
                         grb::complement(n_shortest_paths),
                         grb::NoAccumulate(),
                         grb::ArithmeticSemiring<T>(),
                         frontier,
                         graph,
                         grb::REPLACE);

                ++depth;
            }

            VectorT n_shortest_paths_inv(num_nodes);
            grb::apply(n_shortest_paths_inv,
                       grb::NoMask(),
                       grb::NoAccumulate(),
                       grb::MultiplicativeInverse<T>(),
                       n_shortest_paths);

            VectorT update(num_nodes);
            grb::assign(update,
                        grb::NoMask(),
                        grb::NoAccumulate(),
                        1,
                        grb::AllIndices());

            VectorT weights(num_nodes);
            for (int32_t idx = depth - 1; idx > 0; --idx)
            {
                --depth;

                grb::eWiseMult(weights,
                               *(search[idx]),
                               grb::NoAccumulate(),
                               grb::Times<T>(),
                               n_shortest_paths_inv,
                               update,
                               grb::REPLACE);

                grb::mxv(weights,
                         *(search[idx - 1]),
                         grb::NoAccumulate(),
                         grb::ArithmeticSemiring<T>(),
                         graph,
                         weights,
                         grb::REPLACE);

                grb::eWiseMult(update,
                               grb::NoMask(),
                               grb::Plus<T>(),
                               grb::Times<T>(),
                               weights,
                               n_shortest_paths,
                               grb::REPLACE);
            }

            grb::eWiseAdd(bc_vec,
                          grb::NoMask(),
                          grb::NoAccumulate(),
                          grb::Plus<T>(),
                          bc_vec,
                          update);
        }

        grb::Vector<T> bias(num_nodes);
        grb::assign(bias,
                    grb::NoMask(),
                    grb::NoAccumulate(),
                    static_cast<T>(num_nodes),
                    grb::AllIndices());
        grb::eWiseAdd(bc_vec,
                      grb::NoMask(),
                      grb::NoAccumulate(),
                      grb::Minus<T>(),
                      bc_vec,
                      bias,
                      grb::REPLACE);

        std::vector <typename MatrixT::ScalarType> betweenness_centrality(num_nodes, 0);
        for (grb::IndexType k = 0; k<num_nodes;k++)
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
        using VectorT = grb::Vector<T>;

        grb::IndexType num_nodes(graph.nrows());
        grb::IndexType cols(graph.ncols());

        if (num_nodes != cols)
        {
            throw grb::DimensionException();
        }

        MatrixT score(num_nodes, num_nodes);

        grb::IndexType depth;
        for (grb::IndexType root = 0; root < num_nodes; root++)
        {
            VectorT frontier(num_nodes);
            grb::extract(frontier,
                         grb::NoMask(),
                         grb::NoAccumulate(),
                         grb::transpose(graph),
                         grb::AllIndices(),
                         root);

            VectorT n_shortest_paths(num_nodes);
            n_shortest_paths.setElement(root, 1);

            MatrixT update(num_nodes, num_nodes);
            VectorT flow(num_nodes);

            depth = 0;

            std::vector<std::unique_ptr<grb::Vector<bool>>> search;
            search.emplace_back(std::make_unique<grb::Vector<bool>>(num_nodes));
            search[depth]->setElement(root, 1);

            while (frontier.nvals() != 0)
            {
                ++depth;
                search.emplace_back(std::make_unique<grb::Vector<bool>>(num_nodes));

                // search[depth] = (bool)frontier
                grb::apply(*(search[depth]),
                           grb::NoMask(),
                           grb::NoAccumulate(),
                           grb::Identity<int32_t, bool>(),
                           frontier);

                // n_shortest_paths += frontier
                grb::eWiseAdd(n_shortest_paths,
                              grb::NoMask(),
                              grb::NoAccumulate(),
                              grb::Plus<T>(),
                              n_shortest_paths,
                              frontier,
                              grb::REPLACE);

                // frontier<!n_shortest_paths>{z} = frontier *.+ graph
                grb::vxm(frontier,
                         grb::complement(n_shortest_paths),
                         grb::NoAccumulate(),
                         grb::ArithmeticSemiring<T>(),
                         frontier,
                         graph,
                         grb::REPLACE);
            }

            while (depth >= 1)
            {
                VectorT weights(num_nodes);

                // weights = <search[depth]>(flow ./ n_shortest_paths) + search[depth]
                grb::eWiseMult(weights,
                               *(search[depth]),
                               grb::NoAccumulate(),
                               grb::Div<T>(),
                               flow,
                               n_shortest_paths);
                grb::eWiseAdd(weights,
                              grb::NoMask(),
                              grb::NoAccumulate(),
                              grb::Plus<T>(),
                              weights, *(search[depth]));

                {
                    // update = A *.+ diag(weights)
                    std::vector<T> w(weights.nvals());
                    grb::IndexArrayType widx(weights.nvals());
                    weights.extractTuples(widx, w);
                    MatrixT wmat(num_nodes, num_nodes);
                    wmat.build(widx, widx, w);
                    grb::mxm(update,
                             grb::NoMask(),
                             grb::NoAccumulate(),
                             grb::ArithmeticSemiring<T>(),
                             graph, wmat);
                }

                // weights<search[depth-1]>{z} = n_shortest_paths
                grb::apply(weights,
                           *search[depth-1],
                           grb::NoAccumulate(),
                           grb::Identity<T>(),
                           n_shortest_paths, grb::REPLACE);

                {
                    // update = diag(weights) *.+ update
                    std::vector<T> w(weights.nvals());
                    grb::IndexArrayType widx(weights.nvals());
                    weights.extractTuples(widx, w);
                    MatrixT wmat(num_nodes, num_nodes);
                    wmat.build(widx, widx, w);
                    grb::mxm(update,
                             grb::NoMask(),
                             grb::NoAccumulate(),
                             grb::ArithmeticSemiring<T>(),
                             wmat, update);
                }

                // score += update
                grb::eWiseAdd(score,
                              grb::NoMask(),
                              grb::NoAccumulate(),
                              grb::Plus<T>(),
                              score, update);

                // flow = update +.
                grb::reduce(flow,
                            grb::NoMask(),
                            grb::NoAccumulate(),
                            grb::Plus<T>(),
                            update);

                depth = depth - 1;
            }
        }
        return score;
    }

} // algorithms
