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

#ifndef ALGORITHMS_BC_HPP
#define ALGORITHMS_BC_HPP

#include <iostream>

#define GB_DEBUG
#include <graphblas/graphblas.hpp>
//#include <graphblas/linalg_utils.hpp>


namespace algorithms
{
    /**
     * @brief Compute the vertex betweenness centrality of all vertices in the
     *        given graph from a batch of src vertices.
     *
     * The betweenness centrality of a vertex measures the number of
     * times a vertex acts as a bridge along the shortest path between two
     * vertices.
     *
     * \f$C_b(v)=\sum\limits_{s \\neq v \in V}\sum\limits_{t \\neq v \in V}\\frac{\sigma_{st}(v)}{\sigma_{st}}\f$
     *
     * @param[in]  graph     The graph to compute the betweenness centrality of.
     * @param[in]  s         The set of source vertex indices from which to compute
     *                       BC contributions
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
        GraphBLAS::print_matrix(std::cerr, A, "Graph");

        // nsver = |s| (partition size)
        GraphBLAS::IndexType nsver(s.size());
        if (nsver == 0)
        {

            throw GraphBLAS::DimensionException();
        }

        std::cerr << "batch size (p): " << nsver << std::endl;

        using T = typename MatrixT::ScalarType;

        GraphBLAS::IndexType m(A.nrows());
        GraphBLAS::IndexType n(A.ncols());
        if (m != n)
        {
            throw GraphBLAS::DimensionException();
        }

        std::cerr << "num nodes (n): " << n << std::endl;

        // Placeholder for GrB_ALL where dimension size is nsver=|s|
        ///std::vector<GraphBLAS::IndexType> GrB_ALL_nsver;     // fill with sequence
        GraphBLAS::IndexArrayType GrB_ALL_nsver;     // fill with sequence
        GrB_ALL_nsver.reserve(nsver);
        for (GraphBLAS::IndexType idx = 0; idx < nsver; ++idx)
        {
            GrB_ALL_nsver.push_back(idx); // tilln
        }

        // Placeholder for GrB_ALL where dimension is n
        //std::vector<GraphBLAS::IndexType> GrB_ALL_n;     // fill with sequence
//        GraphBLAS::IndexArrayType GrB_ALL_n;
//        // @todo: Implement GrB_ALL_n support in extract
//        GrB_ALL_n.reserve(n);
//        for (GraphBLAS::IndexType idx = 0; idx < n; ++idx)
//        {
//            GrB_ALL_n.push_back(idx);
//        }

        // The current frontier for all BFS's (from all roots)
        // It is initialized to the out neighbors of the specified roots
        GraphBLAS::Matrix<int32_t> Frontier(nsver, n);     // F is nsver x n (rows)
        GraphBLAS::extract(Frontier,
                           GraphBLAS::NoMask(),
                           GraphBLAS::NoAccumulate(),
                           A,
                           s,
//                           GrB_ALL_n);
                           GraphBLAS::GrB_ALL);
        GraphBLAS::print_matrix(std::cerr, Frontier, "initial frontier");

        // NumSP holds number of shortest paths to a vertex from a given root
        // NumSP is initialized with the each starting root in 's':
        // NumSP[i,s[i]] = 1 where 0 <= i < nsver; implied zero elsewhere
        GraphBLAS::Matrix<int32_t> NumSP(nsver, n);

        // @TODO: We might NOT want GrB_ALL here for the rows, because we want a diagonal
        NumSP.build(GrB_ALL_nsver, s, std::vector<int32_t>(nsver, 1));
        GraphBLAS::print_matrix(std::cerr, NumSP, "initial NumSP");

        // ==================== BFS phase ====================
        // Placeholders for GraphBLAS operators

        // @todo:  Is this right?
        //GraphBLAS::PlusMonoid<int32_t>            Int32Add;
        //GraphBLAS::ArithmeticSemiring<int32_t>    Int32AddMul;

        // This is defined in the new space
        //GraphBLAS::math::Identity<bool, int32_t>  GrB_IDENTITY_BOOL;
        GrB_IDENTITY_BOOL identity_bool;

        std::vector<GraphBLAS::Matrix<bool>* > Sigmas;
        std::cerr << "======= START BFS phase ======" << std::endl;
        int32_t d = 0;

        // For testing purpose we only allow 10 iterations so it doesn't
        // get into an infinite loop
        while (Frontier.nvals() > 0 && d < 10)
        {
            std::cerr << "------- BFS iteration " << d << " --------" << std::endl;

            Sigmas.push_back(new GraphBLAS::Matrix<bool>(nsver, n));
            // Sigma[d] = (bool)
            GraphBLAS::apply(*(Sigmas[d]),
                             GraphBLAS::NoMask(),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Identity<bool>(),
                             Frontier );
            GraphBLAS::print_matrix(std::cerr, *Sigmas[d], "Sigma[d] = (bool)Frontier");

            // P = F + P
            GraphBLAS::eWiseAdd(NumSP,
                                GraphBLAS::NoMask(),
                                GraphBLAS::NoAccumulate(),
                                GraphBLAS::Plus<int32_t>(),
                                NumSP,
                                Frontier);
            GraphBLAS::print_matrix(std::cerr, NumSP, "NumSP");

            // F<!P> = F +.* A
            GraphBLAS::mxm(Frontier,                             // C
                           GraphBLAS::complement(NumSP),         // M
                           GraphBLAS::NoAccumulate(),            // accum
                           GraphBLAS::ArithmeticSemiring<int32_t>(), // op
                           Frontier,                             // A
                           A,                                    // B
                           true);                                // replace
            GraphBLAS::print_matrix(std::cerr, Frontier, "New frontier");

            ++d;
        }
        std::cerr << "======= END BFS phase =======" << std::endl;

        // ================== backprop phase ==================

        GraphBLAS::Matrix<float> NspInv(nsver, n);
        //                  A      C       Unary
        // GraphBLAS::apply(NumSP, NspInv, GrB_MINV_FP32);
        GraphBLAS::apply(NspInv,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::MultiplicativeInverse<float>(),
                         NumSP);
        GraphBLAS::print_matrix(std::cerr, NspInv, "(1 ./ P)");

        GraphBLAS::Matrix<float> BCu(nsver, n);
        GraphBLAS::assign_constant(BCu,
                          GraphBLAS::NoMask(),
                          GraphBLAS::NoAccumulate(),
                          1.0f,
//                          GrB_ALL_nsver,
//                          GrB_ALL_n);
                          GraphBLAS::GrB_ALL,
                          GraphBLAS::GrB_ALL);
        GraphBLAS::print_matrix(std::cerr, BCu, "U");


        GraphBLAS::Matrix<float> W(nsver, n);

        std::cerr << "======= BEGIN BACKPROP phase =======" << std::endl;
        for (int32_t i = d - 1; i > 0; --i)
        {
            std::cerr << "------- BACKPROP iteration " << i << " --------" << std::endl;

            // W<Sigma[i]> = (1 ./ P) .* U
            //                            A       B    C  M           true         op
            // GraphBLAS::ewisemultMasked(NspInv, BCu, W, *Sigmas[i], GrB_REPLACE, FP32Mul);
            GraphBLAS::eWiseMult(W,                            // C
                                 *Sigmas[i],                   // Mask
                                 GraphBLAS::NoAccumulate(),    // accum
                                 GraphBLAS::Times<float>(),    // op
                                 NspInv,                       // A
                                 BCu,                          // B
                                 true);                        // replace
            GraphBLAS::print_matrix(std::cerr, W, "W<Sigma[i]> = (1 ./ P) .* U");

            // W<Sigma[i-1]> = A +.* W
            //                      A  B                        C  M             semiring
            // graphblas::mxmMasked(W, graphblas::transpose(A), W, *Sigmas[i-1], FP32AddMul);
            // TODO: Probably need to transpose
            GraphBLAS::mxm(W,                                        // C
                           *Sigmas[i-1],                             // M
                           GraphBLAS::NoAccumulate(),                // accum
                           GraphBLAS::ArithmeticSemiring<float>(),   // op
                           W,                                        // A
                           GraphBLAS::transpose(A),                  // B
                           true);                                    // replace
            GraphBLAS::print_matrix(std::cerr, W, "W<Sigma[i-1]> = A +.* W");

            // U += W .* P
            //                      A  B      C    monoid   accum
            // graphblas::ewisemult(W, NumSP, BCu, FP32Mul, graphblas::math::Accum<float>());
            GraphBLAS::eWiseMult(BCu,                          // C
                                 GraphBLAS::NoMask(),          // Mask
                                 GraphBLAS::Plus<float>(),     // accum
                                 GraphBLAS::Times<float>(),    // op
                                 W,                            // A
                                 NumSP,                        // B
                                 false);                       // replace
            GraphBLAS::print_matrix(std::cerr, BCu, "U += W .* P");

            --d;
        }
        for (auto it = Sigmas.begin(); it != Sigmas.end(); ++it)
        {
            delete *it;
        }

        std::cerr << "======= END BACKPROP phase =======" << std::endl;
        GraphBLAS::print_matrix(std::cerr, BCu, "BC Updates");

        // @todo replace with GrB_All
        GraphBLAS::Vector<float, GraphBLAS::SparseTag> result(n);
        GraphBLAS::assign_constant(result,
                                   GraphBLAS::NoMask(),
                                   GraphBLAS::NoAccumulate(),
                                   nsver * -1.0f,
                                   //GrB_ALL_n);
                                   GraphBLAS::GrB_ALL);
        GraphBLAS::reduce(result,                          // W
                          GraphBLAS::NoMask(),             // Mask
                          GraphBLAS::Plus<float>(),        // Accum
                          GraphBLAS::Plus<float>(),        // Op
                          GraphBLAS::transpose(BCu),       // A, transpose make col reduce
                          true);                           // replace

        GraphBLAS::print_vector(std::cerr, result, "RESULT");
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


    /**
     * @brief Compute the vertex betweenness centrality of all vertices in the
     *        given graph from a batch of src vertices.
     *
     * The betweenness centrality of a vertex measures the number of
     * times a vertex acts as a bridge along the shortest path between two
     * vertices.
     *
     * \f$C_b(v)=\sum\limits_{s \\neq v \in V}\sum\limits_{t \\neq v \in V}\\frac{\sigma_{st}(v)}{\sigma_{st}}\f$
     *
     * @param[in]  graph     The graph to compute the betweenness centrality of.
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
        GraphBLAS::print_matrix(std::cerr, A, "Graph");

        // nsver = |s| (partition size)
        GraphBLAS::IndexType nsver(s.size());
        if (nsver == 0)
        {
            throw GraphBLAS::DimensionException();
        }

        std::cerr << "batch size (p): " << nsver << std::endl;

        using T = typename MatrixT::ScalarType;

        // GrB_Matrix_nrows(&n, A)
        GraphBLAS::IndexType m(A.nrows());
        GraphBLAS::IndexType n(A.ncols());
        if (m != n)
        {
            throw GraphBLAS::DimensionException();
        }

        std::cerr << "num nodes (n): " << n << std::endl;

        // Placeholder for GrB_ALL where dimension size is nsver=|s|
        std::vector<GraphBLAS::IndexType> GrB_ALL_nsver;     // fill with sequence
        GrB_ALL_nsver.reserve(nsver);
        for (GraphBLAS::IndexType idx = 0; idx < nsver; ++idx)
        {
            GrB_ALL_nsver.push_back(idx); // tilln
        }

        // Placeholder for GrB_ALL where dimension is n
        std::vector<GraphBLAS::IndexType> GrB_ALL_n;     // fill with sequence
        GrB_ALL_n.reserve(n);
        for (GraphBLAS::IndexType idx = 0; idx < n; ++idx)
        {
            GrB_ALL_n.push_back(idx);
        }

        // The current frontier for all BFS's (from all roots)
        // It is initialized to the out neighbors of the specified roots
        GraphBLAS::Matrix<int32_t> Frontier(nsver, n);     // F is nsver x n (rows)
        //GraphBLAS::extract(A, s, GrB_ALL_n, Frontier);
        GraphBLAS::extract(Frontier,
                           GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                           A, s, GrB_ALL_n);
        GraphBLAS::print_matrix(std::cerr, Frontier, "initial frontier");

        // NumSP holds number of shortest paths to a vertex from a given root
        // NumSP is initialized with the each starting root in 's':
        // NumSP[i,s[i]] = 1 where 0 <= i < nsver; implied zero elsewhere
        GraphBLAS::Matrix<int32_t> NumSP(nsver, n);
        NumSP.build(GrB_ALL_nsver, s, std::vector<int32_t>(nsver, 1));
        GraphBLAS::print_matrix(std::cerr, NumSP, "initial NumSP");

        // ==================== BFS phase ====================
        // Placeholders for GraphBLAS operators
        //GraphBLAS::PlusMonoid<int32_t>            Int32Add;
        //GraphBLAS::ArithmeticSemiring<int32_t>    Int32AddMul;

        std::vector<GraphBLAS::Matrix<bool>* > Sigmas;
        std::cerr << "======= START BFS phase ======" << std::endl;
        int32_t d = 0;
        while (Frontier.nvals() > 0)
        {
            std::cerr << "------- BFS iteration " << d << " --------" << std::endl;

            Sigmas.push_back(new GraphBLAS::Matrix<bool>(nsver, n));
            // Sigma[d] = (bool)F
            GraphBLAS::apply(*(Sigmas[d]),
                             GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                             GraphBLAS::Identity<int32_t, bool>(),
                             Frontier);
            GraphBLAS::print_matrix(std::cerr, *Sigmas[d],
                                    "Sigma[d] = (bool)Frontier");
            // P = F + P
            GraphBLAS::eWiseAdd(NumSP,
                                GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                GraphBLAS::Plus<int32_t>(),
                                NumSP, Frontier);
            GraphBLAS::print_matrix(std::cerr, NumSP, "NumSP");
            // F<!P> = F +.* A
            GraphBLAS::mxm(Frontier,
                           GraphBLAS::complement(NumSP),
                           GraphBLAS::NoAccumulate(),
                           GraphBLAS::ArithmeticSemiring<int32_t>(),
                           Frontier, A, true); // Replace?
            GraphBLAS::print_matrix(std::cerr, Frontier, "New frontier");

            ++d;
        }
        std::cerr << "======= END BFS phase =======" << std::endl;

        // ================== backprop phase ==================
        // Placeholders for GraphBLAS operators
        //GraphBLAS::PlusMonoid<float>            FP32Add;
        //GraphBLAS::TimesMonoid<float>           FP32Mul;
        //GraphBLAS::ArithmeticSemiring<float>    FP32AddMul;
        //GraphBLAS::MultiplicativeInverse<float> GrB_MINV_FP32;

        GraphBLAS::Matrix<float> NspInv(nsver, n);
        GraphBLAS::apply(NspInv,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::MultiplicativeInverse<float>(),
                         NumSP,
                         true);
        GraphBLAS::print_matrix(std::cerr, NspInv, "(1 ./ P)");

        //GraphBLAS::Matrix<float> BCu = GraphBLAS::fill<GraphBLAS::Matrix<float>>
        //    (1.0f, nsver, n);
        GraphBLAS::Matrix<float> BCu(nsver, n);
        GraphBLAS::assign_constant(BCu,
                                   GraphBLAS::NoMask(),
                                   GraphBLAS::NoAccumulate(),
                                   1.0f,
                                   GraphBLAS::GrB_ALL,
                                   GraphBLAS::GrB_ALL, true);

        GraphBLAS::print_matrix(std::cerr, BCu, "U");

        GraphBLAS::Matrix<float> W(nsver, n);

        std::cerr << "======= BEGIN BACKPROP phase =======" << std::endl;
        for (int32_t i = d - 1; i > 0; --i)
        {
            std::cerr << "------- BACKPROP iteration " << i << " --------" << std::endl;

            // W<Sigma[i]> = (1 ./ P) .* U
            //bool const GrB_REPLACE = true;
            //GraphBLAS::ewisemultMasked(NspInv, BCu, W, *Sigmas[i],
            //                           GrB_REPLACE, FP32Mul);
            GraphBLAS::eWiseMult(W,
                                 *Sigmas[i], GraphBLAS::NoAccumulate(),
                                 GraphBLAS::Times<float>(),
                                 NspInv, BCu,
                                 true);

            GraphBLAS::print_matrix(std::cerr, W, "W<Sigma[i]> = (1 ./ P) .* U");

            // W<Sigma[i-1]> = A +.* W
            //GraphBLAS::mxmMasked(W, GraphBLAS::transpose(A), W,
            //                     *Sigmas[i-1], FP32AddMul);
            GraphBLAS::mxm(W,
                           *Sigmas[i-1], GraphBLAS::NoAccumulate(),
                           GraphBLAS::ArithmeticSemiring<float>(),
                           W, GraphBLAS::transpose(A),
                           true);

            GraphBLAS::print_matrix(std::cerr, W, "W<Sigma[i-1]> = A +.* W");

            // U += W .* P
            //GraphBLAS::ewisemult(W, NumSP, BCu, FP32Mul,
            //                     GraphBLAS::math::Accum<float>());
            GraphBLAS::eWiseMult(BCu,
                                 GraphBLAS::NoMask(), GraphBLAS::Plus<float>(),
                                 GraphBLAS::Times<float>(),
                                 W, NumSP); // Replace == false?

            GraphBLAS::print_matrix(std::cerr, BCu, "U += W .* P");

            --d;
        }

        for (auto it = Sigmas.begin(); it != Sigmas.end(); ++it)
        {
            delete *it;
        }
        std::cerr << "======= END BACKPROP phase =======" << std::endl;
        GraphBLAS::print_matrix(std::cerr, BCu, "BC Updates");

        //GraphBLAS::Matrix<float> result =
        //    GraphBLAS::fill<GraphBLAS::Matrix<float>>(nsver * -1.0F, 1, n);
        GraphBLAS::Vector<float> result(n);
        GraphBLAS::assign_constant(result,
                                   GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                   -1.0f*nsver,
                                   GraphBLAS::GrB_ALL);

        //GraphBLAS::col_reduce(BCu, result, FP32Add,
        //                      GraphBLAS::math::Accum<float>());
        GraphBLAS::reduce(result,
                          GraphBLAS::NoMask(), GraphBLAS::Plus<float>(),
                          GraphBLAS::Plus<float>(),
                          GraphBLAS::transpose(BCu)); // replace = dont care
        GraphBLAS::print_vector(std::cerr, result, "RESULT");

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

    /**
     * @brief Compute the vertex betweenness centrality of all vertices in the
     *        given graph from a batch of src vertices.
     *
     * @note "alt" in function name means first iteration of BFS is replaced with
     *       a call to extract.
     *
     * The betweenness centrality of a vertex measures the number of
     * times a vertex acts as a bridge along the shortest path between two
     * vertices.
     *
     * \f$C_b(v)=\sum\limits_{s \\neq v \in V}\sum\limits_{t \\neq v \in V}\\frac{\sigma_{st}(v)}{\sigma_{st}}\f$
     *
     * @param[in]  graph     The graph to compute the betweenness centrality of.
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
        GraphBLAS::print_matrix(std::cerr, A, "Graph");

        // nsver = |s| (partition size)
        GraphBLAS::IndexType nsver(s.size());
        if (nsver == 0)
        {
            throw GraphBLAS::DimensionException();
        }

        std::cerr << "batch size (p): " << nsver << std::endl;

        using T = typename MatrixT::ScalarType;

        GraphBLAS::IndexType m(A.nrows());
        GraphBLAS::IndexType n(A.ncols());

        // GrB_Matrix_nrows(&n, A)
        //A.get_shape(m, n);
        if (m != n)
        {
            throw GraphBLAS::DimensionException();
        }

        std::cerr << "num nodes (n): " << n << std::endl;

        // Placeholder for GrB_ALL where dimension size is nsver=|s|
        std::vector<GraphBLAS::IndexType> GrB_ALL_nsver;     // fill with sequence
        GrB_ALL_nsver.reserve(nsver);
        for (GraphBLAS::IndexType idx = 0; idx < nsver; ++idx)
        {
            GrB_ALL_nsver.push_back(idx); // tilln
        }

        // Placeholder for GrB_ALL where dimension is n
        std::vector<GraphBLAS::IndexType> GrB_ALL_n;     // fill with sequence
        GrB_ALL_n.reserve(n);
        for (GraphBLAS::IndexType idx = 0; idx < n; ++idx)
        {
            GrB_ALL_n.push_back(idx);
        }

        // The current frontier for all BFS's (from all roots)
        // It is initialized to the out neighbors of the specified roots
        GraphBLAS::Matrix<int32_t> Frontier(n, nsver);         // F is n x nsver
        //GraphBLAS::extract(GraphBLAS::transpose(A), GrB_ALL_n, s, Frontier);
        GraphBLAS::extract(Frontier,
                           GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                           GraphBLAS::transpose(A),
                           GrB_ALL_n, s);
        GraphBLAS::print_matrix(std::cerr, Frontier, "initial frontier");

        // NumSP holds number of shortest paths to a vertex from a given root
        // NumSP is initialized with the each starting root in 's':
        // NumSP[s[i],i] = 1 where 0 <= i < nsver; implied zero elsewhere
        GraphBLAS::Matrix<int32_t> NumSP(n, nsver);
        NumSP.build(s, GrB_ALL_nsver, std::vector<int32_t>(nsver, 1));
        GraphBLAS::print_matrix(std::cerr, NumSP, "initial NumSP");

        // ==================== BFS phase ====================
        // Placeholders for GraphBLAS operators
        //GraphBLAS::PlusMonoid<int32_t>            Int32Add;
        //GraphBLAS::ArithmeticSemiring<int32_t>    Int32AddMul;

        std::vector<GraphBLAS::Matrix<bool>* > Sigmas;
        std::cerr << "======= START BFS phase ======" << std::endl;
        int32_t d = 0;
        while (Frontier.nvals() > 0)
        {
            std::cerr << "------- BFS iteration " << d << " --------" << std::endl;

            Sigmas.push_back(new GraphBLAS::Matrix<bool>(n, nsver));
            // Sigma[d] = (bool)F
            //GraphBLAS::apply(*(Sigmas[d]),
            //                 GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
            //                 IDENTITY_INT2BOOL, Frontier);
            GraphBLAS::apply(*(Sigmas[d]),
                             GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                             GraphBLAS::Identity<int32_t, bool>(),
                             Frontier);
            GraphBLAS::print_matrix(std::cerr, *Sigmas[d], "Sigma[d] = (bool)Frontier");

            // P = F + P
            //GraphBLAS::ewiseadd(NumSP, Frontier, NumSP, Int32Add);
            GraphBLAS::eWiseAdd(NumSP,
                                GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                GraphBLAS::Plus<int32_t>(),
                                NumSP, Frontier);
            GraphBLAS::print_matrix(std::cerr, NumSP, "NumSP");

            // F<!P> = A' +.* F
            //GraphBLAS::mxmMasked(GraphBLAS::transpose(A), Frontier,
            //                     Frontier,
            //                     GraphBLAS::negate(NumSP, Int32AddMul),
            //                     Int32AddMul);
            GraphBLAS::mxm(Frontier,
                           GraphBLAS::complement(NumSP),
                           GraphBLAS::NoAccumulate(),
                           GraphBLAS::ArithmeticSemiring<int32_t>(),
                           GraphBLAS::transpose(A), Frontier, true); // Replace?
            GraphBLAS::print_matrix(std::cerr, Frontier, "New frontier");

            ++d;
        }
        std::cerr << "======= END BFS phase =======" << std::endl;

        // ================== backprop phase ==================
        // Placeholders for GraphBLAS operators
        //GraphBLAS::PlusMonoid<float>           FP32Add;
        //GraphBLAS::TimesMonoid<float>          FP32Mul;
        //GraphBLAS::ArithmeticSemiring<float>   FP32AddMul;
        //GraphBLAS::math::Inverse<float>        GrB_MINV_FP32;

        GraphBLAS::Matrix<float> NspInv(n, nsver);
        //GraphBLAS::apply(NumSP, NspInv, GrB_MINV_FP32);
        GraphBLAS::apply(NspInv,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::MultiplicativeInverse<float>(),
                         NumSP,
                         true);
        GraphBLAS::print_matrix(std::cerr, NspInv, "(1 ./ P)");

        //GraphBLAS::Matrix<float> BCu = GraphBLAS::fill<GraphBLAS::Matrix<float>>
        //    (1.0f, n, nsver);
        GraphBLAS::Matrix<float> BCu(n, nsver);
        GraphBLAS::assign_constant(BCu,
                                   GraphBLAS::NoMask(),
                                   GraphBLAS::NoAccumulate(),
                                   1.0f,
                                   GraphBLAS::GrB_ALL,
                                   GraphBLAS::GrB_ALL, true);

        GraphBLAS::print_matrix(std::cerr, BCu, "U");

        GraphBLAS::Matrix<float> W(n, nsver);

        std::cerr << "======= BEGIN BACKPROP phase =======" << std::endl;
        for (int32_t i = d - 1; i > 0; --i)
        {
            std::cerr << "------- BACKPROP iteration " << i << " --------" << std::endl;

            // W<Sigma[i]> = (1 ./ P) .* U
            //bool const GrB_REPLACE = true;
            //GraphBLAS::ewisemultMasked(NspInv, BCu, W, *Sigmas[i],
            //                          GrB_REPLACE, FP32Mul);
            GraphBLAS::eWiseMult(W,
                                 *Sigmas[i], GraphBLAS::NoAccumulate(),
                                 GraphBLAS::Times<float>(),
                                 NspInv, BCu,
                                 true);
            GraphBLAS::print_matrix(std::cerr, W, "W<Sigma[i]> = (1 ./ P) .* U");

            // W<Sigma[i-1]> = A +.* W
            //GraphBLAS::mxmMasked(A, W, W, *Sigmas[i-1], FP32AddMul);
            GraphBLAS::mxm(W,
                           *Sigmas[i-1], GraphBLAS::NoAccumulate(),
                           GraphBLAS::ArithmeticSemiring<float>(),
                           A, W,
                           true);
            GraphBLAS::print_matrix(std::cerr, W, "W<Sigma[i-1]> = A +.* W");

            // U += W .* P
            //GraphBLAS::ewisemult(W, NumSP, BCu, FP32Mul,
            //                     GraphBLAS::math::Accum<float>());
            GraphBLAS::eWiseMult(BCu,
                                 GraphBLAS::NoMask(), GraphBLAS::Plus<float>(),
                                 GraphBLAS::Times<float>(),
                                 W, NumSP); // Replace == false?
            GraphBLAS::print_matrix(std::cerr, BCu, "U += W .* P");

            --d;
        }

        for (auto it = Sigmas.begin(); it != Sigmas.end(); ++it)
        {
            delete *it;
        }
        std::cerr << "======= END BACKPROP phase =======" << std::endl;
        GraphBLAS::print_matrix(std::cerr, BCu, "BC Updates");

        //GraphBLAS::Matrix<float> result =
        //    GraphBLAS::fill<GraphBLAS::Matrix<float>>(nsver * -1.0F, n, 1);
        GraphBLAS::Vector<float> result(n);
        GraphBLAS::assign_constant(result,
                                   GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                   -1.0f*nsver,
                                   GraphBLAS::GrB_ALL);

        //GraphBLAS::row_reduce(BCu, result, FP32Add,
        //                      GraphBLAS::math::Accum<float>());
        GraphBLAS::reduce(result,
                          GraphBLAS::NoMask(), GraphBLAS::Plus<float>(),
                          GraphBLAS::Plus<float>(),
                          BCu); // replace = dont care
        GraphBLAS::print_vector(std::cerr, result, "RESULT");

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

    /**
     * @brief Compute the vertex betweenness centrality of all vertices in the
     *        given graph from a batch of src vertices.
     *
     * The betweenness centrality of a vertex measures the number of
     * times a vertex acts as a bridge along the shortest path between two
     * vertices.
     *
     * \f$C_b(v)=\sum\limits_{s \\neq v \in V}\sum\limits_{t \\neq v \in V}\\frac{\sigma_{st}(v)}{\sigma_{st}}\f$
     *
     * @param[in]  graph     The graph to compute the betweenness centrality of.
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
        GraphBLAS::print_matrix(std::cerr, A, "Graph");

        // nsver = |s| (partition size)
        GraphBLAS::IndexType nsver(s.size());
        if (nsver == 0)
        {
            throw GraphBLAS::DimensionException();
        }

        std::cerr << "batch size (p): " << nsver << std::endl;

        using T = typename MatrixT::ScalarType;

        GraphBLAS::IndexType m(A.nrows());
        GraphBLAS::IndexType n(A.ncols());
        if (m != n)
        {
            throw GraphBLAS::DimensionException();
        }

        std::cerr << "num nodes (n): " << n << std::endl;

        // Placeholder for GrB_ALL where dimension size is nsver=|s|
        std::vector<GraphBLAS::IndexType> GrB_ALL_nsver;     // fill with sequence
        GrB_ALL_nsver.reserve(nsver);
        for (GraphBLAS::IndexType idx = 0; idx < nsver; ++idx)
        {
            GrB_ALL_nsver.push_back(idx); // tilln
        }

        // Placeholder for GrB_ALL where dimension is n
        std::vector<GraphBLAS::IndexType> GrB_ALL_n;     // fill with sequence
        GrB_ALL_n.reserve(n);
        for (GraphBLAS::IndexType idx = 0; idx < n; ++idx)
        {
            GrB_ALL_n.push_back(idx);
        }

        // The current frontier for all BFS's (from all roots)
        // It is initialized with the each starting root in 's':
        // F[s[i],i] = 1 where 0 <= i < nsver; implied zero elsewhere
        GraphBLAS::Matrix<int32_t> Frontier(n, nsver);         // F is n x nsver
        Frontier.build(s, GrB_ALL_nsver, std::vector<int32_t>(nsver, 1));
        GraphBLAS::print_matrix(std::cerr, Frontier, "initial frontier");

        // NumSP holds number of shortest paths to a vertex from a given root
        // NumSP is initialized to a copy of Frontier. NumSP is n x nsver
        GraphBLAS::Matrix<int32_t> NumSP(Frontier);
        GraphBLAS::print_matrix(std::cerr, NumSP, "initial NumSP");

        // ==================== BFS phase ====================
        // Placeholders for GraphBLAS operators
        //GraphBLAS::PlusMonoid<int32_t>            Int32Add;
        //GraphBLAS::ArithmeticSemiring<int32_t>    Int32AddMul;
        //GraphBLAS::math::Identity<bool, int32_t>  GrB_IDENTITY_BOOL;

        std::vector<GraphBLAS::Matrix<bool>* > Sigmas;
        std::cerr << "======= START BFS phase ======" << std::endl;
        int32_t d = 0;
        while (Frontier.nvals() > 0)
        {
            std::cerr << "------- BFS iteration " << d << " --------" << std::endl;
            // F<!P> = A' +.* F
            //GraphBLAS::mxmMasked(GraphBLAS::transpose(A), Frontier,
            //                     Frontier,
            //                     GraphBLAS::negate(NumSP, Int32AddMul),
            //                     Int32AddMul);
            GraphBLAS::mxm(Frontier,
                           GraphBLAS::complement(NumSP), GraphBLAS::NoAccumulate(),
                           GraphBLAS::ArithmeticSemiring<int32_t>(),
                           GraphBLAS::transpose(A), Frontier, true);
            GraphBLAS::print_matrix(std::cerr, Frontier, "New frontier");

            // Sigma[d] = (bool)F
            Sigmas.push_back(new GraphBLAS::Matrix<bool>(n, nsver));
            //GraphBLAS::apply(Frontier, *(Sigmas[d]),
            //                 GrB_IDENTITY_BOOL);
            GraphBLAS::apply(*(Sigmas[d]),
                             GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                             GraphBLAS::Identity<int32_t, bool>(),
                             Frontier);
            GraphBLAS::print_matrix(std::cerr, *Sigmas[d], "Sigma[d] = (bool)Frontier");

            // P = F + P
            //GraphBLAS::ewiseadd(NumSP, Frontier, NumSP, Int32Add);
            GraphBLAS::eWiseAdd(NumSP,
                                GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                GraphBLAS::Plus<int32_t>(),
                                NumSP, Frontier);
            GraphBLAS::print_matrix(std::cerr, NumSP, "NumSP");

            ++d;
        }
        std::cerr << "======= END BFS phase =======" << std::endl;

        // ================== backprop phase ==================
        // Placeholders for GraphBLAS operators
        //GraphBLAS::PlusMonoid<float>           FP32Add;
        //GraphBLAS::TimesMonoid<float>          FP32Mul;
        //GraphBLAS::ArithmeticSemiring<float>   FP32AddMul;
        //GraphBLAS::math::Inverse<float>        GrB_MINV_FP32;

        GraphBLAS::Matrix<float> NspInv(n, nsver);
        //GraphBLAS::apply(NumSP, NspInv, GrB_MINV_FP32);
        GraphBLAS::apply(NspInv,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::MultiplicativeInverse<float>(),
                         NumSP,
                         true);
        GraphBLAS::print_matrix(std::cerr, NspInv, "(1 ./ P)");

        //GraphBLAS::Matrix<float> BCu = GraphBLAS::fill<GraphBLAS::Matrix<float>>
        //    (1.0f, n, nsver);
        GraphBLAS::Matrix<float> BCu(n, nsver);
        GraphBLAS::assign_constant(BCu,
                                   GraphBLAS::NoMask(),
                                   GraphBLAS::NoAccumulate(),
                                   1.0f,
                                   GraphBLAS::GrB_ALL,
                                   GraphBLAS::GrB_ALL, true);

        GraphBLAS::print_matrix(std::cerr, BCu, "U");

        GraphBLAS::Matrix<float> W(n, nsver);

        std::cerr << "======= BEGIN BACKPROP phase =======" << std::endl;
        for (int32_t i = d - 1; i > 0; --i)
        {
            std::cerr << "------- BACKPROP iteration " << i << " --------" << std::endl;

            // W<Sigma[i]> = (1 ./ P) .* U
            //bool const GrB_REPLACE = true;
            //GraphBLAS::ewisemultMasked(NspInv, BCu, W, *Sigmas[i],
            //                           GrB_REPLACE, FP32Mul);
            GraphBLAS::eWiseMult(W,
                                 *Sigmas[i], GraphBLAS::NoAccumulate(),
                                 GraphBLAS::Times<float>(),
                                 NspInv, BCu,
                                 true);
            GraphBLAS::print_matrix(std::cerr, W, "W<Sigma[i]> = (1 ./ P) .* U");

            // W<Sigma[i-1]> = A +.* W
            //GraphBLAS::mxmMasked(A, W, W, *Sigmas[i-1], FP32AddMul);
            GraphBLAS::mxm(W,
                           *Sigmas[i-1], GraphBLAS::NoAccumulate(),
                           GraphBLAS::ArithmeticSemiring<float>(),
                           A, W,
                           true);
            GraphBLAS::print_matrix(std::cerr, W, "W<Sigma[i-1]> = A +.* W");

            // U += W .* P
            //GraphBLAS::ewisemult(W, NumSP, BCu, FP32Mul,
            //                     GraphBLAS::math::Accum<float>());
            GraphBLAS::eWiseMult(BCu,
                                 GraphBLAS::NoMask(), GraphBLAS::Plus<float>(),
                                 GraphBLAS::Times<float>(),
                                 W, NumSP); // Replace == false?
            GraphBLAS::print_matrix(std::cerr, BCu, "U += W .* P");

            --d;
        }

        for (auto it = Sigmas.begin(); it != Sigmas.end(); ++it)
        {
            delete *it;
        }
        std::cerr << "======= END BACKPROP phase =======" << std::endl;
        GraphBLAS::print_matrix(std::cerr, BCu, "BC Updates");

        //GraphBLAS::Matrix<float> result =
        //    GraphBLAS::fill<GraphBLAS::Matrix<float>>(nsver * -1.0F, n, 1);
        GraphBLAS::Vector<float> result(n);
        GraphBLAS::assign_constant(result,
                                   GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                   -1.0f*nsver,
                                   GraphBLAS::GrB_ALL);

        //GraphBLAS::row_reduce(BCu, result, FP32Add,
        //                      GraphBLAS::math::Accum<float>());
        GraphBLAS::reduce(result,
                          GraphBLAS::NoMask(), GraphBLAS::Plus<float>(),
                          GraphBLAS::Plus<float>(),
                          BCu); // replace = dont care
        GraphBLAS::print_vector(std::cerr, result, "RESULT");

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



    /**
     * @brief Compute the vertex betweenness centrality of all vertices in the
     *        given graph from a batch of src vertices.
     *
     * The betweenness centrality of a vertex measures the number of
     * times a vertex acts as a bridge along the shortest path between two
     * vertices.
     *
     * \f$C_b(v)=\sum\limits_{s \\neq v \in V}\sum\limits_{t \\neq v \in V}\\frac{\sigma_{st}(v)}{\sigma_{st}}\f$
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
        GraphBLAS::print_matrix(std::cerr, graph, "Graph");

        // p = nsver = |src_nodes| (partition size)
        GraphBLAS::IndexType p(src_nodes.size());
        if (p == 0)
        {
            throw GraphBLAS::DimensionException();
        }

        std::cerr << "batch size (p): " << p << std::endl;

        using T = typename MatrixT::ScalarType;
        GraphBLAS::IndexType N, num_cols;

        // GrB_Matrix_nrows(&N, graph)
        graph.get_shape(N, num_cols);
        if (N != num_cols)
        {
            throw GraphBLAS::DimensionException();
        }

        std::cerr << "num nodes (N): " << N << std::endl;

        // Placeholders for GraphBLAS operators
        //GraphBLAS::PlusMonoid<int32_t>         GrB_Int32Add;
        //GraphBLAS::ArithmeticSemiring<int32_t> GrB_Int32AddMul;
        //GraphBLAS::PlusMonoid<double>          GrB_Flt64Add;
        //GraphBLAS::TimesMonoid<double>         GrB_Flt64Mul;
        //GraphBLAS::DivMonoid<double>           GrB_Flt64Div;
        //GraphBLAS::ArithmeticSemiring<double>  GrB_Flt64AddMul;

        // Placeholder for GrB_ALL where dimension is p=|nsver|
        std::vector<GraphBLAS::IndexType> GrB_ALL_p;     // fill with sequence
        GrB_ALL_p.reserve(p);
        for (GraphBLAS::IndexType idx = 0; idx < p; ++idx)
        {
            GrB_ALL_p.push_back(idx);
        }

        // P holds number of shortest paths to a vertex from a given root
        // P[i,src_nodes[i]] = 1 where 0 <= i < nsver; implied zero elsewher
        GraphBLAS::Matrix<int32_t> P(p, N);         // P is pxN
        std::vector<int32_t> ones(p, 1);
        P.build(GrB_ALL_p, src_nodes, ones);
        GraphBLAS::print_matrix(std::cerr, P, "initial P");

        // Placeholder for GrB_ALL where dimension is N
        std::vector<GraphBLAS::IndexType> GrB_ALL_N;     // fill with sequence
        GrB_ALL_N.reserve(N);
        for (GraphBLAS::IndexType idx = 0; idx < N; ++idx)
        {
            GrB_ALL_N.push_back(idx);
        }

        // F is the current frontier for all BFS's (from all roots)
        // It is initialized with the out neighbors for each root
        GraphBLAS::Matrix<int32_t> F(p, N);         // F is pxN
        //GraphBLAS::extract(graph, src_nodes, GrB_ALL_N, F);
        GraphBLAS::extract(F,
                           GraphBLAS::NoMask(),
                           GraphBLAS::NoAccumulate(),
                           graph,
                           src_nodes,
//                         GrB_ALL_N);
                           GraphBLAS::GrB_ALL);
        GraphBLAS::print_matrix(std::cerr, F, "initial frontier");

        // ==================== BFS phase ====================
        std::vector<GraphBLAS::Matrix<int32_t>* > Sigmas;
        std::cerr << "======= START BFS phase ======" << std::endl;
        int32_t d = 0;
        while (F.nvals() > 0)
        {
            std::cerr << "------- BFS iteration " << d << " --------" << std::endl;
            Sigmas.push_back(new GraphBLAS::Matrix<int32_t>(F));  // Sigma[d] = F
            GraphBLAS::print_matrix(std::cerr, *Sigmas[d], "Sigma[d] = F");

            //GraphBLAS::ewiseadd(P, F, P, GrB_Int32Add);           // P = F + P
            // P = F + P
            GraphBLAS::eWiseAdd(P,
                                GraphBLAS::NoMask(),
                                GraphBLAS::NoAccumulate(),
                                GraphBLAS::Plus<int32_t>(),
                                F, P);
            GraphBLAS::print_matrix(std::cerr, P, "P");

            //GraphBLAS::mxmMasked(F, graph, F,                     // F<!P> = F +.* A
            //                     GraphBLAS::negate(P, GrB_Int32AddMul),
            //                     GrB_Int32AddMul);

            // F<!P> = F +.* A
            GraphBLAS::mxm(F,                                    // C
                           GraphBLAS::complement(P),             // M
                           GraphBLAS::NoAccumulate(),            // accum
                           GraphBLAS::ArithmeticSemiring<int32_t>(), // op
                           F,                                    // A
                           graph,                                // B
                           true);                                // replace
            GraphBLAS::print_matrix(std::cerr, F, "New frontier (F)");
            ++d;
        }
        std::cerr << "======= END BFS phase =======" << std::endl;

        // ================== backprop phase ==================
        GraphBLAS::Matrix<double> U(p, N);        // U is pxN
        //GraphBLAS::Matrix<double> Ones =
        //    GraphBLAS::fill<GraphBLAS::Matrix<double>>(
        //        (double)1.0, p, N);  // pxN filled with 1.0
        GraphBLAS::Matrix<double> Ones(p, N);
        GraphBLAS::assign_constant(Ones,
                                   GraphBLAS::NoMask(),
                                   GraphBLAS::NoAccumulate(),
                                   1.0f,
                                   GraphBLAS::GrB_ALL,
                                   GraphBLAS::GrB_ALL);
        GraphBLAS::print_matrix(std::cerr, Ones, "Ones");

        GraphBLAS::Matrix<double> W(p, N);
        GraphBLAS::Matrix<double> tmp1(p, N);
        GraphBLAS::Matrix<double> tmp2(p, N);

        std::cerr << "======= BEGIN BACKPROP phase =======" << std::endl;
        while (d >= 2)
        {
            std::cerr << "------- BACKPROP iteration " << d << " --------" << std::endl;
            // W = Sigma[d-1].*(1 + U)./P
            //GraphBLAS::ewiseadd(Ones, U, tmp1, GrB_Flt64Add);
            GraphBLAS::eWiseAdd(tmp1,                         // C
                                GraphBLAS::NoMask(),          // Mask
                                GraphBLAS::NoAccumulate(),    // accum
                                GraphBLAS::Plus<double>(),    // op
                                Ones,                         // A
                                U,                            // B
                                true);                        // replace
            GraphBLAS::print_matrix(std::cerr, tmp1, "(1 + U)");
            //GraphBLAS::ewisemult(*Sigmas[d-1], tmp1, tmp2, GrB_Flt64Mul);
            GraphBLAS::eWiseMult(tmp2,                         // C
                                 GraphBLAS::NoMask(),          // Mask
                                 GraphBLAS::NoAccumulate(),    // accum
                                 GraphBLAS::Times<double>(),   // op
                                 *Sigmas[d-1],                 // A
                                 tmp1,                         // B
                                 true);                        // replace
            GraphBLAS::print_matrix(std::cerr, tmp2, "Sigma[d-1] .* (1 + U)");
            //GraphBLAS::ewisemult(tmp2, P, W, GrB_Flt64Div);
            GraphBLAS::eWiseMult(W,                            // C
                                 GraphBLAS::NoMask(),          // Mask
                                 GraphBLAS::NoAccumulate(),    // accum
                                 GraphBLAS::Div<double>(),     // op
                                 tmp2,                         // A
                                 P,                            // B
                                 true);                        // replace
            GraphBLAS::print_matrix(std::cerr, W, "W = Sigma[d-1] .* (1 + U) ./ P");

            // W = (A +.* W')' = W +.* A'
            //GraphBLAS::mxm(W, GraphBLAS::transpose(graph), W, GrB_Flt64AddMul);
            GraphBLAS::mxm(W,                                        // C
                           GraphBLAS::NoMask(),                      // M
                           GraphBLAS::NoAccumulate(),                // accum
                           GraphBLAS::ArithmeticSemiring<double>(),  // op
                           W,                                        // A
                           GraphBLAS::transpose(graph),              // B
                           true);                                    // replace
            GraphBLAS::print_matrix(std::cerr, W, "W = W A'");

            // W = W .* Sigma[d-2] .* P
            //GraphBLAS::ewisemult(W, *Sigmas[d-2], W, GrB_Flt64Mul);
            GraphBLAS::eWiseMult(W,                            // C
                                 GraphBLAS::NoMask(),          // Mask
                                 GraphBLAS::NoAccumulate(),    // accum
                                 GraphBLAS::Times<double>(),   // op
                                 W,                            // A
                                 *Sigmas[d-2],                 // B
                                 false);                       // replace
            //GraphBLAS::ewisemult(W, P, W, GrB_Flt64Mul);
            GraphBLAS::eWiseMult(W,                            // C
                                 GraphBLAS::NoMask(),          // Mask
                                 GraphBLAS::NoAccumulate(),    // accum
                                 GraphBLAS::Times<double>(),   // op
                                 W,                            // A
                                 P,                            // B
                                 false);                       // replace
            GraphBLAS::print_matrix(std::cerr, W, "W = W .* P");

            // U = U + W
            //GraphBLAS::ewiseadd(U, W, U, GrB_Flt64Add);
            GraphBLAS::eWiseAdd(U,                            // C
                                GraphBLAS::NoMask(),          // Mask
                                GraphBLAS::NoAccumulate(),    // accum
                                GraphBLAS::Plus<double>(),    // op
                                U,                            // A
                                W,                            // B
                                true);                        // replace
            GraphBLAS::print_matrix(std::cerr, U, "U += W");

            --d;
        }
        for (auto it = Sigmas.begin(); it != Sigmas.end(); ++it)
        {
            delete *it;
        }

        std::cerr << "======= END BACKPROP phase =======" << std::endl;
        GraphBLAS::print_matrix(std::cerr, U, "U");
        GraphBLAS::Vector<double> result(N);
        //GraphBLAS::col_reduce(U, result, GrB_Flt64Add);
        GraphBLAS::reduce(result,                          // W
                          GraphBLAS::NoMask(),             // Mask
                          GraphBLAS::NoAccumulate(),       // Accum
                          GraphBLAS::Plus<double>(),       // Op
                          GraphBLAS::transpose(U),         // A, transpose make col reduce
                          true);                           // replace
        GraphBLAS::print_matrix(std::cerr, result, "RESULT");

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

    /**
     * @brief Compute the vertex betweenness centrality of all vertices in the
     *        given graph ("for all" one at a time).
     *
     * The betweenness centrality of a vertex measures the number of
     * times a vertex acts as a bridge along the shortest path between two
     * vertices.
     *
     * \f$C_b(v)=\sum\limits_{s \\neq v \in V}\sum\limits_{t \\neq v \in V}\\frac{\sigma_{st}(v)}{\sigma_{st}}\f$
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
                               GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                               GraphBLAS::transpose(graph),
                               GraphBLAS::GrB_ALL, i, true);

            VectorT n_shortest_paths(num_nodes);
            n_shortest_paths.setElement(i, static_cast<T>(1));

            GraphBLAS::IndexType depth = 0;
            while (frontier.nvals() > 0)
            {
                search.push_back(new GraphBLAS::Vector<bool>(num_nodes));

                GraphBLAS::apply(*(search[depth]),
                                 GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                 GraphBLAS::Identity<T, bool>(),
                                 frontier);

                GraphBLAS::eWiseAdd(n_shortest_paths,
                                    GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                    GraphBLAS::Plus<T>(),
                                    n_shortest_paths, frontier);

                GraphBLAS::vxm(frontier,
                               GraphBLAS::complement(n_shortest_paths),
                               GraphBLAS::NoAccumulate(),
                               GraphBLAS::ArithmeticSemiring<T>(),
                               frontier, graph, true);

                ++depth;
            }

            VectorT n_shortest_paths_inv(num_nodes);
            GraphBLAS::apply(n_shortest_paths_inv,
                             GraphBLAS::NoMask(),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::MultiplicativeInverse<T>(),
                             n_shortest_paths);

            VectorT update(num_nodes);
            GraphBLAS::assign_constant(update,
                                       GraphBLAS::NoMask(),
                                       GraphBLAS::NoAccumulate(),
                                       1, GraphBLAS::GrB_ALL);

            VectorT weights(num_nodes);
            for (int32_t idx = depth - 1; idx > 0; --idx)
            {
                --depth;

                GraphBLAS::eWiseMult(weights,
                                     *(search[idx]), GraphBLAS::NoAccumulate(),
                                     GraphBLAS::Times<T>(),
                                     n_shortest_paths_inv, update,
                                     true);

                GraphBLAS::mxv(weights,
                               *(search[idx - 1]), GraphBLAS::NoAccumulate(),
                               GraphBLAS::ArithmeticSemiring<T>(),
                               graph, weights, true);

                GraphBLAS::eWiseMult(update,
                                     GraphBLAS::NoMask(), GraphBLAS::Plus<T>(),
                                     GraphBLAS::Times<T>(),
                                     weights, n_shortest_paths, true);
            }

            for (auto it = search.begin(); it != search.end(); ++it)
            {
                delete *it;
            }

            GraphBLAS::eWiseAdd(bc_vec,
                                GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                GraphBLAS::Plus<T>(),
                                bc_vec, update);

        }

        GraphBLAS::Vector<T> bias(num_nodes);
        GraphBLAS::assign_constant(bias,
                                   GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                   static_cast<T>(num_nodes),
                                   GraphBLAS::GrB_ALL);
        GraphBLAS::eWiseAdd(bc_vec,
                            GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                            GraphBLAS::Minus<T>(),
                            bc_vec, bias, true);

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
     * \f$C_b(v)=\sum\limits_{s \\neq v \in V}\sum\limits_{t \\neq v \in V}\\frac{\sigma_{st}(v)}{\sigma_{st}}\f$
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
                               GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                               GraphBLAS::transpose(graph),
                               GraphBLAS::GrB_ALL, root);

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
                                 GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                 GraphBLAS::Identity<int32_t, bool>(),
                                 frontier);

                // n_shortest_paths += frontier
                GraphBLAS::eWiseAdd(n_shortest_paths,
                                    GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                    GraphBLAS::Plus<T>(),
                                    n_shortest_paths, frontier, true);

                // frontier<!n_shortest_paths>{z} = frontier *.+ graph
                GraphBLAS::vxm(frontier,
                               GraphBLAS::complement(n_shortest_paths),
                               GraphBLAS::NoAccumulate(),
                               GraphBLAS::ArithmeticSemiring<T>(),
                               frontier, graph, true);
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
                                   GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                   GraphBLAS::ArithmeticSemiring<T>(),
                                   graph, wmat);
                }

                // weights<search[depth-1]>{z} = n_shortest_paths
                GraphBLAS::apply(weights,
                                 *search[depth-1], GraphBLAS::NoAccumulate(),
                                 GraphBLAS::Identity<T>(),
                                 n_shortest_paths, true);

                {
                    // update = diag(weights) *.+ update
                    std::vector<T> w(weights.nvals());
                    GraphBLAS::IndexArrayType widx(weights.nvals());
                    weights.extractTuples(widx, w);
                    MatrixT wmat(num_nodes, num_nodes);
                    wmat.build(widx, widx, w);
                    GraphBLAS::mxm(update,
                                   GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                   GraphBLAS::ArithmeticSemiring<T>(),
                                   wmat, update);
                }

                // score += update
                GraphBLAS::eWiseAdd(score,
                                    GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                    GraphBLAS::Plus<T>(),
                                    score, update);

                // flow = update +.
                GraphBLAS::reduce(flow,
                                  GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                  GraphBLAS::Plus<T>(),
                                  update);

                depth = depth - 1;
            }
        }
        return score;
    }

} // algorithms

#endif // METRICS_HPP
