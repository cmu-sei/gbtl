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
#include <graphblas/linalg_utils.hpp>


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
    vertex_betweenness_centrality_batch_alt_trans(
        MatrixT const                   &A,
        graphblas::IndexArrayType const &s)
    {
        graphblas::print_matrix(std::cerr, A, "Graph");

        // nsver = |s| (partition size)
        graphblas::IndexType nsver(s.size());
        if (nsver == 0)
        {
            throw graphblas::DimensionException();
        }

        std::cerr << "batch size (p): " << nsver << std::endl;

        using T = typename MatrixT::ScalarType;
        graphblas::IndexType m, n;

        // GrB_Matrix_nrows(&n, A)
        A.get_shape(m, n);
        if (m != n)
        {
            throw graphblas::DimensionException();
        }

        std::cerr << "num nodes (n): " << n << std::endl;

        // Placeholder for GrB_ALL where dimension size is nsver=|s|
        std::vector<graphblas::IndexType> GrB_ALL_nsver;     // fill with sequence
        GrB_ALL_nsver.reserve(nsver);
        for (graphblas::IndexType idx = 0; idx < nsver; ++idx)
        {
            GrB_ALL_nsver.push_back(idx); // tilln
        }

        // Placeholder for GrB_ALL where dimension is n
        std::vector<graphblas::IndexType> GrB_ALL_n;     // fill with sequence
        GrB_ALL_n.reserve(n);
        for (graphblas::IndexType idx = 0; idx < n; ++idx)
        {
            GrB_ALL_n.push_back(idx);
        }

        // The current frontier for all BFS's (from all roots)
        // It is initialized to the out neighbors of the specified roots
        graphblas::Matrix<int32_t> Frontier(nsver, n);     // F is nsver x n (rows)
        graphblas::extract(A, s, GrB_ALL_n, Frontier);
        graphblas::print_matrix(std::cerr, Frontier, "initial frontier");

        // NumSP holds number of shortest paths to a vertex from a given root
        // NumSP is initialized with the each starting root in 's':
        // NumSP[i,s[i]] = 1 where 0 <= i < nsver; implied zero elsewhere
        graphblas::Matrix<int32_t> NumSP(nsver, n);
        graphblas::buildmatrix(NumSP, GrB_ALL_nsver, s,
                               std::vector<int32_t>(nsver, 1));
        graphblas::print_matrix(std::cerr, NumSP, "initial NumSP");

        // ==================== BFS phase ====================
        // Placeholders for GraphBLAS operators
        graphblas::PlusMonoid<int32_t>            Int32Add;
        graphblas::ArithmeticSemiring<int32_t>    Int32AddMul;
        graphblas::math::Identity<bool, int32_t>  GrB_IDENTITY_BOOL;

        std::vector<graphblas::Matrix<bool>* > Sigmas;
        std::cerr << "======= START BFS phase ======" << std::endl;
        int32_t d = 0;
        while (Frontier.get_nnz() > 0)
        {
            std::cerr << "------- BFS iteration " << d << " --------" << std::endl;

            Sigmas.push_back(new graphblas::Matrix<bool>(nsver, n, false));
            // Sigma[d] = (bool)F
            graphblas::apply(Frontier, *(Sigmas[d]), GrB_IDENTITY_BOOL);
            graphblas::print_matrix(std::cerr, *Sigmas[d], "Sigma[d] = (bool)Frontier");
            // P = F + P
            graphblas::ewiseadd(NumSP, Frontier, NumSP, Int32Add);
            graphblas::print_matrix(std::cerr, NumSP, "NumSP");
            // F<!P> = F +.* A
            graphblas::mxmMasked(Frontier, A, Frontier,
                                 graphblas::negate(NumSP, Int32AddMul),
                                 Int32AddMul);
            graphblas::print_matrix(std::cerr, Frontier, "New frontier");

            ++d;
        }
        std::cerr << "======= END BFS phase =======" << std::endl;

        // ================== backprop phase ==================
        // Placeholders for GraphBLAS operators
        graphblas::PlusMonoid<float>           FP32Add;
        graphblas::TimesMonoid<float>          FP32Mul;
        graphblas::ArithmeticSemiring<float>   FP32AddMul;
        graphblas::math::Inverse<float>        GrB_MINV_FP32;

        graphblas::Matrix<float> NspInv(nsver, n);
        graphblas::apply(NumSP, NspInv, GrB_MINV_FP32);
        graphblas::print_matrix(std::cerr, NspInv, "(1 ./ P)");

        graphblas::Matrix<float> BCu = graphblas::fill<graphblas::Matrix<float>>
            (1.0f, nsver, n);

        graphblas::print_matrix(std::cerr, BCu, "U");

        graphblas::Matrix<float> W(nsver, n);

        std::cerr << "======= BEGIN BACKPROP phase =======" << std::endl;
        for (int32_t i = d - 1; i > 0; --i)
        {
            std::cerr << "------- BACKPROP iteration " << i << " --------" << std::endl;

            // W<Sigma[i]> = (1 ./ P) .* U
            bool const GrB_REPLACE = true;
            graphblas::ewisemultMasked(NspInv, BCu, W, *Sigmas[i], GrB_REPLACE, FP32Mul);
            graphblas::print_matrix(std::cerr, W, "W<Sigma[i]> = (1 ./ P) .* U");

            // W<Sigma[i-1]> = A +.* W
            graphblas::mxmMasked(W, graphblas::transpose(A), W,
                                 *Sigmas[i-1], FP32AddMul);
            graphblas::print_matrix(std::cerr, W, "W<Sigma[i-1]> = A +.* W");

            // U += W .* P
            graphblas::ewisemult(W, NumSP, BCu, FP32Mul,
                                 graphblas::math::Accum<float>());
            graphblas::print_matrix(std::cerr, BCu, "U += W .* P");

            --d;
        }
        std::cerr << "======= END BACKPROP phase =======" << std::endl;
        graphblas::print_matrix(std::cerr, BCu, "BC Updates");
        graphblas::Matrix<float> result =
            graphblas::fill<graphblas::Matrix<float>>(-nsver, 1, n);
        graphblas::col_reduce(BCu, result, FP32Add,
                              graphblas::math::Accum<float>());
        graphblas::print_matrix(std::cerr, result, "RESULT");

        std::vector<float> betweenness_centrality;
        for (graphblas::IndexType k = 0; k < n;k++)
        {
            betweenness_centrality.push_back(result.get_value_at(0, k));
        }

        for (auto it = Sigmas.begin(); it != Sigmas.end(); ++it)
        {
            delete *it;
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
    vertex_betweenness_centrality_batch_alt(
        MatrixT const                   &A,
        graphblas::IndexArrayType const &s)
    {
        graphblas::print_matrix(std::cerr, A, "Graph");

        // nsver = |s| (partition size)
        graphblas::IndexType nsver(s.size());
        if (nsver == 0)
        {
            throw graphblas::DimensionException();
        }

        std::cerr << "batch size (p): " << nsver << std::endl;

        using T = typename MatrixT::ScalarType;
        graphblas::IndexType m, n;

        // GrB_Matrix_nrows(&n, A)
        A.get_shape(m, n);
        if (m != n)
        {
            throw graphblas::DimensionException();
        }

        std::cerr << "num nodes (n): " << n << std::endl;

        // Placeholder for GrB_ALL where dimension size is nsver=|s|
        std::vector<graphblas::IndexType> GrB_ALL_nsver;     // fill with sequence
        GrB_ALL_nsver.reserve(nsver);
        for (graphblas::IndexType idx = 0; idx < nsver; ++idx)
        {
            GrB_ALL_nsver.push_back(idx); // tilln
        }

        // Placeholder for GrB_ALL where dimension is n
        std::vector<graphblas::IndexType> GrB_ALL_n;     // fill with sequence
        GrB_ALL_n.reserve(n);
        for (graphblas::IndexType idx = 0; idx < n; ++idx)
        {
            GrB_ALL_n.push_back(idx);
        }

        // The current frontier for all BFS's (from all roots)
        // It is initialized to the out neighbors of the specified roots
        graphblas::Matrix<int32_t> Frontier(n, nsver);         // F is n x nsver
        graphblas::extract(graphblas::transpose(A), GrB_ALL_n, s, Frontier);
        graphblas::print_matrix(std::cerr, Frontier, "initial frontier");

        // NumSP holds number of shortest paths to a vertex from a given root
        // NumSP is initialized with the each starting root in 's':
        // NumSP[s[i],i] = 1 where 0 <= i < nsver; implied zero elsewhere
        graphblas::Matrix<int32_t> NumSP(n, nsver);
        graphblas::buildmatrix(NumSP, s, GrB_ALL_nsver,
                               std::vector<int32_t>(nsver, 1));
        graphblas::print_matrix(std::cerr, NumSP, "initial NumSP");

        // ==================== BFS phase ====================
        // Placeholders for GraphBLAS operators
        graphblas::PlusMonoid<int32_t>            Int32Add;
        graphblas::ArithmeticSemiring<int32_t>    Int32AddMul;
        graphblas::math::Identity<bool, int32_t>  GrB_IDENTITY_BOOL;

        std::vector<graphblas::Matrix<bool>* > Sigmas;
        std::cerr << "======= START BFS phase ======" << std::endl;
        int32_t d = 0;
        while (Frontier.get_nnz() > 0)
        {
            std::cerr << "------- BFS iteration " << d << " --------" << std::endl;

            Sigmas.push_back(new graphblas::Matrix<bool>(n, nsver, false));
            graphblas::apply(Frontier, *(Sigmas[d]),
                             GrB_IDENTITY_BOOL);  // Sigma[d] = (bool)F
            graphblas::print_matrix(std::cerr, *Sigmas[d], "Sigma[d] = (bool)Frontier");

            graphblas::ewiseadd(NumSP, Frontier, NumSP, Int32Add);   // P = F + P
            graphblas::print_matrix(std::cerr, NumSP, "NumSP");

            graphblas::mxmMasked(graphblas::transpose(A), Frontier,  // F<!P> = A' +.* F
                                 Frontier,
                                 graphblas::negate(NumSP, Int32AddMul),
                                 Int32AddMul);
            graphblas::print_matrix(std::cerr, Frontier, "New frontier");

            ++d;
        }
        std::cerr << "======= END BFS phase =======" << std::endl;

        // ================== backprop phase ==================
        // Placeholders for GraphBLAS operators
        graphblas::PlusMonoid<float>           FP32Add;
        graphblas::TimesMonoid<float>          FP32Mul;
        graphblas::ArithmeticSemiring<float>   FP32AddMul;
        graphblas::math::Inverse<float>        GrB_MINV_FP32;

        graphblas::Matrix<float> NspInv(n, nsver);
        graphblas::apply(NumSP, NspInv, GrB_MINV_FP32);
        graphblas::print_matrix(std::cerr, NspInv, "(1 ./ P)");

        graphblas::Matrix<float> BCu = graphblas::fill<graphblas::Matrix<float>>
            (1.0f, n, nsver);

        graphblas::print_matrix(std::cerr, BCu, "U");

        graphblas::Matrix<float> W(n, nsver);

        std::cerr << "======= BEGIN BACKPROP phase =======" << std::endl;
        for (int32_t i = d - 1; i > 0; --i)
        {
            std::cerr << "------- BACKPROP iteration " << i << " --------" << std::endl;

            // W<Sigma[i]> = (1 ./ P) .* U
            bool const GrB_REPLACE = true;
            graphblas::ewisemultMasked(NspInv, BCu, W, *Sigmas[i], GrB_REPLACE, FP32Mul);
            graphblas::print_matrix(std::cerr, W, "W<Sigma[i]> = (1 ./ P) .* U");

            // W<Sigma[i-1]> = A +.* W
            graphblas::mxmMasked(A, W, W, *Sigmas[i-1], FP32AddMul);
            graphblas::print_matrix(std::cerr, W, "W<Sigma[i-1]> = A +.* W");

            // U += W .* P
            graphblas::ewisemult(W, NumSP, BCu, FP32Mul,
                                 graphblas::math::Accum<float>());
            graphblas::print_matrix(std::cerr, BCu, "U += W .* P");

            --d;
        }
        std::cerr << "======= END BACKPROP phase =======" << std::endl;
        graphblas::print_matrix(std::cerr, BCu, "BC Updates");
        graphblas::Matrix<float> result =
            graphblas::fill<graphblas::Matrix<float>>(-nsver, n, 1);
        graphblas::row_reduce(BCu, result, FP32Add,
                              graphblas::math::Accum<float>());
        graphblas::print_matrix(std::cerr, result, "RESULT");

        std::vector<float> betweenness_centrality;
        for (graphblas::IndexType k = 0; k < n;k++)
        {
            betweenness_centrality.push_back(result.get_value_at(k, 0));
        }

        for (auto it = Sigmas.begin(); it != Sigmas.end(); ++it)
        {
            delete *it;
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
        graphblas::IndexArrayType const &s)
    {
        graphblas::print_matrix(std::cerr, A, "Graph");

        // nsver = |s| (partition size)
        graphblas::IndexType nsver(s.size());
        if (nsver == 0)
        {
            throw graphblas::DimensionException();
        }

        std::cerr << "batch size (p): " << nsver << std::endl;

        using T = typename MatrixT::ScalarType;
        graphblas::IndexType m, n;

        // GrB_Matrix_nrows(&n, A)
        A.get_shape(m, n);
        if (m != n)
        {
            throw graphblas::DimensionException();
        }

        std::cerr << "num nodes (n): " << n << std::endl;

        // Placeholder for GrB_ALL where dimension size is nsver=|s|
        std::vector<graphblas::IndexType> GrB_ALL_nsver;     // fill with sequence
        GrB_ALL_nsver.reserve(nsver);
        for (graphblas::IndexType idx = 0; idx < nsver; ++idx)
        {
            GrB_ALL_nsver.push_back(idx); // tilln
        }

        // Placeholder for GrB_ALL where dimension is n
        std::vector<graphblas::IndexType> GrB_ALL_n;     // fill with sequence
        GrB_ALL_n.reserve(n);
        for (graphblas::IndexType idx = 0; idx < n; ++idx)
        {
            GrB_ALL_n.push_back(idx);
        }

        // The current frontier for all BFS's (from all roots)
        // It is initialized with the each starting root in 's':
        // F[s[i],i] = 1 where 0 <= i < nsver; implied zero elsewhere
        graphblas::Matrix<int32_t> Frontier(n, nsver);         // F is n x nsver
        graphblas::buildmatrix(Frontier, s, GrB_ALL_nsver,
                               std::vector<int32_t>(nsver, 1));
        graphblas::print_matrix(std::cerr, Frontier, "initial frontier");

        // NumSP holds number of shortest paths to a vertex from a given root
        // NumSP is initialized to a copy of Frontier. NumSP is n x nsver
        graphblas::Matrix<int32_t> NumSP(Frontier);
        graphblas::print_matrix(std::cerr, NumSP, "initial NumSP");

        // ==================== BFS phase ====================
        // Placeholders for GraphBLAS operators
        graphblas::PlusMonoid<int32_t>            Int32Add;
        graphblas::ArithmeticSemiring<int32_t>    Int32AddMul;
        graphblas::math::Identity<bool, int32_t>  GrB_IDENTITY_BOOL;

        std::vector<graphblas::Matrix<bool>* > Sigmas;
        std::cerr << "======= START BFS phase ======" << std::endl;
        int32_t d = 0;
        while (Frontier.get_nnz() > 0)
        {
            std::cerr << "------- BFS iteration " << d << " --------" << std::endl;
            graphblas::mxmMasked(graphblas::transpose(A), Frontier,  // F<!P> = A' +.* F
                                 Frontier,
                                 graphblas::negate(NumSP, Int32AddMul),
                                 Int32AddMul);
            graphblas::print_matrix(std::cerr, Frontier, "New frontier");

            Sigmas.push_back(new graphblas::Matrix<bool>(n, nsver, false));
            graphblas::apply(Frontier, *(Sigmas[d]),
                             GrB_IDENTITY_BOOL);  // Sigma[d] = (bool)F
            graphblas::print_matrix(std::cerr, *Sigmas[d], "Sigma[d] = (bool)Frontier");

            graphblas::ewiseadd(NumSP, Frontier, NumSP, Int32Add);   // P = F + P
            graphblas::print_matrix(std::cerr, NumSP, "NumSP");

            ++d;
        }
        std::cerr << "======= END BFS phase =======" << std::endl;

        // ================== backprop phase ==================
        // Placeholders for GraphBLAS operators
        graphblas::PlusMonoid<float>           FP32Add;
        graphblas::TimesMonoid<float>          FP32Mul;
        graphblas::ArithmeticSemiring<float>   FP32AddMul;
        graphblas::math::Inverse<float>        GrB_MINV_FP32;

        graphblas::Matrix<float> NspInv(n, nsver);
        graphblas::apply(NumSP, NspInv, GrB_MINV_FP32);
        graphblas::print_matrix(std::cerr, NspInv, "(1 ./ P)");

        graphblas::Matrix<float> BCu = graphblas::fill<graphblas::Matrix<float>>
            (1.0f, n, nsver);

        graphblas::print_matrix(std::cerr, BCu, "U");

        graphblas::Matrix<float> W(n, nsver);

        std::cerr << "======= BEGIN BACKPROP phase =======" << std::endl;
        for (int32_t i = d - 1; i > 0; --i)
        {
            std::cerr << "------- BACKPROP iteration " << i << " --------" << std::endl;

            // W<Sigma[i]> = (1 ./ P) .* U
            bool const GrB_REPLACE = true;
            graphblas::ewisemultMasked(NspInv, BCu, W, *Sigmas[i], GrB_REPLACE, FP32Mul);
            graphblas::print_matrix(std::cerr, W, "W<Sigma[i]> = (1 ./ P) .* U");

            // W<Sigma[i-1]> = A +.* W
            graphblas::mxmMasked(A, W, W, *Sigmas[i-1], FP32AddMul);
            graphblas::print_matrix(std::cerr, W, "W<Sigma[i-1]> = A +.* W");

            // U += W .* P
            graphblas::ewisemult(W, NumSP, BCu, FP32Mul,
                                 graphblas::math::Accum<float>());
            graphblas::print_matrix(std::cerr, BCu, "U += W .* P");

            --d;
        }
        std::cerr << "======= END BACKPROP phase =======" << std::endl;
        graphblas::print_matrix(std::cerr, BCu, "BC Updates");
        graphblas::Matrix<float> result =
            graphblas::fill<graphblas::Matrix<float>>(-nsver, n, 1);
        graphblas::row_reduce(BCu, result, FP32Add,
                              graphblas::math::Accum<float>());
        graphblas::print_matrix(std::cerr, result, "RESULT");

        std::vector<float> betweenness_centrality;
        for (graphblas::IndexType k = 0; k < n;k++)
        {
            betweenness_centrality.push_back(result.get_value_at(k, 0));
        }

        for (auto it = Sigmas.begin(); it != Sigmas.end(); ++it)
        {
            delete *it;
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
        graphblas::IndexArrayType const &src_nodes)
    {
        graphblas::print_matrix(std::cerr, graph, "Graph");

        // p = nsver = |src_nodes| (partition size)
        graphblas::IndexType p(src_nodes.size());
        if (p == 0)
        {
            throw graphblas::DimensionException();
        }

        std::cerr << "batch size (p): " << p << std::endl;

        using T = typename MatrixT::ScalarType;
        graphblas::IndexType N, num_cols;

        // GrB_Matrix_nrows(&N, graph)
        graph.get_shape(N, num_cols);
        if (N != num_cols)
        {
            throw graphblas::DimensionException();
        }

        std::cerr << "num nodes (N): " << N << std::endl;

        // Placeholders for GraphBLAS operators
        graphblas::PlusMonoid<int32_t>         GrB_Int32Add;
        graphblas::ArithmeticSemiring<int32_t> GrB_Int32AddMul;
        graphblas::PlusMonoid<double>          GrB_Flt64Add;
        graphblas::TimesMonoid<double>         GrB_Flt64Mul;
        graphblas::DivMonoid<double>           GrB_Flt64Div;
        graphblas::ArithmeticSemiring<double>  GrB_Flt64AddMul;

        // Placeholder for GrB_ALL where dimension is p=|nsver|
        std::vector<graphblas::IndexType> GrB_ALL_p;     // fill with sequence
        GrB_ALL_p.reserve(p);
        for (graphblas::IndexType idx = 0; idx < p; ++idx)
        {
            GrB_ALL_p.push_back(idx);
        }

        // P holds number of shortest paths to a vertex from a given root
        // P[i,src_nodes[i]] = 1 where 0 <= i < nsver; implied zero elsewher
        graphblas::Matrix<int32_t> P(p, N);         // P is pxN
        std::vector<int32_t> ones(p, 1);
        graphblas::buildmatrix(P, GrB_ALL_p, src_nodes, ones);
        graphblas::print_matrix(std::cerr, P, "initial P");

        // Placeholder for GrB_ALL where dimension is N
        std::vector<graphblas::IndexType> GrB_ALL_N;     // fill with sequence
        GrB_ALL_N.reserve(N);
        for (graphblas::IndexType idx = 0; idx < N; ++idx)
        {
            GrB_ALL_N.push_back(idx);
        }

        // F is the current frontier for all BFS's (from all roots)
        // It is initialized with the out neighbors for each root
        graphblas::Matrix<int32_t> F(p, N);         // F is pxN
        graphblas::extract(graph, src_nodes, GrB_ALL_N, F);
        graphblas::print_matrix(std::cerr, F, "initial frontier");

        // ==================== BFS phase ====================
        std::vector<graphblas::Matrix<int32_t>* > Sigmas;
        std::cerr << "======= START BFS phase ======" << std::endl;
        int32_t d = 0;
        while (F.get_nnz() > 0)
        {
            std::cerr << "------- BFS iteration " << d << " --------" << std::endl;
            Sigmas.push_back(new graphblas::Matrix<int32_t>(F));  // Sigma[d] = F
            graphblas::print_matrix(std::cerr, *Sigmas[d], "Sigma[d] = F");

            graphblas::ewiseadd(P, F, P, GrB_Int32Add);           // P = F + P
            graphblas::print_matrix(std::cerr, P, "P");

            graphblas::mxmMasked(F, graph, F,                     // F<!P> = F +.* A
                                 graphblas::negate(P, GrB_Int32AddMul),
                                 GrB_Int32AddMul);
            graphblas::print_matrix(std::cerr, F, "New frontier (F)");
            ++d;
        }
        std::cerr << "======= END BFS phase =======" << std::endl;

        // ================== backprop phase ==================
        graphblas::Matrix<double> U(p, N);        // U is pxN
        graphblas::Matrix<double> Ones =
            graphblas::fill<graphblas::Matrix<double>>(
                (double)1.0, p, N);  // pxN filled with 1.0
        graphblas::print_matrix(std::cerr, Ones, "Ones");

        graphblas::Matrix<double> W(p, N);
        graphblas::Matrix<double> tmp1(p, N);
        graphblas::Matrix<double> tmp2(p, N);

        std::cerr << "======= BEGIN BACKPROP phase =======" << std::endl;
        while (d >= 2)
        {
            std::cerr << "------- BACKPROP iteration " << d << " --------" << std::endl;
            // W = Sigma[d-1].*(1 + U)./P
            graphblas::ewiseadd(Ones, U, tmp1, GrB_Flt64Add);
            graphblas::print_matrix(std::cerr, tmp1, "(1 + U)");
            graphblas::ewisemult(*Sigmas[d-1], tmp1, tmp2, GrB_Flt64Mul);
            graphblas::print_matrix(std::cerr, tmp2, "Sigma[d-1] .* (1 + U)");
            graphblas::ewisemult(tmp2, P, W, GrB_Flt64Div);
            graphblas::print_matrix(std::cerr, W, "W = Sigma[d-1] .* (1 + U) ./ P");

            // W = (A +.* W')' = W +.* A'
            graphblas::mxm(W, graphblas::transpose(graph), W, GrB_Flt64AddMul);
            graphblas::print_matrix(std::cerr, W, "W = W A'");

            // W = W .* Sigma[d-2] .* P
            graphblas::ewisemult(W, *Sigmas[d-2], W, GrB_Flt64Mul);
            graphblas::ewisemult(W, P, W, GrB_Flt64Mul);
            graphblas::print_matrix(std::cerr, W, "W = W .* P");

            // U = U + W
            graphblas::ewiseadd(U, W, U, GrB_Flt64Add);
            graphblas::print_matrix(std::cerr, U, "U += W");

            --d;
        }
        std::cerr << "======= END BACKPROP phase =======" << std::endl;
        graphblas::print_matrix(std::cerr, U, "U");
        graphblas::Matrix<double> result(1, N);
        graphblas::col_reduce(U, result, GrB_Flt64Add);
        graphblas::print_matrix(std::cerr, result, "RESULT");

        std::vector<double> betweenness_centrality;
        for (graphblas::IndexType k = 0; k < N;k++)
        {
            betweenness_centrality.push_back(result.get_value_at(0, k));
        }

        return betweenness_centrality;
    }

    /**
     * @brief Compute the vertex betweenness centrality of all vertices in the
     *        given graph.
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
        graphblas::IndexType num_nodes, cols, depth;
        graph.get_shape(num_nodes, cols);
        if (num_nodes != cols)
        {
            throw graphblas::DimensionException();
        }

        MatrixT bc_mat(1, num_nodes);
        for (graphblas::IndexType i = 0; i < num_nodes; ++i)
        {
            depth = 0;
            MatrixT search(num_nodes, num_nodes);
            MatrixT fringe(1, num_nodes);
            /// @todo replaced with extract?
            for (graphblas::IndexType k = 0; k < num_nodes; ++k)
            {
                fringe.set_value_at(0, k, graph.get_value_at(i, k));
            }

            MatrixT n_shortest_paths(1, num_nodes);
            n_shortest_paths.set_value_at(0, i, static_cast<T>(1));

            while(fringe.get_nnz() != 0)
            {
                depth = depth + 1;
                graphblas::ewiseadd(n_shortest_paths,fringe,n_shortest_paths);

                for (graphblas::IndexType k = 0; k < num_nodes; ++k)
                {
                    search.set_value_at(
                        depth, k,
                        (fringe.get_value_at(0, k) != 0));  // cast<T>?
                }

                MatrixT not_in_sps = graphblas::fill<MatrixT>(1, 1, num_nodes);
                //for (graphblas::IndexType k = 0; k < num_nodes; ++k)
                //{
                //    not_in_sps.set_value_at(0, k, 1);
                //}

                for (graphblas::IndexType k = 0; k < num_nodes; ++k)
                {
                    if (n_shortest_paths.get_value_at(0, k) != 0)
                    {
                        not_in_sps.set_value_at(0, k, 0); // get_zero()
                    }
                }
                graphblas::mxm(fringe, graph, fringe);
                graphblas::ewisemult(fringe, not_in_sps, fringe);
            }

            MatrixT update(1, num_nodes);
            MatrixT ones = graphblas::fill<MatrixT>(1, 1, num_nodes);

            while (depth >= 2)
            {
                MatrixT n_shortest_paths_inv(1, num_nodes);
                graphblas::apply(n_shortest_paths,
                                 n_shortest_paths_inv,
                                 graphblas::math::inverse<T>);

                MatrixT weights(num_nodes, 1);
                MatrixT temp(1, num_nodes);

                graphblas::ewiseadd(ones, update, temp);
                graphblas::ewisemult(temp, n_shortest_paths_inv, temp);

                for (graphblas::IndexType k = 0; k < num_nodes; ++k)
                {
                    //weights[k][0] = search[depth][k] * temp[0][k];
                    weights.set_value_at(
                        k, 0,
                        search.get_value_at(depth, k) *
                        temp.get_value_at(0, k));
                }

                mxm(graph, weights, weights);

                for (graphblas::IndexType k = 0; k < num_nodes; ++k)
                {
                    //temp[0][k] = search[depth-1][k] * n_shortest_paths[0][k];
                    temp.set_value_at(
                        0, k,
                        search.get_value_at(depth - 1, k) *
                        n_shortest_paths.get_value_at(0, k));
                }

                graphblas::ewisemult(weights,
                                     //graphblas::TransposeView<MatrixT>(temp),
                                     graphblas::transpose(temp),
                                     weights);

                graphblas::ewiseadd(update,
                                    //graphblas::TransposeView<MatrixT>(weights),
                                    graphblas::transpose(weights),
                                    update);
                depth = depth - 1;
            }

            graphblas::ewiseadd(bc_mat, update, bc_mat);
        }

        std::vector <typename MatrixT::ScalarType> betweenness_centrality;
        for (graphblas::IndexType k = 0; k<num_nodes;k++)
        {
            betweenness_centrality.push_back(bc_mat.get_value_at(0, k));
        }

        return betweenness_centrality;
    }

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
     * @return The betweenness centrality of all vertices in the graph
     */
    template<typename MatrixT>
    MatrixT edge_betweenness_centrality(MatrixT const &graph)
    {
        using T = typename MatrixT::ScalarType;
        graphblas::IndexType num_nodes, cols, depth;
        graph.get_shape(num_nodes, cols);

        if (num_nodes != cols)
        {
            throw graphblas::DimensionException();
        }

        MatrixT score(num_nodes, num_nodes);

        for (graphblas::IndexType root = 0; root < num_nodes; root++)
        {
            depth = 0;
            MatrixT search(num_nodes, num_nodes);
            MatrixT fringe (1,num_nodes);
            for (graphblas::IndexType k = 0; k < num_nodes; ++k)
            {
                //fringe[0][k] = graph[root][k];
                fringe.set_value_at(0, k,
                                    graph.get_value_at(root, k));
            }
            MatrixT n_shortest_paths(1, num_nodes);
            //n_shortest_paths[0][root] = 1;
            n_shortest_paths.set_value_at(0, root, 1);

            MatrixT update(num_nodes, num_nodes);
            MatrixT flow(1, num_nodes);
            //search[depth][root] = 1;
            search.set_value_at(depth, root, 1);

            while (fringe.get_nnz() != 0)
            {
                depth = depth + 1;
                graphblas::ewiseadd(n_shortest_paths,
                                    fringe,
                                    n_shortest_paths);
                for (graphblas::IndexType i = 0; i < num_nodes; ++i)
                {
                    //search[depth][i] = (fringe[0][i] != 0);
                    search.set_value_at(depth, i,
                                        (fringe.get_value_at(0, i) != 0));
                }

                MatrixT not_in_sps = graphblas::fill<MatrixT>(1, 1, num_nodes);
                //for (graphblas::IndexType k = 0; k < num_nodes; k++)
                //{
                //    not_in_sps[0][k] = 1;
                //}

                for (graphblas::IndexType k = 0; k < num_nodes; ++k)
                {
                    if (n_shortest_paths.get_value_at(0, k) != 0)
                    {
                        not_in_sps.set_value_at(0, k, 0);  // get_zero()?
                    }
                }

                graphblas::mxm(fringe, graph, fringe);
                graphblas::ewisemult(fringe, not_in_sps, fringe);
            }

            while (depth >= 1)
            {
                MatrixT n_shortest_paths_inv(1, num_nodes);
                graphblas::apply(n_shortest_paths,
                                 n_shortest_paths_inv,
                                 graphblas::math::inverse<T>);
                MatrixT weights(1, num_nodes);

                for (graphblas::IndexType k = 0; k < num_nodes; ++k)
                {
                    //weights[0][k] =
                    //    search[depth][k] * n_shortest_paths_inv[0][k];
                    weights.set_value_at(
                        0, k,
                        search.get_value_at(depth, k) *
                        n_shortest_paths_inv.get_value_at(0, k));
                }

                graphblas::ewisemult(weights, flow, weights);

                for (graphblas::IndexType k = 0; k < num_nodes; ++k)
                {
                    //weights[0][k] = weights[0][k] + search[depth][k];
                    weights.set_value_at(
                        0, k,
                        weights.get_value_at(0, k) +
                        search.get_value_at(depth, k));
                }


                for (graphblas::IndexType i = 0; i < num_nodes; i++)
                {
                    for (graphblas::IndexType k=0;k<num_nodes; k++)
                    {
                        //update[i][k] = graph[i][k] * weights[0][k];
                        update.set_value_at(
                            i, k,
                            graph.get_value_at(i, k) *
                            weights.get_value_at(0, k));
                    }
                }

                for (graphblas::IndexType k=0;k<num_nodes; k++)
                {
                    //weights[0][k] =
                    //    search[depth-1][k] * n_shortest_paths[0][k];
                    weights.set_value_at(
                        0, k,
                        search.get_value_at(depth - 1, k) *
                        n_shortest_paths.get_value_at(0, k));
                }

                for (graphblas::IndexType i = 0; i < num_nodes; ++i)
                {
                    for (graphblas::IndexType k = 0; k < num_nodes; ++k)
                    {
                        //update[k][i] = weights[0][k] * update[k][i];
                        update.set_value_at(
                            k, i,
                            weights.get_value_at(0, k) *
                            update.get_value_at(k, i));
                    }
                }
                graphblas::ewiseadd(score, update, score);

                MatrixT temp(num_nodes, 1);
                row_reduce(update, temp);
                for (graphblas::IndexType k = 0; k < num_nodes; ++k)
                {
                    //flow[0][k] = temp[k][0];
                    flow.set_value_at(0, k, temp.get_value_at(k, 0));
                }

                depth = depth - 1;
            }
        }
        return score;
    }

} // algorithms

#endif // METRICS_HPP
