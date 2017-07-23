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

#ifndef ALGORITHMS_TRIANGLE_COUNT_HPP
#define ALGORITHMS_TRIANGLE_COUNT_HPP

#include <iostream>
#include <chrono>

#define GB_DEBUG
#include <graphblas/graphblas.hpp>
//#include <graphblas/linalg_utils.hpp>


namespace algorithms
{
    /**
     * @brief Compute the number of triangles in a given graph.
     *
     * This function expects an undirected graph.  If it is desired
     * to count the number of edges in a digraph, just pass in the
     * digraph, and then multiply the resulting number of triangles by 2.
     *
     * Given the adjacency matrix of a graph, the idea behind the
     * triangle counting algorithm used is as follows:
     * <ol>
     * <li>First, split \f$A\f$ into lower and upper triangular matrices such
     * that \f$A = L + U\f$.</li>
     *
     * <li>Because the multiplication of \f$L\f$ and \f$U\f$ counts all the
     * wedges \f$(i, j, k)\f$, where \f$j\f$ is the vetex with the lowest
     * degree, we compute the matrix \f$B = LU\f$.</li>
     *
     * <li>Finally, to determine whether the wedges close or not, we compute
     * \f$C = A \circ B\f$.</li>
     *
     * <li>The final number of triangles is then
     * \f$\sum\limits_i^N\sum\limits_j^N C_{ij}\f$.</li>
     * </ol>
     *
     * However, when implementing the algorithm, various optimizations were
     * made to improve performance.
     *
     * @param[in]  graph  The graph to compute the number of triangles in.
     *
     * @return The number of triangles in graph.
     */
    template<typename MatrixT>
    typename MatrixT::ScalarType triangle_count(MatrixT const &graph)
    {
        using T = typename MatrixT::ScalarType;
        GraphBLAS::IndexType rows(graph.nrows());
        GraphBLAS::IndexType cols(graph.ncols());

        MatrixT L(rows, cols), U(rows, cols);
        GraphBLAS::split(graph, L, U);

        MatrixT B(rows, cols);
        //GraphBLAS::mxm(L, U, B);
        GraphBLAS::mxm(B,
                       GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                       GraphBLAS::ArithmeticSemiring<T>(),
                       L, GraphBLAS::transpose(L)); //U);

        MatrixT C(rows, cols);
        //GraphBLAS::ewisemult(graph, B, C);
        GraphBLAS::eWiseMult(C,
                             GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                             GraphBLAS::Times<T>(),
                             graph, B);

        T sum = 0;
        // for (GraphBLAS::IndexType i = 0; i< rows; ++i)
        // {
        //     for (GraphBLAS::IndexType j = 0; j < cols; ++j)
        //     {
        //         sum = sum + C.extractElement(i, j);
        //     }
        // }
        GraphBLAS::reduce(sum,
                          GraphBLAS::NoAccumulate(),
                          GraphBLAS::PlusMonoid<T>(),
                          C);
        return sum / static_cast<T>(2);
    }

    //************************************************************************
    /**
     * From TzeMeng Low, the FLAME approach #1 to triangle counting
     *
     * @param[in] graph  Must be undirected graph with no self-loops; i.e.,
     *                   matrix is symmetric with zeros on the diagonal.
     */
    template<typename MatrixT>
    typename MatrixT::ScalarType triangle_count_flame1(MatrixT const &graph)
    {
        using T = typename MatrixT::ScalarType;
        using VectorT = GraphBLAS::Vector<T>;

        GraphBLAS::IndexType rows(graph.nrows());
        GraphBLAS::IndexType cols(graph.ncols());

        if (rows != cols)
        {
            throw GraphBLAS::DimensionException(
                "triangle_count_flame1 matrix is not square");
        }

        // A graph less than three vertices cannot have any triangles
        if (rows < 3)
        {
            return 0;
        }

        // "split" is not part of GraphBLAS...placeholder because Masking not
        // completely implemented
        MatrixT L(rows, cols), U(rows, cols);
        GraphBLAS::split(graph, L, U);

        GraphBLAS::IndexArrayType indices = {0};

        T delta(0UL);
        for (GraphBLAS::IndexType idx = 2; idx < rows; ++idx)
        {
            MatrixT A00(idx, idx);
            VectorT a01(idx);
            VectorT tmp1(idx);

            indices.push_back(idx - 1);   // [0, 1, ... i - 1]

            //GraphBLAS::extract(U, indices, indices, A00);
            GraphBLAS::extract(A00,
                               GraphBLAS::NoMask(),
                               GraphBLAS::NoAccumulate(),
                               U, indices, indices);

            //GraphBLAS::extract(U, indices, column_index, a01);
            GraphBLAS::extract(a01,
                               GraphBLAS::NoMask(),
                               GraphBLAS::NoAccumulate(),
                               U, indices, idx); ///@todo try row extract

            //GraphBLAS::mxv(A00, a01, tmp1,
            //               GraphBLAS::ArithmeticSemiring<T>());
            GraphBLAS::mxv(tmp1,
                           GraphBLAS::NoMask(),
                           GraphBLAS::NoAccumulate(),
                           GraphBLAS::ArithmeticSemiring<T>(),
                           A00, a01);

            //GraphBLAS::vxm(transpose(a01), tmp1, tmp2,
            //               GraphBLAS::ArithmeticSemiring<T>());
            //delta += tmp2.extractElement(0,0);
            GraphBLAS::eWiseMult(tmp1,
                                 GraphBLAS::NoMask(),
                                 GraphBLAS::NoAccumulate(),
                                 GraphBLAS::Times<T>(),
                                 a01, tmp1);
            GraphBLAS::reduce(delta,
                              GraphBLAS::Plus<T>(),
                              GraphBLAS::PlusMonoid<T>(),
                              tmp1);

            std::cout << "Processed row " << idx << " of " << rows
                      << ", Running count: " << delta << std::endl;
        }

        return delta;
    }

    //************************************************************************
    /**
     * From TzeMeng Low, the FLAME approach #1 to triangle counting
     *
     * @note Scott: an attempt to 'reduce work at either end'...didn't work
     *
     * @param[in] graph  Must be undirected graph with no self-loops; i.e.,
     *                   matrix is symmetric with zeros on the diagonal.
     */
    template<typename MatrixT>
    typename MatrixT::ScalarType triangle_count_flame1a(MatrixT const &graph)
    {
        using T = typename MatrixT::ScalarType;
        using VectorT = GraphBLAS::Vector<T>;

        GraphBLAS::IndexType rows(graph.nrows());
        GraphBLAS::IndexType cols(graph.ncols());

        if (rows != cols)
        {
            throw GraphBLAS::DimensionException(
                "triangle_count_flame1 matrix is not square");
        }

        // A graph less than three vertices cannot have any triangles
        if (rows < 3)
        {
            return 0;
        }

        // "split" is not part of GraphBLAS...placeholder because Masking not
        // completely implemented
        MatrixT L(rows, cols), U(rows, cols);
        GraphBLAS::split(graph, L, U);

        GraphBLAS::IndexArrayType indices = {0};

        T delta(0UL);
        for (GraphBLAS::IndexType idx = 2; idx < rows/2; ++idx)
        {
            MatrixT A00(idx, idx);
            VectorT a01(idx);
            VectorT tmp1(idx);

            indices.push_back(idx - 1);   // [0, 1, ... i - 1]

            //GraphBLAS::extract(U, indices, indices, A00);
            GraphBLAS::extract(A00,
                               GraphBLAS::NoMask(),
                               GraphBLAS::NoAccumulate(),
                               U, indices, indices);
            //GraphBLAS::extract(U, indices, column_index, a01);
            GraphBLAS::extract(a01,
                               GraphBLAS::NoMask(),
                               GraphBLAS::NoAccumulate(),
                               U, indices, idx);

            //GraphBLAS::mxv(A00, a01, tmp1,
            //               GraphBLAS::ArithmeticSemiring<T>());
            GraphBLAS::mxv(tmp1,
                           GraphBLAS::NoMask(),
                           GraphBLAS::NoAccumulate(),
                           GraphBLAS::ArithmeticSemiring<T>(),
                           A00, a01);

            //GraphBLAS::vxm(transpose(a01), tmp1, tmp2,
            //               GraphBLAS::ArithmeticSemiring<T>());
            //delta += tmp2.extractElement(0,0);
            GraphBLAS::eWiseMult(tmp1,
                                 GraphBLAS::NoMask(),
                                 GraphBLAS::NoAccumulate(),
                                 GraphBLAS::Times<T>(),
                                 a01, tmp1);
            GraphBLAS::reduce(delta,
                              GraphBLAS::Plus<T>(),
                              GraphBLAS::PlusMonoid<T>(),
                              tmp1);

            std::cout << "Processed row " << idx << " of " << rows
                      << ", Running count: " << delta << std::endl;
        }

        for (GraphBLAS::IndexType idx = rows/2; idx < rows; ++idx)
        {
            MatrixT A00(idx, idx);
            VectorT a01(idx);
            VectorT tmp1(idx);

            indices.push_back(idx - 1);   // [0, 1, ... i - 1]

            //GraphBLAS::extract(U, indices, indices, A00);
            GraphBLAS::extract(A00,
                               GraphBLAS::NoMask(),
                               GraphBLAS::NoAccumulate(),
                               U, indices, indices);

            //GraphBLAS::extract(U, indices, column_index, a01);
            GraphBLAS::extract(a01,
                               GraphBLAS::NoMask(),
                               GraphBLAS::NoAccumulate(),
                               U, indices, idx);

            //GraphBLAS::vxm(transpose(a01), A00, tmp1,
            //               GraphBLAS::ArithmeticSemiring<T>());
            GraphBLAS::vxm(tmp1,
                           GraphBLAS::NoMask(),
                           GraphBLAS::NoAccumulate(),
                           GraphBLAS::ArithmeticSemiring<T>(),
                           a01, A00);

            //GraphBLAS::vxm(tmp1, a01, tmp2,
            //               GraphBLAS::ArithmeticSemiring<T>());
            //delta += tmp2.extractElement(0,0);
            GraphBLAS::eWiseMult(tmp1,
                                 GraphBLAS::NoMask(),
                                 GraphBLAS::NoAccumulate(),
                                 GraphBLAS::Times<T>(),
                                 a01, tmp1);
            GraphBLAS::reduce(delta,
                              GraphBLAS::Plus<T>(),
                              GraphBLAS::PlusMonoid<T>(),
                              tmp1);

            std::cout << "Processed row " << idx << " of " << rows
                      << ", Running count: " << delta << std::endl;
        }

        return delta;
    }

    //************************************************************************
    /**
     * From TzeMeng Low, the FLAME approach #2 to triangle counting
     *
     * @param[in] graph  Must be undirected graph with no self-loops; i.e.,
     *                   matrix is symmetric with zeros on the diagonal.
     */
    template<typename MatrixT>
    typename MatrixT::ScalarType triangle_count_flame2(MatrixT const &graph)
    {
        using T = typename MatrixT::ScalarType;
        using VectorT = GraphBLAS::Vector<T>;

        GraphBLAS::IndexType rows(graph.nrows());
        GraphBLAS::IndexType cols(graph.ncols());

        if (rows != cols)
        {
            throw GraphBLAS::DimensionException(
                "triangle_count_flame2 matrix is not square");
        }

        // A graph less than three vertices cannot have any triangles
        if (rows < 3)
        {
            return 0;
        }

        // the rows that a01 and A02 extract from (they grow)
        GraphBLAS::IndexArrayType row_indices;
        row_indices.reserve(rows);

        // the cols that a12 and A02 extract from (they shrink)
        GraphBLAS::IndexArrayType col_indices;
        col_indices.reserve(cols);
        for (GraphBLAS::IndexType idx = 1; idx < cols; ++idx)
        {
            col_indices.push_back(idx);
        }

        T delta(0UL);
        for (GraphBLAS::IndexType idx = 1; idx < rows - 1; ++idx)
        {
            // extract from the upper triangular portion of the adj. matrix only
            MatrixT A02(idx, cols - idx - 1);
            VectorT a01(idx);
            VectorT a12(cols - idx - 1);
            VectorT tmp1(cols - idx - 1);

            //GraphBLAS::IndexArrayType pivot_index = {idx};
            row_indices.push_back(idx - 1);   // [0, 1, ... i - 1]
            col_indices.erase(col_indices.begin());

            //GraphBLAS::extract(graph, row_indices, col_indices, A02);
            GraphBLAS::extract(A02,
                               GraphBLAS::NoMask(),
                               GraphBLAS::NoAccumulate(),
                               graph, row_indices, col_indices);

            //GraphBLAS::extract(graph, row_indices, pivot_index, a01);
            GraphBLAS::extract(a01,
                               GraphBLAS::NoMask(),
                               GraphBLAS::NoAccumulate(),
                               graph, row_indices, idx); ///@todo try row extract

            //GraphBLAS::extract(graph, pivot_index, col_indices, a12);
            GraphBLAS::extract(a12,
                               GraphBLAS::NoMask(),
                               GraphBLAS::NoAccumulate(),
                               GraphBLAS::transpose(graph), col_indices, idx);

            //GraphBLAS::mxv(transpose(A02), a01, tmp1,
            //               GraphBLAS::ArithmeticSemiring<T>());
            GraphBLAS::mxv(tmp1,
                           GraphBLAS::NoMask(),
                           GraphBLAS::NoAccumulate(),
                           GraphBLAS::ArithmeticSemiring<T>(),
                           GraphBLAS::transpose(A02), a01);

            //GraphBLAS::vxm(a12, tmp1, tmp2,
            //               GraphBLAS::ArithmeticSemiring<T>());
            //delta += tmp2.extractElement(0,0);
            GraphBLAS::eWiseMult(tmp1,
                                 GraphBLAS::NoMask(),
                                 GraphBLAS::NoAccumulate(),
                                 GraphBLAS::Times<T>(),
                                 a12, tmp1);
            GraphBLAS::reduce(delta,
                              GraphBLAS::Plus<T>(),
                              GraphBLAS::PlusMonoid<T>(),
                              tmp1);

            // DEBUG
            //std::cerr << "************** Iteration " << idx << " **************"
            //          << std::endl;
            //std::cerr << "A02 dimensions = " << idx << " x " << (cols-idx-1)
            //          << std::endl;
            //GraphBLAS::print_matrix(std::cerr, a01, "a01");
            //GraphBLAS::print_matrix(std::cerr, A02, "A02");
            //GraphBLAS::print_matrix(std::cerr, a12, "a12");
            //std::cerr << "delta = " << delta << std::endl;

            std::cout << "Processed row " << idx << " of " << rows
                      << ", Running count: " << delta << std::endl;
        }

        return delta;
    }


    //************************************************************************
    template<typename LMatrixT, typename MatrixT>
    typename MatrixT::ScalarType triangle_count_newGBTL(LMatrixT const &L,
                                                        MatrixT  const &U)
    {
        auto start = std::chrono::steady_clock::now();

        using T = typename MatrixT::ScalarType;
        GraphBLAS::IndexType rows(L.nrows());
        GraphBLAS::IndexType cols(L.ncols());

        MatrixT B(rows, cols);
        GraphBLAS::mxm(B, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                       GraphBLAS::ArithmeticSemiring<T>(),
                       L,
                       U);  /// @todo can't use transpose(L) here as LMatrix may
                            /// already be a TransposeView (nesting not supported)

        auto finish = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>
            (finish - start);
        start = finish;
        std::cout << "mxm elapsed time: " << duration.count() << " msec." << std::endl;

        T sum = 0;
        MatrixT C(rows, cols);
        GraphBLAS::eWiseMult(C, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                             GraphBLAS::Times<T>(),
                             L, B, true);

        GraphBLAS::reduce(sum, GraphBLAS::NoAccumulate(),
                          GraphBLAS::PlusMonoid<T>(), C);
        finish = std::chrono::steady_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>
            (finish - start);
        start = finish;
        std::cout << "count1 elapsed time: " << duration.count() << " msec." << std::endl;

        // for undirected graph you can stop here and return 'sum'

        GraphBLAS::eWiseMult(C, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                             GraphBLAS::Times<T>(),
                             U, B, true);

        GraphBLAS::reduce(sum, GraphBLAS::Plus<T>(),
                          GraphBLAS::PlusMonoid<T>(), C);
        finish = std::chrono::steady_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>
            (finish - start);
        std::cout << "count2 elapsed time: " << duration.count() << " msec." << std::endl;

        return sum / static_cast<T>(2);
    }

    //************************************************************************
    /**
     * From TzeMeng Low, the FLAME approach #1 to triangle counting using GBTL2
     *
     * @param[in] graph  Assumed to be undirected graph with no self-loops.
     *                   Only upper triangular portion of matrix accessed.
     */
    template<typename MatrixT>
    typename MatrixT::ScalarType triangle_count_flame1_newGBTL(MatrixT const &graph)
    {
        using T = typename MatrixT::ScalarType;
        GraphBLAS::IndexType rows(graph.nrows());
        GraphBLAS::IndexType cols(graph.ncols());

        if (rows != cols)
        {
            throw GraphBLAS::DimensionException(
                "triangle_count_flame1_new matrix is not square");
        }

        // A graph less than three vertices cannot have any triangles
        if (rows < 3)
        {
            return 0;
        }

        // "split" is not part of GraphBLAS...placeholder because Masking not
        // completely implemented
        MatrixT L(rows, cols), U(rows, cols);
        GraphBLAS::split(graph, L, U);

        GraphBLAS::IndexArrayType indices = {0};

        T delta(0UL);
        for (GraphBLAS::IndexType idx = 2; idx < rows; ++idx)
        {
            MatrixT A00(idx, idx);
            GraphBLAS::Vector<T> a01(idx);
            GraphBLAS::Vector<T> tmp1(idx);

            indices.push_back(idx - 1);   // [0, 1, ... i - 1]

            GraphBLAS::extract(A00,
                               GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                               U, indices, indices);
            GraphBLAS::extract(a01,
                               GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                               U, indices, idx);

            GraphBLAS::mxv(tmp1,
                           GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                           GraphBLAS::ArithmeticSemiring<T>(),
                           A00, a01);
            GraphBLAS::eWiseMult(tmp1,
                                 GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                 GraphBLAS::Times<T>(),
                                 tmp1, a01, true);
            GraphBLAS::reduce(delta, GraphBLAS::Plus<T>(),
                              GraphBLAS::PlusMonoid<T>(), tmp1);

            std::cout << "Processed row " << idx << " of " << rows
                      << ", Running count: " << delta << std::endl;
        }

        return delta;
    }

    //************************************************************************
    /**
     * From TzeMeng Low, the FLAME approach #2 to triangle counting
     *
     * @param[in] graph  Must be undirected graph with no self-loops; i.e.,
     *                   matrix is symmetric with zeros on the diagonal.
     *                   Only upper triangular portion is accessed.
     */
    template<typename MatrixT>
    typename MatrixT::ScalarType triangle_count_flame2_newGBTL(MatrixT const &graph)
    {
        using T = typename MatrixT::ScalarType;
        GraphBLAS::IndexType rows(graph.nrows());
        GraphBLAS::IndexType cols(graph.ncols());
        if (rows != cols)
        {
            throw GraphBLAS::DimensionException(
                "triangle_count_flame2 matrix is not square");
        }

        // A graph less than three vertices cannot have any triangles
        if (rows < 3)
        {
            return 0;
        }

        // the rows that a01 and A02 extract from (they grow)
        GraphBLAS::IndexArrayType row_indices;
        row_indices.reserve(rows);

        // the cols that a12 and A02 extract from (they shrink)
        GraphBLAS::IndexArrayType col_indices;
        col_indices.reserve(cols);
        for (GraphBLAS::IndexType idx = 1; idx < cols; ++idx)
        {
            col_indices.push_back(idx);
        }

        T delta(0UL);
        for (GraphBLAS::IndexType idx = 1; idx < rows - 1; ++idx)
        {
            // extract from the upper triangular portion of the adj. matrix only
            MatrixT A02(idx, cols - idx - 1);
            GraphBLAS::Vector<T> a01(idx);
            GraphBLAS::Vector<T> a12(cols - idx - 1);
            GraphBLAS::Vector<T> tmp1(idx);

            row_indices.push_back(idx - 1);   // [0, 1, ... i - 1]
            col_indices.erase(col_indices.begin());

            GraphBLAS::extract(A02, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                               graph, row_indices, col_indices, true);
            GraphBLAS::extract(a01, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                               graph, row_indices, idx, true);
            GraphBLAS::extract(a12, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                               GraphBLAS::transpose(graph), col_indices, idx, true);

            GraphBLAS::mxv(tmp1, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                           GraphBLAS::ArithmeticSemiring<T>(),
                           A02, a12, true);

            GraphBLAS::eWiseMult(tmp1, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                 GraphBLAS::Times<T>(),
                                 tmp1, a01, true);
            GraphBLAS::reduce(delta, GraphBLAS::Plus<T>(),
                              GraphBLAS::PlusMonoid<T>(), tmp1);

            std::cout << "Processed row " << idx << " of " << rows
                      << ", Running count: " << delta << std::endl;
        }

        return delta;
    }


    //************************************************************************
    /**
     * From TzeMeng Low, the FLAME approach #2 to triangle counting with masked access
     *
     * @param[in] graph  Must be undirected graph with no self-loops; i.e.,
     *                   matrix is symmetric with zeros on the diagonal.
     *                   Entire matrix is accessed.
     */
    template<typename MatrixT>
    typename MatrixT::ScalarType triangle_count_flame2_newGBTL_masked(MatrixT const &graph)
    {
        using T = typename MatrixT::ScalarType;
        GraphBLAS::IndexType rows(graph.nrows());
        GraphBLAS::IndexType cols(graph.ncols());
        if (rows != cols)
        {
            throw GraphBLAS::DimensionException(
                "triangle_count_flame2 matrix is not square");
        }

        // A graph less than three vertices cannot have any triangles
        if (rows < 3)
        {
            return 0;
        }

        GraphBLAS::IndexArrayType all_indices;
        all_indices.reserve(rows);
        for (GraphBLAS::IndexType ix = 0; ix < rows; ++ix)
        {
            all_indices.push_back(ix);
        }

        GraphBLAS::Vector<bool> mask(rows);
        mask.setElement(0, true);

        GraphBLAS::Vector<T> a(rows);
        GraphBLAS::Vector<T> a10(rows);
        GraphBLAS::Vector<T> a12(rows);
        GraphBLAS::Vector<T> tmp(rows);
        T delta(0UL);

        for (GraphBLAS::IndexType idx = 1; idx < rows - 1; ++idx)
        {
            // extract the whole column
            GraphBLAS::extract(a,
                               GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                               graph, all_indices, idx, true);

            // extract portions of row using mask.
            /// @todo try apply
            GraphBLAS::extract(a10,
                               mask, GraphBLAS::NoAccumulate(),
                               a, all_indices, true);

            mask.setElement(idx, true);
            GraphBLAS::extract(a12,
                               GraphBLAS::complement(mask), GraphBLAS::NoAccumulate(),
                               a, all_indices, true);

            GraphBLAS::mxv(tmp,
                           GraphBLAS::NoMask(),// mask,
                           GraphBLAS::NoAccumulate(),
                           GraphBLAS::ArithmeticSemiring<T>(),
                           graph, a12, true);

            GraphBLAS::eWiseMult(tmp, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                 GraphBLAS::Times<T>(),
                                 tmp, a10, true);

            GraphBLAS::reduce(delta, GraphBLAS::Plus<T>(),
                              GraphBLAS::PlusMonoid<T>(), tmp);

            std::cout << "Processed row " << idx << " of " << rows
                      << ", Running count: " << delta << std::endl;
        }

        return delta;
    }


    //************************************************************************
    /**
     * From TzeMeng Low, the blocked FLAME approach #2 to triangle counting
     * with masked access
     *
     * @param[in] graph  Must be undirected graph with no self-loops; i.e.,
     *                   matrix is symmetric with zeros on the diagonal.
     *                   Upper triangular portion of matrix is accessed.
     */
    template<typename MatrixT>
    typename MatrixT::ScalarType triangle_count_flame2_newGBTL_blocked(
        MatrixT const        &graph,
        GraphBLAS::IndexType  block_size = 128)
    {
        using T = typename MatrixT::ScalarType;
        T  num_triangles = 0;

        GraphBLAS::IndexType rows(graph.nrows());
        GraphBLAS::IndexType cols(graph.ncols());

        // assert(rows == cols);

        // A graph less than three vertices cannot have any triangles
        if (rows < 3)
        {
            return 0;
        }

        if (rows < 2*block_size)
        {
            return triangle_count_flame2_newGBTL(graph);
        }

        GraphBLAS::IndexType begin_index(0);
        GraphBLAS::IndexType end_index(block_size);

        GraphBLAS::IndexArrayType zero2begin; zero2begin.reserve(rows);
        GraphBLAS::IndexArrayType begin2end;  begin2end.reserve(block_size);
        GraphBLAS::IndexArrayType end2rows;   end2rows.reserve(rows);
        GraphBLAS::IndexArrayType end2rowsMask;   end2rows.reserve(rows);


        while (begin_index < rows)
        {
            GraphBLAS::IndexType bsize = end_index - begin_index;
            std::cerr << begin_index << ":" << end_index
                      << ", work = " << bsize << std::endl;

            zero2begin.clear();
            for (GraphBLAS::IndexType ix = 0; ix < begin_index; ++ix)
            {
                zero2begin.push_back(ix); // start with all indices
            }

            begin2end.clear();
            for (GraphBLAS::IndexType ix = begin_index; ix < end_index; ++ix)
            {
                begin2end.push_back(ix); // start with all indices
            }

            end2rows.clear();
            end2rowsMask.clear();
            for (GraphBLAS::IndexType ix = end_index; ix < rows; ++ix)
            {
                end2rows.push_back(ix); // start with all indices
                end2rowsMask.push_back(ix - end_index);
            }
            std::cerr << "0" << std::endl;

            MatrixT A01(begin_index, bsize);            // n x b
            MatrixT A02(begin_index, rows-end_index);   // n x m
            MatrixT A11(bsize, bsize);                  // b x b
            MatrixT A12(bsize, rows-end_index);         // b x m

            GraphBLAS::Matrix<bool> DiagonalMask(rows - end_index, rows - end_index);
            std::vector<bool> mvals(end2rowsMask.size(), true);
            DiagonalMask.build(end2rowsMask, end2rowsMask, mvals);
            std::cerr << "0" << std::endl;

            //
            GraphBLAS::extract(A01, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                               graph, zero2begin, begin2end, true);
            std::cerr << "a" << std::endl;
            GraphBLAS::extract(A11, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                               graph, begin2end, begin2end, true);
            std::cerr << "a" << std::endl;
            GraphBLAS::extract(A02, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                               graph, zero2begin, end2rows, true);
            std::cerr << "a" << std::endl;

            GraphBLAS::extract(A12, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                               graph, begin2end, end2rows, true);
            std::cerr << "0" << std::endl;

            // ***************************************************************
            // STEP 1: tris += trace(A12' * A01' * A02) = trace(A02' * A01 * A12);
            // ***************************************************************
            MatrixT Tmp1(rows - end_index, rows - end_index);
            if (begin_index > 0)
            {
                MatrixT Tmp0(begin_index, rows - end_index);

                GraphBLAS::mxm(Tmp0, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                               GraphBLAS::ArithmeticSemiring<T>(),
                               A01, A12, true);
                // Compute just the diagonal elements
                GraphBLAS::mxm(Tmp1, DiagonalMask, GraphBLAS::NoAccumulate(),
                               GraphBLAS::ArithmeticSemiring<T>(),
                               GraphBLAS::transpose(A02), Tmp0, true);
                GraphBLAS::reduce(num_triangles, GraphBLAS::Plus<T>(),
                                  GraphBLAS::PlusMonoid<T>(), Tmp1);
            }
            // ***************************************************************
            // STEP 2: tris += trace(A12' * A11 * A12)
            // ***************************************************************
            std::cerr << "2" << std::endl;
            MatrixT Tmp2(bsize, rows - end_index);

            GraphBLAS::mxm(Tmp2, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                           GraphBLAS::ArithmeticSemiring<T>(),
                           A11, A12, true);
            // Compute just the diagonal elements
            std::cerr << "2" << std::endl;
            GraphBLAS::mxm(Tmp1, DiagonalMask, GraphBLAS::NoAccumulate(),
                           GraphBLAS::ArithmeticSemiring<T>(),
                           GraphBLAS::transpose(A12), Tmp2, true);
            std::cerr << "2" << std::endl;
            GraphBLAS::reduce(num_triangles, GraphBLAS::Plus<T>(),
                              GraphBLAS::PlusMonoid<T>(), Tmp1);

            // ***************************************************************
            // STEP 3: tris += trace(A01 * A11 * A01')
            // ***************************************************************
            if (begin_index > 0)
            {
                MatrixT Tmp3(bsize, begin_index);
                MatrixT Tmp4(begin_index, begin_index);

                GraphBLAS::Matrix<bool> DiagonalMask2(begin_index, begin_index);
                std::vector<bool> mvals2(zero2begin.size(), true);
                DiagonalMask2.build(zero2begin, zero2begin, mvals2);
                std::cerr << "0" << std::endl;

                std::cerr << "3" << std::endl;
                GraphBLAS::mxm(Tmp3, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                               GraphBLAS::ArithmeticSemiring<T>(),
                               A11, GraphBLAS::transpose(A01), true);
                // Compute just the diagonal elements
                std::cerr << "3" << std::endl;
                GraphBLAS::mxm(Tmp4, DiagonalMask2, GraphBLAS::NoAccumulate(),
                               GraphBLAS::ArithmeticSemiring<T>(),
                               A01, Tmp3, true);
                std::cerr << "3" << std::endl;
                GraphBLAS::reduce(num_triangles, GraphBLAS::Plus<T>(),
                                  GraphBLAS::PlusMonoid<T>(), Tmp4);
            }

            // ***************************************************************
            // STEP 4: Count triangles in diagonal block
            // ***************************************************************
            std::cerr << "4" << std::endl;
            num_triangles += algorithms::triangle_count_newGBTL(
                GraphBLAS::transpose(A11), A11);

            std::cout << "Running count: " << num_triangles << std::endl;

            // set up for next block
            begin_index += block_size;
            end_index = std::min(rows, begin_index + block_size);
        }
        return num_triangles;
    }
} // algorithms

#endif // ALGORITHMS_TRIANGLE_COUNT_HPP
