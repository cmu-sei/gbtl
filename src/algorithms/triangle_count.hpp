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

#define GB_DEBUG
#include <graphblas/graphblas.hpp>
#include <graphblas/linalg_utils.hpp>


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
        graphblas::IndexType rows, cols;
        graph.get_shape(rows, cols);

        MatrixT L(rows, cols), U(rows, cols);
        graphblas::split(graph, L, U);

        MatrixT B(rows, cols);
        graphblas::mxm(L, U, B);

        MatrixT C(rows, cols);
        graphblas::ewisemult(graph, B, C);

        T sum = 0;
        for (graphblas::IndexType i = 0; i< rows; ++i)
        {
            for (graphblas::IndexType j = 0; j < cols; ++j)
            {
                sum = sum + C.extractElement(i, j);
            }
        }
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
        graphblas::IndexType rows, cols;
        graph.get_shape(rows, cols);
        if (rows != cols)
        {
            throw graphblas::DimensionException(
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
        graphblas::split(graph, L, U);

        graphblas::IndexArrayType indices = {0};

        T delta(0UL);
        for (graphblas::IndexType idx = 2; idx < rows; ++idx)
        {
            MatrixT A00(idx, idx);
            MatrixT a01(idx, 1);   // old GBTL does not have Vectors
            MatrixT tmp1(idx, 1);  // old GBTL does not have Vectors
            MatrixT tmp2(1, 1);    // a matrix of one element != scalar

            graphblas::IndexArrayType column_index = {idx};
            indices.push_back(idx - 1);   // [0, 1, ... i - 1]

            graphblas::extract(U, indices, indices, A00);
            graphblas::extract(U, indices, column_index, a01);

            graphblas::mxv(A00, a01, tmp1,
                           graphblas::ArithmeticSemiring<T>());
            graphblas::vxm(transpose(a01), tmp1, tmp2,
                           graphblas::ArithmeticSemiring<T>());

            delta += tmp2.extractElement(0,0);
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
        graphblas::IndexType rows, cols;
        graph.get_shape(rows, cols);
        if (rows != cols)
        {
            throw graphblas::DimensionException(
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
        graphblas::split(graph, L, U);

        graphblas::IndexArrayType indices = {0};

        T delta(0UL);
        for (graphblas::IndexType idx = 2; idx < rows/2; ++idx)
        {
            MatrixT A00(idx, idx);
            MatrixT a01(idx, 1);   // old GBTL does not have Vectors
            MatrixT tmp1(idx, 1);  // old GBTL does not have Vectors
            MatrixT tmp2(1, 1);    // a matrix of one element != scalar

            graphblas::IndexArrayType column_index = {idx};
            indices.push_back(idx - 1);   // [0, 1, ... i - 1]

            graphblas::extract(U, indices, indices, A00);
            graphblas::extract(U, indices, column_index, a01);

            graphblas::mxv(A00, a01, tmp1,
                           graphblas::ArithmeticSemiring<T>());
            graphblas::vxm(transpose(a01), tmp1, tmp2,
                           graphblas::ArithmeticSemiring<T>());

            delta += tmp2.extractElement(0,0);
            std::cout << "Processed row " << idx << " of " << rows
                      << ", Running count: " << delta << std::endl;
        }

        for (graphblas::IndexType idx = rows/2; idx < rows; ++idx)
        {
            MatrixT A00(idx, idx);
            MatrixT a01(idx, 1);   // old GBTL does not have Vectors
            MatrixT tmp1(1, idx);  // old GBTL does not have Vectors
            MatrixT tmp2(1, 1);    // a matrix of one element != scalar

            graphblas::IndexArrayType column_index = {idx};
            indices.push_back(idx - 1);   // [0, 1, ... i - 1]

            graphblas::extract(U, indices, indices, A00);
            graphblas::extract(U, indices, column_index, a01);

            graphblas::vxm(transpose(a01), A00, tmp1,
                           graphblas::ArithmeticSemiring<T>());
            graphblas::vxm(tmp1, a01, tmp2,
                           graphblas::ArithmeticSemiring<T>());

            delta += tmp2.extractElement(0,0);
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
        graphblas::IndexType rows, cols;
        graph.get_shape(rows, cols);
        if (rows != cols)
        {
            throw graphblas::DimensionException(
                "triangle_count_flame2 matrix is not square");
        }

        // A graph less than three vertices cannot have any triangles
        if (rows < 3)
        {
            return 0;
        }

        // the rows that a01 and A02 extract from (they grow)
        graphblas::IndexArrayType row_indices;
        row_indices.reserve(rows);

        // the cols that a12 and A02 extract from (they shrink)
        graphblas::IndexArrayType col_indices;
        col_indices.reserve(cols);
        for (graphblas::IndexType idx = 1; idx < cols; ++idx)
        {
            col_indices.push_back(idx);
        }

        T delta(0UL);
        for (graphblas::IndexType idx = 1; idx < rows - 1; ++idx)
        {
            // extract from the upper triangular portion of the adj. matrix only
            MatrixT A02(idx, cols - idx - 1);
            MatrixT a01(idx, 1);              // old GBTL does not have Vectors
            MatrixT a12(1, cols - idx - 1);   // old GBTL does not have Vectors
            MatrixT tmp1(cols - idx - 1, 1);  // old GBTL does not have Vectors
            MatrixT tmp2(1, 1);    // a matrix of one element != scalar

            graphblas::IndexArrayType pivot_index = {idx};
            row_indices.push_back(idx - 1);   // [0, 1, ... i - 1]
            col_indices.erase(col_indices.begin());

            graphblas::extract(graph, row_indices, col_indices, A02);
            graphblas::extract(graph, row_indices, pivot_index, a01);
            graphblas::extract(graph, pivot_index, col_indices, a12);

            graphblas::mxv(transpose(A02), a01, tmp1,
                           graphblas::ArithmeticSemiring<T>());

            /// @todo replace vxm+extractElement with eWiseMult(vec)+reduce(scalar)
            graphblas::vxm(a12, tmp1, tmp2,
                           graphblas::ArithmeticSemiring<T>());
            delta += tmp2.extractElement(0,0);

            // DEBUG
            //std::cerr << "************** Iteration " << idx << " **************"
            //          << std::endl;
            //std::cerr << "A02 dimensions = " << idx << " x " << (cols-idx-1)
            //          << std::endl;
            //graphblas::print_matrix(std::cerr, a01, "a01");
            //graphblas::print_matrix(std::cerr, A02, "A02");
            //graphblas::print_matrix(std::cerr, a12, "a12");
            //std::cerr << "delta = " << delta << std::endl;

            std::cout << "Processed row " << idx << " of " << rows
                      << ", Running count: " << delta << std::endl;
        }

        return delta;
    }


    //************************************************************************
    template<typename MatrixT>
    typename MatrixT::ScalarType triangle_count_newGBTL(MatrixT const &L,
                                                        MatrixT const &U)
    {
        using T = typename MatrixT::ScalarType;
        GraphBLAS::IndexType rows(L.nrows());
        GraphBLAS::IndexType cols(L.ncols());

        MatrixT B(rows, cols);
        GraphBLAS::mxm(B, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                       GraphBLAS::ArithmeticSemiring<T>(),
                       L, U, true);

        T sum = 0;
        MatrixT C(rows, cols);
        GraphBLAS::eWiseMult(C, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                             GraphBLAS::Times<T>(),
                             L, B, true);

        GraphBLAS::reduce(sum, GraphBLAS::NoAccumulate(),
                          GraphBLAS::PlusMonoid<T>(), C);

        // for undirected graph you can stop here and return 'sum'

        GraphBLAS::eWiseMult(C, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                             GraphBLAS::Times<T>(),
                             U, B, true);

        GraphBLAS::reduce(sum, GraphBLAS::Plus<T>(),
                          GraphBLAS::PlusMonoid<T>(), C);

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

        GraphBLAS::IndexArrayType indices = {0};

        T delta(0UL);
        for (GraphBLAS::IndexType idx = 2; idx < rows; ++idx)
        {
            MatrixT A00(idx, idx);
            GraphBLAS::Vector<T> a01(idx);
            GraphBLAS::Vector<T> tmp1(idx);

            GraphBLAS::IndexArrayType column_index = {idx};
            indices.push_back(idx - 1);   // [0, 1, ... i - 1]

            GraphBLAS::extract(A00, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                               graph, indices, indices, true);
            GraphBLAS::extract(a01, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                               graph, indices, idx, true);

            GraphBLAS::mxv(tmp1, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                           GraphBLAS::ArithmeticSemiring<T>(),
                           A00, a01, true);
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
} // algorithms

#endif // METRICS_HPP
