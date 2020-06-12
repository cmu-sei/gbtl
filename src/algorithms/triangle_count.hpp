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
#include <chrono>

#include <graphblas/graphblas.hpp>

//****************************************************************************
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
     * <li>First, split \f$graph\f$ into lower and upper triangular matrices such
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
     * @param[in]  graph  The graph to compute the number of triangles in.
     *
     * @return The number of triangles in graph.
     */
    template<typename MatrixT>
    typename MatrixT::ScalarType triangle_count(MatrixT const &graph)
    {
        using T = typename MatrixT::ScalarType;
        grb::IndexType rows(graph.nrows());
        grb::IndexType cols(graph.ncols());

        MatrixT L(rows, cols), U(rows, cols);
        grb::split(graph, L, U);

        MatrixT B(rows, cols);
        grb::mxm(B, grb::NoMask(), grb::NoAccumulate(),
                 grb::ArithmeticSemiring<T>(), L, U);

        MatrixT C(rows, cols);
        grb::eWiseMult(C, grb::NoMask(), grb::NoAccumulate(),
                       grb::Times<T>(), graph, B);

        T sum = 0;
        grb::reduce(sum, grb::NoAccumulate(), grb::PlusMonoid<T>(), C);
        return sum / static_cast<T>(2);
    }

    //************************************************************************
    /**
     * @brief Compute the number of triangles in an undirected graph already
     *        split using |L.*(L +.* U)|.
     */
    template<typename MatrixT>
    typename MatrixT::ScalarType triangle_count_masked(MatrixT const &L,
                                                       MatrixT const &U)
    {
        using T = typename MatrixT::ScalarType;
        grb::IndexType rows(L.nrows());
        grb::IndexType cols(L.ncols());

        MatrixT B(rows, cols);
        grb::mxm(B, L, grb::NoAccumulate(), grb::ArithmeticSemiring<T>(), L, U);

        T sum = 0;
        grb::reduce(sum, grb::NoAccumulate(), grb::PlusMonoid<T>(), B);
        return sum;
    }

    //************************************************************************
    /**
     * @brief Compute the number of triangles in an undirected graph already
     *        split using |L.*(L +.* L')|.
     */
    template<typename MatrixT>
    typename MatrixT::ScalarType triangle_count_masked(MatrixT const &L)
    {
        using T = typename MatrixT::ScalarType;
        grb::IndexType rows(L.nrows());
        grb::IndexType cols(L.ncols());

        MatrixT B(rows, cols);
        grb::mxm(B, L, grb::NoAccumulate(),
                 grb::ArithmeticSemiring<T>(), L, grb::transpose(L));

        T sum = 0;
        grb::reduce(sum, grb::NoAccumulate(), grb::PlusMonoid<T>(), B);
        return sum;
    }

    //************************************************************************
    /**
     * @brief Compute the number of triangles in an undirected graph already
     *        split using |L.*(L +.* L)|.
     */
    template<typename MatrixT>
    typename MatrixT::ScalarType triangle_count_masked_noT(MatrixT const &L)
    {
        using T = typename MatrixT::ScalarType;
        grb::IndexType rows(L.nrows());
        grb::IndexType cols(L.ncols());

        MatrixT B(rows, cols);
        grb::mxm(B, L, grb::NoAccumulate(), grb::ArithmeticSemiring<T>(), L, L);

        T sum = 0;
        grb::reduce(sum, grb::NoAccumulate(), grb::PlusMonoid<T>(), B);
        return sum;
    }

    //************************************************************************
    /**
     * @brief Compute the number of triangles in an undirected graph already
     *        split using
     *            B   = (L +.* U),
     *            CL  = (L .* B),
     *            CU  = (U .* B),
     *            sum = (|CU|+|CL|)/2
     */
    template<typename LMatrixT, typename MatrixT>
    typename MatrixT::ScalarType triangle_count_newGBTL(LMatrixT const &L,
                                                        MatrixT  const &U)
    {
        using T = typename MatrixT::ScalarType;
        grb::IndexType rows(L.nrows());
        grb::IndexType cols(L.ncols());

        MatrixT B(rows, cols);
        grb::mxm(B, grb::NoMask(), grb::NoAccumulate(),
                 grb::ArithmeticSemiring<T>(),
                 L,
                 U);  /// @todo can't use transpose(L) here as LMatrix may
                      /// already be a TransposeView (nesting not supported)

        T sum = 0;
        MatrixT C(rows, cols);
        grb::eWiseMult(C, grb::NoMask(), grb::NoAccumulate(),
                       grb::Times<T>(), L, B, grb::REPLACE);

        grb::reduce(sum, grb::NoAccumulate(), grb::PlusMonoid<T>(), C);

        // for undirected graph you can stop here and return 'sum'

        grb::eWiseMult(C, grb::NoMask(), grb::NoAccumulate(),
                       grb::Times<T>(), U, B, grb::REPLACE);

        grb::reduce(sum, grb::Plus<T>(), grb::PlusMonoid<T>(), C);

        return sum / static_cast<T>(2);
    }
} // algorithms
