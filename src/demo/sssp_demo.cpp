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

#include <iostream>

#include <graphblas/graphblas.hpp>
#include <algorithms/sssp.hpp>

//****************************************************************************
template <typename T>
void run_demo()
{
    // FA Tech Note graph
    //
    //    {-, -, -, 1, -, -, -, -, -},
    //    {-, -, -, 1, -, -, 1, -, -},
    //    {-, -, -, -, 1, 1, 1, -, 1},
    //    {1, 1, -, -, 1, -, 1, -, -},
    //    {-, -, 1, 1, -, -, -, -, 1},
    //    {-, -, 1, -, -, -, -, -, -},
    //    {-, 1, 1, 1, -, -, -, -, -},
    //    {-, -, -, -, -, -, -, -, -},
    //    {-, -, 1, -, 1, -, -, -, -};

    T const INF(std::numeric_limits<T>::max());

    graphblas::IndexType const NUM_NODES = 9;
    graphblas::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    graphblas::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<T> v(i.size(), 1);

    graphblas::Matrix<T, graphblas::DirectedMatrixTag>
        G_tn(NUM_NODES, NUM_NODES, INF);

    graphblas::buildmatrix(G_tn, i.begin(), j.begin(), v.begin(), i.size());

    std::cout << "G_tn: zero = " << G_tn.get_zero() << std::endl;
    graphblas::pretty_print_matrix(std::cout, G_tn);

    auto ident = graphblas::identity<
        graphblas::Matrix<T, graphblas::DirectedMatrixTag > >(NUM_NODES,INF,0);
    std::cout << "ident: zero = " << ident.get_zero() << std::endl;
    graphblas::pretty_print_matrix(std::cout, ident);

    // compute all shortest paths
    graphblas::Matrix<T, graphblas::DirectedMatrixTag>
        G_tn_res(NUM_NODES, NUM_NODES, INF);
    algorithms::sssp(G_tn, ident, G_tn_res);

    std::cout << "Results:" << std::endl;
    graphblas::pretty_print_matrix(std::cout, G_tn_res);
}

//****************************************************************************
int main()
{
    std::cout << "ScalarType = double:" << std::endl;
    run_demo<double>();

    std::cout << "ScalarType = unsigned int" << std::endl;
    run_demo<unsigned int>();

    return 0;
}
