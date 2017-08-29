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

    GraphBLAS::IndexType const NUM_NODES = 9;
    GraphBLAS::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    GraphBLAS::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<T> v(i.size(), 1);

    GraphBLAS::Matrix<T> G_tn(NUM_NODES, NUM_NODES);

    G_tn.build(i.begin(), j.begin(), v.begin(), i.size());
    GraphBLAS::print_matrix(std::cout, G_tn, "Graph");

    // compute one shortest paths
    GraphBLAS::Vector<T> path(NUM_NODES);
    path.setElement(0, 0);
    GraphBLAS::print_vector(std::cout, path, "Source");
    algorithms::sssp(G_tn, path);
    GraphBLAS::print_vector(std::cout, path, "single SSSP results");

    // compute all shortest paths
    auto paths = GraphBLAS::scaled_identity<GraphBLAS::Matrix<T>>(NUM_NODES, 0);
    GraphBLAS::print_matrix(std::cout, paths, "Sources");
    algorithms::batch_sssp(G_tn, paths);
    GraphBLAS::print_matrix(std::cout, paths, "batch SSSP results");
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
