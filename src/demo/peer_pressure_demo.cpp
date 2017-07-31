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
#include <algorithms/cluster.hpp>

using namespace GraphBLAS;

//****************************************************************************
int main()
{
    // {{1, 0, 1, 1, 0, 0, 1, 0},
    //  {0, 1, 1, 1, 0, 0, 0, 1},
    //  {1, 0, 1, 0, 1, 0, 1, 0},
    //  {1, 1, 0, 1, 0, 1, 0, 0},
    //  {1, 0, 1, 0, 1, 0, 1, 0},
    //  {0, 1, 0, 1, 0, 1, 1, 1},
    //  {1, 0, 0, 0, 1, 0, 1, 0},
    //  {0, 1, 0, 1, 0, 1, 0, 1}}

    VectorIndexType i1 = {0, 0, 0, 0, 1, 1, 1, 1,
                         2, 2, 2, 2, 3, 3, 3, 3,
                         4, 4, 4, 4, 5, 5, 5, 5, 5,
                         6, 6, 6, 7, 7, 7, 7};
    VectorIndexType j1 = {0, 2, 3, 6, 1, 2, 3, 7,
                         0, 2, 4, 6, 0, 1, 3, 5,
                         0, 2, 4, 6, 1, 3, 5, 6, 7,
                         0, 4, 6, 1, 3, 5, 7};
    std::vector<float> v1(i1.size(), 1.0);
    Matrix<float> G_m1(8, 8);
    G_m1.build(i1.begin(), j1.begin(), v1.begin(), i1.size());

    auto ans = algorithms::peer_pressure_cluster(G_m1);
    print_matrix(std::cout, ans, "G_m1 cluster matrix");

    auto clusters = algorithms::get_cluster_assignments(ans);
    std::cout << "cluster assignments:";
    for (auto it = clusters.begin(); it != clusters.end(); ++it)
    {
        std::cout << " " << *it;
    }
    std::cout << std::endl;

    // {{1, 1, 1, 1, 0},
    //  {1, 1, 1, 0, 0},
    //  {1, 1, 1, 0, 0},
    //  {0, 0, 0, 1, 1},
    //  {0, 0, 0, 1, 1}}

    VectorIndexType i2 = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4};
    VectorIndexType j2 = {0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 3, 4};
    std::vector<float> v2(i2.size(), 1.0);
    Matrix<float> G_m2(5, 5);
    G_m2.build(i2.begin(), j2.begin(), v2.begin(), i2.size());

    auto ans2 = algorithms::peer_pressure_cluster(G_m2);
    print_matrix(std::cout, ans2, "G_m2 cluster matrix");

    auto clusters2 = algorithms::get_cluster_assignments(ans2);
    std::cout << "cluster assignments:";
    for (auto it = clusters2.begin(); it != clusters2.end(); ++it)
    {
        std::cout << " " << *it;
    }
    std::cout << std::endl;

    return 0;
}
