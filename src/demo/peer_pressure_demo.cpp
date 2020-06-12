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

#include <iostream>

#include <graphblas/graphblas.hpp>
#include <algorithms/cluster.hpp>

using namespace grb;

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

    IndexArrayType i1 = {0, 0, 0, 0, 1, 1, 1, 1,
                         2, 2, 2, 2, 3, 3, 3, 3,
                         4, 4, 4, 4, 5, 5, 5, 5, 5,
                         6, 6, 6, 7, 7, 7, 7};
    IndexArrayType j1 = {0, 2, 3, 6, 1, 2, 3, 7,
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

    IndexArrayType i2 = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4};
    IndexArrayType j2 = {0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 3, 4};
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
