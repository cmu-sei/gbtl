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

//****************************************************************************
grb::IndexType const NUM_NODES = 34;

grb::IndexArrayType i = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,
    2,2,2,2,2,2,2,2,2,2,
    3,3,3,3,3,3,
    4,4,4,
    5,5,5,5,
    6,6,6,6,
    7,7,7,7,
    8,8,8,8,8,
    9,9,
    10,10,10,
    11,
    12,12,
    13,13,13,13,13,
    14,14,
    15,15,
    16,16,
    17,17,
    18,18,
    19,19,19,
    20,20,
    21,21,
    22,22,
    23,23,23,23,23,
    24,24,24,
    25,25,25,
    26,26,
    27,27,27,27,
    28,28,28,
    29,29,29,29,
    30,30,30,30,
    31,31,31,31,31,
    32,32,32,32,32,32,32,32,32,32,32,32,
    33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33};

grb::IndexArrayType j = {
    1,2,3,4,5,6,7,8,10,11,12,13,17,19,21,31,     //1,2,3,4,5,6,7,8,10,11,12,13,19,21,23,31,
    0,2,3,7,13,17,19,21,30,
    0,1,3,7,8,9,13,27,28,32,
    0,1,2,7,12,13,
    0,6,10,
    0,6,10,16,
    0,4,5,16,
    0,1,2,3,
    0,2,30,32,33,
    2,33,
    0,4,5,
    0,
    0,3,
    0,1,2,3,33,
    32,33,
    32,33,
    5,6,
    0,1,
    32,33,
    0,1,33,
    32,33,
    0,1,
    32,33,
    25,27,29,32,33,
    25,27,31,
    23,24,31,
    29,33,
    2,23,24,33,
    2,31,33,
    23,26,32,33,
    1,8,32,33,
    0,24,25,32,33,    //0,24,25,28,32,33,
    2,8,14,15,18,20,22,23,29,30,31,33,
    8,9,13,14,15,18,19,20,22,23,26,27,28,29,30,31,32};

//****************************************************************************
int main(int, char**)
{
    grb::Matrix<uint32_t> A_karate(NUM_NODES, NUM_NODES);
    grb::Matrix<uint32_t> G_karate(NUM_NODES, NUM_NODES);
    if (i.size() != j.size())
    {
        std::cerr << "Index arrays are not the same size: " << i.size()
                  << " != " << j.size() << std::endl;
        return -1;
    }

    std::vector<double> weights(i.size(), 1.0);
    G_karate.build(i.begin(), j.begin(), weights.begin(), i.size());

    auto Clusters =
        grb::scaled_identity<grb::Matrix<bool>>(NUM_NODES);

    // Add self-loops to graph
    grb::eWiseAdd(A_karate,
                  grb::NoMask(), grb::NoAccumulate(),
                  grb::Plus<uint32_t>(),
                  G_karate, Clusters);

    algorithms::peer_pressure_cluster_v2(A_karate, Clusters, 100);

    auto clusters = algorithms::get_cluster_assignments(Clusters);
    std::cout << "p.p. cluster assignments (no normalization): ";
    for (auto it = clusters.begin(); it != clusters.end(); ++it)
    {
        std::cout << " " << *it;
    }
    std::cout << std::endl;

    auto Clusters2 = grb::scaled_identity<grb::Matrix<bool>>(NUM_NODES);

    algorithms::peer_pressure_cluster(A_karate, Clusters2, 100);

    auto clusters2 = algorithms::get_cluster_assignments(Clusters2);
    std::cout << "cluster assignments (with normalization):    ";
    for (auto it = clusters2.begin(); it != clusters2.end(); ++it)
    {
        std::cout << " " << *it;
    }
    std::cout << std::endl;
    return 0;
}
