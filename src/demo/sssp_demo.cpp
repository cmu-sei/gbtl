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

    grb::IndexType const NUM_NODES = 9;
    grb::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    grb::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<T> v(i.size(), 1);

    grb::Matrix<T> G_tn(NUM_NODES, NUM_NODES);

    G_tn.build(i.begin(), j.begin(), v.begin(), i.size());
    grb::print_matrix(std::cout, G_tn, "Graph");

    // compute one shortest paths
    grb::Vector<T> path(NUM_NODES);
    path.setElement(0, 0);
    grb::print_vector(std::cout, path, "Source");
    algorithms::sssp(G_tn, path);
    grb::print_vector(std::cout, path, "single SSSP results");

    // compute all shortest paths
    auto paths = grb::scaled_identity<grb::Matrix<T>>(NUM_NODES, 0);
    grb::print_matrix(std::cout, paths, "Sources");
    algorithms::batch_sssp(G_tn, paths);
    grb::print_matrix(std::cout, paths, "batch SSSP results");
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
