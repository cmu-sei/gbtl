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
#include <fstream>
#include <chrono>

#define GRAPHBLAS_DEBUG 1

#include <graphblas/graphblas.hpp>
#include <algorithms/cluster_louvain.hpp>
#include "Timer.hpp"

//****************************************************************************
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "ERROR: too few arguments." << std::endl;
        exit(1);
    }

    // Read the edgelist and create the tuple arrays
    std::string pathname(argv[1]);
    std::ifstream infile(pathname);
    grb::IndexArrayType iA;
    grb::IndexArrayType jA;
    std::vector<double> weights;
    uint64_t num_rows = 0;
    uint64_t max_id = 0;
    uint64_t src, dst, weight;
    while (infile)
    {
        infile >> src >> dst >> weight;
        src -= 1;
        dst -= 1;
        //std::cout << "Read: " << src << ", " << dst << std::endl;
        if (src > max_id) max_id = src;
        if (dst > max_id) max_id = dst;

        // if (src != dst)  // ignore self loops
        iA.push_back(src);
        jA.push_back(dst);
        weights.push_back((double)weight);

        ++num_rows;
    }
    std::cout << "Read " << num_rows << " rows." << std::endl;
    std::cout << "#Nodes = " << (max_id + 1) << std::endl;

    grb::IndexType NUM_NODES(max_id + 1);

    using MatType = grb::Matrix<double>;

    MatType A(NUM_NODES, NUM_NODES);
    A.build(iA.begin(), jA.begin(), weights.begin(), iA.size());

    std::cout << "Running louvain clustering..." << std::endl;

    Timer<std::chrono::steady_clock> my_timer;

    // Perform clustering with 2 different algorithms
    //===================
    my_timer.start();
    auto cluster_matrix = algorithms::louvain_cluster(A);
    my_timer.stop();

    std::cout << "Elapsed time: " << my_timer.elapsed() << " msec." << std::endl;
    auto cluster_assignments =
        algorithms::get_louvain_cluster_assignments(cluster_matrix);
    print_vector(std::cout, cluster_assignments, "cluster assignments");

    //===================
    my_timer.start();
    auto cluster2_matrix = algorithms::louvain_cluster_masked(A);
    my_timer.stop();

    std::cout << "Elapsed time: " << my_timer.elapsed() << " msec." << std::endl;
    auto cluster2_assignments =
        algorithms::get_louvain_cluster_assignments(cluster2_matrix);
    print_vector(std::cout, cluster2_assignments, "cluster (masked)  assignments");

    return 0;
}
