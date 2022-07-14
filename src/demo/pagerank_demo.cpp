/*
 * GraphBLAS Template Library (GBTL), Version 3.0
 *
 * Copyright 2021 Carnegie Mellon University, Battelle Memorial Institute, and
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

#include <graphblas/graphblas.hpp>
#include <algorithms/page_rank.hpp>
#include "Timer.hpp"
#include "read_edge_list.hpp"

//****************************************************************************
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "ERROR: too few arguments. Provide edge list file with triplets." << std::endl;
        exit(1);
    }

    unsigned int const NUM_ITERS = 100;
    Timer<std::chrono::steady_clock, std::chrono::microseconds> my_timer;

    // Read the edgelist and create the graph
    using T = double;
    grb::IndexArrayType i, j;
    std::vector<T> v;
    my_timer.start();

    grb::IndexType const NUM_NODES(read_triples<T>(argv[1], i, j, v));

    grb::Matrix<T> graph(NUM_NODES, NUM_NODES);
    graph.build(i.begin(), j.begin(), v.begin(), i.size());

    my_timer.stop();

    std::cout << "Elapsed read time: " << my_timer.elapsed() << " usec.\n";
    std::cout << "#Nodes = " << graph.nrows() << std::endl;
    std::cout << "#Edges = " << graph.nvals() << std::endl;

    // Perform page rank
    grb::Vector<T> PR(graph.nrows());
    my_timer.start();
    algorithms::page_rank(graph, PR, 0.85, 1.e-8, NUM_ITERS);
    my_timer.stop();
    std::cout << "Elapsed compute time: " << my_timer.elapsed() << " usec.\n";
    grb::print_vector(std::cout, PR, "PR");

    unsigned int num_iters;
    grb::Vector<T> PR_gap(graph.nrows());
    my_timer.start();
    algorithms::pagerank_gap(graph, PR_gap, num_iters, 0.85, 1.e-8, NUM_ITERS);
    my_timer.stop();
    std::cout << "Elapsed GAP compute time: " << my_timer.elapsed() << " usec.\n";
    std::cout << "Number of iterations: " << num_iters << std::endl;
    grb::print_vector(std::cout, PR_gap, "PR_gap");

    return 0;
}
