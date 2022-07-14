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
#include <algorithms/bfs.hpp>
#include <algorithms/bfs_do.hpp>
#include "Timer.hpp"
#include "read_edge_list.hpp"

//****************************************************************************
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "ERROR: too few arguments. Provide edge list file with triplets."
                  << std::endl;
        exit(1);
    }

    // Read the edgelist and create the tuple arrays
    std::string pathname(argv[1]);

    Timer<std::chrono::steady_clock, std::chrono::microseconds> my_timer;

    /// @todo revisit scalar type
    using T = uint16_t;

    grb::IndexArrayType i, j;
    std::vector<T> v;
    my_timer.start();
    grb::IndexType const NUM_NODES(read_triples<T>(argv[1], i, j, v, true));
    my_timer.stop();

    std::cout << "Elapsed read time: " << my_timer.elapsed() << " usec." << std::endl;
    std::cout << "#Edges = " << v.size()  << std::endl;
    std::cout << "#Nodes = " << NUM_NODES << std::endl;

    grb::Matrix<T> graph(NUM_NODES, NUM_NODES);
    graph.build(i.begin(), j.begin(), v.begin(), i.size());
    std::cerr << "Input adjacency matrix, nvals() = " << graph.nvals()
              << std::endl;

    // Force it to be symmetric
    grb::eWiseAdd(graph, grb::NoMask(), grb::NoAccumulate(),
                  grb::BitwiseOr<T>(), graph, transpose(graph));

    //grb::print_matrix(std::cout, graph, "Symmetric adjacency matrix:");
    std::cerr << "Symmetric adjacency matrix, nvals() = " << graph.nvals()
              << std::endl;

    // Perform a single BFS
    grb::Vector<T> parent_list(NUM_NODES);
    my_timer.start();
    algorithms::bfs(graph, 3UL, parent_list);
    my_timer.stop();
    std::cout << "Elapsed BFS time: " << my_timer.elapsed() << " usec." << std::endl;
    //grb::print_vector(std::cout, parent_list, "Parent list for root at vertex 3");
    std::cout << "parent_list.nvals() = " << parent_list.nvals() << std::endl;

    // Perform a single BFS-DO
    grb::Vector<T> do_parent_list(NUM_NODES);
    my_timer.start();
    algorithms::bfs_do_parents(graph, graph, 3, do_parent_list);
    my_timer.stop();
    std::cout << "Elapsed BFS-DO time: " << my_timer.elapsed() << " usec." << std::endl;
    //grb::print_vector(std::cout, parent_list, "Parent list for root at vertex 3");
    std::cout << "do_parent_list.nvals() = " << do_parent_list.nvals() << std::endl;


    // Check answers
    bool passed = true;
    for (grb::IndexType ix = 0; ix < NUM_NODES; ++ix)
    {
        if (do_parent_list.hasElement(ix) != parent_list.hasElement(ix))
        {
            std::cerr << ix << ": missing element\n";
            passed = false;
        }
        if (do_parent_list.hasElement(ix) &&
            (do_parent_list.extractElement(ix) != parent_list.extractElement(ix)))
        {
            std::cerr << ix << ": parent mismatch: "
                      << do_parent_list.extractElement(ix) << " != "
                      << parent_list.extractElement(ix) << std::endl;
            passed = false;
        }
    }
    if (!passed)
    {
        std::cerr << "ERROR: Parent lists did not match.\n";
    }
    else
    {
        std::cerr << "Parent lists match.\n";
    }
    return passed ? 0 : 1;
}
