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
#include <algorithms/cc.hpp>
#include "Timer.hpp"

//****************************************************************************
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "ERROR: too few arguments. Provide edge list file with triplets." << std::endl;
        exit(1);
    }

    // Read the edgelist and create the tuple arrays
    std::string pathname(argv[1]);

    Timer<std::chrono::steady_clock, std::chrono::microseconds> my_timer;

    /// @todo revisit scalar type
    using T = uint32_t;

    grb::IndexArrayType i, j;
    std::vector<T> v;
    uint64_t num_rows = 0;
    uint64_t max_id = 0;

    my_timer.start();
    {
        std::ifstream infile(pathname);
        while (infile)
        {
            uint64_t src, dst;
            T val;
            infile >> src >> dst >> val;
            //std::cout << "Read: " << src << ", " << dst << ", " << val << std::endl;
            if (src != dst)
            {
                if (src > max_id) max_id = src;
                if (dst > max_id) max_id = dst;

                i.push_back(src);
                j.push_back(dst);
                v.push_back(val);
            }
            // else ignore self loops

            ++num_rows;
        }
    }
    grb::IndexType NUM_NODES = max_id + 1;

    my_timer.stop();
    std::cout << "Elapsed read time: " << my_timer.elapsed() << " usec." << std::endl;

    std::cout << "Read " << num_rows << " rows." << std::endl;
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

    // Perform connected components
    grb::Vector<T> component_ids(NUM_NODES);
    my_timer.start();
    auto num_ccs = algorithms::cc(graph, component_ids);
    my_timer.stop();
    std::cout << "Elapsed read time: " << my_timer.elapsed() << " usec." << std::endl;
    grb::print_vector(std::cout, component_ids, "component IDs");
    std::cout << "number of components = " << num_ccs << std::endl;

    return 0;
}
