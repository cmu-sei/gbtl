/*
 * GraphBLAS Template Library, Version 2.0
 *
 * Copyright 2018 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors. All Rights Reserved.
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
 * RIGHTS..
 *
 * Released under a BSD (SEI)-style license, please see license.txt or contact
 * permission@sei.cmu.edu for full terms.
 *
 * This release is an update of:
 *
 * 1. GraphBLAS Template Library (GBTL)
 * (https://github.com/cmu-sei/gbtl/blob/1.0.0/LICENSE) Copyright 2015 Carnegie
 * Mellon University and The Trustees of Indiana. DM17-0037, DM-0002659
 *
 * DM18-0559
 */

#include <iostream>
#include <fstream>
#include <chrono>

#define GRAPHBLAS_DEBUG 1

#include <graphblas/graphblas.hpp>
#include <algorithms/triangle_count.hpp>

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
    GraphBLAS::IndexArrayType iL, iU, iA;
    GraphBLAS::IndexArrayType jL, jU, jA;
    uint64_t num_rows = 0;
    uint64_t max_id = 0;
    uint64_t src, dst;

    while (infile)
    {
        infile >> src >> dst;
        //std::cout << "Read: " << src << ", " << dst << std::endl;
        if (src > max_id) max_id = src;
        if (dst > max_id) max_id = dst;

        if (src < dst)
        {
            iA.push_back(src);
            jA.push_back(dst);

            iU.push_back(src);
            jU.push_back(dst);
        }
        else if (dst < src)
        {
            iA.push_back(src);
            jA.push_back(dst);

            iL.push_back(src);
            jL.push_back(dst);
        }
        // else ignore self loops

        ++num_rows;
    }
    std::cout << "Read " << num_rows << " rows." << std::endl;
    std::cout << "#Nodes = " << (max_id + 1) << std::endl;

    GraphBLAS::IndexType NUM_NODES(max_id + 1);
    typedef int32_t T;
    std::vector<T> v(iA.size(), 1);

    /// @todo change scalar type to unsigned int or GraphBLAS::IndexType
    typedef GraphBLAS::Matrix<T, GraphBLAS::DirectedMatrixTag> MatType;

    MatType A(NUM_NODES, NUM_NODES);
    MatType L(NUM_NODES, NUM_NODES);
    MatType U(NUM_NODES, NUM_NODES);

    A.build(iA.begin(), jA.begin(), v.begin(), iA.size());
    L.build(iL.begin(), jL.begin(), v.begin(), iL.size());
    U.build(iU.begin(), jU.begin(), v.begin(), iU.size());

    std::cout << "Running algorithm(s)..." << std::endl;
    T count(0);

    auto start = std::chrono::steady_clock::now();

    // Perform triangle counting with three different algorithms
    count = algorithms::triangle_count_newGBTL(L, U);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>
        (std::chrono::steady_clock::now() - start);

    std::cout << "# triangles = " << count << std::endl;
    std::cout << "Elapsed time: " << duration.count() << " msec." << std::endl;

    start = std::chrono::steady_clock::now();

    count = algorithms::triangle_count_masked(L);

    duration = std::chrono::duration_cast<std::chrono::milliseconds>
        (std::chrono::steady_clock::now() - start);
    std::cout << "# triangles (masked) = " << count << std::endl;
    std::cout << "Elapsed time: " << duration.count() << " msec." << std::endl;

    //count = algorithms::triangle_count_flame1_newGBTL(U);
    //std::cout << "# triangles = " << count << std::endl;

    //count = algorithms::triangle_count_flame2_newGBTL(U);
    //std::cout << "# triangles = " << count << std::endl;

    //count = algorithms::triangle_count_flame2_newGBTL_masked(A);
    //std::cout << "# triangles = " << count << std::endl;
    //return 0;

    //count = algorithms::triangle_count_flame2_newGBTL_blocked(U, 256UL);
    //std::cout << "# triangles = " << count << std::endl;
    return 0;
}
