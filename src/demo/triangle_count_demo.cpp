/*
 * Copyright (c) 2017 Carnegie Mellon University.
 * All Rights Reserved.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS," WITH NO WARRANTIES WHATSOEVER. CARNEGIE
 * MELLON UNIVERSITY EXPRESSLY DISCLAIMS
 * TO THE FULLEST EXTENT PERMITTED BY LAW ALL EXPRESS, IMPLIED, AND STATUTORY
 * WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT OF PROPRIETARY RIGHTS.
 *
 * This Program is distributed under a BSD license.  Please see LICENSE file or
 * permission@sei.cmu.edu for more information.  DM-0002659
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
    GraphBLAS::VectorIndexType iL, iU, iA;
    GraphBLAS::VectorIndexType jL, jU, jA;
    int64_t num_rows = 0;
    int64_t max_id = 0;
    uint64_t src, dst;
//    for (std::string row; getline(infile, row, '\n');)
//    {
//        std::cout << "Row: " << row << std::endl;
//        sscanf(row.c_str(), "%ld\t%ld", &src, &dst);
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
