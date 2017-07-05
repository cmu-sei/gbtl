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
#include <cstdio>

#include <graphblas/graphblas.hpp>
#include <algorithms/k_truss.hpp>

//****************************************************************************
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "ERROR: too few arguments." << std::endl;
    }

    // Read the edgelist and create the tuple arrays
    std::string pathname(argv[1]);
    GraphBLAS::IndexArrayType iE, iA;
    GraphBLAS::IndexArrayType jE, jA;
    GraphBLAS::IndexType max_id = 0;
    GraphBLAS::IndexType num_edges = 0;

    GraphBLAS::IndexType src, dst;

    std::ifstream infile(pathname);
    while (infile >> src >> dst)
    {
        std::cout << "Read: " << src << ", " << dst << std::endl;
        if (src != dst) // ignore self loops
        {
            max_id = std::max(src, max_id);
            max_id = std::max(dst, max_id);

            //iA.push_back(src);
            //jA.push_back(dst);

            iE.push_back(num_edges);
            jE.push_back(src);
            iE.push_back(num_edges);
            jE.push_back(dst);

            ++num_edges;
        }
    }
    std::cout << "Read " << num_edges << " edges." << std::endl;
    std::cout << "#Nodes = " << (max_id + 1) << std::endl;

    GraphBLAS::IndexType NUM_NODES(max_id + 1);
    typedef int32_t T;
    std::vector<T> v(iE.size(), 1);

    /// @todo change scalar type to unsigned int or GraphBLAS::IndexType
    typedef GraphBLAS::Matrix<T, GraphBLAS::DirectedMatrixTag> MatType;

    //MatType A(NUM_NODES, NUM_NODES);
    MatType E(num_edges, NUM_NODES);
    MatType Eout(num_edges, NUM_NODES);

    E.build(iE, jE, v);

    std::cout << "Running algorithm(s)..." << std::endl;
    T count(0);

    algorithms::k_truss(Eout, E, 3);
    //std::cout << "# triangles = " << count << std::endl;
    return 0;
}
