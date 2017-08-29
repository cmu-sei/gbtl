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

using namespace GraphBLAS;

//****************************************************************************
namespace
{
    GraphBLAS::IndexArrayType i = {
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

    GraphBLAS::IndexArrayType j = {
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

}

//****************************************************************************
int main(int argc, char **argv)
{
    typedef int T;

    // create an incidence matrix from the data
    IndexType num_edges = 0;
    IndexType num_nodes = 0;
    IndexArrayType edge_array, node_array;
    // count edges in upper triangle of A
    for (IndexType ix = 0; ix < i.size(); ++ix)
    {
        if (i[ix] < j[ix])
        {
            edge_array.push_back(num_edges);
            node_array.push_back(i[ix]);
            edge_array.push_back(num_edges);
            node_array.push_back(j[ix]);
            ++num_edges;

            num_nodes = std::max(num_nodes, i[ix]);
            num_nodes = std::max(num_nodes, j[ix]);
        }
    }
    ++num_nodes;
    std::vector<T> v(edge_array.size(), 1);

    Matrix<T> E(num_edges, num_nodes);
    E.build(edge_array.begin(), node_array.begin(), v.begin(), v.size());
    print_matrix(std::cout, E, "Incidence");

    std::cout << "Running k-truss algorithm..." << std::endl;
    T count(0);

    auto Eout3 = algorithms::k_truss(E, 3);
    std::cout << "===============================================" << std::endl;
    GraphBLAS::print_matrix(std::cout, Eout3, "Edges in 3-trusses");
    std::cout << "===============================================" << std::endl;

    auto Eout4 = algorithms::k_truss(Eout3, 4);
    std::cout << "===============================================" << std::endl;
    GraphBLAS::print_matrix(std::cout, Eout4, "Edges in 4-trusses");
    std::cout << "===============================================" << std::endl;

    auto Eout5 = algorithms::k_truss(Eout4, 5);
    std::cout << "===============================================" << std::endl;
    GraphBLAS::print_matrix(std::cout, Eout5, "Edges in 5-trusses");
    std::cout << "===============================================" << std::endl;

    auto Eout6 = algorithms::k_truss(Eout5, 6);
    std::cout << "===============================================" << std::endl;
    GraphBLAS::print_matrix(std::cout, Eout6, "Edges in 6-trusses");
    std::cout << "===============================================" << std::endl;
    return 0;
}
