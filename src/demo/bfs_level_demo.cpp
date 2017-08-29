/*
 * Copyright (c) 2015 Carnegie Mellon University and The Trustees of Indiana
 * University.
 * All Rights Reserved.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS," WITH NO WARRANTIES WHATSOEVER. CARNEGIE
 * MELLON UNIVERSITY AND THE TRUSTEES OF INDIANA UNIVERSITY EXPRESSLY DISCLAIM
 * TO THE FULLEST EXTENT PERMITTED BY LAW ALL EXPRESS, IMPLIED, AND STATUTORY
 * WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT OF PROPRIETARY RIGHTS.
 *
 * This Program is distributed under a BSD license.  Please see LICENSE file or
 * permission@sei.cmu.edu for more information.  DM-0002659
 */
#include <iostream>

#include <graphblas/graphblas.hpp>
#include <algorithms/bfs.hpp>

GraphBLAS::IndexType const num_nodes = 34;
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
    31,31,31,31,31,31,
    32,32,32,32,32,32,32,32,32,32,32,32,
    33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33};

GraphBLAS::IndexArrayType j = {
    1,2,3,4,5,6,7,8,10,11,12,13,19,21,23,31,
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
    0,24,25,28,32,33,
    2,8,14,15,18,20,22,23,29,30,31,33,
    8,9,13,14,15,18,19,20,22,23,26,27,28,29,30,31,32};


//****************************************************************************
int main()
{
    // TODO Assignment from Initalizer list.
    GraphBLAS::Matrix<unsigned int>  G_karate(num_nodes, num_nodes);
    std::vector<unsigned int> weights(i.size(), 1);

    G_karate.build(i.begin(), j.begin(), weights.begin(), i.size());

    // Trying the row vector approach
    GraphBLAS::Matrix<unsigned int>  root(1, num_nodes);
    // pick an arbitrary root:
    root.setElement(0, 30, 1);

    GraphBLAS::Matrix<unsigned int> levels1(1, num_nodes);
    GraphBLAS::Matrix<unsigned int> levels(1, num_nodes);

    algorithms::bfs_level(G_karate, root, levels1);

//    std::cout << "Graph: " << std::endl;
//    GraphBLAS::print_matrix(std::cout, G_karate);
    std::cout << "bfs_level output" << std::endl;
    std::cout << "root:" << std::endl;
    GraphBLAS::print_matrix(std::cout, root);
    std::cout << "levels:" << std::endl;
    GraphBLAS::print_matrix(std::cout, levels1);

    algorithms::batch_bfs_level_masked(G_karate, root, levels);

//    std::cout << "Graph: " << std::endl;
//    GraphBLAS::print_matrix(std::cout, G_karate);
    std::cout << std::endl;
    std::cout << "root:" << std::endl;
    GraphBLAS::print_matrix(std::cout, root);
    std::cout << "levels:" << std::endl;
    GraphBLAS::print_matrix(std::cout, levels);

    return 0;
}
