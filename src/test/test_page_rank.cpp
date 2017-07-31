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
#include <algorithms/page_rank.hpp>

using namespace GraphBLAS;
using namespace algorithms;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE page_rank_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(page_rank_suite)

//****************************************************************************
BOOST_AUTO_TEST_CASE(page_rank_test_tn_isolated_node)
{
    IndexType const NUM_NODES(9);
    IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                        4, 4, 4, 5, 6, 6, 6, 8, 8};

    IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                        2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<double> v(i.size(), 1.0);
    Matrix<double> m1(NUM_NODES, NUM_NODES);
    m1.build(i, j, v);
    print_matrix(std::cout, m1, "Graph:");

    Vector<double> page_rank(NUM_NODES);
    algorithms::page_rank(m1, page_rank);
    print_vector(std::cout, page_rank, "Technote page rank:");

    /// @todo in the old code the isolated node 7 was given a PR of 1/60
    /// which is (1 - 0.85)/NUM_NODES. In GBTL2, we need to assign implied
    /// zeroes directly.  Is this useful?
    BOOST_CHECK_EQUAL(NUM_NODES, page_rank.nvals());
    std::vector<IndexType> idx(NUM_NODES);
    std::vector<double> results(NUM_NODES);
    page_rank.extractTuples(idx, results);
    std::vector<double> answer = {0.0540983, 0.0899004, 0.170849,
                                  0.170849,  0.129597,  0.0540983,
                                  0.129597,  0.0166667, 0.0899004};
    for (IndexType ix = 0; ix < NUM_NODES; ++ix)
    {
        BOOST_CHECK_CLOSE(results[ix], answer[ix], 0.001);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(page_rank_test_gilbert_directed)
{
    IndexType const NUM_NODES(7);
    IndexArrayType i = {0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6};

    IndexArrayType j = {1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4};
    std::vector<double> v(i.size(), 1.0);
    Matrix<double> m1(NUM_NODES, NUM_NODES);
    m1.build(i, j, v);
    print_matrix(std::cout, m1, "Graph:");

    Vector<double> page_rank(NUM_NODES);
    algorithms::page_rank(m1, page_rank);
    print_vector(std::cout, page_rank, "Gilbert page rank:");

    // CHECK answer
    BOOST_CHECK_EQUAL(NUM_NODES, page_rank.nvals());
    std::vector<IndexType> idx(NUM_NODES);
    std::vector<double> results(NUM_NODES);
    page_rank.extractTuples(idx, results);
    std::vector<double> answer = {0.0449525, 0.0409707, 0.383965,
                                  0.0524507, 0.051156, 0.386829, 0.0396759};
    for (IndexType ix = 0; ix < NUM_NODES; ++ix)
    {
        BOOST_CHECK_CLOSE(results[ix], answer[ix], 0.001);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(page_rank_test_001)
{
    IndexType NUM_NODES = 12;
    IndexArrayType i = {
        0, 0, 0, 0,
        1, 1, 1,
        2, 2, 2,
        3, 3, 3, 3,
        4, 4, 4, 4,
        5, 5,
        6, 6, 6,
        7, 7, 7, 7,
        8, 8, 8, 8,
        9, 9, 9,
        10,10,10,10,
        11,11};

    IndexArrayType j = {
        1, 5, 6, 9,
        0, 2, 4,
        1, 3, 4,
        2, 7, 8, 10,
        1, 2, 6, 7,
        0, 9,
        0, 4, 9,
        3, 4, 8, 10,
        3, 7, 10, 11,
        0, 5, 6,
        3, 7, 8, 11,
        8, 10};
    std::vector<double> v(i.size(), 1.0);
    Matrix<double> m1(NUM_NODES, NUM_NODES);
    m1.build(i, j, v);
    print_matrix(std::cout, m1, "Graph:");

    Vector<double> page_rank(NUM_NODES);
    algorithms::page_rank(m1, page_rank);
    print_vector(std::cout, page_rank, "Page rank:");

    // CHECK answer
    BOOST_CHECK_EQUAL(NUM_NODES, page_rank.nvals());
    std::vector<IndexType> idx(NUM_NODES);
    std::vector<double> results(NUM_NODES);
    page_rank.extractTuples(idx, results);
    std::vector<double> answer = {0.105862,  0.0757762, 0.0746632,
                                  0.0932387, 0.098525, 0.0575081,
                                  0.0777645, 0.0922869, 0.0947498,
                                  0.0826865, 0.0947498, 0.0521893};
    for (IndexType ix = 0; ix < NUM_NODES; ++ix)
    {
        BOOST_CHECK_CLOSE(results[ix], answer[ix], 0.001);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(page_rank_test_karate)
{
    IndexType NUM_NODES = 34;
    IndexArrayType i = {
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

    IndexArrayType j = {
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

    std::vector<double> v(i.size(), 1.0);
    Vector<double> page_rank(NUM_NODES);
    Matrix<double> k_graph(NUM_NODES, NUM_NODES);
    k_graph.build(i, j, v);

    algorithms::page_rank(k_graph, page_rank, 0.85, 1.e-7);
    print_vector(std::cout, page_rank, "Page rank:");

    // CHECK answer
    BOOST_CHECK_EQUAL(NUM_NODES, page_rank.nvals());
    std::vector<IndexType> idx(NUM_NODES);
    std::vector<double> results(NUM_NODES);
    page_rank.extractTuples(idx, results);
    std::vector<double> answer = {0.093175,  0.0496871,  0.0567346, 0.0349464,
                                  0.0215771, 0.0286444,  0.0286444, 0.0237885,
                                  0.0297366, 0.0143716,  0.0215771, 0.00934765,
                                  0.0142961, 0.0289308,  0.0147509, 0.0147509,
                                  0.0166181, 0.00908666, 0.0147509, 0.0191648,
                                  0.0147509, 0.0140225,  0.0147509, 0.0376547,
                                  0.0218512, 0.0224002,  0.0154626, 0.0269799,
                                  0.0197516, 0.0277661,  0.0245012, 0.0377902,
                                  0.0740136, 0.103725};
    for (IndexType ix = 0; ix < NUM_NODES; ++ix)
    {
        BOOST_CHECK_CLOSE(results[ix], answer[ix], 0.001);
    }
}

BOOST_AUTO_TEST_SUITE_END()
