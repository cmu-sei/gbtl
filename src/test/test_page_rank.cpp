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
 * This Software includes and/or makes use of the following Third-Party Software
 * subject to its own license:
 *
 * 1. Boost Unit Test Framework
 * (https://www.boost.org/doc/libs/1_45_0/libs/test/doc/html/utf.html)
 * Copyright 2001 Boost software license, Gennadiy Rozental.
 *
 * DM20-0442
 */

#include <iostream>

#include <graphblas/graphblas.hpp>
#include <algorithms/page_rank.hpp>

using namespace grb;
using namespace algorithms;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE page_rank_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

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
        1,2,3,4,5,6,7,8,10,11,12,13,17,19,21,31,
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
    std::vector<double> answer = {0.0972041, 0.0529611, 0.0570794, 0.0358651,
                                  0.0220110, 0.0291618, 0.0291618, 0.0244569,
                                  0.0297064, 0.0142740, 0.0220110, 0.00955609,
                                  0.0146330, 0.0294733, 0.0144766, 0.0144766,
                                  0.0168304, 0.0145342, 0.0144766, 0.0195506,
                                  0.0144766, 0.0145342, 0.0144766, 0.0314621,
                                  0.0210849, 0.0210187, 0.0150202, 0.0256012,
                                  0.0195538, 0.0262432, 0.0245261, 0.0370686,
                                  0.0718675, 0.101166};
    for (IndexType ix = 0; ix < NUM_NODES; ++ix)
    {
        BOOST_CHECK_CLOSE(results[ix], answer[ix], 0.001);
    }
}

BOOST_AUTO_TEST_SUITE_END()
