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

#define GB_USE_CPU_SIMPLE

#include <iostream>

#include <graphblas/graphblas.hpp>
#include <algorithms/cluster.hpp>
#include <graphblas/linalg_utils.hpp>

using namespace graphblas;
using namespace algorithms;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE cluster_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(cluster_suite)

//****************************************************************************
BOOST_AUTO_TEST_CASE(cluster_test_markov)
{
    IndexType num_nodes = 52;
    graphblas::IndexArrayType i = {
        0, 0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3, 3,
        4, 4, 4, 4, 4,
        5, 5, 5,
        6, 6, 6, 6,
        7, 7, 7, 7, 7,
        8, 8, 8, 8, 8,
        9, 9, 9, 9,
        10,10,10,10,10,
        11,11,11};

    graphblas::IndexArrayType j = {
        0, 1, 5, 6, 9,
        0, 1, 2, 4,
        1, 2, 3, 4,
        2, 3, 7, 8, 10,
        1, 2, 4, 6, 7,
        0, 5, 9,
        0, 4, 6, 9,
        3, 4, 7, 8, 10,
        3, 7, 8, 10, 11,
        0, 5, 6, 9,
        3, 7, 8, 10, 11,
        8, 10, 11};
    std::vector<double> v(num_nodes, 1.0);
    Matrix<double, DirectedMatrixTag> m1(12, 12);
    buildmatrix(m1, i, j, v);

    auto cluster_matrix = markov_cluster(m1);
    //std::cout << cluster_matrix << std::endl;

    // Compare with the example here:
    // https://www.cs.umd.edu/class/fall2009/cmsc858l/lecs/Lec12-mcl.pdf
    auto cluster_assignments = get_cluster_assignments(cluster_matrix);

    BOOST_CHECK_EQUAL(cluster_assignments[0], cluster_assignments[5]);
    BOOST_CHECK_EQUAL(cluster_assignments[0], cluster_assignments[6]);
    BOOST_CHECK_EQUAL(cluster_assignments[0], cluster_assignments[9]);

    BOOST_CHECK_EQUAL(cluster_assignments[1], cluster_assignments[2]);
    BOOST_CHECK_EQUAL(cluster_assignments[1], cluster_assignments[4]);

    BOOST_CHECK_EQUAL(cluster_assignments[3], cluster_assignments[7]);
    BOOST_CHECK_EQUAL(cluster_assignments[3], cluster_assignments[8]);
    BOOST_CHECK_EQUAL(cluster_assignments[3], cluster_assignments[10]);
    BOOST_CHECK_EQUAL(cluster_assignments[3], cluster_assignments[11]);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(cluster_test_peer_pressure1)
{
    graphblas::IndexArrayType i_m1 = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
                                      3, 3, 3, 3, 4, 4, 4, 4,
                                      5, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7};
    graphblas::IndexArrayType j_m1 = {0, 2, 3, 6, 1, 2, 3, 7, 0, 2, 4, 6,
                                      0, 1, 3, 5, 0, 2, 4, 6,
                                      1, 3, 5, 6, 7, 0, 4, 6, 1, 3, 5, 7};
    std::vector<double>       v_m1 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                      1, 1, 1, 1, 1, 1, 1, 1,
                                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> m1(8, 8);
    buildmatrix(m1, i_m1, j_m1, v_m1);

    auto ans = algorithms::peer_pressure_cluster(m1);
    auto cluster_assignments = get_cluster_assignments(ans);

    //std::cout << ans << std::endl;
    BOOST_CHECK_EQUAL(cluster_assignments[0], cluster_assignments[2]);
    BOOST_CHECK_EQUAL(cluster_assignments[0], cluster_assignments[4]);
    BOOST_CHECK_EQUAL(cluster_assignments[0], cluster_assignments[6]);

    BOOST_CHECK_EQUAL(cluster_assignments[1], cluster_assignments[3]);
    BOOST_CHECK_EQUAL(cluster_assignments[1], cluster_assignments[5]);
    BOOST_CHECK_EQUAL(cluster_assignments[1], cluster_assignments[7]);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(cluster_test_peer_pressure2)
{
    graphblas::IndexArrayType i_m2 = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2,
                                      3, 3, 4, 4};
    graphblas::IndexArrayType j_m2 = {0, 1, 2, 3, 0, 1, 2, 0, 1, 2,
                                      3, 4, 3, 4};
    std::vector<double>       v_m2 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                      1, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> m2(5, 5);
    buildmatrix(m2, i_m2, j_m2, v_m2);

    auto ans2 = algorithms::peer_pressure_cluster(m2);
    auto cluster_assignments = get_cluster_assignments(ans2);

    //std::cout << ans << std::endl;
    BOOST_CHECK_EQUAL(cluster_assignments[0], cluster_assignments[1]);
    BOOST_CHECK_EQUAL(cluster_assignments[0], cluster_assignments[2]);

    BOOST_CHECK_EQUAL(cluster_assignments[3], cluster_assignments[4]);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(cluster_test_peer_pressure_karate)
{
    IndexType num_nodes = 34;
    graphblas::IndexArrayType i = {
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

    graphblas::IndexArrayType j = {
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

    Matrix<double, DirectedMatrixTag> m1(num_nodes, num_nodes);
    buildmatrix(m1, i, j, v);

    // add the elements along the diagonal as required for convergence
    auto i34 =
        graphblas::identity<Matrix<double, DirectedMatrixTag> >(num_nodes);

    Matrix<double, DirectedMatrixTag> m1i(num_nodes, num_nodes);
    ewiseadd(m1, i34, m1i);

    auto ans = algorithms::peer_pressure_cluster(m1i);
    auto cluster_assignments = get_cluster_assignments(ans);

    std::cout << "[lousy karate clusters";
    for (auto it = cluster_assignments.begin();
         it != cluster_assignments.end();
         ++it)
    {
        std::cout << ", " << *it;
    }
    std::cout << "]" << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()
