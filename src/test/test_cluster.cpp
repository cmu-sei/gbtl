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
#include <algorithms/cluster.hpp>

using namespace grb;
using namespace algorithms;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE cluster_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

//****************************************************************************
BOOST_AUTO_TEST_CASE(cluster_test_markov)
{
    IndexType const NUM_NODES = 12;
    IndexType const NUM_EDGES = 52;
    grb::IndexArrayType i = {
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

    grb::IndexArrayType j = {
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
    std::vector<double> v(NUM_EDGES, 1.0);
    Matrix<double> m1(NUM_NODES, NUM_NODES);

    // Build matrix containing self loops
    m1.build(i, j, v);
    grb::print_matrix(std::cout, m1, "Graph + self loops");

    auto cluster_matrix = markov_cluster(m1, 2, 2, 1.0e-16, 30);
    grb::print_matrix(std::cout, cluster_matrix,
                            "Cluster matrix (before threshold)");

    // Optional: Apply a threshold to annihilate REALLY small numbers
    grb::Matrix<bool> mask(NUM_NODES, NUM_NODES);
    grb::apply(mask,
               grb::NoMask(), grb::NoAccumulate(),
               std::bind(grb::GreaterThan<double>(),
                         std::placeholders::_1,
                         1.0e-8),
               cluster_matrix);
    grb::print_matrix(std::cout, mask,
                            "Threshold mask");
    grb::apply(cluster_matrix,
               mask, grb::NoAccumulate(),
               grb::Identity<double>(),
               cluster_matrix, grb::REPLACE);
    grb::print_matrix(std::cout, cluster_matrix,
                      "Cluster matrix (after threshold)");

    // Compare with the example here:
    // https://www.cs.umd.edu/class/fall2009/cmsc858l/lecs/Lec12-mcl.pdf
    auto cluster_assignments = get_cluster_assignments(cluster_matrix);
    std::cout << "Cluster assignments: ";
    for (auto it : cluster_assignments)
        std::cout << it << " ";
    std::cout << std::endl;

    BOOST_CHECK_EQUAL(cluster_assignments[0], cluster_assignments[5]);
    BOOST_CHECK_EQUAL(cluster_assignments[0], cluster_assignments[6]);
    BOOST_CHECK_EQUAL(cluster_assignments[0], cluster_assignments[9]);

    BOOST_CHECK_EQUAL(cluster_assignments[1], cluster_assignments[2]);
    BOOST_CHECK_EQUAL(cluster_assignments[1], cluster_assignments[4]);

    BOOST_CHECK_EQUAL(cluster_assignments[3], cluster_assignments[7]);
    BOOST_CHECK_EQUAL(cluster_assignments[3], cluster_assignments[8]);
    BOOST_CHECK_EQUAL(cluster_assignments[3], cluster_assignments[10]);
    BOOST_CHECK_EQUAL(cluster_assignments[3], cluster_assignments[11]);

    BOOST_CHECK(cluster_assignments[0] != cluster_assignments[1]);
    BOOST_CHECK(cluster_assignments[0] != cluster_assignments[3]);
    BOOST_CHECK(cluster_assignments[1] != cluster_assignments[3]);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(cluster_test_peer_pressure1)
{
    std::cout << "============== Peer Pressure 1 ================" << std::endl;
    grb::IndexArrayType i_m2 = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2,
                                      3, 3, 4, 4};
    grb::IndexArrayType j_m2 = {0, 1, 2, 3, 0, 1, 2, 0, 1, 2,
                                      3, 4, 3, 4};
    std::vector<double>       v_m2 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                      1, 1, 1, 1};
    Matrix<double> m2(5, 5);
    m2.build(i_m2, j_m2, v_m2);

    auto ans2 = algorithms::peer_pressure_cluster(m2);
    auto cluster_assignments = get_cluster_assignments_v2(ans2);
    grb::print_vector(std::cout, cluster_assignments, "CLUSTER ASSIGNMENTS");

    BOOST_CHECK_EQUAL(cluster_assignments.extractElement(0),
                      cluster_assignments.extractElement(1));
    BOOST_CHECK_EQUAL(cluster_assignments.extractElement(0),
                      cluster_assignments.extractElement(1));

    BOOST_CHECK_EQUAL(cluster_assignments.extractElement(3),
                      cluster_assignments.extractElement(4));

    BOOST_CHECK(cluster_assignments.extractElement(0) !=
                cluster_assignments.extractElement(3));
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(cluster_test_peer_pressure2)
{
    std::cout << "============== Peer Pressure 2 ================" << std::endl;
    grb::IndexArrayType i_m1 = {0, 0, 0, 0,
                                      1, 1, 1, 1,
                                      2, 2, 2, 2,
                                      3, 3, 3, 3,
                                      4, 4, 4, 4,
                                      5, 5, 5, 5, 5,
                                      6, 6, 6,
                                      7, 7, 7, 7};
    grb::IndexArrayType j_m1 = {0, 2, 3, 6,
                                      1, 2, 3, 7,
                                      0, 2, 4, 6,
                                      0, 1, 3, 5,
                                      0, 2, 4, 6,
                                      1, 3, 5, 6, 7,
                                      0, 4, 6,
                                      1, 3, 5, 7};
    std::vector<double>       v_m1 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                      1, 1, 1, 1, 1, 1, 1, 1,
                                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    Matrix<double> m1(8, 8);
    m1.build(i_m1, j_m1, v_m1);

    auto ans = algorithms::peer_pressure_cluster(m1);

    auto cluster_assignments = get_cluster_assignments_v2(ans);
    grb::print_vector(std::cout, cluster_assignments, "CLUSTER ASSIGNMENTS");

    BOOST_CHECK_EQUAL(cluster_assignments.extractElement(0),
                      cluster_assignments.extractElement(2));
    BOOST_CHECK_EQUAL(cluster_assignments.extractElement(0),
                      cluster_assignments.extractElement(4));
    BOOST_CHECK_EQUAL(cluster_assignments.extractElement(0),
                      cluster_assignments.extractElement(6));

    BOOST_CHECK_EQUAL(cluster_assignments.extractElement(1),
                      cluster_assignments.extractElement(3));
    BOOST_CHECK_EQUAL(cluster_assignments.extractElement(1),
                      cluster_assignments.extractElement(5));
    BOOST_CHECK_EQUAL(cluster_assignments.extractElement(1),
                      cluster_assignments.extractElement(7));

    BOOST_CHECK(cluster_assignments.extractElement(0) !=
                cluster_assignments.extractElement(1));
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(cluster_test_peer_pressure_karate)
{
    IndexType num_nodes = 34;

    grb::IndexArrayType i = {
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

    grb::IndexArrayType j = {
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

    std::vector<double> v(i.size(), 1.0);

    Matrix<double> A(num_nodes, num_nodes);
    A.build(i, j, v);

    // add the elements along the diagonal as required for convergence
    //ewiseadd(A, I34, Ai);
    Matrix<double> Ai(num_nodes, num_nodes);
    auto I34 = grb::scaled_identity<Matrix<double> >(num_nodes);
    grb::eWiseAdd(Ai,
                  grb::NoMask(), grb::NoAccumulate(),
                  grb::Plus<double>(),
                  A, I34);

    auto ans = algorithms::peer_pressure_cluster(Ai);
    auto cluster_assignments = get_cluster_assignments_v2(ans);
    grb::print_vector(std::cout, cluster_assignments,
                      "LOUSY KARATE CLUSTER ASSIGNMENTS");

    auto ans2 = algorithms::markov_cluster(Ai);
    auto cl2 = get_cluster_assignments_v2(ans2);
    grb::print_vector(std::cout, cl2,
                      "MARKOV KARATE CLUSTER ASSIGNMENTS");

    // auto cluster_assignments = get_cluster_assignments(ans);
    // std::cout << "[lousy karate clusters";
    // for (auto it = cluster_assignments.begin();
    //      it != cluster_assignments.end();
    //      ++it)
    // {
    //     std::cout << ", " << *it;
    // }
    // std::cout << "]" << std::endl;
}
BOOST_AUTO_TEST_SUITE_END()
