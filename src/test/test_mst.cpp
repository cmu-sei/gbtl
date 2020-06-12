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

#include <algorithms/mst.hpp>
#include <graphblas/graphblas.hpp>

using namespace grb;
using namespace algorithms;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE mst_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

//****************************************************************************
BOOST_AUTO_TEST_CASE(mst_test_with_weight_one)
{
    IndexType const NUM_NODES = 6;
    IndexArrayType i_m1 = {0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3,
                           4, 4, 4, 4, 4, 5, 5};
    IndexArrayType j_m1 = {1, 3, 4, 0, 3, 4, 4, 5, 0, 1, 4,
                           0, 1, 2, 3, 5, 2, 4};
    std::vector<double> v_m1(i_m1.size(), 1);
    Matrix<double> m1(NUM_NODES, NUM_NODES);
    m1.build(i_m1, j_m1, v_m1);
    grb::print_matrix(std::cout, m1, "GRAPH***");

    std::vector<IndexType> ans = {99, 0, 4, 1, 3, 2};
    grb::Vector<IndexType> answer(ans, 99);

    grb::Vector<grb::IndexType> parents(NUM_NODES);
    auto result = mst(m1, parents);

    BOOST_CHECK_EQUAL(result, NUM_NODES - 1.0);
    BOOST_CHECK_EQUAL(parents, answer);

    std::cout << "MST weight = " << result << std::endl;
    grb::print_vector(std::cout, parents, "MST parent list");
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(mst_test_with_weight_various)
{
    IndexType const NUM_NODES = 6;
    IndexArrayType i_m1 =      {0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3,
                                4, 4, 4, 4, 4, 5, 5};
    IndexArrayType j_m1 =      {1, 3, 4, 0, 3, 4, 4, 5, 0, 1, 4,
                                0, 1, 2, 3, 5, 2, 4};
    std::vector<double> v_m1 = {2, 2, 1, 2, 2, 1, 1, 2, 2, 2, 1,
                                1, 1, 1, 1, 1, 2, 1};
    Matrix<double> m1(NUM_NODES, NUM_NODES);
    m1.build(i_m1, j_m1, v_m1);
    grb::print_matrix(std::cout, m1, "GRAPH***");

    std::vector<IndexType> ans = {99, 4, 4, 4, 0, 4};
    grb::Vector<IndexType> answer(ans, 99);

    grb::Vector<grb::IndexType> parents(NUM_NODES);
    auto result = mst(m1, parents);

    BOOST_CHECK_EQUAL(result, NUM_NODES - 1.0);
    BOOST_CHECK_EQUAL(parents, answer);

    std::cout << "MST weight = " << result << std::endl;
    grb::print_vector(std::cout, parents, "MST parent list");
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(mst_test_with_weights)
{
    //                   m1({{0, 4, 0, 0, 0, 0, 0, 8, 0},
    //                       {4, 0, 8, 0, 0, 0, 0,11, 0},
    //                       {0, 8, 0, 7, 0, 4, 0, 0, 2},
    //                       {0, 0, 7, 0, 9,14, 0, 0, 0},
    //                       {0, 0, 0, 9, 0,10, 0, 0, 0},
    //                       {0, 0, 4,14,10, 0, 2, 0, 0},
    //                       {0, 0, 0, 0, 0, 2, 0, 1, 6},
    //                       {8,11, 0, 0, 0, 0, 1, 0, 7},
    //                       {0, 0, 2, 0, 0, 0, 6, 7, 0}});
    IndexType const NUM_NODES = 9;
    IndexArrayType i_m1 = {0, 0, 1, 1, 1, 2, 2, 2, 2,
                           3, 3, 3, 4, 4, 5, 5, 5, 5,
                           6, 6, 6, 7, 7, 7, 7, 8, 8, 8};
    IndexArrayType j_m1 = {1, 7, 0, 2, 7, 1, 3, 5, 8,
                           2, 4, 5, 3, 5, 2, 3, 4, 6,
                           5, 7, 8, 0, 1, 6, 8, 2, 6, 7};
    std::vector<double> v_m1 = {4, 8, 4, 8,11, 8, 7, 4, 2,
                                7, 9,14, 9,10, 4,14,10, 2,
                                2, 1, 6, 8,11, 1, 7, 2, 6, 7};
    Matrix<double> m1(NUM_NODES, NUM_NODES);
    m1.build(i_m1, j_m1, v_m1);
    grb::print_matrix(std::cout, m1, "GRAPH***");

    std::vector<IndexType> ans = {99, 0, 1, 2, 3, 2, 5, 6, 2};
    grb::Vector<IndexType> answer(ans, 99);

    grb::Vector<IndexType> parents(NUM_NODES);
    auto result = mst(m1, parents);

    BOOST_CHECK_EQUAL(result, 37);
    BOOST_CHECK_EQUAL(parents, answer);

    //BOOST_CHECK_EQUAL(result, correct_weight);
    std::cout << "MST weight = " << result << std::endl;
    grb::print_vector(std::cout, parents, "MST parent list");
}

BOOST_AUTO_TEST_SUITE_END()
