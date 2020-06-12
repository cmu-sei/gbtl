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

#include <algorithms/maxflow.hpp>
#include <graphblas/graphblas.hpp>

using namespace grb;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE maxflow_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

//****************************************************************************
BOOST_AUTO_TEST_CASE(maxflow_push_relabel_test)
{
    IndexArrayType i = {0, 0, 1, 2, 2, 3, 4, 4};
    IndexArrayType j = {1, 3, 2, 3, 5, 4, 1, 5};
    std::vector<double>       v = {15, 4, 12, 3, 7, 10, 5, 10};
    Matrix<double, DirectedMatrixTag> m1(6, 6);
    m1.build(i, j, v);

    //grb::print_matrix(std::cerr, m1, "\nGraph");
    auto result = algorithms::maxflow_push_relabel(m1, 0, 5);
    BOOST_CHECK_EQUAL(result, 14);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(maxflow_push_relabel_test2)
{
    //       s   1  2  3  4  5  6  t
    //  m1({{-, 10, 5,15, -, -, -, -},   // s = 0
    //      {-,  -, 4, -, 9,15, -, -},   // 1
    //      {-,  -, -, 4, -, 8, -, -},   // 2
    //      {-,  -, -, -, -, -,30, -},   // 3
    //      {-,  -, -, -, -,15, -,10},   // 4
    //      {-,  -, -, -, -, -,15,10},   // 5
    //      {-,  -, 6, -, -, -, -,10},   // 6
    //      {-,  -, -, -, -, -, -, -}}); // t = 7

    IndexArrayType i =      {0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6};
    IndexArrayType j =      {1, 2, 3, 2, 4, 5, 3, 5, 6, 5, 7, 6, 7, 2, 7};
    std::vector<double> v = {10,5,15, 4, 9,15, 4, 8,30,15,10,15,10, 6,10};
    Matrix<double, DirectedMatrixTag> m1(8, 8);
    m1.build(i, j, v);

    //grb::print_matrix(std::cerr, m1, "\nGraph");
    auto result = algorithms::maxflow_push_relabel(m1, 0, 7);
    BOOST_CHECK_EQUAL(result, 28);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(maxflow_ford_fulk_test)
{
    IndexArrayType i = {0, 0, 1, 2, 2, 3, 4, 4};
    IndexArrayType j = {1, 3, 2, 3, 5, 4, 1, 5};
    std::vector<double>       v = {15, 4, 12, 3, 7, 10, 5, 10};
    Matrix<double, DirectedMatrixTag> m1(6, 6);
    m1.build(i, j, v);

    //grb::print_matrix(std::cerr, m1, "\nGraph");
    auto result = algorithms::maxflow_ford_fulk(m1, 0, 5);
    BOOST_CHECK_EQUAL(result, 14);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(maxflow_ford_fulk_test_counter_example)
{
    /*       2  2
     *      1--6--3
     *  2  /     / \  10
     *    /     /                               \
     *   0     /     5
     *    \   / 9   /
     *  9  \ /     /  2
     *      2-----4
     *         7
     */
    IndexArrayType      i = {0, 0, 1, 2, 2, 3, 4, 6};
    IndexArrayType      j = {1, 2, 6, 3, 4, 5, 5, 3};
    std::vector<double> v = {2, 9, 2, 9, 7,10, 2, 2};
    Matrix<double, DirectedMatrixTag> m1(7, 7);
    m1.build(i, j, v);

    //grb::print_matrix(std::cerr, m1, "\nGraph");
    auto result = algorithms::maxflow_ford_fulk(m1, 0, 5);
    BOOST_CHECK_EQUAL(result, 11);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(maxflow_ford_fulk_test2)
{
    //       s   1  2  3  4  5  6  t
    //  m1({{-, 10, 5,15, -, -, -, -},   // s = 0
    //      {-,  -, 4, -, 9,15, -, -},   // 1
    //      {-,  4, -, 4, -, 8, -, -},   // 2
    //      {-,  -, -, -, -, -,30, -},   // 3
    //      {-,  -, -, -, -,15, -,10},   // 4
    //      {-,  -, -, -, -, -,15,10},   // 5
    //      {-,  -, 6, -, -, -, -,10},   // 6
    //      {-,  -, -, -, -, -, -, -}}); // t = 7

    IndexArrayType i =      {0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6};
    IndexArrayType j =      {1, 2, 3, 2, 4, 5, 3, 5, 6, 5, 7, 6, 7, 2, 7};
    std::vector<double> v = {10,5,15, 4, 9,15, 4, 8,30,15,10,15,10, 6,10};
    Matrix<double, DirectedMatrixTag> m1(8, 8);
    m1.build(i, j, v);

    //grb::print_matrix(std::cerr, m1, "\nGraph");
    auto result = algorithms::maxflow_ford_fulk(m1, 0, 7);
    BOOST_CHECK_EQUAL(result, 28);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(maxflow_ford_fulk_test2_bidirectional)
{
    //       s   1  2  3  4  5  6  t
    //  m1({{-, 10, 5,15, -, -, -, -},   // s = 0
    //      {-,  -, 4, -, 9,15, -, -},   // 1
    //      {-,  4, -, 4, -, 8, -, -},   // 2
    //      {-,  -, -, -, -, -,30, -},   // 3
    //      {-,  -, -, -, -,15, -,10},   // 4
    //      {-,  -, -, -, 5, -,15,10},   // 5
    //      {-,  -, 6, -, -, -, -,10},   // 6
    //      {-,  -, -, -, -, -, -, -}}); // t = 7

    IndexArrayType i =      {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 4, 5, 5, 5, 6, 6};
    IndexArrayType j =      {1, 2, 3, 2, 4, 5, 1, 3, 5, 6, 5, 7, 4, 6, 7, 2, 7};
    std::vector<double> v = {10,5,15, 4, 9,15, 4, 4, 8,30,15,10, 5,15,10, 6,10};
    Matrix<double, DirectedMatrixTag> m1(8, 8);
    m1.build(i, j, v);

    //grb::print_matrix(std::cerr, m1, "\nGraph");
    auto result = algorithms::maxflow_ford_fulk(m1, 0, 7);
    BOOST_CHECK_EQUAL(result, 30);
}

BOOST_AUTO_TEST_SUITE_END()
