/*
 * GraphBLAS Template Library, Version 2.0
 *
 * Copyright 2018 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors. All Rights Reserved.
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
 * RIGHTS..
 *
 * Released under a BSD (SEI)-style license, please see license.txt or contact
 * permission@sei.cmu.edu for full terms.
 *
 * This release is an update of:
 *
 * 1. GraphBLAS Template Library (GBTL)
 * (https://github.com/cmu-sei/gbtl/blob/1.0.0/LICENSE) Copyright 2015 Carnegie
 * Mellon University and The Trustees of Indiana. DM17-0037, DM-0002659
 *
 * DM18-0559
 */

#include <iostream>

#include <algorithms/maxflow.hpp>
#include <graphblas/graphblas.hpp>

using namespace GraphBLAS;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE maxflow_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

//****************************************************************************
BOOST_AUTO_TEST_CASE(maxflow_test)
{
    IndexArrayType i = {0, 0, 1, 2, 2, 3, 4, 4};
    IndexArrayType j = {1, 3, 2, 3, 5, 4, 1, 5};
    std::vector<double>       v = {15, 4, 12, 3, 7, 10, 5, 10};
    Matrix<double, DirectedMatrixTag> m1(6, 6);
    m1.build(i, j, v);

    std::cerr << "============================== test1 =======================\n";
    GraphBLAS::print_matrix(std::cerr, m1, "\nGraph");
    auto result = algorithms::maxflow(m1, 0, 5);
    BOOST_CHECK_EQUAL(result, 14);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(maxflow_test2)
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

    std::cerr << "============================== test2 =======================\n";
    GraphBLAS::print_matrix(std::cerr, m1, "\nGraph");
    auto result = algorithms::maxflow(m1, 0, 7);
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

    std::cerr << "============================== FF test1 =======================\n";
    GraphBLAS::print_matrix(std::cerr, m1, "\nGraph");
    auto result = algorithms::maxflow_ford_fulk(m1, 0, 5);
    BOOST_CHECK_EQUAL(result, 14);
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

    std::cerr << "============================== FF test1 =======================\n";
    GraphBLAS::print_matrix(std::cerr, m1, "\nGraph");
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

    std::cerr << "========================== FF test2 bidir ====================\n";
    GraphBLAS::print_matrix(std::cerr, m1, "\nGraph");
    auto result = algorithms::maxflow_ford_fulk(m1, 0, 7);
    BOOST_CHECK_EQUAL(result, 30);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(maxflow_ford_fulk2_test1)
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

    std::cerr << "=========================== FF2 test =========================\n";
    GraphBLAS::print_matrix(std::cerr, m1, "\nGraph");
    auto result = algorithms::maxflow_ford_fulk2(m1, 0, 7);
    BOOST_CHECK_EQUAL(result, 28);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(maxflow_ford_fulk2_test2_bidirectional)
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

    std::cerr << "========================== FF2 test2 bidir ====================\n";
    GraphBLAS::print_matrix(std::cerr, m1, "\nGraph");
    auto result = algorithms::maxflow_ford_fulk2(m1, 0, 7);
    BOOST_CHECK_EQUAL(result, 30);
}


BOOST_AUTO_TEST_SUITE_END()
