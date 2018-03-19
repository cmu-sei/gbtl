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

#include <algorithms/maxflow.hpp>
#include <graphblas/graphblas.hpp>

using namespace GraphBLAS;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE maxflow_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(maxflow_suite)

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
