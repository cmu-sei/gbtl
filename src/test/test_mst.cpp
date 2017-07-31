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

#include <algorithms/mst.hpp>
#include <graphblas/graphblas.hpp>

using namespace GraphBLAS;
using namespace algorithms;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE mst_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(mst_suite)

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
    GraphBLAS::print_matrix(std::cout, m1, "GRAPH***");

    std::vector<IndexType> ans = {99, 0, 4, 1, 3, 2};
    GraphBLAS::Vector<IndexType> answer(ans, 99);

    GraphBLAS::Vector<GraphBLAS::IndexType> parents(NUM_NODES);
    auto result = mst(m1, parents);

    BOOST_CHECK_EQUAL(result, NUM_NODES - 1.0);
    BOOST_CHECK_EQUAL(parents, answer);

    std::cout << "MST weight = " << result << std::endl;
    GraphBLAS::print_vector(std::cout, parents, "MST parent list");
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
    GraphBLAS::print_matrix(std::cout, m1, "GRAPH***");

    std::vector<IndexType> ans = {99, 4, 4, 4, 0, 4};
    GraphBLAS::Vector<IndexType> answer(ans, 99);

    GraphBLAS::Vector<GraphBLAS::IndexType> parents(NUM_NODES);
    auto result = mst(m1, parents);

    BOOST_CHECK_EQUAL(result, NUM_NODES - 1.0);
    BOOST_CHECK_EQUAL(parents, answer);

    std::cout << "MST weight = " << result << std::endl;
    GraphBLAS::print_vector(std::cout, parents, "MST parent list");
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
    GraphBLAS::print_matrix(std::cout, m1, "GRAPH***");

    std::vector<IndexType> ans = {99, 0, 1, 2, 3, 2, 5, 6, 2};
    GraphBLAS::Vector<IndexType> answer(ans, 99);

    GraphBLAS::Vector<IndexType> parents(NUM_NODES);
    auto result = mst(m1, parents);

    BOOST_CHECK_EQUAL(result, 37);
    BOOST_CHECK_EQUAL(parents, answer);

    //BOOST_CHECK_EQUAL(result, correct_weight);
    std::cout << "MST weight = " << result << std::endl;
    GraphBLAS::print_vector(std::cout, parents, "MST parent list");
}

BOOST_AUTO_TEST_SUITE_END()
