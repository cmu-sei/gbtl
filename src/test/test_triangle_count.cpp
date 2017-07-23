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
#include <algorithms/triangle_count.hpp>

using namespace GraphBLAS;
using namespace algorithms;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE triangle_count_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(triangle_count_suite)

/*
static double const INF = std::numeric_limits<double>::max();

static std::vector<double> gr={0,1,1,2,2,2,2,3,3,3,3,4,4,4,5,6,6,6,8,8};
static std::vector<double> gc={3,3,6,4,5,6,8,0,1,4,6,2,3,8,2,1,2,3,2,4};
static std::vector<double> gv(gr.size(), 1);

static Matrix<double, DirectedMatrixTag> G_tn(9,9);


//static Matrix<double, DirectedMatrixTag> G_tn_answer(
//    {{2, 2, 3, 1, 2, 4, 2, INF, 3},
//     {2, 2, 2, 1, 2, 3, 1, INF, 3},
//     {3, 2, 2, 2, 1, 1, 1, INF, 1},
//     {1, 1, 2, 2, 1, 3, 1, INF, 2},
//     {2, 2, 1, 1, 2, 2, 2, INF, 1},
//     {4, 3, 1, 3, 2, 2, 2, INF, 2},
//     {2, 1, 1, 1, 2, 2, 2, INF, 2},
//     {INF, INF, INF, INF, INF, INF, INF, INF, INF},
//     {3, 3, 1, 2, 1, 2, 2, INF, 2}},
//    INF);

static std::vector<double> tr={0,0,1,1,2,2,2,2,3,3,4,4};
static std::vector<double> tc={1,2,0,2,0,1,3,4,2,4,2,3};
static std::vector<double> tv(tr.size(), 1);

//static Matrix<double, DirectedMatrixTag> test5x5(
//    {{INF,   1,   1, INF, INF},
//     {  1, INF,   1, INF, INF},
//     {  1,   1, INF,   1,   1},
//     {INF, INF,   1, INF,   1},
//     {INF, INF,   1,   1, INF}},
//    INF);
static Matrix<double, DirectedMatrixTag> test5x5(5,5,INF);
*/

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_triangle_counting_v1)
{
    //Matrix<double, DirectedMatrixTag> testtriangle(
    //                       {{0,1,1,1,0},
    //                        {1,0,1,0,1},
    //                        {1,1,0,1,1},
    //                        {1,0,1,0,1},
    //                        {0,1,1,1,0}});

    std::vector<double> ar={0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4};
    std::vector<double> ac={1, 2, 3, 0, 2, 4, 0, 1, 3, 4, 0, 2, 4, 1, 2, 3};
    std::vector<double> av={1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> testtriangle(5,5);
    testtriangle.build(ar.begin(), ac.begin(), av.begin(), av.size());

    IndexType result = triangle_count(testtriangle);
    BOOST_CHECK_EQUAL(result, 4);
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(test_triangle_counting_flame1)
{
    //Matrix<double, DirectedMatrixTag> testtriangle(
    //                       {{0,1,1,1,0},
    //                        {1,0,1,0,1},
    //                        {1,1,0,1,1},
    //                        {1,0,1,0,1},
    //                        {0,1,1,1,0}});

    std::vector<double> ar={0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4};
    std::vector<double> ac={1, 2, 3, 0, 2, 4, 0, 1, 3, 4, 0, 2, 4, 1, 2, 3};
    std::vector<double> av={1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> testtriangle(5,5);
    testtriangle.build(ar.begin(), ac.begin(), av.begin(), av.size());

    IndexType result = triangle_count_flame1(testtriangle);
    BOOST_CHECK_EQUAL(result, 4);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_triangle_counting_flame1_newGBTL)
{
    //Matrix<double, DirectedMatrixTag> testtriangle(
    //                       {{0,1,1,1,0},
    //                        {1,0,1,0,1},
    //                        {1,1,0,1,1},
    //                        {1,0,1,0,1},
    //                        {0,1,1,1,0}});

    std::vector<double> ar={0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4};
    std::vector<double> ac={1, 2, 3, 0, 2, 4, 0, 1, 3, 4, 0, 2, 4, 1, 2, 3};
    std::vector<double> av={1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> testtriangle(5,5);
    testtriangle.build(ar.begin(), ac.begin(), av.begin(), av.size());

    IndexType result = triangle_count_flame1_newGBTL(testtriangle);
    BOOST_CHECK_EQUAL(result, 4);
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(test_triangle_counting_flame1a)
{
    //Matrix<double, DirectedMatrixTag> testtriangle(
    //                       {{0,1,1,1,0},
    //                        {1,0,1,0,1},
    //                        {1,1,0,1,1},
    //                        {1,0,1,0,1},
    //                        {0,1,1,1,0}});

    std::vector<double> ar={0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4};
    std::vector<double> ac={1, 2, 3, 0, 2, 4, 0, 1, 3, 4, 0, 2, 4, 1, 2, 3};
    std::vector<double> av={1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> testtriangle(5,5);
    testtriangle.build(ar.begin(), ac.begin(), av.begin(), av.size());

    IndexType result = triangle_count_flame1a(testtriangle);
    BOOST_CHECK_EQUAL(result, 4);
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(test_triangle_counting_flame2)
{
    //Matrix<double, DirectedMatrixTag> testtriangle(
    //                       {{0,1,1,1,0},
    //                        {1,0,1,0,1},
    //                        {1,1,0,1,1},
    //                        {1,0,1,0,1},
    //                        {0,1,1,1,0}});

    std::vector<double> ar={0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4};
    std::vector<double> ac={1, 2, 3, 0, 2, 4, 0, 1, 3, 4, 0, 2, 4, 1, 2, 3};
    std::vector<double> av={1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> testtriangle(5,5);
    testtriangle.build(ar.begin(), ac.begin(), av.begin(), av.size());

    IndexType result = triangle_count_flame2(testtriangle);
    BOOST_CHECK_EQUAL(result, 4);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_triangle_counting_flame2_newGBTL)
{
    //Matrix<double, DirectedMatrixTag> testtriangle(
    //                       {{0,1,1,1,0},
    //                        {1,0,1,0,1},
    //                        {1,1,0,1,1},
    //                        {1,0,1,0,1},
    //                        {0,1,1,1,0}});

    std::vector<double> ar={0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4};
    std::vector<double> ac={1, 2, 3, 0, 2, 4, 0, 1, 3, 4, 0, 2, 4, 1, 2, 3};
    std::vector<double> av={1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> testtriangle(5,5);
    testtriangle.build(ar.begin(), ac.begin(), av.begin(), av.size());

    IndexType result = triangle_count_flame2_newGBTL(testtriangle);
    BOOST_CHECK_EQUAL(result, 4);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_triangle_counting_flame2_newGBTL_masked)
{
    //Matrix<double, DirectedMatrixTag> testtriangle(
    //                       {{0,1,1,1,0},
    //                        {1,0,1,0,1},
    //                        {1,1,0,1,1},
    //                        {1,0,1,0,1},
    //                        {0,1,1,1,0}});

    std::vector<double> ar={0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4};
    std::vector<double> ac={1, 2, 3, 0, 2, 4, 0, 1, 3, 4, 0, 2, 4, 1, 2, 3};
    std::vector<double> av={1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> testtriangle(5,5);
    testtriangle.build(ar.begin(), ac.begin(), av.begin(), av.size());

    IndexType result = triangle_count_flame2_newGBTL_masked(testtriangle);
    BOOST_CHECK_EQUAL(result, 4);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_triangle_counting_flame2_newGBTL_blocked)
{
    //Matrix<double, DirectedMatrixTag> testtriangle(
    //                       {{0,1,1,1,0},
    //                        {1,0,1,0,1},
    //                        {1,1,0,1,1},
    //                        {1,0,1,0,1},
    //                        {0,1,1,1,0}});

    std::vector<double> ar={0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4};
    std::vector<double> ac={1, 2, 3, 0, 2, 4, 0, 1, 3, 4, 0, 2, 4, 1, 2, 3};
    std::vector<double> av={1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> testtriangle(5,5);
    testtriangle.build(ar.begin(), ac.begin(), av.begin(), av.size());

    IndexType result = triangle_count_flame2_newGBTL_blocked(testtriangle);
    BOOST_CHECK_EQUAL(result, 4);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_triangle_counting_newGBTL)
{
    //Matrix<double, DirectedMatrixTag> testtriangle(
    //                       {{0,1,1,1,0},
    //                        {1,0,1,0,1},
    //                        {1,1,0,1,1},
    //                        {1,0,1,0,1},
    //                        {0,1,1,1,0}});

    std::vector<double> ar={0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4};
    std::vector<double> ac={1, 2, 3, 0, 2, 4, 0, 1, 3, 4, 0, 2, 4, 1, 2, 3};
    std::vector<double> av={1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> testtriangle(5,5), L(5,5), U(5,5);
    testtriangle.build(ar.begin(), ac.begin(), av.begin(), av.size());
    GraphBLAS::split(testtriangle, L, U);

    IndexType result = triangle_count_newGBTL(L, U);
    BOOST_CHECK_EQUAL(result, 4);
}

BOOST_AUTO_TEST_SUITE_END()
