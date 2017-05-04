/*
 * Copyright (c) 2017 Carnegie Mellon University.
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

using namespace GraphBLAS;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE sparse_extract_suite

#include <boost/test/included/unit_test.hpp>

// @todo:  Why do I have to do this??
//#include <graphblas/system/sequential/sparse_extract.hpp>

BOOST_AUTO_TEST_SUITE(sparse_assign_suite)

//****************************************************************************

BOOST_AUTO_TEST_CASE(sparse_assign_base)
{
    std::vector<std::vector<double>> matA = {{1, 6},
                                             {9, 2}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(matA, 0);

    std::vector<std::vector<double>> matAnswer = {{0, 0, 0},
                                                  {9, 0, 2},
                                                  {1, 0, 6}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(matAnswer, 0);

    // Output space
    GraphBLAS::IndexType M = 3;
    GraphBLAS::IndexType N = 3;

    GraphBLAS::IndexArrayType row_indicies = {2, 1};
    GraphBLAS::IndexArrayType col_indicies = {0, 2};

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(M, N);

    GraphBLAS::assign(result,
                      GraphBLAS::NoMask(),
                      GraphBLAS::NoAccumulate(),
                      mA,
                      row_indicies,
                      col_indicies);

    BOOST_CHECK_EQUAL(result, answer);
}

BOOST_AUTO_TEST_CASE(sparse_assign_mask)
{
    std::vector<std::vector<double>> matA = {{1, 6},
                                             {9, 2}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(matA, 0);

    std::vector<std::vector<double>> matAnswer = {{0, 0, 0},
                                                  {9, 0, 0},
                                                  {1, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(matAnswer, 0);

    std::vector<std::vector<bool>> matMask = {{true, true, false},
                                              {true, true, false},
                                              {true, true, false}};
    GraphBLAS::Matrix<bool, GraphBLAS::DirectedMatrixTag> mask(matMask, 0);


    // Output space
    GraphBLAS::IndexType M = 3;
    GraphBLAS::IndexType N = 3;

    GraphBLAS::IndexArrayType row_indicies = {2, 1};
    GraphBLAS::IndexArrayType col_indicies = {0, 2};

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(M, N);

    GraphBLAS::assign(result,
                      mask,
                      GraphBLAS::NoAccumulate(),
                      mA,
                      row_indicies,
                      col_indicies);

    BOOST_CHECK_EQUAL(result, answer);
}

BOOST_AUTO_TEST_CASE(sparse_assign_accum)
{
    std::vector<std::vector<double>> matA = {{1, 6},
                                             {9, 2}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(matA, 0);

    std::vector<std::vector<double>> matC= {{1, 2, 3},
                                            {4, 5, 6},
                                            {7, 8, 9}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mC(matC, 0);

    std::vector<std::vector<double>> matAnswer = {{1,  2, 3},
                                                  {13, 5, 8},
                                                  {8,  8, 15}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(matAnswer, 0);

    GraphBLAS::IndexArrayType row_indicies = {2, 1};
    GraphBLAS::IndexArrayType col_indicies = {0, 2};

    GraphBLAS::assign(mC,
                      GraphBLAS::NoMask(),
                      GraphBLAS::Plus<double>(),
                      mA,
                      row_indicies,
                      col_indicies);

    BOOST_CHECK_EQUAL(mC, answer);
}


BOOST_AUTO_TEST_SUITE_END()
