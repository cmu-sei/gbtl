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

BOOST_AUTO_TEST_SUITE(sparse_apply_suite)

//****************************************************************************

BOOST_AUTO_TEST_CASE(sparse_apply_vector)
{
    std::vector<double> vecA = {8, 0, 6, 0};
    GraphBLAS::Vector<double> vA(vecA, 0);

    std::vector<double> vecC = {1, 2, 0, 0};
    GraphBLAS::Vector<double> vC(vecC, 0);

    std::vector<double> vecAnswer = {-8, 0, -6,  0};
    GraphBLAS::Vector<double> answer(vecAnswer, 0);

    GraphBLAS::apply(vC,
                     GraphBLAS::NoMask(),
                     GraphBLAS::NoAccumulate(),
                     GraphBLAS::AdditiveInverse<double>(),
                     vA);

    BOOST_CHECK_EQUAL(vC, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_apply_vector_accum)
{
    std::vector<double> vecA = {8, 0, 6, 0};
    GraphBLAS::Vector<double> vA(vecA, 0);

    std::vector<double> vecC = {8, 2, 0, 0};
    GraphBLAS::Vector<double> vC(vecC, 0);

    // NOTE: The accum results in an explicit zero
    std::vector<double> vecAnswer = {0, 2, -6,  666};
    GraphBLAS::Vector<double> answer(vecAnswer, 666);

    GraphBLAS::apply(vC,
                     GraphBLAS::NoMask(),
                     GraphBLAS::Plus<double>(),
                     GraphBLAS::AdditiveInverse<double>(),
                     vA);

    BOOST_CHECK_EQUAL(vC, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_apply_vector_mask)
{
    std::vector<double> vecA = {8, 8, 0, 0, 6, 6, 0, 0};
    GraphBLAS::Vector<double> vA(vecA, 0);

    std::vector<double> vecC = {1, 1, 2, 2, 0, 0, 0, 0};
    GraphBLAS::Vector<double> vC(vecC, 0);

    std::vector<bool> vecMask = {true, false, true, false, true, false, true, false};
    GraphBLAS::Vector<bool> mask(vecMask, false);

    std::vector<double> vecAnswer = {-8, 1, 0, 2, -6, 0, 0, 0};
    GraphBLAS::Vector<double> answer(vecAnswer, 0);

    GraphBLAS::apply(vC,
                     mask,
                     GraphBLAS::NoAccumulate(),
                     GraphBLAS::AdditiveInverse<double>(),
                     vA);

    BOOST_CHECK_EQUAL(vC, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_apply_vector_mask_replace)
{
    std::vector<double> vecA = {8, 8, 0, 0, 6, 6, 0, 0};
    GraphBLAS::Vector<double> vA(vecA, 0);

    std::vector<double> vecC = {1, 1, 2, 2, 0, 0, 0, 0};
    GraphBLAS::Vector<double> vC(vecC, 0);

    std::vector<bool> vecMask = {true, false, true, false, true, false, true, false};
    GraphBLAS::Vector<bool> mask(vecMask, false);

    std::vector<double> vecAnswer = {-8, 0, 0, 0, -6, 0, 0, 0};
    GraphBLAS::Vector<double> answer(vecAnswer, 0);

    GraphBLAS::apply(vC,
                     mask,
                     GraphBLAS::NoAccumulate(),
                     GraphBLAS::AdditiveInverse<double>(),
                     vA,
                     true);

    BOOST_CHECK_EQUAL(vC, answer);
}

//****************************************************************************

BOOST_AUTO_TEST_CASE(sparse_apply_matrix)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(matA, 0);

    std::vector<std::vector<double>> matC = {{1, 2, 3},
                                             {4, 5, 6},
                                             {7, 8, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mC(matC, 0);

    std::vector<std::vector<double>> matAnswer = {{-8, -1, -6},
                                                  {-3, -5, -7},
                                                  {-4, -9,  0}};

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(matAnswer, 0);

    GraphBLAS::apply(mC,
                     GraphBLAS::NoMask(),
                     GraphBLAS::NoAccumulate(),
                     GraphBLAS::AdditiveInverse<double>(),
                     mA);

    BOOST_CHECK_EQUAL(mC, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_apply_matrix_accum)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(matA, 0);

    std::vector<std::vector<double>> matC = {{1, 2, 3},
                                             {4, 5, 6},
                                             {7, 8, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mC(matC, 0);

    std::vector<std::vector<double>> matAnswer = {{-7, 1, -3},
                                                  { 1, 0, -1},
                                                  { 3,-1,  1337}};
    // NOTE: The accum results in an explicit zero
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(matAnswer, 1337);

    GraphBLAS::apply(mC,
                     GraphBLAS::NoMask(),
                     GraphBLAS::Plus<double>(),
                     GraphBLAS::AdditiveInverse<double>(),
                     mA);

    BOOST_CHECK_EQUAL(mC, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_apply_matrix_mask)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(matA, 0);

    std::vector<std::vector<double>> matC = {{1, 2, 3},
                                             {4, 5, 6},
                                             {7, 8, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mC(matC, 0);


    std::vector<std::vector<bool>> matMask = {{true, false, false},
                                              {true, true,  false,},
                                              {true, true,  true}};
    GraphBLAS::Matrix<bool, GraphBLAS::DirectedMatrixTag> mask(matMask, false);

    std::vector<std::vector<double>> matAnswer = {{-8, 2,  3},
                                                  {-3, -5, 6},
                                                  {-4, -9, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(matAnswer, 0);

    GraphBLAS::apply(mC,
                     mask,
                     GraphBLAS::NoAccumulate(),
                     GraphBLAS::AdditiveInverse<double>(),
                     mA);

    BOOST_CHECK_EQUAL(mC, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_apply_matrix_mask_replace)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(matA, 0);

    std::vector<std::vector<double>> matC = {{1, 2, 3},
                                             {4, 5, 6},
                                             {7, 8, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mC(matC, 0);


    std::vector<std::vector<bool>> matMask = {{true, false, false},
                                              {true, true,  false,},
                                              {true, true,  true}};
    GraphBLAS::Matrix<bool, GraphBLAS::DirectedMatrixTag> mask(matMask, false);

    std::vector<std::vector<double>> matAnswer = {{-8, 0,  0},
                                                  {-3, -5, 0},
                                                  {-4, -9, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(matAnswer, 0);

    GraphBLAS::apply(mC,
                     mask,
                     GraphBLAS::NoAccumulate(),
                     GraphBLAS::AdditiveInverse<double>(),
                     mA,
                     true);

    BOOST_CHECK_EQUAL(mC, answer);
}


BOOST_AUTO_TEST_SUITE_END()
