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

using namespace graphblas;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE transpose_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(transpose_suite)

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_two_argument_transpose_bad_dim)
{
    graphblas::IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double>       v_mA    = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    buildmatrix(mA, i_mA, j_mA, v_mA);

    Matrix<double, DirectedMatrixTag> mC(3, 4);

    BOOST_CHECK_THROW(
        (transpose(mA, mC)),
        DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_two_argument_transpose_square)
{
    graphblas::IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double>       v_mA    = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    buildmatrix(mA, i_mA, j_mA, v_mA);

    Matrix<double, DirectedMatrixTag> answer(3, 3);
    buildmatrix(answer, j_mA, i_mA, v_mA);

    Matrix<double, DirectedMatrixTag> mC(3, 3);

    transpose(mA, mC);

    BOOST_CHECK_EQUAL(mC, answer);

}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_two_argument_transpose_nonsquare)
{
    graphblas::IndexArrayType i_mA    = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mA    = {0, 1, 2, 3, 0, 1, 2, 0, 1, 2};
    std::vector<double>       v_mA    = {1, 2, 3,-2, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 4);
    buildmatrix(mA, i_mA, j_mA, v_mA);

    Matrix<double, DirectedMatrixTag> answer(4, 3);
    buildmatrix(answer, j_mA, i_mA, v_mA);

    Matrix<double, DirectedMatrixTag> mC(4, 3);

    transpose(mA, mC);

    BOOST_CHECK_EQUAL(mC, answer);

}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_single_argument_transpose_square)
{
    graphblas::IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double>       v_mA    = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    buildmatrix(mA, i_mA, j_mA, v_mA);

    Matrix<double, DirectedMatrixTag> answer(3, 3);
    buildmatrix(answer, j_mA, i_mA, v_mA);

    auto result = transpose(mA);

    IndexType r, c;
    result.get_shape(r, c);
    BOOST_CHECK_EQUAL(r, 3);
    BOOST_CHECK_EQUAL(c, 3);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_single_argument_transpose_nonsquare)
{
    graphblas::IndexArrayType i_mA    = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mA    = {0, 1, 2, 3, 0, 1, 2, 0, 1, 2};
    std::vector<double>       v_mA    = {1, 2, 3,-2, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 4);
    buildmatrix(mA, i_mA, j_mA, v_mA);

    Matrix<double, DirectedMatrixTag> answer(4, 3);
    buildmatrix(answer, j_mA, i_mA, v_mA);

    auto result = transpose(mA);

    IndexType r, c;
    result.get_shape(r, c);
    BOOST_CHECK_EQUAL(r, 4);
    BOOST_CHECK_EQUAL(c, 3);
    BOOST_CHECK_EQUAL(result, answer);
}

BOOST_AUTO_TEST_SUITE_END()
