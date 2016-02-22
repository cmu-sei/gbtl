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

#include <graphblas/graphblas.hpp>

#include <iostream>

using namespace graphblas;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE negate_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(negate_suite)

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_negate_square)
{
    graphblas::IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double>       v_mA    = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    buildmatrix(mA, i_mA, j_mA, v_mA);

    std::vector<double>   v_answer    = {0, 1, 1, 1, 0, 1, 1, 1, 0};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    buildmatrix(answer, i_mA, j_mA, v_answer);

    auto result = negate(mA);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_negateview_square)
{
    graphblas::IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double>       v_mA    = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    buildmatrix(mA, i_mA, j_mA, v_mA);

    std::vector<double>   v_answer    = {0, 1, 1, 1, 0, 1, 1, 1, 0};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    buildmatrix(answer, i_mA, j_mA, v_answer);

    auto result = negate(mA); //NegateView<Matrix<double, DirectedMatrixTag>,
                              //graphblas::ArithmeticSemiring<double> >(mA);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_negate_nonsquare)
{
    graphblas::IndexArrayType i_mA    = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    graphblas::IndexArrayType j_mA    = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    std::vector<double>       v_mA    = {1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1};
    Matrix<double, DirectedMatrixTag> mA(3, 4, 0);
    buildmatrix(mA, i_mA, j_mA, v_mA);

    std::vector<double>   v_answer    = {0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0};
    Matrix<double, DirectedMatrixTag> answer(3, 4, 0);
    buildmatrix(answer, i_mA, j_mA, v_answer);

    auto result = negate(mA);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_negateview_nonsquare)
{
    graphblas::IndexArrayType i_mA    = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    graphblas::IndexArrayType j_mA    = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    std::vector<double>       v_mA    = {1, 0, 2, 0, 0, 3, 0,-1, 0, 0,-2, 6};
    Matrix<double, DirectedMatrixTag> mA(3, 4, 0);
    buildmatrix(mA, i_mA, j_mA, v_mA);

    std::vector<double>   v_answer    = {0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0};
    Matrix<double, DirectedMatrixTag> answer(3, 4, 0);
    buildmatrix(answer, i_mA, j_mA, v_answer);

    auto result = negate(mA); //NegateView<Matrix<double, DirectedMatrixTag>,
                              //graphblas::ArithmeticSemiring<double> >(mA);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_negate_transpose_nonsquare)
{
    graphblas::IndexArrayType i_mA    = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    graphblas::IndexArrayType j_mA    = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    std::vector<double>       v_mA    = {1, 0, 2, 0, 0, 3, 0,-1, 0, 0,-2, 6};
    Matrix<double, DirectedMatrixTag> mA(3, 4, 0);
    buildmatrix(mA, i_mA, j_mA, v_mA);

    std::vector<double>   v_answer    = {0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0};
    Matrix<double, DirectedMatrixTag> answer(4, 3, 0);
    buildmatrix(answer, j_mA, i_mA, v_answer);

    /// @todo This currently does not work (TransposeView is destructed)
    //auto result = negate(transpose(mA));
    auto tres = transpose(mA);
    auto result = negate(tres);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_transpose_negate_nonsquare)
{
    graphblas::IndexArrayType i_mA    = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    graphblas::IndexArrayType j_mA    = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    std::vector<double>       v_mA    = {1, 0, 2, 0, 0, 3, 0,-1, 0, 0,-2, 6};
    Matrix<double, DirectedMatrixTag> mA(3, 4, 0);
    buildmatrix(mA, i_mA, j_mA, v_mA);

    std::vector<double>   v_answer    = {0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0};
    Matrix<double, DirectedMatrixTag> answer(4, 3, 0);
    buildmatrix(answer, j_mA, i_mA, v_answer);

    /// @todo This currently does not work (NegateView is destructed)
    //auto result = transpose(negate(mA));
    auto nres = negate(mA);
    auto result = transpose(nres);

    BOOST_CHECK_EQUAL(result, answer);
}

/// @todo Need many more tests involving other semirings

BOOST_AUTO_TEST_SUITE_END()
