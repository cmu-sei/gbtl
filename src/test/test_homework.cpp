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
#include <graphblas/system/sequential/NegateView.hpp>

#include <iostream>

using namespace graphblas;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE homework_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(homework_suite)

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_single_argument_transpose)
{
    graphblas::IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    buildmatrix(mA, i_mA, j_mA, v_mA);

    std::vector<double> v_answer = {1, 4, 7, 2, 5, 8, 3, 6, 9};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    buildmatrix(answer, i_mA, j_mA, v_answer);

    auto result = transpose(mA);

    BOOST_CHECK_EQUAL(result, answer);

}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_negate)
{
    graphblas::IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    buildmatrix(mA, i_mA, j_mA, v_mA);

    std::vector<double> v_answer = {0, 1, 1, 1, 0, 1, 1, 1, 0};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    buildmatrix(answer, i_mA, j_mA, v_answer);

    auto result = negate(mA);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_single_argument_negate)
{
    graphblas::IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    buildmatrix(mA, i_mA, j_mA, v_mA);

    std::vector<double> v_answer = {0, 1, 1, 1, 0, 1, 1, 1, 0};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    buildmatrix(answer, i_mA, j_mA, v_answer);

    // @todo:  AOM:  Won't compile for me
//    auto result = NegateView<
//        Matrix<double, DirectedMatrixTag>,
//        graphblas::ArithmeticSemiring<double> >(mA);
//
//    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_masked)
{
    graphblas::IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    buildmatrix(mA, i_mA, j_mA, v_mA);

    graphblas::IndexArrayType i_mB    = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2};
    graphblas::IndexArrayType j_mB    = {0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3};
    std::vector<double> v_mB = {5, 8, 1, 2, 6, 7, 3, 4, 5, 9, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 4);
    buildmatrix(mB, i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 4);

    graphblas::IndexArrayType i_answer = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    graphblas::IndexArrayType j_answer = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    std::vector<double> v_answer = {114, 160, 60, 27, 74, 97,
                                    73, 14, 119, 157, 112, 23};
    Matrix<double, DirectedMatrixTag> answer(3, 4);
    buildmatrix(answer, i_answer, j_answer, v_answer);

    Matrix<unsigned int, DirectedMatrixTag> mask(3,4);
    std::vector<unsigned int> v_mask(i_answer.size(), 1);
    buildmatrix(mask, i_answer, j_answer, v_mask);

    mxmMaskedV2(mA, mB, result, mask);

    BOOST_CHECK_EQUAL(result, answer);
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_transpose)
{
    graphblas::IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    buildmatrix(mA, i_mA, j_mA, v_mA);

    graphblas::IndexArrayType i_mB    = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2};
    graphblas::IndexArrayType j_mB    = {0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3};
    std::vector<double> v_mB = {5, 8, 1, 2, 6, 7, 3, 4, 5, 9, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 4);
    buildmatrix(mB, i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 4);

    graphblas::IndexArrayType i_answer = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    graphblas::IndexArrayType j_answer = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    std::vector<double> v_answer = {112, 159, 87, 31, 97, 131,
                                    94, 22, 87, 111, 102, 15};
    Matrix<double, DirectedMatrixTag> answer(3, 4);
    buildmatrix(answer, i_answer, j_answer, v_answer);

    mxm(transpose(mA), mB, result);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_b_transpose)
{
    graphblas::IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    buildmatrix(mA, i_mA, j_mA, v_mA);

    graphblas::IndexArrayType i_mB    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mB    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mB = {5, 8, 1, 2, 6, 7, 3, 4, 5};
    Matrix<double, DirectedMatrixTag> mB(3, 3);
    buildmatrix(mB, i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 3);

    graphblas::IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer = {119, 87, 79, 66, 80, 62, 108, 125, 98};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    buildmatrix(answer, i_answer, j_answer, v_answer);

    mxm(mA, transpose(mB), result);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_and_b_transpose)
{
    graphblas::IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    buildmatrix(mA, i_mA, j_mA, v_mA);

    graphblas::IndexArrayType i_mB    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mB    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mB = {5, 8, 1, 2, 6, 7, 3, 4, 5};
    Matrix<double, DirectedMatrixTag> mB(3, 3);
    buildmatrix(mB, i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 3);

    graphblas::IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer = {99, 97, 87, 83, 100, 81, 72, 105, 78};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    buildmatrix(answer, i_answer, j_answer, v_answer);

    mxm(transpose(mA), transpose(mB), result);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_equals_c_transpose)
{
    graphblas::IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    buildmatrix(mA, i_mA, j_mA, v_mA);

    graphblas::IndexArrayType i_mB    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mB    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mB = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 3);
    buildmatrix(mB, i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 3);

    graphblas::IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer = {12, 4, 7, 7, 5, 8, 3, 6, 9};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    buildmatrix(answer, i_answer, j_answer, v_answer);

    mxm(transpose(mA), mB, result);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_negate)
{
    graphblas::IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {0, 1, 1, 0, 0, 0, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    buildmatrix(mA, i_mA, j_mA, v_mA);

    graphblas::IndexArrayType i_mB    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mB    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mB = {1, 0, 1, 1, 1, 0, 1, 0, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 3);
    buildmatrix(mB, i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 3);

    graphblas::IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer = {1, 0, 1, 3, 1, 2, 0, 0, 0};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    buildmatrix(answer, i_answer, j_answer, v_answer);

    mxm(negate(mA), mB, result);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_b_negate)
{
    graphblas::IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {0, 1, 1, 0, 0, 0, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    buildmatrix(mA, i_mA, j_mA, v_mA);

    graphblas::IndexArrayType i_mB    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mB    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mB = {1, 0, 1, 1, 1, 0, 1, 0, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 3);
    buildmatrix(mB, i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 3);

    graphblas::IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer = {0, 1, 1, 0, 0, 0, 0, 2, 1};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    buildmatrix(answer, i_answer, j_answer, v_answer);

    mxm(mA, negate(mB), result);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_and_b_negate)
{
    graphblas::IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {0, 1, 1, 0, 0, 0, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    buildmatrix(mA, i_mA, j_mA, v_mA);

    graphblas::IndexArrayType i_mB    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mB    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mB = {1, 0, 1, 1, 1, 0, 1, 0, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 3);
    buildmatrix(mB, i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 3);

    graphblas::IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer = {0, 1, 0, 0, 2, 1, 0, 0, 0};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    buildmatrix(answer, i_answer, j_answer, v_answer);

    mxm(negate(mA), negate(mB), result);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_equals_c_negate)
{
    graphblas::IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {0, 1, 1, 0, 0, 0, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    buildmatrix(mA, i_mA, j_mA, v_mA);

    graphblas::IndexArrayType i_mB    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mB    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mB = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 3);
    buildmatrix(mB, i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 3);

    graphblas::IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer = {1, 0, 0, 1, 1, 1, 0, 0, 0};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    buildmatrix(answer, i_answer, j_answer, v_answer);

    mxm(negate(mA), mB, result);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_transpose_and_negate)
{
    graphblas::IndexArrayType i_mA = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mA = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {0, 1, 1, 0, 0, 0, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    buildmatrix(mA, i_mA, j_mA, v_mA);

    graphblas::IndexArrayType i_mB = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mB = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mB = {1, 0, 1, 1, 1, 0, 1, 0, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 3);
    buildmatrix(mB, i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 3);

    graphblas::IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer = {2, 1, 1, 1, 1, 0, 1, 1, 0};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    buildmatrix(answer, i_answer, j_answer, v_answer);

    mxm(negate(transpose(mA)), mB, result);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_b_transpose_and_negate)
{
    graphblas::IndexArrayType i_mA = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mA = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {0, 1, 1, 0, 0, 0, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    buildmatrix(mA, i_mA, j_mA, v_mA);

    graphblas::IndexArrayType i_mB = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mB = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mB = {1, 0, 1, 1, 1, 0, 1, 0, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 3);
    buildmatrix(mB, i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 3);

    graphblas::IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer = {1, 1, 1, 0, 0, 0, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    buildmatrix(answer, i_answer, j_answer, v_answer);

    mxm(mA, negate(transpose(mB)), result);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_and_b_transpose_and_negate)
{
    graphblas::IndexArrayType i_mA = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mA = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {0, 1, 1, 0, 0, 0, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    buildmatrix(mA, i_mA, j_mA, v_mA);

    graphblas::IndexArrayType i_mB = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_mB = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mB = {1, 0, 1, 1, 1, 0, 1, 0, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 3);
    buildmatrix(mB, i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 3);

    graphblas::IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    graphblas::IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer = {1, 0, 1, 1, 0, 1, 1, 0, 1};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    buildmatrix(answer, i_answer, j_answer, v_answer);

    mxm(negate(transpose(mA)), negate(transpose(mB)), result);
    BOOST_CHECK_EQUAL(result, answer);
}

BOOST_AUTO_TEST_SUITE_END()
