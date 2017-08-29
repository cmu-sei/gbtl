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

using namespace GraphBLAS;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE homework_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(homework_suite)

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_single_argument_transpose)
{
    IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    std::vector<double> v_answer = {1, 4, 7, 2, 5, 8, 3, 6, 9};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    answer.build(i_mA, j_mA, v_answer);

    auto result = transpose(mA);

    for (IndexType ix = 0; ix < 3; ++ix)
        for (IndexType iy = 0; iy < 3; ++iy)
            BOOST_CHECK_EQUAL(result.extractElement(ix, iy),
                              answer.extractElement(ix, iy));
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_complement)
{
    IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_aA    = {0, 0, 1, 1, 2, 2};
    IndexArrayType j_aA    = {1, 2, 0, 2, 0, 1};
    std::vector<bool> v_answer(6, true);
    Matrix<bool, DirectedMatrixTag> answer(3, 3);
    answer.build(i_aA, j_aA, v_answer);

    auto result = complement(mA);

    for (IndexType ix = 0; ix < 3; ++ix)
    {
        for (IndexType iy = 0; iy < 3; ++iy)
        {
            BOOST_CHECK_EQUAL(result.hasElement(ix, iy),
                              answer.hasElement(ix, iy));
            if (answer.hasElement(ix, iy))
            {
                BOOST_CHECK_EQUAL(result.extractElement(ix, iy),
                                  answer.extractElement(ix, iy));
            }
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_masked)
{
    IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mB    = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2};
    IndexArrayType j_mB    = {0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3};
    std::vector<double> v_mB = {5, 8, 1, 2, 6, 7, 3, 4, 5, 9, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 4);
    mB.build(i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 4);

    IndexArrayType i_answer = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    IndexArrayType j_answer = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    std::vector<double> v_answer = {114, 160, 60, 27, 74, 97,
                                    73, 14, 119, 157, 112, 23};
    Matrix<double, DirectedMatrixTag> answer(3, 4);
    answer.build(i_answer, j_answer, v_answer);

    Matrix<unsigned int, DirectedMatrixTag> mask(3,4);
    std::vector<unsigned int> v_mask(i_answer.size(), 1);
    mask.build(i_answer, j_answer, v_mask);

    mxm(result,
        mask, GraphBLAS::NoAccumulate(),
        GraphBLAS::ArithmeticSemiring<double>(),
        mA, mB);

    BOOST_CHECK_EQUAL(result, answer);
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_transpose)
{
    IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mB    = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2};
    IndexArrayType j_mB    = {0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3};
    std::vector<double> v_mB = {5, 8, 1, 2, 6, 7, 3, 4, 5, 9, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 4);
    mB.build(i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 4);

    IndexArrayType i_answer = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    IndexArrayType j_answer = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    std::vector<double> v_answer = {112, 159, 87, 31, 97, 131,
                                    94, 22, 87, 111, 102, 15};
    Matrix<double, DirectedMatrixTag> answer(3, 4);
    answer.build(i_answer, j_answer, v_answer);

    mxm(result,
        GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
        GraphBLAS::ArithmeticSemiring<double>(),
        transpose(mA), mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_b_transpose)
{
    IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mB    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mB    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mB = {5, 8, 1, 2, 6, 7, 3, 4, 5};
    Matrix<double, DirectedMatrixTag> mB(3, 3);
    mB.build(i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 3);

    IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer = {119, 87, 79, 66, 80, 62, 108, 125, 98};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    answer.build(i_answer, j_answer, v_answer);

    mxm(result,
        GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
        GraphBLAS::ArithmeticSemiring<double>(),
        mA, transpose(mB));

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_and_b_transpose)
{
    IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mB    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mB    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mB = {5, 8, 1, 2, 6, 7, 3, 4, 5};
    Matrix<double, DirectedMatrixTag> mB(3, 3);
    mB.build(i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 3);

    IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer = {99, 97, 87, 83, 100, 81, 72, 105, 78};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    answer.build(i_answer, j_answer, v_answer);

    mxm(result,
        GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
        GraphBLAS::ArithmeticSemiring<double>(),
        transpose(mA), transpose(mB));

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_equals_c_transpose)
{
    IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mB    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mB    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mB = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 3);
    mB.build(i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 3);

    IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer = {12, 4, 7, 7, 5, 8, 3, 6, 9};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    answer.build(i_answer, j_answer, v_answer);

    mxm(result,
        GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
        GraphBLAS::ArithmeticSemiring<double>(),
        transpose(mA), mB);

    BOOST_CHECK_EQUAL(result, answer);
}

/// @todo Complementing transpose not supported
/// @todo Complementing operands not supperted
#if 0

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_complement)
{
    IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {0, 1, 1, 0, 0, 0, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mB    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mB    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mB = {1, 0, 1, 1, 1, 0, 1, 0, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 3);
    mB.build(i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 3);

    IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer = {1, 0, 1, 3, 1, 2, 0, 0, 0};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    answer.build(i_answer, j_answer, v_answer);

    mxm(result,
        GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
        GraphBLAS::ArithmeticSemiring<double>(),
        complement(mA), mB);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_b_complement)
{
    IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {0, 1, 1, 0, 0, 0, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mB    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mB    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mB = {1, 0, 1, 1, 1, 0, 1, 0, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 3);
    mB.build(i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 3);

    IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer = {0, 1, 1, 0, 0, 0, 0, 2, 1};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    answer.build(i_answer, j_answer, v_answer);

    mxm(result,
        GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
        GraphBLAS::ArithmeticSemiring<double>(),
        mA, complement(mB));
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_and_b_complement)
{
    IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {0, 1, 1, 0, 0, 0, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mB    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mB    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mB = {1, 0, 1, 1, 1, 0, 1, 0, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 3);
    mB.build(i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 3);

    IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer = {0, 1, 0, 0, 2, 1, 0, 0, 0};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    answer.build(i_answer, j_answer, v_answer);

    mxm(result,
        GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
        GraphBLAS::ArithmeticSemiring<double>(),
        complement(mA), complement(mB));
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_equals_c_complement)
{
    IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {0, 1, 1, 0, 0, 0, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mB    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mB    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mB = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 3);
    mB.build(i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 3);

    IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer = {1, 0, 0, 1, 1, 1, 0, 0, 0};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    answer.build(i_answer, j_answer, v_answer);

    mxm(result,
        GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
        GraphBLAS::ArithmeticSemiring<double>(),
        complement(mA), mB);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_transpose_and_complement)
{
    IndexArrayType i_mA = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {0, 1, 1, 0, 0, 0, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mB = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mB = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mB = {1, 0, 1, 1, 1, 0, 1, 0, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 3);
    mB.build(i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 3);

    IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer = {2, 1, 1, 1, 1, 0, 1, 1, 0};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    answer.build(i_answer, j_answer, v_answer);

    mxm(complement(transpose(mA)), mB, result);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_b_transpose_and_complement)
{
    IndexArrayType i_mA = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {0, 1, 1, 0, 0, 0, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mB = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mB = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mB = {1, 0, 1, 1, 1, 0, 1, 0, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 3);
    mB.build(i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 3);

    IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer = {1, 1, 1, 0, 0, 0, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    answer.build(i_answer, j_answer, v_answer);

    mxm(mA, complement(transpose(mB)), result);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_and_b_transpose_and_complement)
{
    IndexArrayType i_mA = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {0, 1, 1, 0, 0, 0, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mB = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mB = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mB = {1, 0, 1, 1, 1, 0, 1, 0, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 3);
    mB.build(i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 3);

    IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer = {1, 0, 1, 1, 0, 1, 1, 0, 1};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    answer.build(i_answer, j_answer, v_answer);

    mxm(result,
        GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
        complement(transpose(mA)), complement(transpose(mB)));
    BOOST_CHECK_EQUAL(result, answer);
}
#endif

BOOST_AUTO_TEST_SUITE_END()
