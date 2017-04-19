/*
 * Copyright (c) 2017 Carnegie Mellon University and The Trustees of
 * Indiana University.
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

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE sparse_mxm_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(sparse_mxm_suite)

//****************************************************************************

namespace
{
    //static std::vector<std::vector<double> > mA_dense = {{12, 7, 3},
    //                                                     {4,  5, 6},
    //                                                     {7,  8, 9}};
    //static Matrix<double, DirectedMatrixTag> mA(mA_dense);
    GraphBLAS::IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    GraphBLAS::IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};

    //static std::vector<std::vector<double> > mB_dense = {{5, 8, 1, 2},
    //                                                     {6, 7, 3, -},
    //                                                     {4, 5, 9, 1}};
    //static Matrix<double, DirectedMatrixTag> mB(mB_dense);
    GraphBLAS::IndexArrayType i_mB    = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2};
    GraphBLAS::IndexArrayType j_mB    = {0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3};
    std::vector<double> v_mB = {5, 8, 1, 2, 6, 7, 3, 4, 5, 9, 1};

    //static std::vector<std::vector<double> > mAns_dense = {{114, 160,  60, 27},
    //                                                       { 74,  97,  73, 14},
    //                                                       {119, 157, 112, 23}};
    //static Matrix<double, DirectedMatrixTag> mAns(mAns_dense);
    GraphBLAS::IndexArrayType i_answer = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    GraphBLAS::IndexArrayType j_answer = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    std::vector<double> v_answer = {114, 160, 60, 27, 74, 97,
                                    73, 14, 119, 157, 112, 23};

    //static std::vector<std::vector<double> > mC_dense = {{1, 1, 1, 1},
    //                                                     {1, 1, 1, 1},
    //                                                     {1, 1, 1, 1}};
    //static Matrix<double, DirectedMatrixTag> mC(mC_dense);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(mxm_bad_dimensions)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA, GraphBLAS::Second<double>());

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(3, 4);
    mB.build(i_mB, j_mB, v_mB, GraphBLAS::Second<double>());

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 4);

    BOOST_CHECK_THROW(
        (GraphBLAS::mxm(result,
                        GraphBLAS::Second<double>(),
                        GraphBLAS::ArithmeticSemiring<double>(), mB, mA)),
        GraphBLAS::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(mxm_reg)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA, GraphBLAS::Second<double>());

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(3, 4);
    mB.build(i_mB, j_mB, v_mB, GraphBLAS::Second<double>());

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 4);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(3, 4);
    answer.build(i_answer, j_answer, v_answer, GraphBLAS::Second<double>());

    GraphBLAS::mxm(result,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, mB);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(mxm_duplicate_input)
{
    // Build some matrices.
    GraphBLAS::IndexArrayType i = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    GraphBLAS::IndexArrayType j = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double>       v = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mat(4, 4);
    mat.build(i, j, v);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> m3(4, 4);

    GraphBLAS::IndexArrayType i_answer = {0, 0, 0,  1, 1, 1, 1,
                                          2, 2, 2, 2,  3, 3, 3};
    GraphBLAS::IndexArrayType j_answer = {0, 1, 2,  0, 1, 2, 3,
                                          0, 1, 2, 3,  1, 2, 3};
    std::vector<double> v_answer = {2, 3, 2,  3, 9, 10, 6,
                                    2, 10, 22, 21,  6, 21, 25};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(4, 4);
    answer.build(i_answer, j_answer, v_answer);

    GraphBLAS::mxm(m3,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat);

    BOOST_CHECK_EQUAL(m3, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_masked)
{
    GraphBLAS::IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    GraphBLAS::IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    GraphBLAS::IndexArrayType i_mB    = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2};
    GraphBLAS::IndexArrayType j_mB    = {0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3};
    std::vector<double> v_mB = {5, 8, 1, 2, 6, 7, 3, 4, 5, 9, 1};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(3, 4);
    mB.build(i_mB, j_mB, v_mB);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 4);

    GraphBLAS::IndexArrayType i_answer = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    GraphBLAS::IndexArrayType j_answer = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    std::vector<double> v_answer = {114, 160, 60, 27, 74, 97,
                                    73, 14, 119, 157, 112, 23};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(3, 4);
    answer.build(i_answer, j_answer, v_answer);

    GraphBLAS::Matrix<unsigned int, GraphBLAS::DirectedMatrixTag> mask(3,4);
    std::vector<unsigned int> v_mask(i_answer.size(), 1);
    mask.build(i_answer, j_answer, v_mask);

    GraphBLAS::mxm(result,
                   mask,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, mB);

    BOOST_CHECK_EQUAL(result, answer);
}

#if 0
//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_not_all_one_masked)
{
    GraphBLAS::IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    GraphBLAS::IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    GraphBLAS::IndexArrayType i_mB    = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2};
    GraphBLAS::IndexArrayType j_mB    = {0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3};
    std::vector<double> v_mB = {5, 8, 1, 2, 6, 7, 3, 4, 5, 9, 1};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(3, 4);
    mB.build(i_mB, j_mB, v_mB);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 4);

    GraphBLAS::IndexArrayType i_answer = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2};
    GraphBLAS::IndexArrayType j_answer = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2};
    std::vector<double> v_answer = {114, 160, 60, 27, 74, 97,
                                    73, 14, 119, 157, 112};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(3, 4);
    answer.build(i_answer, j_answer, v_answer);

    GraphBLAS::Matrix<unsigned int, GraphBLAS::DirectedMatrixTag> mask(3,4);
    std::vector<unsigned int> v_mask(i_answer.size(), 1);
    .build(mask, i_answer, j_answer, v_mask);

    mxmMasked(mA, mB, result, mask);

    BOOST_CHECK_EQUAL(result, answer);
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_transpose)
{
    GraphBLAS::IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    GraphBLAS::IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    GraphBLAS::IndexArrayType i_mB    = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2};
    GraphBLAS::IndexArrayType j_mB    = {0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3};
    std::vector<double> v_mB = {5, 8, 1, 2, 6, 7, 3, 4, 5, 9, 1};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(3, 4);
    mB.build(i_mB, j_mB, v_mB);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 4);

    GraphBLAS::IndexArrayType i_answer = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    GraphBLAS::IndexArrayType j_answer = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    std::vector<double> v_answer = {112, 159, 87, 31, 97, 131,
                                    94, 22, 87, 111, 102, 15};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(3, 4);
    answer.build(i_answer, j_answer, v_answer);

    mxm(transpose(mA), mB, result);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_b_transpose)
{
    GraphBLAS::IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    GraphBLAS::IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    GraphBLAS::IndexArrayType i_mB    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    GraphBLAS::IndexArrayType j_mB    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mB = {5, 8, 1, 2, 6, 7, 3, 4, 5};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(3, 3);
    mB.build(i_mB, j_mB, v_mB);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 3);

    GraphBLAS::IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    GraphBLAS::IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer = {119, 87, 79, 66, 80, 62, 108, 125, 98};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(3, 3);
    answer.build(i_answer, j_answer, v_answer);

    mxm(mA, transpose(mB), result);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_and_b_transpose)
{
    GraphBLAS::IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    GraphBLAS::IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    GraphBLAS::IndexArrayType i_mB    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    GraphBLAS::IndexArrayType j_mB    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mB = {5, 8, 1, 2, 6, 7, 3, 4, 5};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(3, 3);
    mB.build(i_mB, j_mB, v_mB);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 3);

    GraphBLAS::IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    GraphBLAS::IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer = {99, 97, 87, 83, 100, 81, 72, 105, 78};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(3, 3);
    answer.build(i_answer, j_answer, v_answer);

    mxm(transpose(mA), transpose(mB), result);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_equals_c_transpose)
{
    GraphBLAS::IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    GraphBLAS::IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    GraphBLAS::IndexArrayType i_mB    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    GraphBLAS::IndexArrayType j_mB    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mB = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(3, 3);
    mB.build(i_mB, j_mB, v_mB);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 3);

    GraphBLAS::IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    GraphBLAS::IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer = {12, 4, 7, 7, 5, 8, 3, 6, 9};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(3, 3);
    answer.build(i_answer, j_answer, v_answer);

    mxm(transpose(mA), mB, result);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_negate)
{
    //GraphBLAS::IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    //GraphBLAS::IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    //std::vector<double> v_mA = {0, 1, 1, 0, 0, 0, 1, 1, 1};
    //GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(3, 3);
    //mA.build(i_mA, j_mA, v_mA);

    //GraphBLAS::IndexArrayType i_mB    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    //GraphBLAS::IndexArrayType j_mB    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    //std::vector<double> v_mB = {1, 0, 1, 1, 1, 0, 1, 0, 1};
    //GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(3, 3);
    //mB.build(i_mB, j_mB, v_mB);

    //GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 3);

    //GraphBLAS::IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    //GraphBLAS::IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    //std::vector<double> v_answer = {1, 0, 1, 3, 1, 2, 0, 0, 0};
    //GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(3, 3);
    //answer.build(i_answer, j_answer, v_answer);

    GraphBLAS::IndexArrayType i_mA    = {0, 0, 2, 2, 2};
    GraphBLAS::IndexArrayType j_mA    = {1, 2, 0, 1, 2};
    std::vector<double> v_mA =          {1, 1, 1, 1, 1};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    GraphBLAS::IndexArrayType i_mB    = {0, 0, 1, 1, 2, 2};
    GraphBLAS::IndexArrayType j_mB    = {0, 2, 0, 1, 0, 2};
    std::vector<double> v_mB =          {1, 1, 1, 1, 1, 1};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(3, 3);
    mB.build(i_mB, j_mB, v_mB);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 3);

    GraphBLAS::IndexArrayType i_answer = {0, 0, 1, 1, 1};
    GraphBLAS::IndexArrayType j_answer = {0, 2, 0, 1, 2};
    std::vector<double> v_answer =       {1, 1, 3, 1, 2};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(3, 3);
    answer.build(i_answer, j_answer, v_answer);

    mxm(negate(mA), mB, result);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_b_negate)
{
    //GraphBLAS::IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    //GraphBLAS::IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    //std::vector<double> v_mA = {0, 1, 1, 0, 0, 0, 1, 1, 1};
    //GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(3, 3);
    //mA.build(i_mA, j_mA, v_mA);

    //GraphBLAS::IndexArrayType i_mB    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    //GraphBLAS::IndexArrayType j_mB    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    //std::vector<double> v_mB = {1, 0, 1, 1, 1, 0, 1, 0, 1};
    //GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(3, 3);
    //mB.build(i_mB, j_mB, v_mB);

    //GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 3);

    //GraphBLAS::IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    //GraphBLAS::IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    //std::vector<double> v_answer = {0, 1, 1, 0, 0, 0, 0, 2, 1};
    //GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(3, 3);
    //answer.build(i_answer, j_answer, v_answer);

    GraphBLAS::IndexArrayType i_mA    = {0, 0, 2, 2, 2};
    GraphBLAS::IndexArrayType j_mA    = {1, 2, 0, 1, 2};
    std::vector<double> v_mA =          {1, 1, 1, 1, 1};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    GraphBLAS::IndexArrayType i_mB    = {0, 0, 1, 1, 2, 2};
    GraphBLAS::IndexArrayType j_mB    = {0, 2, 0, 1, 0, 2};
    std::vector<double> v_mB =          {1, 1, 1, 1, 1, 1};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(3, 3);
    mB.build(i_mB, j_mB, v_mB);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 3);

    GraphBLAS::IndexArrayType i_answer = {0, 0, 2, 2};
    GraphBLAS::IndexArrayType j_answer = {1, 2, 1, 2};
    std::vector<double> v_answer =       {1, 1, 2, 1};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(3, 3);
    answer.build(i_answer, j_answer, v_answer);

    mxm(mA, negate(mB), result);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_and_b_negate)
{
    //GraphBLAS::IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    //GraphBLAS::IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    //std::vector<double> v_mA = {0, 1, 1, 0, 0, 0, 1, 1, 1};
    //GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(3, 3);
    //mA.build(i_mA, j_mA, v_mA);

    //GraphBLAS::IndexArrayType i_mB    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    //GraphBLAS::IndexArrayType j_mB    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    //std::vector<double> v_mB = {1, 0, 1, 1, 1, 0, 1, 0, 1};
    //GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(3, 3);
    //mB.build(i_mB, j_mB, v_mB);

    //GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 3);

    //GraphBLAS::IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    //GraphBLAS::IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    //std::vector<double> v_answer = {0, 1, 0, 0, 2, 1, 0, 0, 0};
    //GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(3, 3);
    //answer.build(i_answer, j_answer, v_answer);

    GraphBLAS::IndexArrayType i_mA    = {0, 0, 2, 2, 2};
    GraphBLAS::IndexArrayType j_mA    = {1, 2, 0, 1, 2};
    std::vector<double> v_mA =          {1, 1, 1, 1, 1};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    GraphBLAS::IndexArrayType i_mB    = {0, 0, 1, 1, 2, 2};
    GraphBLAS::IndexArrayType j_mB    = {0, 2, 0, 1, 0, 2};
    std::vector<double> v_mB =          {1, 1, 1, 1, 1, 1};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(3, 3);
    mB.build(i_mB, j_mB, v_mB);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 3);

    GraphBLAS::IndexArrayType i_answer = {0, 1, 1};
    GraphBLAS::IndexArrayType j_answer = {1, 1, 2};
    std::vector<double> v_answer =       {1, 2, 1};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(3, 3);
    answer.build(i_answer, j_answer, v_answer);

    mxm(negate(mA), negate(mB), result);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_equals_c_negate)
{
    //GraphBLAS::IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    //GraphBLAS::IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    //std::vector<double> v_mA = {0, 1, 1, 0, 0, 0, 1, 1, 1};
    //GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(3, 3);
    //mA.build(i_mA, j_mA, v_mA);

    //GraphBLAS::IndexArrayType i_mB    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    //GraphBLAS::IndexArrayType j_mB    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    //std::vector<double> v_mB = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    //GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(3, 3);
    //mB.build(i_mB, j_mB, v_mB);

    //GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 3);

    //GraphBLAS::IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    //GraphBLAS::IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    //std::vector<double> v_answer = {1, 0, 0, 1, 1, 1, 0, 0, 0};
    //GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(3, 3);
    //answer.build(i_answer, j_answer, v_answer);

    GraphBLAS::IndexArrayType i_mA    = {0, 0, 2, 2, 2};
    GraphBLAS::IndexArrayType j_mA    = {1, 2, 0, 1, 2};
    std::vector<double> v_mA =          {1, 1, 1, 1, 1};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    GraphBLAS::IndexArrayType i_mB    = {0, 1, 2};
    GraphBLAS::IndexArrayType j_mB    = {0, 1, 2};
    std::vector<double> v_mB =          {1, 1, 1};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(3, 3);
    mB.build(i_mB, j_mB, v_mB);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 3);

    GraphBLAS::IndexArrayType i_answer = {0, 1, 1, 1};
    GraphBLAS::IndexArrayType j_answer = {0, 0, 1, 2};
    std::vector<double> v_answer =       {1, 1, 1, 1};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(3, 3);
    answer.build(i_answer, j_answer, v_answer);

    mxm(negate(mA), mB, result);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_transpose_and_negate)
{
    //GraphBLAS::IndexArrayType i_mA = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    //GraphBLAS::IndexArrayType j_mA = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    //std::vector<double> v_mA = {0, 1, 1, 0, 0, 0, 1, 1, 1};
    //GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(3, 3);
    //mA.build(i_mA, j_mA, v_mA);

    //GraphBLAS::IndexArrayType i_mB = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    //GraphBLAS::IndexArrayType j_mB = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    //std::vector<double> v_mB = {1, 0, 1, 1, 1, 0, 1, 0, 1};
    //GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(3, 3);
    //mB.build(i_mB, j_mB, v_mB);

    //GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 3);

    //GraphBLAS::IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    //GraphBLAS::IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    //std::vector<double> v_answer = {2, 1, 1, 1, 1, 0, 1, 1, 0};
    //GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(3, 3);
    //answer.build(i_answer, j_answer, v_answer);

    GraphBLAS::IndexArrayType i_mA = {0, 0, 2, 2, 2};
    GraphBLAS::IndexArrayType j_mA = {1, 2, 0, 1, 2};
    std::vector<double> v_mA =       {1, 1, 1, 1, 1};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    GraphBLAS::IndexArrayType i_mB = {0, 0, 1, 1, 2, 2};
    GraphBLAS::IndexArrayType j_mB = {0, 2, 0, 1, 0, 2};
    std::vector<double> v_mB =       {1, 1, 1, 1, 1, 1};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(3, 3);
    mB.build(i_mB, j_mB, v_mB);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 3);

    GraphBLAS::IndexArrayType i_answer = {0, 0, 0, 1, 1, 2, 2};
    GraphBLAS::IndexArrayType j_answer = {0, 1, 2, 0, 1, 0, 1};
    std::vector<double> v_answer =       {2, 1, 1, 1, 1, 1, 1};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(3, 3);
    answer.build(i_answer, j_answer, v_answer);
    mxm(negate(transpose(mA)), mB, result);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_b_transpose_and_negate)
{
    //GraphBLAS::IndexArrayType i_mA = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    //GraphBLAS::IndexArrayType j_mA = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    //std::vector<double> v_mA = {0, 1, 1, 0, 0, 0, 1, 1, 1};
    //GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(3, 3);
    //mA.build(i_mA, j_mA, v_mA);

    //GraphBLAS::IndexArrayType i_mB = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    //GraphBLAS::IndexArrayType j_mB = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    //std::vector<double> v_mB = {1, 0, 1, 1, 1, 0, 1, 0, 1};
    //GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(3, 3);
    //mB.build(i_mB, j_mB, v_mB);

    //GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 3);

    //GraphBLAS::IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    //GraphBLAS::IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    //std::vector<double> v_answer = {1, 1, 1, 0, 0, 0, 1, 1, 1};
    //GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(3, 3);
    //answer.build(i_answer, j_answer, v_answer);

    GraphBLAS::IndexArrayType i_mA = {0, 0, 2, 2, 2};
    GraphBLAS::IndexArrayType j_mA = {1, 2, 0, 1, 2};
    std::vector<double> v_mA =       {1, 1, 1, 1, 1};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    GraphBLAS::IndexArrayType i_mB = {0, 0, 1, 1, 2, 2};
    GraphBLAS::IndexArrayType j_mB = {0, 2, 0, 1, 0, 2};
    std::vector<double> v_mB =       {1, 1, 1, 1, 1, 1};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(3, 3);
    mB.build(i_mB, j_mB, v_mB);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 3);

    GraphBLAS::IndexArrayType i_answer = {0, 0, 0, 2, 2, 2};
    GraphBLAS::IndexArrayType j_answer = {0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer =       {1, 1, 1, 1, 1, 1};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(3, 3);
    answer.build(i_answer, j_answer, v_answer);
    mxm(mA, negate(transpose(mB)), result);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_and_b_transpose_and_negate)
{
    //GraphBLAS::IndexArrayType i_mA = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    //GraphBLAS::IndexArrayType j_mA = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    //std::vector<double> v_mA = {0, 1, 1, 0, 0, 0, 1, 1, 1};
    //GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(3, 3);
    //mA.build(i_mA, j_mA, v_mA);

    //GraphBLAS::IndexArrayType i_mB = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    //GraphBLAS::IndexArrayType j_mB = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    //std::vector<double> v_mB = {1, 0, 1, 1, 1, 0, 1, 0, 1};
    //GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(3, 3);
    //mB.build(i_mB, j_mB, v_mB);

    //GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 3);

    //GraphBLAS::IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    //GraphBLAS::IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    //std::vector<double> v_answer = {1, 0, 1, 1, 0, 1, 1, 0, 1};
    //GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(3, 3);
    //answer.build(i_answer, j_answer, v_answer);

    GraphBLAS::IndexArrayType i_mA = {0, 0, 2, 2, 2};
    GraphBLAS::IndexArrayType j_mA = {1, 2, 0, 1, 2};
    std::vector<double> v_mA =       {1, 1, 1, 1, 1};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    GraphBLAS::IndexArrayType i_mB = {0, 0, 1, 1, 2, 2};
    GraphBLAS::IndexArrayType j_mB = {0, 2, 0, 1, 0, 2};
    std::vector<double> v_mB =       {1, 1, 1, 1, 1, 1};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(3, 3);
    mB.build(i_mB, j_mB, v_mB);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 3);

    GraphBLAS::IndexArrayType i_answer = {0, 0, 1, 1, 2, 2};
    GraphBLAS::IndexArrayType j_answer = {0, 2, 0, 2, 0, 2};
    std::vector<double> v_answer =       {1, 1, 1, 1, 1, 1};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(3, 3);
    answer.build(i_answer, j_answer, v_answer);
    mxm(negate(transpose(mA)), negate(transpose(mB)), result);
    BOOST_CHECK_EQUAL(result, answer);
}
#endif
BOOST_AUTO_TEST_SUITE_END()
