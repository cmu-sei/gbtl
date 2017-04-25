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
    std::vector<std::vector<double> > m3x3_dense = {{12, 1, 3},
                                                    {0,  0, 6},
                                                    {7,  0, 9}};

    std::vector<std::vector<double> > m3x4_dense = {{5, 0, 1, 2},
                                                    {6, 7, 0, 0},
                                                    {4, 5, 0, 1}};

    std::vector<double> u3_dense = {1, 1, 0};
    std::vector<double> u4_dense = {0, 0, 1, 1};

    std::vector<double> ans3_dense = {13, 0, 7};
    std::vector<double> ans4_dense = { 3, 0, 1};

    //static Matrix<double, DirectedMatrixTag> mAns(mAns_dense);
    GraphBLAS::IndexArrayType i_all3x4 = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    GraphBLAS::IndexArrayType j_all3x4 = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};

    static std::vector<double> v4_ones = {1, 1, 1, 1};
    static std::vector<double> v3_ones = {1, 1, 1};

    static std::vector<std::vector<double> > mOnes_3x4 = {{1, 1, 1, 1},
                                                          {1, 1, 1, 1},
                                                          {1, 1, 1, 1}};

    static std::vector<std::vector<double> > mOnes_3x3 = {{1, 1, 1},
                                                          {1, 1, 1},
                                                          {1, 1, 1}};

    static std::vector<std::vector<double> > mIdentity_3x3 = {{1, 0, 0},
                                                              {0, 1, 0},
                                                              {0, 0, 1}};
}

//****************************************************************************
// Tests without mask
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_bad_dimensions)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(m3x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m3x4_dense, 0.);

    GraphBLAS::Vector<double, GraphBLAS::SparseTag> u3(u3_dense, 0.);

    GraphBLAS::Vector<double, GraphBLAS::SparseTag> w4(4);
    GraphBLAS::Vector<double, GraphBLAS::SparseTag> w3(3);

    // incompatible input dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::mxv(w3,
                        GraphBLAS::Second<double>(),
                        GraphBLAS::ArithmeticSemiring<double>(), mB, u3)),
        GraphBLAS::DimensionException);

    // incompatible output matrix dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::mxv(w4,
                        GraphBLAS::Second<double>(),
                        GraphBLAS::ArithmeticSemiring<double>(), mA, u3)),
        GraphBLAS::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_reg)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(m3x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m3x4_dense, 0.);
    GraphBLAS::Vector<double> u3(u3_dense, 0.);
    GraphBLAS::Vector<double> u4(u4_dense, 0.);
    GraphBLAS::Vector<double> result(3);
    GraphBLAS::Vector<double> ansA(ans3_dense, 0.);
    GraphBLAS::Vector<double> ansB(ans4_dense, 0.);

    GraphBLAS::mxv(result,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, u3);
    BOOST_CHECK_EQUAL(result, ansA);

    GraphBLAS::mxv(result,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mB, u4);
    BOOST_CHECK_EQUAL(result, ansB);
}

#if 0

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_stored_zero_result)
{
    // Build some matrices.
    std::vector<std::vector<int> > A = {{1, 1, 0, 0},
                                        {1, 2, 2, 0},
                                        {0, 2, 3, 3},
                                        {0, 0, 3, 4}};
    std::vector<std::vector<int> > B = {{ 1,-2, 0,  0},
                                        {-1, 1, 0,  0},
                                        { 0, 0, 3, -4},
                                        { 0, 0,-3,  3}};
    GraphBLAS::Matrix<int, GraphBLAS::DirectedMatrixTag> mA(A, 0);
    GraphBLAS::Matrix<int, GraphBLAS::DirectedMatrixTag> mB(B, 0);
    GraphBLAS::Matrix<int, GraphBLAS::DirectedMatrixTag> result(4, 4);

    // use a different sentinel value so that stored zeros are preserved.
    int const NIL(666);
    std::vector<std::vector<int> > ans = {{  0,  -1, NIL, NIL},
                                          { -1,   0,  -6,  -8},
                                          { -2,   2,   0,  -3},
                                          {NIL, NIL,  -3,   0}};
    GraphBLAS::Matrix<int, GraphBLAS::DirectedMatrixTag> answer(ans, NIL);

    GraphBLAS::mxv(result,
                   GraphBLAS::Second<int>(),
                   GraphBLAS::ArithmeticSemiring<int>(), mA, mB);
    BOOST_CHECK_EQUAL(result, answer);
    BOOST_CHECK_EQUAL(result.nvals(), 12);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_duplicate_input)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mat(m, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(4, 4);

    std::vector<std::vector<double> > ans = {{2,  3,  2,  0},
                                             {3,  9, 10,  6},
                                             {2, 10, 22, 21},
                                             {0,  6, 21, 25}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxv(result,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 4);

    std::vector<std::vector<double> > ans = {{112, 159,  87, 31},
                                             { 97, 131,  94, 22},
                                             { 87, 111, 102, 15}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxv(result,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), transpose(mA), mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.);

    std::vector<std::vector<double> > B = {{5, 8, 1},
                                           {2, 6, 7},
                                           {3, 4, 5}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(B, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 3);

    std::vector<std::vector<double> > ans = {{119,  87, 79},
                                             { 66,  80, 62},
                                             {108, 125, 98}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxv(result,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, transpose(mB));

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_and_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.);
    std::vector<std::vector<double> > B =  {{5, 8, 1},
                                            {2, 6, 7},
                                            {3, 4, 5}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(B, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 3);

    std::vector<std::vector<double> > ans =  {{99,  97, 87},
                                              {83, 100, 81},
                                              {72, 105, 78}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxv(result,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), transpose(mA), transpose(mB));

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_equals_c_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mIdentity_3x3, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 3);

    std::vector<std::vector<double> > ans = {{12, 4, 7},
                                             { 7, 5, 8},
                                             { 3, 6, 9}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    //auto answer = GraphBLAS::transpose(mA);

    GraphBLAS::mxv(result,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), transpose(mA), mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
// Tests using a mask with REPLACE
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_masked_replace_bad_dimensions)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.); // 3x3
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense, 0.); // 3x4
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result3x4(mOnes_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{1, 0, 0},
                                                          {1, 1, 0},
                                                          {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 6);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::mxv(result3x4,
                        mMask,
                        GraphBLAS::Second<double>(),
                        GraphBLAS::ArithmeticSemiring<double>(), mA, mB,
                        true)),
        GraphBLAS::DimensionException);
}
//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_masked_replace_reg)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{1, 0, 0, 0},
                                                          {1, 1, 0, 0},
                                                          {1, 1, 1, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x4, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 6);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   0,   0,  0},
                                             { 74,  97,   0,  0},
                                             {119, 157, 112,  0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxv(result,
                   mMask,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, mB,
                   true);
    BOOST_CHECK_EQUAL(result.nvals(), 6);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_masked_replace_reg_stored_zero)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{1, 0, 0, 0},
                                                          {1, 1, 0, 0},
                                                          {1, 1, 1, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x4, 0.);
    mMask.setElement(0, 1, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 7);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   0,   0,  0},
                                             { 74,  97,   0,  0},
                                             {119, 157, 112,  0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxv(result,
                   mMask,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, mB,
                   true);
    BOOST_CHECK_EQUAL(result.nvals(), 6);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_masked_replace_duplicate_input)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mat(m, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_4x4, 0.);

    std::vector<std::vector<double> > ans = {{2,  0,  0,  0},
                                             {3,  9,  0,  0},
                                             {2, 10, 22,  0},
                                             {0,  6, 21, 25}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    static std::vector<std::vector<double> > mMask_4x4 = {{1, 0, 0, 0},
                                                          {1, 1, 0, 0},
                                                          {1, 1, 1, 0},
                                                          {1, 1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_4x4, 0.);

    GraphBLAS::mxv(result,
                   mMask,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat,
                   true);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_masked_replace_a_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{1, 0, 0, 0},
                                                          {1, 1, 0, 0},
                                                          {1, 1, 1, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x4, 0.);

    std::vector<std::vector<double> > ans = {{112,   0,   0,  0},
                                             { 97, 131,   0,  0},
                                             { 87, 111, 102,  0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxv(result,
                   mMask,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), transpose(mA), mB,
                   true);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_masked_replace_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.);

    std::vector<std::vector<double> > B = {{5, 8, 1},
                                           {2, 6, 7},
                                           {3, 4, 5}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(B, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{1, 0, 0},
                                                          {1, 1, 0},
                                                          {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans = {{119,   0,  0},
                                             { 66,  80,  0},
                                             {108, 125, 98}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxv(result,
                   mMask,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, transpose(mB),
                   true);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_masked_replace_a_and_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.);
    std::vector<std::vector<double> > B =  {{5, 8, 1},
                                            {2, 6, 7},
                                            {3, 4, 5}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(B, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{1, 0, 0},
                                                          {1, 1, 0},
                                                          {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans =  {{99,   0,  0},
                                              {83, 100,  0},
                                              {72, 105, 78}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxv(result,
                   mMask,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), transpose(mA), transpose(mB),
                   true);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_masked_replace_a_equals_c_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mIdentity_3x3, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{1, 0, 0},
                                                          {1, 1, 0},
                                                          {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans = {{12, 0, 0},
                                             { 7, 5, 0},
                                             { 3, 6, 9}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    //auto answer = GraphBLAS::transpose(mA);

    GraphBLAS::mxv(result,
                   mMask,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), transpose(mA), mB,
                   true);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
// Tests using a mask with MERGE semantics
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_masked_bad_dimensions)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.); // 3x3
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense, 0.); // 3x4
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result3x4(mOnes_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{1, 0, 0},
                                                          {1, 1, 0},
                                                          {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 6);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::mxv(result3x4,
                        mMask,
                        GraphBLAS::Second<double>(),
                        GraphBLAS::ArithmeticSemiring<double>(), mA, mB)),
        GraphBLAS::DimensionException);
}
//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_masked_reg)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{1, 0, 0, 0},
                                                          {1, 1, 0, 0},
                                                          {1, 1, 1, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x4, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 6);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   1,   1,  1},
                                             { 74,  97,   1,  1},
                                             {119, 157, 112,  1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxv(result,
                   mMask,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, mB);
    BOOST_CHECK_EQUAL(result.nvals(), 12);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_masked_reg_stored_zero)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{1, 0, 0, 0},
                                                          {1, 1, 0, 0},
                                                          {1, 1, 1, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x4, 0.);
    mMask.setElement(0, 1, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 7);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   1,   1,  1},
                                             { 74,  97,   1,  1},
                                             {119, 157, 112,  1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxv(result,
                   mMask,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, mB);
    BOOST_CHECK_EQUAL(result.nvals(), 12);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_masked_duplicate_input)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mat(m, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_4x4, 0.);

    std::vector<std::vector<double> > ans = {{2,  1,  1,  1},
                                             {3,  9,  1,  1},
                                             {2, 10, 22,  1},
                                             {0,  6, 21, 25}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    static std::vector<std::vector<double> > mMask_4x4 = {{1, 0, 0, 0},
                                                          {1, 1, 0, 0},
                                                          {1, 1, 1, 0},
                                                          {1, 1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_4x4, 0.);

    GraphBLAS::mxv(result,
                   mMask,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_masked_a_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{1, 0, 0, 0},
                                                          {1, 1, 0, 0},
                                                          {1, 1, 1, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x4, 0.);

    std::vector<std::vector<double> > ans = {{112,   1,   1,  1},
                                             { 97, 131,   1,  1},
                                             { 87, 111, 102,  1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxv(result,
                   mMask,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), transpose(mA), mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_masked_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.);

    std::vector<std::vector<double> > B = {{5, 8, 1},
                                           {2, 6, 7},
                                           {3, 4, 5}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(B, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{1, 0, 0},
                                                          {1, 1, 0},
                                                          {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans = {{119,   1,  1},
                                             { 66,  80,  1},
                                             {108, 125, 98}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxv(result,
                   mMask,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, transpose(mB));

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_masked_a_and_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.);
    std::vector<std::vector<double> > B =  {{5, 8, 1},
                                            {2, 6, 7},
                                            {3, 4, 5}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(B, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{1, 0, 0},
                                                          {1, 1, 0},
                                                          {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans =  {{99,   1,  1},
                                              {83, 100,  1},
                                              {72, 105, 78}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxv(result,
                   mMask,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), transpose(mA), transpose(mB));

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_masked_a_equals_c_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mIdentity_3x3, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{1, 0, 0},
                                                          {1, 1, 0},
                                                          {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans = {{12, 1, 1},
                                             { 7, 5, 1},
                                             { 3, 6, 9}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    //auto answer = GraphBLAS::transpose(mA);

    GraphBLAS::mxv(result,
                   mMask,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), transpose(mA), mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
// Tests using a complemented mask with REPLACE semantics
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_scmp_masked_replace_bad_dimensions)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.); // 3x3
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense, 0.); // 3x4
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result3x4(mOnes_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{1, 0, 0},
                                                          {1, 1, 0},
                                                          {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 6);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::mxv(result3x4,
                        GraphBLAS::complement(mMask),
                        GraphBLAS::Second<double>(),
                        GraphBLAS::ArithmeticSemiring<double>(), mA, mB,
                        true)),
        GraphBLAS::DimensionException);
}
//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_scmp_masked_replace_reg)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{0, 1, 1, 1},
                                                          {0, 0, 1, 1},
                                                          {0, 0, 0, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x4, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 6);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   0,   0,  0},
                                             { 74,  97,   0,  0},
                                             {119, 157, 112,  0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxv(result,
                   GraphBLAS::complement(mMask),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, mB,
                   true);
    BOOST_CHECK_EQUAL(result.nvals(), 6);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_scmp_masked_replace_reg_stored_zero)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{0, 1, 1, 1},
                                                          {0, 0, 1, 1},
                                                          {0, 0, 0, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x4, 0.);
    mMask.setElement(0, 0, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 7);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   0,   0,  0},
                                             { 74,  97,   0,  0},
                                             {119, 157, 112,  0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxv(result,
                   GraphBLAS::complement(mMask),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, mB,
                   true);
    BOOST_CHECK_EQUAL(result.nvals(), 6);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_scmp_masked_replace_duplicate_input)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mat(m, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_4x4, 0.);

    std::vector<std::vector<double> > ans = {{2,  0,  0,  0},
                                             {3,  9,  0,  0},
                                             {2, 10, 22,  0},
                                             {0,  6, 21, 25}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    static std::vector<std::vector<double> > mMask_4x4 = {{0, 1, 1, 1},
                                                          {0, 0, 1, 1},
                                                          {0, 0, 0, 1},
                                                          {0, 0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_4x4, 0.);

    GraphBLAS::mxv(result,
                   GraphBLAS::complement(mMask),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat,
                   true);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_scmp_masked_replace_a_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{0, 1, 1, 1},
                                                          {0, 0, 1, 1},
                                                          {0, 0, 0, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x4, 0.);

    std::vector<std::vector<double> > ans = {{112,   0,   0,  0},
                                             { 97, 131,   0,  0},
                                             { 87, 111, 102,  0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxv(result,
                   GraphBLAS::complement(mMask),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), transpose(mA), mB,
                   true);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_scmp_masked_replace_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.);

    std::vector<std::vector<double> > B = {{5, 8, 1},
                                           {2, 6, 7},
                                           {3, 4, 5}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(B, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{0, 1, 1},
                                                          {0, 0, 1},
                                                          {0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans = {{119,   0,  0},
                                             { 66,  80,  0},
                                             {108, 125, 98}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxv(result,
                   GraphBLAS::complement(mMask),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, transpose(mB),
                   true);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_scmp_masked_replace_a_and_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.);
    std::vector<std::vector<double> > B =  {{5, 8, 1},
                                            {2, 6, 7},
                                            {3, 4, 5}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(B, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{0, 1, 1},
                                                          {0, 0, 1},
                                                          {0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans =  {{99,   0,  0},
                                              {83, 100,  0},
                                              {72, 105, 78}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxv(result,
                   GraphBLAS::complement(mMask),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), transpose(mA), transpose(mB),
                   true);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_scmp_masked_replace_a_equals_c_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mIdentity_3x3, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{0, 1, 1},
                                                          {0, 0, 1},
                                                          {0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans = {{12, 0, 0},
                                             { 7, 5, 0},
                                             { 3, 6, 9}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    //auto answer = GraphBLAS::transpose(mA);

    GraphBLAS::mxv(result,
                   GraphBLAS::complement(mMask),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), transpose(mA), mB,
                   true);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
// Tests using a complemented mask (with merge semantics)
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_scmp_masked_bad_dimensions)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.); // 3x3
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense, 0.); // 3x4
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result3x4(mOnes_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{1, 0, 0},
                                                          {1, 1, 0},
                                                          {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 6);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::mxv(result3x4,
                        GraphBLAS::complement(mMask),
                        GraphBLAS::Second<double>(),
                        GraphBLAS::ArithmeticSemiring<double>(), mA, mB)),
        GraphBLAS::DimensionException);
}
//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_scmp_masked_reg)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{0, 1, 1, 1},
                                                          {0, 0, 1, 1},
                                                          {0, 0, 0, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x4, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 6);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   1,   1,  1},
                                             { 74,  97,   1,  1},
                                             {119, 157, 112,  1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxv(result,
                   GraphBLAS::complement(mMask),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, mB);
    BOOST_CHECK_EQUAL(result.nvals(), 12);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_scmp_masked_reg_stored_zero)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{0, 1, 1, 1},
                                                          {0, 0, 1, 1},
                                                          {0, 0, 0, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x4, 0.);
    mMask.setElement(0, 0, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 7);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   1,   1,  1},
                                             { 74,  97,   1,  1},
                                             {119, 157, 112,  1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxv(result,
                   GraphBLAS::complement(mMask),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, mB);
    BOOST_CHECK_EQUAL(result.nvals(), 12);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_scmp_masked_duplicate_input)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mat(m, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_4x4, 0.);

    std::vector<std::vector<double> > ans = {{2,  1,  1,  1},
                                             {3,  9,  1,  1},
                                             {2, 10, 22,  1},
                                             {0,  6, 21, 25}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    static std::vector<std::vector<double> > mMask_4x4 = {{0, 1, 1, 1},
                                                          {0, 0, 1, 1},
                                                          {0, 0, 0, 1},
                                                          {0, 0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_4x4, 0.);

    GraphBLAS::mxv(result,
                   GraphBLAS::complement(mMask),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_scmp_masked_a_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{0, 1, 1, 1},
                                                          {0, 0, 1, 1},
                                                          {0, 0, 0, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x4, 0.);

    std::vector<std::vector<double> > ans = {{112,   1,   1,  1},
                                             { 97, 131,   1,  1},
                                             { 87, 111, 102,  1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxv(result,
                   GraphBLAS::complement(mMask),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), transpose(mA), mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_scmp_masked_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.);

    std::vector<std::vector<double> > B = {{5, 8, 1},
                                           {2, 6, 7},
                                           {3, 4, 5}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(B, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{0, 1, 1},
                                                          {0, 0, 1},
                                                          {0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans = {{119,   1,  1},
                                             { 66,  80,  1},
                                             {108, 125, 98}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxv(result,
                   GraphBLAS::complement(mMask),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, transpose(mB));

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_scmp_masked_a_and_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.);
    std::vector<std::vector<double> > B =  {{5, 8, 1},
                                            {2, 6, 7},
                                            {3, 4, 5}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(B, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{0, 1, 1},
                                                          {0, 0, 1},
                                                          {0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans =  {{99,   1,  1},
                                              {83, 100,  1},
                                              {72, 105, 78}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxv(result,
                   GraphBLAS::complement(mMask),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), transpose(mA), transpose(mB));

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_scmp_masked_a_equals_c_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mIdentity_3x3, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{0, 1, 1},
                                                          {0, 0, 1},
                                                          {0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans = {{12, 1, 1},
                                             { 7, 5, 1},
                                             { 3, 6, 9}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    //auto answer = GraphBLAS::transpose(mA);

    GraphBLAS::mxv(result,
                   GraphBLAS::complement(mMask),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), transpose(mA), mB);

    BOOST_CHECK_EQUAL(result, answer);
}

#endif

BOOST_AUTO_TEST_SUITE_END()
