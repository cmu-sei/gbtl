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
#define BOOST_TEST_MODULE sparse_ewisemult_matrix_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(sparse_ewisemult_matrix_suite)

//****************************************************************************

namespace
{
    std::vector<std::vector<double> > m3x3_dense = {{12, 0, 7},
                                                    {1,  0, 0},
                                                    {3,  6, 9}};

    std::vector<std::vector<double> > eye3x3_dense = {{1, 0, 0},
                                                      {0, 1, 0},
                                                      {0, 0, 1}};

    std::vector<std::vector<double> > twos3x3_dense = {{2, 2, 2},
                                                       {2, 2, 2},
                                                       {2, 2, 2}};

    std::vector<std::vector<double> > zero3x3_dense = {{0, 0, 0},
                                                       {0, 0, 0},
                                                       {0, 0, 0}};

    std::vector<std::vector<double> > ans_twos3x3_dense = {{24,  0, 14},
                                                           { 2,  0,  0},
                                                           { 6, 12, 18}};

    std::vector<std::vector<double> > ans_eye3x3_dense = {{12,  0,  0},
                                                          { 0,  0,  0},
                                                          { 0,  0,  9}};

    std::vector<std::vector<double> > m3x4_dense = {{5, 0, 1, 2},
                                                    {6, 7, 0, 0},
                                                    {4, 5, 0, 1}};

    std::vector<std::vector<double> > m4x3_dense = {{5, 6, 4},
                                                    {0, 7, 5},
                                                    {1, 0, 0},
                                                    {2, 0, 1}};

    std::vector<std::vector<double> > twos4x3_dense = {{2, 2, 2},
                                                       {2, 2, 2},
                                                       {2, 2, 2},
                                                       {2, 2, 2}};

    std::vector<std::vector<double> > twos3x4_dense = {{2, 2, 2, 2},
                                                       {2, 2, 2, 2},
                                                       {2, 2, 2, 2}};

    std::vector<std::vector<double> > ans_twos4x3_dense = {{10, 12,  8},
                                                           {0,  14, 10},
                                                           {2,   0,  0},
                                                           {4,   0,  2}};
    std::vector<std::vector<double> > ans_twos3x4_dense = {{10,  0,  2, 4},
                                                           {12, 14,  0, 0},
                                                           { 8, 10,  0, 2}};
}

//****************************************************************************
// Tests without mask
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_bad_dimensions)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(m3x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m4x3_dense, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(3, 3);

    // incompatible input dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseMult(Result,
                              GraphBLAS::NoMask(),
                              GraphBLAS::NoAccumulate(),
                              GraphBLAS::Times<double>(), mA, mB)),
        GraphBLAS::DimensionException);

    // incompatible output matrix dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseMult(Result,
                              GraphBLAS::NoMask(),
                              GraphBLAS::NoAccumulate(),
                              GraphBLAS::Times<double>(), mB, mB)),
        GraphBLAS::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_reg)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> A(m3x3_dense, 0.);

    // ewise mult with dense matrix
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> B(twos3x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(
        ans_twos3x3_dense, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(3,3);

    GraphBLAS::eWiseMult(Result,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), A, B);
    BOOST_CHECK_EQUAL(Result, Ans);

    // ewise mult with sparse matrix
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> B2(eye3x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans2(
        ans_eye3x3_dense, 0.);
    GraphBLAS::eWiseMult(Result,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), A, B2);
    BOOST_CHECK_EQUAL(Result, Ans2);

    // ewise mult with empty matrix
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> B3(zero3x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans3(
        zero3x3_dense, 0.);
    GraphBLAS::eWiseMult(Result,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), A, B3);
    BOOST_CHECK_EQUAL(Result, Ans3);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_stored_zero_result)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> A(m3x3_dense, 0.);

    // Add a stored zero on the diagonal
    A.setElement(1, 1, 0);
    BOOST_CHECK_EQUAL(A.nvals(), 7);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> B2(eye3x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans2(
        ans_eye3x3_dense, 0.);
    // Add a stored zero on the diagonal
    Ans2.setElement(1, 1, 0);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(3,3);

    GraphBLAS::eWiseMult(Result,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), A, B2);
    BOOST_CHECK_EQUAL(Result, Ans2);
    BOOST_CHECK_EQUAL(Result.nvals(), 3);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> A(m3x4_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> B(twos4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_twos4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(4,3);

    GraphBLAS::eWiseMult(Result,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(),
                         transpose(A), B);
    BOOST_CHECK_EQUAL(Result, Ans);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> B2(m3x4_dense, 0.);
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseMult(Result,
                              GraphBLAS::NoMask(),
                              GraphBLAS::NoAccumulate(),
                              GraphBLAS::Times<double>(),
                              transpose(A), A)),
        GraphBLAS::DimensionException);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result2(3,3);
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseMult(Result2,
                              GraphBLAS::NoMask(),
                              GraphBLAS::NoAccumulate(),
                              GraphBLAS::Times<double>(),
                              transpose(A), B)),
        GraphBLAS::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> A(m3x4_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> B(twos4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_twos3x4_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(3, 4);


    GraphBLAS::eWiseMult(Result,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(),
                         A, transpose(B));
    BOOST_CHECK_EQUAL(Result, Ans);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> B2(m3x4_dense, 0.);
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseMult(Result,
                              GraphBLAS::NoMask(),
                              GraphBLAS::NoAccumulate(),
                              GraphBLAS::Times<double>(),
                              B, transpose(B))),
        GraphBLAS::DimensionException);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result2(3,3);
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseMult(Result2,
                              GraphBLAS::NoMask(),
                              GraphBLAS::NoAccumulate(),
                              GraphBLAS::Times<double>(),
                              A, transpose(B))),
         GraphBLAS::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_and_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> A(m4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> B(twos4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_twos3x4_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(3, 4);


    GraphBLAS::eWiseMult(Result,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(),
                         transpose(A), transpose(B));
    BOOST_CHECK_EQUAL(Result, Ans);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> B2(m3x4_dense, 0.);
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseMult(Result,
                              GraphBLAS::NoMask(),
                              GraphBLAS::NoAccumulate(),
                              GraphBLAS::Times<double>(),
                              transpose(Ans), transpose(B))),
        GraphBLAS::DimensionException);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result2(3,3);
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseMult(Result2,
                              GraphBLAS::NoMask(),
                              GraphBLAS::NoAccumulate(),
                              GraphBLAS::Times<double>(),
                              transpose(A), transpose(B))),
         GraphBLAS::DimensionException);
}

//****************************************************************************
// Tests using a mask with REPLACE
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_masked_replace_bad_dimensions)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(twos3x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(4, 3);

    // incompatible Mask-Output dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseMult(Result,
                              Mask,
                              GraphBLAS::NoAccumulate(),
                              GraphBLAS::Times<double>(), mA, mB,
                              true)),
        GraphBLAS::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_masked_replace_reg)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m4x3_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{1, 1, 1},
                                                    {0, 1, 1},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(mask_dense, 0.);

    {
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{10, 12,  8},
                                                       {0,  14, 10},
                                                       {0,   0,  0},
                                                       {0,   0,  0}};
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(Result,
                             Mask,
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Times<double>(), mA, mB,
                             true);

        BOOST_CHECK_EQUAL(Result.nvals(), 5);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

    {
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{10, 12,  8},
                                                       {0,  14, 10},
                                                       {0,   0,  2},
                                                       {0,   0,  0}};
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(Result,
                             Mask,
                             GraphBLAS::Second<double>(),
                             GraphBLAS::Times<double>(), mA, mB,
                             true);

        BOOST_CHECK_EQUAL(Result.nvals(), 6);
        BOOST_CHECK_EQUAL(Result, Ans);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_masked_replace_reg_stored_zero)
{
    // tests a computed and stored zero
    // tests a stored zero in the mask
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m4x3_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{1, 1, 1},
                                                    {0, 1, 1},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(mask_dense); //, 0.);
    BOOST_CHECK_EQUAL(Mask.nvals(), 12);

    {
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{10, 12,  8},
                                                       {0,  14, 10},
                                                       {0,   0,  0},
                                                       {0,   0,  0}};
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(Result,
                             Mask,
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Times<double>(), mA, mB,
                             true);

        BOOST_CHECK_EQUAL(Result.nvals(), 5);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

    {
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{10, 12,  8},
                                                       {0,  14, 10},
                                                       {0,   0,  2},
                                                       {0,   0,  0}};
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(Result,
                             Mask,
                             GraphBLAS::Second<double>(),
                             GraphBLAS::Times<double>(), mA, mB,
                             true);

        BOOST_CHECK_EQUAL(Result.nvals(), 6);
        BOOST_CHECK_EQUAL(Result, Ans);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_masked_replace_a_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos3x4_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m4x3_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{1, 1, 1},
                                                    {0, 1, 1},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(mask_dense, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{10, 12,  8},
                                                   {0,  14, 10},
                                                   {0,   0,  0},
                                                   {0,   0,  0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

    GraphBLAS::eWiseMult(Result,
                         Mask,
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), transpose(mA), mB,
                         true);

    BOOST_CHECK_EQUAL(Result.nvals(), 5);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_masked_replace_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m3x4_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{1, 1, 1},
                                                    {0, 1, 1},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(mask_dense, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{10, 12,  8},
                                                   {0,  14, 10},
                                                   {0,   0,  0},
                                                   {0,   0,  0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

    GraphBLAS::eWiseMult(Result,
                         Mask,
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), mA, transpose(mB),
                         true);

    BOOST_CHECK_EQUAL(Result.nvals(), 5);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_masked_replace_a_and_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos3x4_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m3x4_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{1, 1, 1},
                                                    {0, 1, 1},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(mask_dense, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{10, 12,  8},
                                                   {0,  14, 10},
                                                   {0,   0,  0},
                                                   {0,   0,  0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

    GraphBLAS::eWiseMult(Result,
                         Mask,
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), transpose(mA), transpose(mB),
                         true);

    BOOST_CHECK_EQUAL(Result.nvals(), 5);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
// Tests using a mask with MERGE semantics
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_masked_bad_dimensions)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(twos3x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(4, 3);

    // incompatible Mask-Output dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseMult(Result,
                              Mask,
                              GraphBLAS::NoAccumulate(),
                              GraphBLAS::Times<double>(), mA, mB)),
        GraphBLAS::DimensionException)
}

#if 0


//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_masked_reg)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(m3x3_dense, 0.);

    GraphBLAS::Vector<double, GraphBLAS::SparseTag> u3(u3_dense, 0.);
    GraphBLAS::Vector<double, GraphBLAS::SparseTag> m3(u3_dense, 0.); // [1 1 -]
    BOOST_CHECK_EQUAL(m3.nvals(), 2);

    {
        GraphBLAS::Vector<double> result(v3_ones, 0.);

        std::vector<double> ans = {13, 1, 1};
        GraphBLAS::Vector<double> answer(ans, 0.);

        GraphBLAS::eWiseMult(result,
                       m3,
                       GraphBLAS::Second<double>(),
                       GraphBLAS::Times<double>(), u3, mA);

        BOOST_CHECK_EQUAL(result.nvals(), 3);
        BOOST_CHECK_EQUAL(result, answer);
    }

    {
        GraphBLAS::Vector<double> result(v3_ones, 0.);

        std::vector<double> ans = {13, 0, 1};
        GraphBLAS::Vector<double> answer(ans, 0.);

        GraphBLAS::eWiseMult(result,
                       m3,
                       GraphBLAS::NoAccumulate(),
                       GraphBLAS::Times<double>(), u3, mA);

        BOOST_CHECK_EQUAL(result.nvals(), 2);
        BOOST_CHECK_EQUAL(result, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_masked_reg_stored_zero)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(m3x3_dense, 0.);

    GraphBLAS::Vector<double, GraphBLAS::SparseTag> u3(u3_dense, 0.);
    GraphBLAS::Vector<double, GraphBLAS::SparseTag> m3(u3_dense); // [1 1 0]
    BOOST_CHECK_EQUAL(m3.nvals(), 3);

    {
        GraphBLAS::Vector<double> result(v3_ones, 0.);

        std::vector<double> ans = {13, 1, 1};
        GraphBLAS::Vector<double> answer(ans, 0.);

        GraphBLAS::eWiseMult(result,
                       m3,
                       GraphBLAS::Second<double>(),
                       GraphBLAS::Times<double>(), u3, mA);

        BOOST_CHECK_EQUAL(result.nvals(), 3);
        BOOST_CHECK_EQUAL(result, answer);
    }

    {
        GraphBLAS::Vector<double> result(v3_ones, 0.);

        std::vector<double> ans = {13, 0, 1};
        GraphBLAS::Vector<double> answer(ans, 0.);

        GraphBLAS::eWiseMult(result,
                       m3,
                       GraphBLAS::NoAccumulate(),
                       GraphBLAS::Times<double>(), u3, mA);

        BOOST_CHECK_EQUAL(result.nvals(), 2);
        BOOST_CHECK_EQUAL(result, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_masked_a_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(m3x3_dense, 0.);

    GraphBLAS::Vector<double, GraphBLAS::SparseTag> u3(u3_dense, 0.); // [1 1 -]
    GraphBLAS::Vector<double, GraphBLAS::SparseTag> m3(u3_dense, 0.); // [1 1 -]
    GraphBLAS::Vector<double> result(v3_ones, 0.);

    BOOST_CHECK_EQUAL(m3.nvals(), 2);

    std::vector<double> ans = {12, 1, 1};
    GraphBLAS::Vector<double> answer(ans, 0.);

    GraphBLAS::eWiseMult(result,
                   m3,
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::Times<double>(), u3, transpose(mA));

    BOOST_CHECK_EQUAL(result.nvals(), 3);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
// Tests using a complemented mask with REPLACE semantics
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_scmp_masked_replace_bad_dimensions)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(m3x3_dense, 0.);

    GraphBLAS::Vector<double, GraphBLAS::SparseTag> u3(u3_dense, 0.);
    GraphBLAS::Vector<double, GraphBLAS::SparseTag> m4(u4_dense, 0.);
    GraphBLAS::Vector<double, GraphBLAS::SparseTag> w3(3);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseMult(w3,
                        GraphBLAS::complement(m4),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Times<double>(), u3, mA,
                        true)),
        GraphBLAS::DimensionException);
}
//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_scmp_masked_replace_reg)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(m3x3_dense, 0.);

    GraphBLAS::Vector<double, GraphBLAS::SparseTag> u3(u3_dense, 0.);

    std::vector<double> scmp_mask_dense = {0., 0., 1.};
    GraphBLAS::Vector<double, GraphBLAS::SparseTag> m3(scmp_mask_dense, 0.);
    BOOST_CHECK_EQUAL(m3.nvals(), 1);

    {
        GraphBLAS::Vector<double> result(v3_ones, 0.);

        std::vector<double> ans = {13, 1, 0};
        GraphBLAS::Vector<double> answer(ans, 0.);

        GraphBLAS::eWiseMult(result,
                       GraphBLAS::complement(m3),
                       GraphBLAS::Second<double>(),
                       GraphBLAS::Times<double>(), u3, mA,
                       true);

        BOOST_CHECK_EQUAL(result.nvals(), 2);
        BOOST_CHECK_EQUAL(result, answer);
    }

    {
        GraphBLAS::Vector<double> result(v3_ones, 0.);

        std::vector<double> ans = {13, 0, 0};
        GraphBLAS::Vector<double> answer(ans, 0.);

        GraphBLAS::eWiseMult(result,
                       GraphBLAS::complement(m3),
                       GraphBLAS::NoAccumulate(),
                       GraphBLAS::Times<double>(), u3, mA,
                       true);

        BOOST_CHECK_EQUAL(result.nvals(), 1);
        BOOST_CHECK_EQUAL(result, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_scmp_masked_replace_reg_stored_zero)
{
    // tests a computed and stored zero
    // tests a stored zero in the mask
    // Build some matrices.
    std::vector<std::vector<int> > A_dense = {{1, 1, 0, 0},
                                              {1, 2, 2, 0},
                                              {0, 2, 3, 3},
                                              {0, 0, 3, 4}};
    std::vector<int> u4_dense =  { 1,-1, 1, 0};
    std::vector<int> scmp_mask_dense =  { 0, 0, 0, 1 };
    std::vector<int> ones_dense =  { 1, 1, 1, 1};
    GraphBLAS::Matrix<int, GraphBLAS::DirectedMatrixTag> A(A_dense, 0);
    GraphBLAS::Vector<int> u(u4_dense, 0);
    GraphBLAS::Vector<int> mask(scmp_mask_dense); //, 0);
    GraphBLAS::Vector<int> result(ones_dense, 0);

    BOOST_CHECK_EQUAL(mask.nvals(), 4);  // [ 0 0 0 1 ]

    int const NIL(666);
    std::vector<int> ans = { 0, 1, 1,  NIL};
    GraphBLAS::Vector<int> answer(ans, NIL);

    GraphBLAS::eWiseMult(result,
                   GraphBLAS::complement(mask),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::Times<double>(), u, A,
                   true);
    BOOST_CHECK_EQUAL(result.nvals(), 3);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_scmp_masked_replace_a_transpose)
{
    // Build some matrices.
    std::vector<std::vector<int> > A_dense = {{1, 1, 0, 0},
                                              {1, 2, 2, 0},
                                              {0, 2, 3, 3},
                                              {0, 0, 3, 4},
                                              {1, 1, 1, 1}};
    std::vector<int> u4_dense =  { 1,-1, 1, 0};
    std::vector<int> mask_dense =  { 0, 1, 0, 1, 0};
    std::vector<int> ones_dense =  { 1, 1, 1, 1, 1};
    GraphBLAS::Matrix<int, GraphBLAS::DirectedMatrixTag> A(A_dense, 0);
    GraphBLAS::Vector<int> u(u4_dense, 0);
    GraphBLAS::Vector<int> mask(mask_dense, 0);
    GraphBLAS::Vector<int> result(ones_dense, 0);

    int const NIL(666);
    std::vector<int> ans = { 0, NIL, 1,  NIL, 1};
    GraphBLAS::Vector<int> answer(ans, NIL);

    GraphBLAS::eWiseMult(result,
                   GraphBLAS::complement(mask),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::Times<double>(), u, transpose(A),
                   true);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
// Tests using a complemented mask (with merge semantics)
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_scmp_masked_bad_dimensions)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(m3x3_dense, 0.);

    GraphBLAS::Vector<double, GraphBLAS::SparseTag> u3(u3_dense, 0.);
    GraphBLAS::Vector<double, GraphBLAS::SparseTag> m4(u4_dense, 0.);
    GraphBLAS::Vector<double, GraphBLAS::SparseTag> w3(v3_ones, 0.);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseMult(w3,
                        GraphBLAS::complement(m4),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Times<double>(), u3, mA)),
        GraphBLAS::DimensionException);
}
//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_scmp_masked_reg)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(m3x3_dense, 0.);

    GraphBLAS::Vector<double, GraphBLAS::SparseTag> u3(u3_dense, 0.);
    GraphBLAS::Vector<double, GraphBLAS::SparseTag> m3(u3_dense, 0.); // [1 1 -]
    GraphBLAS::Vector<double> result(v3_ones, 0.);

    BOOST_CHECK_EQUAL(m3.nvals(), 2);

    std::vector<double> ans = {1, 1, 7}; // if accum is NULL then {1, 0, 7};
    GraphBLAS::Vector<double> answer(ans, 0.);

    GraphBLAS::eWiseMult(result,
                   GraphBLAS::complement(m3),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::Times<double>(), u3, mA);

    BOOST_CHECK_EQUAL(result.nvals(), 3); //2);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_scmp_masked_reg_stored_zero)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(m3x3_dense, 0.);

    GraphBLAS::Vector<double, GraphBLAS::SparseTag> u3(u3_dense, 0.);
    GraphBLAS::Vector<double, GraphBLAS::SparseTag> m3(u3_dense); // [1 1 0]
    GraphBLAS::Vector<double> result(v3_ones, 0.);

    BOOST_CHECK_EQUAL(m3.nvals(), 3);

    std::vector<double> ans = {1, 1, 7}; // if accum is NULL then {1, 0, 7};
    GraphBLAS::Vector<double> answer(ans, 0.);

    GraphBLAS::eWiseMult(result,
                   GraphBLAS::complement(m3),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::Times<double>(), u3, mA);

    BOOST_CHECK_EQUAL(result.nvals(), 3); //2);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_scmp_masked_a_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(m3x3_dense, 0.);

    GraphBLAS::Vector<double, GraphBLAS::SparseTag> u3(u3_dense, 0.); // [1 1 -]
    GraphBLAS::Vector<double, GraphBLAS::SparseTag> m3(u3_dense, 0.); // [1 1 -]
    GraphBLAS::Vector<double> result(v3_ones, 0.);

    BOOST_CHECK_EQUAL(m3.nvals(), 2);

    std::vector<double> ans = {1, 1, 9};
    GraphBLAS::Vector<double> answer(ans, 0.);

    GraphBLAS::eWiseMult(result,
                   GraphBLAS::complement(m3),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::Times<double>(), u3, transpose(mA));

    BOOST_CHECK_EQUAL(result.nvals(), 3);
    BOOST_CHECK_EQUAL(result, answer);
}
#endif

BOOST_AUTO_TEST_SUITE_END()
