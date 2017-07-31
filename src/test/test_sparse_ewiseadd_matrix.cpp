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
#define BOOST_TEST_MODULE sparse_ewiseadd_matrix_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(sparse_ewiseadd_matrix_suite)

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

    std::vector<std::vector<double> > ans_twos3x3_dense = {{14,  2,  9},
                                                           { 3,  2,  2},
                                                           { 5,  8, 11}};

    std::vector<std::vector<double> > ans_eye3x3_dense = {{13,  0,  7},
                                                          { 1,  1,  0},
                                                          { 3,  6, 10}};

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

    std::vector<std::vector<double> > ans_twos4x3_dense = {{7, 8,  6},
                                                           {2, 9,  7},
                                                           {3, 2,  2},
                                                           {4, 2,  3}};
    std::vector<std::vector<double> > ans_twos3x4_dense = {{7, 2, 3, 4},
                                                           {8, 9, 2, 2},
                                                           {6, 7, 2, 3}};
}

//****************************************************************************
// Tests without mask
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_bad_dimensions)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(m3x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m4x3_dense, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(3, 3);

    // incompatible input dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseAdd(Result,
                             GraphBLAS::NoMask(),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Plus<double>(), mA, mB)),
        GraphBLAS::DimensionException);

    // incompatible output matrix dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseAdd(Result,
                             GraphBLAS::NoMask(),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Plus<double>(), mB, mB)),
        GraphBLAS::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_reg)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> A(m3x3_dense, 0.);

    // ewise add with dense matrix
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> B(twos3x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(
        ans_twos3x3_dense, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(3,3);

    GraphBLAS::eWiseAdd(Result,
                        GraphBLAS::NoMask(),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), A, B);
    BOOST_CHECK_EQUAL(Result, Ans);

    // ewise add with sparse matrix
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> B2(eye3x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans2(
        ans_eye3x3_dense, 0.);
    GraphBLAS::eWiseAdd(Result,
                        GraphBLAS::NoMask(),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), A, B2);
    BOOST_CHECK_EQUAL(Result, Ans2);

    // ewise add with empty matrix
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> B3(zero3x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans3(
        m3x3_dense, 0.);
    GraphBLAS::eWiseAdd(Result,
                        GraphBLAS::NoMask(),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), A, B3);
    BOOST_CHECK_EQUAL(Result, Ans3);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_stored_zero_result)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> A(m3x3_dense, 0.);

    // Add a stored zero
    A.setElement(1, 2, 0);
    BOOST_CHECK_EQUAL(A.nvals(), 7);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> B2(eye3x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans2(
        ans_eye3x3_dense, 0.);
    // Add a stored zero
    Ans2.setElement(1, 2, 0);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(3,3);

    GraphBLAS::eWiseAdd(Result,
                        GraphBLAS::NoMask(),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), A, B2);
    BOOST_CHECK_EQUAL(Result.nvals(), 8);
    BOOST_CHECK_EQUAL(Result, Ans2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> A(m3x4_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> B(twos4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_twos4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(4,3);

    GraphBLAS::eWiseAdd(Result,
                        GraphBLAS::NoMask(),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(),
                        transpose(A), B);
    BOOST_CHECK_EQUAL(Result, Ans);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> B2(m3x4_dense, 0.);
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseAdd(Result,
                             GraphBLAS::NoMask(),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Plus<double>(),
                             transpose(A), A)),
        GraphBLAS::DimensionException);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result2(3,3);
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseAdd(Result2,
                             GraphBLAS::NoMask(),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Plus<double>(),
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


    GraphBLAS::eWiseAdd(Result,
                        GraphBLAS::NoMask(),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(),
                        A, transpose(B));
    BOOST_CHECK_EQUAL(Result, Ans);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> B2(m3x4_dense, 0.);
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseAdd(Result,
                             GraphBLAS::NoMask(),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Plus<double>(),
                             B, transpose(B))),
        GraphBLAS::DimensionException);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result2(3,3);
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseAdd(Result2,
                             GraphBLAS::NoMask(),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Plus<double>(),
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


    GraphBLAS::eWiseAdd(Result,
                        GraphBLAS::NoMask(),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(),
                        transpose(A), transpose(B));
    BOOST_CHECK_EQUAL(Result, Ans);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> B2(m3x4_dense, 0.);
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseAdd(Result,
                             GraphBLAS::NoMask(),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Plus<double>(),
                             transpose(Ans), transpose(B))),
        GraphBLAS::DimensionException);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result2(3,3);
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseAdd(Result2,
                             GraphBLAS::NoMask(),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Plus<double>(),
                             transpose(A), transpose(B))),
        GraphBLAS::DimensionException);
}

//****************************************************************************
// Tests using a mask with REPLACE
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_masked_replace_bad_dimensions)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(twos3x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(4, 3);

    // incompatible Mask-Output dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseAdd(Result,
                             Mask,
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Plus<double>(), mA, mB,
                             true)),
        GraphBLAS::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_masked_replace_reg)
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

        std::vector<std::vector<double> > ans_dense = {{7, 8, 6},
                                                       {0, 9, 7},
                                                       {0, 0, 2},
                                                       {0, 0, 0}};
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            Mask,
                            GraphBLAS::NoAccumulate(),
                            GraphBLAS::Plus<double>(), mA, mB,
                            true);

        BOOST_CHECK_EQUAL(Result.nvals(), 6);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

    {
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{7, 8,  6},
                                                       {0, 9,  7},
                                                       {0, 0,  2},
                                                       {0, 0,  0}};
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            Mask,
                            GraphBLAS::Second<double>(),
                            GraphBLAS::Plus<double>(), mA, mB,
                            true);

        BOOST_CHECK_EQUAL(Result.nvals(), 6);
        BOOST_CHECK_EQUAL(Result, Ans);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_masked_replace_reg_stored_zero)
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

        std::vector<std::vector<double> > ans_dense = {{7, 8, 6},
                                                       {0, 9, 7},
                                                       {0, 0, 2},
                                                       {0, 0, 0}};
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            Mask,
                            GraphBLAS::NoAccumulate(),
                            GraphBLAS::Plus<double>(), mA, mB,
                            true);

        BOOST_CHECK_EQUAL(Result.nvals(), 6);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

    {
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{7,  8,  6},
                                                       {0,  9,  7},
                                                       {0,  0,  2},
                                                       {0,  0,  0}};
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            Mask,
                            GraphBLAS::Second<double>(),
                            GraphBLAS::Plus<double>(), mA, mB,
                            true);

        BOOST_CHECK_EQUAL(Result.nvals(), 6);
        BOOST_CHECK_EQUAL(Result, Ans);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_masked_replace_a_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos3x4_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m4x3_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{1, 1, 1},
                                                    {0, 1, 1},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(mask_dense, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{7, 8, 6},
                                                   {0, 9, 7},
                                                   {0, 0, 2},
                                                   {0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

    GraphBLAS::eWiseAdd(Result,
                        Mask,
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), transpose(mA), mB,
                        true);

    BOOST_CHECK_EQUAL(Result.nvals(), 6);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_masked_replace_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m3x4_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{1, 1, 1},
                                                    {0, 1, 1},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(mask_dense, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{7, 8, 6},
                                                   {0, 9, 7},
                                                   {0, 0, 2},
                                                   {0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

    GraphBLAS::eWiseAdd(Result,
                        Mask,
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), mA, transpose(mB),
                        true);

    BOOST_CHECK_EQUAL(Result.nvals(), 6);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_masked_replace_a_and_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos3x4_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m3x4_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{1, 1, 1},
                                                    {0, 1, 1},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(mask_dense, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{7, 8, 6},
                                                   {0, 9, 7},
                                                   {0, 0, 2},
                                                   {0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

    GraphBLAS::eWiseAdd(Result,
                        Mask,
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), transpose(mA), transpose(mB),
                        true);

    BOOST_CHECK_EQUAL(Result.nvals(), 6);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
// Tests using a mask with MERGE semantics
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_masked_bad_dimensions)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(twos3x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(4, 3);

    // incompatible Mask-Output dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseAdd(Result,
                             Mask,
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Plus<double>(), mA, mB)),
        GraphBLAS::DimensionException)
        }

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_masked_reg)
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

        std::vector<std::vector<double> > ans_dense = {{7,  8,  6},
                                                       {2,  9,  7},
                                                       {2,  2,  2},
                                                       {2,  2,  2}};
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            Mask,
                            GraphBLAS::NoAccumulate(),
                            GraphBLAS::Plus<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 12);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

    {
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{7,  8,  6},
                                                       {2,  9,  7},
                                                       {2,  2,  2},
                                                       {2,  2,  2}};
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            Mask,
                            GraphBLAS::Second<double>(),
                            GraphBLAS::Plus<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 12);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_masked_reg_stored_zero)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m4x3_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{1, 1, 1},
                                                    {0, 1, 1},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(mask_dense); //, 0.);

    {
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{7,  8,  6},
                                                       {2,  9,  7},
                                                       {2,  2,  2},
                                                       {2,  2,  2}};
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            Mask,
                            GraphBLAS::NoAccumulate(),
                            GraphBLAS::Plus<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 12);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

    {
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{7,  8,  6},
                                                       {2,  9,  7},
                                                       {2,  2,  2},
                                                       {2,  2,  2}};
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            Mask,
                            GraphBLAS::Second<double>(),
                            GraphBLAS::Plus<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 12);
        BOOST_CHECK_EQUAL(Result, Ans);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_masked_a_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos3x4_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m4x3_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{1, 1, 1},
                                                    {0, 1, 1},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(mask_dense, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{7,  8,  6},
                                                   {2,  9,  7},
                                                   {2,  2,  2},
                                                   {2,  2,  2}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

    GraphBLAS::eWiseAdd(Result,
                        Mask,
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), transpose(mA), mB);

    BOOST_CHECK_EQUAL(Result.nvals(), 12);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_masked_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m3x4_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{1, 1, 1},
                                                    {0, 1, 1},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(mask_dense, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{7,  8,  6},
                                                   {2,  9,  7},
                                                   {2,  2,  2},
                                                   {2,  2,  2}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

    GraphBLAS::eWiseAdd(Result,
                        Mask,
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), mA, transpose(mB));

    BOOST_CHECK_EQUAL(Result.nvals(), 12);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_masked_a_and_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos3x4_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m3x4_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{1, 1, 1},
                                                    {0, 1, 1},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(mask_dense, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{7,  8,  6},
                                                   {2,  9,  7},
                                                   {2,  2,  2},
                                                   {2,  2,  2}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

    GraphBLAS::eWiseAdd(Result,
                        Mask,
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), transpose(mA), transpose(mB));

    BOOST_CHECK_EQUAL(Result.nvals(), 12);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
// Tests using a complemented mask with REPLACE semantics
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_scmp_masked_replace_bad_dimensions)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(twos3x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(4, 3);

    // incompatible Mask-Output dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseAdd(Result,
                             GraphBLAS::complement(Mask),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Plus<double>(), mA, mB,
                             true)),
        GraphBLAS::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_scmp_masked_replace_reg)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m4x3_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(mask_dense, 0.);

    {
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{7, 8, 6},
                                                       {0, 9, 7},
                                                       {0, 0, 2},
                                                       {0, 0, 0}};
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            GraphBLAS::complement(Mask),
                            GraphBLAS::NoAccumulate(),
                            GraphBLAS::Plus<double>(), mA, mB,
                            true);

        BOOST_CHECK_EQUAL(Result.nvals(), 6);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

    {
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{7, 8, 6},
                                                       {0, 9, 7},
                                                       {0, 0, 2},
                                                       {0, 0, 0}};
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            GraphBLAS::complement(Mask),
                            GraphBLAS::Second<double>(),
                            GraphBLAS::Plus<double>(), mA, mB,
                            true);

        BOOST_CHECK_EQUAL(Result.nvals(), 6);
        BOOST_CHECK_EQUAL(Result, Ans);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_scmp_masked_replace_reg_stored_zero)
{
    // tests a computed and stored zero
    // tests a stored zero in the mask
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m4x3_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(mask_dense); //, 0.);
    BOOST_CHECK_EQUAL(Mask.nvals(), 12);

    {
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{7, 8, 6},
                                                       {0, 9, 7},
                                                       {0, 0, 2},
                                                       {0, 0, 0}};
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            GraphBLAS::complement(Mask),
                            GraphBLAS::NoAccumulate(),
                            GraphBLAS::Plus<double>(), mA, mB,
                            true);

        BOOST_CHECK_EQUAL(Result.nvals(), 6);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

    {
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{7, 8, 6},
                                                       {0, 9, 7},
                                                       {0, 0, 2},
                                                       {0, 0, 0}};
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            GraphBLAS::complement(Mask),
                            GraphBLAS::Second<double>(),
                            GraphBLAS::Plus<double>(), mA, mB,
                            true);

        BOOST_CHECK_EQUAL(Result.nvals(), 6);
        BOOST_CHECK_EQUAL(Result, Ans);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_scmp_masked_replace_a_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos3x4_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m4x3_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(mask_dense, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{7, 8, 6},
                                                   {0, 9, 7},
                                                   {0, 0, 2},
                                                   {0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

    GraphBLAS::eWiseAdd(Result,
                        GraphBLAS::complement(Mask),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), transpose(mA), mB,
                        true);

    BOOST_CHECK_EQUAL(Result.nvals(), 6);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_scmp_masked_replace_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m3x4_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(mask_dense, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{7, 8, 6},
                                                   {0, 9, 7},
                                                   {0, 0, 2},
                                                   {0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

    GraphBLAS::eWiseAdd(Result,
                        GraphBLAS::complement(Mask),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), mA, transpose(mB),
                        true);

    BOOST_CHECK_EQUAL(Result.nvals(), 6);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_scmp_masked_replace_a_and_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos3x4_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m3x4_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(mask_dense, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{7, 8, 6},
                                                   {0, 9, 7},
                                                   {0, 0, 2},
                                                   {0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

    GraphBLAS::eWiseAdd(Result,
                        GraphBLAS::complement(Mask),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), transpose(mA), transpose(mB),
                        true);

    BOOST_CHECK_EQUAL(Result.nvals(), 6);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
// Tests using a complemented mask (with merge semantics)
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_scmp_masked_bad_dimensions)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(twos3x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(4, 3);

    // incompatible Mask-Output dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseAdd(Result,
                             GraphBLAS::complement(Mask),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Plus<double>(), mA, mB)),
        GraphBLAS::DimensionException)
        }

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_scmp_masked_reg)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m4x3_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(mask_dense, 0.);

    {
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{7,  8,  6},
                                                       {2,  9,  7},
                                                       {2,  2,  2},
                                                       {2,  2,  2}};
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            GraphBLAS::complement(Mask),
                            GraphBLAS::NoAccumulate(),
                            GraphBLAS::Plus<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 12);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

    {
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{7,  8,  6},
                                                       {2,  9,  7},
                                                       {2,  2,  2},
                                                       {2,  2,  2}};
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            GraphBLAS::complement(Mask),
                            GraphBLAS::Second<double>(),
                            GraphBLAS::Plus<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 12);
        BOOST_CHECK_EQUAL(Result, Ans);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_scmp_masked_reg_stored_zero)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m4x3_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(mask_dense);//, 0.);

    {
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{7,  8,  6},
                                                       {2,  9,  7},
                                                       {2,  2,  2},
                                                       {2,  2,  2}};
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            GraphBLAS::complement(Mask),
                            GraphBLAS::NoAccumulate(),
                            GraphBLAS::Plus<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 12);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

    {
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{7,  8,  6},
                                                       {2,  9,  7},
                                                       {2,  2,  2},
                                                       {2,  2,  2}};
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            GraphBLAS::complement(Mask),
                            GraphBLAS::Second<double>(),
                            GraphBLAS::Plus<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 12);
        BOOST_CHECK_EQUAL(Result, Ans);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_scmp_masked_a_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos3x4_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m4x3_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(mask_dense, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{7,  8,  6},
                                                   {2,  9,  7},
                                                   {2,  2,  2},
                                                   {2,  2,  2}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

    GraphBLAS::eWiseAdd(Result,
                        GraphBLAS::complement(Mask),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), transpose(mA), mB);

    BOOST_CHECK_EQUAL(Result.nvals(), 12);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_scmp_masked_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m3x4_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(mask_dense, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{7,  8,  6},
                                                   {2,  9,  7},
                                                   {2,  2,  2},
                                                   {2,  2,  2}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

    GraphBLAS::eWiseAdd(Result,
                        GraphBLAS::complement(Mask),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), mA, transpose(mB));

    BOOST_CHECK_EQUAL(Result.nvals(), 12);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_scmp_masked_a_and_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos3x4_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m3x4_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(mask_dense, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{7,  8,  6},
                                                   {2,  9,  7},
                                                   {2,  2,  2},
                                                   {2,  2,  2}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

    GraphBLAS::eWiseAdd(Result,
                        GraphBLAS::complement(Mask),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), transpose(mA), transpose(mB));

    BOOST_CHECK_EQUAL(Result.nvals(), 12);
    BOOST_CHECK_EQUAL(Result, Ans);
}


BOOST_AUTO_TEST_SUITE_END()
