/*
 * GraphBLAS Template Library (GBTL), Version 3.0
 *
 * Copyright 2020 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors.
 *
 * THIS MATERIAL WAS PREPARED AS AN ACCOUNT OF WORK SPONSORED BY AN AGENCY OF
 * THE UNITED STATES GOVERNMENT.  NEITHER THE UNITED STATES GOVERNMENT NOR THE
 * UNITED STATES DEPARTMENT OF ENERGY, NOR THE UNITED STATES DEPARTMENT OF
 * DEFENSE, NOR CARNEGIE MELLON UNIVERSITY, NOR BATTELLE, NOR ANY OF THEIR
 * EMPLOYEES, NOR ANY JURISDICTION OR ORGANIZATION THAT HAS COOPERATED IN THE
 * DEVELOPMENT OF THESE MATERIALS, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS,
 * OR USEFULNESS OR ANY INFORMATION, APPARATUS, PRODUCT, SOFTWARE, OR PROCESS
 * DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED
 * RIGHTS.
 *
 * Released under a BSD-style license, please see LICENSE file or contact
 * permission@sei.cmu.edu for full terms.
 *
 * [DISTRIBUTION STATEMENT A] This material has been approved for public release
 * and unlimited distribution.  Please see Copyright notice for non-US
 * Government use and distribution.
 *
 * This Software includes and/or makes use of the following Third-Party Software
 * subject to its own license:
 *
 * 1. Boost Unit Test Framework
 * (https://www.boost.org/doc/libs/1_45_0/libs/test/doc/html/utf.html)
 * Copyright 2001 Boost software license, Gennadiy Rozental.
 *
 * DM20-0442
 */

#define GRAPHBLAS_LOGGING_LEVEL 0

#include <graphblas/graphblas.hpp>

using namespace grb;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE ewiseadd_matrix_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

/// @todo add better tests with accumulate (only a few using Second)

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
    grb::Matrix<double, grb::DirectedMatrixTag> mA(m3x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> mB(m4x3_dense, 0.);

    grb::Matrix<double, grb::DirectedMatrixTag> Result(3, 3);

    // incompatible input dimensions
    BOOST_CHECK_THROW(
        (grb::eWiseAdd(Result,
                             grb::NoMask(),
                             grb::NoAccumulate(),
                             grb::Plus<double>(), mA, mB)),
        grb::DimensionException);

    // incompatible output matrix dimensions
    BOOST_CHECK_THROW(
        (grb::eWiseAdd(Result,
                             grb::NoMask(),
                             grb::NoAccumulate(),
                             grb::Plus<double>(), mB, mB)),
        grb::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_bad_dimensions2)
{
    IndexArrayType i_m1    = {0, 0, 1, 1, 2, 2, 3};
    IndexArrayType j_m1    = {0, 1, 1, 2, 2, 3, 3};
    std::vector<double> v_m1 = {1, 2, 2, 3, 3, 4, 4};
    Matrix<double, DirectedMatrixTag> m1(4, 4);
    m1.build(i_m1, j_m1, v_m1);

    IndexArrayType i_m2    = {0, 0, 1, 1, 1, 2, 2};
    IndexArrayType j_m2    = {0, 1, 0, 1, 2, 1, 2};
    std::vector<double> v_m2 = {2, 2, 1, 4, 4, 4, 6};
    Matrix<double, DirectedMatrixTag> m2(3, 4);
    m2.build(i_m2, j_m2, v_m2);

    Matrix<double, DirectedMatrixTag> m3(4, 4);

    BOOST_CHECK_THROW(
        eWiseAdd(m3, NoMask(), NoAccumulate(),
                 Plus<double>(), m1, m2),
        DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_eWiseadd_matrix_normal)
{
    // Build some sparse matrices.
    IndexArrayType i_mat    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j_mat    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_mat = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4};
    Matrix<double, DirectedMatrixTag> mat(4, 4);
    mat.build(i_mat, j_mat, v_mat);

    Matrix<double, DirectedMatrixTag> m3(4, 4);

    IndexArrayType i_answer    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j_answer    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_answer = {2, 2, 2, 4, 4, 4, 6, 6, 6, 8};
    Matrix<double, DirectedMatrixTag> answer(4, 4);
    answer.build(i_answer, j_answer, v_answer);

    // Now try simple's ewiseapply.
    eWiseAdd(m3, NoMask(), NoAccumulate(), Plus<double>(), mat, mat);

    BOOST_CHECK_EQUAL(m3, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_eWiseadd_matrix_semiring)
{
    // Build some sparse matrices.
    IndexArrayType i_mat    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j_mat    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_mat = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4};
    Matrix<double, DirectedMatrixTag> mat(4, 4);
    mat.build(i_mat, j_mat, v_mat);

    Matrix<double, DirectedMatrixTag> m3(4, 4);

    IndexArrayType i_answer    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j_answer    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_answer = {2, 2, 2, 4, 4, 4, 6, 6, 6, 8};
    Matrix<double, DirectedMatrixTag> answer(4, 4);
    answer.build(i_answer, j_answer, v_answer);

    eWiseAdd(m3, NoMask(), NoAccumulate(),
             add_monoid(ArithmeticSemiring<double>()),
             mat, mat);

    BOOST_CHECK_EQUAL(m3, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_reg)
{
    grb::Matrix<double, grb::DirectedMatrixTag> A(m3x3_dense, 0.);

    // ewise add with dense matrix
    grb::Matrix<double, grb::DirectedMatrixTag> B(twos3x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> Ans(
        ans_twos3x3_dense, 0.);

    grb::Matrix<double, grb::DirectedMatrixTag> Result(3,3);

    grb::eWiseAdd(Result,
                        grb::NoMask(),
                        grb::NoAccumulate(),
                        grb::Plus<double>(), A, B);
    BOOST_CHECK_EQUAL(Result, Ans);

    // ewise add with sparse matrix
    grb::Matrix<double, grb::DirectedMatrixTag> B2(eye3x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> Ans2(
        ans_eye3x3_dense, 0.);
    grb::eWiseAdd(Result,
                        grb::NoMask(),
                        grb::NoAccumulate(),
                        grb::Plus<double>(), A, B2);
    BOOST_CHECK_EQUAL(Result, Ans2);

    // ewise add with empty matrix
    grb::Matrix<double, grb::DirectedMatrixTag> B3(zero3x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> Ans3(
        m3x3_dense, 0.);
    grb::eWiseAdd(Result,
                        grb::NoMask(),
                        grb::NoAccumulate(),
                        grb::Plus<double>(), A, B3);
    BOOST_CHECK_EQUAL(Result, Ans3);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_semiring_matrix_reg)
{
    grb::Matrix<double, grb::DirectedMatrixTag> A(m3x3_dense, 0.);

    // ewise add with dense matrix
    grb::Matrix<double, grb::DirectedMatrixTag> B(twos3x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> Ans(
        ans_twos3x3_dense, 0.);

    grb::Matrix<double, grb::DirectedMatrixTag> Result(3,3);

    grb::eWiseAdd(Result,
                        grb::NoMask(),
                        grb::NoAccumulate(),
                        add_monoid(grb::ArithmeticSemiring<double>()),
                        A, B);
    BOOST_CHECK_EQUAL(Result, Ans);

    // ewise add with sparse matrix
    grb::Matrix<double, grb::DirectedMatrixTag> B2(eye3x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> Ans2(
        ans_eye3x3_dense, 0.);
    grb::eWiseAdd(Result,
                        grb::NoMask(),
                        grb::NoAccumulate(),
                        add_monoid(grb::ArithmeticSemiring<double>()),
                        A, B2);
    BOOST_CHECK_EQUAL(Result, Ans2);

    // ewise add with empty matrix
    grb::Matrix<double, grb::DirectedMatrixTag> B3(zero3x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> Ans3(
        m3x3_dense, 0.);
    grb::eWiseAdd(Result,
                        grb::NoMask(),
                        grb::NoAccumulate(),
                        add_monoid(grb::ArithmeticSemiring<double>()),
                        A, B3);
    BOOST_CHECK_EQUAL(Result, Ans3);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_stored_zero_result)
{
    grb::Matrix<double, grb::DirectedMatrixTag> A(m3x3_dense, 0.);

    // Add a stored zero
    A.setElement(1, 2, 0);
    BOOST_CHECK_EQUAL(A.nvals(), 7);

    grb::Matrix<double, grb::DirectedMatrixTag> B2(eye3x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> Ans2(
        ans_eye3x3_dense, 0.);
    // Add a stored zero
    Ans2.setElement(1, 2, 0);

    grb::Matrix<double, grb::DirectedMatrixTag> Result(3,3);

    grb::eWiseAdd(Result,
                        grb::NoMask(),
                        grb::NoAccumulate(),
                        grb::Plus<double>(), A, B2);
    BOOST_CHECK_EQUAL(Result.nvals(), 8);
    BOOST_CHECK_EQUAL(Result, Ans2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_transpose)
{
    grb::Matrix<double, grb::DirectedMatrixTag> A(m3x4_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> B(twos4x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> Ans(ans_twos4x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> Result(4,3);

    grb::eWiseAdd(Result,
                        grb::NoMask(),
                        grb::NoAccumulate(),
                        grb::Plus<double>(),
                        transpose(A), B);
    BOOST_CHECK_EQUAL(Result, Ans);

    grb::Matrix<double, grb::DirectedMatrixTag> B2(m3x4_dense, 0.);
    BOOST_CHECK_THROW(
        (grb::eWiseAdd(Result,
                             grb::NoMask(),
                             grb::NoAccumulate(),
                             grb::Plus<double>(),
                             transpose(A), A)),
        grb::DimensionException);

    grb::Matrix<double, grb::DirectedMatrixTag> Result2(3,3);
    BOOST_CHECK_THROW(
        (grb::eWiseAdd(Result2,
                             grb::NoMask(),
                             grb::NoAccumulate(),
                             grb::Plus<double>(),
                             transpose(A), B)),
        grb::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_b_transpose)
{
    grb::Matrix<double, grb::DirectedMatrixTag> A(m3x4_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> B(twos4x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> Ans(ans_twos3x4_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> Result(3, 4);


    grb::eWiseAdd(Result,
                        grb::NoMask(),
                        grb::NoAccumulate(),
                        grb::Plus<double>(),
                        A, transpose(B));
    BOOST_CHECK_EQUAL(Result, Ans);

    grb::Matrix<double, grb::DirectedMatrixTag> B2(m3x4_dense, 0.);
    BOOST_CHECK_THROW(
        (grb::eWiseAdd(Result,
                             grb::NoMask(),
                             grb::NoAccumulate(),
                             grb::Plus<double>(),
                             B, transpose(B))),
        grb::DimensionException);

    grb::Matrix<double, grb::DirectedMatrixTag> Result2(3,3);
    BOOST_CHECK_THROW(
        (grb::eWiseAdd(Result2,
                             grb::NoMask(),
                             grb::NoAccumulate(),
                             grb::Plus<double>(),
                             A, transpose(B))),
        grb::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_and_b_transpose)
{
    grb::Matrix<double, grb::DirectedMatrixTag> A(m4x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> B(twos4x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> Ans(ans_twos3x4_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> Result(3, 4);


    grb::eWiseAdd(Result,
                        grb::NoMask(),
                        grb::NoAccumulate(),
                        grb::Plus<double>(),
                        transpose(A), transpose(B));
    BOOST_CHECK_EQUAL(Result, Ans);

    grb::Matrix<double, grb::DirectedMatrixTag> B2(m3x4_dense, 0.);
    BOOST_CHECK_THROW(
        (grb::eWiseAdd(Result,
                             grb::NoMask(),
                             grb::NoAccumulate(),
                             grb::Plus<double>(),
                             transpose(Ans), transpose(B))),
        grb::DimensionException);

    grb::Matrix<double, grb::DirectedMatrixTag> Result2(3,3);
    BOOST_CHECK_THROW(
        (grb::eWiseAdd(Result2,
                             grb::NoMask(),
                             grb::NoAccumulate(),
                             grb::Plus<double>(),
                             transpose(A), transpose(B))),
        grb::DimensionException);
}

//****************************************************************************
// Tests using a mask with REPLACE
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_masked_replace_bad_dimensions)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> mB(m4x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> Mask(twos3x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> Result(4, 3);

    // incompatible Mask-Output dimensions
    BOOST_CHECK_THROW(
        (grb::eWiseAdd(Result,
                             Mask,
                             grb::NoAccumulate(),
                             grb::Plus<double>(), mA, mB,
                             REPLACE)),
        grb::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_masked_replace_reg)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> mB(m4x3_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{1, 1, 1},
                                                    {0, 1, 1},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    grb::Matrix<double, grb::DirectedMatrixTag> Mask(mask_dense, 0.);

    {
        grb::Matrix<double, grb::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{7, 8, 6},
                                                       {0, 9, 7},
                                                       {0, 0, 2},
                                                       {0, 0, 0}};
        grb::Matrix<double, grb::DirectedMatrixTag> Ans(ans_dense, 0.);

        grb::eWiseAdd(Result,
                            Mask,
                            grb::NoAccumulate(),
                            grb::Plus<double>(), mA, mB,
                            REPLACE);

        BOOST_CHECK_EQUAL(Result.nvals(), 6);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

    {
        grb::Matrix<double, grb::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{7, 8,  6},
                                                       {0, 9,  7},
                                                       {0, 0,  2},
                                                       {0, 0,  0}};
        grb::Matrix<double, grb::DirectedMatrixTag> Ans(ans_dense, 0.);

        grb::eWiseAdd(Result,
                            Mask,
                            grb::Second<double>(),
                            grb::Plus<double>(), mA, mB,
                            REPLACE);

        BOOST_CHECK_EQUAL(Result.nvals(), 6);
        BOOST_CHECK_EQUAL(Result, Ans);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_masked_replace_reg_stored_zero)
{
    // tests a computed and stored zero
    // tests a stored zero in the mask
    grb::Matrix<double, grb::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> mB(m4x3_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{1, 1, 1},
                                                    {0, 1, 1},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    grb::Matrix<double, grb::DirectedMatrixTag> Mask(mask_dense); //, 0.);
    BOOST_CHECK_EQUAL(Mask.nvals(), 12);

    {
        grb::Matrix<double, grb::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{7, 8, 6},
                                                       {0, 9, 7},
                                                       {0, 0, 2},
                                                       {0, 0, 0}};
        grb::Matrix<double, grb::DirectedMatrixTag> Ans(ans_dense, 0.);

        grb::eWiseAdd(Result,
                            Mask,
                            grb::NoAccumulate(),
                            grb::Plus<double>(), mA, mB,
                            REPLACE);

        BOOST_CHECK_EQUAL(Result.nvals(), 6);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

    {
        grb::Matrix<double, grb::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{7,  8,  6},
                                                       {0,  9,  7},
                                                       {0,  0,  2},
                                                       {0,  0,  0}};
        grb::Matrix<double, grb::DirectedMatrixTag> Ans(ans_dense, 0.);

        grb::eWiseAdd(Result,
                            Mask,
                            grb::Second<double>(),
                            grb::Plus<double>(), mA, mB,
                            REPLACE);

        BOOST_CHECK_EQUAL(Result.nvals(), 6);
        BOOST_CHECK_EQUAL(Result, Ans);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_masked_replace_a_transpose)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(twos3x4_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> mB(m4x3_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{1, 1, 1},
                                                    {0, 1, 1},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    grb::Matrix<double, grb::DirectedMatrixTag> Mask(mask_dense, 0.);

    grb::Matrix<double, grb::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{7, 8, 6},
                                                   {0, 9, 7},
                                                   {0, 0, 2},
                                                   {0, 0, 0}};
    grb::Matrix<double, grb::DirectedMatrixTag> Ans(ans_dense, 0.);

    grb::eWiseAdd(Result,
                        Mask,
                        grb::NoAccumulate(),
                        grb::Plus<double>(), transpose(mA), mB,
                        REPLACE);

    BOOST_CHECK_EQUAL(Result.nvals(), 6);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_masked_replace_b_transpose)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> mB(m3x4_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{1, 1, 1},
                                                    {0, 1, 1},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    grb::Matrix<double, grb::DirectedMatrixTag> Mask(mask_dense, 0.);

    grb::Matrix<double, grb::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{7, 8, 6},
                                                   {0, 9, 7},
                                                   {0, 0, 2},
                                                   {0, 0, 0}};
    grb::Matrix<double, grb::DirectedMatrixTag> Ans(ans_dense, 0.);

    grb::eWiseAdd(Result,
                        Mask,
                        grb::NoAccumulate(),
                        grb::Plus<double>(), mA, transpose(mB),
                        REPLACE);

    BOOST_CHECK_EQUAL(Result.nvals(), 6);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_masked_replace_a_and_b_transpose)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(twos3x4_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> mB(m3x4_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{1, 1, 1},
                                                    {0, 1, 1},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    grb::Matrix<double, grb::DirectedMatrixTag> Mask(mask_dense, 0.);

    grb::Matrix<double, grb::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{7, 8, 6},
                                                   {0, 9, 7},
                                                   {0, 0, 2},
                                                   {0, 0, 0}};
    grb::Matrix<double, grb::DirectedMatrixTag> Ans(ans_dense, 0.);

    grb::eWiseAdd(Result,
                        Mask,
                        grb::NoAccumulate(),
                        grb::Plus<double>(), transpose(mA), transpose(mB),
                        REPLACE);

    BOOST_CHECK_EQUAL(Result.nvals(), 6);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
// Tests using a mask with MERGE semantics
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_masked_bad_dimensions)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> mB(m4x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> Mask(twos3x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> Result(4, 3);

    // incompatible Mask-Output dimensions
    BOOST_CHECK_THROW(
        (grb::eWiseAdd(Result,
                             Mask,
                             grb::NoAccumulate(),
                             grb::Plus<double>(), mA, mB)),
        grb::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_masked_reg)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> mB(m4x3_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{1, 1, 1},
                                                    {0, 1, 1},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    grb::Matrix<double, grb::DirectedMatrixTag> Mask(mask_dense, 0.);

    {
        grb::Matrix<double, grb::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{7,  8,  6},
                                                       {2,  9,  7},
                                                       {2,  2,  2},
                                                       {2,  2,  2}};
        grb::Matrix<double, grb::DirectedMatrixTag> Ans(ans_dense, 0.);

        grb::eWiseAdd(Result,
                            Mask,
                            grb::NoAccumulate(),
                            grb::Plus<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 12);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

    {
        grb::Matrix<double, grb::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{7,  8,  6},
                                                       {2,  9,  7},
                                                       {2,  2,  2},
                                                       {2,  2,  2}};
        grb::Matrix<double, grb::DirectedMatrixTag> Ans(ans_dense, 0.);

        grb::eWiseAdd(Result,
                            Mask,
                            grb::Second<double>(),
                            grb::Plus<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 12);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_masked_reg_stored_zero)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> mB(m4x3_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{1, 1, 1},
                                                    {0, 1, 1},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    grb::Matrix<double, grb::DirectedMatrixTag> Mask(mask_dense); //, 0.);

    {
        grb::Matrix<double, grb::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{7,  8,  6},
                                                       {2,  9,  7},
                                                       {2,  2,  2},
                                                       {2,  2,  2}};
        grb::Matrix<double, grb::DirectedMatrixTag> Ans(ans_dense, 0.);

        grb::eWiseAdd(Result,
                            Mask,
                            grb::NoAccumulate(),
                            grb::Plus<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 12);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

    {
        grb::Matrix<double, grb::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{7,  8,  6},
                                                       {2,  9,  7},
                                                       {2,  2,  2},
                                                       {2,  2,  2}};
        grb::Matrix<double, grb::DirectedMatrixTag> Ans(ans_dense, 0.);

        grb::eWiseAdd(Result,
                            Mask,
                            grb::Second<double>(),
                            grb::Plus<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 12);
        BOOST_CHECK_EQUAL(Result, Ans);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_masked_a_transpose)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(twos3x4_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> mB(m4x3_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{1, 1, 1},
                                                    {0, 1, 1},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    grb::Matrix<double, grb::DirectedMatrixTag> Mask(mask_dense, 0.);

    grb::Matrix<double, grb::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{7,  8,  6},
                                                   {2,  9,  7},
                                                   {2,  2,  2},
                                                   {2,  2,  2}};
    grb::Matrix<double, grb::DirectedMatrixTag> Ans(ans_dense, 0.);

    grb::eWiseAdd(Result,
                        Mask,
                        grb::NoAccumulate(),
                        grb::Plus<double>(), transpose(mA), mB);

    BOOST_CHECK_EQUAL(Result.nvals(), 12);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_masked_b_transpose)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> mB(m3x4_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{1, 1, 1},
                                                    {0, 1, 1},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    grb::Matrix<double, grb::DirectedMatrixTag> Mask(mask_dense, 0.);

    grb::Matrix<double, grb::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{7,  8,  6},
                                                   {2,  9,  7},
                                                   {2,  2,  2},
                                                   {2,  2,  2}};
    grb::Matrix<double, grb::DirectedMatrixTag> Ans(ans_dense, 0.);

    grb::eWiseAdd(Result,
                        Mask,
                        grb::NoAccumulate(),
                        grb::Plus<double>(), mA, transpose(mB));

    BOOST_CHECK_EQUAL(Result.nvals(), 12);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_masked_a_and_b_transpose)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(twos3x4_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> mB(m3x4_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{1, 1, 1},
                                                    {0, 1, 1},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    grb::Matrix<double, grb::DirectedMatrixTag> Mask(mask_dense, 0.);

    grb::Matrix<double, grb::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{7,  8,  6},
                                                   {2,  9,  7},
                                                   {2,  2,  2},
                                                   {2,  2,  2}};
    grb::Matrix<double, grb::DirectedMatrixTag> Ans(ans_dense, 0.);

    grb::eWiseAdd(Result,
                        Mask,
                        grb::NoAccumulate(),
                        grb::Plus<double>(), transpose(mA), transpose(mB));

    BOOST_CHECK_EQUAL(Result.nvals(), 12);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
// Tests using a complemented mask with REPLACE semantics
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_scmp_masked_replace_bad_dimensions)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> mB(m4x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> Mask(twos3x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> Result(4, 3);

    // incompatible Mask-Output dimensions
    BOOST_CHECK_THROW(
        (grb::eWiseAdd(Result,
                             grb::complement(Mask),
                             grb::NoAccumulate(),
                             grb::Plus<double>(), mA, mB,
                             REPLACE)),
        grb::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_scmp_masked_replace_reg)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> mB(m4x3_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    grb::Matrix<double, grb::DirectedMatrixTag> Mask(mask_dense, 0.);

    {
        grb::Matrix<double, grb::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{7, 8, 6},
                                                       {0, 9, 7},
                                                       {0, 0, 2},
                                                       {0, 0, 0}};
        grb::Matrix<double, grb::DirectedMatrixTag> Ans(ans_dense, 0.);

        grb::eWiseAdd(Result,
                            grb::complement(Mask),
                            grb::NoAccumulate(),
                            grb::Plus<double>(), mA, mB,
                            REPLACE);

        BOOST_CHECK_EQUAL(Result.nvals(), 6);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

    {
        grb::Matrix<double, grb::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{7, 8, 6},
                                                       {0, 9, 7},
                                                       {0, 0, 2},
                                                       {0, 0, 0}};
        grb::Matrix<double, grb::DirectedMatrixTag> Ans(ans_dense, 0.);

        grb::eWiseAdd(Result,
                            grb::complement(Mask),
                            grb::Second<double>(),
                            grb::Plus<double>(), mA, mB,
                            REPLACE);

        BOOST_CHECK_EQUAL(Result.nvals(), 6);
        BOOST_CHECK_EQUAL(Result, Ans);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_scmp_masked_replace_reg_stored_zero)
{
    // tests a computed and stored zero
    // tests a stored zero in the mask
    grb::Matrix<double, grb::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> mB(m4x3_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    grb::Matrix<double, grb::DirectedMatrixTag> Mask(mask_dense); //, 0.);
    BOOST_CHECK_EQUAL(Mask.nvals(), 12);

    {
        grb::Matrix<double, grb::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{7, 8, 6},
                                                       {0, 9, 7},
                                                       {0, 0, 2},
                                                       {0, 0, 0}};
        grb::Matrix<double, grb::DirectedMatrixTag> Ans(ans_dense, 0.);

        grb::eWiseAdd(Result,
                            grb::complement(Mask),
                            grb::NoAccumulate(),
                            grb::Plus<double>(), mA, mB,
                            REPLACE);

        BOOST_CHECK_EQUAL(Result.nvals(), 6);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

    {
        grb::Matrix<double, grb::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{7, 8, 6},
                                                       {0, 9, 7},
                                                       {0, 0, 2},
                                                       {0, 0, 0}};
        grb::Matrix<double, grb::DirectedMatrixTag> Ans(ans_dense, 0.);

        grb::eWiseAdd(Result,
                            grb::complement(Mask),
                            grb::Second<double>(),
                            grb::Plus<double>(), mA, mB,
                            REPLACE);

        BOOST_CHECK_EQUAL(Result.nvals(), 6);
        BOOST_CHECK_EQUAL(Result, Ans);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_scmp_masked_replace_a_transpose)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(twos3x4_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> mB(m4x3_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    grb::Matrix<double, grb::DirectedMatrixTag> Mask(mask_dense, 0.);

    grb::Matrix<double, grb::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{7, 8, 6},
                                                   {0, 9, 7},
                                                   {0, 0, 2},
                                                   {0, 0, 0}};
    grb::Matrix<double, grb::DirectedMatrixTag> Ans(ans_dense, 0.);

    grb::eWiseAdd(Result,
                        grb::complement(Mask),
                        grb::NoAccumulate(),
                        grb::Plus<double>(), transpose(mA), mB,
                        REPLACE);

    BOOST_CHECK_EQUAL(Result.nvals(), 6);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_scmp_masked_replace_b_transpose)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> mB(m3x4_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    grb::Matrix<double, grb::DirectedMatrixTag> Mask(mask_dense, 0.);

    grb::Matrix<double, grb::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{7, 8, 6},
                                                   {0, 9, 7},
                                                   {0, 0, 2},
                                                   {0, 0, 0}};
    grb::Matrix<double, grb::DirectedMatrixTag> Ans(ans_dense, 0.);

    grb::eWiseAdd(Result,
                        grb::complement(Mask),
                        grb::NoAccumulate(),
                        grb::Plus<double>(), mA, transpose(mB),
                        REPLACE);

    BOOST_CHECK_EQUAL(Result.nvals(), 6);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_scmp_masked_replace_a_and_b_transpose)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(twos3x4_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> mB(m3x4_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    grb::Matrix<double, grb::DirectedMatrixTag> Mask(mask_dense, 0.);

    grb::Matrix<double, grb::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{7, 8, 6},
                                                   {0, 9, 7},
                                                   {0, 0, 2},
                                                   {0, 0, 0}};
    grb::Matrix<double, grb::DirectedMatrixTag> Ans(ans_dense, 0.);

    grb::eWiseAdd(Result,
                        grb::complement(Mask),
                        grb::NoAccumulate(),
                        grb::Plus<double>(), transpose(mA), transpose(mB),
                        REPLACE);

    BOOST_CHECK_EQUAL(Result.nvals(), 6);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
// Tests using a complemented mask (with merge semantics)
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_scmp_masked_bad_dimensions)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> mB(m4x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> Mask(twos3x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> Result(4, 3);

    // incompatible Mask-Output dimensions
    BOOST_CHECK_THROW(
        (grb::eWiseAdd(Result,
                             grb::complement(Mask),
                             grb::NoAccumulate(),
                             grb::Plus<double>(), mA, mB)),
        grb::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_scmp_masked_reg)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> mB(m4x3_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    grb::Matrix<double, grb::DirectedMatrixTag> Mask(mask_dense, 0.);

    {
        grb::Matrix<double, grb::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{7,  8,  6},
                                                       {2,  9,  7},
                                                       {2,  2,  2},
                                                       {2,  2,  2}};
        grb::Matrix<double, grb::DirectedMatrixTag> Ans(ans_dense, 0.);

        grb::eWiseAdd(Result,
                            grb::complement(Mask),
                            grb::NoAccumulate(),
                            grb::Plus<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 12);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

    {
        grb::Matrix<double, grb::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{7,  8,  6},
                                                       {2,  9,  7},
                                                       {2,  2,  2},
                                                       {2,  2,  2}};
        grb::Matrix<double, grb::DirectedMatrixTag> Ans(ans_dense, 0.);

        grb::eWiseAdd(Result,
                            grb::complement(Mask),
                            grb::Second<double>(),
                            grb::Plus<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 12);
        BOOST_CHECK_EQUAL(Result, Ans);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_scmp_masked_reg_stored_zero)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> mB(m4x3_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    grb::Matrix<double, grb::DirectedMatrixTag> Mask(mask_dense);//, 0.);

    {
        grb::Matrix<double, grb::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{7,  8,  6},
                                                       {2,  9,  7},
                                                       {2,  2,  2},
                                                       {2,  2,  2}};
        grb::Matrix<double, grb::DirectedMatrixTag> Ans(ans_dense, 0.);

        grb::eWiseAdd(Result,
                            grb::complement(Mask),
                            grb::NoAccumulate(),
                            grb::Plus<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 12);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

    {
        grb::Matrix<double, grb::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{7,  8,  6},
                                                       {2,  9,  7},
                                                       {2,  2,  2},
                                                       {2,  2,  2}};
        grb::Matrix<double, grb::DirectedMatrixTag> Ans(ans_dense, 0.);

        grb::eWiseAdd(Result,
                      grb::complement(Mask),
                      grb::Second<double>(),
                      grb::Plus<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 12);
        BOOST_CHECK_EQUAL(Result, Ans);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_scmp_masked_a_transpose)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(twos3x4_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> mB(m4x3_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    grb::Matrix<double, grb::DirectedMatrixTag> Mask(mask_dense, 0.);

    grb::Matrix<double, grb::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{7,  8,  6},
                                                   {2,  9,  7},
                                                   {2,  2,  2},
                                                   {2,  2,  2}};
    grb::Matrix<double, grb::DirectedMatrixTag> Ans(ans_dense, 0.);

    grb::eWiseAdd(Result,
                  grb::complement(Mask),
                  grb::NoAccumulate(),
                  grb::Plus<double>(), transpose(mA), mB);

    BOOST_CHECK_EQUAL(Result.nvals(), 12);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_scmp_masked_b_transpose)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> mB(m3x4_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    grb::Matrix<double, grb::DirectedMatrixTag> Mask(mask_dense, 0.);

    grb::Matrix<double, grb::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{7,  8,  6},
                                                   {2,  9,  7},
                                                   {2,  2,  2},
                                                   {2,  2,  2}};
    grb::Matrix<double, grb::DirectedMatrixTag> Ans(ans_dense, 0.);

    grb::eWiseAdd(Result,
                        grb::complement(Mask),
                        grb::NoAccumulate(),
                        grb::Plus<double>(), mA, transpose(mB));

    BOOST_CHECK_EQUAL(Result.nvals(), 12);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_matrix_scmp_masked_a_and_b_transpose)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(twos3x4_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> mB(m3x4_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    grb::Matrix<double, grb::DirectedMatrixTag> Mask(mask_dense, 0.);

    grb::Matrix<double, grb::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{7,  8,  6},
                                                   {2,  9,  7},
                                                   {2,  2,  2},
                                                   {2,  2,  2}};
    grb::Matrix<double, grb::DirectedMatrixTag> Ans(ans_dense, 0.);

    grb::eWiseAdd(Result,
                  grb::complement(Mask),
                  grb::NoAccumulate(),
                  grb::Plus<double>(), transpose(mA), transpose(mB));

    BOOST_CHECK_EQUAL(Result.nvals(), 12);
    BOOST_CHECK_EQUAL(Result, Ans);
}


BOOST_AUTO_TEST_SUITE_END()
