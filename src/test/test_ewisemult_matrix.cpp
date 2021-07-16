/*
 * GraphBLAS Template Library, Version 2.0
 *
 * Copyright 2018 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors. All Rights Reserved.
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
 * RIGHTS..
 *
 * Released under a BSD (SEI)-style license, please see license.txt or contact
 * permission@sei.cmu.edu for full terms.
 *
 * This release is an update of:
 *
 * 1. GraphBLAS Template Library (GBTL)
 * (https://github.com/cmu-sei/gbtl/blob/1.0.0/LICENSE) Copyright 2015 Carnegie
 * Mellon University and The Trustees of Indiana. DM17-0037, DM-0002659
 *
 * DM18-0559
 */

#define GRAPHBLAS_LOGGING_LEVEL 0

#include <graphblas/graphblas.hpp>

using namespace GraphBLAS;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE ewisemult_matrix_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

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

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_bad_dimensions)
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
        eWiseMult(m3, NoMask(), NoAccumulate(),
                  Times<double>(), m1, m2),
        DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_bad_dimensions2)
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
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_overwrite_replace)
{
    using T = int32_t;

    std::vector<std::vector<T> > tmp1_dense = {{0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0}};
    std::vector<std::vector<T> >  a10_dense = {{0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1}};

    GraphBLAS::Matrix<T> tmp1(tmp1_dense, 0);
    GraphBLAS::Matrix<T> a10(a10_dense, 0);

    GraphBLAS::eWiseMult(tmp1, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<T>(),
                         tmp1, a10, true);
    BOOST_CHECK_EQUAL(0, tmp1.nvals());

    //GraphBLAS::print_matrix(std::cerr, tmp1, "tmp1");

    T delta(0);
    GraphBLAS::reduce(delta, GraphBLAS::Plus<T>(),
                      GraphBLAS::PlusMonoid<T>(), tmp1);

    BOOST_CHECK_EQUAL(delta, 0);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_semiring_matrix_overwrite_replace)
{
    using T = int32_t;

    std::vector<std::vector<T> > tmp1_dense = {{0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0}};
    std::vector<std::vector<T> >  a10_dense = {{0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1}};

    GraphBLAS::Matrix<T> tmp1(tmp1_dense, 0);
    GraphBLAS::Matrix<T> a10(a10_dense, 0);

    GraphBLAS::eWiseMult(tmp1, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                         multiply_op(GraphBLAS::ArithmeticSemiring<T>()),
                         tmp1, a10, true);
    BOOST_CHECK_EQUAL(0, tmp1.nvals());

    //GraphBLAS::print_matrix(std::cerr, tmp1, "tmp1");

    T delta(0);
    GraphBLAS::reduce(delta,
                      add_monoid(GraphBLAS::ArithmeticSemiring<T>()),
                      add_monoid(GraphBLAS::ArithmeticSemiring<T>()),
                      tmp1);

    BOOST_CHECK_EQUAL(delta, 0);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_normal)
{
    IndexArrayType i_mat    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j_mat    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_mat = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4};
    Matrix<double, DirectedMatrixTag> mat(4, 4);
    mat.build(i_mat, j_mat, v_mat);

    Matrix<double, DirectedMatrixTag> m3(4, 4);

    IndexArrayType i_answer    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j_answer    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_answer = {1, 1, 1, 4, 4, 4, 9, 9, 9, 16};
    Matrix<double, DirectedMatrixTag> answer(4, 4);
    answer.build(i_answer, j_answer, v_answer);

    eWiseMult(m3, NoMask(), NoAccumulate(),
              Times<double>(),
              mat, mat);

    BOOST_CHECK_EQUAL(m3, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_normal_semiring)
{
    IndexArrayType i_mat    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j_mat    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_mat = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4};
    Matrix<double, DirectedMatrixTag> mat(4, 4);
    mat.build(i_mat, j_mat, v_mat);

    Matrix<double, DirectedMatrixTag> m3(4, 4);

    IndexArrayType i_answer    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j_answer    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_answer = {1, 1, 1, 4, 4, 4, 9, 9, 9, 16};
    Matrix<double, DirectedMatrixTag> answer(4, 4);
    answer.build(i_answer, j_answer, v_answer);

    eWiseMult(m3, NoMask(), NoAccumulate(),
              multiply_op(ArithmeticSemiring<double>()),
              mat, mat);

    BOOST_CHECK_EQUAL(m3, answer);
}

//****************************************************************************
// Tests without mask
//****************************************************************************

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
        GraphBLAS::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_masked_reg)
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
                                                       {2,  14, 10},
                                                       {2,   2,  0},
                                                       {2,   2,  2}};
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(Result,
                             Mask,
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Times<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 11);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

    {
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{10, 12,  8},
                                                       {2,  14, 10},
                                                       {2,   2,  2},
                                                       {2,   2,  2}};
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(Result,
                             Mask,
                             GraphBLAS::Second<double>(),
                             GraphBLAS::Times<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 12);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_masked_reg_stored_zero)
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

        std::vector<std::vector<double> > ans_dense = {{10, 12,  8},
                                                       {2,  14, 10},
                                                       {2,   2,  0},
                                                       {2,   2,  2}};
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(Result,
                             Mask,
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Times<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 11);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

    {
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{10, 12,  8},
                                                       {2,  14, 10},
                                                       {2,   2,  2},
                                                       {2,   2,  2}};
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(Result,
                             Mask,
                             GraphBLAS::Second<double>(),
                             GraphBLAS::Times<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 12);
        BOOST_CHECK_EQUAL(Result, Ans);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_masked_a_transpose)
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
                                                   {2,  14, 10},
                                                   {2,   2,  0},
                                                   {2,   2,  2}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

    GraphBLAS::eWiseMult(Result,
                         Mask,
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), transpose(mA), mB);

    BOOST_CHECK_EQUAL(Result.nvals(), 11);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_masked_b_transpose)
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
                                                   {2,  14, 10},
                                                   {2,   2,  0},
                                                   {2,   2,  2}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

    GraphBLAS::eWiseMult(Result,
                         Mask,
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), mA, transpose(mB));

    BOOST_CHECK_EQUAL(Result.nvals(), 11);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_masked_a_and_b_transpose)
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
                                                   {2,  14, 10},
                                                   {2,   2,  0},
                                                   {2,   2,  2}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

    GraphBLAS::eWiseMult(Result,
                         Mask,
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), transpose(mA), transpose(mB));

    BOOST_CHECK_EQUAL(Result.nvals(), 11);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
// Tests using a complemented mask with REPLACE semantics
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_scmp_masked_replace_bad_dimensions)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(twos3x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(4, 3);

    // incompatible Mask-Output dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseMult(Result,
                              GraphBLAS::complement(Mask),
                              GraphBLAS::NoAccumulate(),
                              GraphBLAS::Times<double>(), mA, mB,
                              true)),
        GraphBLAS::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_scmp_masked_replace_reg)
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

        std::vector<std::vector<double> > ans_dense = {{10, 12,  8},
                                                       {0,  14, 10},
                                                       {0,   0,  0},
                                                       {0,   0,  0}};
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(Result,
                             GraphBLAS::complement(Mask),
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
                             GraphBLAS::complement(Mask),
                             GraphBLAS::Second<double>(),
                             GraphBLAS::Times<double>(), mA, mB,
                             true);

        BOOST_CHECK_EQUAL(Result.nvals(), 6);
        BOOST_CHECK_EQUAL(Result, Ans);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_scmp_masked_replace_reg_stored_zero)
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

        std::vector<std::vector<double> > ans_dense = {{10, 12,  8},
                                                       {0,  14, 10},
                                                       {0,   0,  0},
                                                       {0,   0,  0}};
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(Result,
                             GraphBLAS::complement(Mask),
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
                             GraphBLAS::complement(Mask),
                             GraphBLAS::Second<double>(),
                             GraphBLAS::Times<double>(), mA, mB,
                             true);

        BOOST_CHECK_EQUAL(Result.nvals(), 6);
        BOOST_CHECK_EQUAL(Result, Ans);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_scmp_masked_replace_a_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos3x4_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m4x3_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(mask_dense, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{10, 12,  8},
                                                   {0,  14, 10},
                                                   {0,   0,  0},
                                                   {0,   0,  0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

    GraphBLAS::eWiseMult(Result,
                         GraphBLAS::complement(Mask),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), transpose(mA), mB,
                         true);

    BOOST_CHECK_EQUAL(Result.nvals(), 5);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_scmp_masked_replace_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m3x4_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(mask_dense, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{10, 12,  8},
                                                   {0,  14, 10},
                                                   {0,   0,  0},
                                                   {0,   0,  0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

    GraphBLAS::eWiseMult(Result,
                         GraphBLAS::complement(Mask),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), mA, transpose(mB),
                         true);

    BOOST_CHECK_EQUAL(Result.nvals(), 5);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_scmp_masked_replace_a_and_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos3x4_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m3x4_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(mask_dense, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{10, 12,  8},
                                                   {0,  14, 10},
                                                   {0,   0,  0},
                                                   {0,   0,  0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

    GraphBLAS::eWiseMult(Result,
                         GraphBLAS::complement(Mask),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), transpose(mA), transpose(mB),
                         true);

    BOOST_CHECK_EQUAL(Result.nvals(), 5);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
// Tests using a complemented mask (with merge semantics)
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_scmp_masked_bad_dimensions)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(twos3x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(4, 3);

    // incompatible Mask-Output dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseMult(Result,
                              GraphBLAS::complement(Mask),
                              GraphBLAS::NoAccumulate(),
                              GraphBLAS::Times<double>(), mA, mB)),
        GraphBLAS::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_scmp_masked_reg)
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

        std::vector<std::vector<double> > ans_dense = {{10, 12,  8},
                                                       {2,  14, 10},
                                                       {2,   2,  0},
                                                       {2,   2,  2}};
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(Result,
                             GraphBLAS::complement(Mask),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Times<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 11);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

    {
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{10, 12,  8},
                                                       {2,  14, 10},
                                                       {2,   2,  2},
                                                       {2,   2,  2}};
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(Result,
                             GraphBLAS::complement(Mask),
                             GraphBLAS::Second<double>(),
                             GraphBLAS::Times<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 12);
        BOOST_CHECK_EQUAL(Result, Ans);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_scmp_masked_reg_stored_zero)
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

        std::vector<std::vector<double> > ans_dense = {{10, 12,  8},
                                                       {2,  14, 10},
                                                       {2,   2,  0},
                                                       {2,   2,  2}};
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(Result,
                             GraphBLAS::complement(Mask),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Times<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 11);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

    {
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

        std::vector<std::vector<double> > ans_dense = {{10, 12,  8},
                                                       {2,  14, 10},
                                                       {2,   2,  2},
                                                       {2,   2,  2}};
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(Result,
                             GraphBLAS::complement(Mask),
                             GraphBLAS::Second<double>(),
                             GraphBLAS::Times<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 12);
        BOOST_CHECK_EQUAL(Result, Ans);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_scmp_masked_a_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos3x4_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m4x3_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(mask_dense, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{10, 12,  8},
                                                   {2,  14, 10},
                                                   {2,   2,  0},
                                                   {2,   2,  2}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

    GraphBLAS::eWiseMult(Result,
                         GraphBLAS::complement(Mask),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), transpose(mA), mB);

    BOOST_CHECK_EQUAL(Result.nvals(), 11);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_scmp_masked_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos4x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m3x4_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(mask_dense, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{10, 12,  8},
                                                   {2,  14, 10},
                                                   {2,   2,  0},
                                                   {2,   2,  2}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

    GraphBLAS::eWiseMult(Result,
                         GraphBLAS::complement(Mask),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), mA, transpose(mB));

    BOOST_CHECK_EQUAL(Result.nvals(), 11);
    BOOST_CHECK_EQUAL(Result, Ans);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_matrix_scmp_masked_a_and_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(twos3x4_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m3x4_dense, 0.);

    std::vector<std::vector<double> > mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Mask(mask_dense, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Result(twos4x3_dense, 0.);

    std::vector<std::vector<double> > ans_dense = {{10, 12,  8},
                                                   {2,  14, 10},
                                                   {2,   2,  0},
                                                   {2,   2,  2}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> Ans(ans_dense, 0.);

    GraphBLAS::eWiseMult(Result,
                         GraphBLAS::complement(Mask),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), transpose(mA), transpose(mB));

    BOOST_CHECK_EQUAL(Result.nvals(), 11);
    BOOST_CHECK_EQUAL(Result, Ans);
}


BOOST_AUTO_TEST_SUITE_END()
