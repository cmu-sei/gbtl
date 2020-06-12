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

#include <iostream>
#include <graphblas/graphblas.hpp>

using namespace grb;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE views_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm)
{
    std::vector<std::vector<double> > A ={{12, 7, 3},
                                          { 4, 5, 6},
                                          { 7, 8, 9}};
    Matrix<double, DirectedMatrixTag> mA(A, 0);

    std::vector<std::vector<double> > B = {{5, 8, 1, 2},
                                           {6, 7, 3, 0},
                                           {4, 5, 9, 1}};
    Matrix<double, DirectedMatrixTag> mB(B, 0);

    Matrix<double, DirectedMatrixTag> result(3, 4);

    std::vector<std::vector<double> > ans = {{114, 160,  60,  27},
                                             { 74,  97,  73,  14},
                                             {119, 157, 112,  23}};
    Matrix<double, DirectedMatrixTag> answer(ans, 0);

    mxm(result,
        grb::NoMask(), grb::NoAccumulate(),
        grb::ArithmeticSemiring<double>(),
        mA, mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_masked)
{
    std::vector<std::vector<double> > A ={{12, 7, 3},
                                          { 4, 5, 6},
                                          { 7, 8, 9}};
    Matrix<double, DirectedMatrixTag> mA(A, 0);

    std::vector<std::vector<double> > B = {{5, 8, 1, 2},
                                           {6, 7, 3, 0},
                                           {4, 5, 9, 1}};
    Matrix<double, DirectedMatrixTag> mB(B, 0);

    Matrix<double, DirectedMatrixTag> result(3, 4);

    std::vector<std::vector<double> > ans = {{114,   0,   0,   0},
                                             { 74,  97,   0,   0},
                                             {119, 157, 112,   0}};
    Matrix<double, DirectedMatrixTag> answer(ans, 0);

    std::vector<std::vector<uint8_t> > M = {{1, 0, 0, 0},
                                            {1, 1, 0, 0},
                                            {1, 1, 1, 0}};
    Matrix<uint8_t, DirectedMatrixTag> mask(M, 0);

    mxm(result,
        mask, grb::NoAccumulate(),
        grb::ArithmeticSemiring<double>(),
        mA, mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_complemented_mask)
{
    std::vector<std::vector<double> > A ={{12, 7, 3},
                                          { 4, 5, 6},
                                          { 7, 8, 9}};
    Matrix<double, DirectedMatrixTag> mA(A, 0);

    std::vector<std::vector<double> > B = {{5, 8, 1, 2},
                                           {6, 7, 3, 0},
                                           {4, 5, 9, 1}};
    Matrix<double, DirectedMatrixTag> mB(B, 0);

    Matrix<double, DirectedMatrixTag> result(3, 4);

    std::vector<std::vector<double> > ans = {{114,   0,   0,   0},
                                             { 74,  97,   0,   0},
                                             {119, 157, 112,   0}};
    Matrix<double, DirectedMatrixTag> answer(ans, 0);

    std::vector<std::vector<uint8_t> > M = {{0, 1, 1, 1},
                                            {0, 0, 1, 1},
                                            {0, 0, 0, 1}};
    Matrix<uint8_t, DirectedMatrixTag> mask(M, 0);

    mxm(result,
        grb::complement(mask), grb::NoAccumulate(),
        grb::ArithmeticSemiring<double>(),
        mA, mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_structure_mask)
{
    std::vector<std::vector<double> > A ={{12, 7, 3},
                                          { 4, 5, 6},
                                          { 7, 8, 9}};
    Matrix<double, DirectedMatrixTag> mA(A, 0);

    std::vector<std::vector<double> > B = {{5, 8, 1, 2},
                                           {6, 7, 3, 0},
                                           {4, 5, 9, 1}};
    Matrix<double, DirectedMatrixTag> mB(B, 0);

    Matrix<double, DirectedMatrixTag> result(3, 4);

    std::vector<std::vector<double> > ans = {{114,   0,   0,   0},
                                             { 74,  97,   0,   0},
                                             {119, 157, 112,   0}};
    Matrix<double, DirectedMatrixTag> answer(ans, 0);

    std::vector<std::vector<uint8_t> > M = {{1,99,99,99},
                                            {0, 1,99,99},
                                            {0, 0, 1,99}};
    Matrix<uint8_t, DirectedMatrixTag> mask(M, 99);

    mxm(result,
        grb::structure(mask), grb::NoAccumulate(),
        grb::ArithmeticSemiring<double>(),
        mA, mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_complemented_structure_mask)
{
    std::vector<std::vector<double> > A ={{12, 7, 3},
                                          { 4, 5, 6},
                                          { 7, 8, 9}};
    Matrix<double, DirectedMatrixTag> mA(A, 0);

    std::vector<std::vector<double> > B = {{5, 8, 1, 2},
                                           {6, 7, 3, 0},
                                           {4, 5, 9, 1}};
    Matrix<double, DirectedMatrixTag> mB(B, 0);

    Matrix<double, DirectedMatrixTag> result(3, 4);

    std::vector<std::vector<double> > ans = {{114,   0,   0,   0},
                                             { 74,  97,   0,   0},
                                             {119, 157, 112,   0}};
    Matrix<double, DirectedMatrixTag> answer(ans, 0);

    std::vector<std::vector<uint8_t> > M = {{99, 1, 0, 1},
                                            {99,99, 0, 1},
                                            {99,99,99, 0}};
    Matrix<uint8_t, DirectedMatrixTag> mask(M, 99);

    mxm(result,
        grb::complement(grb::structure(mask)),
        grb::NoAccumulate(),
        grb::ArithmeticSemiring<double>(),
        mA, mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_a_transpose)
{
    std::vector<std::vector<double> > A ={{12, 7, 3},
                                          { 4, 5, 6},
                                          { 7, 8, 9}};
    Matrix<double, DirectedMatrixTag> mA(A, 0);

    std::vector<std::vector<double> > B = {{5, 8, 1, 2},
                                           {6, 7, 3, 0},
                                           {4, 5, 9, 1}};
    Matrix<double, DirectedMatrixTag> mB(B, 0);

    Matrix<double, DirectedMatrixTag> result(3, 4);

    IndexArrayType i_answer = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    IndexArrayType j_answer = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    std::vector<double> v_answer = {112, 159, 87, 31, 97, 131,
                                    94, 22, 87, 111, 102, 15};
    Matrix<double, DirectedMatrixTag> answer(3, 4);
    answer.build(i_answer, j_answer, v_answer);

    mxm(result,
        grb::NoMask(), grb::NoAccumulate(),
        grb::ArithmeticSemiring<double>(),
        transpose(mA), mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_b_transpose)
{
    std::vector<std::vector<double> > A ={{12, 7, 3},
                                          { 4, 5, 6},
                                          { 7, 8, 9}};
    Matrix<double, DirectedMatrixTag> mA(A, 0);

    std::vector<std::vector<double> > B = {{5, 8, 1},
                                           {2, 6, 7},
                                           {3, 4, 5}};
    Matrix<double, DirectedMatrixTag> mB(B, 0);

    Matrix<double, DirectedMatrixTag> result(3, 3);

    IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer = {119, 87, 79, 66, 80, 62, 108, 125, 98};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    answer.build(i_answer, j_answer, v_answer);

    mxm(result,
        grb::NoMask(), grb::NoAccumulate(),
        grb::ArithmeticSemiring<double>(),
        mA, transpose(mB));

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_a_and_b_transpose)
{
    std::vector<std::vector<double> > A ={{12, 7, 3},
                                          { 4, 5, 6},
                                          { 7, 8, 9}};
    Matrix<double, DirectedMatrixTag> mA(A, 0);

    std::vector<std::vector<double> > B = {{5, 8, 1},
                                           {2, 6, 7},
                                           {3, 4, 5}};
    Matrix<double, DirectedMatrixTag> mB(B, 0);

    Matrix<double, DirectedMatrixTag> result(3, 3);

    IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer = {99, 97, 87, 83, 100, 81, 72, 105, 78};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    answer.build(i_answer, j_answer, v_answer);

    mxm(result,
        grb::NoMask(), grb::NoAccumulate(),
        grb::ArithmeticSemiring<double>(),
        transpose(mA), transpose(mB));

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
//****************************************************************************


//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv)
{
    std::vector<std::vector<double> > A ={{12, 7, 3},
                                          { 4, 5, 6},
                                          { 7, 8, 9}};
    Matrix<double> mA(A, 0);

    std::vector<double> B = {5, 6, 4};
    Vector<double> mB(B, 0);

    Vector<double> result(3);

    std::vector<double> ans = {114,  74, 119};
    Vector<double> answer(ans, 0);

    mxv(result,
        grb::NoMask(), grb::NoAccumulate(),
        grb::ArithmeticSemiring<double>(),
        mA, mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_masked)
{
    std::vector<std::vector<double> > A ={{12, 7, 3},
                                          { 4, 5, 6},
                                          { 7, 8, 9}};
    Matrix<double> mA(A, 0);

    std::vector<double> B = {5, 6, 4};
    Vector<double> mB(B, 0);

    Vector<double> result(3);

    std::vector<double> ans = {114,  74, 0};
    Vector<double> answer(ans, 0);

    std::vector<uint8_t> M = {1, 1, 0};
    Vector<uint8_t> mask(M, 0);

    mxv(result,
        mask, grb::NoAccumulate(),
        grb::ArithmeticSemiring<double>(),
        mA, mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_complemented_mask)
{
    std::vector<std::vector<double> > A ={{12, 7, 3},
                                          { 4, 5, 6},
                                          { 7, 8, 9}};
    Matrix<double> mA(A, 0);

    std::vector<double> B = {5, 6, 4};
    Vector<double> mB(B, 0);

    Vector<double> result(3);

    std::vector<double> ans = {114,  74, 0};
    Vector<double> answer(ans, 0);

    std::vector<uint8_t> M = {0, 0, 1};
    Vector<uint8_t> mask(M, 0);

    mxv(result,
        grb::complement(mask), grb::NoAccumulate(),
        grb::ArithmeticSemiring<double>(),
        mA, mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_structure_mask)
{
    std::vector<std::vector<double> > A ={{12, 7, 3},
                                          { 4, 5, 6},
                                          { 7, 8, 9}};
    Matrix<double> mA(A, 0);

    std::vector<double> B = {5, 6, 4};
    Vector<double> mB(B, 0);

    Vector<double> result(3);

    std::vector<double> ans = {114,  74, 0};
    Vector<double> answer(ans, 0);

    std::vector<uint8_t> M = {1, 0, 99};
    Vector<uint8_t> mask(M, 99);

    mxv(result,
        grb::structure(mask), grb::NoAccumulate(),
        grb::ArithmeticSemiring<double>(),
        mA, mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_complemented_structure_mask)
{
    std::vector<std::vector<double> > A ={{12, 7, 3},
                                          { 4, 5, 6},
                                          { 7, 8, 9}};
    Matrix<double> mA(A, 0);

    std::vector<double> B = {5, 6, 4};
    Vector<double> mB(B, 0);

    Vector<double> result(3);

    std::vector<double> ans = {114,  74, 0};
    Vector<double> answer(ans, 0);

    std::vector<uint8_t> M = {99, 99, 0};
    Vector<uint8_t> mask(M, 99);

    mxv(result,
        grb::complement(grb::structure(mask)),
        grb::NoAccumulate(),
        grb::ArithmeticSemiring<double>(),
        mA, mB);

    BOOST_CHECK_EQUAL(result, answer);
}

BOOST_AUTO_TEST_SUITE_END()
