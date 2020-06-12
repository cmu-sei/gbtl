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

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE mxv_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

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

    static std::vector<double> v4_ones = {1, 1, 1, 1};
    static std::vector<double> v3_ones = {1, 1, 1};
}

//****************************************************************************
// Tests without mask
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_bad_dimensions)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(m3x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> mB(m3x4_dense, 0.);

    grb::Vector<double, grb::SparseTag> u3(u3_dense, 0.);

    grb::Vector<double, grb::SparseTag> w4(4);
    grb::Vector<double, grb::SparseTag> w3(3);

    // incompatible input dimensions
    BOOST_CHECK_THROW(
        (grb::mxv(w3,
                  grb::NoMask(),
                  grb::NoAccumulate(),
                  grb::ArithmeticSemiring<double>(),
                  mB,
                  u3)),
        grb::DimensionException);

    // incompatible output matrix dimensions
    BOOST_CHECK_THROW(
        (grb::mxv(w4,
                  grb::NoMask(),
                  grb::NoAccumulate(),
                  grb::ArithmeticSemiring<double>(),
                  mA,
                  u3)),
        grb::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_reg)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(m3x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> mB(m3x4_dense, 0.);
    grb::Vector<double> u3(u3_dense, 0.);
    grb::Vector<double> u4(u4_dense, 0.);
    grb::Vector<double> result(3);
    grb::Vector<double> ansA(ans3_dense, 0.);
    grb::Vector<double> ansB(ans4_dense, 0.);

    grb::mxv(result,
             grb::NoMask(),
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(),
             mA,
             u3);
    BOOST_CHECK_EQUAL(result, ansA);

    grb::mxv(result,
             grb::NoMask(),
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(),
             mB,
             u4);
    BOOST_CHECK_EQUAL(result, ansB);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_stored_zero_result)
{
    // Build some matrices.
    std::vector<std::vector<int> > A = {{1, 1, 0, 0},
                                        {1, 2, 2, 0},
                                        {0, 2, 3, 3},
                                        {0, 0, 3, 4}};
    std::vector<int> u4_dense =  { 1,-1, 0, 0};
    grb::Matrix<int, grb::DirectedMatrixTag> mA(A, 0);
    grb::Vector<int> u(u4_dense, 0);
    grb::Vector<int> result(4);

    // use a different sentinel value so that stored zeros are preserved.
    int const NIL(666);
    std::vector<int> ans = {  0,  -1, -2, NIL};
    grb::Vector<int> answer(ans, NIL);

    grb::mxv(result,
             grb::NoMask(),
             grb::NoAccumulate(), // Second<int>(),
             grb::ArithmeticSemiring<int>(),
             mA,
             u);
    BOOST_CHECK_EQUAL(result, answer);
    BOOST_CHECK_EQUAL(result.nvals(), 3);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_transpose)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(m3x3_dense, 0.);
    grb::Matrix<double, grb::DirectedMatrixTag> mB(m3x4_dense, 0.);
    grb::Vector<double> u3(u3_dense, 0.);
    grb::Vector<double> u4(u4_dense, 0.);
    grb::Vector<double> result3(3);
    grb::Vector<double> result4(4);

    std::vector<double> ansAT_dense = {12, 1, 9};
    std::vector<double> ansBT_dense = {11, 7, 1, 2};

    grb::Vector<double> ansA(ansAT_dense, 0.);
    grb::Vector<double> ansB(ansBT_dense, 0.);

    grb::mxv(result3,
             grb::NoMask(),
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(),
             transpose(mA),
             u3);
    BOOST_CHECK_EQUAL(result3, ansA);

    BOOST_CHECK_THROW(
        (grb::mxv(result4,
                  grb::NoMask(),
                  grb::NoAccumulate(),
                  grb::ArithmeticSemiring<double>(),
                  transpose(mB),
                  u4)),
        grb::DimensionException);

    grb::mxv(result4,
             grb::NoMask(),
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(),
             transpose(mB),
             u3);
    BOOST_CHECK_EQUAL(result4, ansB);
}

//****************************************************************************
// Tests using a mask with REPLACE
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_masked_replace_bad_dimensions)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(m3x3_dense, 0.);

    grb::Vector<double, grb::SparseTag> u3(u3_dense, 0.);
    grb::Vector<double, grb::SparseTag> m4(u4_dense, 0.);
    grb::Vector<double, grb::SparseTag> w3(3);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (grb::mxv(w3,
                  m4,
                  grb::NoAccumulate(),
                  grb::ArithmeticSemiring<double>(), mA, u3,
                  grb::REPLACE)),
        grb::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_masked_replace_reg)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(m3x3_dense, 0.);

    grb::Vector<double, grb::SparseTag> u3(u3_dense, 0.);
    grb::Vector<double, grb::SparseTag> m3(u3_dense, 0.);
    BOOST_CHECK_EQUAL(m3.nvals(), 2);

    {
        grb::Vector<double> result(v3_ones, 0.);

        std::vector<double> ans = {13, 1, 0};
        grb::Vector<double> answer(ans, 0.);

        grb::mxv(result,
                 m3,
                 grb::Second<double>(),
                 grb::ArithmeticSemiring<double>(),
                 mA,
                 u3,
                 grb::REPLACE);

        BOOST_CHECK_EQUAL(result.nvals(), 2);
        BOOST_CHECK_EQUAL(result, answer);
    }

    {
        grb::Vector<double> result(v3_ones, 0.);

        std::vector<double> ans = {13, 0, 0};
        grb::Vector<double> answer(ans, 0.);

        grb::mxv(result,
                 m3,
                 grb::NoAccumulate(),
                 grb::ArithmeticSemiring<double>(),
                 mA,
                 u3,
                 grb::REPLACE);

        BOOST_CHECK_EQUAL(result.nvals(), 1);
        BOOST_CHECK_EQUAL(result, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_masked_replace_reg_stored_zero)
{
    // tests a computed and stored zero
    // tests a stored zero in the mask

    // Build some matrices.
    std::vector<std::vector<int> > A_dense = {{1, 1, 0, 0},
                                              {1, 2, 2, 0},
                                              {0, 2, 3, 3},
                                              {0, 0, 3, 4}};
    std::vector<int> u4_dense =  { 1,-1, 1, 0};
    std::vector<int> mask_dense =  { 1, 1, 1, 0};
    std::vector<int> ones_dense =  { 1, 1, 1, 1};
    grb::Matrix<int, grb::DirectedMatrixTag> A(A_dense, 0);
    grb::Vector<int> u(u4_dense, 0);
    grb::Vector<int> mask(mask_dense, 0);
    grb::Vector<int> result(ones_dense, 0);

    BOOST_CHECK_EQUAL(mask.nvals(), 3);  // [ 1 1 1 - ]
    mask.setElement(3, 0);
    BOOST_CHECK_EQUAL(mask.nvals(), 4);  // [ 1 1 1 0 ]

    int const NIL(666);
    std::vector<int> ans = { 0, 1, 1,  NIL};
    grb::Vector<int> answer(ans, NIL);

    grb::mxv(result,
             mask,
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(), A, u,
             grb::REPLACE);
    BOOST_CHECK_EQUAL(result.nvals(), 3);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_masked_replace_a_transpose)
{
    // Build some matrices.
    std::vector<std::vector<int> > A_dense = {{1, 1, 0, 0, 1},
                                              {1, 2, 2, 0, 1},
                                              {0, 2, 3, 3, 1},
                                              {0, 0, 3, 4, 1}};
    std::vector<int> u4_dense =  { 1,-1, 1, 0};
    std::vector<int> mask_dense =  { 1, 0, 1, 0, 1};
    std::vector<int> ones_dense =  { 1, 1, 1, 1, 1};
    grb::Matrix<int, grb::DirectedMatrixTag> A(A_dense, 0);
    grb::Vector<int> u(u4_dense, 0);
    grb::Vector<int> mask(mask_dense, 0);
    grb::Vector<int> result(ones_dense, 0);

    int const NIL(666);
    std::vector<int> ans = { 0, NIL, 1,  NIL, 1};
    grb::Vector<int> answer(ans, NIL);

    grb::mxv(result,
             mask,
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(),
             transpose(A),
             u,
             grb::REPLACE);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
// Tests using a mask with MERGE semantics
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_masked_bad_dimensions)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(m3x3_dense, 0.);

    grb::Vector<double, grb::SparseTag> u3(u3_dense, 0.);
    grb::Vector<double, grb::SparseTag> m4(u4_dense, 0.);
    grb::Vector<double, grb::SparseTag> w3(v3_ones, 0.);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (grb::mxv(w3,
                  m4,
                  grb::NoAccumulate(),
                  grb::ArithmeticSemiring<double>(),
                  mA,
                  u3)),
        grb::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_masked_reg)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(m3x3_dense, 0.);

    grb::Vector<double, grb::SparseTag> u3(u3_dense, 0.);
    grb::Vector<double, grb::SparseTag> m3(u3_dense, 0.); // [1 1 -]
    BOOST_CHECK_EQUAL(m3.nvals(), 2);

    {
        grb::Vector<double> result(v3_ones, 0.);

        std::vector<double> ans = {13, 1, 1};
        grb::Vector<double> answer(ans, 0.);

        grb::mxv(result,
                 m3,
                 grb::Second<double>(),
                 grb::ArithmeticSemiring<double>(),
                 mA,
                 u3);

        BOOST_CHECK_EQUAL(result.nvals(), 3);
        BOOST_CHECK_EQUAL(result, answer);
    }

    {
        grb::Vector<double> result(v3_ones, 0.);

        std::vector<double> ans = {13, 0, 1};
        grb::Vector<double> answer(ans, 0.);

        grb::mxv(result,
                 m3,
                 grb::NoAccumulate(),
                 grb::ArithmeticSemiring<double>(),
                 mA,
                 u3);

        BOOST_CHECK_EQUAL(result.nvals(), 2);
        BOOST_CHECK_EQUAL(result, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_masked_reg_stored_zero)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(m3x3_dense, 0.);

    grb::Vector<double, grb::SparseTag> u3(u3_dense, 0.);
    grb::Vector<double, grb::SparseTag> m3(u3_dense); // [1 1 0]
    BOOST_CHECK_EQUAL(m3.nvals(), 3);

    {
        grb::Vector<double> result(v3_ones, 0.);

        std::vector<double> ans = {13, 1, 1};
        grb::Vector<double> answer(ans, 0.);

        grb::mxv(result,
                 m3,
                 grb::Second<double>(),
                 grb::ArithmeticSemiring<double>(),
                 mA,
                 u3);

        BOOST_CHECK_EQUAL(result.nvals(), 3);
        BOOST_CHECK_EQUAL(result, answer);
    }

    {
        grb::Vector<double> result(v3_ones, 0.);

        std::vector<double> ans = {13, 0, 1};
        grb::Vector<double> answer(ans, 0.);

        grb::mxv(result,
                 m3,
                 grb::NoAccumulate(),
                 grb::ArithmeticSemiring<double>(),
                 mA,
                 u3);

        BOOST_CHECK_EQUAL(result.nvals(), 2);
        BOOST_CHECK_EQUAL(result, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_masked_a_transpose)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(m3x3_dense, 0.);

    grb::Vector<double, grb::SparseTag> u3(u3_dense, 0.); // [1 1 -]
    grb::Vector<double, grb::SparseTag> m3(u3_dense, 0.); // [1 1 -]
    grb::Vector<double> result(v3_ones, 0.);

    BOOST_CHECK_EQUAL(m3.nvals(), 2);

    std::vector<double> ans = {12, 1, 1};
    grb::Vector<double> answer(ans, 0.);

    grb::mxv(result,
             m3,
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(), transpose(mA), u3);

    BOOST_CHECK_EQUAL(result.nvals(), 3);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
// Tests using a complemented mask with REPLACE semantics
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_scmp_masked_replace_bad_dimensions)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(m3x3_dense, 0.);

    grb::Vector<double, grb::SparseTag> u3(u3_dense, 0.);
    grb::Vector<double, grb::SparseTag> m4(u4_dense, 0.);
    grb::Vector<double, grb::SparseTag> w3(3);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (grb::mxv(w3,
                  grb::complement(m4),
                  grb::NoAccumulate(),
                  grb::ArithmeticSemiring<double>(),
                  mA,
                  u3,
                  grb::REPLACE)),
        grb::DimensionException);
}
//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_scmp_masked_replace_reg)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(m3x3_dense, 0.);

    grb::Vector<double, grb::SparseTag> u3(u3_dense, 0.);

    std::vector<double> scmp_mask_dense = {0., 0., 1.};
    grb::Vector<double, grb::SparseTag> m3(scmp_mask_dense, 0.);
    BOOST_CHECK_EQUAL(m3.nvals(), 1);

    {
        grb::Vector<double> result(v3_ones, 0.);

        std::vector<double> ans = {13, 1, 0};
        grb::Vector<double> answer(ans, 0.);

        grb::mxv(result,
                 grb::complement(m3),
                 grb::Second<double>(),
                 grb::ArithmeticSemiring<double>(),
                 mA,
                 u3,
                 grb::REPLACE);

        BOOST_CHECK_EQUAL(result.nvals(), 2);
        BOOST_CHECK_EQUAL(result, answer);
    }

    {
        grb::Vector<double> result(v3_ones, 0.);

        std::vector<double> ans = {13, 0, 0};
        grb::Vector<double> answer(ans, 0.);

        grb::mxv(result,
                 grb::complement(m3),
                 grb::NoAccumulate(),
                 grb::ArithmeticSemiring<double>(),
                 mA,
                 u3,
                 grb::REPLACE);

        BOOST_CHECK_EQUAL(result.nvals(), 1);
        BOOST_CHECK_EQUAL(result, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_scmp_masked_replace_reg_stored_zero)
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
    grb::Matrix<int, grb::DirectedMatrixTag> A(A_dense, 0);
    grb::Vector<int> u(u4_dense, 0);
    grb::Vector<int> mask(scmp_mask_dense); //, 0);
    grb::Vector<int> result(ones_dense, 0);

    BOOST_CHECK_EQUAL(mask.nvals(), 4);  // [ 0 0 0 1 ]

    int const NIL(666);
    std::vector<int> ans = { 0, 1, 1,  NIL};
    grb::Vector<int> answer(ans, NIL);

    grb::mxv(result,
             grb::complement(mask),
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(),
             A,
             u,
             grb::REPLACE);
    BOOST_CHECK_EQUAL(result.nvals(), 3);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_scmp_masked_replace_a_transpose)
{
    // Build some matrices.
    std::vector<std::vector<int> > A_dense = {{1, 1, 0, 0, 1},
                                              {1, 2, 2, 0, 1},
                                              {0, 2, 3, 3, 1},
                                              {0, 0, 3, 4, 1}};
    std::vector<int> u4_dense =  { 1,-1, 1, 0};
    std::vector<int> mask_dense =  { 0, 1, 0, 1, 0};
    std::vector<int> ones_dense =  { 1, 1, 1, 1, 1};
    grb::Matrix<int, grb::DirectedMatrixTag> A(A_dense, 0);
    grb::Vector<int> u(u4_dense, 0);
    grb::Vector<int> mask(mask_dense, 0);
    grb::Vector<int> result(ones_dense, 0);

    int const NIL(666);
    std::vector<int> ans = { 0, NIL, 1,  NIL, 1};
    grb::Vector<int> answer(ans, NIL);

    grb::mxv(result,
             grb::complement(mask),
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(),
             transpose(A),
             u,
             grb::REPLACE);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
// Tests using a complemented mask (with merge semantics)
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_scmp_masked_bad_dimensions)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(m3x3_dense, 0.);

    grb::Vector<double, grb::SparseTag> u3(u3_dense, 0.);
    grb::Vector<double, grb::SparseTag> m4(u4_dense, 0.);
    grb::Vector<double, grb::SparseTag> w3(v3_ones, 0.);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (grb::mxv(w3,
                  grb::complement(m4),
                  grb::NoAccumulate(),
                  grb::ArithmeticSemiring<double>(),
                  mA,
                  u3)),
        grb::DimensionException);
}
//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_scmp_masked_reg)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(m3x3_dense, 0.);

    grb::Vector<double, grb::SparseTag> u3(u3_dense, 0.);
    grb::Vector<double, grb::SparseTag> m3(u3_dense, 0.); // [1 1 -]
    grb::Vector<double> result(v3_ones, 0.);

    BOOST_CHECK_EQUAL(m3.nvals(), 2);

    std::vector<double> ans = {1, 1, 7}; // if accum is NULL then {1, 0, 7};
    grb::Vector<double> answer(ans, 0.);

    grb::mxv(result,
             grb::complement(m3),
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(),
             mA,
             u3);

    BOOST_CHECK_EQUAL(result.nvals(), 3); //2);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_scmp_masked_reg_stored_zero)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(m3x3_dense, 0.);

    grb::Vector<double, grb::SparseTag> u3(u3_dense, 0.);
    grb::Vector<double, grb::SparseTag> m3(u3_dense); // [1 1 0]
    grb::Vector<double> result(v3_ones, 0.);

    BOOST_CHECK_EQUAL(m3.nvals(), 3);

    std::vector<double> ans = {1, 1, 7}; // if accum is NULL then {1, 0, 7};
    grb::Vector<double> answer(ans, 0.);

    grb::mxv(result,
             grb::complement(m3),
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(),
             mA,
             u3);

    BOOST_CHECK_EQUAL(result.nvals(), 3); //2);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_scmp_masked_a_transpose)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(m3x3_dense, 0.);

    grb::Vector<double, grb::SparseTag> u3(u3_dense, 0.); // [1 1 -]
    grb::Vector<double, grb::SparseTag> m3(u3_dense, 0.); // [1 1 -]
    grb::Vector<double> result(v3_ones, 0.);

    BOOST_CHECK_EQUAL(m3.nvals(), 2);

    std::vector<double> ans = {1, 1, 9};
    grb::Vector<double> answer(ans, 0.);

    grb::mxv(result,
             grb::complement(m3),
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(),
             transpose(mA),
             u3);

    BOOST_CHECK_EQUAL(result.nvals(), 3);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
// Matrix-vector multiply with mask and accumulator but no replace
//****************************************************************************
BOOST_AUTO_TEST_CASE(mxv_reg)
{
    std::vector<std::vector<double>> mat = {{8, 1, 6},
                                            {3, 0, 7},
                                            {0, 0, 2}};

    std::vector<uint32_t> mask = {1, 1, 1};
    grb::Vector<uint32_t, grb::SparseTag> m(mask, 0);

    grb::Matrix<double, grb::DirectedMatrixTag> A(mat, 0.);

    std::vector<double> vec1 = {-1, 2, 1};
    std::vector<double> vec2 = {0, 1, 0};
    std::vector<double> vec3 = {2, 1, 0};
    std::vector<double> vec4 = {0, 1, 1};

    grb::Vector<double, grb::SparseTag> v1(vec1, 0.);
    grb::Vector<double, grb::SparseTag> v2(vec2, 0.);
    grb::Vector<double, grb::SparseTag> v3(vec3, 0.);
    grb::Vector<double, grb::SparseTag> v4(vec4, 0.);

    std::vector<double> a1_dense = {0, 4, 2};
    std::vector<double> a2_dense = {1, 0, 0};
    std::vector<double> a3_dense = {17, 6, 0};
    std::vector<double> a4_dense = {7, 7, 2};

    grb::Vector<double, grb::SparseTag> ans1(a1_dense);
    grb::Vector<double, grb::SparseTag> ans2(a2_dense, 0.);
    grb::Vector<double, grb::SparseTag> ans3(a3_dense, 0.);
    grb::Vector<double, grb::SparseTag> ans4(a4_dense, 0.);

    grb::Vector<double, grb::SparseTag> result(3);

    grb::mxv(result,
             m,                                    // mask
             grb::Second<double>(),                // accum
             grb::ArithmeticSemiring<double>(),    // semiring
             A,
             v1);
    BOOST_CHECK_EQUAL(result, ans1);

    result.clear();
    grb::mxv(result,
             m,                                    // mask
             grb::Second<double>(),                // accum
             grb::ArithmeticSemiring<double>(),    // semiring
             A,
             v2);
    BOOST_CHECK_EQUAL(result, ans2);

    result.clear();
    grb::mxv(result,
             m,                                    // mask
             grb::Second<double>(),                // accum
             grb::ArithmeticSemiring<double>(),    // semiring
             A,
             v3);
    BOOST_CHECK_EQUAL(result, ans3);

    result.clear();
    grb::mxv(result,
             m,                                    // mask
             grb::Second<double>(),                // accum
             grb::ArithmeticSemiring<double>(),    // semiring
             A,
             v4);
    BOOST_CHECK_EQUAL(result, ans4);
}


BOOST_AUTO_TEST_SUITE_END()
