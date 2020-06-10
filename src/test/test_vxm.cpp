/*
 * GraphBLAS Template Library, Version 2.1
 *
 * Copyright 2019 Carnegie Mellon University, Battelle Memorial Institute, and
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

#include <graphblas/graphblas.hpp>

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE vxm_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

//****************************************************************************

namespace
{
    std::vector<std::vector<double> > m3x3_dense = {{12, 0, 7},
                                                    {1,  0, 0},
                                                    {3,  6, 9}};

    std::vector<std::vector<double> > m4x3_dense = {{5, 6, 4},
                                                    {0, 7, 5},
                                                    {1, 0, 0},
                                                    {2, 0, 1}};
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
BOOST_AUTO_TEST_CASE(test_vxm_bad_dimensions)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(m3x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m4x3_dense, 0.);

    GraphBLAS::Vector<double, GraphBLAS::SparseTag> u3(u3_dense, 0.);

    GraphBLAS::Vector<double, GraphBLAS::SparseTag> w4(4);
    GraphBLAS::Vector<double, GraphBLAS::SparseTag> w3(3);

    // incompatible input dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::vxm(w3,
                        GraphBLAS::NoMask(),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::ArithmeticSemiring<double>(), u3, mB)),
        GraphBLAS::DimensionException);

    // incompatible output matrix dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::vxm(w4,
                        GraphBLAS::NoMask(),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::ArithmeticSemiring<double>(), u3, mA)),
        GraphBLAS::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_vxm_reg)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(m3x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m4x3_dense, 0.);
    GraphBLAS::Vector<double> u3(u3_dense, 0.);
    GraphBLAS::Vector<double> u4(u4_dense, 0.);
    GraphBLAS::Vector<double> result(3);
    GraphBLAS::Vector<double> ansA(ans3_dense, 0.);
    GraphBLAS::Vector<double> ansB(ans4_dense, 0.);

    GraphBLAS::vxm(result,
                   GraphBLAS::NoMask(),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), u3, mA);
    BOOST_CHECK_EQUAL(result, ansA);

    GraphBLAS::vxm(result,
                   GraphBLAS::NoMask(),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), u4, mB);
    BOOST_CHECK_EQUAL(result, ansB);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_vxm_stored_zero_result)
{
    // Build some matrices.
    std::vector<std::vector<int> > A = {{1, 1, 0, 0},
                                        {1, 2, 2, 0},
                                        {0, 2, 3, 3},
                                        {0, 0, 3, 4}};
    std::vector<int> u4_dense =  { 1,-1, 0, 0};
    GraphBLAS::Matrix<int, GraphBLAS::DirectedMatrixTag> mA(A, 0);
    GraphBLAS::Vector<int> u(u4_dense, 0);
    GraphBLAS::Vector<int> result(4);

    // use a different sentinel value so that stored zeros are preserved.
    int const NIL(666);
    std::vector<int> ans = {  0,  -1, -2, NIL};
    GraphBLAS::Vector<int> answer(ans, NIL);

    GraphBLAS::vxm(result,
                   GraphBLAS::NoMask(),
                   GraphBLAS::NoAccumulate(), // Second<int>(),
                   GraphBLAS::ArithmeticSemiring<int>(), u, mA);
    BOOST_CHECK_EQUAL(result, answer);
    BOOST_CHECK_EQUAL(result.nvals(), 3);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(m3x3_dense, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(m4x3_dense, 0.);
    GraphBLAS::Vector<double> u3(u3_dense, 0.);
    GraphBLAS::Vector<double> u4(u4_dense, 0.);
    GraphBLAS::Vector<double> result3(3);
    GraphBLAS::Vector<double> result4(4);

    std::vector<double> ansAT_dense = {12, 1, 9};
    std::vector<double> ansBT_dense = {11, 7, 1, 2};

    GraphBLAS::Vector<double> ansA(ansAT_dense, 0.);
    GraphBLAS::Vector<double> ansB(ansBT_dense, 0.);

    GraphBLAS::vxm(result3,
                   GraphBLAS::NoMask(),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(),
                   u3, transpose(mA));
    BOOST_CHECK_EQUAL(result3, ansA);

    BOOST_CHECK_THROW(
        (GraphBLAS::vxm(result4,
                        GraphBLAS::NoMask(),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::ArithmeticSemiring<double>(),
                        u4, transpose(mB))),
        GraphBLAS::DimensionException);

    GraphBLAS::vxm(result4,
                   GraphBLAS::NoMask(),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(),
                   u3, transpose(mB));
    BOOST_CHECK_EQUAL(result4, ansB);
}

//****************************************************************************
// Tests using a mask with REPLACE
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_vxm_masked_replace_bad_dimensions)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(m3x3_dense, 0.);

    GraphBLAS::Vector<double, GraphBLAS::SparseTag> u3(u3_dense, 0.);
    GraphBLAS::Vector<double, GraphBLAS::SparseTag> m4(u4_dense, 0.);
    GraphBLAS::Vector<double, GraphBLAS::SparseTag> w3(3);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::vxm(w3,
                        m4,
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::ArithmeticSemiring<double>(), u3, mA,
                        GraphBLAS::REPLACE)),
        GraphBLAS::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_vxm_masked_replace_reg)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(m3x3_dense, 0.);

    GraphBLAS::Vector<double, GraphBLAS::SparseTag> u3(u3_dense, 0.);
    GraphBLAS::Vector<double, GraphBLAS::SparseTag> m3(u3_dense, 0.);
    BOOST_CHECK_EQUAL(m3.nvals(), 2);

    {
        GraphBLAS::Vector<double> result(v3_ones, 0.);

        std::vector<double> ans = {13, 1, 0};
        GraphBLAS::Vector<double> answer(ans, 0.);

        GraphBLAS::vxm(result,
                       m3,
                       GraphBLAS::Second<double>(),
                       GraphBLAS::ArithmeticSemiring<double>(), u3, mA,
                       GraphBLAS::REPLACE);

        BOOST_CHECK_EQUAL(result.nvals(), 2);
        BOOST_CHECK_EQUAL(result, answer);
    }

    {
        GraphBLAS::Vector<double> result(v3_ones, 0.);

        std::vector<double> ans = {13, 0, 0};
        GraphBLAS::Vector<double> answer(ans, 0.);

        GraphBLAS::vxm(result,
                       m3,
                       GraphBLAS::NoAccumulate(),
                       GraphBLAS::ArithmeticSemiring<double>(), u3, mA,
                       GraphBLAS::REPLACE);

        BOOST_CHECK_EQUAL(result.nvals(), 1);
        BOOST_CHECK_EQUAL(result, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_vxm_masked_replace_reg_stored_zero)
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
    GraphBLAS::Matrix<int, GraphBLAS::DirectedMatrixTag> A(A_dense, 0);
    GraphBLAS::Vector<int> u(u4_dense, 0);
    GraphBLAS::Vector<int> mask(mask_dense, 0);
    GraphBLAS::Vector<int> result(ones_dense, 0);

    BOOST_CHECK_EQUAL(mask.nvals(), 3);  // [ 1 1 1 - ]
    mask.setElement(3, 0);
    BOOST_CHECK_EQUAL(mask.nvals(), 4);  // [ 1 1 1 0 ]

    int const NIL(666);
    std::vector<int> ans = { 0, 1, 1,  NIL};
    GraphBLAS::Vector<int> answer(ans, NIL);

    GraphBLAS::vxm(result,
                   mask,
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), u, A,
                   GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result.nvals(), 3);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_vxm_masked_replace_a_transpose)
{
    // Build some matrices.
    std::vector<std::vector<int> > A_dense = {{1, 1, 0, 0},
                                              {1, 2, 2, 0},
                                              {0, 2, 3, 3},
                                              {0, 0, 3, 4},
                                              {1, 1, 1, 1}};
    std::vector<int> u4_dense =  { 1,-1, 1, 0};
    std::vector<int> mask_dense =  { 1, 0, 1, 0, 1};
    std::vector<int> ones_dense =  { 1, 1, 1, 1, 1};
    GraphBLAS::Matrix<int, GraphBLAS::DirectedMatrixTag> A(A_dense, 0);
    GraphBLAS::Vector<int> u(u4_dense, 0);
    GraphBLAS::Vector<int> mask(mask_dense, 0);
    GraphBLAS::Vector<int> result(ones_dense, 0);

    int const NIL(666);
    std::vector<int> ans = { 0, NIL, 1,  NIL, 1};
    GraphBLAS::Vector<int> answer(ans, NIL);

    GraphBLAS::vxm(result,
                   mask,
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), u, transpose(A),
                   GraphBLAS::REPLACE);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
// Tests using a mask with MERGE semantics
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_vxm_masked_bad_dimensions)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(m3x3_dense, 0.);

    GraphBLAS::Vector<double, GraphBLAS::SparseTag> u3(u3_dense, 0.);
    GraphBLAS::Vector<double, GraphBLAS::SparseTag> m4(u4_dense, 0.);
    GraphBLAS::Vector<double, GraphBLAS::SparseTag> w3(v3_ones, 0.);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::vxm(w3,
                        m4,
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::ArithmeticSemiring<double>(), u3, mA)),
        GraphBLAS::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_vxm_masked_reg)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(m3x3_dense, 0.);

    GraphBLAS::Vector<double, GraphBLAS::SparseTag> u3(u3_dense, 0.);
    GraphBLAS::Vector<double, GraphBLAS::SparseTag> m3(u3_dense, 0.); // [1 1 -]
    BOOST_CHECK_EQUAL(m3.nvals(), 2);

    {
        GraphBLAS::Vector<double> result(v3_ones, 0.);

        std::vector<double> ans = {13, 1, 1};
        GraphBLAS::Vector<double> answer(ans, 0.);

        GraphBLAS::vxm(result,
                       m3,
                       GraphBLAS::Second<double>(),
                       GraphBLAS::ArithmeticSemiring<double>(), u3, mA);

        BOOST_CHECK_EQUAL(result.nvals(), 3);
        BOOST_CHECK_EQUAL(result, answer);
    }

    {
        GraphBLAS::Vector<double> result(v3_ones, 0.);

        std::vector<double> ans = {13, 0, 1};
        GraphBLAS::Vector<double> answer(ans, 0.);

        GraphBLAS::vxm(result,
                       m3,
                       GraphBLAS::NoAccumulate(),
                       GraphBLAS::ArithmeticSemiring<double>(), u3, mA);

        BOOST_CHECK_EQUAL(result.nvals(), 2);
        BOOST_CHECK_EQUAL(result, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_vxm_masked_reg_stored_zero)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(m3x3_dense, 0.);

    GraphBLAS::Vector<double, GraphBLAS::SparseTag> u3(u3_dense, 0.);
    GraphBLAS::Vector<double, GraphBLAS::SparseTag> m3(u3_dense); // [1 1 0]
    BOOST_CHECK_EQUAL(m3.nvals(), 3);

    {
        GraphBLAS::Vector<double> result(v3_ones, 0.);

        std::vector<double> ans = {13, 1, 1};
        GraphBLAS::Vector<double> answer(ans, 0.);

        GraphBLAS::vxm(result,
                       m3,
                       GraphBLAS::Second<double>(),
                       GraphBLAS::ArithmeticSemiring<double>(), u3, mA);

        BOOST_CHECK_EQUAL(result.nvals(), 3);
        BOOST_CHECK_EQUAL(result, answer);
    }

    {
        GraphBLAS::Vector<double> result(v3_ones, 0.);

        std::vector<double> ans = {13, 0, 1};
        GraphBLAS::Vector<double> answer(ans, 0.);

        GraphBLAS::vxm(result,
                       m3,
                       GraphBLAS::NoAccumulate(),
                       GraphBLAS::ArithmeticSemiring<double>(), u3, mA);

        BOOST_CHECK_EQUAL(result.nvals(), 2);
        BOOST_CHECK_EQUAL(result, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_vxm_masked_a_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(m3x3_dense, 0.);

    GraphBLAS::Vector<double, GraphBLAS::SparseTag> u3(u3_dense, 0.); // [1 1 -]
    GraphBLAS::Vector<double, GraphBLAS::SparseTag> m3(u3_dense, 0.); // [1 1 -]
    GraphBLAS::Vector<double> result(v3_ones, 0.);

    BOOST_CHECK_EQUAL(m3.nvals(), 2);

    std::vector<double> ans = {12, 1, 1};
    GraphBLAS::Vector<double> answer(ans, 0.);

    GraphBLAS::vxm(result,
                   m3,
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), u3, transpose(mA));

    BOOST_CHECK_EQUAL(result.nvals(), 3);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
// Tests using a complemented mask with REPLACE semantics
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_vxm_scmp_masked_replace_bad_dimensions)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(m3x3_dense, 0.);

    GraphBLAS::Vector<double, GraphBLAS::SparseTag> u3(u3_dense, 0.);
    GraphBLAS::Vector<double, GraphBLAS::SparseTag> m4(u4_dense, 0.);
    GraphBLAS::Vector<double, GraphBLAS::SparseTag> w3(3);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::vxm(w3,
                        GraphBLAS::complement(m4),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::ArithmeticSemiring<double>(), u3, mA,
                        GraphBLAS::REPLACE)),
        GraphBLAS::DimensionException);
}
//****************************************************************************
BOOST_AUTO_TEST_CASE(test_vxm_scmp_masked_replace_reg)
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

        GraphBLAS::vxm(result,
                       GraphBLAS::complement(m3),
                       GraphBLAS::Second<double>(),
                       GraphBLAS::ArithmeticSemiring<double>(), u3, mA,
                       GraphBLAS::REPLACE);

        BOOST_CHECK_EQUAL(result.nvals(), 2);
        BOOST_CHECK_EQUAL(result, answer);
    }

    {
        GraphBLAS::Vector<double> result(v3_ones, 0.);

        std::vector<double> ans = {13, 0, 0};
        GraphBLAS::Vector<double> answer(ans, 0.);

        GraphBLAS::vxm(result,
                       GraphBLAS::complement(m3),
                       GraphBLAS::NoAccumulate(),
                       GraphBLAS::ArithmeticSemiring<double>(), u3, mA,
                       GraphBLAS::REPLACE);

        BOOST_CHECK_EQUAL(result.nvals(), 1);
        BOOST_CHECK_EQUAL(result, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_vxm_scmp_masked_replace_reg_stored_zero)
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

    GraphBLAS::vxm(result,
                   GraphBLAS::complement(mask),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), u, A,
                   GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result.nvals(), 3);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_vxm_scmp_masked_replace_a_transpose)
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

    GraphBLAS::vxm(result,
                   GraphBLAS::complement(mask),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), u, transpose(A),
                   GraphBLAS::REPLACE);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
// Tests using a complemented mask (with merge semantics)
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_vxm_scmp_masked_bad_dimensions)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(m3x3_dense, 0.);

    GraphBLAS::Vector<double, GraphBLAS::SparseTag> u3(u3_dense, 0.);
    GraphBLAS::Vector<double, GraphBLAS::SparseTag> m4(u4_dense, 0.);
    GraphBLAS::Vector<double, GraphBLAS::SparseTag> w3(v3_ones, 0.);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::vxm(w3,
                        GraphBLAS::complement(m4),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::ArithmeticSemiring<double>(), u3, mA)),
        GraphBLAS::DimensionException);
}
//****************************************************************************
BOOST_AUTO_TEST_CASE(test_vxm_scmp_masked_reg)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(m3x3_dense, 0.);

    GraphBLAS::Vector<double, GraphBLAS::SparseTag> u3(u3_dense, 0.);
    GraphBLAS::Vector<double, GraphBLAS::SparseTag> m3(u3_dense, 0.); // [1 1 -]
    GraphBLAS::Vector<double> result(v3_ones, 0.);

    BOOST_CHECK_EQUAL(m3.nvals(), 2);

    std::vector<double> ans = {1, 1, 7}; // if accum is NULL then {1, 0, 7};
    GraphBLAS::Vector<double> answer(ans, 0.);

    GraphBLAS::vxm(result,
                   GraphBLAS::complement(m3),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), u3, mA);

    BOOST_CHECK_EQUAL(result.nvals(), 3); //2);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_vxm_scmp_masked_reg_stored_zero)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(m3x3_dense, 0.);

    GraphBLAS::Vector<double, GraphBLAS::SparseTag> u3(u3_dense, 0.);
    GraphBLAS::Vector<double, GraphBLAS::SparseTag> m3(u3_dense); // [1 1 0]
    GraphBLAS::Vector<double> result(v3_ones, 0.);

    BOOST_CHECK_EQUAL(m3.nvals(), 3);

    std::vector<double> ans = {1, 1, 7}; // if accum is NULL then {1, 0, 7};
    GraphBLAS::Vector<double> answer(ans, 0.);

    GraphBLAS::vxm(result,
                   GraphBLAS::complement(m3),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), u3, mA);

    BOOST_CHECK_EQUAL(result.nvals(), 3); //2);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_vxm_scmp_masked_a_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(m3x3_dense, 0.);

    GraphBLAS::Vector<double, GraphBLAS::SparseTag> u3(u3_dense, 0.); // [1 1 -]
    GraphBLAS::Vector<double, GraphBLAS::SparseTag> m3(u3_dense, 0.); // [1 1 -]
    GraphBLAS::Vector<double> result(v3_ones, 0.);

    BOOST_CHECK_EQUAL(m3.nvals(), 2);

    std::vector<double> ans = {1, 1, 9};
    GraphBLAS::Vector<double> answer(ans, 0.);

    GraphBLAS::vxm(result,
                   GraphBLAS::complement(m3),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), u3, transpose(mA));

    BOOST_CHECK_EQUAL(result.nvals(), 3);
    BOOST_CHECK_EQUAL(result, answer);
}


BOOST_AUTO_TEST_SUITE_END()
