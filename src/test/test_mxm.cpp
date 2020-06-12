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

//#define GRAPHBLAS_LOGGING_LEVEL 2

#include <graphblas/graphblas.hpp>

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE mxm_test_suite

#include <boost/test/included/unit_test.hpp>

using namespace grb;

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

//****************************************************************************

namespace
{
    static std::vector<std::vector<double> > mA_dense_3x3 =
    {{12, 7, 3},
     {4,  5, 6},
     {7,  8, 9}};

    static std::vector<std::vector<double> > mAT_dense_3x3 =
    {{12, 4, 7},
     {7,  5, 8},
     {3,  6, 9}};

    static std::vector<std::vector<double> > mB_dense_3x4 =
    {{5, 8, 1, 2},
     {6, 7, 3, 0.},
     {4, 5, 9, 1}};

    static std::vector<std::vector<double> > mBT_dense_3x4 =
    {{5, 6, 4},
     {8, 7, 5},
     {1, 3, 9},
     {2, 0, 1}};

    static std::vector<std::vector<double> > mAnswer_dense =
    {{114, 160, 60,  27},
     {74,  97,  73,  14},
     {119, 157, 112, 23}};

    static std::vector<std::vector<double> > mAnswer_plus1_dense =
    {{115, 161, 61,  28},
     {75,  98,  74,  15},
     {120, 158, 113, 24}};

    static std::vector<std::vector<double> > mA_sparse_3x3 =
    {{12.3, 7.5,  0},
     {0,    -5.2, 0},
     {7.0,  0,    9.0}};

    static std::vector<std::vector<double> > mA_sparse_1337zero_3x3 =
    {{12.3, 7.5,  1337},
     {1337, -5.2, 1337},
     {7.0,  1337, 9.0}};

    static std::vector<std::vector<double> > mB_sparse_3x4 =
    {{5.0, 8.5, 0,   -2.1},
     {0.0, -7,  3.8, 0.0},
     {4.0, 0,   0,   1.3}};

    // mA_sparse_3x3 * mA_sparse_3x3
    static std::vector<std::vector<double> > mAmA_answer_sparse =
    {{12.3*12.3,      12.3*7.5-7.5*5.2,  0.0  },
     {0.0,             5.2*5.2,          0.0  },
     {12.3*7. + 7.*9., 7.5*7.,           9.*9.}};

    // mA_sparse_3x3 * mB_sparse_3x4
    static std::vector<std::vector<double> > mAnswer_sparse =
    {{61.5, 52.05, 28.5,   -25.83},
     {0.0,  36.4,  -19.76, 0.0},
     {71.0, 59.5,  0.0,    -3.0}};

    //static Matrix<double, DirectedMatrixTag> mAns(mAns_dense);
    grb::IndexArrayType i_all3x4 = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    grb::IndexArrayType j_all3x4 = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};

    static std::vector<std::vector<double> > mOnes_4x4 =
    {{1, 1, 1, 1},
     {1, 1, 1, 1},
     {1, 1, 1, 1},
     {1, 1, 1, 1}};

    static std::vector<std::vector<double> > mOnes_3x4 =
    {{1, 1, 1, 1},
     {1, 1, 1, 1},
     {1, 1, 1, 1}};

    static std::vector<std::vector<double> > mOnes_3x3 =
    {{1, 1, 1},
     {1, 1, 1},
     {1, 1, 1}};

    static std::vector<std::vector<double> > mIdentity_3x3 =
    {{1, 0, 0},
     {0, 1, 0},
     {0, 0, 1}};

    static std::vector<std::vector<double> > mLowerMask_3x4 =
    {{1, 0,    0,   0},
     {1, 0.5,  0,   0},
     {1, -1.0, 1.5, 0}};

    static std::vector<std::vector<bool> > mLowerBoolMask_3x4 =
    {{true, false, false, false},
     {true, true,  false, false},
     {true, true,  true,  false}};

    static std::vector<std::vector<bool> > mLowerBoolMask_3x3 =
    {{true, false, false},
     {true, true,  false},
     {true, true,  true}};

    static std::vector<std::vector<bool> > mScmpLowerBoolMask_3x3 =
    {{false,  true, true},
     {false, false, true},
     {false, false, false}};

}

//****************************************************************************
// API error tests
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_bad_dimensions)
{
    grb::Matrix<double, grb::DirectedMatrixTag> mA(mA_dense_3x3, 0.); // 3x3
    grb::Matrix<double, grb::DirectedMatrixTag> mB(mB_dense_3x4, 0.); // 3x4
    grb::Matrix<double, grb::DirectedMatrixTag> result3x4(3, 4);
    grb::Matrix<double, grb::DirectedMatrixTag> result3x3(3, 3);
    grb::Matrix<double, grb::DirectedMatrixTag> ones3x4(mOnes_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{1, 0, 0},
                                                          {1, 1, 0},
                                                          {1, 1, 1}};
    grb::Matrix<double, grb::DirectedMatrixTag> mMask(mMask_3x3, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 6);

    // NoMask_NoAccum_AB

    // ncols(A) != nrows(B)
    BOOST_CHECK_THROW(
        (mxm(result3x4,
             grb::NoMask(), grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(),
             mB, mA)),
        DimensionException);

    // dim(C) != dim(A*B)
    BOOST_CHECK_THROW(
        (mxm(result3x3,
             grb::NoMask(), grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(),
             mA, mB)),
        DimensionException);

    // NoMask_Accum_AB

    // incompatible input matrix dimensions
    BOOST_CHECK_THROW(
        (grb::mxm(result3x4,
                  grb::NoMask(),
                  grb::Second<double>(),
                  grb::ArithmeticSemiring<double>(), mB, mA)),
        grb::DimensionException);

    // incompatible output matrix dimensions
    BOOST_CHECK_THROW(
        (grb::mxm(result3x3,
                  grb::NoMask(),
                  grb::Second<double>(),
                  grb::ArithmeticSemiring<double>(), mA, mB)),
        grb::DimensionException);

    // Mask_NoAccum

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (grb::mxm(ones3x4,
                  mMask,
                  grb::NoAccumulate(),
                  grb::ArithmeticSemiring<double>(), mA, mB,
                  REPLACE)),
        grb::DimensionException);

    // Mask_Accum (replace and merge)

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (grb::mxm(ones3x4,
                  mMask,
                  grb::Second<double>(),
                  grb::ArithmeticSemiring<double>(), mA, mB, REPLACE)),
        grb::DimensionException);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (grb::mxm(ones3x4,
                  mMask,
                  grb::Second<double>(),
                  grb::ArithmeticSemiring<double>(), mA, mB)),
        grb::DimensionException);

    // CompMask_NoAccum (replace and merge)

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (grb::mxm(ones3x4,
                  grb::complement(mMask),
                  grb::NoAccumulate(),
                  grb::ArithmeticSemiring<double>(), mA, mB, REPLACE)),
        grb::DimensionException);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (grb::mxm(result3x4,
                  grb::complement(mMask),
                  grb::NoAccumulate(),
                  grb::ArithmeticSemiring<double>(), mA, mB)),
        grb::DimensionException);

    // CompMask_Accum (replace and merge)

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (grb::mxm(ones3x4,
                  grb::complement(mMask),
                  grb::Second<double>(),
                  grb::ArithmeticSemiring<double>(), mA, mB, REPLACE)),
        grb::DimensionException);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (grb::mxm(result3x4,
                  grb::complement(mMask),
                  grb::Second<double>(),
                  grb::ArithmeticSemiring<double>(), mA, mB)),
        grb::DimensionException);
}

//****************************************************************************
// NoMask_NoAccum
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_AB)
{
    grb::Matrix<double> mC(3, 4);
    grb::Matrix<double> mA(mA_sparse_3x3, 0.);
    grb::Matrix<double> mB(mB_sparse_3x4, 0.);

    grb::Matrix<double> answer(mAnswer_sparse, 0.);

    grb::mxm(mC,
             grb::NoMask(),
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(), mA, mB);
    for (grb::IndexType ix = 0; ix < answer.nrows(); ++ix)
    {
        for (grb::IndexType iy = 0; iy < answer.ncols(); ++iy)
        {
            BOOST_CHECK_EQUAL(mC.hasElement(ix, iy), answer.hasElement(ix, iy));
            if (mC.hasElement(ix, iy))
            {
                BOOST_CHECK_CLOSE(mC.extractElement(ix,iy),
                                  answer.extractElement(ix,iy), 0.0001);
            }
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_AB_empty)
{
    grb::Matrix<double> mZero(3, 3);
    grb::Matrix<double> mOnes(mOnes_3x3, 0.);
    grb::Matrix<double> mC(mOnes_3x3, 0.);
    grb::Matrix<double> mD(mOnes_3x3, 0.);

    grb::mxm(mC,
             NoMask(), NoAccumulate(),
             ArithmeticSemiring<double>(), mZero, mOnes);
    BOOST_CHECK_EQUAL(mC, mZero);

    grb::mxm(mD,
             NoMask(), NoAccumulate(),
             ArithmeticSemiring<double>(), mOnes, mZero);
    BOOST_CHECK_EQUAL(mD, mZero);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_AB_dense)
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

    mxm(result,
        grb::NoMask(), grb::NoAccumulate(),
        grb::ArithmeticSemiring<double>(),
        mA, mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_AB_emptyRowA_emptyColB)
{
    std::vector<std::vector<double>> mA_vals = {{8, 1, 6},
                                                {0, 0, 0},
                                                {4, 9, 2}};

    std::vector<std::vector<double>> mB_vals = {{0, 0, 0, 1},
                                                {1, 0, 1, 1},
                                                {0, 0, 1, 1}};

    std::vector<std::vector<double>> answer_vals = {{1, 0, 7, 15},
                                                    {0, 0, 0, 0},
                                                    {9, 0, 11, 15}};

    grb::Matrix<double> mA(mA_vals, 0.);
    grb::Matrix<double> mB(mB_vals, 0.);
    grb::Matrix<double> result(3, 4);
    grb::Matrix<double> answer(answer_vals, 0.);

    grb::mxm(result,
             grb::NoMask(),
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(), mA, mB);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_AB_emptyColA_emptyRowB)
{
    std::vector<std::vector<double>> mA_vals = {{8, 0, 6},
                                                {1, 0, 9},
                                                {4, 0, 2}};

    std::vector<std::vector<double>> mB_vals = {{0, 1, 0, 1},
                                                {1, 0, 1, 1},
                                                {0, 0, 0, 0}};

    std::vector<std::vector<double>> answer_vals = {{0, 8, 0, 8},
                                                    {0, 1, 0, 1},
                                                    {0, 4, 0, 4}};

    grb::Matrix<double> mA(mA_vals, 0.);
    grb::Matrix<double> mB(mB_vals, 0.);
    grb::Matrix<double> result(3, 4);
    grb::Matrix<double> answer(answer_vals, 0.);

    grb::mxm(result,
             grb::NoMask(),
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(), mA, mB);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_AB_ABdup)
{
    // Build some matrices.
    IndexArrayType      i = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType      j = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4};
    Matrix<double, DirectedMatrixTag> mat(4, 4);
    mat.build(i, j, v);

    Matrix<double, DirectedMatrixTag> m3(4, 4);

    IndexArrayType      i_answer = {0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
    IndexArrayType      j_answer = {0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3};
    std::vector<double> v_answer = {2, 3, 2, 3, 9,10, 6, 2,10,22,21, 6,21,25};
    Matrix<double, DirectedMatrixTag> answer(4, 4);
    answer.build(i_answer, j_answer, v_answer);

    mxm(m3,
        grb::NoMask(), grb::NoAccumulate(),
        grb::ArithmeticSemiring<double>(),
        mat, mat);

    BOOST_CHECK_EQUAL(m3, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_AB_ACdup)
{
    grb::Matrix<double> mC(mA_sparse_3x3, 0.);
    grb::Matrix<double> mB(mA_sparse_3x3, 0.);

    grb::Matrix<double> answer(mAmA_answer_sparse, 0.);

    grb::mxm(mC,
             grb::NoMask(),
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(), mC, mB);

    BOOST_CHECK_EQUAL(mC, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_AB_BCdup)
{
    grb::Matrix<double> mC(mA_sparse_3x3, 0.);
    grb::Matrix<double> mA(mA_sparse_3x3, 0.);

    grb::Matrix<double> answer(mAmA_answer_sparse, 0.);

    grb::mxm(mC,
             grb::NoMask(),
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(), mA, mC);

    BOOST_CHECK_EQUAL(mC, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_ATB)
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
        grb::NoMask(), grb::NoAccumulate(),
        grb::ArithmeticSemiring<double>(),
        transpose(mA), mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_ATB_empty)
{
    grb::Matrix<double> mZero(3, 3);
    grb::Matrix<double> mOnes(mOnes_3x3, 0.);
    grb::Matrix<double> mC(mOnes_3x3, 0.);
    grb::Matrix<double> mD(mOnes_3x3, 0.);

    grb::mxm(mC,
             NoMask(), NoAccumulate(),
             ArithmeticSemiring<double>(), transpose(mZero), mOnes);
    BOOST_CHECK_EQUAL(mC, mZero);

    grb::mxm(mD,
             NoMask(), NoAccumulate(),
             ArithmeticSemiring<double>(), transpose(mOnes), mZero);
    BOOST_CHECK_EQUAL(mD, mZero);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_ATB_identity_B)
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
        grb::NoMask(), grb::NoAccumulate(),
        grb::ArithmeticSemiring<double>(),
        transpose(mA), mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_ATB_emptyRowA_emptyColB)
{
    std::vector<std::vector<double>> mA_vals = {{8, 1, 6},
                                                {0, 0, 0},
                                                {4, 9, 2}};

    std::vector<std::vector<double>> mB_vals = {{1, 0, 0},
                                                {1, 0, 1},
                                                {0, 0, 1}};

    std::vector<std::vector<double>> answer_vals = {{8, 0, 4},
                                                    {1, 0, 9},
                                                    {6, 0, 2}};

    grb::Matrix<double> mA(mA_vals, 0.);
    grb::Matrix<double> mB(mB_vals, 0.);
    grb::Matrix<double> result(3, 3);
    grb::Matrix<double> answer(answer_vals, 0.);

    grb::mxm(result,
             grb::NoMask(),
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(), transpose(mA), mB);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_ATB_emptyColA_emptyRowB)
{
    std::vector<std::vector<double>> mA_vals = {{8, 0, 6},
                                                {1, 0, 9},
                                                {4, 0, 2}};

    std::vector<std::vector<double>> mB_vals = {{0, 1, 0},
                                                {1, 0, 1},
                                                {0, 0, 0}};

    std::vector<std::vector<double>> answer_vals = {{1, 8, 1},
                                                    {0, 0, 0},
                                                    {9, 6, 9}};

    grb::Matrix<double> mA(mA_vals, 0.);
    grb::Matrix<double> mB(mB_vals, 0.);
    grb::Matrix<double> result(3, 3);
    grb::Matrix<double> answer(answer_vals, 0.);

    grb::mxm(result,
             grb::NoMask(),
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(), transpose(mA), mB);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_ATB_ABdup)
{
    // Build some matrices.
    static std::vector<std::vector<double> > mat3x3 = {{1, 0, 0},
                                                       {1, 2, 0},
                                                       {0, 2, 3}};
    Matrix<double> mat(mat3x3, 0.);

    Matrix<double> res(3,3);

    static std::vector<std::vector<double> > ans3x3 =  {{2, 2, 0},
                                                        {2, 8, 6},
                                                        {0, 6, 9}};
    Matrix<double> answer(ans3x3, 0.);

    mxm(res,
        grb::NoMask(), grb::NoAccumulate(),
        grb::ArithmeticSemiring<double>(),
        transpose(mat), mat);

    BOOST_CHECK_EQUAL(res, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_ATB_ACdup)
{
    // Build some matrices.
    static std::vector<std::vector<double> > mat3x3 = {{1, 0, 0},
                                                       {1, 2, 0},
                                                       {0, 2, 3}};
    grb::Matrix<double> mC(mat3x3, 0.);
    grb::Matrix<double> mB(mat3x3, 0.);

    static std::vector<std::vector<double> > ans3x3 =  {{2, 2, 0},
                                                        {2, 8, 6},
                                                        {0, 6, 9}};
    Matrix<double> answer(ans3x3, 0.);

    grb::mxm(mC,
             grb::NoMask(),
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(), transpose(mC), mB);

    BOOST_CHECK_EQUAL(mC, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_ATB_BCdup)
{
    static std::vector<std::vector<double> > mat3x3 = {{1, 0, 0},
                                                       {1, 2, 0},
                                                       {0, 2, 3}};
    grb::Matrix<double> mA(mat3x3, 0.);
    grb::Matrix<double> mC(mat3x3, 0.);

    static std::vector<std::vector<double> > ans3x3 =  {{2, 2, 0},
                                                        {2, 8, 6},
                                                        {0, 6, 9}};
    Matrix<double> answer(ans3x3, 0.);
    grb::mxm(mC,
             grb::NoMask(),
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(), transpose(mA), mC);

    BOOST_CHECK_EQUAL(mC, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_ABT)
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
        grb::NoMask(), grb::NoAccumulate(),
        grb::ArithmeticSemiring<double>(),
        mA, transpose(mB));

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_ABT_empty)
{
    grb::Matrix<double> mZero(3, 3);
    grb::Matrix<double> mOnes(mOnes_3x3, 0.);
    grb::Matrix<double> mC(mOnes_3x3, 0.);
    grb::Matrix<double> mD(mOnes_3x3, 0.);

    grb::mxm(mC,
             NoMask(), NoAccumulate(),
             ArithmeticSemiring<double>(), mOnes, transpose(mZero));
    BOOST_CHECK_EQUAL(mC, mZero);

    grb::mxm(mD,
             NoMask(), NoAccumulate(),
             ArithmeticSemiring<double>(), mZero, transpose(mOnes));
    BOOST_CHECK_EQUAL(mD, mZero);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_ABT_emptyRowA_emptyColB)
{
    std::vector<std::vector<double>> mA_vals = {{8, 1, 6},
                                                {0, 0, 0},
                                                {4, 9, 2}};

    std::vector<std::vector<double>> mB_vals = {{1, 0, 0},
                                                {1, 0, 1},
                                                {0, 0, 1}};

    std::vector<std::vector<double>> answer_vals = {{8, 14, 6},
                                                    {0, 0,  0},
                                                    {4, 6,  2}};

    grb::Matrix<double> mA(mA_vals, 0.);
    grb::Matrix<double> mB(mB_vals, 0.);
    grb::Matrix<double> result(3, 3);
    grb::Matrix<double> answer(answer_vals, 0.);

    grb::mxm(result,
             grb::NoMask(),
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(), mA, transpose(mB));
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_ABT_emptyColA_emptyRowB)
{
    std::vector<std::vector<double>> mA_vals = {{8, 0, 6},
                                                {1, 0, 9},
                                                {4, 0, 2}};

    std::vector<std::vector<double>> mB_vals = {{0, 1, 0},
                                                {1, 0, 1},
                                                {0, 0, 0}};

    std::vector<std::vector<double>> answer_vals = {{0, 14, 0},
                                                    {0, 10, 0},
                                                    {0,  6, 0}};

    grb::Matrix<double> mA(mA_vals, 0.);
    grb::Matrix<double> mB(mB_vals, 0.);
    grb::Matrix<double> result(3, 3);
    grb::Matrix<double> answer(answer_vals, 0.);

    grb::mxm(result,
             grb::NoMask(),
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(), mA, transpose(mB));
    BOOST_CHECK_EQUAL(result, answer);
}
//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_ABT_ABdup)
{
    // Build some matrices.
    static std::vector<std::vector<double> > mat3x3 = {{1, 0, 0},
                                                       {1, 2, 0},
                                                       {0, 2, 3}};
    Matrix<double> mat(mat3x3, 0.);

    Matrix<double> res(3,3);

    static std::vector<std::vector<double> > ans3x3 =  {{1, 1, 0},
                                                        {1, 5, 4},
                                                        {0, 4,13}};
    Matrix<double> answer(ans3x3, 0.);

    mxm(res,
        grb::NoMask(), grb::NoAccumulate(),
        grb::ArithmeticSemiring<double>(),
        mat, transpose(mat));

    BOOST_CHECK_EQUAL(res, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_ABT_ACdup)
{
    // Build some matrices.
    static std::vector<std::vector<double> > mat3x3 = {{1, 0, 0},
                                                       {1, 2, 0},
                                                       {0, 2, 3}};
    grb::Matrix<double> mC(mat3x3, 0.);
    grb::Matrix<double> mB(mat3x3, 0.);

    static std::vector<std::vector<double> > ans3x3 =  {{1, 1, 0},
                                                        {1, 5, 4},
                                                        {0, 4,13}};
    Matrix<double> answer(ans3x3, 0.);

    grb::mxm(mC,
             grb::NoMask(),
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(), mC, transpose(mB));

    BOOST_CHECK_EQUAL(mC, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_ABT_BCdup)
{
    static std::vector<std::vector<double> > mat3x3 = {{1, 0, 0},
                                                       {1, 2, 0},
                                                       {0, 2, 3}};
    grb::Matrix<double> mA(mat3x3, 0.);
    grb::Matrix<double> mC(mat3x3, 0.);

    static std::vector<std::vector<double> > ans3x3 =  {{1, 1, 0},
                                                        {1, 5, 4},
                                                        {0, 4,13}};
    Matrix<double> answer(ans3x3, 0.);
    grb::mxm(mC,
             grb::NoMask(),
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(), mA, transpose(mC));

    BOOST_CHECK_EQUAL(mC, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_ATBT)
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
        grb::NoMask(), grb::NoAccumulate(),
        grb::ArithmeticSemiring<double>(),
        transpose(mA), transpose(mB));

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_ATBT_empty)
{
    grb::Matrix<double> mZero(3, 3);
    grb::Matrix<double> mOnes(mOnes_3x3, 0.);
    grb::Matrix<double> mC(mOnes_3x3, 0.);
    grb::Matrix<double> mD(mOnes_3x3, 0.);

    grb::mxm(mC,
             NoMask(), NoAccumulate(),
             ArithmeticSemiring<double>(),
             transpose(mOnes), transpose(mZero));
    BOOST_CHECK_EQUAL(mC, mZero);

    grb::mxm(mD,
             NoMask(), NoAccumulate(),
             ArithmeticSemiring<double>(),
             transpose(mZero), transpose(mOnes));
    BOOST_CHECK_EQUAL(mD, mZero);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_ATBT_emptyRowA_emptyColB)
{
    std::vector<std::vector<double>> mA_vals = {{8, 1, 6},
                                                {0, 0, 0},
                                                {4, 9, 2}};

    std::vector<std::vector<double>> mB_vals = {{1, 0, 0},
                                                {1, 0, 1},
                                                {0, 0, 1}};

    std::vector<std::vector<double>> answer_vals = {{8, 12, 4},
                                                    {1, 10, 9},
                                                    {6, 8,  2}};

    grb::Matrix<double> mA(mA_vals, 0.);
    grb::Matrix<double> mB(mB_vals, 0.);
    grb::Matrix<double> result(3, 3);
    grb::Matrix<double> answer(answer_vals, 0.);

    grb::mxm(result,
             grb::NoMask(),
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(),
             transpose(mA), transpose(mB));
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_ATBT_emptyColA_emptyRowB)
{
    std::vector<std::vector<double>> mA_vals = {{8, 0, 6},
                                                {1, 0, 9},
                                                {4, 0, 2}};

    std::vector<std::vector<double>> mB_vals = {{0, 1, 0},
                                                {1, 0, 1},
                                                {0, 0, 0}};

    std::vector<std::vector<double>> answer_vals = {{1, 12, 0},
                                                    {0,  0, 0},
                                                    {9,  8, 0}};

    grb::Matrix<double> mA(mA_vals, 0.);
    grb::Matrix<double> mB(mB_vals, 0.);
    grb::Matrix<double> result(3, 3);
    grb::Matrix<double> answer(answer_vals, 0.);

    grb::mxm(result,
             grb::NoMask(),
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(),
             transpose(mA), transpose(mB));
    BOOST_CHECK_EQUAL(result, answer);
}
//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_ATBT_ABdup)
{
    // Build some matrices.
    static std::vector<std::vector<double> > mat3x3 = {{1, 0, 0},
                                                       {1, 2, 0},
                                                       {0, 2, 3}};
    Matrix<double> mat(mat3x3, 0.);

    Matrix<double> res(3,3);

    static std::vector<std::vector<double> > ans3x3 =  {{1, 3, 2},
                                                        {0, 4,10},
                                                        {0, 0, 9}};
    Matrix<double> answer(ans3x3, 0.);

    mxm(res,
        grb::NoMask(), grb::NoAccumulate(),
        grb::ArithmeticSemiring<double>(),
        transpose(mat), transpose(mat));

    BOOST_CHECK_EQUAL(res, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_ATBT_ACdup)
{
    // Build some matrices.
    static std::vector<std::vector<double> > mat3x3 = {{1, 0, 0},
                                                       {1, 2, 0},
                                                       {0, 2, 3}};
    grb::Matrix<double> mC(mat3x3, 0.);
    grb::Matrix<double> mB(mat3x3, 0.);

    static std::vector<std::vector<double> > ans3x3 =  {{1, 3, 2},
                                                        {0, 4,10},
                                                        {0, 0, 9}};
    Matrix<double> answer(ans3x3, 0.);

    grb::mxm(mC,
             grb::NoMask(),
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(),
             transpose(mC), transpose(mB));

    BOOST_CHECK_EQUAL(mC, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_ATBT_BCdup)
{
    static std::vector<std::vector<double> > mat3x3 = {{1, 0, 0},
                                                       {1, 2, 0},
                                                       {0, 2, 3}};
    grb::Matrix<double> mA(mat3x3, 0.);
    grb::Matrix<double> mC(mat3x3, 0.);

    static std::vector<std::vector<double> > ans3x3 =  {{1, 3, 2},
                                                        {0, 4,10},
                                                        {0, 0, 9}};
    Matrix<double> answer(ans3x3, 0.);
    grb::mxm(mC,
             grb::NoMask(),
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(),
             transpose(mA), transpose(mC));

    BOOST_CHECK_EQUAL(mC, answer);
}

//****************************************************************************
// NoMask_Accum
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_Accum_AB)
{
    grb::Matrix<double> mA(mA_dense_3x3, 0.); // 3x3
    grb::Matrix<double> mB(mB_dense_3x4, 0.); // 3x4
    grb::Matrix<double> result(3, 4);
    grb::Matrix<double> answer(mAnswer_dense, 0.);

    grb::mxm(result,
             grb::NoMask(),
             grb::Second<double>(),
             grb::ArithmeticSemiring<double>(), mA, mB);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_Accum_AB_empty)
{
    grb::Matrix<double> mZero(3, 3);
    grb::Matrix<double> mOnes(mOnes_3x3, 0.);
    grb::Matrix<double> mC(mOnes_3x3, 0.);
    grb::Matrix<double> mD(mOnes_3x3, 0.);

    grb::mxm(mC,
             NoMask(), Plus<double>(),
             ArithmeticSemiring<double>(), mZero, mOnes);
    BOOST_CHECK_EQUAL(mC, mOnes);

    grb::mxm(mD,
             NoMask(), Plus<double>(),
             ArithmeticSemiring<double>(), mOnes, mZero);
    BOOST_CHECK_EQUAL(mD, mOnes);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_Accum_AB_stored_zero_result)
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
    grb::Matrix<int> mA(A, 0);
    grb::Matrix<int> mB(B, 0);
    grb::Matrix<int> result(4, 4);

    // use a different sentinel value so that stored zeros are preserved.
    int const NIL(666);
    std::vector<std::vector<int> > ans = {{  0,  -1, NIL, NIL},
                                          { -1,   0,   6,  -8},
                                          { -2,   2,   0,  -3},
                                          {NIL, NIL,  -3,   0}};
    grb::Matrix<int> answer(ans, NIL);

    grb::mxm(result,
             grb::NoMask(),
             grb::Second<int>(),
             grb::ArithmeticSemiring<int>(), mA, mB);
    BOOST_CHECK_EQUAL(result, answer);
    BOOST_CHECK_EQUAL(result.nvals(), 12);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_Accum_AB_ABdup_Cempty)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    grb::Matrix<double> mat(m, 0.);

    grb::Matrix<double> result(4, 4);

    std::vector<std::vector<double> > ans = {{2,  3,  2,  0},
                                             {3,  9, 10,  6},
                                             {2, 10, 22, 21},
                                             {0,  6, 21, 25}};
    grb::Matrix<double> answer(ans, 0.);

    grb::mxm(result,
             grb::NoMask(),
             grb::Second<double>(),
             grb::ArithmeticSemiring<double>(), mat, mat);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_Accum_ATB)
{
    grb::Matrix<double> mA(mAT_dense_3x3, 0.); // 3x3
    grb::Matrix<double> mB(mB_dense_3x4, 0.); // 3x4
    grb::Matrix<double> result(3, 4);
    grb::Matrix<double> answer(mAnswer_dense, 0.);
    grb::Matrix<double> answerp1(mAnswer_plus1_dense, 0.);

    grb::mxm(result,
             grb::NoMask(),
             grb::Plus<double>(),
             grb::ArithmeticSemiring<double>(), transpose(mA), mB);
    BOOST_CHECK_EQUAL(result, answer);

    grb::Matrix<double> res1(mOnes_3x4, 0.);
    grb::mxm(res1,
             grb::NoMask(),
             grb::Plus<double>(),
             grb::ArithmeticSemiring<double>(), transpose(mA), mB);
    BOOST_CHECK_EQUAL(res1, answerp1);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_Accum_ATB_empty)
{
    grb::Matrix<double> mZero(3, 3);
    grb::Matrix<double> mOnes(mOnes_3x3, 0.);
    grb::Matrix<double> mC(mOnes_3x3, 0.);
    grb::Matrix<double> mD(mOnes_3x3, 0.);

    grb::mxm(mC,
             NoMask(), Plus<double>(),
             ArithmeticSemiring<double>(),
             transpose(mZero), mOnes);
    BOOST_CHECK_EQUAL(mC, mOnes);

    grb::mxm(mD,
             NoMask(), Plus<double>(),
             ArithmeticSemiring<double>(),
             transpose(mOnes), mZero);
    BOOST_CHECK_EQUAL(mD, mOnes);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_Accum_ATB_Bidentity_Cempty)
{
    grb::Matrix<double> mA(mA_dense_3x3, 0.);
    grb::Matrix<double> mB(mIdentity_3x3, 0.);
    grb::Matrix<double> result(3, 3);

    std::vector<std::vector<double> > ans = {{12, 4, 7},
                                             { 7, 5, 8},
                                             { 3, 6, 9}};
    grb::Matrix<double> answer(ans, 0.);

    //auto answer = grb::transpose(mA);

    grb::mxm(result,
             grb::NoMask(),
             grb::Second<double>(),
             grb::ArithmeticSemiring<double>(), transpose(mA), mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_Accum_ABT)
{
    grb::Matrix<double> mA(mA_dense_3x3, 0.);

    std::vector<std::vector<double> > B = {{5, 8, 1},
                                           {2, 6, 7},
                                           {3, 4, 5}};
    grb::Matrix<double> mB(B, 0.);

    // empty dst
    grb::Matrix<double> result(3, 3);

    std::vector<std::vector<double> > ans = {{119,  87, 79},
                                             { 66,  80, 62},
                                             {108, 125, 98}};
    grb::Matrix<double> answer(ans, 0.);

    grb::mxm(result,
             grb::NoMask(),
             grb::Plus<double>(),
             grb::ArithmeticSemiring<double>(), mA, transpose(mB));

    BOOST_CHECK_EQUAL(result, answer);

    // filled dst
    grb::Matrix<double> res1(mOnes_3x3, 0.);
    std::vector<std::vector<double> > ans1 = {{120,  88, 80},
                                              { 67,  81, 63},
                                              {109, 126, 99}};
    grb::Matrix<double> answer1(ans1, 0.);
    grb::mxm(res1,
             grb::NoMask(),
             grb::Plus<double>(),
             grb::ArithmeticSemiring<double>(), mA, transpose(mB));

    BOOST_CHECK_EQUAL(res1, answer1);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_Accum_ABT_empty)
{
    grb::Matrix<double> mZero(3, 3);
    grb::Matrix<double> mOnes(mOnes_3x3, 0.);
    grb::Matrix<double> mC(mOnes_3x3, 0.);
    grb::Matrix<double> mD(mOnes_3x3, 0.);

    grb::mxm(mC,
             NoMask(), Plus<double>(),
             ArithmeticSemiring<double>(),
             mOnes, transpose(mZero));
    BOOST_CHECK_EQUAL(mC, mOnes);

    grb::mxm(mD,
             NoMask(), Plus<double>(),
             ArithmeticSemiring<double>(),
             mZero, transpose(mOnes));
    BOOST_CHECK_EQUAL(mD, mOnes);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_Accum_ATBT)
{
    grb::Matrix<double> mA(mA_dense_3x3, 0.);
    std::vector<std::vector<double> > B =  {{5, 8, 1},
                                            {2, 6, 7},
                                            {3, 4, 5}};
    grb::Matrix<double> mB(B, 0.);

    // Empty dst
    grb::Matrix<double> result(3, 3);

    std::vector<std::vector<double> > ans =  {{99,  97, 87},
                                              {83, 100, 81},
                                              {72, 105, 78}};
    grb::Matrix<double> answer(ans, 0.);

    grb::mxm(result,
             grb::NoMask(),
             grb::Second<double>(),
             grb::ArithmeticSemiring<double>(),
             transpose(mA), transpose(mB));

    BOOST_CHECK_EQUAL(result, answer);

    // Filled dst
    grb::Matrix<double> res1(mOnes_3x3, 0.);

    std::vector<std::vector<double> > ans1=  {{100,  98, 88},
                                              { 84, 101, 82},
                                              { 73, 106, 79}};
    grb::Matrix<double> answer1(ans1, 0.);

    grb::mxm(res1,
             grb::NoMask(),
             grb::Plus<double>(),
             grb::ArithmeticSemiring<double>(),
             transpose(mA), transpose(mB));

    BOOST_CHECK_EQUAL(res1, answer1);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_Accum_ATBT_empty)
{
    grb::Matrix<double> mZero(3, 3);
    grb::Matrix<double> mOnes(mOnes_3x3, 0.);
    grb::Matrix<double> mC(mOnes_3x3, 0.);
    grb::Matrix<double> mD(mOnes_3x3, 0.);

    grb::mxm(mC,
             NoMask(), Plus<double>(),
             ArithmeticSemiring<double>(),
             transpose(mOnes), transpose(mZero));
    BOOST_CHECK_EQUAL(mC, mOnes);

    grb::mxm(mD,
             NoMask(), Plus<double>(),
             ArithmeticSemiring<double>(),
             transpose(mZero), transpose(mOnes));
    BOOST_CHECK_EQUAL(mD, mOnes);
}

// ****************************************************************************
// ****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_NoAccum_AB_empty)
{
    grb::Matrix<double> mZero(3, 3);
    grb::Matrix<double> mOnes(mOnes_3x3, 0.);
    grb::Matrix<double> mC(3,3);

    // NOTE: The mask is true for any non-zero.
    grb::Matrix<bool> mMask(mLowerBoolMask_3x3, false);

    std::vector<std::vector<double>> upper = {{0, 1, 1},
                                              {0, 0, 1},
                                              {0, 0, 0}};
    grb::Matrix<double> mUpper(upper, 0.);

    // Merge
    mC = mOnes;
    grb::mxm(mC,
             mMask, NoAccumulate(),
             ArithmeticSemiring<double>(), mZero, mOnes);
    BOOST_CHECK_EQUAL(mC, mUpper);

    mC = mOnes;
    grb::mxm(mC,
             mMask, NoAccumulate(),
             ArithmeticSemiring<double>(), mOnes, mZero);
    BOOST_CHECK_EQUAL(mC, mUpper);

    // Replace
    mC = mOnes;
    grb::mxm(mC,
             mMask, NoAccumulate(),
             ArithmeticSemiring<double>(), mZero, mOnes, REPLACE);
    BOOST_CHECK_EQUAL(mC, mZero);

    mC = mOnes;
    grb::mxm(mC,
             mMask, NoAccumulate(),
             ArithmeticSemiring<double>(), mOnes, mZero, REPLACE);
    BOOST_CHECK_EQUAL(mC, mZero);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_NoAccum_AB_Merge_full_mask)
{
    IndexArrayType i_mA      =  {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA      =  {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mB      = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2};
    IndexArrayType j_mB      = {0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3};
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
        mask, grb::NoAccumulate(),
        grb::ArithmeticSemiring<double>(),
        mA, mB);

    BOOST_CHECK_EQUAL(result, answer);
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_NoAccum_AB_mask_not_full)
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

    IndexArrayType i_answer = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_answer = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2};
    std::vector<double> v_answer = {114, 160, 60, 27, 74, 97,
                                    73, 14, 119, 157, 112};
    Matrix<double, DirectedMatrixTag> answer(3, 4);
    answer.build(i_answer, j_answer, v_answer);

    Matrix<unsigned int, DirectedMatrixTag> mask(3,4);
    std::vector<unsigned int> v_mask(i_answer.size(), 1);
    mask.build(i_answer, j_answer, v_mask);

    mxm(result,
        mask, grb::NoAccumulate(),
        grb::ArithmeticSemiring<double>(),
        mA, mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_NoAccum_AB_Merge_Cones_Mlower_stored_zero)
{
    grb::Matrix<double> mA(mA_dense_3x3, 0.);
    grb::Matrix<double> mB(mB_dense_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{1, 0, 0, 0},
                                                          {1, 1, 0, 0},
                                                          {1, 1, 1, 0}};
    grb::Matrix<double> mMask(mMask_3x4, 0.);
    mMask.setElement(0, 1, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 7);

    grb::Matrix<double> result(mOnes_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   1,   1,  1},
                                             { 74,  97,   1,  1},
                                             {119, 157, 112,  1}};
    grb::Matrix<double> answer(ans, 0.);

    grb::mxm(result,
             mMask,
//                   grb::Second<double>(),
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(), mA, mB);
    BOOST_CHECK_EQUAL(result.nvals(), 12);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_NoAccum_AB_Merge_Cones_ABdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    grb::Matrix<double> mat(m, 0.);

    grb::Matrix<double> result(mOnes_4x4, 0.);

    std::vector<std::vector<double> > ans = {{2,  1,  1,  1},
                                             {3,  9,  1,  1},
                                             {2, 10, 22,  1},
                                             {0,  6, 21, 25}};
    grb::Matrix<double> answer(ans, 0.);

    static std::vector<std::vector<double> > mMask_4x4 = {{1, 0, 0, 0},
                                                          {1, 1, 0, 0},
                                                          {1, 1, 1, 0},
                                                          {1, 1, 1, 1}};
    grb::Matrix<double> mMask(mMask_4x4, 0.);

    grb::mxm(result,
             mMask,
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(), mat, mat);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_NoAccum_AB_Replace_ABdup_result_ones)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    grb::Matrix<double> mat(m, 0.);

    grb::Matrix<double> result(mOnes_4x4, 0.);

    std::vector<std::vector<double> > ans = {{2,  0,  0,  0},
                                             {3,  9,  0,  0},
                                             {2, 10, 22,  0},
                                             {0,  6, 21, 25}};
    grb::Matrix<double> answer(ans, 0.);

    static std::vector<std::vector<double> > mMask_4x4 = {{1, 0, 0, 0},
                                                          {1, 1, 0, 0},
                                                          {1, 1, 1, 0},
                                                          {1, 1, 1, 1}};
    grb::Matrix<double> mMask(mMask_4x4, 0.);

    grb::mxm(result,
             mMask,
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(), mat, mat,
             REPLACE);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_Accum_AB_Replace_lower_mask_result_ones)
{
    grb::Matrix<double> mA(mA_dense_3x3, 0.);
    grb::Matrix<double> mB(mB_dense_3x4, 0.);

    // NOTE: The mask is true for any non-zero.
    grb::Matrix<double> mMask(mLowerMask_3x4, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 6);

    grb::Matrix<double> result(mOnes_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   0,   0,  0},
                                             { 74,  97,   0,  0},
                                             {119, 157, 112,  0}};
    grb::Matrix<double> answer(ans, 0.);

    grb::mxm(result,
             mMask,
             grb::Second<double>(),
             grb::ArithmeticSemiring<double>(), mA, mB,
             REPLACE);

    BOOST_CHECK_EQUAL(result.nvals(), 6);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_Accum_AB_Replace_bool_masked_result_ones)
{
    grb::Matrix<double> mA(mA_dense_3x3, 0.);
    grb::Matrix<double> mB(mB_dense_3x4, 0.);

    grb::Matrix<bool> mMask(mLowerBoolMask_3x4, false);
    BOOST_CHECK_EQUAL(mMask.nvals(), 6);

    grb::Matrix<double> result(mOnes_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   0,   0,  0},
                                             { 74,  97,   0,  0},
                                             {119, 157, 112,  0}};
    grb::Matrix<double> answer(ans, 0.);

    grb::mxm(result,
             mMask,
             grb::Second<double>(),
             grb::ArithmeticSemiring<double>(), mA, mB,
             REPLACE);

    BOOST_CHECK_EQUAL(result.nvals(), 6);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_Accum_AB_Replace_mask_stored_zero_result_ones)
{
    grb::Matrix<double> mA(mA_dense_3x3, 0.);
    grb::Matrix<double> mB(mB_dense_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{1, 0, 0, 0},
                                                          {1, 1, 0, 0},
                                                          {1, 1, 1, 0}};
    grb::Matrix<double> mMask(mMask_3x4, 0.);
    mMask.setElement(0, 1, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 7);

    grb::Matrix<double> result(mOnes_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   0,   0,  0},
                                             { 74,  97,   0,  0},
                                             {119, 157, 112,  0}};
    grb::Matrix<double> answer(ans, 0.);

    grb::mxm(result,
             mMask,
             grb::Second<double>(),
             grb::ArithmeticSemiring<double>(), mA, mB,
             REPLACE);
    BOOST_CHECK_EQUAL(result.nvals(), 6);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_Accum_AB_Merge_Cones_Mlower)
{
    grb::Matrix<double> mA(mA_dense_3x3, 0.);
    grb::Matrix<double> mB(mB_dense_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{1, 0, 0, 0},
                                                          {1, 1, 0, 0},
                                                          {1, 1, 1, 0}};
    grb::Matrix<double> mMask(mMask_3x4, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 6);

    grb::Matrix<double> result(mOnes_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   1,   1,  1},
                                             { 74,  97,   1,  1},
                                             {119, 157, 112,  1}};
    grb::Matrix<double> answer(ans, 0.);

    grb::mxm(result,
             mMask,
             grb::Second<double>(),
             grb::ArithmeticSemiring<double>(), mA, mB);
    BOOST_CHECK_EQUAL(result.nvals(), 12);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_Accum_ATB_Replace_Mlower_Cones)
{
    grb::Matrix<double> mA(mA_dense_3x3, 0.);
    grb::Matrix<double> mB(mB_dense_3x4, 0.);

    grb::Matrix<double> result(mOnes_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{1, 0, 0, 0},
                                                          {1, 1, 0, 0},
                                                          {1, 1, 1, 0}};
    grb::Matrix<double> mMask(mMask_3x4, 0.);

    std::vector<std::vector<double> > ans = {{112,   0,   0,  0},
                                             { 97, 131,   0,  0},
                                             { 87, 111, 102,  0}};
    grb::Matrix<double> answer(ans, 0.);

    grb::mxm(result,
             mMask,
             grb::Second<double>(),
             grb::ArithmeticSemiring<double>(), transpose(mA), mB,
             REPLACE);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_Accum_ATB_Replace_Mlower_Cones_Bidentity)
{
    grb::Matrix<double> mA(mA_dense_3x3, 0.);
    grb::Matrix<double> mB(mIdentity_3x3, 0.);
    grb::Matrix<double> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{1, 0, 0},
                                                          {1, 1, 0},
                                                          {1, 1, 1}};
    grb::Matrix<double> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans = {{12, 0, 0},
                                             { 7, 5, 0},
                                             { 3, 6, 9}};
    grb::Matrix<double> answer(ans, 0.);

    //auto answer = grb::transpose(mA);

    grb::mxm(result,
             mMask,
             grb::Second<double>(),
             grb::ArithmeticSemiring<double>(), transpose(mA), mB,
             REPLACE);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_Accum_ATB_Merge_Cones_Mlower)
{
    grb::Matrix<double> mA(mA_dense_3x3, 0.);

    std::vector<std::vector<double> > B = {{5, 8, 1},
                                           {2, 6, 7},
                                           {3, 4, 5}};
    grb::Matrix<double> mB(B, 0.);

    grb::Matrix<double> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{1, 0, 0},
                                                          {1, 1, 0},
                                                          {1, 1, 1}};
    grb::Matrix<double> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans = {{119,   1,  1},
                                             { 66,  80,  1},
                                             {108, 125, 98}};
    grb::Matrix<double> answer(ans, 0.);

    grb::mxm(result,
             mMask,
             grb::Second<double>(),
             grb::ArithmeticSemiring<double>(), mA, transpose(mB));

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_Accum_ATB_Merge_Cones_Mmasked_Bidentity)
{
    grb::Matrix<double> mA(mA_dense_3x3, 0.);
    grb::Matrix<double> mB(mIdentity_3x3, 0.);
    grb::Matrix<double> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{1, 0, 0},
                                                          {1, 1, 0},
                                                          {1, 1, 1}};
    grb::Matrix<double> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans = {{12, 1, 1},
                                             { 7, 5, 1},
                                             { 3, 6, 9}};
    grb::Matrix<double> answer(ans, 0.);

    //auto answer = grb::transpose(mA);

    grb::mxm(result,
             mMask,
             grb::Second<double>(),
             grb::ArithmeticSemiring<double>(), transpose(mA), mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_Accum_ATB_Cones_Mlower)
{
    grb::Matrix<double> mA(mA_dense_3x3, 0.);
    grb::Matrix<double> mB(mB_dense_3x4, 0.);

    grb::Matrix<double> result(mOnes_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{1, 0, 0, 0},
                                                          {1, 1, 0, 0},
                                                          {1, 1, 1, 0}};
    grb::Matrix<double> mMask(mMask_3x4, 0.);

    std::vector<std::vector<double> > ans = {{112,   1,   1,  1},
                                             { 97, 131,   1,  1},
                                             { 87, 111, 102,  1}};
    grb::Matrix<double> answer(ans, 0.);

    grb::mxm(result,
             mMask,
             grb::Second<double>(),
             grb::ArithmeticSemiring<double>(), transpose(mA), mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_Accum_ABT_Mlower_Cones)
{
    grb::Matrix<double> mA(mA_dense_3x3, 0.);

    std::vector<std::vector<double> > B = {{5, 8, 1},
                                           {2, 6, 7},
                                           {3, 4, 5}};
    grb::Matrix<double> mB(B, 0.);

    grb::Matrix<double> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{1, 0, 0},
                                                          {1, 1, 0},
                                                          {1, 1, 1}};
    grb::Matrix<double> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans = {{119,   0,  0},
                                             { 66,  80,  0},
                                             {108, 125, 98}};
    grb::Matrix<double> answer(ans, 0.);

    grb::mxm(result,
             mMask,
             grb::Second<double>(),
             grb::ArithmeticSemiring<double>(), mA, transpose(mB),
             REPLACE);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_Accum_ATBT_Mlower_Cones)
{
    grb::Matrix<double> mA(mA_dense_3x3, 0.);
    std::vector<std::vector<double> > B =  {{5, 8, 1},
                                            {2, 6, 7},
                                            {3, 4, 5}};
    grb::Matrix<double> mB(B, 0.);

    grb::Matrix<double> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{1, 0, 0},
                                                          {1, 1, 0},
                                                          {1, 1, 1}};
    grb::Matrix<double> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans =  {{99,   0,  0},
                                              {83, 100,  0},
                                              {72, 105, 78}};
    grb::Matrix<double> answer(ans, 0.);

    grb::mxm(result,
             mMask,
             grb::Second<double>(),
             grb::ArithmeticSemiring<double>(), transpose(mA), transpose(mB),
             REPLACE);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_Accum_ATBT_Merge_Cones_Mlower)
{
    grb::Matrix<double> mA(mA_dense_3x3, 0.);
    std::vector<std::vector<double> > B =  {{5, 8, 1},
                                            {2, 6, 7},
                                            {3, 4, 5}};
    grb::Matrix<double> mB(B, 0.);

    grb::Matrix<double> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{1, 0, 0},
                                                          {1, 1, 0},
                                                          {1, 1, 1}};
    grb::Matrix<double> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans =  {{99,   1,  1},
                                              {83, 100,  1},
                                              {72, 105, 78}};
    grb::Matrix<double> answer(ans, 0.);

    grb::mxm(result,
             mMask,
             grb::Second<double>(),
             grb::ArithmeticSemiring<double>(), transpose(mA), transpose(mB));

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_NoAccum_AB_Replace_ABdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    grb::Matrix<double> mat(m, 0.);

    grb::Matrix<double> result(mOnes_4x4, 0.);

    std::vector<std::vector<double> > ans = {{2,  0,  0,  0},
                                             {3,  9,  0,  0},
                                             {2, 10, 22,  0},
                                             {0,  6, 21, 25}};
    grb::Matrix<double> answer(ans, 0.);

    static std::vector<std::vector<double> > mMask_4x4 = {{0, 1, 1, 1},
                                                          {0, 0, 1, 1},
                                                          {0, 0, 0, 1},
                                                          {0, 0, 0, 0}};
    grb::Matrix<double> mMask(mMask_4x4, 0.);

    grb::mxm(result,
             grb::complement(mMask),
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(), mat, mat,
             REPLACE);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_Accum_AB_Replace_Cones_Mnlower)
{
    grb::Matrix<double> mA(mA_dense_3x3, 0.);
    grb::Matrix<double> mB(mB_dense_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{0, 1, 1, 1},
                                                          {0, 0, 1, 1},
                                                          {0, 0, 0, 1}};
    grb::Matrix<double> mMask(mMask_3x4, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 6);

    grb::Matrix<double> result(mOnes_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   0,   0,  0},
                                             { 74,  97,   0,  0},
                                             {119, 157, 112,  0}};
    grb::Matrix<double> answer(ans, 0.);

    grb::mxm(result,
             grb::complement(mMask),
             grb::Second<double>(),
             grb::ArithmeticSemiring<double>(), mA, mB,
             REPLACE);
    BOOST_CHECK_EQUAL(result.nvals(), 6);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_Accum_AB_Replace_Mstored_zero)
{
    grb::Matrix<double> mA(mA_dense_3x3, 0.);
    grb::Matrix<double> mB(mB_dense_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{0, 1, 1, 1},
                                                          {0, 0, 1, 1},
                                                          {0, 0, 0, 1}};
    grb::Matrix<double> mMask(mMask_3x4, 0.);
    mMask.setElement(0, 0, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 7);

    grb::Matrix<double> result(mOnes_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   0,   0,  0},
                                             { 74,  97,   0,  0},
                                             {119, 157, 112,  0}};
    grb::Matrix<double> answer(ans, 0.);

    grb::mxm(result,
             grb::complement(mMask),
             grb::Second<double>(),
             grb::ArithmeticSemiring<double>(), mA, mB,
             REPLACE);
    BOOST_CHECK_EQUAL(result.nvals(), 6);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_Accum_AB_Merge)
{
    grb::Matrix<double> mA(mA_dense_3x3, 0.);
    grb::Matrix<double> mB(mB_dense_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{0, 1, 1, 1},
                                                          {0, 0, 1, 1},
                                                          {0, 0, 0, 1}};
    grb::Matrix<double> mMask(mMask_3x4, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 6);

    grb::Matrix<double> result(mOnes_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   1,   1,  1},
                                             { 74,  97,   1,  1},
                                             {119, 157, 112,  1}};
    grb::Matrix<double> answer(ans, 0.);

    grb::mxm(result,
             grb::complement(mMask),
             grb::Second<double>(),
             grb::ArithmeticSemiring<double>(), mA, mB);
    BOOST_CHECK_EQUAL(result.nvals(), 12);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_Accum_AB_Merge_Mstored_zero)
{
    grb::Matrix<double> mA(mA_dense_3x3, 0.);
    grb::Matrix<double> mB(mB_dense_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{0, 1, 1, 1},
                                                          {0, 0, 1, 1},
                                                          {0, 0, 0, 1}};
    grb::Matrix<double> mMask(mMask_3x4, 0.);
    mMask.setElement(0, 0, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 7);

    grb::Matrix<double> result(mOnes_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   1,   1,  1},
                                             { 74,  97,   1,  1},
                                             {119, 157, 112,  1}};
    grb::Matrix<double> answer(ans, 0.);

    grb::mxm(result,
             grb::complement(mMask),
             grb::Second<double>(),
             grb::ArithmeticSemiring<double>(), mA, mB);
    BOOST_CHECK_EQUAL(result.nvals(), 12);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_Accum_AB_Merge_ABdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    grb::Matrix<double> mat(m, 0.);

    grb::Matrix<double> result(mOnes_4x4, 0.);

    std::vector<std::vector<double> > ans = {{2,  1,  1,  1},
                                             {3,  9,  1,  1},
                                             {2, 10, 22,  1},
                                             {0,  6, 21, 25}};
    grb::Matrix<double> answer(ans, 0.);

    static std::vector<std::vector<double> > mMask_4x4 = {{0, 1, 1, 1},
                                                          {0, 0, 1, 1},
                                                          {0, 0, 0, 1},
                                                          {0, 0, 0, 0}};
    grb::Matrix<double> mMask(mMask_4x4, 0.);

    grb::mxm(result,
             grb::complement(mMask),
             grb::NoAccumulate(),
             grb::ArithmeticSemiring<double>(), mat, mat);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_Accum_ATB_Replace)
{
    grb::Matrix<double> mA(mA_dense_3x3, 0.);
    grb::Matrix<double> mB(mB_dense_3x4, 0.);

    grb::Matrix<double> result(mOnes_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{0, 1, 1, 1},
                                                          {0, 0, 1, 1},
                                                          {0, 0, 0, 1}};
    grb::Matrix<double> mMask(mMask_3x4, 0.);

    std::vector<std::vector<double> > ans = {{112,   0,   0,  0},
                                             { 97, 131,   0,  0},
                                             { 87, 111, 102,  0}};
    grb::Matrix<double> answer(ans, 0.);

    grb::mxm(result,
             grb::complement(mMask),
             grb::Second<double>(),
             grb::ArithmeticSemiring<double>(), transpose(mA), mB,
             REPLACE);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_Accum_ATB_Replace_Bidentity)
{
    grb::Matrix<double> mA(mA_dense_3x3, 0.);
    grb::Matrix<double> mB(mIdentity_3x3, 0.);
    grb::Matrix<double> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{0, 1, 1},
                                                          {0, 0, 1},
                                                          {0, 0, 0}};
    grb::Matrix<double> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans = {{12, 0, 0},
                                             { 7, 5, 0},
                                             { 3, 6, 9}};
    grb::Matrix<double> answer(ans, 0.);

    //auto answer = grb::transpose(mA);

    grb::mxm(result,
             grb::complement(mMask),
             grb::Second<double>(),
             grb::ArithmeticSemiring<double>(), transpose(mA), mB,
             REPLACE);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_Accum_ATB_Merge)
{
    grb::Matrix<double> mA(mA_dense_3x3, 0.);
    grb::Matrix<double> mB(mB_dense_3x4, 0.);

    grb::Matrix<double> result(mOnes_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{0, 1, 1, 1},
                                                          {0, 0, 1, 1},
                                                          {0, 0, 0, 1}};
    grb::Matrix<double> mMask(mMask_3x4, 0.);

    std::vector<std::vector<double> > ans = {{112,   1,   1,  1},
                                             { 97, 131,   1,  1},
                                             { 87, 111, 102,  1}};
    grb::Matrix<double> answer(ans, 0.);

    grb::mxm(result,
             grb::complement(mMask),
             grb::Second<double>(),
             grb::ArithmeticSemiring<double>(), transpose(mA), mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_Accum_ATB_Merge_Bidentity)
{
    grb::Matrix<double> mA(mA_dense_3x3, 0.);
    grb::Matrix<double> mB(mIdentity_3x3, 0.);
    grb::Matrix<double> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{0, 1, 1},
                                                          {0, 0, 1},
                                                          {0, 0, 0}};
    grb::Matrix<double> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans = {{12, 1, 1},
                                             { 7, 5, 1},
                                             { 3, 6, 9}};
    grb::Matrix<double> answer(ans, 0.);

    //auto answer = grb::transpose(mA);

    grb::mxm(result,
             grb::complement(mMask),
             grb::Second<double>(),
             grb::ArithmeticSemiring<double>(), transpose(mA), mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_Accum_ABT_Replace)
{
    grb::Matrix<double> mA(mA_dense_3x3, 0.);

    std::vector<std::vector<double> > B = {{5, 8, 1},
                                           {2, 6, 7},
                                           {3, 4, 5}};
    grb::Matrix<double> mB(B, 0.);

    grb::Matrix<double> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{0, 1, 1},
                                                          {0, 0, 1},
                                                          {0, 0, 0}};
    grb::Matrix<double> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans = {{119,   0,  0},
                                             { 66,  80,  0},
                                             {108, 125, 98}};
    grb::Matrix<double> answer(ans, 0.);

    grb::mxm(result,
             grb::complement(mMask),
             grb::Second<double>(),
             grb::ArithmeticSemiring<double>(), mA, transpose(mB),
             REPLACE);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_Accum_ABT_Merge)
{
    grb::Matrix<double> mA(mA_dense_3x3, 0.);

    std::vector<std::vector<double> > B = {{5, 8, 1},
                                           {2, 6, 7},
                                           {3, 4, 5}};
    grb::Matrix<double> mB(B, 0.);

    grb::Matrix<double> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{0, 1, 1},
                                                          {0, 0, 1},
                                                          {0, 0, 0}};
    grb::Matrix<double> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans = {{119,   1,  1},
                                             { 66,  80,  1},
                                             {108, 125, 98}};
    grb::Matrix<double> answer(ans, 0.);

    grb::mxm(result,
             grb::complement(mMask),
             grb::Second<double>(),
             grb::ArithmeticSemiring<double>(), mA, transpose(mB));

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_Accum_ATBT_Replace)
{
    grb::Matrix<double> mA(mA_dense_3x3, 0.);
    std::vector<std::vector<double> > B =  {{5, 8, 1},
                                            {2, 6, 7},
                                            {3, 4, 5}};
    grb::Matrix<double> mB(B, 0.);

    grb::Matrix<double> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{0, 1, 1},
                                                          {0, 0, 1},
                                                          {0, 0, 0}};
    grb::Matrix<double> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans =  {{99,   0,  0},
                                              {83, 100,  0},
                                              {72, 105, 78}};
    grb::Matrix<double> answer(ans, 0.);

    grb::mxm(result,
             grb::complement(mMask),
             grb::Second<double>(),
             grb::ArithmeticSemiring<double>(), transpose(mA), transpose(mB),
             REPLACE);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_Accum_ATBT_Merge)
{
    grb::Matrix<double> mA(mA_dense_3x3, 0.);
    std::vector<std::vector<double> > B =  {{5, 8, 1},
                                            {2, 6, 7},
                                            {3, 4, 5}};
    grb::Matrix<double> mB(B, 0.);

    grb::Matrix<double> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{0, 1, 1},
                                                          {0, 0, 1},
                                                          {0, 0, 0}};
    grb::Matrix<double> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans =  {{99,   1,  1},
                                              {83, 100,  1},
                                              {72, 105, 78}};
    grb::Matrix<double> answer(ans, 0.);

    grb::mxm(result,
             grb::complement(mMask),
             grb::Second<double>(),
             grb::ArithmeticSemiring<double>(), transpose(mA), transpose(mB));

    BOOST_CHECK_EQUAL(result, answer);
}

BOOST_AUTO_TEST_SUITE_END()
