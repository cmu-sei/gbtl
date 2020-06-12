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

    static std::vector<std::vector<double> > mB_dense_3x4 =
            {{5, 8, 1, 2},
             {6, 7, 3, 0.},
             {4, 5, 9, 1}};

    static std::vector<std::vector<double> > mAnswer_dense =
            {{114, 160, 60,  27},
             {74,  97,  73,  14},
             {119, 157, 112, 23}};

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

}

//****************************************************************************
BOOST_AUTO_TEST_CASE(mxm_test_template_specialization_compilation)
{
    IndexArrayType i_A      = { 0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_A      = { 0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_A = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double> A(3, 3);
    A.build(i_A, j_A, v_A);

    IndexArrayType i_B      = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_B      = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_B = {5, 8, 1, 6, 7, 3, 4, 5, 9};
    Matrix<double> B(3, 3);
    B.build(i_B, j_B, v_B);

    IndexArrayType i_M    = {0, 0, 1, 1, 2, 2};
    IndexArrayType j_M    = {0, 2, 0, 2, 0, 2};
    std::vector<bool> v_M = {true, true, true, true, true, true};
    Matrix<bool> M(3, 3);
    M.build(i_M, j_M, v_M);

    Matrix<double> C(3, 3);
    ArithmeticSemiring<double> sr;
    Plus<double> accum;

    // Make sure all combinations compile

    // A*B
    mxm(C, NoMask(),      NoAccumulate(), sr, A, B);
    mxm(C, NoMask(),      accum,          sr, A, B);
    mxm(C, M,             NoAccumulate(), sr, A, B, REPLACE);
    mxm(C, M,             NoAccumulate(), sr, A, B, MERGE);
    mxm(C, complement(M), NoAccumulate(), sr, A, B, REPLACE);
    mxm(C, complement(M), NoAccumulate(), sr, A, B, MERGE);
    mxm(C, M,             accum,          sr, A, B, REPLACE);
    mxm(C, M,             accum,          sr, A, B, MERGE);
    mxm(C, complement(M), accum,          sr, A, B, REPLACE);
    mxm(C, complement(M), accum,          sr, A, B, MERGE);

    // A*B'
    mxm(C, NoMask(),      NoAccumulate(), sr, A, transpose(B));
    mxm(C, NoMask(),      accum,          sr, A, transpose(B));
    mxm(C, M,             NoAccumulate(), sr, A, transpose(B), REPLACE);
    mxm(C, M,             NoAccumulate(), sr, A, transpose(B), MERGE);
    mxm(C, complement(M), NoAccumulate(), sr, A, transpose(B), REPLACE);
    mxm(C, complement(M), NoAccumulate(), sr, A, transpose(B), MERGE);
    mxm(C, M,             accum,          sr, A, transpose(B), REPLACE);
    mxm(C, M,             accum,          sr, A, transpose(B), MERGE);
    mxm(C, complement(M), accum,          sr, A, transpose(B), REPLACE);
    mxm(C, complement(M), accum,          sr, A, transpose(B), MERGE);

    // A'B
    mxm(C, NoMask(),      NoAccumulate(), sr, transpose(A), B);
    mxm(C, NoMask(),      accum,          sr, transpose(A), B);
    mxm(C, M,             NoAccumulate(), sr, transpose(A), B, REPLACE);
    mxm(C, M,             NoAccumulate(), sr, transpose(A), B, MERGE);
    mxm(C, complement(M), NoAccumulate(), sr, transpose(A), B, REPLACE);
    mxm(C, complement(M), NoAccumulate(), sr, transpose(A), B, MERGE);
    mxm(C, M,             accum,          sr, transpose(A), B, REPLACE);
    mxm(C, M,             accum,          sr, transpose(A), B, MERGE);
    mxm(C, complement(M), accum,          sr, transpose(A), B, REPLACE);
    mxm(C, complement(M), accum,          sr, transpose(A), B, MERGE);

    // A'B'
    mxm(C, NoMask(),      NoAccumulate(), sr, transpose(A), transpose(B));
    mxm(C, NoMask(),      accum,          sr, transpose(A), transpose(B));
    mxm(C, M,             NoAccumulate(), sr, transpose(A), transpose(B), REPLACE);
    mxm(C, M,             NoAccumulate(), sr, transpose(A), transpose(B), MERGE);
    mxm(C, complement(M), NoAccumulate(), sr, transpose(A), transpose(B), REPLACE);
    mxm(C, complement(M), NoAccumulate(), sr, transpose(A), transpose(B), MERGE);
    mxm(C, M,             accum,          sr, transpose(A), transpose(B), REPLACE);
    mxm(C, M,             accum,          sr, transpose(A), transpose(B), MERGE);
    mxm(C, complement(M), accum,          sr, transpose(A), transpose(B), REPLACE);
    mxm(C, complement(M), accum,          sr, transpose(A), transpose(B), MERGE);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(mxm_reg)
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

    std::cout << "Test AxB" << std::endl;
    mxm(result,
        grb::NoMask(), grb::NoAccumulate(),
        grb::ArithmeticSemiring<double>(),
        mA, mB);

    BOOST_CHECK_EQUAL(result, answer);
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_transpose)
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

    std::cout << "Test ATxB" << std::endl;
    mxm(result,
        grb::NoMask(), grb::NoAccumulate(),
        grb::ArithmeticSemiring<double>(),
        transpose(mA), mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_b_transpose)
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

    std::cout << "Test AxBT" << std::endl;
    mxm(result,
        grb::NoMask(), grb::NoAccumulate(),
        grb::ArithmeticSemiring<double>(),
        mA, transpose(mB));

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_and_b_transpose)
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

    std::cout << "Test ATxBT" << std::endl;
    mxm(result,
        grb::NoMask(), grb::NoAccumulate(),
        grb::ArithmeticSemiring<double>(),
        transpose(mA), transpose(mB));

    BOOST_CHECK_EQUAL(result, answer);
}

BOOST_AUTO_TEST_SUITE_END()
