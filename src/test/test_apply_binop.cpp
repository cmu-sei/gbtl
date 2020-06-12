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

#include <functional>
#include <iostream>
#include <vector>

#include <graphblas/graphblas.hpp>

using namespace grb;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE apply_binop_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

//****************************************************************************
// apply standard matrix
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(apply_stdmat_test_bad_dimension)
{
    // | 1 1 - - |
    // | 1 2 2 - |
    // | - 2 3 3 |
    // | - - 3 4 |

    // Build some sparse matrices.
    IndexArrayType i    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v          = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4};
    Matrix<double, DirectedMatrixTag> A(4, 4);
    A.build(i, j, v);

    // bind1st
    {
        Matrix<double, DirectedMatrixTag> C(3, 4);
        BOOST_CHECK_THROW(
            (apply(C,
                   NoMask(),
                   NoAccumulate(),
                   Times<double>(),
                   A,
                   2.0)),
            DimensionException);
    }
    {
        Matrix<double, DirectedMatrixTag> C(4, 3);
        BOOST_CHECK_THROW(
            (apply(C,
                   NoMask(),
                   NoAccumulate(),
                   Times<double>(),
                   2.0,
                   A)),
            DimensionException);
    }

    {
        Matrix<double, DirectedMatrixTag> C(4, 4);
        Matrix<bool> M(3, 4);
        BOOST_CHECK_THROW(
            (apply(C,
                   M,
                   NoAccumulate(),
                   Times<double>(),
                   2.0,
                   A)),
            DimensionException);
    }
    {
        Matrix<double, DirectedMatrixTag> C(4, 4);
        Matrix<bool> M(4, 3);
        BOOST_CHECK_THROW(
            (apply(C,
                   M,
                   NoAccumulate(),
                   Times<double>(),
                   2.0,
                   A)),
            DimensionException);
    }

    // bind 2nd
    {
        Matrix<double, DirectedMatrixTag> C(3, 4);
        BOOST_CHECK_THROW(
            (apply(C,
                   NoMask(),
                   NoAccumulate(),
                   Times<double>(),
                   A,
                   2.0)),
            DimensionException);
    }
    {
        Matrix<double, DirectedMatrixTag> C(4, 3);
        BOOST_CHECK_THROW(
            (apply(C,
                   NoMask(),
                   NoAccumulate(),
                   Times<double>(),
                   A,
                   2.0)),
            DimensionException);
    }

    {
        Matrix<double, DirectedMatrixTag> C(4, 4);
        Matrix<bool> M(3, 4);
        BOOST_CHECK_THROW(
            (apply(C,
                   M,
                   NoAccumulate(),
                   Times<double>(),
                   A,
                   2.0)),
            DimensionException);
    }
    {
        Matrix<double, DirectedMatrixTag> C(4, 4);
        Matrix<bool> M(4, 3);
        BOOST_CHECK_THROW(
            (apply(C,
                   M,
                   NoAccumulate(),
                   Times<double>(),
                   A,
                   2.0)),
            DimensionException);
    }

}

//****************************************************************************

BOOST_AUTO_TEST_CASE(sparse_apply_matrix_nomask_noaccum)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 0}};
    grb::Matrix<double, grb::DirectedMatrixTag> mA(matA, 0);

    std::vector<std::vector<double>> matC = {{1, 2, 3},
                                             {4, 5, 6},
                                             {7, 8, 0}};
    grb::Matrix<double, grb::DirectedMatrixTag> mC(matC, 0);

    std::vector<std::vector<double>> matAnswer = {{-8, -1, -6},
                                                  {-3, -5, -7},
                                                  {-4, -9,  0}};

    grb::Matrix<double, grb::DirectedMatrixTag> answer(matAnswer, 0);

    grb::apply(mC,
               grb::NoMask(),
               grb::NoAccumulate(),
               grb::Times<double>(),
               -1.0,
               mA);

    BOOST_CHECK_EQUAL(mC, answer);

    grb::apply(mC,
               grb::NoMask(),
               grb::NoAccumulate(),
               grb::Times<double>(),
               mA,
               -1.0);

    BOOST_CHECK_EQUAL(mC, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_apply_matrix_mask_merge_noaccum)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 0}};
    grb::Matrix<double, grb::DirectedMatrixTag> mA(matA, 0);

    std::vector<std::vector<double>> matC = {{1, 2, 3},
                                             {4, 5, 6},
                                             {7, 8, 0}};

    std::vector<std::vector<bool>> matMask = {{true, false, false},
                                              {true, true,  false,},
                                              {true, true,  true}};
    grb::Matrix<bool, grb::DirectedMatrixTag> mask(matMask, false);

    std::vector<std::vector<double>> matAnswer = {{-8, 2,  3},
                                                  {-3, -5, 6},
                                                  {-4, -9, 0}};
    grb::Matrix<double, grb::DirectedMatrixTag> answer(matAnswer, 0);

    {
        grb::Matrix<double, grb::DirectedMatrixTag> mC(matC, 0);

        grb::apply(mC,
                   mask,
                   grb::NoAccumulate(),
                   grb::Times<double>(),
                   -1.0,
                   mA);

        BOOST_CHECK_EQUAL(mC, answer);
    }
    {
        grb::Matrix<double, grb::DirectedMatrixTag> mC(matC, 0);

        grb::apply(mC,
                   mask,
                   grb::NoAccumulate(),
                   grb::Times<double>(),
                   mA,
                   -1.0);

        BOOST_CHECK_EQUAL(mC, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_apply_matrix_mask_replace_noaccum)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 0}};
    grb::Matrix<double, grb::DirectedMatrixTag> mA(matA, 0);

    std::vector<std::vector<double>> matC = {{1, 2, 3},
                                             {4, 5, 6},
                                             {7, 8, 0}};


    std::vector<std::vector<bool>> matMask = {{true, false, false},
                                              {true, true,  false,},
                                              {true, true,  true}};
    grb::Matrix<bool, grb::DirectedMatrixTag> mask(matMask, false);

    std::vector<std::vector<double>> matAnswer = {{-8, 0,  0},
                                                  {-3, -5, 0},
                                                  {-4, -9, 0}};
    grb::Matrix<double, grb::DirectedMatrixTag> answer(matAnswer, 0);

    {
        grb::Matrix<double, grb::DirectedMatrixTag> mC(matC, 0);
        grb::apply(mC,
                   mask,
                   grb::NoAccumulate(),
                   grb::Times<double>(),
                   -1.0,
                   mA,
                   grb::REPLACE);

        BOOST_CHECK_EQUAL(mC, answer);
    }
    {
        grb::Matrix<double, grb::DirectedMatrixTag> mC(matC, 0);
        grb::apply(mC,
                   mask,
                   grb::NoAccumulate(),
                   grb::Times<double>(),
                   mA,
                   -1.0,
                   grb::REPLACE);

        BOOST_CHECK_EQUAL(mC, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(apply_stdmat_test_noaccum)
{
    // Build input matrix
    // | 1 1 - - |
    // | 1 2 2 - |
    // | - 2 3 3 |

    std::vector<std::vector<double>> Atmp = {{1, 1, 0, 0},
                                             {1, 2, 2, 0},
                                             {0, 2, 3, 3}};
    Matrix<double, DirectedMatrixTag> A(Atmp, 0.0);

    std::vector<std::vector<double>> Ctmp = {{9, 9, 9, 9},
                                             {9, 9, 9, 9},
                                             {9, 9, 9, 9}};

    std::vector<std::vector<uint8_t>> mask = {{1, 1, 1, 1},
                                              {0, 1, 1, 1},
                                              {0, 0, 1, 1}};
    Matrix<uint8_t> M(mask, 0);

    {
        std::vector<std::vector<double>> ans = {{1.5, 1.5,   0,   0},
                                                {1.5, 2.5, 2.5,   0},
                                                {  0, 2.5, 3.5, 3.5}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(3, 4);
        apply(C,
              NoMask(),
              NoAccumulate(),
              Plus<double>(),
              0.5,
              A);
        BOOST_CHECK_EQUAL(C, answer);

        Matrix<double, DirectedMatrixTag> C2(3, 4);
        apply(C2,
              NoMask(),
              NoAccumulate(),
              Plus<double>(),
              A,
              0.5);
        BOOST_CHECK_EQUAL(C2, answer);
    }

    // Mask, replace
    {
        std::vector<std::vector<double>> ans = {{1.5, 1.5,   0,   0},
                                                {  0, 2.5, 2.5,   0},
                                                {  0,   0, 3.5, 3.5}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(Ctmp, 0);
        apply(C,
              M,
              NoAccumulate(),
              Plus<double>(),
              0.5,
              A,
              REPLACE);
        BOOST_CHECK_EQUAL(C, answer);

        Matrix<double, DirectedMatrixTag> C2(Ctmp, 0);
        apply(C2,
              M,
              NoAccumulate(),
              Plus<double>(),
              A,
              0.5,
              REPLACE);
        BOOST_CHECK_EQUAL(C2, answer);
    }

    // Mask, merge
    {
        std::vector<std::vector<double>> ans = {{1.5, 1.5,   0,   0},
                                                {  9, 2.5, 2.5,   0},
                                                {  9,   9, 3.5, 3.5}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(Ctmp, 0);
        apply(C,
              M,
              NoAccumulate(),
              Plus<double>(),
              0.5,
              A,
              MERGE);
        BOOST_CHECK_EQUAL(C, answer);

        Matrix<double, DirectedMatrixTag> C2(Ctmp, 0);
        apply(C2,
              M,
              NoAccumulate(),
              Plus<double>(),
              A,
              0.5,
              MERGE);
        BOOST_CHECK_EQUAL(C2, answer);
    }

    // complement(Mask), replace
    {
        std::vector<std::vector<double>> ans = {{  0,   0,   0,   0},
                                                {1.5,   0,   0,   0},
                                                {  0, 2.5,   0,   0}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(Ctmp, 0);
        apply(C,
              complement(M),
              NoAccumulate(),
              Plus<double>(),
              0.5,
              A,
              REPLACE);
        BOOST_CHECK_EQUAL(C, answer);

        Matrix<double, DirectedMatrixTag> C2(Ctmp, 0);
        apply(C2,
              complement(M),
              NoAccumulate(),
              Plus<double>(),
              A,
              0.5,
              REPLACE);
        BOOST_CHECK_EQUAL(C2, answer);
    }
    // complement(Mask), merge
    {
        std::vector<std::vector<double>> ans = {{  9,   9,   9,   9},
                                                {1.5,   9,   9,   9},
                                                {  0, 2.5,   9,   9}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(Ctmp, 0);
        apply(C,
              complement(M),
              NoAccumulate(),
              Plus<double>(),
              0.5,
              A,
              MERGE);
        BOOST_CHECK_EQUAL(C, answer);

        Matrix<double, DirectedMatrixTag> C2(Ctmp, 0);
        apply(C2,
              complement(M),
              NoAccumulate(),
              Plus<double>(),
              A,
              0.5,
              MERGE);
        BOOST_CHECK_EQUAL(C2, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_apply_matrix_nomask_plus_accum)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 0}};
    grb::Matrix<double, grb::DirectedMatrixTag> mA(matA, 0);

    std::vector<std::vector<double>> matC = {{1, 2, 3},
                                             {4, 5, 6},
                                             {7, 8, 0}};

    std::vector<std::vector<double>> matAnswer = {{-7, 1, -3},
                                                  { 1, 0, -1},
                                                  { 3,-1,  1337}};
    // NOTE: The accum results in an explicit zero
    grb::Matrix<double, grb::DirectedMatrixTag> answer(matAnswer, 1337);

    {
        grb::Matrix<double, grb::DirectedMatrixTag> mC(matC, 0);
        grb::apply(mC,
                   grb::NoMask(),
                   grb::Plus<double>(),
                   grb::Times<double>(),
                   -1.0,
                   mA);

        BOOST_CHECK_EQUAL(mC, answer);
    }
    {
        grb::Matrix<double, grb::DirectedMatrixTag> mC(matC, 0);
        grb::apply(mC,
                   grb::NoMask(),
                   grb::Plus<double>(),
                   grb::Times<double>(),
                   mA,
                   -1.0);

        BOOST_CHECK_EQUAL(mC, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(apply_stdmat_test_plus_accum)
{
    // Build input matrix
    // | 1 1 - - |
    // | 1 2 2 - |
    // | - 2 3 3 |

    std::vector<std::vector<double>> Atmp = {{1, 1, 0, 0},
                                             {1, 2, 2, 0},
                                             {0, 2, 3, 3}};
    Matrix<double, DirectedMatrixTag> A(Atmp, 0.0);

    std::vector<std::vector<double>> Ctmp = {{9, 9, 9, 0},
                                             {9, 9, 0, 0},
                                             {9, 0, 0, 0}};

    std::vector<std::vector<uint8_t>> mask = {{1, 1, 1, 1},
                                              {0, 1, 1, 1},
                                              {0, 0, 1, 1}};
    Matrix<uint8_t> M(mask, 0);

    {
        std::vector<std::vector<double>> ans = {{9.5,  9.5,    9,   0},
                                                {9.5, 10.0,  1.0,   0},
                                                {  9,  1.0,  1.5, 1.5}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(Ctmp, 0.);
        apply(C,
              NoMask(),
              Plus<double>(),
              Times<double>(),
              0.5,
              A);
        BOOST_CHECK_EQUAL(C, answer);

        Matrix<double, DirectedMatrixTag> C2(Ctmp, 0.);
        apply(C2,
              NoMask(),
              Plus<double>(),
              Times<double>(),
              A,
              0.5);
        BOOST_CHECK_EQUAL(C2, answer);
    }

    // Mask, replace
    {
        std::vector<std::vector<double>> ans = {{9.5,  9.5,    9,   0},
                                                {  0, 10.0,  1.0,   0},
                                                {  0,    0,  1.5, 1.5}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(Ctmp, 0);
        apply(C,
              M,
              Plus<double>(),
              Times<double>(),
              0.5,
              A,
              REPLACE);
        BOOST_CHECK_EQUAL(C, answer);

        Matrix<double, DirectedMatrixTag> C2(Ctmp, 0);
        apply(C2,
              M,
              Plus<double>(),
              Times<double>(),
              A,
              0.5,
              REPLACE);
        BOOST_CHECK_EQUAL(C2, answer);
    }
    // Mask, merge
    {
        std::vector<std::vector<double>> ans = {{9.5,  9.5,    9,   0},
                                                {  9, 10.0,  1.0,   0},
                                                {  9,    0,  1.5, 1.5}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(Ctmp, 0);
        apply(C,
              M,
              Plus<double>(),
              Times<double>(),
              0.5,
              A,
              MERGE);
        BOOST_CHECK_EQUAL(C, answer);

        Matrix<double, DirectedMatrixTag> C2(Ctmp, 0);
        apply(C2,
              M,
              Plus<double>(),
              Times<double>(),
              A,
              0.5,
              MERGE);
        BOOST_CHECK_EQUAL(C2, answer);
    }

    // complement(Mask), replace
    {
        std::vector<std::vector<double>> ans = {{  0,    0,    0,   0},
                                                {9.5,    0,    0,   0},
                                                {  9,  1.0,    0,   0}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(Ctmp, 0);
        apply(C,
              complement(M),
              Plus<double>(),
              Times<double>(),
              0.5,
              A,
              REPLACE);
        BOOST_CHECK_EQUAL(C, answer);

        Matrix<double, DirectedMatrixTag> C2(Ctmp, 0);
        apply(C2,
              complement(M),
              Plus<double>(),
              Times<double>(),
              A,
              0.5,
              REPLACE);
        BOOST_CHECK_EQUAL(C2, answer);
    }
    // complement(Mask), merge
    {
        std::vector<std::vector<double>> ans = {{  9,    9,    9,   0},
                                                {9.5,    9,    0,   0},
                                                {  9,  1.0,    0,   0}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(Ctmp, 0);
        apply(C,
              complement(M),
              Plus<double>(),
              Times<double>(),
              0.5,
              A,
              MERGE);
        BOOST_CHECK_EQUAL(C, answer);

        Matrix<double, DirectedMatrixTag> C2(Ctmp, 0);
        apply(C2,
              complement(M),
              Plus<double>(),
              Times<double>(),
              A,
              0.5,
              MERGE);
        BOOST_CHECK_EQUAL(C2, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(apply_stdmat_test_second_accum)
{
    // Build input matrix
    // | 1 1 - - |
    // | 1 2 2 - |
    // | - 2 3 3 |
    // | - - 3 4 |

    IndexArrayType i = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4};
    Matrix<double, DirectedMatrixTag> A(4, 4);
    A.build(i, j, v);

    std::vector<double> v_answer = {0.5, 0.5, 0.5, 1.0, 1.0,
                                    1.0, 1.5, 1.5, 1.5, 2.0};

    Matrix<double, DirectedMatrixTag> answer(4, 4);
    answer.build(i, j, v_answer);

    Matrix<double, DirectedMatrixTag> C(4, 4);
    apply(C,
          NoMask(),
          Second<double>(),
          Times<double>(),
          0.5,
          A);
    BOOST_CHECK_EQUAL(C, answer);

    Matrix<double, DirectedMatrixTag> C2(4, 4);
    apply(C2,
          NoMask(),
          Second<double>(),
          Times<double>(),
          A,
          0.5);
    BOOST_CHECK_EQUAL(C2, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(apply_stdmat_test_noaccum_transpose)
{
    // Build input matrix
    // | 1 1 - - |T
    // | 1 2 2 - |
    // | - 2 3 3 |

    std::vector<std::vector<double>> Atmp = {{1, 1, 0},
                                             {1, 2, 2},
                                             {0, 2, 3},
                                             {0, 0, 3}};
    Matrix<double, DirectedMatrixTag> A(Atmp, 0.0);

    std::vector<std::vector<double>> Ctmp = {{9, 9, 9, 9},
                                             {9, 9, 9, 9},
                                             {9, 9, 9, 9}};

    std::vector<std::vector<uint8_t>> mask = {{1, 1, 1, 1},
                                              {0, 1, 1, 1},
                                              {0, 0, 1, 1}};
    Matrix<uint8_t> M(mask, 0);

    {
        std::vector<std::vector<double>> ans = {{0.5, 0.5,   0,   0},
                                                {0.5, 1.0, 1.0,   0},
                                                {0,   1.0, 1.5, 1.5}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(3, 4);
        apply(C,
              NoMask(),
              NoAccumulate(),
              Times<double>(),
              transpose(A),
              0.5);
        BOOST_CHECK_EQUAL(C, answer);

        Matrix<double, DirectedMatrixTag> C2(3, 4);
        apply(C2,
              NoMask(),
              NoAccumulate(),
              Times<double>(),
              transpose(A),
              0.5);
        BOOST_CHECK_EQUAL(C2, answer);
    }
}

//****************************************************************************
// Apply standard vector
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(apply_stdvec_test_bad_dimension)
{
    // | - 2 3 3 |

    // Build some sparse vectors.
    std::vector<double> v = {0, 2, 3, 3};
    Vector<double> u(v, 0.0);

    {
        Vector<double> w(3);
        BOOST_CHECK_THROW(
            (apply(w,
                   NoMask(),
                   NoAccumulate(),
                   Times<double>(),
                   0.5,
                   u)),
            DimensionException);

        BOOST_CHECK_THROW(
            (apply(w,
                   NoMask(),
                   NoAccumulate(),
                   Times<double>(),
                   u,
                   0.5)),
            DimensionException);
    }

    {
        Vector<double> w(4);
        Vector<bool> m(3);
        BOOST_CHECK_THROW(
            (apply(w,
                   m,
                   NoAccumulate(),
                   Times<double>(),
                   0.5,
                   u)),
            DimensionException);
        BOOST_CHECK_THROW(
            (apply(w,
                   m,
                   NoAccumulate(),
                   Times<double>(),
                   u,
                   0.5)),
            DimensionException);
    }
}

//****************************************************************************

BOOST_AUTO_TEST_CASE(sparse_apply_vector_nomask_noaccum)
{
    std::vector<double> vecA = {8, 0, 6, 0};
    grb::Vector<double> vA(vecA, 0);

    std::vector<double> vecC = {1, 2, 0, 0};

    std::vector<double> vecAnswer = {-8, 0, -6,  0};
    grb::Vector<double> answer(vecAnswer, 0);

    {
        grb::Vector<double> vC(vecC, 0);
        grb::apply(vC,
                   grb::NoMask(),
                   grb::NoAccumulate(),
                   grb::Times<double>(),
                   -1.0,
                   vA);

        BOOST_CHECK_EQUAL(vC, answer);
    }
    {
        grb::Vector<double> vC(vecC, 0);
        grb::apply(vC,
                   grb::NoMask(),
                   grb::NoAccumulate(),
                   grb::Times<double>(),
                   vA,
                   -1.0);

        BOOST_CHECK_EQUAL(vC, answer);
    }
}

//****************************************************************************

BOOST_AUTO_TEST_CASE(sparse_apply_vector_nomask_noaccum_bool)
{
    std::vector<bool> vecA = {true, false, true, false};
    grb::Vector<bool> vA(vecA, false);

    std::vector<bool> vecC = {true, true, false, false};

    std::vector<bool> vecAnswer = {true, false, true, false};
    grb::Vector<bool> answer(vecAnswer, false);

    {
        grb::Vector<bool> vC(vecC, false);
        grb::apply(vC,
                   grb::NoMask(),
                   grb::NoAccumulate(),
                   grb::LogicalXor<bool>(),
                   false,
                   vA);

        BOOST_CHECK_EQUAL(vC, answer);
    }
    {
        grb::Vector<bool> vC(vecC, false);
        grb::apply(vC,
                   grb::NoMask(),
                   grb::NoAccumulate(),
                   grb::LogicalXor<bool>(),
                   vA,
                   false);

        BOOST_CHECK_EQUAL(vC, answer);
    }
}

//****************************************************************************

BOOST_AUTO_TEST_CASE(sparse_apply_vector_nomask_noaccum_int)
{
    std::vector<int> vecA = {1, 0, 1, 0};
    grb::Vector<int> vA(vecA, 0);

    std::vector<int> vecC = {1, 1, 0, 0};

    std::vector<int> vecAnswer = {2, 0, 2, 0};
    grb::Vector<int> answer(vecAnswer, 0);

    {
        grb::Vector<int> vC(vecC, 0);
        grb::apply(vC,
                   grb::NoMask(),
                   grb::NoAccumulate(),
                   grb::Plus<int>(),
                   1,
                   vA);

        BOOST_CHECK_EQUAL(vC, answer);
    }
    {
        grb::Vector<int> vC(vecC, 0);
        grb::apply(vC,
                   grb::NoMask(),
                   grb::NoAccumulate(),
                   grb::Plus<int>(),
                   vA,
                   1);

        BOOST_CHECK_EQUAL(vC, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_apply_vector_mask_merge_noaccum)
{
    std::vector<double> vecA = {8, 8, 0, 0, 6, 6, 0, 0};
    grb::Vector<double> vA(vecA, 0);

    std::vector<double> vecC = {1, 1, 2, 2, 0, 0, 0, 0};

    std::vector<bool> vecMask = {true, false, true, false, true, false, true, false};
    grb::Vector<bool> mask(vecMask, false);

    std::vector<double> vecAnswer = {-8, 1, 0, 2, -6, 0, 0, 0};
    grb::Vector<double> answer(vecAnswer, 0);

    {
        grb::Vector<double> vC(vecC, 0);
        grb::apply(vC,
                   mask,
                   grb::NoAccumulate(),
                   grb::Times<double>(),
                   -1.0,
                   vA);

        BOOST_CHECK_EQUAL(vC, answer);
    }
    {
        grb::Vector<double> vC(vecC, 0);
        grb::apply(vC,
                   mask,
                   grb::NoAccumulate(),
                   grb::Times<double>(),
                   vA,
                   -1.0);

        BOOST_CHECK_EQUAL(vC, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_apply_vector_mask_replace_noaccum)
{
    std::vector<double> vecA = {8, 8, 0, 0, 6, 6, 0, 0};
    grb::Vector<double> vA(vecA, 0);

    std::vector<double> vecC = {1, 1, 2, 2, 0, 0, 0, 0};

    std::vector<bool> vecMask = {true, false, true, false, true, false, true, false};
    grb::Vector<bool> mask(vecMask, false);

    std::vector<double> vecAnswer = {-8, 0, 0, 0, -6, 0, 0, 0};
    grb::Vector<double> answer(vecAnswer, 0);

    {
        grb::Vector<double> vC(vecC, 0);
        grb::apply(vC,
                   mask,
                   grb::NoAccumulate(),
                   grb::Times<double>(),
                   -1.0,
                   vA,
                   REPLACE);

        BOOST_CHECK_EQUAL(vC, answer);
    }

    {
        grb::Vector<double> vC(vecC, 0);
        grb::apply(vC,
                   mask,
                   grb::NoAccumulate(),
                   grb::Times<double>(),
                   vA,
                   -1.0,
                   REPLACE);

        BOOST_CHECK_EQUAL(vC, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(apply_stdvec_test_noaccum)
{
    // Build input vector
    // | 1 2 - - |'

    std::vector<double> utmp  = {4, 5, 2, 3, 0, 0};
    Vector<double> u(utmp, 0.0);

    std::vector<double> wtmp  = {9, 0, 9, 0, 9, 0};
    std::vector<uint8_t> mask = {0, 0, 1, 1, 1, 1};
    Vector<uint8_t> m(mask, 0);

    {
        std::vector<double> ans = {-4., -5., -2., -3.,   0,   0};
        Vector<double> answer(ans, 0.);

        Vector<double> w(6);
        apply(w,
              NoMask(),
              NoAccumulate(),
              Times<double>(),
              -1.,
              u);
        BOOST_CHECK_EQUAL(w, answer);

        Vector<double> w2(6);
        apply(w2,
              NoMask(),
              NoAccumulate(),
              Times<double>(),
              u,
              -1.);
        BOOST_CHECK_EQUAL(w2, answer);
    }

    // Mask, replace
    {
        std::vector<double> ans = {0, 0, -2., -3.,   0,   0};
        Vector<double> answer(ans, 0.);

        Vector<double> w(wtmp, 0);
        apply(w,
              m,
              NoAccumulate(),
              Times<double>(),
              -1.,
              u,
              REPLACE);
        BOOST_CHECK_EQUAL(w, answer);

        Vector<double> w2(wtmp, 0);
        apply(w2,
              m,
              NoAccumulate(),
              Times<double>(),
              u,
              -1.,
              REPLACE);
        BOOST_CHECK_EQUAL(w2, answer);
    }
    // Mask, merge
    {
        std::vector<double> ans = {9, 0, -2., -3., 0, 0};
        Vector<double> answer(ans, 0.);

        Vector<double> w(wtmp, 0);
        apply(w,
              m,
              NoAccumulate(),
              Times<double>(),
              -1.,
              u,
              MERGE);
        BOOST_CHECK_EQUAL(w, answer);

        Vector<double> w2(wtmp, 0);
        apply(w2,
              m,
              NoAccumulate(),
              Times<double>(),
              u,
              -1.,
              MERGE);
        BOOST_CHECK_EQUAL(w2, answer);
    }

    // complement(Mask), replace
    {
        std::vector<double> ans = {-4., -5., 0, 0,   0,   0};
        Vector<double> answer(ans, 0.);

        Vector<double> w(wtmp, 0);
        apply(w,
              complement(m),
              NoAccumulate(),
              Times<double>(),
              -1.,
              u,
              REPLACE);
        BOOST_CHECK_EQUAL(w, answer);

        Vector<double> w2(wtmp, 0);
        apply(w2,
              complement(m),
              NoAccumulate(),
              Times<double>(),
              u,
              -1.,
              REPLACE);
        BOOST_CHECK_EQUAL(w2, answer);
    }
    // complement(Mask), merge
    {
        std::vector<double> ans = {-4., -5., 9, 0, 9,  0};
        Vector<double> answer(ans, 0.);

        Vector<double> w(wtmp, 0);
        apply(w,
              complement(m),
              NoAccumulate(),
              Times<double>(),
              -1.,
              u,
              MERGE);
        BOOST_CHECK_EQUAL(w, answer);

        Vector<double> w2(wtmp, 0);
        apply(w2,
              complement(m),
              NoAccumulate(),
              Times<double>(),
              u,
              -1.,
              MERGE);
        BOOST_CHECK_EQUAL(w2, answer);
    }
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_apply_vector_accum)
{
    std::vector<double> vecA = {8, 0, 6, 0};
    grb::Vector<double> vA(vecA, 0);

    std::vector<double> vecC = {8, 2, 0, 0};

    // NOTE: The accum results in an explicit zero
    std::vector<double> vecAnswer = {0, 2, -6,  666};
    grb::Vector<double> answer(vecAnswer, 666);

    {
        grb::Vector<double> vC(vecC, 0);
        grb::apply(vC,
                   grb::NoMask(),
                   grb::Plus<double>(),
                   grb::Times<double>(),
                   -1.,
                   vA);

        BOOST_CHECK_EQUAL(vC, answer);
    }

    {
        grb::Vector<double> vC(vecC, 0);
        grb::apply(vC,
                   grb::NoMask(),
                   grb::Plus<double>(),
                   grb::Times<double>(),
                   vA,
                   -1.);

        BOOST_CHECK_EQUAL(vC, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(apply_stdvec_test_plus_accum)
{
    // Build input vector
    // | 1 2 - - |'

    std::vector<double> utmp  = {4, 5, 2, 3, 0, 0};
    Vector<double> u(utmp, 0.0);

    std::vector<double> wtmp  = {9, 0, 9, 0, 9, 0};
    std::vector<uint8_t> mask = {0, 0, 1, 1, 1, 1};
    Vector<uint8_t> m(mask, 0);

    {
        std::vector<double> ans = {5., -5., 7., -3.,  9.,   0};
        Vector<double> answer(ans, 0.);

        Vector<double> w(wtmp, 0);
        apply(w,
              NoMask(),
              Plus<double>(),
              Times<double>(),
              -1.,
              u);
        BOOST_CHECK_EQUAL(w, answer);

        Vector<double> w2(wtmp, 0);
        apply(w2,
              NoMask(),
              Plus<double>(),
              Times<double>(),
              u,
              -1.);
        BOOST_CHECK_EQUAL(w2, answer);
    }

    // Mask, replace
    {
        std::vector<double> ans = {0, 0, 7., -3.,  9.,   0};
        Vector<double> answer(ans, 0.);

        Vector<double> w(wtmp, 0);
        apply(w,
              m,
              Plus<double>(),
              Times<double>(),
              -1.,
              u,
              REPLACE);
        BOOST_CHECK_EQUAL(w, answer);

        Vector<double> w2(wtmp, 0);
        apply(w2,
              m,
              Plus<double>(),
              Times<double>(),
              u,
              -1.,
              REPLACE);
        BOOST_CHECK_EQUAL(w2, answer);
    }
    // Mask, merge
    {
        std::vector<double> ans = {9, 0, 7., -3., 9, 0};
        Vector<double> answer(ans, 0.);

        Vector<double> w2(wtmp, 0);
        apply(w2,
              m,
              Plus<double>(),
              Times<double>(),
              -1.,
              u,
              MERGE);
        BOOST_CHECK_EQUAL(w2, answer);

        Vector<double> w(wtmp, 0);
        apply(w,
              m,
              Plus<double>(),
              Times<double>(),
              u,
              -1.,
              MERGE);
        BOOST_CHECK_EQUAL(w, answer);
    }

    // complement(Mask), replace
    {
        std::vector<double> ans = {5., -5., 0, 0,   0,   0};
        Vector<double> answer(ans, 0.);

        Vector<double> w(wtmp, 0);
        apply(w,
              complement(m),
              Plus<double>(),
              Times<double>(),
              -1.,
              u,
              REPLACE);
        BOOST_CHECK_EQUAL(w, answer);

        Vector<double> w2(wtmp, 0);
        apply(w2,
              complement(m),
              Plus<double>(),
              Times<double>(),
              u,
              -1.,
              REPLACE);
        BOOST_CHECK_EQUAL(w2, answer);
    }
    // complement(Mask), merge
    {
        std::vector<double> ans = {5., -5., 9, 0, 9,  0};
        Vector<double> answer(ans, 0.);

        Vector<double> w(wtmp, 0);
        apply(w,
              complement(m),
              Plus<double>(),
              Times<double>(),
              -1.,
              u,
              MERGE);
        BOOST_CHECK_EQUAL(w, answer);

        Vector<double> w2(wtmp, 0);
        apply(w2,
              complement(m),
              Plus<double>(),
              Times<double>(),
              u,
              -1.,
              MERGE);
        BOOST_CHECK_EQUAL(w2, answer);
    }
}


BOOST_AUTO_TEST_SUITE_END()
