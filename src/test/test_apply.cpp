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

using namespace GraphBLAS;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE apply_test_suite

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

    {
        Matrix<double, DirectedMatrixTag> C(3, 4);
        BOOST_CHECK_THROW(
            (apply(C,
                   NoMask(),
                   NoAccumulate(),
                   MultiplicativeInverse<double>(),
                   A)),
            DimensionException);
    }
    {
        Matrix<double, DirectedMatrixTag> C(4, 3);
        BOOST_CHECK_THROW(
            (apply(C,
                   NoMask(),
                   NoAccumulate(),
                   MultiplicativeInverse<double>(),
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
                   MultiplicativeInverse<double>(),
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
                   MultiplicativeInverse<double>(),
                   A)),
            DimensionException);
    }
}

//****************************************************************************

BOOST_AUTO_TEST_CASE(sparse_apply_matrix_nomask_noaccum)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(matA, 0);

    std::vector<std::vector<double>> matC = {{1, 2, 3},
                                             {4, 5, 6},
                                             {7, 8, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mC(matC, 0);

    std::vector<std::vector<double>> matAnswer = {{-8, -1, -6},
                                                  {-3, -5, -7},
                                                  {-4, -9,  0}};

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(matAnswer, 0);

    GraphBLAS::apply(mC,
                     GraphBLAS::NoMask(),
                     GraphBLAS::NoAccumulate(),
                     GraphBLAS::AdditiveInverse<double>(),
                     mA);

    BOOST_CHECK_EQUAL(mC, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_apply_matrix_mask_merge_noaccum)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(matA, 0);

    std::vector<std::vector<double>> matC = {{1, 2, 3},
                                             {4, 5, 6},
                                             {7, 8, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mC(matC, 0);


    std::vector<std::vector<bool>> matMask = {{true, false, false},
                                              {true, true,  false,},
                                              {true, true,  true}};
    GraphBLAS::Matrix<bool, GraphBLAS::DirectedMatrixTag> mask(matMask, false);

    std::vector<std::vector<double>> matAnswer = {{-8, 2,  3},
                                                  {-3, -5, 6},
                                                  {-4, -9, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(matAnswer, 0);

    GraphBLAS::apply(mC,
                     mask,
                     GraphBLAS::NoAccumulate(),
                     GraphBLAS::AdditiveInverse<double>(),
                     mA);

    BOOST_CHECK_EQUAL(mC, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_apply_matrix_mask_replace_noaccum)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(matA, 0);

    std::vector<std::vector<double>> matC = {{1, 2, 3},
                                             {4, 5, 6},
                                             {7, 8, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mC(matC, 0);


    std::vector<std::vector<bool>> matMask = {{true, false, false},
                                              {true, true,  false,},
                                              {true, true,  true}};
    GraphBLAS::Matrix<bool, GraphBLAS::DirectedMatrixTag> mask(matMask, false);

    std::vector<std::vector<double>> matAnswer = {{-8, 0,  0},
                                                  {-3, -5, 0},
                                                  {-4, -9, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(matAnswer, 0);

    GraphBLAS::apply(mC,
                     mask,
                     GraphBLAS::NoAccumulate(),
                     GraphBLAS::AdditiveInverse<double>(),
                     mA,
                     REPLACE);

    BOOST_CHECK_EQUAL(mC, answer);
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
        std::vector<std::vector<double>> ans = {{1, 1,    0,    0},
                                                {1, 1./2, 1./2, 0},
                                                {0, 1./2, 1./3, 1./3}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(3, 4);
        apply(C,
              NoMask(),
              NoAccumulate(),
              MultiplicativeInverse<double>(),
              A);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // Mask, replace
    {
        std::vector<std::vector<double>> ans = {{1, 1,    0,    0},
                                                {0, 1./2, 1./2, 0},
                                                {0, 0,    1./3, 1./3}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(Ctmp, 0);
        apply(C,
              M,
              NoAccumulate(),
              MultiplicativeInverse<double>(),
              A,
              REPLACE);
        BOOST_CHECK_EQUAL(C, answer);
    }
    // Mask, merge
    {
        std::vector<std::vector<double>> ans = {{1, 1,    0,    0},
                                                {9, 1./2, 1./2, 0},
                                                {9, 9,    1./3, 1./3}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(Ctmp, 0);
        apply(C,
              M,
              NoAccumulate(),
              MultiplicativeInverse<double>(),
              A,
              MERGE);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // complement(Mask), replace
    {
        std::vector<std::vector<double>> ans = {{0, 0,    0, 0},
                                                {1, 0,    0, 0},
                                                {0, 1./2, 0, 0}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(Ctmp, 0);
        apply(C,
              complement(M),
              NoAccumulate(),
              MultiplicativeInverse<double>(),
              A,
              REPLACE);
        BOOST_CHECK_EQUAL(C, answer);
    }
    // complement(Mask), merge
    {
        std::vector<std::vector<double>> ans = {{9, 9,    9, 9},
                                                {1, 9,    9, 9},
                                                {0, 1./2, 9, 9}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(Ctmp, 0);
        apply(C,
              complement(M),
              NoAccumulate(),
              MultiplicativeInverse<double>(),
              A,
              MERGE);
        BOOST_CHECK_EQUAL(C, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_apply_matrix_nomask_plus_accum)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(matA, 0);

    std::vector<std::vector<double>> matC = {{1, 2, 3},
                                             {4, 5, 6},
                                             {7, 8, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mC(matC, 0);

    std::vector<std::vector<double>> matAnswer = {{-7, 1, -3},
                                                  { 1, 0, -1},
                                                  { 3,-1,  1337}};
    // NOTE: The accum results in an explicit zero
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(matAnswer, 1337);

    GraphBLAS::apply(mC,
                     GraphBLAS::NoMask(),
                     GraphBLAS::Plus<double>(),
                     GraphBLAS::AdditiveInverse<double>(),
                     mA);

    BOOST_CHECK_EQUAL(mC, answer);
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
        std::vector<std::vector<double>> ans = {{10, 10,    9,    0},
                                                {10, 19./2, 1./2, 0},
                                                {9,  1./2,  1./3, 1./3}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(Ctmp, 0.);
        apply(C,
              NoMask(),
              Plus<double>(),
              MultiplicativeInverse<double>(),
              A);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // Mask, replace
    {
        std::vector<std::vector<double>> ans = {{10,  10,    9,    0},
                                                { 0,  19./2, 1./2, 0},
                                                { 0,  0,     1./3, 1./3}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(Ctmp, 0);
        apply(C,
              M,
              Plus<double>(),
              MultiplicativeInverse<double>(),
              A,
              REPLACE);
        BOOST_CHECK_EQUAL(C, answer);
    }
    // Mask, merge
    {
        std::vector<std::vector<double>> ans = {{10, 10,    9,    0},
                                                {9,  19./2, 1./2, 0},
                                                {9,  0,     1./3, 1./3}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(Ctmp, 0);
        apply(C,
              M,
              Plus<double>(),
              MultiplicativeInverse<double>(),
              A,
              MERGE);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // complement(Mask), replace
    {
        std::vector<std::vector<double>> ans = {{0,  0,    0, 0},
                                                {10, 0,    0, 0},
                                                {9,  1./2, 0, 0}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(Ctmp, 0);
        apply(C,
              complement(M),
              Plus<double>(),
              MultiplicativeInverse<double>(),
              A,
              REPLACE);
        BOOST_CHECK_EQUAL(C, answer);
    }
    // complement(Mask), merge
    {
        std::vector<std::vector<double>> ans = {{ 9, 9,    9, 0},
                                                {10, 9,    0, 0},
                                                { 9, 1./2, 0, 0}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(Ctmp, 0);
        apply(C,
              complement(M),
              Plus<double>(),
              MultiplicativeInverse<double>(),
              A,
              MERGE);
        BOOST_CHECK_EQUAL(C, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(apply_stdmat_test_second_accum)
{
    // Build some matrices.
    IndexArrayType i = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4};
    Matrix<double, DirectedMatrixTag> A(4, 4);
    A.build(i, j, v);
    Matrix<double, DirectedMatrixTag> C(4, 4);

    std::vector<double> v_answer = {1, 1, 1, 1./2, 1./2,
                                    1./2, 1./3, 1./3, 1./3, 1./4};

    Matrix<double, DirectedMatrixTag> answer(4, 4);
    answer.build(i, j, v_answer);

    apply(C,
          NoMask(),
          Second<double>(),
          MultiplicativeInverse<double>(),
          A);
    BOOST_CHECK_EQUAL(C, answer);
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
        std::vector<std::vector<double>> ans = {{1, 1,    0,    0},
                                                {1, 1./2, 1./2, 0},
                                                {0, 1./2, 1./3, 1./3}};
        Matrix<double, DirectedMatrixTag> answer(ans, 0.);

        Matrix<double, DirectedMatrixTag> C(3, 4);
        apply(C,
              NoMask(),
              NoAccumulate(),
              MultiplicativeInverse<double>(),
              transpose(A));
        BOOST_CHECK_EQUAL(C, answer);
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
                   MultiplicativeInverse<double>(),
                   u)),
            DimensionException);
    }

    {
        Vector<double> w(4);
        Vector<bool> m(3);
        BOOST_CHECK_THROW(
            (apply(w,
                   m,
                   NoAccumulate(),
                   MultiplicativeInverse<double>(),
                   u)),
            DimensionException);
    }
}

//****************************************************************************

BOOST_AUTO_TEST_CASE(sparse_apply_vector_nomask_noaccum)
{
    std::vector<double> vecA = {8, 0, 6, 0};
    GraphBLAS::Vector<double> vA(vecA, 0);

    std::vector<double> vecC = {1, 2, 0, 0};
    GraphBLAS::Vector<double> vC(vecC, 0);

    std::vector<double> vecAnswer = {-8, 0, -6,  0};
    GraphBLAS::Vector<double> answer(vecAnswer, 0);

    GraphBLAS::apply(vC,
                     GraphBLAS::NoMask(),
                     GraphBLAS::NoAccumulate(),
                     GraphBLAS::AdditiveInverse<double>(),
                     vA);

    BOOST_CHECK_EQUAL(vC, answer);
}

//****************************************************************************

BOOST_AUTO_TEST_CASE(sparse_apply_vector_nomask_noaccum_bool)
{
    std::vector<bool> vecA = {true, false, true, false};
    GraphBLAS::Vector<bool> vA(vecA, false);

    std::vector<bool> vecC = {true, true, false, false};
    GraphBLAS::Vector<bool> vC(vecC, false);

    std::vector<bool> vecAnswer = {false, true, false, true};
    GraphBLAS::Vector<bool> answer(vecAnswer, true);

    GraphBLAS::apply(vC,
                     GraphBLAS::NoMask(),
                     GraphBLAS::NoAccumulate(),
                     GraphBLAS::LogicalNot<bool>(),
                     vA,
                     GraphBLAS::REPLACE);

    BOOST_CHECK_EQUAL(vC, answer);
}

//****************************************************************************

BOOST_AUTO_TEST_CASE(sparse_apply_vector_nomask_noaccum_int)
{
    std::vector<int> vecA = {1, 0, 1, 0};
    GraphBLAS::Vector<int> vA(vecA, 0);

    std::vector<int> vecC = {1, 1, 0, 0};
    GraphBLAS::Vector<int> vC(vecC, 0);

    std::vector<int> vecAnswer = {-1, 0, -1, 0};
    GraphBLAS::Vector<int> answer(vecAnswer, 0);

    GraphBLAS::apply(vC,
                     GraphBLAS::NoMask(),
                     GraphBLAS::NoAccumulate(),
                     GraphBLAS::AdditiveInverse<int>(),
                     vA,
                     GraphBLAS::MERGE);

    BOOST_CHECK_EQUAL(vC, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_apply_vector_mask_merge_noaccum)
{
    std::vector<double> vecA = {8, 8, 0, 0, 6, 6, 0, 0};
    GraphBLAS::Vector<double> vA(vecA, 0);

    std::vector<double> vecC = {1, 1, 2, 2, 0, 0, 0, 0};
    GraphBLAS::Vector<double> vC(vecC, 0);

    std::vector<bool> vecMask = {true, false, true, false, true, false, true, false};
    GraphBLAS::Vector<bool> mask(vecMask, false);

    std::vector<double> vecAnswer = {-8, 1, 0, 2, -6, 0, 0, 0};
    GraphBLAS::Vector<double> answer(vecAnswer, 0);

    GraphBLAS::apply(vC,
                     mask,
                     GraphBLAS::NoAccumulate(),
                     GraphBLAS::AdditiveInverse<double>(),
                     vA);

    BOOST_CHECK_EQUAL(vC, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_apply_vector_mask_replace_noaccum)
{
    std::vector<double> vecA = {8, 8, 0, 0, 6, 6, 0, 0};
    GraphBLAS::Vector<double> vA(vecA, 0);

    std::vector<double> vecC = {1, 1, 2, 2, 0, 0, 0, 0};
    GraphBLAS::Vector<double> vC(vecC, 0);

    std::vector<bool> vecMask = {true, false, true, false, true, false, true, false};
    GraphBLAS::Vector<bool> mask(vecMask, false);

    std::vector<double> vecAnswer = {-8, 0, 0, 0, -6, 0, 0, 0};
    GraphBLAS::Vector<double> answer(vecAnswer, 0);

    GraphBLAS::apply(vC,
                     mask,
                     GraphBLAS::NoAccumulate(),
                     GraphBLAS::AdditiveInverse<double>(),
                     vA,
                     REPLACE);

    BOOST_CHECK_EQUAL(vC, answer);
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
        std::vector<double> ans = {1/4., 1/5., 1/2., 1/3.,   0,   0};
        Vector<double> answer(ans, 0.);

        Vector<double> w(6);
        apply(w,
              NoMask(),
              NoAccumulate(),
              MultiplicativeInverse<double>(),
              u);
        BOOST_CHECK_EQUAL(w, answer);
    }

    // Mask, replace
    {
        std::vector<double> ans = {0, 0, 1/2., 1/3.,   0,   0};
        Vector<double> answer(ans, 0.);

        Vector<double> w(wtmp, 0);
        apply(w,
              m,
              NoAccumulate(),
              MultiplicativeInverse<double>(),
              u,
              REPLACE);
        BOOST_CHECK_EQUAL(w, answer);
    }
    // Mask, merge
    {
        std::vector<double> ans = {9, 0, 1/2., 1/3., 0, 0};
        Vector<double> answer(ans, 0.);

        Vector<double> w(wtmp, 0);
        apply(w,
              m,
              NoAccumulate(),
              MultiplicativeInverse<double>(),
              u,
              MERGE);
        BOOST_CHECK_EQUAL(w, answer);
    }

    // complement(Mask), replace
    {
        std::vector<double> ans = {1/4., 1/5., 0, 0,   0,   0};
        Vector<double> answer(ans, 0.);

        Vector<double> w(wtmp, 0);
        apply(w,
              complement(m),
              NoAccumulate(),
              MultiplicativeInverse<double>(),
              u,
              REPLACE);
        BOOST_CHECK_EQUAL(w, answer);
    }
    // complement(Mask), merge
    {
        std::vector<double> ans = {1/4., 1/5., 9, 0, 9,  0};
        Vector<double> answer(ans, 0.);

        Vector<double> w(wtmp, 0);
        apply(w,
              complement(m),
              NoAccumulate(),
              MultiplicativeInverse<double>(),
              u,
              MERGE);
        BOOST_CHECK_EQUAL(w, answer);
    }
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_apply_vector_accum)
{
    std::vector<double> vecA = {8, 0, 6, 0};
    GraphBLAS::Vector<double> vA(vecA, 0);

    std::vector<double> vecC = {8, 2, 0, 0};
    GraphBLAS::Vector<double> vC(vecC, 0);

    // NOTE: The accum results in an explicit zero
    std::vector<double> vecAnswer = {0, 2, -6,  666};
    GraphBLAS::Vector<double> answer(vecAnswer, 666);

    GraphBLAS::apply(vC,
                     GraphBLAS::NoMask(),
                     GraphBLAS::Plus<double>(),
                     GraphBLAS::AdditiveInverse<double>(),
                     vA);

    BOOST_CHECK_EQUAL(vC, answer);
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
        std::vector<double> ans = {37/4., 1/5., 19/2., 1/3.,   9,   0};
        Vector<double> answer(ans, 0.);

        Vector<double> w(wtmp, 0);
        apply(w,
              NoMask(),
              Plus<double>(),
              MultiplicativeInverse<double>(),
              u);
        BOOST_CHECK_EQUAL(w, answer);
    }

    // Mask, replace
    {
        std::vector<double> ans = {0, 0, 19/2., 1/3.,   9,   0};
        Vector<double> answer(ans, 0.);

        Vector<double> w(wtmp, 0);
        apply(w,
              m,
              Plus<double>(),
              MultiplicativeInverse<double>(),
              u,
              REPLACE);
        BOOST_CHECK_EQUAL(w, answer);
    }
    // Mask, merge
    {
        std::vector<double> ans = {9, 0, 19/2., 1/3., 9, 0};
        Vector<double> answer(ans, 0.);

        Vector<double> w(wtmp, 0);
        apply(w,
              m,
              Plus<double>(),
              MultiplicativeInverse<double>(),
              u,
              MERGE);
        BOOST_CHECK_EQUAL(w, answer);
    }

    // complement(Mask), replace
    {
        std::vector<double> ans = {37/4., 1/5., 0, 0,   0,   0};
        Vector<double> answer(ans, 0.);

        Vector<double> w(wtmp, 0);
        apply(w,
              complement(m),
              Plus<double>(),
              MultiplicativeInverse<double>(),
              u,
              REPLACE);
        BOOST_CHECK_EQUAL(w, answer);
    }
    // complement(Mask), merge
    {
        std::vector<double> ans = {37/4., 1/5., 9, 0, 9,  0};
        Vector<double> answer(ans, 0.);

        Vector<double> w(wtmp, 0);
        apply(w,
              complement(m),
              Plus<double>(),
              MultiplicativeInverse<double>(),
              u,
              MERGE);
        BOOST_CHECK_EQUAL(w, answer);
    }
}

//****************************************************************************
// test unaryop interface in conjuction with std::bind
//****************************************************************************


//****************************************************************************

BOOST_AUTO_TEST_CASE(bind_sparse_apply_matrix_nomask_noaccum)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(matA, 0);

    std::vector<std::vector<double>> matC = {{1, 2, 3},
                                             {4, 5, 6},
                                             {7, 8, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mC(matC, 0);

    std::vector<std::vector<double>> matAnswer = {{-8, -1, -6},
                                                  {-3, -5, -7},
                                                  {-4, -9,  0}};

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(matAnswer, 0);

    GraphBLAS::apply(mC,
                     GraphBLAS::NoMask(),
                     GraphBLAS::NoAccumulate(),
                     std::bind(GraphBLAS::Times<double>(),
                               -1.0,
                               std::placeholders::_1),
                     mA);

    BOOST_CHECK_EQUAL(mC, answer);

    GraphBLAS::apply(mC,
                     GraphBLAS::NoMask(),
                     GraphBLAS::NoAccumulate(),
                     std::bind(GraphBLAS::Times<double>(),
                               std::placeholders::_1,
                               -1.0),
                     mA);

    BOOST_CHECK_EQUAL(mC, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bind_sparse_apply_matrix_mask_merge_noaccum)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(matA, 0);

    std::vector<std::vector<double>> matC = {{1, 2, 3},
                                             {4, 5, 6},
                                             {7, 8, 0}};

    std::vector<std::vector<bool>> matMask = {{true, false, false},
                                              {true, true,  false,},
                                              {true, true,  true}};
    GraphBLAS::Matrix<bool, GraphBLAS::DirectedMatrixTag> mask(matMask, false);

    std::vector<std::vector<double>> matAnswer = {{-8, 2,  3},
                                                  {-3, -5, 6},
                                                  {-4, -9, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(matAnswer, 0);

    {
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mC(matC, 0);

        GraphBLAS::apply(mC,
                         mask,
                         GraphBLAS::NoAccumulate(),
                         std::bind(GraphBLAS::Times<double>(),
                               -1.0,
                               std::placeholders::_1),
                         mA);

        BOOST_CHECK_EQUAL(mC, answer);
    }
    {
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mC(matC, 0);

        GraphBLAS::apply(mC,
                         mask,
                         GraphBLAS::NoAccumulate(),
                         std::bind(GraphBLAS::Times<double>(),
                                   std::placeholders::_1,
                                   -1.0),
                         mA);

        BOOST_CHECK_EQUAL(mC, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bind_sparse_apply_matrix_mask_replace_noaccum)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(matA, 0);

    std::vector<std::vector<double>> matC = {{1, 2, 3},
                                             {4, 5, 6},
                                             {7, 8, 0}};


    std::vector<std::vector<bool>> matMask = {{true, false, false},
                                              {true, true,  false,},
                                              {true, true,  true}};
    GraphBLAS::Matrix<bool, GraphBLAS::DirectedMatrixTag> mask(matMask, false);

    std::vector<std::vector<double>> matAnswer = {{-8, 0,  0},
                                                  {-3, -5, 0},
                                                  {-4, -9, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(matAnswer, 0);

    {
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mC(matC, 0);
        GraphBLAS::apply(mC,
                         mask,
                         GraphBLAS::NoAccumulate(),
                         std::bind(GraphBLAS::Times<double>(),
                                   -1.0,
                                   std::placeholders::_1),
                         mA,
                         GraphBLAS::REPLACE);

        BOOST_CHECK_EQUAL(mC, answer);
    }
    {
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mC(matC, 0);
        GraphBLAS::apply(mC,
                         mask,
                         GraphBLAS::NoAccumulate(),
                         std::bind(GraphBLAS::Times<double>(),
                                   std::placeholders::_1,
                                   -1.0),
                         mA,
                         GraphBLAS::REPLACE);

        BOOST_CHECK_EQUAL(mC, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bind_apply_stdmat_test_noaccum)
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
              std::bind(GraphBLAS::Plus<double>(),
                        0.5,
                        std::placeholders::_1),
              A);
        BOOST_CHECK_EQUAL(C, answer);

        Matrix<double, DirectedMatrixTag> C2(3, 4);
        apply(C2,
              NoMask(),
              NoAccumulate(),
              std::bind(GraphBLAS::Plus<double>(),
                        std::placeholders::_1,
                        0.5),
              A);
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
              std::bind(GraphBLAS::Plus<double>(),
                        0.5,
                        std::placeholders::_1),
              A,
              REPLACE);
        BOOST_CHECK_EQUAL(C, answer);

        Matrix<double, DirectedMatrixTag> C2(Ctmp, 0);
        apply(C2,
              M,
              NoAccumulate(),
              std::bind(GraphBLAS::Plus<double>(),
                        std::placeholders::_1,
                        0.5),
              A,
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
              std::bind(GraphBLAS::Plus<double>(),
                        0.5,
                        std::placeholders::_1),
              A,
              MERGE);
        BOOST_CHECK_EQUAL(C, answer);

        Matrix<double, DirectedMatrixTag> C2(Ctmp, 0);
        apply(C2,
              M,
              NoAccumulate(),
              std::bind(GraphBLAS::Plus<double>(),
                        std::placeholders::_1,
                        0.5),
              A,
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
              std::bind(GraphBLAS::Plus<double>(),
                        0.5,
                        std::placeholders::_1),
              A,
              REPLACE);
        BOOST_CHECK_EQUAL(C, answer);

        Matrix<double, DirectedMatrixTag> C2(Ctmp, 0);
        apply(C2,
              complement(M),
              NoAccumulate(),
              std::bind(GraphBLAS::Plus<double>(),
                        std::placeholders::_1,
                        0.5),
              A,
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
              std::bind(GraphBLAS::Plus<double>(),
                        0.5,
                        std::placeholders::_1),
              A,
              MERGE);
        BOOST_CHECK_EQUAL(C, answer);

        Matrix<double, DirectedMatrixTag> C2(Ctmp, 0);
        apply(C2,
              complement(M),
              NoAccumulate(),
              std::bind(GraphBLAS::Plus<double>(),
                        std::placeholders::_1,
                        0.5),
              A,
              MERGE);
        BOOST_CHECK_EQUAL(C2, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bind_sparse_apply_matrix_nomask_plus_accum)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(matA, 0);

    std::vector<std::vector<double>> matC = {{1, 2, 3},
                                             {4, 5, 6},
                                             {7, 8, 0}};

    std::vector<std::vector<double>> matAnswer = {{-7, 1, -3},
                                                  { 1, 0, -1},
                                                  { 3,-1,  1337}};
    // NOTE: The accum results in an explicit zero
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(matAnswer, 1337);

    {
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mC(matC, 0);
        GraphBLAS::apply(mC,
                         GraphBLAS::NoMask(),
                         GraphBLAS::Plus<double>(),
                         std::bind(GraphBLAS::Times<double>(),
                                   -1.0,
                                   std::placeholders::_1),
                         mA);

        BOOST_CHECK_EQUAL(mC, answer);
    }
    {
        GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mC(matC, 0);
        GraphBLAS::apply(mC,
                         GraphBLAS::NoMask(),
                         GraphBLAS::Plus<double>(),
                         std::bind(GraphBLAS::Times<double>(),
                                   std::placeholders::_1,
                                   -1.0),
                         mA);

        BOOST_CHECK_EQUAL(mC, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bind_apply_stdmat_test_plus_accum)
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
              std::bind(GraphBLAS::Times<double>(),
                        0.5,
                        std::placeholders::_1),
              A);
        BOOST_CHECK_EQUAL(C, answer);

        Matrix<double, DirectedMatrixTag> C2(Ctmp, 0.);
        apply(C2,
              NoMask(),
              Plus<double>(),
              std::bind(GraphBLAS::Times<double>(),
                        std::placeholders::_1,
                        0.5),
              A);
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
              std::bind(GraphBLAS::Times<double>(),
                        0.5,
                        std::placeholders::_1),
              A,
              REPLACE);
        BOOST_CHECK_EQUAL(C, answer);

        Matrix<double, DirectedMatrixTag> C2(Ctmp, 0);
        apply(C2,
              M,
              Plus<double>(),
              std::bind(GraphBLAS::Times<double>(),
                        std::placeholders::_1,
                        0.5),
              A,
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
              std::bind(GraphBLAS::Times<double>(),
                        0.5,
                        std::placeholders::_1),
              A,
              MERGE);
        BOOST_CHECK_EQUAL(C, answer);

        Matrix<double, DirectedMatrixTag> C2(Ctmp, 0);
        apply(C2,
              M,
              Plus<double>(),
              std::bind(GraphBLAS::Times<double>(),
                        std::placeholders::_1,
                        0.5),
              A,
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
              std::bind(GraphBLAS::Times<double>(),
                        0.5,
                        std::placeholders::_1),
              A,
              REPLACE);
        BOOST_CHECK_EQUAL(C, answer);

        Matrix<double, DirectedMatrixTag> C2(Ctmp, 0);
        apply(C2,
              complement(M),
              Plus<double>(),
              std::bind(GraphBLAS::Times<double>(),
                        std::placeholders::_1,
                        0.5),
              A,
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
              std::bind(GraphBLAS::Times<double>(),
                        0.5,
                        std::placeholders::_1),
              A,
              MERGE);
        BOOST_CHECK_EQUAL(C, answer);

        Matrix<double, DirectedMatrixTag> C2(Ctmp, 0);
        apply(C2,
              complement(M),
              Plus<double>(),
              std::bind(GraphBLAS::Times<double>(),
                        std::placeholders::_1,
                        0.5),
              A,
              MERGE);
        BOOST_CHECK_EQUAL(C2, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bind_apply_stdmat_test_second_accum)
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
          std::bind(GraphBLAS::Times<double>(),
                    0.5,
                    std::placeholders::_1),
          A);
    BOOST_CHECK_EQUAL(C, answer);

    Matrix<double, DirectedMatrixTag> C2(4, 4);
    apply(C2,
          NoMask(),
          Second<double>(),
          std::bind(GraphBLAS::Times<double>(),
                    std::placeholders::_1,
                    0.5),
          A);
    BOOST_CHECK_EQUAL(C2, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bind_apply_stdmat_test_noaccum_transpose)
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
              std::bind(GraphBLAS::Times<double>(),
                        0.5,
                        std::placeholders::_1),
              transpose(A));
        BOOST_CHECK_EQUAL(C, answer);

        Matrix<double, DirectedMatrixTag> C2(3, 4);
        apply(C2,
              NoMask(),
              NoAccumulate(),
              std::bind(GraphBLAS::Times<double>(),
                        std::placeholders::_1,
                        0.5),
              transpose(A));
        BOOST_CHECK_EQUAL(C2, answer);
    }
}

//****************************************************************************
// Apply standard vector
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(bind_apply_stdvec_test_bad_dimension)
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
                   std::bind(GraphBLAS::Times<double>(),
                             0.5,
                             std::placeholders::_1),
                   u)),
            DimensionException);

        BOOST_CHECK_THROW(
            (apply(w,
                   NoMask(),
                   NoAccumulate(),
                   std::bind(GraphBLAS::Times<double>(),
                             std::placeholders::_1,
                             0.5),
                   u)),
            DimensionException);
    }

    {
        Vector<double> w(4);
        Vector<bool> m(3);
        BOOST_CHECK_THROW(
            (apply(w,
                   m,
                   NoAccumulate(),
                   std::bind(GraphBLAS::Times<double>(),
                             0.5,
                             std::placeholders::_1),
                   u)),
            DimensionException);
        BOOST_CHECK_THROW(
            (apply(w,
                   m,
                   NoAccumulate(),
                   std::bind(GraphBLAS::Times<double>(),
                             std::placeholders::_1,
                             0.5),
                   u)),
            DimensionException);
    }
}

//****************************************************************************

BOOST_AUTO_TEST_CASE(bind_sparse_apply_vector_nomask_noaccum)
{
    std::vector<double> vecA = {8, 0, 6, 0};
    GraphBLAS::Vector<double> vA(vecA, 0);

    std::vector<double> vecC = {1, 2, 0, 0};

    std::vector<double> vecAnswer = {-8, 0, -6,  0};
    GraphBLAS::Vector<double> answer(vecAnswer, 0);

    {
        GraphBLAS::Vector<double> vC(vecC, 0);
        GraphBLAS::apply(vC,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         std::bind(GraphBLAS::Times<double>(),
                                   -1.0,
                                   std::placeholders::_1),
                         vA);

        BOOST_CHECK_EQUAL(vC, answer);
    }
    {
        GraphBLAS::Vector<double> vC(vecC, 0);
        GraphBLAS::apply(vC,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         std::bind(GraphBLAS::Times<double>(),
                                   std::placeholders::_1,
                                   -1.0),
                         vA);

        BOOST_CHECK_EQUAL(vC, answer);
    }
}

//****************************************************************************

BOOST_AUTO_TEST_CASE(bind_sparse_apply_vector_nomask_noaccum_bool)
{
    std::vector<bool> vecA = {true, false, true, false};
    GraphBLAS::Vector<bool> vA(vecA, false);

    std::vector<bool> vecC = {true, true, false, false};

    std::vector<bool> vecAnswer = {true, false, true, false};
    GraphBLAS::Vector<bool> answer(vecAnswer, false);

    {
        GraphBLAS::Vector<bool> vC(vecC, false);
        GraphBLAS::apply(vC,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         std::bind(GraphBLAS::LogicalXor<bool>(),
                                   false,
                                   std::placeholders::_1),
                         vA);

        BOOST_CHECK_EQUAL(vC, answer);
    }
    {
        GraphBLAS::Vector<bool> vC(vecC, false);
        GraphBLAS::apply(vC,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         std::bind(GraphBLAS::LogicalXor<bool>(),
                                   std::placeholders::_1,
                                   false),
                         vA);

        BOOST_CHECK_EQUAL(vC, answer);
    }
}

//****************************************************************************

BOOST_AUTO_TEST_CASE(bind_sparse_apply_vector_nomask_noaccum_int)
{
    std::vector<int> vecA = {1, 0, 1, 0};
    GraphBLAS::Vector<int> vA(vecA, 0);

    std::vector<int> vecC = {1, 1, 0, 0};

    std::vector<int> vecAnswer = {2, 0, 2, 0};
    GraphBLAS::Vector<int> answer(vecAnswer, 0);

    {
        GraphBLAS::Vector<int> vC(vecC, 0);
        GraphBLAS::apply(vC,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         std::bind(GraphBLAS::Plus<int>(),
                                   1,
                                   std::placeholders::_1),
                         vA);

        BOOST_CHECK_EQUAL(vC, answer);
    }
    {
        GraphBLAS::Vector<int> vC(vecC, 0);
        GraphBLAS::apply(vC,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         std::bind(GraphBLAS::Plus<int>(),
                                   std::placeholders::_1,
                                   1),
                         vA);

        BOOST_CHECK_EQUAL(vC, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bind_sparse_apply_vector_mask_merge_noaccum)
{
    std::vector<double> vecA = {8, 8, 0, 0, 6, 6, 0, 0};
    GraphBLAS::Vector<double> vA(vecA, 0);

    std::vector<double> vecC = {1, 1, 2, 2, 0, 0, 0, 0};

    std::vector<bool> vecMask = {true, false, true, false, true, false, true, false};
    GraphBLAS::Vector<bool> mask(vecMask, false);

    std::vector<double> vecAnswer = {-8, 1, 0, 2, -6, 0, 0, 0};
    GraphBLAS::Vector<double> answer(vecAnswer, 0);

    {
        GraphBLAS::Vector<double> vC(vecC, 0);
        GraphBLAS::apply(vC,
                         mask,
                         GraphBLAS::NoAccumulate(),
                         std::bind(GraphBLAS::Times<double>(),
                                   -1.0,
                                   std::placeholders::_1),
                         vA);

        BOOST_CHECK_EQUAL(vC, answer);
    }
    {
        GraphBLAS::Vector<double> vC(vecC, 0);
        GraphBLAS::apply(vC,
                         mask,
                         GraphBLAS::NoAccumulate(),
                         std::bind(GraphBLAS::Times<double>(),
                                   std::placeholders::_1,
                                   -1.0),
                         vA);

        BOOST_CHECK_EQUAL(vC, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bind_sparse_apply_vector_mask_replace_noaccum)
{
    std::vector<double> vecA = {8, 8, 0, 0, 6, 6, 0, 0};
    GraphBLAS::Vector<double> vA(vecA, 0);

    std::vector<double> vecC = {1, 1, 2, 2, 0, 0, 0, 0};

    std::vector<bool> vecMask = {true, false, true, false, true, false, true, false};
    GraphBLAS::Vector<bool> mask(vecMask, false);

    std::vector<double> vecAnswer = {-8, 0, 0, 0, -6, 0, 0, 0};
    GraphBLAS::Vector<double> answer(vecAnswer, 0);

    {
        GraphBLAS::Vector<double> vC(vecC, 0);
        GraphBLAS::apply(vC,
                         mask,
                         GraphBLAS::NoAccumulate(),
                         std::bind(GraphBLAS::Times<double>(),
                                   -1.0,
                                   std::placeholders::_1),
                         vA,
                         REPLACE);

        BOOST_CHECK_EQUAL(vC, answer);
    }

    {
        GraphBLAS::Vector<double> vC(vecC, 0);
        GraphBLAS::apply(vC,
                         mask,
                         GraphBLAS::NoAccumulate(),
                         std::bind(GraphBLAS::Times<double>(),
                                   std::placeholders::_1,
                                   -1.0),
                         vA,
                         REPLACE);

        BOOST_CHECK_EQUAL(vC, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bind_apply_stdvec_test_noaccum)
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
              std::bind(GraphBLAS::Times<double>(),
                        -1.0,
                        std::placeholders::_1),
              u);
        BOOST_CHECK_EQUAL(w, answer);

        Vector<double> w2(6);
        apply(w2,
              NoMask(),
              NoAccumulate(),
              std::bind(GraphBLAS::Times<double>(),
                        std::placeholders::_1,
                        -1.0),
              u);
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
              std::bind(GraphBLAS::Times<double>(),
                        -1.0,
                        std::placeholders::_1),
              u,
              REPLACE);
        BOOST_CHECK_EQUAL(w, answer);

        Vector<double> w2(wtmp, 0);
        apply(w2,
              m,
              NoAccumulate(),
              std::bind(GraphBLAS::Times<double>(),
                        std::placeholders::_1,
                        -1.0),
              u,
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
              std::bind(GraphBLAS::Times<double>(),
                        -1.0,
                        std::placeholders::_1),
              u,
              MERGE);
        BOOST_CHECK_EQUAL(w, answer);

        Vector<double> w2(wtmp, 0);
        apply(w2,
              m,
              NoAccumulate(),
              std::bind(GraphBLAS::Times<double>(),
                        std::placeholders::_1,
                        -1.0),
              u,
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
              std::bind(GraphBLAS::Times<double>(),
                        -1.0,
                        std::placeholders::_1),
              u,
              REPLACE);
        BOOST_CHECK_EQUAL(w, answer);

        Vector<double> w2(wtmp, 0);
        apply(w2,
              complement(m),
              NoAccumulate(),
              std::bind(GraphBLAS::Times<double>(),
                        std::placeholders::_1,
                        -1.0),
              u,
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
              std::bind(GraphBLAS::Times<double>(),
                        -1.0,
                        std::placeholders::_1),
              u,
              MERGE);
        BOOST_CHECK_EQUAL(w, answer);

        Vector<double> w2(wtmp, 0);
        apply(w2,
              complement(m),
              NoAccumulate(),
              std::bind(GraphBLAS::Times<double>(),
                        std::placeholders::_1,
                        -1.0),
              u,
              MERGE);
        BOOST_CHECK_EQUAL(w2, answer);
    }
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(bind_sparse_apply_vector_accum)
{
    std::vector<double> vecA = {8, 0, 6, 0};
    GraphBLAS::Vector<double> vA(vecA, 0);

    std::vector<double> vecC = {8, 2, 0, 0};

    // NOTE: The accum results in an explicit zero
    std::vector<double> vecAnswer = {0, 2, -6,  666};
    GraphBLAS::Vector<double> answer(vecAnswer, 666);

    {
        GraphBLAS::Vector<double> vC(vecC, 0);
        GraphBLAS::apply(vC,
                         GraphBLAS::NoMask(),
                         GraphBLAS::Plus<double>(),
                         std::bind(GraphBLAS::Times<double>(),
                                   -1.0,
                                   std::placeholders::_1),
                         vA);

        BOOST_CHECK_EQUAL(vC, answer);
    }

    {
        GraphBLAS::Vector<double> vC(vecC, 0);
        GraphBLAS::apply(vC,
                         GraphBLAS::NoMask(),
                         GraphBLAS::Plus<double>(),
                         std::bind(GraphBLAS::Times<double>(),
                                   std::placeholders::_1,
                                   -1.0),
                         vA);

        BOOST_CHECK_EQUAL(vC, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bind_apply_stdvec_test_plus_accum)
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
              std::bind(GraphBLAS::Times<double>(),
                        -1.0,
                        std::placeholders::_1),
              u);
        BOOST_CHECK_EQUAL(w, answer);

        Vector<double> w2(wtmp, 0);
        apply(w2,
              NoMask(),
              Plus<double>(),
              std::bind(GraphBLAS::Times<double>(),
                        std::placeholders::_1,
                        -1.0),
              u);
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
              std::bind(GraphBLAS::Times<double>(),
                        -1.0,
                        std::placeholders::_1),
              u,
              REPLACE);
        BOOST_CHECK_EQUAL(w, answer);

        Vector<double> w2(wtmp, 0);
        apply(w2,
              m,
              Plus<double>(),
              std::bind(GraphBLAS::Times<double>(),
                        std::placeholders::_1,
                        -1.0),
              u,
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
              std::bind(GraphBLAS::Times<double>(),
                        -1.0,
                        std::placeholders::_1),
              u,
              MERGE);
        BOOST_CHECK_EQUAL(w2, answer);

        Vector<double> w(wtmp, 0);
        apply(w,
              m,
              Plus<double>(),
              std::bind(GraphBLAS::Times<double>(),
                        std::placeholders::_1,
                        -1.0),
              u,
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
              std::bind(GraphBLAS::Times<double>(),
                        -1.0,
                        std::placeholders::_1),
              u,
              REPLACE);
        BOOST_CHECK_EQUAL(w, answer);

        Vector<double> w2(wtmp, 0);
        apply(w2,
              complement(m),
              Plus<double>(),
              std::bind(GraphBLAS::Times<double>(),
                        std::placeholders::_1,
                        -1.0),
              u,
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
              std::bind(GraphBLAS::Times<double>(),
                        -1.0,
                        std::placeholders::_1),
              u,
              MERGE);
        BOOST_CHECK_EQUAL(w, answer);

        Vector<double> w2(wtmp, 0);
        apply(w2,
              complement(m),
              Plus<double>(),
              std::bind(GraphBLAS::Times<double>(),
                        std::placeholders::_1,
                        -1.0),
              u,
              MERGE);
        BOOST_CHECK_EQUAL(w2, answer);
    }
}


BOOST_AUTO_TEST_SUITE_END()
