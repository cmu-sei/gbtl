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
                     true);

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
              true);
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
              false);
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
              true);
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
              false);
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
              true);
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
              false);
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
              true);
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
              false);
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
                     true);

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
              true);
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
              false);
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
              true);
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
              false);
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
              true);
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
              false);
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
              true);
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
              false);
        BOOST_CHECK_EQUAL(w, answer);
    }
}


BOOST_AUTO_TEST_SUITE_END()
