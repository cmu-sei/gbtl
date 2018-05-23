/*
 * Copyright (c) 2015 Carnegie Mellon University and The Trustees of Indiana
 * University.
 * All Rights Reserved.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS," WITH NO WARRANTIES WHATSOEVER. CARNEGIE
 * MELLON UNIVERSITY AND THE TRUSTEES OF INDIANA UNIVERSITY EXPRESSLY DISCLAIM
 * TO THE FULLEST EXTENT PERMITTED BY LAW ALL EXPRESS, IMPLIED, AND STATUTORY
 * WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT OF PROPRIETARY RIGHTS.
 *
 * This Program is distributed under a BSD license.  Please see LICENSE file or
 * permission@sei.cmu.edu for more information.  DM-0002659
 */

#include <functional>
#include <iostream>
#include <vector>

#include <graphblas/graphblas.hpp>

using namespace GraphBLAS;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE apply_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(apply_suite)

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

    // Build some sparse matrices.
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
