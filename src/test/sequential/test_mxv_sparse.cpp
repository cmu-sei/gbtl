/*
 * Copyright (c) 2017 Carnegie Mellon University.
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

#include <iostream>
#include <graphblas/graphblas.hpp>

using namespace GraphBLAS;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE mxm_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(mxm_suite)

//****************************************************************************
// Matrix-vector multiply
/// @todo move to front end tests.
BOOST_AUTO_TEST_CASE(mxv_reg)
{
    std::vector<std::vector<double>> mat = {{8, 1, 6},
                                            {3, 0, 7},
                                            {0, 0, 2}};

    std::vector<uint32_t> mask = {1, 1, 1};
    std::vector<double> vec1 = {-1, 2, 1};
    std::vector<double> vec2 = {0, 1, 0};
    std::vector<double> vec3 = {2, 1, 0};
    std::vector<double> vec4 = {0, 1, 1};

    // Must use tuple notation because some zeros are stored
    std::vector<IndexType> i1 = {0, 1, 2}; std::vector<double> a1 = {0, 4, 2};
    std::vector<IndexType> i2 = {0};       std::vector<double> a2 = {1};
    std::vector<IndexType> i3 = {0, 1};    std::vector<double> a3 = {17, 6};
    std::vector<IndexType> i4 = {0, 1, 2}; std::vector<double> a4 = {7, 7, 2};

    backend::BitmapSparseVector<uint32_t> m(mask, 0);

    backend::LilSparseMatrix<double> A(mat, 0);

    backend::BitmapSparseVector<double> v1(vec1, 0);
    backend::BitmapSparseVector<double> v2(vec2, 0);
    backend::BitmapSparseVector<double> v3(vec3, 0);
    backend::BitmapSparseVector<double> v4(vec4, 0);

    backend::BitmapSparseVector<double> ans1(3, i1, a1);
    backend::BitmapSparseVector<double> ans2(3, i2, a2);
    backend::BitmapSparseVector<double> ans3(3, i3, a3);
    backend::BitmapSparseVector<double> ans4(3, i4, a4);

    backend::BitmapSparseVector<double> result(3);

    backend::mxv(result,
                 m,                                          // mask
                 Second<double>(),                // accum
                 ArithmeticSemiring<double>(),    // semiring
                 A,
                 v1);
    BOOST_CHECK_EQUAL(result, ans1);

    result.clear();
    backend::mxv(result,
                 m,                                          // mask
                 Second<double>(),                // accum
                 ArithmeticSemiring<double>(),    // semiring
                 A,
                 v2);
    BOOST_CHECK_EQUAL(result, ans2);

    result.clear();
    backend::mxv(result,
                 m,                                          // mask
                 Second<double>(),                // accum
                 ArithmeticSemiring<double>(),    // semiring
                 A,
                 v3);
    BOOST_CHECK_EQUAL(result, ans3);

    result.clear();
    backend::mxv(result,
                 m,                                          // mask
                 Second<double>(),                // accum
                 ArithmeticSemiring<double>(),    // semiring
                 A,
                 v4);
    BOOST_CHECK_EQUAL(result, ans4);
}

#if 0
//****************************************************************************
// Matrix multiply, empty rows and columns
BOOST_AUTO_TEST_CASE(mxm_reg_empty)
{
    std::vector<std::vector<double>> mat1 = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 2}};

    std::vector<std::vector<double>> mat2 = {{0, 0, 0},
                                             {1, 0, 1},
                                             {0, 0, 1}};

    std::vector<std::vector<double>> mat3 = {{1, 0, 7},
                                             {5, 0, 12},
                                             {9, 0, 11}};

    backend::LilSparseMatrix<double> m1(mat1, 0);
    backend::LilSparseMatrix<double> m2(mat2, 0);
    backend::LilSparseMatrix<double> answer(mat3, 0);
    IndexType M = 3;
    IndexType N = 3;
    backend::LilSparseMatrix<double> result(M, N);
    backend::mxm(result,
                 Second<double>(),
                 ArithmeticSemiring<double>(),
                 m1,
                 m2);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(mxm2_empty_rows_and_columns)
{
    std::vector<std::vector<double>> mat1 = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 2}};

    std::vector<std::vector<double>> mat2 = {{0, 0, 0},
                                             {1, 0, 1},
                                             {0, 0, 1}};

    std::vector<std::vector<double>> mat3 = {{1, 0, 7},
                                             {5, 0, 12},
                                             {9, 0, 11}};

    backend::LilSparseMatrix<double> m1(mat1, 0);
    backend::LilSparseMatrix<double> m2(mat2, 0);
    backend::LilSparseMatrix<double> answer(mat3, 0);
    IndexType M = 3;
    IndexType N = 3;
    backend::LilSparseMatrix<double> result(M, N);
    backend::mxm_v2(result,
                    Second<double>(),
                    ArithmeticSemiring<double>(),
                    m1,
                    m2);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(mxm2_accum_with_empty_rows_and_columns)
{
    std::vector<std::vector<double>> mat0 = {{1, 1, 1},
                                             {1, 1, 1},
                                             {1, 1, 1}};

    std::vector<std::vector<double>> mat1 = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 2}};

    std::vector<std::vector<double>> mat2 = {{0, 0, 0},
                                             {1, 0, 1},
                                             {0, 0, 1}};

    std::vector<std::vector<double>> mat3 = {{2,  1, 8},
                                             {6,  1, 13},
                                             {10, 1, 12}};

    backend::LilSparseMatrix<double> A(mat1, 0);
    backend::LilSparseMatrix<double> B(mat2, 0);
    backend::LilSparseMatrix<double> answer(mat3, 0);
    IndexType M = 3;
    IndexType N = 3;
    backend::LilSparseMatrix<double> C(mat0, 0);
    backend::mxm_v2(C,
                    Plus<double>(),
                    ArithmeticSemiring<double>(),
                    A,
                    B);
    BOOST_CHECK_EQUAL(C, answer);
}
#endif

BOOST_AUTO_TEST_SUITE_END()
