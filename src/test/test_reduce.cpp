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

using namespace graphblas;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE cpu_simple_reduce_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(cpu_simple_reduce_suite)

//****************************************************************************
BOOST_AUTO_TEST_CASE(row_reduce_test_bad_dimension)
{
    // Build some matrices.
    graphblas::IndexArrayType i_A    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    graphblas::IndexArrayType j_A    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_A = {1, 1, 1, 2, 2, 1, 3, 5, 3, 4};
    Matrix<double, DirectedMatrixTag> A(4, 4);
    buildmatrix(A, i_A, j_A, v_A);

    graphblas::IndexArrayType i_C    = {0, 1, 2};
    graphblas::IndexArrayType j_C    = {0, 0, 0};
    std::vector<double> v_C = {1, 1, 1};
    Matrix<double, DirectedMatrixTag> C(3, 1);
    buildmatrix(C, i_C, j_C, v_C);

    BOOST_CHECK_THROW(
        (row_reduce(A, C)),
        graphblas::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(row_reduce_test_default_accum)
{
    // Build some matrices.
    graphblas::IndexArrayType i_A    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    graphblas::IndexArrayType j_A    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_A = {1, 1, 1, 2, 2, 1, 3, 5, 3, 4};
    Matrix<double, DirectedMatrixTag> A(4, 4);
    buildmatrix(A, i_A, j_A, v_A);

    graphblas::IndexArrayType i_C    = {0, 1, 2, 3};
    graphblas::IndexArrayType j_C    = {0, 0, 0, 0};
    std::vector<double> v_C = {1, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> C(4, 1);
    buildmatrix(C, i_C, j_C, v_C);

    graphblas::IndexArrayType i_answer    = {0, 1, 2, 3};
    graphblas::IndexArrayType j_answer    = {0, 0, 0, 0};
    std::vector<double> v_answer = {2, 5, 9, 7};
    Matrix<double, DirectedMatrixTag> answer(4, 1);
    buildmatrix(answer, i_answer, j_answer, v_answer);

    row_reduce(A, C);

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(row_reduce_test_assign)
{
    // Build some matrices.
    graphblas::IndexArrayType i_A    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    graphblas::IndexArrayType j_A    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_A = {1, 1, 1, 2, 2, 1, 3, 5, 3, 4};
    Matrix<double, DirectedMatrixTag> A(4, 4);
    buildmatrix(A, i_A, j_A, v_A);

    graphblas::IndexArrayType i_C    = {0, 1, 2, 3};
    graphblas::IndexArrayType j_C    = {0, 0, 0, 0};
    std::vector<double> v_C = {1, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> C(4, 1);
    buildmatrix(C, i_C, j_C, v_C);

    graphblas::IndexArrayType i_answer    = {0, 1, 2, 3};
    graphblas::IndexArrayType j_answer    = {0, 0, 0, 0};
    std::vector<double> v_answer = {2, 5, 9, 7};
    Matrix<double, DirectedMatrixTag> answer(4, 1);
    buildmatrix(answer, i_answer, j_answer, v_answer);

    row_reduce(A, C,
               graphblas::PlusMonoid<double>(),
               graphblas::math::Assign<double>());

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(row_reduce_test_accum)
{
    // Build some matrices.
    graphblas::IndexArrayType i_A    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    graphblas::IndexArrayType j_A    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_A = {1, 1, 1, 2, 2, 1, 3, 5, 3, 4};
    Matrix<double, DirectedMatrixTag> A(4, 4);
    buildmatrix(A, i_A, j_A, v_A);

    graphblas::IndexArrayType i_C    = {0, 1, 2, 3};
    graphblas::IndexArrayType j_C    = {0, 0, 0, 0};
    std::vector<double> v_C = {1, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> C(4, 1);
    buildmatrix(C, i_C, j_C, v_C);

    graphblas::IndexArrayType i_answer    = {0, 1, 2, 3};
    graphblas::IndexArrayType j_answer    = {0, 0, 0, 0};
    std::vector<double> v_answer = {3, 6, 10, 8};
    Matrix<double, DirectedMatrixTag> answer(4, 1);
    buildmatrix(answer, i_answer, j_answer, v_answer);

    row_reduce(A, C,
               graphblas::PlusMonoid<double>(),
               graphblas::math::Accum<double>());

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(col_reduce_test_bad_dimensions)
{
    // Build some matrices.
    graphblas::IndexArrayType i_A    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    graphblas::IndexArrayType j_A    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_A = {1, 1, 1, 2, 2, 1, 3, 5, 3, 4};
    Matrix<double, DirectedMatrixTag> A(4, 4);
    buildmatrix(A, i_A, j_A, v_A);

    graphblas::IndexArrayType i_C    = {0, 0, 0, 0, 0};
    graphblas::IndexArrayType j_C    = {0, 1, 2, 3, 4};
    std::vector<double> v_C = {1, 1, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> C(1, 5);
    buildmatrix(C, i_C, j_C, v_C);

    BOOST_CHECK_THROW(
        (col_reduce(A, C)),
        graphblas::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(col_reduce_test_default_accum)
{
    // Build some matrices.
    graphblas::IndexArrayType i_A    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    graphblas::IndexArrayType j_A    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_A = {1, 1, 1, 2, 2, 1, 3, 5, 3, 4};
    Matrix<double, DirectedMatrixTag> A(4, 4);
    buildmatrix(A, i_A, j_A, v_A);

    graphblas::IndexArrayType i_C    = {0, 0, 0, 0};
    graphblas::IndexArrayType j_C    = {0, 1, 2, 3};
    std::vector<double> v_C = {1, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> C(1, 4);
    buildmatrix(C, i_C, j_C, v_C);

    graphblas::IndexArrayType i_answer    = {0, 0, 0, 0};
    graphblas::IndexArrayType j_answer    = {0, 1, 2, 3};
    std::vector<double> v_answer = {2, 4, 8, 9};
    Matrix<double, DirectedMatrixTag> answer(1, 4);
    buildmatrix(answer, i_answer, j_answer, v_answer);

    col_reduce(A, C);

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(col_reduce_test_assign)
{
    // Build some matrices.
    graphblas::IndexArrayType i_A    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    graphblas::IndexArrayType j_A    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_A = {1, 1, 1, 2, 2, 1, 3, 5, 3, 4};
    Matrix<double, DirectedMatrixTag> A(4, 4);
    buildmatrix(A, i_A, j_A, v_A);

    graphblas::IndexArrayType i_C    = {0, 0, 0, 0};
    graphblas::IndexArrayType j_C    = {0, 1, 2, 3};
    std::vector<double> v_C = {1, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> C(1, 4);
    buildmatrix(C, i_C, j_C, v_C);

    graphblas::IndexArrayType i_answer    = {0, 0, 0, 0};
    graphblas::IndexArrayType j_answer    = {0, 1, 2, 3};
    std::vector<double> v_answer = {2, 4, 8, 9};
    Matrix<double, DirectedMatrixTag> answer(1, 4);
    buildmatrix(answer, i_answer, j_answer, v_answer);

    col_reduce(A, C);

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(col_reduce_test_accum)
{
    // Build some matrices.
    graphblas::IndexArrayType i_A    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    graphblas::IndexArrayType j_A    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_A = {1, 1, 1, 2, 2, 1, 3, 5, 3, 4};
    Matrix<double, DirectedMatrixTag> A(4, 4);
    buildmatrix(A, i_A, j_A, v_A);

    graphblas::IndexArrayType i_C    = {0, 0, 0, 0};
    graphblas::IndexArrayType j_C    = {0, 1, 2, 3};
    std::vector<double> v_C = {1, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> C(1, 4);
    buildmatrix(C, i_C, j_C, v_C);

    graphblas::IndexArrayType i_answer    = {0, 0, 0, 0};
    graphblas::IndexArrayType j_answer    = {0, 1, 2, 3};
    std::vector<double> v_answer = {3, 5, 9, 10};
    Matrix<double, DirectedMatrixTag> answer(1, 4);
    buildmatrix(answer, i_answer, j_answer, v_answer);

    col_reduce(A, C,
               graphblas::PlusMonoid<double>(),
               graphblas::math::Accum<double>());

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(masked_col_reduce_test_default_accum)
{
    // Build some matrices.
    graphblas::IndexArrayType i_A    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    graphblas::IndexArrayType j_A    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_A = {1, 1, 1, 2, 2, 1, 3, 5, 3, 4};
    Matrix<double, DirectedMatrixTag> A(4, 4);
    buildmatrix(A, i_A, j_A, v_A);

    graphblas::IndexArrayType i_C    = {0, 0, 0, 0};
    graphblas::IndexArrayType j_C    = {0, 1, 2, 3};
    std::vector<double> v_C = {1, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> C(1, 4);
    buildmatrix(C, i_C, j_C, v_C);

    graphblas::IndexArrayType i_mask    = {0, 0, 0, 0};
    graphblas::IndexArrayType j_mask    = {0, 1, 2, 3};
    std::vector<double> v_mask = {1, 0, 1, 0};
    Matrix<double, DirectedMatrixTag> mask(1, 4);
    buildmatrix(mask, i_mask, j_mask, v_mask);

    graphblas::IndexArrayType i_answer    = {0, 0, 0, 0};
    graphblas::IndexArrayType j_answer    = {0, 1, 2, 3};
    std::vector<double> v_answer = {2, 1, 8, 1};
    Matrix<double, DirectedMatrixTag> answer(1, 4);
    buildmatrix(answer, i_answer, j_answer, v_answer);

    colReduceMasked(A, C, mask);

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************

BOOST_AUTO_TEST_CASE(masked_row_reduce_test_default_accum)
{
    // Build some matrices.
    graphblas::IndexArrayType i_A    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    graphblas::IndexArrayType j_A    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_A = {1, 1, 1, 2, 2, 1, 3, 5, 3, 4};
    Matrix<double, DirectedMatrixTag> A(4, 4);
    buildmatrix(A, i_A, j_A, v_A);

    graphblas::IndexArrayType i_C    = {0, 1, 2, 3};
    graphblas::IndexArrayType j_C    = {0, 0, 0, 0};
    std::vector<double> v_C = {1, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> C(4, 1);
    buildmatrix(C, i_C, j_C, v_C);

    graphblas::IndexArrayType i_mask    = {0, 1, 2, 3};
    graphblas::IndexArrayType j_mask    = {0, 0, 0, 0};
    std::vector<double> v_mask = {1, 0, 1, 0};
    Matrix<double, DirectedMatrixTag> mask(4, 1);
    buildmatrix(mask, i_mask, j_mask, v_mask);


    graphblas::IndexArrayType i_answer    = {0, 1, 2, 3};
    graphblas::IndexArrayType j_answer    = {0, 0, 0, 0};
    std::vector<double> v_answer = {2, 1, 9, 1};
    Matrix<double, DirectedMatrixTag> answer(4, 1);
    buildmatrix(answer, i_answer, j_answer, v_answer);

    rowReduceMasked(A, C, mask);

    BOOST_CHECK_EQUAL(C, answer);
}

BOOST_AUTO_TEST_SUITE_END()
