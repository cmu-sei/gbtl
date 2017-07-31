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
#define BOOST_TEST_MODULE cpu_simple_reduce_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(cpu_simple_reduce_suite)

//****************************************************************************
BOOST_AUTO_TEST_CASE(row_reduce_test_bad_dimension)
{
    // Build some matrices.
    IndexArrayType i_A    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j_A    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_A = {1, 1, 1, 2, 2, 1, 3, 5, 3, 4};
    Matrix<double, DirectedMatrixTag> A(4, 4);
    A.build(i_A, j_A, v_A);

    IndexArrayType i_C    = {0, 1, 2};
    std::vector<double> v_C = {1, 1, 1};
    Vector<double> C(3);
    C.build(i_C, v_C);

    BOOST_CHECK_THROW(
        (reduce(C, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                GraphBLAS::PlusMonoid<double>(), A)),
        DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(row_reduce_test_default_accum)
{
    // Build some matrices.
    IndexArrayType i_A    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j_A    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_A = {1, 1, 1, 2, 2, 1, 3, 5, 3, 4};
    Matrix<double, DirectedMatrixTag> A(4, 4);
    A.build(i_A, j_A, v_A);

    IndexArrayType i_C    = {0, 1, 2, 3};
    std::vector<double> v_C = {1, 1, 1, 1};
    Vector<double> C(4);
    C.build(i_C, v_C);

    IndexArrayType i_answer    = {0, 1, 2, 3};
    std::vector<double> v_answer = {2, 5, 9, 7};
    Vector<double> answer(4);
    answer.build(i_answer, v_answer);

    reduce(C, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
           GraphBLAS::PlusMonoid<double>(), A);
    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(row_reduce_test_accum)
{
    // Build some matrices.
    IndexArrayType i_A    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j_A    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_A = {1, 1, 1, 2, 2, 1, 3, 5, 3, 4};
    Matrix<double, DirectedMatrixTag> A(4, 4);
    A.build(i_A, j_A, v_A);

    IndexArrayType i_C    = {0, 1, 2, 3};
    std::vector<double> v_C = {1, 1, 1, 1};
    Vector<double> C(4);
    C.build(i_C, v_C);

    IndexArrayType i_answer    = {0, 1, 2, 3};
    std::vector<double> v_answer = {3, 6, 10, 8};
    Vector<double> answer(4);
    answer.build(i_answer, v_answer);

    reduce(C, GraphBLAS::NoMask(), GraphBLAS::Plus<double>(),
           GraphBLAS::PlusMonoid<double>(), A);

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(col_reduce_test_bad_dimensions)
{
    // Build some matrices.
    IndexArrayType i_A    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j_A    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_A = {1, 1, 1, 2, 2, 1, 3, 5, 3, 4};
    Matrix<double, DirectedMatrixTag> A(4, 4);
    A.build(i_A, j_A, v_A);

    IndexArrayType j_C    = {0, 1, 2, 3, 4};
    std::vector<double> v_C = {1, 1, 1, 1, 1};
    Vector<double> C(5);
    C.build(j_C, v_C);

    BOOST_CHECK_THROW(
        (reduce(C, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                GraphBLAS::PlusMonoid<double>(), GraphBLAS::transpose(A))),
        DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(col_reduce_test_default_accum)
{
    // Build some matrices.
    IndexArrayType i_A    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j_A    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_A = {1, 1, 1, 2, 2, 1, 3, 5, 3, 4};
    Matrix<double, DirectedMatrixTag> A(4, 4);
    A.build(i_A, j_A, v_A);

    IndexArrayType j_C    = {0, 1, 2, 3};
    std::vector<double> v_C = {1, 1, 1, 1};
    Vector<double> C(4);
    C.build(j_C, v_C);

    IndexArrayType j_answer    = {0, 1, 2, 3};
    std::vector<double> v_answer = {2, 4, 8, 9};
    Vector<double> answer(4);
    answer.build(j_answer, v_answer);

    //col_reduce(A, C);
    reduce(C, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
           GraphBLAS::PlusMonoid<double>(), GraphBLAS::transpose(A));

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(col_reduce_test_assign)
{
    // Build some matrices.
    IndexArrayType i_A    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j_A    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_A = {1, 1, 1, 2, 2, 1, 3, 5, 3, 4};
    Matrix<double, DirectedMatrixTag> A(4, 4);
    A.build(i_A, j_A, v_A);

    IndexArrayType j_C    = {0, 1, 2, 3};
    std::vector<double> v_C = {1, 1, 1, 1};
    Vector<double> C(4);
    C.build(j_C, v_C);

    IndexArrayType j_answer    = {0, 1, 2, 3};
    std::vector<double> v_answer = {2, 4, 8, 9};
    Vector<double> answer(4);
    answer.build(j_answer, v_answer);

    //col_reduce(A, C);
    reduce(C, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
           GraphBLAS::PlusMonoid<double>(), GraphBLAS::transpose(A));

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(col_reduce_test_accum)
{
    // Build some matrices.
    IndexArrayType i_A    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j_A    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_A = {1, 1, 1, 2, 2, 1, 3, 5, 3, 4};
    Matrix<double, DirectedMatrixTag> A(4, 4);
    A.build(i_A, j_A, v_A);

    IndexArrayType j_C    = {0, 1, 2, 3};
    std::vector<double> v_C = {1, 1, 1, 1};
    Vector<double> C(4);
    C.build(j_C, v_C);

    IndexArrayType j_answer    = {0, 1, 2, 3};
    std::vector<double> v_answer = {3, 5, 9, 10};
    Vector<double> answer(4);
    answer.build(j_answer, v_answer);

    reduce(C, GraphBLAS::NoMask(), GraphBLAS::Plus<double>(),
           GraphBLAS::PlusMonoid<double>(), GraphBLAS::transpose(A));


    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(masked_col_reduce_test_default_accum)
{
    // Build some matrices.
    IndexArrayType i_A    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j_A    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_A = {1, 1, 1, 2, 2, 1, 3, 5, 3, 4};
    Matrix<double, DirectedMatrixTag> A(4, 4);
    A.build(i_A, j_A, v_A);

    IndexArrayType j_C    = {0, 1, 2, 3};
    std::vector<double> v_C = {1, 1, 1, 1};
    Vector<double> C(4);
    C.build(j_C, v_C);

    IndexArrayType j_mask    = {0, 1, 2, 3};
    std::vector<double> v_mask = {1, 0, 1, 0};
    Vector<bool> mask(4);
    mask.build(j_mask, v_mask);

    IndexArrayType j_answer    = {0, 1, 2, 3};
    std::vector<double> v_answer = {2, 1, 8, 1};
    Vector<double> answer(4);
    answer.build(j_answer, v_answer);

    //colReduceMasked(A, C, mask);
    reduce(C, mask, GraphBLAS::NoAccumulate(),
           GraphBLAS::PlusMonoid<double>(), GraphBLAS::transpose(A));

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************

BOOST_AUTO_TEST_CASE(masked_row_reduce_test_default_accum)
{
    // Build some matrices.
    IndexArrayType i_A    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j_A    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_A = {1, 1, 1, 2, 2, 1, 3, 5, 3, 4};
    Matrix<double, DirectedMatrixTag> A(4, 4);
    A.build(i_A, j_A, v_A);

    IndexArrayType i_C    = {0, 1, 2, 3};
    std::vector<double> v_C = {1, 1, 1, 1};
    Vector<double> C(4);
    C.build(i_C, v_C);

    IndexArrayType i_mask    = {0, 1, 2, 3};
    std::vector<double> v_mask = {1, 0, 1, 0};
    Vector<bool> mask(4);
    mask.build(i_mask, v_mask);

    IndexArrayType i_answer    = {0, 1, 2, 3};
    std::vector<double> v_answer = {2, 1, 9, 1};
    Vector<double> answer(4);
    answer.build(i_answer, v_answer);

    //rowReduceMasked(A, C, mask);
    reduce(C, mask, GraphBLAS::NoAccumulate(),
           GraphBLAS::PlusMonoid<double>(), A);

    BOOST_CHECK_EQUAL(C, answer);
}

BOOST_AUTO_TEST_SUITE_END()
