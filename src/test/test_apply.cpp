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

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE apply_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(apply_suite)

//****************************************************************************
BOOST_AUTO_TEST_CASE(apply_test_bad_dimension)
{
    // Build some sparse matrices.
    GraphBLAS::IndexArrayType i    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    GraphBLAS::IndexArrayType j    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> A(4, 4);
    A.build(i, j, v);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> C(3, 4);

    BOOST_CHECK_THROW(
        (apply(C,
               GraphBLAS::NoMask(),
               GraphBLAS::NoAccumulate(),
               GraphBLAS::MultiplicativeInverse<double>(),
               A)),
        GraphBLAS::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(apply_test_default_accum)
{
    // Build some matrices.
    GraphBLAS::IndexArrayType i = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    GraphBLAS::IndexArrayType j = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> A(4, 4);
    A.build(i, j, v);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> C(4, 4);

    std::vector<double> v_answer = {1, 1, 1, 1./2, 1./2,
                                    1./2, 1./3, 1./3, 1./3, 1./4};

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(4, 4);
    answer.build(i, j, v_answer);

    GraphBLAS::apply(C,
                     GraphBLAS::NoMask(),
                     GraphBLAS::NoAccumulate(),
                     GraphBLAS::MultiplicativeInverse<double>(),
                     A);
    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(apply_test_assign_accum)
{
    // Build some matrices.
    GraphBLAS::IndexArrayType i = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    GraphBLAS::IndexArrayType j = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> A(4, 4);
    A.build(i, j, v);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> C(4, 4);

    std::vector<double> v_answer = {1, 1, 1, 1./2, 1./2,
                                    1./2, 1./3, 1./3, 1./3, 1./4};

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(4, 4);
    answer.build(i, j, v_answer);

    GraphBLAS::apply(C,
                     GraphBLAS::NoMask(),
                     GraphBLAS::Second<double>(),
                     GraphBLAS::MultiplicativeInverse<double>(),
                     A);
    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(apply_test_accum)
{
    // Build some matrices.
    GraphBLAS::IndexArrayType i_A    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    GraphBLAS::IndexArrayType j_A    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_A = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> A(4, 4);
    A.build(i_A, j_A, v_A);

    GraphBLAS::IndexArrayType i_C    = {3, 3, 3};
    GraphBLAS::IndexArrayType j_C    = {0, 2, 3};
    std::vector<double> v_C = {9, 1, 1};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> C(4, 4);
    C.build(i_C, j_C, v_C);


    GraphBLAS::IndexArrayType i_answer = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3};
    GraphBLAS::IndexArrayType j_answer = {0, 1, 0, 1, 2, 1, 2, 3, 0, 2, 3};
    std::vector<double> v_answer = {1, 1, 1, 1./2, 1./2,
                                    1./2, 1./3, 1./3, 9, 4./3, 5./4};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(4, 4);
    answer.build(i_answer, j_answer, v_answer);

    GraphBLAS::apply(C,
                     GraphBLAS::NoMask(),
                     GraphBLAS::Plus<double>(),
                     GraphBLAS::MultiplicativeInverse<double>(),
                     A);
    BOOST_CHECK_EQUAL(C, answer);
}

/// @todo test mask/replace combinations

BOOST_AUTO_TEST_SUITE_END()
