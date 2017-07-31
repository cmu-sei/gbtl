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

#include <graphblas/graphblas.hpp>

using namespace GraphBLAS;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE extract_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(extract_suite)

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_test_bad_dimensions)
{
    IndexArrayType i    = {0, 0, 0, 1, 1, 1, 2, 2};
    IndexArrayType j    = {1, 2, 3, 0, 2, 3, 0, 1};
    std::vector<double> v = {1, 2, 3, 4, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> m1(3, 4);
    m1.build(i, j, v);

    Matrix<double, DirectedMatrixTag> c(2,3);

    IndexArrayType vect_I({0, 2});
    IndexArrayType vect_J({0, 4, 1, 2});

    // nvcc requires that the acccumulator be explicitly specified to compile.
    BOOST_CHECK_THROW(extract(c, NoMask(), NoAccumulate(), m1, vect_I, vect_J),
                      DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_test_no_accum)
{
    IndexArrayType i_m1    = {0, 0, 0, 1, 1, 1, 2, 2};
    IndexArrayType j_m1    = {1, 2, 3, 0, 2, 3, 0, 1};
    std::vector<double> v_m1 = {1, 2, 3, 4, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> m1(3, 4);
    m1.build(i_m1, j_m1, v_m1);


    IndexArrayType vect_I({0,2});
    IndexArrayType vect_J({0,1,3});

    Matrix<double, DirectedMatrixTag> c(2,3);

    IndexArrayType i_result    = {0, 0, 1, 1};
    IndexArrayType j_result    = {1, 2, 0, 1};
    std::vector<double> v_result = {1, 3, 8, 9};
    Matrix<double, DirectedMatrixTag> result(2, 3);
    result.build(i_result, j_result, v_result);

    // nvcc requires that the acccumulator be explicitly specified to compile.
    extract(c, NoMask(), NoAccumulate(), m1, vect_I, vect_J);

    BOOST_CHECK_EQUAL(c, result);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_test_accum)
{
    IndexArrayType i_A    = {0, 0, 0, 1, 1, 1, 2, 2};
    IndexArrayType j_A    = {1, 2, 3, 0, 2, 3, 0, 1};
    std::vector<double> v_A = {1, 2, 3, 4, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> A(3, 4);
    A.build(i_A, j_A, v_A);

    IndexArrayType i_C    = {0, 0, 1, 1};
    IndexArrayType j_C    = {1, 2, 0, 1};
    std::vector<double> v_C = {1, 3, 8, 9};
    Matrix<double, DirectedMatrixTag> C(2, 3);
    C.build(i_C, j_C, v_C);

    IndexArrayType i_result    = {0, 0, 1, 1};
    IndexArrayType j_result    = {1, 2, 0, 1};
    std::vector<double> v_result = {2, 6, 16, 18};
    Matrix<double, DirectedMatrixTag> result(2, 3);
    result.build(i_result, j_result, v_result);

    IndexArrayType vect_I({0,2});
    IndexArrayType vect_J({0,1,3});

    //extract(A, vect_I, vect_J, C, math::Accum<double>());
    extract(C, NoMask(), Plus<double>(), A, vect_I, vect_J);
    BOOST_CHECK_EQUAL(C, result);
}

BOOST_AUTO_TEST_SUITE_END()
