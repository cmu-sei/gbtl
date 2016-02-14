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

using namespace graphblas;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE extract_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(extract_suite)

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_test_bad_dimensions)
{
    graphblas::IndexArrayType i    = {0, 0, 0, 1, 1, 1, 2, 2};
    graphblas::IndexArrayType j    = {1, 2, 3, 0, 2, 3, 0, 1};
    std::vector<double> v = {1, 2, 3, 4, 6, 7, 8, 9};
    graphblas::Matrix<double, DirectedMatrixTag> m1(3, 4);
    buildmatrix(m1, i, j, v);

    graphblas::Matrix<double, DirectedMatrixTag> c(2,3);

    graphblas::IndexArrayType vect_I({0, 2});
    graphblas::IndexArrayType vect_J({0, 1, 3, 2});

    // nvcc requires that the acccumulator be explicitly specified to compile.
    BOOST_CHECK_THROW(graphblas::extract(m1, vect_I, vect_J, c,
                                         graphblas::math::Assign<double>()),
                      graphblas::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_test_default_accum)
{
    graphblas::IndexArrayType i_m1    = {0, 0, 0, 1, 1, 1, 2, 2};
    graphblas::IndexArrayType j_m1    = {1, 2, 3, 0, 2, 3, 0, 1};
    std::vector<double> v_m1 = {1, 2, 3, 4, 6, 7, 8, 9};
    graphblas::Matrix<double, DirectedMatrixTag> m1(3, 4);
    buildmatrix(m1, i_m1, j_m1, v_m1);


    graphblas::IndexArrayType vect_I({0,2});
    graphblas::IndexArrayType vect_J({0,1,3});

    graphblas::Matrix<double, DirectedMatrixTag> c(2,3);

    graphblas::IndexArrayType i_result    = {0, 0, 1, 1};
    graphblas::IndexArrayType j_result    = {1, 2, 0, 1};
    std::vector<double> v_result = {1, 3, 8, 9};
    graphblas::Matrix<double, DirectedMatrixTag> result(2, 3);
    buildmatrix(result, i_result, j_result, v_result);

    // nvcc requires that the acccumulator be explicitly specified to compile.
    graphblas::extract(m1, vect_I, vect_J, c,
                       graphblas::math::Assign<double>());

    BOOST_CHECK_EQUAL(c, result);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_test_assign_accum)
{
    graphblas::IndexArrayType i_m1    = {0, 0, 0, 1, 1, 1, 2, 2};
    graphblas::IndexArrayType j_m1    = {1, 2, 3, 0, 2, 3, 0, 1};
    std::vector<double> v_m1 = {1, 2, 3, 4, 6, 7, 8, 9};
    graphblas::Matrix<double, DirectedMatrixTag> m1(3, 4);
    buildmatrix(m1, i_m1, j_m1, v_m1);


    graphblas::IndexArrayType vect_I({0,2});
    graphblas::IndexArrayType vect_J({0,1,3});

    graphblas::Matrix<double, DirectedMatrixTag> c(2,3);

    graphblas::IndexArrayType i_result    = {0, 0, 1, 1};
    graphblas::IndexArrayType j_result    = {1, 2, 0, 1};
    std::vector<double> v_result = {1, 3, 8, 9};
    graphblas::Matrix<double, DirectedMatrixTag> result(2, 3);
    buildmatrix(result, i_result, j_result, v_result);

    graphblas::extract(m1, vect_I, vect_J, c,
                       graphblas::math::Assign<double>());

    BOOST_CHECK_EQUAL(c, result);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_test_accum)
{
    graphblas::IndexArrayType i_A    = {0, 0, 0, 1, 1, 1, 2, 2};
    graphblas::IndexArrayType j_A    = {1, 2, 3, 0, 2, 3, 0, 1};
    std::vector<double> v_A = {1, 2, 3, 4, 6, 7, 8, 9};
    graphblas::Matrix<double, DirectedMatrixTag> A(3, 4);
    buildmatrix(A, i_A, j_A, v_A);

    graphblas::IndexArrayType i_C    = {0, 0, 1, 1};
    graphblas::IndexArrayType j_C    = {1, 2, 0, 1};
    std::vector<double> v_C = {1, 3, 8, 9};
    graphblas::Matrix<double, DirectedMatrixTag> C(2, 3);
    buildmatrix(C, i_C, j_C, v_C);

    graphblas::IndexArrayType i_result    = {0, 0, 1, 1};
    graphblas::IndexArrayType j_result    = {1, 2, 0, 1};
    std::vector<double> v_result = {2, 6, 16, 18};
    graphblas::Matrix<double, DirectedMatrixTag> result(2, 3);
    buildmatrix(result, i_result, j_result, v_result);

    graphblas::IndexArrayType vect_I({0,2});
    graphblas::IndexArrayType vect_J({0,1,3});

    graphblas::extract(A, vect_I, vect_J, C, graphblas::math::Accum<double>());
    BOOST_CHECK_EQUAL(C, result);
}

BOOST_AUTO_TEST_SUITE_END()
