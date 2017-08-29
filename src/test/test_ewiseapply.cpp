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
#define BOOST_TEST_MODULE ewiseaddmult_suite

#include <boost/test/included/unit_test.hpp>

static ArithmeticSemiring<double> sr;

BOOST_AUTO_TEST_SUITE(ewiseaddmult_suite)

//****************************************************************************
BOOST_AUTO_TEST_CASE(ewiseadd_bad_dimensions)
{
    IndexArrayType i_m1    = {0, 0, 1, 1, 2, 2, 3};
    IndexArrayType j_m1    = {0, 1, 1, 2, 2, 3, 3};
    std::vector<double> v_m1 = {1, 2, 2, 3, 3, 4, 4};
    Matrix<double, DirectedMatrixTag> m1(4, 4);
    m1.build(i_m1, j_m1, v_m1);

    IndexArrayType i_m2    = {0, 0, 1, 1, 1, 2, 2};
    IndexArrayType j_m2    = {0, 1, 0, 1, 2, 1, 2};
    std::vector<double> v_m2 = {2, 2, 1, 4, 4, 4, 6};
    Matrix<double, DirectedMatrixTag> m2(3, 4);
    m2.build(i_m2, j_m2, v_m2);

    Matrix<double, DirectedMatrixTag> m3(4, 4);

    BOOST_CHECK_THROW(
        eWiseAdd(m3, NoMask(), NoAccumulate(),
                 Plus<double>(), m1, m2),
        DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(eWiseAdd_normal)
{
    // Build some sparse matrices.
    IndexArrayType i_mat    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j_mat    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_mat = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4};
    Matrix<double, DirectedMatrixTag> mat(4, 4);
    mat.build(i_mat, j_mat, v_mat);

    Matrix<double, DirectedMatrixTag> m3(4, 4);

    IndexArrayType i_answer    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j_answer    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_answer = {2, 2, 2, 4, 4, 4, 6, 6, 6, 8};
    Matrix<double, DirectedMatrixTag> answer(4, 4);
    answer.build(i_answer, j_answer, v_answer);

    // Now try simple's ewiseapply.
    eWiseAdd(m3, NoMask(), NoAccumulate(), Plus<double>(), mat, mat);

    BOOST_CHECK_EQUAL(m3, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(eWiseMult_bad_dimensions)
{
    IndexArrayType i_m1    = {0, 0, 1, 1, 2, 2, 3};
    IndexArrayType j_m1    = {0, 1, 1, 2, 2, 3, 3};
    std::vector<double> v_m1 = {1, 2, 2, 3, 3, 4, 4};
    Matrix<double, DirectedMatrixTag> m1(4, 4);
    m1.build(i_m1, j_m1, v_m1);

    IndexArrayType i_m2    = {0, 0, 1, 1, 1, 2, 2};
    IndexArrayType j_m2    = {0, 1, 0, 1, 2, 1, 2};
    std::vector<double> v_m2 = {2, 2, 1, 4, 4, 4, 6};
    Matrix<double, DirectedMatrixTag> m2(3, 4);
    m2.build(i_m2, j_m2, v_m2);

    Matrix<double, DirectedMatrixTag> m3(4, 4);

    BOOST_CHECK_THROW(
        eWiseMult(m3, NoMask(), NoAccumulate(),
                  Times<double>(), m1, m2),
        DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(eWiseMult_normal)
{
    IndexArrayType i_mat    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j_mat    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_mat = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4};
    Matrix<double, DirectedMatrixTag> mat(4, 4);
    mat.build(i_mat, j_mat, v_mat);

    Matrix<double, DirectedMatrixTag> m3(4, 4);

    IndexArrayType i_answer    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j_answer    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_answer = {1, 1, 1, 4, 4, 4, 9, 9, 9, 16};
    Matrix<double, DirectedMatrixTag> answer(4, 4);
    answer.build(i_answer, j_answer, v_answer);

    eWiseMult(m3, NoMask(), NoAccumulate(),
              Times<double>(),
              mat, mat);

    BOOST_CHECK_EQUAL(m3, answer);
}

BOOST_AUTO_TEST_SUITE_END()
