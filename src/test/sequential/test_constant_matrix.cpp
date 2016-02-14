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

#include <iostream>

#include <graphblas/graphblas.hpp>

using namespace graphblas;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE constant_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(constant_suite)

BOOST_AUTO_TEST_CASE(constant_test_construction)
{
    graphblas::IndexType M = 4;
    graphblas::IndexType N = 3;
    double value = 6.8;
    graphblas::ConstantMatrix<double> m1(M, N, value);

    graphblas::IndexType num_rows, num_cols;
    m1.get_shape(num_rows, num_cols);
    BOOST_CHECK_EQUAL(num_rows, M);
    BOOST_CHECK_EQUAL(num_cols, N);

    for (graphblas::IndexType i = 0; i < M; i++)
    {
        for (graphblas::IndexType j = 0; j < N; j++)
        {
            BOOST_CHECK_EQUAL(m1[i][j], value);
            BOOST_CHECK_EQUAL(m1.get_value_at(i, j), value);
        }
    }
}

BOOST_AUTO_TEST_CASE(constant_test_copy_construction)
{
    graphblas::IndexType M = 4;
    graphblas::IndexType N = 3;
    double value = 6.8;
    graphblas::ConstantMatrix<double> m1(M, N, value);
    graphblas::ConstantMatrix<double> m2(m1);

    graphblas::IndexType num_rows, num_cols;
    m2.get_shape(num_rows, num_cols);
    BOOST_CHECK_EQUAL(num_rows, M);
    BOOST_CHECK_EQUAL(num_cols, N);

    for (graphblas::IndexType i = 0; i < M; i++)
    {
        for (graphblas::IndexType j = 0; j < N; j++)
        {
            BOOST_CHECK_EQUAL(m2[i][j], value);
            BOOST_CHECK_EQUAL(m2.get_value_at(i, j), value);
        }
    }
}

BOOST_AUTO_TEST_CASE(constant_test_assign)
{
    graphblas::IndexType M = 4;
    graphblas::IndexType N = 3;
    double value = 6.8;
    graphblas::ConstantMatrix<double> m1(M, N, value);
    graphblas::ConstantMatrix<double> m2(N, M, -value);

    graphblas::IndexType num_rows, num_cols;
    m2.get_shape(num_rows, num_cols);
    BOOST_CHECK_EQUAL(num_rows, N);
    BOOST_CHECK_EQUAL(num_cols, M);

    for (graphblas::IndexType i = 0; i < M; i++)
    {
        for (graphblas::IndexType j = 0; j < N; j++)
        {
            BOOST_CHECK_EQUAL(m2[j][i], -value);
            BOOST_CHECK_EQUAL(m2.get_value_at(j, i), -value);
        }
    }

    m2 = m1;

    m2.get_shape(num_rows, num_cols);
    BOOST_CHECK_EQUAL(num_rows, M);
    BOOST_CHECK_EQUAL(num_cols, N);

    for (graphblas::IndexType i = 0; i < M; i++)
    {
        for (graphblas::IndexType j = 0; j < N; j++)
        {
            BOOST_CHECK_EQUAL(m2[i][j], value);
            BOOST_CHECK_EQUAL(m2.get_value_at(i, j), value);
        }
    }
}

// Assignment to a location with a previous value
BOOST_AUTO_TEST_CASE(constant_test_assign_to_nonzero_element)
{
    graphblas::IndexType M = 4;
    graphblas::IndexType N = 3;
    double value = 6.8;
    graphblas::ConstantMatrix<double> m1(M, N, value);

    double new_value = 15.4;
    m1.set_value_at(0, 0, new_value);

    for (graphblas::IndexType i = 0; i < M; i++)
    {
        for (graphblas::IndexType j = 0; j < N; j++)
        {
            BOOST_CHECK_EQUAL(m1.get_value_at(i, j), new_value);
        }
    }

    new_value = 21.9;
    m1[0][0] = new_value;

    for (graphblas::IndexType i = 0; i < M; i++)
    {
        for (graphblas::IndexType j = 0; j < N; j++)
        {
            BOOST_CHECK_EQUAL(m1.get_value_at(i, j), new_value);
        }
    }
}

// Removing a value
BOOST_AUTO_TEST_CASE(constant_test_create_a_structural_zero)
{
    graphblas::IndexType M = 4;
    graphblas::IndexType N = 3;
    double value = 6.8;
    graphblas::ConstantMatrix<double> m1(M, N, value);

    double new_value = 15.4;
    m1.set_value_at(0, 0, m1.get_zero());

    for (graphblas::IndexType i = 0; i < M; i++)
    {
        for (graphblas::IndexType j = 0; j < N; j++)
        {
            BOOST_CHECK_EQUAL(m1.get_value_at(i, j), m1.get_zero());
        }
    }
}

BOOST_AUTO_TEST_CASE(constant_test_get_rows)
{
    graphblas::IndexType M = 4;
    graphblas::IndexType N = 3;
    double value = 6.8;
    graphblas::ConstantMatrix<double> m1(M, N, value);

    graphblas::IndexType row_index = 2;
    auto mat_row2a = m1.get_row(row_index);
    auto mat_row2b = m1[row_index];

    BOOST_CHECK_EQUAL(mat_row2a.size(), N);
    BOOST_CHECK_EQUAL(mat_row2b.size(), N);

    for (graphblas::IndexType j = 0; j < N; ++j)
    {
        BOOST_CHECK_EQUAL(mat_row2a[j], m1.get_value_at(row_index, j));
        BOOST_CHECK_EQUAL(mat_row2b[j], m1.get_value_at(row_index, j));
    }
}

BOOST_AUTO_TEST_SUITE_END()
