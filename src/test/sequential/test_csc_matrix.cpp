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

#define GB_DEBUG
#include <graphblas/graphblas.hpp>

using namespace graphblas;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE csc_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(csc_suite)

//****************************************************************************
BOOST_AUTO_TEST_CASE(csc_constructor)
{
    std::vector<std::vector<double> > mat = {{6, 0, 0, 4},
                                             {7, 0, 0, 0},
                                             {0, 0, 9, 4},
                                             {2, 5, 0, 3},
                                             {2, 0, 0, 1},
                                             {0, 0, 0, 0},
                                             {0, 1, 0, 2}};

    graphblas::CscMatrix<double> m1(mat);

    graphblas::IndexType M = mat.size();
    graphblas::IndexType N = mat[0].size();
    graphblas::IndexType num_rows, num_cols;
    m1.get_shape(num_rows, num_cols);
    BOOST_CHECK_EQUAL(num_rows, M);
    BOOST_CHECK_EQUAL(num_cols, N);
    for (graphblas::IndexType i = 0; i < M; i++)
    {
        for (graphblas::IndexType j = 0; j < N; j++)
        {
            BOOST_CHECK_EQUAL(m1.get_value_at(i, j), mat[i][j]);
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(csc_tuple_constructor)
{
    std::vector<std::vector<double> > mat = {{6, 0, 0, 4},
                                          {7, 0, 0, 0},
                                          {0, 0, 9, 4},
                                          {2, 5, 0, 3},
                                          {2, 0, 0, 1},
                                          {0, 0, 0, 0},
                                          {0, 1, 0, 2}};

    std::vector<std::vector<std::tuple<graphblas::IndexType, double> > > mat2 = {
      {std::make_tuple(0,6),
        std::make_tuple(3,4)},
      {std::make_tuple(0,7)},
      {std::make_tuple(2,9),
        std::make_tuple(3,4)},
      {std::make_tuple(0,2),
        std::make_tuple(1,5),
        std::make_tuple(3,3)},
      {std::make_tuple(0,2),
        std::make_tuple(3,1)},
      {},
      {std::make_tuple(1,1),
        std::make_tuple(3,2)}
    };
    graphblas::CscMatrix<double> m2(0, mat2);

    graphblas::IndexType M = mat.size();
    graphblas::IndexType N = mat[0].size();
    graphblas::IndexType num_rows, num_cols;
    m2.get_shape(num_rows, num_cols);
    BOOST_CHECK_EQUAL(num_rows, M);
    BOOST_CHECK_EQUAL(num_cols, N);
    for (graphblas::IndexType i = 0; i < M; i++)
    {
        for (graphblas::IndexType j = 0; j < N; j++)
        {
            BOOST_CHECK_EQUAL(m2.get_value_at(i, j), mat[i][j]);
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(csc_assignment_empty_location)
{
    std::vector<std::vector<double> > mat = {{6, 0, 0, 4},
                                          {7, 0, 0, 0},
                                          {0, 0, 9, 4},
                                          {2, 5, 0, 3},
                                          {2, 0, 0, 1},
                                          {0, 0, 0, 0},
                                          {0, 1, 0, 2}};

    graphblas::CscMatrix<double> m1(mat);

    graphblas::IndexType M = mat.size();
    graphblas::IndexType N = mat[0].size();
    graphblas::IndexType num_rows, num_cols;
    m1.get_shape(num_rows, num_cols);
    BOOST_CHECK_EQUAL(num_rows, M);
    BOOST_CHECK_EQUAL(num_cols, N);

    mat[0][1] = 8;
    m1.set_value_at(0, 1, 8);

    for (graphblas::IndexType i = 0; i < M; i++)
    {
        for (graphblas::IndexType j = 0; j < N; j++)
        {
            BOOST_CHECK_EQUAL(m1.get_value_at(i, j), mat[i][j]);
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(csc_assignment_over_previous_value)
{
    std::vector<std::vector<double> > mat = {{6, 0, 0, 4},
                                          {7, 0, 0, 0},
                                          {0, 0, 9, 4},
                                          {2, 5, 0, 3},
                                          {2, 0, 0, 1},
                                          {0, 0, 0, 0},
                                          {0, 1, 0, 2}};

    graphblas::CscMatrix<double> m1(mat);

    graphblas::IndexType M = mat.size();
    graphblas::IndexType N = mat[0].size();
    graphblas::IndexType num_rows, num_cols;
    m1.get_shape(num_rows, num_cols);
    BOOST_CHECK_EQUAL(num_rows, M);
    BOOST_CHECK_EQUAL(num_cols, N);

    mat[0][0] = 8;
    m1.set_value_at(0, 0, 8);
    for (graphblas::IndexType i = 0; i < M; i++)
    {
        for (graphblas::IndexType j = 0; j < N; j++)
        {
            BOOST_CHECK_EQUAL(m1.get_value_at(i, j), mat[i][j]);
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(csc_double_assignment)
{
    std::vector<std::vector<double> > mat = {{6, 0, 0, 4},
                                          {7, 0, 0, 0},
                                          {0, 0, 9, 4},
                                          {2, 5, 0, 3},
                                          {2, 0, 0, 1},
                                          {0, 0, 0, 0},
                                          {0, 1, 0, 2}};

    graphblas::CscMatrix<double> m1(mat);

    graphblas::IndexType M = mat.size();
    graphblas::IndexType N = mat[0].size();
    graphblas::IndexType num_rows, num_cols;
    m1.get_shape(num_rows, num_cols);
    BOOST_CHECK_EQUAL(num_rows, M);
    BOOST_CHECK_EQUAL(num_cols, N);

    mat[1][1] = 1;
    mat[2][2] = 0;
    m1.set_value_at(1, 1, 1);
    m1.set_value_at(2, 2, 0);
    for (graphblas::IndexType i = 0; i < M; i++)
    {
        for (graphblas::IndexType j = 0; j < N; j++)
        {
            BOOST_CHECK_EQUAL(m1.get_value_at(i, j), mat[i][j]);
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(csc_remove_value)
{
    std::vector<std::vector<double> > mat = {{6, 0, 0, 4},
                                          {7, 0, 0, 0},
                                          {0, 0, 9, 4},
                                          {2, 5, 0, 3},
                                          {2, 0, 0, 1},
                                          {0, 0, 0, 0},
                                          {0, 1, 0, 2}};

    graphblas::CscMatrix<double> m1(mat);

    graphblas::IndexType M = mat.size();
    graphblas::IndexType N = mat[0].size();
    graphblas::IndexType num_rows, num_cols;
    m1.get_shape(num_rows, num_cols);
    BOOST_CHECK_EQUAL(num_rows, M);
    BOOST_CHECK_EQUAL(num_cols, N);

    mat[0][2] = 0;
    m1.set_value_at(0, 2, 0);
    for (graphblas::IndexType i = 0; i < M; i++)
    {
        for (graphblas::IndexType j = 0; j < N; j++)
        {
            BOOST_CHECK_EQUAL(m1.get_value_at(i, j), mat[i][j]);
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(csc_zero_column)
{
    std::vector<std::vector<double> > mat = {{0, 0, 0, 4},
                                          {0, 0, 0, 0},
                                          {0, 0, 9, 4},
                                          {0, 5, 0, 3},
                                          {0, 0, 0, 1},
                                          {0, 0, 0, 0},
                                          {0, 1, 0, 2}};

    graphblas::CscMatrix<double> m1(mat);

    graphblas::IndexType M = mat.size();
    graphblas::IndexType N = mat[0].size();
    graphblas::IndexType num_rows, num_cols;
    m1.get_shape(num_rows, num_cols);
    BOOST_CHECK_EQUAL(num_rows, M);
    BOOST_CHECK_EQUAL(num_cols, N);

    for (graphblas::IndexType i = 0; i < M; i++)
    {
        for (graphblas::IndexType j = 0; j < N; j++)
        {
            BOOST_CHECK_EQUAL(m1.get_value_at(i, j), mat[i][j]);
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(csc_test_copy_assignment)
{
    std::vector<std::vector<double> > mat = {{6, 0, 0, 4},
                                             {7, 0, 0, 0},
                                             {0, 0, 9, 4},
                                             {2, 5, 0, 3},
                                             {2, 0, 0, 1},
                                             {0, 0, 0, 0},
                                             {0, 1, 0, 2}};

    graphblas::CscMatrix<double> m1(mat);
    graphblas::IndexType M, N;
    m1.get_shape(M, N);
    graphblas::CscMatrix<double> m2(M, N);

    m2 = m1;

    BOOST_CHECK_EQUAL(m1, m2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(csc_test_get_rows)
{
/*
    std::vector<std::vector<double> > mat = {{6, 0, 0, 4},
                                             {7, 0, 0, 0},
                                             {0, 0, 9, 4},
                                             {2, 5, 0, 3},
                                             {2, 0, 0, 1},
                                             {0, 0, 0, 0},
                                             {0, 1, 0, 2}};

    graphblas::CscMatrix<double> m1(mat);
    graphblas::IndexType M, N;
    m1.get_shape(M, N);

    graphblas::IndexType row_index = 2;
    auto mat_row2a = m1.get_row(row_index);
    auto mat_row2b = m1[row_index];

    for (graphblas::IndexType j = 0; j < N; ++j)
    {
        BOOST_CHECK_EQUAL(mat_row2a[j], m1.get_value_at(row_index, j));
        BOOST_CHECK_EQUAL(mat_row2b[j], m1.get_value_at(row_index, j));
    }
*/
}

//****************************************************************************
// @todo add this to a TransposeView test suite (if the class survives).
BOOST_AUTO_TEST_CASE(csc_test_transpose_view)
{
    std::vector<std::vector<double> > mat = {{6, 0, 0, 4},
                                             {7, 0, 0, 0},
                                             {0, 0, 9, 4},
                                             {2, 5, 0, 3},
                                             {2, 0, 0, 1},
                                             {0, 0, 0, 0},
                                             {0, 1, 0, 2}};

    graphblas::CscMatrix<double> m1(mat);
    graphblas::IndexType M, N;
    m1.get_shape(M, N);

    graphblas::backend::TransposeView<graphblas::CscMatrix<double> > mT(m1);

    graphblas::IndexType m, n;
    mT.get_shape(m, n);

    BOOST_CHECK_EQUAL(m, N);
    BOOST_CHECK_EQUAL(M, n);

    for (graphblas::IndexType i = 0; i < M; i++)
    {
        for (graphblas::IndexType j = 0; j < N; j++)
        {
            BOOST_CHECK_EQUAL(m1.get_value_at(i, j), mT.get_value_at(j, i));
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
