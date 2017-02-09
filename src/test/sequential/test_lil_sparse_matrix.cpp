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

using namespace GraphBLAS;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE lil_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(lil_suite)

//****************************************************************************
// LIL basic constructor
BOOST_AUTO_TEST_CASE(lil_test_construction_vector_of_vector_of_scalar)
{
    std::vector<std::vector<double> > mat = {{6, 0, 0, 4},
                                          {7, 0, 0, 0},
                                          {0, 0, 9, 4},
                                          {2, 5, 0, 3},
                                          {2, 0, 0, 1},
                                          {0, 0, 0, 0},
                                          {0, 1, 0, 2}};

    GraphBLAS::LilSparseMatrix<double> m1(mat);

    graphblas::IndexType M = mat.size();
    graphblas::IndexType N = mat[0].size();
    graphblas::IndexType num_rows, num_cols;
    m1.nrows(num_rows);
    m1.ncols(num_cols);
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
#if 1

//****************************************************************************
// Tuple constructor
/*
BOOST_AUTO_TEST_CASE(lil_test_construction_vector_of_vector_of_tuple)
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
    GraphBLAS::LilSparseMatrix<double> m2(0, mat2);

    graphblas::IndexType M = mat.size();
    graphblas::IndexType N = mat[0].size();
    graphblas::IndexType num_rows, num_cols;
    m2.nrows(num_rows);
    m2.ncols(num_cols);
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
*/
//****************************************************************************
// Assignment to an empty location
BOOST_AUTO_TEST_CASE(lil_test_assign_to_structural_zero)
{
    std::vector<std::vector<double> > mat = {{6, 0, 0, 4},
                                             {7, 0, 0, 0},
                                             {0, 0, 9, 4},
                                             {2, 5, 0, 3},
                                             {2, 0, 0, 1},
                                             {0, 0, 0, 0},
                                             {0, 1, 0, 2}};

    GraphBLAS::LilSparseMatrix<double> m1(mat);

    graphblas::IndexType M = mat.size();
    graphblas::IndexType N = mat[0].size();
    graphblas::IndexType num_rows, num_cols;
    m1.nrows(num_rows);
    m1.ncols(num_cols);
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
// Assignment to a location with a previous value
BOOST_AUTO_TEST_CASE(lil_test_assign_to_nonzero_element)
{
    std::vector<std::vector<double> > mat = {{6, 0, 0, 4},
                                          {7, 0, 0, 0},
                                          {0, 0, 9, 4},
                                          {2, 5, 0, 3},
                                          {2, 0, 0, 1},
                                          {0, 0, 0, 0},
                                          {0, 1, 0, 2}};

    GraphBLAS::LilSparseMatrix<double> m1(mat);

    graphblas::IndexType M = mat.size();
    graphblas::IndexType N = mat[0].size();
    graphblas::IndexType num_rows, num_cols;
    m1.nrows(num_rows);
    m1.ncols(num_cols);
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
// Removing a value
BOOST_AUTO_TEST_CASE(lil_test_create_a_structural_zero)
{
    std::vector<std::vector<double> > mat = {{6, 0, 0, 4},
                                          {7, 0, 0, 0},
                                          {0, 0, 9, 4},
                                          {2, 5, 0, 3},
                                          {2, 0, 0, 1},
                                          {0, 0, 0, 0},
                                          {0, 1, 0, 2}};

    GraphBLAS::LilSparseMatrix<double> m1(mat);

    graphblas::IndexType M = mat.size();
    graphblas::IndexType N = mat[0].size();
    graphblas::IndexType num_rows, num_cols;
    m1.nrows(num_rows);
    m1.ncols(num_cols);
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
BOOST_AUTO_TEST_CASE(lil_test_copy_assignment)
{
    std::vector<std::vector<double> > mat = {{6, 0, 0, 4},
                                             {7, 0, 0, 0},
                                             {0, 0, 9, 4},
                                             {2, 5, 0, 3},
                                             {2, 0, 0, 1},
                                             {0, 0, 0, 0},
                                             {0, 1, 0, 2}};

    GraphBLAS::LilSparseMatrix<double> m1(mat);
    graphblas::IndexType M, N;
    m1.nrows(M);
    m1.ncols(N);
    GraphBLAS::LilSparseMatrix<double> m2(M, N);

    m2 = m1;

    BOOST_CHECK_EQUAL(m1, m2);
/*    for (graphblas::IndexType i = 0; i < M; i++)
    {
        for (graphblas::IndexType j = 0; j < N; j++)
        {
            BOOST_CHECK_EQUAL(m1.get_value_at(i, j), m2[i][j]);
        }
    }
*/
}
#endif
BOOST_AUTO_TEST_SUITE_END()
