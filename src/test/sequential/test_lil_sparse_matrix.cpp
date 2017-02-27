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
#define BOOST_TEST_MODULE lil_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(lil_suite)

//****************************************************************************
// LIL basic constructor
BOOST_AUTO_TEST_CASE(lil_test_construction_basic)
{
    GraphBLAS::IndexType M = 7;
    GraphBLAS::IndexType N = 4;
    GraphBLAS::LilSparseMatrix<double> m1(M, N);

    GraphBLAS::IndexType num_rows, num_cols;
    m1.nrows(num_rows);
    m1.ncols(num_cols);
    BOOST_CHECK_EQUAL(num_rows, M);
    BOOST_CHECK_EQUAL(num_cols, N);
}


//****************************************************************************
// LIL constructor from dense matrix
BOOST_AUTO_TEST_CASE(lil_test_construction_dense)
{
    std::vector<std::vector<double>> mat = {{6, 0, 0, 4},
                                            {7, 0, 0, 0},
                                            {0, 0, 9, 4},
                                            {2, 5, 0, 3},
                                            {2, 0, 0, 1},
                                            {0, 0, 0, 0},
                                            {0, 1, 0, 2}};
    
    GraphBLAS::LilSparseMatrix<double> m1(mat);
    
    GraphBLAS::IndexType M = mat.size();
    GraphBLAS::IndexType N = mat[0].size();
    GraphBLAS::IndexType num_rows, num_cols;
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

//****************************************************************************
// LIL constructor from dense matrix, implied zeros
BOOST_AUTO_TEST_CASE(lil_test_construction_dense_zero)
{
    std::vector<std::vector<double>> mat = {{6, 0, 0, 4},
                                            {7, 0, 0, 0},
                                            {0, 0, 9, 4},
                                            {2, 5, 0, 3},
                                            {2, 0, 0, 1},
                                            {0, 0, 0, 0},
                                            {0, 1, 0, 2}};
    
    GraphBLAS::LilSparseMatrix<double> m1(mat, 0);
    
    GraphBLAS::IndexType M = mat.size();
    GraphBLAS::IndexType N = mat[0].size();
    GraphBLAS::IndexType num_rows, num_cols;
    m1.nrows(num_rows);
    m1.ncols(num_cols);
    BOOST_CHECK_EQUAL(num_rows, M);
    BOOST_CHECK_EQUAL(num_cols, N);
    for (graphblas::IndexType i = 0; i < M; i++)
    {
        for (graphblas::IndexType j = 0; j < N; j++)
        {
            if (mat[i][j] != 0)
            {
                BOOST_CHECK_EQUAL(m1.get_value_at(i, j), mat[i][j]);
            }
        }
    }
}

//****************************************************************************
// LIL constructor from copy
BOOST_AUTO_TEST_CASE(lil_test_construction_copy)
{
    std::vector<std::vector<double>> mat = {{6, 0, 0, 4},
                                            {7, 0, 0, 0},
                                            {0, 0, 9, 4},
                                            {2, 5, 0, 3},
                                            {2, 0, 0, 1},
                                            {0, 0, 0, 0},
                                            {0, 1, 0, 2}};
    
    GraphBLAS::LilSparseMatrix<double> m1(mat, 0);
    
    GraphBLAS::LilSparseMatrix<double> m2(m1);
    
    BOOST_CHECK_EQUAL(m1, m2);
}
 
//****************************************************************************
// Assignment to empty location
BOOST_AUTO_TEST_CASE(lil_test_assign_to_implied_zero)
{
    std::vector<std::vector<double>> mat = {{6, 0, 0, 4},
                                            {7, 0, 0, 0},
                                            {0, 0, 9, 4},
                                            {2, 5, 0, 3},
                                            {2, 0, 0, 1},
                                            {0, 0, 0, 0},
                                            {0, 1, 0, 2}};
    
    GraphBLAS::LilSparseMatrix<double> m1(mat, 0);
    
    mat[0][1] = 8;
    m1.set_value_at(0, 1, 8);
    BOOST_CHECK_EQUAL(m1.get_value_at(0, 1), mat[0][1]);
    
    GraphBLAS::LilSparseMatrix<double> m2(mat, 0);
    BOOST_CHECK_EQUAL(m1, m2);
}

//****************************************************************************
// Assignment to a location with a previous value
BOOST_AUTO_TEST_CASE(lil_test_assign_to_nonzero_element)
{
    std::vector<std::vector<double>> mat = {{6, 0, 0, 4},
                                            {7, 0, 0, 0},
                                            {0, 0, 9, 4},
                                            {2, 5, 0, 3},
                                            {2, 0, 0, 1},
                                            {0, 0, 0, 0},
                                            {0, 1, 0, 2}};
    
    GraphBLAS::LilSparseMatrix<double> m1(mat, 0);
    
    mat[0][0] = 8;
    m1.set_value_at(0, 0, 8);
    BOOST_CHECK_EQUAL(m1.get_value_at(0, 0), mat[0][0]);
    
    GraphBLAS::LilSparseMatrix<double> m2(mat, 0);
    BOOST_CHECK_EQUAL(m1, m2);
}
 
BOOST_AUTO_TEST_SUITE_END()
