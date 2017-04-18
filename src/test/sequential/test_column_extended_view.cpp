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
#define BOOST_TEST_MODULE column_extended_view_suite

#include <boost/test/included/unit_test.hpp>


BOOST_AUTO_TEST_SUITE(column_extended_view_suite)

//****************************************************************************
// basic constructor
BOOST_AUTO_TEST_CASE(cev_test_construction_vector)
{
    // Build a matrix that is a column vector
    std::vector<std::vector<double> > row_vector = {{6, 1, -1, 4}};
    graphblas::CooMatrix<double> row_matrix(row_vector);

    graphblas::IndexType num_rows = 3;
    graphblas::ColumnExtendedView<graphblas::CooMatrix<double> >
        m1(0,
           row_matrix,
           num_rows);

    graphblas::IndexType M, N;
    m1.get_shape(M, N);
    BOOST_CHECK_EQUAL(num_rows, M);
    BOOST_CHECK_EQUAL(row_vector[0].size(), N);
    for (graphblas::IndexType i = 0; i < M; i++)
    {
        for (graphblas::IndexType j = 0; j < N; j++)
        {
            BOOST_CHECK_EQUAL(m1.extractElement(i, j), row_vector[0][j]);
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(cev_test_construction_row_view)
{
/*
    std::vector<std::vector<double> > mat = {{6, 0, 0, 4},
                                             {7, 0, 0, 0},
                                             {0, 0, 9, 4},
                                             {2, 5, 0, 3},
                                             {2, 0, 0, 1},
                                             {0, 0, 0, 0},
                                             {0, 1, 0, 2}};

    // Create a row view of a matrix (4th row).
    graphblas::IndexType orig_row_index = 3;
    graphblas::CooMatrix<double> matrix(mat);
    graphblas::RowView<graphblas::CooMatrix<double> >
        row_matrix = matrix[orig_row_index];

    // Replicate that column 3 times
    graphblas::IndexType num_rows = 3;
    graphblas::ColumnExtendedView<
        graphblas::RowView<
            graphblas::CooMatrix<double> > > cev_mat(0, row_matrix, num_rows);

    graphblas::IndexType M, N;
    cev_mat.get_shape(M, N);

    BOOST_CHECK_EQUAL(M, num_rows);
    BOOST_CHECK_EQUAL(N, mat[orig_row_index].size());
    for (graphblas::IndexType i = 0; i < M; i++)
    {
        for (graphblas::IndexType j = 0; j < N; j++)
        {
            BOOST_CHECK_EQUAL(cev_mat.extractElement(i, j),
                              mat[orig_row_index][j]);
        }
    }
*/
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(cev_test_construction_matrix_row_index)
{
    std::vector<std::vector<double> > mat = {{6, 0, 0, 4},
                                             {7, 0, 0, 0},
                                             {0, 0, 9, 4},
                                             {2, 5, 0, 3},
                                             {2, 0, 0, 1},
                                             {0, 0, 0, 0},
                                             {0, 1, 0, 2}};

    // Create a column view of a matrix (4th col).
    graphblas::IndexType orig_row_index = 3;
    graphblas::CooMatrix<double> matrix(mat);

    // Replicate the 4th column 3 times
    graphblas::IndexType num_rows = 3;
    graphblas::ColumnExtendedView<
        graphblas::CooMatrix<double> > cev_mat(orig_row_index,
                                               matrix,
                                               num_rows);

    graphblas::IndexType M, N;
    cev_mat.get_shape(M, N);

    BOOST_CHECK_EQUAL(M, num_rows);
    BOOST_CHECK_EQUAL(N, mat[orig_row_index].size());
    for (graphblas::IndexType i = 0; i < M; i++)
    {
        for (graphblas::IndexType j = 0; j < N; j++)
        {
            BOOST_CHECK_EQUAL(cev_mat.extractElement(i, j),
                              mat[orig_row_index][j]);
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(cev_test_modify_element)
{
    std::vector<std::vector<double> > mat = {{6, 0, 0, 4},
                                             {7, 0, 0, 0},
                                             {0, 0, 9, 4},
                                             {2, 5, 0, 3},
                                             {2, 0, 0, 1},
                                             {0, 0, 0, 0},
                                             {0, 1, 0, 2}};

    // Create a column view of a matrix (4th col).
    graphblas::IndexType orig_row_index = 3;
    graphblas::CooMatrix<double> matrix(mat);

    // Replicate the 4th column 3 times
    graphblas::IndexType num_rows = 3;
    graphblas::ColumnExtendedView<
        graphblas::CooMatrix<double> > cev_mat(orig_row_index,
                                               matrix,
                                               num_rows);


    graphblas::IndexType M, N;
    cev_mat.get_shape(M, N);

    BOOST_CHECK_EQUAL(M, num_rows);
    BOOST_CHECK_EQUAL(N, mat[orig_row_index].size());

    // **** modify the second value ****
    double new_value = 55.0;
    cev_mat.setElement(0, 1, new_value);
    mat[orig_row_index][1] = new_value;

    for (graphblas::IndexType i = 0; i < M; ++i)
    {
        for (graphblas::IndexType j = 0; j < N; ++j)
        {
            BOOST_CHECK_EQUAL(cev_mat.extractElement(i, j),
                              mat[orig_row_index][j]);
        }
    }

    // Show that it modified the underlying matrix
    matrix.get_shape(M, N);
    for (graphblas::IndexType i = 0; i < M; ++i)
    {
        for (graphblas::IndexType j = 0; j < N; ++j)
        {
            BOOST_CHECK_EQUAL(matrix.extractElement(i, j), mat[i][j]);
        }
    }
}



BOOST_AUTO_TEST_SUITE_END()
