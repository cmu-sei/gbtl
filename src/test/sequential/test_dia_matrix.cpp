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
#define BOOST_TEST_MODULE DiaMatrix_suite

#include <boost/test/included/unit_test.hpp>

/**
* TODO: Increase test coverage when we actually decide whether or not this matrix is useful
*
*/
BOOST_AUTO_TEST_SUITE(DiaMatrix_suite)

BOOST_AUTO_TEST_CASE(DiaMatrix_test_rect_matrix)
{
    std::vector<std::vector<double> > mat = {{0, 1, 2, 3},
                                             {4, 0, 6, 7},
                                             {8, 9, 0, 0}};

    graphblas::DiaMatrix<double> m1(mat);
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
            BOOST_CHECK_EQUAL(m1[i][j], mat[i][j]);
        }
    }
}



BOOST_AUTO_TEST_SUITE_END()
