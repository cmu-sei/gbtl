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
#define BOOST_TEST_MODULE mxm_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(mxm_suite)

//****************************************************************************
// Matrix multiply
BOOST_AUTO_TEST_CASE(mxm_reg)
{
    std::vector<std::vector<double>> mat1 = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 2}};
    
    std::vector<std::vector<double>> mat2 = {{0, 1, 1},
                                             {1, 0, 1},
                                             {1, 1, 0}};
    
    std::vector<std::vector<double>> mat3 = {{7, 14, 9},
                                             {12, 10, 8},
                                             {11, 6, 13}};
    
    GraphBLAS::LilSparseMatrix<double> m1(mat1, 0);
    GraphBLAS::LilSparseMatrix<double> m2(mat2, 0);
    GraphBLAS::LilSparseMatrix<double> answer(mat3, 0);
    GraphBLAS::IndexType M = 3;
    GraphBLAS::IndexType N = 3;
    GraphBLAS::LilSparseMatrix<double> result(M, N);
    GraphBLAS::backend::mxm(m1, m2, result,
                            GraphBLAS::ArithmeticSemiring<double>(),
                            GraphBLAS::PlusMonoid<double>());
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
// Matrix multiply, bad dimensions
/*
BOOST_AUTO_TEST_CASE(mxm_bad_dimensions)
{
    std::vector<std::vector<double>> mat1 = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 2}};
    
    std::vector<std::vector<double>> mat2 = {{0, 1, 1},
                                             {1, 0, 1},
                                             {1, 1, 0},
                                             {1, 1, 1}};
    
    std::vector<std::vector<double>> mat3 = {{7, 14, 9},
                                             {12, 10, 8},
                                             {11, 6, 13}};
    
    GraphBLAS::LilSparseMatrix<double> m1(mat1, 0);
    GraphBLAS::LilSparseMatrix<double> m2(mat2, 0);
    GraphBLAS::LilSparseMatrix<double> answer(mat3, 0);
    GraphBLAS::IndexType M = 3;
    GraphBLAS::IndexType N = 3;
    GraphBLAS::LilSparseMatrix<double> result(M, N);
    std::cout << "\nMatrices built";
    BOOST_CHECK_THROW(
        (GraphBLAS::backend::mxm(m1, m2, result,
                                 GraphBLAS::ArithmeticSemiring<double>(),
                                 GraphBLAS::PlusMonoid<double>())),
        GraphBLAS::DimensionException);
    std::cout << "\nTest 2 done";
}
*/
BOOST_AUTO_TEST_SUITE_END()
