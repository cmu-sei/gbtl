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
BOOST_AUTO_TEST_CASE(mxm_reg_square)
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

    std::vector<std::vector<bool>>   matMask = {{true,  false, true},
                                                {false, true,  false},
                                                {true,  false, true}};
    GraphBLAS::LilSparseMatrix<bool> mask(matMask, 0);

//    GraphBLAS::backend::mxm_v3(result,
//                               GraphBLAS::LilSparseNoMask(),
//                               GraphBLAS::NoAccumulate(),               // accum
//                               GraphBLAS::ArithmeticSemiring<double>(), // op
//                               m1,
//                               m2);

    GraphBLAS::backend::mxm_v3(result,
                               GraphBLAS::LilSparseNoMask(),
                               GraphBLAS::NoAccumulate(),               // accum
                               GraphBLAS::ArithmeticSemiring<double>(), // op
                               m1,
                               m2);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
// Matrix multiply with a rectangular matrix
BOOST_AUTO_TEST_CASE(mxm_reg_rect)
{
    std::vector<std::vector<double>> mat1 = {{8, 1},
                                             {3, 5},
                                             {4, 9}};

    std::vector<std::vector<double>> mat2 = {{0, 1, 1},
                                             {1, 0, 1}};

    std::vector<std::vector<double>> mat3 = {{1, 8, 9},
                                             {5, 3, 8},
                                             {9, 4, 13}};

    GraphBLAS::LilSparseMatrix<double> m1(mat1, 0);
    GraphBLAS::LilSparseMatrix<double> m2(mat2, 0);
    GraphBLAS::LilSparseMatrix<double> answer(mat3, 0);
    GraphBLAS::IndexType M = 3;
    GraphBLAS::IndexType N = 3;
    GraphBLAS::LilSparseMatrix<double> result(M, N);

    GraphBLAS::backend::mxm_v3(result,
                               GraphBLAS::LilSparseNoMask(),
                               GraphBLAS::NoAccumulate(),                // accum
                               GraphBLAS::ArithmeticSemiring<double>(),    // semiring
                               m1,
                               m2);


    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
// Matrix multiply with a boolean mask
BOOST_AUTO_TEST_CASE(mxm_bool)
{
    std::vector<std::vector<double>> mat1 = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 2}};

    std::vector<std::vector<double>> mat2 = {{0, 1, 1},
                                             {1, 0, 1},
                                             {1, 1, 0}};

    std::vector<std::vector<bool>>   matMask = {{true,  false, true},
                                                {false, true,  false},
                                                {true,  false, true}};

    std::vector<std::vector<double>> matAnswer = {{7, 0, 9},
                                                  {0, 10, 0},
                                                  {11, 0, 13}};

    GraphBLAS::LilSparseMatrix<double> m1(mat1, 0);
    GraphBLAS::LilSparseMatrix<double> m2(mat2, 0);
    GraphBLAS::LilSparseMatrix<bool> mask(matMask, 0);
    GraphBLAS::LilSparseMatrix<double> answer(matAnswer, 0);
    GraphBLAS::IndexType M = 3;
    GraphBLAS::IndexType N = 3;
    GraphBLAS::LilSparseMatrix<double> result(M, N);

    GraphBLAS::backend::mxm_v3(result,
                               mask,
                               GraphBLAS::NoAccumulate(),                // accum
                               GraphBLAS::ArithmeticSemiring<double>(),    // semiring
                               m1,
                               m2);

    BOOST_CHECK_EQUAL(result, answer);
}


//****************************************************************************
// Matrix multiply with a boolean mask
BOOST_AUTO_TEST_CASE(mxm_bool_accum)
{
    std::vector<std::vector<double>> mat1 = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 2}};

    std::vector<std::vector<double>> mat2 = {{0, 1, 1},
                                             {1, 0, 1},
                                             {1, 1, 0}};

    std::vector<std::vector<bool>>   matMask = {{true,  false, true},
                                                {false, true,  false},
                                                {true,  false, true}};

    std::vector<std::vector<double>> matExisting = {{1, 1, 1},
                                                    {0, 0, 0},
                                                    {0, 0, 0}};

    // without pre-existing but with mask                                              
    // std::vector<std::vector<double>> matAnswer = {{7, 0, 9},
    //                                               {0, 10, 0},
    //                                               {11, 0, 13}};
    std::vector<std::vector<double>> matAnswer = {{8,  1,  10},
                                                  {0,  10, 0},
                                                  {11, 0,  13}};

    GraphBLAS::LilSparseMatrix<double> m1(mat1, 0);
    GraphBLAS::LilSparseMatrix<double> m2(mat2, 0);
    GraphBLAS::LilSparseMatrix<bool> mask(matMask, 0);
    GraphBLAS::LilSparseMatrix<double> answer(matAnswer, 0);
    GraphBLAS::LilSparseMatrix<double> result(matExisting, 0);

    GraphBLAS::backend::mxm_v3(result,
                               mask,
                               GraphBLAS::Plus<double>(),                // accum
                               GraphBLAS::ArithmeticSemiring<double>(),  // semiring
                               m1,
                               m2);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
// Matrix multiply with a boolean mask
BOOST_AUTO_TEST_CASE(mxm_bool_accum_replace)
{
    std::vector<std::vector<double>> mat1 = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 2}};

    std::vector<std::vector<double>> mat2 = {{0, 1, 1},
                                             {1, 0, 1},
                                             {1, 1, 0}};

    std::vector<std::vector<bool>>   matMask = {{true,  false, true},
                                                {false, true,  false},
                                                {true,  false, true}};

    std::vector<std::vector<double>> matExisting = {{1, 2, 3},
                                                    {4, 5, 6},
                                                    {7, 8, 9}};

    // without pre-existing                                                
    // std::vector<std::vector<double>> matAnswer = {{7, 0, 9},
    //                                               {0, 10, 0},
    //                                               {11, 0, 13}};
    std::vector<std::vector<double>> matAnswer = {{8,  0,  12},
                                                  {0,  15, 0},
                                                  {18, 0,  22}};

    GraphBLAS::LilSparseMatrix<double> m1(mat1, 0);
    GraphBLAS::LilSparseMatrix<double> m2(mat2, 0);
    GraphBLAS::LilSparseMatrix<bool> mask(matMask, 0);
    GraphBLAS::LilSparseMatrix<double> answer(matAnswer, 0);
    GraphBLAS::LilSparseMatrix<double> result(matExisting, 0);

    GraphBLAS::backend::mxm_v3(result,
                               mask,
                               GraphBLAS::Plus<double>(),                // accum
                               GraphBLAS::ArithmeticSemiring<double>(),  // semiring
                               m1,
                               m2,
                               true);

    BOOST_CHECK_EQUAL(result, answer);

}

//****************************************************************************
// Matrix multiply with a non-boolean type
BOOST_AUTO_TEST_CASE(mxm_other)
{
    std::vector<std::vector<double>> mat1 = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 2}};

    std::vector<std::vector<double>> mat2 = {{0, 1, 1},
                                             {1, 0, 1},
                                             {1, 1, 0}};

    std::vector<std::vector<double>> matMask = {{ 0.0, 1.0, 0.0},
                                                {-1.0, 0.5, -1.0},
                                                { 0.0, 1.0, 0.0}};

    std::vector<std::vector<double>> matAnswer = {{0,  14, 0},
                                                  {12, 10, 8},
                                                  {0,  6,  0}};

    GraphBLAS::LilSparseMatrix<double> m1(mat1, 0);
    GraphBLAS::LilSparseMatrix<double> m2(mat2, 0);
    GraphBLAS::LilSparseMatrix<double> mask(matMask, 0);
    GraphBLAS::LilSparseMatrix<double> answer(matAnswer, 0);
    GraphBLAS::IndexType M = 3;
    GraphBLAS::IndexType N = 3;
    GraphBLAS::LilSparseMatrix<double> result(M, N);

    GraphBLAS::backend::mxm_v3(result,
                               mask,
                               GraphBLAS::NoAccumulate(),                // accum
                               GraphBLAS::ArithmeticSemiring<double>(),    // semiring
                               m1,
                               m2);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
// Matrix multiply, empty rows and columns
BOOST_AUTO_TEST_CASE(mxm_reg_empty)
{
    std::vector<std::vector<double>> mat1 = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 2}};

    std::vector<std::vector<double>> mat2 = {{0, 0, 0},
                                             {1, 0, 1},
                                             {0, 0, 1}};

    std::vector<std::vector<double>> mat3 = {{1, 0, 7},
                                             {5, 0, 12},
                                             {9, 0, 11}};

    GraphBLAS::LilSparseMatrix<double> m1(mat1, 0);
    GraphBLAS::LilSparseMatrix<double> m2(mat2, 0);
    GraphBLAS::LilSparseMatrix<double> answer(mat3, 0);
    GraphBLAS::IndexType M = 3;
    GraphBLAS::IndexType N = 3;
    GraphBLAS::LilSparseMatrix<double> result(M, N);

    GraphBLAS::backend::mxm_v3(result,
                               GraphBLAS::LilSparseNoMask(),
                               GraphBLAS::NoAccumulate(),
                               GraphBLAS::ArithmeticSemiring<double>(),
                               m1,
                               m2);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(mxm2_empty_rows_and_columns)
{
    std::vector<std::vector<double>> mat1 = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 2}};

    std::vector<std::vector<double>> mat2 = {{0, 0, 0},
                                             {1, 0, 1},
                                             {0, 0, 1}};

    std::vector<std::vector<double>> mat3 = {{1, 0, 7},
                                             {5, 0, 12},
                                             {9, 0, 11}};

    GraphBLAS::LilSparseMatrix<double> m1(mat1, 0);
    GraphBLAS::LilSparseMatrix<double> m2(mat2, 0);
    GraphBLAS::LilSparseMatrix<double> answer(mat3, 0);
    GraphBLAS::IndexType M = 3;
    GraphBLAS::IndexType N = 3;
    GraphBLAS::LilSparseMatrix<double> result(M, N);

    GraphBLAS::backend::mxm_v3(result,
                               GraphBLAS::LilSparseNoMask(),
                               GraphBLAS::NoAccumulate(),
                               GraphBLAS::ArithmeticSemiring<double>(),
                               m1,
                               m2);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(mxm2_accum_with_empty_rows_and_columns)
{
    std::vector<std::vector<double>> mat0 = {{1, 1, 1},
                                             {1, 1, 1},
                                             {1, 1, 1}};

    std::vector<std::vector<double>> mat1 = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 2}};

    std::vector<std::vector<double>> mat2 = {{0, 0, 0},
                                             {1, 0, 1},
                                             {0, 0, 1}};

    std::vector<std::vector<double>> mat3 = {{2,  1, 8},
                                             {6,  1, 13},
                                             {10, 1, 12}};

    GraphBLAS::LilSparseMatrix<double> A(mat1, 0);
    GraphBLAS::LilSparseMatrix<double> B(mat2, 0);
    GraphBLAS::LilSparseMatrix<double> answer(mat3, 0);
    GraphBLAS::LilSparseMatrix<double> C(mat0, 0);

    GraphBLAS::backend::mxm_v3(C,
                               GraphBLAS::LilSparseNoMask(),
                               GraphBLAS::Plus<double>(),
                               GraphBLAS::ArithmeticSemiring<double>(),
                               A,
                               B);

    BOOST_CHECK_EQUAL(C, answer);
}


BOOST_AUTO_TEST_SUITE_END()
