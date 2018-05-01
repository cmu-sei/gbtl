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

#define GRAPHBLAS_LOGGING_LEVEL 2

#include <graphblas/graphblas.hpp>

using namespace GraphBLAS;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE extract_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(extract_suite)

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_test_bad_dimensions)
{
    IndexArrayType i    = {0, 0, 0, 1, 1, 1, 2, 2};
    IndexArrayType j    = {1, 2, 3, 0, 2, 3, 0, 1};
    std::vector<double> v = {1, 2, 3, 4, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> A(3, 4);
    A.build(i, j, v);

    Matrix<double, DirectedMatrixTag> C(2, 3);

    IndexArrayType vect_I({0, 2});
    IndexArrayType vect_J({0, 4, 1, 2});

    // Standard matrix version:
    // 1. nrows(C) != nrows(M)

    // 2. ncols(C) != ncols(M)

    // 3. nrows(C) != |I|

    // 4. ncols(C) != |J|
    BOOST_CHECK_THROW(extract(C, NoMask(), NoAccumulate(), A, vect_I, vect_J),
                      DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_test_no_accum)
{
    IndexArrayType i_A    = {0, 0, 0, 1, 1, 1, 2, 2};
    IndexArrayType j_A    = {1, 2, 3, 0, 2, 3, 0, 1};
    std::vector<double> v_A = {1, 2, 3, 4, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> A(3, 4);
    A.build(i_A, j_A, v_A);


    IndexArrayType vect_I({0,2});
    IndexArrayType vect_J({0,1,3});

    Matrix<double, DirectedMatrixTag> C(2,3);

    IndexArrayType i_result    = {0, 0, 1, 1};
    IndexArrayType j_result    = {1, 2, 0, 1};
    std::vector<double> v_result = {1, 3, 8, 9};
    Matrix<double, DirectedMatrixTag> result(2, 3);
    result.build(i_result, j_result, v_result);

    extract(C, NoMask(), NoAccumulate(), A, vect_I, vect_J);

    BOOST_CHECK_EQUAL(C, result);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_test_accum)
{
    IndexArrayType i_A    = {0, 0, 0, 1, 1, 1, 2, 2};
    IndexArrayType j_A    = {1, 2, 3, 0, 2, 3, 0, 1};
    std::vector<double> v_A = {1, 2, 3, 4, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> A(3, 4);
    A.build(i_A, j_A, v_A);

    IndexArrayType i_C    = {0, 0, 1, 1};
    IndexArrayType j_C    = {1, 2, 0, 1};
    std::vector<double> v_C = {1, 3, 8, 9};
    Matrix<double, DirectedMatrixTag> C(2, 3);
    C.build(i_C, j_C, v_C);

    IndexArrayType i_result    = {0, 0, 1, 1};
    IndexArrayType j_result    = {1, 2, 0, 1};
    std::vector<double> v_result = {2, 6, 16, 18};
    Matrix<double, DirectedMatrixTag> result(2, 3);
    result.build(i_result, j_result, v_result);

    IndexArrayType vect_I({0,2});
    IndexArrayType vect_J({0,1,3});

    //extract(A, vect_I, vect_J, C, math::Accum<double>());
    extract(C, NoMask(), Plus<double>(), A, vect_I, vect_J);
    BOOST_CHECK_EQUAL(C, result);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_extract_4_3_6_1_base)
{
    std::vector<double> vecU = {1,2,3,4,5,6};
    GraphBLAS::Vector<double> vU(vecU);

    std::vector<double> vecAnswer = {1,3,5};
    GraphBLAS::Vector<double> answer(vecAnswer);

    // Output space
    GraphBLAS::IndexType M = 3;

    IndexArrayType vec_indices = {0, 2, 4};
    GraphBLAS::IndexArrayType indices(vec_indices);

    GraphBLAS::Vector<double> result(M);

    GraphBLAS::extract(result,
                       GraphBLAS::NoMask(),
                       GraphBLAS::NoAccumulate(),
                       vU,
                       indices);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_extract_4_3_6_1_bounds_check1)
{
    std::vector<double> vecU = {1,2,3,4,5,6};
    GraphBLAS::Vector<double> vU(vecU);

    std::vector<double> vecAnswer = {1,3,5};
    GraphBLAS::Vector<double> answer(vecAnswer);

    // Output space
    GraphBLAS::IndexType M = 3;

    IndexArrayType vec_indices = {0, 2, 4};
    GraphBLAS::IndexArrayType indices(vec_indices);

    GraphBLAS::Vector<double> result(M);

    // =======
    // Check w.size != mask.size
    GraphBLAS::Vector<double> bad_result(M-1);

    std::vector<bool> vecFullMask = {true, true, true};
    GraphBLAS::Vector<bool> mask(vecFullMask);


    BOOST_CHECK_THROW(
            (GraphBLAS::extract(bad_result,
                       mask,
                       GraphBLAS::NoAccumulate(),
                       vU,
                       indices)),
        DimensionException);

    // =======

    // Also check nindices mismatch
    IndexArrayType bad_vec_indices = {2, 4};
    GraphBLAS::IndexArrayType bad_indices(bad_vec_indices);

    BOOST_CHECK_THROW(
            (GraphBLAS::extract(result,
                                GraphBLAS::NoMask(),
                                GraphBLAS::NoAccumulate(),
                                vU,
                                bad_indices)),
            DimensionException);
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_extract_base)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 2}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(matA, 0);

    std::vector<std::vector<double>> matAnswer = {{1, 6},
                                                  {9, 2}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(matAnswer, 0);

    // Output space
    GraphBLAS::IndexType M = 2;
    GraphBLAS::IndexType N = 2;

    IndexArrayType vec_row_indices = {0, 2};
    IndexArrayType vec_col_indices = {1, 2};
    GraphBLAS::IndexArrayType row_indices(vec_row_indices);
    GraphBLAS::IndexArrayType col_indices(vec_col_indices);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(M, N);

    GraphBLAS::extract(result,
                       GraphBLAS::NoMask(),
                       GraphBLAS::NoAccumulate(),
                       mA,
                       row_indices,
                       col_indices);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_extract_duplicate)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 2}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(matA, 0);

    std::vector<std::vector<double>> matAnswer = {{1, 1, 6},
                                                  {9, 9, 2},
                                                  {9, 9, 2}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(matAnswer, 0);

    // Output space
    GraphBLAS::IndexType M = 3;
    GraphBLAS::IndexType N = 3;

    GraphBLAS::Matrix<bool, GraphBLAS::DirectedMatrixTag> mask(M, N);

    IndexArrayType vec_row_indices = {0, 2, 2};
    IndexArrayType vec_col_indices = {1, 1, 2};
    GraphBLAS::IndexArrayType row_indices(vec_row_indices);
    GraphBLAS::IndexArrayType col_indices(vec_col_indices);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(M, N);

    GraphBLAS::extract(result,
                       GraphBLAS::NoMask(),
                       GraphBLAS::NoAccumulate(),
                       mA,
                       row_indices,
                       col_indices);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_extract_permute)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 2}};

    std::vector<std::vector<double>> matAnswer = {{9, 2, 4},
                                                  {1, 6, 8},
                                                  {5, 7, 3}};


    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(matA, 0);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(matAnswer, 0);

    // Output space
    GraphBLAS::IndexType M = 3;
    GraphBLAS::IndexType N = 3;


    IndexArrayType vec_row_indices = {2, 0, 1};
    IndexArrayType vec_col_indices = {1, 2, 0};
    GraphBLAS::IndexArrayType row_indices(vec_row_indices);
    GraphBLAS::IndexArrayType col_indices(vec_col_indices);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(M, N);

    GraphBLAS::extract(result,
                       GraphBLAS::NoMask(),
                       GraphBLAS::NoAccumulate(),
                       mA,
                       row_indices,
                       col_indices);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************

BOOST_AUTO_TEST_CASE(sparse_extract_mask)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 2}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(matA, 0);

    std::vector<std::vector<double>> matAnswer = {{1, 0},
                                                  {9, 2}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(matAnswer, 0);

    IndexArrayType vec_row_indices = {0, 2};
    IndexArrayType vec_col_indices = {1, 2};
    GraphBLAS::IndexArrayType row_indices(vec_row_indices);
    GraphBLAS::IndexArrayType col_indices(vec_col_indices);

    // Output space
    GraphBLAS::IndexType M = 2;
    GraphBLAS::IndexType N = 2;

    std::vector<std::vector<bool>> matMask = {{true, false},
                                              {true, true}};

    GraphBLAS::Matrix<bool, GraphBLAS::DirectedMatrixTag> mask(matMask, false);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(M, N);

    GraphBLAS::extract(result,
                       mask,
                       GraphBLAS::NoAccumulate(),
                       mA,
                       row_indices,
                       col_indices);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************

BOOST_AUTO_TEST_CASE(sparse_extract_mask_replace)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(matA, 0);

    std::vector<std::vector<double>> matAnswer = {{1, 0},
                                                  {9, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(matAnswer, 0);

    IndexArrayType vec_row_indices = {0, 2};
    IndexArrayType vec_col_indices = {1, 2};
    GraphBLAS::IndexArrayType row_indices(vec_row_indices);
    GraphBLAS::IndexArrayType col_indices(vec_col_indices);

    // Output space
    GraphBLAS::IndexType M = 2;
    GraphBLAS::IndexType N = 2;

    std::vector<std::vector<bool>> matMask = {{true, false},
                                              {true, true}};

    GraphBLAS::Matrix<bool, GraphBLAS::DirectedMatrixTag> mask(matMask, false);

    // Replace with mask overwrites everything so we shouldn't see any of these.
    std::vector<std::vector<double>> matResult = {{20, 20},
                                                  {20, 20}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(matResult, 0);

    GraphBLAS::extract(result,
                       mask,
                       GraphBLAS::NoAccumulate(),
                       mA,
                       row_indices,
                       col_indices,
                       true);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************

BOOST_AUTO_TEST_CASE(sparse_extract_accum)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 2}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(matA, 0);

    std::vector<std::vector<double>> matAnswer = {{21, 26},
                                                  {29, 22}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(matAnswer, 0);

    // Output space
    GraphBLAS::IndexType M = 2;
    GraphBLAS::IndexType N = 2;

    IndexArrayType vec_row_indices = {0, 2};
    IndexArrayType vec_col_indices = {1, 2};
    GraphBLAS::IndexArrayType row_indices(vec_row_indices);
    GraphBLAS::IndexArrayType col_indices(vec_col_indices);

    std::vector<std::vector<double>> matResult = {{20, 20},
                                                  {20, 20}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(matResult, 0);

    GraphBLAS::extract(result,
                       GraphBLAS::NoMask(),
                       GraphBLAS::Plus<double>(),
                       mA,
                       row_indices,
                       col_indices);

    BOOST_CHECK_EQUAL(result, answer);
}


//****************************************************************************

BOOST_AUTO_TEST_CASE(sparse_extract_column)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 2}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(matA, 0);

    std::vector<double> vecAnswer = {1,9};
    GraphBLAS::Vector<double> answer(vecAnswer, 0);

    // Output space
    GraphBLAS::IndexType M = 2;

    IndexArrayType vec_row_indices = {0, 2};
    GraphBLAS::IndexArrayType row_indices(vec_row_indices);

    GraphBLAS::Vector<double> result(M);

    GraphBLAS::extract(result,
                       GraphBLAS::NoMask(),
                       GraphBLAS::NoAccumulate(),
                       mA,
                       row_indices,
                       (GraphBLAS::IndexType)1);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************

BOOST_AUTO_TEST_CASE(sparse_extract_column_all)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 2}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(matA, 0);

    std::vector<double> vecAnswer = {1,5,9};
    GraphBLAS::Vector<double> answer(vecAnswer, 0);

    GraphBLAS::Vector<double> result(3);

    GraphBLAS::extract(result,
                       GraphBLAS::NoMask(),
                       GraphBLAS::NoAccumulate(),
                       mA,
                       GraphBLAS::AllIndices(),
                       1,
                       false);

    BOOST_CHECK_EQUAL(result, answer);
}

BOOST_AUTO_TEST_SUITE_END()
