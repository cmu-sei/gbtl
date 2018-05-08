/*
 * Copyright (c) 2018 Carnegie Mellon University and The Trustees of Indiana
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

#define GRAPHBLAS_LOGGING_LEVEL 0

#include <graphblas/graphblas.hpp>

using namespace GraphBLAS;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE extract_stdmat_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(extract_stdmat_suite)

//****************************************************************************
// extract standard matrix error tests
//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_stdmat_test_bad_dimensions)
{
    IndexArrayType i      = {0, 0, 0, 1, 1, 1, 2, 2};
    IndexArrayType j      = {1, 2, 3, 0, 2, 3, 0, 1};
    std::vector<double> v = {1, 2, 3, 4, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> A(3, 4);
    A.build(i, j, v);

    Matrix<double, DirectedMatrixTag> C(2, 3);


    // Standard matrix version:
    // 1. nrows(C) != nrows(M)
    {
        // Too many mask rows
        std::vector<std::vector<bool>> matMask = {{true, false, true},
                                                  {true, true,  false},
                                                  {false, true, true}};
        Matrix<bool, DirectedMatrixTag> M(matMask, false);
        IndexArrayType vect_I({0, 2});
        IndexArrayType vect_J({0, 1, 2});
        BOOST_CHECK_THROW(
            extract(C, M, NoAccumulate(), A, vect_I, vect_J),
            DimensionException);
    }

    {
        // Too few mask rows
        std::vector<std::vector<bool>> matMask = {{false, true, true}};
        Matrix<bool, DirectedMatrixTag> M(matMask, false);
        IndexArrayType vect_I({0, 2});
        IndexArrayType vect_J({0, 1, 2});
        BOOST_CHECK_THROW(
            extract(C, M, NoAccumulate(), A, vect_I, vect_J),
            DimensionException);
    }

    // 2. ncols(C) != ncols(M)
    {
        // Too many mask cols
        std::vector<std::vector<bool>> matMask = {{false, true, false, true},
                                                  {true, true, false, false}};
        Matrix<bool, DirectedMatrixTag> M(matMask, false);
        IndexArrayType vect_I({0, 2});
        IndexArrayType vect_J({0, 1, 2});
        BOOST_CHECK_THROW(
            extract(C, M, NoAccumulate(), A, vect_I, vect_J),
            DimensionException);
    }

    {
        // Too few mask cols
        std::vector<std::vector<bool>> matMask = {{false, true},
                                                  {true, true}};
        Matrix<bool, DirectedMatrixTag> M(matMask, false);
        IndexArrayType vect_I({0, 2});
        IndexArrayType vect_J({0, 1, 2});
        BOOST_CHECK_THROW(
            extract(C, M, NoAccumulate(), A, vect_I, vect_J),
            DimensionException);
    }


    // 3. nrows(C) != |I|
    {
        IndexArrayType vect_I({0, 2, 1}); // too many rows for C
        IndexArrayType vect_J({0, 1, 2});
        BOOST_CHECK_THROW(
            extract(C, NoMask(), NoAccumulate(), A, vect_I, vect_J),
            DimensionException);
    }

    {
        IndexArrayType vect_I({0}); // too few rows for C
        IndexArrayType vect_J({0, 1, 2});
        BOOST_CHECK_THROW(
            extract(C, NoMask(), NoAccumulate(), A, vect_I, vect_J),
            DimensionException);
    }

    // 4. ncols(C) != |J|
    {
        IndexArrayType vect_I({0, 2});
        IndexArrayType vect_J({0, 2, 1, 2}); // too many cols for C
        BOOST_CHECK_THROW(
            extract(C, NoMask(), NoAccumulate(), A, vect_I, vect_J),
            DimensionException);
    }

    {
        IndexArrayType vect_I({0, 2});
        IndexArrayType vect_J({0, 1}); // too many cols for C
        BOOST_CHECK_THROW(
            extract(C, NoMask(), NoAccumulate(), A, vect_I, vect_J),
            DimensionException);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_stdmat_index_out_of_bounds)
{
    IndexArrayType i    = {0, 0, 0, 1, 1, 1, 2, 2};
    IndexArrayType j    = {1, 2, 3, 0, 2, 3, 0, 1};
    std::vector<double> v = {1, 2, 3, 4, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> A(3, 4);
    A.build(i, j, v);

    Matrix<double, DirectedMatrixTag> C(2, 3);


    // Standard matrix version:
    // J index out of range
    {
        std::vector<std::vector<bool>> matMask = {{true, false, true},
                                                  {true, true,  false}};
        Matrix<bool, DirectedMatrixTag> M(matMask, false);
        IndexArrayType vect_I({0, 2});
        IndexArrayType vect_J({0, 4, 2});
        BOOST_CHECK_THROW(
            extract(C, M, NoAccumulate(), A, vect_I, vect_J),
            IndexOutOfBoundsException);

        BOOST_CHECK_THROW(
            extract(C, NoMask(), NoAccumulate(), A, vect_I, vect_J),
            IndexOutOfBoundsException);
    }

    // I index out of range
    {
        std::vector<std::vector<bool>> matMask = {{true, false, true},
                                                  {true, true,  false}};
        Matrix<bool, DirectedMatrixTag> M(matMask, false);
        IndexArrayType vect_I({3, 2});
        IndexArrayType vect_J({0, 1, 2});
        BOOST_CHECK_THROW(
            extract(C, M, NoAccumulate(), A, vect_I, vect_J),
            IndexOutOfBoundsException);
        BOOST_CHECK_THROW(
            extract(C, NoMask(), NoAccumulate(), A, vect_I, vect_J),
            IndexOutOfBoundsException);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_stdmat_test_allindices_too_small)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6, 0},
                                             {0, 5, 7, 9},
                                             {4, 0, 2, 0}};
    Matrix<double, DirectedMatrixTag> mA(matA, 0);

    // result is too small when I = AllIndices will pass
    {
        std::vector<std::vector<double>> matAnswer = {{8, 1, 6},
                                                      {0, 5, 7}};
        Matrix<double> answer(matAnswer, 0);
        Matrix<double> result(2, 3);

        extract(result, NoMask(), NoAccumulate(),
                mA, AllIndices(), AllIndices());

        BOOST_CHECK_EQUAL(result, answer);
    }

    // result is too big when I = AllIndices will pass
    {
        std::vector<std::vector<double>> matAnswer = {{8, 1, 6, 0, 0, 0},
                                                      {0, 5, 7, 9, 0, 0},
                                                      {4, 0, 2, 0, 0, 0},
                                                      {0, 0, 0, 0, 0, 0}};
        Matrix<double> answer(matAnswer, 0);
        Matrix<double> result(4, 6);

        extract(result, NoMask(), NoAccumulate(),
                mA, AllIndices(), AllIndices());

        BOOST_CHECK_EQUAL(result, answer);
    }
}

//****************************************************************************
// Standard matrix passing test cases:
//
// Simplified test structure (12 test cases total):
//
// Mask cases (x3):  nomask,  noscmp, scmp
// Accum cases (x2): noaccum, accum
// Trans cases (x2): notrans, trans
//
// Within each test case: 5 checks will be run:
//
// row_ind: AllIndices, nodup/ordered, nodup/permute, dup/ordered, dup/permute
// col_ind: AllIndices, nodup/ordered, nodup/permute, dup/ordered, dup/permute
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_stdmat_test_nomask_noaccum_notrans)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6, 0},
                                             {0, 5, 7, 9},
                                             {4, 0, 2, 0}};
    Matrix<double> A(matA, 0);

    // I,J - AllIndices
    {
        Matrix<double> C(3, 4);
        extract(C, NoMask(), NoAccumulate(), A, AllIndices(), AllIndices());

        Matrix<double> answer(A);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - no dup, ordered
    {
        IndexArrayType arrayI({0,2});
        IndexArrayType arrayJ({0,1,3});

        Matrix<double> C(2,3);
        extract(C, NoMask(), NoAccumulate(), A, arrayI, arrayJ);

        std::vector<std::vector<double>> ansMat ={{8, 1, 0},
                                                  {4, 0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - no dup, permuted
    {
        IndexArrayType arrayI({2,0});
        IndexArrayType arrayJ({3,0,1});

        Matrix<double> C(2,3);
        extract(C, NoMask(), NoAccumulate(), A, arrayI, arrayJ);

        std::vector<std::vector<double>> ansMat ={{0, 4, 0},
                                                  {0, 8, 1}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - dup, ordered
    {
        IndexArrayType arrayI({0,0,2});
        IndexArrayType arrayJ({0,1,1,3});

        Matrix<double> C(3,4);
        extract(C, NoMask(), NoAccumulate(), A, arrayI, arrayJ);

        std::vector<std::vector<double>> ansMat ={{8, 1, 1, 0},
                                                  {8, 1, 1, 0},
                                                  {4, 0, 0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - dup, permuted
    {
        IndexArrayType arrayI({2,0,2});
        IndexArrayType arrayJ({3,0,1,0});

        Matrix<double> C(3,4);
        extract(C, NoMask(), NoAccumulate(), A, arrayI, arrayJ);

        std::vector<std::vector<double>> ansMat ={{0, 4, 0, 4},
                                                  {0, 8, 1, 8},
                                                  {0, 4, 0, 4}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_stdmat_test_nomask_noaccum_trans)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6, 0},
                                             {0, 5, 7, 9},
                                             {4, 0, 2, 0}};
    Matrix<double> A(matA, 0);

    // I,J - AllIndices
    {
        Matrix<double> C(4,3);
        extract(C, NoMask(), NoAccumulate(), transpose(A),
                AllIndices(), AllIndices());

        std::vector<std::vector<double>> ansMat = {{8, 0, 4},
                                                   {1, 5, 0},
                                                   {6, 7, 2},
                                                   {0, 9, 0}};
        Matrix<double> answer(ansMat, 0);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - no dup, ordered
    {
        IndexArrayType arrayI({0,1,3});
        IndexArrayType arrayJ({0,2});

        Matrix<double> C(3,2);
        extract(C, NoMask(), NoAccumulate(), transpose(A), arrayI, arrayJ);

        std::vector<std::vector<double>> ansMat ={{8, 4},
                                                  {1, 0},
                                                  {0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - no dup, permuted
    {
        IndexArrayType arrayI({3,0,1});
        IndexArrayType arrayJ({2,0});

        Matrix<double> C(3,2);
        extract(C, NoMask(), NoAccumulate(), transpose(A), arrayI, arrayJ);

        std::vector<std::vector<double>> ansMat ={{0, 0},
                                                  {4, 8},
                                                  {0, 1}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - dup, ordered
    {
        IndexArrayType arrayI({0,1,1,3});
        IndexArrayType arrayJ({0,0,2});

        Matrix<double> C(4,3);
        extract(C, NoMask(), NoAccumulate(), transpose(A), arrayI, arrayJ);

        std::vector<std::vector<double>> ansMat ={{8, 8, 4},
                                                  {1, 1, 0},
                                                  {1, 1, 0},
                                                  {0, 0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - dup, permuted
    {
        IndexArrayType arrayI({3,0,1,0});
        IndexArrayType arrayJ({2,0,2});

        Matrix<double> C(4,3);
        extract(C, NoMask(), NoAccumulate(), transpose(A), arrayI, arrayJ);

        std::vector<std::vector<double>> ansMat ={{0, 0, 0},
                                                  {4, 8, 4},
                                                  {0, 1, 0},
                                                  {4, 8, 4}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_stdmat_test_nomask_accum_notrans)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6, 0},
                                             {0, 5, 7, 9},
                                             {4, 0, 2, 0}};
    std::vector<std::vector<double>> matC3x4 = {{9, 9, 9, 9},
                                                {9, 9, 9, 9},
                                                {9, 9, 9, 0}};
    std::vector<std::vector<double>> matC2x3 = {{9, 9, 9},
                                                {9, 9, 0}};
    Matrix<double> A(matA, 0);

    // I,J - AllIndices
    {
        Matrix<double> C(matC3x4, 0);
        extract(C, NoMask(), Plus<double>(), A, AllIndices(), AllIndices());

        std::vector<std::vector<double>> ansMat = {{17, 10, 15,  9},
                                                   { 9, 14, 16, 18},
                                                   {13,  9, 11,  0}};
        Matrix<double> answer(ansMat, 0);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - no dup, ordered
    {
        IndexArrayType arrayI({0,2});
        IndexArrayType arrayJ({0,1,3});

        Matrix<double> C(matC2x3, 0);
        extract(C, NoMask(), Plus<double>(), A, arrayI, arrayJ);

        std::vector<std::vector<double>> ansMat ={{17, 10, 9},
                                                  {13,  9, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - no dup, permuted
    {
        IndexArrayType arrayI({2,0});
        IndexArrayType arrayJ({3,0,1});

        Matrix<double> C(matC2x3, 0);
        extract(C, NoMask(), Plus<double>(), A, arrayI, arrayJ);

        std::vector<std::vector<double>> ansMat ={{9, 13,  9},
                                                  {9, 17,  1}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - dup, ordered
    {
        IndexArrayType arrayI({0,0,2});
        IndexArrayType arrayJ({0,1,1,3});

        Matrix<double> C(matC3x4, 0);
        extract(C, NoMask(), Plus<double>(), A, arrayI, arrayJ);

        std::vector<std::vector<double>> ansMat ={{17, 10, 10, 9},
                                                  {17, 10, 10, 9},
                                                  {13,  9,  9, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - dup, permuted
    {
        IndexArrayType arrayI({2,0,2});
        IndexArrayType arrayJ({3,0,1,0});

        Matrix<double> C(matC3x4, 0);
        extract(C, NoMask(), Plus<double>(), A, arrayI, arrayJ);

        std::vector<std::vector<double>> ansMat ={{9, 13,  9, 13},
                                                  {9, 17, 10, 17},
                                                  {9, 13,  9, 4}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_stdmat_test_nomask_accum_trans)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6, 0},
                                             {0, 5, 7, 9},
                                             {4, 0, 2, 0}};
    std::vector<std::vector<double>> matC4x3 = {{9, 9, 9},
                                                {9, 9, 9},
                                                {9, 9, 9},
                                                {9, 9, 0}};
    std::vector<std::vector<double>> matC3x2 = {{9, 9},
                                                {9, 9},
                                                {9, 0}};
    Matrix<double> A(matA, 0);

    // I,J - AllIndices
    {
        Matrix<double> C(matC4x3, 0);
        extract(C, NoMask(), Plus<double>(), transpose(A),
                AllIndices(), AllIndices());

        std::vector<std::vector<double>> ansMat = {{17,  9, 13},
                                                   {10, 14,  9},
                                                   {15, 16, 11},
                                                   { 9, 18,  0}};
        Matrix<double> answer(ansMat, 0);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - no dup, ordered
    {
        IndexArrayType arrayI({0,1,3});
        IndexArrayType arrayJ({0,2});

        Matrix<double> C(matC3x2, 0);
        extract(C, NoMask(), Plus<double>(), transpose(A), arrayI, arrayJ);

        std::vector<std::vector<double>> ansMat ={{17, 13},
                                                  {10,  9},
                                                  { 9,  0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - no dup, permuted
    {
        IndexArrayType arrayI({3,0,1});
        IndexArrayType arrayJ({2,0});

        Matrix<double> C(matC3x2, 0);
        extract(C, NoMask(), Plus<double>(), transpose(A), arrayI, arrayJ);

        std::vector<std::vector<double>> ansMat ={{ 9,  9},
                                                  {13, 17},
                                                  { 9, 1}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - dup, ordered
    {
        IndexArrayType arrayI({0,1,1,3});
        IndexArrayType arrayJ({0,0,2});

        Matrix<double> C(matC4x3, 0);
        extract(C, NoMask(), Plus<double>(), transpose(A), arrayI, arrayJ);

        std::vector<std::vector<double>> ansMat ={{17, 17, 13},
                                                  {10, 10,  9},
                                                  {10, 10,  9},
                                                  { 9,  9,  0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - dup, permuted
    {
        IndexArrayType arrayI({3,0,1,0});
        IndexArrayType arrayJ({2,0,2});

        Matrix<double> C(matC4x3, 0);
        extract(C, NoMask(), Plus<double>(), transpose(A), arrayI, arrayJ);

        std::vector<std::vector<double>> ansMat ={{ 9,  9,  9},
                                                  {13, 17, 13},
                                                  { 9, 10,  9},
                                                  {13, 17,  4}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_stdmat_test_noscmp_noaccum_notrans)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6, 0},
                                             {0, 5, 7, 9},
                                             {4, 0, 2, 0}};
    Matrix<double> A(matA, 0);

    std::vector<std::vector<uint8_t>> matMask2x3 = {{1, 1, 0},    // stored 0
                                                    {1,99, 0}};
    std::vector<std::vector<uint8_t>> matMask3x4 = {{1, 1, 1, 0}, // stored 0
                                                    {1, 1,99, 0},
                                                    {1,99,99, 0}};
    Matrix<uint8_t> mask2x3(matMask2x3, 99); // turn 99's into implicit 0's
    Matrix<uint8_t> mask3x4(matMask3x4, 99);

    std::vector<std::vector<double>> matC3x4 = {{9, 9, 9, 9},
                                                {9, 9, 9, 9},
                                                {9, 9, 9, 0}};
    std::vector<std::vector<double>> matC2x3 = {{9, 9, 9},
                                                {9, 9, 0}};

    // I,J - AllIndices
    {
        Matrix<double> C(matC3x4, 0);
        extract(C, mask3x4, NoAccumulate(), A, AllIndices(), AllIndices(), true);

        std::vector<std::vector<double>> ansMat = {{8, 1, 6, 0},
                                                   {0, 5, 0, 0},
                                                   {4, 0, 0, 0}};
        Matrix<double> answer(ansMat, 0);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        Matrix<double> C(matC3x4, 0);
        extract(C, mask3x4, NoAccumulate(), A, AllIndices(), AllIndices(), false);

        std::vector<std::vector<double>> ansMat = {{8, 1, 6, 9},
                                                   {0, 5, 9, 9},
                                                   {4, 9, 9, 0}};
        Matrix<double> answer(ansMat, 0);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - no dup, ordered
    {
        IndexArrayType arrayI({0,2});
        IndexArrayType arrayJ({0,1,3});

        Matrix<double> C(matC2x3, 0);
        extract(C, mask2x3, NoAccumulate(), A, arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{8, 1, 0},
                                                  {4, 0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({0,2});
        IndexArrayType arrayJ({0,1,3});

        Matrix<double> C(matC2x3, 0);
        extract(C, mask2x3, NoAccumulate(), A, arrayI, arrayJ, false);

        std::vector<std::vector<double>> ansMat ={{8, 1, 9},
                                                  {4, 9, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - no dup, permuted
    {
        IndexArrayType arrayI({2,0});
        IndexArrayType arrayJ({3,0,1});

        Matrix<double> C(matC2x3, 0);
        extract(C, mask2x3, NoAccumulate(), A, arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{0, 4, 0},
                                                  {0, 0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({2,0});
        IndexArrayType arrayJ({3,0,1});

        Matrix<double> C(matC2x3, 0);
        extract(C, mask2x3, NoAccumulate(), A, arrayI, arrayJ, false);

        std::vector<std::vector<double>> ansMat ={{0, 4, 9},
                                                  {0, 9, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - dup, ordered
    {
        IndexArrayType arrayI({0,0,2});
        IndexArrayType arrayJ({0,1,1,3});

        Matrix<double> C(matC3x4, 0);
        extract(C, mask3x4, NoAccumulate(), A, arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{8, 1, 1, 0},
                                                  {8, 1, 0, 0},
                                                  {4, 0, 0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({0,0,2});
        IndexArrayType arrayJ({0,1,1,3});

        Matrix<double> C(matC3x4, 0);
        extract(C, mask3x4, NoAccumulate(), A, arrayI, arrayJ, false);

        std::vector<std::vector<double>> ansMat ={{8, 1, 1, 9},
                                                  {8, 1, 9, 9},
                                                  {4, 9, 9, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - dup, permuted
    {
        IndexArrayType arrayI({2,0,2});
        IndexArrayType arrayJ({3,0,1,0});

        Matrix<double> C(matC3x4, 0);
        extract(C, mask3x4, NoAccumulate(), A, arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{0, 4, 0, 0},
                                                  {0, 8, 0, 0},
                                                  {0, 0, 0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({2,0,2});
        IndexArrayType arrayJ({3,0,1,0});

        Matrix<double> C(matC3x4, 0);
        extract(C, mask3x4, NoAccumulate(), A, arrayI, arrayJ, false);

        std::vector<std::vector<double>> ansMat ={{0, 4, 0, 9},
                                                  {0, 8, 9, 9},
                                                  {0, 9, 9, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_stdmat_test_noscmp_noaccum_trans)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6, 0},
                                             {0, 5, 7, 9},
                                             {4, 0, 2, 0}};
    Matrix<double> A(matA, 0);
    std::vector<std::vector<uint8_t>> matMask3x2 = {{1, 1},    // stored 0
                                                    {1,99},    // stored 0
                                                    {0, 0}};
    std::vector<std::vector<uint8_t>> matMask4x3 = {{1, 1, 1}, // stored 0
                                                    {1, 1,99}, // stored 0
                                                    {1,99,99},
                                                    {0, 0, 0}};
    Matrix<uint8_t> mask3x2(matMask3x2, 99); // turn 99's into implicit 0's
    Matrix<uint8_t> mask4x3(matMask4x3, 99);

    std::vector<std::vector<double>> matC4x3 = {{9, 9, 9},
                                                {9, 9, 9},
                                                {9, 9, 9},
                                                {9, 9, 0}};
    std::vector<std::vector<double>> matC3x2 = {{9, 9},
                                                {9, 9},
                                                {9, 0}};

    // I,J - AllIndices
    {
        Matrix<double> C(matC4x3, 0);
        extract(C, mask4x3, NoAccumulate(), transpose(A),
                AllIndices(), AllIndices(), true);

        std::vector<std::vector<double>> ansMat = {{8, 0, 4},
                                                   {1, 5, 0},
                                                   {6, 0, 0},
                                                   {0, 0, 0}};
        Matrix<double> answer(ansMat, 0);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        Matrix<double> C(matC4x3, 0);
        extract(C, mask4x3, NoAccumulate(), transpose(A),
                AllIndices(), AllIndices(), false);

        std::vector<std::vector<double>> ansMat = {{8, 0, 4},
                                                   {1, 5, 9},
                                                   {6, 9, 9},
                                                   {9, 9, 0}};
        Matrix<double> answer(ansMat, 0);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - no dup, ordered
    {
        IndexArrayType arrayI({0,1,3});
        IndexArrayType arrayJ({0,2});

        Matrix<double> C(matC3x2, 0);
        extract(C, mask3x2, NoAccumulate(), transpose(A), arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{8, 4},
                                                  {1, 0},
                                                  {0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({0,1,3});
        IndexArrayType arrayJ({0,2});

        Matrix<double> C(matC3x2, 0);
        extract(C, mask3x2, NoAccumulate(), transpose(A), arrayI, arrayJ,false);

        std::vector<std::vector<double>> ansMat ={{8, 4},
                                                  {1, 9},
                                                  {9, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - no dup, permuted
    {
        IndexArrayType arrayI({3,0,1});
        IndexArrayType arrayJ({2,0});

        Matrix<double> C(matC3x2, 0);
        extract(C, mask3x2, NoAccumulate(), transpose(A), arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{0, 0},
                                                  {4, 0},
                                                  {0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({3,0,1});
        IndexArrayType arrayJ({2,0});

        Matrix<double> C(matC3x2, 0);
        extract(C, mask3x2, NoAccumulate(), transpose(A), arrayI, arrayJ,false);

        std::vector<std::vector<double>> ansMat ={{0, 0},
                                                  {4, 9},
                                                  {9, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - dup, ordered
    {
        IndexArrayType arrayI({0,1,1,3});
        IndexArrayType arrayJ({0,0,2});

        Matrix<double> C(matC4x3, 0);
        extract(C, mask4x3, NoAccumulate(), transpose(A), arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{8, 8, 4},
                                                  {1, 1, 0},
                                                  {1, 0, 0},
                                                  {0, 0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({0,1,1,3});
        IndexArrayType arrayJ({0,0,2});

        Matrix<double> C(matC4x3, 0);
        extract(C, mask4x3, NoAccumulate(), transpose(A), arrayI, arrayJ,false);

        std::vector<std::vector<double>> ansMat ={{8, 8, 4},
                                                  {1, 1, 9},
                                                  {1, 9, 9},
                                                  {9, 9, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - dup, permuted
    {
        IndexArrayType arrayI({3,0,1,0});
        IndexArrayType arrayJ({2,0,2});

        Matrix<double> C(matC4x3, 0);
        extract(C, mask4x3, NoAccumulate(), transpose(A), arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{0, 0, 0},
                                                  {4, 8, 0},
                                                  {0, 0, 0},
                                                  {0, 0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({3,0,1,0});
        IndexArrayType arrayJ({2,0,2});

        Matrix<double> C(matC4x3, 0);
        extract(C, mask4x3, NoAccumulate(), transpose(A), arrayI, arrayJ,false);

        std::vector<std::vector<double>> ansMat ={{0, 0, 0},
                                                  {4, 8, 9},
                                                  {0, 9, 9},
                                                  {9, 9, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_stdmat_test_noscmp_accum_notrans)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6, 0},
                                             {0, 5, 7, 9},
                                             {4, 0, 2, 0}};
    Matrix<double> A(matA, 0);

    std::vector<std::vector<uint8_t>> matMask2x3 = {{1, 1, 0},    // stored 0
                                                    {1,99, 0}};
    std::vector<std::vector<uint8_t>> matMask3x4 = {{1, 1, 1, 0}, // stored 0
                                                    {1, 1,99, 0},
                                                    {1,99,99, 0}};
    Matrix<uint8_t> mask2x3(matMask2x3, 99); // turn 99's into implicit 0's
    Matrix<uint8_t> mask3x4(matMask3x4, 99);

    std::vector<std::vector<double>> matC3x4 = {{9, 9, 9, 9},
                                                {9, 9, 9, 9},
                                                {9, 9, 9, 0}};
    std::vector<std::vector<double>> matC2x3 = {{9, 9, 9},
                                                {9, 9, 0}};

    // I,J - AllIndices
    {
        Matrix<double> C(matC3x4, 0);
        extract(C, mask3x4, Plus<double>(), A, AllIndices(), AllIndices(), true);

        std::vector<std::vector<double>> ansMat = {{17, 10, 15, 0},
                                                   { 9, 14,  0, 0},
                                                   {13,  0,  0, 0}};
        Matrix<double> answer(ansMat, 0);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        Matrix<double> C(matC3x4, 0);
        extract(C, mask3x4, Plus<double>(), A, AllIndices(), AllIndices(), false);

        std::vector<std::vector<double>> ansMat = {{17, 10, 15, 9},
                                                   { 9, 14,  9, 9},
                                                   {13,  9,  9, 0}};
        Matrix<double> answer(ansMat, 0);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - no dup, ordered
    {
        IndexArrayType arrayI({0,2});
        IndexArrayType arrayJ({0,1,3});

        Matrix<double> C(matC2x3, 0);
        extract(C, mask2x3, Plus<double>(), A, arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{17, 10, 0},
                                                  {13,  0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({0,2});
        IndexArrayType arrayJ({0,1,3});

        Matrix<double> C(matC2x3, 0);
        extract(C, mask2x3, Plus<double>(), A, arrayI, arrayJ, false);

        std::vector<std::vector<double>> ansMat ={{17, 10, 9},
                                                  {13,  9, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - no dup, permuted
    {
        IndexArrayType arrayI({2,0});
        IndexArrayType arrayJ({3,0,1});

        Matrix<double> C(matC2x3, 0);
        extract(C, mask2x3, Plus<double>(), A, arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{9,13, 0},
                                                  {9, 0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({2,0});
        IndexArrayType arrayJ({3,0,1});

        Matrix<double> C(matC2x3, 0);
        extract(C, mask2x3, Plus<double>(), A, arrayI, arrayJ, false);

        std::vector<std::vector<double>> ansMat ={{9,13, 9},
                                                  {9, 9, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - dup, ordered
    {
        IndexArrayType arrayI({0,0,2});
        IndexArrayType arrayJ({0,1,1,3});

        Matrix<double> C(matC3x4, 0);
        extract(C, mask3x4, Plus<double>(), A, arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{17, 10, 10, 0},
                                                  {17, 10,  0, 0},
                                                  {13,  0,  0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({0,0,2});
        IndexArrayType arrayJ({0,1,1,3});

        Matrix<double> C(matC3x4, 0);
        extract(C, mask3x4, Plus<double>(), A, arrayI, arrayJ, false);

        std::vector<std::vector<double>> ansMat ={{17, 10, 10, 9},
                                                  {17, 10,  9, 9},
                                                  {13,  9,  9, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - dup, permuted
    {
        IndexArrayType arrayI({2,0,2});
        IndexArrayType arrayJ({3,0,1,0});

        Matrix<double> C(matC3x4, 0);
        extract(C, mask3x4, Plus<double>(), A, arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{9,13, 9, 0},
                                                  {9,17, 0, 0},
                                                  {9, 0, 0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({2,0,2});
        IndexArrayType arrayJ({3,0,1,0});

        Matrix<double> C(matC3x4, 0);
        extract(C, mask3x4, Plus<double>(), A, arrayI, arrayJ, false);

        std::vector<std::vector<double>> ansMat ={{9,13, 9, 9},
                                                  {9,17, 9, 9},
                                                  {9, 9, 9, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_stdmat_test_noscmp_accum_trans)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6, 0},
                                             {0, 5, 7, 9},
                                             {4, 0, 2, 0}};
    Matrix<double> A(matA, 0);
    std::vector<std::vector<uint8_t>> matMask3x2 = {{1, 1},    // stored 0
                                                    {1,99},    // stored 0
                                                    {0, 0}};
    std::vector<std::vector<uint8_t>> matMask4x3 = {{1, 1, 1}, // stored 0
                                                    {1, 1,99}, // stored 0
                                                    {1,99,99},
                                                    {0, 0, 0}};
    Matrix<uint8_t> mask3x2(matMask3x2, 99); // turn 99's into implicit 0's
    Matrix<uint8_t> mask4x3(matMask4x3, 99);

    std::vector<std::vector<double>> matC4x3 = {{9, 9, 9},
                                                {9, 9, 9},
                                                {9, 9, 9},
                                                {9, 9, 0}};
    std::vector<std::vector<double>> matC3x2 = {{9, 9},
                                                {9, 9},
                                                {9, 0}};

    // I,J - AllIndices
    {
        Matrix<double> C(matC4x3, 0);
        extract(C, mask4x3, Plus<double>(), transpose(A),
                AllIndices(), AllIndices(), true);

        std::vector<std::vector<double>> ansMat = {{17,  9, 13},
                                                   {10, 14,  0},
                                                   {15,  0,  0},
                                                   { 0,  0,  0}};
        Matrix<double> answer(ansMat, 0);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        Matrix<double> C(matC4x3, 0);
        extract(C, mask4x3, Plus<double>(), transpose(A),
                AllIndices(), AllIndices(), false);

        std::vector<std::vector<double>> ansMat = {{17,  9, 13},
                                                   {10, 14,  9},
                                                   {15,  9,  9},
                                                   { 9,  9,  0}};
        Matrix<double> answer(ansMat, 0);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - no dup, ordered
    {
        IndexArrayType arrayI({0,1,3});
        IndexArrayType arrayJ({0,2});

        Matrix<double> C(matC3x2, 0);
        extract(C, mask3x2, Plus<double>(), transpose(A), arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{17, 13},
                                                  {10,  0},
                                                  { 0,  0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({0,1,3});
        IndexArrayType arrayJ({0,2});

        Matrix<double> C(matC3x2, 0);
        extract(C, mask3x2, Plus<double>(), transpose(A), arrayI, arrayJ,false);

        std::vector<std::vector<double>> ansMat ={{17, 13},
                                                  {10,  9},
                                                  { 9,  0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - no dup, permuted
    {
        IndexArrayType arrayI({3,0,1});
        IndexArrayType arrayJ({2,0});

        Matrix<double> C(matC3x2, 0);
        extract(C, mask3x2, Plus<double>(), transpose(A), arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{ 9, 9},
                                                  {13, 0},
                                                  { 0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({3,0,1});
        IndexArrayType arrayJ({2,0});

        Matrix<double> C(matC3x2, 0);
        extract(C, mask3x2, Plus<double>(), transpose(A), arrayI, arrayJ,false);

        std::vector<std::vector<double>> ansMat ={{ 9, 9},
                                                  {13, 9},
                                                  { 9, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - dup, ordered
    {
        IndexArrayType arrayI({0,1,1,3});
        IndexArrayType arrayJ({0,0,2});

        Matrix<double> C(matC4x3, 0);
        extract(C, mask4x3, Plus<double>(), transpose(A), arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{17, 17, 13},
                                                  {10, 10,  0},
                                                  {10,  0,  0},
                                                  { 0,  0,  0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({0,1,1,3});
        IndexArrayType arrayJ({0,0,2});

        Matrix<double> C(matC4x3, 0);
        extract(C, mask4x3, Plus<double>(), transpose(A), arrayI, arrayJ,false);

        std::vector<std::vector<double>> ansMat ={{17, 17, 13},
                                                  {10, 10,  9},
                                                  {10,  9,  9},
                                                  { 9,  9,  0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - dup, permuted
    {
        IndexArrayType arrayI({3,0,1,0});
        IndexArrayType arrayJ({2,0,2});

        Matrix<double> C(matC4x3, 0);
        extract(C, mask4x3, Plus<double>(), transpose(A), arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{ 9,  9, 9},
                                                  {13, 17, 0},
                                                  { 9,  0, 0},
                                                  { 0,  0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({3,0,1,0});
        IndexArrayType arrayJ({2,0,2});

        Matrix<double> C(matC4x3, 0);
        extract(C, mask4x3, Plus<double>(), transpose(A), arrayI, arrayJ,false);

        std::vector<std::vector<double>> ansMat ={{ 9,  9, 9},
                                                  {13, 17, 9},
                                                  { 9,  9, 9},
                                                  { 9,  9, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_stdmat_test_scmp_noaccum_notrans)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6, 0},
                                             {0, 5, 7, 9},
                                             {4, 0, 2, 0}};
    Matrix<double> A(matA, 0);

    // complements of masks from the noscmp case
    std::vector<std::vector<uint8_t>> matMask2x3 = {{0,99, 1},    // stored 0
                                                    {0, 1, 1}};
    std::vector<std::vector<uint8_t>> matMask3x4 = {{0,99,99, 1}, // stored 0
                                                    {0,99, 1, 1},
                                                    {0, 1, 1, 1}};
    Matrix<uint8_t> mask2x3(matMask2x3, 99); // turn 99's into implicit 0's
    Matrix<uint8_t> mask3x4(matMask3x4, 99);

    std::vector<std::vector<double>> matC3x4 = {{9, 9, 9, 9},
                                                {9, 9, 9, 9},
                                                {9, 9, 9, 0}};
    std::vector<std::vector<double>> matC2x3 = {{9, 9, 9},
                                                {9, 9, 0}};

    // I,J - AllIndices
    {
        Matrix<double> C(matC3x4, 0);
        extract(C, complement(mask3x4), NoAccumulate(), A, AllIndices(), AllIndices(), true);

        std::vector<std::vector<double>> ansMat = {{8, 1, 6, 0},
                                                   {0, 5, 0, 0},
                                                   {4, 0, 0, 0}};
        Matrix<double> answer(ansMat, 0);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        Matrix<double> C(matC3x4, 0);
        extract(C, complement(mask3x4), NoAccumulate(), A, AllIndices(), AllIndices(), false);

        std::vector<std::vector<double>> ansMat = {{8, 1, 6, 9},
                                                   {0, 5, 9, 9},
                                                   {4, 9, 9, 0}};
        Matrix<double> answer(ansMat, 0);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - no dup, ordered
    {
        IndexArrayType arrayI({0,2});
        IndexArrayType arrayJ({0,1,3});

        Matrix<double> C(matC2x3, 0);
        extract(C, complement(mask2x3), NoAccumulate(), A, arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{8, 1, 0},
                                                  {4, 0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({0,2});
        IndexArrayType arrayJ({0,1,3});

        Matrix<double> C(matC2x3, 0);
        extract(C, complement(mask2x3), NoAccumulate(), A, arrayI, arrayJ, false);

        std::vector<std::vector<double>> ansMat ={{8, 1, 9},
                                                  {4, 9, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - no dup, permuted
    {
        IndexArrayType arrayI({2,0});
        IndexArrayType arrayJ({3,0,1});

        Matrix<double> C(matC2x3, 0);
        extract(C, complement(mask2x3), NoAccumulate(), A, arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{0, 4, 0},
                                                  {0, 0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({2,0});
        IndexArrayType arrayJ({3,0,1});

        Matrix<double> C(matC2x3, 0);
        extract(C, complement(mask2x3), NoAccumulate(), A, arrayI, arrayJ, false);

        std::vector<std::vector<double>> ansMat ={{0, 4, 9},
                                                  {0, 9, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - dup, ordered
    {
        IndexArrayType arrayI({0,0,2});
        IndexArrayType arrayJ({0,1,1,3});

        Matrix<double> C(matC3x4, 0);
        extract(C, complement(mask3x4), NoAccumulate(), A, arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{8, 1, 1, 0},
                                                  {8, 1, 0, 0},
                                                  {4, 0, 0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({0,0,2});
        IndexArrayType arrayJ({0,1,1,3});

        Matrix<double> C(matC3x4, 0);
        extract(C, complement(mask3x4), NoAccumulate(), A, arrayI, arrayJ, false);

        std::vector<std::vector<double>> ansMat ={{8, 1, 1, 9},
                                                  {8, 1, 9, 9},
                                                  {4, 9, 9, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - dup, permuted
    {
        IndexArrayType arrayI({2,0,2});
        IndexArrayType arrayJ({3,0,1,0});

        Matrix<double> C(matC3x4, 0);
        extract(C, complement(mask3x4), NoAccumulate(), A, arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{0, 4, 0, 0},
                                                  {0, 8, 0, 0},
                                                  {0, 0, 0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({2,0,2});
        IndexArrayType arrayJ({3,0,1,0});

        Matrix<double> C(matC3x4, 0);
        extract(C, complement(mask3x4), NoAccumulate(), A, arrayI, arrayJ, false);

        std::vector<std::vector<double>> ansMat ={{0, 4, 0, 9},
                                                  {0, 8, 9, 9},
                                                  {0, 9, 9, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_stdmat_test_scmp_noaccum_trans)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6, 0},
                                             {0, 5, 7, 9},
                                             {4, 0, 2, 0}};
    Matrix<double> A(matA, 0);
    std::vector<std::vector<uint8_t>> matMask3x2 = {{ 0, 0},    // stored 0
                                                    {99, 1},    // stored 0
                                                    { 1, 1}};
    std::vector<std::vector<uint8_t>> matMask4x3 = {{ 0, 0, 0}, // stored 0
                                                    {99,99, 1}, // stored 0
                                                    {99, 1, 1},
                                                    { 1, 1, 1}};
    Matrix<uint8_t> mask3x2(matMask3x2, 99); // turn 99's into implicit 0's
    Matrix<uint8_t> mask4x3(matMask4x3, 99);

    std::vector<std::vector<double>> matC4x3 = {{9, 9, 9},
                                                {9, 9, 9},
                                                {9, 9, 9},
                                                {9, 9, 0}};
    std::vector<std::vector<double>> matC3x2 = {{9, 9},
                                                {9, 9},
                                                {9, 0}};

    // I,J - AllIndices
    {
        Matrix<double> C(matC4x3, 0);
        extract(C, complement(mask4x3), NoAccumulate(), transpose(A),
                AllIndices(), AllIndices(), true);

        std::vector<std::vector<double>> ansMat = {{8, 0, 4},
                                                   {1, 5, 0},
                                                   {6, 0, 0},
                                                   {0, 0, 0}};
        Matrix<double> answer(ansMat, 0);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        Matrix<double> C(matC4x3, 0);
        extract(C, complement(mask4x3), NoAccumulate(), transpose(A),
                AllIndices(), AllIndices(), false);

        std::vector<std::vector<double>> ansMat = {{8, 0, 4},
                                                   {1, 5, 9},
                                                   {6, 9, 9},
                                                   {9, 9, 0}};
        Matrix<double> answer(ansMat, 0);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - no dup, ordered
    {
        IndexArrayType arrayI({0,1,3});
        IndexArrayType arrayJ({0,2});

        Matrix<double> C(matC3x2, 0);
        extract(C, complement(mask3x2), NoAccumulate(), transpose(A), arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{8, 4},
                                                  {1, 0},
                                                  {0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({0,1,3});
        IndexArrayType arrayJ({0,2});

        Matrix<double> C(matC3x2, 0);
        extract(C, complement(mask3x2), NoAccumulate(), transpose(A), arrayI, arrayJ,false);

        std::vector<std::vector<double>> ansMat ={{8, 4},
                                                  {1, 9},
                                                  {9, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - no dup, permuted
    {
        IndexArrayType arrayI({3,0,1});
        IndexArrayType arrayJ({2,0});

        Matrix<double> C(matC3x2, 0);
        extract(C, complement(mask3x2), NoAccumulate(), transpose(A), arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{0, 0},
                                                  {4, 0},
                                                  {0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({3,0,1});
        IndexArrayType arrayJ({2,0});

        Matrix<double> C(matC3x2, 0);
        extract(C, complement(mask3x2), NoAccumulate(), transpose(A), arrayI, arrayJ,false);

        std::vector<std::vector<double>> ansMat ={{0, 0},
                                                  {4, 9},
                                                  {9, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - dup, ordered
    {
        IndexArrayType arrayI({0,1,1,3});
        IndexArrayType arrayJ({0,0,2});

        Matrix<double> C(matC4x3, 0);
        extract(C, complement(mask4x3), NoAccumulate(), transpose(A), arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{8, 8, 4},
                                                  {1, 1, 0},
                                                  {1, 0, 0},
                                                  {0, 0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({0,1,1,3});
        IndexArrayType arrayJ({0,0,2});

        Matrix<double> C(matC4x3, 0);
        extract(C, complement(mask4x3), NoAccumulate(), transpose(A), arrayI, arrayJ,false);

        std::vector<std::vector<double>> ansMat ={{8, 8, 4},
                                                  {1, 1, 9},
                                                  {1, 9, 9},
                                                  {9, 9, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - dup, permuted
    {
        IndexArrayType arrayI({3,0,1,0});
        IndexArrayType arrayJ({2,0,2});

        Matrix<double> C(matC4x3, 0);
        extract(C, complement(mask4x3), NoAccumulate(), transpose(A), arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{0, 0, 0},
                                                  {4, 8, 0},
                                                  {0, 0, 0},
                                                  {0, 0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({3,0,1,0});
        IndexArrayType arrayJ({2,0,2});

        Matrix<double> C(matC4x3, 0);
        extract(C, complement(mask4x3), NoAccumulate(), transpose(A), arrayI, arrayJ,false);

        std::vector<std::vector<double>> ansMat ={{0, 0, 0},
                                                  {4, 8, 9},
                                                  {0, 9, 9},
                                                  {9, 9, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_stdmat_test_scmp_accum_notrans)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6, 0},
                                             {0, 5, 7, 9},
                                             {4, 0, 2, 0}};
    Matrix<double> A(matA, 0);

    std::vector<std::vector<uint8_t>> matMask2x3 = {{0,99, 1},    // stored 0
                                                    {0, 1, 1}};
    std::vector<std::vector<uint8_t>> matMask3x4 = {{0,99,99, 1}, // stored 0
                                                    {0,99, 1, 1},
                                                    {0, 1, 1, 1}};
    Matrix<uint8_t> mask2x3(matMask2x3, 99); // turn 99's into implicit 0's
    Matrix<uint8_t> mask3x4(matMask3x4, 99);

    std::vector<std::vector<double>> matC3x4 = {{9, 9, 9, 9},
                                                {9, 9, 9, 9},
                                                {9, 9, 9, 0}};
    std::vector<std::vector<double>> matC2x3 = {{9, 9, 9},
                                                {9, 9, 0}};

    // I,J - AllIndices
    {
        Matrix<double> C(matC3x4, 0);
        extract(C, complement(mask3x4), Plus<double>(), A, AllIndices(), AllIndices(), true);

        std::vector<std::vector<double>> ansMat = {{17, 10, 15, 0},
                                                   { 9, 14,  0, 0},
                                                   {13,  0,  0, 0}};
        Matrix<double> answer(ansMat, 0);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        Matrix<double> C(matC3x4, 0);
        extract(C, complement(mask3x4), Plus<double>(), A, AllIndices(), AllIndices(), false);

        std::vector<std::vector<double>> ansMat = {{17, 10, 15, 9},
                                                   { 9, 14,  9, 9},
                                                   {13,  9,  9, 0}};
        Matrix<double> answer(ansMat, 0);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - no dup, ordered
    {
        IndexArrayType arrayI({0,2});
        IndexArrayType arrayJ({0,1,3});

        Matrix<double> C(matC2x3, 0);
        extract(C, complement(mask2x3), Plus<double>(), A, arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{17, 10, 0},
                                                  {13,  0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({0,2});
        IndexArrayType arrayJ({0,1,3});

        Matrix<double> C(matC2x3, 0);
        extract(C, complement(mask2x3), Plus<double>(), A, arrayI, arrayJ, false);

        std::vector<std::vector<double>> ansMat ={{17, 10, 9},
                                                  {13,  9, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - no dup, permuted
    {
        IndexArrayType arrayI({2,0});
        IndexArrayType arrayJ({3,0,1});

        Matrix<double> C(matC2x3, 0);
        extract(C, complement(mask2x3), Plus<double>(), A, arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{9,13, 0},
                                                  {9, 0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({2,0});
        IndexArrayType arrayJ({3,0,1});

        Matrix<double> C(matC2x3, 0);
        extract(C, complement(mask2x3), Plus<double>(), A, arrayI, arrayJ, false);

        std::vector<std::vector<double>> ansMat ={{9,13, 9},
                                                  {9, 9, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - dup, ordered
    {
        IndexArrayType arrayI({0,0,2});
        IndexArrayType arrayJ({0,1,1,3});

        Matrix<double> C(matC3x4, 0);
        extract(C, complement(mask3x4), Plus<double>(), A, arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{17, 10, 10, 0},
                                                  {17, 10,  0, 0},
                                                  {13,  0,  0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({0,0,2});
        IndexArrayType arrayJ({0,1,1,3});

        Matrix<double> C(matC3x4, 0);
        extract(C, complement(mask3x4), Plus<double>(), A, arrayI, arrayJ, false);

        std::vector<std::vector<double>> ansMat ={{17, 10, 10, 9},
                                                  {17, 10,  9, 9},
                                                  {13,  9,  9, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - dup, permuted
    {
        IndexArrayType arrayI({2,0,2});
        IndexArrayType arrayJ({3,0,1,0});

        Matrix<double> C(matC3x4, 0);
        extract(C, complement(mask3x4), Plus<double>(), A, arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{9,13, 9, 0},
                                                  {9,17, 0, 0},
                                                  {9, 0, 0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({2,0,2});
        IndexArrayType arrayJ({3,0,1,0});

        Matrix<double> C(matC3x4, 0);
        extract(C, complement(mask3x4), Plus<double>(), A, arrayI, arrayJ, false);

        std::vector<std::vector<double>> ansMat ={{9,13, 9, 9},
                                                  {9,17, 9, 9},
                                                  {9, 9, 9, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_stdmat_test_scmp_accum_trans)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6, 0},
                                             {0, 5, 7, 9},
                                             {4, 0, 2, 0}};
    Matrix<double> A(matA, 0);
    std::vector<std::vector<uint8_t>> matMask3x2 = {{ 0, 0},    // stored 0
                                                    {99, 1},    // stored 0
                                                    { 1, 1}};
    std::vector<std::vector<uint8_t>> matMask4x3 = {{ 0, 0, 0}, // stored 0
                                                    {99,99, 1}, // stored 0
                                                    {99, 1, 1},
                                                    { 1, 1, 1}};
    Matrix<uint8_t> mask3x2(matMask3x2, 99); // turn 99's into implicit 0's
    Matrix<uint8_t> mask4x3(matMask4x3, 99);

    std::vector<std::vector<double>> matC4x3 = {{9, 9, 9},
                                                {9, 9, 9},
                                                {9, 9, 9},
                                                {9, 9, 0}};
    std::vector<std::vector<double>> matC3x2 = {{9, 9},
                                                {9, 9},
                                                {9, 0}};

    // I,J - AllIndices
    {
        Matrix<double> C(matC4x3, 0);
        extract(C, complement(mask4x3), Plus<double>(), transpose(A),
                AllIndices(), AllIndices(), true);

        std::vector<std::vector<double>> ansMat = {{17,  9, 13},
                                                   {10, 14,  0},
                                                   {15,  0,  0},
                                                   { 0,  0,  0}};
        Matrix<double> answer(ansMat, 0);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        Matrix<double> C(matC4x3, 0);
        extract(C, complement(mask4x3), Plus<double>(), transpose(A),
                AllIndices(), AllIndices(), false);

        std::vector<std::vector<double>> ansMat = {{17,  9, 13},
                                                   {10, 14,  9},
                                                   {15,  9,  9},
                                                   { 9,  9,  0}};
        Matrix<double> answer(ansMat, 0);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - no dup, ordered
    {
        IndexArrayType arrayI({0,1,3});
        IndexArrayType arrayJ({0,2});

        Matrix<double> C(matC3x2, 0);
        extract(C, complement(mask3x2), Plus<double>(), transpose(A), arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{17, 13},
                                                  {10,  0},
                                                  { 0,  0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({0,1,3});
        IndexArrayType arrayJ({0,2});

        Matrix<double> C(matC3x2, 0);
        extract(C, complement(mask3x2), Plus<double>(), transpose(A), arrayI, arrayJ,false);

        std::vector<std::vector<double>> ansMat ={{17, 13},
                                                  {10,  9},
                                                  { 9,  0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - no dup, permuted
    {
        IndexArrayType arrayI({3,0,1});
        IndexArrayType arrayJ({2,0});

        Matrix<double> C(matC3x2, 0);
        extract(C, complement(mask3x2), Plus<double>(), transpose(A), arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{ 9, 9},
                                                  {13, 0},
                                                  { 0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({3,0,1});
        IndexArrayType arrayJ({2,0});

        Matrix<double> C(matC3x2, 0);
        extract(C, complement(mask3x2), Plus<double>(), transpose(A), arrayI, arrayJ,false);

        std::vector<std::vector<double>> ansMat ={{ 9, 9},
                                                  {13, 9},
                                                  { 9, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - dup, ordered
    {
        IndexArrayType arrayI({0,1,1,3});
        IndexArrayType arrayJ({0,0,2});

        Matrix<double> C(matC4x3, 0);
        extract(C, complement(mask4x3), Plus<double>(), transpose(A), arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{17, 17, 13},
                                                  {10, 10,  0},
                                                  {10,  0,  0},
                                                  { 0,  0,  0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({0,1,1,3});
        IndexArrayType arrayJ({0,0,2});

        Matrix<double> C(matC4x3, 0);
        extract(C, complement(mask4x3), Plus<double>(), transpose(A), arrayI, arrayJ,false);

        std::vector<std::vector<double>> ansMat ={{17, 17, 13},
                                                  {10, 10,  9},
                                                  {10,  9,  9},
                                                  { 9,  9,  0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }

    // I,J - dup, permuted
    {
        IndexArrayType arrayI({3,0,1,0});
        IndexArrayType arrayJ({2,0,2});

        Matrix<double> C(matC4x3, 0);
        extract(C, complement(mask4x3), Plus<double>(), transpose(A), arrayI, arrayJ, true);

        std::vector<std::vector<double>> ansMat ={{ 9,  9, 9},
                                                  {13, 17, 0},
                                                  { 9,  0, 0},
                                                  { 0,  0, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
    {
        IndexArrayType arrayI({3,0,1,0});
        IndexArrayType arrayJ({2,0,2});

        Matrix<double> C(matC4x3, 0);
        extract(C, complement(mask4x3), Plus<double>(), transpose(A), arrayI, arrayJ,false);

        std::vector<std::vector<double>> ansMat ={{ 9,  9, 9},
                                                  {13, 17, 9},
                                                  { 9,  9, 9},
                                                  { 9,  9, 0}};
        Matrix<double> answer(ansMat, 0.);
        BOOST_CHECK_EQUAL(C, answer);
    }
}

BOOST_AUTO_TEST_SUITE_END()
