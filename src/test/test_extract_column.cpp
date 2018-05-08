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
#define BOOST_TEST_MODULE extract_column_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(extract_column_suite)

//****************************************************************************
// Extract column tests
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_col_test_bad_dimensions)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6, 0},
                                             {3, 0, 7, 9},
                                             {4, 9, 2, 0}};
    Matrix<double, DirectedMatrixTag> mA(matA, 0);

    // Output space
    IndexType M = 2;

    IndexArrayType vec_row_indices = {0, 2};
    IndexArrayType row_indices(vec_row_indices);

    // result vector too small
    {
        Vector<double> result(M-1);

        BOOST_CHECK_THROW(
            (extract(result,
                     NoMask(),
                     NoAccumulate(),
                     mA,
                     row_indices,
                     (IndexType)1)),
            DimensionException);
    }

    // result vector too large
    {
        Vector<double> result(M+1);

        BOOST_CHECK_THROW(
            (extract(result,
                     NoMask(),
                     NoAccumulate(),
                     mA,
                     row_indices,
                     (IndexType)1)),
            DimensionException);
    }

    // Mask too small
    {
        Vector<double> result(M);
        Vector<bool> mask(M-1);

        BOOST_CHECK_THROW(
            (extract(result,
                     mask,
                     NoAccumulate(),
                     mA,
                     row_indices,
                     (IndexType)1)),
            DimensionException);
    }

    // Mask too large
    {
        Vector<double> result(M);
        Vector<bool> mask(M+1);

        BOOST_CHECK_THROW(
            (extract(result,
                     mask,
                     NoAccumulate(),
                     mA,
                     row_indices,
                     (IndexType)1)),
            DimensionException);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_col_test_invalid_index)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6},
                                             {3, 0, 7},
                                             {4, 9, 2}};
    Matrix<double, DirectedMatrixTag> mA(matA, 0);

    // Output space
    IndexType M = 2;

    IndexArrayType vec_row_indices = {0, 2};
    IndexArrayType row_indices(vec_row_indices);

    // result vector too small
    {
        Vector<double> result(M);

        BOOST_CHECK_THROW(
            (extract(result,
                     NoMask(),
                     NoAccumulate(),
                     mA,
                     row_indices,
                     (IndexType)4)),
            InvalidIndexException);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_col_test_index_out_of_bounds)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6},
                                             {3, 0, 7},
                                             {4, 9, 2}};
    Matrix<double, DirectedMatrixTag> mA(matA, 0);

    // Output space
    IndexType M = 2;

    IndexArrayType vec_row_indices = {0, 4};
    IndexArrayType row_indices(vec_row_indices);

    // result vector too small
    {
        Vector<double> result(M);

        BOOST_CHECK_THROW(
            (extract(result,
                     NoMask(),
                     NoAccumulate(),
                     mA,
                     row_indices,
                     (IndexType)1)),
            IndexOutOfBoundsException);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_col_test_allindices_too_small)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6, 0},
                                             {0, 5, 7, 9},
                                             {4, 0, 2, 0}};
    Matrix<double, DirectedMatrixTag> mA(matA, 0);

    // result is too small when I = AllIndices will pass
    {
        std::vector<double> vecAnswer = {0, 5, 7};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        IndexType M = 3;
        Vector<double> result(M);

        extract(result,
                NoMask(),
                NoAccumulate(),
                transpose(mA),
                AllIndices(),
                (IndexType)1);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // result is too large when I = AllIndices will pass
    {
        std::vector<double> vecAnswer = {0, 5, 7, 9, 0, 0};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        IndexType M = 6;
        Vector<double> result(M);

        extract(result,
                NoMask(),
                NoAccumulate(),
                transpose(mA),
                AllIndices(),
                (IndexType)1);

        BOOST_CHECK_EQUAL(result, answer);
    }
}


//****************************************************************************
// extract column passing test cases:
//
// Simplified test structure (12 test cases total):
//
// Mask cases (x3):  nomask,  noscmp, scmp
// Accum cases (x2): noaccum, accum
// Trans cases (x2): notrans, trans
//
// Within each test case: 5 checks will be run:
//
// indices: AllIndices, nodup/ordered, nodup/permute, dup/ordered, dup/permute
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_col_test_nomask_noaccum_notrans)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6, 0},
                                             {0, 5, 7, 9},
                                             {4, 0, 2, 0}};
    Matrix<double, DirectedMatrixTag> mA(matA, 0);

    // I - AllIndices
    {
        std::vector<double> vecAnswer = {1, 5, 0};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        IndexType M = 3;
        Vector<double> result(M);

        extract(result, NoMask(), NoAccumulate(),
                mA, AllIndices(), (IndexType)1);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - no dup, ordered
    {
        IndexArrayType row_indices = {0, 2};

        std::vector<double> vecAnswer = {6, 2};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        IndexType M = 2;
        Vector<double> result(M);

        extract(result, NoMask(), NoAccumulate(),
                mA, row_indices, (IndexType)2);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - no dup, permuted
    {
        IndexArrayType row_indices = {2, 0};

        std::vector<double> vecAnswer = {2, 6};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        IndexType M = 2;
        Vector<double> result(M);

        extract(result, NoMask(), NoAccumulate(),
                mA, row_indices, (IndexType)2);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - dup, ordered
    {
        IndexArrayType row_indices = {0, 1, 1, 2};

        std::vector<double> vecAnswer = {0, 9, 9, 0};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        IndexType M = 4;
        Vector<double> result(M);

        extract(result, NoMask(), NoAccumulate(),
                mA, row_indices, (IndexType)3);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - dup, permuted
    {
        IndexArrayType row_indices = {0, 2, 2, 1};

        std::vector<double> vecAnswer = {8, 4, 4, 0};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        IndexType M = 4;
        Vector<double> result(M);

        extract(result, NoMask(), NoAccumulate(),
                mA, row_indices, (IndexType)0);

        BOOST_CHECK_EQUAL(result, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_col_test_nomask_noaccum_trans)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6, 0},
                                             {0, 5, 7, 9},
                                             {4, 0, 2, 0}};
    Matrix<double, DirectedMatrixTag> mA(matA, 0);

    // I - AllIndices
    {
        std::vector<double> vecAnswer = {0, 5, 7, 9};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        IndexType M = 4;
        Vector<double> result(M);

        extract(result, NoMask(), NoAccumulate(),
                transpose(mA), AllIndices(), (IndexType)1);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - no dup, ordered
    {
        IndexArrayType row_indices = {0, 2, 3};

        std::vector<double> vecAnswer = {4, 2, 0};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        IndexType M = 3;
        Vector<double> result(M);

        extract(result, NoMask(), NoAccumulate(),
                transpose(mA), row_indices, (IndexType)2);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - no dup, permuted
    {
        IndexArrayType row_indices = {0, 3, 2};

        std::vector<double> vecAnswer = {4, 0, 2};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        IndexType M = 3;
        Vector<double> result(M);

        extract(result,
                NoMask(),
                NoAccumulate(),
                transpose(mA),
                row_indices,
                (IndexType)2);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - dup, ordered
    {
        IndexArrayType row_indices = {0, 2, 3, 3};

        std::vector<double> vecAnswer = {4, 2, 0, 0};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        IndexType M = 4;
        Vector<double> result(M);

        extract(result,
                NoMask(),
                NoAccumulate(),
                transpose(mA),
                row_indices,
                (IndexType)2);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - dup, permuted
    {
        IndexArrayType row_indices = {0, 2, 3, 0};

        std::vector<double> vecAnswer = {4, 2, 0, 4};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        IndexType M = 4;
        Vector<double> result(M);

        extract(result,
                NoMask(),
                NoAccumulate(),
                transpose(mA),
                row_indices,
                (IndexType)2);

        BOOST_CHECK_EQUAL(result, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_col_test_nomask_accum_notrans)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6, 0},
                                             {0, 5, 7, 9},
                                             {4, 0, 2, 0}};
    Matrix<double, DirectedMatrixTag> mA(matA, 0);

    // I - AllIndices
    {
        std::vector<double> vecAnswer = {2, 5, 1};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        IndexType M = 3;
        std::vector<double> res = {1, 0, 1};
        Vector<double> result(res, 0.);

        extract(result,
                NoMask(),
                Plus<double>(),
                mA,
                AllIndices(),
                (IndexType)1);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - no dup, ordered
    {
        IndexArrayType row_indices = {0, 2};

        std::vector<double> vecAnswer = {7, 2};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        IndexType M = 2;
        std::vector<double> res = {1, 0};
        Vector<double> result(res, 0.);

        extract(result,
                NoMask(),
                Plus<double>(),
                mA,
                row_indices,
                (IndexType)2);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - no dup, permuted
    {
        IndexArrayType row_indices = {2, 0};

        std::vector<double> vecAnswer = {3, 6};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        IndexType M = 2;
        std::vector<double> res = {1, 0};
        Vector<double> result(res, 0.);

        extract(result,
                NoMask(),
                Plus<double>(),
                mA,
                row_indices,
                (IndexType)2);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - dup, ordered
    {
        IndexArrayType row_indices = {0, 1, 1, 2};

        std::vector<double> vecAnswer = {1, 9, 10, 0};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        IndexType M = 4;
        std::vector<double> res = {1, 0, 1, 0};
        Vector<double> result(res, 0.);

        extract(result,
                NoMask(),
                Plus<double>(),
                mA,
                row_indices,
                (IndexType)3);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - dup, permuted
    {
        IndexArrayType row_indices = {0, 2, 2, 1};

        std::vector<double> vecAnswer = {8, 4, 5, 1};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        IndexType M = 4;
        std::vector<double> res = {0, 0, 1, 1};
        Vector<double> result(res, 0.);

        extract(result,
                NoMask(),
                Plus<double>(),
                mA,
                row_indices,
                (IndexType)0);

        BOOST_CHECK_EQUAL(result, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_col_test_nomask_accum_trans)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6, 0},
                                             {0, 5, 7, 9},
                                             {4, 0, 2, 0}};
    Matrix<double, DirectedMatrixTag> mA(matA, 0);

    // I - AllIndices
    {
        std::vector<double> vecAnswer = {1, 5, 8, 10};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        IndexType M = 3;
        std::vector<double> res = {1, 0, 1, 1};
        Vector<double> result(res, 0.);

        extract(result,
                NoMask(),
                Plus<double>(),
                transpose(mA),
                AllIndices(),
                (IndexType)1);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - no dup, ordered
    {
        IndexArrayType row_indices = {0, 2};

        std::vector<double> vecAnswer = {5, 2};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        IndexType M = 2;
        std::vector<double> res = {1, 0};
        Vector<double> result(res, 0.);

        extract(result,
                NoMask(),
                Plus<double>(),
                transpose(mA),
                row_indices,
                (IndexType)2);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - no dup, permuted
    {
        IndexArrayType row_indices = {2, 0};

        std::vector<double> vecAnswer = {3, 4};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        IndexType M = 2;
        std::vector<double> res = {1, 0};
        Vector<double> result(res, 0.);

        extract(result,
                NoMask(),
                Plus<double>(),
                transpose(mA),
                row_indices,
                (IndexType)2);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - dup, ordered
    {
        IndexArrayType row_indices = {0, 1, 1, 2};

        std::vector<double> vecAnswer = {5, 0, 1, 2};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        IndexType M = 4;
        std::vector<double> res = {1, 0, 1, 0};
        Vector<double> result(res, 0.);

        extract(result,
                NoMask(),
                Plus<double>(),
                transpose(mA),
                row_indices,
                (IndexType)2);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - dup, permuted
    {
        IndexArrayType row_indices = {0, 2, 2, 1};

        std::vector<double> vecAnswer = {8, 6, 7, 2};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        IndexType M = 4;
        std::vector<double> res = {0, 0, 1, 1};
        Vector<double> result(res, 0.);

        extract(result,
                NoMask(),
                Plus<double>(),
                transpose(mA),
                row_indices,
                (IndexType)0);

        BOOST_CHECK_EQUAL(result, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_col_test_noscmp_noaccum_notrans)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6, 0},
                                             {0, 5, 0, 9},
                                             {4, 0, 2, 0}};
    Matrix<double> mA(matA, 0);

    std::vector<uint8_t> vecm3 = {1, 0, 99};    // one stored zero:
    Vector<uint8_t> mask3(vecm3, 99);           // [1, 0, -]
    std::vector<uint8_t> vecm2 = {1, 0};        // one stored zero:
    Vector<uint8_t> mask2(vecm2, 99);           // [1, 0]

    // I - AllIndices
    {
        std::vector<double> vecAnswer = {6, 0, 0};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        std::vector<double> res = {1, 1, 1};

        // replace
        Vector<double> result(res, 0.);
        extract(result, mask3, NoAccumulate(),
                mA, AllIndices(), (IndexType)2, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    // I - AllIndices
    {
        std::vector<double> vecAnswer = {6, 1, 1};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        std::vector<double> res = {1, 1, 1};

        // merge
        Vector<double> result(res, 0.);
        extract(result, mask3, NoAccumulate(),
                mA, AllIndices(), (IndexType)2, false);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - no dup, ordered
    {
        IndexArrayType row_indices = {0, 2};

        std::vector<double> vecAnswer = {1, 0};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        std::vector<double> res = {1, 1};

        // replace
        Vector<double> result(res, 0.);
        extract(result, mask2, NoAccumulate(),
                mA, row_indices, (IndexType)1, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    // I - no dup, ordered
    {
        IndexArrayType row_indices = {0, 2};

        std::vector<double> vecAnswer = {1, 1};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        std::vector<double> res = {1, 1};

        // merge
        Vector<double> result(res, 0.);
        extract(result, mask2, NoAccumulate(),
                mA, row_indices, (IndexType)1, false);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - no dup, permuted
    {
        IndexArrayType row_indices = {2, 0};

        std::vector<double> vecAnswer = {0, 0};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        std::vector<double> res = {0, 1};

        // replace
        Vector<double> result(res, 0.);
        extract(result, mask2, NoAccumulate(),
                mA, row_indices, (IndexType)1, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        IndexArrayType row_indices = {2, 0};

        std::vector<double> vecAnswer = {0, 1};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        std::vector<double> res = {1, 1};

        // merge
        Vector<double> result(res, 0.);
        extract(result, mask2, NoAccumulate(),
                mA, row_indices, (IndexType)1, false);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - dup, ordered
    {
        IndexArrayType row_indices = {0, 0, 2};

        std::vector<double> vecAnswer = {1, 0, 0};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        std::vector<double> res = {9, 9, 9};

        // replace
        Vector<double> result(res, 0.);
        extract(result, mask3, NoAccumulate(),
                mA, row_indices, (IndexType)1, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        IndexArrayType row_indices = {0, 0, 2};

        std::vector<double> vecAnswer = {1, 9, 9};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        std::vector<double> res = {9, 9, 9};

        // merge
        Vector<double> result(res, 0.);
        extract(result, mask3, NoAccumulate(),
                mA, row_indices, (IndexType)1, false);

        BOOST_CHECK_EQUAL(result, answer);
    }
    // I - dup, permuted
    {
        IndexArrayType row_indices = {2, 0, 0};

        std::vector<double> vecAnswer = {0, 0, 0};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        std::vector<double> res = {9, 9, 9};

        // replace
        Vector<double> result(res, 0.);
        extract(result, mask3, NoAccumulate(),
                mA, row_indices, (IndexType)1, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        IndexArrayType row_indices = {2, 0, 0};

        std::vector<double> vecAnswer = {0, 9, 9};
        Vector<double> answer(vecAnswer, 0);

        // Output rank
        std::vector<double> res = {9, 9, 9};

        // merge
        Vector<double> result(res, 0.);
        extract(result, mask3, NoAccumulate(),
                mA, row_indices, (IndexType)1, false);

        BOOST_CHECK_EQUAL(result, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_col_test_noscmp_noaccum_trans)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6, 0},
                                             {0, 5, 7, 9},
                                             {4, 0, 2, 0}};
    Matrix<double> mA(matA, 0);

    std::vector<uint8_t> vecm3 = {1, 0, 99};    // one stored zero:
    Vector<uint8_t> mask3(vecm3, 99);           // [1, 0, -]
    std::vector<uint8_t> vecm4 = {1, 0, 99, 1}; // one stored zero:
    Vector<uint8_t> mask4(vecm4, 99);           // [1, 0, -, 1]


    // I - AllIndices
    {
        std::vector<double> vecAnswer = {4, 0, 0, 0};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9, 9};
        Vector<double> result(res);

        extract(result, mask4, NoAccumulate(),
                transpose(mA), AllIndices(), (IndexType)2, true);  // replace

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        std::vector<double> vecAnswer = {4, 9, 9, 0};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9, 9};
        Vector<double> result(res);

        extract(result, mask4, NoAccumulate(),
                transpose(mA), AllIndices(), (IndexType)2, false);  // merge

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - no dup, ordered
    {
        IndexArrayType row_indices = {0, 2, 3};

        std::vector<double> vecAnswer = {4, 0, 0};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res);

        extract(result, mask3, NoAccumulate(),
                transpose(mA), row_indices, (IndexType)2, true);  // replace

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        IndexArrayType row_indices = {0, 2, 3};

        std::vector<double> vecAnswer = {4, 9, 9};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res);

        extract(result, mask3, NoAccumulate(),
                transpose(mA), row_indices, (IndexType)2, false);  // merge

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - no dup, permuted
    {
        IndexArrayType row_indices = {2, 3, 0};

        std::vector<double> vecAnswer = {2, 0, 0};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res);

        extract(result, mask3,  NoAccumulate(),
                transpose(mA), row_indices, (IndexType)2, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        IndexArrayType row_indices = {2, 3, 0};

        std::vector<double> vecAnswer = {2, 9, 9};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res);

        extract(result, mask3,  NoAccumulate(),
                transpose(mA), row_indices, (IndexType)2, false);

        BOOST_CHECK_EQUAL(result, answer);
    }


    // I - dup, ordered
    {
        IndexArrayType row_indices = {0, 2, 3, 3};

        std::vector<double> vecAnswer = {4, 0, 0, 0};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9, 9};
        Vector<double> result(res);

        extract(result, mask4, NoAccumulate(),
                transpose(mA), row_indices, (IndexType)2, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        IndexArrayType row_indices = {0, 2, 3, 3};

        std::vector<double> vecAnswer = {4, 9, 9, 0};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9, 9};
        Vector<double> result(res);

        extract(result, mask4, NoAccumulate(),
                transpose(mA), row_indices, (IndexType)2, false);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - dup, permuted
    {
        IndexArrayType row_indices = {0, 2, 3, 0};

        std::vector<double> vecAnswer = {4, 0, 0, 4};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9, 9};
        Vector<double> result(res);

        extract(result, mask4, NoAccumulate(),
                transpose(mA), row_indices, (IndexType)2, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        IndexArrayType row_indices = {0, 2, 3, 0};

        std::vector<double> vecAnswer = {4, 9, 9, 4};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9, 9};
        Vector<double> result(res);

        extract(result, mask4, NoAccumulate(),
                transpose(mA), row_indices, (IndexType)2, false);

        BOOST_CHECK_EQUAL(result, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_col_test_noscmp_accum_notrans)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6, 0},
                                             {0, 5, 0, 9},
                                             {4, 0, 2, 0}};
    Matrix<double> mA(matA, 0);

    std::vector<uint8_t> vecm3 = {1, 0, 99};    // one stored zero:
    Vector<uint8_t> mask3(vecm3, 99);           // [1, 0, -]
    std::vector<uint8_t> vecm2 = {1, 0};        // one stored zero:
    Vector<uint8_t> mask2(vecm2, 99);           // [1, 0]

    // I - AllIndices
    {
        std::vector<double> vecAnswer = {15, 0, 0};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res, 0.);

        extract(result, mask3, Plus<double>(),
                mA, AllIndices(), (IndexType)2, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        std::vector<double> vecAnswer = {15, 9, 9};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res, 0.);

        extract(result, mask3, Plus<double>(),
                mA, AllIndices(), (IndexType)2, false);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - no dup, ordered
    {
        IndexArrayType row_indices = {0, 2};

        std::vector<double> vecAnswer = {10, 0};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9};
        Vector<double> result(res, 0.);

        extract(result, mask2, Plus<double>(),
                mA, row_indices, (IndexType)1, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        IndexArrayType row_indices = {0, 2};

        std::vector<double> vecAnswer = {10, 9};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9};
        Vector<double> result(res, 0.);

        extract(result, mask2, Plus<double>(),
                mA, row_indices, (IndexType)1, false);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - no dup, permuted
    {
        IndexArrayType row_indices = {2, 0};

        std::vector<double> vecAnswer = {9, 0};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9};
        Vector<double> result(res, 0.);

        extract(result, mask2, Plus<double>(),
                mA, row_indices, (IndexType)1, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        IndexArrayType row_indices = {2, 0};

        std::vector<double> vecAnswer = {9, 9};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9};
        Vector<double> result(res, 0.);

        extract(result, mask2, Plus<double>(),
                mA, row_indices, (IndexType)1, false);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - dup, ordered
    {
        IndexArrayType row_indices = {0, 0, 2};

        std::vector<double> vecAnswer = {10, 0, 0};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res, 0.);

        extract(result, mask3, Plus<double>(),
                mA, row_indices, (IndexType)1, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        IndexArrayType row_indices = {0, 0, 2};

        std::vector<double> vecAnswer = {10, 9, 9};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res, 0.);

        extract(result, mask3, Plus<double>(),
                mA, row_indices, (IndexType)1, false);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - dup, permuted
    {
        IndexArrayType row_indices = {2, 0, 0};

        std::vector<double> vecAnswer = {9, 0, 0};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res, 0.);

        extract(result, mask3, Plus<double>(),
                mA, row_indices, (IndexType)1, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        IndexArrayType row_indices = {2, 0, 0};

        std::vector<double> vecAnswer = {9, 9, 9};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res, 0.);

        extract(result, mask3, Plus<double>(),
                mA, row_indices, (IndexType)1, false);

        BOOST_CHECK_EQUAL(result, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_col_test_noscmp_accum_trans)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6, 0},
                                             {0, 5, 7, 9},
                                             {4, 0, 2, 0}};
    Matrix<double> mA(matA, 0);

    std::vector<uint8_t> vecm3 = {1, 0, 99};    // one stored zero:
    Vector<uint8_t> mask3(vecm3, 99);           // [1, 0, -]
    std::vector<uint8_t> vecm4 = {1, 0, 99, 1}; // one stored zero:
    Vector<uint8_t> mask4(vecm4, 99);           // [1, 0, -, 1]

    // I - AllIndices
    {
        std::vector<double> vecAnswer = {13, 0, 0, 9};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9, 9};
        Vector<double> result(res);

        extract(result, mask4, Plus<double>(),
                transpose(mA), AllIndices(), (IndexType)2, true);  // replace

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        std::vector<double> vecAnswer = {13, 9, 9, 9};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9, 9};
        Vector<double> result(res);

        extract(result, mask4, Plus<double>(),
                transpose(mA), AllIndices(), (IndexType)2, false);  // merge

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - no dup, ordered
    {
        IndexArrayType row_indices = {0, 2, 3};

        std::vector<double> vecAnswer = {13, 0, 0};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res);

        extract(result, mask3, Plus<double>(),
                transpose(mA), row_indices, (IndexType)2, true);  // replace

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        IndexArrayType row_indices = {0, 2, 3};

        std::vector<double> vecAnswer = {13, 9, 9};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res);

        extract(result, mask3, Plus<double>(),
                transpose(mA), row_indices, (IndexType)2, false);  // merge

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - no dup, permuted
    {
        IndexArrayType row_indices = {2, 3, 0};

        std::vector<double> vecAnswer = {11, 0, 0};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res);

        extract(result, mask3,  Plus<double>(),
                transpose(mA), row_indices, (IndexType)2, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        IndexArrayType row_indices = {2, 3, 0};

        std::vector<double> vecAnswer = {11, 9, 9};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res);

        extract(result, mask3,  Plus<double>(),
                transpose(mA), row_indices, (IndexType)2, false);

        BOOST_CHECK_EQUAL(result, answer);
    }


    // I - dup, ordered
    {
        IndexArrayType row_indices = {0, 2, 3, 3};

        std::vector<double> vecAnswer = {13, 0, 0, 9};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9, 9};
        Vector<double> result(res);

        extract(result, mask4, Plus<double>(),
                transpose(mA), row_indices, (IndexType)2, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        IndexArrayType row_indices = {0, 2, 3, 3};

        std::vector<double> vecAnswer = {13, 9, 9, 9};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9, 9};
        Vector<double> result(res);

        extract(result, mask4, Plus<double>(),
                transpose(mA), row_indices, (IndexType)2, false);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - dup, permuted
    {
        IndexArrayType row_indices = {0, 2, 3, 0};

        std::vector<double> vecAnswer = {13, 0, 0, 13};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9, 9};
        Vector<double> result(res);

        extract(result, mask4, Plus<double>(),
                transpose(mA), row_indices, (IndexType)2, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        IndexArrayType row_indices = {0, 2, 3, 0};

        std::vector<double> vecAnswer = {13, 9, 9, 13};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9, 9};
        Vector<double> result(res);

        extract(result, mask4, Plus<double>(),
                transpose(mA), row_indices, (IndexType)2, false);

        BOOST_CHECK_EQUAL(result, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_col_test_scmp_noaccum_notrans)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6, 0},
                                             {0, 5, 0, 9},
                                             {4, 0, 2, 0}};
    Matrix<double> mA(matA, 0);

    std::vector<uint8_t> vecm3 = {1, 0, 99};    // one stored zero:
    Vector<uint8_t> mask3(vecm3, 99);           // [1, 0, -]
    std::vector<uint8_t> vecm2 = {1, 0};        // one stored zero:
    Vector<uint8_t> mask2(vecm2, 99);           // [1, 0]

    // I - AllIndices
    {
        std::vector<double> vecAnswer = {0, 0, 2};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res, 0.);
        extract(result, complement(mask3), NoAccumulate(),
                mA, AllIndices(), (IndexType)2, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        std::vector<double> vecAnswer = {9, 0, 2};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res, 0.);
        extract(result, complement(mask3), NoAccumulate(),
                mA, AllIndices(), (IndexType)2, false);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - no dup, ordered
    {
        IndexArrayType row_indices = {0, 2};

        std::vector<double> vecAnswer = {0, 0};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9};
        Vector<double> result(res, 0.);
        extract(result, complement(mask2), NoAccumulate(),
                mA, row_indices, (IndexType)1, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        IndexArrayType row_indices = {0, 2};

        std::vector<double> vecAnswer = {9, 0};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9};
        Vector<double> result(res, 0.);
        extract(result, complement(mask2), NoAccumulate(),
                mA, row_indices, (IndexType)1, false);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - no dup, permuted
    {
        IndexArrayType row_indices = {2, 0};

        std::vector<double> vecAnswer = {0, 1};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9};
        Vector<double> result(res, 0.);
        extract(result, complement(mask2), NoAccumulate(),
                mA, row_indices, (IndexType)1, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        IndexArrayType row_indices = {2, 0};

        std::vector<double> vecAnswer = {9, 1};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9};
        Vector<double> result(res, 0.);
        extract(result, complement(mask2), NoAccumulate(),
                mA, row_indices, (IndexType)1, false);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - dup, ordered
    {
        IndexArrayType row_indices = {0, 0, 2};

        std::vector<double> vecAnswer = {0, 1, 0};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res, 0.);
        extract(result, complement(mask3), NoAccumulate(),
                mA, row_indices, (IndexType)1, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        IndexArrayType row_indices = {0, 0, 2};

        std::vector<double> vecAnswer = {9, 1, 0};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res, 0.);
        extract(result, complement(mask3), NoAccumulate(),
                mA, row_indices, (IndexType)1, false);

        BOOST_CHECK_EQUAL(result, answer);
    }
    // I - dup, permuted
    {
        IndexArrayType row_indices = {2, 0, 0};

        std::vector<double> vecAnswer = {0, 1, 1};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res, 0.);
        extract(result, complement(mask3), NoAccumulate(),
                mA, row_indices, (IndexType)1, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        IndexArrayType row_indices = {2, 0, 0};

        std::vector<double> vecAnswer = {9, 1, 1};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res, 0.);
        extract(result, complement(mask3), NoAccumulate(),
                mA, row_indices, (IndexType)1, false);

        BOOST_CHECK_EQUAL(result, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_col_test_scmp_noaccum_trans)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6, 0},
                                             {0, 5, 7, 9},
                                             {4, 0, 2, 0}};
    Matrix<double> mA(matA, 0);

    std::vector<uint8_t> vecm3 = {1, 0, 99};    // one stored zero:
    Vector<uint8_t> mask3(vecm3, 99);           // [1, 0, -]
    std::vector<uint8_t> vecm4 = {1, 0, 99, 1}; // one stored zero:
    Vector<uint8_t> mask4(vecm4, 99);           // [1, 0, -, 1]


    // I - AllIndices
    {
        std::vector<double> vecAnswer = {0, 0, 2, 0};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9, 9};
        Vector<double> result(res);

        extract(result, complement(mask4), NoAccumulate(),
                transpose(mA), AllIndices(), (IndexType)2, true);  // replace

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        std::vector<double> vecAnswer = {9, 0, 2, 9};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9, 9};
        Vector<double> result(res);

        extract(result, complement(mask4), NoAccumulate(),
                transpose(mA), AllIndices(), (IndexType)2, false);  // merge

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - no dup, ordered
    {
        IndexArrayType row_indices = {0, 2, 3};

        std::vector<double> vecAnswer = {0, 2, 0};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res);

        extract(result, complement(mask3), NoAccumulate(),
                transpose(mA), row_indices, (IndexType)2, true);  // replace

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        IndexArrayType row_indices = {0, 2, 3};

        std::vector<double> vecAnswer = {9, 2, 0};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res);

        extract(result, complement(mask3), NoAccumulate(),
                transpose(mA), row_indices, (IndexType)2, false);  // merge

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - no dup, permuted
    {
        IndexArrayType row_indices = {2, 3, 0};

        std::vector<double> vecAnswer = {0, 0, 4};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res);

        extract(result, complement(mask3),  NoAccumulate(),
                transpose(mA), row_indices, (IndexType)2, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        IndexArrayType row_indices = {2, 3, 0};

        std::vector<double> vecAnswer = {9, 0, 4};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res);

        extract(result, complement(mask3),  NoAccumulate(),
                transpose(mA), row_indices, (IndexType)2, false);

        BOOST_CHECK_EQUAL(result, answer);
    }


    // I - dup, ordered
    {
        IndexArrayType row_indices = {0, 2, 3, 3};

        std::vector<double> vecAnswer = {0, 2, 0, 0};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9, 9};
        Vector<double> result(res);

        extract(result, complement(mask4), NoAccumulate(),
                transpose(mA), row_indices, (IndexType)2, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        IndexArrayType row_indices = {0, 2, 3, 3};

        std::vector<double> vecAnswer = {9, 2, 0, 9};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9, 9};
        Vector<double> result(res);

        extract(result, complement(mask4), NoAccumulate(),
                transpose(mA), row_indices, (IndexType)2, false);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - dup, permuted
    {
        IndexArrayType row_indices = {0, 2, 3, 0};

        std::vector<double> vecAnswer = {0, 2, 0, 0};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9, 9};
        Vector<double> result(res);

        extract(result, complement(mask4), NoAccumulate(),
                transpose(mA), row_indices, (IndexType)2, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        IndexArrayType row_indices = {0, 2, 3, 0};

        std::vector<double> vecAnswer = {9, 2, 0, 9};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9, 9};
        Vector<double> result(res);

        extract(result, complement(mask4), NoAccumulate(),
                transpose(mA), row_indices, (IndexType)2, false);

        BOOST_CHECK_EQUAL(result, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_col_test_scmp_accum_notrans)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6, 0},
                                             {0, 5, 0, 9},
                                             {4, 0, 2, 0}};
    Matrix<double> mA(matA, 0);

    std::vector<uint8_t> vecm3 = {1, 0, 99};    // one stored zero:
    Vector<uint8_t> mask3(vecm3, 99);           // [1, 0, -]
    std::vector<uint8_t> vecm2 = {1, 0};        // one stored zero:
    Vector<uint8_t> mask2(vecm2, 99);           // [1, 0]

    // I - AllIndices
    {
        std::vector<double> vecAnswer = {0, 14, 9};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res, 0.);

        extract(result, complement(mask3), Plus<double>(),
                mA, AllIndices(), (IndexType)1, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        std::vector<double> vecAnswer = {9, 14, 9};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res, 0.);

        extract(result, complement(mask3), Plus<double>(),
                mA, AllIndices(), (IndexType)1, false);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - no dup, ordered
    {
        IndexArrayType row_indices = {0, 2};

        std::vector<double> vecAnswer = {0, 9};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9};
        Vector<double> result(res, 0.);

        extract(result, complement(mask2), Plus<double>(),
                mA, row_indices, (IndexType)1, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        IndexArrayType row_indices = {0, 2};

        std::vector<double> vecAnswer = {9, 9};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9};
        Vector<double> result(res, 0.);

        extract(result, complement(mask2), Plus<double>(),
                mA, row_indices, (IndexType)1, false);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - no dup, permuted
    {
        IndexArrayType row_indices = {2, 0};

        std::vector<double> vecAnswer = {0, 10};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9};
        Vector<double> result(res, 0.);

        extract(result, complement(mask2), Plus<double>(),
                mA, row_indices, (IndexType)1, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        IndexArrayType row_indices = {2, 0};

        std::vector<double> vecAnswer = {9, 10};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9};
        Vector<double> result(res, 0.);

        extract(result, complement(mask2), Plus<double>(),
                mA, row_indices, (IndexType)1, false);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - dup, ordered
    {
        IndexArrayType row_indices = {0, 0, 2};

        std::vector<double> vecAnswer = {0, 10, 9};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res, 0.);

        extract(result, complement(mask3), Plus<double>(),
                mA, row_indices, (IndexType)1, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        IndexArrayType row_indices = {0, 0, 2};

        std::vector<double> vecAnswer = {9, 10, 9};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res, 0.);

        extract(result, complement(mask3), Plus<double>(),
                mA, row_indices, (IndexType)1, false);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - dup, permuted
    {
        IndexArrayType row_indices = {2, 0, 0};

        std::vector<double> vecAnswer = {0, 10, 10};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res, 0.);

        extract(result, complement(mask3), Plus<double>(),
                mA, row_indices, (IndexType)1, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        IndexArrayType row_indices = {2, 0, 0};

        std::vector<double> vecAnswer = {9, 10, 10};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res, 0.);

        extract(result, complement(mask3), Plus<double>(),
                mA, row_indices, (IndexType)1, false);

        BOOST_CHECK_EQUAL(result, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(extract_col_test_scmp_accum_trans)
{
    std::vector<std::vector<double>> matA = {{8, 1, 6, 0},
                                             {0, 5, 7, 9},
                                             {4, 0, 2, 0}};
    Matrix<double> mA(matA, 0);

    std::vector<uint8_t> vecm3 = {1, 0, 99};    // one stored zero:
    Vector<uint8_t> mask3(vecm3, 99);           // [1, 0, -]
    std::vector<uint8_t> vecm4 = {1, 0, 99, 1}; // one stored zero:
    Vector<uint8_t> mask4(vecm4, 99);           // [1, 0, -, 1]

    // I - AllIndices
    {
        std::vector<double> vecAnswer = {0, 9, 11, 0};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9, 9};
        Vector<double> result(res);

        extract(result, complement(mask4), Plus<double>(),
                transpose(mA), AllIndices(), (IndexType)2, true);  // replace

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        std::vector<double> vecAnswer = {9, 9, 11, 9};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9, 9};
        Vector<double> result(res);

        extract(result, complement(mask4), Plus<double>(),
                transpose(mA), AllIndices(), (IndexType)2, false);  // merge

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - no dup, ordered
    {
        IndexArrayType row_indices = {0, 2, 3};

        std::vector<double> vecAnswer = {0, 11, 9};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res);

        extract(result, complement(mask3), Plus<double>(),
                transpose(mA), row_indices, (IndexType)2, true);  // replace

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        IndexArrayType row_indices = {0, 2, 3};

        std::vector<double> vecAnswer = {9, 11, 9};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res);

        extract(result, complement(mask3), Plus<double>(),
                transpose(mA), row_indices, (IndexType)2, false);  // merge

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - no dup, permuted
    {
        IndexArrayType row_indices = {2, 3, 0};

        std::vector<double> vecAnswer = {0, 9, 13};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res);

        extract(result, complement(mask3),  Plus<double>(),
                transpose(mA), row_indices, (IndexType)2, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        IndexArrayType row_indices = {2, 3, 0};

        std::vector<double> vecAnswer = {9, 9, 13};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9};
        Vector<double> result(res);

        extract(result, complement(mask3),  Plus<double>(),
                transpose(mA), row_indices, (IndexType)2, false);

        BOOST_CHECK_EQUAL(result, answer);
    }


    // I - dup, ordered
    {
        IndexArrayType row_indices = {0, 2, 3, 3};

        std::vector<double> vecAnswer = {0, 11, 9, 0};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9, 9};
        Vector<double> result(res);

        extract(result, complement(mask4), Plus<double>(),
                transpose(mA), row_indices, (IndexType)2, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        IndexArrayType row_indices = {0, 2, 3, 3};

        std::vector<double> vecAnswer = {9, 11, 9, 9};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9, 9};
        Vector<double> result(res);

        extract(result, complement(mask4), Plus<double>(),
                transpose(mA), row_indices, (IndexType)2, false);

        BOOST_CHECK_EQUAL(result, answer);
    }

    // I - dup, permuted
    {
        IndexArrayType row_indices = {0, 2, 3, 0};

        std::vector<double> vecAnswer = {0, 11, 9, 0};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9, 9};
        Vector<double> result(res);

        extract(result, complement(mask4), Plus<double>(),
                transpose(mA), row_indices, (IndexType)2, true);

        BOOST_CHECK_EQUAL(result, answer);
    }
    {
        IndexArrayType row_indices = {0, 2, 3, 0};

        std::vector<double> vecAnswer = {9, 11, 9, 9};
        Vector<double> answer(vecAnswer, 0);

        std::vector<double> res = {9, 9, 9, 9};
        Vector<double> result(res);

        extract(result, complement(mask4), Plus<double>(),
                transpose(mA), row_indices, (IndexType)2, false);

        BOOST_CHECK_EQUAL(result, answer);
    }
}

BOOST_AUTO_TEST_SUITE_END()
