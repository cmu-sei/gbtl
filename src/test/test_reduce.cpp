/*
 * GraphBLAS Template Library (GBTL), Version 3.0
 *
 * Copyright 2020 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors.
 *
 * THIS MATERIAL WAS PREPARED AS AN ACCOUNT OF WORK SPONSORED BY AN AGENCY OF
 * THE UNITED STATES GOVERNMENT.  NEITHER THE UNITED STATES GOVERNMENT NOR THE
 * UNITED STATES DEPARTMENT OF ENERGY, NOR THE UNITED STATES DEPARTMENT OF
 * DEFENSE, NOR CARNEGIE MELLON UNIVERSITY, NOR BATTELLE, NOR ANY OF THEIR
 * EMPLOYEES, NOR ANY JURISDICTION OR ORGANIZATION THAT HAS COOPERATED IN THE
 * DEVELOPMENT OF THESE MATERIALS, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS,
 * OR USEFULNESS OR ANY INFORMATION, APPARATUS, PRODUCT, SOFTWARE, OR PROCESS
 * DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED
 * RIGHTS.
 *
 * Released under a BSD-style license, please see LICENSE file or contact
 * permission@sei.cmu.edu for full terms.
 *
 * [DISTRIBUTION STATEMENT A] This material has been approved for public release
 * and unlimited distribution.  Please see Copyright notice for non-US
 * Government use and distribution.
 *
 * This Software includes and/or makes use of the following Third-Party Software
 * subject to its own license:
 *
 * 1. Boost Unit Test Framework
 * (https://www.boost.org/doc/libs/1_45_0/libs/test/doc/html/utf.html)
 * Copyright 2001 Boost software license, Gennadiy Rozental.
 *
 * DM20-0442
 */

#define GRAPHBLAS_LOGGING_LEVEL 0

#include <functional>
#include <iostream>
#include <vector>

#include <graphblas/graphblas.hpp>

using namespace grb;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE reduce_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

/// @todo This is a merge of two test files for reduce.  Many are redundant.

//****************************************************************************
// reduce matrix to vector
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_reduce_matvec_bad_dimensions)
{
    std::vector<std::vector<double> > m3x3_dense = {{12, 1, 3},
                                                    {0,  0, 0},
                                                    {7,  0, 9}};

    std::vector<std::vector<double> > m6x4_dense = {{5, 0, 1, 0},
                                                    {6, 7, 0, 0},
                                                    {6, 7, 0, 0},
                                                    {0, 5, 0, 0},
                                                    {6, 7, 0, 0},
                                                    {0, 0, 0, 0}};

    Matrix<double, DirectedMatrixTag> mA(m3x3_dense, 0.);
    Matrix<double, DirectedMatrixTag> mB(m6x4_dense, 0.);

    Vector<double, SparseTag> w4(4);

    // incompatible output vector dimensions
    BOOST_CHECK_THROW(
        (reduce(w4,
                NoMask(),
                NoAccumulate(),
                Plus<double>(), mA)),
        DimensionException);

    BOOST_CHECK_THROW(
        (reduce(w4,
                NoMask(),
                NoAccumulate(),
                Plus<double>(), mB)),
        DimensionException);

    // vector-scalar version - no bad dimensions

    // matrix-scalar version - no bad dimensions
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(reduce_matvec_test_bad_dimension)
{
    // Build some matrices.
    IndexArrayType i_A    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j_A    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_A = {1, 1, 1, 2, 2, 1, 3, 5, 3, 4};
    Matrix<double, DirectedMatrixTag> A(4, 4);
    A.build(i_A, j_A, v_A);

    IndexArrayType i_C    = {0, 1, 2};
    std::vector<double> v_C = {1, 1, 1};
    Vector<double> C(3);
    C.build(i_C, v_C);

    BOOST_CHECK_THROW(
        (reduce(C, NoMask(), NoAccumulate(),
                PlusMonoid<double>(), A)),
        DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_reduce_matvec_mask_replace_bad_dimensions)
{
    std::vector<std::vector<double> > m3x3_dense = {{12, 1, 3},
                                                    {0,  0, 0},
                                                    {7,  0, 9}};
    std::vector<double> u3_dense = {1, 1, 0};
    std::vector<double> u4_dense = {0, 0, 13, 1};

    Matrix<double, DirectedMatrixTag> mA(m3x3_dense, 0.);

    Vector<double, SparseTag> u3(u3_dense, 0.);
    Vector<double, SparseTag> m4(u4_dense, 0.);
    Vector<double, SparseTag> w3(3);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (reduce(w3,
                m4,
                NoAccumulate(),
                Plus<double>(), mA,
                REPLACE)),
        DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_reduce_matvec_mask_merge_bad_dimensions)
{
    std::vector<std::vector<double> > m3x3_dense = {{12, 1, 3},
                                                    {0,  0, 0},
                                                    {7,  0, 9}};
    std::vector<double> v3_ones = {1, 1, 1};
    std::vector<double> u4_dense = {0, 0, 13, 1};

    Matrix<double, DirectedMatrixTag> mA(m3x3_dense, 0.);
    Vector<double, SparseTag> m4(u4_dense, 0.);
    Vector<double, SparseTag> w3(v3_ones, 0.);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (reduce(w3,
                m4,
                NoAccumulate(),
                Plus<double>(), mA)),
        DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(reduce_matvec_test_transpose_bad_dimensions)
{
    // Build some matrices.
    IndexArrayType i_A    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j_A    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_A = {1, 1, 1, 2, 2, 1, 3, 5, 3, 4};
    Matrix<double, DirectedMatrixTag> A(4, 4);
    A.build(i_A, j_A, v_A);

    IndexArrayType j_C    = {0, 1, 2, 3, 4};
    std::vector<double> v_C = {1, 1, 1, 1, 1};
    Vector<double> C(5);
    C.build(j_C, v_C);

    BOOST_CHECK_THROW(
        (reduce(C, NoMask(), NoAccumulate(),
                PlusMonoid<double>(), transpose(A))),
        DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_reduce_matvec_scmp_replace_bad_dimensions)
{
    std::vector<std::vector<double> > m3x3_dense = {{12, 1, 3},
                                                    {0,  0, 0},
                                                    {7,  0, 9}};
    std::vector<double> u4_dense = {0, 0, 13, 1};

    Matrix<double, DirectedMatrixTag> mA(m3x3_dense, 0.);
    Vector<double, SparseTag> m4(u4_dense, 0.);
    Vector<double, SparseTag> w3(3);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (reduce(w3,
                complement(m4),
                NoAccumulate(),
                Plus<double>(), mA,
                REPLACE)),
        DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_reduce_matvec_scmp_merge_bad_dimensions)
{
    std::vector<std::vector<double> > m3x3_dense = {{12, 1, 3},
                                                    {0,  0, 0},
                                                    {7,  0, 9}};
    std::vector<double> u4_dense = {0, 0, 13, 1};
    std::vector<double> v3_ones = {1, 1, 1};

    Matrix<double, DirectedMatrixTag> mA(m3x3_dense, 0.);

    Vector<double, SparseTag> m4(u4_dense, 0.);
    Vector<double, SparseTag> w3(v3_ones, 0.);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (reduce(w3,
                complement(m4),
                NoAccumulate(),
                Plus<double>(), mA)),
        DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(reduce_matvec_test_noaccum)
{
    // Build some matrices.
    IndexArrayType i_A    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j_A    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_A = {1, 1, 1, 2, 2, 1, 3, 5, 3, 4};
    Matrix<double, DirectedMatrixTag> A(4, 4);
    A.build(i_A, j_A, v_A);

    IndexArrayType i_C    = {0, 1, 2, 3};
    std::vector<double> v_C = {1, 1, 1, 1};
    Vector<double> C(4);
    C.build(i_C, v_C);

    IndexArrayType i_answer    = {0, 1, 2, 3};
    std::vector<double> v_answer = {2, 5, 9, 7};
    Vector<double> answer(4);
    answer.build(i_answer, v_answer);

    reduce(C, NoMask(), NoAccumulate(),
           PlusMonoid<double>(), A);
    BOOST_CHECK_EQUAL(C, answer);

    reduce(C, NoMask(), NoAccumulate(),
           Plus<double>(), A);
    BOOST_CHECK_EQUAL(C, answer);

    reduce(C, NoMask(), NoAccumulate(),
           add_monoid(ArithmeticSemiring<double>()), A);
    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_reduce_matvec_reg)
{
    std::vector<std::vector<double> > m3x3_dense = {{12, 1, 3},
                                                    {0,  0, 0},
                                                    {7,  0, 9}};

    std::vector<std::vector<double> > m6x4_dense = {{5, 0, 1, 0},
                                                    {6, 7, 0, 0},
                                                    {6, 7, 0, 0},
                                                    {0, 5, 0, 0},
                                                    {6, 7, 0, 0},
                                                    {0, 0, 0, 0}};

    std::vector<double> ans3x3_dense  = {16, 0, 16};
    std::vector<double> ans6x4_dense = {6, 13, 13, 5, 13, 0};

    // Matrix to vector reduction
    Matrix<double, DirectedMatrixTag> mA(m3x3_dense, 0.);
    Matrix<double, DirectedMatrixTag> mB(m6x4_dense, 0.);
    Matrix<double, DirectedMatrixTag> empty6x4(6, 4);
    Vector<double> result3(3);
    Vector<double> result6(6);
    Vector<double> empty_result6(6);
    Vector<double> ansA(ans3x3_dense, 0.);
    Vector<double> ansB(ans6x4_dense, 0.);

    reduce(result3,
           NoMask(),
           NoAccumulate(),
           Plus<double>(), mA);
    BOOST_CHECK_EQUAL(result3, ansA);

    reduce(result6,
           NoMask(),
           NoAccumulate(),
           Plus<double>(), mB);
    BOOST_CHECK_EQUAL(result6, ansB);

    reduce(empty_result6,
           NoMask(),
           NoAccumulate(),
           Plus<double>(), empty6x4);
    BOOST_CHECK_EQUAL(empty_result6.nvals(), 0);

}

//****************************************************************************
BOOST_AUTO_TEST_CASE(reduce_matvec_test_nomask_accum)
{
    // Build some matrices.
    IndexArrayType i_A    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j_A    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_A = {1, 1, 1, 2, 2, 1, 3, 5, 3, 4};
    Matrix<double, DirectedMatrixTag> A(4, 4);
    A.build(i_A, j_A, v_A);

    IndexArrayType i_C    = {0, 1, 2, 3};
    std::vector<double> v_C = {1, 1, 1, 1};
    Vector<double> C(4);
    C.build(i_C, v_C);

    IndexArrayType i_answer    = {0, 1, 2, 3};
    std::vector<double> v_answer = {3, 6, 10, 8};
    Vector<double> answer(4);
    answer.build(i_answer, v_answer);

    reduce(C, NoMask(), Plus<double>(),
           PlusMonoid<double>(), A);

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_reduce_matvec_mask_replace_second_accum)
{
    std::vector<std::vector<double> > m3x3_dense = {{12, 1, 3},
                                                    {0,  0, 0},
                                                    {7,  0, 9}};
    std::vector<double> u3_dense = {1, 1, 0};
    std::vector<double> v3_ones = {1, 1, 1};

    Matrix<double, DirectedMatrixTag> mA(m3x3_dense, 0.);
    Vector<double, SparseTag> m3(u3_dense, 0.);
    BOOST_CHECK_EQUAL(m3.nvals(), 2);

    {
        Vector<double> result(v3_ones, 0.);

        std::vector<double> ans = {16, 1, 0};
        Vector<double> answer(ans, 0.);

        reduce(result,
               m3,
               Second<double>(),
               Plus<double>(), mA,
               REPLACE);

        BOOST_CHECK_EQUAL(result.nvals(), 2);
        BOOST_CHECK_EQUAL(result, answer);
    }

    {
        Vector<double> result(v3_ones, 0.);

        std::vector<double> ans = {16, 0, 0};
        Vector<double> answer(ans, 0.);

        reduce(result,
               m3,
               NoAccumulate(),
               Plus<double>(), mA,
               REPLACE);

        BOOST_CHECK_EQUAL(result.nvals(), 1);
        BOOST_CHECK_EQUAL(result, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_reduce_matvec_mask_replace_stored_zero)
{
    // tests a computed and stored zero
    // tests a stored zero in the mask

    // Build some matrices.
    std::vector<std::vector<int> > A_dense = {{1, 1, 0, 0},
                                              {1, 2, 2, 0},
                                              {0, 2, 3, 3},
                                              {0, 0, 3, 4}};
    std::vector<int> mask_dense =  { 1, 1, 1, 0};
    std::vector<int> ones_dense =  { 1, 1, 1, 1};
    Matrix<int, DirectedMatrixTag> A(A_dense, 0);
    Vector<int> mask(mask_dense, 0);
    Vector<int> result(ones_dense, 0);

    BOOST_CHECK_EQUAL(mask.nvals(), 3);  // [ 1 1 1 - ]
    mask.setElement(3, 0);
    BOOST_CHECK_EQUAL(mask.nvals(), 4);  // [ 1 1 1 0 ]

    int const NIL(666);
    std::vector<int> ans = { 2, 5, 8,  NIL};
    Vector<int> answer(ans, NIL);

    reduce(result,
           mask,
           NoAccumulate(),
           Plus<double>(), A,
           REPLACE);
    BOOST_CHECK_EQUAL(result.nvals(), 3);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_reduce_matvec_mask_replace_transpose)
{
    // Build some matrices.
    std::vector<std::vector<int> > A_dense = {{1, 1, 0, 0, 1},
                                              {-1, 2, 2, 0, 1},
                                              {0, 2, 3, 3, 1},
                                              {0, 0, 3, 4, 1}};

    std::vector<int> mask_dense =  { 1, 0, 1, 0, 1};
    std::vector<int> ones_dense =  { 1, 1, 1, 1, 1};
    Matrix<int, DirectedMatrixTag> A(A_dense, 0);
    Vector<int> mask(mask_dense, 0);
    Vector<int> result(ones_dense, 0);

    int const NIL(666);
    std::vector<int> ans = { 0, NIL, 8,  NIL, 4};
    Vector<int> answer(ans, NIL);

    reduce(result,
           mask,
           NoAccumulate(),
           Plus<double>(), transpose(A),
           REPLACE);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_reduce_matvec_mask_merge)
{
    std::vector<std::vector<double> > m3x3_dense = {{12, 1, 3},
                                                    {0,  0, 0},
                                                    {7,  0, 9}};
    std::vector<double> u3_dense = {1, 1, 0};
    std::vector<double> v3_ones = {1, 1, 1};

    Matrix<double, DirectedMatrixTag> mA(m3x3_dense, 0.);
    Vector<double, SparseTag> m3(u3_dense, 0.); // [1 1 -]
    BOOST_CHECK_EQUAL(m3.nvals(), 2);

    {
        Vector<double> result(v3_ones, 0.);

        std::vector<double> ans = {16, 1, 1};
        Vector<double> answer(ans, 0.);

        reduce(result,
               m3,
               Second<double>(),
               Plus<double>(), mA);

        BOOST_CHECK_EQUAL(result.nvals(), 3);
        BOOST_CHECK_EQUAL(result, answer);
    }

    {
        Vector<double> result(v3_ones, 0.);

        std::vector<double> ans = {16, 0, 1};
        Vector<double> answer(ans, 0.);

        reduce(result,
               m3,
               NoAccumulate(),
               Plus<double>(), mA);

        BOOST_CHECK_EQUAL(result.nvals(), 2);
        BOOST_CHECK_EQUAL(result, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_reduce_matvec_mask_merge_stored_zero)
{
    std::vector<std::vector<double> > m3x3_dense = {{12, 1, 3},
                                                    {0,  0, 0},
                                                    {7,  0, 9}};
    std::vector<double> u3_dense = {1, 1, 0};
    std::vector<double> v3_ones = {1, 1, 1};

    Matrix<double, DirectedMatrixTag> mA(m3x3_dense, 0.);

    Vector<double, SparseTag> u3(u3_dense, 0.);
    Vector<double, SparseTag> m3(u3_dense); // [1 1 0]
    BOOST_CHECK_EQUAL(m3.nvals(), 3);

    {
        Vector<double> result(v3_ones, 0.);

        std::vector<double> ans = {16, 1, 1};
        Vector<double> answer(ans, 0.);

        reduce(result,
               m3,
               Second<double>(),
               Plus<double>(), mA);

        BOOST_CHECK_EQUAL(result.nvals(), 3);
        BOOST_CHECK_EQUAL(result, answer);
    }

    {
        Vector<double> result(v3_ones, 0.);

        std::vector<double> ans = {16, 0, 1};
        Vector<double> answer(ans, 0.);

        reduce(result,
               m3,
               NoAccumulate(),
               Plus<double>(), mA);

        BOOST_CHECK_EQUAL(result.nvals(), 2);
        BOOST_CHECK_EQUAL(result, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_reduce_matvec_mask_transpose)
{
    std::vector<std::vector<double> > m3x3_dense = {{12, 1, 3},
                                                    {0,  0, 0},
                                                    {7,  0, 9}};
    std::vector<double> u3_dense = {1, 1, 0};
    std::vector<double> v3_ones = {1, 1, 1};

    Matrix<double, DirectedMatrixTag> mA(m3x3_dense, 0.);
    Vector<double, SparseTag> m3(u3_dense, 0.); // [1 1 -]
    Vector<double> result(v3_ones, 0.);

    BOOST_CHECK_EQUAL(m3.nvals(), 2);

    std::vector<double> ans = {19, 1, 1};
    Vector<double> answer(ans, 0.);

    reduce(result,
           m3,
           NoAccumulate(),
           Plus<double>(), transpose(mA));

    BOOST_CHECK_EQUAL(result.nvals(), 3);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(reduce_matvec_test_nomask_noaccum)
{
    // Build some matrices.
    IndexArrayType i_A    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j_A    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_A = {1, 1, 1, 2, 2, 1, 3, 5, 3, 4};
    Matrix<double, DirectedMatrixTag> A(4, 4);
    A.build(i_A, j_A, v_A);

    IndexArrayType j_C    = {0, 1, 2, 3};
    std::vector<double> v_C = {1, 1, 1, 1};
    Vector<double> C(4);
    C.build(j_C, v_C);

    IndexArrayType j_answer    = {0, 1, 2, 3};
    std::vector<double> v_answer = {2, 4, 8, 9};
    Vector<double> answer(4);
    answer.build(j_answer, v_answer);

    //col_reduce(A, C);
    reduce(C, NoMask(), NoAccumulate(),
           PlusMonoid<double>(), transpose(A));

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(reduce_matvec_test_transpose_nomask_noaccum)
{
    // Build some matrices.
    IndexArrayType i_A    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j_A    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_A = {1, 1, 1, 2, 2, 1, 3, 5, 3, 4};
    Matrix<double, DirectedMatrixTag> A(4, 4);
    A.build(i_A, j_A, v_A);

    IndexArrayType j_C    = {0, 1, 2, 3};
    std::vector<double> v_C = {1, 1, 1, 1};
    Vector<double> C(4);
    C.build(j_C, v_C);

    IndexArrayType j_answer    = {0, 1, 2, 3};
    std::vector<double> v_answer = {2, 4, 8, 9};
    Vector<double> answer(4);
    answer.build(j_answer, v_answer);

    //col_reduce(A, C);
    reduce(C, NoMask(), NoAccumulate(),
           PlusMonoid<double>(), transpose(A));

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(reduce_matvec_test_transpose_nomask_accum)
{
    // Build some matrices.
    IndexArrayType i_A    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j_A    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_A = {1, 1, 1, 2, 2, 1, 3, 5, 3, 4};
    Matrix<double, DirectedMatrixTag> A(4, 4);
    A.build(i_A, j_A, v_A);

    IndexArrayType j_C    = {0, 1, 2, 3};
    std::vector<double> v_C = {1, 1, 1, 1};
    Vector<double> C(4);
    C.build(j_C, v_C);

    IndexArrayType j_answer    = {0, 1, 2, 3};
    std::vector<double> v_answer = {3, 5, 9, 10};
    Vector<double> answer(4);
    answer.build(j_answer, v_answer);

    reduce(C, NoMask(), Plus<double>(),
           PlusMonoid<double>(), transpose(A));


    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(reduce_matvec_test_transpose_masked_accum)
{
    // Build some matrices.
    IndexArrayType i_A    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j_A    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_A = {1, 1, 1, 2, 2, 1, 3, 5, 3, 4};
    Matrix<double, DirectedMatrixTag> A(4, 4);
    A.build(i_A, j_A, v_A);

    IndexArrayType j_C    = {0, 1, 2, 3};
    std::vector<double> v_C = {1, 1, 1, 1};
    Vector<double> C(4);
    C.build(j_C, v_C);

    IndexArrayType j_mask    = {0, 1, 2, 3};
    std::vector<double> v_mask = {1, 0, 1, 0};
    Vector<bool> mask(4);
    mask.build(j_mask, v_mask);

    IndexArrayType j_answer    = {0, 1, 2, 3};
    std::vector<double> v_answer = {2, 1, 8, 1};
    Vector<double> answer(4);
    answer.build(j_answer, v_answer);

    //colReduceMasked(A, C, mask);
    reduce(C, mask, NoAccumulate(),
           PlusMonoid<double>(), transpose(A));

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************

BOOST_AUTO_TEST_CASE(reduce_matvec_test_masked_noaccum)
{
    // Build some matrices.
    IndexArrayType i_A    = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j_A    = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v_A = {1, 1, 1, 2, 2, 1, 3, 5, 3, 4};
    Matrix<double, DirectedMatrixTag> A(4, 4);
    A.build(i_A, j_A, v_A);

    IndexArrayType i_C    = {0, 1, 2, 3};
    std::vector<double> v_C = {1, 1, 1, 1};
    Vector<double> C(4);
    C.build(i_C, v_C);

    IndexArrayType i_mask    = {0, 1, 2, 3};
    std::vector<double> v_mask = {1, 0, 1, 0};
    Vector<bool> mask(4);
    mask.build(i_mask, v_mask);

    IndexArrayType i_answer    = {0, 1, 2, 3};
    std::vector<double> v_answer = {2, 1, 9, 1};
    Vector<double> answer(4);
    answer.build(i_answer, v_answer);

    reduce(C, mask, NoAccumulate(),
           PlusMonoid<double>(), A);

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_reduce_matvec_transpose_scmp_noaccum)
{
    std::vector<std::vector<double> > m3x3_dense = {{12, 1, 3},
                                                    {0,  0, 0},
                                                    {7,  0, 9}};
    std::vector<double> u3_dense = {1, 1, 0};
    std::vector<double> v3_ones = {1, 1, 1};

    Matrix<double, DirectedMatrixTag> mA(m3x3_dense, 0.);

    Vector<double, SparseTag> u3(u3_dense, 0.); // [1 1 -]
    Vector<double, SparseTag> m3(u3_dense, 0.); // [1 1 -]
    Vector<double> result(v3_ones, 0.);

    BOOST_CHECK_EQUAL(m3.nvals(), 2);

    std::vector<double> ans = {1, 1, 12};
    Vector<double> answer(ans, 0.);

    reduce(result,
           complement(m3),
           NoAccumulate(),
           Plus<double>(), transpose(mA));

    BOOST_CHECK_EQUAL(result.nvals(), 3);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_reduce_matvec_nomask_noaccum_stored_zero_result)
{
    // Build some matrices.
    std::vector<std::vector<int> > A = {{1, -1, 0, 0},
                                        {-1, -2, 2, 0},
                                        {0, -2, -3, 3},
                                        {0, 0, 0, 0}};
    Matrix<int, DirectedMatrixTag> mA(A, 0);
    Vector<int> result(4);

    // use a different sentinel value so that stored zeros are preserved.
    int const NIL(666);
    std::vector<int> ans = {  0,  -1, -2, NIL};
    Vector<int> answer(ans, NIL);

    reduce(result,
           NoMask(),
           NoAccumulate(), // Second<int>(),
           Plus<int>(), mA);
    BOOST_CHECK_EQUAL(result, answer);
    BOOST_CHECK_EQUAL(result.nvals(), 3);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_reduce_matvec_transpose_nomask_noaccum)
{
    std::vector<std::vector<double> > m3x3_dense = {{12, 1, 3},
                                                    {0,  0, 0},
                                                    {7,  0, 9}};

    std::vector<std::vector<double> > m3x4_dense = {{5, 0, 1, 0},
                                                    {6, 7, 0, 0},
                                                    {6, 7, 0, 0}};

    std::vector<double> u3_dense = {1, 1, 0};
    std::vector<double> u4_dense = {0, 0, 13, 1};

    Matrix<double, DirectedMatrixTag> mA(m3x3_dense, 0.);
    Matrix<double, DirectedMatrixTag> mB(m3x4_dense, 0.);
    Vector<double> u3(u3_dense, 0.);
    Vector<double> u4(u4_dense, 0.);
    Vector<double> result3(3);
    Vector<double> result4(4);

    std::vector<double> ansAT_dense = {19, 1, 12};
    std::vector<double> ansBT_dense = {17, 14, 1, 0};

    Vector<double> ansA(ansAT_dense, 0.);
    Vector<double> ansB(ansBT_dense, 0.);

    reduce(result3,
           NoMask(),
           NoAccumulate(),
           Plus<double>(),
           transpose(mA));
    BOOST_CHECK_EQUAL(result3, ansA);

    reduce(result4,
           NoMask(),
           NoAccumulate(),
           Plus<double>(),
           transpose(mB));
    BOOST_CHECK_EQUAL(result4, ansB);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_reduce_matvec_scmp_replace)
{
    std::vector<std::vector<double> > m3x3_dense = {{12, 1, 3},
                                                    {0,  0, 0},
                                                    {7,  0, 9}};
    std::vector<double> v3_ones = {1, 1, 1};

    Matrix<double, DirectedMatrixTag> mA(m3x3_dense, 0.);
    std::vector<double> scmp_mask_dense = {0., 0., 1.};
    Vector<double, SparseTag> m3(scmp_mask_dense, 0.);
    BOOST_CHECK_EQUAL(m3.nvals(), 1);

    {
        Vector<double> result(v3_ones, 0.);

        std::vector<double> ans = {16, 1, 0};
        Vector<double> answer(ans, 0.);

        reduce(result,
               complement(m3),
               Second<double>(),
               Plus<double>(), mA,
               REPLACE);

        BOOST_CHECK_EQUAL(result.nvals(), 2);
        BOOST_CHECK_EQUAL(result, answer);
    }

    {
        Vector<double> result(v3_ones, 0.);

        std::vector<double> ans = {16, 0, 0};
        Vector<double> answer(ans, 0.);

        reduce(result,
               complement(m3),
               NoAccumulate(),
               Plus<double>(), mA,
               REPLACE);

        BOOST_CHECK_EQUAL(result.nvals(), 1);
        BOOST_CHECK_EQUAL(result, answer);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_reduce_matvec_scmp_replace_stored_zero)
{
    // tests a computed and stored zero
    // tests a stored zero in the mask
    // Build some matrices.
    std::vector<std::vector<int> > A_dense = {{1,-1, 0, 0},
                                              {1, 2, 2, 0},
                                              {0, 2, 3, 3},
                                              {0, 0, 3, 4}};
    std::vector<int> scmp_mask_dense =  { 0, 0, 0, 1 };
    std::vector<int> ones_dense =  { 1, 1, 1, 1};
    Matrix<int, DirectedMatrixTag> A(A_dense, 0);
    Vector<int> mask(scmp_mask_dense); //, 0);
    Vector<int> result(ones_dense, 0);

    BOOST_CHECK_EQUAL(mask.nvals(), 4);  // [ 0 0 0 1 ]

    int const NIL(666);
    std::vector<int> ans = { 0, 5, 8,  NIL};
    Vector<int> answer(ans, NIL);

    reduce(result,
           complement(mask),
           NoAccumulate(),
           Plus<double>(), A,
           REPLACE);
    BOOST_CHECK_EQUAL(result.nvals(), 3);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_reduce_matvec_scmp_replace_transpose)
{
    // Build some matrices.
    std::vector<std::vector<int> > A_dense = {{1, 1, 0, 0, 1},
                                              {-1, 2, 2, 0, 1},
                                              {0, 2, 3, 3, 1},
                                              {0, 0, 3, 4, 1}};
    std::vector<int> mask_dense =  { 0, 1, 0, 1, 0};
    std::vector<int> ones_dense =  { 1, 1, 1, 1, 1};
    Matrix<int, DirectedMatrixTag> A(A_dense, 0);
    Vector<int> mask(mask_dense, 0);
    Vector<int> result(ones_dense, 0);

    int const NIL(666);
    std::vector<int> ans = { 0, NIL, 8,  NIL, 4};
    Vector<int> answer(ans, NIL);

    reduce(result,
           complement(mask),
           NoAccumulate(),
           Plus<double>(), transpose(A),
           REPLACE);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_reduce_matvec_scmp_merge)
{
    std::vector<std::vector<double> > m3x3_dense = {{12, 1, 3},
                                                    {0,  0, 0},
                                                    {7,  0, 9}};
    std::vector<double> u3_dense = {1, 1, 0};
    std::vector<double> v3_ones = {1, 1, 1};

    Matrix<double, DirectedMatrixTag> mA(m3x3_dense, 0.);
    Vector<double, SparseTag> m3(u3_dense, 0.); // [1 1 -]
    Vector<double> result(v3_ones, 0.);

    BOOST_CHECK_EQUAL(m3.nvals(), 2);

    std::vector<double> ans = {1, 1, 16}; // if accum is NULL then {1, 0, 7};
    Vector<double> answer(ans, 0.);

    reduce(result,
           complement(m3),
           NoAccumulate(),
           Plus<double>(), mA);

    BOOST_CHECK_EQUAL(result.nvals(), 3); //2);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_reduce_matvec_scmp_merge_stored_zero)
{
    std::vector<std::vector<double> > m3x3_dense = {{12, 1, 3},
                                                    {0,  0, 0},
                                                    {7,  0, 9}};
    std::vector<double> u3_dense = {1, 1, 0};
    std::vector<double> v3_ones = {1, 1, 1};

    Matrix<double, DirectedMatrixTag> mA(m3x3_dense, 0.);
    Vector<double, SparseTag> m3(u3_dense); // [1 1 0]
    Vector<double> result(v3_ones, 0.);

    BOOST_CHECK_EQUAL(m3.nvals(), 3);

    std::vector<double> ans = {1, 1, 16};
    Vector<double> answer(ans, 0.);

    reduce(result,
           complement(m3),
           NoAccumulate(),
           Plus<double>(), mA);

    BOOST_CHECK_EQUAL(result.nvals(), 3); //2);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
// reduce vector to scalar
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_reduce_vecscalar_reg)
{
    std::vector<double> ans3x3_dense  = {16, 0, 16};

    Vector<double> empty_result6(6);
    Vector<double> ansA(ans3x3_dense, 0.);

    // Vector to scalar reduction
    double val = 22;
    reduce(val,
           NoAccumulate(),
           PlusMonoid<double>(), empty_result6);
    BOOST_CHECK_EQUAL(val, PlusMonoid<double>().identity());

    val = 22;
    reduce(val,
           NoAccumulate(),
           PlusMonoid<double>(), ansA);
    BOOST_CHECK_EQUAL(val, 32.0);

    val = 22;
    reduce(val,
           NoAccumulate(),
           add_monoid(ArithmeticSemiring<double>()), ansA);
    BOOST_CHECK_EQUAL(val, 32.0);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_reduce_vecscalar_noaccum)
{
    // | - 2 3 3 |

    // Build some sparse vectors.
    std::vector<double> v = {0, 2, 3, 3};
    Vector<double> u(v, 0.0);

    {
        double val = 99.;
        reduce(val,
               NoAccumulate(),
               PlusMonoid<double>(),
               u);
        BOOST_CHECK_EQUAL(val, 8.0);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_reduce_vecscalar_noaccum_empty)
{
    // | - 2 3 3 |

    // Build some sparse vectors.
    Vector<double> u(4);

    {
        double val = 99.;
        reduce(val,
               NoAccumulate(),
               PlusMonoid<double>(),
               u);
        BOOST_CHECK_EQUAL(val, 0.0);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_reduce_vecscalar_accum)
{
    // | - 2 3 3 |

    // Build some sparse vectors.
    std::vector<double> v = {0, 2, 3, 3};
    Vector<double> u(v, 0.0);

    {
        double val = 99.;
        reduce(val,
               Plus<double>(),
               PlusMonoid<double>(),
               u);
        BOOST_CHECK_EQUAL(val, 107.0);
    }

    {
        double val = 99.;
        reduce(val,
               Second<double>(),
               PlusMonoid<double>(),
               u);
        BOOST_CHECK_EQUAL(val, 8.0);
    }
}

//****************************************************************************
// reduce matrix to scalar
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_reduce_matscalar_noaccum)
{
    std::vector<std::vector<double> > m6x4_dense = {{5, 0, 1, 0},
                                                    {6, 7, 0, 0},
                                                    {6, 7, 0, 0},
                                                    {0, 5, 0, 0},
                                                    {6, 7, 0, 0},
                                                    {0, 0, 0, 0}};

    Matrix<double, DirectedMatrixTag> mB(m6x4_dense, 0.);

    // Matrix to scalar reduction
    double val = 22;
    reduce(val,
           NoAccumulate(),
           PlusMonoid<double>(), mB);
    BOOST_CHECK_EQUAL(val, 50.0);

    val = 22;
    reduce(val,
           NoAccumulate(),
           add_monoid(ArithmeticSemiring<double>()), mB);
    BOOST_CHECK_EQUAL(val, 50.0);
}

//****************************************************************************
//
// Transpose not supported
//
// BOOST_AUTO_TEST_CASE(test_reduce_matscalar_noaccum_transpose)
// {
//     std::vector<std::vector<double> > m6x4_dense = {{5, 0, 1, 0},
//                                                     {6, 7, 0, 0},
//                                                     {6, 7, 0, 0},
//                                                     {0, 5, 0, 0},
//                                                     {6, 7, 0, 0},
//                                                     {0, 0, 0, 0}};
//
//     Matrix<double, DirectedMatrixTag> mB(m6x4_dense, 0.);
//     // Matrix to scalar reduction
//     double val = 22;
//     reduce(val,
//            NoAccumulate(),
//            PlusMonoid<double>(), transpose(mB));
//     BOOST_CHECK_EQUAL(val, 50.0);
// }

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_reduce_matscalar_noaccum_empty)
{
    Matrix<double, DirectedMatrixTag> empty6x4(6, 4);

    // Matrix to scalar reduction
    double val = 22;
    reduce(val,
           NoAccumulate(),
           PlusMonoid<double>(), empty6x4);
    BOOST_CHECK_EQUAL(val, PlusMonoid<double>().identity());
}



//****************************************************************************
BOOST_AUTO_TEST_CASE(test_reduce_matscalar_accum)
{
    std::vector<std::vector<double> > m6x4_dense = {{5, 0, 1, 0},
                                                    {6, 7, 0, 0},
                                                    {6, 7, 0, 0},
                                                    {0, 5, 0, 0},
                                                    {6, 7, 0, 0},
                                                    {0, 0, 0, 0}};

    Matrix<double, DirectedMatrixTag> mB(m6x4_dense, 0.);

    // Matrix to scalar reduction
    double val = 22;
    reduce(val,
           Plus<double>(),
           PlusMonoid<double>(), mB);
    BOOST_CHECK_EQUAL(val, 72.0);
}

//****************************************************************************
//
// Transpose not supported
//
// BOOST_AUTO_TEST_CASE(test_reduce_matscalar_accum_transpose)
// {
//     std::vector<std::vector<double> > m6x4_dense = {{5, 0, 1, 0},
//                                                     {6, 7, 0, 0},
//                                                     {6, 7, 0, 0},
//                                                     {0, 5, 0, 0},
//                                                     {6, 7, 0, 0},
//                                                     {0, 0, 0, 0}};
//     Matrix<double, DirectedMatrixTag> mB(m6x4_dense, 0.);
//     // Matrix to scalar reduction
//     double val = 22;
//     reduce(val,
//            NoAccumulate(),
//            PlusMonoid<double>(), transpose(mB));
//     BOOST_CHECK_EQUAL(val, 77.0);
// }

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_reduce_matscalar_accum_empty)
{
    Matrix<double, DirectedMatrixTag> empty6x4(6, 4);

    // Matrix to scalar reduction
    double val = 22;
    reduce(val,
           Plus<double>(),
           PlusMonoid<double>(), empty6x4);
    BOOST_CHECK_EQUAL(val, 22.0);
}

BOOST_AUTO_TEST_SUITE_END()
