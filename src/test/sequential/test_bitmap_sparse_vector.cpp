/*
 * Copyright (c) 2017 Carnegie Mellon University.
 * All Rights Reserved.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS," WITH NO WARRANTIES WHATSOEVER. CARNEGIE
 * MELLON UNIVERSITY EXPRESSLY DISCLAIMS
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
#define BOOST_TEST_MODULE bitmap_sparse_vector_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(bitmap_sparse_vector_suite)

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_construction_basic)
{
    GraphBLAS::IndexType M = 7;
    GraphBLAS::backend::BitmapSparseVector<double> v1(M);

    BOOST_CHECK_EQUAL(v1.size(), M);
    BOOST_CHECK_EQUAL(v1.nvals(), 0);
    BOOST_CHECK_THROW(v1.extractElement(0), NoValueException);
    BOOST_CHECK_THROW(v1.extractElement(M-1), NoValueException);
    BOOST_CHECK_THROW(v1.extractElement(M), IndexOutOfBoundsException);

    BOOST_CHECK_THROW(GraphBLAS::backend::BitmapSparseVector<double>(0),
                      InvalidValueException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_construction_from_dense)
{
    std::vector<double> vec = {6, 0, 0, 4, 7, 0, 9, 4};

    GraphBLAS::backend::BitmapSparseVector<double> v1(vec);

    BOOST_CHECK_EQUAL(v1.size(), vec.size());
    BOOST_CHECK_EQUAL(v1.nvals(), vec.size());
    for (GraphBLAS::IndexType i = 0; i < vec.size(); ++i)
    {
        BOOST_CHECK_EQUAL(v1.extractElement(i), vec[i]);
    }
    BOOST_CHECK_THROW(v1.extractElement(v1.size()), IndexOutOfBoundsException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_sparse_construction_from_dense)
{
    double zero(0);
    std::vector<double> vec = {6, zero, zero, 4, 7, zero, 9, 4};

    GraphBLAS::backend::BitmapSparseVector<double> v1(vec, zero);

    BOOST_CHECK_EQUAL(v1.size(), vec.size());
    BOOST_CHECK_EQUAL(v1.nvals(), 5);
    for (GraphBLAS::IndexType i = 0; i < vec.size(); ++i)
    {
        if (vec[i] == zero)
        {
            BOOST_CHECK_THROW(v1.extractElement(i), NoValueException);
        }
        else
        {
            BOOST_CHECK_EQUAL(v1.extractElement(i), vec[i]);
        }
    }
    BOOST_CHECK_THROW(v1.extractElement(v1.size()), IndexOutOfBoundsException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_copy_construction_equality_operator)
{
    double zero(0);
    std::vector<double> vec = {6, zero, zero, 4, 7, zero, 9, 4};

    GraphBLAS::backend::BitmapSparseVector<double> v1(vec, zero);

    GraphBLAS::backend::BitmapSparseVector<double> v2(v1);

    BOOST_CHECK_EQUAL(v1, v2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_access_novalue_in_non_empty_vector)
{
    std::vector<double> vec = {6, 0, 0, 4, 7, 0, 9, 4};

    GraphBLAS::backend::BitmapSparseVector<double> v1(vec, 0);

    BOOST_CHECK_EQUAL(v1.nvals(), 5);
    BOOST_CHECK_THROW(v1.extractElement(1), NoValueException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_assign_to_implied_zero_and_stored_value)
{
    double zero(0);
    std::vector<double> vec = {6, zero, zero, 4, 7, zero, 9, 4};

    GraphBLAS::backend::BitmapSparseVector<double> v1(vec, zero);

    v1.setElement(1, 2.);

    GraphBLAS::backend::BitmapSparseVector<double> v2(vec, zero);
    BOOST_CHECK(v1 != v2);

    v2.setElement(1, 3.);
    BOOST_CHECK(v1 != v2);

    v2.setElement(1, 2.);
    BOOST_CHECK_EQUAL(v1, v2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_index_array_constrution)
{
    double zero(0);
    std::vector<double> vec = {6, zero, zero, 4, 7, zero, 9, 4};

    GraphBLAS::backend::BitmapSparseVector<double> v1(vec, zero);

    std::vector<IndexType> indices = {0, 3, 4, 6, 7};
    std::vector<double>    values  = {6 ,4, 7, 9, 4};

    GraphBLAS::backend::BitmapSparseVector<double> v2(8, indices, values);
    BOOST_CHECK_EQUAL(v1, v2);

    GraphBLAS::backend::BitmapSparseVector<double> v3(9, indices, values);
    BOOST_CHECK(v1 != v3);

    BOOST_CHECK_THROW(
        GraphBLAS::backend::BitmapSparseVector<double>(7, indices, values),
        DimensionException);
    BOOST_CHECK_EQUAL(v1, v2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_assignment)
{
    std::vector<IndexType> indices = {0, 3, 4, 6, 7};
    std::vector<double>    values  = {6 ,4, 7, 9, 4};

    GraphBLAS::backend::BitmapSparseVector<double> v1(8, indices, values);
    GraphBLAS::backend::BitmapSparseVector<double> v2(8);

    v2 = v1;
    BOOST_CHECK_EQUAL(v1, v2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_sparse_nomask_noaccum)
{
    std::vector<std::vector<double>> mat = {{0, 0, 0, 0},
                                            {0, 0, 0, 4},
                                            {0, 0, 9, 0},
                                            {0, 0, 5, 3},
                                            {0, 2, 0, 0},
                                            {0, 3, 0, 1},
                                            {0, 3, 3, 0},
                                            {0, 1, 4, 2},
                                            {6, 0, 0, 0},
                                            {6, 0, 0, 2},
                                            {6, 0, 4, 0},
                                            {6, 0, 4, 2},
                                            {6, 1, 0, 0},
                                            {6, 1, 0, 2},
                                            {6, 1, 4, 0},
                                            {6, 1, 4, 2}};
    GraphBLAS::backend::LilSparseMatrix<double> m1(mat, 0);

    std::vector<double> vec = {6, 0, 0, 4};
    GraphBLAS::backend::BitmapSparseVector<double> v1(vec, 0);

    GraphBLAS::backend::BitmapSparseVector<double> w(16);

    GraphBLAS::backend::mxv(w,
                            GraphBLAS::backend::NoMask(),
                            GraphBLAS::Second<double>(),
                            GraphBLAS::ArithmeticSemiring<double>(),
                            m1, v1, true);

    std::vector<double> answer = { 0, 16,  0, 12,  0,  4,  0,  8,
                                  36, 44, 36, 44, 36, 44, 36, 44};
    GraphBLAS::backend::BitmapSparseVector<double> ans(answer, 0);
    BOOST_CHECK_EQUAL(ans, w);
    w.printInfo(std::cerr);
}

BOOST_AUTO_TEST_SUITE_END()
