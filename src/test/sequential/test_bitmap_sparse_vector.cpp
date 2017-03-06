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
    GraphBLAS::BitmapSparseVector<double> v1(M);

    BOOST_CHECK_EQUAL(v1.get_size(), M);
    BOOST_CHECK_EQUAL(v1.get_nvals(), 0);
    BOOST_CHECK_THROW(v1.get_value_at(0), NoValueException);
    BOOST_CHECK_THROW(v1.get_value_at(M-1), NoValueException);
    BOOST_CHECK_THROW(v1.get_value_at(M), IndexOutOfBoundsException);

    BOOST_CHECK_THROW(GraphBLAS::BitmapSparseVector<double>(0),
                      InvalidValueException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_construction_from_dense)
{
    std::vector<double> vec = {6, 0, 0, 4, 7, 0, 9, 4};

    GraphBLAS::BitmapSparseVector<double> v1(vec);

    BOOST_CHECK_EQUAL(v1.get_size(), vec.size());
    BOOST_CHECK_EQUAL(v1.get_nvals(), vec.size());
    for (GraphBLAS::IndexType i = 0; i < vec.size(); ++i)
    {
        BOOST_CHECK_EQUAL(v1.get_value_at(i), vec[i]);
    }
    BOOST_CHECK_THROW(v1.get_value_at(v1.get_size()), IndexOutOfBoundsException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_sparse_construction_from_dense)
{
    double zero(0);
    std::vector<double> vec = {6, zero, zero, 4, 7, zero, 9, 4};

    GraphBLAS::BitmapSparseVector<double> v1(vec, zero);

    BOOST_CHECK_EQUAL(v1.get_size(), vec.size());
    BOOST_CHECK_EQUAL(v1.get_nvals(), 5);
    for (GraphBLAS::IndexType i = 0; i < vec.size(); ++i)
    {
        if (vec[i] == zero)
        {
            BOOST_CHECK_THROW(v1.get_value_at(i), NoValueException);
        }
        else
        {
            BOOST_CHECK_EQUAL(v1.get_value_at(i), vec[i]);
        }
    }
    BOOST_CHECK_THROW(v1.get_value_at(v1.get_size()), IndexOutOfBoundsException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_copy_construction_equality_operator)
{
    std::vector<double> vec = {6, 0, 0, 4, 7, 0, 9, 4};

    GraphBLAS::BitmapSparseVector<double> v1(vec, 0);

    GraphBLAS::BitmapSparseVector<double> v2(v1);

    BOOST_CHECK_EQUAL(v1, v2);
}

#if 0

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

    GraphBLAS::BitmapSparseVector<double> m1(mat, 0);

    mat[0][1] = 8;
    m1.set_value_at(0, 1, 8);
    BOOST_CHECK_EQUAL(m1.get_value_at(0, 1), mat[0][1]);

    GraphBLAS::BitmapSparseVector<double> m2(mat, 0);
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

    GraphBLAS::BitmapSparseVector<double> m1(mat, 0);

    mat[0][0] = 8;
    m1.set_value_at(0, 0, 8);
    BOOST_CHECK_EQUAL(m1.get_value_at(0, 0), mat[0][0]);

    GraphBLAS::BitmapSparseVector<double> m2(mat, 0);
    BOOST_CHECK_EQUAL(m1, m2);
}
#endif
BOOST_AUTO_TEST_SUITE_END()
