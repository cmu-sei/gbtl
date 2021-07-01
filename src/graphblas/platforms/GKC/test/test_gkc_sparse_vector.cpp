/*
 * GraphBLAS Template Library, Version 2.1
 *
 * Copyright 2019 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors. All Rights Reserved.
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
 * RIGHTS..
 *
 * Released under a BSD (SEI)-style license, please see license.txt or contact
 * permission@sei.cmu.edu for full terms.
 *
 * This release is an update of:
 *
 * 1. GraphBLAS Template Library (GBTL)
 * (https://github.com/cmu-sei/gbtl/blob/1.0.0/LICENSE) Copyright 2015 Carnegie
 * Mellon University and The Trustees of Indiana. DM17-0037, DM-0002659
 *
 * DM18-0559
 */

#include <iostream>

#include <graphblas/graphblas.hpp>

using namespace grb;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE gkc_sparse_vector_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_construction_basic)
{
    grb::IndexType M = 7;
    grb::backend::GKCSparseVector<double> v1(M);

    BOOST_CHECK_EQUAL(v1.size(), M);
    BOOST_CHECK_EQUAL(v1.nvals(), 0);
    BOOST_CHECK_THROW(v1.extractElement(0), NoValueException);
    BOOST_CHECK_THROW(v1.extractElement(M-1), NoValueException);
    BOOST_CHECK_THROW(v1.extractElement(M), IndexOutOfBoundsException);

    BOOST_CHECK_THROW(grb::backend::GKCSparseVector<double>(0),
                      InvalidValueException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_construction_from_dense)
{
    std::vector<double> vec = {6, 0, 0, 4, 7, 0, 9, 4};

    grb::backend::GKCSparseVector<double> v1(vec);

    BOOST_CHECK_EQUAL(v1.size(), vec.size());
    BOOST_CHECK_EQUAL(v1.nvals(), vec.size());
    for (grb::IndexType i = 0; i < vec.size(); ++i)
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

    grb::backend::GKCSparseVector<double> v1(vec, zero);

    BOOST_CHECK_EQUAL(v1.size(), vec.size());
    BOOST_CHECK_EQUAL(v1.nvals(), 5);
    for (grb::IndexType i = 0; i < vec.size(); ++i)
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

    grb::backend::GKCSparseVector<double> v1(vec, zero);

    grb::backend::GKCSparseVector<double> v2(v1);

    BOOST_CHECK_EQUAL(v1, v2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_access_novalue_in_non_empty_vector)
{
    std::vector<double> vec = {6, 0, 0, 4, 7, 0, 9, 4};

    grb::backend::GKCSparseVector<double> v1(vec, 0);

    BOOST_CHECK_EQUAL(v1.nvals(), 5);
    BOOST_CHECK_THROW(v1.extractElement(1), NoValueException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_assign_to_implied_zero_and_stored_value)
{
    double zero(0);
    std::vector<double> vec = {6, zero, zero, 4, 7, zero, 9, 4};

    grb::backend::GKCSparseVector<double> v1(vec, zero);

    v1.setElement(1, 2.);

    grb::backend::GKCSparseVector<double> v2(vec, zero);
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

    grb::backend::GKCSparseVector<double> v1(vec, zero);

    std::vector<IndexType> indices = {0, 3, 4, 6, 7};
    std::vector<double>    values  = {6 ,4, 7, 9, 4};

    grb::backend::GKCSparseVector<double> v2(indices.begin(), 
    values.begin(), indices.size());

    BOOST_CHECK_EQUAL(v1, v2);

    //grb::backend::GKCSparseVector<double> v3(indices.begin(), values.begin(), 9);
    //BOOST_CHECK(v1 != v3);

    //BOOST_CHECK_THROW(
    //    grb::backend::GKCSparseVector<double>(indices.begin(), values.begin(), 3),
    //    DimensionException);
    //BOOST_CHECK_EQUAL(v1, v2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_assignment)
{
    std::vector<IndexType> indices = {0, 3, 4, 6, 7};
    std::vector<double>    values  = {6 ,4, 7, 9, 4};

    grb::backend::GKCSparseVector<double> v1(indices.begin(), values.begin(), indices.size());
    grb::backend::GKCSparseVector<double> v2(8);

    v2 = v1;
    BOOST_CHECK_EQUAL(v1, v2);
}

BOOST_AUTO_TEST_CASE(test_sorting)
{
    std::vector<IndexType> indices  = {0, 3, 4, 6, 7};
    std::vector<double>    values   = {6, 4, 7, 9, 4};
    std::vector<IndexType> indices2 = {3, 4, 0, 6, 7};
    std::vector<double>    values2  = {4, 7, 6, 9, 4};
    
    grb::backend::GKCSparseVector<double> v1(indices.begin(), values.begin(), indices.size());
    grb::backend::GKCSparseVector<double> v2(indices2.begin(), values2.begin(), indices2.size());

    // Current semantic meaning of neq is that vector with different order but same elements is not equal.
    BOOST_CHECK(v1 != v2);

    v2.sortSelf();

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
    grb::backend::GKCMatrix<double> m1(mat, 0);

    std::vector<double> vec = {6, 0, 0, 4};
    grb::backend::GKCSparseVector<double> v1(vec, 0);

    grb::backend::GKCSparseVector<double> w(16);

    grb::backend::mxv(w,
                      grb::NoMask(),
                      grb::Second<double>(),
                      grb::ArithmeticSemiring<double>(),
                      m1, v1, REPLACE);

    std::vector<double> answer = { 0, 16,  0, 12,  0,  4,  0,  8,
                                  36, 44, 36, 44, 36, 44, 36, 44};
    grb::backend::GKCSparseVector<double> ans(answer, 0);
    BOOST_CHECK_EQUAL(ans, w);
    w.printInfo(std::cerr);
    ans.printInfo(std::cerr);
}


BOOST_AUTO_TEST_SUITE_END()
