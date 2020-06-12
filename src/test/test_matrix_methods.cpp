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

#include <iostream>

#include <graphblas/graphblas.hpp>

using namespace grb;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE matrix_methods_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

//****************************************************************************
BOOST_AUTO_TEST_CASE(matrix_resize_test)
{
    IndexArrayType      i = {1, 2, 3, 4};
    std::vector<double> v = {1, 2, 3, 4};

    // matrix = diag{ -, 1, 2, 3, 4, -};

    IndexType const NSIZE = 6;
    Matrix<double> m1(NSIZE, NSIZE);
    m1.build(i, i, v);
    //print_matrix(std::cerr, m1, "m1(6,6)");

    BOOST_CHECK_EQUAL(m1.nvals(), i.size());
    BOOST_CHECK_EQUAL(m1.nrows(), NSIZE);
    BOOST_CHECK_EQUAL(m1.ncols(), NSIZE);

    // Make it bigger and set an element
    m1.resize(2*NSIZE, 2*NSIZE);
    //print_matrix(std::cerr, m1, "m1(12,12)");
    BOOST_CHECK_EQUAL(m1.nvals(), i.size());
    BOOST_CHECK_EQUAL(m1.nrows(), 2*NSIZE);
    BOOST_CHECK_EQUAL(m1.ncols(), 2*NSIZE);

    m1.setElement(NSIZE + 1, NSIZE + 1, 99);

    BOOST_CHECK_EQUAL(m1.nvals(), i.size() + 1);
    BOOST_CHECK_EQUAL(m1.extractElement(NSIZE + 1, NSIZE + 1), 99);

    // Reduce num columns check remaining elements
    m1.resize(2*NSIZE, 4UL);
    //print_matrix(std::cerr, m1, "m1(12, 4)");
    BOOST_CHECK_EQUAL(m1.nvals(), 3);
    BOOST_CHECK_EQUAL(m1.nrows(), 2*NSIZE);
    BOOST_CHECK_EQUAL(m1.ncols(), 4);
    BOOST_CHECK_EQUAL(m1.extractElement(1, 1), 1);
    BOOST_CHECK_EQUAL(m1.extractElement(2, 2), 2);
    BOOST_CHECK_EQUAL(m1.extractElement(3, 3), 3);

    // Reduce num rows check remaining elements
    m1.resize(3UL, 4UL);
    //print_matrix(std::cerr, m1, "m1(3, 4)");
    BOOST_CHECK_EQUAL(m1.nvals(), 2);
    BOOST_CHECK_EQUAL(m1.nrows(), 3);
    BOOST_CHECK_EQUAL(m1.ncols(), 4);
    BOOST_CHECK_EQUAL(m1.extractElement(1, 1), 1);
    BOOST_CHECK_EQUAL(m1.extractElement(2, 2), 2);

    // reduce both and check remaining elements
    m1.resize(2UL, 2UL);
    //print_matrix(std::cerr, m1, "m1(2,2)");
    BOOST_CHECK_EQUAL(m1.nvals(), 1);
    BOOST_CHECK_EQUAL(m1.nrows(), 2);
    BOOST_CHECK_EQUAL(m1.ncols(), 2);
    BOOST_CHECK_EQUAL(m1.extractElement(1, 1), 1);

    // Make it bigger show that elements don't reappear
    m1.resize(2*NSIZE, 2*NSIZE);
    //print_matrix(std::cerr, m1, "m1(12,12)");
    BOOST_CHECK_EQUAL(m1.nvals(), 1);
    BOOST_CHECK_EQUAL(m1.nrows(), 2*NSIZE);
    BOOST_CHECK_EQUAL(m1.ncols(), 2*NSIZE);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(matrix_removeElement_test)
{
    IndexArrayType      i = {0, 0, 0, 1, 1, 1, 2, 2};
    IndexArrayType      j = {1, 2, 3, 0, 2, 3, 0, 1};
    std::vector<double> v = {1, 2, 3, 4, 6, 7, 8, 9};

    //std::vector<std::vector<double> > mat = {{0, 1, 2, 3},
    //                                         {4, 0, 6, 7},
    //                                         {8, 9, 0, 0}};

    Matrix<double, DirectedMatrixTag> m1(3, 4);
    m1.build(i, j, v);

    BOOST_CHECK_EQUAL(m1.nvals(), i.size());

    // remove something that does not exist
    BOOST_CHECK(!m1.hasElement(1, 1));
    m1.removeElement(1, 1);
    BOOST_CHECK_EQUAL(m1.nvals(), i.size());
    BOOST_CHECK(!m1.hasElement(1, 1));

    // remove something that exists
    BOOST_CHECK(m1.hasElement(1, 2));
    m1.removeElement(1, 2);
    BOOST_CHECK_EQUAL(m1.nvals(), i.size() - 1);
    BOOST_CHECK(!m1.hasElement(1, 2));
}

BOOST_AUTO_TEST_SUITE_END()
