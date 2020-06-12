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
#define BOOST_TEST_MODULE vector_methods_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

//****************************************************************************
BOOST_AUTO_TEST_CASE(vector_resize_test)
{
    IndexArrayType      i = {1, 2, 3};
    std::vector<double> v = {1, 2, 3};

    //std::vector<double> vec = { -, 1, 2, 3, -, -};

    IndexType const NSIZE = 6;
    Vector<double> v1(NSIZE);
    v1.build(i, v);

    BOOST_CHECK_EQUAL(v1.nvals(), i.size());
    BOOST_CHECK_EQUAL(v1.size(), NSIZE);

    // Make it bigger and set an element
    v1.resize(2*NSIZE);
    BOOST_CHECK_EQUAL(v1.nvals(), i.size());
    BOOST_CHECK_EQUAL(v1.size(), 2*NSIZE);

    v1.setElement(NSIZE + 1, 99);

    BOOST_CHECK_EQUAL(v1.extractElement(NSIZE + 1), 99);

    // Make it smaller and check remaining elements
    v1.resize(3UL);
    BOOST_CHECK_EQUAL(v1.size(), 3);
    BOOST_CHECK_EQUAL(v1.nvals(), 2);
    BOOST_CHECK_EQUAL(v1.extractElement(1), 1);
    BOOST_CHECK_EQUAL(v1.extractElement(2), 2);

    // Make it bigger show that elements don't reappear
    v1.resize(2*NSIZE);
    BOOST_CHECK_EQUAL(v1.nvals(), 2);
    BOOST_CHECK_EQUAL(v1.size(), 2*NSIZE);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(vector_removeElement_test)
{
    IndexArrayType      i = {1, 2, 3};
    std::vector<double> v = {1, 2, 3};

    //std::vector<double> vec = { -, 1, 2, 3, -, -};

    Vector<double> v1(6);
    v1.build(i, v);

    BOOST_CHECK_EQUAL(v1.nvals(), i.size());

    // remove something that does not exist
    BOOST_CHECK(!v1.hasElement(4));
    v1.removeElement(4);
    BOOST_CHECK_EQUAL(v1.nvals(), i.size());
    BOOST_CHECK(!v1.hasElement(4));

    // remove something that exists
    BOOST_CHECK(v1.hasElement(1));
    v1.removeElement(1);
    BOOST_CHECK_EQUAL(v1.nvals(), i.size() - 1);
    BOOST_CHECK(!v1.hasElement(1));
}

BOOST_AUTO_TEST_SUITE_END()
