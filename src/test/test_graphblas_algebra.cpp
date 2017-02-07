/*
 * Copyright (c) 2017 Carnegie Mellon University
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

#include <functional>
#include <iostream>
#include <vector>

#include <graphblas/GraphBLAS_algebra.hpp>

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE graphblas_algebra_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

//****************************************************************************
BOOST_AUTO_TEST_CASE(Identity_test)
{
    int8_t  i8  = 3;
    int16_t i16 = -400;
    int32_t i32 = -1001000;
    int64_t i64 = 5123123123;
    BOOST_CHECK_EQUAL(GraphBLAS::Identity<int8_t>()(i8), i8);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(negate_test01)
{
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(negate_test02)
{
}

BOOST_AUTO_TEST_SUITE_END()
