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

#include <functional>
#include <iostream>
#include <vector>

#include <graphblas/graphblas.hpp>

using namespace GraphBLAS;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE math_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(math_suite)

//****************************************************************************
BOOST_AUTO_TEST_CASE(power_test)
{
    BOOST_CHECK_EQUAL(Power<double>()(2, 5), 32.0);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(negate_test01)
{
    BOOST_CHECK_EQUAL(AdditiveInverse<double>()(-10), 10);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(negate_test02)
{
    BOOST_CHECK_EQUAL(AdditiveInverse<double>()(500), -500);
}

//****************************************************************************
//BOOST_AUTO_TEST_CASE(inverse_test01)
//{
//    BOOST_CHECK_EQUAL(math::inverse<double>(0), 0);
//}

//****************************************************************************
BOOST_AUTO_TEST_CASE(inverse_test02)
{
    BOOST_CHECK_EQUAL(MultiplicativeInverse<double>()(5), 1./5);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(plus_test)
{
    BOOST_CHECK_EQUAL(Plus<double>()(2, 6), 8);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sub_test)
{
    BOOST_CHECK_EQUAL(Minus<double>()(2, 6), -4);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(times_test)
{
    BOOST_CHECK_EQUAL(Times<double>()(3, 10), 30);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(div_test)
{
    BOOST_CHECK_EQUAL(Div<double>()(500, 5), 100);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(arithmetic_min_test01)
{
    BOOST_CHECK_EQUAL(Min<double>()(0, 1000000), 0);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(arithmetic_min_test02)
{
    BOOST_CHECK_EQUAL(Min<double>()(-5, 0), -5);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(arithmetic_min_test03)
{
    BOOST_CHECK_EQUAL(Min<double>()(7, 3), 3);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(select2nd_test04)
{
    BOOST_CHECK_EQUAL(Second<double>()(5, 1337), 1337);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(is_equal_test01)
{
    BOOST_CHECK_EQUAL(Equal<double>()(1, 1), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(is_equal_test02)
{
    BOOST_CHECK_EQUAL(Equal<double>()(0xC0FFEE, 0xCAFE), false);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(xor_fn_test)
{
    BOOST_CHECK_EQUAL(Xor<int>()(1, 2), 3);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(or_fn_test)
{
    BOOST_CHECK_EQUAL(LogicalOr<int>()(1, 0), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(and_fn_test)
{
    BOOST_CHECK_EQUAL(LogicalAnd<int>()(0, 2), false);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(not_fn_test)
{
    BOOST_CHECK_EQUAL(LogicalNot<int>()(1), false);
}

BOOST_AUTO_TEST_SUITE_END()
