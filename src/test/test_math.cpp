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

using namespace graphblas;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE math_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(math_suite)

BOOST_AUTO_TEST_CASE(power_test)
{
    BOOST_CHECK_EQUAL(math::power<double>(2, 5), 32.0);
}

BOOST_AUTO_TEST_CASE(negate_test01)
{
    BOOST_CHECK_EQUAL(math::negate<double>(-10), 10);
}

BOOST_AUTO_TEST_CASE(negate_test02)
{
    BOOST_CHECK_EQUAL(math::negate<double>(500), -500);
}

BOOST_AUTO_TEST_CASE(inverse_test01)
{
    BOOST_CHECK_EQUAL(math::inverse<double>(0), 0);
}

BOOST_AUTO_TEST_CASE(inverse_test02)
{
    BOOST_CHECK_EQUAL(math::inverse<double>(5), 1./5);
}

BOOST_AUTO_TEST_CASE(plus_test)
{
    BOOST_CHECK_EQUAL(math::plus<double>(2, 6), 8);
}

BOOST_AUTO_TEST_CASE(sub_test)
{
    BOOST_CHECK_EQUAL(math::sub<double>(2, 6), -4);
}

BOOST_AUTO_TEST_CASE(times_test)
{
    BOOST_CHECK_EQUAL(math::times<double>(3, 10), 30);
}

BOOST_AUTO_TEST_CASE(div_test)
{
    BOOST_CHECK_EQUAL(math::div<double>(500, 5), 100);
}

BOOST_AUTO_TEST_CASE(arithmetic_min_test01)
{
    BOOST_CHECK_EQUAL(math::arithmetic_min<double>(0, 1000000), 0);
}

BOOST_AUTO_TEST_CASE(arithmetic_min_test02)
{
    BOOST_CHECK_EQUAL(math::arithmetic_min<double>(-5, 0), -5);
}

BOOST_AUTO_TEST_CASE(arithmetic_min_test03)
{
    BOOST_CHECK_EQUAL(math::arithmetic_min<double>(7, 3), 3);
}

BOOST_AUTO_TEST_CASE(incr_second_test01)
{
    BOOST_CHECK_EQUAL(math::incr_second<double>(5, 20), 21);
}

BOOST_AUTO_TEST_CASE(incr_second_test02)
{
    BOOST_CHECK_EQUAL(math::incr_second<double>(0, 20), 0);
}

BOOST_AUTO_TEST_CASE(incr_second_test03)
{
    BOOST_CHECK_EQUAL(math::incr_second<double>(0, 0), 0);
}

BOOST_AUTO_TEST_CASE(incr_second_test04)
{
    BOOST_CHECK_EQUAL(math::incr_second<double>(5, 0), 0);
}

BOOST_AUTO_TEST_CASE(min_test01)
{
    BOOST_CHECK_EQUAL(math::annihilator_min<double>(5, 10), 5);
}

BOOST_AUTO_TEST_CASE(min_test02)
{
    BOOST_CHECK_EQUAL(math::annihilator_min<double>(
                          std::numeric_limits<double>::max(), 10), 10);
}

BOOST_AUTO_TEST_CASE(min_test03)
{
    BOOST_CHECK_EQUAL(math::annihilator_min<double>(
                          5, std::numeric_limits<double>::max()), 5);
}

BOOST_AUTO_TEST_CASE(min_test04)
{
    BOOST_CHECK_EQUAL(math::annihilator_min<double>(
                          std::numeric_limits<double>::max(),
                          std::numeric_limits<double>::max()),
                      std::numeric_limits<double>::max());
}

BOOST_AUTO_TEST_CASE(select2nd_test02)
{
    BOOST_CHECK_EQUAL(math::select2nd<double>(
                          std::numeric_limits<double>::max(), 20),
                      std::numeric_limits<double>::max());
}

BOOST_AUTO_TEST_CASE(select2nd_test03)
{
    BOOST_CHECK_EQUAL(math::select2nd<double>(
                          std::numeric_limits<double>::max(),
                          std::numeric_limits<double>::max()),
                      std::numeric_limits<double>::max());
}

BOOST_AUTO_TEST_CASE(select2nd_test04)
{
    BOOST_CHECK_EQUAL(math::select2nd<double>(5, 1337), 1337);
}

BOOST_AUTO_TEST_CASE(is_equal_test01)
{
    BOOST_CHECK_EQUAL(math::is_equal<double>(1, 1), true);
}

BOOST_AUTO_TEST_CASE(is_equal_test02)
{
    BOOST_CHECK_EQUAL(math::is_equal<double>(0xC0FFEE, 0xCAFE), false);
}

BOOST_AUTO_TEST_CASE(is_zero_test01)
{
    BOOST_CHECK_EQUAL(math::is_zero<double>(0), true);
}

BOOST_AUTO_TEST_CASE(is_zero_test02)
{
    BOOST_CHECK_EQUAL(math::is_zero<double>(77), false);
}

BOOST_AUTO_TEST_CASE(remove_item_test)
{
    BOOST_CHECK_EQUAL(math::remove_item<double>(1, 0), 0);
}

BOOST_AUTO_TEST_CASE(xor_fn_test)
{
    BOOST_CHECK_EQUAL(math::xor_fn<int>(1, 2), 3);
}

BOOST_AUTO_TEST_CASE(or_fn_test)
{
    BOOST_CHECK_EQUAL(math::or_fn<int>(1, 0), true);
}

BOOST_AUTO_TEST_CASE(and_fn_test)
{
    BOOST_CHECK_EQUAL(math::and_fn<int>(0, 2), false);
}

BOOST_AUTO_TEST_CASE(not_fn_test)
{
    BOOST_CHECK_EQUAL(math::not_fn<int>(1), false);
}

BOOST_AUTO_TEST_SUITE_END()
