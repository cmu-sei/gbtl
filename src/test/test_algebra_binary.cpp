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

#include <functional>
#include <iostream>
#include <vector>

#include <graphblas/graphblas.hpp>

using namespace grb;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE algebra_binary_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

//****************************************************************************
// Summary
//****************************************************************************
BOOST_AUTO_TEST_CASE(misc_math_tests)
{
    BOOST_CHECK_EQUAL(Equal<double>()(1, 1), true);
    BOOST_CHECK_EQUAL(Equal<double>()(0xC0FFEE, 0xCAFE), false);
    BOOST_CHECK_EQUAL(NotEqual<double>()(1, 1), false);
    BOOST_CHECK_EQUAL(NotEqual<double>()(0xC0FFEE, 0xCAFE), true);

    BOOST_CHECK_EQUAL(GreaterThan<double>()(1, 2), false);
    BOOST_CHECK_EQUAL(LessThan<double>()(1, 2), true);
    BOOST_CHECK_EQUAL(GreaterEqual<double>()(1, 2), false);
    BOOST_CHECK_EQUAL(LessEqual<double>()(1, 1), true);

    BOOST_CHECK_EQUAL(First<double>()(5, 1337), 5);
    BOOST_CHECK_EQUAL(Second<double>()(5, 1337), 1337);

    BOOST_CHECK_EQUAL(Min<double>()(0, 1000000), 0);
    BOOST_CHECK_EQUAL(Min<double>()(-5, 0), -5);
    BOOST_CHECK_EQUAL(Min<double>()(7, 3), 3);
    BOOST_CHECK_EQUAL(Max<double>()(0, 1000000), 1000000);
    BOOST_CHECK_EQUAL(Max<double>()(-5, 0), 0);
    BOOST_CHECK_EQUAL(Max<double>()(7, 3), 7);

    BOOST_CHECK_EQUAL(Plus<double>()(2, 6), 8);
    BOOST_CHECK_EQUAL(Minus<double>()(2, 6), -4);
    BOOST_CHECK_EQUAL(Times<double>()(3, 10), 30);
    BOOST_CHECK_EQUAL(Div<double>()(500, 5), 100);
    BOOST_CHECK_EQUAL(Power<double>()(2, 5), 32.0);
}

//****************************************************************************
// Test Binary Operators
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(equal_same_domain_test)
{
    BOOST_CHECK_EQUAL(Equal<double>()(0.0, 0.0), true);
    BOOST_CHECK_EQUAL(Equal<double>()(1.0, 0.0), false);
    BOOST_CHECK_EQUAL(Equal<double>()(0.0, 1.0), false);
    BOOST_CHECK_EQUAL(Equal<double>()(1.0, 1.0), true);

    BOOST_CHECK_EQUAL(Equal<float>()(0.0f, 0.0f), true);
    BOOST_CHECK_EQUAL(Equal<float>()(1.0f, 0.0f), false);
    BOOST_CHECK_EQUAL(Equal<float>()(0.0f, 1.0f), false);
    BOOST_CHECK_EQUAL(Equal<float>()(1.0f, 1.0f), true);

    BOOST_CHECK_EQUAL(Equal<uint64_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(Equal<uint64_t>()(1, 0), false);
    BOOST_CHECK_EQUAL(Equal<uint64_t>()(0, 1), false);
    BOOST_CHECK_EQUAL(Equal<uint64_t>()(1, 1), true);

    BOOST_CHECK_EQUAL(Equal<uint32_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(Equal<uint32_t>()(1, 0), false);
    BOOST_CHECK_EQUAL(Equal<uint32_t>()(0, 1), false);
    BOOST_CHECK_EQUAL(Equal<uint32_t>()(1, 1), true);

    BOOST_CHECK_EQUAL(Equal<uint16_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(Equal<uint16_t>()(1, 0), false);
    BOOST_CHECK_EQUAL(Equal<uint16_t>()(0, 1), false);
    BOOST_CHECK_EQUAL(Equal<uint16_t>()(1, 1), true);

    BOOST_CHECK_EQUAL(Equal<uint8_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(Equal<uint8_t>()(1, 0), false);
    BOOST_CHECK_EQUAL(Equal<uint8_t>()(0, 1), false);
    BOOST_CHECK_EQUAL(Equal<uint8_t>()(1, 1), true);

    BOOST_CHECK_EQUAL(Equal<int64_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(Equal<int64_t>()(-1, 0), false);
    BOOST_CHECK_EQUAL(Equal<int64_t>()(0, -1), false);
    BOOST_CHECK_EQUAL(Equal<int64_t>()(-1, -1), true);

    BOOST_CHECK_EQUAL(Equal<int32_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(Equal<int32_t>()(-1, 0), false);
    BOOST_CHECK_EQUAL(Equal<int32_t>()(0, -1), false);
    BOOST_CHECK_EQUAL(Equal<int32_t>()(-1, -1), true);

    BOOST_CHECK_EQUAL(Equal<int16_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(Equal<int16_t>()(-1, 0), false);
    BOOST_CHECK_EQUAL(Equal<int16_t>()(0, -1), false);
    BOOST_CHECK_EQUAL(Equal<int16_t>()(-1, -1), true);

    BOOST_CHECK_EQUAL(Equal<int8_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(Equal<int8_t>()(-1, 0), false);
    BOOST_CHECK_EQUAL(Equal<int8_t>()(0, -1), false);
    BOOST_CHECK_EQUAL(Equal<int8_t>()(-1, -1), true);

    BOOST_CHECK_EQUAL(Equal<bool>()(false, false), true);
    BOOST_CHECK_EQUAL(Equal<bool>()(false, true),  false);
    BOOST_CHECK_EQUAL(Equal<bool>()(true, false),  false);
    BOOST_CHECK_EQUAL(Equal<bool>()(true, true),   true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(equal_different_domain_test)
{
    BOOST_CHECK_EQUAL((Equal<double,bool    >()(0.0, 0.0)), true);
    BOOST_CHECK_EQUAL((Equal<double,int32_t >()(1.0, 0.0)), false);
    BOOST_CHECK_EQUAL((Equal<double,uint64_t>()(0.0, 1.0)), false);
    BOOST_CHECK_EQUAL((Equal<double,float   >()(1.0, 1.0f)), true);

    BOOST_CHECK_EQUAL((Equal<float,bool    >()(0.0f, false)), true);
    BOOST_CHECK_EQUAL((Equal<float,double  >()(1.0f, 0.0)), false);
    BOOST_CHECK_EQUAL((Equal<float,uint64_t>()(0.0f, 1)), false);
    BOOST_CHECK_EQUAL((Equal<float,int32_t >()(1.0f, 1)), true);

    BOOST_CHECK_EQUAL((Equal<uint64_t,bool    >()(0, false)), true);
    BOOST_CHECK_EQUAL((Equal<uint64_t,int32_t >()(1, 0)), false);
    BOOST_CHECK_EQUAL((Equal<uint64_t,uint64_t>()(0, 1)), false);
    BOOST_CHECK_EQUAL((Equal<uint64_t,float   >()(1, 1.0f)), true);

    BOOST_CHECK_EQUAL((Equal<uint32_t,bool    >()(0, false)), true);
    BOOST_CHECK_EQUAL((Equal<uint32_t,int32_t >()(1, 0)), false);
    BOOST_CHECK_EQUAL((Equal<uint32_t,uint64_t>()(0, 1)), false);
    BOOST_CHECK_EQUAL((Equal<uint32_t,float   >()(1, 1)), true);

    BOOST_CHECK_EQUAL((Equal<uint16_t,bool    >()(0, false)), true);
    BOOST_CHECK_EQUAL((Equal<uint16_t,int32_t >()(1, 0)), false);
    BOOST_CHECK_EQUAL((Equal<uint16_t,uint64_t>()(0, 1)), false);
    BOOST_CHECK_EQUAL((Equal<uint16_t,float   >()(1, 1.0f)), true);

    BOOST_CHECK_EQUAL((Equal<uint8_t,bool    >()(0, false)), true);
    BOOST_CHECK_EQUAL((Equal<uint8_t,int32_t >()(1, 0)), false);
    BOOST_CHECK_EQUAL((Equal<uint8_t,uint64_t>()(0, 1)), false);
    BOOST_CHECK_EQUAL((Equal<uint8_t,float   >()(1, 1.0f)), true);

    BOOST_CHECK_EQUAL((Equal<int64_t,bool    >()( 0,  false)), true);
    BOOST_CHECK_EQUAL((Equal<int64_t,uint32_t>()(-1,  0)), false);
    BOOST_CHECK_EQUAL((Equal<int64_t,int64_t >()( 0, -1)), false);
    BOOST_CHECK_EQUAL((Equal<int64_t,float   >()(-1, -1.0f)), true);

    BOOST_CHECK_EQUAL((Equal<int32_t,bool    >()( 0,  false)), true);
    BOOST_CHECK_EQUAL((Equal<int32_t,uint32_t>()(-1,  0)), false);
    BOOST_CHECK_EQUAL((Equal<int32_t,int64_t >()( 0, -1)), false);
    BOOST_CHECK_EQUAL((Equal<int32_t,float   >()(-1, -1.0f)), true);

    BOOST_CHECK_EQUAL((Equal<int16_t,bool    >()( 0,  false)), true);
    BOOST_CHECK_EQUAL((Equal<int16_t,uint32_t>()(-1,  0)),     false);
    BOOST_CHECK_EQUAL((Equal<int16_t,int64_t >()( 0, -1)),     false);
    BOOST_CHECK_EQUAL((Equal<int16_t,float   >()(-1, -1.0f)),   true);

    BOOST_CHECK_EQUAL((Equal<int8_t,bool    >()( 0,  false)), true);
    BOOST_CHECK_EQUAL((Equal<int8_t,uint32_t>()(-1,  0)),     false);
    BOOST_CHECK_EQUAL((Equal<int8_t,int64_t> ()( 0, -1)),     false);
    BOOST_CHECK_EQUAL((Equal<int8_t,float   >()(-1, -1.f)),   true);

    BOOST_CHECK_EQUAL((Equal<bool,int8_t  >()(false, 0)),   true);
    BOOST_CHECK_EQUAL((Equal<bool,int32_t >()(false, 1)),   false);
    BOOST_CHECK_EQUAL((Equal<bool,uint64_t>()(true,  0UL)), false);
    BOOST_CHECK_EQUAL((Equal<bool,float   >()(true,  1.0f)),true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(not_equal_same_domain_test)
{
    BOOST_CHECK_EQUAL(NotEqual<double>()(0.0, 0.0), false);
    BOOST_CHECK_EQUAL(NotEqual<double>()(1.0, 0.0), true);
    BOOST_CHECK_EQUAL(NotEqual<double>()(0.0, 1.0), true);
    BOOST_CHECK_EQUAL(NotEqual<double>()(1.0, 1.0), false);

    BOOST_CHECK_EQUAL(NotEqual<float>()(0.0f, 0.0f), false);
    BOOST_CHECK_EQUAL(NotEqual<float>()(1.0f, 0.0f), true);
    BOOST_CHECK_EQUAL(NotEqual<float>()(0.0f, 1.0f), true);
    BOOST_CHECK_EQUAL(NotEqual<float>()(1.0f, 1.0f), false);

    BOOST_CHECK_EQUAL(NotEqual<uint64_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(NotEqual<uint64_t>()(1, 0), true);
    BOOST_CHECK_EQUAL(NotEqual<uint64_t>()(0, 1), true);
    BOOST_CHECK_EQUAL(NotEqual<uint64_t>()(1, 1), false);

    BOOST_CHECK_EQUAL(NotEqual<uint32_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(NotEqual<uint32_t>()(1, 0), true);
    BOOST_CHECK_EQUAL(NotEqual<uint32_t>()(0, 1), true);
    BOOST_CHECK_EQUAL(NotEqual<uint32_t>()(1, 1), false);

    BOOST_CHECK_EQUAL(NotEqual<uint16_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(NotEqual<uint16_t>()(1, 0), true);
    BOOST_CHECK_EQUAL(NotEqual<uint16_t>()(0, 1), true);
    BOOST_CHECK_EQUAL(NotEqual<uint16_t>()(1, 1), false);

    BOOST_CHECK_EQUAL(NotEqual<uint8_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(NotEqual<uint8_t>()(1, 0), true);
    BOOST_CHECK_EQUAL(NotEqual<uint8_t>()(0, 1), true);
    BOOST_CHECK_EQUAL(NotEqual<uint8_t>()(1, 1), false);

    BOOST_CHECK_EQUAL(NotEqual<int64_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(NotEqual<int64_t>()(-1, 0), true);
    BOOST_CHECK_EQUAL(NotEqual<int64_t>()(0, -1), true);
    BOOST_CHECK_EQUAL(NotEqual<int64_t>()(-1, -1), false);

    BOOST_CHECK_EQUAL(NotEqual<int32_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(NotEqual<int32_t>()(-1, 0), true);
    BOOST_CHECK_EQUAL(NotEqual<int32_t>()(0, -1), true);
    BOOST_CHECK_EQUAL(NotEqual<int32_t>()(-1, -1), false);

    BOOST_CHECK_EQUAL(NotEqual<int16_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(NotEqual<int16_t>()(-1, 0), true);
    BOOST_CHECK_EQUAL(NotEqual<int16_t>()(0, -1), true);
    BOOST_CHECK_EQUAL(NotEqual<int16_t>()(-1, -1), false);

    BOOST_CHECK_EQUAL(NotEqual<int8_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(NotEqual<int8_t>()(-1, 0), true);
    BOOST_CHECK_EQUAL(NotEqual<int8_t>()(0, -1), true);
    BOOST_CHECK_EQUAL(NotEqual<int8_t>()(-1, -1), false);

    BOOST_CHECK_EQUAL(NotEqual<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(NotEqual<bool>()(false, true),  true);
    BOOST_CHECK_EQUAL(NotEqual<bool>()(true, false),  true);
    BOOST_CHECK_EQUAL(NotEqual<bool>()(true, true),   false);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(not_equal_different_domain_test)
{
    BOOST_CHECK_EQUAL((NotEqual<double,bool    >()(0.0, 0.0)), false);
    BOOST_CHECK_EQUAL((NotEqual<double,int32_t >()(1.0, 0.0)), true);
    BOOST_CHECK_EQUAL((NotEqual<double,uint64_t>()(0.0, 1.0)), true);
    BOOST_CHECK_EQUAL((NotEqual<double,float   >()(1.0, 1.0f)), false);

    BOOST_CHECK_EQUAL((NotEqual<float,bool    >()(0.0f, false)), false);
    BOOST_CHECK_EQUAL((NotEqual<float,double  >()(1.0f, 0.0)), true);
    BOOST_CHECK_EQUAL((NotEqual<float,uint64_t>()(0.0f, 1)), true);
    BOOST_CHECK_EQUAL((NotEqual<float,int32_t >()(1.0f, 1)), false);

    BOOST_CHECK_EQUAL((NotEqual<uint64_t,bool    >()(0, false)), false);
    BOOST_CHECK_EQUAL((NotEqual<uint64_t,int32_t >()(1, 0)), true);
    BOOST_CHECK_EQUAL((NotEqual<uint64_t,uint64_t>()(0, 1)), true);
    BOOST_CHECK_EQUAL((NotEqual<uint64_t,float   >()(1, 1.0f)), false);

    BOOST_CHECK_EQUAL((NotEqual<uint32_t,bool    >()(0, false)), false);
    BOOST_CHECK_EQUAL((NotEqual<uint32_t,int32_t >()(1, 0)), true);
    BOOST_CHECK_EQUAL((NotEqual<uint32_t,uint64_t>()(0, 1)), true);
    BOOST_CHECK_EQUAL((NotEqual<uint32_t,float   >()(1, 1)), false);

    BOOST_CHECK_EQUAL((NotEqual<uint16_t,bool    >()(0, false)), false);
    BOOST_CHECK_EQUAL((NotEqual<uint16_t,int32_t >()(1, 0)), true);
    BOOST_CHECK_EQUAL((NotEqual<uint16_t,uint64_t>()(0, 1)), true);
    BOOST_CHECK_EQUAL((NotEqual<uint16_t,float   >()(1, 1.0f)), false);

    BOOST_CHECK_EQUAL((NotEqual<uint8_t,bool    >()(0, false)), false);
    BOOST_CHECK_EQUAL((NotEqual<uint8_t,int32_t >()(1, 0)), true);
    BOOST_CHECK_EQUAL((NotEqual<uint8_t,uint64_t>()(0, 1)), true);
    BOOST_CHECK_EQUAL((NotEqual<uint8_t,float   >()(1, 1.0f)), false);

    BOOST_CHECK_EQUAL((NotEqual<int64_t,bool    >()( 0,  false)), false);
    BOOST_CHECK_EQUAL((NotEqual<int64_t,uint32_t>()(-1,  0)), true);
    BOOST_CHECK_EQUAL((NotEqual<int64_t,int64_t >()( 0, -1)), true);
    BOOST_CHECK_EQUAL((NotEqual<int64_t,float   >()(-1, -1.0f)), false);

    BOOST_CHECK_EQUAL((NotEqual<int32_t,bool    >()( 0,  false)), false);
    BOOST_CHECK_EQUAL((NotEqual<int32_t,uint32_t>()(-1,  0)), true);
    BOOST_CHECK_EQUAL((NotEqual<int32_t,int64_t >()( 0, -1)), true);
    BOOST_CHECK_EQUAL((NotEqual<int32_t,float   >()(-1, -1.0f)), false);

    BOOST_CHECK_EQUAL((NotEqual<int16_t,bool    >()( 0,  false)), false);
    BOOST_CHECK_EQUAL((NotEqual<int16_t,uint32_t>()(-1,  0)),     true);
    BOOST_CHECK_EQUAL((NotEqual<int16_t,int64_t >()( 0, -1)),     true);
    BOOST_CHECK_EQUAL((NotEqual<int16_t,float   >()(-1, -1.0f)),   false);

    BOOST_CHECK_EQUAL((NotEqual<int8_t,bool    >()( 0,  false)), false);
    BOOST_CHECK_EQUAL((NotEqual<int8_t,uint32_t>()(-1,  0)),     true);
    BOOST_CHECK_EQUAL((NotEqual<int8_t,int64_t> ()( 0, -1)),     true);
    BOOST_CHECK_EQUAL((NotEqual<int8_t,float   >()(-1, -1.f)),   false);

    BOOST_CHECK_EQUAL((NotEqual<bool,int8_t  >()(false, 0)),   false);
    BOOST_CHECK_EQUAL((NotEqual<bool,int32_t >()(false, 1)),   true);
    BOOST_CHECK_EQUAL((NotEqual<bool,uint64_t>()(true,  0UL)), true);
    BOOST_CHECK_EQUAL((NotEqual<bool,float   >()(true,  1.0f)),false);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(greater_than_same_domain_test)
{
    BOOST_CHECK_EQUAL(GreaterThan<double>()(0.0, 0.0), false);
    BOOST_CHECK_EQUAL(GreaterThan<double>()(1.0, 0.0), true);
    BOOST_CHECK_EQUAL(GreaterThan<double>()(0.0, 1.0), false);
    BOOST_CHECK_EQUAL(GreaterThan<double>()(1.0, 1.0), false);

    BOOST_CHECK_EQUAL(GreaterThan<float>()(0.0f, 0.0f), false);
    BOOST_CHECK_EQUAL(GreaterThan<float>()(1.0f, 0.0f), true);
    BOOST_CHECK_EQUAL(GreaterThan<float>()(0.0f, 1.0f), false);
    BOOST_CHECK_EQUAL(GreaterThan<float>()(1.0f, 1.0f), false);

    BOOST_CHECK_EQUAL(GreaterThan<uint64_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GreaterThan<uint64_t>()(1, 0), true);
    BOOST_CHECK_EQUAL(GreaterThan<uint64_t>()(0, 1), false);
    BOOST_CHECK_EQUAL(GreaterThan<uint64_t>()(1, 1), false);

    BOOST_CHECK_EQUAL(GreaterThan<uint32_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GreaterThan<uint32_t>()(1, 0), true);
    BOOST_CHECK_EQUAL(GreaterThan<uint32_t>()(0, 1), false);
    BOOST_CHECK_EQUAL(GreaterThan<uint32_t>()(1, 1), false);

    BOOST_CHECK_EQUAL(GreaterThan<uint16_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GreaterThan<uint16_t>()(1, 0), true);
    BOOST_CHECK_EQUAL(GreaterThan<uint16_t>()(0, 1), false);
    BOOST_CHECK_EQUAL(GreaterThan<uint16_t>()(1, 1), false);

    BOOST_CHECK_EQUAL(GreaterThan<uint8_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GreaterThan<uint8_t>()(1, 0), true);
    BOOST_CHECK_EQUAL(GreaterThan<uint8_t>()(0, 1), false);
    BOOST_CHECK_EQUAL(GreaterThan<uint8_t>()(1, 1), false);

    BOOST_CHECK_EQUAL(GreaterThan<int64_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GreaterThan<int64_t>()(-1, 0), false);
    BOOST_CHECK_EQUAL(GreaterThan<int64_t>()(0, -1), true);
    BOOST_CHECK_EQUAL(GreaterThan<int64_t>()(-1, -1), false);

    BOOST_CHECK_EQUAL(GreaterThan<int32_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GreaterThan<int32_t>()(-1, 0), false);
    BOOST_CHECK_EQUAL(GreaterThan<int32_t>()(0, -1), true);
    BOOST_CHECK_EQUAL(GreaterThan<int32_t>()(-1, -1), false);

    BOOST_CHECK_EQUAL(GreaterThan<int16_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GreaterThan<int16_t>()(-1, 0), false);
    BOOST_CHECK_EQUAL(GreaterThan<int16_t>()(0, -1), true);
    BOOST_CHECK_EQUAL(GreaterThan<int16_t>()(-1, -1), false);

    BOOST_CHECK_EQUAL(GreaterThan<int8_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GreaterThan<int8_t>()(-1, 0), false);
    BOOST_CHECK_EQUAL(GreaterThan<int8_t>()(0, -1), true);
    BOOST_CHECK_EQUAL(GreaterThan<int8_t>()(-1, -1), false);

    BOOST_CHECK_EQUAL(GreaterThan<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(GreaterThan<bool>()(false, true),  false);
    BOOST_CHECK_EQUAL(GreaterThan<bool>()(true, false),  true);
    BOOST_CHECK_EQUAL(GreaterThan<bool>()(true, true),   false);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(greater_than_different_domain_test)
{
    BOOST_CHECK_EQUAL((GreaterThan<double,bool    >()(0.0, 0.0)), false);
    BOOST_CHECK_EQUAL((GreaterThan<double,int32_t >()(1.0, 0.0)), true);
    BOOST_CHECK_EQUAL((GreaterThan<double,uint64_t>()(0.0, 1.0)), false);
    BOOST_CHECK_EQUAL((GreaterThan<double,float   >()(1.0, 1.0f)), false);

    BOOST_CHECK_EQUAL((GreaterThan<float,bool    >()(0.0f, false)), false);
    BOOST_CHECK_EQUAL((GreaterThan<float,double  >()(1.0f, 0.0)), true);
    BOOST_CHECK_EQUAL((GreaterThan<float,uint64_t>()(0.0f, 1)), false);
    BOOST_CHECK_EQUAL((GreaterThan<float,int32_t >()(1.0f, 1)), false);

    BOOST_CHECK_EQUAL((GreaterThan<uint64_t,bool    >()(0, false)), false);
    BOOST_CHECK_EQUAL((GreaterThan<uint64_t,int32_t >()(1, 0)), true);
    BOOST_CHECK_EQUAL((GreaterThan<uint64_t,uint64_t>()(0, 1)), false);
    BOOST_CHECK_EQUAL((GreaterThan<uint64_t,float   >()(1, 1.0f)), false);

    BOOST_CHECK_EQUAL((GreaterThan<uint32_t,bool    >()(0, false)), false);
    BOOST_CHECK_EQUAL((GreaterThan<uint32_t,int32_t >()(1, 0)), true);
    BOOST_CHECK_EQUAL((GreaterThan<uint32_t,uint64_t>()(0, 1)), false);
    BOOST_CHECK_EQUAL((GreaterThan<uint32_t,float   >()(1, 1)), false);

    BOOST_CHECK_EQUAL((GreaterThan<uint16_t,bool    >()(0, false)), false);
    BOOST_CHECK_EQUAL((GreaterThan<uint16_t,int32_t >()(1, 0)), true);
    BOOST_CHECK_EQUAL((GreaterThan<uint16_t,uint64_t>()(0, 1)), false);
    BOOST_CHECK_EQUAL((GreaterThan<uint16_t,float   >()(1, 1.0f)), false);

    BOOST_CHECK_EQUAL((GreaterThan<uint8_t,bool    >()(0, false)), false);
    BOOST_CHECK_EQUAL((GreaterThan<uint8_t,int32_t >()(1, 0)), true);
    BOOST_CHECK_EQUAL((GreaterThan<uint8_t,uint64_t>()(0, 1)), false);
    BOOST_CHECK_EQUAL((GreaterThan<uint8_t,float   >()(1, 1.0f)), false);

    BOOST_CHECK_EQUAL((GreaterThan<int64_t,bool    >()( 0,  false)), false);
    BOOST_CHECK_EQUAL((GreaterThan<int64_t,uint32_t>()(-1,  0)), false);
    BOOST_CHECK_EQUAL((GreaterThan<int64_t,int64_t >()( 0, -1)), true);
    BOOST_CHECK_EQUAL((GreaterThan<int64_t,float   >()(-1, -1.0f)), false);

    BOOST_CHECK_EQUAL((GreaterThan<int32_t,bool    >()( 0,  false)), false);
    // STRANGE RESULT...-1 must be converted to uint32_t before comparison
    BOOST_CHECK_EQUAL((GreaterThan<int32_t,uint32_t>()(-1,  0)), true);
    BOOST_CHECK_EQUAL((GreaterThan<int32_t,int64_t >()( 0, -1)), true);
    BOOST_CHECK_EQUAL((GreaterThan<int32_t,float   >()(-1, -1.0f)), false);

    BOOST_CHECK_EQUAL((GreaterThan<int16_t,bool    >()( 0,  false)), false);
    // STRANGE RESULT...-1 must be converted to uint32_t before comparison
    BOOST_CHECK_EQUAL((GreaterThan<int16_t,uint32_t>()(-1,  0)),     true);
    BOOST_CHECK_EQUAL((GreaterThan<int16_t,int64_t >()( 0, -1)),     true);
    BOOST_CHECK_EQUAL((GreaterThan<int16_t,float   >()(-1, -1.0f)),   false);

    BOOST_CHECK_EQUAL((GreaterThan<int8_t,bool    >()( 0,  false)), false);
    // STRANGE RESULT...-1 must be converted to uint32_t before comparison
    BOOST_CHECK_EQUAL((GreaterThan<int8_t,uint32_t>()(-1,  0)),     true);
    BOOST_CHECK_EQUAL((GreaterThan<int8_t,int64_t> ()( 0, -1)),     true);
    BOOST_CHECK_EQUAL((GreaterThan<int8_t,float   >()(-1, -1.f)),   false);

    BOOST_CHECK_EQUAL((GreaterThan<bool,int8_t  >()(false, 0)),   false);
    BOOST_CHECK_EQUAL((GreaterThan<bool,int32_t >()(false, 1)),   false);
    BOOST_CHECK_EQUAL((GreaterThan<bool,uint64_t>()(true,  0UL)), true);
    BOOST_CHECK_EQUAL((GreaterThan<bool,float   >()(true,  1.0f)),false);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(less_than_same_domain_test)
{
    BOOST_CHECK_EQUAL(LessThan<double>()(0.0, 0.0), false);
    BOOST_CHECK_EQUAL(LessThan<double>()(1.0, 0.0), false);
    BOOST_CHECK_EQUAL(LessThan<double>()(0.0, 1.0), true);
    BOOST_CHECK_EQUAL(LessThan<double>()(1.0, 1.0), false);

    BOOST_CHECK_EQUAL(LessThan<float>()(0.0f, 0.0f), false);
    BOOST_CHECK_EQUAL(LessThan<float>()(1.0f, 0.0f), false);
    BOOST_CHECK_EQUAL(LessThan<float>()(0.0f, 1.0f), true);
    BOOST_CHECK_EQUAL(LessThan<float>()(1.0f, 1.0f), false);

    BOOST_CHECK_EQUAL(LessThan<uint64_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(LessThan<uint64_t>()(1, 0), false);
    BOOST_CHECK_EQUAL(LessThan<uint64_t>()(0, 1), true);
    BOOST_CHECK_EQUAL(LessThan<uint64_t>()(1, 1), false);

    BOOST_CHECK_EQUAL(LessThan<uint32_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(LessThan<uint32_t>()(1, 0), false);
    BOOST_CHECK_EQUAL(LessThan<uint32_t>()(0, 1), true);
    BOOST_CHECK_EQUAL(LessThan<uint32_t>()(1, 1), false);

    BOOST_CHECK_EQUAL(LessThan<uint16_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(LessThan<uint16_t>()(1, 0), false);
    BOOST_CHECK_EQUAL(LessThan<uint16_t>()(0, 1), true);
    BOOST_CHECK_EQUAL(LessThan<uint16_t>()(1, 1), false);

    BOOST_CHECK_EQUAL(LessThan<uint8_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(LessThan<uint8_t>()(1, 0), false);
    BOOST_CHECK_EQUAL(LessThan<uint8_t>()(0, 1), true);
    BOOST_CHECK_EQUAL(LessThan<uint8_t>()(1, 1), false);

    BOOST_CHECK_EQUAL(LessThan<int64_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(LessThan<int64_t>()(-1, 0), true);
    BOOST_CHECK_EQUAL(LessThan<int64_t>()(0, -1), false);
    BOOST_CHECK_EQUAL(LessThan<int64_t>()(-1, -1), false);

    BOOST_CHECK_EQUAL(LessThan<int32_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(LessThan<int32_t>()(-1, 0), true);
    BOOST_CHECK_EQUAL(LessThan<int32_t>()(0, -1), false);
    BOOST_CHECK_EQUAL(LessThan<int32_t>()(-1, -1), false);

    BOOST_CHECK_EQUAL(LessThan<int16_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(LessThan<int16_t>()(-1, 0), true);
    BOOST_CHECK_EQUAL(LessThan<int16_t>()(0, -1), false);
    BOOST_CHECK_EQUAL(LessThan<int16_t>()(-1, -1), false);

    BOOST_CHECK_EQUAL(LessThan<int8_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(LessThan<int8_t>()(-1, 0), true);
    BOOST_CHECK_EQUAL(LessThan<int8_t>()(0, -1), false);
    BOOST_CHECK_EQUAL(LessThan<int8_t>()(-1, -1), false);

    BOOST_CHECK_EQUAL(LessThan<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(LessThan<bool>()(false, true),  true);
    BOOST_CHECK_EQUAL(LessThan<bool>()(true, false),  false);
    BOOST_CHECK_EQUAL(LessThan<bool>()(true, true),   false);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(less_than_different_domain_test)
{
    BOOST_CHECK_EQUAL((LessThan<double,bool    >()(0.0, 0.0)), false);
    BOOST_CHECK_EQUAL((LessThan<double,int32_t >()(1.0, 0.0)), false);
    BOOST_CHECK_EQUAL((LessThan<double,uint64_t>()(0.0, 1.0)), true);
    BOOST_CHECK_EQUAL((LessThan<double,float   >()(1.0, 1.0f)), false);

    BOOST_CHECK_EQUAL((LessThan<float,bool    >()(0.0f, false)), false);
    BOOST_CHECK_EQUAL((LessThan<float,double  >()(1.0f, 0.0)), false);
    BOOST_CHECK_EQUAL((LessThan<float,uint64_t>()(0.0f, 1)), true);
    BOOST_CHECK_EQUAL((LessThan<float,int32_t >()(1.0f, 1)), false);

    BOOST_CHECK_EQUAL((LessThan<uint64_t,bool    >()(0, false)), false);
    BOOST_CHECK_EQUAL((LessThan<uint64_t,int32_t >()(1, 0)), false);
    BOOST_CHECK_EQUAL((LessThan<uint64_t,uint64_t>()(0, 1)), true);
    BOOST_CHECK_EQUAL((LessThan<uint64_t,float   >()(1, 1.0f)), false);

    BOOST_CHECK_EQUAL((LessThan<uint32_t,bool    >()(0, false)), false);
    BOOST_CHECK_EQUAL((LessThan<uint32_t,int32_t >()(1, 0)), false);
    BOOST_CHECK_EQUAL((LessThan<uint32_t,uint64_t>()(0, 1)), true);
    BOOST_CHECK_EQUAL((LessThan<uint32_t,float   >()(1, 1)), false);

    BOOST_CHECK_EQUAL((LessThan<uint16_t,bool    >()(0, false)), false);
    BOOST_CHECK_EQUAL((LessThan<uint16_t,int32_t >()(1, 0)), false);
    BOOST_CHECK_EQUAL((LessThan<uint16_t,uint64_t>()(0, 1)), true);
    BOOST_CHECK_EQUAL((LessThan<uint16_t,float   >()(1, 1.0f)), false);

    BOOST_CHECK_EQUAL((LessThan<uint8_t,bool    >()(0, false)), false);
    BOOST_CHECK_EQUAL((LessThan<uint8_t,int32_t >()(1, 0)), false);
    BOOST_CHECK_EQUAL((LessThan<uint8_t,uint64_t>()(0, 1)), true);
    BOOST_CHECK_EQUAL((LessThan<uint8_t,float   >()(1, 1.0f)), false);

    BOOST_CHECK_EQUAL((LessThan<int64_t,bool    >()( 0,  false)), false);
    BOOST_CHECK_EQUAL((LessThan<int64_t,uint32_t>()(-1,  0)), true);
    BOOST_CHECK_EQUAL((LessThan<int64_t,int64_t >()( 0, -1)), false);
    BOOST_CHECK_EQUAL((LessThan<int64_t,float   >()(-1, -1.0f)), false);

    BOOST_CHECK_EQUAL((LessThan<int32_t,bool    >()( 0,  false)), false);
    // STRANGE RESULT...-1 must be converted to uint32_t before comparison
    BOOST_CHECK_EQUAL((LessThan<int32_t,uint32_t>()(-1,  0)), false);
    BOOST_CHECK_EQUAL((LessThan<int32_t,int64_t >()( 0, -1)), false);
    BOOST_CHECK_EQUAL((LessThan<int32_t,float   >()(-1, -1.0f)), false);

    BOOST_CHECK_EQUAL((LessThan<int16_t,bool    >()( 0,  false)), false);
    // STRANGE RESULT...-1 must be converted to uint32_t before comparison
    BOOST_CHECK_EQUAL((LessThan<int16_t,uint32_t>()(-1,  0)),     false);
    BOOST_CHECK_EQUAL((LessThan<int16_t,int64_t >()( 0, -1)),     false);
    BOOST_CHECK_EQUAL((LessThan<int16_t,float   >()(-1, -1.0f)),   false);

    BOOST_CHECK_EQUAL((LessThan<int8_t,bool    >()( 0,  false)), false);
    // STRANGE RESULT...-1 must be converted to uint32_t before comparison
    BOOST_CHECK_EQUAL((LessThan<int8_t,uint32_t>()(-1,  0)),     false);
    BOOST_CHECK_EQUAL((LessThan<int8_t,int64_t> ()( 0, -1)),     false);
    BOOST_CHECK_EQUAL((LessThan<int8_t,float   >()(-1, -1.f)),   false);

    BOOST_CHECK_EQUAL((LessThan<bool,int8_t  >()(false, 0)),   false);
    BOOST_CHECK_EQUAL((LessThan<bool,int32_t >()(false, 1)),   true);
    BOOST_CHECK_EQUAL((LessThan<bool,uint64_t>()(true,  0UL)), false);
    BOOST_CHECK_EQUAL((LessThan<bool,float   >()(true,  1.0f)),false);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(greater_equal_same_domain_test)
{
    BOOST_CHECK_EQUAL(GreaterEqual<double>()(0.0, 0.0), true);
    BOOST_CHECK_EQUAL(GreaterEqual<double>()(1.0, 0.0), true);
    BOOST_CHECK_EQUAL(GreaterEqual<double>()(0.0, 1.0), false);
    BOOST_CHECK_EQUAL(GreaterEqual<double>()(1.0, 1.0), true);

    BOOST_CHECK_EQUAL(GreaterEqual<float>()(0.0f, 0.0f), true);
    BOOST_CHECK_EQUAL(GreaterEqual<float>()(1.0f, 0.0f), true);
    BOOST_CHECK_EQUAL(GreaterEqual<float>()(0.0f, 1.0f), false);
    BOOST_CHECK_EQUAL(GreaterEqual<float>()(1.0f, 1.0f), true);

    BOOST_CHECK_EQUAL(GreaterEqual<uint64_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GreaterEqual<uint64_t>()(1, 0), true);
    BOOST_CHECK_EQUAL(GreaterEqual<uint64_t>()(0, 1), false);
    BOOST_CHECK_EQUAL(GreaterEqual<uint64_t>()(1, 1), true);

    BOOST_CHECK_EQUAL(GreaterEqual<uint32_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GreaterEqual<uint32_t>()(1, 0), true);
    BOOST_CHECK_EQUAL(GreaterEqual<uint32_t>()(0, 1), false);
    BOOST_CHECK_EQUAL(GreaterEqual<uint32_t>()(1, 1), true);

    BOOST_CHECK_EQUAL(GreaterEqual<uint16_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GreaterEqual<uint16_t>()(1, 0), true);
    BOOST_CHECK_EQUAL(GreaterEqual<uint16_t>()(0, 1), false);
    BOOST_CHECK_EQUAL(GreaterEqual<uint16_t>()(1, 1), true);

    BOOST_CHECK_EQUAL(GreaterEqual<uint8_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GreaterEqual<uint8_t>()(1, 0), true);
    BOOST_CHECK_EQUAL(GreaterEqual<uint8_t>()(0, 1), false);
    BOOST_CHECK_EQUAL(GreaterEqual<uint8_t>()(1, 1), true);

    BOOST_CHECK_EQUAL(GreaterEqual<int64_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GreaterEqual<int64_t>()(-1, 0), false);
    BOOST_CHECK_EQUAL(GreaterEqual<int64_t>()(0, -1), true);
    BOOST_CHECK_EQUAL(GreaterEqual<int64_t>()(-1, -1), true);

    BOOST_CHECK_EQUAL(GreaterEqual<int32_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GreaterEqual<int32_t>()(-1, 0), false);
    BOOST_CHECK_EQUAL(GreaterEqual<int32_t>()(0, -1), true);
    BOOST_CHECK_EQUAL(GreaterEqual<int32_t>()(-1, -1), true);

    BOOST_CHECK_EQUAL(GreaterEqual<int16_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GreaterEqual<int16_t>()(-1, 0), false);
    BOOST_CHECK_EQUAL(GreaterEqual<int16_t>()(0, -1), true);
    BOOST_CHECK_EQUAL(GreaterEqual<int16_t>()(-1, -1), true);

    BOOST_CHECK_EQUAL(GreaterEqual<int8_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GreaterEqual<int8_t>()(-1, 0), false);
    BOOST_CHECK_EQUAL(GreaterEqual<int8_t>()(0, -1), true);
    BOOST_CHECK_EQUAL(GreaterEqual<int8_t>()(-1, -1), true);

    BOOST_CHECK_EQUAL(GreaterEqual<bool>()(false, false), true);
    BOOST_CHECK_EQUAL(GreaterEqual<bool>()(false, true),  false);
    BOOST_CHECK_EQUAL(GreaterEqual<bool>()(true, false),  true);
    BOOST_CHECK_EQUAL(GreaterEqual<bool>()(true, true),   true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(greater_equal_different_domain_test)
{
    BOOST_CHECK_EQUAL((GreaterEqual<double,bool    >()(0.0, 0.0)), true);
    BOOST_CHECK_EQUAL((GreaterEqual<double,int32_t >()(1.0, 0.0)), true);
    BOOST_CHECK_EQUAL((GreaterEqual<double,uint64_t>()(0.0, 1.0)), false);
    BOOST_CHECK_EQUAL((GreaterEqual<double,float   >()(1.0, 1.0f)), true);

    BOOST_CHECK_EQUAL((GreaterEqual<float,bool    >()(0.0f, false)), true);
    BOOST_CHECK_EQUAL((GreaterEqual<float,double  >()(1.0f, 0.0)), true);
    BOOST_CHECK_EQUAL((GreaterEqual<float,uint64_t>()(0.0f, 1)), false);
    BOOST_CHECK_EQUAL((GreaterEqual<float,int32_t >()(1.0f, 1)), true);

    BOOST_CHECK_EQUAL((GreaterEqual<uint64_t,bool    >()(0, false)), true);
    BOOST_CHECK_EQUAL((GreaterEqual<uint64_t,int32_t >()(1, 0)), true);
    BOOST_CHECK_EQUAL((GreaterEqual<uint64_t,uint64_t>()(0, 1)), false);
    BOOST_CHECK_EQUAL((GreaterEqual<uint64_t,float   >()(1, 1.0f)), true);

    BOOST_CHECK_EQUAL((GreaterEqual<uint32_t,bool    >()(0, false)), true);
    BOOST_CHECK_EQUAL((GreaterEqual<uint32_t,int32_t >()(1, 0)), true);
    BOOST_CHECK_EQUAL((GreaterEqual<uint32_t,uint64_t>()(0, 1)), false);
    BOOST_CHECK_EQUAL((GreaterEqual<uint32_t,float   >()(1, 1)), true);

    BOOST_CHECK_EQUAL((GreaterEqual<uint16_t,bool    >()(0, false)), true);
    BOOST_CHECK_EQUAL((GreaterEqual<uint16_t,int32_t >()(1, 0)), true);
    BOOST_CHECK_EQUAL((GreaterEqual<uint16_t,uint64_t>()(0, 1)), false);
    BOOST_CHECK_EQUAL((GreaterEqual<uint16_t,float   >()(1, 1.0f)), true);

    BOOST_CHECK_EQUAL((GreaterEqual<uint8_t,bool    >()(0, false)), true);
    BOOST_CHECK_EQUAL((GreaterEqual<uint8_t,int32_t >()(1, 0)), true);
    BOOST_CHECK_EQUAL((GreaterEqual<uint8_t,uint64_t>()(0, 1)), false);
    BOOST_CHECK_EQUAL((GreaterEqual<uint8_t,float   >()(1, 1.0f)), true);

    BOOST_CHECK_EQUAL((GreaterEqual<int64_t,bool    >()( 0,  false)), true);
    BOOST_CHECK_EQUAL((GreaterEqual<int64_t,uint32_t>()(-1,  0)), false);
    BOOST_CHECK_EQUAL((GreaterEqual<int64_t,int64_t >()( 0, -1)), true);
    BOOST_CHECK_EQUAL((GreaterEqual<int64_t,float   >()(-1, -1.0f)), true);

    BOOST_CHECK_EQUAL((GreaterEqual<int32_t,bool    >()( 0,  false)), true);
    // STRANGE RESULT...-1 must be converted to uint32_t before comparison
    BOOST_CHECK_EQUAL((GreaterEqual<int32_t,uint32_t>()(-1,  0)), true);
    BOOST_CHECK_EQUAL((GreaterEqual<int32_t,int64_t >()( 0, -1)), true);
    BOOST_CHECK_EQUAL((GreaterEqual<int32_t,float   >()(-1, -1.0f)), true);

    BOOST_CHECK_EQUAL((GreaterEqual<int16_t,bool    >()( 0,  false)), true);
    // STRANGE RESULT...-1 must be converted to uint32_t before comparison
    BOOST_CHECK_EQUAL((GreaterEqual<int16_t,uint32_t>()(-1,  0)),     true);
    BOOST_CHECK_EQUAL((GreaterEqual<int16_t,int64_t >()( 0, -1)),     true);
    BOOST_CHECK_EQUAL((GreaterEqual<int16_t,float   >()(-1, -1.0f)),   true);

    BOOST_CHECK_EQUAL((GreaterEqual<int8_t,bool    >()( 0,  false)), true);
    // STRANGE RESULT...-1 must be converted to uint32_t before comparison
    BOOST_CHECK_EQUAL((GreaterEqual<int8_t,uint32_t>()(-1,  0)),     true);
    BOOST_CHECK_EQUAL((GreaterEqual<int8_t,int64_t> ()( 0, -1)),     true);
    BOOST_CHECK_EQUAL((GreaterEqual<int8_t,float   >()(-1, -1.f)),   true);

    BOOST_CHECK_EQUAL((GreaterEqual<bool,int8_t  >()(false, 0)),   true);
    BOOST_CHECK_EQUAL((GreaterEqual<bool,int32_t >()(false, 1)),   false);
    BOOST_CHECK_EQUAL((GreaterEqual<bool,uint64_t>()(true,  0UL)), true);
    BOOST_CHECK_EQUAL((GreaterEqual<bool,float   >()(true,  1.0f)),true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(less_equal_same_domain_test)
{
    BOOST_CHECK_EQUAL(LessEqual<double>()(0.0, 0.0), true);
    BOOST_CHECK_EQUAL(LessEqual<double>()(1.0, 0.0), false);
    BOOST_CHECK_EQUAL(LessEqual<double>()(0.0, 1.0), true);
    BOOST_CHECK_EQUAL(LessEqual<double>()(1.0, 1.0), true);

    BOOST_CHECK_EQUAL(LessEqual<float>()(0.0f, 0.0f), true);
    BOOST_CHECK_EQUAL(LessEqual<float>()(1.0f, 0.0f), false);
    BOOST_CHECK_EQUAL(LessEqual<float>()(0.0f, 1.0f), true);
    BOOST_CHECK_EQUAL(LessEqual<float>()(1.0f, 1.0f), true);

    BOOST_CHECK_EQUAL(LessEqual<uint64_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(LessEqual<uint64_t>()(1, 0), false);
    BOOST_CHECK_EQUAL(LessEqual<uint64_t>()(0, 1), true);
    BOOST_CHECK_EQUAL(LessEqual<uint64_t>()(1, 1), true);

    BOOST_CHECK_EQUAL(LessEqual<uint32_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(LessEqual<uint32_t>()(1, 0), false);
    BOOST_CHECK_EQUAL(LessEqual<uint32_t>()(0, 1), true);
    BOOST_CHECK_EQUAL(LessEqual<uint32_t>()(1, 1), true);

    BOOST_CHECK_EQUAL(LessEqual<uint16_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(LessEqual<uint16_t>()(1, 0), false);
    BOOST_CHECK_EQUAL(LessEqual<uint16_t>()(0, 1), true);
    BOOST_CHECK_EQUAL(LessEqual<uint16_t>()(1, 1), true);

    BOOST_CHECK_EQUAL(LessEqual<uint8_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(LessEqual<uint8_t>()(1, 0), false);
    BOOST_CHECK_EQUAL(LessEqual<uint8_t>()(0, 1), true);
    BOOST_CHECK_EQUAL(LessEqual<uint8_t>()(1, 1), true);

    BOOST_CHECK_EQUAL(LessEqual<int64_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(LessEqual<int64_t>()(-1, 0), true);
    BOOST_CHECK_EQUAL(LessEqual<int64_t>()(0, -1), false);
    BOOST_CHECK_EQUAL(LessEqual<int64_t>()(-1, -1), true);

    BOOST_CHECK_EQUAL(LessEqual<int32_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(LessEqual<int32_t>()(-1, 0), true);
    BOOST_CHECK_EQUAL(LessEqual<int32_t>()(0, -1), false);
    BOOST_CHECK_EQUAL(LessEqual<int32_t>()(-1, -1), true);

    BOOST_CHECK_EQUAL(LessEqual<int16_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(LessEqual<int16_t>()(-1, 0), true);
    BOOST_CHECK_EQUAL(LessEqual<int16_t>()(0, -1), false);
    BOOST_CHECK_EQUAL(LessEqual<int16_t>()(-1, -1), true);

    BOOST_CHECK_EQUAL(LessEqual<int8_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(LessEqual<int8_t>()(-1, 0), true);
    BOOST_CHECK_EQUAL(LessEqual<int8_t>()(0, -1), false);
    BOOST_CHECK_EQUAL(LessEqual<int8_t>()(-1, -1), true);

    BOOST_CHECK_EQUAL(LessEqual<bool>()(false, false), true);
    BOOST_CHECK_EQUAL(LessEqual<bool>()(false, true),  true);
    BOOST_CHECK_EQUAL(LessEqual<bool>()(true, false),  false);
    BOOST_CHECK_EQUAL(LessEqual<bool>()(true, true),   true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(less_equal_different_domain_test)
{
    BOOST_CHECK_EQUAL((LessEqual<double,bool    >()(0.0, 0.0)), true);
    BOOST_CHECK_EQUAL((LessEqual<double,int32_t >()(1.0, 0.0)), false);
    BOOST_CHECK_EQUAL((LessEqual<double,uint64_t>()(0.0, 1.0)), true);
    BOOST_CHECK_EQUAL((LessEqual<double,float   >()(1.0, 1.0f)), true);

    BOOST_CHECK_EQUAL((LessEqual<float,bool    >()(0.0f, false)), true);
    BOOST_CHECK_EQUAL((LessEqual<float,double  >()(1.0f, 0.0)), false);
    BOOST_CHECK_EQUAL((LessEqual<float,uint64_t>()(0.0f, 1)), true);
    BOOST_CHECK_EQUAL((LessEqual<float,int32_t >()(1.0f, 1)), true);

    BOOST_CHECK_EQUAL((LessEqual<uint64_t,bool    >()(0, false)), true);
    BOOST_CHECK_EQUAL((LessEqual<uint64_t,int32_t >()(1, 0)), false);
    BOOST_CHECK_EQUAL((LessEqual<uint64_t,uint64_t>()(0, 1)), true);
    BOOST_CHECK_EQUAL((LessEqual<uint64_t,float   >()(1, 1.0f)), true);

    BOOST_CHECK_EQUAL((LessEqual<uint32_t,bool    >()(0, false)), true);
    BOOST_CHECK_EQUAL((LessEqual<uint32_t,int32_t >()(1, 0)), false);
    BOOST_CHECK_EQUAL((LessEqual<uint32_t,uint64_t>()(0, 1)), true);
    BOOST_CHECK_EQUAL((LessEqual<uint32_t,float   >()(1, 1)), true);

    BOOST_CHECK_EQUAL((LessEqual<uint16_t,bool    >()(0, false)), true);
    BOOST_CHECK_EQUAL((LessEqual<uint16_t,int32_t >()(1, 0)), false);
    BOOST_CHECK_EQUAL((LessEqual<uint16_t,uint64_t>()(0, 1)), true);
    BOOST_CHECK_EQUAL((LessEqual<uint16_t,float   >()(1, 1.0f)), true);

    BOOST_CHECK_EQUAL((LessEqual<uint8_t,bool    >()(0, false)), true);
    BOOST_CHECK_EQUAL((LessEqual<uint8_t,int32_t >()(1, 0)), false);
    BOOST_CHECK_EQUAL((LessEqual<uint8_t,uint64_t>()(0, 1)), true);
    BOOST_CHECK_EQUAL((LessEqual<uint8_t,float   >()(1, 1.0f)), true);

    BOOST_CHECK_EQUAL((LessEqual<int64_t,bool    >()( 0,  false)), true);
    BOOST_CHECK_EQUAL((LessEqual<int64_t,uint32_t>()(-1,  0)), true);
    BOOST_CHECK_EQUAL((LessEqual<int64_t,int64_t >()( 0, -1)), false);
    BOOST_CHECK_EQUAL((LessEqual<int64_t,float   >()(-1, -1.0f)), true);

    BOOST_CHECK_EQUAL((LessEqual<int32_t,bool    >()( 0,  false)), true);
    // STRANGE RESULT...-1 must be converted to uint32_t before comparison
    BOOST_CHECK_EQUAL((LessEqual<int32_t,uint32_t>()(-1,  0)), false);
    BOOST_CHECK_EQUAL((LessEqual<int32_t,int64_t >()( 0, -1)), false);
    BOOST_CHECK_EQUAL((LessEqual<int32_t,float   >()(-1, -1.0f)), true);

    BOOST_CHECK_EQUAL((LessEqual<int16_t,bool    >()( 0,  false)), true);
    // STRANGE RESULT...-1 must be converted to uint32_t before comparison
    BOOST_CHECK_EQUAL((LessEqual<int16_t,uint32_t>()(-1,  0)),     false);
    BOOST_CHECK_EQUAL((LessEqual<int16_t,int64_t >()( 0, -1)),     false);
    BOOST_CHECK_EQUAL((LessEqual<int16_t,float   >()(-1, -1.0f)),   true);

    BOOST_CHECK_EQUAL((LessEqual<int8_t,bool    >()( 0,  false)), true);
    // STRANGE RESULT...-1 must be converted to uint32_t before comparison
    BOOST_CHECK_EQUAL((LessEqual<int8_t,uint32_t>()(-1,  0)),     false);
    BOOST_CHECK_EQUAL((LessEqual<int8_t,int64_t> ()( 0, -1)),     false);
    BOOST_CHECK_EQUAL((LessEqual<int8_t,float   >()(-1, -1.f)),   true);

    BOOST_CHECK_EQUAL((LessEqual<bool,int8_t  >()(false, 0)),   true);
    BOOST_CHECK_EQUAL((LessEqual<bool,int32_t >()(false, 1)),   true);
    BOOST_CHECK_EQUAL((LessEqual<bool,uint64_t>()(true,  0UL)), false);
    BOOST_CHECK_EQUAL((LessEqual<bool,float   >()(true,  1.0f)),true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(first_same_domain_test)
{
    BOOST_CHECK_EQUAL(First<double>()(0.0, 0.0), 0.0);
    BOOST_CHECK_EQUAL(First<double>()(1.0, 0.0), 1.0);
    BOOST_CHECK_EQUAL(First<double>()(0.0, 1.0), 0.0);
    BOOST_CHECK_EQUAL(First<double>()(1.0, 1.0), 1.0);

    BOOST_CHECK_EQUAL(First<float>()(0.0f, 0.0f), 0.0f);
    BOOST_CHECK_EQUAL(First<float>()(1.0f, 0.0f), 1.0f);
    BOOST_CHECK_EQUAL(First<float>()(0.0f, 1.0f), 0.0f);
    BOOST_CHECK_EQUAL(First<float>()(1.0f, 1.0f), 1.0f);

    BOOST_CHECK_EQUAL(First<uint64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(First<uint64_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(First<uint64_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(First<uint64_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(First<uint32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(First<uint32_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(First<uint32_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(First<uint32_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(First<uint16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(First<uint16_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(First<uint16_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(First<uint16_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(First<uint8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(First<uint8_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(First<uint8_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(First<uint8_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(First<int64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(First<int64_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(First<int64_t>()(0, -1), 0);
    BOOST_CHECK_EQUAL(First<int64_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(First<int32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(First<int32_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(First<int32_t>()(0, -1), 0);
    BOOST_CHECK_EQUAL(First<int32_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(First<int16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(First<int16_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(First<int16_t>()(0, -1), 0);
    BOOST_CHECK_EQUAL(First<int16_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(First<int8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(First<int8_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(First<int8_t>()(0, -1), 0);
    BOOST_CHECK_EQUAL(First<int8_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(First<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(First<bool>()(false, true),  false);
    BOOST_CHECK_EQUAL(First<bool>()(true, false),  true);
    BOOST_CHECK_EQUAL(First<bool>()(true, true),   true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(first_different_domain_test)
{
    BOOST_CHECK_EQUAL((First<double,bool    >()(0.2, true)), 0.2);
    BOOST_CHECK_EQUAL((First<double,int32_t >()(1.2, 0)), 1.2);
    BOOST_CHECK_EQUAL((First<double,uint64_t>()(0.2, 1UL)), 0.2);
    BOOST_CHECK_EQUAL((First<double,float   >()(1.2, 1.0f)), 1.2);

    BOOST_CHECK_EQUAL((First<float,bool    >()(0.1f, false)), 0.1f);
    BOOST_CHECK_EQUAL((First<float,double  >()(1.1f, 0.2)), 1.1f);
    BOOST_CHECK_EQUAL((First<float,uint64_t>()(0.1f, 1UL)), 0.1f);
    BOOST_CHECK_EQUAL((First<float,int32_t >()(1.1f, 1)), 1.1f);

    BOOST_CHECK_EQUAL((First<uint64_t,bool    >()(0, false)), 0);
    BOOST_CHECK_EQUAL((First<uint64_t,int32_t >()(1, 0)), 1);
    BOOST_CHECK_EQUAL((First<uint64_t,uint64_t>()(0, 1)), 0);
    BOOST_CHECK_EQUAL((First<uint64_t,float   >()(1, 1.1f)), 1);

    BOOST_CHECK_EQUAL((First<uint32_t,bool    >()(0, false)), 0);
    BOOST_CHECK_EQUAL((First<uint32_t,int32_t >()(1, 0)), 1);
    BOOST_CHECK_EQUAL((First<uint32_t,uint64_t>()(0, 1)), 0);
    BOOST_CHECK_EQUAL((First<uint32_t,float   >()(1, 1)), 1);

    BOOST_CHECK_EQUAL((First<uint16_t,bool    >()(0, false)), 0);
    BOOST_CHECK_EQUAL((First<uint16_t,int32_t >()(1, 0)), 1);
    BOOST_CHECK_EQUAL((First<uint16_t,uint64_t>()(0, 1)), 0);
    BOOST_CHECK_EQUAL((First<uint16_t,float   >()(1, 1.1f)), 1);

    BOOST_CHECK_EQUAL((First<uint8_t,bool    >()(0, false)), 0);
    BOOST_CHECK_EQUAL((First<uint8_t,int32_t >()(1, 0)), 1);
    BOOST_CHECK_EQUAL((First<uint8_t,uint64_t>()(0, 1)), 0);
    BOOST_CHECK_EQUAL((First<uint8_t,float   >()(1, 1.1f)), 1);

    BOOST_CHECK_EQUAL((First<int64_t,bool    >()( 0,  false)), 0);
    BOOST_CHECK_EQUAL((First<int64_t,uint32_t>()(-1,  0)), -1);
    BOOST_CHECK_EQUAL((First<int64_t,int64_t >()( 0, -1)), 0);
    BOOST_CHECK_EQUAL((First<int64_t,float   >()(-1, -1.1f)), -1);

    BOOST_CHECK_EQUAL((First<int32_t,bool    >()( 0,  false)), 0);
    BOOST_CHECK_EQUAL((First<int32_t,uint32_t>()(-1,  0)), -1);
    BOOST_CHECK_EQUAL((First<int32_t,int64_t >()( 0, -1)), 0);
    BOOST_CHECK_EQUAL((First<int32_t,float   >()(-1, -1.1f)), -1);

    BOOST_CHECK_EQUAL((First<int16_t,bool    >()( 0,  false)), 0);
    BOOST_CHECK_EQUAL((First<int16_t,uint32_t>()(-1,  0)), -1);
    BOOST_CHECK_EQUAL((First<int16_t,int64_t >()( 0, -1)), 0);
    BOOST_CHECK_EQUAL((First<int16_t,float   >()(-1, -1.1f)), -1);

    BOOST_CHECK_EQUAL((First<int8_t,bool    >()( 0,  false)), 0);
    BOOST_CHECK_EQUAL((First<int8_t,uint32_t>()(-1,  0)),     -1);
    BOOST_CHECK_EQUAL((First<int8_t,int64_t> ()( 0, -1)),     0);
    BOOST_CHECK_EQUAL((First<int8_t,float   >()(-1, -1.f)),   -1);

    BOOST_CHECK_EQUAL((First<bool,int8_t  >()(false, 0)),   false);
    BOOST_CHECK_EQUAL((First<bool,int32_t >()(false, 1)),   false);
    BOOST_CHECK_EQUAL((First<bool,uint64_t>()(true,  0UL)), true);
    BOOST_CHECK_EQUAL((First<bool,float   >()(true,  1.1f)),true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(second_same_domain_test)
{
    BOOST_CHECK_EQUAL(Second<double>()(0.0, 0.0), 0.0);
    BOOST_CHECK_EQUAL(Second<double>()(1.0, 0.0), 0.0);
    BOOST_CHECK_EQUAL(Second<double>()(0.0, 1.0), 1.0);
    BOOST_CHECK_EQUAL(Second<double>()(1.0, 1.0), 1.0);

    BOOST_CHECK_EQUAL(Second<float>()(0.0f, 0.0f), 0.0f);
    BOOST_CHECK_EQUAL(Second<float>()(1.0f, 0.0f), 0.0f);
    BOOST_CHECK_EQUAL(Second<float>()(0.0f, 1.0f), 1.0f);
    BOOST_CHECK_EQUAL(Second<float>()(1.0f, 1.0f), 1.0f);

    BOOST_CHECK_EQUAL(Second<uint64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Second<uint64_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(Second<uint64_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(Second<uint64_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(Second<uint32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Second<uint32_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(Second<uint32_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(Second<uint32_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(Second<uint16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Second<uint16_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(Second<uint16_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(Second<uint16_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(Second<uint8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Second<uint8_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(Second<uint8_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(Second<uint8_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(Second<int64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Second<int64_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(Second<int64_t>()(0, -1), -1);
    BOOST_CHECK_EQUAL(Second<int64_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(Second<int32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Second<int32_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(Second<int32_t>()(0, -1), -1);
    BOOST_CHECK_EQUAL(Second<int32_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(Second<int16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Second<int16_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(Second<int16_t>()(0, -1), -1);
    BOOST_CHECK_EQUAL(Second<int16_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(Second<int8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Second<int8_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(Second<int8_t>()(0, -1), -1);
    BOOST_CHECK_EQUAL(Second<int8_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(Second<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(Second<bool>()(false, true),  true);
    BOOST_CHECK_EQUAL(Second<bool>()(true, false),  false);
    BOOST_CHECK_EQUAL(Second<bool>()(true, true),   true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(second_different_domain_test)
{
    BOOST_CHECK_EQUAL((Second<double,bool    >()(0.2, true)), true);
    BOOST_CHECK_EQUAL((Second<double,int32_t >()(1.2, 0)), 0);
    BOOST_CHECK_EQUAL((Second<double,uint64_t>()(0.2, 1UL)), 1UL);
    BOOST_CHECK_EQUAL((Second<double,float   >()(1.2, 1.1f)), 1.1f);

    BOOST_CHECK_EQUAL((Second<float,bool    >()(0.1f, false)), false);
    BOOST_CHECK_EQUAL((Second<float,double  >()(1.1f, 0.0)), 0.0);
    BOOST_CHECK_EQUAL((Second<float,uint64_t>()(0.1f, 1UL)), 1UL);
    BOOST_CHECK_EQUAL((Second<float,int32_t >()(1.1f, 1)), 1U);

    BOOST_CHECK_EQUAL((Second<uint64_t,bool    >()(0, false)), false);
    BOOST_CHECK_EQUAL((Second<uint64_t,int32_t >()(1, 0)), 0);
    BOOST_CHECK_EQUAL((Second<uint64_t,uint64_t>()(0, 1UL)), 1UL);
    BOOST_CHECK_EQUAL((Second<uint64_t,float   >()(1, 1.1f)), 1.1f);

    BOOST_CHECK_EQUAL((Second<uint32_t,bool    >()(0, false)), false);
    BOOST_CHECK_EQUAL((Second<uint32_t,int32_t >()(1, 0)), 0);
    BOOST_CHECK_EQUAL((Second<uint32_t,uint64_t>()(0, 1UL)), 1UL);
    BOOST_CHECK_EQUAL((Second<uint32_t,float   >()(1, 1.1f)), 1.1f);

    BOOST_CHECK_EQUAL((Second<uint16_t,bool    >()(0, false)), false);
    BOOST_CHECK_EQUAL((Second<uint16_t,int32_t >()(1, 0)), 0);
    BOOST_CHECK_EQUAL((Second<uint16_t,uint64_t>()(0, 1)), 1);
    BOOST_CHECK_EQUAL((Second<uint16_t,float   >()(1, 1.1f)), 1.1f);

    BOOST_CHECK_EQUAL((Second<uint8_t,bool    >()(0, false)), false);
    BOOST_CHECK_EQUAL((Second<uint8_t,int32_t >()(1, 0)), 0);
    BOOST_CHECK_EQUAL((Second<uint8_t,uint64_t>()(0, 1)), 1);
    BOOST_CHECK_EQUAL((Second<uint8_t,float   >()(1, 1.1f)), 1.1f);

    BOOST_CHECK_EQUAL((Second<int64_t,bool    >()( 0,  false)), false);
    BOOST_CHECK_EQUAL((Second<int64_t,uint32_t>()(-1,  0)), 0);
    BOOST_CHECK_EQUAL((Second<int64_t,int64_t >()( 0, -1)), -1);
    BOOST_CHECK_EQUAL((Second<int64_t,float   >()(-1, -1.1f)), -1.1f);

    BOOST_CHECK_EQUAL((Second<int32_t,bool    >()( 0,  false)), false);
    BOOST_CHECK_EQUAL((Second<int32_t,uint32_t>()(-1,  0)), 0);
    BOOST_CHECK_EQUAL((Second<int32_t,int64_t >()( 0, -1)), -1);
    BOOST_CHECK_EQUAL((Second<int32_t,float   >()(-1, -1.1f)), -1.1f);

    BOOST_CHECK_EQUAL((Second<int16_t,bool    >()( 0,  false)), false);
    BOOST_CHECK_EQUAL((Second<int16_t,uint32_t>()(-1,  0)), 0);
    BOOST_CHECK_EQUAL((Second<int16_t,int64_t >()( 0, -1)), -1);
    BOOST_CHECK_EQUAL((Second<int16_t,float   >()(-1, -1.1f)), -1.1f);

    BOOST_CHECK_EQUAL((Second<int8_t,bool    >()( 0,  false)), false);
    BOOST_CHECK_EQUAL((Second<int8_t,uint32_t>()(-1,  0)),     0);
    BOOST_CHECK_EQUAL((Second<int8_t,int64_t> ()( 0, -1)),    -1);
    BOOST_CHECK_EQUAL((Second<int8_t,float   >()(-1, -1.f)),  -1.f);

    BOOST_CHECK_EQUAL((Second<bool,int8_t  >()(false, 0)),   0);
    BOOST_CHECK_EQUAL((Second<bool,int32_t >()(false, 1)),   1);
    BOOST_CHECK_EQUAL((Second<bool,uint64_t>()(true,  0UL)), 0UL);
    BOOST_CHECK_EQUAL((Second<bool,float   >()(true,  1.1f)),1.1f);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(min_same_domain_test)
{
    BOOST_CHECK_EQUAL(Min<double>()(0.0, 0.0), 0.0);
    BOOST_CHECK_EQUAL(Min<double>()(1.0, 0.0), 0.0);
    BOOST_CHECK_EQUAL(Min<double>()(0.0, 1.0), 0.0);
    BOOST_CHECK_EQUAL(Min<double>()(1.0, 1.0), 1.0);

    BOOST_CHECK_EQUAL(Min<float>()(0.0f, 0.0f), 0.0f);
    BOOST_CHECK_EQUAL(Min<float>()(1.0f, 0.0f), 0.0f);
    BOOST_CHECK_EQUAL(Min<float>()(0.0f, 1.0f), 0.0f);
    BOOST_CHECK_EQUAL(Min<float>()(1.0f, 1.0f), 1.0f);

    BOOST_CHECK_EQUAL(Min<uint64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Min<uint64_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(Min<uint64_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(Min<uint64_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(Min<uint32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Min<uint32_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(Min<uint32_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(Min<uint32_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(Min<uint16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Min<uint16_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(Min<uint16_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(Min<uint16_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(Min<uint8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Min<uint8_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(Min<uint8_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(Min<uint8_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(Min<int64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Min<int64_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(Min<int64_t>()(0, -1), -1);
    BOOST_CHECK_EQUAL(Min<int64_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(Min<int32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Min<int32_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(Min<int32_t>()(0, -1), -1);
    BOOST_CHECK_EQUAL(Min<int32_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(Min<int16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Min<int16_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(Min<int16_t>()(0, -1), -1);
    BOOST_CHECK_EQUAL(Min<int16_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(Min<int8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Min<int8_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(Min<int8_t>()(0, -1), -1);
    BOOST_CHECK_EQUAL(Min<int8_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(Min<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(Min<bool>()(false, true),  false);
    BOOST_CHECK_EQUAL(Min<bool>()(true, false),  false);
    BOOST_CHECK_EQUAL(Min<bool>()(true, true),   true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(min_different_domain_test)
{
    BOOST_CHECK_EQUAL((Min<double,bool    >()(0.2, true)), 0.2);
    BOOST_CHECK_EQUAL((Min<double,int32_t >()(1.2, 0)), 0.0);
    BOOST_CHECK_EQUAL((Min<double,uint64_t>()(0.2, 1UL)), 0.2);
    BOOST_CHECK_EQUAL((Min<double,float   >()(1.2, 1.1f)), ((double)1.1f));

    BOOST_CHECK_EQUAL((Min<float,bool    >()(0.1f, false)), 0.0f);
    BOOST_CHECK_EQUAL((Min<float,double  >()(1.1f, 0.0)), 0.0f);
    BOOST_CHECK_EQUAL((Min<float,uint64_t>()(0.1f, 1UL)), 0.1f);
    BOOST_CHECK_EQUAL((Min<float,int32_t >()(1.1f, 1)), 1.0f);

    BOOST_CHECK_EQUAL((Min<uint64_t,bool    >()(0, false)), 0);
    BOOST_CHECK_EQUAL((Min<uint64_t,int32_t >()(1, 0)), 0);
    BOOST_CHECK_EQUAL((Min<uint64_t,uint64_t>()(0, 1UL)), 0);
    BOOST_CHECK_EQUAL((Min<uint64_t,float   >()(1, 1.1f)), 1);

    BOOST_CHECK_EQUAL((Min<uint32_t,bool    >()(0, false)), 0);
    BOOST_CHECK_EQUAL((Min<uint32_t,int32_t >()(1, 0)), 0);
    BOOST_CHECK_EQUAL((Min<uint32_t,uint64_t>()(0, 1UL)), 0U);
    BOOST_CHECK_EQUAL((Min<uint32_t,float   >()(1, 1.1f)), 1U);

    BOOST_CHECK_EQUAL((Min<uint16_t,bool    >()(0, false)), 0);
    BOOST_CHECK_EQUAL((Min<uint16_t,int32_t >()(1, 0)), 0);
    BOOST_CHECK_EQUAL((Min<uint16_t,uint64_t>()(0, 1)), 0);
    BOOST_CHECK_EQUAL((Min<uint16_t,float   >()(1, 1.1f)), 1);

    BOOST_CHECK_EQUAL((Min<uint8_t,bool    >()(0, false)), 0);
    BOOST_CHECK_EQUAL((Min<uint8_t,int32_t >()(1, 0)), 0);
    BOOST_CHECK_EQUAL((Min<uint8_t,uint64_t>()(0, 1)), 0);
    BOOST_CHECK_EQUAL((Min<uint8_t,float   >()(1, 1.1f)), 1);

    BOOST_CHECK_EQUAL((Min<int64_t,bool    >()( 0,  false)), 0);
    BOOST_CHECK_EQUAL((Min<int64_t,uint32_t>()(-1,  0)), -1);
    BOOST_CHECK_EQUAL((Min<int64_t,int64_t >()( 0, -1)), -1);
    BOOST_CHECK_EQUAL((Min<int64_t,float   >()(-1, -1.1f)), -1);

    BOOST_CHECK_EQUAL((Min<int32_t,bool    >()( 0,  false)), 0);
    // STRANGE RESULT...-1 must be converted to uint32_t before comparison
    BOOST_CHECK_EQUAL((Min<int32_t,uint32_t>()(-1,  0)), 0);
    BOOST_CHECK_EQUAL((Min<int32_t,int64_t >()( 0, -1)), -1);
    BOOST_CHECK_EQUAL((Min<int32_t,float   >()(-1, -1.1f)), -1);

    BOOST_CHECK_EQUAL((Min<int16_t,bool    >()( 0,  false)), 0);
    // STRANGE RESULT...-1 must be converted to uint32_t before comparison
    BOOST_CHECK_EQUAL((Min<int16_t,uint32_t>()(-1,  0)), 0);
    BOOST_CHECK_EQUAL((Min<int16_t,int64_t >()( 0, -1)), -1);
    BOOST_CHECK_EQUAL((Min<int16_t,float   >()(-1, -1.1f)), -1);

    BOOST_CHECK_EQUAL((Min<int8_t,bool    >()( 0,  false)), 0);
    // STRANGE RESULT...-1 must be converted to uint32_t before comparison
    BOOST_CHECK_EQUAL((Min<int8_t,uint32_t>()(-1,  0)),    0);
    BOOST_CHECK_EQUAL((Min<int8_t,int64_t> ()( 0, -1)),    -1);
    BOOST_CHECK_EQUAL((Min<int8_t,float   >()(-1, -1.f)),  -1);

    BOOST_CHECK_EQUAL((Min<bool,int8_t  >()(false, 0)),   false);
    BOOST_CHECK_EQUAL((Min<bool,int32_t >()(false, 1)),   false);
    BOOST_CHECK_EQUAL((Min<bool,uint64_t>()(true,  0UL)), false);
    BOOST_CHECK_EQUAL((Min<bool,float   >()(true,  1.1f)),true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(max_same_domain_test)
{
    BOOST_CHECK_EQUAL(Max<double>()(0.0, 0.0), 0.0);
    BOOST_CHECK_EQUAL(Max<double>()(1.0, 0.0), 1.0);
    BOOST_CHECK_EQUAL(Max<double>()(0.0, 1.0), 1.0);
    BOOST_CHECK_EQUAL(Max<double>()(1.0, 1.0), 1.0);

    BOOST_CHECK_EQUAL(Max<float>()(0.0f, 0.0f), 0.0f);
    BOOST_CHECK_EQUAL(Max<float>()(1.0f, 0.0f), 1.0f);
    BOOST_CHECK_EQUAL(Max<float>()(0.0f, 1.0f), 1.0f);
    BOOST_CHECK_EQUAL(Max<float>()(1.0f, 1.0f), 1.0f);

    BOOST_CHECK_EQUAL(Max<uint64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Max<uint64_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(Max<uint64_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(Max<uint64_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(Max<uint32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Max<uint32_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(Max<uint32_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(Max<uint32_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(Max<uint16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Max<uint16_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(Max<uint16_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(Max<uint16_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(Max<uint8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Max<uint8_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(Max<uint8_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(Max<uint8_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(Max<int64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Max<int64_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(Max<int64_t>()(0, -1), 0);
    BOOST_CHECK_EQUAL(Max<int64_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(Max<int32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Max<int32_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(Max<int32_t>()(0, -1), 0);
    BOOST_CHECK_EQUAL(Max<int32_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(Max<int16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Max<int16_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(Max<int16_t>()(0, -1), 0);
    BOOST_CHECK_EQUAL(Max<int16_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(Max<int8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Max<int8_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(Max<int8_t>()(0, -1), 0);
    BOOST_CHECK_EQUAL(Max<int8_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(Max<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(Max<bool>()(false, true),  true);
    BOOST_CHECK_EQUAL(Max<bool>()(true, false),  true);
    BOOST_CHECK_EQUAL(Max<bool>()(true, true),   true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(max_different_domain_test)
{
    BOOST_CHECK_EQUAL((Max<double,bool    >()(0.2, true)), 1.0);
    BOOST_CHECK_EQUAL((Max<double,int32_t >()(1.2, 0)), 1.2);
    BOOST_CHECK_EQUAL((Max<double,uint64_t>()(0.2, 1UL)), 1.0);
    BOOST_CHECK_EQUAL((Max<double,float   >()(1.2, 1.1f)), 1.2);

    BOOST_CHECK_EQUAL((Max<float,bool    >()(0.1f, false)), 0.1f);
    BOOST_CHECK_EQUAL((Max<float,double  >()(1.1f, 0.0)), 1.1f);
    BOOST_CHECK_EQUAL((Max<float,uint64_t>()(0.1f, 1UL)), 1.0f);
    BOOST_CHECK_EQUAL((Max<float,int32_t >()(1.1f, 1)), 1.1f);

    BOOST_CHECK_EQUAL((Max<uint64_t,bool    >()(0, false)), 0);
    BOOST_CHECK_EQUAL((Max<uint64_t,int32_t >()(1, 0)), 1);
    BOOST_CHECK_EQUAL((Max<uint64_t,uint64_t>()(0, 1UL)), 1);
    BOOST_CHECK_EQUAL((Max<uint64_t,float   >()(1, 1.1f)), 1);

    BOOST_CHECK_EQUAL((Max<uint32_t,bool    >()(0, false)), 0);
    BOOST_CHECK_EQUAL((Max<uint32_t,int32_t >()(1, 0)), 1);
    BOOST_CHECK_EQUAL((Max<uint32_t,uint64_t>()(0, 1UL)), 1U);
    BOOST_CHECK_EQUAL((Max<uint32_t,float   >()(1, 1.1f)), 1U);

    BOOST_CHECK_EQUAL((Max<uint16_t,bool    >()(0, false)), 0);
    BOOST_CHECK_EQUAL((Max<uint16_t,int32_t >()(1, 0)), 1);
    BOOST_CHECK_EQUAL((Max<uint16_t,uint64_t>()(0, 1)), 1);
    BOOST_CHECK_EQUAL((Max<uint16_t,float   >()(1, 1.1f)), 1);

    BOOST_CHECK_EQUAL((Max<uint8_t,bool    >()(0, false)), 0);
    BOOST_CHECK_EQUAL((Max<uint8_t,int32_t >()(1, 0)), 1);
    BOOST_CHECK_EQUAL((Max<uint8_t,uint64_t>()(0, 1)), 1);
    BOOST_CHECK_EQUAL((Max<uint8_t,float   >()(1, 1.1f)), 1);

    BOOST_CHECK_EQUAL((Max<int64_t,bool    >()( 0,  false)), 0);
    BOOST_CHECK_EQUAL((Max<int64_t,uint32_t>()(-1,  0)),  0);
    BOOST_CHECK_EQUAL((Max<int64_t,int64_t >()( 0, -1)),  0);
    BOOST_CHECK_EQUAL((Max<int64_t,float   >()(-1, -1.1f)), -1);

    BOOST_CHECK_EQUAL((Max<int32_t,bool    >()( 0,  false)), 0);
    // STRANGE RESULT...-1 must be converted to uint32_t before comparison
    BOOST_CHECK_EQUAL((Max<int32_t,uint32_t>()(-1,  0)), -1);
    BOOST_CHECK_EQUAL((Max<int32_t,int64_t >()( 0, -1)),  0);
    BOOST_CHECK_EQUAL((Max<int32_t,float   >()(-1, -1.1f)), -1);

    BOOST_CHECK_EQUAL((Max<int16_t,bool    >()( 0,  false)), 0);
    // STRANGE RESULT...-1 must be converted to uint32_t before comparison
    BOOST_CHECK_EQUAL((Max<int16_t,uint32_t>()(-1,  0)), -1);
    BOOST_CHECK_EQUAL((Max<int16_t,int64_t >()( 0, -1)), 0);
    BOOST_CHECK_EQUAL((Max<int16_t,float   >()(-1, -1.1f)), -1);

    BOOST_CHECK_EQUAL((Max<int8_t,bool    >()( 0,  false)), 0);
    // STRANGE RESULT...-1 must be converted to uint32_t before comparison
    BOOST_CHECK_EQUAL((Max<int8_t,uint32_t>()(-1,  0)),  -1);
    BOOST_CHECK_EQUAL((Max<int8_t,int64_t> ()( 0, -1)),   0);
    BOOST_CHECK_EQUAL((Max<int8_t,float   >()(-1, -1.f)),  -1);

    BOOST_CHECK_EQUAL((Max<bool,int8_t  >()(false, 0)),   false);
    BOOST_CHECK_EQUAL((Max<bool,int32_t >()(false, 1)),   true);
    BOOST_CHECK_EQUAL((Max<bool,uint64_t>()(true,  0UL)), true);
    BOOST_CHECK_EQUAL((Max<bool,float   >()(true,  1.1f)),true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(plus_same_domain_test)
{
    BOOST_CHECK_EQUAL(Plus<double>()(0.0, 0.0), 0.0);
    BOOST_CHECK_EQUAL(Plus<double>()(1.0, 0.0), 1.0);
    BOOST_CHECK_EQUAL(Plus<double>()(0.0, 1.0), 1.0);
    BOOST_CHECK_EQUAL(Plus<double>()(1.0, 1.0), 2.0);

    BOOST_CHECK_EQUAL(Plus<float>()(0.0f, 0.0f), 0.0f);
    BOOST_CHECK_EQUAL(Plus<float>()(1.0f, 0.0f), 1.0f);
    BOOST_CHECK_EQUAL(Plus<float>()(0.0f, 1.0f), 1.0f);
    BOOST_CHECK_EQUAL(Plus<float>()(1.0f, 1.0f), 2.0f);

    BOOST_CHECK_EQUAL(Plus<uint64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Plus<uint64_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(Plus<uint64_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(Plus<uint64_t>()(1, 1), 2);

    BOOST_CHECK_EQUAL(Plus<uint32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Plus<uint32_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(Plus<uint32_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(Plus<uint32_t>()(1, 1), 2);

    BOOST_CHECK_EQUAL(Plus<uint16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Plus<uint16_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(Plus<uint16_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(Plus<uint16_t>()(1, 1), 2);

    BOOST_CHECK_EQUAL(Plus<uint8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Plus<uint8_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(Plus<uint8_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(Plus<uint8_t>()(1, 1), 2);

    BOOST_CHECK_EQUAL(Plus<int64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Plus<int64_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(Plus<int64_t>()(0, -1), -1);
    BOOST_CHECK_EQUAL(Plus<int64_t>()(-1, -1), -2);

    BOOST_CHECK_EQUAL(Plus<int32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Plus<int32_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(Plus<int32_t>()(0, -1), -1);
    BOOST_CHECK_EQUAL(Plus<int32_t>()(-1, -1), -2);

    BOOST_CHECK_EQUAL(Plus<int16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Plus<int16_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(Plus<int16_t>()(0, -1), -1);
    BOOST_CHECK_EQUAL(Plus<int16_t>()(-1, -1), -2);

    BOOST_CHECK_EQUAL(Plus<int8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Plus<int8_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(Plus<int8_t>()(0, -1), -1);
    BOOST_CHECK_EQUAL(Plus<int8_t>()(-1, -1), -2);

    BOOST_CHECK_EQUAL(Plus<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(Plus<bool>()(false, true),  true);
    BOOST_CHECK_EQUAL(Plus<bool>()(true, false),  true);
    BOOST_CHECK_EQUAL(Plus<bool>()(true, true),   true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(plus_different_domain_test)
{
    BOOST_CHECK_EQUAL((Plus<double,bool    >()(0.2, true)), 1.2);
    BOOST_CHECK_EQUAL((Plus<double,int32_t >()(1.2, 0)), 1.2);
    BOOST_CHECK_EQUAL((Plus<double,uint64_t>()(0.2, 1UL)), 1.2);
    BOOST_CHECK_EQUAL((Plus<double,float   >()(1.2, 1.1f)),
                      (double)(1.2 + 1.1f));

    BOOST_CHECK_EQUAL((Plus<float,bool    >()(0.1f, false)), 0.1f);
    BOOST_CHECK_EQUAL((Plus<float,double  >()(1.1f, 0.0)), 1.1f);
    BOOST_CHECK_EQUAL((Plus<float,uint64_t>()(0.1f, 1UL)), 1.1f);
    BOOST_CHECK_EQUAL((Plus<float,int32_t >()(1.1f, 1)), 2.1f);

    BOOST_CHECK_EQUAL((Plus<uint64_t,bool    >()(0, false)), 0);
    BOOST_CHECK_EQUAL((Plus<uint64_t,int32_t >()(1, -1)), 0);
    BOOST_CHECK_EQUAL((Plus<uint64_t,uint64_t>()(0, 1UL)), 1);
    BOOST_CHECK_EQUAL((Plus<uint64_t,float   >()(1, -1.1f)), 0);

    BOOST_CHECK_EQUAL((Plus<uint32_t,bool    >()(0, false)), 0);
    BOOST_CHECK_EQUAL((Plus<uint32_t,int32_t >()(1, -1)), 0U);
    BOOST_CHECK_EQUAL((Plus<uint32_t,uint64_t>()(0, 1UL)), 1U);
    BOOST_CHECK_EQUAL((Plus<uint32_t,float   >()(1, -1.1f)), 0U);

    BOOST_CHECK_EQUAL((Plus<uint16_t,bool    >()(0, false)), 0);
    BOOST_CHECK_EQUAL((Plus<uint16_t,int32_t >()(1, -1)), 0);
    BOOST_CHECK_EQUAL((Plus<uint16_t,uint64_t>()(0, 1)), 1);
    BOOST_CHECK_EQUAL((Plus<uint16_t,float   >()(1, -1.1f)), 0);

    BOOST_CHECK_EQUAL((Plus<uint8_t,bool    >()(0, false)), 0);
    BOOST_CHECK_EQUAL((Plus<uint8_t,int32_t >()(1, -1)), 0);
    BOOST_CHECK_EQUAL((Plus<uint8_t,uint64_t>()(0, 1)), 1);
    BOOST_CHECK_EQUAL((Plus<uint8_t,float   >()(1, -1.1f)), 0);

    BOOST_CHECK_EQUAL((Plus<int64_t,bool    >()( 0,  false)), 0);
    BOOST_CHECK_EQUAL((Plus<int64_t,uint32_t>()(-1,  0)), -1);
    BOOST_CHECK_EQUAL((Plus<int64_t,int64_t >()( 0, -1)), -1);
    BOOST_CHECK_EQUAL((Plus<int64_t,float   >()(-1, -1.1f)), -2);

    BOOST_CHECK_EQUAL((Plus<int32_t,bool    >()( 0,  false)), 0);
    // STRANGE RESULT...-1 must be converted to uint32_t before comparison
    BOOST_CHECK_EQUAL((Plus<int32_t,uint32_t>()(-1,  0)), -1);
    BOOST_CHECK_EQUAL((Plus<int32_t,int64_t >()( 0, -1)),  -1);
    BOOST_CHECK_EQUAL((Plus<int32_t,float   >()(-1, -1.1f)), -2);

    BOOST_CHECK_EQUAL((Plus<int16_t,bool    >()( 0,  false)), 0);
    // STRANGE RESULT...-1 must be converted to uint32_t before comparison
    BOOST_CHECK_EQUAL((Plus<int16_t,uint32_t>()(-1,  0)), -1);
    BOOST_CHECK_EQUAL((Plus<int16_t,int64_t >()( 0, -1)), -1);
    BOOST_CHECK_EQUAL((Plus<int16_t,float   >()(-1, -1.1f)), -2);

    BOOST_CHECK_EQUAL((Plus<int8_t,bool    >()( 0,  false)), 0);
    // STRANGE RESULT...-1 must be converted to uint32_t before comparison
    BOOST_CHECK_EQUAL((Plus<int8_t,uint32_t>()(-1,  0)),  -1);
    BOOST_CHECK_EQUAL((Plus<int8_t,int64_t> ()( 0, -1)),   -1);
    BOOST_CHECK_EQUAL((Plus<int8_t,float   >()(-1, -1.f)),  -2);

    BOOST_CHECK_EQUAL((Plus<bool,int8_t  >()(false, 0)),   false);
    BOOST_CHECK_EQUAL((Plus<bool,int32_t >()(false, 1)),   true);
    BOOST_CHECK_EQUAL((Plus<bool,uint64_t>()(true,  0UL)), true);
    BOOST_CHECK_EQUAL((Plus<bool,float   >()(true,  1.1f)),true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(minus_same_domain_test)
{
    BOOST_CHECK_EQUAL(Minus<double>()(0.0, 0.0), 0.0);
    BOOST_CHECK_EQUAL(Minus<double>()(1.0, 0.0), 1.0);
    BOOST_CHECK_EQUAL(Minus<double>()(2.0, 1.0), 1.0);
    BOOST_CHECK_EQUAL(Minus<double>()(1.0, 1.0), 0.0);

    BOOST_CHECK_EQUAL(Minus<float>()(0.0f, 0.0f), 0.0f);
    BOOST_CHECK_EQUAL(Minus<float>()(1.0f, 0.0f), 1.0f);
    BOOST_CHECK_EQUAL(Minus<float>()(2.0f, 1.0f), 1.0f);
    BOOST_CHECK_EQUAL(Minus<float>()(1.0f, 1.0f), 0.0f);

    BOOST_CHECK_EQUAL(Minus<uint64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Minus<uint64_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(Minus<uint64_t>()(2, 1), 1);
    BOOST_CHECK_EQUAL(Minus<uint64_t>()(1, 1), 0);

    BOOST_CHECK_EQUAL(Minus<uint32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Minus<uint32_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(Minus<uint32_t>()(2, 1), 1);
    BOOST_CHECK_EQUAL(Minus<uint32_t>()(1, 1), 0);

    BOOST_CHECK_EQUAL(Minus<uint16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Minus<uint16_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(Minus<uint16_t>()(2, 1), 1);
    BOOST_CHECK_EQUAL(Minus<uint16_t>()(1, 1), 0);

    BOOST_CHECK_EQUAL(Minus<uint8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Minus<uint8_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(Minus<uint8_t>()(2, 1), 1);
    BOOST_CHECK_EQUAL(Minus<uint8_t>()(1, 1), 0);

    BOOST_CHECK_EQUAL(Minus<int64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Minus<int64_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(Minus<int64_t>()(-2, -1), -1);
    BOOST_CHECK_EQUAL(Minus<int64_t>()(-1, -1), 0);

    BOOST_CHECK_EQUAL(Minus<int32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Minus<int32_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(Minus<int32_t>()(-2, -1), -1);
    BOOST_CHECK_EQUAL(Minus<int32_t>()(-1, -1), 0);

    BOOST_CHECK_EQUAL(Minus<int16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Minus<int16_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(Minus<int16_t>()(-2, -1), -1);
    BOOST_CHECK_EQUAL(Minus<int16_t>()(-1, -1), 0);

    BOOST_CHECK_EQUAL(Minus<int8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Minus<int8_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(Minus<int8_t>()(-2, -1), -1);
    BOOST_CHECK_EQUAL(Minus<int8_t>()(-1, -1), 0);

    BOOST_CHECK_EQUAL(Minus<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(Minus<bool>()(false, true),  true);
    BOOST_CHECK_EQUAL(Minus<bool>()(true, false),  true);
    BOOST_CHECK_EQUAL(Minus<bool>()(true, true),   false);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(minus_different_domain_test)
{
    BOOST_CHECK_EQUAL((Minus<double,bool    >()(0.2, true)), -0.8);
    BOOST_CHECK_EQUAL((Minus<double,int32_t >()(1.2, 0)), 1.2);
    BOOST_CHECK_EQUAL((Minus<double,uint64_t>()(0.2, 1UL)), -0.8);
    BOOST_CHECK_EQUAL((Minus<double,float   >()(1.2, 1.1f)),
                      (double)(1.2 - 1.1f));

    BOOST_CHECK_EQUAL((Minus<float,bool    >()(0.1f, false)), 0.1f);
    BOOST_CHECK_EQUAL((Minus<float,double  >()(1.1f, 0.0)),   1.1f);
    BOOST_CHECK_EQUAL((Minus<float,uint64_t>()(0.1f, 1UL)),  -0.9f);
    BOOST_CHECK_EQUAL((Minus<float,int32_t >()(1.1f, 1)), float(1.1f - 1));

    BOOST_CHECK_EQUAL((Minus<uint64_t,bool    >()(0, false)), 0);
    BOOST_CHECK_EQUAL((Minus<uint64_t,int32_t >()(1, -1)), 2);
    BOOST_CHECK_EQUAL((Minus<uint64_t,uint64_t>()(0, 1UL)), -1);
    BOOST_CHECK_EQUAL((Minus<uint64_t,float   >()(1, -1.1f)), 2);

    BOOST_CHECK_EQUAL((Minus<uint32_t,bool    >()(0, false)), 0);
    BOOST_CHECK_EQUAL((Minus<uint32_t,int32_t >()(1, -1)), 2U);
    BOOST_CHECK_EQUAL((Minus<uint32_t,uint64_t>()(0, 1UL)), -1U);
    BOOST_CHECK_EQUAL((Minus<uint32_t,float   >()(1, -1.1f)), 2U);

    BOOST_CHECK_EQUAL((Minus<uint16_t,bool    >()(0, false)), 0);
    BOOST_CHECK_EQUAL((Minus<uint16_t,int32_t >()(1, -1)), 2);
    BOOST_CHECK_EQUAL((Minus<uint16_t,uint64_t>()(0, 1)), (uint16_t)-1);
    BOOST_CHECK_EQUAL((Minus<uint16_t,float   >()(1, -1.1f)), 2);

    BOOST_CHECK_EQUAL((Minus<uint8_t,bool    >()(0, false)), 0);
    BOOST_CHECK_EQUAL((Minus<uint8_t,int32_t >()(1, -1)), 2);
    BOOST_CHECK_EQUAL((Minus<uint8_t,uint64_t>()(0, 1)), (uint8_t)-1);
    BOOST_CHECK_EQUAL((Minus<uint8_t,float   >()(1, -1.1f)), 2);

    BOOST_CHECK_EQUAL((Minus<int64_t,bool    >()( 0,  false)), 0);
    BOOST_CHECK_EQUAL((Minus<int64_t,uint32_t>()(-1,  0)), -1);
    BOOST_CHECK_EQUAL((Minus<int64_t,int64_t >()( 0, -1)), 1);
    BOOST_CHECK_EQUAL((Minus<int64_t,float   >()(-1, -1.1f)), 0);

    BOOST_CHECK_EQUAL((Minus<int32_t,bool    >()( 0,  false)), 0);
    BOOST_CHECK_EQUAL((Minus<int32_t,uint32_t>()(-1,  0)), -1);
    BOOST_CHECK_EQUAL((Minus<int32_t,int64_t >()( 0, -1)),  1);
    BOOST_CHECK_EQUAL((Minus<int32_t,float   >()(-1, -1.1f)), 0);

    BOOST_CHECK_EQUAL((Minus<int16_t,bool    >()( 0,  false)), 0);
    BOOST_CHECK_EQUAL((Minus<int16_t,uint32_t>()(-1,  0)), -1);
    BOOST_CHECK_EQUAL((Minus<int16_t,int64_t >()( 0, -1)), 1);
    BOOST_CHECK_EQUAL((Minus<int16_t,float   >()(-1, -1.1f)), 0);

    BOOST_CHECK_EQUAL((Minus<int8_t,bool    >()( 0,  false)), 0);
    BOOST_CHECK_EQUAL((Minus<int8_t,uint32_t>()(-1,  0)),  -1);
    BOOST_CHECK_EQUAL((Minus<int8_t,int64_t> ()( 0, -1)),   1);
    BOOST_CHECK_EQUAL((Minus<int8_t,float   >()(-1, -1.f)), 0);

    BOOST_CHECK_EQUAL((Minus<bool,int8_t  >()(false, 0)),   false);
    BOOST_CHECK_EQUAL((Minus<bool,int32_t >()(false, 1)),   true);
    BOOST_CHECK_EQUAL((Minus<bool,uint64_t>()(true,  0UL)), true);
    BOOST_CHECK_EQUAL((Minus<bool,float   >()(true,  1.1f)),true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(times_same_domain_test)
{
    BOOST_CHECK_EQUAL(Times<double>()(0.0, 0.0), 0.0);
    BOOST_CHECK_EQUAL(Times<double>()(1.0, 0.0), 0.0);
    BOOST_CHECK_EQUAL(Times<double>()(2.0, 1.0), 2.0);
    BOOST_CHECK_EQUAL(Times<double>()(1.0, 1.0), 1.0);

    BOOST_CHECK_EQUAL(Times<float>()(0.0f, 0.0f), 0.0f);
    BOOST_CHECK_EQUAL(Times<float>()(1.0f, 0.0f), 0.0f);
    BOOST_CHECK_EQUAL(Times<float>()(2.0f, 1.0f), 2.0f);
    BOOST_CHECK_EQUAL(Times<float>()(1.0f, 1.0f), 1.0f);

    BOOST_CHECK_EQUAL(Times<uint64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Times<uint64_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(Times<uint64_t>()(2, 1), 2);
    BOOST_CHECK_EQUAL(Times<uint64_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(Times<uint32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Times<uint32_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(Times<uint32_t>()(2, 1), 2);
    BOOST_CHECK_EQUAL(Times<uint32_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(Times<uint16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Times<uint16_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(Times<uint16_t>()(2, 1), 2);
    BOOST_CHECK_EQUAL(Times<uint16_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(Times<uint8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Times<uint8_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(Times<uint8_t>()(2, 1), 2);
    BOOST_CHECK_EQUAL(Times<uint8_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(Times<int64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Times<int64_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(Times<int64_t>()(-2, 1), -2);
    BOOST_CHECK_EQUAL(Times<int64_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(Times<int32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Times<int32_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(Times<int32_t>()(-2, 1), -2);
    BOOST_CHECK_EQUAL(Times<int32_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(Times<int16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Times<int16_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(Times<int16_t>()(-2, 1), -2);
    BOOST_CHECK_EQUAL(Times<int16_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(Times<int8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(Times<int8_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(Times<int8_t>()(-2, 1), -2);
    BOOST_CHECK_EQUAL(Times<int8_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(Times<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(Times<bool>()(false, true),  false);
    BOOST_CHECK_EQUAL(Times<bool>()(true, false),  false);
    BOOST_CHECK_EQUAL(Times<bool>()(true, true),   true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(times_different_domain_test)
{
    BOOST_CHECK_EQUAL((Times<double,bool    >()(0.1,  false)), 0.);
    BOOST_CHECK_EQUAL((Times<double,int32_t >()(1.2, 0)), 0.);
    BOOST_CHECK_EQUAL((Times<double,uint64_t>()(0.2, 1UL)), 0.2);
    BOOST_CHECK_EQUAL((Times<double,float   >()(1.2, 1.1f)),
                      (double)(1.2*1.1f));

    BOOST_CHECK_EQUAL((Times<float,bool    >()(0.1f, false)), 0.f);
    BOOST_CHECK_EQUAL((Times<float,double  >()(1.1f, 0.0)),   0.f);
    BOOST_CHECK_EQUAL((Times<float,uint64_t>()(0.1f, 1UL)),   0.1f);
    BOOST_CHECK_EQUAL((Times<float,int32_t >()(1.1f, 1)),     1.1f);

    BOOST_CHECK_EQUAL((Times<uint64_t,bool    >()(0, false)), 0);
    BOOST_CHECK_EQUAL((Times<uint64_t,int32_t >()(1, -1)), (uint64_t)-1L);
    BOOST_CHECK_EQUAL((Times<uint64_t,uint64_t>()(0, 1UL)),   0);
    float    opfl = -1.1f;
    uint64_t op64 = 1;
    uint64_t ans64 = op64*opfl;  // debug and opt builds are different.
    BOOST_CHECK_EQUAL((Times<uint64_t,float   >()(1, -1.1f)), ans64);

    BOOST_CHECK_EQUAL((Times<uint32_t,bool    >()(0, false)), 0);
    BOOST_CHECK_EQUAL((Times<uint32_t,int32_t >()(1, -1)), (uint32_t)-1);
    BOOST_CHECK_EQUAL((Times<uint32_t,uint64_t>()(0, 1UL)),   0U);
    uint32_t op32 = 1;
    uint32_t ans32 = op32*opfl;  // debug and opt builds are different.
    BOOST_CHECK_EQUAL((Times<uint32_t,float   >()(1, -1.1f)), ans32);

    BOOST_CHECK_EQUAL((Times<uint16_t,bool    >()(0, false)), 0);
    BOOST_CHECK_EQUAL((Times<uint16_t,int32_t >()(1, -1)), (uint16_t)-1);
    BOOST_CHECK_EQUAL((Times<uint16_t,uint64_t>()(0, 1)), 0);
    uint16_t op16 = 1;
    uint16_t ans16 = op16*opfl;  // debug and opt builds are different.
    BOOST_CHECK_EQUAL((Times<uint16_t,float   >()(1, -1.1f)), ans16);

    BOOST_CHECK_EQUAL((Times<uint8_t,bool    >()(0, false)), 0);
    BOOST_CHECK_EQUAL((Times<uint8_t,int32_t >()(1, -1)), (uint8_t)-1);
    BOOST_CHECK_EQUAL((Times<uint8_t,uint64_t>()(0, 1)), 0);
    uint8_t op8 = 1;
    uint8_t ans8 = op8*opfl;  // debug and opt builds are different.
    BOOST_CHECK_EQUAL((Times<uint8_t,float   >()(1, -1.1f)), ans8);

    BOOST_CHECK_EQUAL((Times<int64_t,bool    >()( 0,  false)), 0);
    BOOST_CHECK_EQUAL((Times<int64_t,uint32_t>()(-1,  0)), 0);
    BOOST_CHECK_EQUAL((Times<int64_t,uint64_t >()( 0, -1)), 0);
    BOOST_CHECK_EQUAL((Times<int64_t,float   >()(-1, -1.1f)), 1);

    BOOST_CHECK_EQUAL((Times<int32_t,bool    >()( 0,  false)), 0);
    BOOST_CHECK_EQUAL((Times<int32_t,uint32_t>()(-1,  0)), 0);
    BOOST_CHECK_EQUAL((Times<int32_t,int64_t >()( 1, -1)), -1);
    BOOST_CHECK_EQUAL((Times<int32_t,float   >()(-1, -1.1f)), 1);

    BOOST_CHECK_EQUAL((Times<int16_t,bool    >()( 0,  false)), 0);
    BOOST_CHECK_EQUAL((Times<int16_t,uint32_t>()(-1,  0)), 0);
    BOOST_CHECK_EQUAL((Times<int16_t,int64_t >()( 0, -1)), 0);
    BOOST_CHECK_EQUAL((Times<int16_t,float   >()(-1, -1.1f)), 1);

    BOOST_CHECK_EQUAL((Times<int8_t,bool    >()( 0,  false)), 0);
    BOOST_CHECK_EQUAL((Times<int8_t,uint32_t>()(-1,  0)),  0);
    BOOST_CHECK_EQUAL((Times<int8_t,int64_t> ()( 0, -1)),   0);
    BOOST_CHECK_EQUAL((Times<int8_t,float   >()(-1, -1.f)), 1);

    BOOST_CHECK_EQUAL((Times<bool,int8_t  >()(false, 0)),   false);
    BOOST_CHECK_EQUAL((Times<bool,int32_t >()(false, 1)),   false);
    BOOST_CHECK_EQUAL((Times<bool,uint64_t>()(true,  0UL)), false);
    BOOST_CHECK_EQUAL((Times<bool,float   >()(true,  1.1f)),true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(div_same_domain_test)
{
    BOOST_CHECK_EQUAL(Div<double>()(0.0, 1.0), 0.0);
    BOOST_CHECK_EQUAL(Div<double>()(1.0, 2.0), 0.5);
    BOOST_CHECK_EQUAL(Div<double>()(2.0, 1.0), 2.0);
    BOOST_CHECK_EQUAL(Div<double>()(1.0, 1.0), 1.0);

    BOOST_CHECK_EQUAL(Div<float>()(0.0f, 1.0f), 0.0f);
    BOOST_CHECK_EQUAL(Div<float>()(1.0f, 2.0f), 0.5f);
    BOOST_CHECK_EQUAL(Div<float>()(2.0f, 1.0f), 2.0f);
    BOOST_CHECK_EQUAL(Div<float>()(1.0f, 1.0f), 1.0f);

    BOOST_CHECK_EQUAL(Div<uint64_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(Div<uint64_t>()(1, 2), 0);
    BOOST_CHECK_EQUAL(Div<uint64_t>()(2, 1), 2);
    BOOST_CHECK_EQUAL(Div<uint64_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(Div<uint32_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(Div<uint32_t>()(1, 2), 0);
    BOOST_CHECK_EQUAL(Div<uint32_t>()(2, 1), 2);
    BOOST_CHECK_EQUAL(Div<uint32_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(Div<uint16_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(Div<uint16_t>()(1, 2), 0);
    BOOST_CHECK_EQUAL(Div<uint16_t>()(2, 1), 2);
    BOOST_CHECK_EQUAL(Div<uint16_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(Div<uint8_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(Div<uint8_t>()(1, 2), 0);
    BOOST_CHECK_EQUAL(Div<uint8_t>()(2, 1), 2);
    BOOST_CHECK_EQUAL(Div<uint8_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(Div<int64_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(Div<int64_t>()(-1, 2), 0);
    BOOST_CHECK_EQUAL(Div<int64_t>()(-2, 1), -2);
    BOOST_CHECK_EQUAL(Div<int64_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(Div<int32_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(Div<int32_t>()(-1, 2), 0);
    BOOST_CHECK_EQUAL(Div<int32_t>()(-2, 1), -2);
    BOOST_CHECK_EQUAL(Div<int32_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(Div<int16_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(Div<int16_t>()(-1, 2), 0);
    BOOST_CHECK_EQUAL(Div<int16_t>()(-2, 1), -2);
    BOOST_CHECK_EQUAL(Div<int16_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(Div<int8_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(Div<int8_t>()(-1, 2), 0);
    BOOST_CHECK_EQUAL(Div<int8_t>()(-2, 1), -2);
    BOOST_CHECK_EQUAL(Div<int8_t>()(-1, -1), 1);

    //BOOST_CHECK_EQUAL(Div<bool>()(false, false), ?);
    BOOST_CHECK_EQUAL(Div<bool>()(false, true),  false);
    //BOOST_CHECK_EQUAL(Div<bool>()(true, false),  ?);
    BOOST_CHECK_EQUAL(Div<bool>()(true, true),   true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(div_different_domain_test)
{
    BOOST_CHECK_EQUAL((Div<double,bool    >()(0.1, true)), 0.1);
    BOOST_CHECK_EQUAL((Div<double,int32_t >()(1.2, -2)),  -0.6);
    BOOST_CHECK_EQUAL((Div<double,uint64_t>()(0.2, 1UL)),  0.2);
    BOOST_CHECK_EQUAL((Div<double,float   >()(1.2, -1.f)),-1.2);

    BOOST_CHECK_EQUAL((Div<float,bool    >()(0.1f, true)),  0.1f);
    BOOST_CHECK_EQUAL((Div<float,double  >()(1.1f, -2.0)), -0.55f);
    BOOST_CHECK_EQUAL((Div<float,uint64_t>()(0.1f, 1UL)),   0.1f);
    BOOST_CHECK_EQUAL((Div<float,int32_t >()(1.1f, -1)),   -1.1f);

    BOOST_CHECK_EQUAL((Div<uint64_t,bool    >()(0, true)),  0);
    BOOST_CHECK_EQUAL((Div<uint64_t,int32_t >()(1, -2)),    0);
    BOOST_CHECK_EQUAL((Div<uint64_t,uint64_t>()(0, 1UL)),   0);
    BOOST_CHECK_EQUAL((Div<uint64_t,float   >()(1, -1.1f)), 0);

    BOOST_CHECK_EQUAL((Div<uint32_t,bool    >()(0, true)),  0);
    BOOST_CHECK_EQUAL((Div<uint32_t,int32_t >()(1, -2)),    0);
    BOOST_CHECK_EQUAL((Div<uint32_t,uint64_t>()(0, 1UL)),   0U);
    BOOST_CHECK_EQUAL((Div<uint32_t,float   >()(1, -1.1f)), 0);

    BOOST_CHECK_EQUAL((Div<uint16_t,bool    >()(0, true)),  0);
    BOOST_CHECK_EQUAL((Div<uint16_t,int32_t >()(1, -2)),    0);
    BOOST_CHECK_EQUAL((Div<uint16_t,uint64_t>()(0, 1)),     0);
    BOOST_CHECK_EQUAL((Div<uint16_t,float   >()(1, -1.1f)), 0);

    BOOST_CHECK_EQUAL((Div<uint8_t,bool    >()(0, true)),  0);
    BOOST_CHECK_EQUAL((Div<uint8_t,int32_t >()(1, -2)),    0);
    BOOST_CHECK_EQUAL((Div<uint8_t,uint64_t>()(0, 1)),     0);
    BOOST_CHECK_EQUAL((Div<uint8_t,float   >()(1, -1.1f)), 0);

    BOOST_CHECK_EQUAL((Div<int64_t,bool    >()( 0,  true)), 0);
    BOOST_CHECK_EQUAL((Div<int64_t,uint32_t>()(-1,  2)),    0);
    BOOST_CHECK_EQUAL((Div<int64_t,uint64_t >()( 0, 1)),    0);
    BOOST_CHECK_EQUAL((Div<int64_t,float   >()(-1, -1.1f)), 0);

    BOOST_CHECK_EQUAL((Div<int32_t,bool    >()( 0,  true)), 0);
    BOOST_CHECK_EQUAL((Div<int32_t,uint32_t>()(-1,  2)),     ((uint32_t)-1)/2);
    BOOST_CHECK_EQUAL((Div<int32_t,int64_t >()( 1, -1)),   -1);
    BOOST_CHECK_EQUAL((Div<int32_t,float   >()(-1, -1.1f)), 0);

    BOOST_CHECK_EQUAL((Div<int16_t,bool    >()( 0,  true)), 0);
    BOOST_CHECK_EQUAL((Div<int16_t,uint32_t>()(-1,  2)),    -1);
    BOOST_CHECK_EQUAL((Div<int16_t,int64_t >()( 0, -1)),    0);
    BOOST_CHECK_EQUAL((Div<int16_t,float   >()(-1, -1.1f)), 0);

    BOOST_CHECK_EQUAL((Div<int8_t,bool    >()( 0,  true)), 0);
    BOOST_CHECK_EQUAL((Div<int8_t,uint32_t>()(-1,  2)),   -1);
    BOOST_CHECK_EQUAL((Div<int8_t,int64_t> ()( 0, -1)),    0);
    BOOST_CHECK_EQUAL((Div<int8_t,float   >()(-1, -1.f)),  1);

    BOOST_CHECK_EQUAL((Div<bool,int8_t  >()(true,  1)),   true);
    BOOST_CHECK_EQUAL((Div<bool,int32_t >()(false,-2)),   false);
    BOOST_CHECK_EQUAL((Div<bool,uint64_t>()(true,  1UL)), true);
    BOOST_CHECK_EQUAL((Div<bool,float   >()(true, -1.1f)),true);
}

BOOST_AUTO_TEST_SUITE_END()
