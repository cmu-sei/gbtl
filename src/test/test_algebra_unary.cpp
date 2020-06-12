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
#define BOOST_TEST_MODULE algebra_unary_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

//****************************************************************************
// Summary
//****************************************************************************
BOOST_AUTO_TEST_CASE(misc_math_tests)
{
    /// @todo Boost Test Framework does not handle commas in template
    ///       parameter lists without extra parentheses
    BOOST_CHECK_EQUAL((Identity<float, int>()(2.5)), 2);

    BOOST_CHECK_EQUAL(Abs<float>()(-2.5), 2.5);
    BOOST_CHECK_EQUAL(Abs<int>()(-2), 2);
    BOOST_CHECK_EQUAL(AdditiveInverse<double>()(-10), 10);
    BOOST_CHECK_EQUAL(AdditiveInverse<double>()(500), -500);
    BOOST_CHECK_EQUAL(MultiplicativeInverse<double>()(5), 1./5);
    BOOST_CHECK_EQUAL(LogicalNot<int>()(1), false);
    BOOST_CHECK_EQUAL(BitwiseNot<unsigned int>()(1), 0xfffffffe);
    //BOOST_CHECK_EQUAL(BitwiseNot<float>()(1.0), 0xfffffffe);  // does not compile

    BOOST_CHECK_EQUAL(LogicalOr<bool>()(true, false), true);
    BOOST_CHECK_EQUAL(LogicalAnd<bool>()(false, true), false);
    BOOST_CHECK_EQUAL(LogicalXor<bool>()(false, true), true);
    BOOST_CHECK_EQUAL(LogicalXnor<bool>()(false, true), false);

    BOOST_CHECK_EQUAL(BitwiseOr<unsigned short>()(0x00ff, 0x0f0f), 0x0fff);
    BOOST_CHECK_EQUAL(BitwiseAnd<unsigned short>()(0x00ff, 0x0f0f), 0x000f);
    BOOST_CHECK_EQUAL(BitwiseXor<unsigned short>()(0x00ff, 0x0f0f), 0x0ff0);
    BOOST_CHECK_EQUAL(BitwiseXnor<unsigned short>()(0x00ff, 0x0f0f), 0xf00f);
}

//****************************************************************************
// Test Unary Operators
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(Identity_test)
{
    int8_t  i8  = 3;
    int16_t i16 = -400;
    int32_t i32 = -1001000;
    int64_t i64 = 5123123123;
    BOOST_CHECK_EQUAL(Identity<int8_t>()(i8),  i8);
    BOOST_CHECK_EQUAL(Identity<int16_t>()(i8),  i8);
    BOOST_CHECK_EQUAL(Identity<int32_t>()(i8),  i8);
    BOOST_CHECK_EQUAL(Identity<int64_t>()(i8),  i8);

    BOOST_CHECK_EQUAL(Identity<int16_t>()(i16), i16);
    BOOST_CHECK_EQUAL(Identity<int32_t>()(i32), i32);
    BOOST_CHECK_EQUAL(Identity<int64_t>()(i64), i64);

    uint8_t  ui8  = 3;
    uint16_t ui16 = 400;
    uint32_t ui32 = 1001000;
    uint64_t ui64 = 5123123123;
    BOOST_CHECK_EQUAL(Identity<uint8_t>()(ui8),  ui8);
    BOOST_CHECK_EQUAL(Identity<uint16_t>()(ui16), ui16);
    BOOST_CHECK_EQUAL(Identity<uint32_t>()(ui32), ui32);
    BOOST_CHECK_EQUAL(Identity<uint64_t>()(ui64), ui64);

    float   f32 = 3.14159;
    double  f64 = 2.718832654;
    BOOST_CHECK_EQUAL(Identity<float>()(f32), f32);
    BOOST_CHECK_EQUAL(Identity<double>()(f64), f64);

    BOOST_CHECK_EQUAL(Identity<uint32_t>()(f32), 3U);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(identity_same_domain_test)
{
    BOOST_CHECK_EQUAL(Identity<double>()(2.2), 2.2);
    BOOST_CHECK_EQUAL(Identity<float>()(2.2f), 2.2f);

    BOOST_CHECK_EQUAL(Identity<uint64_t>()(2UL), 2UL);
    BOOST_CHECK_EQUAL(Identity<uint32_t>()(2U), 2U);
    BOOST_CHECK_EQUAL(Identity<uint16_t>()(2), 2);
    BOOST_CHECK_EQUAL(Identity<uint8_t>()(2), 2);

    BOOST_CHECK_EQUAL(Identity<int64_t>()(-2L), -2L);
    BOOST_CHECK_EQUAL(Identity<int32_t>()(-2), -2);
    BOOST_CHECK_EQUAL(Identity<int16_t>()(-2), -2);
    BOOST_CHECK_EQUAL(Identity<int8_t>()(-2), -2);

    BOOST_CHECK_EQUAL(Identity<bool>()(false), false);
    BOOST_CHECK_EQUAL(Identity<bool>()(true), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(identity_different_domain_test)
{
    /// @todo this is an incomplete set of pairs and does not cover the
    /// corner cases at the numeric limits that actually vary by platform
    BOOST_CHECK_EQUAL((Identity<double,float>()(2.2)), 2.2f);
    BOOST_CHECK_EQUAL((Identity<double,uint64_t>()(2.2)), 2UL);
    BOOST_CHECK_EQUAL((Identity<double,int64_t>()(-2.2)), -2UL);
    BOOST_CHECK_EQUAL((Identity<double,bool>()(2.2)), true);
    BOOST_CHECK_EQUAL((Identity<double,bool>()(0.0)), false);

    BOOST_CHECK_EQUAL((Identity<float,double>()(2.2f)), static_cast<double>(2.2f));
    BOOST_CHECK_EQUAL((Identity<float,uint32_t>()(2.2f)), 2U);
    BOOST_CHECK_EQUAL((Identity<float,int32_t>()(-2.2f)), -2);
    BOOST_CHECK_EQUAL((Identity<float,bool>()(2.2f)), true);
    BOOST_CHECK_EQUAL((Identity<float,bool>()(0.0f)), false);

    BOOST_CHECK_EQUAL((Identity<uint64_t, double>()(2UL)), 2.0);
    BOOST_CHECK_EQUAL((Identity<uint64_t, float>()(2UL)), 2.0f);
    BOOST_CHECK_EQUAL((Identity<uint64_t, int64_t>()(2UL)), 2UL);
    BOOST_CHECK_EQUAL((Identity<uint64_t,uint32_t>()(2UL)), 2UL);
    BOOST_CHECK_EQUAL((Identity<uint64_t,bool>()(2UL)), true);
    BOOST_CHECK_EQUAL((Identity<uint64_t,bool>()(0UL)), false);

    BOOST_CHECK_EQUAL((Identity<uint32_t, double>()(2U)), 2.0);
    BOOST_CHECK_EQUAL((Identity<uint32_t, float>()(2U)), 2.0f);
    BOOST_CHECK_EQUAL((Identity<uint32_t,uint64_t>()(2U)), 2UL);
    BOOST_CHECK_EQUAL((Identity<uint32_t, int64_t>()(2U)), 2L);
    BOOST_CHECK_EQUAL((Identity<uint32_t, int32_t>()(2U)), 2U);
    BOOST_CHECK_EQUAL((Identity<uint32_t,uint16_t>()(2U)), 2U);
    BOOST_CHECK_EQUAL((Identity<uint32_t, int16_t>()(2U)), 2);
    BOOST_CHECK_EQUAL((Identity<uint32_t,bool>()(2U)), true);
    BOOST_CHECK_EQUAL((Identity<uint32_t,bool>()(0U)), false);

    BOOST_CHECK_EQUAL((Identity<uint16_t, double>()(2U)), 2.0);
    BOOST_CHECK_EQUAL((Identity<uint16_t, float>()(2U)), 2.0f);
    BOOST_CHECK_EQUAL((Identity<uint16_t,uint32_t>()(2U)), 2U);
    BOOST_CHECK_EQUAL((Identity<uint16_t, int32_t>()(2U)), 2);
    BOOST_CHECK_EQUAL((Identity<uint16_t, int16_t>()(2)), 2);
    BOOST_CHECK_EQUAL((Identity<uint16_t,uint8_t>()(2)), 2);
    BOOST_CHECK_EQUAL((Identity<uint16_t, int8_t>()(2)), 2);
    BOOST_CHECK_EQUAL((Identity<uint16_t,bool>()(2U)), true);
    BOOST_CHECK_EQUAL((Identity<uint16_t,bool>()(0U)), false);

    BOOST_CHECK_EQUAL((Identity<uint8_t,double>()(2U)), 2.0);
    BOOST_CHECK_EQUAL((Identity<uint8_t,float>()(2U)), 2.0f);
    BOOST_CHECK_EQUAL((Identity<uint8_t,int8_t>()(2)), 2);
    BOOST_CHECK_EQUAL((Identity<uint8_t,uint16_t>()(2)), 2);
    BOOST_CHECK_EQUAL((Identity<uint8_t,int16_t>()(2)), 2);
    BOOST_CHECK_EQUAL((Identity<uint8_t,bool>()(2U)), true);
    BOOST_CHECK_EQUAL((Identity<uint8_t,bool>()(0U)), false);

    BOOST_CHECK_EQUAL((Identity<int64_t,double>()(-2L)), -2.0);
    BOOST_CHECK_EQUAL((Identity<int64_t,float>()(-2L)), -2.0f);
    BOOST_CHECK_EQUAL((Identity<int64_t,int32_t>()(-2L)), -2);
    BOOST_CHECK_EQUAL((Identity<int64_t,int16_t>()(-2L)), -2);
    BOOST_CHECK_EQUAL((Identity<int64_t,bool>()(-2L)), true);
    BOOST_CHECK_EQUAL((Identity<int64_t,bool>()(0L)), false);

    BOOST_CHECK_EQUAL((Identity<int32_t,double>()(-2)), -2.0);
    BOOST_CHECK_EQUAL((Identity<int32_t,float>()(-2)), -2.0f);
    BOOST_CHECK_EQUAL((Identity<int32_t,int64_t>()(-2)), -2);
    BOOST_CHECK_EQUAL((Identity<int32_t,int16_t>()(-2)), -2);
    BOOST_CHECK_EQUAL((Identity<int32_t,bool>()(-2)), true);
    BOOST_CHECK_EQUAL((Identity<int32_t,bool>()(0)), false);

    BOOST_CHECK_EQUAL((Identity<int16_t,double>()(-2)), -2.0);
    BOOST_CHECK_EQUAL((Identity<int16_t,float>()(-2)), -2.0f);
    BOOST_CHECK_EQUAL((Identity<int16_t,int32_t>()(-2)), -2);
    BOOST_CHECK_EQUAL((Identity<int16_t,int8_t>()(-2)), -2);
    BOOST_CHECK_EQUAL((Identity<int16_t,bool>()(-2)), true);
    BOOST_CHECK_EQUAL((Identity<int16_t,bool>()(0)), false);

    BOOST_CHECK_EQUAL((Identity<int8_t,double>()(-2)), -2.0);
    BOOST_CHECK_EQUAL((Identity<int8_t,float>()(-2)), -2.0f);
    BOOST_CHECK_EQUAL((Identity<int8_t,int16_t>()(-2)), -2);
    BOOST_CHECK_EQUAL((Identity<int8_t,bool>()(-2)), true);
    BOOST_CHECK_EQUAL((Identity<int8_t,bool>()(0)), false);

    BOOST_CHECK_EQUAL((Identity<bool,double>()(false)), 0.0);
    BOOST_CHECK_EQUAL((Identity<bool,double>()(true)), 1.0);
    BOOST_CHECK_EQUAL((Identity<bool,float>()(false)), 0.0f);
    BOOST_CHECK_EQUAL((Identity<bool,float>()(true)), 1.0f);
    BOOST_CHECK_EQUAL((Identity<bool,uint32_t>()(false)), 0U);
    BOOST_CHECK_EQUAL((Identity<bool,uint32_t>()(true)), 1U);
    BOOST_CHECK_EQUAL((Identity<bool,int32_t>()(false)), 0);
    BOOST_CHECK_EQUAL((Identity<bool,int32_t>()(true)), 1);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(Abs_test)
{
    int8_t  i8  = -8;
    int16_t i16 = -16;
    int32_t i32 = -32;
    int64_t i64 = -64;

    signed char   ci  = -64;
    short         si  = -128;
    int           i   = -256;
    long int      li  = -512;

    uint8_t  ui8  = 8;
    uint16_t ui16 = 16;
    uint32_t ui32 = 32;
    uint64_t ui64 = 64;

    unsigned char     uci  = 64;
    unsigned short    usi  = 128;
    unsigned int      ui   = 256;
    unsigned long int uli  = 512;

    float f = -45.45f;
    double d = -1048.36;

    BOOST_CHECK_EQUAL(Abs<int8_t>()(i8),   abs(i8));
    BOOST_CHECK_EQUAL(Abs<int16_t>()(i16), abs(i16));
    BOOST_CHECK_EQUAL(Abs<int32_t>()(i32), abs(i32));
    BOOST_CHECK_EQUAL(Abs<int64_t>()(i64), labs(i64));

    BOOST_CHECK_EQUAL(Abs<char>()(ci),     abs(ci));
    BOOST_CHECK_EQUAL(Abs<short>()(si),    abs(si));
    BOOST_CHECK_EQUAL(Abs<int>()(i),       abs(i));
    BOOST_CHECK_EQUAL(Abs<long int>()(li), labs(li));

    BOOST_CHECK_EQUAL(Abs<uint8_t>()(ui8),   ui8);
    BOOST_CHECK_EQUAL(Abs<uint16_t>()(ui16), ui16);
    BOOST_CHECK_EQUAL(Abs<uint32_t>()(ui32), ui32);
    BOOST_CHECK_EQUAL(Abs<uint64_t>()(ui64), ui64);

    BOOST_CHECK_EQUAL(Abs<bool>()(true), true);

    BOOST_CHECK_EQUAL(Abs<unsigned char>()(uci),  uci);
    BOOST_CHECK_EQUAL(Abs<unsigned short>()(usi), usi);
    BOOST_CHECK_EQUAL(Abs<unsigned int>()(ui),    ui);
    BOOST_CHECK_EQUAL(Abs<unsigned long>()(uli),  uli);

    BOOST_CHECK_EQUAL(Abs<unsigned int>()(ui),   ui);

    BOOST_CHECK_EQUAL(Abs<float>()(f), fabsf(f));
    BOOST_CHECK_EQUAL(Abs<double>()(d), fabs(d));
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(additive_inverse_test)
{
    BOOST_CHECK_EQUAL(AdditiveInverse<double>()(2.2), -2.2);
    BOOST_CHECK_EQUAL(AdditiveInverse<double>()(0.0), 0.0);
    BOOST_CHECK_EQUAL(AdditiveInverse<double>()(-2.0), 2.0);

    BOOST_CHECK_EQUAL(AdditiveInverse<float>()(2.2f), -2.2f);
    BOOST_CHECK_EQUAL(AdditiveInverse<float>()(0.0f), 0.0f);
    BOOST_CHECK_EQUAL(AdditiveInverse<float>()(-2.2f), 2.2f);

    BOOST_CHECK_EQUAL(AdditiveInverse<uint64_t>()(2UL), static_cast<uint64_t>(-2L));
    BOOST_CHECK_EQUAL(AdditiveInverse<uint64_t>()(0UL), 0UL);
    BOOST_CHECK_EQUAL(AdditiveInverse<uint64_t>()(-2), 2UL);
    BOOST_CHECK_EQUAL(AdditiveInverse<uint32_t>()(2U), static_cast<uint32_t>(-2));
    BOOST_CHECK_EQUAL(AdditiveInverse<uint32_t>()(0U), 0U);
    BOOST_CHECK_EQUAL(AdditiveInverse<uint32_t>()(-2), 2U);
    BOOST_CHECK_EQUAL(AdditiveInverse<uint16_t>()(2), static_cast<uint16_t>(-2));
    BOOST_CHECK_EQUAL(AdditiveInverse<uint16_t>()(0), 0);
    BOOST_CHECK_EQUAL(AdditiveInverse<uint16_t>()(-2), 2);
    BOOST_CHECK_EQUAL(AdditiveInverse<uint8_t>()(2), static_cast<uint8_t>(-2));
    BOOST_CHECK_EQUAL(AdditiveInverse<uint8_t>()(0), 0);
    BOOST_CHECK_EQUAL(AdditiveInverse<uint8_t>()(-2), 2);

    BOOST_CHECK_EQUAL(AdditiveInverse<int64_t>()(-2L), 2L);
    BOOST_CHECK_EQUAL(AdditiveInverse<int64_t>()(0L), 0L);
    BOOST_CHECK_EQUAL(AdditiveInverse<int64_t>()(2L), -2L);
    BOOST_CHECK_EQUAL(AdditiveInverse<int32_t>()(-2), 2);
    BOOST_CHECK_EQUAL(AdditiveInverse<int32_t>()(0), 0);
    BOOST_CHECK_EQUAL(AdditiveInverse<int32_t>()(2), -2);
    BOOST_CHECK_EQUAL(AdditiveInverse<int16_t>()(-2), 2);
    BOOST_CHECK_EQUAL(AdditiveInverse<int16_t>()(0), 0);
    BOOST_CHECK_EQUAL(AdditiveInverse<int16_t>()(2), -2);
    BOOST_CHECK_EQUAL(AdditiveInverse<int8_t>()(-2), 2);
    BOOST_CHECK_EQUAL(AdditiveInverse<int8_t>()(0), 0);
    BOOST_CHECK_EQUAL(AdditiveInverse<int8_t>()(2), -2);

    BOOST_CHECK_EQUAL(AdditiveInverse<bool>()(false), false);
    BOOST_CHECK_EQUAL(AdditiveInverse<bool>()(true), true);

    // different domain tests

    BOOST_CHECK_EQUAL((AdditiveInverse<double, int>()(2.2)), -2);
    BOOST_CHECK_EQUAL((AdditiveInverse<double, int>()(0.0)),  0);
    BOOST_CHECK_EQUAL((AdditiveInverse<double, int>()(-2.0)), 2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(multiplicative_inverse_same_domain_test)
{
    BOOST_CHECK_EQUAL(MultiplicativeInverse<double>()(2.2), 1./2.2);
    BOOST_CHECK_EQUAL(MultiplicativeInverse<double>()(-2.2), -1./2.2);

    BOOST_CHECK_EQUAL(MultiplicativeInverse<float>()(2.2f), 1.f/2.2f);
    BOOST_CHECK_EQUAL(MultiplicativeInverse<float>()(-2.2f), -1.f/2.2f);

    BOOST_CHECK_EQUAL(MultiplicativeInverse<uint64_t>()(2UL), 0UL);
    BOOST_CHECK_EQUAL(MultiplicativeInverse<uint32_t>()(2U), 0U);
    BOOST_CHECK_EQUAL(MultiplicativeInverse<uint16_t>()(2), 0);
    BOOST_CHECK_EQUAL(MultiplicativeInverse<uint8_t>()(2), 0);

    BOOST_CHECK_EQUAL(MultiplicativeInverse<int64_t>()(-2L), 0L);
    BOOST_CHECK_EQUAL(MultiplicativeInverse<int32_t>()(-2), 0);
    BOOST_CHECK_EQUAL(MultiplicativeInverse<int16_t>()(-2), 0);
    BOOST_CHECK_EQUAL(MultiplicativeInverse<int8_t>()(-2), 0);

    BOOST_CHECK_EQUAL(MultiplicativeInverse<bool>()(true), true);

    // divide by zero exception:
    //BOOST_CHECK_EQUAL(MultiplicativeInverse<bool>()(false), false);

    // different domain tests

    BOOST_CHECK_EQUAL((MultiplicativeInverse<int,double>()(2)),   0.5);
    BOOST_CHECK_EQUAL((MultiplicativeInverse<int,double>()(1)),   1.0);
    BOOST_CHECK_EQUAL((MultiplicativeInverse<int,double>()(-3)), -1./3.);

    BOOST_CHECK_EQUAL((MultiplicativeInverse<double,int>()(0.5)),      2);
    BOOST_CHECK_EQUAL((MultiplicativeInverse<double,int>()(1.)),       1);
    BOOST_CHECK_EQUAL((MultiplicativeInverse<double,int>()(-0.3333)), -3);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(logical_not_same_domain_test)
{
    BOOST_CHECK_EQUAL(LogicalNot<double>()(2.2), 0.0);
    BOOST_CHECK_EQUAL(LogicalNot<double>()(0.0), 1.0);

    BOOST_CHECK_EQUAL(LogicalNot<float>()(2.2f), 0.0f);
    BOOST_CHECK_EQUAL(LogicalNot<float>()(1.0f), 0.0f);
    BOOST_CHECK_EQUAL(LogicalNot<float>()(0.0f), 1.0f);

    BOOST_CHECK_EQUAL(LogicalNot<uint64_t>()(2UL), 0UL);
    BOOST_CHECK_EQUAL(LogicalNot<uint64_t>()(0UL), 1UL);
    BOOST_CHECK_EQUAL(LogicalNot<uint32_t>()(2U), 0U);
    BOOST_CHECK_EQUAL(LogicalNot<uint32_t>()(0U), 1U);
    BOOST_CHECK_EQUAL(LogicalNot<uint16_t>()(2), 0);
    BOOST_CHECK_EQUAL(LogicalNot<uint16_t>()(0), 1);
    BOOST_CHECK_EQUAL(LogicalNot<uint8_t>()(2), 0);
    BOOST_CHECK_EQUAL(LogicalNot<uint8_t>()(0), 1);

    BOOST_CHECK_EQUAL(LogicalNot<int64_t>()(-2L), 0L);
    BOOST_CHECK_EQUAL(LogicalNot<int64_t>()(0L), 1L);
    BOOST_CHECK_EQUAL(LogicalNot<int32_t>()(-2), 0);
    BOOST_CHECK_EQUAL(LogicalNot<int32_t>()(0), 1);
    BOOST_CHECK_EQUAL(LogicalNot<int16_t>()(-2), 0);
    BOOST_CHECK_EQUAL(LogicalNot<int16_t>()(0), 1);
    BOOST_CHECK_EQUAL(LogicalNot<int8_t>()(-2), 0);
    BOOST_CHECK_EQUAL(LogicalNot<int8_t>()(0), 1);

    BOOST_CHECK_EQUAL(LogicalNot<int>()(1), 0);
    BOOST_CHECK_EQUAL(LogicalNot<int>()(0), 1);

    BOOST_CHECK_EQUAL(LogicalNot<bool>()(false), true);
    BOOST_CHECK_EQUAL(LogicalNot<bool>()(true), false);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(logical_not_different_domain_test)
{
    /// @todo this is an incomplete set of pairs and does not cover the
    /// corner cases at the numeric limits that actually vary by platform
    BOOST_CHECK_EQUAL((LogicalNot<double,float>()(2.2)), 0.0f);
    BOOST_CHECK_EQUAL((LogicalNot<double,float>()(0.0)), 1.0f);
    BOOST_CHECK_EQUAL((LogicalNot<double,uint64_t>()(2.2)), 0UL);
    BOOST_CHECK_EQUAL((LogicalNot<double,uint64_t>()(0.0)), 1UL);
    BOOST_CHECK_EQUAL((LogicalNot<double,int64_t>()(-2.2)), 0L);
    BOOST_CHECK_EQUAL((LogicalNot<double,int64_t>()(0.0)), 1L);
    BOOST_CHECK_EQUAL((LogicalNot<double,bool>()(2.2)), false);
    BOOST_CHECK_EQUAL((LogicalNot<double,bool>()(0.0)), true);

    BOOST_CHECK_EQUAL((LogicalNot<float,double>()(2.2f)), 0.0);
    BOOST_CHECK_EQUAL((LogicalNot<float,double>()(0.0f)), 1.0);
    BOOST_CHECK_EQUAL((LogicalNot<float,uint32_t>()(2.2f)), 0U);
    BOOST_CHECK_EQUAL((LogicalNot<float,uint32_t>()(0.0f)), 1U);
    BOOST_CHECK_EQUAL((LogicalNot<float,int32_t>()(-2.2f)), 0);
    BOOST_CHECK_EQUAL((LogicalNot<float,int32_t>()( 0.0f)), 1);
    BOOST_CHECK_EQUAL((LogicalNot<float,bool>()(2.2f)), false);
    BOOST_CHECK_EQUAL((LogicalNot<float,bool>()(0.0f)), true);

    BOOST_CHECK_EQUAL((LogicalNot<uint64_t, double>()(2UL)), 0.0);
    BOOST_CHECK_EQUAL((LogicalNot<uint64_t, double>()(0UL)), 1.0);
    BOOST_CHECK_EQUAL((LogicalNot<uint64_t, float>()(2UL)), 0.0f);
    BOOST_CHECK_EQUAL((LogicalNot<uint64_t, float>()(0UL)), 1.0f);
    BOOST_CHECK_EQUAL((LogicalNot<uint64_t, int64_t>()(2UL)), 0L);
    BOOST_CHECK_EQUAL((LogicalNot<uint64_t, int64_t>()(0UL)), 1L);
    BOOST_CHECK_EQUAL((LogicalNot<uint64_t,uint32_t>()(2UL)), 0UL);
    BOOST_CHECK_EQUAL((LogicalNot<uint64_t,uint32_t>()(0UL)), 1UL);
    BOOST_CHECK_EQUAL((LogicalNot<uint64_t,bool>()(2UL)), false);
    BOOST_CHECK_EQUAL((LogicalNot<uint64_t,bool>()(0UL)), true);

    BOOST_CHECK_EQUAL((LogicalNot<uint32_t, double>()(2U)), 0.0);
    BOOST_CHECK_EQUAL((LogicalNot<uint32_t, double>()(0U)), 1.0);
    BOOST_CHECK_EQUAL((LogicalNot<uint32_t, float>()(2U)), 0.0f);
    BOOST_CHECK_EQUAL((LogicalNot<uint32_t, float>()(0U)), 1.0f);
    BOOST_CHECK_EQUAL((LogicalNot<uint32_t,uint64_t>()(2U)), 0UL);
    BOOST_CHECK_EQUAL((LogicalNot<uint32_t,uint64_t>()(0U)), 1UL);
    BOOST_CHECK_EQUAL((LogicalNot<uint32_t, int64_t>()(2U)), 0L);
    BOOST_CHECK_EQUAL((LogicalNot<uint32_t, int64_t>()(0U)), 1L);
    BOOST_CHECK_EQUAL((LogicalNot<uint32_t, int32_t>()(2U)), 0U);
    BOOST_CHECK_EQUAL((LogicalNot<uint32_t, int32_t>()(0U)), 1U);
    BOOST_CHECK_EQUAL((LogicalNot<uint32_t,uint16_t>()(2U)), 0U);
    BOOST_CHECK_EQUAL((LogicalNot<uint32_t,uint16_t>()(0U)), 1U);
    BOOST_CHECK_EQUAL((LogicalNot<uint32_t, int16_t>()(2U)), 0);
    BOOST_CHECK_EQUAL((LogicalNot<uint32_t, int16_t>()(0U)), 1);
    BOOST_CHECK_EQUAL((LogicalNot<uint32_t,bool>()(2U)), false);
    BOOST_CHECK_EQUAL((LogicalNot<uint32_t,bool>()(0U)), true);

    BOOST_CHECK_EQUAL((LogicalNot<uint16_t, double>()(2U)), 0.0);
    BOOST_CHECK_EQUAL((LogicalNot<uint16_t, double>()(0U)), 1.0);
    BOOST_CHECK_EQUAL((LogicalNot<uint16_t, float>()(2U)), 0.0f);
    BOOST_CHECK_EQUAL((LogicalNot<uint16_t, float>()(0U)), 1.0f);
    BOOST_CHECK_EQUAL((LogicalNot<uint16_t,uint32_t>()(2U)), 0U);
    BOOST_CHECK_EQUAL((LogicalNot<uint16_t,uint32_t>()(0U)), 1U);
    BOOST_CHECK_EQUAL((LogicalNot<uint16_t, int32_t>()(2U)), 0);
    BOOST_CHECK_EQUAL((LogicalNot<uint16_t, int32_t>()(0U)), 1);
    BOOST_CHECK_EQUAL((LogicalNot<uint16_t, int16_t>()(2)), 0);
    BOOST_CHECK_EQUAL((LogicalNot<uint16_t, int16_t>()(0)), 1);
    BOOST_CHECK_EQUAL((LogicalNot<uint16_t,uint8_t>()(2)), 0);
    BOOST_CHECK_EQUAL((LogicalNot<uint16_t,uint8_t>()(0)), 1);
    BOOST_CHECK_EQUAL((LogicalNot<uint16_t, int8_t>()(2)), 0);
    BOOST_CHECK_EQUAL((LogicalNot<uint16_t, int8_t>()(0)), 1);
    BOOST_CHECK_EQUAL((LogicalNot<uint16_t,bool>()(2U)), false);
    BOOST_CHECK_EQUAL((LogicalNot<uint16_t,bool>()(0U)), true);

    BOOST_CHECK_EQUAL((LogicalNot<uint8_t,double>()(2U)), 0.0);
    BOOST_CHECK_EQUAL((LogicalNot<uint8_t,double>()(0U)), 1.0);
    BOOST_CHECK_EQUAL((LogicalNot<uint8_t,float>()(2U)), 0.0f);
    BOOST_CHECK_EQUAL((LogicalNot<uint8_t,float>()(0U)), 1.0f);
    BOOST_CHECK_EQUAL((LogicalNot<uint8_t,int8_t>()(2)), 0);
    BOOST_CHECK_EQUAL((LogicalNot<uint8_t,int8_t>()(0)), 1);
    BOOST_CHECK_EQUAL((LogicalNot<uint8_t,uint16_t>()(2)), 0);
    BOOST_CHECK_EQUAL((LogicalNot<uint8_t,uint16_t>()(0)), 1);
    BOOST_CHECK_EQUAL((LogicalNot<uint8_t,int16_t>()(2)), 0);
    BOOST_CHECK_EQUAL((LogicalNot<uint8_t,int16_t>()(0)), 1);
    BOOST_CHECK_EQUAL((LogicalNot<uint8_t,bool>()(2U)), false);
    BOOST_CHECK_EQUAL((LogicalNot<uint8_t,bool>()(0U)), true);

    BOOST_CHECK_EQUAL((LogicalNot<int64_t,double>()(-2L)), 0.0);
    BOOST_CHECK_EQUAL((LogicalNot<int64_t,double>()(0L)), 1.0);
    BOOST_CHECK_EQUAL((LogicalNot<int64_t,float>()(-2L)), 0.0f);
    BOOST_CHECK_EQUAL((LogicalNot<int64_t,float>()(0L)), 1.0f);
    BOOST_CHECK_EQUAL((LogicalNot<int64_t,int32_t>()(-2L)), 0);
    BOOST_CHECK_EQUAL((LogicalNot<int64_t,int32_t>()(0L)), 1);
    BOOST_CHECK_EQUAL((LogicalNot<int64_t,int16_t>()(-2L)), 0);
    BOOST_CHECK_EQUAL((LogicalNot<int64_t,int16_t>()(0L)), 1);
    BOOST_CHECK_EQUAL((LogicalNot<int64_t,bool>()(-2L)), false);
    BOOST_CHECK_EQUAL((LogicalNot<int64_t,bool>()(0L)), true);

    BOOST_CHECK_EQUAL((LogicalNot<int32_t,double>()(-2)), 0.0);
    BOOST_CHECK_EQUAL((LogicalNot<int32_t,double>()(0)), 1.0);
    BOOST_CHECK_EQUAL((LogicalNot<int32_t,float>()(-2)), 0.0f);
    BOOST_CHECK_EQUAL((LogicalNot<int32_t,float>()(0)), 1.0f);
    BOOST_CHECK_EQUAL((LogicalNot<int32_t,int64_t>()(-2)), 0);
    BOOST_CHECK_EQUAL((LogicalNot<int32_t,int64_t>()(0)), 1);
    BOOST_CHECK_EQUAL((LogicalNot<int32_t,int16_t>()(-2)), 0);
    BOOST_CHECK_EQUAL((LogicalNot<int32_t,int16_t>()(0)), 1);
    BOOST_CHECK_EQUAL((LogicalNot<int32_t,bool>()(-2)), false);
    BOOST_CHECK_EQUAL((LogicalNot<int32_t,bool>()(0)), true);

    BOOST_CHECK_EQUAL((LogicalNot<int16_t,double>()(-2)), 0.0);
    BOOST_CHECK_EQUAL((LogicalNot<int16_t,double>()(0)), 1.0);
    BOOST_CHECK_EQUAL((LogicalNot<int16_t,float>()(-2)), 0.0f);
    BOOST_CHECK_EQUAL((LogicalNot<int16_t,float>()(0)), 1.0f);
    BOOST_CHECK_EQUAL((LogicalNot<int16_t,int32_t>()(-2)), 0);
    BOOST_CHECK_EQUAL((LogicalNot<int16_t,int32_t>()(0)), 1);
    BOOST_CHECK_EQUAL((LogicalNot<int16_t,int8_t>()(-2)), 0);
    BOOST_CHECK_EQUAL((LogicalNot<int16_t,int8_t>()(0)), 1);
    BOOST_CHECK_EQUAL((LogicalNot<int16_t,bool>()(-2)), false);
    BOOST_CHECK_EQUAL((LogicalNot<int16_t,bool>()(0)), true);

    BOOST_CHECK_EQUAL((LogicalNot<int8_t,double>()(-2)), 0.0);
    BOOST_CHECK_EQUAL((LogicalNot<int8_t,double>()(0)), 1.0);
    BOOST_CHECK_EQUAL((LogicalNot<int8_t,float>()(-2)), 0.0f);
    BOOST_CHECK_EQUAL((LogicalNot<int8_t,float>()(0)), 1.0f);
    BOOST_CHECK_EQUAL((LogicalNot<int8_t,int16_t>()(-2)), 0);
    BOOST_CHECK_EQUAL((LogicalNot<int8_t,int16_t>()(0)), 1);
    BOOST_CHECK_EQUAL((LogicalNot<int8_t,bool>()(-2)), false);
    BOOST_CHECK_EQUAL((LogicalNot<int8_t,bool>()(0)), true);

    BOOST_CHECK_EQUAL((LogicalNot<bool,double>()(false)), 1.0);
    BOOST_CHECK_EQUAL((LogicalNot<bool,double>()(true)), 0.0);
    BOOST_CHECK_EQUAL((LogicalNot<bool,float>()(false)), 1.0f);
    BOOST_CHECK_EQUAL((LogicalNot<bool,float>()(true)), 0.0f);
    BOOST_CHECK_EQUAL((LogicalNot<bool,uint32_t>()(false)), 1U);
    BOOST_CHECK_EQUAL((LogicalNot<bool,uint32_t>()(true)), 0U);
    BOOST_CHECK_EQUAL((LogicalNot<bool,int32_t>()(false)), 1);
    BOOST_CHECK_EQUAL((LogicalNot<bool,int32_t>()(true)), 0);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(BitwiseNot_test)
{
    bool    b   = true;

    int8_t  i8  = -8;
    int16_t i16 = -16;
    int32_t i32 = -32;
    int64_t i64 = -64;

    uint8_t  ui8  = 8;
    uint16_t ui16 = 16;
    uint32_t ui32 = 32;
    uint64_t ui64 = 64;

    // odd result for bool (but correct)
    BOOST_CHECK_EQUAL(BitwiseNot<bool>()(b), true);

    BOOST_CHECK_EQUAL(BitwiseNot<int8_t>()(i8), 7);
    BOOST_CHECK_EQUAL(BitwiseNot<int16_t>()(i16), 15);
    BOOST_CHECK_EQUAL(BitwiseNot<int32_t>()(i32), 31);
    BOOST_CHECK_EQUAL(BitwiseNot<int64_t>()(i64), 63);
    BOOST_CHECK_EQUAL(BitwiseNot<uint8_t>()(ui8),   0xf7);
    BOOST_CHECK_EQUAL(BitwiseNot<uint16_t>()(ui16), 0xffef);
    BOOST_CHECK_EQUAL(BitwiseNot<uint32_t>()(ui32), 0xffffffdf);
    BOOST_CHECK_EQUAL(BitwiseNot<uint64_t>()(ui64),
                      0xffffffffffffffbf);

    // different domains
    BOOST_CHECK_EQUAL((BitwiseNot<uint64_t,uint32_t>()(ui64)),
                      0xffffffbf);
    BOOST_CHECK_EQUAL((BitwiseNot<uint64_t,uint16_t>()(ui64)),
                      0xffbf);
    BOOST_CHECK_EQUAL((BitwiseNot<uint64_t,uint8_t>()(ui64)),
                      0xbf);
}

//****************************************************************************
// Binary Operator tests
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(logical_or_same_domain_test)
{
    BOOST_CHECK_EQUAL(LogicalOr<double>()(0.0, 0.0), 0.0);
    BOOST_CHECK_EQUAL(LogicalOr<double>()(1.0, 0.0), 1.0);
    BOOST_CHECK_EQUAL(LogicalOr<double>()(0.0, 1.0), 1.0);
    BOOST_CHECK_EQUAL(LogicalOr<double>()(1.0, 1.0), 1.0);

    BOOST_CHECK_EQUAL(LogicalOr<float>()(0.0f, 0.0f), 0.0f);
    BOOST_CHECK_EQUAL(LogicalOr<float>()(1.0f, 0.0f), 1.0f);
    BOOST_CHECK_EQUAL(LogicalOr<float>()(0.0f, 1.0f), 1.0f);
    BOOST_CHECK_EQUAL(LogicalOr<float>()(1.0f, 1.0f), 1.0f);

    BOOST_CHECK_EQUAL(LogicalOr<uint64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalOr<uint64_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(LogicalOr<uint64_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(LogicalOr<uint64_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(LogicalOr<uint32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalOr<uint32_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(LogicalOr<uint32_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(LogicalOr<uint32_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(LogicalOr<uint16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalOr<uint16_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(LogicalOr<uint16_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(LogicalOr<uint16_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(LogicalOr<uint8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalOr<uint8_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(LogicalOr<uint8_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(LogicalOr<uint8_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(LogicalOr<int64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalOr<int64_t>()(-1, 0), 1);
    BOOST_CHECK_EQUAL(LogicalOr<int64_t>()(0, -1), 1);
    BOOST_CHECK_EQUAL(LogicalOr<int64_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(LogicalOr<int32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalOr<int32_t>()(-1, 0), 1);
    BOOST_CHECK_EQUAL(LogicalOr<int32_t>()(0, -1), 1);
    BOOST_CHECK_EQUAL(LogicalOr<int32_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(LogicalOr<int16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalOr<int16_t>()(-1, 0), 1);
    BOOST_CHECK_EQUAL(LogicalOr<int16_t>()(0, -1), 1);
    BOOST_CHECK_EQUAL(LogicalOr<int16_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(LogicalOr<int8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalOr<int8_t>()(-1, 0), 1);
    BOOST_CHECK_EQUAL(LogicalOr<int8_t>()(0, -1), 1);
    BOOST_CHECK_EQUAL(LogicalOr<int8_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(LogicalOr<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(LogicalOr<bool>()(false, true),  true);
    BOOST_CHECK_EQUAL(LogicalOr<bool>()(true, false),  true);
    BOOST_CHECK_EQUAL(LogicalOr<bool>()(true, true),   true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(logical_or_different_domain_test)
{
    BOOST_CHECK_EQUAL((LogicalOr<int,double>()( 0, 0.0)), 0.0);
    BOOST_CHECK_EQUAL((LogicalOr<int,double>()(-1, 0.0)), 1.0);
    BOOST_CHECK_EQUAL((LogicalOr<int,double>()( 0, -1.0)), 1.0);
    BOOST_CHECK_EQUAL((LogicalOr<int,double>()(-1, 1.0)), 1.0);

    BOOST_CHECK_EQUAL((LogicalOr<float,int>()(0.0f, 0)), 0.0f);
    BOOST_CHECK_EQUAL((LogicalOr<float,int>()(1.0f, 0)), 1.0f);
    BOOST_CHECK_EQUAL((LogicalOr<float,int>()(0.0f, -1)), 1.0f);
    BOOST_CHECK_EQUAL((LogicalOr<float,int>()(1.0f, -1)), 1.0f);

    BOOST_CHECK_EQUAL((LogicalOr<float, uint64_t>()(  0.f, 0)), 0.f);
    BOOST_CHECK_EQUAL((LogicalOr<float, uint64_t>()( 0.1f, 0)), 1.f);
    BOOST_CHECK_EQUAL((LogicalOr<float, uint64_t>()(  0.f, 1)), 1.f);
    BOOST_CHECK_EQUAL((LogicalOr<float, uint64_t>()(-.05f, 1)), 1.f);

    BOOST_CHECK_EQUAL((LogicalOr<int32_t,float>()( 0, 0.f)), 0);
    BOOST_CHECK_EQUAL((LogicalOr<int32_t,float>()(-1, 0.f)), 1);
    BOOST_CHECK_EQUAL((LogicalOr<int32_t,float>()( 0, 1.f)), 1);
    BOOST_CHECK_EQUAL((LogicalOr<int32_t,float>()(-1, 1.f)), 1);

    BOOST_CHECK_EQUAL((LogicalOr<double, uint16_t>()(0., 0)), 0.);
    BOOST_CHECK_EQUAL((LogicalOr<double, uint16_t>()(1., 0)), 1.);
    BOOST_CHECK_EQUAL((LogicalOr<double, uint16_t>()(0., 1)), 1.);
    BOOST_CHECK_EQUAL((LogicalOr<double, uint16_t>()(1., 1)), 1.);

    BOOST_CHECK_EQUAL((LogicalOr<uint8_t, float>()(0, 0.f)), 0);
    BOOST_CHECK_EQUAL((LogicalOr<uint8_t, float>()(1, 0.f)), 1);
    BOOST_CHECK_EQUAL((LogicalOr<uint8_t, float>()(0, 1.f)), 1);
    BOOST_CHECK_EQUAL((LogicalOr<uint8_t, float>()(1, 1.f)), 1);

    BOOST_CHECK_EQUAL((LogicalOr<bool, float>()(false, 0.f)), false);
    BOOST_CHECK_EQUAL((LogicalOr<bool, float>()(true,  0.f)), true);
    BOOST_CHECK_EQUAL((LogicalOr<bool, float>()(false, 1.f)), true);
    BOOST_CHECK_EQUAL((LogicalOr<bool, float>()(true,  1.f)), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(logical_and_same_domain_test)
{
    BOOST_CHECK_EQUAL(LogicalAnd<double>()(0.0, 0.0), 0.0);
    BOOST_CHECK_EQUAL(LogicalAnd<double>()(1.0, 0.0), 0.0);
    BOOST_CHECK_EQUAL(LogicalAnd<double>()(0.0, 1.0), 0.0);
    BOOST_CHECK_EQUAL(LogicalAnd<double>()(1.0, 1.0), 1.0);

    BOOST_CHECK_EQUAL(LogicalAnd<float>()(0.0f, 0.0f), 0.0f);
    BOOST_CHECK_EQUAL(LogicalAnd<float>()(1.0f, 0.0f), 0.0f);
    BOOST_CHECK_EQUAL(LogicalAnd<float>()(0.0f, 1.0f), 0.0f);
    BOOST_CHECK_EQUAL(LogicalAnd<float>()(1.0f, 1.0f), 1.0f);

    BOOST_CHECK_EQUAL(LogicalAnd<uint64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalAnd<uint64_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(LogicalAnd<uint64_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(LogicalAnd<uint64_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(LogicalAnd<uint32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalAnd<uint32_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(LogicalAnd<uint32_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(LogicalAnd<uint32_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(LogicalAnd<uint16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalAnd<uint16_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(LogicalAnd<uint16_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(LogicalAnd<uint16_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(LogicalAnd<uint8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalAnd<uint8_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(LogicalAnd<uint8_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(LogicalAnd<uint8_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(LogicalAnd<int64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalAnd<int64_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(LogicalAnd<int64_t>()(0, -1), 0);
    BOOST_CHECK_EQUAL(LogicalAnd<int64_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(LogicalAnd<int32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalAnd<int32_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(LogicalAnd<int32_t>()(0, -1), 0);
    BOOST_CHECK_EQUAL(LogicalAnd<int32_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(LogicalAnd<int16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalAnd<int16_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(LogicalAnd<int16_t>()(0, -1), 0);
    BOOST_CHECK_EQUAL(LogicalAnd<int16_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(LogicalAnd<int8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalAnd<int8_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(LogicalAnd<int8_t>()(0, -1), 0);
    BOOST_CHECK_EQUAL(LogicalAnd<int8_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(LogicalAnd<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(LogicalAnd<bool>()(false, true),  false);
    BOOST_CHECK_EQUAL(LogicalAnd<bool>()(true, false),  false);
    BOOST_CHECK_EQUAL(LogicalAnd<bool>()(true, true),   true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(logical_and_different_domain_test)
{
    BOOST_CHECK_EQUAL((LogicalAnd<int,double>()( 0, 0.0)), 0.0);
    BOOST_CHECK_EQUAL((LogicalAnd<int,double>()(-1, 0.0)), 0.0);
    BOOST_CHECK_EQUAL((LogicalAnd<int,double>()( 0,-1.0)), 0.0);
    BOOST_CHECK_EQUAL((LogicalAnd<int,double>()(-1, 1.0)), 1.0);

    BOOST_CHECK_EQUAL((LogicalAnd<float,int>()(0.0f,  0)), 0.0f);
    BOOST_CHECK_EQUAL((LogicalAnd<float,int>()(1.0f,  0)), 0.0f);
    BOOST_CHECK_EQUAL((LogicalAnd<float,int>()(0.0f, -1)), 0.0f);
    BOOST_CHECK_EQUAL((LogicalAnd<float,int>()(1.0f, -1)), 1.0f);

    BOOST_CHECK_EQUAL((LogicalAnd<float, uint64_t>()(  0.f, 0)), 0.f);
    BOOST_CHECK_EQUAL((LogicalAnd<float, uint64_t>()( 0.1f, 0)), 0.f);
    BOOST_CHECK_EQUAL((LogicalAnd<float, uint64_t>()(  0.f, 1)), 0.f);
    BOOST_CHECK_EQUAL((LogicalAnd<float, uint64_t>()(-.05f, 1)), 1.f);

    BOOST_CHECK_EQUAL((LogicalAnd<int32_t,float>()( 0, 0.f)), 0);
    BOOST_CHECK_EQUAL((LogicalAnd<int32_t,float>()(-1, 0.f)), 0);
    BOOST_CHECK_EQUAL((LogicalAnd<int32_t,float>()( 0, 1.f)), 0);
    BOOST_CHECK_EQUAL((LogicalAnd<int32_t,float>()(-1, 1.f)), 1);

    BOOST_CHECK_EQUAL((LogicalAnd<double, uint16_t>()(0., 0)), 0.);
    BOOST_CHECK_EQUAL((LogicalAnd<double, uint16_t>()(1., 0)), 0.);
    BOOST_CHECK_EQUAL((LogicalAnd<double, uint16_t>()(0., 1)), 0.);
    BOOST_CHECK_EQUAL((LogicalAnd<double, uint16_t>()(1., 1)), 1.);

    BOOST_CHECK_EQUAL((LogicalAnd<uint8_t, float>()(0, 0.f)), 0);
    BOOST_CHECK_EQUAL((LogicalAnd<uint8_t, float>()(1, 0.f)), 0);
    BOOST_CHECK_EQUAL((LogicalAnd<uint8_t, float>()(0, 1.f)), 0);
    BOOST_CHECK_EQUAL((LogicalAnd<uint8_t, float>()(1, 1.f)), 1);

    BOOST_CHECK_EQUAL((LogicalAnd<bool, float>()(false, 0.f)), false);
    BOOST_CHECK_EQUAL((LogicalAnd<bool, float>()(true,  0.f)), false);
    BOOST_CHECK_EQUAL((LogicalAnd<bool, float>()(false, 1.f)), false);
    BOOST_CHECK_EQUAL((LogicalAnd<bool, float>()(true,  1.f)), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(logical_xor_same_domain_test)
{
    BOOST_CHECK_EQUAL(LogicalXor<double>()(0.0, 0.0), 0.0);
    BOOST_CHECK_EQUAL(LogicalXor<double>()(1.0, 0.0), 1.0);
    BOOST_CHECK_EQUAL(LogicalXor<double>()(0.0, 1.0), 1.0);
    BOOST_CHECK_EQUAL(LogicalXor<double>()(1.0, 1.0), 0.0);

    BOOST_CHECK_EQUAL(LogicalXor<float>()(0.0f, 0.0f), 0.0f);
    BOOST_CHECK_EQUAL(LogicalXor<float>()(1.0f, 0.0f), 1.0f);
    BOOST_CHECK_EQUAL(LogicalXor<float>()(0.0f, 1.0f), 1.0f);
    BOOST_CHECK_EQUAL(LogicalXor<float>()(1.0f, 1.0f), 0.0f);

    BOOST_CHECK_EQUAL(LogicalXor<uint64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalXor<uint64_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(LogicalXor<uint64_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(LogicalXor<uint64_t>()(1, 1), 0);

    BOOST_CHECK_EQUAL(LogicalXor<uint32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalXor<uint32_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(LogicalXor<uint32_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(LogicalXor<uint32_t>()(1, 1), 0);

    BOOST_CHECK_EQUAL(LogicalXor<uint16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalXor<uint16_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(LogicalXor<uint16_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(LogicalXor<uint16_t>()(1, 1), 0);

    BOOST_CHECK_EQUAL(LogicalXor<uint8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalXor<uint8_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(LogicalXor<uint8_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(LogicalXor<uint8_t>()(1, 1), 0);

    BOOST_CHECK_EQUAL(LogicalXor<int64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalXor<int64_t>()(-1, 0), 1);
    BOOST_CHECK_EQUAL(LogicalXor<int64_t>()(0, -1), 1);
    BOOST_CHECK_EQUAL(LogicalXor<int64_t>()(-1, -1), 0);

    BOOST_CHECK_EQUAL(LogicalXor<int32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalXor<int32_t>()(-1, 0), 1);
    BOOST_CHECK_EQUAL(LogicalXor<int32_t>()(0, -1), 1);
    BOOST_CHECK_EQUAL(LogicalXor<int32_t>()(-1, -1), 0);

    BOOST_CHECK_EQUAL(LogicalXor<int16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalXor<int16_t>()(-1, 0), 1);
    BOOST_CHECK_EQUAL(LogicalXor<int16_t>()(0, -1), 1);
    BOOST_CHECK_EQUAL(LogicalXor<int16_t>()(-1, -1), 0);

    BOOST_CHECK_EQUAL(LogicalXor<int8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalXor<int8_t>()(-1, 0), 1);
    BOOST_CHECK_EQUAL(LogicalXor<int8_t>()(0, -1), 1);
    BOOST_CHECK_EQUAL(LogicalXor<int8_t>()(-1, -1), 0);

    BOOST_CHECK_EQUAL(LogicalXor<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(LogicalXor<bool>()(false, true),  true);
    BOOST_CHECK_EQUAL(LogicalXor<bool>()(true, false),  true);
    BOOST_CHECK_EQUAL(LogicalXor<bool>()(true, true),   false);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(logical_xor_different_domain_test)
{
    BOOST_CHECK_EQUAL((LogicalXor<int,double>()( 0, 0.0)), 0.0);
    BOOST_CHECK_EQUAL((LogicalXor<int,double>()(-1, 0.0)), 1.0);
    BOOST_CHECK_EQUAL((LogicalXor<int,double>()( 0,-1.0)), 1.0);
    BOOST_CHECK_EQUAL((LogicalXor<int,double>()(-1, 1.0)), 0.0);

    BOOST_CHECK_EQUAL((LogicalXor<float,int>()(0.0f,  0)), 0.0f);
    BOOST_CHECK_EQUAL((LogicalXor<float,int>()(1.0f,  0)), 1.0f);
    BOOST_CHECK_EQUAL((LogicalXor<float,int>()(0.0f, -1)), 1.0f);
    BOOST_CHECK_EQUAL((LogicalXor<float,int>()(1.0f, -1)), 0.0f);

    BOOST_CHECK_EQUAL((LogicalXor<float, uint64_t>()(  0.f, 0)), 0.f);
    BOOST_CHECK_EQUAL((LogicalXor<float, uint64_t>()( 0.1f, 0)), 1.f);
    BOOST_CHECK_EQUAL((LogicalXor<float, uint64_t>()(  0.f, 1)), 1.f);
    BOOST_CHECK_EQUAL((LogicalXor<float, uint64_t>()(-.05f, 1)), 0.f);

    BOOST_CHECK_EQUAL((LogicalXor<int32_t,float>()( 0, 0.f)), 0);
    BOOST_CHECK_EQUAL((LogicalXor<int32_t,float>()(-1, 0.f)), 1);
    BOOST_CHECK_EQUAL((LogicalXor<int32_t,float>()( 0, 1.f)), 1);
    BOOST_CHECK_EQUAL((LogicalXor<int32_t,float>()(-1, 1.f)), 0);

    BOOST_CHECK_EQUAL((LogicalXor<double, uint16_t>()(0., 0)), 0.);
    BOOST_CHECK_EQUAL((LogicalXor<double, uint16_t>()(1., 0)), 1.);
    BOOST_CHECK_EQUAL((LogicalXor<double, uint16_t>()(0., 1)), 1.);
    BOOST_CHECK_EQUAL((LogicalXor<double, uint16_t>()(1., 1)), 0.);

    BOOST_CHECK_EQUAL((LogicalXor<uint8_t, float>()(0, 0.f)), 0);
    BOOST_CHECK_EQUAL((LogicalXor<uint8_t, float>()(1, 0.f)), 1);
    BOOST_CHECK_EQUAL((LogicalXor<uint8_t, float>()(0, 1.f)), 1);
    BOOST_CHECK_EQUAL((LogicalXor<uint8_t, float>()(1, 1.f)), 0);

    BOOST_CHECK_EQUAL((LogicalXor<bool, float>()(false, 0.f)), false);
    BOOST_CHECK_EQUAL((LogicalXor<bool, float>()(true,  0.f)),  true);
    BOOST_CHECK_EQUAL((LogicalXor<bool, float>()(false, 1.f)),  true);
    BOOST_CHECK_EQUAL((LogicalXor<bool, float>()(true,  1.f)), false);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(logical_xnor_same_domain_test)
{
    BOOST_CHECK_EQUAL(LogicalXnor<double>()(0.0, 0.0), 1.0);
    BOOST_CHECK_EQUAL(LogicalXnor<double>()(1.0, 0.0), 0.0);
    BOOST_CHECK_EQUAL(LogicalXnor<double>()(0.0, 1.0), 0.0);
    BOOST_CHECK_EQUAL(LogicalXnor<double>()(1.0, 1.0), 1.0);

    BOOST_CHECK_EQUAL(LogicalXnor<float>()(0.0f, 0.0f), 1.0f);
    BOOST_CHECK_EQUAL(LogicalXnor<float>()(1.0f, 0.0f), 0.0f);
    BOOST_CHECK_EQUAL(LogicalXnor<float>()(0.0f, 1.0f), 0.0f);
    BOOST_CHECK_EQUAL(LogicalXnor<float>()(1.0f, 1.0f), 1.0f);

    BOOST_CHECK_EQUAL(LogicalXnor<uint64_t>()(0, 0), 1);
    BOOST_CHECK_EQUAL(LogicalXnor<uint64_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(LogicalXnor<uint64_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(LogicalXnor<uint64_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(LogicalXnor<uint32_t>()(0, 0), 1);
    BOOST_CHECK_EQUAL(LogicalXnor<uint32_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(LogicalXnor<uint32_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(LogicalXnor<uint32_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(LogicalXnor<uint16_t>()(0, 0), 1);
    BOOST_CHECK_EQUAL(LogicalXnor<uint16_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(LogicalXnor<uint16_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(LogicalXnor<uint16_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(LogicalXnor<uint8_t>()(0, 0), 1);
    BOOST_CHECK_EQUAL(LogicalXnor<uint8_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(LogicalXnor<uint8_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(LogicalXnor<uint8_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(LogicalXnor<int64_t>()(0, 0), 1);
    BOOST_CHECK_EQUAL(LogicalXnor<int64_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(LogicalXnor<int64_t>()(0, -1), 0);
    BOOST_CHECK_EQUAL(LogicalXnor<int64_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(LogicalXnor<int32_t>()(0, 0), 1);
    BOOST_CHECK_EQUAL(LogicalXnor<int32_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(LogicalXnor<int32_t>()(0, -1), 0);
    BOOST_CHECK_EQUAL(LogicalXnor<int32_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(LogicalXnor<int16_t>()(0, 0), 1);
    BOOST_CHECK_EQUAL(LogicalXnor<int16_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(LogicalXnor<int16_t>()(0, -1), 0);
    BOOST_CHECK_EQUAL(LogicalXnor<int16_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(LogicalXnor<int8_t>()(0, 0), 1);
    BOOST_CHECK_EQUAL(LogicalXnor<int8_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(LogicalXnor<int8_t>()(0, -1), 0);
    BOOST_CHECK_EQUAL(LogicalXnor<int8_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(LogicalXnor<bool>()(false, false), true);
    BOOST_CHECK_EQUAL(LogicalXnor<bool>()(false, true), false);
    BOOST_CHECK_EQUAL(LogicalXnor<bool>()(true, false), false);
    BOOST_CHECK_EQUAL(LogicalXnor<bool>()(true, true),   true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(logical_xnor_different_domain_test)
{
    BOOST_CHECK_EQUAL((LogicalXnor<int,double>()( 0, 0.0)), 1.0);
    BOOST_CHECK_EQUAL((LogicalXnor<int,double>()(-1, 0.0)), 0.0);
    BOOST_CHECK_EQUAL((LogicalXnor<int,double>()( 0,-1.0)), 0.0);
    BOOST_CHECK_EQUAL((LogicalXnor<int,double>()(-1, 1.0)), 1.0);

    BOOST_CHECK_EQUAL((LogicalXnor<float,int>()(0.0f,  0)), 1.0f);
    BOOST_CHECK_EQUAL((LogicalXnor<float,int>()(1.0f,  0)), 0.0f);
    BOOST_CHECK_EQUAL((LogicalXnor<float,int>()(0.0f, -1)), 0.0f);
    BOOST_CHECK_EQUAL((LogicalXnor<float,int>()(1.0f, -1)), 1.0f);

    BOOST_CHECK_EQUAL((LogicalXnor<float, uint64_t>()(  0.f, 0)), 1.f);
    BOOST_CHECK_EQUAL((LogicalXnor<float, uint64_t>()( 0.1f, 0)), 0.f);
    BOOST_CHECK_EQUAL((LogicalXnor<float, uint64_t>()(  0.f, 1)), 0.f);
    BOOST_CHECK_EQUAL((LogicalXnor<float, uint64_t>()(-.05f, 1)), 1.f);

    BOOST_CHECK_EQUAL((LogicalXnor<int32_t,float>()( 0, 0.f)), 1);
    BOOST_CHECK_EQUAL((LogicalXnor<int32_t,float>()(-1, 0.f)), 0);
    BOOST_CHECK_EQUAL((LogicalXnor<int32_t,float>()( 0, 1.f)), 0);
    BOOST_CHECK_EQUAL((LogicalXnor<int32_t,float>()(-1, 1.f)), 1);

    BOOST_CHECK_EQUAL((LogicalXnor<double, uint16_t>()(0., 0)), 1.);
    BOOST_CHECK_EQUAL((LogicalXnor<double, uint16_t>()(1., 0)), 0.);
    BOOST_CHECK_EQUAL((LogicalXnor<double, uint16_t>()(0., 1)), 0.);
    BOOST_CHECK_EQUAL((LogicalXnor<double, uint16_t>()(1., 1)), 1.);

    BOOST_CHECK_EQUAL((LogicalXnor<uint8_t, float>()(0, 0.f)), 1);
    BOOST_CHECK_EQUAL((LogicalXnor<uint8_t, float>()(1, 0.f)), 0);
    BOOST_CHECK_EQUAL((LogicalXnor<uint8_t, float>()(0, 1.f)), 0);
    BOOST_CHECK_EQUAL((LogicalXnor<uint8_t, float>()(1, 1.f)), 1);

    BOOST_CHECK_EQUAL((LogicalXnor<bool, float>()(false, 0.f)), true);
    BOOST_CHECK_EQUAL((LogicalXnor<bool, float>()(true,  0.f)), false);
    BOOST_CHECK_EQUAL((LogicalXnor<bool, float>()(false, 1.f)), false);
    BOOST_CHECK_EQUAL((LogicalXnor<bool, float>()(true,  1.f)), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bitwise_or_test)
{
    //BOOST_CHECK_EQUAL(BitwiseOr<float>()(1.0f, 0.2f), 0.3f); // doesn't compile
    BOOST_CHECK_EQUAL(BitwiseOr<    bool>()(true, false), true);
    BOOST_CHECK_EQUAL(BitwiseOr< uint8_t>()(0x33, 0x3c), 0x03f);
    BOOST_CHECK_EQUAL(BitwiseOr<uint16_t>()(0x00ff, 0x0f0f), 0x0fff);
    BOOST_CHECK_EQUAL(BitwiseOr<uint32_t>()(0x00ff00ff,
                                            0x0f0f0f0f), 0x0fff0fff);
    BOOST_CHECK_EQUAL(BitwiseOr<uint64_t>()(0x00000000ffffffff,
                                            0x0000ffff0000ffff),
                      0x0000ffffffffffff);
    BOOST_CHECK_EQUAL(BitwiseOr< int8_t>()(0x33, 0x3c), 0x03f);
    BOOST_CHECK_EQUAL(BitwiseOr<int16_t>()(0x00ff, 0x0f0f), 0x0fff);
    BOOST_CHECK_EQUAL(BitwiseOr<int32_t>()(0x00ff00ff,
                                           0x0f0f0f0f), 0x0fff0fff);
    BOOST_CHECK_EQUAL(BitwiseOr<int64_t>()(0x00000000ffffffff,
                                           0x0000ffff0000ffff),
                      0x0000ffffffffffff);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bitwise_and_test)
{
    BOOST_CHECK_EQUAL(BitwiseAnd<    bool>()(true, false), false);
    BOOST_CHECK_EQUAL(BitwiseAnd< uint8_t>()(0x33, 0x3c), 0x30);
    BOOST_CHECK_EQUAL(BitwiseAnd<uint16_t>()(0x00ff, 0x0f0f), 0x000f);
    BOOST_CHECK_EQUAL(BitwiseAnd<uint32_t>()(0x00ff00ff,
                                             0x0f0f0f0f), 0x000f000f);
    BOOST_CHECK_EQUAL(BitwiseAnd<uint64_t>()(0x00000000ffffffff,
                                             0x0000ffff0000ffff),
                      0x000000000000ffff);
    BOOST_CHECK_EQUAL(BitwiseAnd< int8_t>()(0x33, 0x3c), 0x30);
    BOOST_CHECK_EQUAL(BitwiseAnd<int16_t>()(0x00ff, 0x0f0f), 0x000f);
    BOOST_CHECK_EQUAL(BitwiseAnd<int32_t>()(0x00ff00ff,
                                            0x0f0f0f0f), 0x000f000f);
    BOOST_CHECK_EQUAL(BitwiseAnd<int64_t>()(0x00000000ffffffff,
                                            0x0000ffff0000ffff),
                      0x000000000000ffff);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bitwise_xor_test)
{
    BOOST_CHECK_EQUAL(BitwiseXor<    bool>()(true, false), true);
    BOOST_CHECK_EQUAL(BitwiseXor< uint8_t>()(0x33, 0x3c), 0x0f);
    BOOST_CHECK_EQUAL(BitwiseXor<uint16_t>()(0x00ff, 0x0f0f), 0x0ff0);
    BOOST_CHECK_EQUAL(BitwiseXor<uint32_t>()(0x00ff00ff,
                                             0x0f0f0f0f), 0x0ff00ff0);
    BOOST_CHECK_EQUAL(BitwiseXor<uint64_t>()(0x00000000ffffffff,
                                             0x0000ffff0000ffff),
                      0x0000ffffffff0000);
    BOOST_CHECK_EQUAL(BitwiseXor< int8_t>()(0x33, 0x3c), 0x0f);
    BOOST_CHECK_EQUAL(BitwiseXor<int16_t>()(0x00ff, 0x0f0f), 0x0ff0);
    BOOST_CHECK_EQUAL(BitwiseXor<int32_t>()(0x00ff00ff,
                                            0x0f0f0f0f), 0x0ff00ff0);
    BOOST_CHECK_EQUAL(BitwiseXor<int64_t>()(0x00000000ffffffff,
                                            0x0000ffff0000ffff),
                      0x0000ffffffff0000);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bitwise_xnor_test)
{
    //weird but true
    BOOST_CHECK_EQUAL(BitwiseXnor<    bool>()(true, false), true);

    BOOST_CHECK_EQUAL(BitwiseXnor< uint8_t>()(0x33, 0x3c), 0xf0);
    BOOST_CHECK_EQUAL(BitwiseXnor<uint16_t>()(0x00ff, 0x0f0f), 0xf00f);
    BOOST_CHECK_EQUAL(BitwiseXnor<uint32_t>()(0x00ff00ff,
                                              0x0f0f0f0f), 0xf00ff00f);
    BOOST_CHECK_EQUAL(BitwiseXnor<uint64_t>()(0x00000000ffffffff,
                                              0x0000ffff0000ffff),
                      0xffff00000000ffff);
    BOOST_CHECK_EQUAL(BitwiseXnor< int8_t>()(0x33, 0x3c), (int8_t)0xf0);
    BOOST_CHECK_EQUAL(BitwiseXnor<int16_t>()(0x00ff, 0x0f0f), (int16_t)0xf00f);
    BOOST_CHECK_EQUAL(BitwiseXnor<int32_t>()(0x00ff00ff,
                                             0x0f0f0f0f), 0xf00ff00f);
    BOOST_CHECK_EQUAL(BitwiseXnor<int64_t>()(0x00000000ffffffff,
                                             0x0000ffff0000ffff),
                      0xffff00000000ffff);
}

BOOST_AUTO_TEST_SUITE_END()
