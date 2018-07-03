/*
 * GraphBLAS Template Library, Version 2.0
 *
 * Copyright 2018 Carnegie Mellon University, Battelle Memorial Institute, and
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

#include <functional>
#include <iostream>
#include <vector>

#include <graphblas/graphblas.hpp>

using namespace GraphBLAS;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE algebra_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

//****************************************************************************
// Misc tests from deprecated test_math.cpp file
BOOST_AUTO_TEST_CASE(misc_math_tests)
{
    BOOST_CHECK_EQUAL(Power<double>()(2, 5), 32.0);
    BOOST_CHECK_EQUAL(AdditiveInverse<double>()(-10), 10);
    BOOST_CHECK_EQUAL(AdditiveInverse<double>()(500), -500);
    BOOST_CHECK_EQUAL(MultiplicativeInverse<double>()(5), 1./5);
    BOOST_CHECK_EQUAL(Plus<double>()(2, 6), 8);
    BOOST_CHECK_EQUAL(Minus<double>()(2, 6), -4);
    BOOST_CHECK_EQUAL(Times<double>()(3, 10), 30);
    BOOST_CHECK_EQUAL(Div<double>()(500, 5), 100);
    BOOST_CHECK_EQUAL(Min<double>()(0, 1000000), 0);
    BOOST_CHECK_EQUAL(Min<double>()(-5, 0), -5);
    BOOST_CHECK_EQUAL(Min<double>()(7, 3), 3);
    BOOST_CHECK_EQUAL(Second<double>()(5, 1337), 1337);
    BOOST_CHECK_EQUAL(Equal<double>()(1, 1), true);
    BOOST_CHECK_EQUAL(Equal<double>()(0xC0FFEE, 0xCAFE), false);
    BOOST_CHECK_EQUAL(Xor<int>()(1, 2), 3);
    BOOST_CHECK_EQUAL(LogicalOr<int>()(1, 0), true);
    BOOST_CHECK_EQUAL(LogicalAnd<int>()(0, 2), false);
    BOOST_CHECK_EQUAL(LogicalNot<int>()(1), false);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(Identity_test)
{
    int8_t  i8  = 3;
    int16_t i16 = -400;
    int32_t i32 = -1001000;
    int64_t i64 = 5123123123;
    BOOST_CHECK_EQUAL(GraphBLAS::Identity<int8_t>()(i8),  i8);
    BOOST_CHECK_EQUAL(GraphBLAS::Identity<int16_t>()(i16), i16);
    BOOST_CHECK_EQUAL(GraphBLAS::Identity<int32_t>()(i32), i32);
    BOOST_CHECK_EQUAL(GraphBLAS::Identity<int64_t>()(i64), i64);

    uint8_t  ui8  = 3;
    uint16_t ui16 = 400;
    uint32_t ui32 = 1001000;
    uint64_t ui64 = 5123123123;
    BOOST_CHECK_EQUAL(GraphBLAS::Identity<uint8_t>()(ui8),  ui8);
    BOOST_CHECK_EQUAL(GraphBLAS::Identity<uint16_t>()(ui16), ui16);
    BOOST_CHECK_EQUAL(GraphBLAS::Identity<uint32_t>()(ui32), ui32);
    BOOST_CHECK_EQUAL(GraphBLAS::Identity<uint64_t>()(ui64), ui64);

    float   f32 = 3.14159;
    double  f64 = 2.718832654;
    BOOST_CHECK_EQUAL(GraphBLAS::Identity<float>()(f32), f32);
    BOOST_CHECK_EQUAL(GraphBLAS::Identity<double>()(f64), f64);

    BOOST_CHECK_EQUAL(GraphBLAS::Identity<uint32_t>()(f32), 3U);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(LogicalNot_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalNot<bool>()(true), false);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(PlusMonoid_test)
{
    uint8_t i8[3]={15, 22, 37};

    GraphBLAS::PlusMonoid<uint8_t> GrB_PLUS_INT8;
    BOOST_CHECK_EQUAL(GrB_PLUS_INT8.identity(), static_cast<uint8_t>(0));
    BOOST_CHECK_EQUAL(GrB_PLUS_INT8(i8[0], i8[1]), i8[2]);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(ArithmeticSemiring_test)
{
    uint32_t i32[]={15, 22, 15+22, 15*22};

    GraphBLAS::ArithmeticSemiring<uint32_t> GrB_PlusTimes_INT32;
    BOOST_CHECK_EQUAL(GrB_PlusTimes_INT32.zero(), static_cast<uint32_t>(0));
    BOOST_CHECK_EQUAL(GrB_PlusTimes_INT32.add(i32[0], i32[1]), i32[2]);
    BOOST_CHECK_EQUAL(GrB_PlusTimes_INT32.mult(i32[0], i32[1]), i32[3]);
}

//****************************************************************************
// Test Unary Operators
//****************************************************************************

BOOST_AUTO_TEST_CASE(identity_same_domain_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::Identity<double>()(2.2), 2.2);
    BOOST_CHECK_EQUAL(GraphBLAS::Identity<float>()(2.2f), 2.2f);

    BOOST_CHECK_EQUAL(GraphBLAS::Identity<uint64_t>()(2UL), 2UL);
    BOOST_CHECK_EQUAL(GraphBLAS::Identity<uint32_t>()(2U), 2U);
    BOOST_CHECK_EQUAL(GraphBLAS::Identity<uint16_t>()(2), 2);
    BOOST_CHECK_EQUAL(GraphBLAS::Identity<uint8_t>()(2), 2);

    BOOST_CHECK_EQUAL(GraphBLAS::Identity<int64_t>()(-2L), -2L);
    BOOST_CHECK_EQUAL(GraphBLAS::Identity<int32_t>()(-2), -2);
    BOOST_CHECK_EQUAL(GraphBLAS::Identity<int16_t>()(-2), -2);
    BOOST_CHECK_EQUAL(GraphBLAS::Identity<int8_t>()(-2), -2);

    BOOST_CHECK_EQUAL(GraphBLAS::Identity<bool>()(false), false);
    BOOST_CHECK_EQUAL(GraphBLAS::Identity<bool>()(true), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(identity_different_domain_test)
{
    /// @todo this is an incomplete set of pairs and does not cover the
    /// corner cases at the numeric limits that actually vary by platform
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<double,float>()(2.2)), 2.2f);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<double,uint64_t>()(2.2)), 2UL);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<double,int64_t>()(-2.2)), -2UL);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<double,bool>()(2.2)), true);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<double,bool>()(0.0)), false);

    BOOST_CHECK_EQUAL((GraphBLAS::Identity<float,double>()(2.2f)), static_cast<double>(2.2f));
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<float,uint32_t>()(2.2f)), 2U);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<float,int32_t>()(-2.2f)), -2);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<float,bool>()(2.2f)), true);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<float,bool>()(0.0f)), false);

    BOOST_CHECK_EQUAL((GraphBLAS::Identity<uint64_t, double>()(2UL)), 2.0);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<uint64_t, float>()(2UL)), 2.0f);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<uint64_t, int64_t>()(2UL)), 2UL);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<uint64_t,uint32_t>()(2UL)), 2UL);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<uint64_t,bool>()(2UL)), true);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<uint64_t,bool>()(0UL)), false);

    BOOST_CHECK_EQUAL((GraphBLAS::Identity<uint32_t, double>()(2U)), 2.0);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<uint32_t, float>()(2U)), 2.0f);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<uint32_t,uint64_t>()(2U)), 2UL);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<uint32_t, int64_t>()(2U)), 2L);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<uint32_t, int32_t>()(2U)), 2U);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<uint32_t,uint16_t>()(2U)), 2U);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<uint32_t, int16_t>()(2U)), 2);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<uint32_t,bool>()(2U)), true);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<uint32_t,bool>()(0U)), false);

    BOOST_CHECK_EQUAL((GraphBLAS::Identity<uint16_t, double>()(2U)), 2.0);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<uint16_t, float>()(2U)), 2.0f);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<uint16_t,uint32_t>()(2U)), 2U);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<uint16_t, int32_t>()(2U)), 2);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<uint16_t, int16_t>()(2)), 2);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<uint16_t,uint8_t>()(2)), 2);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<uint16_t, int8_t>()(2)), 2);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<uint16_t,bool>()(2U)), true);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<uint16_t,bool>()(0U)), false);

    BOOST_CHECK_EQUAL((GraphBLAS::Identity<uint8_t,double>()(2U)), 2.0);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<uint8_t,float>()(2U)), 2.0f);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<uint8_t,int8_t>()(2)), 2);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<uint8_t,uint16_t>()(2)), 2);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<uint8_t,int16_t>()(2)), 2);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<uint8_t,bool>()(2U)), true);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<uint8_t,bool>()(0U)), false);

    BOOST_CHECK_EQUAL((GraphBLAS::Identity<int64_t,double>()(-2L)), -2.0);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<int64_t,float>()(-2L)), -2.0f);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<int64_t,int32_t>()(-2L)), -2);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<int64_t,int16_t>()(-2L)), -2);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<int64_t,bool>()(-2L)), true);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<int64_t,bool>()(0L)), false);

    BOOST_CHECK_EQUAL((GraphBLAS::Identity<int32_t,double>()(-2)), -2.0);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<int32_t,float>()(-2)), -2.0f);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<int32_t,int64_t>()(-2)), -2);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<int32_t,int16_t>()(-2)), -2);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<int32_t,bool>()(-2)), true);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<int32_t,bool>()(0)), false);

    BOOST_CHECK_EQUAL((GraphBLAS::Identity<int16_t,double>()(-2)), -2.0);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<int16_t,float>()(-2)), -2.0f);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<int16_t,int32_t>()(-2)), -2);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<int16_t,int8_t>()(-2)), -2);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<int16_t,bool>()(-2)), true);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<int16_t,bool>()(0)), false);

    BOOST_CHECK_EQUAL((GraphBLAS::Identity<int8_t,double>()(-2)), -2.0);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<int8_t,float>()(-2)), -2.0f);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<int8_t,int16_t>()(-2)), -2);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<int8_t,bool>()(-2)), true);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<int8_t,bool>()(0)), false);

    BOOST_CHECK_EQUAL((GraphBLAS::Identity<bool,double>()(false)), 0.0);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<bool,double>()(true)), 1.0);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<bool,float>()(false)), 0.0f);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<bool,float>()(true)), 1.0f);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<bool,uint32_t>()(false)), 0U);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<bool,uint32_t>()(true)), 1U);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<bool,int32_t>()(false)), 0);
    BOOST_CHECK_EQUAL((GraphBLAS::Identity<bool,int32_t>()(true)), 1);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(logical_not_same_domain_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalNot<double>()(2.2), 0.0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalNot<double>()(0.0), 1.0);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalNot<float>()(2.2f), 0.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalNot<float>()(0.0f), 1.0f);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalNot<uint64_t>()(2UL), 0UL);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalNot<uint64_t>()(0UL), 1UL);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalNot<uint32_t>()(2U), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalNot<uint32_t>()(0U), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalNot<uint16_t>()(2), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalNot<uint16_t>()(0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalNot<uint8_t>()(2), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalNot<uint8_t>()(0), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalNot<int64_t>()(-2L), 0L);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalNot<int64_t>()(0L), 1L);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalNot<int32_t>()(-2), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalNot<int32_t>()(0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalNot<int16_t>()(-2), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalNot<int16_t>()(0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalNot<int8_t>()(-2), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalNot<int8_t>()(0), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalNot<bool>()(false), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalNot<bool>()(true), false);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(logical_not_different_domain_test)
{
    /// @todo this is an incomplete set of pairs and does not cover the
    /// corner cases at the numeric limits that actually vary by platform
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<double,float>()(2.2)), 0.0f);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<double,float>()(0.0)), 1.0f);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<double,uint64_t>()(2.2)), 0UL);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<double,uint64_t>()(0.0)), 1UL);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<double,int64_t>()(-2.2)), 0L);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<double,int64_t>()(0.0)), 1L);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<double,bool>()(2.2)), false);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<double,bool>()(0.0)), true);

    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<float,double>()(2.2f)), 0.0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<float,double>()(0.0f)), 1.0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<float,uint32_t>()(2.2f)), 0U);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<float,uint32_t>()(0.0f)), 1U);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<float,int32_t>()(-2.2f)), 0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<float,int32_t>()( 0.0f)), 1);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<float,bool>()(2.2f)), false);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<float,bool>()(0.0f)), true);

    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint64_t, double>()(2UL)), 0.0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint64_t, double>()(0UL)), 1.0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint64_t, float>()(2UL)), 0.0f);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint64_t, float>()(0UL)), 1.0f);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint64_t, int64_t>()(2UL)), 0L);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint64_t, int64_t>()(0UL)), 1L);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint64_t,uint32_t>()(2UL)), 0UL);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint64_t,uint32_t>()(0UL)), 1UL);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint64_t,bool>()(2UL)), false);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint64_t,bool>()(0UL)), true);

    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint32_t, double>()(2U)), 0.0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint32_t, double>()(0U)), 1.0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint32_t, float>()(2U)), 0.0f);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint32_t, float>()(0U)), 1.0f);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint32_t,uint64_t>()(2U)), 0UL);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint32_t,uint64_t>()(0U)), 1UL);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint32_t, int64_t>()(2U)), 0L);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint32_t, int64_t>()(0U)), 1L);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint32_t, int32_t>()(2U)), 0U);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint32_t, int32_t>()(0U)), 1U);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint32_t,uint16_t>()(2U)), 0U);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint32_t,uint16_t>()(0U)), 1U);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint32_t, int16_t>()(2U)), 0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint32_t, int16_t>()(0U)), 1);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint32_t,bool>()(2U)), false);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint32_t,bool>()(0U)), true);

    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint16_t, double>()(2U)), 0.0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint16_t, double>()(0U)), 1.0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint16_t, float>()(2U)), 0.0f);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint16_t, float>()(0U)), 1.0f);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint16_t,uint32_t>()(2U)), 0U);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint16_t,uint32_t>()(0U)), 1U);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint16_t, int32_t>()(2U)), 0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint16_t, int32_t>()(0U)), 1);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint16_t, int16_t>()(2)), 0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint16_t, int16_t>()(0)), 1);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint16_t,uint8_t>()(2)), 0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint16_t,uint8_t>()(0)), 1);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint16_t, int8_t>()(2)), 0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint16_t, int8_t>()(0)), 1);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint16_t,bool>()(2U)), false);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint16_t,bool>()(0U)), true);

    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint8_t,double>()(2U)), 0.0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint8_t,double>()(0U)), 1.0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint8_t,float>()(2U)), 0.0f);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint8_t,float>()(0U)), 1.0f);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint8_t,int8_t>()(2)), 0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint8_t,int8_t>()(0)), 1);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint8_t,uint16_t>()(2)), 0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint8_t,uint16_t>()(0)), 1);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint8_t,int16_t>()(2)), 0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint8_t,int16_t>()(0)), 1);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint8_t,bool>()(2U)), false);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<uint8_t,bool>()(0U)), true);

    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int64_t,double>()(-2L)), 0.0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int64_t,double>()(0L)), 1.0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int64_t,float>()(-2L)), 0.0f);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int64_t,float>()(0L)), 1.0f);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int64_t,int32_t>()(-2L)), 0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int64_t,int32_t>()(0L)), 1);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int64_t,int16_t>()(-2L)), 0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int64_t,int16_t>()(0L)), 1);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int64_t,bool>()(-2L)), false);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int64_t,bool>()(0L)), true);

    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int32_t,double>()(-2)), 0.0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int32_t,double>()(0)), 1.0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int32_t,float>()(-2)), 0.0f);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int32_t,float>()(0)), 1.0f);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int32_t,int64_t>()(-2)), 0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int32_t,int64_t>()(0)), 1);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int32_t,int16_t>()(-2)), 0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int32_t,int16_t>()(0)), 1);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int32_t,bool>()(-2)), false);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int32_t,bool>()(0)), true);

    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int16_t,double>()(-2)), 0.0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int16_t,double>()(0)), 1.0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int16_t,float>()(-2)), 0.0f);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int16_t,float>()(0)), 1.0f);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int16_t,int32_t>()(-2)), 0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int16_t,int32_t>()(0)), 1);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int16_t,int8_t>()(-2)), 0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int16_t,int8_t>()(0)), 1);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int16_t,bool>()(-2)), false);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int16_t,bool>()(0)), true);

    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int8_t,double>()(-2)), 0.0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int8_t,double>()(0)), 1.0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int8_t,float>()(-2)), 0.0f);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int8_t,float>()(0)), 1.0f);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int8_t,int16_t>()(-2)), 0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int8_t,int16_t>()(0)), 1);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int8_t,bool>()(-2)), false);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<int8_t,bool>()(0)), true);

    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<bool,double>()(false)), 1.0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<bool,double>()(true)), 0.0);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<bool,float>()(false)), 1.0f);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<bool,float>()(true)), 0.0f);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<bool,uint32_t>()(false)), 1U);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<bool,uint32_t>()(true)), 0U);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<bool,int32_t>()(false)), 1);
    BOOST_CHECK_EQUAL((GraphBLAS::LogicalNot<bool,int32_t>()(true)), 0);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(additive_inverse_same_domain_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<double>()(2.2), -2.2);
    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<double>()(0.0), 0.0);
    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<double>()(-2.0), 2.0);

    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<float>()(2.2f), -2.2f);
    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<float>()(0.0f), 0.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<float>()(-2.2f), 2.2f);

    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<uint64_t>()(2UL), static_cast<uint64_t>(-2L));
    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<uint64_t>()(0UL), 0UL);
    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<uint64_t>()(-2), 2UL);
    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<uint32_t>()(2U), static_cast<uint32_t>(-2));
    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<uint32_t>()(0U), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<uint32_t>()(-2), 2U);
    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<uint16_t>()(2), static_cast<uint16_t>(-2));
    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<uint16_t>()(0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<uint16_t>()(-2), 2);
    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<uint8_t>()(2), static_cast<uint8_t>(-2));
    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<uint8_t>()(0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<uint8_t>()(-2), 2);

    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<int64_t>()(-2L), 2L);
    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<int64_t>()(0L), 0L);
    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<int64_t>()(2L), -2L);
    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<int32_t>()(-2), 2);
    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<int32_t>()(0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<int32_t>()(2), -2);
    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<int16_t>()(-2), 2);
    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<int16_t>()(0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<int16_t>()(2), -2);
    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<int8_t>()(-2), 2);
    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<int8_t>()(0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<int8_t>()(2), -2);

    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<bool>()(false), false);
    BOOST_CHECK_EQUAL(GraphBLAS::AdditiveInverse<bool>()(true), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(additive_inverse_different_domain_test)
{
    /// @todo
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(multiplicative_inverse_same_domain_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::MultiplicativeInverse<double>()(2.2), 1./2.2);
    BOOST_CHECK_EQUAL(GraphBLAS::MultiplicativeInverse<double>()(-2.2), -1./2.2);

    BOOST_CHECK_EQUAL(GraphBLAS::MultiplicativeInverse<float>()(2.2f), 1.f/2.2f);
    BOOST_CHECK_EQUAL(GraphBLAS::MultiplicativeInverse<float>()(-2.2f), -1.f/2.2f);

    BOOST_CHECK_EQUAL(GraphBLAS::MultiplicativeInverse<uint64_t>()(2UL), 0UL);
    BOOST_CHECK_EQUAL(GraphBLAS::MultiplicativeInverse<uint32_t>()(2U), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::MultiplicativeInverse<uint16_t>()(2), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::MultiplicativeInverse<uint8_t>()(2), 0);

    BOOST_CHECK_EQUAL(GraphBLAS::MultiplicativeInverse<int64_t>()(-2L), 0L);
    BOOST_CHECK_EQUAL(GraphBLAS::MultiplicativeInverse<int32_t>()(-2), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::MultiplicativeInverse<int16_t>()(-2), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::MultiplicativeInverse<int8_t>()(-2), 0);

    BOOST_CHECK_EQUAL(GraphBLAS::MultiplicativeInverse<bool>()(true), true);

    // divide by zero
    //BOOST_CHECK_EQUAL(GraphBLAS::MultiplicativeInverse<bool>()(false), false);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(multiplicative_inverse_different_domain_test)
{
    //// @todo
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(logical_or_same_domain_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<double>()(0.0, 0.0), 0.0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<double>()(1.0, 0.0), 1.0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<double>()(0.0, 1.0), 1.0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<double>()(1.0, 1.0), 1.0);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<float>()(0.0f, 0.0f), 0.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<float>()(1.0f, 0.0f), 1.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<float>()(0.0f, 1.0f), 1.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<float>()(1.0f, 1.0f), 1.0f);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<uint64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<uint64_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<uint64_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<uint64_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<uint32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<uint32_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<uint32_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<uint32_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<uint16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<uint16_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<uint16_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<uint16_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<uint8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<uint8_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<uint8_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<uint8_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<int64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<int64_t>()(-1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<int64_t>()(0, -1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<int64_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<int32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<int32_t>()(-1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<int32_t>()(0, -1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<int32_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<int16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<int16_t>()(-1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<int16_t>()(0, -1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<int16_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<int8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<int8_t>()(-1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<int8_t>()(0, -1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<int8_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<bool>()(false, true),  true);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<bool>()(true, false),  true);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOr<bool>()(true, true),   true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(logical_and_same_domain_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<double>()(0.0, 0.0), 0.0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<double>()(1.0, 0.0), 0.0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<double>()(0.0, 1.0), 0.0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<double>()(1.0, 1.0), 1.0);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<float>()(0.0f, 0.0f), 0.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<float>()(1.0f, 0.0f), 0.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<float>()(0.0f, 1.0f), 0.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<float>()(1.0f, 1.0f), 1.0f);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<uint64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<uint64_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<uint64_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<uint64_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<uint32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<uint32_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<uint32_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<uint32_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<uint16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<uint16_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<uint16_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<uint16_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<uint8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<uint8_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<uint8_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<uint8_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<int64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<int64_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<int64_t>()(0, -1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<int64_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<int32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<int32_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<int32_t>()(0, -1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<int32_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<int16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<int16_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<int16_t>()(0, -1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<int16_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<int8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<int8_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<int8_t>()(0, -1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<int8_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<bool>()(false, true),  false);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<bool>()(true, false),  false);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalAnd<bool>()(true, true),   true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(logical_xor_same_domain_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<double>()(0.0, 0.0), 0.0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<double>()(1.0, 0.0), 1.0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<double>()(0.0, 1.0), 1.0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<double>()(1.0, 1.0), 0.0);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<float>()(0.0f, 0.0f), 0.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<float>()(1.0f, 0.0f), 1.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<float>()(0.0f, 1.0f), 1.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<float>()(1.0f, 1.0f), 0.0f);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<uint64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<uint64_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<uint64_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<uint64_t>()(1, 1), 0);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<uint32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<uint32_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<uint32_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<uint32_t>()(1, 1), 0);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<uint16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<uint16_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<uint16_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<uint16_t>()(1, 1), 0);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<uint8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<uint8_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<uint8_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<uint8_t>()(1, 1), 0);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<int64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<int64_t>()(-1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<int64_t>()(0, -1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<int64_t>()(-1, -1), 0);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<int32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<int32_t>()(-1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<int32_t>()(0, -1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<int32_t>()(-1, -1), 0);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<int16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<int16_t>()(-1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<int16_t>()(0, -1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<int16_t>()(-1, -1), 0);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<int8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<int8_t>()(-1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<int8_t>()(0, -1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<int8_t>()(-1, -1), 0);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<bool>()(false, true),  true);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<bool>()(true, false),  true);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalXor<bool>()(true, true),   false);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(equal_same_domain_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<double>()(0.0, 0.0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<double>()(1.0, 0.0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<double>()(0.0, 1.0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<double>()(1.0, 1.0), true);

    BOOST_CHECK_EQUAL(GraphBLAS::Equal<float>()(0.0f, 0.0f), true);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<float>()(1.0f, 0.0f), false);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<float>()(0.0f, 1.0f), false);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<float>()(1.0f, 1.0f), true);

    BOOST_CHECK_EQUAL(GraphBLAS::Equal<uint64_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<uint64_t>()(1, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<uint64_t>()(0, 1), false);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<uint64_t>()(1, 1), true);

    BOOST_CHECK_EQUAL(GraphBLAS::Equal<uint32_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<uint32_t>()(1, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<uint32_t>()(0, 1), false);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<uint32_t>()(1, 1), true);

    BOOST_CHECK_EQUAL(GraphBLAS::Equal<uint16_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<uint16_t>()(1, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<uint16_t>()(0, 1), false);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<uint16_t>()(1, 1), true);

    BOOST_CHECK_EQUAL(GraphBLAS::Equal<uint8_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<uint8_t>()(1, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<uint8_t>()(0, 1), false);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<uint8_t>()(1, 1), true);

    BOOST_CHECK_EQUAL(GraphBLAS::Equal<int64_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<int64_t>()(-1, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<int64_t>()(0, -1), false);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<int64_t>()(-1, -1), true);

    BOOST_CHECK_EQUAL(GraphBLAS::Equal<int32_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<int32_t>()(-1, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<int32_t>()(0, -1), false);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<int32_t>()(-1, -1), true);

    BOOST_CHECK_EQUAL(GraphBLAS::Equal<int16_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<int16_t>()(-1, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<int16_t>()(0, -1), false);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<int16_t>()(-1, -1), true);

    BOOST_CHECK_EQUAL(GraphBLAS::Equal<int8_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<int8_t>()(-1, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<int8_t>()(0, -1), false);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<int8_t>()(-1, -1), true);

    BOOST_CHECK_EQUAL(GraphBLAS::Equal<bool>()(false, false), true);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<bool>()(false, true),  false);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<bool>()(true, false),  false);
    BOOST_CHECK_EQUAL(GraphBLAS::Equal<bool>()(true, true),   true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(not_equal_same_domain_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<double>()(0.0, 0.0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<double>()(1.0, 0.0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<double>()(0.0, 1.0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<double>()(1.0, 1.0), false);

    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<float>()(0.0f, 0.0f), false);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<float>()(1.0f, 0.0f), true);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<float>()(0.0f, 1.0f), true);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<float>()(1.0f, 1.0f), false);

    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<uint64_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<uint64_t>()(1, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<uint64_t>()(0, 1), true);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<uint64_t>()(1, 1), false);

    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<uint32_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<uint32_t>()(1, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<uint32_t>()(0, 1), true);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<uint32_t>()(1, 1), false);

    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<uint16_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<uint16_t>()(1, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<uint16_t>()(0, 1), true);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<uint16_t>()(1, 1), false);

    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<uint8_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<uint8_t>()(1, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<uint8_t>()(0, 1), true);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<uint8_t>()(1, 1), false);

    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<int64_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<int64_t>()(-1, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<int64_t>()(0, -1), true);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<int64_t>()(-1, -1), false);

    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<int32_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<int32_t>()(-1, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<int32_t>()(0, -1), true);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<int32_t>()(-1, -1), false);

    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<int16_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<int16_t>()(-1, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<int16_t>()(0, -1), true);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<int16_t>()(-1, -1), false);

    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<int8_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<int8_t>()(-1, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<int8_t>()(0, -1), true);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<int8_t>()(-1, -1), false);

    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<bool>()(false, true),  true);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<bool>()(true, false),  true);
    BOOST_CHECK_EQUAL(GraphBLAS::NotEqual<bool>()(true, true),   false);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(greater_than_same_domain_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<double>()(0.0, 0.0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<double>()(1.0, 0.0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<double>()(0.0, 1.0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<double>()(1.0, 1.0), false);

    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<float>()(0.0f, 0.0f), false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<float>()(1.0f, 0.0f), true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<float>()(0.0f, 1.0f), false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<float>()(1.0f, 1.0f), false);

    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<uint64_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<uint64_t>()(1, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<uint64_t>()(0, 1), false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<uint64_t>()(1, 1), false);

    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<uint32_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<uint32_t>()(1, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<uint32_t>()(0, 1), false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<uint32_t>()(1, 1), false);

    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<uint16_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<uint16_t>()(1, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<uint16_t>()(0, 1), false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<uint16_t>()(1, 1), false);

    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<uint8_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<uint8_t>()(1, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<uint8_t>()(0, 1), false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<uint8_t>()(1, 1), false);

    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<int64_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<int64_t>()(-1, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<int64_t>()(0, -1), true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<int64_t>()(-1, -1), false);

    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<int32_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<int32_t>()(-1, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<int32_t>()(0, -1), true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<int32_t>()(-1, -1), false);

    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<int16_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<int16_t>()(-1, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<int16_t>()(0, -1), true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<int16_t>()(-1, -1), false);

    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<int8_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<int8_t>()(-1, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<int8_t>()(0, -1), true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<int8_t>()(-1, -1), false);

    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<bool>()(false, true),  false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<bool>()(true, false),  true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterThan<bool>()(true, true),   false);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(less_than_same_domain_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<double>()(0.0, 0.0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<double>()(1.0, 0.0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<double>()(0.0, 1.0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<double>()(1.0, 1.0), false);

    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<float>()(0.0f, 0.0f), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<float>()(1.0f, 0.0f), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<float>()(0.0f, 1.0f), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<float>()(1.0f, 1.0f), false);

    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<uint64_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<uint64_t>()(1, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<uint64_t>()(0, 1), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<uint64_t>()(1, 1), false);

    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<uint32_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<uint32_t>()(1, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<uint32_t>()(0, 1), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<uint32_t>()(1, 1), false);

    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<uint16_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<uint16_t>()(1, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<uint16_t>()(0, 1), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<uint16_t>()(1, 1), false);

    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<uint8_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<uint8_t>()(1, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<uint8_t>()(0, 1), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<uint8_t>()(1, 1), false);

    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<int64_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<int64_t>()(-1, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<int64_t>()(0, -1), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<int64_t>()(-1, -1), false);

    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<int32_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<int32_t>()(-1, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<int32_t>()(0, -1), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<int32_t>()(-1, -1), false);

    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<int16_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<int16_t>()(-1, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<int16_t>()(0, -1), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<int16_t>()(-1, -1), false);

    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<int8_t>()(0, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<int8_t>()(-1, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<int8_t>()(0, -1), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<int8_t>()(-1, -1), false);

    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<bool>()(false, true),  true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<bool>()(true, false),  false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessThan<bool>()(true, true),   false);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(greater_equal_same_domain_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<double>()(0.0, 0.0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<double>()(1.0, 0.0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<double>()(0.0, 1.0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<double>()(1.0, 1.0), true);

    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<float>()(0.0f, 0.0f), true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<float>()(1.0f, 0.0f), true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<float>()(0.0f, 1.0f), false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<float>()(1.0f, 1.0f), true);

    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<uint64_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<uint64_t>()(1, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<uint64_t>()(0, 1), false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<uint64_t>()(1, 1), true);

    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<uint32_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<uint32_t>()(1, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<uint32_t>()(0, 1), false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<uint32_t>()(1, 1), true);

    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<uint16_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<uint16_t>()(1, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<uint16_t>()(0, 1), false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<uint16_t>()(1, 1), true);

    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<uint8_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<uint8_t>()(1, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<uint8_t>()(0, 1), false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<uint8_t>()(1, 1), true);

    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<int64_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<int64_t>()(-1, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<int64_t>()(0, -1), true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<int64_t>()(-1, -1), true);

    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<int32_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<int32_t>()(-1, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<int32_t>()(0, -1), true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<int32_t>()(-1, -1), true);

    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<int16_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<int16_t>()(-1, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<int16_t>()(0, -1), true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<int16_t>()(-1, -1), true);

    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<int8_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<int8_t>()(-1, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<int8_t>()(0, -1), true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<int8_t>()(-1, -1), true);

    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<bool>()(false, false), true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<bool>()(false, true),  false);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<bool>()(true, false),  true);
    BOOST_CHECK_EQUAL(GraphBLAS::GreaterEqual<bool>()(true, true),   true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(less_equal_same_domain_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<double>()(0.0, 0.0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<double>()(1.0, 0.0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<double>()(0.0, 1.0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<double>()(1.0, 1.0), true);

    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<float>()(0.0f, 0.0f), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<float>()(1.0f, 0.0f), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<float>()(0.0f, 1.0f), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<float>()(1.0f, 1.0f), true);

    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<uint64_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<uint64_t>()(1, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<uint64_t>()(0, 1), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<uint64_t>()(1, 1), true);

    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<uint32_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<uint32_t>()(1, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<uint32_t>()(0, 1), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<uint32_t>()(1, 1), true);

    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<uint16_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<uint16_t>()(1, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<uint16_t>()(0, 1), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<uint16_t>()(1, 1), true);

    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<uint8_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<uint8_t>()(1, 0), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<uint8_t>()(0, 1), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<uint8_t>()(1, 1), true);

    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<int64_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<int64_t>()(-1, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<int64_t>()(0, -1), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<int64_t>()(-1, -1), true);

    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<int32_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<int32_t>()(-1, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<int32_t>()(0, -1), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<int32_t>()(-1, -1), true);

    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<int16_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<int16_t>()(-1, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<int16_t>()(0, -1), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<int16_t>()(-1, -1), true);

    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<int8_t>()(0, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<int8_t>()(-1, 0), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<int8_t>()(0, -1), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<int8_t>()(-1, -1), true);

    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<bool>()(false, false), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<bool>()(false, true),  true);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<bool>()(true, false),  false);
    BOOST_CHECK_EQUAL(GraphBLAS::LessEqual<bool>()(true, true),   true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(first_same_domain_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::First<double>()(0.0, 0.0), 0.0);
    BOOST_CHECK_EQUAL(GraphBLAS::First<double>()(1.0, 0.0), 1.0);
    BOOST_CHECK_EQUAL(GraphBLAS::First<double>()(0.0, 1.0), 0.0);
    BOOST_CHECK_EQUAL(GraphBLAS::First<double>()(1.0, 1.0), 1.0);

    BOOST_CHECK_EQUAL(GraphBLAS::First<float>()(0.0f, 0.0f), 0.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::First<float>()(1.0f, 0.0f), 1.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::First<float>()(0.0f, 1.0f), 0.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::First<float>()(1.0f, 1.0f), 1.0f);

    BOOST_CHECK_EQUAL(GraphBLAS::First<uint64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::First<uint64_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::First<uint64_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::First<uint64_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::First<uint32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::First<uint32_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::First<uint32_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::First<uint32_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::First<uint16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::First<uint16_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::First<uint16_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::First<uint16_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::First<uint8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::First<uint8_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::First<uint8_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::First<uint8_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::First<int64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::First<int64_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::First<int64_t>()(0, -1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::First<int64_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(GraphBLAS::First<int32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::First<int32_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::First<int32_t>()(0, -1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::First<int32_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(GraphBLAS::First<int16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::First<int16_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::First<int16_t>()(0, -1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::First<int16_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(GraphBLAS::First<int8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::First<int8_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::First<int8_t>()(0, -1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::First<int8_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(GraphBLAS::First<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(GraphBLAS::First<bool>()(false, true),  false);
    BOOST_CHECK_EQUAL(GraphBLAS::First<bool>()(true, false),  true);
    BOOST_CHECK_EQUAL(GraphBLAS::First<bool>()(true, true),   true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(second_same_domain_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::Second<double>()(0.0, 0.0), 0.0);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<double>()(1.0, 0.0), 0.0);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<double>()(0.0, 1.0), 1.0);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<double>()(1.0, 1.0), 1.0);

    BOOST_CHECK_EQUAL(GraphBLAS::Second<float>()(0.0f, 0.0f), 0.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<float>()(1.0f, 0.0f), 0.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<float>()(0.0f, 1.0f), 1.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<float>()(1.0f, 1.0f), 1.0f);

    BOOST_CHECK_EQUAL(GraphBLAS::Second<uint64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<uint64_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<uint64_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<uint64_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::Second<uint32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<uint32_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<uint32_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<uint32_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::Second<uint16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<uint16_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<uint16_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<uint16_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::Second<uint8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<uint8_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<uint8_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<uint8_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::Second<int64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<int64_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<int64_t>()(0, -1), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<int64_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(GraphBLAS::Second<int32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<int32_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<int32_t>()(0, -1), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<int32_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(GraphBLAS::Second<int16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<int16_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<int16_t>()(0, -1), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<int16_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(GraphBLAS::Second<int8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<int8_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<int8_t>()(0, -1), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<int8_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(GraphBLAS::Second<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<bool>()(false, true),  true);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<bool>()(true, false),  false);
    BOOST_CHECK_EQUAL(GraphBLAS::Second<bool>()(true, true),   true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(min_same_domain_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::Min<double>()(0.0, 0.0), 0.0);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<double>()(1.0, 0.0), 0.0);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<double>()(0.0, 1.0), 0.0);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<double>()(1.0, 1.0), 1.0);

    BOOST_CHECK_EQUAL(GraphBLAS::Min<float>()(0.0f, 0.0f), 0.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<float>()(1.0f, 0.0f), 0.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<float>()(0.0f, 1.0f), 0.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<float>()(1.0f, 1.0f), 1.0f);

    BOOST_CHECK_EQUAL(GraphBLAS::Min<uint64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<uint64_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<uint64_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<uint64_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::Min<uint32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<uint32_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<uint32_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<uint32_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::Min<uint16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<uint16_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<uint16_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<uint16_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::Min<uint8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<uint8_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<uint8_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<uint8_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::Min<int64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<int64_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<int64_t>()(0, -1), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<int64_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(GraphBLAS::Min<int32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<int32_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<int32_t>()(0, -1), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<int32_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(GraphBLAS::Min<int16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<int16_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<int16_t>()(0, -1), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<int16_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(GraphBLAS::Min<int8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<int8_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<int8_t>()(0, -1), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<int8_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(GraphBLAS::Min<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<bool>()(false, true),  false);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<bool>()(true, false),  false);
    BOOST_CHECK_EQUAL(GraphBLAS::Min<bool>()(true, true),   true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(max_same_domain_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::Max<double>()(0.0, 0.0), 0.0);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<double>()(1.0, 0.0), 1.0);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<double>()(0.0, 1.0), 1.0);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<double>()(1.0, 1.0), 1.0);

    BOOST_CHECK_EQUAL(GraphBLAS::Max<float>()(0.0f, 0.0f), 0.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<float>()(1.0f, 0.0f), 1.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<float>()(0.0f, 1.0f), 1.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<float>()(1.0f, 1.0f), 1.0f);

    BOOST_CHECK_EQUAL(GraphBLAS::Max<uint64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<uint64_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<uint64_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<uint64_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::Max<uint32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<uint32_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<uint32_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<uint32_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::Max<uint16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<uint16_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<uint16_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<uint16_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::Max<uint8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<uint8_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<uint8_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<uint8_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::Max<int64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<int64_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<int64_t>()(0, -1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<int64_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(GraphBLAS::Max<int32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<int32_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<int32_t>()(0, -1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<int32_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(GraphBLAS::Max<int16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<int16_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<int16_t>()(0, -1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<int16_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(GraphBLAS::Max<int8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<int8_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<int8_t>()(0, -1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<int8_t>()(-1, -1), -1);

    BOOST_CHECK_EQUAL(GraphBLAS::Max<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<bool>()(false, true),  true);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<bool>()(true, false),  true);
    BOOST_CHECK_EQUAL(GraphBLAS::Max<bool>()(true, true),   true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(plus_same_domain_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<double>()(0.0, 0.0), 0.0);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<double>()(1.0, 0.0), 1.0);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<double>()(0.0, 1.0), 1.0);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<double>()(1.0, 1.0), 2.0);

    BOOST_CHECK_EQUAL(GraphBLAS::Plus<float>()(0.0f, 0.0f), 0.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<float>()(1.0f, 0.0f), 1.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<float>()(0.0f, 1.0f), 1.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<float>()(1.0f, 1.0f), 2.0f);

    BOOST_CHECK_EQUAL(GraphBLAS::Plus<uint64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<uint64_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<uint64_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<uint64_t>()(1, 1), 2);

    BOOST_CHECK_EQUAL(GraphBLAS::Plus<uint32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<uint32_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<uint32_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<uint32_t>()(1, 1), 2);

    BOOST_CHECK_EQUAL(GraphBLAS::Plus<uint16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<uint16_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<uint16_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<uint16_t>()(1, 1), 2);

    BOOST_CHECK_EQUAL(GraphBLAS::Plus<uint8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<uint8_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<uint8_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<uint8_t>()(1, 1), 2);

    BOOST_CHECK_EQUAL(GraphBLAS::Plus<int64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<int64_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<int64_t>()(0, -1), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<int64_t>()(-1, -1), -2);

    BOOST_CHECK_EQUAL(GraphBLAS::Plus<int32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<int32_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<int32_t>()(0, -1), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<int32_t>()(-1, -1), -2);

    BOOST_CHECK_EQUAL(GraphBLAS::Plus<int16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<int16_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<int16_t>()(0, -1), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<int16_t>()(-1, -1), -2);

    BOOST_CHECK_EQUAL(GraphBLAS::Plus<int8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<int8_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<int8_t>()(0, -1), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<int8_t>()(-1, -1), -2);

    BOOST_CHECK_EQUAL(GraphBLAS::Plus<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<bool>()(false, true),  true);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<bool>()(true, false),  true);
    BOOST_CHECK_EQUAL(GraphBLAS::Plus<bool>()(true, true),   true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(minus_same_domain_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<double>()(0.0, 0.0), 0.0);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<double>()(1.0, 0.0), 1.0);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<double>()(2.0, 1.0), 1.0);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<double>()(1.0, 1.0), 0.0);

    BOOST_CHECK_EQUAL(GraphBLAS::Minus<float>()(0.0f, 0.0f), 0.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<float>()(1.0f, 0.0f), 1.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<float>()(2.0f, 1.0f), 1.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<float>()(1.0f, 1.0f), 0.0f);

    BOOST_CHECK_EQUAL(GraphBLAS::Minus<uint64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<uint64_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<uint64_t>()(2, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<uint64_t>()(1, 1), 0);

    BOOST_CHECK_EQUAL(GraphBLAS::Minus<uint32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<uint32_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<uint32_t>()(2, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<uint32_t>()(1, 1), 0);

    BOOST_CHECK_EQUAL(GraphBLAS::Minus<uint16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<uint16_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<uint16_t>()(2, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<uint16_t>()(1, 1), 0);

    BOOST_CHECK_EQUAL(GraphBLAS::Minus<uint8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<uint8_t>()(1, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<uint8_t>()(2, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<uint8_t>()(1, 1), 0);

    BOOST_CHECK_EQUAL(GraphBLAS::Minus<int64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<int64_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<int64_t>()(-2, -1), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<int64_t>()(-1, -1), 0);

    BOOST_CHECK_EQUAL(GraphBLAS::Minus<int32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<int32_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<int32_t>()(-2, -1), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<int32_t>()(-1, -1), 0);

    BOOST_CHECK_EQUAL(GraphBLAS::Minus<int16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<int16_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<int16_t>()(-2, -1), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<int16_t>()(-1, -1), 0);

    BOOST_CHECK_EQUAL(GraphBLAS::Minus<int8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<int8_t>()(-1, 0), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<int8_t>()(-2, -1), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<int8_t>()(-1, -1), 0);

    BOOST_CHECK_EQUAL(GraphBLAS::Minus<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<bool>()(false, true),  true);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<bool>()(true, false),  true);
    BOOST_CHECK_EQUAL(GraphBLAS::Minus<bool>()(true, true),   false);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(times_same_domain_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::Times<double>()(0.0, 0.0), 0.0);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<double>()(1.0, 0.0), 0.0);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<double>()(2.0, 1.0), 2.0);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<double>()(1.0, 1.0), 1.0);

    BOOST_CHECK_EQUAL(GraphBLAS::Times<float>()(0.0f, 0.0f), 0.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<float>()(1.0f, 0.0f), 0.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<float>()(2.0f, 1.0f), 2.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<float>()(1.0f, 1.0f), 1.0f);

    BOOST_CHECK_EQUAL(GraphBLAS::Times<uint64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<uint64_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<uint64_t>()(2, 1), 2);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<uint64_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::Times<uint32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<uint32_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<uint32_t>()(2, 1), 2);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<uint32_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::Times<uint16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<uint16_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<uint16_t>()(2, 1), 2);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<uint16_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::Times<uint8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<uint8_t>()(1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<uint8_t>()(2, 1), 2);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<uint8_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::Times<int64_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<int64_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<int64_t>()(-2, 1), -2);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<int64_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::Times<int32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<int32_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<int32_t>()(-2, 1), -2);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<int32_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::Times<int16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<int16_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<int16_t>()(-2, 1), -2);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<int16_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::Times<int8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<int8_t>()(-1, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<int8_t>()(-2, 1), -2);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<int8_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::Times<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<bool>()(false, true),  false);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<bool>()(true, false),  false);
    BOOST_CHECK_EQUAL(GraphBLAS::Times<bool>()(true, true),   true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(div_same_domain_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::Div<double>()(0.0, 1.0), 0.0);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<double>()(1.0, 2.0), 0.5);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<double>()(2.0, 1.0), 2.0);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<double>()(1.0, 1.0), 1.0);

    BOOST_CHECK_EQUAL(GraphBLAS::Div<float>()(0.0f, 1.0f), 0.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<float>()(1.0f, 2.0f), 0.5f);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<float>()(2.0f, 1.0f), 2.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<float>()(1.0f, 1.0f), 1.0f);

    BOOST_CHECK_EQUAL(GraphBLAS::Div<uint64_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<uint64_t>()(1, 2), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<uint64_t>()(2, 1), 2);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<uint64_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::Div<uint32_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<uint32_t>()(1, 2), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<uint32_t>()(2, 1), 2);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<uint32_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::Div<uint16_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<uint16_t>()(1, 2), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<uint16_t>()(2, 1), 2);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<uint16_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::Div<uint8_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<uint8_t>()(1, 2), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<uint8_t>()(2, 1), 2);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<uint8_t>()(1, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::Div<int64_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<int64_t>()(-1, 2), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<int64_t>()(-2, 1), -2);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<int64_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::Div<int32_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<int32_t>()(-1, 2), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<int32_t>()(-2, 1), -2);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<int32_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::Div<int16_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<int16_t>()(-1, 2), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<int16_t>()(-2, 1), -2);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<int16_t>()(-1, -1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::Div<int8_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<int8_t>()(-1, 2), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<int8_t>()(-2, 1), -2);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<int8_t>()(-1, -1), 1);

    //BOOST_CHECK_EQUAL(GraphBLAS::Div<bool>()(false, false), ?);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<bool>()(false, true),  false);
    //BOOST_CHECK_EQUAL(GraphBLAS::Div<bool>()(true, false),  ?);
    BOOST_CHECK_EQUAL(GraphBLAS::Div<bool>()(true, true),   true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(plus_monoid_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::PlusMonoid<double>().identity(), 0.0);
    BOOST_CHECK_EQUAL(GraphBLAS::PlusMonoid<double>()(-2., 1.), -1.0);
    BOOST_CHECK_EQUAL(GraphBLAS::PlusMonoid<float>().identity(), 0.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::PlusMonoid<float>()(-2.f, 1.f), -1.0f);

    BOOST_CHECK_EQUAL(GraphBLAS::PlusMonoid<uint64_t>().identity(), 0UL);
    BOOST_CHECK_EQUAL(GraphBLAS::PlusMonoid<uint64_t>()(2UL, 1UL), 3UL);
    BOOST_CHECK_EQUAL(GraphBLAS::PlusMonoid<uint32_t>().identity(), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::PlusMonoid<uint32_t>()(2U, 1U), 3U);
    BOOST_CHECK_EQUAL(GraphBLAS::PlusMonoid<uint16_t>().identity(), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::PlusMonoid<uint16_t>()(2U, 1U), 3U);
    BOOST_CHECK_EQUAL(GraphBLAS::PlusMonoid<uint8_t>().identity(), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::PlusMonoid<uint8_t>()(2U, 1U), 3U);

    BOOST_CHECK_EQUAL(GraphBLAS::PlusMonoid<int64_t>().identity(), 0L);
    BOOST_CHECK_EQUAL(GraphBLAS::PlusMonoid<int64_t>()(-2L, 1L), -1L);
    BOOST_CHECK_EQUAL(GraphBLAS::PlusMonoid<int32_t>().identity(), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::PlusMonoid<int32_t>()(-2, 1), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::PlusMonoid<int16_t>().identity(), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::PlusMonoid<int16_t>()(-2, 1), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::PlusMonoid<int8_t>().identity(), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::PlusMonoid<int8_t>()(-2, 1), -1);

    BOOST_CHECK_EQUAL(GraphBLAS::PlusMonoid<bool>().identity(), false);
    BOOST_CHECK_EQUAL(GraphBLAS::PlusMonoid<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(GraphBLAS::PlusMonoid<bool>()(true, true), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(times_monoid_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::TimesMonoid<double>().identity(), 1.0);
    BOOST_CHECK_EQUAL(GraphBLAS::TimesMonoid<double>()(-2., 1.), -2.0);
    BOOST_CHECK_EQUAL(GraphBLAS::TimesMonoid<float>().identity(), 1.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::TimesMonoid<float>()(-2.f, 1.f), -2.0f);

    BOOST_CHECK_EQUAL(GraphBLAS::TimesMonoid<uint64_t>().identity(), 1UL);
    BOOST_CHECK_EQUAL(GraphBLAS::TimesMonoid<uint64_t>()(2UL, 1UL), 2UL);
    BOOST_CHECK_EQUAL(GraphBLAS::TimesMonoid<uint32_t>().identity(), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::TimesMonoid<uint32_t>()(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(GraphBLAS::TimesMonoid<uint16_t>().identity(), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::TimesMonoid<uint16_t>()(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(GraphBLAS::TimesMonoid<uint8_t>().identity(), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::TimesMonoid<uint8_t>()(2U, 1U), 2U);

    BOOST_CHECK_EQUAL(GraphBLAS::TimesMonoid<int64_t>().identity(), 1L);
    BOOST_CHECK_EQUAL(GraphBLAS::TimesMonoid<int64_t>()(-2L, 1L), -2L);
    BOOST_CHECK_EQUAL(GraphBLAS::TimesMonoid<int32_t>().identity(), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::TimesMonoid<int32_t>()(-2, 1), -2);
    BOOST_CHECK_EQUAL(GraphBLAS::TimesMonoid<int16_t>().identity(), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::TimesMonoid<int16_t>()(-2, 1), -2);
    BOOST_CHECK_EQUAL(GraphBLAS::TimesMonoid<int8_t>().identity(), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::TimesMonoid<int8_t>()(-2, 1), -2);

    BOOST_CHECK_EQUAL(GraphBLAS::TimesMonoid<bool>().identity(), true);
    BOOST_CHECK_EQUAL(GraphBLAS::TimesMonoid<bool>()(false, true), false);
    BOOST_CHECK_EQUAL(GraphBLAS::TimesMonoid<bool>()(true, true), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(min_monoid_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::MinMonoid<double>().identity(),
                      std::numeric_limits<double>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinMonoid<double>()(-2., 1.), -2.0);
    BOOST_CHECK_EQUAL(GraphBLAS::MinMonoid<float>().identity(),
                      std::numeric_limits<float>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinMonoid<float>()(-2.f, 1.f), -2.0f);

    BOOST_CHECK_EQUAL(GraphBLAS::MinMonoid<uint64_t>().identity(),
                      std::numeric_limits<uint64_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinMonoid<uint64_t>()(2UL, 1UL), 1UL);
    BOOST_CHECK_EQUAL(GraphBLAS::MinMonoid<uint32_t>().identity(),
                      std::numeric_limits<uint32_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinMonoid<uint32_t>()(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::MinMonoid<uint16_t>().identity(),
                      std::numeric_limits<uint16_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinMonoid<uint16_t>()(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::MinMonoid<uint8_t>().identity(),
                      std::numeric_limits<uint8_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinMonoid<uint8_t>()(2U, 1U), 1U);

    BOOST_CHECK_EQUAL(GraphBLAS::MinMonoid<int64_t>().identity(),
                      std::numeric_limits<int64_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinMonoid<int64_t>()(-2L, 1L), -2L);
    BOOST_CHECK_EQUAL(GraphBLAS::MinMonoid<int32_t>().identity(),
                      std::numeric_limits<int32_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinMonoid<int32_t>()(-2, 1), -2);
    BOOST_CHECK_EQUAL(GraphBLAS::MinMonoid<int16_t>().identity(),
                      std::numeric_limits<int16_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinMonoid<int16_t>()(-2, 1), -2);
    BOOST_CHECK_EQUAL(GraphBLAS::MinMonoid<int8_t>().identity(),
                      std::numeric_limits<int8_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinMonoid<int8_t>()(-2, 1), -2);

    BOOST_CHECK_EQUAL(GraphBLAS::MinMonoid<bool>().identity(), true);
    BOOST_CHECK_EQUAL(GraphBLAS::MinMonoid<bool>()(false, true), false);
    BOOST_CHECK_EQUAL(GraphBLAS::MinMonoid<bool>()(true, true), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(max_monoid_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::MaxMonoid<double>().identity(), 0.0);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxMonoid<double>()(-2., 1.), 1.0);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxMonoid<float>().identity(), 0.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxMonoid<float>()(-2.f, 1.f), 1.0f);

    BOOST_CHECK_EQUAL(GraphBLAS::MaxMonoid<uint64_t>().identity(), 0UL);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxMonoid<uint64_t>()(2UL, 1UL), 2UL);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxMonoid<uint32_t>().identity(), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxMonoid<uint32_t>()(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxMonoid<uint16_t>().identity(), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxMonoid<uint16_t>()(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxMonoid<uint8_t>().identity(), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxMonoid<uint8_t>()(2U, 1U), 2U);

    BOOST_CHECK_EQUAL(GraphBLAS::MaxMonoid<int64_t>().identity(), 0L);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxMonoid<int64_t>()(-2L, 1L), 1L);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxMonoid<int32_t>().identity(), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxMonoid<int32_t>()(-2, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxMonoid<int16_t>().identity(), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxMonoid<int16_t>()(-2, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxMonoid<int8_t>().identity(), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxMonoid<int8_t>()(-2, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::MaxMonoid<bool>().identity(), false);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxMonoid<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxMonoid<bool>()(false, true), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(logical_or_monoid_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<double>().identity(), 0.0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<double>()(-2., 0.), 1.0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<double>()(0., 0.), 0.0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<float>().identity(), 0.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<float>()(-2.f, 0.f), 1.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<float>()(0.f, 0.f), 0.0f);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<uint64_t>().identity(), 0UL);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<uint64_t>()(2UL, 0UL), 1UL);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<uint64_t>()(0UL, 0UL), 0UL);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<uint32_t>().identity(), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<uint32_t>()(2U, 0U), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<uint32_t>()(0U, 0U), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<uint16_t>().identity(), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<uint16_t>()(2U, 0U), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<uint16_t>()(0U, 0U), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<uint8_t>().identity(), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<uint8_t>()(2U, 0U), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<uint8_t>()(0U, 0U), 0U);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<int64_t>().identity(), 0L);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<int64_t>()(-2L, 0L), 1L);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<int64_t>()(0L, 0L), 0L);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<int32_t>().identity(), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<int32_t>()(-2, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<int32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<int16_t>().identity(), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<int16_t>()(-2, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<int16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<int8_t>().identity(), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<int8_t>()(-2, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<int8_t>()(0, 0), 0);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<bool>().identity(), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalOrMonoid<bool>()(false, true), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(arithmetic_semiring_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<double>().zero(), 0.0);
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<double>().add(-2., 1.), -1.0);
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<double>().mult(-2., 1.), -2.0);
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<float>().zero(), 0.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<float>().add(-2.f, 1.f), -1.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<float>().mult(-2.f, 1.f), -2.0f);

    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<uint64_t>().zero(), 0UL);
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<uint64_t>().add(2UL, 1UL), 3UL);
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<uint64_t>().mult(2UL, 1UL), 2UL);
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<uint32_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<uint32_t>().add(2U, 1U), 3U);
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<uint32_t>().mult(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<uint16_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<uint16_t>().add(2U, 1U), 3U);
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<uint16_t>().mult(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<uint8_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<uint8_t>().add(2U, 1U), 3U);
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<uint8_t>().mult(2U, 1U), 2U);

    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<int64_t>().zero(), 0L);
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<int64_t>().add(-2L, 1L), -1L);
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<int64_t>().mult(-2L, 1L), -2L);
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<int32_t>().zero(), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<int32_t>().add(-2, 1), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<int32_t>().mult(-2, 1), -2);
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<int16_t>().zero(), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<int16_t>().add(-2, 1), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<int16_t>().mult(-2, 1), -2);
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<int8_t>().zero(), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<int8_t>().add(-2, 1), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<int8_t>().mult(-2, 1), -2);

    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<bool>().zero(), false);
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<bool>().add(false, false), false);
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<bool>().add(false, true), true);
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<bool>().mult(false, true), false);
    BOOST_CHECK_EQUAL(GraphBLAS::ArithmeticSemiring<bool>().mult(true, true), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(logical_semiring_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<double>().zero(), 0.0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<double>().add(-2., 1.), 1.0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<double>().add(0., 1.), 1.0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<double>().add(-2., 0.), 1.0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<double>().add(0., 0.), 0.0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<double>().mult(-2., 1.), 1.0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<double>().mult(0., 1.), 0.0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<double>().mult(-2., 0.), 0.0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<double>().mult(0., 0.), 0.0);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<float>().zero(), 0.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<float>().add(-2.f, 1.f), 1.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<float>().add(0.f, 1.f), 1.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<float>().add(-2.f, 0.f), 1.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<float>().add(0.f, 0.f), 0.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<float>().mult(-2.f, 1.f), 1.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<float>().mult(0.f, 1.f), 0.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<float>().mult(-2.f, 0.f), 0.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<float>().mult(0.f, 0.f), 0.0f);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint64_t>().zero(), 0UL);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint64_t>().add(2UL, 1UL), 1UL);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint64_t>().add(2UL, 0UL), 1UL);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint64_t>().add(0UL, 1UL), 1UL);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint64_t>().add(0UL, 0UL), 0UL);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint64_t>().mult(2UL, 1UL), 1UL);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint64_t>().mult(2UL, 0UL), 0UL);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint64_t>().mult(0UL, 1UL), 0UL);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint64_t>().mult(0UL, 0UL), 0UL);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint32_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint32_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint32_t>().add(2U, 0U), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint32_t>().add(0U, 1U), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint32_t>().add(0U, 0U), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint32_t>().mult(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint32_t>().mult(2U, 0U), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint32_t>().mult(0U, 1U), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint32_t>().mult(0U, 0U), 0U);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint16_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint16_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint16_t>().add(2U, 0U), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint16_t>().add(0U, 1U), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint16_t>().add(0U, 0U), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint16_t>().mult(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint16_t>().mult(2U, 0U), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint16_t>().mult(0U, 1U), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint16_t>().mult(0U, 0U), 0U);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint8_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint8_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint8_t>().add(2U, 0U), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint8_t>().add(0U, 1U), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint8_t>().add(0U, 0U), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint8_t>().mult(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint8_t>().mult(2U, 0U), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint8_t>().mult(0U, 1U), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<uint8_t>().mult(0U, 0U), 0U);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int64_t>().zero(), 0L);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int64_t>().add(-2L, 1L), 1L);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int64_t>().add(-2L, 0L), 1L);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int64_t>().add(0L, 1L), 1L);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int64_t>().add(0L, 0L), 0L);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int64_t>().mult(-2L, 1L), 1L);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int64_t>().mult(-2L, 0L), 0L);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int64_t>().mult(0L, 1L), 0L);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int64_t>().mult(0L, 0L), 0L);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int32_t>().zero(), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int32_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int32_t>().add(-2, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int32_t>().add(0, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int32_t>().add(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int32_t>().mult(-2, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int32_t>().mult(-2, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int32_t>().mult(0, 1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int32_t>().mult(0, 0), 0);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int16_t>().zero(), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int16_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int16_t>().add(-2, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int16_t>().add(0, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int16_t>().add(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int16_t>().mult(-2, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int16_t>().mult(-2, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int16_t>().mult(0, 1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int16_t>().mult(0, 0), 0);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int8_t>().zero(), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int8_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int8_t>().add(-2, 0), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int8_t>().add(0, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int8_t>().add(0, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int8_t>().mult(-2, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int8_t>().mult(-2, 0), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int8_t>().mult(0, 1), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<int8_t>().mult(0, 0), 0);

    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<bool>().zero(), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<bool>().add(false, false), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<bool>().add(false, true), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<bool>().add(true, false), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<bool>().add(true, true), true);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<bool>().mult(false, false), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<bool>().mult(false, true), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<bool>().mult(true, false), false);
    BOOST_CHECK_EQUAL(GraphBLAS::LogicalSemiring<bool>().mult(true, true), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(min_plus_semiring_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<double>().zero(),
                      std::numeric_limits<double>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<double>().add(-2., 1.), -2.0);
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<double>().add(2., 1.), 1.0);
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<double>().mult(-2., 1.), -1.0);

    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<float>().zero(),
                      std::numeric_limits<float>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<float>().add(-2.f, 1.f), -2.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<float>().add(2.f, 1.f), 1.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<float>().mult(-2.f, 1.f), -1.0f);

    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<uint64_t>().zero(),
                      std::numeric_limits<uint64_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<uint64_t>().add(2UL, 1UL), 1UL);
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<uint64_t>().add(2UL, 3UL), 2UL);
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<uint64_t>().mult(2UL, 1UL), 3UL);

    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<uint32_t>().zero(),
                      std::numeric_limits<uint32_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<uint32_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<uint32_t>().add(2U, 3U), 2U);
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<uint32_t>().mult(2U, 1U), 3U);

    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<uint16_t>().zero(),
                      std::numeric_limits<uint16_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<uint16_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<uint16_t>().add(2U, 3U), 2U);
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<uint16_t>().mult(2U, 1U), 3U);

    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<uint8_t>().zero(),
                      std::numeric_limits<uint8_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<uint8_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<uint8_t>().add(2U, 3U), 2U);
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<uint8_t>().mult(2U, 1U), 3U);

    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<int64_t>().zero(),
                      std::numeric_limits<int64_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<int64_t>().add(-2L, 1L), -2L);
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<int64_t>().add(2L, -1L), -1L);
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<int64_t>().mult(-2L, 1L), -1L);

    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<int32_t>().zero(),
                      std::numeric_limits<int32_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<int32_t>().add(-2, 1), -2);
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<int32_t>().add(2, -1), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<int32_t>().mult(-2, 1), -1);

    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<int16_t>().zero(),
                      std::numeric_limits<int16_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<int16_t>().add(-2, 1), -2);
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<int16_t>().add(2, -1), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<int16_t>().mult(-2, 1), -1);

    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<int8_t>().zero(),
                      std::numeric_limits<int8_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<int8_t>().add(-2, 1), -2);
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<int8_t>().add(2, -1), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<int8_t>().mult(-2, 1), -1);

    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<bool>().zero(), true);
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<bool>().add(false, true), false);
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<bool>().add(true, true), true);
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<bool>().mult(false, false), false);
    BOOST_CHECK_EQUAL(GraphBLAS::MinPlusSemiring<bool>().mult(true, false), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(max_times_semiring_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<double>().zero(), 0.0);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<double>().add(-2., 1.), 1.0);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<double>().mult(-2., 1.), -2.0);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<float>().zero(), 0.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<float>().add(-2.f, 1.f), 1.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<float>().mult(-2.f, 1.f), -2.0f);

    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<uint64_t>().zero(), 0UL);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<uint64_t>().add(2UL, 1UL), 2UL);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<uint64_t>().mult(2UL, 1UL), 2UL);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<uint32_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<uint32_t>().add(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<uint32_t>().mult(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<uint16_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<uint16_t>().add(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<uint16_t>().mult(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<uint8_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<uint8_t>().add(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<uint8_t>().mult(2U, 1U), 2U);

    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<int64_t>().zero(), 0L);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<int64_t>().add(-2L, 1L), 1L);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<int64_t>().mult(-2L, 1L), -2L);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<int32_t>().zero(), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<int32_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<int32_t>().mult(-2, 1), -2);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<int16_t>().zero(), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<int16_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<int16_t>().mult(-2, 1), -2);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<int8_t>().zero(), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<int8_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<int8_t>().mult(-2, 1), -2);

    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<bool>().zero(), false);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<bool>().add(false, false), false);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<bool>().add(false, true), true);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<bool>().mult(false, true), false);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxTimesSemiring<bool>().mult(true, true), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(min_select2nd_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<double>().zero(),
                      std::numeric_limits<double>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<double>().add(-2., 1.), -2.0);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<double>().add(2., 1.), 1.0);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<double>().mult(-2., 1.), 1.0);

    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<float>().zero(),
                      std::numeric_limits<float>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<float>().add(-2.f, 1.f), -2.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<float>().add(2.f, 1.f), 1.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<float>().mult(-2.f, 1.f), 1.0f);

    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<uint64_t>().zero(),
                      std::numeric_limits<uint64_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<uint64_t>().add(2UL, 1UL), 1UL);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<uint64_t>().add(2UL, 3UL), 2UL);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<uint64_t>().mult(2UL, 1UL), 1UL);

    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<uint32_t>().zero(),
                      std::numeric_limits<uint32_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<uint32_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<uint32_t>().add(2U, 3U), 2U);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<uint32_t>().mult(2U, 1U), 1U);

    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<uint16_t>().zero(),
                      std::numeric_limits<uint16_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<uint16_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<uint16_t>().add(2U, 3U), 2U);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<uint16_t>().mult(2U, 1U), 1U);

    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<uint8_t>().zero(),
                      std::numeric_limits<uint8_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<uint8_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<uint8_t>().add(2U, 3U), 2U);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<uint8_t>().mult(2U, 1U), 1U);

    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<int64_t>().zero(),
                      std::numeric_limits<int64_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<int64_t>().add(-2L, 1L), -2L);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<int64_t>().add(2L, -1L), -1L);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<int64_t>().mult(-2L, 1L), 1L);

    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<int32_t>().zero(),
                      std::numeric_limits<int32_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<int32_t>().add(-2, 1), -2);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<int32_t>().add(2, -1), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<int32_t>().mult(-2, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<int16_t>().zero(),
                      std::numeric_limits<int16_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<int16_t>().add(-2, 1), -2);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<int16_t>().add(2, -1), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<int16_t>().mult(-2, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<int8_t>().zero(),
                      std::numeric_limits<int8_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<int8_t>().add(-2, 1), -2);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<int8_t>().add(2, -1), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<int8_t>().mult(-2, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<bool>().zero(), true);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<bool>().add(false, true), false);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<bool>().add(true, true), true);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<bool>().mult(true, false), false);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect2ndSemiring<bool>().mult(false, true), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(max_select2nd_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<double>().zero(), 0.0);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<double>().add(-2., 1.), 1.0);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<double>().mult(-2., 1.), 1.0);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<float>().zero(), 0.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<float>().add(-2.f, 1.f), 1.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<float>().mult(-2.f, 1.f), 1.0f);

    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<uint64_t>().zero(), 0UL);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<uint64_t>().add(2UL, 1UL), 2UL);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<uint64_t>().mult(2UL, 1UL), 1UL);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<uint32_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<uint32_t>().add(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<uint32_t>().mult(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<uint16_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<uint16_t>().add(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<uint16_t>().mult(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<uint8_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<uint8_t>().add(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<uint8_t>().mult(2U, 1U), 1U);

    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<int64_t>().zero(), 0L);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<int64_t>().add(-2L, 1L), 1L);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<int64_t>().mult(-2L, 1L), 1L);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<int32_t>().zero(), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<int32_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<int32_t>().mult(-2, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<int16_t>().zero(), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<int16_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<int16_t>().mult(-2, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<int8_t>().zero(), 0);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<int8_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<int8_t>().mult(-2, 1), 1);

    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<bool>().zero(), false);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<bool>().add(false, false), false);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<bool>().add(false, true), true);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<bool>().mult(false, true), true);
    BOOST_CHECK_EQUAL(GraphBLAS::MaxSelect2ndSemiring<bool>().mult(true, false), false);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(min_select1st_test)
{
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<double>().zero(),
                      std::numeric_limits<double>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<double>().add(-2., 1.), -2.0);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<double>().add(2., 1.), 1.0);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<double>().mult(-2., 1.), -2.0);

    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<float>().zero(),
                      std::numeric_limits<float>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<float>().add(-2.f, 1.f), -2.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<float>().add(2.f, 1.f), 1.0f);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<float>().mult(-2.f, 1.f), -2.0f);

    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<uint64_t>().zero(),
                      std::numeric_limits<uint64_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<uint64_t>().add(2UL, 1UL), 1UL);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<uint64_t>().add(2UL, 3UL), 2UL);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<uint64_t>().mult(2UL, 1UL), 2UL);

    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<uint32_t>().zero(),
                      std::numeric_limits<uint32_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<uint32_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<uint32_t>().add(2U, 3U), 2U);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<uint32_t>().mult(2U, 1U), 2U);

    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<uint16_t>().zero(),
                      std::numeric_limits<uint16_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<uint16_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<uint16_t>().add(2U, 3U), 2U);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<uint16_t>().mult(2U, 1U), 2U);

    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<uint8_t>().zero(),
                      std::numeric_limits<uint8_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<uint8_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<uint8_t>().add(2U, 3U), 2U);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<uint8_t>().mult(2U, 1U), 2U);

    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<int64_t>().zero(),
                      std::numeric_limits<int64_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<int64_t>().add(-2L, 1L), -2L);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<int64_t>().add(2L, -1L), -1L);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<int64_t>().mult(-2L, 1L), -2L);

    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<int32_t>().zero(),
                      std::numeric_limits<int32_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<int32_t>().add(-2, 1), -2);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<int32_t>().add(2, -1), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<int32_t>().mult(-2, 1), -2);

    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<int16_t>().zero(),
                      std::numeric_limits<int16_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<int16_t>().add(-2, 1), -2);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<int16_t>().add(2, -1), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<int16_t>().mult(-2, 1), -2);

    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<int8_t>().zero(),
                      std::numeric_limits<int8_t>::max());
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<int8_t>().add(-2, 1), -2);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<int8_t>().add(2, -1), -1);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<int8_t>().mult(-2, 1), -2);

    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<bool>().zero(), true);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<bool>().add(false, true), false);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<bool>().add(true, true), true);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<bool>().mult(true, false), true);
    BOOST_CHECK_EQUAL(GraphBLAS::MinSelect1stSemiring<bool>().mult(false, true), false);
}

BOOST_AUTO_TEST_SUITE_END()
