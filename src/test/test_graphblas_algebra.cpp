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

#include <graphblas/graphblas.hpp>

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

BOOST_AUTO_TEST_SUITE_END()
