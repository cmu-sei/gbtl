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
#define BOOST_TEST_MODULE algebra_semiring_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

//****************************************************************************
BOOST_AUTO_TEST_CASE(arithmetic_semiring_test)
{
    BOOST_CHECK_EQUAL(ArithmeticSemiring<double>().zero(), 0.0);
    BOOST_CHECK_EQUAL(ArithmeticSemiring<double>().add(-2., 1.), -1.0);
    BOOST_CHECK_EQUAL(ArithmeticSemiring<double>().mult(-2., 1.), -2.0);
    BOOST_CHECK_EQUAL(ArithmeticSemiring<float>().zero(), 0.0f);
    BOOST_CHECK_EQUAL(ArithmeticSemiring<float>().add(-2.f, 1.f), -1.0f);
    BOOST_CHECK_EQUAL(ArithmeticSemiring<float>().mult(-2.f, 1.f), -2.0f);

    BOOST_CHECK_EQUAL(ArithmeticSemiring<uint64_t>().zero(), 0UL);
    BOOST_CHECK_EQUAL(ArithmeticSemiring<uint64_t>().add(2UL, 1UL), 3UL);
    BOOST_CHECK_EQUAL(ArithmeticSemiring<uint64_t>().mult(2UL, 1UL), 2UL);
    BOOST_CHECK_EQUAL(ArithmeticSemiring<uint32_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(ArithmeticSemiring<uint32_t>().add(2U, 1U), 3U);
    BOOST_CHECK_EQUAL(ArithmeticSemiring<uint32_t>().mult(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(ArithmeticSemiring<uint16_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(ArithmeticSemiring<uint16_t>().add(2U, 1U), 3U);
    BOOST_CHECK_EQUAL(ArithmeticSemiring<uint16_t>().mult(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(ArithmeticSemiring<uint8_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(ArithmeticSemiring<uint8_t>().add(2U, 1U), 3U);
    BOOST_CHECK_EQUAL(ArithmeticSemiring<uint8_t>().mult(2U, 1U), 2U);

    BOOST_CHECK_EQUAL(ArithmeticSemiring<int64_t>().zero(), 0L);
    BOOST_CHECK_EQUAL(ArithmeticSemiring<int64_t>().add(-2L, 1L), -1L);
    BOOST_CHECK_EQUAL(ArithmeticSemiring<int64_t>().mult(-2L, 1L), -2L);
    BOOST_CHECK_EQUAL(ArithmeticSemiring<int32_t>().zero(), 0);
    BOOST_CHECK_EQUAL(ArithmeticSemiring<int32_t>().add(-2, 1), -1);
    BOOST_CHECK_EQUAL(ArithmeticSemiring<int32_t>().mult(-2, 1), -2);
    BOOST_CHECK_EQUAL(ArithmeticSemiring<int16_t>().zero(), 0);
    BOOST_CHECK_EQUAL(ArithmeticSemiring<int16_t>().add(-2, 1), -1);
    BOOST_CHECK_EQUAL(ArithmeticSemiring<int16_t>().mult(-2, 1), -2);
    BOOST_CHECK_EQUAL(ArithmeticSemiring<int8_t>().zero(), 0);
    BOOST_CHECK_EQUAL(ArithmeticSemiring<int8_t>().add(-2, 1), -1);
    BOOST_CHECK_EQUAL(ArithmeticSemiring<int8_t>().mult(-2, 1), -2);

    BOOST_CHECK_EQUAL(ArithmeticSemiring<bool>().zero(), false);
    BOOST_CHECK_EQUAL(ArithmeticSemiring<bool>().add(false, false), false);
    BOOST_CHECK_EQUAL(ArithmeticSemiring<bool>().add(false, true), true);
    BOOST_CHECK_EQUAL(ArithmeticSemiring<bool>().mult(false, true), false);
    BOOST_CHECK_EQUAL(ArithmeticSemiring<bool>().mult(true, true), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(min_plus_semiring_test)
{
    BOOST_CHECK_EQUAL(MinPlusSemiring<double>().zero(),
                      std::numeric_limits<double>::infinity());
    BOOST_CHECK_EQUAL(MinPlusSemiring<double>().add(-2., 1.), -2.0);
    BOOST_CHECK_EQUAL(MinPlusSemiring<double>().add(2., 1.), 1.0);
    BOOST_CHECK_EQUAL(MinPlusSemiring<double>().mult(-2., 1.), -1.0);

    BOOST_CHECK_EQUAL(MinPlusSemiring<float>().zero(),
                      std::numeric_limits<float>::infinity());
    BOOST_CHECK_EQUAL(MinPlusSemiring<float>().add(-2.f, 1.f), -2.0f);
    BOOST_CHECK_EQUAL(MinPlusSemiring<float>().add(2.f, 1.f), 1.0f);
    BOOST_CHECK_EQUAL(MinPlusSemiring<float>().mult(-2.f, 1.f), -1.0f);

    BOOST_CHECK_EQUAL(MinPlusSemiring<uint64_t>().zero(),
                      std::numeric_limits<uint64_t>::max());
    BOOST_CHECK_EQUAL(MinPlusSemiring<uint64_t>().add(2UL, 1UL), 1UL);
    BOOST_CHECK_EQUAL(MinPlusSemiring<uint64_t>().add(2UL, 3UL), 2UL);
    BOOST_CHECK_EQUAL(MinPlusSemiring<uint64_t>().mult(2UL, 1UL), 3UL);

    BOOST_CHECK_EQUAL(MinPlusSemiring<uint32_t>().zero(),
                      std::numeric_limits<uint32_t>::max());
    BOOST_CHECK_EQUAL(MinPlusSemiring<uint32_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(MinPlusSemiring<uint32_t>().add(2U, 3U), 2U);
    BOOST_CHECK_EQUAL(MinPlusSemiring<uint32_t>().mult(2U, 1U), 3U);

    BOOST_CHECK_EQUAL(MinPlusSemiring<uint16_t>().zero(),
                      std::numeric_limits<uint16_t>::max());
    BOOST_CHECK_EQUAL(MinPlusSemiring<uint16_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(MinPlusSemiring<uint16_t>().add(2U, 3U), 2U);
    BOOST_CHECK_EQUAL(MinPlusSemiring<uint16_t>().mult(2U, 1U), 3U);

    BOOST_CHECK_EQUAL(MinPlusSemiring<uint8_t>().zero(),
                      std::numeric_limits<uint8_t>::max());
    BOOST_CHECK_EQUAL(MinPlusSemiring<uint8_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(MinPlusSemiring<uint8_t>().add(2U, 3U), 2U);
    BOOST_CHECK_EQUAL(MinPlusSemiring<uint8_t>().mult(2U, 1U), 3U);

    BOOST_CHECK_EQUAL(MinPlusSemiring<int64_t>().zero(),
                      std::numeric_limits<int64_t>::max());
    BOOST_CHECK_EQUAL(MinPlusSemiring<int64_t>().add(-2L, 1L), -2L);
    BOOST_CHECK_EQUAL(MinPlusSemiring<int64_t>().add(2L, -1L), -1L);
    BOOST_CHECK_EQUAL(MinPlusSemiring<int64_t>().mult(-2L, 1L), -1L);

    BOOST_CHECK_EQUAL(MinPlusSemiring<int32_t>().zero(),
                      std::numeric_limits<int32_t>::max());
    BOOST_CHECK_EQUAL(MinPlusSemiring<int32_t>().add(-2, 1), -2);
    BOOST_CHECK_EQUAL(MinPlusSemiring<int32_t>().add(2, -1), -1);
    BOOST_CHECK_EQUAL(MinPlusSemiring<int32_t>().mult(-2, 1), -1);

    BOOST_CHECK_EQUAL(MinPlusSemiring<int16_t>().zero(),
                      std::numeric_limits<int16_t>::max());
    BOOST_CHECK_EQUAL(MinPlusSemiring<int16_t>().add(-2, 1), -2);
    BOOST_CHECK_EQUAL(MinPlusSemiring<int16_t>().add(2, -1), -1);
    BOOST_CHECK_EQUAL(MinPlusSemiring<int16_t>().mult(-2, 1), -1);

    BOOST_CHECK_EQUAL(MinPlusSemiring<int8_t>().zero(),
                      std::numeric_limits<int8_t>::max());
    BOOST_CHECK_EQUAL(MinPlusSemiring<int8_t>().add(-2, 1), -2);
    BOOST_CHECK_EQUAL(MinPlusSemiring<int8_t>().add(2, -1), -1);
    BOOST_CHECK_EQUAL(MinPlusSemiring<int8_t>().mult(-2, 1), -1);

    BOOST_CHECK_EQUAL(MinPlusSemiring<bool>().zero(), true);
    BOOST_CHECK_EQUAL(MinPlusSemiring<bool>().add(false, true), false);
    BOOST_CHECK_EQUAL(MinPlusSemiring<bool>().add(true, true), true);
    BOOST_CHECK_EQUAL(MinPlusSemiring<bool>().mult(false, false), false);
    BOOST_CHECK_EQUAL(MinPlusSemiring<bool>().mult(true, false), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(max_plus_semiring_test)
{
    BOOST_CHECK_EQUAL(MaxPlusSemiring<double>().zero(),
                      -std::numeric_limits<double>::infinity());
    BOOST_CHECK_EQUAL(MaxPlusSemiring<double>().add(-2., 1.), 1.0);
    BOOST_CHECK_EQUAL(MaxPlusSemiring<double>().add(2., 1.), 2.0);
    BOOST_CHECK_EQUAL(MaxPlusSemiring<double>().mult(-2., 1.), -1.0);

    BOOST_CHECK_EQUAL(MaxPlusSemiring<float>().zero(),
                      -std::numeric_limits<float>::infinity());
    BOOST_CHECK_EQUAL(MaxPlusSemiring<float>().add(-2.f, 1.f), 1.0f);
    BOOST_CHECK_EQUAL(MaxPlusSemiring<float>().add(2.f, 1.f), 2.0f);
    BOOST_CHECK_EQUAL(MaxPlusSemiring<float>().mult(-2.f, 1.f), -1.0f);

    BOOST_CHECK_EQUAL(MaxPlusSemiring<uint64_t>().zero(),
                      std::numeric_limits<uint64_t>::min());
    BOOST_CHECK_EQUAL(MaxPlusSemiring<uint64_t>().add(2UL, 1UL), 2UL);
    BOOST_CHECK_EQUAL(MaxPlusSemiring<uint64_t>().add(2UL, 3UL), 3UL);
    BOOST_CHECK_EQUAL(MaxPlusSemiring<uint64_t>().mult(2UL, 1UL), 3UL);

    BOOST_CHECK_EQUAL(MaxPlusSemiring<uint32_t>().zero(),
                      std::numeric_limits<uint32_t>::min());
    BOOST_CHECK_EQUAL(MaxPlusSemiring<uint32_t>().add(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(MaxPlusSemiring<uint32_t>().add(2U, 3U), 3U);
    BOOST_CHECK_EQUAL(MaxPlusSemiring<uint32_t>().mult(2U, 1U), 3U);

    BOOST_CHECK_EQUAL(MaxPlusSemiring<uint16_t>().zero(),
                      std::numeric_limits<uint16_t>::min());
    BOOST_CHECK_EQUAL(MaxPlusSemiring<uint16_t>().add(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(MaxPlusSemiring<uint16_t>().add(2U, 3U), 3U);
    BOOST_CHECK_EQUAL(MaxPlusSemiring<uint16_t>().mult(2U, 1U), 3U);

    BOOST_CHECK_EQUAL(MaxPlusSemiring<uint8_t>().zero(),
                      std::numeric_limits<uint8_t>::min());
    BOOST_CHECK_EQUAL(MaxPlusSemiring<uint8_t>().add(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(MaxPlusSemiring<uint8_t>().add(2U, 3U), 3U);
    BOOST_CHECK_EQUAL(MaxPlusSemiring<uint8_t>().mult(2U, 1U), 3U);

    BOOST_CHECK_EQUAL(MaxPlusSemiring<int64_t>().zero(),
                      std::numeric_limits<int64_t>::min());
    BOOST_CHECK_EQUAL(MaxPlusSemiring<int64_t>().add(-2L, 1L), 1L);
    BOOST_CHECK_EQUAL(MaxPlusSemiring<int64_t>().add(2L, -1L), 2L);
    BOOST_CHECK_EQUAL(MaxPlusSemiring<int64_t>().mult(-2L, 1L), -1L);

    BOOST_CHECK_EQUAL(MaxPlusSemiring<int32_t>().zero(),
                      std::numeric_limits<int32_t>::min());
    BOOST_CHECK_EQUAL(MaxPlusSemiring<int32_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(MaxPlusSemiring<int32_t>().add(2, -1), 2);
    BOOST_CHECK_EQUAL(MaxPlusSemiring<int32_t>().mult(-2, 1), -1);

    BOOST_CHECK_EQUAL(MaxPlusSemiring<int16_t>().zero(),
                      std::numeric_limits<int16_t>::min());
    BOOST_CHECK_EQUAL(MaxPlusSemiring<int16_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(MaxPlusSemiring<int16_t>().add(2, -1), 2);
    BOOST_CHECK_EQUAL(MaxPlusSemiring<int16_t>().mult(-2, 1), -1);

    BOOST_CHECK_EQUAL(MaxPlusSemiring<int8_t>().zero(),
                      std::numeric_limits<int8_t>::min());
    BOOST_CHECK_EQUAL(MaxPlusSemiring<int8_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(MaxPlusSemiring<int8_t>().add(2, -1), 2);
    BOOST_CHECK_EQUAL(MaxPlusSemiring<int8_t>().mult(-2, 1), -1);

    BOOST_CHECK_EQUAL(MaxPlusSemiring<bool>().zero(), false);
    BOOST_CHECK_EQUAL(MaxPlusSemiring<bool>().add(false, true), true);
    BOOST_CHECK_EQUAL(MaxPlusSemiring<bool>().add(true, true), true);
    BOOST_CHECK_EQUAL(MaxPlusSemiring<bool>().mult(false, false), false);
    BOOST_CHECK_EQUAL(MaxPlusSemiring<bool>().mult(true, false), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(min_times_semiring_test)
{
    BOOST_CHECK_EQUAL(MinTimesSemiring<double>().zero(),
                      std::numeric_limits<double>::infinity());
    BOOST_CHECK_EQUAL(MinTimesSemiring<double>().add(-2., 1.), -2.0);
    BOOST_CHECK_EQUAL(MinTimesSemiring<double>().mult(-2., 1.), -2.0);
    BOOST_CHECK_EQUAL(MinTimesSemiring<float>().zero(),
                      std::numeric_limits<float>::infinity());
    BOOST_CHECK_EQUAL(MinTimesSemiring<float>().add(-2.f, 1.f), -2.0f);
    BOOST_CHECK_EQUAL(MinTimesSemiring<float>().mult(-2.f, 1.f), -2.0f);

    BOOST_CHECK_EQUAL(MinTimesSemiring<uint64_t>().zero(),
                      std::numeric_limits<uint64_t>::max());
    BOOST_CHECK_EQUAL(MinTimesSemiring<uint64_t>().add(2UL, 1UL), 1UL);
    BOOST_CHECK_EQUAL(MinTimesSemiring<uint64_t>().mult(2UL, 1UL), 2UL);
    BOOST_CHECK_EQUAL(MinTimesSemiring<uint32_t>().zero(),
                      std::numeric_limits<uint32_t>::max());
    BOOST_CHECK_EQUAL(MinTimesSemiring<uint32_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(MinTimesSemiring<uint32_t>().mult(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(MinTimesSemiring<uint16_t>().zero(),
                      std::numeric_limits<uint16_t>::max());
    BOOST_CHECK_EQUAL(MinTimesSemiring<uint16_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(MinTimesSemiring<uint16_t>().mult(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(MinTimesSemiring<uint8_t>().zero(),
                      std::numeric_limits<uint8_t>::max());
    BOOST_CHECK_EQUAL(MinTimesSemiring<uint8_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(MinTimesSemiring<uint8_t>().mult(2U, 1U), 2U);

    BOOST_CHECK_EQUAL(MinTimesSemiring<int64_t>().zero(),
                      std::numeric_limits<int64_t>::max());
    BOOST_CHECK_EQUAL(MinTimesSemiring<int64_t>().add(-2L, 1L), -2L);
    BOOST_CHECK_EQUAL(MinTimesSemiring<int64_t>().mult(-2L, 1L), -2L);
    BOOST_CHECK_EQUAL(MinTimesSemiring<int32_t>().zero(),
                      std::numeric_limits<int32_t>::max());
    BOOST_CHECK_EQUAL(MinTimesSemiring<int32_t>().add(-2, 1), -2);
    BOOST_CHECK_EQUAL(MinTimesSemiring<int32_t>().mult(-2, 1), -2);
    BOOST_CHECK_EQUAL(MinTimesSemiring<int16_t>().zero(),
                      std::numeric_limits<int16_t>::max());
    BOOST_CHECK_EQUAL(MinTimesSemiring<int16_t>().add(-2, 1), -2);
    BOOST_CHECK_EQUAL(MinTimesSemiring<int16_t>().mult(-2, 1), -2);
    BOOST_CHECK_EQUAL(MinTimesSemiring<int8_t>().zero(),
                      std::numeric_limits<int8_t>::max());
    BOOST_CHECK_EQUAL(MinTimesSemiring<int8_t>().add(-2, 1), -2);
    BOOST_CHECK_EQUAL(MinTimesSemiring<int8_t>().mult(-2, 1), -2);

    BOOST_CHECK_EQUAL(MinTimesSemiring<bool>().zero(), true);
    BOOST_CHECK_EQUAL(MinTimesSemiring<bool>().add(false, false), false);
    BOOST_CHECK_EQUAL(MinTimesSemiring<bool>().add(false, true), false);
    BOOST_CHECK_EQUAL(MinTimesSemiring<bool>().mult(false, true), false);
    BOOST_CHECK_EQUAL(MinTimesSemiring<bool>().mult(true, true), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(max_times_semiring_test)
{
    BOOST_CHECK_EQUAL(MaxTimesSemiring<double>().zero(),
                      -std::numeric_limits<double>::infinity());
    BOOST_CHECK_EQUAL(MaxTimesSemiring<double>().add(-2., 1.), 1.0);
    BOOST_CHECK_EQUAL(MaxTimesSemiring<double>().mult(-2., 1.), -2.0);
    BOOST_CHECK_EQUAL(MaxTimesSemiring<float>().zero(),
                      -std::numeric_limits<float>::infinity());
    BOOST_CHECK_EQUAL(MaxTimesSemiring<float>().add(-2.f, 1.f), 1.0f);
    BOOST_CHECK_EQUAL(MaxTimesSemiring<float>().mult(-2.f, 1.f), -2.0f);

    BOOST_CHECK_EQUAL(MaxTimesSemiring<uint64_t>().zero(), 0UL);
    BOOST_CHECK_EQUAL(MaxTimesSemiring<uint64_t>().add(2UL, 1UL), 2UL);
    BOOST_CHECK_EQUAL(MaxTimesSemiring<uint64_t>().mult(2UL, 1UL), 2UL);
    BOOST_CHECK_EQUAL(MaxTimesSemiring<uint32_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(MaxTimesSemiring<uint32_t>().add(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(MaxTimesSemiring<uint32_t>().mult(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(MaxTimesSemiring<uint16_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(MaxTimesSemiring<uint16_t>().add(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(MaxTimesSemiring<uint16_t>().mult(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(MaxTimesSemiring<uint8_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(MaxTimesSemiring<uint8_t>().add(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(MaxTimesSemiring<uint8_t>().mult(2U, 1U), 2U);

    BOOST_CHECK_EQUAL(MaxTimesSemiring<int64_t>().zero(),
                      std::numeric_limits<int64_t>::min());
    BOOST_CHECK_EQUAL(MaxTimesSemiring<int64_t>().add(-2L, 1L), 1L);
    BOOST_CHECK_EQUAL(MaxTimesSemiring<int64_t>().mult(-2L, 1L), -2L);
    BOOST_CHECK_EQUAL(MaxTimesSemiring<int32_t>().zero(),
                      std::numeric_limits<int32_t>::min());
    BOOST_CHECK_EQUAL(MaxTimesSemiring<int32_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(MaxTimesSemiring<int32_t>().mult(-2, 1), -2);
    BOOST_CHECK_EQUAL(MaxTimesSemiring<int16_t>().zero(),
                      std::numeric_limits<int16_t>::min());
    BOOST_CHECK_EQUAL(MaxTimesSemiring<int16_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(MaxTimesSemiring<int16_t>().mult(-2, 1), -2);
    BOOST_CHECK_EQUAL(MaxTimesSemiring<int8_t>().zero(),
                      std::numeric_limits<int8_t>::min());
    BOOST_CHECK_EQUAL(MaxTimesSemiring<int8_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(MaxTimesSemiring<int8_t>().mult(-2, 1), -2);

    BOOST_CHECK_EQUAL(MaxTimesSemiring<bool>().zero(), false);
    BOOST_CHECK_EQUAL(MaxTimesSemiring<bool>().add(false, false), false);
    BOOST_CHECK_EQUAL(MaxTimesSemiring<bool>().add(false, true), true);
    BOOST_CHECK_EQUAL(MaxTimesSemiring<bool>().mult(false, true), false);
    BOOST_CHECK_EQUAL(MaxTimesSemiring<bool>().mult(true, true), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(min_max_semiring_test)
{
    BOOST_CHECK_EQUAL(MinMaxSemiring<double>().zero(),
                      std::numeric_limits<double>::infinity());
    BOOST_CHECK_EQUAL(MinMaxSemiring<double>().add(-2., 1.), -2.0);
    BOOST_CHECK_EQUAL(MinMaxSemiring<double>().mult(-2., 1.),  1.0);
    BOOST_CHECK_EQUAL(MinMaxSemiring<float>().zero(),
                      std::numeric_limits<float>::infinity());
    BOOST_CHECK_EQUAL(MinMaxSemiring<float>().add(-2.f, 1.f), -2.0f);
    BOOST_CHECK_EQUAL(MinMaxSemiring<float>().mult(-2.f, 1.f),  1.0f);

    BOOST_CHECK_EQUAL(MinMaxSemiring<uint64_t>().zero(),
                      std::numeric_limits<uint64_t>::max());
    BOOST_CHECK_EQUAL(MinMaxSemiring<uint64_t>().add(2UL, 1UL), 1UL);
    BOOST_CHECK_EQUAL(MinMaxSemiring<uint64_t>().mult(2UL, 1UL), 2UL);
    BOOST_CHECK_EQUAL(MinMaxSemiring<uint32_t>().zero(),
                      std::numeric_limits<uint32_t>::max());
    BOOST_CHECK_EQUAL(MinMaxSemiring<uint32_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(MinMaxSemiring<uint32_t>().mult(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(MinMaxSemiring<uint16_t>().zero(),
                      std::numeric_limits<uint16_t>::max());
    BOOST_CHECK_EQUAL(MinMaxSemiring<uint16_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(MinMaxSemiring<uint16_t>().mult(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(MinMaxSemiring<uint8_t>().zero(),
                      std::numeric_limits<uint8_t>::max());
    BOOST_CHECK_EQUAL(MinMaxSemiring<uint8_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(MinMaxSemiring<uint8_t>().mult(2U, 1U), 2U);

    BOOST_CHECK_EQUAL(MinMaxSemiring<int64_t>().zero(),
                      std::numeric_limits<int64_t>::max());
    BOOST_CHECK_EQUAL(MinMaxSemiring<int64_t>().add(-2L, 1L),-2L);
    BOOST_CHECK_EQUAL(MinMaxSemiring<int64_t>().mult(-2L, 1L),  1L);
    BOOST_CHECK_EQUAL(MinMaxSemiring<int32_t>().zero(),
                      std::numeric_limits<int32_t>::max());
    BOOST_CHECK_EQUAL(MinMaxSemiring<int32_t>().add(-2, 1),-2);
    BOOST_CHECK_EQUAL(MinMaxSemiring<int32_t>().mult(-2, 1),  1);
    BOOST_CHECK_EQUAL(MinMaxSemiring<int16_t>().zero(),
                      std::numeric_limits<int16_t>::max());
    BOOST_CHECK_EQUAL(MinMaxSemiring<int16_t>().add(-2, 1),-2);
    BOOST_CHECK_EQUAL(MinMaxSemiring<int16_t>().mult(-2, 1),  1);
    BOOST_CHECK_EQUAL(MinMaxSemiring<int8_t>().zero(),
                      std::numeric_limits<int8_t>::max());
    BOOST_CHECK_EQUAL(MinMaxSemiring<int8_t>().add(-2, 1),-2);
    BOOST_CHECK_EQUAL(MinMaxSemiring<int8_t>().mult(-2, 1),  1);

    BOOST_CHECK_EQUAL(MinMaxSemiring<bool>().zero(), true);
    BOOST_CHECK_EQUAL(MinMaxSemiring<bool>().add(false, false), false);
    BOOST_CHECK_EQUAL(MinMaxSemiring<bool>().add(false, true), false);
    BOOST_CHECK_EQUAL(MinMaxSemiring<bool>().mult(false, true), true);
    BOOST_CHECK_EQUAL(MinMaxSemiring<bool>().mult(true, true), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(max_min_semiring_test)
{
    BOOST_CHECK_EQUAL(MaxMinSemiring<double>().zero(),
                      -std::numeric_limits<double>::infinity());
    BOOST_CHECK_EQUAL(MaxMinSemiring<double>().add(-2., 1.), 1.0);
    BOOST_CHECK_EQUAL(MaxMinSemiring<double>().mult(-2., 1.), -2.0);
    BOOST_CHECK_EQUAL(MaxMinSemiring<float>().zero(),
                      -std::numeric_limits<float>::infinity());
    BOOST_CHECK_EQUAL(MaxMinSemiring<float>().add(-2.f, 1.f), 1.0f);
    BOOST_CHECK_EQUAL(MaxMinSemiring<float>().mult(-2.f, 1.f), -2.0f);

    BOOST_CHECK_EQUAL(MaxMinSemiring<uint64_t>().zero(), 0UL);
    BOOST_CHECK_EQUAL(MaxMinSemiring<uint64_t>().add(2UL, 1UL), 2UL);
    BOOST_CHECK_EQUAL(MaxMinSemiring<uint64_t>().mult(2UL, 1UL), 1UL);
    BOOST_CHECK_EQUAL(MaxMinSemiring<uint32_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(MaxMinSemiring<uint32_t>().add(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(MaxMinSemiring<uint32_t>().mult(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(MaxMinSemiring<uint16_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(MaxMinSemiring<uint16_t>().add(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(MaxMinSemiring<uint16_t>().mult(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(MaxMinSemiring<uint8_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(MaxMinSemiring<uint8_t>().add(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(MaxMinSemiring<uint8_t>().mult(2U, 1U), 1U);

    BOOST_CHECK_EQUAL(MaxMinSemiring<int64_t>().zero(),
                      std::numeric_limits<int64_t>::min());
    BOOST_CHECK_EQUAL(MaxMinSemiring<int64_t>().add(-2L, 1L), 1L);
    BOOST_CHECK_EQUAL(MaxMinSemiring<int64_t>().mult(-2L, 1L), -2L);
    BOOST_CHECK_EQUAL(MaxMinSemiring<int32_t>().zero(),
                      std::numeric_limits<int32_t>::min());
    BOOST_CHECK_EQUAL(MaxMinSemiring<int32_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(MaxMinSemiring<int32_t>().mult(-2, 1), -2);
    BOOST_CHECK_EQUAL(MaxMinSemiring<int16_t>().zero(),
                      std::numeric_limits<int16_t>::min());
    BOOST_CHECK_EQUAL(MaxMinSemiring<int16_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(MaxMinSemiring<int16_t>().mult(-2, 1), -2);
    BOOST_CHECK_EQUAL(MaxMinSemiring<int8_t>().zero(),
                      std::numeric_limits<int8_t>::min());
    BOOST_CHECK_EQUAL(MaxMinSemiring<int8_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(MaxMinSemiring<int8_t>().mult(-2, 1), -2);

    BOOST_CHECK_EQUAL(MaxMinSemiring<bool>().zero(), false);
    BOOST_CHECK_EQUAL(MaxMinSemiring<bool>().add(false, false), false);
    BOOST_CHECK_EQUAL(MaxMinSemiring<bool>().add(false, true), true);
    BOOST_CHECK_EQUAL(MaxMinSemiring<bool>().mult(false, true), false);
    BOOST_CHECK_EQUAL(MaxMinSemiring<bool>().mult(true, true), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(plus_min_semiring_test)
{
    BOOST_CHECK_EQUAL(PlusMinSemiring<double>().zero(), 0.);
    BOOST_CHECK_EQUAL(PlusMinSemiring<double>().add(-2., 1.), -1.0);
    BOOST_CHECK_EQUAL(PlusMinSemiring<double>().add(2., 1.), 3.0);
    BOOST_CHECK_EQUAL(PlusMinSemiring<double>().mult(-2., 1.), -2.0);

    BOOST_CHECK_EQUAL(PlusMinSemiring<float>().zero(), 0.f);
    BOOST_CHECK_EQUAL(PlusMinSemiring<float>().add(-2.f, 1.f), -1.0f);
    BOOST_CHECK_EQUAL(PlusMinSemiring<float>().add(2.f, 1.f), 3.0f);
    BOOST_CHECK_EQUAL(PlusMinSemiring<float>().mult(-2.f, 1.f), -2.0f);

    BOOST_CHECK_EQUAL(PlusMinSemiring<uint64_t>().zero(), 0UL);
    BOOST_CHECK_EQUAL(PlusMinSemiring<uint64_t>().add(2UL, 1UL), 3UL);
    BOOST_CHECK_EQUAL(PlusMinSemiring<uint64_t>().add(2UL, 3UL), 5UL);
    BOOST_CHECK_EQUAL(PlusMinSemiring<uint64_t>().mult(2UL, 1UL), 1UL);

    BOOST_CHECK_EQUAL(PlusMinSemiring<uint32_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(PlusMinSemiring<uint32_t>().add(2U, 1U), 3U);
    BOOST_CHECK_EQUAL(PlusMinSemiring<uint32_t>().add(2U, 3U), 5U);
    BOOST_CHECK_EQUAL(PlusMinSemiring<uint32_t>().mult(2U, 1U), 1U);

    BOOST_CHECK_EQUAL(PlusMinSemiring<uint16_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(PlusMinSemiring<uint16_t>().add(2U, 1U), 3U);
    BOOST_CHECK_EQUAL(PlusMinSemiring<uint16_t>().add(2U, 3U), 5U);
    BOOST_CHECK_EQUAL(PlusMinSemiring<uint16_t>().mult(2U, 1U), 1U);

    BOOST_CHECK_EQUAL(PlusMinSemiring<uint8_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(PlusMinSemiring<uint8_t>().add(2U, 1U), 3U);
    BOOST_CHECK_EQUAL(PlusMinSemiring<uint8_t>().add(2U, 3U), 5U);
    BOOST_CHECK_EQUAL(PlusMinSemiring<uint8_t>().mult(2U, 1U), 1U);

    BOOST_CHECK_EQUAL(PlusMinSemiring<int64_t>().zero(), 0L);
    BOOST_CHECK_EQUAL(PlusMinSemiring<int64_t>().add(-2L, 1L), -1L);
    BOOST_CHECK_EQUAL(PlusMinSemiring<int64_t>().add(2L, -1L), 1L);
    BOOST_CHECK_EQUAL(PlusMinSemiring<int64_t>().mult(-2L, 1L), -2L);

    BOOST_CHECK_EQUAL(PlusMinSemiring<int32_t>().zero(), 0);
    BOOST_CHECK_EQUAL(PlusMinSemiring<int32_t>().add(-2, 1), -1);
    BOOST_CHECK_EQUAL(PlusMinSemiring<int32_t>().add(2, -1), 1);
    BOOST_CHECK_EQUAL(PlusMinSemiring<int32_t>().mult(-2, 1), -2);

    BOOST_CHECK_EQUAL(PlusMinSemiring<int16_t>().zero(), 0);
    BOOST_CHECK_EQUAL(PlusMinSemiring<int16_t>().add(-2, 1), -1);
    BOOST_CHECK_EQUAL(PlusMinSemiring<int16_t>().add(2, -1), 1);
    BOOST_CHECK_EQUAL(PlusMinSemiring<int16_t>().mult(-2, 1), -2);

    BOOST_CHECK_EQUAL(PlusMinSemiring<int8_t>().zero(), 0);
    BOOST_CHECK_EQUAL(PlusMinSemiring<int8_t>().add(-2, 1), -1);
    BOOST_CHECK_EQUAL(PlusMinSemiring<int8_t>().add(2, -1), 1);
    BOOST_CHECK_EQUAL(PlusMinSemiring<int8_t>().mult(-2, 1), -2);

    BOOST_CHECK_EQUAL(PlusMinSemiring<bool>().zero(), false);
    BOOST_CHECK_EQUAL(PlusMinSemiring<bool>().add(false, true), true);
    BOOST_CHECK_EQUAL(PlusMinSemiring<bool>().add(true, true), true);
    BOOST_CHECK_EQUAL(PlusMinSemiring<bool>().mult(false, false), false);
    BOOST_CHECK_EQUAL(PlusMinSemiring<bool>().mult(true, false), false);
    BOOST_CHECK_EQUAL(PlusMinSemiring<bool>().mult(true, true), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(logical_semiring_test)
{
    BOOST_CHECK_EQUAL(LogicalSemiring<double>().zero(), 0.0);
    BOOST_CHECK_EQUAL(LogicalSemiring<double>().add(-2., 1.), 1.0);
    BOOST_CHECK_EQUAL(LogicalSemiring<double>().add(0., 1.), 1.0);
    BOOST_CHECK_EQUAL(LogicalSemiring<double>().add(-2., 0.), 1.0);
    BOOST_CHECK_EQUAL(LogicalSemiring<double>().add(0., 0.), 0.0);
    BOOST_CHECK_EQUAL(LogicalSemiring<double>().mult(-2., 1.), 1.0);
    BOOST_CHECK_EQUAL(LogicalSemiring<double>().mult(0., 1.), 0.0);
    BOOST_CHECK_EQUAL(LogicalSemiring<double>().mult(-2., 0.), 0.0);
    BOOST_CHECK_EQUAL(LogicalSemiring<double>().mult(0., 0.), 0.0);

    BOOST_CHECK_EQUAL(LogicalSemiring<float>().zero(), 0.0f);
    BOOST_CHECK_EQUAL(LogicalSemiring<float>().add(-2.f, 1.f), 1.0f);
    BOOST_CHECK_EQUAL(LogicalSemiring<float>().add(0.f, 1.f), 1.0f);
    BOOST_CHECK_EQUAL(LogicalSemiring<float>().add(-2.f, 0.f), 1.0f);
    BOOST_CHECK_EQUAL(LogicalSemiring<float>().add(0.f, 0.f), 0.0f);
    BOOST_CHECK_EQUAL(LogicalSemiring<float>().mult(-2.f, 1.f), 1.0f);
    BOOST_CHECK_EQUAL(LogicalSemiring<float>().mult(0.f, 1.f), 0.0f);
    BOOST_CHECK_EQUAL(LogicalSemiring<float>().mult(-2.f, 0.f), 0.0f);
    BOOST_CHECK_EQUAL(LogicalSemiring<float>().mult(0.f, 0.f), 0.0f);

    BOOST_CHECK_EQUAL(LogicalSemiring<uint64_t>().zero(), 0UL);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint64_t>().add(2UL, 1UL), 1UL);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint64_t>().add(2UL, 0UL), 1UL);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint64_t>().add(0UL, 1UL), 1UL);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint64_t>().add(0UL, 0UL), 0UL);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint64_t>().mult(2UL, 1UL), 1UL);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint64_t>().mult(2UL, 0UL), 0UL);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint64_t>().mult(0UL, 1UL), 0UL);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint64_t>().mult(0UL, 0UL), 0UL);

    BOOST_CHECK_EQUAL(LogicalSemiring<uint32_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint32_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint32_t>().add(2U, 0U), 1U);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint32_t>().add(0U, 1U), 1U);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint32_t>().add(0U, 0U), 0U);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint32_t>().mult(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint32_t>().mult(2U, 0U), 0U);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint32_t>().mult(0U, 1U), 0U);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint32_t>().mult(0U, 0U), 0U);

    BOOST_CHECK_EQUAL(LogicalSemiring<uint16_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint16_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint16_t>().add(2U, 0U), 1U);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint16_t>().add(0U, 1U), 1U);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint16_t>().add(0U, 0U), 0U);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint16_t>().mult(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint16_t>().mult(2U, 0U), 0U);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint16_t>().mult(0U, 1U), 0U);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint16_t>().mult(0U, 0U), 0U);

    BOOST_CHECK_EQUAL(LogicalSemiring<uint8_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint8_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint8_t>().add(2U, 0U), 1U);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint8_t>().add(0U, 1U), 1U);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint8_t>().add(0U, 0U), 0U);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint8_t>().mult(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint8_t>().mult(2U, 0U), 0U);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint8_t>().mult(0U, 1U), 0U);
    BOOST_CHECK_EQUAL(LogicalSemiring<uint8_t>().mult(0U, 0U), 0U);

    BOOST_CHECK_EQUAL(LogicalSemiring<int64_t>().zero(), 0L);
    BOOST_CHECK_EQUAL(LogicalSemiring<int64_t>().add(-2L, 1L), 1L);
    BOOST_CHECK_EQUAL(LogicalSemiring<int64_t>().add(-2L, 0L), 1L);
    BOOST_CHECK_EQUAL(LogicalSemiring<int64_t>().add(0L, 1L), 1L);
    BOOST_CHECK_EQUAL(LogicalSemiring<int64_t>().add(0L, 0L), 0L);
    BOOST_CHECK_EQUAL(LogicalSemiring<int64_t>().mult(-2L, 1L), 1L);
    BOOST_CHECK_EQUAL(LogicalSemiring<int64_t>().mult(-2L, 0L), 0L);
    BOOST_CHECK_EQUAL(LogicalSemiring<int64_t>().mult(0L, 1L), 0L);
    BOOST_CHECK_EQUAL(LogicalSemiring<int64_t>().mult(0L, 0L), 0L);

    BOOST_CHECK_EQUAL(LogicalSemiring<int32_t>().zero(), 0);
    BOOST_CHECK_EQUAL(LogicalSemiring<int32_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(LogicalSemiring<int32_t>().add(-2, 0), 1);
    BOOST_CHECK_EQUAL(LogicalSemiring<int32_t>().add(0, 1), 1);
    BOOST_CHECK_EQUAL(LogicalSemiring<int32_t>().add(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalSemiring<int32_t>().mult(-2, 1), 1);
    BOOST_CHECK_EQUAL(LogicalSemiring<int32_t>().mult(-2, 0), 0);
    BOOST_CHECK_EQUAL(LogicalSemiring<int32_t>().mult(0, 1), 0);
    BOOST_CHECK_EQUAL(LogicalSemiring<int32_t>().mult(0, 0), 0);

    BOOST_CHECK_EQUAL(LogicalSemiring<int16_t>().zero(), 0);
    BOOST_CHECK_EQUAL(LogicalSemiring<int16_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(LogicalSemiring<int16_t>().add(-2, 0), 1);
    BOOST_CHECK_EQUAL(LogicalSemiring<int16_t>().add(0, 1), 1);
    BOOST_CHECK_EQUAL(LogicalSemiring<int16_t>().add(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalSemiring<int16_t>().mult(-2, 1), 1);
    BOOST_CHECK_EQUAL(LogicalSemiring<int16_t>().mult(-2, 0), 0);
    BOOST_CHECK_EQUAL(LogicalSemiring<int16_t>().mult(0, 1), 0);
    BOOST_CHECK_EQUAL(LogicalSemiring<int16_t>().mult(0, 0), 0);

    BOOST_CHECK_EQUAL(LogicalSemiring<int8_t>().zero(), 0);
    BOOST_CHECK_EQUAL(LogicalSemiring<int8_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(LogicalSemiring<int8_t>().add(-2, 0), 1);
    BOOST_CHECK_EQUAL(LogicalSemiring<int8_t>().add(0, 1), 1);
    BOOST_CHECK_EQUAL(LogicalSemiring<int8_t>().add(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalSemiring<int8_t>().mult(-2, 1), 1);
    BOOST_CHECK_EQUAL(LogicalSemiring<int8_t>().mult(-2, 0), 0);
    BOOST_CHECK_EQUAL(LogicalSemiring<int8_t>().mult(0, 1), 0);
    BOOST_CHECK_EQUAL(LogicalSemiring<int8_t>().mult(0, 0), 0);

    BOOST_CHECK_EQUAL(LogicalSemiring<bool>().zero(), false);
    BOOST_CHECK_EQUAL(LogicalSemiring<bool>().add(false, false), false);
    BOOST_CHECK_EQUAL(LogicalSemiring<bool>().add(false, true), true);
    BOOST_CHECK_EQUAL(LogicalSemiring<bool>().add(true, false), true);
    BOOST_CHECK_EQUAL(LogicalSemiring<bool>().add(true, true), true);
    BOOST_CHECK_EQUAL(LogicalSemiring<bool>().mult(false, false), false);
    BOOST_CHECK_EQUAL(LogicalSemiring<bool>().mult(false, true), false);
    BOOST_CHECK_EQUAL(LogicalSemiring<bool>().mult(true, false), false);
    BOOST_CHECK_EQUAL(LogicalSemiring<bool>().mult(true, true), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(and_or_semiring_test)
{
    BOOST_CHECK_EQUAL(AndOrSemiring<double>().zero(), 1.0);
    BOOST_CHECK_EQUAL(AndOrSemiring<double>().mult(-2., 1.), 1.0);
    BOOST_CHECK_EQUAL(AndOrSemiring<double>().mult(0., 1.), 1.0);
    BOOST_CHECK_EQUAL(AndOrSemiring<double>().mult(-2., 0.), 1.0);
    BOOST_CHECK_EQUAL(AndOrSemiring<double>().mult(0., 0.), 0.0);
    BOOST_CHECK_EQUAL(AndOrSemiring<double>().add(-2., 1.), 1.0);
    BOOST_CHECK_EQUAL(AndOrSemiring<double>().add(0., 1.), 0.0);
    BOOST_CHECK_EQUAL(AndOrSemiring<double>().add(-2., 0.), 0.0);
    BOOST_CHECK_EQUAL(AndOrSemiring<double>().add(0., 0.), 0.0);

    BOOST_CHECK_EQUAL(AndOrSemiring<float>().zero(), 1.0f);
    BOOST_CHECK_EQUAL(AndOrSemiring<float>().mult(-2.f, 1.f), 1.0f);
    BOOST_CHECK_EQUAL(AndOrSemiring<float>().mult(0.f, 1.f), 1.0f);
    BOOST_CHECK_EQUAL(AndOrSemiring<float>().mult(-2.f, 0.f), 1.0f);
    BOOST_CHECK_EQUAL(AndOrSemiring<float>().mult(0.f, 0.f), 0.0f);
    BOOST_CHECK_EQUAL(AndOrSemiring<float>().add(-2.f, 1.f), 1.0f);
    BOOST_CHECK_EQUAL(AndOrSemiring<float>().add(0.f, 1.f), 0.0f);
    BOOST_CHECK_EQUAL(AndOrSemiring<float>().add(-2.f, 0.f), 0.0f);
    BOOST_CHECK_EQUAL(AndOrSemiring<float>().add(0.f, 0.f), 0.0f);

    BOOST_CHECK_EQUAL(AndOrSemiring<uint64_t>().zero(), 1UL);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint64_t>().mult(2UL, 1UL), 1UL);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint64_t>().mult(2UL, 0UL), 1UL);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint64_t>().mult(0UL, 1UL), 1UL);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint64_t>().mult(0UL, 0UL), 0UL);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint64_t>().add(2UL, 1UL), 1UL);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint64_t>().add(2UL, 0UL), 0UL);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint64_t>().add(0UL, 1UL), 0UL);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint64_t>().add(0UL, 0UL), 0UL);

    BOOST_CHECK_EQUAL(AndOrSemiring<uint32_t>().zero(), 1U);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint32_t>().mult(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint32_t>().mult(2U, 0U), 1U);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint32_t>().mult(0U, 1U), 1U);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint32_t>().mult(0U, 0U), 0U);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint32_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint32_t>().add(2U, 0U), 0U);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint32_t>().add(0U, 1U), 0U);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint32_t>().add(0U, 0U), 0U);

    BOOST_CHECK_EQUAL(AndOrSemiring<uint16_t>().zero(), 1U);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint16_t>().mult(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint16_t>().mult(2U, 0U), 1U);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint16_t>().mult(0U, 1U), 1U);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint16_t>().mult(0U, 0U), 0U);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint16_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint16_t>().add(2U, 0U), 0U);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint16_t>().add(0U, 1U), 0U);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint16_t>().add(0U, 0U), 0U);

    BOOST_CHECK_EQUAL(AndOrSemiring<uint8_t>().zero(), 1U);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint8_t>().mult(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint8_t>().mult(2U, 0U), 1U);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint8_t>().mult(0U, 1U), 1U);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint8_t>().mult(0U, 0U), 0U);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint8_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint8_t>().add(2U, 0U), 0U);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint8_t>().add(0U, 1U), 0U);
    BOOST_CHECK_EQUAL(AndOrSemiring<uint8_t>().add(0U, 0U), 0U);

    BOOST_CHECK_EQUAL(AndOrSemiring<int64_t>().zero(), 1L);
    BOOST_CHECK_EQUAL(AndOrSemiring<int64_t>().mult(-2L, 1L), 1L);
    BOOST_CHECK_EQUAL(AndOrSemiring<int64_t>().mult(-2L, 0L), 1L);
    BOOST_CHECK_EQUAL(AndOrSemiring<int64_t>().mult(0L, 1L), 1L);
    BOOST_CHECK_EQUAL(AndOrSemiring<int64_t>().mult(0L, 0L), 0L);
    BOOST_CHECK_EQUAL(AndOrSemiring<int64_t>().add(-2L, 1L), 1L);
    BOOST_CHECK_EQUAL(AndOrSemiring<int64_t>().add(-2L, 0L), 0L);
    BOOST_CHECK_EQUAL(AndOrSemiring<int64_t>().add(0L, 1L), 0L);
    BOOST_CHECK_EQUAL(AndOrSemiring<int64_t>().add(0L, 0L), 0L);

    BOOST_CHECK_EQUAL(AndOrSemiring<int32_t>().zero(), 1);
    BOOST_CHECK_EQUAL(AndOrSemiring<int32_t>().mult(-2, 1), 1);
    BOOST_CHECK_EQUAL(AndOrSemiring<int32_t>().mult(-2, 0), 1);
    BOOST_CHECK_EQUAL(AndOrSemiring<int32_t>().mult(0, 1), 1);
    BOOST_CHECK_EQUAL(AndOrSemiring<int32_t>().mult(0, 0), 0);
    BOOST_CHECK_EQUAL(AndOrSemiring<int32_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(AndOrSemiring<int32_t>().add(-2, 0), 0);
    BOOST_CHECK_EQUAL(AndOrSemiring<int32_t>().add(0, 1), 0);
    BOOST_CHECK_EQUAL(AndOrSemiring<int32_t>().add(0, 0), 0);

    BOOST_CHECK_EQUAL(AndOrSemiring<int16_t>().zero(), 1);
    BOOST_CHECK_EQUAL(AndOrSemiring<int16_t>().mult(-2, 1), 1);
    BOOST_CHECK_EQUAL(AndOrSemiring<int16_t>().mult(-2, 0), 1);
    BOOST_CHECK_EQUAL(AndOrSemiring<int16_t>().mult(0, 1), 1);
    BOOST_CHECK_EQUAL(AndOrSemiring<int16_t>().mult(0, 0), 0);
    BOOST_CHECK_EQUAL(AndOrSemiring<int16_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(AndOrSemiring<int16_t>().add(-2, 0), 0);
    BOOST_CHECK_EQUAL(AndOrSemiring<int16_t>().add(0, 1), 0);
    BOOST_CHECK_EQUAL(AndOrSemiring<int16_t>().add(0, 0), 0);

    BOOST_CHECK_EQUAL(AndOrSemiring<int8_t>().zero(), 1);
    BOOST_CHECK_EQUAL(AndOrSemiring<int8_t>().mult(-2, 1), 1);
    BOOST_CHECK_EQUAL(AndOrSemiring<int8_t>().mult(-2, 0), 1);
    BOOST_CHECK_EQUAL(AndOrSemiring<int8_t>().mult(0, 1), 1);
    BOOST_CHECK_EQUAL(AndOrSemiring<int8_t>().mult(0, 0), 0);
    BOOST_CHECK_EQUAL(AndOrSemiring<int8_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(AndOrSemiring<int8_t>().add(-2, 0), 0);
    BOOST_CHECK_EQUAL(AndOrSemiring<int8_t>().add(0, 1), 0);
    BOOST_CHECK_EQUAL(AndOrSemiring<int8_t>().add(0, 0), 0);

    BOOST_CHECK_EQUAL(AndOrSemiring<bool>().zero(), true);
    BOOST_CHECK_EQUAL(AndOrSemiring<bool>().mult(false, false), false);
    BOOST_CHECK_EQUAL(AndOrSemiring<bool>().mult(false, true), true);
    BOOST_CHECK_EQUAL(AndOrSemiring<bool>().mult(true, false), true);
    BOOST_CHECK_EQUAL(AndOrSemiring<bool>().mult(true, true), true);
    BOOST_CHECK_EQUAL(AndOrSemiring<bool>().add(false, false), false);
    BOOST_CHECK_EQUAL(AndOrSemiring<bool>().add(false, true), false);
    BOOST_CHECK_EQUAL(AndOrSemiring<bool>().add(true, false), false);
    BOOST_CHECK_EQUAL(AndOrSemiring<bool>().add(true, true), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(xor_and_semiring_test)
{
    BOOST_CHECK_EQUAL(XorAndSemiring<double>().zero(), 0.0);
    BOOST_CHECK_EQUAL(XorAndSemiring<double>().add(-2., 1.), 0.0);
    BOOST_CHECK_EQUAL(XorAndSemiring<double>().add(0., 1.), 1.0);
    BOOST_CHECK_EQUAL(XorAndSemiring<double>().add(-2., 0.), 1.0);
    BOOST_CHECK_EQUAL(XorAndSemiring<double>().add(0., 0.), 0.0);
    BOOST_CHECK_EQUAL(XorAndSemiring<double>().mult(-2., 1.), 1.0);
    BOOST_CHECK_EQUAL(XorAndSemiring<double>().mult(0., 1.), 0.0);
    BOOST_CHECK_EQUAL(XorAndSemiring<double>().mult(-2., 0.), 0.0);
    BOOST_CHECK_EQUAL(XorAndSemiring<double>().mult(0., 0.), 0.0);

    BOOST_CHECK_EQUAL(XorAndSemiring<float>().zero(), 0.0f);
    BOOST_CHECK_EQUAL(XorAndSemiring<float>().add(-2.f, 1.f), 0.0f);
    BOOST_CHECK_EQUAL(XorAndSemiring<float>().add(0.f, 1.f), 1.0f);
    BOOST_CHECK_EQUAL(XorAndSemiring<float>().add(-2.f, 0.f), 1.0f);
    BOOST_CHECK_EQUAL(XorAndSemiring<float>().add(0.f, 0.f), 0.0f);
    BOOST_CHECK_EQUAL(XorAndSemiring<float>().mult(-2.f, 1.f), 1.0f);
    BOOST_CHECK_EQUAL(XorAndSemiring<float>().mult(0.f, 1.f), 0.0f);
    BOOST_CHECK_EQUAL(XorAndSemiring<float>().mult(-2.f, 0.f), 0.0f);
    BOOST_CHECK_EQUAL(XorAndSemiring<float>().mult(0.f, 0.f), 0.0f);

    BOOST_CHECK_EQUAL(XorAndSemiring<uint64_t>().zero(), 0UL);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint64_t>().add(2UL, 1UL), 0UL);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint64_t>().add(2UL, 0UL), 1UL);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint64_t>().add(0UL, 1UL), 1UL);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint64_t>().add(0UL, 0UL), 0UL);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint64_t>().mult(2UL, 1UL), 1UL);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint64_t>().mult(2UL, 0UL), 0UL);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint64_t>().mult(0UL, 1UL), 0UL);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint64_t>().mult(0UL, 0UL), 0UL);

    BOOST_CHECK_EQUAL(XorAndSemiring<uint32_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint32_t>().add(2U, 1U), 0U);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint32_t>().add(2U, 0U), 1U);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint32_t>().add(0U, 1U), 1U);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint32_t>().add(0U, 0U), 0U);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint32_t>().mult(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint32_t>().mult(2U, 0U), 0U);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint32_t>().mult(0U, 1U), 0U);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint32_t>().mult(0U, 0U), 0U);

    BOOST_CHECK_EQUAL(XorAndSemiring<uint16_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint16_t>().add(2U, 1U), 0U);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint16_t>().add(2U, 0U), 1U);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint16_t>().add(0U, 1U), 1U);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint16_t>().add(0U, 0U), 0U);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint16_t>().mult(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint16_t>().mult(2U, 0U), 0U);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint16_t>().mult(0U, 1U), 0U);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint16_t>().mult(0U, 0U), 0U);

    BOOST_CHECK_EQUAL(XorAndSemiring<uint8_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint8_t>().add(2U, 1U), 0U);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint8_t>().add(2U, 0U), 1U);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint8_t>().add(0U, 1U), 1U);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint8_t>().add(0U, 0U), 0U);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint8_t>().mult(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint8_t>().mult(2U, 0U), 0U);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint8_t>().mult(0U, 1U), 0U);
    BOOST_CHECK_EQUAL(XorAndSemiring<uint8_t>().mult(0U, 0U), 0U);

    BOOST_CHECK_EQUAL(XorAndSemiring<int64_t>().zero(), 0L);
    BOOST_CHECK_EQUAL(XorAndSemiring<int64_t>().add(-2L, 1L), 0L);
    BOOST_CHECK_EQUAL(XorAndSemiring<int64_t>().add(-2L, 0L), 1L);
    BOOST_CHECK_EQUAL(XorAndSemiring<int64_t>().add(0L, 1L), 1L);
    BOOST_CHECK_EQUAL(XorAndSemiring<int64_t>().add(0L, 0L), 0L);
    BOOST_CHECK_EQUAL(XorAndSemiring<int64_t>().mult(-2L, 1L), 1L);
    BOOST_CHECK_EQUAL(XorAndSemiring<int64_t>().mult(-2L, 0L), 0L);
    BOOST_CHECK_EQUAL(XorAndSemiring<int64_t>().mult(0L, 1L), 0L);
    BOOST_CHECK_EQUAL(XorAndSemiring<int64_t>().mult(0L, 0L), 0L);

    BOOST_CHECK_EQUAL(XorAndSemiring<int32_t>().zero(), 0);
    BOOST_CHECK_EQUAL(XorAndSemiring<int32_t>().add(-2, 1), 0);
    BOOST_CHECK_EQUAL(XorAndSemiring<int32_t>().add(-2, 0), 1);
    BOOST_CHECK_EQUAL(XorAndSemiring<int32_t>().add(0, 1), 1);
    BOOST_CHECK_EQUAL(XorAndSemiring<int32_t>().add(0, 0), 0);
    BOOST_CHECK_EQUAL(XorAndSemiring<int32_t>().mult(-2, 1), 1);
    BOOST_CHECK_EQUAL(XorAndSemiring<int32_t>().mult(-2, 0), 0);
    BOOST_CHECK_EQUAL(XorAndSemiring<int32_t>().mult(0, 1), 0);
    BOOST_CHECK_EQUAL(XorAndSemiring<int32_t>().mult(0, 0), 0);

    BOOST_CHECK_EQUAL(XorAndSemiring<int16_t>().zero(), 0);
    BOOST_CHECK_EQUAL(XorAndSemiring<int16_t>().add(-2, 1), 0);
    BOOST_CHECK_EQUAL(XorAndSemiring<int16_t>().add(-2, 0), 1);
    BOOST_CHECK_EQUAL(XorAndSemiring<int16_t>().add(0, 1), 1);
    BOOST_CHECK_EQUAL(XorAndSemiring<int16_t>().add(0, 0), 0);
    BOOST_CHECK_EQUAL(XorAndSemiring<int16_t>().mult(-2, 1), 1);
    BOOST_CHECK_EQUAL(XorAndSemiring<int16_t>().mult(-2, 0), 0);
    BOOST_CHECK_EQUAL(XorAndSemiring<int16_t>().mult(0, 1), 0);
    BOOST_CHECK_EQUAL(XorAndSemiring<int16_t>().mult(0, 0), 0);

    BOOST_CHECK_EQUAL(XorAndSemiring<int8_t>().zero(), 0);
    BOOST_CHECK_EQUAL(XorAndSemiring<int8_t>().add(-2, 1), 0);
    BOOST_CHECK_EQUAL(XorAndSemiring<int8_t>().add(-2, 0), 1);
    BOOST_CHECK_EQUAL(XorAndSemiring<int8_t>().add(0, 1), 1);
    BOOST_CHECK_EQUAL(XorAndSemiring<int8_t>().add(0, 0), 0);
    BOOST_CHECK_EQUAL(XorAndSemiring<int8_t>().mult(-2, 1), 1);
    BOOST_CHECK_EQUAL(XorAndSemiring<int8_t>().mult(-2, 0), 0);
    BOOST_CHECK_EQUAL(XorAndSemiring<int8_t>().mult(0, 1), 0);
    BOOST_CHECK_EQUAL(XorAndSemiring<int8_t>().mult(0, 0), 0);

    BOOST_CHECK_EQUAL(XorAndSemiring<bool>().zero(), false);
    BOOST_CHECK_EQUAL(XorAndSemiring<bool>().add(false, false), false);
    BOOST_CHECK_EQUAL(XorAndSemiring<bool>().add(false, true), true);
    BOOST_CHECK_EQUAL(XorAndSemiring<bool>().add(true, false), true);
    BOOST_CHECK_EQUAL(XorAndSemiring<bool>().add(true, true), false);
    BOOST_CHECK_EQUAL(XorAndSemiring<bool>().mult(false, false), false);
    BOOST_CHECK_EQUAL(XorAndSemiring<bool>().mult(false, true), false);
    BOOST_CHECK_EQUAL(XorAndSemiring<bool>().mult(true, false), false);
    BOOST_CHECK_EQUAL(XorAndSemiring<bool>().mult(true, true), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(xnor_or_semiring_test)
{
    BOOST_CHECK_EQUAL(XnorOrSemiring<double>().zero(), 1.0);
    BOOST_CHECK_EQUAL(XnorOrSemiring<double>().mult(-2., 1.), 1.0);
    BOOST_CHECK_EQUAL(XnorOrSemiring<double>().mult(0., 1.), 1.0);
    BOOST_CHECK_EQUAL(XnorOrSemiring<double>().mult(-2., 0.), 1.0);
    BOOST_CHECK_EQUAL(XnorOrSemiring<double>().mult(0., 0.), 0.0);
    BOOST_CHECK_EQUAL(XnorOrSemiring<double>().add(-2., 1.), 1.0);
    BOOST_CHECK_EQUAL(XnorOrSemiring<double>().add(0., 1.), 0.0);
    BOOST_CHECK_EQUAL(XnorOrSemiring<double>().add(-2., 0.), 0.0);
    BOOST_CHECK_EQUAL(XnorOrSemiring<double>().add(0., 0.), 1.0);

    BOOST_CHECK_EQUAL(XnorOrSemiring<float>().zero(), 1.0f);
    BOOST_CHECK_EQUAL(XnorOrSemiring<float>().mult(-2.f, 1.f), 1.0f);
    BOOST_CHECK_EQUAL(XnorOrSemiring<float>().mult(0.f, 1.f), 1.0f);
    BOOST_CHECK_EQUAL(XnorOrSemiring<float>().mult(-2.f, 0.f), 1.0f);
    BOOST_CHECK_EQUAL(XnorOrSemiring<float>().mult(0.f, 0.f), 0.0f);
    BOOST_CHECK_EQUAL(XnorOrSemiring<float>().add(-2.f, 1.f), 1.0f);
    BOOST_CHECK_EQUAL(XnorOrSemiring<float>().add(0.f, 1.f), 0.0f);
    BOOST_CHECK_EQUAL(XnorOrSemiring<float>().add(-2.f, 0.f), 0.0f);
    BOOST_CHECK_EQUAL(XnorOrSemiring<float>().add(0.f, 0.f), 1.0f);

    BOOST_CHECK_EQUAL(XnorOrSemiring<uint64_t>().zero(), 1UL);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint64_t>().mult(2UL, 1UL), 1UL);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint64_t>().mult(2UL, 0UL), 1UL);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint64_t>().mult(0UL, 1UL), 1UL);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint64_t>().mult(0UL, 0UL), 0UL);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint64_t>().add(2UL, 1UL), 1UL);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint64_t>().add(2UL, 0UL), 0UL);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint64_t>().add(0UL, 1UL), 0UL);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint64_t>().add(0UL, 0UL), 1UL);

    BOOST_CHECK_EQUAL(XnorOrSemiring<uint32_t>().zero(), 1U);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint32_t>().mult(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint32_t>().mult(2U, 0U), 1U);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint32_t>().mult(0U, 1U), 1U);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint32_t>().mult(0U, 0U), 0U);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint32_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint32_t>().add(2U, 0U), 0U);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint32_t>().add(0U, 1U), 0U);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint32_t>().add(0U, 0U), 1U);

    BOOST_CHECK_EQUAL(XnorOrSemiring<uint16_t>().zero(), 1U);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint16_t>().mult(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint16_t>().mult(2U, 0U), 1U);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint16_t>().mult(0U, 1U), 1U);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint16_t>().mult(0U, 0U), 0U);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint16_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint16_t>().add(2U, 0U), 0U);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint16_t>().add(0U, 1U), 0U);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint16_t>().add(0U, 0U), 1U);

    BOOST_CHECK_EQUAL(XnorOrSemiring<uint8_t>().zero(), 1U);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint8_t>().mult(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint8_t>().mult(2U, 0U), 1U);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint8_t>().mult(0U, 1U), 1U);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint8_t>().mult(0U, 0U), 0U);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint8_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint8_t>().add(2U, 0U), 0U);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint8_t>().add(0U, 1U), 0U);
    BOOST_CHECK_EQUAL(XnorOrSemiring<uint8_t>().add(0U, 0U), 1U);

    BOOST_CHECK_EQUAL(XnorOrSemiring<int64_t>().zero(), 1L);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int64_t>().mult(-2L, 1L), 1L);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int64_t>().mult(-2L, 0L), 1L);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int64_t>().mult(0L, 1L), 1L);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int64_t>().mult(0L, 0L), 0L);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int64_t>().add(-2L, 1L), 1L);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int64_t>().add(-2L, 0L), 0L);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int64_t>().add(0L, 1L), 0L);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int64_t>().add(0L, 0L), 1L);

    BOOST_CHECK_EQUAL(XnorOrSemiring<int32_t>().zero(), 1);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int32_t>().mult(-2, 1), 1);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int32_t>().mult(-2, 0), 1);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int32_t>().mult(0, 1), 1);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int32_t>().mult(0, 0), 0);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int32_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int32_t>().add(-2, 0), 0);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int32_t>().add(0, 1), 0);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int32_t>().add(0, 0), 1);

    BOOST_CHECK_EQUAL(XnorOrSemiring<int16_t>().zero(), 1);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int16_t>().mult(-2, 1), 1);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int16_t>().mult(-2, 0), 1);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int16_t>().mult(0, 1), 1);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int16_t>().mult(0, 0), 0);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int16_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int16_t>().add(-2, 0), 0);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int16_t>().add(0, 1), 0);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int16_t>().add(0, 0), 1);

    BOOST_CHECK_EQUAL(XnorOrSemiring<int8_t>().zero(), 1);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int8_t>().mult(-2, 1), 1);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int8_t>().mult(-2, 0), 1);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int8_t>().mult(0, 1), 1);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int8_t>().mult(0, 0), 0);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int8_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int8_t>().add(-2, 0), 0);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int8_t>().add(0, 1), 0);
    BOOST_CHECK_EQUAL(XnorOrSemiring<int8_t>().add(0, 0), 1);

    BOOST_CHECK_EQUAL(XnorOrSemiring<bool>().zero(), true);
    BOOST_CHECK_EQUAL(XnorOrSemiring<bool>().mult(false, false), false);
    BOOST_CHECK_EQUAL(XnorOrSemiring<bool>().mult(false, true), true);
    BOOST_CHECK_EQUAL(XnorOrSemiring<bool>().mult(true, false), true);
    BOOST_CHECK_EQUAL(XnorOrSemiring<bool>().mult(true, true), true);
    BOOST_CHECK_EQUAL(XnorOrSemiring<bool>().add(false, false), true);
    BOOST_CHECK_EQUAL(XnorOrSemiring<bool>().add(false, true), false);
    BOOST_CHECK_EQUAL(XnorOrSemiring<bool>().add(true, false), false);
    BOOST_CHECK_EQUAL(XnorOrSemiring<bool>().add(true, true), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(min_first_test)
{
    BOOST_CHECK_EQUAL(MinFirstSemiring<double>().zero(),
                      std::numeric_limits<double>::infinity());
    BOOST_CHECK_EQUAL(MinFirstSemiring<double>().add(-2., 1.), -2.0);
    BOOST_CHECK_EQUAL(MinFirstSemiring<double>().add(2., 1.), 1.0);
    BOOST_CHECK_EQUAL(MinFirstSemiring<double>().mult(-2., 1.), -2.0);

    BOOST_CHECK_EQUAL(MinFirstSemiring<float>().zero(),
                      std::numeric_limits<float>::infinity());
    BOOST_CHECK_EQUAL(MinFirstSemiring<float>().add(-2.f, 1.f), -2.0f);
    BOOST_CHECK_EQUAL(MinFirstSemiring<float>().add(2.f, 1.f), 1.0f);
    BOOST_CHECK_EQUAL(MinFirstSemiring<float>().mult(-2.f, 1.f), -2.0f);

    BOOST_CHECK_EQUAL(MinFirstSemiring<uint64_t>().zero(),
                      std::numeric_limits<uint64_t>::max());
    BOOST_CHECK_EQUAL(MinFirstSemiring<uint64_t>().add(2UL, 1UL), 1UL);
    BOOST_CHECK_EQUAL(MinFirstSemiring<uint64_t>().add(2UL, 3UL), 2UL);
    BOOST_CHECK_EQUAL(MinFirstSemiring<uint64_t>().mult(2UL, 1UL), 2UL);

    BOOST_CHECK_EQUAL(MinFirstSemiring<uint32_t>().zero(),
                      std::numeric_limits<uint32_t>::max());
    BOOST_CHECK_EQUAL(MinFirstSemiring<uint32_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(MinFirstSemiring<uint32_t>().add(2U, 3U), 2U);
    BOOST_CHECK_EQUAL(MinFirstSemiring<uint32_t>().mult(2U, 1U), 2U);

    BOOST_CHECK_EQUAL(MinFirstSemiring<uint16_t>().zero(),
                      std::numeric_limits<uint16_t>::max());
    BOOST_CHECK_EQUAL(MinFirstSemiring<uint16_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(MinFirstSemiring<uint16_t>().add(2U, 3U), 2U);
    BOOST_CHECK_EQUAL(MinFirstSemiring<uint16_t>().mult(2U, 1U), 2U);

    BOOST_CHECK_EQUAL(MinFirstSemiring<uint8_t>().zero(),
                      std::numeric_limits<uint8_t>::max());
    BOOST_CHECK_EQUAL(MinFirstSemiring<uint8_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(MinFirstSemiring<uint8_t>().add(2U, 3U), 2U);
    BOOST_CHECK_EQUAL(MinFirstSemiring<uint8_t>().mult(2U, 1U), 2U);

    BOOST_CHECK_EQUAL(MinFirstSemiring<int64_t>().zero(),
                      std::numeric_limits<int64_t>::max());
    BOOST_CHECK_EQUAL(MinFirstSemiring<int64_t>().add(-2L, 1L), -2L);
    BOOST_CHECK_EQUAL(MinFirstSemiring<int64_t>().add(2L, -1L), -1L);
    BOOST_CHECK_EQUAL(MinFirstSemiring<int64_t>().mult(-2L, 1L), -2L);

    BOOST_CHECK_EQUAL(MinFirstSemiring<int32_t>().zero(),
                      std::numeric_limits<int32_t>::max());
    BOOST_CHECK_EQUAL(MinFirstSemiring<int32_t>().add(-2, 1), -2);
    BOOST_CHECK_EQUAL(MinFirstSemiring<int32_t>().add(2, -1), -1);
    BOOST_CHECK_EQUAL(MinFirstSemiring<int32_t>().mult(-2, 1), -2);

    BOOST_CHECK_EQUAL(MinFirstSemiring<int16_t>().zero(),
                      std::numeric_limits<int16_t>::max());
    BOOST_CHECK_EQUAL(MinFirstSemiring<int16_t>().add(-2, 1), -2);
    BOOST_CHECK_EQUAL(MinFirstSemiring<int16_t>().add(2, -1), -1);
    BOOST_CHECK_EQUAL(MinFirstSemiring<int16_t>().mult(-2, 1), -2);

    BOOST_CHECK_EQUAL(MinFirstSemiring<int8_t>().zero(),
                      std::numeric_limits<int8_t>::max());
    BOOST_CHECK_EQUAL(MinFirstSemiring<int8_t>().add(-2, 1), -2);
    BOOST_CHECK_EQUAL(MinFirstSemiring<int8_t>().add(2, -1), -1);
    BOOST_CHECK_EQUAL(MinFirstSemiring<int8_t>().mult(-2, 1), -2);

    BOOST_CHECK_EQUAL(MinFirstSemiring<bool>().zero(), true);
    BOOST_CHECK_EQUAL(MinFirstSemiring<bool>().add(false, true), false);
    BOOST_CHECK_EQUAL(MinFirstSemiring<bool>().add(true, true), true);
    BOOST_CHECK_EQUAL(MinFirstSemiring<bool>().mult(true, false), true);
    BOOST_CHECK_EQUAL(MinFirstSemiring<bool>().mult(false, true), false);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(min_second_test)
{
    BOOST_CHECK_EQUAL(MinSecondSemiring<double>().zero(),
                      std::numeric_limits<double>::infinity());
    BOOST_CHECK_EQUAL(MinSecondSemiring<double>().add(-2., 1.), -2.0);
    BOOST_CHECK_EQUAL(MinSecondSemiring<double>().add(2., 1.), 1.0);
    BOOST_CHECK_EQUAL(MinSecondSemiring<double>().mult(-2., 1.), 1.0);

    BOOST_CHECK_EQUAL(MinSecondSemiring<float>().zero(),
                      std::numeric_limits<float>::infinity());
    BOOST_CHECK_EQUAL(MinSecondSemiring<float>().add(-2.f, 1.f), -2.0f);
    BOOST_CHECK_EQUAL(MinSecondSemiring<float>().add(2.f, 1.f), 1.0f);
    BOOST_CHECK_EQUAL(MinSecondSemiring<float>().mult(-2.f, 1.f), 1.0f);

    BOOST_CHECK_EQUAL(MinSecondSemiring<uint64_t>().zero(),
                      std::numeric_limits<uint64_t>::max());
    BOOST_CHECK_EQUAL(MinSecondSemiring<uint64_t>().add(2UL, 1UL), 1UL);
    BOOST_CHECK_EQUAL(MinSecondSemiring<uint64_t>().add(2UL, 3UL), 2UL);
    BOOST_CHECK_EQUAL(MinSecondSemiring<uint64_t>().mult(2UL, 1UL), 1UL);

    BOOST_CHECK_EQUAL(MinSecondSemiring<uint32_t>().zero(),
                      std::numeric_limits<uint32_t>::max());
    BOOST_CHECK_EQUAL(MinSecondSemiring<uint32_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(MinSecondSemiring<uint32_t>().add(2U, 3U), 2U);
    BOOST_CHECK_EQUAL(MinSecondSemiring<uint32_t>().mult(2U, 1U), 1U);

    BOOST_CHECK_EQUAL(MinSecondSemiring<uint16_t>().zero(),
                      std::numeric_limits<uint16_t>::max());
    BOOST_CHECK_EQUAL(MinSecondSemiring<uint16_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(MinSecondSemiring<uint16_t>().add(2U, 3U), 2U);
    BOOST_CHECK_EQUAL(MinSecondSemiring<uint16_t>().mult(2U, 1U), 1U);

    BOOST_CHECK_EQUAL(MinSecondSemiring<uint8_t>().zero(),
                      std::numeric_limits<uint8_t>::max());
    BOOST_CHECK_EQUAL(MinSecondSemiring<uint8_t>().add(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(MinSecondSemiring<uint8_t>().add(2U, 3U), 2U);
    BOOST_CHECK_EQUAL(MinSecondSemiring<uint8_t>().mult(2U, 1U), 1U);

    BOOST_CHECK_EQUAL(MinSecondSemiring<int64_t>().zero(),
                      std::numeric_limits<int64_t>::max());
    BOOST_CHECK_EQUAL(MinSecondSemiring<int64_t>().add(-2L, 1L), -2L);
    BOOST_CHECK_EQUAL(MinSecondSemiring<int64_t>().add(2L, -1L), -1L);
    BOOST_CHECK_EQUAL(MinSecondSemiring<int64_t>().mult(-2L, 1L), 1L);

    BOOST_CHECK_EQUAL(MinSecondSemiring<int32_t>().zero(),
                      std::numeric_limits<int32_t>::max());
    BOOST_CHECK_EQUAL(MinSecondSemiring<int32_t>().add(-2, 1), -2);
    BOOST_CHECK_EQUAL(MinSecondSemiring<int32_t>().add(2, -1), -1);
    BOOST_CHECK_EQUAL(MinSecondSemiring<int32_t>().mult(-2, 1), 1);

    BOOST_CHECK_EQUAL(MinSecondSemiring<int16_t>().zero(),
                      std::numeric_limits<int16_t>::max());
    BOOST_CHECK_EQUAL(MinSecondSemiring<int16_t>().add(-2, 1), -2);
    BOOST_CHECK_EQUAL(MinSecondSemiring<int16_t>().add(2, -1), -1);
    BOOST_CHECK_EQUAL(MinSecondSemiring<int16_t>().mult(-2, 1), 1);

    BOOST_CHECK_EQUAL(MinSecondSemiring<int8_t>().zero(),
                      std::numeric_limits<int8_t>::max());
    BOOST_CHECK_EQUAL(MinSecondSemiring<int8_t>().add(-2, 1), -2);
    BOOST_CHECK_EQUAL(MinSecondSemiring<int8_t>().add(2, -1), -1);
    BOOST_CHECK_EQUAL(MinSecondSemiring<int8_t>().mult(-2, 1), 1);

    BOOST_CHECK_EQUAL(MinSecondSemiring<bool>().zero(), true);
    BOOST_CHECK_EQUAL(MinSecondSemiring<bool>().add(false, true), false);
    BOOST_CHECK_EQUAL(MinSecondSemiring<bool>().add(true, true), true);
    BOOST_CHECK_EQUAL(MinSecondSemiring<bool>().mult(true, false), false);
    BOOST_CHECK_EQUAL(MinSecondSemiring<bool>().mult(false, true), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(max_first_test)
{
    BOOST_CHECK_EQUAL(MaxFirstSemiring<double>().zero(),
                      -std::numeric_limits<double>::infinity());
    BOOST_CHECK_EQUAL(MaxFirstSemiring<double>().add(-2., 1.), 1.0);
    BOOST_CHECK_EQUAL(MaxFirstSemiring<double>().mult(-2., 1.), -2.0);
    BOOST_CHECK_EQUAL(MaxFirstSemiring<float>().zero(),
                      -std::numeric_limits<float>::infinity());
    BOOST_CHECK_EQUAL(MaxFirstSemiring<float>().add(-2.f, 1.f), 1.0f);
    BOOST_CHECK_EQUAL(MaxFirstSemiring<float>().mult(-2.f, 1.f), -2.0f);

    BOOST_CHECK_EQUAL(MaxFirstSemiring<uint64_t>().zero(), 0UL);
    BOOST_CHECK_EQUAL(MaxFirstSemiring<uint64_t>().add(2UL, 1UL), 2UL);
    BOOST_CHECK_EQUAL(MaxFirstSemiring<uint64_t>().mult(2UL, 1UL), 2UL);
    BOOST_CHECK_EQUAL(MaxFirstSemiring<uint32_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(MaxFirstSemiring<uint32_t>().add(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(MaxFirstSemiring<uint32_t>().mult(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(MaxFirstSemiring<uint16_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(MaxFirstSemiring<uint16_t>().add(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(MaxFirstSemiring<uint16_t>().mult(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(MaxFirstSemiring<uint8_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(MaxFirstSemiring<uint8_t>().add(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(MaxFirstSemiring<uint8_t>().mult(2U, 1U), 2U);

    BOOST_CHECK_EQUAL(MaxFirstSemiring<int64_t>().zero(),
                      std::numeric_limits<int64_t>::min());
    BOOST_CHECK_EQUAL(MaxFirstSemiring<int64_t>().add(-2L, 1L), 1L);
    BOOST_CHECK_EQUAL(MaxFirstSemiring<int64_t>().mult(-2L, 1L), -2L);
    BOOST_CHECK_EQUAL(MaxFirstSemiring<int32_t>().zero(),
                      std::numeric_limits<int32_t>::min());
    BOOST_CHECK_EQUAL(MaxFirstSemiring<int32_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(MaxFirstSemiring<int32_t>().mult(-2, 1), -2);
    BOOST_CHECK_EQUAL(MaxFirstSemiring<int16_t>().zero(),
                      std::numeric_limits<int16_t>::min());
    BOOST_CHECK_EQUAL(MaxFirstSemiring<int16_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(MaxFirstSemiring<int16_t>().mult(-2, 1), -2);
    BOOST_CHECK_EQUAL(MaxFirstSemiring<int8_t>().zero(),
                      std::numeric_limits<int8_t>::min());
    BOOST_CHECK_EQUAL(MaxFirstSemiring<int8_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(MaxFirstSemiring<int8_t>().mult(-2, 1), -2);

    BOOST_CHECK_EQUAL(MaxFirstSemiring<bool>().zero(), false);
    BOOST_CHECK_EQUAL(MaxFirstSemiring<bool>().add(false, false), false);
    BOOST_CHECK_EQUAL(MaxFirstSemiring<bool>().add(false, true), true);
    BOOST_CHECK_EQUAL(MaxFirstSemiring<bool>().mult(false, true), false);
    BOOST_CHECK_EQUAL(MaxFirstSemiring<bool>().mult(true, false), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(max_second_test)
{
    BOOST_CHECK_EQUAL(MaxSecondSemiring<double>().zero(),
                      -std::numeric_limits<double>::infinity());
    BOOST_CHECK_EQUAL(MaxSecondSemiring<double>().add(-2., 1.), 1.0);
    BOOST_CHECK_EQUAL(MaxSecondSemiring<double>().mult(-2., 1.), 1.0);
    BOOST_CHECK_EQUAL(MaxSecondSemiring<float>().zero(),
                      -std::numeric_limits<float>::infinity());
    BOOST_CHECK_EQUAL(MaxSecondSemiring<float>().add(-2.f, 1.f), 1.0f);
    BOOST_CHECK_EQUAL(MaxSecondSemiring<float>().mult(-2.f, 1.f), 1.0f);

    BOOST_CHECK_EQUAL(MaxSecondSemiring<uint64_t>().zero(), 0UL);
    BOOST_CHECK_EQUAL(MaxSecondSemiring<uint64_t>().add(2UL, 1UL), 2UL);
    BOOST_CHECK_EQUAL(MaxSecondSemiring<uint64_t>().mult(2UL, 1UL), 1UL);
    BOOST_CHECK_EQUAL(MaxSecondSemiring<uint32_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(MaxSecondSemiring<uint32_t>().add(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(MaxSecondSemiring<uint32_t>().mult(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(MaxSecondSemiring<uint16_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(MaxSecondSemiring<uint16_t>().add(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(MaxSecondSemiring<uint16_t>().mult(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(MaxSecondSemiring<uint8_t>().zero(), 0U);
    BOOST_CHECK_EQUAL(MaxSecondSemiring<uint8_t>().add(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(MaxSecondSemiring<uint8_t>().mult(2U, 1U), 1U);

    BOOST_CHECK_EQUAL(MaxSecondSemiring<int64_t>().zero(),
                      std::numeric_limits<int64_t>::min());
    BOOST_CHECK_EQUAL(MaxSecondSemiring<int64_t>().add(-2L, 1L), 1L);
    BOOST_CHECK_EQUAL(MaxSecondSemiring<int64_t>().mult(-2L, 1L), 1L);
    BOOST_CHECK_EQUAL(MaxSecondSemiring<int32_t>().zero(),
                      std::numeric_limits<int32_t>::min());
    BOOST_CHECK_EQUAL(MaxSecondSemiring<int32_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(MaxSecondSemiring<int32_t>().mult(-2, 1), 1);
    BOOST_CHECK_EQUAL(MaxSecondSemiring<int16_t>().zero(),
                      std::numeric_limits<int16_t>::min());
    BOOST_CHECK_EQUAL(MaxSecondSemiring<int16_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(MaxSecondSemiring<int16_t>().mult(-2, 1), 1);
    BOOST_CHECK_EQUAL(MaxSecondSemiring<int8_t>().zero(),
                      std::numeric_limits<int8_t>::min());
    BOOST_CHECK_EQUAL(MaxSecondSemiring<int8_t>().add(-2, 1), 1);
    BOOST_CHECK_EQUAL(MaxSecondSemiring<int8_t>().mult(-2, 1), 1);

    BOOST_CHECK_EQUAL(MaxSecondSemiring<bool>().zero(), false);
    BOOST_CHECK_EQUAL(MaxSecondSemiring<bool>().add(false, false), false);
    BOOST_CHECK_EQUAL(MaxSecondSemiring<bool>().add(false, true), true);
    BOOST_CHECK_EQUAL(MaxSecondSemiring<bool>().mult(false, true), true);
    BOOST_CHECK_EQUAL(MaxSecondSemiring<bool>().mult(true, false), false);
}

BOOST_AUTO_TEST_SUITE_END()
