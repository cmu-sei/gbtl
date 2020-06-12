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
#define BOOST_TEST_MODULE algebra_monoid_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

//****************************************************************************
// Monoid tests
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(plus_monoid_test)
{
    BOOST_CHECK_EQUAL(PlusMonoid<double>().identity(), 0.0);
    BOOST_CHECK_EQUAL(PlusMonoid<double>()(-2., 1.), -1.0);
    BOOST_CHECK_EQUAL(PlusMonoid<float>().identity(), 0.0f);
    BOOST_CHECK_EQUAL(PlusMonoid<float>()(-2.f, 1.f), -1.0f);

    BOOST_CHECK_EQUAL(PlusMonoid<uint64_t>().identity(), 0UL);
    BOOST_CHECK_EQUAL(PlusMonoid<uint64_t>()(2UL, 1UL), 3UL);
    BOOST_CHECK_EQUAL(PlusMonoid<uint32_t>().identity(), 0U);
    BOOST_CHECK_EQUAL(PlusMonoid<uint32_t>()(2U, 1U), 3U);
    BOOST_CHECK_EQUAL(PlusMonoid<uint16_t>().identity(), 0U);
    BOOST_CHECK_EQUAL(PlusMonoid<uint16_t>()(2U, 1U), 3U);
    BOOST_CHECK_EQUAL(PlusMonoid<uint8_t>().identity(), 0U);
    BOOST_CHECK_EQUAL(PlusMonoid<uint8_t>()(2U, 1U), 3U);

    BOOST_CHECK_EQUAL(PlusMonoid<int64_t>().identity(), 0L);
    BOOST_CHECK_EQUAL(PlusMonoid<int64_t>()(-2L, 1L), -1L);
    BOOST_CHECK_EQUAL(PlusMonoid<int32_t>().identity(), 0);
    BOOST_CHECK_EQUAL(PlusMonoid<int32_t>()(-2, 1), -1);
    BOOST_CHECK_EQUAL(PlusMonoid<int16_t>().identity(), 0);
    BOOST_CHECK_EQUAL(PlusMonoid<int16_t>()(-2, 1), -1);
    BOOST_CHECK_EQUAL(PlusMonoid<int8_t>().identity(), 0);
    BOOST_CHECK_EQUAL(PlusMonoid<int8_t>()(-2, 1), -1);

    BOOST_CHECK_EQUAL(PlusMonoid<bool>().identity(), false);
    BOOST_CHECK_EQUAL(PlusMonoid<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(PlusMonoid<bool>()(true, true), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(times_monoid_test)
{
    BOOST_CHECK_EQUAL(TimesMonoid<double>().identity(), 1.0);
    BOOST_CHECK_EQUAL(TimesMonoid<double>()(-2., 1.), -2.0);
    BOOST_CHECK_EQUAL(TimesMonoid<float>().identity(), 1.0f);
    BOOST_CHECK_EQUAL(TimesMonoid<float>()(-2.f, 1.f), -2.0f);

    BOOST_CHECK_EQUAL(TimesMonoid<uint64_t>().identity(), 1UL);
    BOOST_CHECK_EQUAL(TimesMonoid<uint64_t>()(2UL, 1UL), 2UL);
    BOOST_CHECK_EQUAL(TimesMonoid<uint32_t>().identity(), 1U);
    BOOST_CHECK_EQUAL(TimesMonoid<uint32_t>()(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(TimesMonoid<uint16_t>().identity(), 1U);
    BOOST_CHECK_EQUAL(TimesMonoid<uint16_t>()(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(TimesMonoid<uint8_t>().identity(), 1U);
    BOOST_CHECK_EQUAL(TimesMonoid<uint8_t>()(2U, 1U), 2U);

    BOOST_CHECK_EQUAL(TimesMonoid<int64_t>().identity(), 1L);
    BOOST_CHECK_EQUAL(TimesMonoid<int64_t>()(-2L, 1L), -2L);
    BOOST_CHECK_EQUAL(TimesMonoid<int32_t>().identity(), 1);
    BOOST_CHECK_EQUAL(TimesMonoid<int32_t>()(-2, 1), -2);
    BOOST_CHECK_EQUAL(TimesMonoid<int16_t>().identity(), 1);
    BOOST_CHECK_EQUAL(TimesMonoid<int16_t>()(-2, 1), -2);
    BOOST_CHECK_EQUAL(TimesMonoid<int8_t>().identity(), 1);
    BOOST_CHECK_EQUAL(TimesMonoid<int8_t>()(-2, 1), -2);

    BOOST_CHECK_EQUAL(TimesMonoid<bool>().identity(), true);
    BOOST_CHECK_EQUAL(TimesMonoid<bool>()(false, true), false);
    BOOST_CHECK_EQUAL(TimesMonoid<bool>()(true, true), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(min_monoid_test)
{
    BOOST_CHECK_EQUAL(MinMonoid<double>().identity(),
                      std::numeric_limits<double>::infinity());
    BOOST_CHECK_EQUAL(MinMonoid<double>()(-2., 1.), -2.0);
    BOOST_CHECK_EQUAL(MinMonoid<float>().identity(),
                      std::numeric_limits<float>::infinity());
    BOOST_CHECK_EQUAL(MinMonoid<float>()(-2.f, 1.f), -2.0f);

    BOOST_CHECK_EQUAL(MinMonoid<uint64_t>().identity(),
                      std::numeric_limits<uint64_t>::max());
    BOOST_CHECK_EQUAL(MinMonoid<uint64_t>()(2UL, 1UL), 1UL);
    BOOST_CHECK_EQUAL(MinMonoid<uint32_t>().identity(),
                      std::numeric_limits<uint32_t>::max());
    BOOST_CHECK_EQUAL(MinMonoid<uint32_t>()(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(MinMonoid<uint16_t>().identity(),
                      std::numeric_limits<uint16_t>::max());
    BOOST_CHECK_EQUAL(MinMonoid<uint16_t>()(2U, 1U), 1U);
    BOOST_CHECK_EQUAL(MinMonoid<uint8_t>().identity(),
                      std::numeric_limits<uint8_t>::max());
    BOOST_CHECK_EQUAL(MinMonoid<uint8_t>()(2U, 1U), 1U);

    BOOST_CHECK_EQUAL(MinMonoid<int64_t>().identity(),
                      std::numeric_limits<int64_t>::max());
    BOOST_CHECK_EQUAL(MinMonoid<int64_t>()(-2L, 1L), -2L);
    BOOST_CHECK_EQUAL(MinMonoid<int32_t>().identity(),
                      std::numeric_limits<int32_t>::max());
    BOOST_CHECK_EQUAL(MinMonoid<int32_t>()(-2, 1), -2);
    BOOST_CHECK_EQUAL(MinMonoid<int16_t>().identity(),
                      std::numeric_limits<int16_t>::max());
    BOOST_CHECK_EQUAL(MinMonoid<int16_t>()(-2, 1), -2);
    BOOST_CHECK_EQUAL(MinMonoid<int8_t>().identity(),
                      std::numeric_limits<int8_t>::max());
    BOOST_CHECK_EQUAL(MinMonoid<int8_t>()(-2, 1), -2);

    BOOST_CHECK_EQUAL(MinMonoid<bool>().identity(), true);
    BOOST_CHECK_EQUAL(MinMonoid<bool>()(false, true), false);
    BOOST_CHECK_EQUAL(MinMonoid<bool>()(true, true), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(max_monoid_test)
{
    BOOST_CHECK_EQUAL(MaxMonoid<double>().identity(),
                      -std::numeric_limits<double>::infinity());
    BOOST_CHECK_EQUAL(MaxMonoid<double>()(-2., 1.), 1.0);
    BOOST_CHECK_EQUAL(MaxMonoid<float>().identity(),
                      -std::numeric_limits<float>::infinity());
    BOOST_CHECK_EQUAL(MaxMonoid<float>()(-2.f, 1.f), 1.0f);

    BOOST_CHECK_EQUAL(MaxMonoid<uint64_t>().identity(), 0UL);
    BOOST_CHECK_EQUAL(MaxMonoid<uint64_t>()(2UL, 1UL), 2UL);
    BOOST_CHECK_EQUAL(MaxMonoid<uint32_t>().identity(), 0U);
    BOOST_CHECK_EQUAL(MaxMonoid<uint32_t>()(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(MaxMonoid<uint16_t>().identity(), 0U);
    BOOST_CHECK_EQUAL(MaxMonoid<uint16_t>()(2U, 1U), 2U);
    BOOST_CHECK_EQUAL(MaxMonoid<uint8_t>().identity(), 0U);
    BOOST_CHECK_EQUAL(MaxMonoid<uint8_t>()(2U, 1U), 2U);

    BOOST_CHECK_EQUAL(MaxMonoid<int64_t>().identity(),
                      std::numeric_limits<int64_t>::min());
    BOOST_CHECK_EQUAL(MaxMonoid<int64_t>()(-2L, 1L), 1L);
    BOOST_CHECK_EQUAL(MaxMonoid<int32_t>().identity(),
                      std::numeric_limits<int32_t>::min());
    BOOST_CHECK_EQUAL(MaxMonoid<int32_t>()(-2, 1), 1);
    BOOST_CHECK_EQUAL(MaxMonoid<int16_t>().identity(),
                      std::numeric_limits<int16_t>::min());
    BOOST_CHECK_EQUAL(MaxMonoid<int16_t>()(-2, 1), 1);
    BOOST_CHECK_EQUAL(MaxMonoid<int8_t>().identity(),
                      std::numeric_limits<int8_t>::min());
    BOOST_CHECK_EQUAL(MaxMonoid<int8_t>()(-2, 1), 1);

    BOOST_CHECK_EQUAL(MaxMonoid<bool>().identity(), false);
    BOOST_CHECK_EQUAL(MaxMonoid<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(MaxMonoid<bool>()(false, true), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(logical_or_monoid_test)
{
    BOOST_CHECK_EQUAL(LogicalOrMonoid<double>().identity(), 0.0);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<double>()(0., 0.),  0.0);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<double>()(0., 1.),  1.0);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<double>()(-2., 0.), 1.0);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<double>()(1., -1.), 1.0);

    BOOST_CHECK_EQUAL(LogicalOrMonoid<float>().identity(), 0.0f);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<float>()(0.f, 0.f),  0.0f);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<float>()(0.f, 1.f),  1.0f);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<float>()(-2.f, 0.f), 1.0f);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<float>()(1.f, -1.f), 1.0f);

    BOOST_CHECK_EQUAL(LogicalOrMonoid<uint64_t>().identity(), 0UL);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<uint64_t>()(0UL, 0UL), 0UL);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<uint64_t>()(0UL, 1UL), 1UL);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<uint64_t>()(2UL, 0UL), 1UL);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<uint64_t>()(2UL, 1UL), 1UL);

    BOOST_CHECK_EQUAL(LogicalOrMonoid<uint32_t>().identity(), 0U);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<uint32_t>()(0U, 0U), 0U);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<uint32_t>()(0U, 1U), 1U);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<uint32_t>()(2U, 0U), 1U);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<uint32_t>()(2U, 1U), 1U);

    BOOST_CHECK_EQUAL(LogicalOrMonoid<uint16_t>().identity(), 0U);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<uint16_t>()(0U, 0U), 0U);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<uint16_t>()(0U, 1U), 1U);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<uint16_t>()(2U, 0U), 1U);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<uint16_t>()(2U, 1U), 1U);

    BOOST_CHECK_EQUAL(LogicalOrMonoid<uint8_t>().identity(), 0U);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<uint8_t>()(0U, 0U), 0U);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<uint8_t>()(0U, 1U), 1U);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<uint8_t>()(2U, 0U), 1U);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<uint8_t>()(2U, 1U), 1U);

    BOOST_CHECK_EQUAL(LogicalOrMonoid<int64_t>().identity(), 0L);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<int64_t>()(0L, 0L), 0L);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<int64_t>()(0L, 1L), 1L);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<int64_t>()(-2L, 0L), 1L);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<int64_t>()(-2L, 1L), 1L);

    BOOST_CHECK_EQUAL(LogicalOrMonoid<int32_t>().identity(), 0);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<int32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<int32_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<int32_t>()(-2, 0), 1);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<int32_t>()(-2, 1), 1);

    BOOST_CHECK_EQUAL(LogicalOrMonoid<int16_t>().identity(), 0);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<int16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<int16_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<int16_t>()(-2, 0), 1);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<int16_t>()(-2, 1), 1);

    BOOST_CHECK_EQUAL(LogicalOrMonoid<int8_t>().identity(), 0);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<int8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<int8_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<int8_t>()(-2, 0), 1);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<int8_t>()(-2, 1), 1);

    BOOST_CHECK_EQUAL(LogicalOrMonoid<bool>().identity(), false);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<bool>()(false, true), true);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<bool>()(true, false), true);
    BOOST_CHECK_EQUAL(LogicalOrMonoid<bool>()(true, true), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(logical_and_monoid_test)
{
    BOOST_CHECK_EQUAL(LogicalAndMonoid<double>().identity(), 1.0);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<double>()(0., 0.),  0.0);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<double>()(0., 1.),  0.0);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<double>()(-2., 0.), 0.0);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<double>()(1., -1.), 1.0);

    BOOST_CHECK_EQUAL(LogicalAndMonoid<float>().identity(), 1.0f);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<float>()(0.f, 0.f),  0.0f);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<float>()(0.f, 1.f),  0.0f);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<float>()(-2.f, 0.f), 0.0f);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<float>()(1.f, -1.f), 1.0f);

    BOOST_CHECK_EQUAL(LogicalAndMonoid<uint64_t>().identity(), 1UL);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<uint64_t>()(0UL, 0UL), 0UL);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<uint64_t>()(0UL, 1UL), 0UL);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<uint64_t>()(2UL, 0UL), 0UL);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<uint64_t>()(2UL, 1UL), 1UL);

    BOOST_CHECK_EQUAL(LogicalAndMonoid<uint32_t>().identity(), 1U);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<uint32_t>()(0U, 0U), 0U);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<uint32_t>()(0U, 1U), 0U);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<uint32_t>()(2U, 0U), 0U);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<uint32_t>()(2U, 1U), 1U);

    BOOST_CHECK_EQUAL(LogicalAndMonoid<uint16_t>().identity(), 1U);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<uint16_t>()(0U, 0U), 0U);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<uint16_t>()(0U, 1U), 0U);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<uint16_t>()(2U, 0U), 0U);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<uint16_t>()(2U, 1U), 1U);

    BOOST_CHECK_EQUAL(LogicalAndMonoid<uint8_t>().identity(), 1U);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<uint8_t>()(0U, 0U), 0U);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<uint8_t>()(0U, 1U), 0U);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<uint8_t>()(2U, 0U), 0U);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<uint8_t>()(2U, 1U), 1U);

    BOOST_CHECK_EQUAL(LogicalAndMonoid<int64_t>().identity(), 1L);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<int64_t>()(0L, 0L), 0L);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<int64_t>()(0L, 1L), 0L);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<int64_t>()(-2L, 0L), 0L);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<int64_t>()(-2L, 1L), 1L);

    BOOST_CHECK_EQUAL(LogicalAndMonoid<int32_t>().identity(), 1);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<int32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<int32_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<int32_t>()(-2, 0), 0);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<int32_t>()(-2, 1), 1);

    BOOST_CHECK_EQUAL(LogicalAndMonoid<int16_t>().identity(), 1);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<int16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<int16_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<int16_t>()(-2, 0), 0);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<int16_t>()(-2, 1), 1);

    BOOST_CHECK_EQUAL(LogicalAndMonoid<int8_t>().identity(), 1);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<int8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<int8_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<int8_t>()(-2, 0), 0);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<int8_t>()(-2, 1), 1);

    BOOST_CHECK_EQUAL(LogicalAndMonoid<bool>().identity(),  true);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<bool>()(false, true), false);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<bool>()(true, false), false);
    BOOST_CHECK_EQUAL(LogicalAndMonoid<bool>()(true, true), true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(logical_xor_monoid_test)
{
    BOOST_CHECK_EQUAL(LogicalXorMonoid<double>().identity(), 0.0);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<double>()(0., 0.),  0.0);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<double>()(0., 1.),  1.0);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<double>()(-2., 0.), 1.0);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<double>()(1., -1.), 0.0);

    BOOST_CHECK_EQUAL(LogicalXorMonoid<float>().identity(), 0.0f);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<float>()(0.f, 0.f),  0.0f);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<float>()(0.f, 1.f),  1.0f);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<float>()(-2.f, 0.f), 1.0f);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<float>()(1.f, -1.f), 0.0f);

    BOOST_CHECK_EQUAL(LogicalXorMonoid<uint64_t>().identity(), 0UL);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<uint64_t>()(0UL, 0UL), 0UL);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<uint64_t>()(0UL, 1UL), 1UL);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<uint64_t>()(2UL, 0UL), 1UL);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<uint64_t>()(2UL, 1UL), 0UL);

    BOOST_CHECK_EQUAL(LogicalXorMonoid<uint32_t>().identity(), 0U);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<uint32_t>()(0U, 0U), 0U);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<uint32_t>()(0U, 1U), 1U);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<uint32_t>()(2U, 0U), 1U);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<uint32_t>()(2U, 1U), 0U);

    BOOST_CHECK_EQUAL(LogicalXorMonoid<uint16_t>().identity(), 0U);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<uint16_t>()(0U, 0U), 0U);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<uint16_t>()(0U, 1U), 1U);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<uint16_t>()(2U, 0U), 1U);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<uint16_t>()(2U, 1U), 0U);

    BOOST_CHECK_EQUAL(LogicalXorMonoid<uint8_t>().identity(), 0U);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<uint8_t>()(0U, 0U), 0U);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<uint8_t>()(0U, 1U), 1U);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<uint8_t>()(2U, 0U), 1U);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<uint8_t>()(2U, 1U), 0U);

    BOOST_CHECK_EQUAL(LogicalXorMonoid<int64_t>().identity(), 0L);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<int64_t>()(0L, 0L), 0L);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<int64_t>()(0L, 1L), 1L);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<int64_t>()(-2L, 0L), 1L);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<int64_t>()(-2L, 1L), 0L);

    BOOST_CHECK_EQUAL(LogicalXorMonoid<int32_t>().identity(), 0);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<int32_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<int32_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<int32_t>()(-2, 0), 1);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<int32_t>()(-2, 1), 0);

    BOOST_CHECK_EQUAL(LogicalXorMonoid<int16_t>().identity(), 0);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<int16_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<int16_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<int16_t>()(-2, 0), 1);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<int16_t>()(-2, 1), 0);

    BOOST_CHECK_EQUAL(LogicalXorMonoid<int8_t>().identity(), 0);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<int8_t>()(0, 0), 0);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<int8_t>()(0, 1), 1);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<int8_t>()(-2, 0), 1);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<int8_t>()(-2, 1), 0);

    BOOST_CHECK_EQUAL(LogicalXorMonoid<bool>().identity(), false);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<bool>()(false, false), false);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<bool>()(false, true),  true);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<bool>()(true, false),  true);
    BOOST_CHECK_EQUAL(LogicalXorMonoid<bool>()(true, true), false);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(logical_xnor_monoid_test)
{
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<double>().identity(), 1.0);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<double>()(0., 0.),  1.0);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<double>()(0., 1.),  0.0);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<double>()(-2., 0.), 0.0);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<double>()(1., -1.), 1.0);

    BOOST_CHECK_EQUAL(LogicalXnorMonoid<float>().identity(), 1.0f);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<float>()(0.f, 0.f),  1.0f);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<float>()(0.f, 1.f),  0.0f);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<float>()(-2.f, 0.f), 0.0f);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<float>()(1.f, -1.f), 1.0f);

    BOOST_CHECK_EQUAL(LogicalXnorMonoid<uint64_t>().identity(), 1UL);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<uint64_t>()(0UL, 0UL), 1UL);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<uint64_t>()(0UL, 1UL), 0UL);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<uint64_t>()(2UL, 0UL), 0UL);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<uint64_t>()(2UL, 1UL), 1UL);

    BOOST_CHECK_EQUAL(LogicalXnorMonoid<uint32_t>().identity(), 1U);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<uint32_t>()(0U, 0U), 1U);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<uint32_t>()(0U, 1U), 0U);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<uint32_t>()(2U, 0U), 0U);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<uint32_t>()(2U, 1U), 1U);

    BOOST_CHECK_EQUAL(LogicalXnorMonoid<uint16_t>().identity(), 1U);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<uint16_t>()(0U, 0U), 1U);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<uint16_t>()(0U, 1U), 0U);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<uint16_t>()(2U, 0U), 0U);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<uint16_t>()(2U, 1U), 1U);

    BOOST_CHECK_EQUAL(LogicalXnorMonoid<uint8_t>().identity(), 1U);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<uint8_t>()(0U, 0U), 1U);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<uint8_t>()(0U, 1U), 0U);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<uint8_t>()(2U, 0U), 0U);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<uint8_t>()(2U, 1U), 1U);

    BOOST_CHECK_EQUAL(LogicalXnorMonoid<int64_t>().identity(), 1L);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<int64_t>()(0L, 0L), 1L);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<int64_t>()(0L, 1L), 0L);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<int64_t>()(-2L, 0L), 0L);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<int64_t>()(-2L, 1L), 1L);

    BOOST_CHECK_EQUAL(LogicalXnorMonoid<int32_t>().identity(), 1);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<int32_t>()(0, 0), 1);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<int32_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<int32_t>()(-2, 0), 0);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<int32_t>()(-2, 1), 1);

    BOOST_CHECK_EQUAL(LogicalXnorMonoid<int16_t>().identity(), 1);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<int16_t>()(0, 0), 1);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<int16_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<int16_t>()(-2, 0), 0);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<int16_t>()(-2, 1), 1);

    BOOST_CHECK_EQUAL(LogicalXnorMonoid<int8_t>().identity(), 1);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<int8_t>()(0, 0), 1);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<int8_t>()(0, 1), 0);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<int8_t>()(-2, 0), 0);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<int8_t>()(-2, 1), 1);

    BOOST_CHECK_EQUAL(LogicalXnorMonoid<bool>().identity(), true);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<bool>()(false, false), true);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<bool>()(false, true), false);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<bool>()(true, false), false);
    BOOST_CHECK_EQUAL(LogicalXnorMonoid<bool>()(true, true),  true);
}

BOOST_AUTO_TEST_SUITE_END()
