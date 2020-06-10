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

#include <graphblas/graphblas.hpp>

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE ewisemult_vector_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

// ***************************************************************************

namespace
{
    std::vector<double> v3a_dense = {12, 0, 7};

    std::vector<double> v4a_dense = {0, 0, 12, 7};
    std::vector<double> v4b_dense = {0,  1, 0, 2};
    std::vector<double> v4c_dense = {3,  6, 9, 1};

    std::vector<double> zero3_dense = {0, 0, 0};
    std::vector<double> zero4_dense = {0, 0, 0, 0};

    std::vector<double> twos3_dense = {2, 2, 2};
    std::vector<double> twos4_dense = {2, 2, 2, 2};

    std::vector<double> ans_4atwos4_dense = {0, 0, 24, 14};
    std::vector<double> ans_4a4b_dense = {0, 0, 0, 14};
}

// ***************************************************************************
// Tests without mask
// ***************************************************************************

// ***************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_vector_bad_dimensions)
{
    GraphBLAS::Vector<double> v(v3a_dense, 0.);
    GraphBLAS::Vector<double> u(v4a_dense, 0.);

    GraphBLAS::Vector<double> result(3);

    // incompatible input dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseMult(result,
                              GraphBLAS::NoMask(),
                              GraphBLAS::NoAccumulate(),
                              GraphBLAS::Times<double>(), v, u)),
        GraphBLAS::DimensionException);

    // incompatible output vector dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseMult(result,
                              GraphBLAS::NoMask(),
                              GraphBLAS::NoAccumulate(),
                              GraphBLAS::Times<double>(), u, u)),
        GraphBLAS::DimensionException);
}

// ***************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_vector_reg)
{
    GraphBLAS::Vector<double> u(v4a_dense, 0.);

    // ewise mult with dense vector
    GraphBLAS::Vector<double> v(twos4_dense, 0.);
    GraphBLAS::Vector<double> ans(ans_4atwos4_dense, 0.);

    GraphBLAS::Vector<double> result(4);

    GraphBLAS::eWiseMult(result,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), u, v);
    BOOST_CHECK_EQUAL(result, ans);

    // ewise mult with sparse vector
    GraphBLAS::Vector<double> v2(v4b_dense, 0.);
    GraphBLAS::Vector<double> ans2(ans_4a4b_dense, 0.);
    GraphBLAS::eWiseMult(result,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), u, v2);
    BOOST_CHECK_EQUAL(result, ans2);

    // ewise mult with empty vector
    GraphBLAS::Vector<double> v3(zero4_dense, 0.);
    GraphBLAS::Vector<double> ans3(zero4_dense, 0.);
    GraphBLAS::eWiseMult(result,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), u, v3);
    BOOST_CHECK_EQUAL(result, ans3);
}

// ***************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_vector_stored_zero_result)
{
    GraphBLAS::Vector<double> u(v4a_dense); //, 0.);

    // ewise mult with dense vector
    GraphBLAS::Vector<double> v(twos4_dense, 0.);
    GraphBLAS::Vector<double> ans(ans_4atwos4_dense); //, 0.);

    GraphBLAS::Vector<double> result(4);

    GraphBLAS::eWiseMult(result,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), u, v);
    BOOST_CHECK_EQUAL(result, ans);

    // ewise mult with sparse vector
    GraphBLAS::Vector<double> v2(v4b_dense, 0.);
    GraphBLAS::Vector<double> ans2(ans_4a4b_dense, 0.);
    ans2.setElement(1, 0.);
    GraphBLAS::eWiseMult(result,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), u, v2);
    BOOST_CHECK_EQUAL(result, ans2);

    // ewise mult with empty vector
    GraphBLAS::Vector<double> v3(zero4_dense, 0.);
    GraphBLAS::Vector<double> ans3(zero4_dense, 0.);
    GraphBLAS::eWiseMult(result,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), u, v3);
    BOOST_CHECK_EQUAL(result, ans3);
}

// ***************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_semiring_vector_stored_zero_result)
{
    GraphBLAS::Vector<double> u(v4a_dense); //, 0.);

    // ewise mult with dense vector
    GraphBLAS::Vector<double> v(twos4_dense, 0.);
    GraphBLAS::Vector<double> ans(ans_4atwos4_dense); //, 0.);

    GraphBLAS::Vector<double> result(4);

    GraphBLAS::eWiseMult(result,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         multiply_op(GraphBLAS::ArithmeticSemiring<double>()),
                         u, v);
    BOOST_CHECK_EQUAL(result, ans);

    // ewise mult with sparse vector
    GraphBLAS::Vector<double> v2(v4b_dense, 0.);
    GraphBLAS::Vector<double> ans2(ans_4a4b_dense, 0.);
    ans2.setElement(1, 0.);
    GraphBLAS::eWiseMult(result,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         multiply_op(GraphBLAS::ArithmeticSemiring<double>()),
                         u, v2);
    BOOST_CHECK_EQUAL(result, ans2);

    // ewise mult with empty vector
    GraphBLAS::Vector<double> v3(zero4_dense, 0.);
    GraphBLAS::Vector<double> ans3(zero4_dense, 0.);
    GraphBLAS::eWiseMult(result,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         multiply_op(GraphBLAS::ArithmeticSemiring<double>()),
                         u, v3);
    BOOST_CHECK_EQUAL(result, ans3);
}

// ***************************************************************************
// Tests using a mask with REPLACE
// ***************************************************************************

// ***************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_vector_masked_replace_bad_dimensions)
{
    GraphBLAS::Vector<double> v(v4a_dense, 0.);
    GraphBLAS::Vector<double> u(v4b_dense, 0.);
    GraphBLAS::Vector<double> mask(v3a_dense, 0.);
    GraphBLAS::Vector<double> result(4);

    // incompatible Mask-Output dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseMult(result,
                              mask,
                              GraphBLAS::NoAccumulate(),
                              GraphBLAS::Times<double>(), v, u, GraphBLAS::REPLACE)),
        GraphBLAS::DimensionException);
}

// ***************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_vector_masked_replace_reg)
{
    std::vector<int> mask4_dense = {0, 1, 1, 0};
    GraphBLAS::Vector<int> mask(mask4_dense, 0);

    GraphBLAS::Vector<double> u(v4a_dense, 0.);

    // ewise mult with dense vector
    GraphBLAS::Vector<double> v(twos4_dense, 0.);
    std::vector<double> ans_4atwos4_dense2 = {0, 0, 24, 0};
    GraphBLAS::Vector<double> ans(ans_4atwos4_dense2, 0.);

    GraphBLAS::Vector<double> result(4);

    GraphBLAS::eWiseMult(result,
                         mask,
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), u, v, GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result, ans);

    // ewise mult with sparse vector
    GraphBLAS::Vector<double> v2(v4b_dense, 0.);
    GraphBLAS::Vector<double> ans2(zero4_dense, 0.);
    GraphBLAS::eWiseMult(result,
                         mask,
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), u, v2, GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result, ans2);

    // ewise mult with empty vector
    GraphBLAS::Vector<double> v3(zero4_dense, 0.);
    GraphBLAS::Vector<double> ans3(zero4_dense, 0.);
    GraphBLAS::eWiseMult(result,
                         mask,
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), u, v3, GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result, ans3);
}

// ***************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_vector_masked_replace_reg_stored_zero)
{
    std::vector<int> mask4_dense = {0, 1, 1, 0};
    GraphBLAS::Vector<int> mask(mask4_dense);//, 0);

    GraphBLAS::Vector<double> u(v4a_dense, 0.);

    // ewise mult with dense vector
    GraphBLAS::Vector<double> v(twos4_dense, 0.);
    std::vector<double> ans_4atwos4_dense2 = {0, 0, 24, 0};
    GraphBLAS::Vector<double> ans(ans_4atwos4_dense2, 0.);

    GraphBLAS::Vector<double> result(4);

    GraphBLAS::eWiseMult(result,
                         mask,
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), u, v, GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result, ans);

    // ewise mult with sparse vector
    GraphBLAS::Vector<double> v2(v4b_dense, 0.);
    GraphBLAS::Vector<double> ans2(zero4_dense, 0.);
    GraphBLAS::eWiseMult(result,
                         mask,
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), u, v2, GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result, ans2);

    // ewise mult with empty vector
    GraphBLAS::Vector<double> v3(zero4_dense, 0.);
    GraphBLAS::Vector<double> ans3(zero4_dense, 0.);
    GraphBLAS::eWiseMult(result,
                         mask,
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), u, v3, GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result, ans3);
}

// ***************************************************************************
// Tests using a mask with MERGE semantics
// ***************************************************************************

// ***************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_vector_masked_bad_dimensions)
{
    GraphBLAS::Vector<double> v(v4a_dense, 0.);
    GraphBLAS::Vector<double> u(v4b_dense, 0.);
    GraphBLAS::Vector<double> mask(v3a_dense, 0.);
    GraphBLAS::Vector<double> result(4);

    // incompatible Mask-Output dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseMult(result,
                              mask,
                              GraphBLAS::NoAccumulate(),
                              GraphBLAS::Times<double>(), v, u)),
        GraphBLAS::DimensionException);
}

// ***************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_vector_masked_reg)
{
    std::vector<int> mask4_dense = {0, 1, 1, 0};
    GraphBLAS::Vector<int> mask(mask4_dense, 0);

    GraphBLAS::Vector<double> u(v4a_dense, 0.);

    {
        // ewise mult with dense vector
        GraphBLAS::Vector<double> v(twos4_dense, 0.);

        GraphBLAS::Vector<double> result(twos4_dense);

        std::vector<double> ans_dense = {2, 0, 24, 2};
        GraphBLAS::Vector<double> ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(result,
                             mask,
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Times<double>(), u, v, GraphBLAS::MERGE);
        BOOST_CHECK_EQUAL(result, ans);
    }

    {
        // ewise mult with sparse vector
        GraphBLAS::Vector<double> v2(v4b_dense, 0.);

        GraphBLAS::Vector<double> result(twos4_dense);

        std::vector<double> ans_dense = {2, 0, 0, 2};
        GraphBLAS::Vector<double> ans2(ans_dense, 0.);

        GraphBLAS::eWiseMult(result,
                             mask,
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Times<double>(), u, v2, GraphBLAS::MERGE);
        BOOST_CHECK_EQUAL(result, ans2);
    }

    {
        // ewise mult with empty vector
        GraphBLAS::Vector<double> v3(zero4_dense, 0.);

        GraphBLAS::Vector<double> result(twos4_dense);

        std::vector<double> ans_dense = {2, 0, 0, 2};
        GraphBLAS::Vector<double> ans3(ans_dense, 0.);

        GraphBLAS::eWiseMult(result,
                             mask,
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Times<double>(), u, v3, GraphBLAS::MERGE);
        BOOST_CHECK_EQUAL(result, ans3);
    }
}

// ***************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_vector_masked_reg_stored_zero)
{

    std::vector<int> mask4_dense = {0, 1, 1, 0};
    GraphBLAS::Vector<int> mask(mask4_dense); //, 0);

    GraphBLAS::Vector<double> u(v4a_dense, 0.);

    {
        // ewise mult with dense vector
        GraphBLAS::Vector<double> v(twos4_dense, 0.);

        GraphBLAS::Vector<double> result(twos4_dense);

        std::vector<double> ans_dense = {2, 0, 24, 2};
        GraphBLAS::Vector<double> ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(result,
                             mask,
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Times<double>(), u, v, GraphBLAS::MERGE);
        BOOST_CHECK_EQUAL(result, ans);
    }

    {
        // ewise mult with sparse vector
        GraphBLAS::Vector<double> v2(v4b_dense, 0.);

        GraphBLAS::Vector<double> result(twos4_dense);

        std::vector<double> ans_dense = {2, 0, 0, 2};
        GraphBLAS::Vector<double> ans2(ans_dense, 0.);

        GraphBLAS::eWiseMult(result,
                             mask,
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Times<double>(), u, v2, GraphBLAS::MERGE);
        BOOST_CHECK_EQUAL(result, ans2);
    }

    {
        // ewise mult with empty vector
        GraphBLAS::Vector<double> v3(zero4_dense, 0.);

        GraphBLAS::Vector<double> result(twos4_dense);

        std::vector<double> ans_dense = {2, 0, 0, 2};
        GraphBLAS::Vector<double> ans3(ans_dense, 0.);

        GraphBLAS::eWiseMult(result,
                             mask,
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Times<double>(), u, v3, GraphBLAS::MERGE);
        BOOST_CHECK_EQUAL(result, ans3);
    }
}

// ***************************************************************************
// Tests using a complemented mask with REPLACE semantics
// ***************************************************************************

// ***************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_vector_scmp_masked_replace_bad_dimensions)
{
    GraphBLAS::Vector<double> v(v4a_dense, 0.);
    GraphBLAS::Vector<double> u(v4b_dense, 0.);
    GraphBLAS::Vector<double> mask(v3a_dense, 0.);
    GraphBLAS::Vector<double> result(4);

    // incompatible Mask-Output dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseMult(result,
                              GraphBLAS::complement(mask),
                              GraphBLAS::NoAccumulate(),
                              GraphBLAS::Times<double>(), v, u, GraphBLAS::REPLACE)),
        GraphBLAS::DimensionException);
}

// ***************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_vector_scmp_masked_replace_reg)
{
    std::vector<int> mask4_dense = {1, 0, 0, 1};
    GraphBLAS::Vector<int> mask(mask4_dense, 0);

    GraphBLAS::Vector<double> u(v4a_dense, 0.);

    // ewise mult with dense vector
    GraphBLAS::Vector<double> v(twos4_dense, 0.);
    std::vector<double> ans_4atwos4_dense2 = {0, 0, 24, 0};
    GraphBLAS::Vector<double> ans(ans_4atwos4_dense2, 0.);

    GraphBLAS::Vector<double> result(4);

    GraphBLAS::eWiseMult(result,
                         GraphBLAS::complement(mask),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), u, v, GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result, ans);

    // ewise mult with sparse vector
    GraphBLAS::Vector<double> v2(v4b_dense, 0.);
    GraphBLAS::Vector<double> ans2(zero4_dense, 0.);
    GraphBLAS::eWiseMult(result,
                         GraphBLAS::complement(mask),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), u, v2, GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result, ans2);

    // ewise mult with empty vector
    GraphBLAS::Vector<double> v3(zero4_dense, 0.);
    GraphBLAS::Vector<double> ans3(zero4_dense, 0.);
    GraphBLAS::eWiseMult(result,
                         GraphBLAS::complement(mask),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), u, v3, GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result, ans3);
}

// ***************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_vector_scmp_masked_replace_reg_stored_zero)
{
    std::vector<int> mask4_dense = {1, 0, 0, 1};
    GraphBLAS::Vector<int> mask(mask4_dense);//, 0);

    GraphBLAS::Vector<double> u(v4a_dense, 0.);

    // ewise mult with dense vector
    GraphBLAS::Vector<double> v(twos4_dense, 0.);
    std::vector<double> ans_4atwos4_dense2 = {0, 0, 24, 0};
    GraphBLAS::Vector<double> ans(ans_4atwos4_dense2, 0.);

    GraphBLAS::Vector<double> result(4);

    GraphBLAS::eWiseMult(result,
                         GraphBLAS::complement(mask),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), u, v, GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result, ans);

    // ewise mult with sparse vector
    GraphBLAS::Vector<double> v2(v4b_dense, 0.);
    GraphBLAS::Vector<double> ans2(zero4_dense, 0.);
    GraphBLAS::eWiseMult(result,
                         GraphBLAS::complement(mask),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), u, v2, GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result, ans2);

    // ewise mult with empty vector
    GraphBLAS::Vector<double> v3(zero4_dense, 0.);
    GraphBLAS::Vector<double> ans3(zero4_dense, 0.);
    GraphBLAS::eWiseMult(result,
                         GraphBLAS::complement(mask),
                         GraphBLAS::NoAccumulate(),
                         GraphBLAS::Times<double>(), u, v3, GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result, ans3);
}

// ***************************************************************************
// Tests using a complemented mask (with merge semantics)
// ***************************************************************************

// ***************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_vector_scmp_masked_bad_dimensions)
{
    GraphBLAS::Vector<double> v(v4a_dense, 0.);
    GraphBLAS::Vector<double> u(v4b_dense, 0.);
    GraphBLAS::Vector<double> mask(v3a_dense, 0.);
    GraphBLAS::Vector<double> result(4);

    // incompatible Mask-Output dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseMult(result,
                              GraphBLAS::complement(mask),
                              GraphBLAS::NoAccumulate(),
                              GraphBLAS::Times<double>(), v, u)),
        GraphBLAS::DimensionException);
}

// ***************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_vector_scmp_masked_reg)
{
    std::vector<int> mask4_dense = {1, 0, 0, 1};
    GraphBLAS::Vector<int> mask(mask4_dense, 0);

    GraphBLAS::Vector<double> u(v4a_dense, 0.);

    {
        // ewise mult with dense vector
        GraphBLAS::Vector<double> v(twos4_dense, 0.);

        GraphBLAS::Vector<double> result(twos4_dense);

        std::vector<double> ans_dense = {2, 0, 24, 2};
        GraphBLAS::Vector<double> ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(result,
                             GraphBLAS::complement(mask),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Times<double>(), u, v, GraphBLAS::MERGE);
        BOOST_CHECK_EQUAL(result, ans);
    }

    {
        // ewise mult with sparse vector
        GraphBLAS::Vector<double> v2(v4b_dense, 0.);

        GraphBLAS::Vector<double> result(twos4_dense);

        std::vector<double> ans_dense = {2, 0, 0, 2};
        GraphBLAS::Vector<double> ans2(ans_dense, 0.);

        GraphBLAS::eWiseMult(result,
                             GraphBLAS::complement(mask),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Times<double>(), u, v2, GraphBLAS::MERGE);
        BOOST_CHECK_EQUAL(result, ans2);
    }

    {
        // ewise mult with empty vector
        GraphBLAS::Vector<double> v3(zero4_dense, 0.);

        GraphBLAS::Vector<double> result(twos4_dense);

        std::vector<double> ans_dense = {2, 0, 0, 2};
        GraphBLAS::Vector<double> ans3(ans_dense, 0.);

        GraphBLAS::eWiseMult(result,
                             GraphBLAS::complement(mask),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Times<double>(), u, v3, GraphBLAS::MERGE);
        BOOST_CHECK_EQUAL(result, ans3);
    }
}

// ***************************************************************************
BOOST_AUTO_TEST_CASE(test_ewisemult_vector_scmp_masked_reg_stored_zero)
{
    std::vector<int> mask4_dense = {1, 0, 0, 1};
    GraphBLAS::Vector<int> mask(mask4_dense); //0);

    GraphBLAS::Vector<double> u(v4a_dense, 0.);

    {
        // ewise mult with dense vector
        GraphBLAS::Vector<double> v(twos4_dense, 0.);

        GraphBLAS::Vector<double> result(twos4_dense);

        std::vector<double> ans_dense = {2, 0, 24, 2};
        GraphBLAS::Vector<double> ans(ans_dense, 0.);

        GraphBLAS::eWiseMult(result,
                             GraphBLAS::complement(mask),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Times<double>(), u, v, GraphBLAS::MERGE);
        BOOST_CHECK_EQUAL(result, ans);
    }

    {
        // ewise mult with sparse vector
        GraphBLAS::Vector<double> v2(v4b_dense, 0.);

        GraphBLAS::Vector<double> result(twos4_dense);

        std::vector<double> ans_dense = {2, 0, 0, 2};
        GraphBLAS::Vector<double> ans2(ans_dense, 0.);

        GraphBLAS::eWiseMult(result,
                             GraphBLAS::complement(mask),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Times<double>(), u, v2, GraphBLAS::MERGE);
        BOOST_CHECK_EQUAL(result, ans2);
    }

    {
        // ewise mult with empty vector
        GraphBLAS::Vector<double> v3(zero4_dense, 0.);

        GraphBLAS::Vector<double> result(twos4_dense);

        std::vector<double> ans_dense = {2, 0, 0, 2};
        GraphBLAS::Vector<double> ans3(ans_dense, 0.);

        GraphBLAS::eWiseMult(result,
                             GraphBLAS::complement(mask),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Times<double>(), u, v3, GraphBLAS::MERGE);
        BOOST_CHECK_EQUAL(result, ans3);
    }
}


BOOST_AUTO_TEST_SUITE_END()
