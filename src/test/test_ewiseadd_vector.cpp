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
#define BOOST_TEST_MODULE sparse_ewiseadd_vector_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

/// @todo add tests with accumulate

//****************************************************************************

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

    std::vector<double> ans_4atwos4_dense = {2, 2, 14, 9};
    std::vector<double> ans_4a4b_dense = {0, 1, 12, 9};
}

//****************************************************************************
// Tests without mask
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_vector_bad_dimensions)
{
    GraphBLAS::Vector<double> v(v3a_dense, 0.);
    GraphBLAS::Vector<double> u(v4a_dense, 0.);

    GraphBLAS::Vector<double> result(3);

    // incompatible input dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseAdd(result,
                             GraphBLAS::NoMask(),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Plus<double>(), v, u)),
        GraphBLAS::DimensionException);

    // incompatible output vector dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseAdd(result,
                             GraphBLAS::NoMask(),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Plus<double>(), u, u)),
        GraphBLAS::DimensionException);

    // Additional tests.
    GraphBLAS::Vector<double> a(v3a_dense, 0.);
    GraphBLAS::Vector<double> b(v4a_dense, 0.);
    GraphBLAS::Vector<double> result_a(3);
    GraphBLAS::Vector<double> result_b(4);

    // w0 m0 u1 v1
    BOOST_CHECK_THROW(
            (GraphBLAS::eWiseAdd(result_a,
                                 a,
                                 GraphBLAS::NoAccumulate(),
                                 GraphBLAS::Plus<double>(),
                                 b,
                                 b)),
            GraphBLAS::DimensionException);

    // w0 m1 u0 v0
    BOOST_CHECK_THROW(
            (GraphBLAS::eWiseAdd(result_a,
                                 b,
                                 GraphBLAS::NoAccumulate(),
                                 GraphBLAS::Plus<double>(),
                                 a,
                                 a)),
            GraphBLAS::DimensionException);

    // w1 m0 u0 v1
    BOOST_CHECK_THROW(
            (GraphBLAS::eWiseAdd(result_b,
                                 a,
                                 GraphBLAS::NoAccumulate(),
                                 GraphBLAS::Plus<double>(),
                                 a,
                                 b)),
            GraphBLAS::DimensionException);

    // w1 m1 u1 v0
    BOOST_CHECK_THROW(
            (GraphBLAS::eWiseAdd(result_b,
                                 b,
                                 GraphBLAS::NoAccumulate(),
                                 GraphBLAS::Plus<double>(),
                                 b,
                                 a)),
            GraphBLAS::DimensionException);

    // w1 m0 u1 v1
    BOOST_CHECK_THROW(
            (GraphBLAS::eWiseAdd(result_b,
                                 a,
                                 GraphBLAS::NoAccumulate(),
                                 GraphBLAS::Plus<double>(),
                                 b,
                                 b)),
            GraphBLAS::DimensionException);

    // w0 m1 u1 v1
    BOOST_CHECK_THROW(
            (GraphBLAS::eWiseAdd(result_a,
                                 b,
                                 GraphBLAS::NoAccumulate(),
                                 GraphBLAS::Plus<double>(),
                                 b,
                                 b)),
            GraphBLAS::DimensionException);

    // w0 no u1 v0
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseAdd(result_a,
                             GraphBLAS::NoMask(),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Plus<double>(),
                             b,
                             a)),
        GraphBLAS::DimensionException);

    // w1 no u1 v0
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseAdd(result_b,
                             GraphBLAS::NoMask(),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Plus<double>(),
                             b,
                             a)),
        GraphBLAS::DimensionException);

    // w1 no u0 v0
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseAdd(result_b,
                             GraphBLAS::NoMask(),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Plus<double>(),
                             a,
                             a)),
        GraphBLAS::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_vector_reg)
{
    GraphBLAS::Vector<double> u(v4a_dense, 0.);

    // ewise add with dense vector
    GraphBLAS::Vector<double> v(twos4_dense, 0.);
    GraphBLAS::Vector<double> ans(ans_4atwos4_dense, 0.);

    GraphBLAS::Vector<double> result(4);

    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::NoMask(),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v);
    BOOST_CHECK_EQUAL(result, ans);

    // ewise add with sparse vector
    GraphBLAS::Vector<double> v2(v4b_dense, 0.);
    GraphBLAS::Vector<double> ans2(ans_4a4b_dense, 0.);
    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::NoMask(),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v2);
    BOOST_CHECK_EQUAL(result, ans2);

    // ewise add with empty vector
    GraphBLAS::Vector<double> v3(zero4_dense, 0.);
    GraphBLAS::Vector<double> ans3(v4a_dense, 0.);
    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::NoMask(),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v3);
    BOOST_CHECK_EQUAL(result, ans3);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_vector_stored_zero_result)
{
    GraphBLAS::Vector<double> u(v4a_dense); //, 0.);

    // ewise add with dense vector
    GraphBLAS::Vector<double> v(twos4_dense, 0.);
    GraphBLAS::Vector<double> ans(ans_4atwos4_dense); //, 0.);

    GraphBLAS::Vector<double> result(4);

    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::NoMask(),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v);
    BOOST_CHECK_EQUAL(result, ans);

    // ewise add with sparse vector
    GraphBLAS::Vector<double> v2(v4b_dense, 0.);
    GraphBLAS::Vector<double> ans2(ans_4a4b_dense); //, 0.);
    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::NoMask(),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v2);
    BOOST_CHECK_EQUAL(result, ans2);

    // ewise add with empty vector
    GraphBLAS::Vector<double> v3(zero4_dense, 0.);
    GraphBLAS::Vector<double> ans3(v4a_dense);//, 0.);
    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::NoMask(),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v3);
    BOOST_CHECK_EQUAL(result, ans3);
}

//****************************************************************************
// Tests using a mask with REPLACE
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_vector_masked_replace_bad_dimensions)
{
    GraphBLAS::Vector<double> v(v4a_dense, 0.);
    GraphBLAS::Vector<double> u(v4b_dense, 0.);
    GraphBLAS::Vector<double> mask(v3a_dense, 0.);
    GraphBLAS::Vector<double> result(4);

    // incompatible Mask-Output dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseAdd(result,
                             mask,
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Plus<double>(), v, u,
                             GraphBLAS::REPLACE)),
        GraphBLAS::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_vector_masked_replace_reg)
{
    std::vector<int> mask4_dense = {0, 1, 1, 0};
    GraphBLAS::Vector<int> mask(mask4_dense, 0);

    GraphBLAS::Vector<double> u(v4a_dense, 0.);

    // ewise add with dense vector
    GraphBLAS::Vector<double> v(twos4_dense, 0.);
    std::vector<double> ans_4atwos4_dense2 = {0, 2, 14, 0};
    GraphBLAS::Vector<double> ans(ans_4atwos4_dense2, 0.);

    GraphBLAS::Vector<double> result(4);

    GraphBLAS::eWiseAdd(result,
                        mask,
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v, GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result, ans);

    // ewise add with sparse vector
    GraphBLAS::Vector<double> v2(v4b_dense, 0.);
    std::vector<double> answer2 = {0, 1, 12, 0};
    GraphBLAS::Vector<double> ans2(answer2, 0.);
    GraphBLAS::eWiseAdd(result,
                        mask,
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v2, GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result, ans2);

    // ewise add with empty vector
    GraphBLAS::Vector<double> v3(zero4_dense, 0.);
    std::vector<double> answer3 = {0, 0, 12, 0};
    GraphBLAS::Vector<double> ans3(answer3, 0.);
    GraphBLAS::eWiseAdd(result,
                        mask,
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v3, GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result, ans3);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_vector_masked_replace_reg_stored_zero)
{
    std::vector<int> mask4_dense = {0, 1, 1, 0};
    GraphBLAS::Vector<int> mask(mask4_dense); //, 0);

    GraphBLAS::Vector<double> u(v4a_dense, 0.);

    // ewise add with dense vector
    GraphBLAS::Vector<double> v(twos4_dense, 0.);
    std::vector<double> ans_4atwos4_dense2 = {0, 2, 14, 0};
    GraphBLAS::Vector<double> ans(ans_4atwos4_dense2, 0.);

    GraphBLAS::Vector<double> result(4);

    GraphBLAS::eWiseAdd(result,
                        mask,
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v, GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result, ans);

    // ewise add with sparse vector
    GraphBLAS::Vector<double> v2(v4b_dense, 0.);
    std::vector<double> answer2 = {0, 1, 12, 0};
    GraphBLAS::Vector<double> ans2(answer2, 0.);
    GraphBLAS::eWiseAdd(result,
                        mask,
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v2, GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result, ans2);

    // ewise add with empty vector
    GraphBLAS::Vector<double> v3(zero4_dense, 0.);
    std::vector<double> answer3 = {0, 0, 12, 0};
    GraphBLAS::Vector<double> ans3(answer3, 0.);
    GraphBLAS::eWiseAdd(result,
                        mask,
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v3, GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result, ans3);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_semiring_vector_masked_replace_reg_stored_zero)
{
    std::vector<int> mask4_dense = {0, 1, 1, 0};
    GraphBLAS::Vector<int> mask(mask4_dense); //, 0);

    GraphBLAS::Vector<double> u(v4a_dense, 0.);

    // ewise add with dense vector
    GraphBLAS::Vector<double> v(twos4_dense, 0.);
    std::vector<double> ans_4atwos4_dense2 = {0, 2, 14, 0};
    GraphBLAS::Vector<double> ans(ans_4atwos4_dense2, 0.);

    GraphBLAS::Vector<double> result(4);

    GraphBLAS::eWiseAdd(result,
                        mask,
                        GraphBLAS::NoAccumulate(),
                        add_monoid(GraphBLAS::ArithmeticSemiring<double>()),
                        u, v, GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result, ans);

    // ewise add with sparse vector
    GraphBLAS::Vector<double> v2(v4b_dense, 0.);
    std::vector<double> answer2 = {0, 1, 12, 0};
    GraphBLAS::Vector<double> ans2(answer2, 0.);
    GraphBLAS::eWiseAdd(result,
                        mask,
                        GraphBLAS::NoAccumulate(),
                        add_monoid(GraphBLAS::ArithmeticSemiring<double>()),
                        u, v2, GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result, ans2);

    // ewise add with empty vector
    GraphBLAS::Vector<double> v3(zero4_dense, 0.);
    std::vector<double> answer3 = {0, 0, 12, 0};
    GraphBLAS::Vector<double> ans3(answer3, 0.);
    GraphBLAS::eWiseAdd(result,
                        mask,
                        GraphBLAS::NoAccumulate(),
                        add_monoid(GraphBLAS::ArithmeticSemiring<double>()),
                        u, v3, GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result, ans3);
}

//****************************************************************************
// Tests using a mask with MERGE semantics
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_vector_masked_bad_dimensions)
{
    GraphBLAS::Vector<double> v(v4a_dense, 0.);
    GraphBLAS::Vector<double> u(v4b_dense, 0.);
    GraphBLAS::Vector<double> mask(v3a_dense, 0.);
    GraphBLAS::Vector<double> result(4);

    // incompatible Mask-Output dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseAdd(result,
                             mask,
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Plus<double>(), v, u, GraphBLAS::MERGE)),
        GraphBLAS::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_vector_masked_reg)
{
    std::vector<int> mask4_dense = {0, 1, 1, 0};
    GraphBLAS::Vector<int> mask(mask4_dense, 0);

    GraphBLAS::Vector<double> u(v4a_dense, 0.);

    // ewise add with dense vector
    GraphBLAS::Vector<double> v(twos4_dense, 0.);
    std::vector<double> ans_4atwos4_dense2 = {0, 2, 14, 0};
    GraphBLAS::Vector<double> ans(ans_4atwos4_dense2, 0.);

    GraphBLAS::Vector<double> result(4);

    GraphBLAS::eWiseAdd(result,
                        mask,
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v, GraphBLAS::MERGE);
    BOOST_CHECK_EQUAL(result, ans);

    // ewise add with sparse vector
    GraphBLAS::Vector<double> v2(v4b_dense, 0.);
    std::vector<double> answer2 = {0, 1, 12, 0};
    GraphBLAS::Vector<double> ans2(answer2, 0.);
    GraphBLAS::eWiseAdd(result,
                        mask,
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v2, GraphBLAS::MERGE);
    BOOST_CHECK_EQUAL(result, ans2);

    // ewise add with empty vector
    GraphBLAS::Vector<double> v3(zero4_dense, 0.);
    std::vector<double> answer3 = {0, 0, 12, 0};
    GraphBLAS::Vector<double> ans3(answer3, 0.);
    GraphBLAS::eWiseAdd(result,
                        mask,
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v3, GraphBLAS::MERGE);
    BOOST_CHECK_EQUAL(result, ans3);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_vector_masked_reg_stored_zero)
{
    std::vector<int> mask4_dense = {0, 1, 1, 0};
    GraphBLAS::Vector<int> mask(mask4_dense); //, 0);

    GraphBLAS::Vector<double> u(v4a_dense, 0.);

    // ewise add with dense vector
    GraphBLAS::Vector<double> v(twos4_dense, 0.);
    std::vector<double> ans_4atwos4_dense2 = {0, 2, 14, 0};
    GraphBLAS::Vector<double> ans(ans_4atwos4_dense2, 0.);

    GraphBLAS::Vector<double> result(4);

    GraphBLAS::eWiseAdd(result,
                        mask,
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v, GraphBLAS::MERGE);
    BOOST_CHECK_EQUAL(result, ans);

    // ewise add with sparse vector
    GraphBLAS::Vector<double> v2(v4b_dense, 0.);
    std::vector<double> answer2 = {0, 1, 12, 0};
    GraphBLAS::Vector<double> ans2(answer2, 0.);
    GraphBLAS::eWiseAdd(result,
                        mask,
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v2, GraphBLAS::MERGE);
    BOOST_CHECK_EQUAL(result, ans2);

    // ewise add with empty vector
    GraphBLAS::Vector<double> v3(zero4_dense, 0.);
    std::vector<double> answer3 = {0, 0, 12, 0};
    GraphBLAS::Vector<double> ans3(answer3, 0.);
    GraphBLAS::eWiseAdd(result,
                        mask,
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v3, GraphBLAS::MERGE);
    BOOST_CHECK_EQUAL(result, ans3);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_semiring_vector_masked_reg_stored_zero)
{
    std::vector<int> mask4_dense = {0, 1, 1, 0};
    GraphBLAS::Vector<int> mask(mask4_dense); //, 0);

    GraphBLAS::Vector<double> u(v4a_dense, 0.);

    // ewise add with dense vector
    GraphBLAS::Vector<double> v(twos4_dense, 0.);
    std::vector<double> ans_4atwos4_dense2 = {0, 2, 14, 0};
    GraphBLAS::Vector<double> ans(ans_4atwos4_dense2, 0.);

    GraphBLAS::Vector<double> result(4);

    GraphBLAS::eWiseAdd(result,
                        mask,
                        GraphBLAS::NoAccumulate(),
                        add_monoid(GraphBLAS::ArithmeticSemiring<double>()),
                        u, v, GraphBLAS::MERGE);
    BOOST_CHECK_EQUAL(result, ans);

    // ewise add with sparse vector
    GraphBLAS::Vector<double> v2(v4b_dense, 0.);
    std::vector<double> answer2 = {0, 1, 12, 0};
    GraphBLAS::Vector<double> ans2(answer2, 0.);
    GraphBLAS::eWiseAdd(result,
                        mask,
                        GraphBLAS::NoAccumulate(),
                        add_monoid(GraphBLAS::ArithmeticSemiring<double>()),
                        u, v2, GraphBLAS::MERGE);
    BOOST_CHECK_EQUAL(result, ans2);

    // ewise add with empty vector
    GraphBLAS::Vector<double> v3(zero4_dense, 0.);
    std::vector<double> answer3 = {0, 0, 12, 0};
    GraphBLAS::Vector<double> ans3(answer3, 0.);
    GraphBLAS::eWiseAdd(result,
                        mask,
                        GraphBLAS::NoAccumulate(),
                        add_monoid(GraphBLAS::ArithmeticSemiring<double>()),
                        u, v3, GraphBLAS::MERGE);
    BOOST_CHECK_EQUAL(result, ans3);
}

//****************************************************************************
// Tests using a complemented mask with REPLACE semantics
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_vector_scmp_masked_replace_bad_dimensions)
{
    GraphBLAS::Vector<double> v(v4a_dense, 0.);
    GraphBLAS::Vector<double> u(v4b_dense, 0.);
    GraphBLAS::Vector<double> mask(v3a_dense, 0.);
    GraphBLAS::Vector<double> result(4);

    // incompatible Mask-Output dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseAdd(result,
                             GraphBLAS::complement(mask),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Plus<double>(), v, u,
                             GraphBLAS::MERGE)),
        GraphBLAS::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_vector_scmp_masked_replace_reg)
{
    std::vector<int> mask4_dense = {1, 0, 0, 1};
    GraphBLAS::Vector<int> mask(mask4_dense, 0);

    GraphBLAS::Vector<double> u(v4a_dense, 0.);

    // ewise add with dense vector
    GraphBLAS::Vector<double> v(twos4_dense, 0.);
    std::vector<double> ans_4atwos4_dense2 = {0, 2, 14, 0};
    GraphBLAS::Vector<double> ans(ans_4atwos4_dense2, 0.);

    GraphBLAS::Vector<double> result(4);

    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::complement(mask),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v, GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result, ans);

    // ewise add with sparse vector
    GraphBLAS::Vector<double> v2(v4b_dense, 0.);
    std::vector<double> answer2 = {0, 1, 12, 0};
    GraphBLAS::Vector<double> ans2(answer2, 0.);
    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::complement(mask),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v2, GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result, ans2);

    // ewise add with empty vector
    GraphBLAS::Vector<double> v3(zero4_dense, 0.);
    std::vector<double> answer3 = {0, 0, 12, 0};
    GraphBLAS::Vector<double> ans3(answer3, 0.);
    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::complement(mask),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v3, GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result, ans3);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_vector_scmp_masked_replace_reg_stored_zero)
{
    std::vector<int> mask4_dense = {1, 0, 0, 1};
    GraphBLAS::Vector<int> mask(mask4_dense); //, 0);

    GraphBLAS::Vector<double> u(v4a_dense, 0.);

    // ewise add with dense vector
    GraphBLAS::Vector<double> v(twos4_dense, 0.);
    std::vector<double> ans_4atwos4_dense2 = {0, 2, 14, 0};
    GraphBLAS::Vector<double> ans(ans_4atwos4_dense2, 0.);

    GraphBLAS::Vector<double> result(4);

    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::complement(mask),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v, GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result, ans);

    // ewise add with sparse vector
    GraphBLAS::Vector<double> v2(v4b_dense, 0.);
    std::vector<double> answer2 = {0, 1, 12, 0};
    GraphBLAS::Vector<double> ans2(answer2, 0.);
    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::complement(mask),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v2, GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result, ans2);

    // ewise add with empty vector
    GraphBLAS::Vector<double> v3(zero4_dense, 0.);
    std::vector<double> answer3 = {0, 0, 12, 0};
    GraphBLAS::Vector<double> ans3(answer3, 0.);
    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::complement(mask),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v3, GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result, ans3);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_semiring_vector_scmp_masked_replace_reg_stored_zero)
{
    std::vector<int> mask4_dense = {1, 0, 0, 1};
    GraphBLAS::Vector<int> mask(mask4_dense); //, 0);

    GraphBLAS::Vector<double> u(v4a_dense, 0.);

    // ewise add with dense vector
    GraphBLAS::Vector<double> v(twos4_dense, 0.);
    std::vector<double> ans_4atwos4_dense2 = {0, 2, 14, 0};
    GraphBLAS::Vector<double> ans(ans_4atwos4_dense2, 0.);

    GraphBLAS::Vector<double> result(4);

    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::complement(mask),
                        GraphBLAS::NoAccumulate(),
                        add_monoid(GraphBLAS::ArithmeticSemiring<double>()),
                        u, v, GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result, ans);

    // ewise add with sparse vector
    GraphBLAS::Vector<double> v2(v4b_dense, 0.);
    std::vector<double> answer2 = {0, 1, 12, 0};
    GraphBLAS::Vector<double> ans2(answer2, 0.);
    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::complement(mask),
                        GraphBLAS::NoAccumulate(),
                        add_monoid(GraphBLAS::ArithmeticSemiring<double>()),
                        u, v2, GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result, ans2);

    // ewise add with empty vector
    GraphBLAS::Vector<double> v3(zero4_dense, 0.);
    std::vector<double> answer3 = {0, 0, 12, 0};
    GraphBLAS::Vector<double> ans3(answer3, 0.);
    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::complement(mask),
                        GraphBLAS::NoAccumulate(),
                        add_monoid(GraphBLAS::ArithmeticSemiring<double>()),
                        u, v3, GraphBLAS::REPLACE);
    BOOST_CHECK_EQUAL(result, ans3);
}

//****************************************************************************
// Tests using a complemented mask (with merge semantics)
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_vector_scmp_masked_bad_dimensions)
{
    GraphBLAS::Vector<double> v(v4a_dense, 0.);
    GraphBLAS::Vector<double> u(v4b_dense, 0.);
    GraphBLAS::Vector<double> mask(v3a_dense, 0.);
    GraphBLAS::Vector<double> result(4);

    // incompatible Mask-Output dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseAdd(result,
                             GraphBLAS::complement(mask),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Plus<double>(), v, u, GraphBLAS::MERGE)),
        GraphBLAS::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_vector_scmp_masked_reg)
{
    std::vector<int> mask4_dense = {1, 0, 0, 1};
    GraphBLAS::Vector<int> mask(mask4_dense, 0);

    GraphBLAS::Vector<double> u(v4a_dense, 0.);

    // ewise add with dense vector
    GraphBLAS::Vector<double> v(twos4_dense, 0.);
    std::vector<double> ans_4atwos4_dense2 = {0, 2, 14, 0};
    GraphBLAS::Vector<double> ans(ans_4atwos4_dense2, 0.);

    GraphBLAS::Vector<double> result(4);

    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::complement(mask),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v, GraphBLAS::MERGE);
    BOOST_CHECK_EQUAL(result, ans);

    // ewise add with sparse vector
    GraphBLAS::Vector<double> v2(v4b_dense, 0.);
    std::vector<double> answer2 = {0, 1, 12, 0};
    GraphBLAS::Vector<double> ans2(answer2, 0.);
    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::complement(mask),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v2, GraphBLAS::MERGE);
    BOOST_CHECK_EQUAL(result, ans2);

    // ewise add with empty vector
    GraphBLAS::Vector<double> v3(zero4_dense, 0.);
    std::vector<double> answer3 = {0, 0, 12, 0};
    GraphBLAS::Vector<double> ans3(answer3, 0.);
    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::complement(mask),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v3, GraphBLAS::MERGE);
    BOOST_CHECK_EQUAL(result, ans3);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_vector_scmp_masked_reg_stored_zero)
{
    std::vector<int> mask4_dense = {1, 0, 0, 1};
    GraphBLAS::Vector<int> mask(mask4_dense); //, 0);

    GraphBLAS::Vector<double> u(v4a_dense, 0.);

    // ewise add with dense vector
    GraphBLAS::Vector<double> v(twos4_dense, 0.);
    std::vector<double> ans_4atwos4_dense2 = {0, 2, 14, 0};
    GraphBLAS::Vector<double> ans(ans_4atwos4_dense2, 0.);

    GraphBLAS::Vector<double> result(4);

    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::complement(mask),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v, GraphBLAS::MERGE);
    BOOST_CHECK_EQUAL(result, ans);

    // ewise add with sparse vector
    GraphBLAS::Vector<double> v2(v4b_dense, 0.);
    std::vector<double> answer2 = {0, 1, 12, 0};
    GraphBLAS::Vector<double> ans2(answer2, 0.);
    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::complement(mask),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v2, GraphBLAS::MERGE);
    BOOST_CHECK_EQUAL(result, ans2);

    // ewise add with empty vector
    GraphBLAS::Vector<double> v3(zero4_dense, 0.);
    std::vector<double> answer3 = {0, 0, 12, 0};
    GraphBLAS::Vector<double> ans3(answer3, 0.);
    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::complement(mask),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v3, GraphBLAS::MERGE);
    BOOST_CHECK_EQUAL(result, ans3);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_semiring_vector_scmp_masked_reg_stored_zero)
{
    std::vector<int> mask4_dense = {1, 0, 0, 1};
    GraphBLAS::Vector<int> mask(mask4_dense); //, 0);

    GraphBLAS::Vector<double> u(v4a_dense, 0.);

    // ewise add with dense vector
    GraphBLAS::Vector<double> v(twos4_dense, 0.);
    std::vector<double> ans_4atwos4_dense2 = {0, 2, 14, 0};
    GraphBLAS::Vector<double> ans(ans_4atwos4_dense2, 0.);

    GraphBLAS::Vector<double> result(4);

    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::complement(mask),
                        GraphBLAS::NoAccumulate(),
                        add_monoid(GraphBLAS::ArithmeticSemiring<double>()),
                        u, v, GraphBLAS::MERGE);
    BOOST_CHECK_EQUAL(result, ans);

    // ewise add with sparse vector
    GraphBLAS::Vector<double> v2(v4b_dense, 0.);
    std::vector<double> answer2 = {0, 1, 12, 0};
    GraphBLAS::Vector<double> ans2(answer2, 0.);
    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::complement(mask),
                        GraphBLAS::NoAccumulate(),
                        add_monoid(GraphBLAS::ArithmeticSemiring<double>()),
                        u, v2, GraphBLAS::MERGE);
    BOOST_CHECK_EQUAL(result, ans2);

    // ewise add with empty vector
    GraphBLAS::Vector<double> v3(zero4_dense, 0.);
    std::vector<double> answer3 = {0, 0, 12, 0};
    GraphBLAS::Vector<double> ans3(answer3, 0.);
    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::complement(mask),
                        GraphBLAS::NoAccumulate(),
                        add_monoid(GraphBLAS::ArithmeticSemiring<double>()),
                        u, v3, GraphBLAS::MERGE);
    BOOST_CHECK_EQUAL(result, ans3);
}

//****************************************************************************
// Tests using an Accumulator (incomplete).
//****************************************************************************

//***************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_vector_accum)
{
    GraphBLAS::Vector<double> v(v3a_dense, 0.);
    GraphBLAS::Vector<double> u(twos3_dense, 0.);

    GraphBLAS::Vector<double> result1({1, 1, 1}, 0.);
    GraphBLAS::Vector<double> ans1({15, 3, 10}, 0.);
    GraphBLAS::eWiseAdd(result1,
                        GraphBLAS::NoMask(),
                        GraphBLAS::Plus<double>(),
                        GraphBLAS::Plus<double>(), u, v);
    BOOST_CHECK_EQUAL(result1, ans1);

    GraphBLAS::Vector<double> result2({1, 1, 1}, 0.);
    GraphBLAS::Vector<double> ans2({-13, -1, -8}, 0.);
    GraphBLAS::eWiseAdd(result2,
                        GraphBLAS::NoMask(),
                        GraphBLAS::Minus<double>(),
                        GraphBLAS::Plus<double>(), u, v);
    BOOST_CHECK_EQUAL(result2, ans2);
}

BOOST_AUTO_TEST_SUITE_END()
