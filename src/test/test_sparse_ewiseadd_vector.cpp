/*
 * Copyright (c) 2017 Carnegie Mellon University and The Trustees of
 * Indiana University.
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

#include <graphblas/graphblas.hpp>

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE sparse_ewiseadd_vector_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(sparse_ewiseadd_vector_suite)

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
                             true)),
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
                        GraphBLAS::Plus<double>(), u, v, true);
    BOOST_CHECK_EQUAL(result, ans);

    // ewise add with sparse vector
    GraphBLAS::Vector<double> v2(v4b_dense, 0.);
    std::vector<double> answer2 = {0, 1, 12, 0};
    GraphBLAS::Vector<double> ans2(answer2, 0.);
    GraphBLAS::eWiseAdd(result,
                        mask,
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v2, true);
    BOOST_CHECK_EQUAL(result, ans2);

    // ewise add with empty vector
    GraphBLAS::Vector<double> v3(zero4_dense, 0.);
    std::vector<double> answer3 = {0, 0, 12, 0};
    GraphBLAS::Vector<double> ans3(answer3, 0.);
    GraphBLAS::eWiseAdd(result,
                        mask,
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v3, true);
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
                        GraphBLAS::Plus<double>(), u, v, true);
    BOOST_CHECK_EQUAL(result, ans);

    // ewise add with sparse vector
    GraphBLAS::Vector<double> v2(v4b_dense, 0.);
    std::vector<double> answer2 = {0, 1, 12, 0};
    GraphBLAS::Vector<double> ans2(answer2, 0.);
    GraphBLAS::eWiseAdd(result,
                        mask,
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v2, true);
    BOOST_CHECK_EQUAL(result, ans2);

    // ewise add with empty vector
    GraphBLAS::Vector<double> v3(zero4_dense, 0.);
    std::vector<double> answer3 = {0, 0, 12, 0};
    GraphBLAS::Vector<double> ans3(answer3, 0.);
    GraphBLAS::eWiseAdd(result,
                        mask,
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v3, true);
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
                             GraphBLAS::Plus<double>(), v, u, false)),
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
                        GraphBLAS::Plus<double>(), u, v, false);
    BOOST_CHECK_EQUAL(result, ans);

    // ewise add with sparse vector
    GraphBLAS::Vector<double> v2(v4b_dense, 0.);
    std::vector<double> answer2 = {0, 1, 12, 0};
    GraphBLAS::Vector<double> ans2(answer2, 0.);
    GraphBLAS::eWiseAdd(result,
                        mask,
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v2, false);
    BOOST_CHECK_EQUAL(result, ans2);

    // ewise add with empty vector
    GraphBLAS::Vector<double> v3(zero4_dense, 0.);
    std::vector<double> answer3 = {0, 0, 12, 0};
    GraphBLAS::Vector<double> ans3(answer3, 0.);
    GraphBLAS::eWiseAdd(result,
                        mask,
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v3, false);
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
                        GraphBLAS::Plus<double>(), u, v, false);
    BOOST_CHECK_EQUAL(result, ans);

    // ewise add with sparse vector
    GraphBLAS::Vector<double> v2(v4b_dense, 0.);
    std::vector<double> answer2 = {0, 1, 12, 0};
    GraphBLAS::Vector<double> ans2(answer2, 0.);
    GraphBLAS::eWiseAdd(result,
                        mask,
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v2, false);
    BOOST_CHECK_EQUAL(result, ans2);

    // ewise add with empty vector
    GraphBLAS::Vector<double> v3(zero4_dense, 0.);
    std::vector<double> answer3 = {0, 0, 12, 0};
    GraphBLAS::Vector<double> ans3(answer3, 0.);
    GraphBLAS::eWiseAdd(result,
                        mask,
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v3, false);
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
                             false)),
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
                        GraphBLAS::Plus<double>(), u, v, true);
    BOOST_CHECK_EQUAL(result, ans);

    // ewise add with sparse vector
    GraphBLAS::Vector<double> v2(v4b_dense, 0.);
    std::vector<double> answer2 = {0, 1, 12, 0};
    GraphBLAS::Vector<double> ans2(answer2, 0.);
    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::complement(mask),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v2, true);
    BOOST_CHECK_EQUAL(result, ans2);

    // ewise add with empty vector
    GraphBLAS::Vector<double> v3(zero4_dense, 0.);
    std::vector<double> answer3 = {0, 0, 12, 0};
    GraphBLAS::Vector<double> ans3(answer3, 0.);
    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::complement(mask),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v3, true);
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
                        GraphBLAS::Plus<double>(), u, v, true);
    BOOST_CHECK_EQUAL(result, ans);

    // ewise add with sparse vector
    GraphBLAS::Vector<double> v2(v4b_dense, 0.);
    std::vector<double> answer2 = {0, 1, 12, 0};
    GraphBLAS::Vector<double> ans2(answer2, 0.);
    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::complement(mask),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v2, true);
    BOOST_CHECK_EQUAL(result, ans2);

    // ewise add with empty vector
    GraphBLAS::Vector<double> v3(zero4_dense, 0.);
    std::vector<double> answer3 = {0, 0, 12, 0};
    GraphBLAS::Vector<double> ans3(answer3, 0.);
    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::complement(mask),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v3, true);
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
                             GraphBLAS::Plus<double>(), v, u, false)),
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
                        GraphBLAS::Plus<double>(), u, v, false);
    BOOST_CHECK_EQUAL(result, ans);

    // ewise add with sparse vector
    GraphBLAS::Vector<double> v2(v4b_dense, 0.);
    std::vector<double> answer2 = {0, 1, 12, 0};
    GraphBLAS::Vector<double> ans2(answer2, 0.);
    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::complement(mask),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v2, false);
    BOOST_CHECK_EQUAL(result, ans2);

    // ewise add with empty vector
    GraphBLAS::Vector<double> v3(zero4_dense, 0.);
    std::vector<double> answer3 = {0, 0, 12, 0};
    GraphBLAS::Vector<double> ans3(answer3, 0.);
    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::complement(mask),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v3, false);
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
                        GraphBLAS::Plus<double>(), u, v, false);
    BOOST_CHECK_EQUAL(result, ans);

    // ewise add with sparse vector
    GraphBLAS::Vector<double> v2(v4b_dense, 0.);
    std::vector<double> answer2 = {0, 1, 12, 0};
    GraphBLAS::Vector<double> ans2(answer2, 0.);
    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::complement(mask),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v2, false);
    BOOST_CHECK_EQUAL(result, ans2);

    // ewise add with empty vector
    GraphBLAS::Vector<double> v3(zero4_dense, 0.);
    std::vector<double> answer3 = {0, 0, 12, 0};
    GraphBLAS::Vector<double> ans3(answer3, 0.);
    GraphBLAS::eWiseAdd(result,
                        GraphBLAS::complement(mask),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), u, v3, false);
    BOOST_CHECK_EQUAL(result, ans3);
}

BOOST_AUTO_TEST_SUITE_END()
