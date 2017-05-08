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

#if 0
//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_vector_stored_zero_result)
{
    GraphBLAS::Vector<double> A(m3x3_dense, 0.);

    // Add a stored zero on the diagonal
    A.setElement(1, 1, 0);
    BOOST_CHECK_EQUAL(A.nvals(), 7);

    GraphBLAS::Vector<double> B2(eye3x3_dense, 0.);
    GraphBLAS::Vector<double> Ans2(
        ans_eye3x3_dense, 0.);
    // Add a stored zero on the diagonal
    Ans2.setElement(1, 1, 0);

    GraphBLAS::Vector<double> Result(3,3);

    GraphBLAS::eWiseAdd(Result,
                        GraphBLAS::NoMask(),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::Plus<double>(), A, B2);
    BOOST_CHECK_EQUAL(Result, Ans2);
    BOOST_CHECK_EQUAL(Result.nvals(), 7);
}

//****************************************************************************
// Tests using a mask with REPLACE
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_vector_masked_replace_bad_dimensions)
{
    GraphBLAS::Vector<double> mA(twos4x3_dense, 0.);
    GraphBLAS::Vector<double> mB(m4x3_dense, 0.);
    GraphBLAS::Vector<double> Mask(twos3x3_dense, 0.);
    GraphBLAS::Vector<double> Result(4, 3);

    // incompatible Mask-Output dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseAdd(Result,
                             Mask,
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Plus<double>(), mA, mB,
                             true)),
        GraphBLAS::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_vector_masked_replace_reg)
{
    GraphBLAS::Vector<double> mA(twos4x3_dense, 0.);
    GraphBLAS::Vector<double> mB(m4x3_dense, 0.);

    std::vector<double> mask_dense = {{1, 1, 1},
                                                    {0, 1, 1},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    GraphBLAS::Vector<double> Mask(mask_dense, 0.);

    {
        GraphBLAS::Vector<double> Result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{7, 8, 6},
                                                       {0, 9, 7},
                                                       {0, 0, 2},
                                                       {0, 0, 0}};
        GraphBLAS::Vector<double> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            Mask,
                            GraphBLAS::NoAccumulate(),
                            GraphBLAS::Plus<double>(), mA, mB,
                            true);

        BOOST_CHECK_EQUAL(Result.nvals(), 6);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

    {
        GraphBLAS::Vector<double> Result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{10, 12,  8},
                                                       {0,  14, 10},
                                                       {0,   0,  2},
                                                       {0,   0,  0}};
        GraphBLAS::Vector<double> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            Mask,
                            GraphBLAS::Second<double>(),
                            GraphBLAS::Plus<double>(), mA, mB,
                            true);

        BOOST_CHECK_EQUAL(Result.nvals(), 6);
        BOOST_CHECK_EQUAL(Result, Ans);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_vector_masked_replace_reg_stored_zero)
{
    // tests a computed and stored zero
    // tests a stored zero in the mask
    GraphBLAS::Vector<double> mA(twos4x3_dense, 0.);
    GraphBLAS::Vector<double> mB(m4x3_dense, 0.);

    std::vector<double> mask_dense = {{1, 1, 1},
                                                    {0, 1, 1},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    GraphBLAS::Vector<double> Mask(mask_dense); //, 0.);
    BOOST_CHECK_EQUAL(Mask.nvals(), 12);

    {
        GraphBLAS::Vector<double> Result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{7, 8, 6},
                                                       {0, 9, 7},
                                                       {0, 0, 2},
                                                       {0, 0, 0}};
        GraphBLAS::Vector<double> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            Mask,
                            GraphBLAS::NoAccumulate(),
                            GraphBLAS::Plus<double>(), mA, mB,
                            true);

        BOOST_CHECK_EQUAL(Result.nvals(), 6);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

    {
        GraphBLAS::Vector<double> Result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{10, 12,  8},
                                                       {0,  14, 10},
                                                       {0,   0,  2},
                                                       {0,   0,  0}};
        GraphBLAS::Vector<double> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            Mask,
                            GraphBLAS::Second<double>(),
                            GraphBLAS::Plus<double>(), mA, mB,
                            true);

        BOOST_CHECK_EQUAL(Result.nvals(), 6);
        BOOST_CHECK_EQUAL(Result, Ans);
    }
}

//****************************************************************************
// Tests using a mask with MERGE semantics
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_vector_masked_bad_dimensions)
{
    GraphBLAS::Vector<double> mA(twos4x3_dense, 0.);
    GraphBLAS::Vector<double> mB(m4x3_dense, 0.);
    GraphBLAS::Vector<double> Mask(twos3x3_dense, 0.);
    GraphBLAS::Vector<double> Result(4, 3);

    // incompatible Mask-Output dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseAdd(Result,
                             Mask,
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Plus<double>(), mA, mB)),
        GraphBLAS::DimensionException)
        }

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_vector_masked_reg)
{
    GraphBLAS::Vector<double> mA(twos4x3_dense, 0.);
    GraphBLAS::Vector<double> mB(m4x3_dense, 0.);

    std::vector<double> mask_dense = {{1, 1, 1},
                                                    {0, 1, 1},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    GraphBLAS::Vector<double> Mask(mask_dense, 0.);

    {
        GraphBLAS::Vector<double> Result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{7,  8,  6},
                                                       {2,  9,  7},
                                                       {2,  2,  2},
                                                       {2,  2,  2}};
        GraphBLAS::Vector<double> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            Mask,
                            GraphBLAS::NoAccumulate(),
                            GraphBLAS::Plus<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 12);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

    {
        GraphBLAS::Vector<double> Result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{7,  8,  6},
                                                       {2,  9,  7},
                                                       {2,  2,  2},
                                                       {2,  2,  2}};
        GraphBLAS::Vector<double> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            Mask,
                            GraphBLAS::Second<double>(),
                            GraphBLAS::Plus<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 12);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_vector_masked_reg_stored_zero)
{
    GraphBLAS::Vector<double> mA(twos4x3_dense, 0.);
    GraphBLAS::Vector<double> mB(m4x3_dense, 0.);

    std::vector<double> mask_dense = {{1, 1, 1},
                                                    {0, 1, 1},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    GraphBLAS::Vector<double> Mask(mask_dense); //, 0.);

    {
        GraphBLAS::Vector<double> Result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{7,  8,  6},
                                                       {2,  9,  7},
                                                       {2,  2,  2},
                                                       {2,  2,  2}};
        GraphBLAS::Vector<double> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            Mask,
                            GraphBLAS::NoAccumulate(),
                            GraphBLAS::Plus<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 12);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

    {
        GraphBLAS::Vector<double> Result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{7,  8,  6},
                                                       {2,  9,  7},
                                                       {2,  2,  2},
                                                       {2,  2,  2}};
        GraphBLAS::Vector<double> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            Mask,
                            GraphBLAS::Second<double>(),
                            GraphBLAS::Plus<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 12);
        BOOST_CHECK_EQUAL(Result, Ans);
    }
}

//****************************************************************************
// Tests using a complemented mask with REPLACE semantics
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_vector_scmp_masked_replace_bad_dimensions)
{
    GraphBLAS::Vector<double> mA(twos4x3_dense, 0.);
    GraphBLAS::Vector<double> mB(m4x3_dense, 0.);
    GraphBLAS::Vector<double> Mask(twos3x3_dense, 0.);
    GraphBLAS::Vector<double> Result(4, 3);

    // incompatible Mask-Output dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseAdd(Result,
                             GraphBLAS::complement(Mask),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Plus<double>(), mA, mB,
                             true)),
        GraphBLAS::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_vector_scmp_masked_replace_reg)
{
    GraphBLAS::Vector<double> mA(twos4x3_dense, 0.);
    GraphBLAS::Vector<double> mB(m4x3_dense, 0.);

    std::vector<double> mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    GraphBLAS::Vector<double> Mask(mask_dense, 0.);

    {
        GraphBLAS::Vector<double> Result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{7, 8, 6},
                                                       {0, 9, 7},
                                                       {0, 0, 2},
                                                       {0, 0, 0}};
        GraphBLAS::Vector<double> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            GraphBLAS::complement(Mask),
                            GraphBLAS::NoAccumulate(),
                            GraphBLAS::Plus<double>(), mA, mB,
                            true);

        BOOST_CHECK_EQUAL(Result.nvals(), 6);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

    {
        GraphBLAS::Vector<double> Result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{7, 8, 6},
                                                       {0, 9, 7},
                                                       {0, 0, 2},
                                                       {0, 0, 0}};
        GraphBLAS::Vector<double> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            GraphBLAS::complement(Mask),
                            GraphBLAS::Second<double>(),
                            GraphBLAS::Plus<double>(), mA, mB,
                            true);

        BOOST_CHECK_EQUAL(Result.nvals(), 6);
        BOOST_CHECK_EQUAL(Result, Ans);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_vector_scmp_masked_replace_reg_stored_zero)
{
    // tests a computed and stored zero
    // tests a stored zero in the mask
    GraphBLAS::Vector<double> mA(twos4x3_dense, 0.);
    GraphBLAS::Vector<double> mB(m4x3_dense, 0.);

    std::vector<double> mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    GraphBLAS::Vector<double> Mask(mask_dense); //, 0.);
    BOOST_CHECK_EQUAL(Mask.nvals(), 12);

    {
        GraphBLAS::Vector<double> Result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{7, 8, 6},
                                                       {0, 9, 7},
                                                       {0, 0, 2},
                                                       {0, 0, 0}};
        GraphBLAS::Vector<double> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            GraphBLAS::complement(Mask),
                            GraphBLAS::NoAccumulate(),
                            GraphBLAS::Plus<double>(), mA, mB,
                            true);

        BOOST_CHECK_EQUAL(Result.nvals(), 6);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

    {
        GraphBLAS::Vector<double> Result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{7, 8, 6},
                                                       {0, 9, 7},
                                                       {0, 0, 2},
                                                       {0, 0, 0}};
        GraphBLAS::Vector<double> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            GraphBLAS::complement(Mask),
                            GraphBLAS::Second<double>(),
                            GraphBLAS::Plus<double>(), mA, mB,
                            true);

        BOOST_CHECK_EQUAL(Result.nvals(), 6);
        BOOST_CHECK_EQUAL(Result, Ans);
    }
}

//****************************************************************************
// Tests using a complemented mask (with merge semantics)
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_vector_scmp_masked_bad_dimensions)
{
    GraphBLAS::Vector<double> mA(twos4x3_dense, 0.);
    GraphBLAS::Vector<double> mB(m4x3_dense, 0.);
    GraphBLAS::Vector<double> Mask(twos3x3_dense, 0.);
    GraphBLAS::Vector<double> Result(4, 3);

    // incompatible Mask-Output dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::eWiseAdd(Result,
                             GraphBLAS::complement(Mask),
                             GraphBLAS::NoAccumulate(),
                             GraphBLAS::Plus<double>(), mA, mB)),
        GraphBLAS::DimensionException)
        }

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_vector_scmp_masked_reg)
{
    GraphBLAS::Vector<double> mA(twos4x3_dense, 0.);
    GraphBLAS::Vector<double> mB(m4x3_dense, 0.);

    std::vector<double> mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    GraphBLAS::Vector<double> Mask(mask_dense, 0.);

    {
        GraphBLAS::Vector<double> Result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{7,  8,  6},
                                                       {2,  9,  7},
                                                       {2,  2,  2},
                                                       {2,  2,  2}};
        GraphBLAS::Vector<double> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            GraphBLAS::complement(Mask),
                            GraphBLAS::NoAccumulate(),
                            GraphBLAS::Plus<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 12);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

    {
        GraphBLAS::Vector<double> Result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{7,  8,  6},
                                                       {2,  9,  7},
                                                       {2,  2,  2},
                                                       {2,  2,  2}};
        GraphBLAS::Vector<double> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            GraphBLAS::complement(Mask),
                            GraphBLAS::Second<double>(),
                            GraphBLAS::Plus<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 12);
        BOOST_CHECK_EQUAL(Result, Ans);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_ewiseadd_vector_scmp_masked_reg_stored_zero)
{
    GraphBLAS::Vector<double> mA(twos4x3_dense, 0.);
    GraphBLAS::Vector<double> mB(m4x3_dense, 0.);

    std::vector<double> mask_dense = {{0, 0, 0},
                                                    {1, 0, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 1}};
    GraphBLAS::Vector<double> Mask(mask_dense);//, 0.);

    {
        GraphBLAS::Vector<double> Result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{7,  8,  6},
                                                       {2,  9,  7},
                                                       {2,  2,  2},
                                                       {2,  2,  2}};
        GraphBLAS::Vector<double> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            GraphBLAS::complement(Mask),
                            GraphBLAS::NoAccumulate(),
                            GraphBLAS::Plus<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 12);
        BOOST_CHECK_EQUAL(Result, Ans);
    }

    {
        GraphBLAS::Vector<double> Result(twos4x3_dense, 0.);

        std::vector<double> ans_dense = {{7,  8,  6},
                                                       {2,  9,  7},
                                                       {2,  2,  2},
                                                       {2,  2,  2}};
        GraphBLAS::Vector<double> Ans(ans_dense, 0.);

        GraphBLAS::eWiseAdd(Result,
                            GraphBLAS::complement(Mask),
                            GraphBLAS::Second<double>(),
                            GraphBLAS::Plus<double>(), mA, mB);

        BOOST_CHECK_EQUAL(Result.nvals(), 12);
        BOOST_CHECK_EQUAL(Result, Ans);
    }
}
#endif
BOOST_AUTO_TEST_SUITE_END()
