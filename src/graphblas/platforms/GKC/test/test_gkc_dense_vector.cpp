/*
 * GraphBLAS Template Library, Version 2.1
 *
 * Copyright 2019 Carnegie Mellon University, Battelle Memorial Institute, and
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

#include <iostream>

#include <graphblas/graphblas.hpp>

using namespace grb;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE gkc_dense_vector_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

//****************************************************************************
// GKC constructor from dense matrix
/*
BOOST_AUTO_TEST_CASE(gkc_test_tags)
{
    IndexType M = 3;
    grb::Vector<double> v1(M);
    // std::cout << "\nv1\n";
    // v1.printInfo(std::cout);

    grb::Vector<double, grb::OrigTag> v2(M);
    // std::cout << "\nv2\n";
    // v2.printInfo(std::cout);

    grb::Vector<double, grb::DenseTag> v2a(M);
    // std::cout << "\nv2a\n";
    // v2a.printInfo(std::cout);

    grb::Vector<double, grb::SparseTag> v2c(M);
    // std::cout << "\nv2c\n";
    // v2c.printInfo(std::cout);

    grb::Vector<double, grb::GKCTag> v3(M);
    // std::cout << "\nv3\n";
    // v3.printInfo(std::cout);

    grb::Vector<double, grb::SparseTag, grb::GKCTag> v3c(M);
    // std::cout << "\nv3c\n";
    // v3c.printInfo(std::cout);

    grb::Vector<double, grb::GKCTag, grb::SparseTag> v3d(M);
    // std::cout << "\nv3d\n";
    // v3d.printInfo(std::cout);

    grb::Vector<double, grb::OrigTag, grb::GKCTag> v4(M);
    // std::cout << "\nv4\n";
    // v4.printInfo(std::cout);

    grb::Vector<double, grb::GKCTag, grb::OrigTag> v4a(M);
    // std::cout << "\nv4a\n";
    // v4a.printInfo(std::cout);
}
*/

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_construction_basic)
{
    grb::IndexType M = 7;
    grb::backend::GKCDenseVector<double> v1(M);

    BOOST_CHECK_EQUAL(v1.size(), M);
    BOOST_CHECK_EQUAL(v1.nvals(), 0);
    BOOST_CHECK_THROW(v1.extractElement(0), NoValueException);
    BOOST_CHECK_THROW(v1.extractElement(M-1), NoValueException);
    BOOST_CHECK_THROW(v1.extractElement(M), IndexOutOfBoundsException);

    BOOST_CHECK_THROW(grb::backend::GKCDenseVector<double>(0),
                      InvalidValueException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_construction_from_dense)
{
    std::vector<double> vec = {6, 0, 0, 4, 7, 0, 9, 4};

    grb::backend::GKCDenseVector<double> v1(vec);

    BOOST_CHECK_EQUAL(v1.size(), vec.size());
    BOOST_CHECK_EQUAL(v1.nvals(), vec.size());
    for (grb::IndexType i = 0; i < vec.size(); ++i)
    {
        BOOST_CHECK_EQUAL(v1.extractElement(i), vec[i]);
    }
    BOOST_CHECK_THROW(v1.extractElement(v1.size()), IndexOutOfBoundsException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_sparse_construction_from_dense)
{
    double zero(0);
    std::vector<double> vec = {6, zero, zero, 4, 7, zero, 9, 4};

    grb::backend::GKCDenseVector<double> v1(vec, zero);

    BOOST_CHECK_EQUAL(v1.size(), vec.size());
    BOOST_CHECK_EQUAL(v1.nvals(), 5);
    for (grb::IndexType i = 0; i < vec.size(); ++i)
    {
        if (vec[i] == zero)
        {
            BOOST_CHECK_THROW(v1.extractElement(i), NoValueException);
        }
        else
        {
            BOOST_CHECK_EQUAL(v1.extractElement(i), vec[i]);
        }
    }
    BOOST_CHECK_THROW(v1.extractElement(v1.size()), IndexOutOfBoundsException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_copy_construction_equality_operator)
{
    double zero(0);
    std::vector<double> vec = {6, zero, zero, 4, 7, zero, 9, 4};

    grb::backend::GKCDenseVector<double> v1(vec, zero);

    grb::backend::GKCDenseVector<double> v2(v1);

    BOOST_CHECK_EQUAL(v1, v2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_access_novalue_in_non_empty_vector)
{
    std::vector<double> vec = {6, 0, 0, 4, 7, 0, 9, 4};

    grb::backend::GKCDenseVector<double> v1(vec, 0);

    BOOST_CHECK_EQUAL(v1.nvals(), 5);
    BOOST_CHECK_THROW(v1.extractElement(1), NoValueException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_assign_to_implied_zero_and_stored_value)
{

    double zero(0);
    std::vector<double> vec = {6, zero, zero, 4, 7, zero, 9, 4};

    grb::backend::GKCDenseVector<double> v1(vec, zero);

    v1.setElement(1, 2.);

    grb::backend::GKCDenseVector<double> v2(vec, zero);
    BOOST_CHECK(v1 != v2);

    v2.setElement(1, 3.);
    // v2.printInfo(std::cerr);
    // v1.printInfo(std::cerr);
    BOOST_CHECK(v1 != v2);

    v2.setElement(1, 2.);
    BOOST_CHECK_EQUAL(v1, v2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_index_array_constrution)
{
    double zero(0);
    std::vector<double> vec = {6, zero, zero, 4, 7, zero, 9, 4};

    grb::backend::GKCDenseVector<double> v1(vec, zero);

    std::vector<IndexType> indices = {0, 3, 4, 6, 7};
    std::vector<double>    values  = {6 ,4, 7, 9, 4};

    grb::backend::GKCDenseVector<double> v2(indices.begin(),
    values.begin(), indices.size());

    BOOST_CHECK_EQUAL(v1, v2);

    //grb::backend::GKCDenseVector<double> v3(indices.begin(), values.begin(), 9);
    //BOOST_CHECK(v1 != v3);

    //BOOST_CHECK_THROW(
    //    grb::backend::GKCDenseVector<double>(indices.begin(), values.begin(), 3),
    //    DimensionException);
    //BOOST_CHECK_EQUAL(v1, v2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_assignment)
{
    std::vector<IndexType> indices = {0, 3, 4, 6, 7};
    std::vector<double>    values  = {6 ,4, 7, 9, 4};

    grb::backend::GKCDenseVector<double> v1(indices.begin(), values.begin(), indices.size());
    grb::backend::GKCDenseVector<double> v2(8);

    v2 = v1;
    BOOST_CHECK_EQUAL(v1, v2);
}

// THIS TEST IS NO LONGER VALID AS LONG AS EQUALITY CHECK PERFORMS A SORT INTERNALLY.
// BOOST_AUTO_TEST_CASE(test_sorting)
// {
//     std::vector<IndexType> indices  = {0, 3, 4, 6, 7};
//     std::vector<double>    values   = {6, 4, 7, 9, 4};
//     std::vector<IndexType> indices2 = {3, 4, 0, 6, 7};
//     std::vector<double>    values2  = {4, 7, 6, 9, 4};
    
//     grb::backend::GKCDenseVector<double> v1(indices.begin(), values.begin(), indices.size());
//     grb::backend::GKCDenseVector<double> v2(indices2.begin(), values2.begin(), indices2.size());

//     // Current semantic meaning of neq is that vector with different order but same elements is not equal.
//     //BOOST_CHECK(v1 != v2);


//     BOOST_CHECK_EQUAL(v1, v2); 
// }

//****************************************************************************
// MxV Tests
//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxv_sparse_nomask_noaccum)
{
    std::vector<std::vector<double>> mat = {{0, 0, 0, 0},
                                            {0, 0, 0, 4},
                                            {0, 0, 9, 0},
                                            {0, 0, 5, 3},
                                            {0, 2, 0, 0},
                                            {0, 3, 0, 1},
                                            {0, 3, 3, 0},
                                            {0, 1, 4, 2},
                                            {6, 0, 0, 0},
                                            {6, 0, 0, 2},
                                            {6, 0, 4, 0},
                                            {6, 0, 4, 2},
                                            {6, 1, 0, 0},
                                            {6, 1, 0, 2},
                                            {6, 1, 4, 0},
                                            {6, 1, 4, 2}};
    grb::backend::GKCMatrix<double> m1(mat, 0);

    std::vector<double> vec = {6, 0, 0, 4};
    grb::backend::GKCDenseVector<double> v1(vec, 0);

    grb::backend::GKCDenseVector<double> w(16);

    grb::backend::mxv(w,
                      grb::NoMask(),
                      grb::NoAccumulate(),
                      grb::ArithmeticSemiring<double>(),
                      m1, v1, REPLACE);

    std::vector<double> answer = { 0, 16,  0, 12,  0,  4,  0,  8,
                                  36, 44, 36, 44, 36, 44, 36, 44};
    grb::backend::GKCDenseVector<double> ans(answer, 0);
    BOOST_CHECK_EQUAL(ans, w);
    //w.printInfo(std::cerr);
    //ans.printInfo(std::cerr);
}

BOOST_AUTO_TEST_CASE(test_mxv_sparse_nomask_noaccum_transpose)
{
    using TEST_TYPE = int;
    std::vector<std::vector<TEST_TYPE>> mat = {{0, 0, 0, 1},
                                            {0, 0, 0, 4},
                                            {0, 0, 9, 0},
                                            {0, 3, 0, 1},
                                            {0, 3, 3, 1},
                                            {0, 1, 4, 2}};
    grb::backend::GKCMatrix<TEST_TYPE> m1(mat, 0);

    std::vector<TEST_TYPE> vec = {6, 0, 0, 4, -12, 0};
    grb::backend::GKCDenseVector<TEST_TYPE> v1(vec, 0);

    grb::backend::GKCDenseVector<TEST_TYPE> w(4);

    grb::backend::mxv(w,
                      grb::NoMask(),
                      grb::NoAccumulate(),
                      grb::ArithmeticSemiring<TEST_TYPE>(),
                      grb::TransposeView(m1), v1, REPLACE);

    std::vector<TEST_TYPE> answer = { 0, -24, -36, -2};
    grb::backend::GKCDenseVector<TEST_TYPE> ans(answer, 0);
    BOOST_CHECK_EQUAL(ans, w);
    //w.printInfo(std::cerr);
    //ans.printInfo(std::cerr);
}

BOOST_AUTO_TEST_CASE(test_vxm_sparse_nomask_noaccum_transpose)
{
    std::vector<std::vector<double>> mat = {{0, 0, 0, 0},
                                            {0, 0, 0, 4},
                                            {0, 0, 9, 0},
                                            {0, 0, 5, 3},
                                            {0, 2, 0, 0},
                                            {0, 3, 0, 1},
                                            {0, 3, 3, 0},
                                            {0, 1, 4, 2},
                                            {6, 0, 0, 0},
                                            {6, 0, 0, 2},
                                            {6, 0, 4, 0},
                                            {6, 0, 4, 2},
                                            {6, 1, 0, 0},
                                            {6, 1, 0, 2},
                                            {6, 1, 4, 0},
                                            {6, 1, 4, 2}};
    grb::backend::GKCMatrix<double> m1(mat, 0);

    std::vector<double> vec = {6, 0, 0, 4};
    grb::backend::GKCDenseVector<double> v1(vec, 0);

    grb::backend::GKCDenseVector<double> w(16);

    grb::backend::vxm(w,
                      grb::NoMask(),
                      grb::NoAccumulate(),
                      grb::ArithmeticSemiring<double>(),
                      v1, grb::TransposeView(m1), REPLACE);

    std::vector<double> answer = { 0, 16,  0, 12,  0,  4,  0,  8,
                                  36, 44, 36, 44, 36, 44, 36, 44};
    grb::backend::GKCDenseVector<double> ans(answer, 0);
    BOOST_CHECK_EQUAL(ans, w);
    //w.printInfo(std::cerr);
    //ans.printInfo(std::cerr);
}

BOOST_AUTO_TEST_CASE(test_vxm_sparse_nomask_noaccum)
{
    using TEST_TYPE = int;
    std::vector<std::vector<TEST_TYPE>> mat = {{0, 0, 0, 1},
                                            {0, 0, 0, 4},
                                            {0, 0, 9, 0},
                                            {0, 3, 0, 1},
                                            {0, 3, 3, 1},
                                            {0, 1, 4, 2}};
    grb::backend::GKCMatrix<TEST_TYPE> m1(mat, 0);

    std::vector<TEST_TYPE> vec = {6, 0, 0, 4, -12, 0};
    grb::backend::GKCDenseVector<TEST_TYPE> v1(vec, 0);

    std::vector<TEST_TYPE> w_vec = {1, 1, 1, 1};
    grb::backend::GKCDenseVector<TEST_TYPE> w(w_vec, 0);

    grb::backend::vxm(w,
                      grb::NoMask(),
                      grb::NoAccumulate(),
                      grb::ArithmeticSemiring<TEST_TYPE>(),
                      v1, m1, REPLACE);

    std::vector<TEST_TYPE> answer = { 0, -24, -36, -2};
    grb::backend::GKCDenseVector<TEST_TYPE> ans(answer, 0);
    BOOST_CHECK_EQUAL(ans, w);
    //w.printInfo(std::cerr);
    //ans.printInfo(std::cerr);
}

BOOST_AUTO_TEST_CASE(test_vxm_sparse_nomask_accum_transpose)
{
    std::vector<std::vector<double>> mat = {{0, 0, 0, 0},
                                            {0, 0, 0, 4},
                                            {0, 0, 9, 0},
                                            {0, 0, 5, 3},
                                            {0, 2, 0, 0},
                                            {0, 3, 0, 1},
                                            {0, 3, 3, 0},
                                            {0, 1, 4, 2},
                                            {6, 0, 0, 0},
                                            {6, 0, 0, 2},
                                            {6, 0, 4, 0},
                                            {6, 0, 4, 2},
                                            {6, 1, 0, 0},
                                            {6, 1, 0, 2},
                                            {6, 1, 4, 0},
                                            {6, 1, 4, 2}};
    grb::backend::GKCMatrix<double> m1(mat, 0);

    std::vector<double> vec = {6, 0, 0, 4};
    grb::backend::GKCDenseVector<double> v1(vec, 0);

    std::vector<double> w_vec = { 1, 1, 1, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0};
    grb::backend::GKCDenseVector<double> w(w_vec, 0);

    grb::backend::vxm(w,
                      grb::NoMask(),
                      grb::Plus<double>(),
                      grb::ArithmeticSemiring<double>(),
                      v1, grb::TransposeView(m1), REPLACE);

    std::vector<double> answer = { 1, 17,  1, 12,  0,  4,  0,  8,
                                  36, 44, 36, 44, 36, 44, 36, 44};
    grb::backend::GKCDenseVector<double> ans(answer, 0);
    BOOST_CHECK_EQUAL(ans, w);
    //w.printInfo(std::cerr);
    //ans.printInfo(std::cerr);
}

BOOST_AUTO_TEST_CASE(test_vxm_sparse_nomask_accum)
{
    using TEST_TYPE = int;
    std::vector<std::vector<TEST_TYPE>> mat = {{0, 0, 0, 1},
                                            {0, 0, 0, 4},
                                            {0, 0, 9, 0},
                                            {0, 3, 0, 1},
                                            {0, 3, 3, 1},
                                            {0, 1, 4, 2}};
    grb::backend::GKCMatrix<TEST_TYPE> m1(mat, 0);

    std::vector<TEST_TYPE> vec = {6, 0, 0, 4, -12, 0};
    // std::vector<TEST_TYPE> mask = {0, 1, 1, 0};
    grb::backend::GKCDenseVector<TEST_TYPE> v1(vec, 0);

    std::vector<TEST_TYPE> w_vec = {1, 0, 1, 0};
    grb::backend::GKCDenseVector<TEST_TYPE> w(w_vec, 0);

    grb::backend::vxm(w,
                      grb::NoMask(),
                      grb::Plus<TEST_TYPE>(),
                      grb::ArithmeticSemiring<TEST_TYPE>(),
                      v1, m1, MERGE);

    std::vector<TEST_TYPE> answer = { 1, -24, -35, -2};
    grb::backend::GKCDenseVector<TEST_TYPE> ans(answer, 0);
    BOOST_CHECK_EQUAL(ans, w);
}


//******************************************************************
// Mask VXM Tests (Mask with Replace)
//******************************************************************

BOOST_AUTO_TEST_CASE(test_mxv_sparse_mask_noaccum)
{
    std::vector<std::vector<double>> mat = {{0, 0, 0, 0},
                                            {0, 0, 0, 4},
                                            {0, 0, 9, 0},
                                            {0, 0, 5, 3},
                                            {0, 2, 0, 0},
                                            {0, 3, 0, 1},
                                            {0, 3, 3, 0},
                                            {0, 1, 4, 2},
                                            {6, 0, 0, 0},
                                            {6, 0, 0, 2},
                                            {6, 0, 4, 0},
                                            {6, 0, 4, 2},
                                            {6, 1, 0, 0},
                                            {6, 1, 0, 2},
                                            {6, 1, 4, 0},
                                            {6, 1, 4, 2}};
    grb::backend::GKCMatrix<double> m1(mat, 0);

    std::vector<double> vec = {6, 0, 0, 4};
    grb::backend::GKCDenseVector<double> v1(vec, 0);

    std::vector<double> mask_vec = { 0, 1, 0, 1, 0, 1, 0, 1,
                                     1, 0, 1, 0, 1, 0, 1, 0};
    grb::backend::GKCDenseVector<double> mask(mask_vec, 0);

    grb::backend::GKCDenseVector<double> w(16);

    grb::backend::mxv(w,
                      mask,
                      grb::NoAccumulate(),
                      grb::ArithmeticSemiring<double>(),
                      m1, v1, REPLACE);

    std::vector<double> answer = { 0, 16, 0, 12, 0, 4, 0, 8,
                                  36, 0, 36, 0, 36, 0, 36, 0};
    grb::backend::GKCDenseVector<double> ans(answer, 0);
    BOOST_CHECK_EQUAL(ans, w);
    //w.printInfo(std::cerr);
    //ans.printInfo(std::cerr);
}

BOOST_AUTO_TEST_CASE(test_mxv_sparse_mask_noaccum_transpose)
{
    using TEST_TYPE = int;
    std::vector<std::vector<TEST_TYPE>> mat = {{0, 0, 0, 1},
                                            {0, 0, 0, 4},
                                            {0, 0, 9, 0},
                                            {0, 3, 0, 1},
                                            {0, 3, 3, 1},
                                            {0, 1, 4, 2}};
    grb::backend::GKCMatrix<TEST_TYPE> m1(mat, 0);

    std::vector<TEST_TYPE> vec = {6, 0, 0, 4, -12, 0};
    grb::backend::GKCDenseVector<TEST_TYPE> v1(vec, 0);

    std::vector<TEST_TYPE> w_vec = {100, 0, -100, 0};
    grb::backend::GKCDenseVector<TEST_TYPE> w(w_vec, 0);

    std::vector<double> mask_vec = {0, 1, 0, 1};
    grb::backend::GKCDenseVector<double> mask(mask_vec, 0);

    grb::backend::mxv(w,
                      mask,
                      grb::NoAccumulate(),
                      grb::ArithmeticSemiring<TEST_TYPE>(),
                      grb::TransposeView(m1), v1, REPLACE);

    std::vector<TEST_TYPE> answer = { 0, -24, 0, -2};
    grb::backend::GKCDenseVector<TEST_TYPE> ans(answer, 0);
    BOOST_CHECK_EQUAL(ans, w);
    //w.printInfo(std::cerr);
    //ans.printInfo(std::cerr);
}

BOOST_AUTO_TEST_CASE(test_vxm_sparse_mask_noaccum_transpose)
{
    std::vector<std::vector<double>> mat = {{0, 0, 0, 0},
                                            {0, 0, 0, 4},
                                            {0, 0, 9, 0},
                                            {0, 0, 5, 3},
                                            {0, 2, 0, 0},
                                            {0, 3, 0, 1},
                                            {0, 3, 3, 0},
                                            {0, 1, 4, 2},
                                            {6, 0, 0, 0},
                                            {6, 0, 0, 2},
                                            {6, 0, 4, 0},
                                            {6, 0, 4, 2},
                                            {6, 1, 0, 0},
                                            {6, 1, 0, 2},
                                            {6, 1, 4, 0},
                                            {6, 1, 4, 2}};
    grb::backend::GKCMatrix<double> m1(mat, 0);

    std::vector<double> vec = {6, 0, 0, 4};
    grb::backend::GKCDenseVector<double> v1(vec, 0);

    grb::backend::GKCDenseVector<double> w(16);

    std::vector<double> mask_vec = {0, 1, 0, 1, 0, 1, 0, 1,
                                    1, 0, 1, 0, 1, 0, 1, 0};
    grb::backend::GKCDenseVector<double> mask(mask_vec, 0);

    grb::backend::vxm(w,
                      mask,
                      grb::NoAccumulate(),
                      grb::ArithmeticSemiring<double>(),
                      v1, grb::TransposeView(m1), REPLACE);

    std::vector<double> answer = { 0, 16,  0, 12,  0,  4,  0,  8,
                                  36, 0, 36, 0, 36, 0, 36, 0};
    grb::backend::GKCDenseVector<double> ans(answer, 0);
    BOOST_CHECK_EQUAL(ans, w);
    //w.printInfo(std::cerr);
    //ans.printInfo(std::cerr);
}

BOOST_AUTO_TEST_CASE(test_vxm_sparse_mask_noaccum)
{
    using TEST_TYPE = int;
    std::vector<std::vector<TEST_TYPE>> mat = {{0, 0, 0, 1},
                                            {0, 0, 0, 4},
                                            {0, 0, 9, 0},
                                            {0, 3, 0, 1},
                                            {0, 3, 3, 1},
                                            {0, 1, 4, 2}};
    grb::backend::GKCMatrix<TEST_TYPE> m1(mat, 0);

    std::vector<TEST_TYPE> vec = {6, 0, 0, 4, -12, 0};
    grb::backend::GKCDenseVector<TEST_TYPE> v1(vec, 0);

    std::vector<TEST_TYPE> w_vec = {1, 1, 1, 1};
    grb::backend::GKCDenseVector<TEST_TYPE> w(w_vec, 0);

    std::vector<double> mask_vec = {0, 1, 0, 1};
    grb::backend::GKCDenseVector<double> mask(mask_vec, 0);

    grb::backend::vxm(w,
                      mask,
                      grb::NoAccumulate(),
                      grb::ArithmeticSemiring<TEST_TYPE>(),
                      v1, m1, REPLACE);

    std::vector<TEST_TYPE> answer = { 0, -24, 0, -2};
    grb::backend::GKCDenseVector<TEST_TYPE> ans(answer, 0);
    BOOST_CHECK_EQUAL(ans, w);
    //w.printInfo(std::cerr);
    //ans.printInfo(std::cerr);
}

BOOST_AUTO_TEST_CASE(test_vxm_sparse_mask_accum_transpose)
{
    std::vector<std::vector<double>> mat = {{0, 0, 0, 0},
                                            {0, 0, 0, 4},
                                            {0, 0, 9, 0},
                                            {0, 0, 5, 3},
                                            {0, 2, 0, 0},
                                            {0, 3, 0, 1},
                                            {0, 3, 3, 0},
                                            {0, 1, 4, 2},
                                            {6, 0, 0, 0},
                                            {6, 0, 0, 2},
                                            {6, 0, 4, 0},
                                            {6, 0, 4, 2},
                                            {6, 1, 0, 0},
                                            {6, 1, 0, 2},
                                            {6, 1, 4, 0},
                                            {6, 1, 4, 2}};
    grb::backend::GKCMatrix<double> m1(mat, 0);

    std::vector<double> vec = {6, 0, 0, 4};
    grb::backend::GKCDenseVector<double> v1(vec, 0);

    std::vector<double> w_vec = { 1, 1, 1, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0};
    grb::backend::GKCDenseVector<double> w(w_vec, 0);

    std::vector<char> mask_vec = {0, 1, 0, 1, 0, 1, 0, 1,
                                    1, 0, 1, 0, 1, 0, 1, 0};
    grb::backend::GKCDenseVector<char> mask(mask_vec, 0);

    grb::backend::vxm(w,
                      mask,  
                      grb::Plus<double>(),
                      grb::ArithmeticSemiring<double>(),
                      v1, grb::TransposeView(m1), REPLACE);

    std::vector<double> answer = { 0, 17,  0, 12,  0,  4,  0,  8,
                                  36, 0, 36, 0, 36, 0, 36, 0};
    grb::backend::GKCDenseVector<double> ans(answer, 0);
    BOOST_CHECK_EQUAL(ans, w);
    //w.printInfo(std::cerr);
    //ans.printInfo(std::cerr);
}

BOOST_AUTO_TEST_CASE(test_vxm_sparse_mask_accum)
{
    using TEST_TYPE = int;
    std::vector<std::vector<TEST_TYPE>> mat = {{0, 0, 0, 1},
                                            {0, 0, 0, 4},
                                            {0, 0, 9, 0},
                                            {0, 3, 0, 1},
                                            {0, 3, 3, 1},
                                            {0, 1, 4, 2}};
    grb::backend::GKCMatrix<TEST_TYPE> m1(mat, 0);

    std::vector<TEST_TYPE> vec = {6, 0, 0, 4, -12, 0};
    // std::vector<TEST_TYPE> mask = {0, 1, 1, 0};
    grb::backend::GKCDenseVector<TEST_TYPE> v1(vec, 0);

    std::vector<TEST_TYPE> w_vec = {1, 0, 1, 0};
    grb::backend::GKCDenseVector<TEST_TYPE> w(w_vec, 0);

    std::vector<double> mask_vec = {0, 1, 0, 1};
    grb::backend::GKCDenseVector<double> mask(mask_vec, 0);

    grb::backend::vxm(w,
                      mask,
                      grb::Plus<TEST_TYPE>(),
                      grb::ArithmeticSemiring<TEST_TYPE>(),
                      v1, m1, REPLACE);

    std::vector<TEST_TYPE> answer = { 0, -24, 0, -2};
    grb::backend::GKCDenseVector<TEST_TYPE> ans(answer, 0);
    BOOST_CHECK_EQUAL(ans, w);
}

BOOST_AUTO_TEST_SUITE_END()
