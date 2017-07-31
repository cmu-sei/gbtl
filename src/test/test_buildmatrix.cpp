/*
 * Copyright (c) 2015 Carnegie Mellon University and The Trustees of Indiana
 * University.
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

#include <iostream>

#include <graphblas/graphblas.hpp>

using namespace GraphBLAS;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE matrix_build_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(matrix_build_suite)

//****************************************************************************
BOOST_AUTO_TEST_CASE(matrix_build_test)
{
    IndexArrayType i = {0, 0, 0, 1, 1, 1, 2, 2};
    IndexArrayType j = {1, 2, 3, 0, 2, 3, 0, 1};
    std::vector<double>       v = {1, 2, 3, 4, 6, 7, 8, 9};

    std::vector<std::vector<double> > mat = {{0, 1, 2, 3},
                                             {4, 0, 6, 7},
                                             {8, 9, 0, 0}};

    Matrix<double, DirectedMatrixTag> m1(3, 4);
    m1.build(i, j, v);

    //std::cerr << m1 << std::endl;
    IndexType nnz = m1.nvals();
    BOOST_CHECK_EQUAL(nnz, i.size());

    IndexArrayType ii(nnz), jj(nnz);
    std::vector<double> val(nnz);
    m1.extractTuples(ii, jj, val);
    BOOST_CHECK_EQUAL(nnz, ii.size());
    BOOST_CHECK_EQUAL(nnz, jj.size());
    BOOST_CHECK_EQUAL(nnz, val.size());

    for (IndexType idx = 0; idx < val.size(); ++idx)
    {
        BOOST_CHECK_EQUAL(val[idx], mat[ii[idx]][jj[idx]]);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(matrix_build_test_iterators)
{
    IndexArrayType i = {0, 0, 0, 1, 1, 1, 2, 2};
    IndexArrayType j = {1, 2, 3, 0, 2, 3, 0, 1};
    std::vector<double>       v = {1, 2, 3, 4, 6, 7, 8, 9};

    std::vector<std::vector<double> > mat = {{0, 1, 2, 3},
                                             {4, 0, 6, 7},
                                             {8, 9, 0, 0}};

    Matrix<double, DirectedMatrixTag> m1(3, 4);
    m1.build(i.begin(), j.begin(), v.begin(), i.size());

    //std::cerr << m1 << std::endl;
    IndexType nnz = m1.nvals();
    BOOST_CHECK_EQUAL(nnz, i.size());

    IndexArrayType ii(nnz), jj(nnz);
    std::vector<double> val(nnz);
    m1.extractTuples(ii, jj, val);
    BOOST_CHECK_EQUAL(nnz, ii.size());
    BOOST_CHECK_EQUAL(nnz, jj.size());
    BOOST_CHECK_EQUAL(nnz, val.size());

    for (IndexType idx = 0; idx < val.size(); ++idx)
    {
        BOOST_CHECK_EQUAL(val[idx], mat[ii[idx]][jj[idx]]);
    }
}

BOOST_AUTO_TEST_SUITE_END()
