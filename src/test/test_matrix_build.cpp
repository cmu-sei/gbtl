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

#include <iostream>

#include <graphblas/graphblas.hpp>

using namespace GraphBLAS;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE matrix_build_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

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
