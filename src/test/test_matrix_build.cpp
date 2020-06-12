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

#include <iostream>

#include <graphblas/graphblas.hpp>

using namespace grb;

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
