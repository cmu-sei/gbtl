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

using namespace grb;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE extracttuples_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

//****************************************************************************
BOOST_AUTO_TEST_CASE(extracttuples_test_simple)
{
    IndexArrayType rows    = {0, 0, 0, 1, 1, 1, 2, 2};
    IndexArrayType columns = {1, 2, 3, 0, 2, 3, 0, 1};
    std::vector<double> values        = {1, 2, 3, 4, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> m1(3, 4);
    m1.build(rows, columns, values);

    IndexType nnz = m1.nvals();
    IndexArrayType r(nnz), c(nnz);
    std::vector<double> v(nnz);

    m1.extractTuples(r, c, v);

    /// @todo the following check should be a utility function

    // Check the result, but it may be out of order.
    bool success = true;
    for (IndexType ix = 0; ix < values.size(); ++ix)
    {
        // Note: no semantics defined for extractTuples regarding the
        // order of returned values, so using an O(N^2) approach
        // without sorting:
        bool found = false;
        for (IndexType iy = 0; iy < v.size(); ++iy)
        {
            if ((r[iy] == rows[ix]) && (c[iy] == columns[ix]))
            {
                found = true;
                if (v[iy] != values[ix])
                {
                    success = false;
                }
                break;
            }
        }
        if (!found)
        {
            success = false;
        }
    }

    BOOST_CHECK_EQUAL(success, true);
}

BOOST_AUTO_TEST_SUITE_END()
