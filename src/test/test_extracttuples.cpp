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

#include <graphblas/graphblas.hpp>

using namespace GraphBLAS;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE extracttuples_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(extracttuples_suite)

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
