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
 * DM20-0442
 */

#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <vector>

#include <graphblas/graphblas.hpp>
using namespace grb;

//****************************************************************************
int main(int, char**)
{
    // syntatic sugar
    using ScalarType = double;

    IndexType const NUM_ROWS = 3;
    IndexType const NUM_COLS = 3;

    // Note: size of dimensions require at ccnstruction
    Matrix<ScalarType> a(NUM_ROWS, NUM_COLS);
    Matrix<ScalarType> b(NUM_ROWS, NUM_COLS);
    Matrix<ScalarType> c(NUM_ROWS, NUM_COLS);

    // initialize matrices
    IndexArrayType i = {0,  1,  2};
    IndexArrayType j = {0,  1,  2};
    std::vector<ScalarType>   v = {1., 1., 1.};

    a.build(i.begin(), j.begin(), v.begin(), i.size());
    b.build(i.begin(), j.begin(), v.begin(), i.size());

    print_matrix(std::cout, a, "Matrix A");
    print_matrix(std::cout, b, "Matrix B");

    // matrix multiply (default parameter values used for some)
    mxm(c, NoMask(), NoAccumulate(), ArithmeticSemiring<ScalarType>(), a, b);

    print_matrix(std::cout, c, "A +.* B");

    // extract the results: nvals() method tells us how big
    IndexType nvals = c.nvals();
    IndexArrayType rows(nvals), cols(nvals);
    std::vector<ScalarType> result(nvals);

    c.extractTuples(rows, cols, result);

    IndexArrayType i_ans = {0,  1,  2};
    IndexArrayType j_ans = {0,  1,  2};
    std::vector<ScalarType>   v_ans = {1., 1., 1.};

    bool success = true;
    for (IndexType ix = 0; ix < result.size(); ++ix)
    {
        // Note: no semantics defined for extractTuples regarding the
        // order of returned values, so using an O(N^2) approach
        // without sorting:
        bool found = false;
        for (IndexType iy = 0; iy < v_ans.size(); ++iy)
        {
            if ((i_ans[iy] == rows[ix]) && (j_ans[iy] == cols[ix]))
            {
                std::cout << "Found result: result index, answer index = "
                          << ix << ", " << iy
                          << ": res("    << rows[ix] << "," << cols[ix]
                          << ") ?= ans(" << i_ans[iy] << "," << j_ans[iy] << "), "
                          << result[ix] << " ?= " << v_ans[iy] << std::endl;
                found = true;
                if (v_ans[iy] != result[ix])
                {
                    std::cerr << "ERROR" << std::endl;
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

    return (success ? 0 : 1);
}
