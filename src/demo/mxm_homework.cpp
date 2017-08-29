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

#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <vector>

#include <graphblas/graphblas.hpp>
using namespace GraphBLAS;

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
