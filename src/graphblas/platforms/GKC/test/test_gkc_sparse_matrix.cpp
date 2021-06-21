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

using namespace grb;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE gkc_sparse_matrix_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

//****************************************************************************
// GKC Matrix basic constructor
BOOST_AUTO_TEST_CASE(gkc_test_construction_basic)
{
    IndexType M = 7;
    IndexType N = 4;
    backend::GKCMatrix<double> m1(M, N);

    IndexType num_rows(m1.nrows());
    IndexType num_cols(m1.ncols());

    BOOST_CHECK_EQUAL(num_rows, M);
    BOOST_CHECK_EQUAL(num_cols, N);
}

//****************************************************************************
// GKC constructor from dense matrix
BOOST_AUTO_TEST_CASE(gkc_test_construction_dense)
{
    std::vector<std::vector<double>> mat = {{6, 0, 0, 4},
                                            {7, 0, 0, 0},
                                            {0, 0, 9, 4},
                                            {2, 5, 0, 3},
                                            {2, 0, 0, 1},
                                            {0, 0, 0, 0},
                                            {0, 1, 0, 2}};

    backend::GKCMatrix<double> m1(mat);

    IndexType M = mat.size();
    IndexType N = mat[0].size();
    IndexType num_rows(m1.nrows());
    IndexType num_cols(m1.ncols());

    BOOST_CHECK_EQUAL(num_rows, M);
    BOOST_CHECK_EQUAL(num_cols, N);
    for (IndexType i = 0; i < M; i++)
    {
        for (IndexType j = 0; j < N; j++)
        {
            BOOST_CHECK_EQUAL(m1.extractElement(i, j), mat[i][j]);
        }
    }
}

//****************************************************************************
// GKC constructor from dense matrix, implied zeros
BOOST_AUTO_TEST_CASE(gkc_test_construction_dense_zero)
{
    std::vector<std::vector<double>> mat = {{6, 0, 0, 4},
                                            {7, 0, 0, 0},
                                            {0, 0, 9, 4},
                                            {2, 5, 0, 3},
                                            {2, 0, 0, 1},
                                            {0, 0, 0, 0},
                                            {0, 1, 0, 2}};

    backend::GKCMatrix<double> m1(mat, 0);

    IndexType M = mat.size();
    IndexType N = mat[0].size();
    IndexType num_rows(m1.nrows());
    IndexType num_cols(m1.ncols());

    BOOST_CHECK_EQUAL(num_rows, M);
    BOOST_CHECK_EQUAL(num_cols, N);
    for (IndexType i = 0; i < M; i++)
    {
        for (IndexType j = 0; j < N; j++)
        {
            if (mat[i][j] != 0)
            {
                BOOST_CHECK_EQUAL(m1.extractElement(i, j), mat[i][j]);
            }
        }
    }
}

//****************************************************************************
// GKC constructor from copy
BOOST_AUTO_TEST_CASE(gkc_test_construction_copy)
{
    std::vector<std::vector<double>> mat = {{6, 0, 0, 4},
                                            {7, 0, 0, 0},
                                            {0, 0, 9, 4},
                                            {2, 5, 0, 3},
                                            {2, 0, 0, 1},
                                            {0, 0, 0, 0},
                                            {0, 1, 0, 2}};

    backend::GKCMatrix<double> m1(mat, 0);

    backend::GKCMatrix<double> m2(m1);

    BOOST_CHECK_EQUAL(m1, m2);
}

/*
//****************************************************************************
BOOST_AUTO_TEST_CASE(gkc_test_assign_to_implied_zero)
{
    std::vector<std::vector<double>> mat = {{6, 0, 0, 4},
                                            {7, 0, 0, 0},
                                            {0, 0, 9, 4},
                                            {2, 5, 0, 3},
                                            {2, 0, 0, 1},
                                            {0, 0, 0, 0},
                                            {0, 1, 0, 2}};

    backend::GKCMatrix<double> m1(mat, 0);

    mat[0][1] = 8;
    m1.setElement(0, 1, 8);
    BOOST_CHECK_EQUAL(m1.extractElement(0, 1), mat[0][1]);

    backend::GKCMatrix<double> m2(mat, 0);
    BOOST_CHECK_EQUAL(m1, m2);
}
*/

//****************************************************************************
BOOST_AUTO_TEST_CASE(gkc_test_assign_to_nonzero_element)
{
    std::vector<std::vector<double>> mat = {{6, 0, 0, 4},
                                            {7, 0, 0, 0},
                                            {0, 0, 9, 4},
                                            {2, 5, 0, 3},
                                            {2, 0, 0, 1},
                                            {0, 0, 0, 0},
                                            {0, 1, 0, 2}};

    backend::GKCMatrix<double> m1(mat, 0);

    mat[0][0] = 8;
    m1.setElement(0, 0, 8);
    BOOST_CHECK_EQUAL(m1.extractElement(0, 0), mat[0][0]);

    backend::GKCMatrix<double> m2(mat, 0);
    BOOST_CHECK_EQUAL(m1, m2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(gkc_test_construct_with_build)
{
    // This test is contingent on the dense implicit constructor working
    // and on equality checks working.
    std::vector<std::vector<double>> mat = {{6, 0, 0, 4},
                                            {7, 0, 0, 0},
                                            {0, 0, 9, 4},
                                            {2, 5, 0, 3},
                                            {2, 0, 0, 1},
                                            {0, 0, 0, 0},
                                            {0, 1, 0, 2}};
    // This is transposed coordinate data.
    std::vector<IndexType> matI = {0,2,3,4,6,0,1,3,4,3,6,2};
    std::vector<IndexType> matJ = {3,3,3,3,3,0,0,0,0,1,1,2};
    std::vector<double> matV    = {4,4,3,1,2,6,7,2,2,5,1,9};

    backend::GKCMatrix<double> m1(mat, 0);
    backend::GKCMatrix<double> m2(matI.begin(), matJ.begin(), matV.begin(), matV.size(), [=](double w1, double w2){return w1;});

    BOOST_CHECK_EQUAL(m1, m2);
}

BOOST_AUTO_TEST_SUITE_END()
