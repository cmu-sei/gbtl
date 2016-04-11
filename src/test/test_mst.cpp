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

#include <algorithms/mst.hpp>
#include <graphblas/graphblas.hpp>

using namespace graphblas;
using namespace algorithms;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE mst_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(mst_suite)

//****************************************************************************
BOOST_AUTO_TEST_CASE(mst_test_with_weight_one)
{
    graphblas::IndexArrayType i_m1 = {0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3,
                                      4, 4, 4, 4, 4, 5, 5};
    graphblas::IndexArrayType j_m1 = {1, 3, 4, 0, 3, 4, 4, 5, 0, 1, 4,
                                      0, 1, 2, 3, 5, 2, 4};
    std::vector<double>       v_m1 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                      1, 1, 1, 1, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> m1(6, 6);
    buildmatrix(m1, i_m1, j_m1, v_m1);

    std::vector<std::tuple<graphblas::IndexType,
                           graphblas::IndexType,
                           double> > ans({
    	std::make_tuple(1, 0, 1.0),
    	std::make_tuple(1, 3, 1.0),
    	std::make_tuple(1, 4, 1.0),
    	std::make_tuple(4, 2, 1.0),
    	std::make_tuple(4, 5, 1.0)
    });

    std::pair<double,
              std::vector<std::tuple<graphblas::IndexType,
                                     graphblas::IndexType,
                                     double> > > answer =
        std::make_pair(5.0, ans);

    auto result = mst(m1);

    bool valid = true;
    valid = valid && (result.first == answer.first);
    valid = valid && (result.second.size() == answer.second.size());
    for (IndexType index=0; index < result.second.size(); ++index)
    {
    	auto tuple = result.second[index];
    	auto ans_tuple = answer.second[index];

    	graphblas::IndexType i, j, ans_i, ans_j;
    	double weight, ans_weight;

    	std::tie(i, j, weight) = tuple;
    	std::tie(ans_i, ans_j, ans_weight) = ans_tuple;

    	valid = valid && (i == ans_i);
    	valid = valid && (j == ans_j);
    	valid = valid && (weight == ans_weight);
    }

    BOOST_CHECK_EQUAL(valid, true);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(mst_test_with_weights)
{
    // LilMatrix<double> m1({{0, 4, 0, 0, 0, 0, 0, 8, 0},
    //                       {4, 0, 8, 0, 0, 0, 0,11, 0},
    //                       {0, 8, 0, 7, 0, 4, 0, 0, 2},
    //                       {0, 0, 7, 0, 9,14, 0, 0, 0},
    //                       {0, 0, 0, 9, 0,10, 0, 0, 0},
    //                       {0, 0, 4,14,10, 0, 2, 0, 0},
    //                       {0, 0, 0, 0, 0, 2, 0, 1, 6},
    //                       {8,11, 0, 0, 0, 0, 1, 0, 7},
    //                       {0, 0, 2, 0, 0, 0, 6, 7, 0}});

    graphblas::IndexArrayType i_m1 = {0, 0, 1, 1, 1, 2, 2, 2, 2,
                                      3, 3, 3, 4, 4, 5, 5, 5, 5,
                                      6, 6, 6, 7, 7, 7, 7, 8, 8, 8};
    graphblas::IndexArrayType j_m1 = {1, 7, 0, 2, 7, 1, 3, 5, 8,
                                      2, 4, 5, 3, 5, 2, 3, 4, 6,
                                      5, 7, 8, 0, 1, 6, 8, 2, 6, 7};
    std::vector<double>       v_m1 = {4, 8, 4, 8,11, 8, 7, 4, 2,
                                      7, 9,14, 9,10, 4,14,10, 2,
                                      2, 1, 6, 8,11, 1, 7, 2, 6, 7};
    Matrix<double, DirectedMatrixTag> m1(9, 9);
    buildmatrix(m1, i_m1, j_m1, v_m1);

    auto result = mst(m1);

    std::vector<std::tuple<graphblas::IndexType,
                           graphblas::IndexType,
                           double> > ans({
        std::make_tuple(1,0,4),
        std::make_tuple(1,2,8),
        std::make_tuple(2,8,2),
        std::make_tuple(2,5,4),
        std::make_tuple(5,6,2),
        std::make_tuple(6,7,1),
        std::make_tuple(2,3,7),
        std::make_tuple(3,4,9)
    });

    std::pair<double,
              std::vector<std::tuple<graphblas::IndexType,
                                     graphblas::IndexType,
                                     double> > > answer =
        std::make_pair(37.0, ans);

    bool valid = true;
    valid = valid && (result.first == answer.first);
    valid = valid && (result.second.size() == answer.second.size());
    for (IndexType index = 0; index < result.second.size(); ++index)
    {
        auto tuple = result.second[index];
        auto ans_tuple = answer.second[index];

        graphblas::IndexType i, j, ans_i, ans_j;
        double weight, ans_weight;

        std::tie(i, j, weight) = tuple;
        std::tie(ans_i, ans_j, ans_weight) = ans_tuple;

        valid = valid && (i == ans_i);
        valid = valid && (j == ans_j);
        valid = valid && (weight == ans_weight);
    }

    BOOST_CHECK_EQUAL(valid, true);
}

BOOST_AUTO_TEST_SUITE_END()
