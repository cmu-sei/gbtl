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

#include <algorithms/bc.hpp>
#include <graphblas/graphblas.hpp>

using namespace GraphBLAS;
using namespace algorithms;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE bc_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(bc_suite)

static std::vector<double> br={0, 0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 6};
static std::vector<double> bc={1, 2, 3, 2, 4, 4, 2, 4, 5, 6, 7, 7};
static std::vector<double> bv(br.size(), 1);

//static Matrix<double, DirectedMatrixTag> betweenness(
//    {{0, 1, 1, 1, 0, 0, 0, 0},
//     {0, 0, 1, 0, 1, 0, 0, 0},
//     {0, 0, 0, 0, 1, 0, 0, 0},
//     {0, 0, 1, 0, 1, 0, 0, 0},
//     {0, 0, 0, 0, 0, 1, 1, 0},
//     {0, 0, 0, 0, 0, 0, 0, 1},
//     {0, 0, 0, 0, 0, 0, 0, 1},
//     {0, 0, 0, 0, 0, 0, 0, 0}});
//

//****************************************************************************
BOOST_AUTO_TEST_CASE(bc_test_vertex_betweenness_centrality)
{
    std::cerr << "test" << std::endl;
    Matrix<double, DirectedMatrixTag> betweenness(8,8);
    betweenness.build(br.begin(), bc.begin(), bv.begin(), bv.size());
    std::vector<double> result = vertex_betweenness_centrality(betweenness);
    std::vector<double> answer = {0.0, 4.0/3, 4.0/3, 4.0/3, 12.0, 2.5, 2.5, 0.0};

    BOOST_CHECK_EQUAL(result.size(), answer.size());
    for (unsigned int ix = 0; ix < result.size(); ++ix)
        BOOST_CHECK_CLOSE(result[ix], answer[ix], 0.0001);

    /// DEBUG
    exit(1);
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(bc_test_edge_betweenness_centrality)
{
    std::cerr << "test" << std::endl;
    Matrix<double, DirectedMatrixTag> betweenness(8,8);
    Matrix<double, DirectedMatrixTag> result(8,8);
    //Matrix<double, DirectedMatrixTag> answer ({{0,  6,  6,  6,  0,  0,  0,  0},
    //                            {0,  0,  1,  0, 10,  0,  0,  0},
    //                            {0,  0,  0,  0, 10,  0,  0,  0},
    //                            {0,  0,  1,  0, 10,  0,  0,  0},
    //                            {0,  0,  0,  0,  0, 14, 14,  0},
    //                            {0,  0,  0,  0,  0,  0,  0,  8},
    //                            {0,  0,  0,  0,  0,  0,  0,  8},
    //                            {0,  0,  0,  0,  0,  0,  0,  0}});

    std::vector<double> ar={0, 0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 6};
    std::vector<double> ac={1, 2, 3, 2, 4, 4, 2, 4, 5, 6, 7, 7};
    std::vector<double> av={6,  6,  6,  1, 10, 10,  1, 10, 14, 14,  8,  8};
    Matrix<double, DirectedMatrixTag> answer(8,8);
    answer.build(ar.begin(), ac.begin(), av.begin(), av.size());

    betweenness.build(br.begin(), bc.begin(), bv.begin(), bv.size());
    result = edge_betweenness_centrality(betweenness);

    //std::cout << result << std::endl;
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bc_test_vertex_betweennes_centrality_batch)
{
    std::cerr << "test" << std::endl;
    Matrix<double, DirectedMatrixTag> betweenness(8,8);
    betweenness.build(br.begin(), bc.begin(), bv.begin(), bv.size());

    //IndexArrayType seed_set={0, 1, 2, 3, 4, 5, 6, 7};
    IndexArrayType seed_set={0};
    std::vector<float> answer = {0.0, 4.0/3, 4.0/3, 4.0/3, 3.0, 0.5, 0.5, 0.0};

    std::vector<float> result =
        vertex_betweenness_centrality_batch(betweenness, seed_set);

    BOOST_CHECK_EQUAL(result.size(), answer.size());
    for (unsigned int ix = 0; ix < result.size(); ++ix)
        BOOST_CHECK_CLOSE(result[ix], answer[ix], 0.0001);

    //==========
    IndexArrayType seed_set3={3};
    std::vector<float> answer3 = {0.0, 0.0, 0.0, 0.0, 3.0, 0.5, 0.5, 0.0};

    std::vector<float> result3 =
        vertex_betweenness_centrality_batch(betweenness, seed_set3);

    BOOST_CHECK_EQUAL(result3.size(), answer3.size());
    for (unsigned int ix = 0; ix < result3.size(); ++ix)
        BOOST_CHECK_CLOSE(result3[ix], answer3[ix], 0.0001);

    //==========
    IndexArrayType seed_set03={0,3};

    std::vector<float> result03 =
        vertex_betweenness_centrality_batch(betweenness, seed_set03);

    BOOST_CHECK_EQUAL(result03.size(), answer3.size());
    for (unsigned int ix = 0; ix < result3.size(); ++ix)
        BOOST_CHECK_CLOSE(result03[ix], answer[ix] + answer3[ix], 0.0001);

    //==========
    IndexArrayType seed_set_all={0,1,2,3,4,5,6,7};
    std::vector<double> answer_all = {0.0, 4.0/3, 4.0/3, 4.0/3, 12.0, 2.5, 2.5, 0.0};

    std::vector<float> result_all =
        vertex_betweenness_centrality_batch(betweenness, seed_set_all);

    BOOST_CHECK_EQUAL(result_all.size(), answer_all.size());
    for (unsigned int ix = 0; ix < result_all.size(); ++ix)
        BOOST_CHECK_CLOSE(result_all[ix], answer_all[ix], 0.0001);

}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bc_test_vertex_betweennes_centrality_batch_alt)
{
    std::cerr << "test" << std::endl;
    // Trying the same graph where each node has self loop.     vvvvvvvvvvvvvvv
    std::vector<double> br1={0, 0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 6,0,1,2,3,4,5,6,7};
    std::vector<double> bc1={1, 2, 3, 2, 4, 4, 2, 4, 5, 6, 7, 7,0,1,2,3,4,5,6,7};
    std::vector<double> bv1(br1.size(), 1);

    Matrix<double, DirectedMatrixTag> betweenness(8,8);
    betweenness.build(br1.begin(), bc1.begin(), bv1.begin(), bv1.size());

    //IndexArrayType seed_set={0, 1, 2, 3, 4, 5, 6, 7};
    IndexArrayType seed_set={0};
    std::vector<float> answer = {0.0, 4.0/3, 4.0/3, 4.0/3, 3.0, 0.5, 0.5, 0.0};
    std::cerr << "******************** START ALT ***********************" << std::endl;
    std::vector<float> result =
        vertex_betweenness_centrality_batch_alt(betweenness, seed_set);

    BOOST_CHECK_EQUAL(result.size(), answer.size());
    for (unsigned int ix = 0; ix < result.size(); ++ix)
        BOOST_CHECK_CLOSE(result[ix], answer[ix], 0.0001);

    //==========
    IndexArrayType seed_set3={3};
    std::vector<float> answer3 = {0.0, 0.0, 0.0, 0.0, 3.0, 0.5, 0.5, 0.0};

    std::vector<float> result3 =
        vertex_betweenness_centrality_batch_alt(betweenness, seed_set3);

    BOOST_CHECK_EQUAL(result3.size(), answer3.size());
    for (unsigned int ix = 0; ix < result3.size(); ++ix)
        BOOST_CHECK_CLOSE(result3[ix], answer3[ix], 0.0001);

    //==========
    IndexArrayType seed_set03={0,3};

    std::vector<float> result03 =
        vertex_betweenness_centrality_batch_alt(betweenness, seed_set03);

    BOOST_CHECK_EQUAL(result03.size(), answer3.size());
    for (unsigned int ix = 0; ix < result3.size(); ++ix)
        BOOST_CHECK_CLOSE(result03[ix], answer[ix] + answer3[ix], 0.0001);

    //==========
    IndexArrayType seed_set_all={0,1,2,3,4,5,6,7};
    std::vector<double> answer_all = {0.0, 4.0/3, 4.0/3, 4.0/3, 12.0, 2.5, 2.5, 0.0};

    std::vector<float> result_all =
        vertex_betweenness_centrality_batch_alt(betweenness, seed_set_all);

    BOOST_CHECK_EQUAL(result_all.size(), answer_all.size());
    for (unsigned int ix = 0; ix < result_all.size(); ++ix)
        BOOST_CHECK_CLOSE(result_all[ix], answer_all[ix], 0.0001);

}

//****************************************************************************
//BOOST_AUTO_TEST_CASE(bc_test_vertex_betweennes_centrality_batch_alt_trans)
BOOST_AUTO_TEST_CASE(bc_batch_alt_trans)
{
    std::cerr << "test" << std::endl;
    Matrix<double, DirectedMatrixTag> betweenness(8,8);
    betweenness.build(br.begin(), bc.begin(), bv.begin(), bv.size());

    //IndexArrayType seed_set={0, 1, 2, 3, 4, 5, 6, 7};
    IndexArrayType seed_set={0};
    std::vector<float> answer = {0.0, 4.0/3, 4.0/3, 4.0/3, 3.0, 0.5, 0.5, 0.0};
    std::cerr << "******************** START ALT TRANS ***********************" << std::endl;
    std::vector<float> result =
        vertex_betweenness_centrality_batch_alt_trans(betweenness, seed_set);

    BOOST_CHECK_EQUAL(result.size(), answer.size());
    for (unsigned int ix = 0; ix < result.size(); ++ix)
        BOOST_CHECK_CLOSE(result[ix], answer[ix], 0.0001);

    //==========
    IndexArrayType seed_set3={3};
    std::vector<float> answer3 = {0.0, 0.0, 0.0, 0.0, 3.0, 0.5, 0.5, 0.0};

    std::vector<float> result3 =
        vertex_betweenness_centrality_batch_alt_trans(betweenness, seed_set3);

    BOOST_CHECK_EQUAL(result3.size(), answer3.size());
    for (unsigned int ix = 0; ix < result3.size(); ++ix)
        BOOST_CHECK_CLOSE(result3[ix], answer3[ix], 0.0001);

    //==========
    IndexArrayType seed_set03={0,3};

    std::vector<float> result03 =
        vertex_betweenness_centrality_batch_alt_trans(betweenness, seed_set03);

    BOOST_CHECK_EQUAL(result03.size(), answer3.size());
    for (unsigned int ix = 0; ix < result3.size(); ++ix)
        BOOST_CHECK_CLOSE(result03[ix], answer[ix] + answer3[ix], 0.0001);

    //==========
    IndexArrayType seed_set_all={0,1,2,3,4,5,6,7};
    std::vector<double> answer_all = {0.0, 4.0/3, 4.0/3, 4.0/3, 12.0, 2.5, 2.5, 0.0};

    std::vector<float> result_all =
        vertex_betweenness_centrality_batch_alt_trans(betweenness, seed_set_all);

    BOOST_CHECK_EQUAL(result_all.size(), answer_all.size());
    for (unsigned int ix = 0; ix < result_all.size(); ++ix)
        BOOST_CHECK_CLOSE(result_all[ix], answer_all[ix], 0.0001);

    std::cerr << "******************** END ALT TRANS ***********************" << std::endl;

}

//****************************************************************************
//BOOST_AUTO_TEST_CASE(bc_test_vertex_betweennes_centrality_batch_alt_trans_v2)
BOOST_AUTO_TEST_CASE(bc_test_batch_alt_trans_v2)
{
    std::cerr << "test" << std::endl;
//    Matrix<double, DirectedMatrixTag> betweenness(8,8);
//    betweenness.build(br.begin(), bc.begin(), bv.begin(), bv.size());
    Matrix<double, DirectedMatrixTag> betweenness(8,8);
    betweenness.build(br.begin(), bc.begin(), bv.begin(), bv.size());

    //IndexArrayType seed_set={0, 1, 2, 3, 4, 5, 6, 7};
    IndexArrayType seed_set={0};
    std::vector<float> answer = {0.0, 4.0/3, 4.0/3, 4.0/3, 3.0, 0.5, 0.5, 0.0};
    std::cerr << "******************** START ALT ++ V2 ++ TRANS ***********************" << std::endl;
    std::vector<float> result = vertex_betweenness_centrality_batch_alt_trans_v2(
            betweenness, seed_set);

    BOOST_CHECK_EQUAL(result.size(), answer.size());
    for (unsigned int ix = 0; ix < result.size(); ++ix)
        BOOST_CHECK_CLOSE(result[ix], answer[ix], 0.0001);

    //==========
    // IndexArrayType seed_set3={3};
    // std::vector<float> answer3 = {0.0, 0.0, 0.0, 0.0, 3.0, 0.5, 0.5, 0.0};

    // std::vector<float> result3 =
    //     vertex_betweenness_centrality_batch_alt_trans_v2(betweenness, seed_set3);

    // BOOST_CHECK_EQUAL(result3.size(), answer3.size());
    // for (unsigned int ix = 0; ix < result3.size(); ++ix)
    //     BOOST_CHECK_CLOSE(result3[ix], answer3[ix], 0.0001);

    // //==========
    // IndexArrayType seed_set03={0,3};

    // std::vector<float> result03 =
    //     vertex_betweenness_centrality_batch_alt_trans_v2(betweenness, seed_set03);

    // BOOST_CHECK_EQUAL(result03.size(), answer3.size());
    // for (unsigned int ix = 0; ix < result3.size(); ++ix)
    //     BOOST_CHECK_CLOSE(result03[ix], answer[ix] + answer3[ix], 0.0001);

    // //==========
    // IndexArrayType seed_set_all={0,1,2,3,4,5,6,7};
    // std::vector<double> answer_all = {0.0, 4.0/3, 4.0/3, 4.0/3, 12.0, 2.5, 2.5, 0.0};

    // std::vector<float> result_all =
    //     vertex_betweenness_centrality_batch_alt_trans_v2(betweenness, seed_set_all);

    // BOOST_CHECK_EQUAL(result_all.size(), answer_all.size());
    // for (unsigned int ix = 0; ix < result_all.size(); ++ix)
    //     BOOST_CHECK_CLOSE(result_all[ix], answer_all[ix], 0.0001);

    std::cerr << "******************** END ALT ++ V2 ++ TRANS ***********************" << std::endl;

}


BOOST_AUTO_TEST_SUITE_END()
