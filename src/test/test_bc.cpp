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

#define GRAPHBLAS_LOGGING_LEVEL 0
#define GRAPHBLAS_BC_DEBUG 0

#include <graphblas/graphblas.hpp>
#include <algorithms/bc.hpp>

using namespace grb;
using namespace algorithms;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE bc_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

//****************************************************************************

static std::vector<double> br={0, 0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 6};
static std::vector<double> bc={1, 2, 3, 2, 4, 4, 2, 4, 5, 6, 7, 7};
static std::vector<double> bv(br.size(), 1);

//static Matrix<double, DirectedMatrixTag> betweenness(
//    {{0, 1, 1, 1, 0, 0, 0, 0},
//     {0, 0, 1, 0, 1, 0, 0, 0},
//     {0, 0, 0, 0, 1, 0, 0, 0},vertex_betweenness_centrality_batch_alt_trans_v2
//     {0, 0, 1, 0, 1, 0, 0, 0},
//     {0, 0, 0, 0, 0, 1, 1, 0},
//     {0, 0, 0, 0, 0, 0, 0, 1},
//     {0, 0, 0, 0, 0, 0, 0, 1},
//     {0, 0, 0, 0, 0, 0, 0, 0}});

//****************************************************************************
BOOST_AUTO_TEST_CASE(bc_test_vertex_betweenness_centrality_gilbert)
{
    IndexType const NUM_NODES = 7;
    std::vector<IndexType> row_indices = {0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6};
    std::vector<IndexType> col_indices = {1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4};
    std::vector<double> values(row_indices.size(), 1.);

    Matrix<double> graph(NUM_NODES, NUM_NODES);
    graph.build(row_indices.begin(), col_indices.begin(),
                      values.begin(), values.size());

    IndexArrayType seed_set={0, 1, 2, 3, 4, 5, 6};

    //std::vector<float> result = vertex_betweenness_centrality(graph);
    std::vector<float> answer = {4.0, 4.5, 2, 4.5, 2.0, 1.0, 3.0};

    {
        std::vector<double> result =
            vertex_betweenness_centrality(graph);

        BOOST_CHECK_EQUAL(result.size(), answer.size());
        for (unsigned int ix = 0; ix < result.size(); ++ix)
            BOOST_CHECK_CLOSE(result[ix], answer[ix], 0.0001);
    }
    {
        std::vector<double> result =
            vertex_betweenness_centrality_batch_old(graph, seed_set);

        BOOST_CHECK_EQUAL(result.size(), answer.size());
        for (unsigned int ix = 0; ix < result.size(); ++ix)
            BOOST_CHECK_CLOSE(result[ix], answer[ix], 0.0001);
    }
    {
        std::vector<float> result =
            vertex_betweenness_centrality_batch(graph, seed_set);

        BOOST_CHECK_EQUAL(result.size(), answer.size());
        for (unsigned int ix = 0; ix < result.size(); ++ix)
            BOOST_CHECK_CLOSE(result[ix], answer[ix], 0.0001);
    }
    {
        std::vector<float> result =
            vertex_betweenness_centrality_batch_alt(graph, seed_set);

        BOOST_CHECK_EQUAL(result.size(), answer.size());
        for (unsigned int ix = 0; ix < result.size(); ++ix)
            BOOST_CHECK_CLOSE(result[ix], answer[ix], 0.0001);
    }
    {
        std::vector<float> result =
            vertex_betweenness_centrality_batch_alt_trans(graph, seed_set);

        BOOST_CHECK_EQUAL(result.size(), answer.size());
        for (unsigned int ix = 0; ix < result.size(); ++ix)
            BOOST_CHECK_CLOSE(result[ix], answer[ix], 0.0001);
    }
    {
        std::vector<float> result =
            vertex_betweenness_centrality_batch_alt_trans_v2(graph, seed_set);

        BOOST_CHECK_EQUAL(result.size(), answer.size());
        for (unsigned int ix = 0; ix < result.size(); ++ix)
            BOOST_CHECK_CLOSE(result[ix], answer[ix], 0.0001);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bc_test_vertex_betweenness_centrality)
{
    Matrix<double, DirectedMatrixTag> betweenness(8,8);
    betweenness.build(br.begin(), bc.begin(), bv.begin(), bv.size());
    std::vector<double> result = vertex_betweenness_centrality(betweenness);
    std::vector<double> answer = {0.0, 4.0/3, 4.0/3, 4.0/3, 12.0, 2.5, 2.5, 0.0};

    BOOST_CHECK_EQUAL(result.size(), answer.size());
    for (unsigned int ix = 0; ix < result.size(); ++ix)
        BOOST_CHECK_CLOSE(result[ix], answer[ix], 0.0001);
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(bc_test_edge_betweenness_centrality_2)
{
    IndexType const NUM_NODES = 11;
    std::vector<IndexType> row_indices = {0, 0, 0, 0, 1, 1, 2, 3, 3, 4, 4, 5,
                                          5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 9, 10};
    std::vector<IndexType> col_indices = {1, 2, 3, 4, 0, 9, 0, 0, 9, 0, 5, 4,
                                          6, 7, 5, 8, 5, 8, 6, 7, 1, 3, 10, 9};
    std::vector<double> values(row_indices.size(), 1.);

    Matrix<double> graph(NUM_NODES, NUM_NODES);
    graph.build(row_indices.begin(), col_indices.begin(),
                values.begin(), values.size());

    IndexArrayType seed_set={0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    std::vector<float> result =
        vertex_betweenness_centrality_batch_alt_trans_v2(graph, seed_set);
    std::cout << "Milcom 2016 Graph:";
    for (auto a : result)
        std::cout << " " << a;
    std::cout << std::endl;
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(bc_test_edge_betweenness_centrality)
{
    Matrix<double, DirectedMatrixTag> betweenness(8,8);
    Matrix<double, DirectedMatrixTag> result(8,8);
    //Matrix<double, DirectedMatrixTag> answer ({
    //                            {0,  6,  6,  6,  0,  0,  0,  0},
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

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bc_test_vertex_betweennes_centrality_batch_old)
{
    Matrix<double, DirectedMatrixTag> betweenness(8,8);
    betweenness.build(br.begin(), bc.begin(), bv.begin(), bv.size());

    //IndexArrayType seed_set={0, 1, 2, 3, 4, 5, 6, 7};
    IndexArrayType seed_set={0};
    std::vector<double> answer = {0.0, 4.0/3, 4.0/3, 4.0/3, 3.0, 0.5, 0.5, 0.0};

    std::vector<double> result =
        vertex_betweenness_centrality_batch_old(betweenness, seed_set);

    BOOST_CHECK_EQUAL(result.size(), answer.size());
    for (unsigned int ix = 0; ix < result.size(); ++ix)
        BOOST_CHECK_CLOSE(result[ix], answer[ix], 0.0001);

    //==========
    IndexArrayType seed_set3={3};
    std::vector<double> answer3 = {0.0, 0.0, 0.0, 0.0, 3.0, 0.5, 0.5, 0.0};

    std::vector<double> result3 =
        vertex_betweenness_centrality_batch_old(betweenness, seed_set3);

    BOOST_CHECK_EQUAL(result3.size(), answer3.size());
    for (unsigned int ix = 0; ix < result3.size(); ++ix)
        BOOST_CHECK_CLOSE(result3[ix], answer3[ix], 0.0001);

    //==========
    IndexArrayType seed_set03={0,3};

    std::vector<double> result03 =
        vertex_betweenness_centrality_batch_old(betweenness, seed_set03);

    BOOST_CHECK_EQUAL(result03.size(), answer3.size());
    for (unsigned int ix = 0; ix < result3.size(); ++ix)
        BOOST_CHECK_CLOSE(result03[ix], answer[ix] + answer3[ix], 0.0001);

    //==========
    IndexArrayType seed_set_all={0,1,2,3,4,5,6,7};
    std::vector<double> answer_all = {0.0, 4.0/3, 4.0/3, 4.0/3, 12.0, 2.5, 2.5, 0.0};

    std::vector<double> result_all =
        vertex_betweenness_centrality_batch_old(betweenness, seed_set_all);

    BOOST_CHECK_EQUAL(result_all.size(), answer_all.size());
    for (unsigned int ix = 0; ix < result_all.size(); ++ix)
        BOOST_CHECK_CLOSE(result_all[ix], answer_all[ix], 0.0001);

}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bc_test_vertex_betweennes_centrality_batch)
{
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
    // Trying the same graph where each node has self loop.     vvvvvvvvvvvvvvv
    std::vector<double> br1={0, 0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 6,0,1,2,3,4,5,6,7};
    std::vector<double> bc1={1, 2, 3, 2, 4, 4, 2, 4, 5, 6, 7, 7,0,1,2,3,4,5,6,7};
    std::vector<double> bv1(br1.size(), 1);

    Matrix<double, DirectedMatrixTag> betweenness(8,8);
    betweenness.build(br1.begin(), bc1.begin(), bv1.begin(), bv1.size());

    //IndexArrayType seed_set={0, 1, 2, 3, 4, 5, 6, 7};
    IndexArrayType seed_set={0};
    std::vector<float> answer = {0.0, 4.0/3, 4.0/3, 4.0/3, 3.0, 0.5, 0.5, 0.0};

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
    //std::cout << "======================= BEGIN0 ==========================\n";
    Matrix<double, DirectedMatrixTag> betweenness(8,8);
    betweenness.build(br.begin(), bc.begin(), bv.begin(), bv.size());

    //IndexArrayType seed_set={0, 1, 2, 3, 4, 5, 6, 7};
    IndexArrayType seed_set={0};
    std::vector<float> answer = {0.0, 4.0/3, 4.0/3, 4.0/3, 3.0, 0.5, 0.5, 0.0};

    std::vector<float> result =
        vertex_betweenness_centrality_batch_alt_trans(betweenness, seed_set);

    BOOST_CHECK_EQUAL(result.size(), answer.size());
    for (unsigned int ix = 0; ix < result.size(); ++ix)
        BOOST_CHECK_CLOSE(result[ix], answer[ix], 0.0001);

    //==========
    //std::cout << "======================= BEGIN3 ==========================\n";
    IndexArrayType seed_set3={3};
    std::vector<float> answer3 = {0.0, 0.0, 0.0, 0.0, 3.0, 0.5, 0.5, 0.0};

    std::vector<float> result3 =
        vertex_betweenness_centrality_batch_alt_trans(betweenness, seed_set3);

    BOOST_CHECK_EQUAL(result3.size(), answer3.size());
    for (unsigned int ix = 0; ix < result3.size(); ++ix)
        BOOST_CHECK_CLOSE(result3[ix], answer3[ix], 0.0001);

    //==========
    //std::cout << "======================= BEGIN0,3 ==========================\n";
    IndexArrayType seed_set03={0,3};

    std::vector<float> result03 =
        vertex_betweenness_centrality_batch_alt_trans(betweenness, seed_set03);

    BOOST_CHECK_EQUAL(result03.size(), answer3.size());
    for (unsigned int ix = 0; ix < result3.size(); ++ix)
        BOOST_CHECK_CLOSE(result03[ix], answer[ix] + answer3[ix], 0.0001);

    //==========
    //std::cout << "======================= BEGIN0-7 ==========================\n";
    IndexArrayType seed_set_all={0,1,2,3,4,5,6,7};
    std::vector<double> answer_all = {0.0, 4.0/3, 4.0/3, 4.0/3, 12.0, 2.5, 2.5, 0.0};

    std::vector<float> result_all =
        vertex_betweenness_centrality_batch_alt_trans(betweenness, seed_set_all);

    BOOST_CHECK_EQUAL(result_all.size(), answer_all.size());
    for (unsigned int ix = 0; ix < result_all.size(); ++ix)
        BOOST_CHECK_CLOSE(result_all[ix], answer_all[ix], 0.0001);
    //std::cout << "=======================  END  ==========================\n";
}

//****************************************************************************
//BOOST_AUTO_TEST_CASE(bc_test_vertex_betweennes_centrality_batch_alt_trans_v2)
BOOST_AUTO_TEST_CASE(bc_test_batch_alt_trans_v2)
{
    Matrix<double, DirectedMatrixTag> betweenness(8,8);
    betweenness.build(br.begin(), bc.begin(), bv.begin(), bv.size());

    //IndexArrayType seed_set={0, 1, 2, 3, 4, 5, 6, 7};
    IndexArrayType seed_set={0};
    std::vector<float> answer = {0.0, 4.0/3, 4.0/3, 4.0/3, 3.0, 0.5, 0.5, 0.0};

    std::vector<float> result = vertex_betweenness_centrality_batch_alt_trans_v2(
            betweenness, seed_set);

    BOOST_CHECK_EQUAL(result.size(), answer.size());
    for (unsigned int ix = 0; ix < result.size(); ++ix)
        BOOST_CHECK_CLOSE(result[ix], answer[ix], 0.0001);

    //==========
    IndexArrayType seed_set3={3};
    std::vector<float> answer3 = {0.0, 0.0, 0.0, 0.0, 3.0, 0.5, 0.5, 0.0};

    std::vector<float> result3 =
        vertex_betweenness_centrality_batch_alt_trans_v2(betweenness, seed_set3);

    BOOST_CHECK_EQUAL(result3.size(), answer3.size());
    for (unsigned int ix = 0; ix < result3.size(); ++ix)
        BOOST_CHECK_CLOSE(result3[ix], answer3[ix], 0.0001);

    //==========
    IndexArrayType seed_set03={0,3};

    std::vector<float> result03 =
        vertex_betweenness_centrality_batch_alt_trans_v2(betweenness, seed_set03);

    BOOST_CHECK_EQUAL(result03.size(), answer3.size());
    for (unsigned int ix = 0; ix < result3.size(); ++ix)
        BOOST_CHECK_CLOSE(result03[ix], answer[ix] + answer3[ix], 0.0001);

    //==========
    IndexArrayType seed_set_all={0,1,2,3,4,5,6,7};
    std::vector<double> answer_all = {0.0, 4.0/3, 4.0/3, 4.0/3, 12.0, 2.5, 2.5, 0.0};

    std::vector<float> result_all =
        vertex_betweenness_centrality_batch_alt_trans_v2(betweenness, seed_set_all);

    BOOST_CHECK_EQUAL(result_all.size(), answer_all.size());
    for (unsigned int ix = 0; ix < result_all.size(); ++ix)
        BOOST_CHECK_CLOSE(result_all[ix], answer_all[ix], 0.0001);
}


BOOST_AUTO_TEST_SUITE_END()
