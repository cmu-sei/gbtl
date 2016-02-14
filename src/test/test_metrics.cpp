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

#include <algorithms/metrics.hpp>
#include <graphblas/graphblas.hpp>

using namespace graphblas;
using namespace algorithms;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE metrics_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(metrics_suite)

static double const INF = std::numeric_limits<double>::max();

/// @todo Switch to using buildmatrix interface.
static Matrix<double, DirectedMatrixTag> G_tn(
    {{0, 0, 0, 1, 0, 0, 0, 0, 0},
     {0, 0, 0, 1, 0, 0, 1, 0, 0},
     {0, 0, 0, 0, 1, 1, 1, 0, 1},
     {1, 1, 0, 0, 1, 0, 1, 0, 0},
     {0, 0, 1, 1, 0, 0, 0, 0, 1},
     {0, 0, 1, 0, 0, 0, 0, 0, 0},
     {0, 1, 1, 1, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 0, 0, 0},
     {0, 0, 1, 0, 1, 0, 0, 0, 0}});


static Matrix<double, DirectedMatrixTag> G_tn_answer(
    {{2, 2, 3, 1, 2, 4, 2, INF, 3},
     {2, 2, 2, 1, 2, 3, 1, INF, 3},
     {3, 2, 2, 2, 1, 1, 1, INF, 1},
     {1, 1, 2, 2, 1, 3, 1, INF, 2},
     {2, 2, 1, 1, 2, 2, 2, INF, 1},
     {4, 3, 1, 3, 2, 2, 2, INF, 2},
     {2, 1, 1, 1, 2, 2, 2, INF, 2},
     {INF, INF, INF, INF, INF, INF, INF, INF, INF},
     {3, 3, 1, 2, 1, 2, 2, INF, 2}},
    INF);

static Matrix<double, DirectedMatrixTag> test5x5(
    {{INF,   1,   1, INF, INF},
     {  1, INF,   1, INF, INF},
     {  1,   1, INF,   1,   1},
     {INF, INF,   1, INF,   1},
     {INF, INF,   1,   1, INF}},
    INF);

static Matrix<double, DirectedMatrixTag> betweenness(
    {{0, 1, 1, 1, 0, 0, 0, 0},
     {0, 0, 1, 0, 1, 0, 0, 0},
     {0, 0, 0, 0, 1, 0, 0, 0},
     {0, 0, 1, 0, 1, 0, 0, 0},
     {0, 0, 0, 0, 0, 1, 1, 0},
     {0, 0, 0, 0, 0, 0, 0, 1},
     {0, 0, 0, 0, 0, 0, 0, 1},
     {0, 0, 0, 0, 0, 0, 0, 0}});

BOOST_AUTO_TEST_CASE(metrics_test_vertex_count)
{
    IndexType result = vertex_count(G_tn);
    BOOST_CHECK_EQUAL(result, 9);
}

BOOST_AUTO_TEST_CASE(metrics_test_edge_count)
{
    IndexType result = edge_count(G_tn);
    BOOST_CHECK_EQUAL(result, 20);
}

BOOST_AUTO_TEST_CASE(metrics_test_in_degree)
{
    IndexType result = vertex_in_degree(G_tn, 1);
    BOOST_CHECK_EQUAL(result, 2);
}

BOOST_AUTO_TEST_CASE(metrics_test_out_degree)
{
    IndexType result = vertex_out_degree(G_tn, 2);
    BOOST_CHECK_EQUAL(result, 4);
}

BOOST_AUTO_TEST_CASE(metrics_test_vertex_degree)
{
    IndexType result = vertex_degree(G_tn, 0);
    BOOST_CHECK_EQUAL(result, 2);
}

BOOST_AUTO_TEST_CASE(metrics_test_graph_distance)
{
    Matrix<double, DirectedMatrixTag> result(5, 5, INF);

    Matrix<double, DirectedMatrixTag> answer(
        {{INF, INF, INF, INF, INF},
         {INF, INF, INF, INF, INF},
         {1, 1, 0, 1, 1},
         {INF, INF, INF, INF, INF},
         {INF, INF, INF, INF, INF}},
        INF);

    graph_distance(test5x5, 2, result);
    BOOST_CHECK_EQUAL(result, answer);
}

BOOST_AUTO_TEST_CASE(metrics_test_graph_distance_matrix)
{
    Matrix<double, DirectedMatrixTag> result(5,5,INF);

    Matrix<double, DirectedMatrixTag> answer(
        {{0, 1, 1, 2, 2},
         {1, 0, 1, 2, 2},
         {1, 1, 0, 1, 1},
         {2, 2, 1, 0, 1},
         {2, 2, 1, 1, 0}},
        INF);

    graph_distance_matrix(test5x5, result);
    BOOST_CHECK_EQUAL(result, answer);
}

BOOST_AUTO_TEST_CASE(metrics_test_graph_eccentricity)
{
    double result = vertex_eccentricity(test5x5, 2);
    BOOST_CHECK_EQUAL(result, 1);
}

BOOST_AUTO_TEST_CASE(metrics_test_graph_radius)
{
    double result = graph_radius(test5x5);
    BOOST_CHECK_EQUAL(result, 1);
}

BOOST_AUTO_TEST_CASE(metrics_test_graph_diameter)
{
    double result = graph_diameter(test5x5);
    BOOST_CHECK_EQUAL(result, 2);
}

BOOST_AUTO_TEST_CASE(metrics_test_degree_centrality)
{
    IndexType result = degree_centrality(test5x5, 0);
    BOOST_CHECK_EQUAL(result, 4);
}

BOOST_AUTO_TEST_CASE(metrics_test_closeness_centrality)
{
    double result = closeness_centrality(test5x5, 2);
    BOOST_CHECK_EQUAL(result, 4);
}


BOOST_AUTO_TEST_CASE(metrics_test_vertex_betweenness_centrality)
{
    std::vector<double> result = vertex_betweenness_centrality(betweenness);
    std::vector<double> answer = {0.0, 4.0/3, 4.0/3, 4.0/3, 12.0, 2.5, 2.5, 0.0};

    BOOST_CHECK_EQUAL_COLLECTIONS(result.begin(), result.end(), answer.begin(), answer.end());
}


BOOST_AUTO_TEST_CASE(metrics_test_edge_betweenness_centrality)
{
    Matrix<double, DirectedMatrixTag> result(8,8);
    Matrix<double, DirectedMatrixTag> answer ({{0,  6,  6,  6,  0,  0,  0,  0},
                                {0,  0,  1,  0, 10,  0,  0,  0},
                                {0,  0,  0,  0, 10,  0,  0,  0},
                                {0,  0,  1,  0, 10,  0,  0,  0},
                                {0,  0,  0,  0,  0, 14, 14,  0},
                                {0,  0,  0,  0,  0,  0,  0,  8},
                                {0,  0,  0,  0,  0,  0,  0,  8},
                                {0,  0,  0,  0,  0,  0,  0,  0}});
    result = edge_betweenness_centrality(betweenness);
    //std::cout << result << std::endl;
    BOOST_CHECK_EQUAL(result, answer);
}

BOOST_AUTO_TEST_CASE(metrics_test_triangle_counting)
{
    Matrix<double, DirectedMatrixTag> testtriangle({{0,1,1,1,0},
                            {1,0,1,0,1},
                            {1,1,0,1,1},
                            {1,0,1,0,1},
                            {0,1,1,1,0}});

    IndexType result = triangle_count(testtriangle);
    BOOST_CHECK_EQUAL(result, 4);
}


BOOST_AUTO_TEST_SUITE_END()
