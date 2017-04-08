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
#include <limits>

#include <algorithms/sssp.hpp>
#include <graphblas/graphblas.hpp>
#include <graphblas/linalg_utils.hpp>

using namespace algorithms;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE sssp_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(sssp_suite)

//****************************************************************************
template <typename T>
graphblas::Matrix<T, graphblas::DirectedMatrixTag> get_tn_answer(T const &INF)
{
    std::vector<graphblas::IndexType> rows = {
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
        2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
        5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8};
    std::vector<graphblas::IndexType> cols = {
        0, 1, 2, 3, 4, 5, 6, 8, 0, 1, 2, 3, 4, 5, 6, 8, 0, 1, 2, 3, 4, 5, 6,
        8, 0, 1, 2, 3, 4, 5, 6, 8, 0, 1, 2, 3, 4, 5, 6, 8, 0, 1, 2, 3, 4, 5,
        6, 8, 0, 1, 2, 3, 4, 5, 6, 8, 7, 0, 1, 2, 3, 4, 5, 6, 8
    };

    std::vector<T> vals = {
        0,   2,   3,   1,   2,   4,   2,   3,   2, 0,   2,   1,   2,
        3,   1,   3,   3,   2, 0,   2,   1,   1,   1,   1,   1,   1,
        2, 0,   1,   3,   1,   2,   2,   2,   1,   1, 0,   2,   2,
        1,   4,   3,   1,   3,   2, 0,   2,   2,   2,   1,   1,   1,
        2,   2, 0,   2, 0,   3,   3,   1,   2,   1,   2,   2, 0};

    //std::vector<std::vector<T> > G_tn_answer_dense =
    //    {{0, 2, 3, 1, 2, 4, 2, INF, 3},
    //     {2, 0, 2, 1, 2, 3, 1, INF, 3},
    //     {3, 2, 0, 2, 1, 1, 1, INF, 1},
    //     {1, 1, 2, 0, 1, 3, 1, INF, 2},
    //     {2, 2, 1, 1, 0, 2, 2, INF, 1},
    //     {4, 3, 1, 3, 2, 0, 2, INF, 2},
    //     {2, 1, 1, 1, 2, 2, 0, INF, 2},
    //     {INF, INF, INF, INF, INF, INF, INF, 0, INF},
    //     {3, 3, 1, 2, 1, 2, 2, INF, 0}};
    graphblas::Matrix<T, graphblas::DirectedMatrixTag> temp(9,9, INF);
    graphblas::buildmatrix(temp, rows.begin(), cols.begin(), vals.begin(), rows.size());
    return temp;
}

//****************************************************************************
template <typename T>
graphblas::Matrix<T, graphblas::DirectedMatrixTag> get_gilbert_answer(T const &INF)
{
    //std::vector<std::vector<T> > G_gilbert_answer_dense =
    //    {{  0,   1,   2,   1,   2,   3,   2},
    //     {  3,   0,   2,   2,   1,   2,   1},
    //     {INF, INF,   0, INF, INF,   1, INF},
    //     {  1,   2,   1,   0,   3,   2,   3},
    //     {INF, INF,   2, INF,   0,   1, INF},
    //     {INF, INF,   1, INF, INF,   0, INF},
    //     {  2,   3,   1,   1,   1,   2,   0}};
    //return Matrix<T, DirectedMatrixTag>(G_gilbert_answer_dense, INF);
    std::vector<graphblas::IndexType> rows = {
        0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 5, 5, 6, 6, 6, 6, 6, 6, 6
    };
    std::vector<graphblas::IndexType> cols = {
        0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 2, 5, 0, 1, 2, 3, 4, 5, 6,
        2, 4, 5, 2, 5, 0, 1, 2, 3, 4, 5, 6
    };
    std::vector<T> vals = {
        0,   1,   2,   1,   2,   3,   2,   3, 0,   2,   2,   1,   2,
        1, 0,   1,   1,   2,   1, 0,   3,   2,   3,   2, 0,   1,
        1, 0,   2,   3,   1,   1,   1,   2, 0
    };
    graphblas::Matrix<T, graphblas::DirectedMatrixTag> temp(7,7, INF);
    graphblas::buildmatrix(temp, rows.begin(), cols.begin(), vals.begin(), rows.size());
    return temp;
}

#if 0

//****************************************************************************
BOOST_AUTO_TEST_CASE(sssp_basic_double_one_root)
{
    double const INF(std::numeric_limits<double>::max());
    graphblas::IndexType const NUM_NODES(9);
    graphblas::IndexType const start_index(3);

    graphblas::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    graphblas::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<double>       v(i.size(), 1);
    graphblas::Matrix<double, graphblas::DirectedMatrixTag> G_tn(NUM_NODES, NUM_NODES, INF);
    buildmatrix(G_tn, i, j, v);

    graphblas::Matrix<double, graphblas::DirectedMatrixTag> root(1, NUM_NODES, INF);
    std::vector<graphblas::IndexType> r={0}, c={start_index};
    std::vector<double> v_r={0};
    graphblas::buildmatrix(root, r.begin(), c.begin(), v_r.begin(), v_r.size());
    //root.set_value_at(0, start_index, 0);

    graphblas::Matrix<double, graphblas::DirectedMatrixTag> distance(1, NUM_NODES, INF);
    sssp(G_tn, root, distance);

    auto G_tn_answer(get_tn_answer(INF));

    for (graphblas::IndexType ix = 0; ix < NUM_NODES; ++ix)
    {
        BOOST_CHECK_EQUAL(distance.get_value_at(0, ix),
                          G_tn_answer.get_value_at(start_index, ix));
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sssp_basic_double)
{
    double const INF(std::numeric_limits<double>::max());

    graphblas::IndexType const NUM_NODES(9);
    graphblas::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    graphblas::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<double>       v(i.size(), 1);
    graphblas::Matrix<double, graphblas::DirectedMatrixTag> G_tn(NUM_NODES, NUM_NODES, INF);
    graphblas::buildmatrix(G_tn, i, j, v);

    auto identity_9x9 =
        graphblas::identity<graphblas::Matrix<double, graphblas::DirectedMatrixTag> >(
            NUM_NODES, INF, 0);

    graphblas::Matrix<double, graphblas::DirectedMatrixTag>
        G_tn_res(NUM_NODES, NUM_NODES, INF);
    sssp(G_tn, identity_9x9, G_tn_res);

    auto G_tn_answer(get_tn_answer(INF));

    BOOST_CHECK_EQUAL(G_tn_res, G_tn_answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sssp_basic_uint)
{
    unsigned int const INF(std::numeric_limits<unsigned int>::max());

    graphblas::IndexType const NUM_NODES(9);
    graphblas::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    graphblas::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<unsigned int> v(i.size(), 1);
    graphblas::Matrix<unsigned int, graphblas::DirectedMatrixTag>
        G_tn(NUM_NODES, NUM_NODES, INF);
    graphblas::buildmatrix(G_tn, i, j, v);

    auto identity_9x9 =
        graphblas::identity<graphblas::Matrix<unsigned int, graphblas::DirectedMatrixTag> >(
            NUM_NODES, INF, 0);

    graphblas::Matrix<unsigned int, graphblas::DirectedMatrixTag>
        G_tn_res(NUM_NODES, NUM_NODES, INF);
    sssp(G_tn, identity_9x9, G_tn_res);

    auto G_tn_answer(get_tn_answer(INF));

    BOOST_CHECK_EQUAL(G_tn_res, G_tn_answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sssp_gilbert_double)
{
    double const INF(std::numeric_limits<double>::max());

    graphblas::IndexType const NUM_NODES(7);
    graphblas::IndexArrayType i = {0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6};
    graphblas::IndexArrayType j = {1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4};
    std::vector<double>       v(i.size(), 1);
    graphblas::Matrix<double, graphblas::DirectedMatrixTag>
        G_gilbert(NUM_NODES, NUM_NODES, INF);
    graphblas::buildmatrix(G_gilbert, i, j, v);

    auto identity_7x7 =
        graphblas::identity<graphblas::Matrix<double, graphblas::DirectedMatrixTag> >(
            NUM_NODES, INF, 0);

    graphblas::Matrix<double, graphblas::DirectedMatrixTag>
        G_gilbert_res(NUM_NODES, NUM_NODES, INF);
    sssp(G_gilbert, identity_7x7, G_gilbert_res);

    auto G_gilbert_answer(get_gilbert_answer(INF));

    BOOST_CHECK_EQUAL(G_gilbert_res, G_gilbert_answer);
}

#endif

//****************************************************************************
BOOST_AUTO_TEST_CASE(sssp_gilbert_uint)
{
    unsigned int const INF(std::numeric_limits<unsigned int>::max());

    graphblas::IndexType const NUM_NODES(7);
    graphblas::IndexArrayType i = {0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6};
    graphblas::IndexArrayType j = {1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4};
    std::vector<unsigned int> v(i.size(), 1);
    graphblas::Matrix<unsigned int, graphblas::DirectedMatrixTag>
        G_gilbert(NUM_NODES, NUM_NODES, INF);
    graphblas::buildmatrix(G_gilbert, i, j, v);

    auto identity_7x7 =
        graphblas::identity<graphblas::Matrix<unsigned int, graphblas::DirectedMatrixTag> >(
            NUM_NODES, INF, 0);

    graphblas::Matrix<unsigned int, graphblas::DirectedMatrixTag> G_gilbert_res(
        NUM_NODES, NUM_NODES, INF);
    sssp(G_gilbert, identity_7x7, G_gilbert_res);

    auto G_gilbert_answer(get_gilbert_answer(INF));

    BOOST_CHECK_EQUAL(G_gilbert_res, G_gilbert_answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(new_sssp_gilbert_uint)
{
    std::cerr << "NEW SSSP TEST" << std::endl;

    unsigned int const INF(std::numeric_limits<unsigned int>::max());

    // The correct answer for all starting points (in order)
    std::vector<std::vector<unsigned int> > G_gilbert_answer_dense =
        {{  0,   1,   2,   1,   2,   3,   2},
         {  3,   0,   2,   2,   1,   2,   1},
         {INF, INF,   0, INF, INF,   1, INF},
         {  1,   2,   1,   0,   3,   2,   3},
         {INF, INF,   2, INF,   0,   1, INF},
         {INF, INF,   1, INF, INF,   0, INF},
         {  2,   3,   1,   1,   1,   2,   0}};
    GraphBLAS::Matrix<unsigned int, GraphBLAS::DirectedMatrixTag>
        G_gilbert_answer(G_gilbert_answer_dense, INF);

    GraphBLAS::IndexType const NUM_NODES(7);
    GraphBLAS::IndexArrayType i = {0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6};
    GraphBLAS::IndexArrayType j = {1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4};
    std::vector<unsigned int> v(i.size(), 1);
    GraphBLAS::Matrix<unsigned int, GraphBLAS::DirectedMatrixTag>
        G_gilbert(NUM_NODES, NUM_NODES);
    G_gilbert.build(i.begin(), j.begin(), v.begin(), i.size());

    //auto identity_7x7 =
    //    GraphBLAS::identity<GraphBLAS::Matrix<unsigned int, GraphBLAS::DirectedMatrixTag> >(
    //        NUM_NODES, INF, 0);
    GraphBLAS::IndexArrayType ii = {0, 1, 2, 3, 4, 5, 6};
    std::vector<unsigned int> vi(ii.size(), 1);
    GraphBLAS::Matrix<unsigned int, GraphBLAS::DirectedMatrixTag> G_gilbert_res(
        NUM_NODES, NUM_NODES);
    G_gilbert_res.build(ii.begin(), ii.begin(), vi.begin(), ii.size());

    GrB_sssp(G_gilbert, G_gilbert_res);

    BOOST_CHECK_EQUAL(G_gilbert_res, G_gilbert_answer);
}

BOOST_AUTO_TEST_SUITE_END()
