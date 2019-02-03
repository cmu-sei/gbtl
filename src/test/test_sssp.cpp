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
#include <limits>

#include <algorithms/sssp.hpp>
#include <graphblas/graphblas.hpp>

using namespace algorithms;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE sssp_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

//****************************************************************************
template <typename T>
GraphBLAS::Matrix<T> get_tn_answer()
{
    std::vector<GraphBLAS::IndexType> rows = {
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
        2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
        5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8};
    std::vector<GraphBLAS::IndexType> cols = {
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
    //    {{0, 2, 3, 1, 2, 4, 2, , 3},
    //     {2, 0, 2, 1, 2, 3, 1, , 3},
    //     {3, 2, 0, 2, 1, 1, 1, , 1},
    //     {1, 1, 2, 0, 1, 3, 1, , 2},
    //     {2, 2, 1, 1, 0, 2, 2, , 1},
    //     {4, 3, 1, 3, 2, 0, 2, , 2},
    //     {2, 1, 1, 1, 2, 2, 0, , 2},
    //     { ,  ,  ,  ,  ,  ,  , 0, },
    //     {3, 3, 1, 2, 1, 2, 2, , 0}};
    GraphBLAS::Matrix<T> temp(9,9);
    temp.build(rows.begin(), cols.begin(), vals.begin(), rows.size());
    return temp;
}

//****************************************************************************
template <typename T>
GraphBLAS::Matrix<T> get_gilbert_answer()
{
    //std::vector<std::vector<T> > G_gilbert_answer_dense =
    //    {{  0,   1,   2,   1,   2,   3,   2},
    //     {  3,   0,   2,   2,   1,   2,   1},
    //     {   ,    ,   0,    ,    ,   1,    },
    //     {  1,   2,   1,   0,   3,   2,   3},
    //     {   ,    ,   2,    ,   0,   1,    },
    //     {   ,    ,   1,    ,    ,   0,    },
    //     {  2,   3,   1,   1,   1,   2,   0}};
    //return Matrix<T>(G_gilbert_answer_dense, );
    std::vector<GraphBLAS::IndexType> rows = {
        0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 5, 5, 6, 6, 6, 6, 6, 6, 6
    };
    std::vector<GraphBLAS::IndexType> cols = {
        0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 2, 5, 0, 1, 2, 3, 4, 5, 6,
        2, 4, 5, 2, 5, 0, 1, 2, 3, 4, 5, 6
    };
    std::vector<T> vals = {
        0,   1,   2,   1,   2,   3,   2,   3, 0,   2,   2,   1,   2,
        1, 0,   1,   1,   2,   1, 0,   3,   2,   3,   2, 0,   1,
        1, 0,   2,   3,   1,   1,   1,   2, 0
    };
    GraphBLAS::Matrix<T> temp(7,7);
    temp.build(rows.begin(), cols.begin(), vals.begin(), rows.size());
    return temp;
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(sssp_basic_mincost_one_root)
{
    GraphBLAS::IndexType const NUM_NODES(4);
    GraphBLAS::IndexType const start_index(0);

    GraphBLAS::IndexArrayType i = {0, 0, 1, 2};
    GraphBLAS::IndexArrayType j = {1, 3, 2, 3};
    std::vector<double>       v = {1, 5, 1, 1};
    GraphBLAS::Matrix<double> G(NUM_NODES, NUM_NODES);
    G.build(i, j, v);

    GraphBLAS::Vector<double> path(NUM_NODES);
    path.setElement(start_index, 0);

    sssp(G, path);
    //GraphBLAS::print_vector(std::cout, path, "MINCOST path");
    BOOST_CHECK_EQUAL(path.extractElement(0), 0);
    BOOST_CHECK_EQUAL(path.extractElement(1), 1);
    BOOST_CHECK_EQUAL(path.extractElement(2), 2);
    BOOST_CHECK_EQUAL(path.extractElement(3), 3);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sssp_basic_mincost2_one_root)
{
    GraphBLAS::IndexType const NUM_NODES(4);
    GraphBLAS::IndexType const start_index(0);

    GraphBLAS::IndexArrayType i = {0, 0, 1, 2};
    GraphBLAS::IndexArrayType j = {1, 3, 2, 3};
    std::vector<double>       v = {1, 2, 1, 1};
    GraphBLAS::Matrix<double> G(NUM_NODES, NUM_NODES);
    G.build(i, j, v);

    GraphBLAS::Vector<double> path(NUM_NODES);
    path.setElement(start_index, 0);

    sssp(G, path);
    //GraphBLAS::print_vector(std::cout, path, "MINCOST path");
    BOOST_CHECK_EQUAL(path.extractElement(0), 0);
    BOOST_CHECK_EQUAL(path.extractElement(1), 1);
    BOOST_CHECK_EQUAL(path.extractElement(2), 2);
    BOOST_CHECK_EQUAL(path.extractElement(3), 2);
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(sssp_basic_double_one_root)
{
    GraphBLAS::IndexType const NUM_NODES(9);
    GraphBLAS::IndexType const start_index(3);

    GraphBLAS::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    GraphBLAS::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<double>       v(i.size(), 1);
    GraphBLAS::Matrix<double> G_tn(NUM_NODES, NUM_NODES);
    G_tn.build(i, j, v);

    GraphBLAS::Vector<double> path(NUM_NODES);
    path.setElement(start_index, 0);

    sssp(G_tn, path);

    auto G_tn_answer(get_tn_answer<double>());

    for (GraphBLAS::IndexType ix = 0; ix < NUM_NODES; ++ix)
    {
        if (G_tn_answer.hasElement(start_index, ix))
        {
            BOOST_CHECK_EQUAL(path.hasElement(ix), true);
            BOOST_CHECK_EQUAL(path.extractElement(ix),
                              G_tn_answer.extractElement(start_index, ix));
        }
        else
        {
            BOOST_CHECK_EQUAL(path.hasElement(ix), false);
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sssp_basic_double_batch)
{
    GraphBLAS::IndexType const NUM_NODES(9);
    GraphBLAS::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    GraphBLAS::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<double>       v(i.size(), 1);
    GraphBLAS::Matrix<double> G_tn(NUM_NODES, NUM_NODES);
    G_tn.build(i, j, v);

    // Perform for all roots simultaneously
    auto paths =
        GraphBLAS::scaled_identity<GraphBLAS::Matrix<double> >(NUM_NODES, 0);
    batch_sssp(G_tn, paths);

    auto G_tn_answer(get_tn_answer<double>());

    //print_matrix(std::cout, paths, "result");
    //print_matrix(std::cout, G_tn_answer, "correct answer");
    BOOST_CHECK_EQUAL(paths, G_tn_answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sssp_basic_uint_batch)
{
    GraphBLAS::IndexType const NUM_NODES(9);
    GraphBLAS::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    GraphBLAS::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<unsigned int> v(i.size(), 1);
    GraphBLAS::Matrix<unsigned int> G_tn(NUM_NODES, NUM_NODES);
    G_tn.build(i, j, v);

    // Perform for all roots simultaneously
    auto paths =
        GraphBLAS::scaled_identity<GraphBLAS::Matrix<unsigned int> >(NUM_NODES, 0);
    batch_sssp(G_tn, paths);

    auto G_tn_answer(get_tn_answer<unsigned int>());

    BOOST_CHECK_EQUAL(paths, G_tn_answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sssp_gilbert_double_batch)
{
    GraphBLAS::IndexType const NUM_NODES(7);
    GraphBLAS::IndexArrayType i = {0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6};
    GraphBLAS::IndexArrayType j = {1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4};
    std::vector<double>       v(i.size(), 1);
    GraphBLAS::Matrix<double> G_gilbert(NUM_NODES, NUM_NODES);
    G_gilbert.build(i, j, v);

    // Perform for all roots simultaneously
    auto paths =
        GraphBLAS::scaled_identity<GraphBLAS::Matrix<double> >(NUM_NODES, 0);
    batch_sssp(G_gilbert, paths);

    auto G_gilbert_answer(get_gilbert_answer<double>());

    BOOST_CHECK_EQUAL(paths, G_gilbert_answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sssp_gilbert_uint)
{
    GraphBLAS::IndexType const NUM_NODES(7);
    GraphBLAS::IndexArrayType i = {0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6};
    GraphBLAS::IndexArrayType j = {1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4};
    std::vector<unsigned int> v(i.size(), 1);
    GraphBLAS::Matrix<unsigned int> G_gilbert(NUM_NODES, NUM_NODES);
    G_gilbert.build(i, j, v);

    // Perform for all roots simultaneously
    auto paths =
        GraphBLAS::scaled_identity<GraphBLAS::Matrix<unsigned int> >(NUM_NODES, 0);
    batch_sssp(G_gilbert, paths);

    auto G_gilbert_answer(get_gilbert_answer<unsigned int>());

    BOOST_CHECK_EQUAL(paths, G_gilbert_answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(new_sssp_gilbert_uint)
{
    unsigned int const INF = 666666;
    // The correct answer for all starting points (in order)
    std::vector<std::vector<unsigned int> > G_gilbert_answer_dense =
        {{  0,   1,   2,   1,   2,   3,   2},
         {  3,   0,   2,   2,   1,   2,   1},
         {INF, INF,   0, INF, INF,   1, INF},
         {  1,   2,   1,   0,   3,   2,   3},
         {INF, INF,   2, INF,   0,   1, INF},
         {INF, INF,   1, INF, INF,   0, INF},
         {  2,   3,   1,   1,   1,   2,   0}};
    GraphBLAS::Matrix<unsigned int>
        G_gilbert_answer(G_gilbert_answer_dense, INF);

    GraphBLAS::IndexType const NUM_NODES(7);
    GraphBLAS::IndexArrayType i = {0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6};
    GraphBLAS::IndexArrayType j = {1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4};
    std::vector<unsigned int> v(i.size(), 1);
    GraphBLAS::Matrix<unsigned int> G_gilbert(NUM_NODES, NUM_NODES);
    G_gilbert.build(i.begin(), j.begin(), v.begin(), i.size());

    //auto identity_7x7 =
    //    GraphBLAS::identity<GraphBLAS::Matrix<unsigned int> >(
    //        NUM_NODES, INF, 0);
    GraphBLAS::IndexArrayType ii = {0, 1, 2, 3, 4, 5, 6};
    std::vector<unsigned int> vi(ii.size(), 0);
    GraphBLAS::Matrix<unsigned int> G_gilbert_res(NUM_NODES, NUM_NODES);
    G_gilbert_res.build(ii.begin(), ii.begin(), vi.begin(), ii.size());

    batch_sssp(G_gilbert, G_gilbert_res);

    BOOST_CHECK_EQUAL(G_gilbert_res, G_gilbert_answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_sssp_delta_step_gilbert_uint)
{
    unsigned int const INF = 666666;

    // The correct answer for all starting points (in order by row)
    //std::vector<std::vector<unsigned int> > G_gilbert_answer_dense =
    //    {{  0,   1,   2,   1,   2,   3,   2},
    //     {  3,   0,   2,   2,   1,   2,   1},
    //     {INF, INF,   0, INF, INF,   1, INF},
    //     {  1,   2,   1,   0,   3,   2,   3},
    //     {INF, INF,   2, INF,   0,   1, INF},
    //     {INF, INF,   1, INF, INF,   0, INF},
    //     {  2,   3,   1,   1,   1,   2,   0}};

    GraphBLAS::IndexType const SOURCE = 3;
    std::vector<unsigned int> dense_ans3 =  {3,   4,   1,   0,   5,   2,   7};
    GraphBLAS::Vector<unsigned int> answer3(dense_ans3);

    GraphBLAS::IndexType const NUM_NODES(7);
    GraphBLAS::IndexArrayType i = {0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6};
    GraphBLAS::IndexArrayType j = {1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4};
    std::vector<unsigned int> v = {1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3};
    GraphBLAS::Matrix<unsigned int> G_gilbert(NUM_NODES, NUM_NODES);
    G_gilbert.build(i.begin(), j.begin(), v.begin(), i.size());

    //auto identity_7x7 =
    //    GraphBLAS::identity<GraphBLAS::Matrix<unsigned int> >(
    //        NUM_NODES, INF, 0);
    GraphBLAS::Vector<unsigned int> result(NUM_NODES);
    sssp_delta_step(G_gilbert, 2U, SOURCE, result);

    BOOST_CHECK_EQUAL(result, answer3);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_sssp_delta_step_marcin_uint)
{
    unsigned int const INF = 666666;

    //  - 7 1 - - - - -
    //  7 - - - - 1 1 -
    //  1 - - 1 - - - -
    //  - - 1 - 1 - - -
    //  - - - 1 - 1 - -
    //  - 1 - - 1 - - 1
    //  - 1 - - - - - 1
    //  - - - - - 1 1 -

    // The correct answer for source vertex 0
    //
    //    {0, 5, 1, 2, 3, 4, 6, 5};

    GraphBLAS::IndexType const SOURCE = 0;
    std::vector<unsigned int> dense_ans = {0, 5, 1, 2, 3, 4, 6, 5};
    GraphBLAS::Vector<unsigned int> answer(dense_ans);

    GraphBLAS::IndexType const NUM_NODES(8);
    GraphBLAS::IndexArrayType i = {0,0,1,1,1,2,2,3,3,4,4,5,5,5,6,6,7,7};
    GraphBLAS::IndexArrayType j = {1,2,0,5,6,0,3,2,4,3,5,1,4,7,1,7,5,6};
    std::vector<unsigned int> v = {7,1,7,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    GraphBLAS::Matrix<unsigned int> G(NUM_NODES, NUM_NODES);
    G.build(i.begin(), j.begin(), v.begin(), i.size());

    std::cout << "\n*************************************\nStarting delta = 2" << std::endl;
    GraphBLAS::Vector<unsigned int> result2(NUM_NODES);
    sssp_delta_step(G, 2U, SOURCE, result2);
    BOOST_CHECK_EQUAL(result2, answer);
    std::cout << "\n*************************************\nStarting delta = 7" << std::endl;
    GraphBLAS::Vector<unsigned int> result7(NUM_NODES);
    sssp_delta_step(G, 7U, SOURCE, result7);
    BOOST_CHECK_EQUAL(result7, answer);
    std::cout << "\n*************************************\nStarting delta = 8" << std::endl;
    GraphBLAS::Vector<unsigned int> result8(NUM_NODES);
    sssp_delta_step(G, 8U, SOURCE, result8);
    BOOST_CHECK_EQUAL(result8, answer);
    std::cout << "\n*************************************\nStarting delta = 9" << std::endl;
    GraphBLAS::Vector<unsigned int> result9(NUM_NODES);
    sssp_delta_step(G, 9U, SOURCE, result9);
    BOOST_CHECK_EQUAL(result9, answer);
    std::cout << "\n*************************************\nStarting delta = 10" << std::endl;
    GraphBLAS::Vector<unsigned int> result10(NUM_NODES);
    sssp_delta_step(G, 10U, SOURCE, result10);
    BOOST_CHECK_EQUAL(result10, answer);
}

BOOST_AUTO_TEST_SUITE_END()
