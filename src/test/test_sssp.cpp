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
grb::Matrix<T> get_tn_answer()
{
    std::vector<grb::IndexType> rows = {
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
        2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
        5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8};
    std::vector<grb::IndexType> cols = {
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
    grb::Matrix<T> temp(9,9);
    temp.build(rows.begin(), cols.begin(), vals.begin(), rows.size());
    return temp;
}

//****************************************************************************
template <typename T>
grb::Matrix<T> get_gilbert_answer()
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
    std::vector<grb::IndexType> rows = {
        0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 5, 5, 6, 6, 6, 6, 6, 6, 6
    };
    std::vector<grb::IndexType> cols = {
        0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 2, 5, 0, 1, 2, 3, 4, 5, 6,
        2, 4, 5, 2, 5, 0, 1, 2, 3, 4, 5, 6
    };
    std::vector<T> vals = {
        0,   1,   2,   1,   2,   3,   2,   3, 0,   2,   2,   1,   2,
        1, 0,   1,   1,   2,   1, 0,   3,   2,   3,   2, 0,   1,
        1, 0,   2,   3,   1,   1,   1,   2, 0
    };
    grb::Matrix<T> temp(7,7);
    temp.build(rows.begin(), cols.begin(), vals.begin(), rows.size());
    return temp;
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(test_sssp_negative_cycle_detection)
{
    grb::IndexType const NUM_NODES(4);
    grb::IndexType const start_index(0);

    grb::IndexArrayType i = {0, 3, 1, 2};
    grb::IndexArrayType j = {1, 1, 2, 3};
    std::vector<double>       v = {1,-5, 1, 3};
    grb::Matrix<double> G(NUM_NODES, NUM_NODES);
    G.build(i, j, v);

    grb::Vector<double> dist(NUM_NODES);
    dist.setElement(start_index, 0);

    BOOST_CHECK_EQUAL(false, sssp(G, dist));
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_batch_sssp_negative_cycle_detection)
{
    grb::IndexType const NUM_NODES(4);

    grb::IndexArrayType i = {0, 3, 1, 2};
    grb::IndexArrayType j = {1, 1, 2, 3};
    std::vector<double>       v = {1,-5, 1, 3};
    grb::Matrix<double> G(NUM_NODES, NUM_NODES);
    G.build(i, j, v);

    // Perform for all roots simultaneously
    auto dists =
        grb::scaled_identity<grb::Matrix<double> >(NUM_NODES, 0);

    BOOST_CHECK_EQUAL(false, batch_sssp(G, dists));
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_filtered_sssp_negative_cycle_detection)
{
    grb::IndexType const NUM_NODES(4);
    grb::IndexType const start_index(0);

    grb::IndexArrayType i = {0, 3, 1, 2};
    grb::IndexArrayType j = {1, 1, 2, 3};
    std::vector<double>       v = {1,-5, 1, 2};  // TODO won't work if 0 distance
    grb::Matrix<double> G(NUM_NODES, NUM_NODES);
    G.build(i, j, v);

    grb::Vector<double> dist(NUM_NODES);
    dist.setElement(start_index, 0);

    BOOST_CHECK_EQUAL(false, filtered_sssp(G, dist));
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_sssp_basic_mincost_one_root)
{
    grb::IndexType const NUM_NODES(4);
    grb::IndexType const start_index(0);

    grb::IndexArrayType i = {0, 0, 1, 2};
    grb::IndexArrayType j = {1, 3, 2, 3};
    std::vector<double>       v = {1, 5, 1, 1};
    grb::Matrix<double> G(NUM_NODES, NUM_NODES);
    G.build(i, j, v);

    grb::Vector<double> dist(NUM_NODES);
    dist.setElement(start_index, 0);

    BOOST_CHECK_EQUAL(true, sssp(G, dist));
    //grb::print_vector(std::cout, dist, "MINCOST dist");
    BOOST_CHECK_EQUAL(dist.extractElement(0), 0);
    BOOST_CHECK_EQUAL(dist.extractElement(1), 1);
    BOOST_CHECK_EQUAL(dist.extractElement(2), 2);
    BOOST_CHECK_EQUAL(dist.extractElement(3), 3);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_sssp_basic_mincost2_one_root)
{
    grb::IndexType const NUM_NODES(4);
    grb::IndexType const start_index(0);

    grb::IndexArrayType i = {0, 0, 1, 2};
    grb::IndexArrayType j = {1, 3, 2, 3};
    std::vector<double>       v = {1, 2, 1, 1};
    grb::Matrix<double> G(NUM_NODES, NUM_NODES);
    G.build(i, j, v);

    grb::Vector<double> dist(NUM_NODES);
    dist.setElement(start_index, 0);

    BOOST_CHECK_EQUAL(true, sssp(G, dist));
    //grb::print_vector(std::cout, dist, "MINCOST dist");
    BOOST_CHECK_EQUAL(dist.extractElement(0), 0);
    BOOST_CHECK_EQUAL(dist.extractElement(1), 1);
    BOOST_CHECK_EQUAL(dist.extractElement(2), 2);
    BOOST_CHECK_EQUAL(dist.extractElement(3), 2);
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(test_sssp_basic_double_one_root)
{
    grb::IndexType const NUM_NODES(9);
    grb::IndexType const start_index(3);

    grb::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    grb::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<double>       v(i.size(), 1);
    grb::Matrix<double> G_tn(NUM_NODES, NUM_NODES);
    G_tn.build(i, j, v);

    grb::Vector<double> dist(NUM_NODES);
    dist.setElement(start_index, 0);

    BOOST_CHECK_EQUAL(true, sssp(G_tn, dist));

    auto G_tn_answer(get_tn_answer<double>());

    for (grb::IndexType ix = 0; ix < NUM_NODES; ++ix)
    {
        if (G_tn_answer.hasElement(start_index, ix))
        {
            BOOST_CHECK_EQUAL(dist.hasElement(ix), true);
            BOOST_CHECK_EQUAL(dist.extractElement(ix),
                              G_tn_answer.extractElement(start_index, ix));
        }
        else
        {
            BOOST_CHECK_EQUAL(dist.hasElement(ix), false);
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_batch_sssp_basic_double)
{
    grb::IndexType const NUM_NODES(9);
    grb::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    grb::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<double>       v(i.size(), 1);
    grb::Matrix<double> G_tn(NUM_NODES, NUM_NODES);
    G_tn.build(i, j, v);

    // Perform for all roots simultaneously
    auto paths =
        grb::scaled_identity<grb::Matrix<double> >(NUM_NODES, 0);
    BOOST_CHECK_EQUAL(true, batch_sssp(G_tn, paths));

    auto G_tn_answer(get_tn_answer<double>());

    //print_matrix(std::cout, paths, "result");
    //print_matrix(std::cout, G_tn_answer, "correct answer");
    BOOST_CHECK_EQUAL(paths, G_tn_answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_batch_sssp_basic_uint)
{
    grb::IndexType const NUM_NODES(9);
    grb::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    grb::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<unsigned int> v(i.size(), 1);
    grb::Matrix<unsigned int> G_tn(NUM_NODES, NUM_NODES);
    G_tn.build(i, j, v);

    // Perform for all roots simultaneously
    auto paths =
        grb::scaled_identity<grb::Matrix<unsigned int> >(NUM_NODES, 0);
    BOOST_CHECK_EQUAL(true, batch_sssp(G_tn, paths));

    auto G_tn_answer(get_tn_answer<unsigned int>());

    BOOST_CHECK_EQUAL(paths, G_tn_answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_batch_sssp_gilbert_double)
{
    grb::IndexType const NUM_NODES(7);
    grb::IndexArrayType i = {0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6};
    grb::IndexArrayType j = {1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4};
    std::vector<double>       v(i.size(), 1);
    grb::Matrix<double> G_gilbert(NUM_NODES, NUM_NODES);
    G_gilbert.build(i, j, v);

    // Perform for all roots simultaneously
    auto paths =
        grb::scaled_identity<grb::Matrix<double> >(NUM_NODES, 0);
    BOOST_CHECK_EQUAL(true, batch_sssp(G_gilbert, paths));

    auto G_gilbert_answer(get_gilbert_answer<double>());

    BOOST_CHECK_EQUAL(paths, G_gilbert_answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_batch_sssp_gilbert_uint)
{
    grb::IndexType const NUM_NODES(7);
    grb::IndexArrayType i = {0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6};
    grb::IndexArrayType j = {1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4};
    std::vector<unsigned int> v(i.size(), 1);
    grb::Matrix<unsigned int> G_gilbert(NUM_NODES, NUM_NODES);
    G_gilbert.build(i, j, v);

    // Perform for all roots simultaneously
    auto paths =
        grb::scaled_identity<grb::Matrix<unsigned int> >(NUM_NODES, 0);
    BOOST_CHECK_EQUAL(true, batch_sssp(G_gilbert, paths));

    auto G_gilbert_answer(get_gilbert_answer<unsigned int>());

    BOOST_CHECK_EQUAL(paths, G_gilbert_answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_filtered_sssp_basic_mincost_one_root)
{
    grb::IndexType const NUM_NODES(4);
    grb::IndexType const start_index(0);

    grb::IndexArrayType i = {0, 0, 1, 2};
    grb::IndexArrayType j = {1, 3, 2, 3};
    std::vector<double>       v = {1, 5, 1, 1};
    grb::Matrix<double> G(NUM_NODES, NUM_NODES);
    G.build(i, j, v);

    grb::Vector<double> dist(NUM_NODES);
    dist.setElement(start_index, 0);

    BOOST_CHECK_EQUAL(true, filtered_sssp(G, dist));
    //grb::print_vector(std::cout, dist, "MINCOST dist");
    BOOST_CHECK_EQUAL(dist.extractElement(0), 0);
    BOOST_CHECK_EQUAL(dist.extractElement(1), 1);
    BOOST_CHECK_EQUAL(dist.extractElement(2), 2);
    BOOST_CHECK_EQUAL(dist.extractElement(3), 3);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_filtered_sssp_basic_mincost2_one_root)
{
    grb::IndexType const NUM_NODES(4);
    grb::IndexType const start_index(0);

    grb::IndexArrayType i = {0, 0, 1, 2};
    grb::IndexArrayType j = {1, 3, 2, 3};
    std::vector<double>       v = {1, 2, 1, 1};
    grb::Matrix<double> G(NUM_NODES, NUM_NODES);
    G.build(i, j, v);

    grb::Vector<double> dist(NUM_NODES);
    dist.setElement(start_index, 0);

    BOOST_CHECK_EQUAL(true, filtered_sssp(G, dist));
    //grb::print_vector(std::cout, dist, "MINCOST dist");
    BOOST_CHECK_EQUAL(dist.extractElement(0), 0);
    BOOST_CHECK_EQUAL(dist.extractElement(1), 1);
    BOOST_CHECK_EQUAL(dist.extractElement(2), 2);
    BOOST_CHECK_EQUAL(dist.extractElement(3), 2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_filtered_sssp_basic_double_one_root)
{
    grb::IndexType const NUM_NODES(9);
    grb::IndexType const start_index(3);

    grb::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    grb::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<double>       v(i.size(), 1);
    grb::Matrix<double> G_tn(NUM_NODES, NUM_NODES);
    G_tn.build(i, j, v);

    grb::Vector<double> dist(NUM_NODES);
    dist.setElement(start_index, 0);

    BOOST_CHECK_EQUAL(true, filtered_sssp(G_tn, dist));

    auto G_tn_answer(get_tn_answer<double>());

    for (grb::IndexType ix = 0; ix < NUM_NODES; ++ix)
    {
        if (G_tn_answer.hasElement(start_index, ix))
        {
            BOOST_CHECK_EQUAL(dist.hasElement(ix), true);
            BOOST_CHECK_EQUAL(dist.extractElement(ix),
                              G_tn_answer.extractElement(start_index, ix));
        }
        else
        {
            BOOST_CHECK_EQUAL(dist.hasElement(ix), false);
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_filtered_sssp_marcin_uint)
{
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

    grb::IndexType const SOURCE = 0;
    std::vector<unsigned int> dense_ans = {0, 5, 1, 2, 3, 4, 6, 5};
    grb::Vector<unsigned int> answer(dense_ans);

    grb::IndexType const NUM_NODES(8);
    grb::IndexArrayType i = {0,0,1,1,1,2,2,3,3,4,4,5,5,5,6,6,7,7};
    grb::IndexArrayType j = {1,2,0,5,6,0,3,2,4,3,5,1,4,7,1,7,5,6};
    std::vector<unsigned int> v = {7,1,7,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    grb::Matrix<unsigned int> G(NUM_NODES, NUM_NODES);
    G.build(i.begin(), j.begin(), v.begin(), i.size());

    grb::Vector<unsigned int> result(NUM_NODES);
    result.setElement(SOURCE, 0);
    filtered_sssp(G, result);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_sssp_delta2_step_gilbert_uint)
{
    grb::IndexType const SOURCE = 3;
    std::vector<unsigned int> dense_ans3 =  {3,   4,   1,   0,   5,   2,   7};
    grb::Vector<unsigned int> answer3(dense_ans3);

    grb::IndexType const NUM_NODES(7);
    grb::IndexArrayType i = {0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6};
    grb::IndexArrayType j = {1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4};
    std::vector<unsigned int> v = {1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3};
    grb::Matrix<unsigned int> G_gilbert(NUM_NODES, NUM_NODES);
    G_gilbert.build(i.begin(), j.begin(), v.begin(), i.size());

    grb::Vector<unsigned int> result(NUM_NODES);
    sssp_delta_step(G_gilbert, 2U, SOURCE, result);

    BOOST_CHECK_EQUAL(result, answer3);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_sssp_delta_step_marcin_uint)
{
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

    grb::IndexType const SOURCE = 0;
    std::vector<unsigned int> dense_ans = {0, 5, 1, 2, 3, 4, 6, 5};
    grb::Vector<unsigned int> answer(dense_ans);

    grb::IndexType const NUM_NODES(8);
    grb::IndexArrayType i = {0,0,1,1,1,2,2,3,3,4,4,5,5,5,6,6,7,7};
    grb::IndexArrayType j = {1,2,0,5,6,0,3,2,4,3,5,1,4,7,1,7,5,6};
    std::vector<unsigned int> v = {7,1,7,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    grb::Matrix<unsigned int> G(NUM_NODES, NUM_NODES);
    G.build(i.begin(), j.begin(), v.begin(), i.size());

    grb::Vector<unsigned int> result2(NUM_NODES);
    sssp_delta_step(G, 2U, SOURCE, result2);
    BOOST_CHECK_EQUAL(result2, answer);
    grb::Vector<unsigned int> result7(NUM_NODES);
    sssp_delta_step(G, 7U, SOURCE, result7);
    BOOST_CHECK_EQUAL(result7, answer);
    grb::Vector<unsigned int> result8(NUM_NODES);
    sssp_delta_step(G, 8U, SOURCE, result8);
    BOOST_CHECK_EQUAL(result8, answer);
    grb::Vector<unsigned int> result9(NUM_NODES);
    sssp_delta_step(G, 9U, SOURCE, result9);
    BOOST_CHECK_EQUAL(result9, answer);
    grb::Vector<unsigned int> result10(NUM_NODES);
    sssp_delta_step(G, 10U, SOURCE, result10);
    BOOST_CHECK_EQUAL(result10, answer);
}

BOOST_AUTO_TEST_SUITE_END()
