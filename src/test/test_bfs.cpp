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

#include <graphblas/graphblas.hpp>
#include <algorithms/bfs.hpp>

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE bfs_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(bfs_suite)

//****************************************************************************
    /// @todo Use a dense matrix type?
namespace
{
    template <typename T>
    std::vector<std::vector<T> > get_tn_answer(T const &INF)
    {
        return std::vector<std::vector<T> >(
            {{  0,   3,   4,   0,   3,   2,   3, INF,   4},
             {  3,   1,   6,   1,   3,   2,   1, INF,   2},
             {  3,   6,   2,   4,   2,   2,   2, INF,   2},
             {  3,   3,   4,   3,   3,   2,   3, INF,   4},
             {  3,   3,   4,   4,   4,   2,   2, INF,   4},
             {  3,   6,   5,   4,   2,   5,   2, INF,   2},
             {  3,   6,   6,   6,   2,   2,   6, INF,   2},
             {INF, INF, INF, INF, INF, INF, INF,   7, INF},
             {  3,   3,   8,   4,   8,   2,   2, INF,   8}});
    }
    /*
     *
     * template:
    template <typename T>
    Matrix<T, DirectedMatrixTag> get_tn_answer(T const &INF)
    {
        std::vector<graphblas::IndexType> rows = {
        };
        std::vector<graphblas::IndexType> cols = {
        };
        std::vector<T> vals = {
        };

        Matrix<T, DirectedMatrixTag> temp(9,9, INF);
        buildmatrix(temp, rows.begin(), cols.begin(), vals.begin(), rows.size());
        return temp;
    }
    */
    //template <typename T>
    //graphblas::Matrix<T> get_tn_answer(T const &INF)
    //{
    //    std::vector<graphblas::IndexType> rows = {
    //        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
    //        2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
    //        5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8
    //    };
    //    std::vector<graphblas::IndexType> cols = {
    //        0, 1, 2, 3, 4, 5, 6, 8, 0, 1, 2, 3, 4, 5, 6, 8, 0, 1, 2, 3, 4, 5, 6,
    //        8, 0, 1, 2, 3, 4, 5, 6, 8, 0, 1, 2, 3, 4, 5, 6, 8, 0, 1, 2, 3, 4, 5,
    //        6, 8, 0, 1, 2, 3, 4, 5, 6, 8, 7, 0, 1, 2, 3, 4, 5, 6, 8
    //    };
    //    std::vector<T> vals = {
    //        0,   3,   4,   0,   3,   2,   3,   4,   3,   1,   6,   1,   3,
    //        2,   1,   2,   3,   6,   2,   4,   2,   2,   2,   2,   3,   3,
    //        4,   3,   3,   2,   3,   4,   3,   3,   4,   4,   4,   2,   2,
    //        4,   3,   6,   5,   4,   2,   5,   2,   2,   3,   6,   6,   6,
    //        2,   2,   6,   2,   7,   3,   3,   8,   4,   8,   2,   2,   8
    //    };

    //    graphblas::Matrix<T> temp(9,9, INF);
    //    graphblas::buildmatrix(temp, rows.begin(), cols.begin(), vals.begin(), rows.size());
    //    return temp;
    //}

    template <typename T>
    std::vector<std::vector<T> > get_gilbert_answer(T const &INF)
    {
        return std::vector<std::vector<T> >(
            {{  0,  0,  3,  0,  1,  2,  1},
             {  3,  1,  6,  6,  1,  4,  1},
             {INF,INF,  2,INF,INF,  2,INF},
             {  3,  0,  3,  3,  1,  2,  1},
             {INF,INF,  5,INF,  4,  4,INF},
             {INF,INF,  5,INF,INF,  5,INF},
             {  3,  0,  6,  6,  6,  2,  6}});
    }
    //template <typename T>
    //graphblas::Matrix<T> get_gilbert_answer(T const &INF)
    //{
    //    std::vector<graphblas::IndexType> rows = {
    //        0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3,
    //        4, 4, 4, 5, 5, 6, 6, 6, 6, 6, 6, 6
    //    };
    //    std::vector<graphblas::IndexType> cols = {
    //        0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 2, 5, 0, 1, 2, 3, 4, 5, 6,
    //        2, 4, 5, 2, 5, 0, 1, 2, 3, 4, 5, 6
    //    };
    //    std::vector<T> vals = {
    //        0, 0,   3, 0,   1,   2,   1,   3,   1,   6,   6,   1,   4,
    //        1,   2,   2,   3, 0,   3,   3,   1,   2,   1,   5,   4,   4,
    //        5,   5,   3, 0,   6,   6,   6,   2,   6
    //    };

    //    graphblas::Matrix<T> temp(9,9, INF);
    //    graphblas::buildmatrix(temp, rows.begin(), cols.begin(), vals.begin(), rows.size());
    //    return temp;
    //}
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bfs_test_basic_one_root)
{
    typedef double T;
    typedef graphblas::Matrix<T, graphblas::DirectedMatrixTag> GrBMatrix;
    T const INF(std::numeric_limits<T>::max());
    graphblas::IndexType const NUM_NODES(9);
    graphblas::IndexType const START_INDEX(5);

    graphblas::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    graphblas::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<T> v(i.size(), 1);

    GrBMatrix G_tn(NUM_NODES, NUM_NODES, INF);
    buildmatrix(G_tn, i, j, v);

    GrBMatrix root(1, NUM_NODES, INF);
    std::vector<graphblas::IndexType> r={0}, c={START_INDEX};
    std::vector<T> v_r={0};
    graphblas::buildmatrix(root, r.begin(), c.begin(), v_r.begin(), v_r.size());
    //root.setElement(0, START_INDEX, 0);  // multiplicative identity

    auto G_tn_answer(get_tn_answer(INF));

    GrBMatrix parent_list(1, NUM_NODES, INF);
    algorithms::bfs(G_tn, root, parent_list);

    for (graphblas::IndexType ix = 0; ix < NUM_NODES; ++ix)
    {
        BOOST_CHECK_EQUAL(parent_list.extractElement(0, ix),
                          G_tn_answer[START_INDEX][ix]);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bfs_test_basic_one_root_integer)
{
    typedef unsigned int T;
    typedef graphblas::Matrix<T, graphblas::DirectedMatrixTag> GrBMatrix;
    T const INF(std::numeric_limits<T>::max());
    graphblas::IndexType const NUM_NODES(9);
    graphblas::IndexType const START_INDEX(5);

    graphblas::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    graphblas::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<T> v(i.size(), 1);

    GrBMatrix G_tn(NUM_NODES, NUM_NODES, INF);
    graphblas::buildmatrix(G_tn, i, j, v);

    GrBMatrix root(1, NUM_NODES, INF);
    std::vector<graphblas::IndexType> r={0}, c={START_INDEX};
    std::vector<T> v_r={0};
    graphblas::buildmatrix(root, r.begin(), c.begin(), v_r.begin(), v_r.size());
    //root.setElement(0, START_INDEX, 0);  // multiplicative identity

    auto G_tn_answer(get_tn_answer(INF));

    GrBMatrix parent_list(1, NUM_NODES, INF);
    algorithms::bfs(G_tn, root, parent_list);

    for (graphblas::IndexType ix = 0; ix < NUM_NODES; ++ix)
    {
        BOOST_CHECK_EQUAL(parent_list.extractElement(0, ix),
                          G_tn_answer[START_INDEX][ix]);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bfs_test_basic)
{
    typedef double T;
    typedef graphblas::Matrix<T, graphblas::DirectedMatrixTag> GrBMatrix;
    T const INF(std::numeric_limits<T>::max());
    graphblas::IndexType const NUM_NODES(9);

    graphblas::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    graphblas::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<T> v(i.size(), 1);

    GrBMatrix G_tn(NUM_NODES, NUM_NODES, INF);
    graphblas::buildmatrix(G_tn, i, j, v);

    auto roots = graphblas::identity<GrBMatrix>(NUM_NODES, INF, 0);

    /// @todo Use a dense matrix type?
    auto G_tn_answer(get_tn_answer(INF));

    GrBMatrix parent_lists(NUM_NODES, NUM_NODES, INF);

    algorithms::bfs(G_tn, roots, parent_lists);

    for (graphblas::IndexType r = 0; r < NUM_NODES; ++r)
    {
        for (graphblas::IndexType c = 0; c < NUM_NODES; ++c)
        {
            BOOST_CHECK_EQUAL(parent_lists.extractElement(r, c),
                              G_tn_answer[r][c]);
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bfs_test_gilbert)
{
    typedef double T;
    typedef graphblas::Matrix<T, graphblas::DirectedMatrixTag> GrBMatrix;
    T const INF(std::numeric_limits<T>::max());
    graphblas::IndexType const NUM_NODES(7);

    graphblas::IndexArrayType i = {0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6};
    graphblas::IndexArrayType j = {1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4};
    std::vector<T> v(i.size(), 1);

    GrBMatrix G_gilbert(NUM_NODES, NUM_NODES, INF);
    graphblas::buildmatrix(G_gilbert, i, j, v);

    auto roots = graphblas::identity<GrBMatrix>(NUM_NODES, INF, 0);

    /// @todo Replace with a dense matrix type
    auto G_gilbert_answer(get_gilbert_answer(INF));

    GrBMatrix parent_lists(NUM_NODES, NUM_NODES, INF);

    algorithms::bfs(G_gilbert, roots, parent_lists);

    for (graphblas::IndexType r = 0; r < NUM_NODES; ++r)
    {
        for (graphblas::IndexType c = 0; c < NUM_NODES; ++c)
        {
            BOOST_CHECK_EQUAL(parent_lists.extractElement(r, c),
                              G_gilbert_answer[r][c]);
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bfs_test_level_one_root)
{
    typedef double T;
    typedef graphblas::Matrix<T, graphblas::DirectedMatrixTag> GrBMatrix;

    graphblas::IndexType const NUM_NODES(9);
    graphblas::IndexType const START_INDEX(5);

    graphblas::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    graphblas::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<T> v(i.size(), 1);

    GrBMatrix G_tn(NUM_NODES, NUM_NODES, 0);
    buildmatrix(G_tn, i, j, v);

    GrBMatrix root(1, NUM_NODES, 0);
    std::vector<graphblas::IndexType> r={0}, c={START_INDEX};
    std::vector<T> v_r={1};
    graphblas::buildmatrix(root, r.begin(), c.begin(), v_r.begin(), v_r.size());
    //root.setElement(0, START_INDEX, 1);  // multiplicative identity

    GrBMatrix levels(1, NUM_NODES, 0);
    algorithms::bfs_level(G_tn, root, levels);

    std::vector<T> answer = {5, 4, 2, 4, 3, 1, 3, 0, 3};

    for (graphblas::IndexType ix = 0; ix < NUM_NODES; ++ix)
    {
        BOOST_CHECK_EQUAL(levels.extractElement(0, ix),
                          answer[ix]);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bfs_test_level_one_root_integer)
{
    typedef unsigned int T;
    typedef graphblas::Matrix<T, graphblas::DirectedMatrixTag> GrBMatrix;

    graphblas::IndexType const NUM_NODES(9);
    graphblas::IndexType const START_INDEX(5);

    graphblas::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    graphblas::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<T> v(i.size(), 1);

    GrBMatrix G_tn(NUM_NODES, NUM_NODES, 0);
    graphblas::buildmatrix(G_tn, i, j, v);

    GrBMatrix root(1, NUM_NODES, 0);
    std::vector<graphblas::IndexType> r={0}, c={START_INDEX};
    std::vector<T> v_r={1};
    graphblas::buildmatrix(root, r.begin(), c.begin(), v_r.begin(), v_r.size());
    //root.setElement(0, START_INDEX, 1);  // multiplicative identity

    GrBMatrix levels(1, NUM_NODES, 0);
    algorithms::bfs_level(G_tn, root, levels);

    std::vector<T> answer = {5, 4, 2, 4, 3, 1, 3, 0, 3};
    for (graphblas::IndexType ix = 0; ix < NUM_NODES; ++ix)
    {
        BOOST_CHECK_EQUAL(levels.extractElement(0, ix),
                          answer[ix]);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bfs_test_level_masked_one_root)
{
    typedef double T;
    typedef graphblas::Matrix<T, graphblas::DirectedMatrixTag> GrBMatrix;

    graphblas::IndexType const NUM_NODES(9);
    graphblas::IndexType const START_INDEX(5);

    graphblas::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    graphblas::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<T> v(i.size(), 1);

    GrBMatrix G_tn(NUM_NODES, NUM_NODES, 0);
    buildmatrix(G_tn, i, j, v);

    GrBMatrix root(1, NUM_NODES, 0);
    std::vector<graphblas::IndexType> r={0}, c={START_INDEX};
    std::vector<T> v_r={1};
    graphblas::buildmatrix(root, r.begin(), c.begin(), v_r.begin(), v_r.size());
    //root.setElement(0, START_INDEX, 1);  // multiplicative identity

    GrBMatrix levels(1, NUM_NODES, 0);
    algorithms::bfs_level_masked(G_tn, root, levels);

    std::vector<T> answer = {5, 4, 2, 4, 3, 1, 3, 0, 3};
    for (graphblas::IndexType ix = 0; ix < NUM_NODES; ++ix)
    {
        BOOST_CHECK_EQUAL(levels.extractElement(0, ix),
                          answer[ix]);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bfs_test_level_masked_one_root_integer)
{
    typedef unsigned int T;
    typedef graphblas::Matrix<T, graphblas::DirectedMatrixTag> GrBMatrix;

    graphblas::IndexType const NUM_NODES(9);
    graphblas::IndexType const START_INDEX(5);

    graphblas::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    graphblas::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<T> v(i.size(), 1);

    GrBMatrix G_tn(NUM_NODES, NUM_NODES, 0);
    graphblas::buildmatrix(G_tn, i, j, v);

    GrBMatrix root(1, NUM_NODES, 0);
    std::vector<graphblas::IndexType> r={0}, c={START_INDEX};
    std::vector<T> v_r={1};
    graphblas::buildmatrix(root, r.begin(), c.begin(), v_r.begin(), v_r.size());
    //root.setElement(0, START_INDEX, 1);  // multiplicative identity

    GrBMatrix levels(1, NUM_NODES, 0);
    algorithms::bfs_level_masked(G_tn, root, levels);

    std::vector<T> answer = {5, 4, 2, 4, 3, 1, 3, 0, 3};
    for (graphblas::IndexType ix = 0; ix < NUM_NODES; ++ix)
    {
        BOOST_CHECK_EQUAL(levels.extractElement(0, ix),
                          answer[ix]);
    }
}

BOOST_AUTO_TEST_SUITE_END()
