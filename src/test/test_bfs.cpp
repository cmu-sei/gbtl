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
        std::vector<GraphBLAS::IndexType> rows = {
        };
        std::vector<GraphBLAS::IndexType> cols = {
        };
        std::vector<T> vals = {
        };

        Matrix<T, DirectedMatrixTag> temp(9,9, INF);
        buildmatrix(temp, rows.begin(), cols.begin(), vals.begin(), rows.size());
        return temp;
    }
    */
    //template <typename T>
    //GraphBLAS::Matrix<T> get_tn_answer(T const &INF)
    //{
    //    std::vector<GraphBLAS::IndexType> rows = {
    //        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
    //        2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
    //        5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8
    //    };
    //    std::vector<GraphBLAS::IndexType> cols = {
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

    //    GraphBLAS::Matrix<T> temp(9,9, INF);
    //    GraphBLAS::buildmatrix(temp, rows.begin(), cols.begin(), vals.begin(), rows.size());
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
    //GraphBLAS::Matrix<T> get_gilbert_answer(T const &INF)
    //{
    //    std::vector<GraphBLAS::IndexType> rows = {
    //        0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3,
    //        4, 4, 4, 5, 5, 6, 6, 6, 6, 6, 6, 6
    //    };
    //    std::vector<GraphBLAS::IndexType> cols = {
    //        0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 2, 5, 0, 1, 2, 3, 4, 5, 6,
    //        2, 4, 5, 2, 5, 0, 1, 2, 3, 4, 5, 6
    //    };
    //    std::vector<T> vals = {
    //        0, 0,   3, 0,   1,   2,   1,   3,   1,   6,   6,   1,   4,
    //        1,   2,   2,   3, 0,   3,   3,   1,   2,   1,   5,   4,   4,
    //        5,   5,   3, 0,   6,   6,   6,   2,   6
    //    };

    //    GraphBLAS::Matrix<T> temp(9,9, INF);
    //    GraphBLAS::buildmatrix(temp, rows.begin(), cols.begin(), vals.begin(), rows.size());
    //    return temp;
    //}
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bfs_test_basic_one_root)
{
    typedef double T;
    typedef GraphBLAS::Matrix<T, GraphBLAS::DirectedMatrixTag> GrBMatrix;

    GraphBLAS::IndexType const NUM_NODES(9);
    GraphBLAS::IndexType const START_INDEX(5);

    GraphBLAS::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    GraphBLAS::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<T> v(i.size(), 1);

    GrBMatrix G_tn(NUM_NODES, NUM_NODES);
    G_tn.build(i, j, v);

    GraphBLAS::Vector<T> root(NUM_NODES);
    root.setElement(START_INDEX, 0);

    GraphBLAS::Vector<T> parent_list(NUM_NODES);
    algorithms::bfs(G_tn, root, parent_list);

    T const INF(std::numeric_limits<T>::max());
    auto G_tn_answer(get_tn_answer(INF));
    for (GraphBLAS::IndexType ix = 0; ix < NUM_NODES; ++ix)
    {
        if (parent_list.hasElement(ix))
        {
            BOOST_CHECK_EQUAL(parent_list.extractElement(ix),
                              G_tn_answer[START_INDEX][ix]);
        }
        else
        {
            BOOST_CHECK_EQUAL(INF, G_tn_answer[START_INDEX][ix]);
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bfs_batch_test_basic_one_root)
{
    typedef double T;
    typedef GraphBLAS::Matrix<T, GraphBLAS::DirectedMatrixTag> GrBMatrix;

    GraphBLAS::IndexType const NUM_NODES(9);
    GraphBLAS::IndexType const START_INDEX(5);

    GraphBLAS::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    GraphBLAS::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<T> v(i.size(), 1);

    GrBMatrix G_tn(NUM_NODES, NUM_NODES);
    G_tn.build(i, j, v);

    GrBMatrix root(1, NUM_NODES);
    root.setElement(0, START_INDEX, 0);

    GrBMatrix parent_list(1, NUM_NODES);
    algorithms::bfs_batch(G_tn, root, parent_list);

    T const INF(std::numeric_limits<T>::max());
    auto G_tn_answer(get_tn_answer(INF));
    for (GraphBLAS::IndexType ix = 0; ix < NUM_NODES; ++ix)
    {
        if (parent_list.hasElement(0, ix))
        {
            BOOST_CHECK_EQUAL(parent_list.extractElement(0, ix),
                              G_tn_answer[START_INDEX][ix]);
        }
        else
        {
            BOOST_CHECK_EQUAL(INF, G_tn_answer[START_INDEX][ix]);
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bfs_batch_test_basic_one_root_integer)
{
    typedef unsigned int T;
    typedef GraphBLAS::Matrix<T, GraphBLAS::DirectedMatrixTag> GrBMatrix;

    GraphBLAS::IndexType const NUM_NODES(9);
    GraphBLAS::IndexType const START_INDEX(5);

    GraphBLAS::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    GraphBLAS::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<T> v(i.size(), 1);

    GrBMatrix G_tn(NUM_NODES, NUM_NODES);
    G_tn.build(i, j, v);

    GrBMatrix root(1, NUM_NODES);
    root.setElement(0, START_INDEX, 0);  // multiplicative identity

    GrBMatrix parent_list(1, NUM_NODES);
    algorithms::bfs_batch(G_tn, root, parent_list);

    T const INF(std::numeric_limits<T>::max());
    auto G_tn_answer(get_tn_answer(INF));
    for (GraphBLAS::IndexType ix = 0; ix < NUM_NODES; ++ix)
    {
        if (parent_list.hasElement(0, ix))
        {
            BOOST_CHECK_EQUAL(parent_list.extractElement(0, ix),
                              G_tn_answer[START_INDEX][ix]);
        }
        else
        {
            BOOST_CHECK_EQUAL(INF, G_tn_answer[START_INDEX][ix]);
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bfs_batch_test_basic)
{
    typedef double T;
    typedef GraphBLAS::Matrix<T, GraphBLAS::DirectedMatrixTag> GrBMatrix;
    GraphBLAS::IndexType const NUM_NODES(9);

    GraphBLAS::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    GraphBLAS::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<T> v(i.size(), 1);

    GrBMatrix G_tn(NUM_NODES, NUM_NODES);
    G_tn.build(i, j, v);

    GrBMatrix roots(NUM_NODES, NUM_NODES);
    GraphBLAS::IndexArrayType ii, jj, vv;
    for (GraphBLAS::IndexType ix = 0; ix < NUM_NODES; ++ix)
    {
        ii.push_back(ix);
        jj.push_back(ix);
        vv.push_back(1);
    }
    roots.build(ii, jj, vv);

    GrBMatrix parent_lists(NUM_NODES, NUM_NODES);

    algorithms::bfs_batch(G_tn, roots, parent_lists);

    /// @todo Use a dense matrix type?
    T const INF(std::numeric_limits<T>::max());
    auto G_tn_answer(get_tn_answer(INF));

    for (GraphBLAS::IndexType r = 0; r < NUM_NODES; ++r)
    {
        for (GraphBLAS::IndexType c = 0; c < NUM_NODES; ++c)
        {
            if (parent_lists.hasElement(r, c))
            {
                BOOST_CHECK_EQUAL(parent_lists.extractElement(r,c),
                                  G_tn_answer[r][c]);
            }
            else
            {
                BOOST_CHECK_EQUAL(INF, G_tn_answer[r][c]);
            }
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bfs_batch_test_gilbert)
{
    typedef double T;
    typedef GraphBLAS::Matrix<T, GraphBLAS::DirectedMatrixTag> GrBMatrix;

    GraphBLAS::IndexType const NUM_NODES(7);

    GraphBLAS::IndexArrayType i = {0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6};
    GraphBLAS::IndexArrayType j = {1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4};
    std::vector<T> v(i.size(), 1);

    GrBMatrix G_gilbert(NUM_NODES, NUM_NODES);
    G_gilbert.build(i, j, v);

    GrBMatrix roots(NUM_NODES, NUM_NODES);
    GraphBLAS::IndexArrayType ii, jj, vv;
    for (GraphBLAS::IndexType ix = 0; ix < NUM_NODES; ++ix)
    {
        ii.push_back(ix);
        jj.push_back(ix);
        vv.push_back(1);
    }
    roots.build(ii, jj, vv);

    GrBMatrix parent_lists(NUM_NODES, NUM_NODES);

    algorithms::bfs_batch(G_gilbert, roots, parent_lists);

    /// @todo Replace with a dense matrix type
    T const INF(std::numeric_limits<T>::max());
    auto G_gilbert_answer(get_gilbert_answer(INF));
    for (GraphBLAS::IndexType r = 0; r < NUM_NODES; ++r)
    {
        for (GraphBLAS::IndexType c = 0; c < NUM_NODES; ++c)
        {
            if (parent_lists.hasElement(r, c))
            {
                BOOST_CHECK_EQUAL(parent_lists.extractElement(r,c),
                                  G_gilbert_answer[r][c]);
            }
            else
            {
                BOOST_CHECK_EQUAL(INF, G_gilbert_answer[r][c]);
            }
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bfs_level_test_one_root)
{
    typedef double T;
    typedef GraphBLAS::Matrix<T, GraphBLAS::DirectedMatrixTag> GrBMatrix;

    GraphBLAS::IndexType const NUM_NODES(9);
    GraphBLAS::IndexType const START_INDEX(5);

    GraphBLAS::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    GraphBLAS::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<T> v(i.size(), 1);

    GrBMatrix G_tn(NUM_NODES, NUM_NODES);
    G_tn.build(i, j, v);

    GrBMatrix root(1, NUM_NODES);
    root.setElement(0, START_INDEX, 1);  // multiplicative identity

    GrBMatrix levels(1, NUM_NODES);
    algorithms::bfs_level(G_tn, root, levels);

    std::vector<T> answer = {5, 4, 2, 4, 3, 1, 3, 0, 3};

    for (GraphBLAS::IndexType ix = 0; ix < NUM_NODES; ++ix)
    {
        if (levels.hasElement(0, ix))
        {
            BOOST_CHECK_EQUAL(levels.extractElement(0, ix),
                              answer[ix]);
        }
        else
        {
            BOOST_CHECK_EQUAL(0, answer[ix]);
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bfs_level_test_one_root_integer)
{
    typedef unsigned int T;
    typedef GraphBLAS::Matrix<T, GraphBLAS::DirectedMatrixTag> GrBMatrix;

    GraphBLAS::IndexType const NUM_NODES(9);
    GraphBLAS::IndexType const START_INDEX(5);

    GraphBLAS::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    GraphBLAS::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<T> v(i.size(), 1);

    GrBMatrix G_tn(NUM_NODES, NUM_NODES);
    G_tn.build(i, j, v);

    GrBMatrix root(1, NUM_NODES);
    root.setElement(0, START_INDEX, 1);  // multiplicative identity

    GrBMatrix levels(1, NUM_NODES);
    algorithms::bfs_level(G_tn, root, levels);

    std::vector<T> answer = {5, 4, 2, 4, 3, 1, 3, 0, 3};
    for (GraphBLAS::IndexType ix = 0; ix < NUM_NODES; ++ix)
    {
        if (levels.hasElement(0, ix))
        {
            BOOST_CHECK_EQUAL(levels.extractElement(0, ix),
                              answer[ix]);
        }
        else
        {
            BOOST_CHECK_EQUAL(0, answer[ix]);
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bfs_level_masked_test_one_root)
{
    typedef double T;
    typedef GraphBLAS::Matrix<T, GraphBLAS::DirectedMatrixTag> GrBMatrix;
    typedef GraphBLAS::Vector<T> GrBVector;

    GraphBLAS::IndexType const NUM_NODES(9);
    GraphBLAS::IndexType const START_INDEX(5);

    GraphBLAS::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    GraphBLAS::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<T> v(i.size(), 1);

    GrBMatrix G_tn(NUM_NODES, NUM_NODES);
    G_tn.build(i, j, v);

    GrBVector root(NUM_NODES);
    root.setElement(START_INDEX, 1);  // multiplicative identity

    GrBVector levels(NUM_NODES);
    algorithms::bfs_level_masked(G_tn, root, levels);

    std::vector<T> answer = {5, 4, 2, 4, 3, 1, 3, 0, 3};
    for (GraphBLAS::IndexType ix = 0; ix < NUM_NODES; ++ix)
    {
        if (levels.hasElement(ix))
        {
            BOOST_CHECK_EQUAL(levels.extractElement(ix),
                              answer[ix]);
        }
        else
        {
            BOOST_CHECK_EQUAL(0, answer[ix]);
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bfs_level_masked_test_one_root_integer)
{
    typedef unsigned int T;
    typedef GraphBLAS::Matrix<T, GraphBLAS::DirectedMatrixTag> GrBMatrix;
    typedef GraphBLAS::Vector<T> GrBVector;

    GraphBLAS::IndexType const NUM_NODES(9);
    GraphBLAS::IndexType const START_INDEX(5);

    GraphBLAS::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    GraphBLAS::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<T> v(i.size(), 1);

    GrBMatrix G_tn(NUM_NODES, NUM_NODES);
    G_tn.build(i, j, v);

    GrBVector root(NUM_NODES);
    root.setElement(START_INDEX, 1);  // multiplicative identity

    GrBVector levels(NUM_NODES);
    algorithms::bfs_level_masked(G_tn, root, levels);

    std::vector<T> answer = {5, 4, 2, 4, 3, 1, 3, 0, 3};
    for (GraphBLAS::IndexType ix = 0; ix < NUM_NODES; ++ix)
    {
        if (levels.hasElement(ix))
        {
            BOOST_CHECK_EQUAL(levels.extractElement(ix),
                              answer[ix]);
        }
        else
        {
            BOOST_CHECK_EQUAL(0, answer[ix]);
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(batch_bfs_level_masked_test_one_root)
{
    typedef double T;
    typedef GraphBLAS::Matrix<T, GraphBLAS::DirectedMatrixTag> GrBMatrix;

    GraphBLAS::IndexType const NUM_NODES(9);
    GraphBLAS::IndexType const START_INDEX(5);

    GraphBLAS::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    GraphBLAS::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<T> v(i.size(), 1);

    GrBMatrix G_tn(NUM_NODES, NUM_NODES);
    G_tn.build(i, j, v);

    GrBMatrix root(1, NUM_NODES);
    root.setElement(0, START_INDEX, 1);  // multiplicative identity

    GrBMatrix levels(1, NUM_NODES);
    algorithms::batch_bfs_level_masked(G_tn, root, levels);

    std::vector<T> answer = {5, 4, 2, 4, 3, 1, 3, 0, 3};
    for (GraphBLAS::IndexType ix = 0; ix < NUM_NODES; ++ix)
    {
        if (levels.hasElement(0, ix))
        {
            BOOST_CHECK_EQUAL(levels.extractElement(0, ix),
                              answer[ix]);
        }
        else
        {
            BOOST_CHECK_EQUAL(0, answer[ix]);
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(batch_bfs_level_masked_test_one_root_integer)
{
    typedef unsigned int T;
    typedef GraphBLAS::Matrix<T, GraphBLAS::DirectedMatrixTag> GrBMatrix;

    GraphBLAS::IndexType const NUM_NODES(9);
    GraphBLAS::IndexType const START_INDEX(5);

    GraphBLAS::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    GraphBLAS::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<T> v(i.size(), 1);

    GrBMatrix G_tn(NUM_NODES, NUM_NODES);
    G_tn.build(i, j, v);

    GrBMatrix root(1, NUM_NODES);
    root.setElement(0, START_INDEX, 1);  // multiplicative identity

    GrBMatrix levels(1, NUM_NODES);
    algorithms::batch_bfs_level_masked(G_tn, root, levels);

    std::vector<T> answer = {5, 4, 2, 4, 3, 1, 3, 0, 3};
    for (GraphBLAS::IndexType ix = 0; ix < NUM_NODES; ++ix)
    {
        if (levels.hasElement(0, ix))
        {
            BOOST_CHECK_EQUAL(levels.extractElement(0, ix),
                              answer[ix]);
        }
        else
        {
            BOOST_CHECK_EQUAL(0, answer[ix]);
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bfs_level_masked_v2_test_one_root)
{
    typedef double T;
    typedef GraphBLAS::Matrix<T, GraphBLAS::DirectedMatrixTag> GrBMatrix;
    typedef GraphBLAS::Vector<T> GrBVector;

    GraphBLAS::IndexType const NUM_NODES(9);
    GraphBLAS::IndexType const START_INDEX(5);

    GraphBLAS::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    GraphBLAS::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<T> v(i.size(), 1);

    GrBMatrix G_tn(NUM_NODES, NUM_NODES);
    G_tn.build(i, j, v);

    GrBVector root(NUM_NODES);
    root.setElement(START_INDEX, 1);  // multiplicative identity

    GrBVector levels(NUM_NODES);
    algorithms::bfs_level_masked_v2(G_tn, root, levels);

    std::vector<T> answer = {5, 4, 2, 4, 3, 1, 3, 0, 3};
    for (GraphBLAS::IndexType ix = 0; ix < NUM_NODES; ++ix)
    {
        if (levels.hasElement(ix))
        {
            BOOST_CHECK_EQUAL(levels.extractElement(ix),
                              answer[ix]);
        }
        else
        {
            BOOST_CHECK_EQUAL(0, answer[ix]);
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(bfs_level_masked_v2_test_one_root_integer)
{
    typedef unsigned int T;
    typedef GraphBLAS::Matrix<T, GraphBLAS::DirectedMatrixTag> GrBMatrix;
    typedef GraphBLAS::Vector<T> GrBVector;

    GraphBLAS::IndexType const NUM_NODES(9);
    GraphBLAS::IndexType const START_INDEX(5);

    GraphBLAS::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    GraphBLAS::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<T> v(i.size(), 1);

    GrBMatrix G_tn(NUM_NODES, NUM_NODES);
    G_tn.build(i, j, v);

    GrBVector root(NUM_NODES);
    root.setElement(START_INDEX, 1);  // multiplicative identity

    GrBVector levels(NUM_NODES);
    algorithms::bfs_level_masked_v2(G_tn, root, levels);

    std::vector<T> answer = {5, 4, 2, 4, 3, 1, 3, 0, 3};
    for (GraphBLAS::IndexType ix = 0; ix < NUM_NODES; ++ix)
    {
        if (levels.hasElement(ix))
        {
            BOOST_CHECK_EQUAL(levels.extractElement(ix),
                              answer[ix]);
        }
        else
        {
            BOOST_CHECK_EQUAL(0, answer[ix]);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
