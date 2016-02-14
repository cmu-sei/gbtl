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

#include <algorithms/mis.hpp>
#include <algorithms/bfs.hpp>
#include <graphblas/graphblas.hpp>

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE mis_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(mis_suite)

//****************************************************************************
BOOST_AUTO_TEST_CASE(mis_test)
{
    double const INF(std::numeric_limits<double>::max());
    typedef graphblas::Matrix<double, graphblas::DirectedMatrixTag> GBMatrix;
    graphblas::IndexType const NUM_NODES(7);
    graphblas::IndexArrayType i = {0, 0, 1, 1, 2, 2, 3, 3, 3,
                                   4, 4, 5, 5, 5, 6, 6};
    graphblas::IndexArrayType j = {1, 2, 0, 3, 0, 5, 1, 4, 6,
                                   3, 5, 2, 4, 6, 3, 5};
    std::vector<double> v(i.size(), 1);
    GBMatrix graph(NUM_NODES, NUM_NODES, 0);
    graphblas::buildmatrix(graph, i, j, v);

    // Run MANY expeeriments to get different IS's
    for (unsigned int seed = 1000; seed < 1200; ++seed)
    {
        GBMatrix independent_set(NUM_NODES, 1, 0);
        algorithms::mis(graph, independent_set, float(seed));
        //graphblas::print_matrix(std::cout, independent_set,
        //                        "independent_set (flags)");

        // Get the set of vertex ID's
        graphblas::IndexArrayType result(
            algorithms::get_vertex_IDs(independent_set));
        //for (auto it = result.begin(); it != result.end(); ++it)
        //    std::cerr << *it << " ";
        //std::cerr << std::endl;

        // Check the result by performing bfs_level() from the independent_set
        // All levels should be 1 (if in IS) or 2 (if not in IS)
        /// @todo I was unable to use transpose(independent_set) for wavefront.
        GBMatrix levels(1, NUM_NODES, 0);
        GBMatrix isT(1, NUM_NODES, 0);
        graphblas::transpose(independent_set, isT);
        algorithms::bfs_level_masked(graph, isT, levels);

        //graphblas::print_matrix(std::cout, levels, "BFS levels from IS");
        for (graphblas::IndexType ix = 0; ix < NUM_NODES; ++ix)
        {
            double lvl = levels.get_value_at(0, ix);
            BOOST_CHECK((lvl < 3) && (lvl > 0));
            if (lvl == 1)
                BOOST_CHECK_EQUAL(isT.get_value_at(0, ix), 1.0);
            else
                BOOST_CHECK_EQUAL(isT.get_value_at(0, ix), 0.0);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
