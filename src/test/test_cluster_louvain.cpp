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

#include <graphblas/graphblas.hpp>
#include <algorithms/cluster_louvain.hpp>

using namespace grb;
using namespace algorithms;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE cluster_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

//****************************************************************************
BOOST_AUTO_TEST_CASE(cluster_test_louvain)
{
    std::cout << "============== Louvain ================" << std::endl;
    grb::IndexArrayType i_m1 = {0, 0, 0, 0,
                                1, 1, 1, 1,
                                2, 2, 2, 2,
                                3, 3, 3, 3,
                                4, 4, 4, 4,
                                5, 5, 5, 5, 5,
                                6, 6, 6,
                                7, 7, 7, 7};
    grb::IndexArrayType j_m1 = {0, 2, 3, 6,
                                1, 2, 3, 7,
                                0, 2, 4, 6,
                                0, 1, 3, 5,
                                0, 2, 4, 6,
                                1, 3, 5, 6, 7,
                                0, 4, 6,
                                1, 3, 5, 7};
    std::vector<double> v_m1 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    Matrix<double> m1(8, 8);
    m1.build(i_m1, j_m1, v_m1);

    auto ans = algorithms::louvain_cluster(m1);

    auto cluster_assignments = get_louvain_cluster_assignments(ans);
    grb::print_vector(std::cout, cluster_assignments, "CLUSTER ASSIGNMENTS");

    BOOST_CHECK_EQUAL(cluster_assignments.extractElement(0),
                      cluster_assignments.extractElement(2));
    BOOST_CHECK_EQUAL(cluster_assignments.extractElement(0),
                      cluster_assignments.extractElement(4));
    BOOST_CHECK_EQUAL(cluster_assignments.extractElement(0),
                      cluster_assignments.extractElement(6));

    BOOST_CHECK_EQUAL(cluster_assignments.extractElement(1),
                      cluster_assignments.extractElement(3));
    BOOST_CHECK_EQUAL(cluster_assignments.extractElement(1),
                      cluster_assignments.extractElement(5));
    BOOST_CHECK_EQUAL(cluster_assignments.extractElement(1),
                      cluster_assignments.extractElement(7));

    BOOST_CHECK(cluster_assignments.extractElement(0) !=
                cluster_assignments.extractElement(1));
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(cluster_test_louvain_masked)
{
    std::cout << "============== Louvain ================" << std::endl;
    grb::IndexArrayType i_m1 = {0, 0, 0, 0,
                                1, 1, 1, 1,
                                2, 2, 2, 2,
                                3, 3, 3, 3,
                                4, 4, 4, 4,
                                5, 5, 5, 5, 5,
                                6, 6, 6,
                                7, 7, 7, 7};
    grb::IndexArrayType j_m1 = {0, 2, 3, 6,
                                1, 2, 3, 7,
                                0, 2, 4, 6,
                                0, 1, 3, 5,
                                0, 2, 4, 6,
                                1, 3, 5, 6, 7,
                                0, 4, 6,
                                1, 3, 5, 7};
    std::vector<double> v_m1 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    Matrix<double> m1(8, 8);
    m1.build(i_m1, j_m1, v_m1);

    auto ans = algorithms::louvain_cluster_masked(m1);

    auto cluster_assignments = get_louvain_cluster_assignments(ans);
    grb::print_vector(std::cout, cluster_assignments, "CLUSTER ASSIGNMENTS");

    BOOST_CHECK_EQUAL(cluster_assignments.extractElement(0),
                      cluster_assignments.extractElement(2));
    BOOST_CHECK_EQUAL(cluster_assignments.extractElement(0),
                      cluster_assignments.extractElement(4));
    BOOST_CHECK_EQUAL(cluster_assignments.extractElement(0),
                      cluster_assignments.extractElement(6));

    BOOST_CHECK_EQUAL(cluster_assignments.extractElement(1),
                      cluster_assignments.extractElement(3));
    BOOST_CHECK_EQUAL(cluster_assignments.extractElement(1),
                      cluster_assignments.extractElement(5));
    BOOST_CHECK_EQUAL(cluster_assignments.extractElement(1),
                      cluster_assignments.extractElement(7));

    BOOST_CHECK(cluster_assignments.extractElement(0) !=
                cluster_assignments.extractElement(1));
}

BOOST_AUTO_TEST_SUITE_END()
