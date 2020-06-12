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
#include <algorithms/k_truss.hpp>

using namespace grb;
using namespace algorithms;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE k_truss_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

//****************************************************************************
BOOST_AUTO_TEST_CASE(k_truss_test_basic)
{
    using T = int32_t;
    IndexType num_nodes = 5;
    IndexType num_edges = 6;
    IndexArrayType edge_num = {0, 1, 2, 3, 4, 5,  0, 1, 2, 3, 4, 5};
    IndexArrayType node_num = {0, 1, 0, 2, 0, 1,  1, 2, 3, 3, 2, 4};
    std::vector<T> val(edge_num.size(), 1);

    // build the incidence matrix
    Matrix<T> E(num_edges, num_nodes);
    E.build(edge_num.begin(), node_num.begin(), val.begin(), val.size());
    //print_matrix(std::cout, E, "Incidence");

    auto Eout3 = k_truss(E, 3);
    BOOST_CHECK_EQUAL(Eout3.nrows(), 5); // only removed one edge
    BOOST_CHECK_EQUAL(Eout3.ncols(), 5);

    auto Eout4 = k_truss(Eout3, 4);
    BOOST_CHECK_EQUAL(Eout4.nrows(), 0); // removed all edges
    BOOST_CHECK_EQUAL(Eout4.ncols(), 5);

    auto Etmp4 = k_truss(E, 4);
    BOOST_CHECK_EQUAL(Etmp4.nrows(), 0); // removed all edges
    BOOST_CHECK_EQUAL(Etmp4.ncols(), 5);

    // TODO test for correct contents
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(k_truss_test2)
{
    grb::IndexArrayType i = {
        0, 0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3, 3,
        4, 4, 4, 4, 4,
        5, 5, 5,
        6, 6, 6, 6,
        7, 7, 7, 7, 7,
        8, 8, 8, 8, 8,
        9, 9, 9, 9,
        10,10,10,10,10,
        11,11,11};

    grb::IndexArrayType j = {
        0, 1, 5, 6, 9,
        0, 1, 2, 4,
        1, 2, 3, 4,
        2, 3, 7, 8, 10,
        1, 2, 4, 6, 7,
        0, 5, 9,
        0, 4, 6, 9,
        3, 4, 7, 8, 10,
        3, 7, 8, 10, 11,
        0, 5, 6, 9,
        3, 7, 8, 10, 11,
        8, 10, 11};

    // create an incidence matrix from the data
    IndexType num_edges = 0;
    IndexType num_nodes = 0;
    IndexArrayType edge_array, node_array;
    // count edges in upper triangle of A
    for (IndexType ix = 0; ix < i.size(); ++ix)
    {
        if (i[ix] < j[ix])
        {
            edge_array.push_back(num_edges);
            node_array.push_back(i[ix]);
            edge_array.push_back(num_edges);
            node_array.push_back(j[ix]);
            ++num_edges;

            num_nodes = std::max(num_nodes, i[ix]);
            num_nodes = std::max(num_nodes, j[ix]);
        }
    }
    ++num_nodes;
    std::vector<int> v(edge_array.size(), 1);
    Matrix<int> E(num_edges, num_nodes);
    E.build(edge_array.begin(), node_array.begin(), v.begin(), v.size());
    //print_matrix(std::cout, E, "Incidence(test 2)");

    auto E3out = algorithms::k_truss(E, 3);
    //print_matrix(std::cout, E3out, "3-truss edges");
    BOOST_CHECK_EQUAL(E3out.nrows(), 16);
    BOOST_CHECK_EQUAL(E3out.ncols(), num_nodes);

    auto E4out = algorithms::k_truss(E3out, 4);
    //print_matrix(std::cout, E4out, "4-truss edges");
    BOOST_CHECK_EQUAL(E4out.nrows(), 6);
    BOOST_CHECK_EQUAL(E4out.ncols(), num_nodes);

    auto E5out = algorithms::k_truss(E4out, 5);
    //print_matrix(std::cout, E5out, "5-truss edges");
    BOOST_CHECK_EQUAL(E5out.nrows(), 0);
    BOOST_CHECK_EQUAL(E5out.ncols(), num_nodes);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(k_truss_test_peer_pressure1)
{
    IndexType num_nodes(5);
    IndexType num_edges(5);
    grb::IndexArrayType i = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4};
    grb::IndexArrayType j = {0, 1, 1, 2, 0, 2, 0, 3, 3, 4};
    std::vector<int> v(i.size(), 1);
    Matrix<double> E(num_edges, num_nodes);
    E.build(i.begin(), j.begin(), v.begin(), v.size());

    auto E3out = algorithms::k_truss(E, 3);
    //print_matrix(std::cout, E3out, "3-truss edges");
    BOOST_CHECK_EQUAL(E3out.nrows(), 3);
    BOOST_CHECK_EQUAL(E3out.ncols(), num_nodes);

    auto E4out = algorithms::k_truss(E3out, 4);
    //print_matrix(std::cout, E4out, "4-truss edges");
    BOOST_CHECK_EQUAL(E4out.nrows(), 0);
    BOOST_CHECK_EQUAL(E4out.ncols(), num_nodes);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(k_truss_test_peer_pressure2)
{
    grb::IndexArrayType i = {0, 0, 0, 0,
                             1, 1, 1, 1,
                             2, 2, 2, 2,
                             3, 3, 3, 3,
                             4, 4, 4, 4,
                             5, 5, 5, 5, 5,
                             6, 6, 6,
                             7, 7, 7, 7};
    grb::IndexArrayType j = {0, 2, 3, 6,
                             1, 2, 3, 7,
                             0, 2, 4, 6,
                             0, 1, 3, 5,
                             0, 2, 4, 6,
                             1, 3, 5, 6, 7,
                             0, 4, 6,
                             1, 3, 5, 7};

    // create an incidence matrix from the data
    IndexType num_edges = 0;
    IndexType num_nodes = 0;
    IndexArrayType edge_array, node_array;
    // count edges in upper triangle of A
    for (IndexType ix = 0; ix < i.size(); ++ix)
    {
        if (i[ix] < j[ix])
        {
            edge_array.push_back(num_edges);
            node_array.push_back(i[ix]);
            edge_array.push_back(num_edges);
            node_array.push_back(j[ix]);
            ++num_edges;

            num_nodes = std::max(num_nodes, i[ix]);
            num_nodes = std::max(num_nodes, j[ix]);
        }
    }
    ++num_nodes;
    std::vector<int> v(edge_array.size(), 1);

    Matrix<int> E(num_edges, num_nodes);
    E.build(edge_array.begin(), node_array.begin(), v.begin(), v.size());
    //print_matrix(std::cout, E, "Incidence");

    auto Eout3 = algorithms::k_truss(E, 3);
    //print_matrix(std::cout, Eout3, "3-truss edges");
    BOOST_CHECK_EQUAL(num_nodes, Eout3.ncols());
    BOOST_CHECK_EQUAL(5, Eout3.nrows());

    auto Eout4 = algorithms::k_truss(Eout3, 4);
    //print_matrix(std::cout, Eout4, "4-truss edges");
    BOOST_CHECK_EQUAL(num_nodes, Eout4.ncols());
    BOOST_CHECK_EQUAL(0, Eout4.nrows());
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(k_truss2_test_basic)
{
    using T = int32_t;
    IndexType num_nodes = 5;
    IndexArrayType i = {0, 1, 0, 2, 0, 1, 1, 2, 3, 3, 2, 4};
    IndexArrayType j = {1, 2, 3, 3, 2, 4, 0, 1, 0, 2, 0, 1};
    std::vector<T> val(i.size(), 1);

    // build the adjacency matrix
    Matrix<T> A(num_nodes, num_nodes);
    A.build(i.begin(), j.begin(), val.begin(), val.size());
    //print_matrix(std::cout, A, "Adjacency");

    auto Aout3 = k_truss2(A, 3);
    BOOST_CHECK_EQUAL(Aout3.nvals(), 10); // only removed one edge
    //print_matrix(std::cout, Aout3, "Adjacency (k=3)");

    auto Aout4 = k_truss2(Aout3, 4);
    BOOST_CHECK_EQUAL(Aout4.nvals(), 0); // removed all edges
    //print_matrix(std::cout, Aout4, "Adjacency (k=4)");

    auto Atmp4 = k_truss2(A, 4);
    BOOST_CHECK_EQUAL(Atmp4.nvals(), 0); // removed all edges

    // TODO test for correct contents
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(k_truss2_test2)
{
    grb::IndexArrayType i = {
        0, 0, 0, 0,
        1,    1, 1,
        2,    2, 2,
        3,    3, 3, 3,
        4, 4,    4, 4,
        5,    5,
        6, 6,    6,
        7, 7,    7, 7,
        8, 8,    8, 8,
        9, 9, 9,
        10,10,10,   10,
        11,11   };

    grb::IndexArrayType j = {
        1, 5, 6, 9,
        0,    2, 4,
        1,    3, 4,
        2,    7, 8, 10,
        1, 2,    6, 7,
        0,    9,
        0, 4,    9,
        3, 4,    8, 10,
        3, 7,    10, 11,
        0, 5, 6,
        3, 7, 8,     11,
        8, 10    };

    // create an incidence matrix from the data
    IndexType num_edges = i.size();
    IndexType num_nodes = 12;
    std::vector<int> val(num_edges, 1);
    Matrix<int> A(num_nodes, num_nodes);
    A.build(i.begin(), j.begin(), val.begin(), num_edges);

    auto A3out = algorithms::k_truss2(A, 3);
    //print_matrix(std::cout, A3out, "3-truss edges");
    BOOST_CHECK_EQUAL(A3out.nvals(), 32);

    auto A4out = algorithms::k_truss2(A3out, 4);
    //print_matrix(std::cout, A4out, "4-truss edges");
    BOOST_CHECK_EQUAL(A4out.nvals(), 12);

    auto A5out = algorithms::k_truss2(A4out, 5);
    //print_matrix(std::cout, A5out, "5-truss edges");
    BOOST_CHECK_EQUAL(A5out.nvals(), 0);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(k_truss2_test_peer_pressure1)
{
    IndexType num_nodes(5);

    grb::IndexArrayType i = {0, 1, 2, 1, 2, 0, 3, 0, 4, 3};
    grb::IndexArrayType j = {1, 0, 1, 2, 0, 2, 0, 3, 3, 4};
    std::vector<int> v(i.size(), 1);
    Matrix<double> A(num_nodes, num_nodes);
    A.build(i.begin(), j.begin(), v.begin(), v.size());

    auto A3out = algorithms::k_truss2(A, 3);
    //print_matrix(std::cout, A3out, "3-truss edges");
    BOOST_CHECK_EQUAL(A3out.nvals(), 6);

    auto A4out = algorithms::k_truss2(A3out, 4);
    //print_matrix(std::cout, A4out, "4-truss edges");
    BOOST_CHECK_EQUAL(A4out.nvals(), 0);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(k_truss2_test_peer_pressure2)
{

    grb::IndexArrayType i = {0, 0, 0, 0,
                             1, 1, 1, 1,
                             2, 2, 2, 2,
                             3, 3, 3, 3,
                             4, 4, 4, 4,
                             5, 5, 5, 5, 5,
                             6, 6, 6,
                             7, 7, 7, 7};
    grb::IndexArrayType j = {0, 2, 3, 6,
                             1, 2, 3, 7,
                             0, 2, 4, 6,
                             0, 1, 3, 5,
                             0, 2, 4, 6,
                             1, 3, 5, 6, 7,
                             0, 4, 6,
                             1, 3, 5, 7};

    // create an adjacency matrix equivalent to the incidence matrix produced in a previous test
    IndexType num_nodes = 0;
    IndexArrayType iv, jv;
    // count edges in upper triangle of A
    for (IndexType ix = 0; ix < i.size(); ++ix)
    {
        if (i[ix] < j[ix])
        {
            iv.push_back(i[ix]); jv.push_back(j[ix]);
            iv.push_back(j[ix]); jv.push_back(i[ix]);  //reverse link

            num_nodes = std::max(num_nodes, i[ix]);
            num_nodes = std::max(num_nodes, j[ix]);
        }
    }
    ++num_nodes;
    IndexType num_edges(iv.size());
    std::vector<int> v(num_edges, 1);

    Matrix<int> A(num_nodes, num_nodes);
    A.build(iv.begin(), jv.begin(), v.begin(), num_edges);
    //print_matrix(std::cout, A, "graph");

    auto Aout3 = algorithms::k_truss2(A, 3);
    //print_matrix(std::cout, Aout3, "3-truss edges");
    BOOST_CHECK_EQUAL(10, Aout3.nvals());

    auto Aout4 = algorithms::k_truss2(Aout3, 4);
    //print_matrix(std::cout, Aout4, "4-truss edges");
    BOOST_CHECK_EQUAL(0, Aout4.nvals());
}

BOOST_AUTO_TEST_SUITE_END()
