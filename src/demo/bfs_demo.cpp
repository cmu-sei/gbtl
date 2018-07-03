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

#include <graphblas/graphblas.hpp>
#include <algorithms/bfs.hpp>

//****************************************************************************
int main()
{
    // FA Tech Note graph
    //
    //    {-, -, -, 1, -, -, -, -, -},
    //    {-, -, -, 1, -, -, 1, -, -},
    //    {-, -, -, -, 1, 1, 1, -, 1},
    //    {1, 1, -, -, 1, -, 1, -, -},
    //    {-, -, 1, 1, -, -, -, -, 1},
    //    {-, -, 1, -, -, -, -, -, -},
    //    {-, 1, 1, 1, -, -, -, -, -},
    //    {-, -, -, -, -, -, -, -, -},
    //    {-, -, 1, -, 1, -, -, -, -};

    /// @todo change scalar type to unsigned int or GraphBLAS::IndexType
    using T = GraphBLAS::IndexType;
    typedef GraphBLAS::Matrix<T, GraphBLAS::DirectedMatrixTag> GBMatrix;
    //T const INF(std::numeric_limits<T>::max());

    GraphBLAS::IndexType const NUM_NODES = 9;
    GraphBLAS::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    GraphBLAS::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<T> v(i.size(), 1);

    GBMatrix G_tn(NUM_NODES, NUM_NODES);
    G_tn.build(i.begin(), j.begin(), v.begin(), i.size());
    GraphBLAS::print_matrix(std::cout, G_tn, "Graph adjacency matrix:");

    // Perform a single BFS
    GraphBLAS::Vector<T> parent_list(NUM_NODES);
    GraphBLAS::Vector<T> root(NUM_NODES);
    root.setElement(3, 1);
    algorithms::bfs(G_tn, root, parent_list);
    GraphBLAS::print_vector(std::cout, parent_list,
                            "Parent list for root at vertex 3");

    // Perform BFS from all roots simultaneously (should the value be 0?)
    //auto roots = GraphBLAS::identity<GBMatrix>(NUM_NODES, INF, 0);
    GBMatrix roots(NUM_NODES, NUM_NODES);
    GraphBLAS::IndexArrayType ii, jj, vv;
    for (GraphBLAS::IndexType ix = 0; ix < NUM_NODES; ++ix)
    {
        ii.push_back(ix);
        jj.push_back(ix);
        vv.push_back(1);
    }
    roots.build(ii, jj, vv);

    GBMatrix G_tn_res(NUM_NODES, NUM_NODES);

    algorithms::bfs_batch(G_tn, roots, G_tn_res);

    GraphBLAS::print_matrix(std::cout, G_tn_res,
                            "Parents for each root (by rows):");

    return 0;
}
