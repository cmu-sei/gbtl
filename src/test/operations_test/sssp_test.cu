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
#include <cusp/print.h>
#include <cusp/io/matrix_market.h>
#include <cusp/io/dimacs.h>

#define readmm(MTX, FILE_NAME) cusp::io::read_matrix_market_file(MTX, FILE_NAME)
#define readdi(MTX, FILE_NAME) cusp::io::read_dimacs_file(MTX, FILE_NAME)

using namespace graphblas;

template<typename MatrixT,
         typename PathMatrixT>
void sssp(MatrixT const     &graph,
          PathMatrixT const &start,
          PathMatrixT       &paths)
{
    using T = typename MatrixT::ScalarType;
    using MinAccum =
        graphblas::math::Accum<T, graphblas::math::ArithmeticMin<T> >;

    paths = start;

    graphblas::IndexType rows, cols, prows, pcols, rrows, rcols;
    graph.get_shape(rows, cols);
    start.get_shape(prows, pcols);
    paths.get_shape(rrows, rcols);

    if ((rows != pcols) || (prows != rrows) || (pcols != rcols))
    {
        throw graphblas::DimensionException();
    }

    /// @todo why num_rows iterations?
    for (graphblas::IndexType k = 0; k < rows; ++k)
    {
        //cusp::print(paths.getBackendMatrix());
        graphblas::mxm<MatrixT, PathMatrixT, PathMatrixT,
                       graphblas::MinPlusSemiring<T>,
                       MinAccum>(paths, graph, paths);
        //std::cout << "Iteration " << k << std::endl;
        //pretty_print_matrix(std::cout, paths);
        //std::cout << std::endl;
    }
    // paths holds return value
}

//hardcoded graph
#if 0
int main()
{
    unsigned int const INF(std::numeric_limits<unsigned int>::max());

    graphblas::IndexType const NUM_NODES(9);
    graphblas::VectorIndexType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    graphblas::VectorIndexType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<unsigned int> v(i.size(), 1);
    Matrix<unsigned int, DirectedMatrixTag> G_tn(NUM_NODES, NUM_NODES, i.size());
    G_tn.set_zero(INF);
    buildmatrix(G_tn, i.begin(), j.begin(), v.begin(), i.size());

    auto identity_9x9 =
        graphblas::identity<graphblas::Matrix<unsigned int, DirectedMatrixTag> >(
            NUM_NODES, INF, 0);

    Matrix<unsigned int, DirectedMatrixTag> G_tn_res(NUM_NODES, NUM_NODES,
            NUM_NODES*NUM_NODES);

    G_tn_res.set_zero(INF);

    sssp(G_tn, identity_9x9, G_tn_res);
}
#endif
int main(int argc, char ** argv){
    typedef unsigned int T;
    unsigned int const INF(std::numeric_limits<unsigned int>::max());
    typedef graphblas::Matrix<T, graphblas::DirectedMatrixTag> GrBMatrix;

    GrBMatrix G_tn;

    G_tn.set_zero(INF);
    //cusp::io::read_matrix_market_file(G_tn.getBackendMatrix(), argv[1]);


    if (atoi(argv[2]) == 1){
        readdi(G_tn.getBackendMatrix(),
                std::string(argv[1]));
    } else {
        readmm(G_tn.getBackendMatrix(),
                std::string(argv[1]));
    }

    graphblas::IndexType NUM_NODES;
    G_tn.get_shape(NUM_NODES, NUM_NODES);

    auto identity_nxn =
        graphblas::identity<graphblas::Matrix<unsigned int, DirectedMatrixTag> >(
            NUM_NODES, INF, 0);

    Matrix<unsigned int, DirectedMatrixTag> G_tn_res(NUM_NODES, NUM_NODES,
            NUM_NODES*NUM_NODES);

    G_tn_res.set_zero(INF);

    graphblas::start_timer();
    sssp(G_tn, identity_nxn, G_tn_res);
    graphblas::stop_timer();
    std::cout<<graphblas::get_elapsed_time()<<std::endl;
}
