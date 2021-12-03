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
 * DM20-0442
 */

#include <iostream>
#include <fstream>
#include <chrono>

#define GRAPHBLAS_DEBUG 1

#include <graphblas/graphblas.hpp>
#include <algorithms/triangle_count.hpp>
#include "Timer.hpp"

using namespace grb;

//****************************************************************************
IndexType read_edge_list(std::string const &pathname,
                         IndexArrayType &row_indices,
                         IndexArrayType &col_indices)
{
    std::ifstream infile(pathname);
    IndexType max_id = 0;
    uint64_t num_rows = 0;
    uint64_t src, dst;

    while (true)
    {
        infile >> src >> dst;
        if (infile.eof()) break;
        //std::cout << "Read: " << src << ", " << dst << std::endl;
        max_id = std::max(max_id, src);
        max_id = std::max(max_id, dst);

        //if (src > max_id) max_id = src;
        //if (dst > max_id) max_id = dst;

        row_indices.push_back(src);
        col_indices.push_back(dst);

        ++num_rows;
    }
    std::cout << "Read " << num_rows << " rows." << std::endl;
    std::cout << "#Nodes = " << (max_id + 1) << std::endl;

    return (max_id + 1);
}


//****************************************************************************
int main(int argc, char **argv)
{
    Timer<std::chrono::steady_clock, std::chrono::microseconds> my_timer;
    my_timer.start();

#if 0
    if (argc < 2)
    {
        std::cerr << "ERROR: too few arguments." << std::endl;
        std::cerr << "Usage: " << argv[0] << " <edge list file>" << std::endl;
        exit(1);
    }

    // Read the edgelist and create the tuple arrays
    std::string pathname(argv[1]);
    IndexArrayType iA, jA;

    IndexType const NUM_NODES(read_edge_list(pathname, iA, jA));
    using T = int32_t;
    using MatType = Matrix<T>; //, NWGraphTag>;
    using BoolMatType = Matrix<bool>;
    std::vector<T> v(iA.size(), 1);
    //std::vector<bool> bv(iA.size(), true);
    MatType A(NUM_NODES, NUM_NODES);
    //MatType B(NUM_NODES, NUM_NODES);
    //BoolMatType M(NUM_NODES, NUM_NODES);

    A.build(iA.begin(), jA.begin(), v.begin(), iA.size());
    //B.build(iA.begin(), jA.begin(), v.begin(), iA.size());
    //M.build(iA.begin(), jA.begin(), bv.begin(), iA.size());
    //MatType C(NUM_NODES, NUM_NODES);
#else
    IndexType const NUM_ROWS = 3;
    IndexType const NUM_COLS = 3;
    using T = double;

    // Note: size of dimensions require at ccnstruction
    Matrix<T> A(NUM_ROWS, NUM_COLS);
    Matrix<T> B(NUM_ROWS, NUM_COLS);
    Matrix<T> C(NUM_ROWS, NUM_COLS);

    // initialize matrices
    IndexArrayType Ai = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType Aj = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<T> Av = {3, 1, 4, 1, 5, 9, 2, 6, 7};

    // initialize matrices
    IndexArrayType Bi = {0, 0, 0, 1, 1,    2,    2};
    IndexArrayType Bj = {0, 1, 2, 0, 1,    0,    2};
    std::vector<T> Bv = {8, 6, 7, 5, 3,    9,    9};

    A.build(Ai.begin(), Aj.begin(), Av.begin(), Ai.size());
    B.build(Bi.begin(), Bj.begin(), Bv.begin(), Bi.size());

    print_matrix(std::cout, A, "Matrix A");
    print_matrix(std::cout, B, "Matrix B");

//  SparseMatrix C {
//    { { 0,  65 }, { 1, 21 }, { 2, 57 } },
//    { { 0, 114 }, { 1, 21 }, { 2, 88 } },
//    { { 0, 109 }, { 1, 30 }, { 2, 77 } },
//  };
#endif

    std::cout << "A: " << A.nvals() << std::endl;
    std::cout << "B: " << B.nvals() << std::endl;

#if 1
    mxm(C,
        NoMask(),
        NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, B);
    print_matrix(std::cout, C, "Matrix C");
#endif
    my_timer.stop();
    std::cout << "Elapsed time: " << my_timer.elapsed()/1000.0 << " sec."
              << std::endl;
    return 0;
}
