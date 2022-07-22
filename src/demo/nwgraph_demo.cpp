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
#include "read_edge_list.hpp"

using namespace grb;

//****************************************************************************
int test_lil()
{
    std::cout << "Testing LilSparseMatrix..." << std::endl;
    IndexType const NUM_ROWS = 3;
    IndexType const NUM_COLS = 3;
    using T = double;

    // Note: size of dimensions require at ccnstruction
    Matrix<T, grb::OrigTag> A(NUM_ROWS, NUM_COLS);
    Matrix<T, grb::OrigTag> B(NUM_ROWS, NUM_COLS);
    Matrix<bool, grb::OrigTag> M(NUM_ROWS, NUM_COLS);

    // initialize matrices
    IndexArrayType Ai = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType Aj = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<T> Av = {3, 1, 4, 1, 5, 9, 2, 6, 7};

    // initialize matrices
    IndexArrayType Bi = {0, 0, 0, 1, 1,    2,    2};
    IndexArrayType Bj = {0, 1, 2, 0, 1,    0,    2};
    std::vector<T> Bv = {8, 6, 7, 5, 3,    9,    9};

    // initialize matrices
    IndexArrayType    Mi = {0, 1, 1, 2};
    IndexArrayType    Mj = {1, 0, 2, 1};
    std::vector<bool> Mv = {true, true, true, true};

    A.build(Ai.begin(), Aj.begin(), Av.begin(), Ai.size());
    B.build(Bi.begin(), Bj.begin(), Bv.begin(), Bi.size());
    M.build(Mi.begin(), Mj.begin(), Mv.begin(), Mi.size());

    print_matrix(std::cout, A, "Matrix A");
    print_matrix(std::cout, B, "Matrix B");
    print_matrix(std::cout, M, "Matrix M");


    {
        Matrix<int, grb::OrigTag> C(NUM_ROWS, NUM_COLS);
        mxm(C,
            NoMask(),
            NoAccumulate(),
            ArithmeticSemiring<double>(),
            A, B);
        print_matrix(std::cout, C, "Matrix C = A +.* B");
//  Answer:
//  SparseMatrix C {
//    { { 0,  65 }, { 1, 21 }, { 2, 57 } },
//    { { 0, 114 }, { 1, 21 }, { 2, 88 } },
//    { { 0, 109 }, { 1, 30 }, { 2, 77 } },
//  };
    }

    {
        Matrix<int, grb::OrigTag> C(NUM_ROWS, NUM_COLS);
        mxm(C,
            M,
            NoAccumulate(),
            ArithmeticSemiring<double>(),
            A, B, grb::REPLACE);
        print_matrix(std::cout, C, "Masked Matrix C<M> = A +.* B");
//  Answer:
//  SparseMatrix C {
//    {             { 1, 21 }            },
//    { { 0, 114 },            { 2, 88 } },
//    {             { 1, 30 }            },
//  };
    }

    return 0;
}

//****************************************************************************
int test_nwgraph()
{
    std::cout << "Testing NWGraphMatrix..." << std::endl;
    IndexType const NUM_ROWS = 3;
    IndexType const NUM_COLS = 3;
    using T = double;

    // Note: size of dimensions require at ccnstruction
    Matrix<T, grb::NWGraphTag> A(NUM_ROWS, NUM_COLS);
    Matrix<T, grb::NWGraphTag> B(NUM_ROWS, NUM_COLS);
    Matrix<char, grb::NWGraphTag> M(NUM_ROWS, NUM_COLS);

    // initialize matrices
    IndexArrayType Ai = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType Aj = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<T> Av = {3, 1, 4, 1, 5, 9, 2, 6, 7};

    // initialize matrices
    IndexArrayType Bi = {0, 0, 0, 1, 1,    2,    2};
    IndexArrayType Bj = {0, 1, 2, 0, 1,    0,    2};
    std::vector<T> Bv = {8, 6, 7, 5, 3,    9,    9};

    // initialize matrices
    IndexArrayType    Mi = {0, 1, 1, 2};
    IndexArrayType    Mj = {1, 0, 2, 1};
    std::vector<bool> Mv = {true, true, true, true};
std::cerr << "A\n";
    A.build(Ai.begin(), Aj.begin(), Av.begin(), Ai.size());
std::cerr << "A\n";
    B.build(Bi.begin(), Bj.begin(), Bv.begin(), Bi.size());
std::cerr << "A\n";
    M.build(Mi.begin(), Mj.begin(), Mv.begin(), Mi.size());

std::cerr << "B\n";
    print_matrix(std::cout, A, "Matrix A");
    print_matrix(std::cout, B, "Matrix B");
    print_matrix(std::cout, M, "Matrix M");

//  Answer:
//  SparseMatrix C {
//    { { 0,  65 }, { 1, 21 }, { 2, 57 } },
//    { { 0, 114 }, { 1, 21 }, { 2, 88 } },
//    { { 0, 109 }, { 1, 30 }, { 2, 77 } },
//  };

    {
        Matrix<int, grb::NWGraphTag> C(NUM_ROWS, NUM_COLS);
        mxm(C,
            NoMask(),
            NoAccumulate(),
            ArithmeticSemiring<double>(),
            A, B);
        print_matrix(std::cout, C, "Matrix C = A +.* B");
    }

    {
        Matrix<int, grb::NWGraphTag> C(NUM_ROWS, NUM_COLS);
        mxm(C,
            M,
            NoAccumulate(),
            ArithmeticSemiring<double>(),
            A, B, grb::REPLACE);
        print_matrix(std::cout, C, "Masked Matrix C<M> = A +.* B");
    }

    return 0;
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
    IndexArrayType iA, jA;

    IndexType const NUM_NODES(read_edge_list(pathname, iA, jA));

    // Read the edgelist and create the tuple arrays
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
#endif

    test_lil();
    test_nwgraph();

    my_timer.stop();
    std::cout << "Elapsed time: " << my_timer.elapsed()/1000.0 << " sec."
              << std::endl;
    return 0;
}
