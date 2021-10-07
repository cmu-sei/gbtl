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

#include "Timer.hpp"
#include <graphblas/graphblas.hpp>
#include "read_edge_list.hpp"

using namespace grb;


//****************************************************************************
int main(int argc, char **argv)
{
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
    using MatType = Matrix<T>;
    using BoolMatType = Matrix<bool>;
    std::vector<T> v(iA.size(), 1);
    std::vector<bool> bv(iA.size(), true);
    MatType A(NUM_NODES, NUM_NODES);
    MatType B(NUM_NODES, NUM_NODES);
    BoolMatType M(NUM_NODES, NUM_NODES);

    A.build(iA.begin(), jA.begin(), v.begin(), iA.size());
    B.build(iA.begin(), jA.begin(), v.begin(), iA.size());
    M.build(iA.begin(), jA.begin(), bv.begin(), iA.size());

    std::cout << "Running algorithm(s)... nvals = " << M.nvals() << std::endl;

    Timer<std::chrono::steady_clock, std::chrono::microseconds> my_timer;
    MatType C(NUM_NODES, NUM_NODES);
    mxm(C,
        NoMask(),
        NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, B);

    //=====================================================
    // Perform 18 different matrix multiplies for each of 4
    // different combinations of input transposes.
    //=====================================================

    //===================
    // A*B
    //===================
    std::cout << "OPTIMIZED IMPLEMENTATION: A*B" << std::endl;
    C.clear();
    my_timer.start();
    mxm(C, NoMask(), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, B);
    my_timer.stop();
    std::cout << "C := A+.*B                : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, NoMask(), Plus<double>(),
        ArithmeticSemiring<double>(),
        A, B);
    my_timer.stop();
    std::cout << "C := C + A+.*B            : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, B);
    my_timer.stop();
    std::cout << "C<M,merge> := A+.*B       : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, B, REPLACE);
    my_timer.stop();
    std::cout << "C<M,replace> := A+.*B     : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, Plus<double>(),
        ArithmeticSemiring<double>(),
        A, B);
    my_timer.stop();
    std::cout << "C<M,merge> := C + A+.*B   : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, Plus<double>(),
        ArithmeticSemiring<double>(),
        A, B, REPLACE);
    my_timer.stop();
    std::cout << "C<M,replace> := C + A+.*B : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, B);
    my_timer.stop();
    std::cout << "C<!M,merge> := A+.*B      : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, B, REPLACE);
    my_timer.stop();
    std::cout << "C<!M,replace> := A+.*B    : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        A, B);
    my_timer.stop();
    std::cout << "C<!M,merge> := C + A+.*B  : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        A, B, REPLACE);
    my_timer.stop();
    std::cout << "C<!M,replace> := C + A+.*B: " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    //-------------------- structure only

    my_timer.start();
    mxm(C, structure(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, B);
    my_timer.stop();
    std::cout << "C<s(M),merge> := A+.*B    : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, structure(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, B, REPLACE);
    my_timer.stop();
    std::cout << "C<s(M),replace> := A+.*B  : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, structure(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        A, B);
    my_timer.stop();
    std::cout << "C<s(M),merge> := C + A+.*B   : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, structure(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        A, B, REPLACE);
    my_timer.stop();
    std::cout << "C<s(M),replace> := C + A+.*B : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, B);
    my_timer.stop();
    std::cout << "C<!s(M),merge> := A+.*B      : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, B, REPLACE);
    my_timer.stop();
    std::cout << "C<!s(M),replace> := A+.*B    : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), Plus<double>(),
        ArithmeticSemiring<double>(),
        A, B);
    my_timer.stop();
    std::cout << "C<!s(M),merge> := C + A+.*B  : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), Plus<double>(),
        ArithmeticSemiring<double>(),
        A, B, REPLACE);
    my_timer.stop();
    std::cout << "C<!s(M),replace> := C + A+.*B: " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    //===================
    // A'*B
    //===================
    std::cout << "OPTIMIZED IMPLEMENTATION: A'*B" << std::endl;
    C.clear();
    my_timer.start();
    mxm(C, NoMask(), NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), B);
    my_timer.stop();
    std::cout << "C := A'+.*B                : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, NoMask(), Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), B);
    my_timer.stop();
    std::cout << "C := C + A'+.*B            : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), B);
    my_timer.stop();
    std::cout << "C<M,merge> := A'+.*B       : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), B, REPLACE);
    my_timer.stop();
    std::cout << "C<M,replace> := A+.*B      : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), B);
    my_timer.stop();
    std::cout << "C<M,merge> := C + A'+.*B   : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), B, REPLACE);
    my_timer.stop();
    std::cout << "C<M,replace> := C + A'+.*B : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), B);
    my_timer.stop();
    std::cout << "C<!M,merge> := A'+.*B      : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), B, REPLACE);
    my_timer.stop();
    std::cout << "C<!M,replace> := A'+.*B    : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), B);
    my_timer.stop();
    std::cout << "C<!M,merge> := C + A'+.*B  : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), B, REPLACE);
    my_timer.stop();
    std::cout << "C<!M,replace> := C + A'+.*B: " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    //-------------------- structure only

    my_timer.start();
    mxm(C, structure(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), B);
    my_timer.stop();
    std::cout << "C<s(M),merge> := A'+.*B       : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, structure(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), B, REPLACE);
    my_timer.stop();
    std::cout << "C<s(M),replace> := A+.*B      : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, structure(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), B);
    my_timer.stop();
    std::cout << "C<s(M),merge> := C + A'+.*B   : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, structure(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), B, REPLACE);
    my_timer.stop();
    std::cout << "C<s(M),replace> := C + A'+.*B : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), B);
    my_timer.stop();
    std::cout << "C<!s(M),merge> := A'+.*B      : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), B, REPLACE);
    my_timer.stop();
    std::cout << "C<!s(M),replace> := A'+.*B    : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), B);
    my_timer.stop();
    std::cout << "C<!s(M),merge> := C + A'+.*B  : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), B, REPLACE);
    my_timer.stop();
    std::cout << "C<!s(M),replace> := C + A'+.*B: " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    //===================
    // A*B'
    //===================
    std::cout << "OPTIMIZED IMPLEMENTATION: A*B'" << std::endl;
    C.clear();
    my_timer.start();
    mxm(C, NoMask(), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, transpose(B));
    my_timer.stop();
    std::cout << "C := A+.*B'                : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, NoMask(), Plus<double>(),
        ArithmeticSemiring<double>(),
        A, transpose(B));
    my_timer.stop();
    std::cout << "C := C + A+.*B'            : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, transpose(B));
    my_timer.stop();
    std::cout << "C<M,merge> := A+.*B'       : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<M,replace> := A+.*B'     : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, Plus<double>(),
        ArithmeticSemiring<double>(),
        A, transpose(B));
    my_timer.stop();
    std::cout << "C<M,merge> := C + A+.*B'   : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, Plus<double>(),
        ArithmeticSemiring<double>(),
        A, transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<M,replace> := C + A+.*B' : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, transpose(B));
    my_timer.stop();
    std::cout << "C<!M,merge> := A+.*B'      : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<!M,replace> := A+.*B'    : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        A, transpose(B));
    my_timer.stop();
    std::cout << "C<!M,merge> := C + A+.*B'  : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        A, transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<!M,replace> := C + A+.*B': " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    //-------------------- structure only

    my_timer.start();
    mxm(C, structure(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, transpose(B));
    my_timer.stop();
    std::cout << "C<s(M),merge> := A+.*B'       : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, structure(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<s(M),replace> := A+.*B'     : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, structure(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        A, transpose(B));
    my_timer.stop();
    std::cout << "C<s(M),merge> := C + A+.*B'   : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, structure(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        A, transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<s(M),replace> := C + A+.*B' : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, transpose(B));
    my_timer.stop();
    std::cout << "C<!s(M),merge> := A+.*B'      : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<!s(M),replace> := A+.*B'    : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), Plus<double>(),
        ArithmeticSemiring<double>(),
        A, transpose(B));
    my_timer.stop();
    std::cout << "C<!s(M),merge> := C + A+.*B'  : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), Plus<double>(),
        ArithmeticSemiring<double>(),
        A, transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<!s(M),replace> := C + A+.*B': " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    //===================
    // A'*B'
    //===================
    std::cout << "OPTIMIZED IMPLEMENTATION: A'*B'" << std::endl;
    C.clear();
    my_timer.start();
    mxm(C, NoMask(), NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B));
    my_timer.stop();
    std::cout << "C := A'+.*B'                : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, NoMask(), Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B));
    my_timer.stop();
    std::cout << "C := C + A'+.*B'            : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B));
    my_timer.stop();
    std::cout << "C<M,merge> := A'+.*B'       : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<M,replace> := A'+.*B'     : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B));
    my_timer.stop();
    std::cout << "C<M,merge> := C + A'+.*B'   : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<M,replace> := C + A'+.*B' : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B));
    my_timer.stop();
    std::cout << "C<!M,merge> := A'+.*B'      : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<!M,replace> := A'+.*B'    : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B));
    my_timer.stop();
    std::cout << "C<!M,merge> := C + A'+.*B'  : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<!M,replace> := C + A'+.*B': " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    //-------------------- structure only

    my_timer.start();
    mxm(C, structure(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B));
    my_timer.stop();
    std::cout << "C<s(M),merge> := A'+.*B'       : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, structure(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<s(M),replace> := A'+.*B'     : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, structure(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B));
    my_timer.stop();
    std::cout << "C<s(M),merge> := C + A'+.*B'   : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, structure(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<s(M),replace> := C + A'+.*B' : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B));
    my_timer.stop();
    std::cout << "C<!s(M),merge> := A'+.*B'      : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<!s(M),replace> := A'+.*B'    : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B));
    my_timer.stop();
    std::cout << "C<!s(M),merge> := C + A'+.*B'  : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<!s(M),replace> := C + A'+.*B': " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    return 0;
}
