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
#include <random>

#define GRAPHBLAS_DEBUG 1

#include <graphblas/graphblas.hpp>
#include <algorithms/triangle_count.hpp>
#include "Timer.hpp"

using namespace grb;

std::default_random_engine  generator;
std::uniform_real_distribution<double> distribution;

//****************************************************************************
IndexType read_edge_list(std::string const &pathname,
                         IndexArrayType &Arow_indices,
                         IndexArrayType &Acol_indices,
                         IndexArrayType &Brow_indices,
                         IndexArrayType &Bcol_indices)
{
    std::ifstream infile(pathname);
    IndexType max_id = 0;
    uint64_t num_rows = 0;
    uint64_t src, dst;

    while (infile)
    {
        infile >> src >> dst;
        //std::cout << "Read: " << src << ", " << dst << std::endl;
        max_id = std::max(max_id, src);
        max_id = std::max(max_id, dst);

        //if (src > max_id) max_id = src;
        //if (dst > max_id) max_id = dst;

        if (distribution(generator) < 0.85)
        {
            Arow_indices.push_back(src);
            Acol_indices.push_back(dst);
        }

        if (distribution(generator) < 0.85)
        {
            Brow_indices.push_back(src);
            Bcol_indices.push_back(dst);
        }

        ++num_rows;
    }
    std::cout << "Read " << num_rows << " rows." << std::endl;
    std::cout << "#Nodes = " << (max_id + 1) << std::endl;

    return (max_id + 1);
}


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
    IndexArrayType iA, jA, iB, jB;

    IndexType const NUM_NODES(read_edge_list(pathname, iA, jA, iB, jB));

    using T = int32_t;
    using MatType = Matrix<T>;
    using BoolMatType = Matrix<bool>;
    std::vector<T> vA(iA.size(), 1);
    std::vector<T> vB(iB.size(), 1);
    MatType A(NUM_NODES, NUM_NODES);
    MatType B(NUM_NODES, NUM_NODES);
    BoolMatType M(NUM_NODES, NUM_NODES);

    A.build(iA.begin(), jA.begin(), vA.begin(), iA.size());
    B.build(iB.begin(), jB.begin(), vB.begin(), iB.size());
    for (IndexType iM = 0; iM < NUM_NODES; ++iM)
    {
        for (IndexType jM = 0; jM < NUM_NODES; ++jM)
        {
            if (distribution(generator) < 0.15)
                M.setElement(iM, jM, true);
        }
    }
    std::cout << "A.nvals = " << A.nvals() << std::endl;
    std::cout << "B.nvals = " << B.nvals() << std::endl;
    std::cout << "M.nvals = " << M.nvals() << std::endl;
    T count(0);

    Timer<std::chrono::steady_clock, std::chrono::microseconds> my_timer;
    MatType C(NUM_NODES, NUM_NODES);
    eWiseMult(C,
              NoMask(),
              NoAccumulate(),
              Times<double>(),
              A, B);

    //===================
    // A*B
    //===================
    std::cout << "IMPLEMENTATION: A.*B" << std::endl;
    C.clear();
    my_timer.start();
    eWiseMult(C, NoMask(), NoAccumulate(),
              Times<double>(),
              A, B);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C := A.*B                \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, NoMask(), Plus<double>(),
              Times<double>(),
              A, B);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C := C + A.*B            \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, M, NoAccumulate(),
              Times<double>(),
              A, B);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<M,merge> := A.*B       \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, M, NoAccumulate(),
              Times<double>(),
              A, B, REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<M,replace> := A.*B     \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, M, Plus<double>(),
              Times<double>(),
              A, B);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<M,merge> := C + A.*B   \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, M, Plus<double>(),
              Times<double>(),
              A, B, REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<M,replace> := C + A.*B \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(M), NoAccumulate(),
              Times<double>(),
              A, B);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!M,merge> := A.*B      \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(M), NoAccumulate(),
              Times<double>(),
              A, B, REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!M,replace> := A.*B    \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(M), Plus<double>(),
              Times<double>(),
              A, B);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!M,merge> := C + A.*B  \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(M), Plus<double>(),
              Times<double>(),
              A, B, REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!M,replace> := C + A.*B\tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    //-------------------- structure only

    my_timer.start();
    eWiseMult(C, structure(M), NoAccumulate(),
              Times<double>(),
              A, B);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<s(M),merge> := A.*B    \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, structure(M), NoAccumulate(),
              Times<double>(),
              A, B, REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<s(M),replace> := A.*B  \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, structure(M), Plus<double>(),
              Times<double>(),
              A, B);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<s(M),merge> := C + A.*B   \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, structure(M), Plus<double>(),
              Times<double>(),
              A, B, REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<s(M),replace> := C + A.*B \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(structure(M)), NoAccumulate(),
              Times<double>(),
              A, B);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!s(M),merge> := A.*B      \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(structure(M)), NoAccumulate(),
              Times<double>(),
              A, B, REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!s(M),replace> := A.*B    \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(structure(M)), Plus<double>(),
              Times<double>(),
              A, B);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!s(M),merge> := C + A.*B  \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(structure(M)), Plus<double>(),
              Times<double>(),
              A, B, REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!s(M),replace> := C + A.*B\tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    //===================
    // A'*B
    //===================
    std::cout << "OPTIMIZED IMPLEMENTATION: A'*B" << std::endl;
    C.clear();
    my_timer.start();
    eWiseMult(C, NoMask(), NoAccumulate(),
              Times<double>(),
              transpose(A), B);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C := A'.*B                \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, NoMask(), Plus<double>(),
              Times<double>(),
              transpose(A), B);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C := C + A'.*B            \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, M, NoAccumulate(),
              Times<double>(),
              transpose(A), B);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<M,merge> := A'.*B       \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, M, NoAccumulate(),
              Times<double>(),
              transpose(A), B, REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<M,replace> := A.*B      \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, M, Plus<double>(),
              Times<double>(),
              transpose(A), B);
    my_timer.stop();
     reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
   std::cout << "C<M,merge> := C + A'.*B   \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, M, Plus<double>(),
              Times<double>(),
              transpose(A), B, REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<M,replace> := C + A'.*B \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(M), NoAccumulate(),
              Times<double>(),
              transpose(A), B);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!M,merge> := A'.*B      \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(M), NoAccumulate(),
              Times<double>(),
              transpose(A), B, REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!M,replace> := A'.*B    \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(M), Plus<double>(),
              Times<double>(),
              transpose(A), B);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!M,merge> := C + A'.*B  \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(M), Plus<double>(),
              Times<double>(),
              transpose(A), B, REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!M,replace> := C + A'.*B\tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    //-------------------- structure only

    my_timer.start();
    eWiseMult(C, structure(M), NoAccumulate(),
              Times<double>(),
              transpose(A), B);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<s(M),merge> := A'.*B       \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, structure(M), NoAccumulate(),
              Times<double>(),
              transpose(A), B, REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<s(M),replace> := A.*B      \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, structure(M), Plus<double>(),
              Times<double>(),
              transpose(A), B);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<s(M),merge> := C + A'.*B   \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, structure(M), Plus<double>(),
              Times<double>(),
              transpose(A), B, REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<s(M),replace> := C + A'.*B \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(structure(M)), NoAccumulate(),
              Times<double>(),
              transpose(A), B);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!s(M),merge> := A'.*B      \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(structure(M)), NoAccumulate(),
              Times<double>(),
              transpose(A), B, REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!s(M),replace> := A'.*B    \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(structure(M)), Plus<double>(),
              Times<double>(),
              transpose(A), B);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!s(M),merge> := C + A'.*B  \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(structure(M)), Plus<double>(),
              Times<double>(),
              transpose(A), B, REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!s(M),replace> := C + A'.*B\tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    //===================
    // A*B'
    //===================
    std::cout << "OPTIMIZED IMPLEMENTATION: A*B'" << std::endl;
    C.clear();
    my_timer.start();
    eWiseMult(C, NoMask(), NoAccumulate(),
              Times<double>(),
              A, transpose(B));
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C := A.*B'                \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, NoMask(), Plus<double>(),
              Times<double>(),
              A, transpose(B));
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C := C + A.*B'            \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, M, NoAccumulate(),
              Times<double>(),
              A, transpose(B));
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<M,merge> := A.*B'       \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, M, NoAccumulate(),
              Times<double>(),
              A, transpose(B), REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<M,replace> := A.*B'     \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, M, Plus<double>(),
              Times<double>(),
              A, transpose(B));
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<M,merge> := C + A.*B'   \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, M, Plus<double>(),
              Times<double>(),
              A, transpose(B), REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<M,replace> := C + A.*B' \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(M), NoAccumulate(),
              Times<double>(),
              A, transpose(B));
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!M,merge> := A.*B'      \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(M), NoAccumulate(),
              Times<double>(),
              A, transpose(B), REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!M,replace> := A.*B'    \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(M), Plus<double>(),
              Times<double>(),
              A, transpose(B));
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!M,merge> := C + A.*B'  \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(M), Plus<double>(),
              Times<double>(),
              A, transpose(B), REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!M,replace> := C + A.*B'\tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    //-------------------- structure only

    my_timer.start();
    eWiseMult(C, structure(M), NoAccumulate(),
              Times<double>(),
              A, transpose(B));
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<s(M),merge> := A.*B'       \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, structure(M), NoAccumulate(),
              Times<double>(),
              A, transpose(B), REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<s(M),replace> := A.*B'     \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, structure(M), Plus<double>(),
              Times<double>(),
              A, transpose(B));
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<s(M),merge> := C + A.*B'   \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, structure(M), Plus<double>(),
              Times<double>(),
              A, transpose(B), REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<s(M),replace> := C + A.*B' \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(structure(M)), NoAccumulate(),
              Times<double>(),
              A, transpose(B));
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!s(M),merge> := A.*B'      \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(structure(M)), NoAccumulate(),
              Times<double>(),
              A, transpose(B), REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!s(M),replace> := A.*B'    \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(structure(M)), Plus<double>(),
              Times<double>(),
              A, transpose(B));
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!s(M),merge> := C + A.*B'  \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(structure(M)), Plus<double>(),
              Times<double>(),
              A, transpose(B), REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!s(M),replace> := C + A.*B'\tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    //===================
    // A'*B'
    //===================
    std::cout << "OPTIMIZED IMPLEMENTATION: A'*B'" << std::endl;
    C.clear();
    my_timer.start();
    eWiseMult(C, NoMask(), NoAccumulate(),
              Times<double>(),
              transpose(A), transpose(B));
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C := A'.*B'                \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, NoMask(), Plus<double>(),
              Times<double>(),
              transpose(A), transpose(B));
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C := C + A'.*B'            \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, M, NoAccumulate(),
              Times<double>(),
              transpose(A), transpose(B));
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<M,merge> := A'.*B'       \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, M, NoAccumulate(),
              Times<double>(),
              transpose(A), transpose(B), REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<M,replace> := A'.*B'     \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, M, Plus<double>(),
              Times<double>(),
              transpose(A), transpose(B));
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<M,merge> := C + A'.*B'   \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, M, Plus<double>(),
              Times<double>(),
              transpose(A), transpose(B), REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<M,replace> := C + A'.*B' \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(M), NoAccumulate(),
              Times<double>(),
              transpose(A), transpose(B));
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!M,merge> := A'.*B'      \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(M), NoAccumulate(),
              Times<double>(),
              transpose(A), transpose(B), REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!M,replace> := A'.*B'    \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(M), Plus<double>(),
              Times<double>(),
              transpose(A), transpose(B));
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!M,merge> := C + A'.*B'  \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(M), Plus<double>(),
              Times<double>(),
              transpose(A), transpose(B), REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!M,replace> := C + A'.*B'\tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    //-------------------- structure only

    my_timer.start();
    eWiseMult(C, structure(M), NoAccumulate(),
              Times<double>(),
              transpose(A), transpose(B));
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<s(M),merge> := A'.*B'       \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, structure(M), NoAccumulate(),
              Times<double>(),
              transpose(A), transpose(B), REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<s(M),replace> := A'.*B'     \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, structure(M), Plus<double>(),
              Times<double>(),
              transpose(A), transpose(B));
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<s(M),merge> := C + A'.*B'   \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, structure(M), Plus<double>(),
              Times<double>(),
              transpose(A), transpose(B), REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<s(M),replace> := C + A'.*B' \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(structure(M)), NoAccumulate(),
              Times<double>(),
              transpose(A), transpose(B));
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!s(M),merge> := A'.*B'      \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(structure(M)), NoAccumulate(),
              Times<double>(),
              transpose(A), transpose(B), REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!s(M),replace> := A'.*B'    \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(structure(M)), Plus<double>(),
              Times<double>(),
              transpose(A), transpose(B));
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!s(M),merge> := C + A'.*B'  \tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    my_timer.start();
    eWiseMult(C, complement(structure(M)), Plus<double>(),
              Times<double>(),
              transpose(A), transpose(B), REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cout << "C<!s(M),replace> := C + A'.*B'\tC.nvals = " << C.nvals()
              << "\t" << my_timer.elapsed() << " usec\treduce = " << count << std::endl;

    return 0;
}
