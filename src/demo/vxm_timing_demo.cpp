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
#include "Timer.hpp"

using namespace grb;

//****************************************************************************
IndexType read_edge_list(std::string const &pathname,
                         IndexArrayType    &row_indices,
                         IndexArrayType    &col_indices)
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
    if (argc < 2)
    {
        std::cerr << "ERROR: too few arguments." << std::endl;
        std::cerr << "Usage: " << argv[0] << " <edge list file>" << std::endl;
        exit(1);
    }

    // Read the edgelist and create the tuple arrays
    std::string pathname(argv[1]);
    IndexArrayType iA, jA, iu;

    IndexType const NUM_NODES(read_edge_list(pathname, iA, jA));

    using T = int32_t;
    using MatType = Matrix<T>;
    using VecType = Vector<T>;
    using BoolVecType = Vector<bool>;
    std::vector<T> v(iA.size(), 1);
    std::vector<bool> bv(iA.size(), true);
    MatType A(NUM_NODES, NUM_NODES);
    MatType AT(NUM_NODES, NUM_NODES);
    VecType u(NUM_NODES);
    VecType w(NUM_NODES);
    VecType w1(NUM_NODES);
    BoolVecType M(NUM_NODES);

    A.build(iA.begin(), jA.begin(), v.begin(), iA.size());
    transpose(AT, NoMask(), NoAccumulate(), A);

    std::default_random_engine  generator;
    std::uniform_real_distribution<double> distribution;
    for (IndexType iu = 0; iu < NUM_NODES; ++iu)
    {
        if (distribution(generator) < 0.15)
            M.setElement(iu, true);
        if (distribution(generator) < 0.1)
            u.setElement(iu, 1);
    }

    std::cout << "Running algorithm(s)... M.nvals = " << M.nvals() << std::endl;
    std::cout << "u.nvals = " << u.nvals() << std::endl;
    T count(0);

    Timer<std::chrono::steady_clock, std::chrono::microseconds> my_timer;

    // warm up
    vxm(w, NoMask(), NoAccumulate(), ArithmeticSemiring<double>(), u, A);

    //=====================================================
    // Perform matrix vector multiplies
    //=====================================================

    //===================
    // u'*A
    //===================
    std::cout << "IMPLEMENTATION: u'*A" << std::endl;
    w.clear();
    my_timer.start();
    vxm(w, NoMask(), NoAccumulate(),
        ArithmeticSemiring<double>(),
        u, AT);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cout << "w' := u'+.*A                : " << my_timer.elapsed()
              << " usec, w.nvals = " << w.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w, NoMask(), Plus<double>(),
        ArithmeticSemiring<double>(),
        u, AT);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cout << "w' := w' + u'+.*A            : " << my_timer.elapsed()
              << " usec, w.nvals = " << w.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w, M, NoAccumulate(),
        ArithmeticSemiring<double>(),
        u, AT);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cout << "w'<m',merge> := u'+.*A       : " << my_timer.elapsed()
              << " usec, w.nvals = " << w.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w, M, NoAccumulate(),
        ArithmeticSemiring<double>(),
        u, AT, REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cout << "w'<m',replace> := u'+.*A     : " << my_timer.elapsed()
              << " usec, w.nvals = " << w.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w, M, Plus<double>(),
        ArithmeticSemiring<double>(),
        u, AT);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cout << "w'<m',merge> := w' + u'+.*A   : " << my_timer.elapsed()
              << " usec, w.nvals = " << w.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w, M, Plus<double>(),
        ArithmeticSemiring<double>(),
        u, AT, REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cout << "w'<m',replace> := w' + u'+.*A : " << my_timer.elapsed()
              << " usec, w.nvals = " << w.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w, complement(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        u, AT);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cout << "w'<!m',merge> := u'+.*A      : " << my_timer.elapsed()
              << " usec, w.nvals = " << w.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w, complement(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        u, AT, REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cout << "w'<!m',replace> := u'+.*A    : " << my_timer.elapsed()
              << " usec, w.nvals = " << w.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w, complement(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        u, AT);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cout << "w'<!m',merge> := w' + u'+.*A  : " << my_timer.elapsed()
              << " usec, w.nvals = " << w.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w, complement(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        u, AT, REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cout << "w'<!m',replace> := w' + u'+.*A: " << my_timer.elapsed()
              << " usec, w.nvals = " << w.nvals()
              << " reduce = " << count << std::endl;

    //----
    my_timer.start();
    vxm(w, structure(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        u, AT);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cout << "w'<s(m'),merge> := u'+.*A    : " << my_timer.elapsed()
              << " usec, w.nvals = " << w.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w, structure(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        u, AT, REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cout << "w'<s(m'),replace> := u'+.*A  : " << my_timer.elapsed()
              << " usec, w.nvals = " << w.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w, structure(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        u, AT);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cout << "w'<s(m'),merge> := w' + u'+.*A   : " << my_timer.elapsed()
              << " usec, w.nvals = " << w.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w, structure(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        u, AT, REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cout << "w'<s(m'),replace> := w' + u'+.*A : " << my_timer.elapsed()
              << " usec, w.nvals = " << w.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w, complement(structure(M)), NoAccumulate(),
        ArithmeticSemiring<double>(),
        u, AT);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cout << "w'<!s(m'),merge> := u'+.*A      : " << my_timer.elapsed()
              << " usec, w.nvals = " << w.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w, complement(structure(M)), NoAccumulate(),
        ArithmeticSemiring<double>(),
        u, AT, REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cout << "w'<!s(m'),replace> := u'+.*A    : " << my_timer.elapsed()
              << " usec, w.nvals = " << w.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w, complement(structure(M)), Plus<double>(),
        ArithmeticSemiring<double>(),
        u, AT);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cout << "w'<!s(m'),merge> := w' + u'+.*A  : " << my_timer.elapsed()
              << " usec, w.nvals = " << w.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w, complement(structure(M)), Plus<double>(),
        ArithmeticSemiring<double>(),
        u, AT, REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cout << "w'<!s(m'),replace> := w' + u'+.*A: " << my_timer.elapsed()
              << " usec, w.nvals = " << w.nvals()
              << " reduce = " << count << std::endl;

    //===================
    // u'*A'
    //===================
    std::cout << "IMPLEMENTATION: u'*A'" << std::endl;
    w1.clear();
    my_timer.start();
    vxm(w1, NoMask(), NoAccumulate(),
        ArithmeticSemiring<double>(),
        u, transpose(A));
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w1);
    std::cout << "w' := u'+.*A'                : " << my_timer.elapsed()
              << " usec, w1.nvals = " << w1.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w1, NoMask(), Plus<double>(),
        ArithmeticSemiring<double>(),
        u, transpose(A));
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w1);
    std::cout << "w' := w' + u'+.*A'            : " << my_timer.elapsed()
              << " usec, w1.nvals = " << w1.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w1, M, NoAccumulate(),
        ArithmeticSemiring<double>(),
        u, transpose(A));
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w1);
    std::cout << "w'<m',merge> := u'+.*A'       : " << my_timer.elapsed()
              << " usec, w1.nvals = " << w1.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w1, M, NoAccumulate(),
        ArithmeticSemiring<double>(),
        u, transpose(A), REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w1);
    std::cout << "w'<m',replace> := u'+.*A      : " << my_timer.elapsed()
              << " usec, w1.nvals = " << w1.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w1, M, Plus<double>(),
        ArithmeticSemiring<double>(),
        u, transpose(A));
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w1);
    std::cout << "w'<m',merge> := w' + u'+.*A'   : " << my_timer.elapsed()
              << " usec, w1.nvals = " << w1.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w1, M, Plus<double>(),
        ArithmeticSemiring<double>(),
        u, transpose(A), REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w1);
    std::cout << "w'<m',replace> := w' + u'+.*A' : " << my_timer.elapsed()
              << " usec, w1.nvals = " << w1.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w1, complement(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        u, transpose(A));
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w1);
    std::cout << "w'<!m',merge> := u'+.*A'      : " << my_timer.elapsed()
              << " usec, w1.nvals = " << w1.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w1, complement(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        u, transpose(A), REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w1);
    std::cout << "w'<!m',replace> := u'+.*A'    : " << my_timer.elapsed()
              << " usec, w1.nvals = " << w1.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w1, complement(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        u, transpose(A));
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w1);
    std::cout << "w'<!m',merge> := w' + u'+.*A'  : " << my_timer.elapsed()
              << " usec, w1.nvals = " << w1.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w1, complement(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        u, transpose(A), REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w1);
    std::cout << "w'<!m',replace> := w' + u'+.*A': " << my_timer.elapsed()
              << " usec, w1.nvals = " << w1.nvals()
              << " reduce = " << count << std::endl;

    //-----

    my_timer.start();
    vxm(w1, structure(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        u, transpose(A));
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w1);
    std::cout << "w'<s(m'),merge> := u'+.*A'       : " << my_timer.elapsed()
              << " usec, w1.nvals = " << w1.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w1, structure(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        u, transpose(A), REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w1);
    std::cout << "w'<s(m'),replace> := u'+.*A'      : " << my_timer.elapsed()
              << " usec, w1.nvals = " << w1.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w1, structure(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        u, transpose(A));
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w1);
    std::cout << "w'<s(m'),merge> := w' + u'+.*A'   : " << my_timer.elapsed()
              << " usec, w1.nvals = " << w1.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w1, structure(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        u, transpose(A), REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w1);
    std::cout << "w'<s(m'),replace> := w' + u'+.*A' : " << my_timer.elapsed()
              << " usec, w1.nvals = " << w1.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w1, complement(structure(M)), NoAccumulate(),
        ArithmeticSemiring<double>(),
        u, transpose(A));
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w1);
    std::cout << "w'<!s(m'),merge> := u'+.*A'      : " << my_timer.elapsed()
              << " usec, w1.nvals = " << w1.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w1, complement(structure(M)), NoAccumulate(),
        ArithmeticSemiring<double>(),
        u, transpose(A), REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w1);
    std::cout << "w'<!s(m'),replace> := u'+.*A'    : " << my_timer.elapsed()
              << " usec, w1.nvals = " << w1.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w1, complement(structure(M)), Plus<double>(),
        ArithmeticSemiring<double>(),
        u, transpose(A));
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w1);
    std::cout << "w'<!s(m'),merge> := w' + u'+.*A'  : " << my_timer.elapsed()
              << " usec, w1.nvals = " << w1.nvals()
              << " reduce = " << count << std::endl;

    my_timer.start();
    vxm(w1, complement(structure(M)), Plus<double>(),
        ArithmeticSemiring<double>(),
        u, transpose(A), REPLACE);
    my_timer.stop();
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w1);
    std::cout << "w'<!s(m'),replace> := w' + u'+.*A': " << my_timer.elapsed()
              << " usec, w1.nvals = " << w1.nvals()
              << " reduce = " << count << std::endl;

    bool passed = (w == w1);
    std::cout << "Results " << (passed ? "PASSED" : "FAILED") << std::endl;
    return 0;
}
