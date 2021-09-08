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

    while (true)
    {
        infile >> src >> dst;
        if (infile.eof()) break;
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

    using T = int32_t;
    using MatType = Matrix<T>;
    using BoolMatType = Matrix<bool>;
    using VecType = Vector<T>;
    using BoolVecType = Vector<bool>;

    IndexType const NUM_NODES(read_edge_list(pathname, iA, jA, iB, jB));
    std::vector<T> vA(iA.size(), 1);
    std::vector<T> vB(iB.size(), 1);

    MatType A(NUM_NODES, NUM_NODES);
    MatType AT(NUM_NODES, NUM_NODES);
    MatType B(NUM_NODES, NUM_NODES);
    MatType BT(NUM_NODES, NUM_NODES);
    MatType C(NUM_NODES, NUM_NODES);
    MatType C1(NUM_NODES, NUM_NODES);
    MatType Ctmp(NUM_NODES, NUM_NODES);
    BoolMatType M(NUM_NODES, NUM_NODES);

    VecType u(NUM_NODES);
    VecType v(NUM_NODES);
    VecType w(NUM_NODES);
    VecType w1(NUM_NODES);
    VecType wtmp(NUM_NODES);
    BoolVecType m(NUM_NODES);

    A.build(iA.begin(), jA.begin(), vA.begin(), iA.size());
    AT.build(jA.begin(), iA.begin(), vA.begin(), iA.size());
    B.build(iB.begin(), jB.begin(), vB.begin(), iB.size());
    BT.build(jB.begin(), iB.begin(), vB.begin(), iB.size());
    for (IndexType ix = 0; ix < NUM_NODES; ++ix)
    {
        //std::cerr << ix << std::endl;
        if (distribution(generator) < 0.15)
            m.setElement(ix, true);
        if (distribution(generator) < 0.40)
            u.setElement(ix, 1);
        if (distribution(generator) < 0.40)
            v.setElement(ix, 1);
        for (IndexType jM = 0; jM < NUM_NODES; ++jM)
        {
            if (distribution(generator) < 0.001)
                M.setElement(ix, jM, true);
        }
    }

    std::cerr << "A.nvals = " << A.nvals() << std::endl;
    std::cerr << "B.nvals = " << B.nvals() << std::endl;
    std::cerr << "M.nvals = " << M.nvals() << std::endl;
    std::cerr << "u.nvals = " << u.nvals() << std::endl;
    std::cerr << "v.nvals = " << v.nvals() << std::endl;
    std::cerr << "m.nvals = " << m.nvals() << std::endl;
    T count(0);

    Timer<std::chrono::steady_clock, std::chrono::microseconds> my_timer;

    // warm up
    eWiseAdd (C,
              NoMask(),
              NoAccumulate(),
              Plus<double>(),
              A, B);

    int const NUM_TRIALS=10;
    double accum_time, min_time, max_time;
    //=====================================================
    // Perform element wise multiplication
    //=====================================================

    //===================
    // u.+v
    //===================
    std::cerr << "IMPLEMENTATION: u.+v" << std::endl;
    std::cerr << "Function                      \tmin\tavg\tmax\tnvals\tchecksum\n";

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseAdd (w, NoMask(), NoAccumulate(),
                  Plus<double>(),
                  u, v);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w := u.+v                    \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseAdd (w, NoMask(), Plus<double>(),
                  Plus<double>(),
                  u, v);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w := w + u.+v                \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseAdd (w, m, NoAccumulate(),
                  Plus<double>(),
                  u, v);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<m,merge> := u.+v           \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseAdd (w, m, NoAccumulate(),
                  Plus<double>(),
                  u, v, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<m,replace> := u.+v         \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseAdd (w, m, Plus<double>(),
                  Plus<double>(),
                  u, v);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<m,merge> := w + u.+v       \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseAdd (w, m, Plus<double>(),
                  Plus<double>(),
                  u, v, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<m,replace> := w + u.+v     \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseAdd (w, complement(m), NoAccumulate(),
                  Plus<double>(),
                  u, v);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<!m,merge> := u.+v          \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseAdd (w, complement(m), NoAccumulate(),
                  Plus<double>(),
                  u, v, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<!m,replace> := u.+v        \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseAdd (w, complement(m), Plus<double>(),
                  Plus<double>(),
                  u, v);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<!m,merge> := w + u.+v      \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseAdd (w, complement(m), Plus<double>(),
                  Plus<double>(),
                  u, v, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<!m,replace> := w + u.+v    \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----
    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseAdd (w, structure(m), NoAccumulate(),
                  Plus<double>(),
                  u, v);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<s(m),merge> := u.+v        \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseAdd (w, structure(m), NoAccumulate(),
                  Plus<double>(),
                  u, v, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<s(m),replace> := u.+v      \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseAdd (w, structure(m), Plus<double>(),
                  Plus<double>(),
                  u, v);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<s(m),merge> := w + u.+v    \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseAdd (w, structure(m), Plus<double>(),
                  Plus<double>(),
                  u, v, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<s(m),replace> := w + u.+v  \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseAdd (w, complement(structure(m)), NoAccumulate(),
                  Plus<double>(),
                  u, v);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<!s(m),merge> := u.+v       \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseAdd (w, complement(structure(m)), NoAccumulate(),
                  Plus<double>(),
                  u, v, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<!s(m),replace> := u.+v     \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseAdd (w, complement(structure(m)), Plus<double>(),
                  Plus<double>(),
                  u, v);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<!s(m),merge> := w + u.+v   \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseAdd (w, complement(structure(m)), Plus<double>(),
                  Plus<double>(),
                  u, v, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<!s(m),replace> := w + u.+v \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //===================
    // A.+B
    //===================
    std::cerr << "IMPLEMENTATION: A.+B" << std::endl;
    Ctmp.clear();

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, NoMask(), NoAccumulate(),
                  Plus<double>(),
                  A, B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C := A.+B                     \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, NoMask(), Plus<double>(),
                  Plus<double>(),
                  A, B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C := C + A.+B                 \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, M, NoAccumulate(),
                  Plus<double>(),
                  A, B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,merge> := A.+B            \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, M, NoAccumulate(),
                  Plus<double>(),
                  A, B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,replace> := A.+B          \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, M, Plus<double>(),
                  Plus<double>(),
                  A, B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,merge> := C + A.+B        \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, M, Plus<double>(),
                  Plus<double>(),
                  A, B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,replace> := C + A.+B      \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(M), NoAccumulate(),
                  Plus<double>(),
                  A, B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,merge> := A.+B           \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(M), NoAccumulate(),
                  Plus<double>(),
                  A, B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,replace> := A.+B         \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(M), Plus<double>(),
                  Plus<double>(),
                  A, B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,merge> := C + A.+B       \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(M), Plus<double>(),
                  Plus<double>(),
                  A, B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,replace> := C + A.+B     \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;

    //-------------------- structure only


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, structure(M), NoAccumulate(),
                  Plus<double>(),
                  A, B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),merge> := A.+B         \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, structure(M), NoAccumulate(),
                  Plus<double>(),
                  A, B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),replace> := A.+B       \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, structure(M), Plus<double>(),
                  Plus<double>(),
                  A, B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),merge> := C + A.+B     \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, structure(M), Plus<double>(),
                  Plus<double>(),
                  A, B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),replace> := C + A.+B   \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(structure(M)), NoAccumulate(),
                  Plus<double>(),
                  A, B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),merge> := A.+B        \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(structure(M)), NoAccumulate(),
                  Plus<double>(),
                  A, B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),replace> := A.+B      \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(structure(M)), Plus<double>(),
                  Plus<double>(),
                  A, B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),merge> := C + A.+B    \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(structure(M)), Plus<double>(),
                  Plus<double>(),
                  A, B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),replace> := C + A.+B  \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;

    //===================
    // A'*B
    //===================
    std::cerr << "OPTIMIZED IMPLEMENTATION: A'*B" << std::endl;
    Ctmp.clear();

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, NoMask(), NoAccumulate(),
                  Plus<double>(),
                  transpose(AT), B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C := A'.+B                    \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, NoMask(), Plus<double>(),
                  Plus<double>(),
                  transpose(AT), B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C := C + A'.+B                \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, M, NoAccumulate(),
                  Plus<double>(),
                  transpose(AT), B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,merge> := A'.+B           \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, M, NoAccumulate(),
                  Plus<double>(),
                  transpose(AT), B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,replace> := A.+B          \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, M, Plus<double>(),
                  Plus<double>(),
                  transpose(AT), B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,merge> := C + A'.+B       \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, M, Plus<double>(),
                  Plus<double>(),
                  transpose(AT), B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,replace> := C + A'.+B     \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(M), NoAccumulate(),
                  Plus<double>(),
                  transpose(AT), B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,merge> := A'.+B          \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(M), NoAccumulate(),
                  Plus<double>(),
                  transpose(AT), B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,replace> := A'.+B        \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(M), Plus<double>(),
                  Plus<double>(),
                  transpose(AT), B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,merge> := C + A'.+B      \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(M), Plus<double>(),
                  Plus<double>(),
                  transpose(AT), B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,replace> := C + A'.+B    \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;

    //-------------------- structure only


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, structure(M), NoAccumulate(),
                  Plus<double>(),
                  transpose(AT), B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),merge> := A'.+B        \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, structure(M), NoAccumulate(),
                  Plus<double>(),
                  transpose(AT), B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),replace> := A.+B       \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, structure(M), Plus<double>(),
                  Plus<double>(),
                  transpose(AT), B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),merge> := C + A'.+B    \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, structure(M), Plus<double>(),
                  Plus<double>(),
                  transpose(AT), B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),replace> := C + A'.+B  \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(structure(M)), NoAccumulate(),
                  Plus<double>(),
                  transpose(AT), B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),merge> := A'.+B       \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(structure(M)), NoAccumulate(),
                  Plus<double>(),
                  transpose(AT), B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),replace> := A'.+B     \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(structure(M)), Plus<double>(),
                  Plus<double>(),
                  transpose(AT), B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),merge> := C + A'.+B   \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(structure(M)), Plus<double>(),
                  Plus<double>(),
                  transpose(AT), B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),replace> := C + A'.+B \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;

    //===================
    // A*B'
    //===================
    std::cerr << "OPTIMIZED IMPLEMENTATION: A*B'" << std::endl;
    Ctmp.clear();

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, NoMask(), NoAccumulate(),
                  Plus<double>(),
                  A, transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C := A.+B'                    \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, NoMask(), Plus<double>(),
                  Plus<double>(),
                  A, transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C := C + A.+B'                \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, M, NoAccumulate(),
                  Plus<double>(),
                  A, transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,merge> := A.+B'           \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, M, NoAccumulate(),
                  Plus<double>(),
                  A, transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,replace> := A.+B'         \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, M, Plus<double>(),
                  Plus<double>(),
                  A, transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,merge> := C + A.+B'       \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, M, Plus<double>(),
                  Plus<double>(),
                  A, transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,replace> := C + A.+B'     \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(M), NoAccumulate(),
                  Plus<double>(),
                  A, transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,merge> := A.+B'          \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(M), NoAccumulate(),
                  Plus<double>(),
                  A, transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,replace> := A.+B'        \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(M), Plus<double>(),
                  Plus<double>(),
                  A, transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,merge> := C + A.+B'      \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(M), Plus<double>(),
                  Plus<double>(),
                  A, transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,replace> := C + A.+B'    \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;

    //-------------------- structure only


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, structure(M), NoAccumulate(),
                  Plus<double>(),
                  A, transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),merge> := A.+B'        \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, structure(M), NoAccumulate(),
                  Plus<double>(),
                  A, transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),replace> := A.+B'      \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, structure(M), Plus<double>(),
                  Plus<double>(),
                  A, transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),merge> := C + A.+B'    \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, structure(M), Plus<double>(),
                  Plus<double>(),
                  A, transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),replace> := C + A.+B'  \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(structure(M)), NoAccumulate(),
                  Plus<double>(),
                  A, transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),merge> := A.+B'       \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(structure(M)), NoAccumulate(),
                  Plus<double>(),
                  A, transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),replace> := A.+B'     \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(structure(M)), Plus<double>(),
                  Plus<double>(),
                  A, transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),merge> := C + A.+B'   \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(structure(M)), Plus<double>(),
                  Plus<double>(),
                  A, transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),replace> := C + A.+B' \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;

    //===================
    // A'*B'
    //===================
    std::cerr << "OPTIMIZED IMPLEMENTATION: A'*B'" << std::endl;
    Ctmp.clear();

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, NoMask(), NoAccumulate(),
                  Plus<double>(),
                  transpose(AT), transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C := A'.+B'                   \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, NoMask(), Plus<double>(),
                  Plus<double>(),
                  transpose(AT), transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C := C + A'.+B'               \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, M, NoAccumulate(),
                  Plus<double>(),
                  transpose(AT), transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,merge> := A'.+B'          \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, M, NoAccumulate(),
                  Plus<double>(),
                  transpose(AT), transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,replace> := A'.+B'        \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, M, Plus<double>(),
                  Plus<double>(),
                  transpose(AT), transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,merge> := C + A'.+B'      \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, M, Plus<double>(),
                  Plus<double>(),
                  transpose(AT), transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,replace> := C + A'.+B'    \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(M), NoAccumulate(),
                  Plus<double>(),
                  transpose(AT), transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,merge> := A'.+B'         \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(M), NoAccumulate(),
                  Plus<double>(),
                  transpose(AT), transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,replace> := A'.+B'       \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(M), Plus<double>(),
                  Plus<double>(),
                  transpose(AT), transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,merge> := C + A'.+B'     \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(M), Plus<double>(),
                  Plus<double>(),
                  transpose(AT), transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,replace> := C + A'.+B'   \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;

    //-------------------- structure only


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, structure(M), NoAccumulate(),
                  Plus<double>(),
                  transpose(AT), transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),merge> := A'.+B'       \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, structure(M), NoAccumulate(),
                  Plus<double>(),
                  transpose(AT), transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),replace> := A'.+B'     \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, structure(M), Plus<double>(),
                  Plus<double>(),
                  transpose(AT), transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),merge> := C + A'.+B'   \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, structure(M), Plus<double>(),
                  Plus<double>(),
                  transpose(AT), transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),replace> := C + A'.+B' \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(structure(M)), NoAccumulate(),
                  Plus<double>(),
                  transpose(AT), transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),merge> := A'.+B'      \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(structure(M)), NoAccumulate(),
                  Plus<double>(),
                  transpose(AT), transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),replace> := A'.+B'    \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(structure(M)), Plus<double>(),
                  Plus<double>(),
                  transpose(AT), transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),merge> := C + A'.+B'  \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseAdd (C, complement(structure(M)), Plus<double>(),
                  Plus<double>(),
                  transpose(AT), transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),replace> := C + A'.+B'\t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;

    return 0;
}
