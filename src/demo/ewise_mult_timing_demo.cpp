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

#include "Timer.hpp"
#include <graphblas/graphblas.hpp>
#include "read_edge_list.hpp"

using namespace grb;

std::default_random_engine  generator;
std::uniform_real_distribution<double> distribution;

//****************************************************************************
void separate_indices(IndexArrayType const &row_indices,
                      IndexArrayType const &col_indices,
                      IndexArrayType &Arow_indices,
                      IndexArrayType &Acol_indices,
                      IndexArrayType &Brow_indices,
                      IndexArrayType &Bcol_indices)
{
    for (size_t ix = 0; ix < row_indices.size(); ++ix)
    {
        if (distribution(generator) < 0.85)
        {
            Arow_indices.push_back(row_indices[ix]);
            Acol_indices.push_back(col_indices[ix]);
        }

        if (distribution(generator) < 0.85)
        {
            Brow_indices.push_back(row_indices[ix]);
            Bcol_indices.push_back(col_indices[ix]);
        }
    }
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
    IndexArrayType ii, jj, iA, jA, iB, jB;

    using T = int32_t;
    using MatType = Matrix<T>;
    using BoolMatType = Matrix<bool>;
    using VecType = Vector<T>;
    using BoolVecType = Vector<bool>;

    IndexType const NUM_NODES(read_edge_list(pathname, ii, jj));
    separate_indices(ii, jj, iA, jA, iB, jB);
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
    eWiseMult(C,
              NoMask(),
              NoAccumulate(),
              Times<double>(),
              A, B);

    int const NUM_TRIALS=10;
    double accum_time, min_time, max_time;
    //=====================================================
    // Perform element wise multiplication
    //=====================================================

    //===================
    // u.*v
    //===================
    std::cerr << "IMPLEMENTATION: u.*v" << std::endl;
    std::cerr << "Function                      \tmin\tavg\tmax\tnvals\tchecksum\n";

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseMult(w, NoMask(), NoAccumulate(),
                  Times<double>(),
                  u, v);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w := u.*v                    \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseMult(w, NoMask(), Plus<double>(),
                  Times<double>(),
                  u, v);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w := w + u.*v                \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseMult(w, m, NoAccumulate(),
                  Times<double>(),
                  u, v);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<m,merge> := u.*v           \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseMult(w, m, NoAccumulate(),
                  Times<double>(),
                  u, v, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<m,replace> := u.*v         \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseMult(w, m, Plus<double>(),
                  Times<double>(),
                  u, v);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<m,merge> := w + u.*v       \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseMult(w, m, Plus<double>(),
                  Times<double>(),
                  u, v, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<m,replace> := w + u.*v     \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseMult(w, complement(m), NoAccumulate(),
                  Times<double>(),
                  u, v);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<!m,merge> := u.*v          \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseMult(w, complement(m), NoAccumulate(),
                  Times<double>(),
                  u, v, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<!m,replace> := u.*v        \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseMult(w, complement(m), Plus<double>(),
                  Times<double>(),
                  u, v);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<!m,merge> := w + u.*v      \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseMult(w, complement(m), Plus<double>(),
                  Times<double>(),
                  u, v, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<!m,replace> := w + u.*v    \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----
    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseMult(w, structure(m), NoAccumulate(),
                  Times<double>(),
                  u, v);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<s(m),merge> := u.*v        \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseMult(w, structure(m), NoAccumulate(),
                  Times<double>(),
                  u, v, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<s(m),replace> := u.*v      \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseMult(w, structure(m), Plus<double>(),
                  Times<double>(),
                  u, v);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<s(m),merge> := w + u.*v    \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseMult(w, structure(m), Plus<double>(),
                  Times<double>(),
                  u, v, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<s(m),replace> := w + u.*v  \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseMult(w, complement(structure(m)), NoAccumulate(),
                  Times<double>(),
                  u, v);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<!s(m),merge> := u.*v       \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseMult(w, complement(structure(m)), NoAccumulate(),
                  Times<double>(),
                  u, v, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<!s(m),replace> := u.*v     \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseMult(w, complement(structure(m)), Plus<double>(),
                  Times<double>(),
                  u, v);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<!s(m),merge> := w + u.*v   \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        w = wtmp;
        my_timer.start();
        eWiseMult(w, complement(structure(m)), Plus<double>(),
                  Times<double>(),
                  u, v, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    wtmp = w;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    std::cerr << "w<!s(m),replace> := w + u.*v \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << w.nvals() << "\t" << count << std::endl;

    //===================
    // A.*B
    //===================
    std::cerr << "IMPLEMENTATION: A.*B" << std::endl;
    Ctmp.clear();

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, NoMask(), NoAccumulate(),
                  Times<double>(),
                  A, B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C := A.*B                     \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, NoMask(), Plus<double>(),
                  Times<double>(),
                  A, B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C := C + A.*B                 \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, M, NoAccumulate(),
                  Times<double>(),
                  A, B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,merge> := A.*B            \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, M, NoAccumulate(),
                  Times<double>(),
                  A, B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,replace> := A.*B          \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, M, Plus<double>(),
                  Times<double>(),
                  A, B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,merge> := C + A.*B        \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, M, Plus<double>(),
                  Times<double>(),
                  A, B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,replace> := C + A.*B      \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(M), NoAccumulate(),
                  Times<double>(),
                  A, B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,merge> := A.*B           \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(M), NoAccumulate(),
                  Times<double>(),
                  A, B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,replace> := A.*B         \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(M), Plus<double>(),
                  Times<double>(),
                  A, B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,merge> := C + A.*B       \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(M), Plus<double>(),
                  Times<double>(),
                  A, B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,replace> := C + A.*B     \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;

    //-------------------- structure only


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, structure(M), NoAccumulate(),
                  Times<double>(),
                  A, B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),merge> := A.*B         \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, structure(M), NoAccumulate(),
                  Times<double>(),
                  A, B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),replace> := A.*B       \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, structure(M), Plus<double>(),
                  Times<double>(),
                  A, B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),merge> := C + A.*B     \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, structure(M), Plus<double>(),
                  Times<double>(),
                  A, B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),replace> := C + A.*B   \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(structure(M)), NoAccumulate(),
                  Times<double>(),
                  A, B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),merge> := A.*B        \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(structure(M)), NoAccumulate(),
                  Times<double>(),
                  A, B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),replace> := A.*B      \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(structure(M)), Plus<double>(),
                  Times<double>(),
                  A, B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),merge> := C + A.*B    \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(structure(M)), Plus<double>(),
                  Times<double>(),
                  A, B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),replace> := C + A.*B  \t"
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
        eWiseMult(C, NoMask(), NoAccumulate(),
                  Times<double>(),
                  transpose(AT), B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C := A'.*B                    \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, NoMask(), Plus<double>(),
                  Times<double>(),
                  transpose(AT), B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C := C + A'.*B                \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, M, NoAccumulate(),
                  Times<double>(),
                  transpose(AT), B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,merge> := A'.*B           \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, M, NoAccumulate(),
                  Times<double>(),
                  transpose(AT), B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,replace> := A.*B          \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, M, Plus<double>(),
                  Times<double>(),
                  transpose(AT), B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,merge> := C + A'.*B       \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, M, Plus<double>(),
                  Times<double>(),
                  transpose(AT), B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,replace> := C + A'.*B     \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(M), NoAccumulate(),
                  Times<double>(),
                  transpose(AT), B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,merge> := A'.*B          \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(M), NoAccumulate(),
                  Times<double>(),
                  transpose(AT), B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,replace> := A'.*B        \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(M), Plus<double>(),
                  Times<double>(),
                  transpose(AT), B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,merge> := C + A'.*B      \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(M), Plus<double>(),
                  Times<double>(),
                  transpose(AT), B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,replace> := C + A'.*B    \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;

    //-------------------- structure only


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, structure(M), NoAccumulate(),
                  Times<double>(),
                  transpose(AT), B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),merge> := A'.*B        \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, structure(M), NoAccumulate(),
                  Times<double>(),
                  transpose(AT), B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),replace> := A.*B       \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, structure(M), Plus<double>(),
                  Times<double>(),
                  transpose(AT), B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),merge> := C + A'.*B    \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, structure(M), Plus<double>(),
                  Times<double>(),
                  transpose(AT), B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),replace> := C + A'.*B  \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(structure(M)), NoAccumulate(),
                  Times<double>(),
                  transpose(AT), B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),merge> := A'.*B       \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(structure(M)), NoAccumulate(),
                  Times<double>(),
                  transpose(AT), B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),replace> := A'.*B     \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(structure(M)), Plus<double>(),
                  Times<double>(),
                  transpose(AT), B);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),merge> := C + A'.*B   \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(structure(M)), Plus<double>(),
                  Times<double>(),
                  transpose(AT), B, REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),replace> := C + A'.*B \t"
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
        eWiseMult(C, NoMask(), NoAccumulate(),
                  Times<double>(),
                  A, transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C := A.*B'                    \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, NoMask(), Plus<double>(),
                  Times<double>(),
                  A, transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C := C + A.*B'                \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, M, NoAccumulate(),
                  Times<double>(),
                  A, transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,merge> := A.*B'           \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, M, NoAccumulate(),
                  Times<double>(),
                  A, transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,replace> := A.*B'         \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, M, Plus<double>(),
                  Times<double>(),
                  A, transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,merge> := C + A.*B'       \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, M, Plus<double>(),
                  Times<double>(),
                  A, transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,replace> := C + A.*B'     \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(M), NoAccumulate(),
                  Times<double>(),
                  A, transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,merge> := A.*B'          \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(M), NoAccumulate(),
                  Times<double>(),
                  A, transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,replace> := A.*B'        \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(M), Plus<double>(),
                  Times<double>(),
                  A, transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,merge> := C + A.*B'      \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(M), Plus<double>(),
                  Times<double>(),
                  A, transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,replace> := C + A.*B'    \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;

    //-------------------- structure only


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, structure(M), NoAccumulate(),
                  Times<double>(),
                  A, transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),merge> := A.*B'        \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, structure(M), NoAccumulate(),
                  Times<double>(),
                  A, transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),replace> := A.*B'      \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, structure(M), Plus<double>(),
                  Times<double>(),
                  A, transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),merge> := C + A.*B'    \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, structure(M), Plus<double>(),
                  Times<double>(),
                  A, transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),replace> := C + A.*B'  \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(structure(M)), NoAccumulate(),
                  Times<double>(),
                  A, transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),merge> := A.*B'       \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(structure(M)), NoAccumulate(),
                  Times<double>(),
                  A, transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),replace> := A.*B'     \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(structure(M)), Plus<double>(),
                  Times<double>(),
                  A, transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),merge> := C + A.*B'   \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(structure(M)), Plus<double>(),
                  Times<double>(),
                  A, transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),replace> := C + A.*B' \t"
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
        eWiseMult(C, NoMask(), NoAccumulate(),
                  Times<double>(),
                  transpose(AT), transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C := A'.*B'                   \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, NoMask(), Plus<double>(),
                  Times<double>(),
                  transpose(AT), transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C := C + A'.*B'               \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, M, NoAccumulate(),
                  Times<double>(),
                  transpose(AT), transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,merge> := A'.*B'          \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, M, NoAccumulate(),
                  Times<double>(),
                  transpose(AT), transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,replace> := A'.*B'        \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, M, Plus<double>(),
                  Times<double>(),
                  transpose(AT), transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,merge> := C + A'.*B'      \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, M, Plus<double>(),
                  Times<double>(),
                  transpose(AT), transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<M,replace> := C + A'.*B'    \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(M), NoAccumulate(),
                  Times<double>(),
                  transpose(AT), transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,merge> := A'.*B'         \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(M), NoAccumulate(),
                  Times<double>(),
                  transpose(AT), transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,replace> := A'.*B'       \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(M), Plus<double>(),
                  Times<double>(),
                  transpose(AT), transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,merge> := C + A'.*B'     \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(M), Plus<double>(),
                  Times<double>(),
                  transpose(AT), transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!M,replace> := C + A'.*B'   \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;

    //-------------------- structure only


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, structure(M), NoAccumulate(),
                  Times<double>(),
                  transpose(AT), transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),merge> := A'.*B'       \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, structure(M), NoAccumulate(),
                  Times<double>(),
                  transpose(AT), transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),replace> := A'.*B'     \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, structure(M), Plus<double>(),
                  Times<double>(),
                  transpose(AT), transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),merge> := C + A'.*B'   \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, structure(M), Plus<double>(),
                  Times<double>(),
                  transpose(AT), transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<s(M),replace> := C + A'.*B' \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(structure(M)), NoAccumulate(),
                  Times<double>(),
                  transpose(AT), transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),merge> := A'.*B'      \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(structure(M)), NoAccumulate(),
                  Times<double>(),
                  transpose(AT), transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),replace> := A'.*B'    \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(structure(M)), Plus<double>(),
                  Times<double>(),
                  transpose(AT), transpose(BT));
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),merge> := C + A'.*B'  \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        C = Ctmp;
        my_timer.start();
        eWiseMult(C, complement(structure(M)), Plus<double>(),
                  Times<double>(),
                  transpose(AT), transpose(BT), REPLACE);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    Ctmp = C;
    reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), C);
    std::cerr << "C<!s(M),replace> := C + A'.*B'\t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << C.nvals() << "\t" << count << std::endl;

    return 0;
}
