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

//****************************************************************************
grb::IndexType const NUM_NODES = 34;

grb::IndexArrayType iA = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,
    2,2,2,2,2,2,2,2,2,2,
    3,3,3,3,3,3,
    4,4,4,
    5,5,5,5,
    6,6,6,6,
    7,7,7,7,
    8,8,8,8,8,
    9,9,
    10,10,10,
    11,
    12,12,
    13,13,13,13,13,
    14,14,
    15,15,
    16,16,
    17,17,
    18,18,
    19,19,19,
    20,20,
    21,21,
    22,22,
    23,23,23,23,23,
    24,24,24,
    25,25,25,
    26,26,
    27,27,27,27,
    28,28,28,
    29,29,29,29,
    30,30,30,30,
    31,31,31,31,31,31,
    32,32,32,32,32,32,32,32,32,32,32,32,
    33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33};

grb::IndexArrayType jA = {
    1,2,3,4,5,6,7,8,10,11,12,13,17,19,21,31,     //1,2,3,4,5,6,7,8,10,11,12,13,19,21,23,31,
    0,2,3,7,13,17,19,21,30,
    0,1,3,7,8,9,13,27,28,32,
    0,1,2,7,12,13,
    0,6,10,
    0,6,10,16,
    0,4,5,16,
    0,1,2,3,
    0,2,30,32,33,
    2,33,
    0,4,5,
    0,
    0,3,
    0,1,2,3,33,
    32,33,
    32,33,
    5,6,
    0,1,
    32,33,
    0,1,33,
    32,33,
    0,1,
    32,33,
    25,27,29,32,33,
    25,27,31,
    23,24,31,
    29,33,
    2,23,24,33,
    2,31,33,
    23,26,32,33,
    1,8,32,33,
    0,24,25,28,32,33,
    2,8,14,15,18,20,22,23,29,30,31,33,
    8,9,13,14,15,18,19,20,22,23,26,27,28,29,30,31,32};

//****************************************************************************
int main(int argc, char **argv)
{
    if (iA.size() != jA.size())
    {
        std::cerr << "Index arrays are not the same size: " << iA.size()
                  << " != " << jA.size() << std::endl;
        return -1;
    }

    Timer<std::chrono::steady_clock, std::chrono::microseconds> my_timer;

    grb::IndexArrayType iL, iU;
    grb::IndexArrayType jL, jU;
    uint64_t num_rows = 0;
    uint64_t max_id = 0;
    uint64_t src, dst;

    my_timer.start();
    {
        for (grb::IndexType ix = 0; ix < iA.size(); ++ix)
        {
            src = iA[ix];
            dst = jA[ix];
            //std::cout << "Read: " << src << ", " << dst << std::endl;
            if (src > max_id) max_id = src;
            if (dst > max_id) max_id = dst;

            if (src < dst)
            {
                iU.push_back(src);
                jU.push_back(dst);
            }
            else if (dst < src)
            {
                iL.push_back(src);
                jL.push_back(dst);
            }
            // else ignore self loops

            ++num_rows;
        }
    }
    my_timer.stop();
    std::cout << "Elapsed read time: " << my_timer.elapsed() << " usec." << std::endl;

    std::cout << "Read " << num_rows << " rows." << std::endl;
    std::cout << "#Nodes = " << (max_id + 1) << std::endl;

    // sort the
    using DegIdx = std::tuple<grb::IndexType,grb::IndexType>;
    my_timer.start();
    std::vector<DegIdx> degrees(max_id + 1);
    for (grb::IndexType idx = 0; idx <= max_id; ++idx)
    {
        degrees[idx] = {0UL, idx};
    }

    {
        for (grb::IndexType ix = 0; ix < iA.size(); ++ix)
        {
            src = iA[ix];
            dst = jA[ix];
            if (src != dst)
            {
                std::get<0>(degrees[src]) += 1;
            }
        }
    }

    std::sort(degrees.begin(), degrees.end(),
              [](DegIdx a, DegIdx b) { return std::get<0>(b) < std::get<0>(a); });

    //relabel
    for (auto &idx : iA) { idx = std::get<1>(degrees[idx]); }
    for (auto &idx : jA) { idx = std::get<1>(degrees[idx]); }
    for (auto &idx : iL) { idx = std::get<1>(degrees[idx]); }
    for (auto &idx : jL) { idx = std::get<1>(degrees[idx]); }
    for (auto &idx : iU) { idx = std::get<1>(degrees[idx]); }
    for (auto &idx : jU) { idx = std::get<1>(degrees[idx]); }

    my_timer.stop();
    std::cout << "Elapsed sort/relabel time: " << my_timer.elapsed() << " usec." << std::endl;

    grb::IndexType idx(0);
    for (auto&& row : degrees)
    {
        std::cout << idx << " <-- " << std::get<1>(row)
                  << ": deg = " << std::get<0>(row) << std::endl;
        idx++;
    }

    grb::IndexType NUM_NODES(max_id + 1);
    using T = int32_t;
    std::vector<T> v(iA.size(), 1);

    /// @todo change scalar type to unsigned int or grb::IndexType
    using MatType = grb::Matrix<T, grb::DirectedMatrixTag>;

    MatType A(NUM_NODES, NUM_NODES);
    MatType L(NUM_NODES, NUM_NODES);
    MatType U(NUM_NODES, NUM_NODES);

    A.build(iA.begin(), jA.begin(), v.begin(), iA.size());
    L.build(iL.begin(), jL.begin(), v.begin(), iL.size());
    U.build(iU.begin(), jU.begin(), v.begin(), iU.size());

    std::cout << "Running algorithm(s)..." << std::endl;
    T count(0);

    // Perform triangle counting with different algorithms
    //===================
    my_timer.start();
    count = algorithms::triangle_count(A);
    my_timer.stop();

    std::cout << "# triangles (L,U=split(A); B=LL'; C=A.*B; #=|C|/2) = " << count << std::endl;
    std::cout << "Elapsed time: " << my_timer.elapsed() << " usec." << std::endl;

    //===================
    my_timer.start();
    count = algorithms::triangle_count_masked(L, U);
    my_timer.stop();

    std::cout << "# triangles (C<L> = L +.* U; #=|C|) = " << count << std::endl;
    std::cout << "Elapsed time: " << my_timer.elapsed() << " usec." << std::endl;

    //===================
    my_timer.start();
    count = algorithms::triangle_count_masked(L);
    my_timer.stop();

    std::cout << "# triangles (C<L> = L +.* L'; #=|C|) = " << count << std::endl;
    std::cout << "Elapsed time: " << my_timer.elapsed() << " usec." << std::endl;

    //===================
    my_timer.start();
    count = algorithms::triangle_count_masked_noT(L);
    my_timer.stop();

    std::cout << "# triangles (C<L> = L +.* L; #=|C|) = " << count << std::endl;
    std::cout << "Elapsed time: " << my_timer.elapsed() << " usec." << std::endl;

    //===================
    my_timer.start();
    count = algorithms::triangle_count_newGBTL(L, U);
    my_timer.stop();

    std::cout << "# triangles (B=LU; C=L.*B; #=|C|) = " << count << std::endl;
    std::cout << "Elapsed time: " << my_timer.elapsed() << " usec." << std::endl;

    return 0;
}
