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
#include <chrono>
#include <graphblas/graphblas.hpp>
#include "raw_gkc_mxv.hpp"
#include "Timer_gkc.hpp"

using namespace grb;

//****************************************************************************

grb::IndexType const NUM_NODES = 34;
grb::IndexArrayType i = {
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
    31,31,31,31,31,
    32,32,32,32,32,32,32,32,32,32,32,32,
    33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33};

grb::IndexArrayType j = {
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
    0,24,25,32,33,    //0,24,25,28,32,33,
    2,8,14,15,18,20,22,23,29,30,31,33,
    8,9,13,14,15,18,19,20,22,23,26,27,28,29,30,31,32};

//****************************************************************************

//****************************************************************************
IndexType read_edge_list(std::string const &pathname,
                         IndexArrayType    &row_indices,
                         IndexArrayType    &col_indices)
{
    std::ifstream infile(pathname);
    IndexType max_id = 0;
    uint64_t num_rows = 0;
    uint64_t src, dst;

    std::string line;

    while (std::getline( infile, line) )
    {
        if (infile.eof()) break;
        
        std::istringstream l(line);
        l >> src >> dst; // And discard the rest (weights)

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
#define EXT_FILE 1
#if EXT_FILE
    if (argc < 2)
    {
        std::cerr << "ERROR: too few arguments." << std::endl;
        std::cerr << "Usage: " << argv[0] << " <edge list file>" << std::endl;
        exit(1);
    }
    
    // Read the edgelist and create the tuple arrays
    IndexArrayType iA, jA, iu;
    std::string pathname(argv[1]);
    IndexType const NUM_NODES(read_edge_list(pathname, iA, jA));
#else
    // NUM_NODES defined at top of file if reading internal data.
#endif

    using T = int32_t;

    using MatType = Matrix<T,GKCTag>;
    using VecType = Vector<T,GKCTag>;
    using BoolVecType = Vector<bool,GKCTag>;
    MatType A(NUM_NODES, NUM_NODES);
    MatType AT(NUM_NODES, NUM_NODES);
    VecType u(NUM_NODES);
    VecType w(NUM_NODES);
    VecType w1(NUM_NODES);
    VecType wtmp(NUM_NODES);
    BoolVecType M(NUM_NODES);

#if EXT_FILE
    std::vector<T> v(iA.size(), 1);
    std::vector<bool> bv(iA.size(), true);
    A.build(iA.begin(), jA.begin(), v.begin(), iA.size());
    AT.build(jA.begin(), iA.begin(), v.begin(), iA.size());
    // transpose(AT, NoMask(), NoAccumulate(), A);
#else
    std::vector<T> weights(i.size(), 1);
    A.build(i.begin(), j.begin(), weights.begin(), i.size());
    AT.build(j.begin(), i.begin(), weights.begin(), i.size());
#endif


    T * vec1_dense = (T*)calloc(NUM_NODES, sizeof(T));
    T * vec2_dense = (T*)calloc(NUM_NODES, sizeof(T));
    IndexType * vec1_sparse = (IndexType*)calloc(NUM_NODES, sizeof(IndexType));
    IndexType * vec2_sparse = (IndexType*)calloc(NUM_NODES, sizeof(IndexType));
    IndexType vec1_elems = 0;
    IndexType vec2_elems = 0;
    std::vector<bool> vec1_bmap(NUM_NODES);
    std::vector<bool> vec2_bmap(NUM_NODES);

    std::default_random_engine  generator;
    std::uniform_real_distribution<double> distribution;
    for (IndexType iu = 0; iu < NUM_NODES; ++iu)
    {
        if (distribution(generator) < 0.15)
        {
            M.setElement(iu, true);
            // Not using mask for this short test.
        }
        if (distribution(generator) < 0.1)
        {
            u.setElement(iu, 1);
            vec1_dense[iu] = 1;
            vec1_sparse[vec1_elems] = iu;
            vec1_bmap[iu] = true;
            vec1_elems++;
        }

    }

    std::cout << "Running algorithm(s)... M.nvals = " << M.nvals() << std::endl;
    std::cout << "u.nvals = " << u.nvals() << std::endl;
    T count(0);

    Timer<std::chrono::steady_clock, std::chrono::microseconds> my_timer;

    // warm up
    // mxv(w, NoMask(), NoAccumulate(), ArithmeticSemiring<double>(), A, u);

    int const NUM_TRIALS=20;
    double accum_time, min_time, max_time;
    //=====================================================
    // Perform matrix vector multiplies
    //=====================================================
    T num_vals = 0;
    T reduc_total = 0;

    //===================
    // A*u
    //===================
    /*
    std::cout << "RAW DENSE GKC: A*u" << std::endl;
    std::cout << "Function                      \tmin\tavg\tmax\tw.nvals\tchecksum\n";


    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        //w = wtmp;
        vec2_elems = 0;
        for (size_t idx = 0; idx < NUM_NODES; idx++)
        {
            vec2_dense[idx] = 0;
            vec2_bmap[idx] = 0;
            vec2_elems = 0;
        }
        my_timer.start();
        // mxv(w, NoMask(), NoAccumulate(),
        //     ArithmeticSemiring<double>(),
        //     A, u);
        
        pr_dot_weights_kernel(A.m_mat.getIA(), A.m_mat.getJA(), A.m_mat.getVA(), NUM_NODES, vec1_dense, vec2_dense);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    // wtmp = w;
    // vec2_elems = 0;
    // for (size_t idx = 0; idx < NUM_NODES; idx++)
    // {
    //     vec2_dense[idx] = 0;
    // }
    // reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    // T num_vals = w.nvals();
    num_vals = 0;
    reduc_total = 0;
    for (size_t idx = 0; idx < NUM_NODES; idx++)
    {
        if (vec2_dense[idx] != 0) num_vals++;
        reduc_total += vec2_dense[idx];
    }
    count = reduc_total;
    std::cout << "w := A+.*u                    \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << num_vals << "\t" << count << std::endl;

    //----------
    */

    //===================
    // A*u
    //===================
    std::cout << "RAW SPARSE GKC: A*u" << std::endl;
    std::cout << "Function                      \tmin\tavg\tmax\tw.nvals\tchecksum\n";

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        //w = wtmp;
        vec2_elems = 0;
        for (size_t idx = 0; idx < NUM_NODES; idx++)
        {
            vec2_dense[idx] = 0;
            vec2_bmap[idx] = 0;
            vec2_elems = 0;
        }
        my_timer.start();
        // mxv(w, NoMask(), NoAccumulate(),
        //     ArithmeticSemiring<double>(),
        //     A, u);
        dot_kernel_bfs(A.m_mat.getIA(), A.m_mat.getJA(), A.m_mat.getVA(), NUM_NODES, 
                         vec1_dense, vec2_dense, vec1_sparse, vec2_sparse, vec1_elems, vec2_elems);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    // wtmp = w;
    // vec2_elems = 0;
    // for (size_t idx = 0; idx < NUM_NODES; idx++)
    // {
    //     vec2_dense[idx] = 0;
    // }
    // reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    // T num_vals = w.nvals();
    num_vals = vec2_elems;
    reduc_total = 0;
    for (size_t idx = 0; idx < vec2_elems; idx++)
    {
        reduc_total += vec2_dense[vec2_sparse[idx]];
    }
    count = reduc_total;
    std::cout << "w := A+.*u                    \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << num_vals << "\t" << count << std::endl;

    //----------
    
    //===================
    // A*u
    //===================
    std::cout << "RAW DENSE W/ BITMAP GKC: A*u" << std::endl;
    std::cout << "Function                      \tmin\tavg\tmax\tw.nvals\tchecksum\n";

    //----------
    accum_time=0.; min_time=1.0e38; max_time=0.;
    for (int ix=0; ix<NUM_TRIALS; ++ix)
    {
        //w = wtmp;
        vec2_elems = 0;
        for (size_t idx = 0; idx < NUM_NODES; idx++)
        {
            vec2_dense[idx] = 0;
            vec2_bmap[idx] = 0;
            vec2_elems = 0;
        }
        my_timer.start();
        // mxv(w, NoMask(), NoAccumulate(),
        //     ArithmeticSemiring<double>(),
        //     A, u);
        
        pr_dot_weights_kernel_bmask(A.m_mat.getIA(), A.m_mat.getJA(), A.m_mat.getVA(), NUM_NODES, 
                                     vec1_dense, vec2_dense, vec1_bmap, vec2_bmap);
        my_timer.stop();
        double t = my_timer.elapsed();
        accum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    // wtmp = w;
    // vec2_elems = 0;
    // for (size_t idx = 0; idx < NUM_NODES; idx++)
    // {
    //     vec2_dense[idx] = 0;
    // }
    // reduce(count, NoAccumulate(), PlusMonoid<int32_t>(), w);
    // T num_vals = w.nvals();
    num_vals = 0;
    reduc_total = 0;
    for (size_t idx = 0; idx < NUM_NODES; idx++)
    {
        if (vec2_bmap[idx])
        {
            num_vals++;
            reduc_total += vec2_dense[idx];
        }  
    }
    count = reduc_total;
    std::cout << "w := A+.*u                    \t"
              << min_time << "\t" << accum_time/NUM_TRIALS << "\t" << max_time
              << "\t" << num_vals << "\t" << count << std::endl;

    //----------



    // bool passed = (w == w1);
    // std::cout << "Results " << (passed ? "PASSED" : "FAILED") << std::endl;
    
    
    free(vec1_dense  );
    free(vec2_dense  );
    free(vec1_sparse );
    free(vec2_sparse );
              
    return 0;
}
