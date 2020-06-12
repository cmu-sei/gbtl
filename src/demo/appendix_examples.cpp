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

#include <graphblas/graphblas.hpp>
#include <algorithms/appendix_algorithms.hpp>

using namespace grb;

//****************************************************************************
IndexType const NUM_NODES = 34;

IndexArrayType i = {
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

IndexArrayType j = {
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
void test_bfs_level_B1()
{
    std::cout << "======== Testing Appendix B.1 code" << std::endl;
    {
        IndexType const NUM_NODES(9);
        IndexType const START_INDEX(5);

        IndexArrayType r = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                       4, 4, 4, 5, 6, 6, 6, 8, 8};
        IndexArrayType c = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                       2, 3, 8, 2, 1, 2, 3, 2, 4};
        std::vector<IndexType> v(r.size(), 1);

        Matrix<IndexType> G_tn(NUM_NODES, NUM_NODES);
        G_tn.build(r, c, v);

        Vector<IndexType> levels(NUM_NODES);
        algorithms::bfs_level_appendixB1(levels, G_tn, START_INDEX);

        print_vector(std::cout, levels, "bfs_level (B1 test):");

    }

    // ---------------------

    {
        Matrix<uint32_t> A(NUM_NODES,NUM_NODES);
        std::vector<uint32_t> weights(i.size(), 1U);
        A.build(i.begin(), j.begin(), weights.begin(), i.size());

        Vector<IndexType> levels(NUM_NODES);

        algorithms::bfs_level_appendixB1(levels, A, 0UL);

        print_vector(std::cout, levels, "Levels (B1, karate, s=0)");
    }
}

//****************************************************************************
void test_bfs_level_B2()
{
    std::cout << "======== Testing Appendix B.2 code" << std::endl;
    {
        IndexType const NUM_NODES(9);
        IndexType const START_INDEX(5);

        IndexArrayType r = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                       4, 4, 4, 5, 6, 6, 6, 8, 8};
        IndexArrayType c = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                       2, 3, 8, 2, 1, 2, 3, 2, 4};
        std::vector<IndexType> v(r.size(), 1);

        Matrix<IndexType> G_tn(NUM_NODES, NUM_NODES);
        G_tn.build(r, c, v);

        Vector<IndexType> levels(NUM_NODES);
        algorithms::bfs_level_appendixB2(levels, G_tn, START_INDEX);

        print_vector(std::cout, levels, "bfs_level (B2 test):");

    }

    // ---------------------

    {
        Matrix<uint32_t> A(NUM_NODES,NUM_NODES);
        std::vector<uint32_t> weights(i.size(), 1U);
        A.build(i.begin(), j.begin(), weights.begin(), i.size());

        Vector<IndexType> levels(NUM_NODES);

        algorithms::bfs_level_appendixB2(levels, A, 0UL);

        print_vector(std::cout, levels, "Levels (B2, karate, s=0)");
    }
}

//****************************************************************************
void test_bfs_parent_B3()
{
    std::cout << "======== Testing Appendix B.3 code" << std::endl;
    {
        IndexType const NUM_NODES(9);
        IndexType const START_INDEX(5);

        IndexArrayType r = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                       4, 4, 4, 5, 6, 6, 6, 8, 8};
        IndexArrayType c = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                       2, 3, 8, 2, 1, 2, 3, 2, 4};
        std::vector<IndexType> v(r.size(), 1);

        Matrix<IndexType> G_tn(NUM_NODES, NUM_NODES);
        G_tn.build(r, c, v);

        Vector<IndexType> parents(NUM_NODES);
        algorithms::bfs_parent_appendixB3(parents, G_tn, START_INDEX);

        print_vector(std::cout, parents, "bfs_parent (B3 test):");

    }

    // ---------------------

    {
        Matrix<uint32_t> A(NUM_NODES,NUM_NODES);
        std::vector<uint32_t> weights(i.size(), 1U);
        A.build(i.begin(), j.begin(), weights.begin(), i.size());

        Vector<IndexType> parents(NUM_NODES);

        algorithms::bfs_parent_appendixB3(parents, A, 0UL);

        print_vector(std::cout, parents, "Parents (B3, karate, s=0)");
    }
}

//****************************************************************************
void test_BC_B4()
{
    std::cout << "======== Testing Appendix B.4 code" << std::endl;
    {
        IndexType const NUM_NODES(9);
        IndexType const START_INDEX(5);

        IndexArrayType r = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                       4, 4, 4, 5, 6, 6, 6, 8, 8};
        IndexArrayType c = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                       2, 3, 8, 2, 1, 2, 3, 2, 4};
        std::vector<IndexType> v(r.size(), 1);

        Matrix<IndexType> G_tn(NUM_NODES, NUM_NODES);
        G_tn.build(r, c, v);

        Vector<double> delta(NUM_NODES);
        algorithms::BC_appendixB4(delta, G_tn, START_INDEX);

        print_vector(std::cout, delta, "BC delta (B4 test)");
    }

    // ---------------------
    Vector<double> bc(NUM_NODES);
    for (IndexType source = 0; source < 30; source += 9)
    {
        Matrix<uint32_t> A(NUM_NODES,NUM_NODES);
        std::vector<uint32_t> weights(i.size(), 1U);
        A.build(i.begin(), j.begin(), weights.begin(), i.size());

        Vector<IndexType> parents(NUM_NODES);

        Vector<double> delta(NUM_NODES);
        algorithms::BC_appendixB4(delta, A, source);
        eWiseAdd(bc, NoMask(), NoAccumulate(), Plus<double>(), bc, delta);

        std::cout << "BC delta (B4, karate, s=" << source << ")";
        print_vector(std::cout, delta, "");
    }

    print_vector(std::cout, bc, "Aggregate BC of for sources");
}

//****************************************************************************
void test_BC_batch_B5()
{
    std::cout << "======== Testing Appendix B.5 code" << std::endl;
    {
        IndexType const NUM_NODES(9);
        IndexType const START_INDEX(5);

        IndexArrayType r = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                       4, 4, 4, 5, 6, 6, 6, 8, 8};
        IndexArrayType c = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                       2, 3, 8, 2, 1, 2, 3, 2, 4};
        std::vector<IndexType> v(r.size(), 1);

        Matrix<IndexType> G_tn(NUM_NODES, NUM_NODES);
        G_tn.build(r, c, v);

        IndexArrayType sources = {START_INDEX};
        Vector<double> delta(NUM_NODES);
        algorithms::BC_update_appendixB5(delta, G_tn, sources);

        print_vector(std::cout, delta, "BC_batch delta (B5 test)");
    }

    // ---------------------

    {
        Matrix<uint32_t> A(NUM_NODES,NUM_NODES);
        std::vector<uint32_t> weights(i.size(), 1U);
        A.build(i.begin(), j.begin(), weights.begin(), i.size());

        Vector<IndexType> parents(NUM_NODES);

        IndexArrayType sources = {0, 9, 18, 27};
        Vector<double> delta(NUM_NODES);
        algorithms::BC_update_appendixB5(delta, A, sources);

        print_vector(std::cout, delta, "BC_batch delta (B5, karate, s={0,9,18,27})");
    }
}

//****************************************************************************
void test_MIS_B6()
{
    std::cout << "======== Testing Appendix B.6 code" << std::endl;
    {
        IndexType const NUM_NODES(9);

        IndexArrayType r = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                       4, 4, 4, 5, 6, 6, 6, 8, 8};
        IndexArrayType c = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                       2, 3, 8, 2, 1, 2, 3, 2, 4};
        std::vector<IndexType> v(r.size(), 1);

        Matrix<IndexType> G_tn(NUM_NODES, NUM_NODES);
        G_tn.build(r, c, v);

        Vector<bool> iset(NUM_NODES);
        algorithms::mis_appendixB6(iset, G_tn);

        print_vector(std::cout, iset, "MIS i-set (B6 test)");
    }

    // ---------------------

    {
        Matrix<uint32_t> A(NUM_NODES,NUM_NODES);
        std::vector<uint32_t> weights(i.size(), 1U);
        A.build(i.begin(), j.begin(), weights.begin(), i.size());

        Vector<IndexType> parents(NUM_NODES);

        Vector<bool> iset(NUM_NODES);
        algorithms::mis_appendixB6(iset, A);

        print_vector(std::cout, iset, "MIS i-set (B6 karate)");
    }
}

//****************************************************************************
void test_triangle_count_B7()
{
    std::cout << "======== Testing Appendix B.7 code" << std::endl;
    {
        //Matrix<double, DirectedMatrixTag> testtriangle(
        //                       {{0,1,1,1,0},
        //                        {1,0,1,0,1},
        //                        {1,1,0,1,1},
        //                        {1,0,1,0,1},
        //                        {0,1,1,1,0}});

        std::vector<double> ar={0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4};
        std::vector<double> ac={1, 2, 3, 0, 2, 4, 0, 1, 3, 4, 0, 2, 4, 1, 2, 3};
        std::vector<double> av={1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        Matrix<double> A(5,5);
        A.build(ar.begin(), ac.begin(), av.begin(), av.size());

        Matrix<double> L(5,5), U(5,5);
        split(A, L, U);

        uint64_t tc = algorithms::triangle_count_appendixB7(L);
        std::cout << "Number of triangles (test 4): " << tc << std::endl;
    }

    //--------------------

    {
        Matrix<uint32_t> A(NUM_NODES,NUM_NODES);
        std::vector<uint32_t> weights(i.size(), 1U);
        A.build(i.begin(), j.begin(), weights.begin(), i.size());

        Matrix<uint32_t> L(NUM_NODES,NUM_NODES), U(NUM_NODES,NUM_NODES);
        split(A, L, U);

        uint64_t tc = algorithms::triangle_count_appendixB7(L);
        std::cout << "Number of triangles (karate): " << tc << std::endl;
    }
}

//****************************************************************************
int main(int, char**)
{
    if (i.size() != j.size())
    {
        std::cerr << "Index arrays are not the same size: " << i.size()
                  << " != " << j.size() << std::endl;
        return -1;
    }

    test_bfs_level_B1();
    test_bfs_level_B2();
    test_bfs_parent_B3();
    test_BC_B4();
    test_BC_batch_B5();
    test_MIS_B6();
    test_triangle_count_B7();
    return 0;
}
