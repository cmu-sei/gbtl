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
    31,31,31,31,31,31,
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
    0,24,25,28,32,33,
    2,8,14,15,18,20,22,23,29,30,31,33,
    8,9,13,14,15,18,19,20,22,23,26,27,28,29,30,31,32};

//****************************************************************************
bool test_bfs_level_C1()
{
    bool passed = true;

    std::cout << "======== Testing Appendix C.1 code" << std::endl;
    {
        IndexType const NUM_NODES(9);
        IndexType const START_INDEX(5);

        IndexArrayType r = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3,
                            3, 4, 4, 4, 5, 6, 6, 6, 8, 8};
        IndexArrayType c = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4,
                            6, 2, 3, 8, 2, 1, 2, 3, 2, 4};
        std::vector<IndexType> v(r.size(), 1);

        std::vector<IndexType> ans = {5, 4, 2, 4, 3, 1, 3, 0, 3};
        Vector<IndexType> answer(ans, 0);

        Matrix<IndexType> G_tn(NUM_NODES, NUM_NODES);
        G_tn.build(r, c, v);

        Vector<IndexType> levels(NUM_NODES);
        algorithms::bfs_level_appendixC1(levels, G_tn, START_INDEX);

        print_vector(std::cout, levels, "bfs_level (C1 test):");

        if (levels != answer)
        {
            std::cout << "Test failed: correct answer: " << answer << std::endl;
            passed = false;
        }
    }

    // ---------------------

    {
        Matrix<uint32_t> A(NUM_NODES,NUM_NODES);
        std::vector<uint32_t> weights(i.size(), 1U);
        A.build(i.begin(), j.begin(), weights.begin(), i.size());

        Vector<IndexType> levels(NUM_NODES);

        std::vector<IndexType> ans = {1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 4, 4, 3,
                                      2, 4, 2, 4, 2, 4, 4, 3, 3, 4, 3, 3, 4, 3, 2, 3, 3};
        Vector<IndexType> answer(ans, 0);

        algorithms::bfs_level_appendixC1(levels, A, 0UL);

        print_vector(std::cout, levels, "Levels (C1, karate, s=0)");

        if (levels != answer)
        {
            std::cout << "Test failed: correct answer: " << answer << std::endl;
            passed = false;
        }
    }

    std::cerr << "test_bfs_level_C1: " << (passed ? "PASSED\n" : "FAILED\n");
    return passed;
}

//****************************************************************************
bool test_bfs_level_C2()
{
    bool passed = true;

    std::cout << "======== Testing Appendix C.2 code" << std::endl;
    {
        IndexType const NUM_NODES(9);
        IndexType const START_INDEX(5);

        IndexArrayType r = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3,
                            3, 4, 4, 4, 5, 6, 6, 6, 8, 8};
        IndexArrayType c = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4,
                            6, 2, 3, 8, 2, 1, 2, 3, 2, 4};
        std::vector<IndexType> v(r.size(), 1);

        Matrix<IndexType> G_tn(NUM_NODES, NUM_NODES);
        G_tn.build(r, c, v);

        Vector<IndexType> levels(NUM_NODES);

        std::vector<IndexType> ans = {5, 4, 2, 4, 3, 1, 3, 0, 3};
        Vector<IndexType> answer(ans, 0);

        algorithms::bfs_level_appendixC2(levels, G_tn, START_INDEX);

        print_vector(std::cout, levels, "bfs_level (C2 test):");

        if (levels != answer)
        {
            std::cout << "Test failed: correct answer: " << answer << std::endl;
            passed = false;
        }
    }

    // ---------------------

    {
        Matrix<uint32_t> A(NUM_NODES,NUM_NODES);
        std::vector<uint32_t> weights(i.size(), 1U);
        A.build(i.begin(), j.begin(), weights.begin(), i.size());

        Vector<IndexType> levels(NUM_NODES);

        std::vector<IndexType> ans = {1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 4, 4, 3,
                                      2, 4, 2, 4, 2, 4, 4, 3, 3, 4, 3, 3, 4, 3, 2, 3, 3};
        Vector<IndexType> answer(ans, 0);

        algorithms::bfs_level_appendixC2(levels, A, 0UL);

        print_vector(std::cout, levels, "Levels (C2, karate, s=0)");

        if (levels != answer)
        {
            std::cout << "Test failed: correct answer: " << answer << std::endl;
            passed = false;
        }
    }

    std::cerr << "test_bfs_level_C2: " << (passed ? "PASSED\n" : "FAILED\n");
    return passed;
}

//****************************************************************************
bool test_bfs_parent_C3()
{
    bool passed = true;

    std::cout << "======== Testing Appendix C.3 code" << std::endl;
    {
        IndexType const NUM_NODES(9);
        IndexType const START_INDEX(5);

        IndexArrayType r = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3,
                            3, 4, 4, 4, 5, 6, 6, 6, 8, 8};
        IndexArrayType c = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4,
                            6, 2, 3, 8, 2, 1, 2, 3, 2, 4};
        std::vector<IndexType> v(r.size(), 1);

        Matrix<IndexType> G_tn(NUM_NODES, NUM_NODES);
        G_tn.build(r, c, v);

        Vector<IndexType> parents(NUM_NODES);

        std::vector<IndexType> ans = {3, 6, 5, 4, 2, 5, 2,99, 2};
        Vector<IndexType> answer(ans, 99);

        algorithms::bfs_parent_appendixC3(parents, G_tn, START_INDEX);

        print_vector(std::cout, parents, "bfs_parent (C3 test):");

        if (parents != answer)
        {
            std::cout << "Test failed: correct answer: " << answer << std::endl;
            passed = false;
        }
    }

    // ---------------------

    {
        Matrix<uint32_t> A(NUM_NODES,NUM_NODES);
        std::vector<uint32_t> weights(i.size(), 1U);
        A.build(i.begin(), j.begin(), weights.begin(), i.size());

        Vector<IndexType> parents(NUM_NODES);

        std::vector<IndexType> ans = {0,  0, 0,  0, 0,  0,  0,  0,  0,  2, 0, 0,  0, 0, 32, 32, 5,
                                      0, 32, 0, 32, 0, 32, 25, 31, 31, 33, 2, 2, 32, 1,  0,  2, 8};
        Vector<IndexType> answer(ans, 99);

        algorithms::bfs_parent_appendixC3(parents, A, 0UL);

        print_vector(std::cout, parents, "Parents (C3, karate, s=0)");

        if (parents != answer)
        {
            std::cout << "Test failed: correct answer: " << answer << std::endl;
            passed = false;
        }
    }

    std::cerr << "test_bfs_parent_C3: " << (passed ? "PASSED\n" : "FAILED\n");
    return passed;
}

//****************************************************************************
bool test_BC_C4()
{
    bool passed = true;

    std::cout << "======== Testing Appendix C.4 code" << std::endl;
    {
        IndexType const NUM_NODES(9);
        IndexType const START_INDEX(5);

        IndexArrayType r = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3,
                            3, 4, 4, 4, 5, 6, 6, 6, 8, 8};
        IndexArrayType c = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4,
                            6, 2, 3, 8, 2, 1, 2, 3, 2, 4};
        std::vector<IndexType> v(r.size(), 1);

        Matrix<IndexType> G_tn(NUM_NODES, NUM_NODES);
        G_tn.build(r, c, v);

        Vector<double> delta(NUM_NODES);
        std::vector<double> ans = {-1, -1, 6, 1, 1, -1, 2, -1, -1};
        Vector<double> answer(ans, -1);

        algorithms::BC_appendixC4(delta, G_tn, START_INDEX);

        print_vector(std::cout, delta, "BC delta (C4 test)");
        if (delta != answer)
        {
            std::cout << "Test failed: correct answer: " << answer << std::endl;
            passed = false;
        }
    }

    // ---------------------
    Vector<double> bc(NUM_NODES);
    {
        IndexType source = 0;
        Matrix<uint32_t> A(NUM_NODES,NUM_NODES);
        std::vector<uint32_t> weights(i.size(), 1U);
        A.build(i.begin(), j.begin(), weights.begin(), i.size());

        Vector<IndexType> parents(NUM_NODES);

        Vector<double> delta(NUM_NODES);
        std::vector<double> ans = {-1, 0.5, 3.91269826889038086, -1, -1, 0.5, 0.5, -1,
                                    3.26984119415283203, -1, -1, -1, -1, 1.46825397014617920,
                                    -1, -1, -1, -1, -1, 1.46825397014617920, -1, -1, -1, -1,
                                    -1, 0.11111111193895340, -1, 0.11111111193895340, -1, -1,
                                    -1, 5.38095235824584961, 2.90476179122924805,
                                    4.87301588058471680};
        Vector<double> answer(ans, -1);

        algorithms::BC_appendixC4(delta, A, source);
        eWiseAdd(bc, NoMask(), NoAccumulate(), Plus<double>(), bc, delta);

        std::cout << "BC delta (C4, karate, s=" << source << ")";
        print_vector(std::cout, delta, "");
        if (delta != answer)
        {
            std::cout << "Test failed: correct answer: " << answer << std::endl;
            passed = false;
        }
    }

    print_vector(std::cout, bc, "Aggregate BC of for sources");

    std::cerr << "test_BC_C4: " << (passed ? "PASSED\n" : "FAILED\n");
    return passed;
}

//****************************************************************************
bool test_BC_batch_C5()
{
    bool passed = true;

    std::cout << "======== Testing Appendix C.5 code" << std::endl;
    {
        IndexType const NUM_NODES(9);
        IndexType const START_INDEX(5);

        IndexArrayType r = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3,
                            3, 4, 4, 4, 5, 6, 6, 6, 8, 8};
        IndexArrayType c = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4,
                            6, 2, 3, 8, 2, 1, 2, 3, 2, 4};
        std::vector<IndexType> v(r.size(), 1);

        Matrix<IndexType> G_tn(NUM_NODES, NUM_NODES);
        G_tn.build(r, c, v);

        IndexArrayType sources = {START_INDEX};
        Vector<double> delta(NUM_NODES);
        std::vector<double> ans = {0, 0, 6, 1, 1, 0, 2, 0, 0};
        Vector<double> answer(ans, -1);
        algorithms::BC_update_appendixC5(delta, G_tn, sources);

        print_vector(std::cout, delta, "BC_batch delta (C5 test)");
        if (delta != answer)
        {
            std::cout << "Test failed: correct answer: " << answer << std::endl;
            passed = false;
        }
    }

    // ---------------------

    {
        Matrix<uint32_t> A(NUM_NODES,NUM_NODES);
        std::vector<uint32_t> weights(i.size(), 1U);
        A.build(i.begin(), j.begin(), weights.begin(), i.size());

        Vector<IndexType> parents(NUM_NODES);

        IndexArrayType sources = {0};
        Vector<double> delta(NUM_NODES);
        std::vector<double> ans = {0, 0.5, 3.9126987457275391, 0, 0, 0.5, 0.5,
                                   0, 3.269841194152832, 0, 0, 0, 0,
                                   1.4682540893554688, 0, 0, 0, 0, 0,
                                   1.4682540893554688, 0, 0, 0, 0, 0,
                                   0.11111116409301758, 0, 0.11111116409301758,
                                   0, 0, 0, 5.3809523582458496, 2.904761791229248,
                                   4.8730158805847168};
        Vector<double> answer(ans, -1);
        algorithms::BC_update_appendixC5(delta, A, sources);

        print_vector(std::cout, delta, "BC_batch delta (C5, karate, s={0,9,18,27})");
        if (delta != answer)
        {
            std::cout << "Test failed: correct answer: " << answer << std::endl;
            passed = false;
        }
    }

    std::cerr << "test_BC_batch_C5: " << (passed ? "PASSED\n" : "FAILED\n");
    return passed;
}

//****************************************************************************
bool test_MIS_C6()
{
    bool passed = true;

    std::cout << "======== Testing Appendix C.6 code" << std::endl;
    {
        IndexType const NUM_NODES(9);

        IndexArrayType r = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3,
                            3, 4, 4, 4, 5, 6, 6, 6, 8, 8};
        IndexArrayType c = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4,
                            6, 2, 3, 8, 2, 1, 2, 3, 2, 4};
        std::vector<IndexType> v(r.size(), 1);

        Matrix<IndexType> G_tn(NUM_NODES, NUM_NODES);
        G_tn.build(r, c, v);

        Vector<bool> iset(NUM_NODES);
        std::vector<bool> ans = {true, true, false, false, true, true, false, true, false};
        Vector<bool> answer(ans, false);
        algorithms::mis_appendixC6(iset, G_tn);

        print_vector(std::cout, iset, "MIS i-set (C6 test)");
    }

    // ---------------------

    {
        Matrix<uint32_t> A(NUM_NODES,NUM_NODES);
        std::vector<uint32_t> weights(i.size(), 1U);
        A.build(i.begin(), j.begin(), weights.begin(), i.size());

        Vector<IndexType> parents(NUM_NODES);

        Vector<bool> iset(NUM_NODES);
        std::vector<bool> ans = {false, false, false, false,  true, false, false,  true,
                                 false,  true, false,  true,  true,  true,  true,  true,
                                  true,  true,  true,  true,  true,  true,  true,  true,
                                  true, false,  true, false,  true, false,  true, false,
                                 false, false};
        Vector<bool> answer(ans, false);
        algorithms::mis_appendixC6(iset, A);

        print_vector(std::cout, iset, "MIS i-set (C6 karate)");
    }

    std::cerr << "test_MIS_C6: " << (passed ? "PASSED\n" : "FAILED\n");
    return passed;
}

//****************************************************************************
bool test_triangle_count_C7()
{
    bool passed = true;

    std::cout << "======== Testing Appendix C.7 code" << std::endl;
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

        uint64_t tc = algorithms::triangle_count_appendixC7(A);
        std::cout << "Number of triangles (test, 4): " << tc << std::endl;

        if (tc != 4)
        {
            std::cout << "Test failed: correct answer: 4\n";
            passed = false;
        }
    }

    //--------------------

    {
        Matrix<uint32_t> A(NUM_NODES,NUM_NODES);
        std::vector<uint32_t> weights(i.size(), 1U);
        A.build(i.begin(), j.begin(), weights.begin(), i.size());

        uint64_t tc = algorithms::triangle_count_appendixC7(A);
        std::cout << "Number of triangles (karate, 45): " << tc << std::endl;

        if (tc != 45)
        {
            std::cout << "Test failed: correct answer: 45\n";
            passed = false;
        }
    }

    std::cerr << "test_triangle_count_C7: " << (passed ? "PASSED\n" : "FAILED\n");
    return passed;
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
    std::cout.precision(17);

    bool passed = true;
    passed &= test_bfs_level_C1();
    passed &= test_bfs_level_C2();
    passed &= test_bfs_parent_C3();
    passed &= test_BC_C4();
    passed &= test_BC_batch_C5();
    passed &= test_MIS_C6();
    passed &= test_triangle_count_C7();

    std::cerr << "appendix_examples: " << (passed ? "PASSED\n" : "FAILED\n");
    return (passed ? 0 : -2);
}
