/*
 * Copyright (c) 2015 Carnegie Mellon University and The Trustees of Indiana
 * University.
 * All Rights Reserved.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS," WITH NO WARRANTIES WHATSOEVER. CARNEGIE
 * MELLON UNIVERSITY AND THE TRUSTEES OF INDIANA UNIVERSITY EXPRESSLY DISCLAIM
 * TO THE FULLEST EXTENT PERMITTED BY LAW ALL EXPRESS, IMPLIED, AND STATUTORY
 * WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT OF PROPRIETARY RIGHTS.
 *
 * This Program is distributed under a BSD license.  Please see LICENSE file or
 * permission@sei.cmu.edu for more information.  DM-0002659
 */

#include <graphblas/graphblas.hpp>

using namespace GraphBLAS;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE assign_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(assign_suite)

//****************************************************************************
// Standard Vector tests
//****************************************************************************
BOOST_AUTO_TEST_CASE(assign_vec_test_bad_dimensions)
{
    IndexArrayType i_c = {1, 2, 3};
    std::vector<double> v_c  = {1, 2, 3};
    Vector<double> w(4);
    w.build(i_c, v_c);

    IndexArrayType i_a    = {1};
    std::vector<double> v_a = {99};
    Vector<double> u(2);
    u.build(i_a, v_a);

    IndexArrayType vect_I({1,3,2});

    BOOST_CHECK_THROW(
        (assign(w, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                u, vect_I)),
        DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(assign_vec_test_index_out_of_range)
{
    IndexArrayType i_c = {1, 2, 3};
    std::vector<double> v_c  = {1, 2, 3};
    Vector<double> w(4);
    w.build(i_c, v_c);

    IndexArrayType i_a    = {1};
    std::vector<double> v_a = {99};
    Vector<double> u(2);
    u.build(i_a, v_a);

    IndexArrayType vect_I({0,4});

    BOOST_CHECK_THROW(
        (assign(w, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                u, vect_I)),
        IndexOutOfBoundsException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(assign_vec_test_no_accum)
{
    IndexArrayType i_c = {0, 1, 2};
    std::vector<double> v_c = {8, 9, 1   };
    Vector<double> c(4);
    c.build(i_c, v_c);

    IndexArrayType i_a = {1, 2};
    std::vector<double> v_a = {   98, 99};
    Vector<double> a(3);
    a.build(i_a, v_a);

    IndexArrayType vect_I({1,2,3});

    IndexArrayType i_result      = {0,     2,  3};
    std::vector<double> v_result = {8,    98, 99};
    Vector<double> result(4);
    result.build(i_result, v_result);

    assign(c, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
           a, vect_I);

    BOOST_CHECK_EQUAL(c, result);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(assign_vec_test_accum)
{
    IndexArrayType i_c = {0, 1, 2};
    std::vector<double> v_c = {8, 9, 1   };
    Vector<double> c(4);
    c.build(i_c, v_c);

    IndexArrayType i_a = {1, 2};
    std::vector<double> v_a = {   98, 99};
    Vector<double> a(3);
    a.build(i_a, v_a);

    IndexArrayType vect_I({1,2,3});

    IndexArrayType i_result      = {0, 1,  2,  3};
    std::vector<double> v_result = {8, 9, 99, 99};
    Vector<double> result(4);
    result.build(i_result, v_result);

    assign(c, GraphBLAS::NoMask(), GraphBLAS::Plus<double>(),
           a, vect_I);

    BOOST_CHECK_EQUAL(c, result);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(assign_vec_test_allindices_no_accum)
{
    IndexArrayType i_c = {0, 1, 2};
    std::vector<double> v_c = {8, 9, 1   };
    Vector<double> c(4);
    c.build(i_c, v_c);

    IndexArrayType i_a = {1, 2, 3};
    std::vector<double> v_a = {   98, 99, 100};
    Vector<double> a(4);
    a.build(i_a, v_a);

    assign(c, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
           a, GraphBLAS::AllIndices());

    BOOST_CHECK_EQUAL(c, a);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(assign_vec_test_allindices_accum)
{
    IndexArrayType i_c = {0, 1, 2};
    std::vector<double> v_c = {8, 9, 1   };
    Vector<double> c(4);
    c.build(i_c, v_c);

    IndexArrayType i_a = {1, 2, 3};
    std::vector<double> v_a = {   98, 99, 100};
    Vector<double> a(4);
    a.build(i_a, v_a);

    assign(c, GraphBLAS::NoMask(), GraphBLAS::Plus<double>(),
           a, GraphBLAS::AllIndices());

    IndexArrayType i_result      = {0, 1,  2,  3};
    std::vector<double> v_result = {8, 107, 100, 100};
    Vector<double> result(4);
    result.build(i_result, v_result);

    BOOST_CHECK_EQUAL(c, result);
}

/// @todo add tests with masks

//****************************************************************************
// Standard Matrix tests
//****************************************************************************
BOOST_AUTO_TEST_CASE(assign_mat_test_bad_dimensions)
{
    IndexArrayType      i_c = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType      j_c = {1, 2, 3, 0, 2, 3, 0, 1, 2};
    std::vector<double> v_c = {1, 2, 3, 4, 6, 7, 8, 9, 1};
    Matrix<double, DirectedMatrixTag> c(3, 4);
    c.build(i_c, j_c, v_c);

    IndexArrayType i_a    = {0, 0, 1};
    IndexArrayType j_a    = {0, 1, 0};
    std::vector<double> v_a = {1, 99, 99};
    Matrix<double, DirectedMatrixTag> a(2, 2);
    a.build(i_a, j_a, v_a);

    IndexArrayType vect_I({1,0,1});
    IndexArrayType vect_J({1,2});

    BOOST_CHECK_THROW(
        (assign(c, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                a, vect_I, vect_J)),
        DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(assign_mat_test_index_out_of_range)
{
    IndexArrayType      i_c = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType      j_c = {1, 2, 3, 0, 2, 3, 0, 1, 2};
    std::vector<double> v_c = {1, 2, 3, 4, 6, 7, 8, 9, 1};
    Matrix<double, DirectedMatrixTag> c(3, 4);
    c.build(i_c, j_c, v_c);

    IndexArrayType i_a    = {0, 0, 1};
    IndexArrayType j_a    = {0, 1, 0};
    std::vector<double> v_a = {1, 99, 99};
    Matrix<double, DirectedMatrixTag> a(2, 2);
    a.build(i_a, j_a, v_a);

    IndexArrayType vect_I({1,5});
    IndexArrayType vect_J({1,2});

    BOOST_CHECK_THROW(
        (assign(c, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                a, vect_I, vect_J)),
        IndexOutOfBoundsException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(assign_mat_test_no_accum)
{
    IndexArrayType i_c    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_c    = {1, 2, 3, 0, 2, 3, 0, 1, 2};
    std::vector<double> v_c = {   1, 2, 3,
                               4,    6, 7,
                               8, 9, 1   };
    Matrix<double, DirectedMatrixTag> c(3, 4);
    c.build(i_c, j_c, v_c);

    IndexArrayType i_a    = {0, 0, 1};
    IndexArrayType j_a    = {0, 1, 0};
    std::vector<double> v_a = {1, 99,
                               99   };
    Matrix<double, DirectedMatrixTag> a(2, 2);
    a.build(i_a, j_a, v_a);

    IndexArrayType vect_I({1,2});
    IndexArrayType vect_J({1,2});

    IndexArrayType i_result    = {0, 0, 0, 1, 1, 1, 1, 2, 2};
    IndexArrayType j_result    = {1, 2, 3, 0, 1, 2, 3, 0, 1};
    std::vector<double> v_result = {   1, 2,  3,
                                    4, 1, 99, 7,
                                    8, 99      };
    Matrix<double, DirectedMatrixTag> result(3, 4);
    result.build(i_result, j_result, v_result);

    assign(c, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
           a, vect_I, vect_J);

    BOOST_CHECK_EQUAL(c, result);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(assign_mat_test_accum)
{
    IndexArrayType i_c    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_c    = {1, 2, 3, 0, 2, 3, 0, 1, 2};
    std::vector<double> v_c = {   1, 2, 3,
                               4,    6, 7,
                               8, 9, 1   };
    Matrix<double, DirectedMatrixTag> c(3, 4);
    c.build(i_c, j_c, v_c);

    IndexArrayType i_a    = {0, 0, 1};
    IndexArrayType j_a    = {0, 1, 0};
    std::vector<double> v_a = {1, 99,
                               99   };
    Matrix<double, DirectedMatrixTag> a(2, 2);
    a.build(i_a, j_a, v_a);

    IndexArrayType vect_I({1,2});
    IndexArrayType vect_J({1,2});

    IndexArrayType i_result    = {0, 0, 0, 1, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_result    = {1, 2, 3, 0, 1, 2, 3, 0, 1, 2};
    std::vector<double> v_result = {   1,     2, 3,
                                    4, 1,   105, 7,
                                    8, 108,   1   };
    Matrix<double, DirectedMatrixTag> result(3, 4);
    result.build(i_result, j_result, v_result);

    assign(c, GraphBLAS::NoMask(), GraphBLAS::Plus<double>(),
           a, vect_I, vect_J);

    BOOST_CHECK_EQUAL(c, result);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(assign_mat_test_allrows_no_accum)
{
    IndexArrayType i_c    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_c    = {1, 2, 3, 0, 2, 3, 0, 1, 2};
    std::vector<double> v_c = {   1, 2, 3,
                               4,    6, 7,
                               8, 9, 1   };
    Matrix<double, DirectedMatrixTag> c(3, 4);
    c.build(i_c, j_c, v_c);

    IndexArrayType i_a    = {0, 0,
                             1,
                             2};
    IndexArrayType j_a    = {0, 1,
                             0,
                             0};
    std::vector<double> v_a = {1,  99,
                               1,
                               99};
    Matrix<double, DirectedMatrixTag> a(3, 2);
    a.build(i_a, j_a, v_a);

    //IndexArrayType vect_I({1,2});
    IndexArrayType vect_J({0,1});

    IndexArrayType i_result    = {0, 0, 0, 0, 1, 1, 1, 2, 2};
    IndexArrayType j_result    = {0, 1, 2, 3, 0, 2, 3, 0, 2};
    std::vector<double> v_result = {1, 99, 2,  3,
                                    1,     6,  7,
                                    99,    1   };
    Matrix<double, DirectedMatrixTag> result(3, 4);
    result.build(i_result, j_result, v_result);

    assign(c, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
           a, GraphBLAS::AllIndices(), vect_J);

    BOOST_CHECK_EQUAL(c, result);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(assign_mat_test_allcols_no_accum)
{
    IndexArrayType i_c    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_c    = {1, 2, 3, 0, 2, 3, 0, 1, 2};
    std::vector<double> v_c = {   1, 2, 3,
                               4,    6, 7,
                               8, 9, 1   };
    Matrix<double, DirectedMatrixTag> c(3, 4);
    c.build(i_c, j_c, v_c);

    IndexArrayType i_a    = {    0,      0,
                             1,  1,  1    };
    IndexArrayType j_a    = {    1,      3,
                             0,  1,  2    };
    std::vector<double> v_a = {   99,     99,
                               1, 99, 99    };
    Matrix<double, DirectedMatrixTag> a(2, 4);
    a.build(i_a, j_a, v_a);

    IndexArrayType vect_I({1,2});
    //IndexArrayType vect_J({0,1});

    IndexArrayType i_result    = {0, 0, 0, 1, 1, 2, 2, 2};
    IndexArrayType j_result    = {1, 2, 3, 1, 3, 0, 1, 2};
    std::vector<double> v_result = {    1,  2,  3,
                                       99,     99,
                                    1, 99, 99    };
    Matrix<double, DirectedMatrixTag> result(3, 4);
    result.build(i_result, j_result, v_result);

    assign(c, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
           a, vect_I, GraphBLAS::AllIndices());

    BOOST_CHECK_EQUAL(c, result);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(assign_mat_test_allrowscols_no_accum)
{
    IndexArrayType i_c    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_c    = {1, 2, 3, 0, 2, 3, 0, 1, 2};
    std::vector<double> v_c = {   1, 2, 3,
                               4,    6, 7,
                               8, 9, 1   };
    Matrix<double, DirectedMatrixTag> c(3, 4);
    c.build(i_c, j_c, v_c);

    IndexArrayType i_a    = {    0,      0,
                             1,  1,  1,
                             2,  2};
    IndexArrayType j_a    = {    1,      3,
                             0,  1,  2,
                             0,  1    };
    std::vector<double> v_a = {    99,     99,
                                1, 99, 99,
                               97, 96    };
    Matrix<double, DirectedMatrixTag> a(3, 4);
    a.build(i_a, j_a, v_a);

    assign(c, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
           a, GraphBLAS::AllIndices(), GraphBLAS::AllIndices());

    BOOST_CHECK_EQUAL(c, a);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(assign_mat_test_allrowscols_accum)
{
    IndexArrayType i_c    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_c    = {1, 2, 3, 0, 2, 3, 0, 1, 2};
    std::vector<double> v_c = {   1, 2, 3,
                               4,    6, 7,
                               8, 9, 1   };
    Matrix<double, DirectedMatrixTag> c(3, 4);
    c.build(i_c, j_c, v_c);

    IndexArrayType i_a    = {    0,      0,
                             1,  1,  1,
                             2,  2};
    IndexArrayType j_a    = {    1,      3,
                             0,  1,  2,
                             0,  1    };
    std::vector<double> v_a = {    99,     99,
                                1, 99, 99,
                               97, 96    };
    Matrix<double, DirectedMatrixTag> a(3, 4);
    a.build(i_a, j_a, v_a);

    assign(c, GraphBLAS::NoMask(), GraphBLAS::Plus<double>(),
           a, GraphBLAS::AllIndices(), GraphBLAS::AllIndices());

    IndexArrayType i_result    = {0, 0, 0, 1, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_result    = {1, 2, 3, 0, 1, 2, 3, 0, 1, 2};
    std::vector<double> v_result = {     100,   2, 102,
                                      5,  99, 105,   7,
                                    105, 105,   1     };
    Matrix<double, DirectedMatrixTag> result(3, 4);
    result.build(i_result, j_result, v_result);

    BOOST_CHECK_EQUAL(c, result);
}

/// @todo add tests with masks

//****************************************************************************
// 4.3.7.5 Vector constant tests
//****************************************************************************
BOOST_AUTO_TEST_CASE(assign_vec_constant_test_bad_dimensions)
{
    IndexArrayType i_w = {1, 2, 3};
    std::vector<double> v_w  = {1, 2, 3};
    Vector<double> w(4);
    w.build(i_w, v_w);

    IndexArrayType i_m    = {1};  // Dimension mismatch only possible with mask
    std::vector<double> v_m = {99};
    Vector<double> m(2);
    m.build(i_m, v_m);

    IndexArrayType vect_I({1,3,2});

    BOOST_CHECK_THROW(
        (assign(w, m, GraphBLAS::NoAccumulate(), 3, vect_I)),
        DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(assign_vec_constant_test_index_out_of_range)
{
    IndexArrayType i_w = {1, 2, 3};
    std::vector<double> v_w  = {1, 2, 3};
    Vector<double> w(4);
    w.build(i_w, v_w);

    IndexArrayType i_a    = {1};
    std::vector<double> v_a = {99};
    Vector<double> u(2);
    u.build(i_a, v_a);

    IndexArrayType vect_I({0,4});

    BOOST_CHECK_THROW(
        (assign(w, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                u, vect_I)),
        IndexOutOfBoundsException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(assign_vec_constant_test_no_accum)
{
    IndexArrayType i_w = {0, 1, 2};
    std::vector<double> v_w = {8, 9, 1   };
    Vector<double> c(4);
    c.build(i_w, v_w);

    IndexArrayType vect_I({1,3});

    IndexArrayType i_result      = {0, 1, 2, 3};
    std::vector<double> v_result = {8,99, 1,99};
    Vector<double> result(4);
    result.build(i_result, v_result);

    assign(c, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(), 99, vect_I);

    BOOST_CHECK_EQUAL(c, result);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(assign_vec_constant_test_accum)
{
    IndexArrayType i_w = {0, 1, 2};
    std::vector<double> v_w = {8, 9, 1   };
    Vector<double> c(4);
    c.build(i_w, v_w);

    IndexArrayType vect_I({1,3});

    IndexArrayType i_result      = {0,  1,  2,  3};
    std::vector<double> v_result = {8,108,  1, 99};
    Vector<double> result(4);
    result.build(i_result, v_result);

    assign(c, GraphBLAS::NoMask(), GraphBLAS::Plus<double>(), 99, vect_I);

    BOOST_CHECK_EQUAL(c, result);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(assign_vec_constant_test_allindices_no_accum)
{
    IndexArrayType i_w = {0, 1, 2};
    std::vector<double> v_w = {8, 9, 1   };
    Vector<double> c(4);
    c.build(i_w, v_w);

    IndexArrayType i_result      = {0,  1,  2,  3};
    std::vector<double> v_result = {99, 99, 99, 99};
    Vector<double> result(4);
    result.build(i_result, v_result);

    assign(c, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
           99, GraphBLAS::AllIndices());

    BOOST_CHECK_EQUAL(c, result);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(assign_vec_constant_test_allindices_accum)
{
    IndexArrayType i_w = {0, 1, 2};
    std::vector<double> v_w = {8, 9, 1   };
    Vector<double> c(4);
    c.build(i_w, v_w);

    assign(c, GraphBLAS::NoMask(), GraphBLAS::Plus<double>(),
           99, GraphBLAS::AllIndices());

    IndexArrayType i_result      = {0, 1,  2,  3};
    std::vector<double> v_result = {107, 108, 100, 99};
    Vector<double> result(4);
    result.build(i_result, v_result);

    BOOST_CHECK_EQUAL(c, result);
}

/// @todo add tests with masks

//****************************************************************************
// 4.3.7.6 Matrix constant tests
//****************************************************************************
BOOST_AUTO_TEST_CASE(assign_mat_constant_test_bad_dimensions)
{
    IndexArrayType      i_c = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType      j_c = {1, 2, 3, 0, 2, 3, 0, 1, 2};
    std::vector<double> v_c = {1, 2, 3, 4, 6, 7, 8, 9, 1};
    Matrix<double, DirectedMatrixTag> c(3, 4);
    c.build(i_c, j_c, v_c);

    IndexArrayType i_m    = {0, 0, 1};
    IndexArrayType j_m    = {0, 1, 0};
    std::vector<double> v_m = {1, 99, 99};
    Matrix<double, DirectedMatrixTag> m(2, 2);
    m.build(i_m, j_m, v_m);

    IndexArrayType vect_I({1,0,1});
    IndexArrayType vect_J({1,2});

    BOOST_CHECK_THROW(
        (assign(c, m, GraphBLAS::NoAccumulate(),
                99, vect_I, vect_J)),
        DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(assign_mat_constant_test_index_out_of_range)
{
    IndexArrayType      i_c = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType      j_c = {1, 2, 3, 0, 2, 3, 0, 1, 2};
    std::vector<double> v_c = {1, 2, 3, 4, 6, 7, 8, 9, 1};
    Matrix<double, DirectedMatrixTag> c(3, 4);
    c.build(i_c, j_c, v_c);

    IndexArrayType vect_I({1,5});
    IndexArrayType vect_J({1,2});

    BOOST_CHECK_THROW(
        (assign(c, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                99, vect_I, vect_J)),
        IndexOutOfBoundsException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(assign_mat_constant_test_no_accum)
{
    IndexArrayType i_c    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_c    = {1, 2, 3, 0, 2, 3, 0, 1, 2};
    std::vector<double> v_c = {   1, 2, 3,
                               4,    6, 7,
                               8, 9, 1   };
    Matrix<double, DirectedMatrixTag> c(3, 4);
    c.build(i_c, j_c, v_c);

    IndexArrayType vect_I({1,2});
    IndexArrayType vect_J({1,2});

    IndexArrayType i_result    = {0, 0, 0, 1, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_result    = {1, 2, 3, 0, 1, 2, 3, 0, 1, 2};
    std::vector<double> v_result = {    1,  2,  3,
                                    4, 99, 99,  7,
                                    8, 99, 99    };
    Matrix<double, DirectedMatrixTag> result(3, 4);
    result.build(i_result, j_result, v_result);

    assign(c, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
           99, vect_I, vect_J);

    BOOST_CHECK_EQUAL(c, result);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(assign_mat_constant_test_accum)
{
    IndexArrayType i_c    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_c    = {1, 2, 3, 0, 2, 3, 0, 1, 2};
    std::vector<double> v_c = {   1, 2, 3,
                               4,    6, 7,
                               8, 9, 1   };
    Matrix<double, DirectedMatrixTag> c(3, 4);
    c.build(i_c, j_c, v_c);

    IndexArrayType vect_I({1,2});
    IndexArrayType vect_J({1,2});

    IndexArrayType i_result    = {0, 0, 0, 1, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_result    = {1, 2, 3, 0, 1, 2, 3, 0, 1, 2};
    std::vector<double> v_result = {     1,   2, 3,
                                    4,  99, 105, 7,
                                    8, 108, 100   };
    Matrix<double, DirectedMatrixTag> result(3, 4);
    result.build(i_result, j_result, v_result);

    assign(c, GraphBLAS::NoMask(), GraphBLAS::Plus<double>(),
           99, vect_I, vect_J);

    BOOST_CHECK_EQUAL(c, result);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(assign_mat_constant_test_allrows_no_accum)
{
    IndexArrayType i_c    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_c    = {1, 2, 3, 0, 2, 3, 0, 1, 2};
    std::vector<double> v_c = {   1, 2, 3,
                               4,    6, 7,
                               8, 9, 1   };
    Matrix<double, DirectedMatrixTag> c(3, 4);
    c.build(i_c, j_c, v_c);

    //IndexArrayType vect_I({1,2});
    IndexArrayType vect_J({0,1});

    IndexArrayType i_result    = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_result    = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2};
    std::vector<double> v_result = {99, 99,  2,  3,
                                    99, 99,  6,  7,
                                    99, 99,  1    };
    Matrix<double, DirectedMatrixTag> result(3, 4);
    result.build(i_result, j_result, v_result);

    assign(c, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
           99, GraphBLAS::AllIndices(), vect_J);

    BOOST_CHECK_EQUAL(c, result);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(assign_mat_constant_test_allcols_no_accum)
{
    IndexArrayType i_c    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_c    = {1, 2, 3, 0, 2, 3, 0, 1, 2};
    std::vector<double> v_c = {   1, 2, 3,
                               4,    6, 7,
                               8, 9, 1   };
    Matrix<double, DirectedMatrixTag> c(3, 4);
    c.build(i_c, j_c, v_c);

    IndexArrayType vect_I({1,2});
    //IndexArrayType vect_J({0,1});

    IndexArrayType i_result    = {0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    IndexArrayType j_result    = {1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    std::vector<double> v_result = {     1,  2,  3,
                                    99, 99, 99, 99,
                                    99, 99, 99, 99};
    Matrix<double, DirectedMatrixTag> result(3, 4);
    result.build(i_result, j_result, v_result);

    assign(c, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
           99, vect_I, GraphBLAS::AllIndices());

    BOOST_CHECK_EQUAL(c, result);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(assign_mat_constant_test_allrowscols_no_accum)
{
    IndexArrayType i_c    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_c    = {1, 2, 3, 0, 2, 3, 0, 1, 2};
    std::vector<double> v_c = {   1, 2, 3,
                               4,    6, 7,
                               8, 9, 1   };
    Matrix<double, DirectedMatrixTag> c(3, 4);
    c.build(i_c, j_c, v_c);

    IndexArrayType i_result    = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    IndexArrayType j_result    = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    std::vector<double> v_result = {99, 99, 99, 99,
                                    99, 99, 99, 99,
                                    99, 99, 99, 99};
    Matrix<double, DirectedMatrixTag> result(3, 4);
    result.build(i_result, j_result, v_result);

    assign(c, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
           99, GraphBLAS::AllIndices(), GraphBLAS::AllIndices());

    BOOST_CHECK_EQUAL(c, result);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(assign_mat_constant_test_allrowscols_accum)
{
    IndexArrayType i_c    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_c    = {1, 2, 3, 0, 2, 3, 0, 1, 2};
    std::vector<double> v_c = {   1, 2, 3,
                               4,    6, 7,
                               8, 9, 1   };
    Matrix<double, DirectedMatrixTag> c(3, 4);
    c.build(i_c, j_c, v_c);

    assign(c, GraphBLAS::NoMask(), GraphBLAS::Plus<double>(),
           99, GraphBLAS::AllIndices(), GraphBLAS::AllIndices());

    IndexArrayType i_result    = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    IndexArrayType j_result    = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    std::vector<double> v_result = {99, 100, 101, 102,
                                    103, 99, 105, 106,
                                    107, 108,100,  99};
    Matrix<double, DirectedMatrixTag> result(3, 4);
    result.build(i_result, j_result, v_result);

    BOOST_CHECK_EQUAL(c, result);
}

/// @todo add tests with masks

BOOST_AUTO_TEST_SUITE_END()
