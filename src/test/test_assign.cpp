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

BOOST_AUTO_TEST_SUITE(_assign_suite)

//****************************************************************************
BOOST_AUTO_TEST_CASE(assign_test_bad_dimensions)
{
    IndexArrayType i_c = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_c = {1, 2, 3, 0, 2, 3, 0, 1, 2};
    std::vector<double> v_c       = {1, 2, 3, 4, 6, 7, 8, 9, 1};
    Matrix<double, DirectedMatrixTag> c(3, 4);
    c.build(i_c, j_c, v_c);

    IndexArrayType i_a    = {0, 0, 1};
    IndexArrayType j_a    = {0, 1, 0};
    std::vector<double> v_a = {1, 99, 99};
    Matrix<double, DirectedMatrixTag> a(2, 2);
    a.build(i_a, j_a, v_a);

    IndexArrayType vect_I({1,3,2});
    IndexArrayType vect_J({1,2});

    std::cerr << "A" << std::endl;
    BOOST_CHECK_THROW(
        (assign(c, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                a, vect_I, vect_J)),
        DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(assign_test_default_accum)
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
                                    8,    99   };
    Matrix<double, DirectedMatrixTag> result(3, 4);
    result.build(i_result, j_result, v_result);

    std::cerr << "B" << std::endl;
    assign(c, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
           a, vect_I, vect_J);

    BOOST_CHECK_EQUAL(c, result);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(assign_test_accum)
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

    std::cerr << "B" << std::endl;
    assign(c, GraphBLAS::NoMask(), GraphBLAS::Plus<double>(),
           a, vect_I, vect_J);

    BOOST_CHECK_EQUAL(c, result);
}

BOOST_AUTO_TEST_SUITE_END()
