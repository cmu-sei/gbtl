/*
 * GraphBLAS Template Library, Version 2.0
 *
 * Copyright 2018 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors. All Rights Reserved.
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
 * RIGHTS..
 *
 * Released under a BSD (SEI)-style license, please see license.txt or contact
 * permission@sei.cmu.edu for full terms.
 *
 * This release is an update of:
 *
 * 1. GraphBLAS Template Library (GBTL)
 * (https://github.com/cmu-sei/gbtl/blob/1.0.0/LICENSE) Copyright 2015 Carnegie
 * Mellon University and The Trustees of Indiana. DM17-0037, DM-0002659
 *
 * DM18-0559
 */

#include <graphblas/graphblas.hpp>

#include <iostream>

using namespace GraphBLAS;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE complement_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_complement_square)
{
    //IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    //IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    //std::vector<double>       v_mA    = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    //Matrix<double, DirectedMatrixTag> mA(3, 3);
    //mA.build(i_mA, j_mA, v_mA);

    //std::vector<double>   v_answer    = {0, 1, 1, 1, 0, 1, 1, 1, 0};
    //Matrix<double, DirectedMatrixTag> answer(3, 3);
    //answer.build(i_mA, j_mA, v_answer);

    IndexArrayType i_mA    = {0, 1, 2};
    IndexArrayType j_mA    = {0, 1, 2};
    std::vector<double>       v_mA    = {1, 1, 1};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mA_n    = {0, 0, 1, 1, 2, 2 };
    IndexArrayType j_mA_n    = {1, 2, 0, 2, 0, 1 };
    std::vector<double>   v_answer    = {1, 1, 1, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    answer.build(i_mA_n, j_mA_n, v_answer);

    auto result = complement(mA);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_complementview_square)
{
    //IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    //IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    //std::vector<double>       v_mA    = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    //Matrix<double, DirectedMatrixTag> mA(3, 3);
    //mA.build(i_mA, j_mA, v_mA);

    //std::vector<double>   v_answer    = {0, 1, 1, 1, 0, 1, 1, 1, 0};
    //Matrix<double, DirectedMatrixTag> answer(3, 3);
    //answer.build(i_mA, j_mA, v_answer);

    IndexArrayType i_mA    = {0, 1, 2};
    IndexArrayType j_mA    = {0, 1, 2};
    std::vector<double>       v_mA    = {1, 1, 1};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mA_n    = {0, 0, 1, 1, 2, 2 };
    IndexArrayType j_mA_n    = {1, 2, 0, 2, 0, 1 };
    std::vector<double>   v_answer    = {1, 1, 1,1, 1, 1};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    answer.build(i_mA_n, j_mA_n, v_answer);

    auto result = complement(mA); //ComplementView<Matrix<double, DirectedMatrixTag>,
                              //ArithmeticSemiring<double> >(mA);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_complement_nonsquare)
{
    //IndexArrayType i_mA    = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    //IndexArrayType j_mA    = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    //std::vector<double>       v_mA    = {1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1};
    //Matrix<double, DirectedMatrixTag> mA(3, 4);
    //mA.build(i_mA, j_mA, v_mA);

    //std::vector<double>   v_answer    = {0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0};
    //Matrix<double, DirectedMatrixTag> answer(3, 4);
    //answer.build(i_mA, j_mA, v_answer);

    IndexArrayType i_mA    = {0, 0, 1, 1, 2, 2};
    IndexArrayType j_mA    = {0, 2, 1, 3, 2, 3};
    std::vector<double>       v_mA    = {1, 1, 1, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> mA(3, 4);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mA_n    = {0, 0, 1, 1, 2, 2};
    IndexArrayType j_mA_n   =  {1, 3, 0, 2, 0, 1};
    std::vector<double>   v_answer    =   {1, 1, 1, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> answer(3, 4);
    answer.build(i_mA_n, j_mA_n, v_answer);
    auto result = complement(mA);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_complementview_nonsquare)
{
    //IndexArrayType i_mA    = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    //IndexArrayType j_mA    = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    //std::vector<double>       v_mA    = {1, 0, 2, 0, 0, 3, 0,-1, 0, 0,-2, 6};
    //Matrix<double, DirectedMatrixTag> mA(3, 4);
    //mA.build(i_mA, j_mA, v_mA);

    //std::vector<double>   v_answer    = {0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0};
    //Matrix<double, DirectedMatrixTag> answer(3, 4);
    //answer.build(i_mA, j_mA, v_answer);

    IndexArrayType i_mA    = {0, 0, 1, 1, 2, 2};
    IndexArrayType j_mA    = {0, 2, 1, 3, 2, 3};
    std::vector<double>       v_mA    = {1, 1, 1, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> mA(3, 4);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mA_n    = {0, 0, 1, 1, 2, 2};
    IndexArrayType j_mA_n   =  {1, 3, 0, 2, 0, 1};
    std::vector<double>   v_answer    =   {1, 1, 1, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> answer(3, 4);
    answer.build(i_mA_n, j_mA_n, v_answer);

    auto result = complement(mA); //ComplementView<Matrix<double, DirectedMatrixTag>,
                              //ArithmeticSemiring<double> >(mA);

    BOOST_CHECK_EQUAL(result, answer);
}

/// @todo transpose of complement and complement of transpose not supported

#if 0
//****************************************************************************
BOOST_AUTO_TEST_CASE(test_complement_transpose_nonsquare)
{
    //IndexArrayType i_mA    = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    //IndexArrayType j_mA    = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    //std::vector<double>       v_mA    = {1, 0, 2, 0, 0, 3, 0,-1, 0, 0,-2, 6};
    //Matrix<double, DirectedMatrixTag> mA(3, 4);
    //mA.build(i_mA, j_mA, v_mA);

    //std::vector<double>   v_answer    = {0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0};
    //Matrix<double, DirectedMatrixTag> answer(4, 3);
    //answer.build(j_mA, i_mA, v_answer);

    /// @todo This currently does not work (TransposeView is destructed)
    //auto result = complement(transpose(mA));

    IndexArrayType i_mA    = {0, 0, 1, 1, 2, 2};
    IndexArrayType j_mA    = {0, 2, 1, 3, 2, 3};
    std::vector<double>       v_mA    = {1,  2,   3, -1,  -2, 6};
    Matrix<double, DirectedMatrixTag> mA(3, 4);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mA_n    = {0, 0, 1, 1, 2, 2};
    IndexArrayType j_mA_n   =  {1, 3, 0, 2, 0, 1};
    std::vector<double>   v_answer    =   {1, 1, 1, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> answer(4, 3);
    answer.build(j_mA_n, i_mA_n, v_answer);
    auto tres = transpose(mA);
    auto result = complement(tres);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_transpose_complement_nonsquare)
{
    //IndexArrayType i_mA    = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    //IndexArrayType j_mA    = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    //std::vector<double>       v_mA    = {1, 0, 2, 0, 0, 3, 0,-1, 0, 0,-2, 6};
    //Matrix<double, DirectedMatrixTag> mA(3, 4);
    //mA.build(i_mA, j_mA, v_mA);

    //std::vector<double>   v_answer    = {0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0};
    //Matrix<double, DirectedMatrixTag> answer(4, 3);
    //answer.build(j_mA, i_mA, v_answer);

    /// @todo This currently does not work (ComplementView is destructed)
    //auto result = transpose(complement(mA));
    IndexArrayType i_mA    = {0, 0, 1, 1, 2, 2};
    IndexArrayType j_mA    = {0, 2, 1, 3, 2, 3};
    std::vector<double>       v_mA    = {1, 2, 3,-1,-2, 6};
    Matrix<double, DirectedMatrixTag> mA(3, 4);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mA_n    = {0, 0, 1, 1, 2, 2};
    IndexArrayType j_mA_n   =  {1, 3, 0, 2, 0, 1};
    std::vector<double>   v_answer    =   {1, 1, 1, 1, 1, 1};
    Matrix<double, DirectedMatrixTag> answer(4, 3);
    answer.build(j_mA_n, i_mA_n, v_answer);
    auto nres = complement(mA);
    auto result = transpose(nres);

    BOOST_CHECK_EQUAL(result, answer);
}
#endif

/// @todo Need many more tests involving vector masks

BOOST_AUTO_TEST_SUITE_END()
