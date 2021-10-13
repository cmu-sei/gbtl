/*
 * GraphBLAS Template Library (GBTL), Version 3.0
 *
 * Copyright 2021 Carnegie Mellon University, Battelle Memorial Institute, and
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
 * This Software includes and/or makes use of the following Third-Party Software
 * subject to its own license:
 *
 * 1. Boost Unit Test Framework
 * (https://www.boost.org/doc/libs/1_45_0/libs/test/doc/html/utf.html)
 * Copyright 2001 Boost software license, Gennadiy Rozental.
 *
 * DM20-0442
 */

#define GRAPHBLAS_LOGGING_LEVEL 0

#include <functional>
#include <iostream>
#include <vector>

#include <graphblas/graphblas.hpp>

using namespace grb;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE apply_indexunaryop_test_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

//****************************************************************************
// apply iu-op matrix
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(applyi_mat_rowindex)
{
    // | 1 1 - - |
    // | 1 2 2 - |
    // | - 2 3 3 |
    // | - - 3 4 |

    // Build some sparse matrices.
    IndexArrayType i      = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j      = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4};
    Matrix<double> A(4, 4);
    A.build(i, j, v);

    IndexArrayType ia      = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType ja      = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> va = {-2, -2, -1, -1, -1, 0, 0, 0, 1, 1};
    grb::Matrix<double> answer(4, 4);
    answer.build(ia, ja, va);

    IndexArrayType im      = {0, 1, 2, 3, 3};
    IndexArrayType jm      = {0, 0, 1, 2, 3};
    std::vector<double> vm = {true, true, true, true, true};
    grb::Matrix<bool> Mask(4, 4);
    Mask.build(im, jm, vm);

    {
        Matrix<double> C(4, 4);
        grb::apply(C, grb::NoMask(), grb::NoAccumulate(),
                   grb::RowIndex<double, int64_t>(),
                   A, -2);

        BOOST_CHECK_EQUAL(C, answer);
    }

    {
        // Transpose
        Matrix<double> C(4, 4);
        grb::apply(C, grb::NoMask(), grb::NoAccumulate(),
                   grb::RowIndex<double, int64_t>(),
                   grb::transpose(A), -2);

        BOOST_CHECK_EQUAL(C, answer);
    }

    {
        Matrix<double> C(4, 4);
        grb::apply(C, grb::NoMask(), grb::NoAccumulate(),
                   [](float a_ij, grb::IndexType i, grb::IndexType j,
                      int64_t offset)
                   { return static_cast<int64_t>(i) + offset; },
                   A, -2);

        BOOST_CHECK_EQUAL(C, answer);
    }

    {
        // Masked
        IndexArrayType ib      = { 0,  1, 2, 3, 3};
        IndexArrayType jb      = { 0,  0, 1, 2, 3};
        std::vector<double> vb = {-2, -1, 0, 1, 1};
        grb::Matrix<double> answer2(4, 4);
        answer2.build(ib, jb, vb);

        Matrix<double> C(4, 4);
        grb::apply(C, Mask, grb::NoAccumulate(),
                   grb::RowIndex<double, int64_t>(),
                   A, -2);

        BOOST_CHECK_EQUAL(C, answer2);

        // Accumlate
        IndexArrayType ic      = { 0,  0,  1,  1,  1, 2, 2, 2, 3, 3};
        IndexArrayType jc      = { 0,  1,  0,  1,  2, 1, 2, 3, 2, 3};
        std::vector<double> vc = {-4, -2, -2, -1, -1, 0, 0, 0, 2, 2};
        grb::Matrix<double> answer3(4, 4);
        answer2.build(ic, jc, vc);
        grb::apply(C, grb::NoMask(), grb::Plus<double>(),
                   grb::RowIndex<double, int64_t>(),
                   A, -2);

        BOOST_CHECK_EQUAL(C, answer2);
    }
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(applyi_mat_colindex)
{
    // | 1 1 - - |
    // | 1 2 2 - |
    // | - 2 3 3 |
    // | - - 3 4 |

    // Build some sparse matrices.
    IndexArrayType i      = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j      = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4};
    Matrix<double> A(4, 4);
    A.build(i, j, v);

    IndexArrayType ia      = { 0,  0,  1,  1,  1,  2, 2, 2, 3, 3};
    IndexArrayType ja      = { 0,  1,  0,  1,  2,  1, 2, 3, 2, 3};
    std::vector<double> va = {-2, -1, -2, -1,  0, -1, 0, 1, 0, 1};
    grb::Matrix<double> answer(4, 4);
    answer.build(ia, ja, va);

    IndexArrayType im      = {0, 1, 2, 3, 3};
    IndexArrayType jm      = {0, 0, 1, 2, 3};
    std::vector<double> vm = {true, true, true, true, true};
    grb::Matrix<bool> Mask(4, 4);
    Mask.build(im, jm, vm);

    {
        Matrix<double> C(4, 4);
        grb::apply(C, grb::NoMask(), grb::NoAccumulate(),
                   grb::ColIndex<double, int64_t>(),
                   A, -2);

        BOOST_CHECK_EQUAL(C, answer);
    }

    {
        // Transpose
        Matrix<double> C(4, 4);
        grb::apply(C, grb::NoMask(), grb::NoAccumulate(),
                   grb::ColIndex<double, int64_t>(),
                   grb::transpose(A), -2);

        BOOST_CHECK_EQUAL(C, answer);
    }

    {
        Matrix<double> C(4, 4);
        grb::apply(C, grb::NoMask(), grb::NoAccumulate(),
                   [](float a_ij, grb::IndexType i, grb::IndexType j,
                      int64_t offset)
                   { return static_cast<int64_t>(j) + offset; },
                   A, -2);

        BOOST_CHECK_EQUAL(C, answer);
    }

    {
        // Masked
        IndexArrayType ib      = { 0,  1,  2, 3, 3};
        IndexArrayType jb      = { 0,  0,  1, 2, 3};
        std::vector<double> vb = {-2, -2, -1, 0, 1};
        grb::Matrix<double> answer2(4, 4);
        answer2.build(ib, jb, vb);

        Matrix<double> C(4, 4);
        grb::apply(C, Mask, grb::NoAccumulate(),
                   grb::ColIndex<double, int64_t>(),
                   A, -2);

        BOOST_CHECK_EQUAL(C, answer2);

        // Accumlate
        IndexArrayType ic      = { 0,  0,  1,  1,  1,  2, 2, 2, 3, 3};
        IndexArrayType jc      = { 0,  1,  0,  1,  2,  1, 2, 3, 2, 3};
        std::vector<double> vc = {-4, -1, -4, -1,  0, -2, 0, 1, 0, 2};
        grb::Matrix<double> answer3(4, 4);
        answer3.build(ic, jc, vc);
        grb::apply(C, grb::NoMask(), grb::Plus<double>(),
                   grb::ColIndex<double, int64_t>(),
                   A, -2);

        BOOST_CHECK_EQUAL(C, answer3);
    }
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(applyi_mat_diagindex)
{
    // | 1 1 - - |
    // | 1 2 2 - |
    // | - 2 3 3 |
    // | - - 3 4 |

    // Build some sparse matrices.
    IndexArrayType i      = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j      = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4};
    Matrix<double> A(4, 4);
    A.build(i, j, v);

    IndexArrayType ia      = { 0,  0,  1,  1,  1,  2,  2,  2,  3,  3};
    IndexArrayType ja      = { 0,  1,  0,  1,  2,  1,  2,  3,  2,  3};
    std::vector<double> va = {-2, -1, -3, -2, -1, -3, -2, -1, -3, -2};
    grb::Matrix<double> answer(4, 4);
    answer.build(ia, ja, va);

    IndexArrayType im      = {0, 1, 2, 3, 3};
    IndexArrayType jm      = {0, 0, 1, 2, 3};
    std::vector<double> vm = {true, true, true, true, true};
    grb::Matrix<bool> Mask(4, 4);
    Mask.build(im, jm, vm);

    {
        Matrix<double> C(4, 4);
        grb::apply(C, grb::NoMask(), grb::NoAccumulate(),
                   grb::DiagIndex<double, int64_t>(),
                   A, -2);

        BOOST_CHECK_EQUAL(C, answer);
    }

    {
        // Transpose
        Matrix<double> C(4, 4);
        grb::apply(C, grb::NoMask(), grb::NoAccumulate(),
                   grb::DiagIndex<double, int64_t>(),
                  grb::transpose(A), -2);

        BOOST_CHECK_EQUAL(C, answer);
    }

    {
        Matrix<double> C(4, 4);
        grb::apply(C, grb::NoMask(), grb::NoAccumulate(),
                   [](float a_ij, grb::IndexType i, grb::IndexType j,
                      int64_t offset)
                   { return (static_cast<int64_t>(j) -
                             static_cast<int64_t>(i) + offset); },
                   A, -2);

        BOOST_CHECK_EQUAL(C, answer);
    }

    {
        // Masked
        IndexArrayType ib      = { 0,  1,  2,  3,  3};
        IndexArrayType jb      = { 0,  0,  1,  2,  3};
        std::vector<double> vb = {-2, -3, -3, -3, -2};
        grb::Matrix<double> answer2(4, 4);
        answer2.build(ib, jb, vb);

        Matrix<double> C(4, 4);
        grb::apply(C, Mask, grb::NoAccumulate(),
                   grb::DiagIndex<double, int64_t>(),
                   A, -2);

        BOOST_CHECK_EQUAL(C, answer2);

        // Accumlate
        IndexArrayType ic      = { 0,  0,  1,  1,  1,  2,  2,  2,  3,  3};
        IndexArrayType jc      = { 0,  1,  0,  1,  2,  1,  2,  3,  2,  3};
        std::vector<double> vc = {-4, -1, -6, -2, -1, -6, -2, -1, -6, -4};
        grb::Matrix<double> answer3(4, 4);
        answer3.build(ic, jc, vc);
        grb::apply(C, grb::NoMask(), grb::Plus<double>(),
                   grb::DiagIndex<double, int64_t>(),
                   A, -2);

        BOOST_CHECK_EQUAL(C, answer3);
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(applyi_vec_rowindex)
{
    // | 1 1 - - |
    // | 1 2 2 - |
    // | - 2 3 3 |
    // | - - 3 4 |

    // Build some sparse vectors.
    std::vector<double> v = {0, 2, 3, 3};
    Vector<double> u(v, 0.0);

    std::vector<bool> m = {false, true, false, true};
    Vector<bool> mask(m, false);

    std::vector<double> ans = {0, 2, 3, 4};
    Vector<double> answer(ans, 0.);
    Vector<double> w(4);

    {
        grb::apply(w, grb::NoMask(), grb::NoAccumulate(),
                   grb::RowIndex<double, int64_t>(),
                   u, 1);

        BOOST_CHECK_EQUAL(w, answer);
    }

    {
        grb::apply(w, grb::NoMask(), grb::NoAccumulate(),
                   [](float w_i, grb::IndexType idx, int64_t offset)
                   { return static_cast<int64_t>(idx) + offset; },
                   u, 1);

        BOOST_CHECK_EQUAL(w, answer);
    }

    {
        // Masked
        std::vector<double> vb = {0, -1, 0, 1};
        grb::Vector<double> answer2(vb, 0.);

        Vector<double> w(4);
        grb::apply(w, mask, grb::NoAccumulate(),
                   grb::RowIndex<double, int64_t>(),
                   u, -2);

        BOOST_CHECK_EQUAL(w, answer2);

        // Accumlate
        std::vector<double> vc = {666, -2, 0, 2};
        grb::Vector<double> answer3(vc, 666.);

        grb::apply(w, grb::NoMask(), grb::Plus<double>(),
                   grb::RowIndex<double, int64_t>(),
                   u, -2);

        BOOST_CHECK_EQUAL(w, answer3);
    }
}

BOOST_AUTO_TEST_SUITE_END()
