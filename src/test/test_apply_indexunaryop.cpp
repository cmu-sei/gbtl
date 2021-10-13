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
BOOST_AUTO_TEST_CASE(apply_mat_rowindex)
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

    {
        Matrix<double> C(4, 4);
        grb::apply(C, grb::NoMask(), grb::NoAccumulate(),
                   grb::RowIndex<double, int64_t>(),
                   -2, A);

        BOOST_CHECK_EQUAL(C, answer);
    }

    {
        Matrix<double> C(4, 4);
        grb::apply(C, grb::NoMask(), grb::NoAccumulate(),
                   [](float a_ij, grb::IndexType i, grb::IndexType j,
                      int64_t offset)
                   { return static_cast<int64_t>(i) + offset; },
                   -2, A);

        BOOST_CHECK_EQUAL(C, answer);
    }
}
#if 0
//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_select_mat_rowgreater)
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

    IndexArrayType ia      = {2, 2, 2, 3, 3};
    IndexArrayType ja      = {1, 2, 3, 2, 3};
    std::vector<double> va = {2, 3, 3, 3, 4};
    grb::Matrix<double> answer(4, 4);
    answer.build(ia, ja, va);

    Matrix<double> C(4, 4);
    grb::apply(C, grb::NoMask(), grb::NoAccumulate(),
                grb::RowGreater<double>(),
                A, 1);

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_select_mat_colle)
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

    IndexArrayType ia      = {0, 0, 1, 1, 2};
    IndexArrayType ja      = {0, 1, 0, 1, 1};
    std::vector<double> va = {1, 1, 1, 2, 2};
    grb::Matrix<double> answer(4, 4);
    answer.build(ia, ja, va);

    Matrix<double> C(4, 4);
    grb::apply(C, grb::NoMask(), grb::NoAccumulate(),
                grb::ColLessEqual<double>(),
                A, 1);

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_select_mat_colgreater)
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

    IndexArrayType ia      = {1, 2, 2, 3, 3};
    IndexArrayType ja      = {2, 2, 3, 2, 3};
    std::vector<double> va = {2, 3, 3, 3, 4};
    grb::Matrix<double> answer(4, 4);
    answer.build(ia, ja, va);

    Matrix<double> C(4, 4);
    grb::apply(C, grb::NoMask(), grb::NoAccumulate(),
                grb::ColGreater<double>(),
                A, 1);

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_select_mat_valequal)
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

    IndexArrayType ia      = {1, 1, 2};
    IndexArrayType ja      = {1, 2, 1};
    std::vector<double> va = {2, 2, 2};
    grb::Matrix<double> answer(4, 4);
    answer.build(ia, ja, va);

    Matrix<double> C(4, 4);
    grb::apply(C, grb::NoMask(), grb::NoAccumulate(),
                grb::ValueEqual<double>(),
                A, 2);

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_select_mat_valneq)
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

    IndexArrayType ia      = {0, 0, 1, 2, 2, 3, 3};
    IndexArrayType ja      = {0, 1, 0, 2, 3, 2, 3};
    std::vector<double> va = {1, 1, 1, 3, 3, 3, 4};
    grb::Matrix<double> answer(4, 4);
    answer.build(ia, ja, va);

    Matrix<double> C(4, 4);
    grb::apply(C, grb::NoMask(), grb::NoAccumulate(),
                grb::ValueNotEqual<double>(),
                A, 2);

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_select_mat_valless)
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

    IndexArrayType ia      = {0, 0, 1, 1, 1, 2};
    IndexArrayType ja      = {0, 1, 0, 1, 2, 1};
    std::vector<double> va = {1, 1, 1, 2, 2, 2};
    grb::Matrix<double> answer(4, 4);
    answer.build(ia, ja, va);

    Matrix<double> C(4, 4);
    grb::apply(C, grb::NoMask(), grb::NoAccumulate(),
                grb::ValueLess<double>(),
                A, 3);

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_select_mat_valle)
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

    IndexArrayType ia      = {0, 0, 1, 1, 1, 2};
    IndexArrayType ja      = {0, 1, 0, 1, 2, 1};
    std::vector<double> va = {1, 1, 1, 2, 2, 2};
    grb::Matrix<double> answer(4, 4);
    answer.build(ia, ja, va);

    Matrix<double> C(4, 4);
    grb::apply(C, grb::NoMask(), grb::NoAccumulate(),
                grb::ValueLessEqual<double>(),
                A, 2);

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_select_mat_valgreater)
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

    IndexArrayType ia      = {1, 1, 2, 2, 2, 3, 3};
    IndexArrayType ja      = {1, 2, 1, 2, 3, 2, 3};
    std::vector<double> va = {2, 2, 2, 3, 3, 3, 4};
    grb::Matrix<double> answer(4, 4);
    answer.build(ia, ja, va);

    Matrix<double> C(4, 4);
    grb::apply(C, grb::NoMask(), grb::NoAccumulate(),
                grb::ValueGreater<double>(),
                A, 1);

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_select_mat_valge)
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

    IndexArrayType ia      = {1, 1, 2, 2, 2, 3, 3};
    IndexArrayType ja      = {1, 2, 1, 2, 3, 2, 3};
    std::vector<double> va = {2, 2, 2, 3, 3, 3, 4};
    grb::Matrix<double> answer(4, 4);
    answer.build(ia, ja, va);

    Matrix<double> C(4, 4);
    grb::apply(C, grb::NoMask(), grb::NoAccumulate(),
                grb::ValueGreaterEqual<double>(),
                A, 2);

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
// select standard vector
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_select_vec_rowle)
{
    // | - 2 3 3 |

    // Build some sparse vectors.
    std::vector<double> v = {0, 2, 3, 3};
    Vector<double> u(v, 0.0);

    std::vector<double> ans = {0, 2, 0, 0};
    Vector<double> answer(ans, 0.);
    Vector<double> w(4);

    grb::apply(w, grb::NoMask(), grb::NoAccumulate(),
                grb::RowLessEqual<double>(),
                u, 1);

    BOOST_CHECK_EQUAL(w, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_select_vec_rowgreater)
{
    // | - 2 3 3 |

    // Build some sparse vectors.
    std::vector<double> v = {0, 2, 3, 3};
    Vector<double> u(v, 0.0);

    std::vector<double> ans = {0, 0, 3, 3};
    Vector<double> answer(ans, 0.);
    Vector<double> w(4);

    grb::apply(w, grb::NoMask(), grb::NoAccumulate(),
                grb::RowGreater<double>(),
                u, 1);

    BOOST_CHECK_EQUAL(w, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_select_vec_valequal)
{
    // | - 2 3 3 |

    // Build some sparse vectors.
    std::vector<double> v = {0, 2, 3, 3};
    Vector<double> u(v, 0.0);

    std::vector<double> ans = {0, 2, 0, 0};
    Vector<double> answer(ans, 0.);
    Vector<double> w(4);

    grb::apply(w, grb::NoMask(), grb::NoAccumulate(),
                grb::ValueEqual<double>(),
                u, 2);

    BOOST_CHECK_EQUAL(w, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_select_vec_valnotequal)
{
    // | - 2 3 3 |

    // Build some sparse vectors.
    std::vector<double> v = {0, 2, 3, 3};
    Vector<double> u(v, 0.0);

    std::vector<double> ans = {0, 0, 3, 3};
    Vector<double> answer(ans, 0.);
    Vector<double> w(4);

    grb::apply(w, grb::NoMask(), grb::NoAccumulate(),
                grb::ValueNotEqual<double>(),
                u, 2);

    BOOST_CHECK_EQUAL(w, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_select_vec_valless)
{
    // | - 2 3 3 |

    // Build some sparse vectors.
    std::vector<double> v = {0, 2, 3, 3};
    Vector<double> u(v, 0.0);

    std::vector<double> ans = {0, 2, 0, 0};
    Vector<double> answer(ans, 0.);
    Vector<double> w(4);

    grb::apply(w, grb::NoMask(), grb::NoAccumulate(),
                grb::ValueLess<double>(),
                u, 3);

    BOOST_CHECK_EQUAL(w, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_select_vec_valle)
{
    // | - 2 3 3 |

    // Build some sparse vectors.
    std::vector<double> v = {0, 2, 3, 3};
    Vector<double> u(v, 0.0);

    std::vector<double> ans = {0, 0, 0, 0};
    Vector<double> answer(ans, 0.);
    Vector<double> w(4);

    grb::apply(w, grb::NoMask(), grb::NoAccumulate(),
                grb::ValueLessEqual<double>(),
                u, 1);

    BOOST_CHECK_EQUAL(w, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_select_vec_valgreater)
{
    // | - 2 3 3 |

    // Build some sparse vectors.
    std::vector<double> v = {0, 2, 3, 3};
    Vector<double> u(v, 0.0);

    std::vector<double> ans = {0, 0, 3, 3};
    Vector<double> answer(ans, 0.);
    Vector<double> w(4);

    grb::apply(w, grb::NoMask(), grb::NoAccumulate(),
                grb::ValueGreater<double>(),
                u, 2);

    BOOST_CHECK_EQUAL(w, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(sparse_select_vec_valge)
{
    // | - 2 3 3 |

    // Build some sparse vectors.
    std::vector<double> v = {0, 2, 3, 3};
    Vector<double> u(v, 0.0);

    std::vector<double> ans = {0, 2, 3, 3};
    Vector<double> answer(ans, 0.);
    Vector<double> w(4);

    grb::apply(w, grb::NoMask(), grb::NoAccumulate(),
                grb::ValueGreaterEqual<double>(),
                u, 1);

    BOOST_CHECK_EQUAL(w, answer);
}
#endif
BOOST_AUTO_TEST_SUITE_END()
