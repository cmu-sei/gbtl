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
 * This Software includes and/or makes use of the following Third-Party Software
 * subject to its own license:
 *
 * 1. Boost Unit Test Framework
 * (https://www.boost.org/doc/libs/1_45_0/libs/test/doc/html/utf.html)
 * Copyright 2001 Boost software license, Gennadiy Rozental.
 *
 * DM20-0442
 */

//#define GRAPHBLAS_LOGGING_LEVEL 2

#include <graphblas/graphblas.hpp>

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE kronecker_ABT_test_suite

#include <boost/test/included/unit_test.hpp>

using namespace grb;

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

//****************************************************************************

namespace
{
    static std::vector<std::vector<double> > A_sparse_3x3 =
    {{1,  0,  0},
     {0,  2,  0},
     {3,  0,  4}};

    static std::vector<std::vector<double> > AT_sparse_3x3 =
    {{1,  0,  3},
     {0,  2,  0},
     {0,  0,  4}};

    static std::vector<std::vector<double> > AA_sparse_9x9 =
    {{1,  0,  0,   0,  0,  0,   0,  0,  0},
     {0,  2,  0,   0,  0,  0,   0,  0,  0},
     {3,  0,  4,   0,  0,  0,   0,  0,  0},
     {0,  0,  0,   2,  0,  0,   0,  0,  0},
     {0,  0,  0,   0,  4,  0,   0,  0,  0},
     {0,  0,  0,   6,  0,  8,   0,  0,  0},
     {3,  0,  0,   0,  0,  0,   4,  0,  0},
     {0,  6,  0,   0,  0,  0,   0,  8,  0},
     {9,  0, 12,   0,  0,  0,  12,  0, 16}
    };

    static std::vector<std::vector<double> > AAT_sparse_9x9 =
    {{1,  0,  3,   0,  0,  0,   0,  0,  0},
     {0,  2,  0,   0,  0,  0,   0,  0,  0},
     {0,  0,  4,   0,  0,  0,   0,  0,  0},
     {0,  0,  0,   2,  0,  6,   0,  0,  0},
     {0,  0,  0,   0,  4,  0,   0,  0,  0},
     {0,  0,  0,   0,  0,  8,   0,  0,  0},
     {3,  0,  9,   0,  0,  0,   4,  0, 12},
     {0,  6,  0,   0,  0,  0,   0,  8,  0},
     {0,  0, 12,   0,  0,  0,   0,  0, 16}
    };

    static std::vector<std::vector<double> > Ar_sparse_3x3 =
    {{1,  0,  0},
     {0,  2,  0},
     {0,  0,  0}};

    static std::vector<std::vector<double> > Ac_sparse_3x3 =
    {{0,  0,  0},
     {0,  2,  0},
     {0,  0,  4}};

    static std::vector<std::vector<double> > B_sparse_3x4 =
    {{1,  1,  0,  0},
     {0,  2,  2,  0},
     {3,  0,  0,  3}};

    static std::vector<std::vector<double> > B_sparse_4x3 =
    {{1,  0,  3},
     {1,  2,  0},
     {0,  2,  0},
     {0,  0,  3}};

    static std::vector<std::vector<double> > Br_sparse_3x4 =
    {{0,  0,  0,  0},
     {0,  2,  2,  0},
     {3,  0,  0,  3}};

    static std::vector<std::vector<double> > Br_sparse_4x3 =
    {{0,  0,  3},
     {0,  2,  0},
     {0,  2,  0},
     {0,  0,  3}};

    static std::vector<std::vector<double> > Bc_sparse_3x4 =
    {{1,  0,  0,  0},
     {0,  0,  2,  0},
     {3,  0,  0,  3}};

    static std::vector<std::vector<double> > Bc_sparse_4x3 =
    {{1,  0,  3},
     {0,  0,  0},
     {0,  2,  0},
     {0,  0,  3}};

    static std::vector<std::vector<double> > BT_sparse_3x4 =
    {{1,  0,  3},
     {1,  2,  0},
     {0,  2,  0},
     {0,  0,  3}};

    static std::vector<std::vector<double> > Answer_rr_sparse_9x12 =
    {{0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {0, 2, 2, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {3, 0, 0, 3,  0, 0, 0, 0,  0, 0, 0, 0},

     {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  0, 4, 4, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  6, 0, 0, 6,  0, 0, 0, 0},

     {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0}
    };

    static std::vector<std::vector<double> > Answer_sparse_9x12 =
    {{1, 1, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {0, 2, 2, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {3, 0, 0, 3,  0, 0, 0, 0,  0, 0, 0, 0},

     {0, 0, 0, 0,  2, 2, 0, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  0, 4, 4, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  6, 0, 0, 6,  0, 0, 0, 0},

     {3, 3, 0, 0,  0, 0, 0, 0,  4, 4, 0, 0},
     {0, 6, 6, 0,  0, 0, 0, 0,  0, 8, 8, 0},
     {9, 0, 0, 9,  0, 0, 0, 0, 12, 0, 0,12}
    };

    static std::vector<std::vector<double> > Answer_sparse_9x12_Lower =
    {{1, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {0, 2, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {3, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},

     {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  6, 0, 0, 0,  0, 0, 0, 0},

     {3, 3, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {0, 6, 6, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {9, 0, 0, 9,  0, 0, 0, 0, 12, 0, 0, 0}
    };

    static std::vector<std::vector<double> > Answer_sparse_9x12_Lower_Ones =
    {{1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1},
     {0, 2, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1},
     {3, 0, 0, 1,  1, 1, 1, 1,  1, 1, 1, 1},

     {0, 0, 0, 0,  1, 1, 1, 1,  1, 1, 1, 1},
     {0, 0, 0, 0,  0, 1, 1, 1,  1, 1, 1, 1},
     {0, 0, 0, 0,  6, 0, 1, 1,  1, 1, 1, 1},

     {3, 3, 0, 0,  0, 0, 0, 1,  1, 1, 1, 1},
     {0, 6, 6, 0,  0, 0, 0, 0,  1, 1, 1, 1},
     {9, 0, 0, 9,  0, 0, 0, 0, 12, 1, 1, 1}
    };

    static std::vector<std::vector<double> > Answer_sparse_9x12_NotLower
    {{0, 1, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {0, 0, 2, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {0, 0, 0, 3,  0, 0, 0, 0,  0, 0, 0, 0},

     {0, 0, 0, 0,  2, 2, 0, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  0, 4, 4, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  0, 0, 0, 6,  0, 0, 0, 0},

     {0, 0, 0, 0,  0, 0, 0, 0,  4, 4, 0, 0},
     {0, 0, 0, 0,  0, 0, 0, 0,  0, 8, 8, 0},
     {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0,12}
    };

    static std::vector<std::vector<double> > Answer_sparse_9x12_NotLower_Ones
    {{1, 1, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {1, 1, 2, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {1, 1, 1, 3,  0, 0, 0, 0,  0, 0, 0, 0},

     {1, 1, 1, 1,  2, 2, 0, 0,  0, 0, 0, 0},
     {1, 1, 1, 1,  1, 4, 4, 0,  0, 0, 0, 0},
     {1, 1, 1, 1,  1, 1, 0, 6,  0, 0, 0, 0},

     {1, 1, 1, 1,  1, 1, 1, 0,  4, 4, 0, 0},
     {1, 1, 1, 1,  1, 1, 1, 1,  0, 8, 8, 0},
     {1, 1, 1, 1,  1, 1, 1, 1,  1, 0, 0,12}
    };

    static std::vector<std::vector<double> > Answer_rc_sparse_9x12 =
    {{1, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {0, 0, 2, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {3, 0, 0, 3,  0, 0, 0, 0,  0, 0, 0, 0},

     {0, 0, 0, 0,  2, 0, 0, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  0, 0, 4, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  6, 0, 0, 6,  0, 0, 0, 0},

     {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0}
    };

    static std::vector<std::vector<double> > Answer_cr_sparse_9x12 =
    {{0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},

     {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  0, 4, 4, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  6, 0, 0, 6,  0, 0, 0, 0},

     {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  0, 0, 0, 0,  0, 8, 8, 0},
     {0, 0, 0, 0,  0, 0, 0, 0, 12, 0, 0,12}
    };

    static std::vector<std::vector<double> > Ones_4x4 =
    {{1, 1, 1, 1},
     {1, 1, 1, 1},
     {1, 1, 1, 1},
     {1, 1, 1, 1}};

    static std::vector<std::vector<double> > Ones_3x4 =
    {{1, 1, 1, 1},
     {1, 1, 1, 1},
     {1, 1, 1, 1}};

    static std::vector<std::vector<double> > Ones_3x3 =
    {{1, 1, 1},
     {1, 1, 1},
     {1, 1, 1}};

    static std::vector<std::vector<double> > Ones_4x3 =
    {{1, 1, 1},
     {1, 1, 1},
     {1, 1, 1},
     {1, 1, 1}};

    static std::vector<std::vector<double> > Ones_9x9 =
    {{1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1}};

    static std::vector<std::vector<double> > Ones_9x12 =
    {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

    static std::vector<std::vector<double> > Identity_3x3 =
    {{1, 0, 0},
     {0, 1, 0},
     {0, 0, 1}};

    static std::vector<std::vector<double> > Lower_3x3 =
    {{1, 0, 0},
     {1, 1, 0},
     {1, 1, 1}};

    static std::vector<std::vector<double> > Lower_3x4 =
    {{1, 0, 0, 0},
     {1, 1, 0, 0},
     {1, 1, 1, 0}};

    static std::vector<std::vector<double> > Lower_4x4 =
    {{1, 0, 0, 0},
     {1, 1, 0, 0},
     {1, 1, 1, 0},
     {1, 1, 1, 1}};

    static std::vector<std::vector<double> > NotLower_3x3 =
    {{0, 1, 1},
     {0, 0, 1},
     {0, 0, 0}};

    static std::vector<std::vector<double> > NotLower_3x4 =
    {{0, 1, 1, 1},
     {0, 0, 1, 1},
     {0, 0, 0, 1}};

    static std::vector<std::vector<double> > NotLower_4x4 =
    {{0, 1, 1, 1},
     {0, 0, 1, 1},
     {0, 0, 0, 1},
     {0, 0, 0, 0}};

    static std::vector<std::vector<double> > LowerMask_3x4 =
    {{1, 0,    0,   0},
     {1, 0.5,  0,   0},
     {1, -1.0, 1.5, 0}};

    static std::vector<std::vector<bool> > LowerBool_3x4 =
    {{true, false, false, false},
     {true, true,  false, false},
     {true, true,  true,  false}};

    static std::vector<std::vector<bool> > LowerBool_3x3 =
    {{true, false, false},
     {true, true,  false},
     {true, true,  true}};

    static std::vector<std::vector<bool> > NotLowerBool_3x3 =
    {{false,  true, true},
     {false, false, true},
     {false, false, false}};

    static std::vector<std::vector<double> > Lower_9x9 =
    {{1, 0, 0, 0,  0, 0, 0, 0,  0},
     {1, 1, 0, 0,  0, 0, 0, 0,  0},
     {1, 1, 1, 0,  0, 0, 0, 0,  0},
     {1, 1, 1, 1,  0, 0, 0, 0,  0},
     {1, 1, 1, 1,  1, 0, 0, 0,  0},
     {1, 1, 1, 1,  1, 1, 0, 0,  0},
     {1, 1, 1, 1,  1, 1, 1, 0,  0},
     {1, 1, 1, 1,  1, 1, 1, 1,  0},
     {1, 1, 1, 1,  1, 1, 1, 1,  1}};

    static std::vector<std::vector<double> > NotLower_9x9 =
    {{0, 1, 1, 1,  1, 1, 1, 1,  1},
     {0, 0, 1, 1,  1, 1, 1, 1,  1},
     {0, 0, 0, 1,  1, 1, 1, 1,  1},
     {0, 0, 0, 0,  1, 1, 1, 1,  1},
     {0, 0, 0, 0,  0, 1, 1, 1,  1},
     {0, 0, 0, 0,  0, 0, 1, 1,  1},
     {0, 0, 0, 0,  0, 0, 0, 1,  1},
     {0, 0, 0, 0,  0, 0, 0, 0,  1},
     {0, 0, 0, 0,  0, 0, 0, 0,  0}};

    static std::vector<std::vector<double> > Lower_9x12 =
    {{1, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {1, 1, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {1, 1, 1, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {1, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 0},
     {1, 1, 1, 1,  1, 0, 0, 0,  0, 0, 0, 0},
     {1, 1, 1, 1,  1, 1, 0, 0,  0, 0, 0, 0},
     {1, 1, 1, 1,  1, 1, 1, 0,  0, 0, 0, 0},
     {1, 1, 1, 1,  1, 1, 1, 1,  0, 0, 0, 0},
     {1, 1, 1, 1,  1, 1, 1, 1,  1, 0, 0, 0}};

    static std::vector<std::vector<double> > NotLower_9x12 =
    {{0, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1},
     {0, 0, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1},
     {0, 0, 0, 1,  1, 1, 1, 1,  1, 1, 1, 1},
     {0, 0, 0, 0,  1, 1, 1, 1,  1, 1, 1, 1},
     {0, 0, 0, 0,  0, 1, 1, 1,  1, 1, 1, 1},
     {0, 0, 0, 0,  0, 0, 1, 1,  1, 1, 1, 1},
     {0, 0, 0, 0,  0, 0, 0, 1,  1, 1, 1, 1},
     {0, 0, 0, 0,  0, 0, 0, 0,  1, 1, 1, 1},
     {0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 1, 1}};

    static std::vector<std::vector<double> > M_9x9 =
        {{1, 0, 0, 1, 0, 0, 1, 0, 0},
         {1, 1, 0, 1, 1, 0, 1, 1, 0},
         {1, 1, 1, 1, 1, 1, 1, 1, 1},
         {1, 0, 0, 1, 0, 0, 1, 0, 0},
         {1, 1, 0, 1, 1, 0, 1, 1, 0},
         {1, 1, 1, 1, 1, 1, 1, 1, 1},
         {1, 0, 0, 1, 0, 0, 1, 0, 0},
         {1, 1, 0, 1, 1, 0, 1, 1, 0},
         {1, 1, 1, 1, 1, 1, 1, 1, 1}};

    static std::vector<std::vector<double> > M_9x12 =
        {{1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0},
         {1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0},
         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
         {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0},
         {1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0},
         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
         {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0},
         {1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0},
         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

    static std::vector<std::vector<double> > M_10x12 =
        {{1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0},
         {1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0},
         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
         {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0},
         {1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0},
         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
         {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0},
         {1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0},
         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
}

//****************************************************************************
// API error tests
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_ABT_bad_dimensions)
{
    Matrix<double> A(A_sparse_3x3, 0.); // 3x3
    Matrix<double> B(B_sparse_4x3, 0.); // 4x3
    Matrix<double> ones9x12(Ones_9x12, 0.);

    Matrix<double> M9x9(M_9x9, 0.);
    Matrix<double> M9x12(M_9x12, 0.);
    Matrix<double> M10x12(M_10x12, 0.);

    Matrix<double> result9x12(9, 12);
    Matrix<double> result9x9(9, 9);
    Matrix<double> result12x12(12, 12);

    // NoMask_NoAccum_AB

    // dim(C) != dim(A)*dim(B)
    BOOST_CHECK_THROW(
        (kronecker(result9x9, NoMask(), NoAccumulate(),
                   Times<double>(), A, transpose(B))),
        DimensionException);

    BOOST_CHECK_THROW(
        (kronecker(result12x12, NoMask(), NoAccumulate(),
                   Times<double>(), A, transpose(B))),
        DimensionException);

    // NoMask_Accum_AB

    // dim(C) != dim(A)*dim(B)
    BOOST_CHECK_THROW(
        (kronecker(result9x9, NoMask(), Plus<double>(),
                   Times<double>(), A, transpose(B))),
        DimensionException);

    BOOST_CHECK_THROW(
        (kronecker(result12x12, NoMask(), Plus<double>(),
                   Times<double>(), A, transpose(B))),
        DimensionException);

    // Mask_NoAccum

    // incompatible mask matrix dimensions
    // nrows(C) != nrows(M)
    BOOST_CHECK_THROW(
        (kronecker(result9x12, M10x12, NoAccumulate(),
                   Times<double>(), A, transpose(B))),
        DimensionException);

    // ncols(C) != ncols(M)
    BOOST_CHECK_THROW(
        (kronecker(result9x12, M9x9, NoAccumulate(),
                   Times<double>(), A, transpose(B))),
        DimensionException);

    BOOST_CHECK_THROW(
        (kronecker(ones9x12, M9x12, NoAccumulate(),
                   Times<double>(), A, transpose(A),
                   REPLACE)),
        DimensionException);

    // Mask_Accum (replace and merge)

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (kronecker(ones9x12, M10x12, Second<double>(),
                   Times<double>(), A, transpose(B), REPLACE)),
        DimensionException);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (kronecker(ones9x12, M9x9, Second<double>(),
                   Times<double>(), A, transpose(B), MERGE)),
        DimensionException);

    // CompMask_NoAccum (replace and merge)

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (kronecker(ones9x12, complement(M10x12), NoAccumulate(),
                   Times<double>(), A, transpose(B), REPLACE)),
        DimensionException);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (kronecker(ones9x12, complement(M9x9), NoAccumulate(),
                   Times<double>(), A, transpose(B), MERGE)),
        DimensionException);

    // CompMask_Accum (replace and merge)

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (kronecker(ones9x12, complement(M10x12), Plus<double>(),
                   Times<double>(), A, transpose(B), REPLACE)),
        DimensionException);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (kronecker(ones9x12, complement(M9x9), Plus<double>(),
                   Times<double>(), A, transpose(B), MERGE)),
        DimensionException);
}

//****************************************************************************
// NoMask_NoAccum
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_NoAccum_ABT)
{
    Matrix<double> C(9, 12);
    Matrix<double> A(A_sparse_3x3, 0.);
    Matrix<double> B(B_sparse_4x3, 0.);

    Matrix<double> answer(Answer_sparse_9x12, 0.);

    kronecker(C, NoMask(), NoAccumulate(),
              Times<double>(), A, transpose(B));

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_NoAccum_ABT_empty)
{
    Matrix<double> Zero(3, 3);
    Matrix<double> Ones(Ones_3x3, 0.);
    Matrix<double> C(Ones_9x9, 0.);
    Matrix<double> mD(Ones_9x9, 0.);
    Matrix<double> Zero9x9(9, 9);

    kronecker(C, NoMask(), NoAccumulate(), Times<double>(), Zero, transpose(Ones));
    BOOST_CHECK_EQUAL(C, Zero9x9);

    kronecker(mD, NoMask(), NoAccumulate(), Times<double>(), Ones, transpose(Zero));
    BOOST_CHECK_EQUAL(mD, Zero9x9);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_NoAccum_ABT_dense)
{
    Matrix<double> Ones3x3(Ones_3x3, 0.);
    Matrix<double> Ones4x3(Ones_4x3, 0.);
    Matrix<double> Ones9x12(Ones_9x12, 0.);
    Matrix<double> result(9, 12);

    kronecker(result, NoMask(), NoAccumulate(),
              Times<double>(), Ones3x3, transpose(Ones4x3));

    BOOST_CHECK_EQUAL(result, Ones9x12);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_NoAccum_ABT_empty_rows)
{
    Matrix<double> C(9, 12);
    Matrix<double> A(Ar_sparse_3x3, 0.);
    Matrix<double> B(Br_sparse_4x3, 0.);

    Matrix<double> answer(Answer_rr_sparse_9x12, 0.);

    kronecker(C, NoMask(), NoAccumulate(),
              Times<double>(), A, transpose(B));

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_NoAccum_ABT_emptyRowA_emptyColB)
{
    Matrix<double> result(9, 12);
    Matrix<double> A(Ar_sparse_3x3, 0.);
    Matrix<double> B(Bc_sparse_4x3, 0.);
    Matrix<double> answer(Answer_rc_sparse_9x12, 0.);

    kronecker(result, NoMask(), NoAccumulate(),
              Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_NoAccum_ABT_emptyColA_emptyRowB)
{
    Matrix<double> result(9, 12);
    Matrix<double> A(Ac_sparse_3x3, 0.);
    Matrix<double> B(Br_sparse_4x3, 0.);
    Matrix<double> answer(Answer_cr_sparse_9x12, 0.);

    kronecker(result, NoMask(), NoAccumulate(),
              Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_NoAccum_ABT_ABdup)
{
    // Build some matrices.
    Matrix<double, DirectedMatrixTag> mat(A_sparse_3x3, 0.);
    Matrix<double, DirectedMatrixTag> m3(9, 9);
    Matrix<double, DirectedMatrixTag> answer(AAT_sparse_9x9, 0.);

    kronecker(m3, NoMask(), NoAccumulate(),
              Times<double>(), mat, transpose(mat));

    BOOST_CHECK_EQUAL(m3, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_NoAccum_ABT_ACdup)
{
    Matrix<double> C(A_sparse_3x3, 0.);
    Matrix<double> B(1, 1);
    B.setElement(0, 0, 1.0);

    Matrix<double> answer(A_sparse_3x3, 0.);

    kronecker(C, NoMask(), NoAccumulate(),
              Times<double>(), C, transpose(B));

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_NoAccum_ABT_BCdup)
{
    Matrix<double> A(1, 1);
    Matrix<double> C(A_sparse_3x3, 0.);
    A.setElement(0, 0, 1.0);

    Matrix<double> answer(AT_sparse_3x3, 0.);

    kronecker(C, NoMask(), NoAccumulate(),
              Times<double>(), A, transpose(C));

    BOOST_CHECK_EQUAL(C, answer);
}


//****************************************************************************
// NoMask_Accum
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_Accum_ABT)
{
    Matrix<double> C(Ones_9x12, 0.);
    Matrix<double> A(A_sparse_3x3, 0.); // 3x3
    Matrix<double> B(B_sparse_4x3, 0.); // 3x4

    Matrix<double> answer(Answer_sparse_9x12, 0.);
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), answer, C);

    kronecker(C, NoMask(), Plus<double>(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_Accum_ABT_empty)
{
    Matrix<double> Zero(3, 3);
    Matrix<double> Ones(Ones_3x3, 0.);
    Matrix<double> C(Ones_9x9, 0.);
    Matrix<double> mD(Ones_9x9, 0.);
    Matrix<double> answer(Ones_9x9, 0.);

    kronecker(C, NoMask(), Plus<double>(), Times<double>(), Zero, transpose(Ones));
    BOOST_CHECK_EQUAL(C, answer);

    kronecker(mD, NoMask(), Plus<double>(), Times<double>(), Ones, transpose(Zero));
    BOOST_CHECK_EQUAL(mD, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_Accum_ABT_dense)
{
    Matrix<double> Ones3x3(Ones_3x3, 0.);
    Matrix<double> Ones4x3(Ones_4x3, 0.);
    Matrix<double> Ones9x12(Ones_9x12, 0.);
    Matrix<double> result(Ones_9x12, 0.);
    Matrix<double> answer(9, 12);
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(),
             Ones9x12, Ones9x12);

    kronecker(result, NoMask(), Plus<double>(),
              Times<double>(), Ones3x3, transpose(Ones4x3));

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_Accum_ABT_empty_rows)
{
    Matrix<double> C(Ones_9x12, 0.);
    Matrix<double> A(Ar_sparse_3x3, 0.);
    Matrix<double> B(Br_sparse_4x3, 0.);

    Matrix<double> answer(Answer_rr_sparse_9x12, 0.);
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), C, answer);

    kronecker(C, NoMask(), Plus<double>(), Times<double>(), A, transpose(B));

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_Accum_ABT_emptyRowA_emptyColB)
{
    Matrix<double> result(Ones_9x12, 0.);
    Matrix<double> A(Ar_sparse_3x3, 0.);
    Matrix<double> B(Bc_sparse_4x3, 0.);
    Matrix<double> answer(Answer_rc_sparse_9x12, 0.);
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), answer, result);

    kronecker(result, NoMask(), Plus<double>(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_Accum_ABT_emptyColA_emptyRowB)
{
    Matrix<double> result(Ones_9x12, 0.);
    Matrix<double> A(Ac_sparse_3x3, 0.);
    Matrix<double> B(Br_sparse_4x3, 0.);
    Matrix<double> answer(Answer_cr_sparse_9x12, 0.);
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), answer, result);

    kronecker(result, NoMask(), Plus<double>(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_Accum_ABT_ABdup)
{
    // Build some matrices.
    Matrix<double, DirectedMatrixTag> mat(A_sparse_3x3, 0.);
    Matrix<double, DirectedMatrixTag> m3(Ones_9x9, 0.);
    Matrix<double, DirectedMatrixTag> answer(AAT_sparse_9x9, 0.);
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), answer, m3);

    kronecker(m3, NoMask(), Plus<double>(), Times<double>(), mat, transpose(mat));

    BOOST_CHECK_EQUAL(m3, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_Accum_ABT_ACdup)
{
    Matrix<double> C(A_sparse_3x3, 0.);
    Matrix<double> B(1, 1);
    B.setElement(0, 0, 1.0);
    Matrix<double> answer(A_sparse_3x3, 0.);
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), answer, answer);

    kronecker(C, NoMask(), Plus<double>(), Times<double>(), C, transpose(B));

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_Accum_ABT_BCdup)
{
    Matrix<double> A(1, 1);
    Matrix<double> C(A_sparse_3x3, 0.);
    A.setElement(0, 0, 1.0);
    Matrix<double> answer(A_sparse_3x3, 0.);
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(),
             answer, transpose(answer));

    kronecker(C, NoMask(), Plus<double>(), Times<double>(), A, transpose(C));

    BOOST_CHECK_EQUAL(C, answer);
}

// ****************************************************************************
// Mask_NoAccum
// ****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_NoAccum_ABT)
{
    Matrix<double> C(9, 12);
    Matrix<double> A(A_sparse_3x3, 0.); // 1 0 0 / 0 2 0 / 3 0 4
    Matrix<double> B(B_sparse_4x3, 0.); // 1 1 0 0 / 0 2 2 0 / 3 0 0 3

    Matrix<double> Identity(Identity_3x3, 0.0);

    Matrix<double> Empty(9, 12);
    Matrix<double> Ones(Ones_9x12, 0.);

    Matrix<double> MLower(Lower_9x12, 0.);
    Matrix<double> MNotLower(NotLower_9x12, 0.);

    Matrix<double> Answer_9x12(Answer_sparse_9x12, 0.);
    Matrix<double> Answer_9x12_Lower(Answer_sparse_9x12_Lower, 0.);
    Matrix<double> Answer_9x12_NotLower(Answer_sparse_9x12_NotLower, 0.);
    Matrix<double> Answer_9x12_Lower_Ones(Answer_sparse_9x12_Lower_Ones, 0.);
    Matrix<double> Answer_9x12_NotLower_Ones(Answer_sparse_9x12_NotLower_Ones, 0.);

    // Merge
    // Mempty vs Mfull vs Mlower

    C = Ones;
    kronecker(C, Empty, NoAccumulate(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, Ones);

    C = Ones;
    kronecker(C, Ones,  NoAccumulate(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, Answer_9x12);

    C = Ones;
    kronecker(C, MLower, NoAccumulate(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, Answer_9x12_Lower_Ones);

    C = Ones;
    kronecker(C, MNotLower, NoAccumulate(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, Answer_9x12_NotLower_Ones);

    // Replace
    // Mempty vs Mfull vs Mlower

    C = Ones;
    kronecker(C, Empty, NoAccumulate(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, Empty);

    C = Ones;
    kronecker(C, Ones,  NoAccumulate(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, Answer_9x12);

    C = Ones;
    kronecker(C, MLower, NoAccumulate(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, Answer_9x12_Lower);

    C = Ones;
    kronecker(C, MNotLower, NoAccumulate(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, Answer_9x12_NotLower);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_NoAccum_ABTM_empty)
{
    Matrix<double> C(9, 12);
    Matrix<double> A(A_sparse_3x3, 0.); // 1 0 0 / 0 2 0 / 3 0 4
    Matrix<double> B(B_sparse_4x3, 0.); // 1 1 0 0 / 0 2 2 0 / 3 0 0 3

    Matrix<double> Ones(Ones_9x12, 0.);
    Matrix<double> Empty(9, 12);
    Matrix<double> Empty3x3(3, 3);
    Matrix<double> Empty4x3(4, 3);

    Matrix<double> MLower(Lower_9x12, 0.);
    Matrix<double> MNotLower(NotLower_9x12, 0.);

    // Merge
    C = Ones;
    kronecker(C, Empty, NoAccumulate(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, Ones);

    C = Ones;
    kronecker(C, MLower, NoAccumulate(), Times<double>(), Empty3x3, transpose(B));
    BOOST_CHECK_EQUAL(C, MNotLower);

    C = Ones;
    kronecker(C, MLower, NoAccumulate(), Times<double>(), A, transpose(Empty4x3));
    BOOST_CHECK_EQUAL(C, MNotLower);

    // Replace
    C = Ones;
    kronecker(C, Empty, NoAccumulate(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, Empty);

    C = Ones;
    kronecker(C, MLower, NoAccumulate(), Times<double>(), Empty3x3, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, Empty);

    C = Ones;
    kronecker(C, MLower, NoAccumulate(), Times<double>(), A, transpose(Empty4x3), REPLACE);
    BOOST_CHECK_EQUAL(C, Empty);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_NoAccum_ABTM_dense)
{
    Matrix<double> A(Ones_3x3, 0.);
    Matrix<double> B(Ones_4x3, 0.);

    Matrix<double> MLower(Lower_9x12, 0.);
    Matrix<double> MNotLower(NotLower_9x12, 0.);

    Matrix<double> Ones(Ones_9x12, 0.);
    Matrix<double> C(9, 12);

    C = Ones;
    kronecker(C, MLower, NoAccumulate(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, Ones);

    C = Ones;
    kronecker(C, MLower, NoAccumulate(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, MLower);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_NoAccum_ABT_empty_rows)
{
    Matrix<double> Ones(Ones_9x12, 0.);
    Matrix<double> A(Ar_sparse_3x3, 0.);
    Matrix<double> B(Br_sparse_4x3, 0.);
    Matrix<double> answer(Answer_rr_sparse_9x12, 0.);
    Matrix<double> C(9, 12);

    Matrix<double> MLower(Lower_9x12, 0.);
    Matrix<double> MNotLower(NotLower_9x12, 0.);

    C = Ones;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MLower, answer);
    kronecker(C, MLower, NoAccumulate(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, answer);

    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MNotLower, answer);
    kronecker(C, MLower, NoAccumulate(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, answer);

}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_NoAccum_ABT_emptyRowA_emptyColB)
{
    Matrix<double> C(9,12);
    Matrix<double> Ones(Ones_9x12, 0.);
    Matrix<double> A(Ar_sparse_3x3, 0.);
    Matrix<double> B(Bc_sparse_4x3, 0.);
    Matrix<double> answer(Answer_rc_sparse_9x12, 0.);

    Matrix<double> MLower(Lower_9x12, 0.);
    Matrix<double> MNotLower(NotLower_9x12, 0.);


    // REPLACE
    C = Ones;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MLower, answer);
    kronecker(C, MLower, NoAccumulate(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MNotLower, answer);
    kronecker(C, MLower, NoAccumulate(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_NoAccum_ABT_emptyColA_emptyRowB)
{
    Matrix<double> C(9,12);
    Matrix<double> Ones(Ones_9x12, 0.);
    Matrix<double> A(Ac_sparse_3x3, 0.);
    Matrix<double> B(Br_sparse_4x3, 0.);
    Matrix<double> answer(Answer_cr_sparse_9x12, 0.);

    Matrix<double> MLower(Lower_9x12, 0.);
    Matrix<double> MNotLower(NotLower_9x12, 0.);

    // REPLACE
    C = Ones;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MLower, answer);
    kronecker(C, MLower, NoAccumulate(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MNotLower, answer);
    kronecker(C, MLower, NoAccumulate(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_NoAccum_ABT_emptyRowM)
{
    Matrix<double> C(9,12);
    Matrix<double> Ones(Ones_9x12, 0.);
    Matrix<double> A(A_sparse_3x3, 0.);
    Matrix<double> B(B_sparse_4x3, 0.);
    Matrix<double> answer(Answer_sparse_9x12, 0.);

    Matrix<double> MLower(Lower_9x12, 0.);
    Matrix<double> MNotLower(NotLower_9x12, 0.);


    // Replace
    C = Ones;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MLower, answer);
    kronecker(C, MLower, NoAccumulate(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MNotLower, answer);
    kronecker(C, MLower, NoAccumulate(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_NoAccum_ABT_ABdup)
{
    Matrix<double> C(9,9);
    Matrix<double> Ones(Ones_9x9, 0.);
    Matrix<double> A(A_sparse_3x3, 0.);
    Matrix<double> answer(AAT_sparse_9x9, 0.);

    Matrix<double> MLower(Lower_9x9, 0.);
    Matrix<double> MNotLower(NotLower_9x9, 0.);

    // Replace
    C = Ones;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MLower, answer);
    kronecker(C, MLower, NoAccumulate(), Times<double>(), A, transpose(A), REPLACE);

    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MNotLower, answer);
    kronecker(C, MLower, NoAccumulate(), Times<double>(), A, transpose(A));
    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_NoAccum_ABT_ACdup)
{
    Matrix<double> Ones(Ones_3x4, 0.);
    Matrix<double> mat(B_sparse_3x4, 0.);
    Matrix<double> C(3, 4);
    Matrix<double> B(1, 1);    B.setElement(0, 0, 1.0);
    Matrix<double> answer(mat);

    Matrix<double> MLower(Lower_3x4, 0.);
    Matrix<double> MNotLower(NotLower_3x4, 0.);

    // Build some matrices.

    // Replace
    C = mat;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MLower, answer);
    kronecker(C, MLower, NoAccumulate(), Times<double>(), C, transpose(B), REPLACE);

    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    C = mat;
    kronecker(C, MLower, NoAccumulate(), Times<double>(), C, transpose(B));

    BOOST_CHECK_EQUAL(C, mat);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_NoAccum_ABT_BCdup)
{
    // Build some matrices.
    Matrix<double> mat(A_sparse_3x3, 0.);
    Matrix<double> C(3,3);
    Matrix<double> A(1, 1);    A.setElement(0, 0, 1.0);
    Matrix<double> answer(mat);

    Matrix<double> MLower(Lower_3x3, 0.);
    Matrix<double> MNotLower(NotLower_3x3, 0.);

    // Replace
    C = mat;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(),
              MLower, transpose(answer));
    kronecker(C, MLower, NoAccumulate(), Times<double>(), A, transpose(C), REPLACE);

    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    C = mat;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(),
              transpose(MLower), transpose(answer));
    kronecker(C, MLower, NoAccumulate(), Times<double>(), A, transpose(C));

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_NoAccum_ABT_MCdup)
{
    // Build some matrices.
    Matrix<double> C(9,12);
    Matrix<double> Ones(Ones_9x12, 0.);
    Matrix<double> A(A_sparse_3x3, 0.);
    Matrix<double> B(B_sparse_4x3, 0.);
    Matrix<double> answer(Answer_sparse_9x12, 0.);

    Matrix<double> MLower(Lower_9x12, 0.);
    Matrix<double> MNotLower(NotLower_9x12, 0.);

    // Replace
    C = MLower;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MLower, answer);
    kronecker(C, C, NoAccumulate(), Times<double>(), A, transpose(B), REPLACE);

    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    C = MLower;
    kronecker(C, C, NoAccumulate(), Times<double>(), A, transpose(B));

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
// Mask_Accum
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_Accum_ABT)
{
    Matrix<double> C(9, 12);
    Matrix<double> A(A_sparse_3x3, 0.); // 1 0 0 / 0 2 0 / 3 0 4
    Matrix<double> B(B_sparse_4x3, 0.); // 1 1 0 0 / 0 2 2 0 / 3 0 0 3

    Matrix<double> Identity(Identity_3x3, 0.0);

    Matrix<double> answer(9, 12);
    Matrix<double> Empty(9, 12);
    Matrix<double> Ones(Ones_9x12, 0.);

    Matrix<double> MLower(Lower_9x12, 0.);
    Matrix<double> MNotLower(NotLower_9x12, 0.);

    Matrix<double> Answer_9x12(Answer_sparse_9x12, 0.);
    Matrix<double> Answer_9x12_Lower(Answer_sparse_9x12_Lower, 0.);
    Matrix<double> Answer_9x12_NotLower(Answer_sparse_9x12_NotLower, 0.);
    Matrix<double> Answer_9x12_Lower_Ones(Answer_sparse_9x12_Lower_Ones, 0.);
    Matrix<double> Answer_9x12_NotLower_Ones(Answer_sparse_9x12_NotLower_Ones, 0.);

    // Merge
    // Mempty vs Mfull vs Mlower

    C = Ones;
    kronecker(C, Empty, Plus<double>(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, Ones);

    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(),
             Ones, Answer_9x12);
    kronecker(C, Ones,  Plus<double>(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, answer); //Answer_9x12);

    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(),
             Ones, Answer_9x12_Lower);
    kronecker(C, MLower, Plus<double>(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, answer);

    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(),
             Ones, Answer_9x12_NotLower);
    kronecker(C, MNotLower, Plus<double>(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    // Mempty vs Mfull vs Mlower

    C = Ones;
    kronecker(C, Empty, Plus<double>(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, Empty);

    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(),
             Ones, Answer_9x12);
    kronecker(C, Ones,  Plus<double>(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, answer);

    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(),
             MLower, Answer_9x12_Lower);
    kronecker(C, MLower, Plus<double>(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, answer);

    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(),
             MNotLower, Answer_9x12_NotLower);
    kronecker(C, MNotLower, Plus<double>(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_Accum_ABTM_empty)
{
    Matrix<double> C(9, 12);
    Matrix<double> A(A_sparse_3x3, 0.); // 1 0 0 / 0 2 0 / 3 0 4
    Matrix<double> B(B_sparse_4x3, 0.); // 1 1 0 0 / 0 2 2 0 / 3 0 0 3

    Matrix<double> Ones(Ones_9x12, 0.);
    Matrix<double> Empty(9, 12);
    Matrix<double> Empty3x3(3, 3);
    Matrix<double> Empty4x3(4, 3);

    Matrix<double> MLower(Lower_9x12, 0.);

    // Merge
    C = Ones;
    kronecker(C, Empty, Plus<double>(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, Ones);

    C = Ones;
    kronecker(C, MLower, Plus<double>(), Times<double>(), Empty3x3, transpose(B));
    BOOST_CHECK_EQUAL(C, Ones);

    C = Ones;
    kronecker(C, MLower, Plus<double>(), Times<double>(), A, transpose(Empty4x3));
    BOOST_CHECK_EQUAL(C, Ones);

    // Replace
    C = Ones;
    kronecker(C, Empty, Plus<double>(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, Empty);

    C = Ones;
    kronecker(C, MLower, Plus<double>(), Times<double>(), Empty3x3, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, MLower);

    C = Ones;
    kronecker(C, MLower, Plus<double>(), Times<double>(), A, transpose(Empty4x3), REPLACE);
    BOOST_CHECK_EQUAL(C, MLower);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_Accum_ABTM_dense)
{
    Matrix<double> A(Ones_3x3, 0.);
    Matrix<double> B(Ones_4x3, 0.);

    Matrix<double> MLower(Lower_9x12, 0.);
    Matrix<double> MNotLower(NotLower_9x12, 0.);

    Matrix<double> Ones(Ones_9x12, 0.);
    Matrix<double> C(9, 12);
    Matrix<double> answer(9, 12);

    C = Ones;
    kronecker(C, MLower, Plus<double>(), Times<double>(), A, transpose(B));
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(),
             Ones, MLower);
    BOOST_CHECK_EQUAL(C, answer);

    C = Ones;
    kronecker(C, MLower, Plus<double>(), Times<double>(), A, transpose(B), REPLACE);
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(),
             MLower, MLower);
    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_Accum_ABT_empty_rows)
{
    Matrix<double> Ones(Ones_9x12, 0.);
    Matrix<double> A(Ar_sparse_3x3, 0.);
    Matrix<double> B(Br_sparse_4x3, 0.);
    Matrix<double> answer(Answer_rr_sparse_9x12, 0.);
    Matrix<double> C(9, 12);

    Matrix<double> MLower(Lower_9x12, 0.);
    Matrix<double> MNotLower(NotLower_9x12, 0.);

    C = Ones;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MLower, answer);
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MLower, answer);
    kronecker(C, MLower, Plus<double>(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, answer);

    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MNotLower, answer);
    kronecker(C, MLower, Plus<double>(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_Accum_ABT_emptyRowA_emptyColB)
{
    Matrix<double> C(9,12);
    Matrix<double> Ones(Ones_9x12, 0.);
    Matrix<double> A(Ar_sparse_3x3, 0.);
    Matrix<double> B(Bc_sparse_4x3, 0.);
    Matrix<double> answer(Answer_rc_sparse_9x12, 0.);

    Matrix<double> MLower(Lower_9x12, 0.);
    Matrix<double> MNotLower(NotLower_9x12, 0.);

    // REPLACE
    C = Ones;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MLower, answer);
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MLower, answer);
    kronecker(C, MLower, Plus<double>(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MNotLower, answer);
    kronecker(C, MLower, Plus<double>(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_Accum_ABT_emptyColA_emptyRowB)
{
    Matrix<double> C(9,12);
    Matrix<double> Ones(Ones_9x12, 0.);
    Matrix<double> A(Ac_sparse_3x3, 0.);
    Matrix<double> B(Br_sparse_4x3, 0.);
    Matrix<double> answer(Answer_cr_sparse_9x12, 0.);

    Matrix<double> MLower(Lower_9x12, 0.);
    Matrix<double> MNotLower(NotLower_9x12, 0.);

    // REPLACE
    C = Ones;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MLower, answer);
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MLower, answer);
    kronecker(C, MLower, Plus<double>(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MNotLower, answer);
    kronecker(C, MLower, Plus<double>(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_Accum_ABT_emptyRowM)
{
    Matrix<double> C(9,12);
    Matrix<double> Ones(Ones_9x12, 0.);
    Matrix<double> A(A_sparse_3x3, 0.);
    Matrix<double> B(B_sparse_4x3, 0.);
    Matrix<double> answer(Answer_sparse_9x12, 0.);

    Matrix<double> MLower(Lower_9x12, 0.);
    Matrix<double> MNotLower(NotLower_9x12, 0.);

    // Replace
    C = Ones;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MLower, answer);
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MLower, answer);
    kronecker(C, MLower, Plus<double>(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MNotLower, answer);
    kronecker(C, MLower, Plus<double>(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_Accum_ABT_ABdup)
{
    Matrix<double> C(9,9);
    Matrix<double> Ones(Ones_9x9, 0.);
    Matrix<double> A(A_sparse_3x3, 0.);
    Matrix<double> answer(AAT_sparse_9x9, 0.);

    Matrix<double> MLower(Lower_9x9, 0.);
    Matrix<double> MNotLower(NotLower_9x9, 0.);

    // Replace
    C = Ones;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MLower, answer);
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MLower, answer);
    kronecker(C, MLower, Plus<double>(), Times<double>(), A, transpose(A), REPLACE);

    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MNotLower, answer);
    kronecker(C, MLower, Plus<double>(), Times<double>(), A, transpose(A));
    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_Accum_ABT_ACdup)
{
    // Build some matrices.
    Matrix<double> Ones(Ones_3x4, 0.);
    Matrix<double> mat(B_sparse_3x4, 0.);
    Matrix<double> C(3, 4);
    Matrix<double> B(1, 1);    B.setElement(0, 0, 1.0);
    Matrix<double> answer(mat);

    Matrix<double> MLower(Lower_3x4, 0.);
    Matrix<double> MNotLower(NotLower_3x4, 0.);

    // Replace
    C = mat;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MLower, answer);
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), answer, answer);
    kronecker(C, MLower, Plus<double>(), Times<double>(), C, transpose(B), REPLACE);

    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    C = mat;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MLower, mat);
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), mat, answer);
    kronecker(C, MLower, Plus<double>(), Times<double>(), C, transpose(B));
    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_Accum_ABT_BCdup)
{
    // Build some matrices.
    Matrix<double> mat(A_sparse_3x3, 0.);
    Matrix<double> C(3,3);
    Matrix<double> A(1, 1);    A.setElement(0, 0, 1.0);
    Matrix<double> answer(mat);

    Matrix<double> MLower(Lower_3x3, 0.);
    Matrix<double> MNotLower(NotLower_3x3, 0.);

    // Replace
    C = mat;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MLower,transpose(mat));
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), answer, mat);
    kronecker(C, MLower, Plus<double>(), Times<double>(), A, transpose(C), REPLACE);

    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    C = mat;
    kronecker(C, MLower, Plus<double>(), Times<double>(), A, transpose(C));

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_Accum_ABT_MCdup)
{
    // Build some matrices.
    Matrix<double> C(9,12);
    Matrix<double> Ones(Ones_9x12, 0.);
    Matrix<double> A(A_sparse_3x3, 0.);
    Matrix<double> B(B_sparse_4x3, 0.);
    Matrix<double> answer(Answer_sparse_9x12, 0.);

    Matrix<double> MLower(Lower_9x12, 0.);
    Matrix<double> MNotLower(NotLower_9x12, 0.);

    // Replace
    C = MLower;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MLower, answer);
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MLower, answer);
    kronecker(C, C, Plus<double>(), Times<double>(), A, transpose(B), REPLACE);

    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    C = MLower;
    kronecker(C, C, Plus<double>(), Times<double>(), A, transpose(B));

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
// CompMask_NoAccum
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_NoAccum_ABT)
{
    Matrix<double> C(9, 12);
    Matrix<double> A(A_sparse_3x3, 0.); // 1 0 0 / 0 2 0 / 3 0 4
    Matrix<double> B(B_sparse_4x3, 0.); // 1 1 0 0 / 0 2 2 0 / 3 0 0 3

    Matrix<double> Identity(Identity_3x3, 0.0);

    Matrix<double> Empty(9, 12);
    Matrix<double> Ones(Ones_9x12, 0.);

    Matrix<double> MLower(Lower_9x12, 0.);
    Matrix<double> MNotLower(NotLower_9x12, 0.);

    Matrix<double> Answer_9x12(Answer_sparse_9x12, 0.);
    Matrix<double> Answer_9x12_Lower(Answer_sparse_9x12_Lower, 0.);
    Matrix<double> Answer_9x12_NotLower(Answer_sparse_9x12_NotLower, 0.);
    Matrix<double> Answer_9x12_Lower_Ones(Answer_sparse_9x12_Lower_Ones, 0.);
    Matrix<double> Answer_9x12_NotLower_Ones(Answer_sparse_9x12_NotLower_Ones, 0.);

    // Merge
    // Mempty vs Mfull vs Mlower

    C = Ones;
    kronecker(C, complement(Ones), NoAccumulate(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, Ones);

    C = Ones;
    kronecker(C, complement(Empty),  NoAccumulate(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, Answer_9x12);

    C = Ones;
    kronecker(C, complement(MNotLower), NoAccumulate(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, Answer_9x12_Lower_Ones);

    C = Ones;
    kronecker(C, complement(MLower), NoAccumulate(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, Answer_9x12_NotLower_Ones);

    // Replace
    // Mempty vs Mfull vs Mlower

    C = Ones;
    kronecker(C, complement(Ones), NoAccumulate(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, Empty);

    C = Ones;
    kronecker(C, complement(Empty),  NoAccumulate(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, Answer_9x12);

    C = Ones;
    kronecker(C, complement(MNotLower), NoAccumulate(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, Answer_9x12_Lower);

    C = Ones;
    kronecker(C, complement(MLower), NoAccumulate(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, Answer_9x12_NotLower);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_NoAccum_ABTM_empty)
{
    Matrix<double> C(9, 12);
    Matrix<double> A(A_sparse_3x3, 0.); // 1 0 0 / 0 2 0 / 3 0 4
    Matrix<double> B(B_sparse_4x3, 0.); // 1 1 0 0 / 0 2 2 0 / 3 0 0 3

    Matrix<double> Ones(Ones_9x12, 0.);
    Matrix<double> Empty(9, 12);
    Matrix<double> Empty3x3(3, 3);
    Matrix<double> Empty4x3(4, 3);

    Matrix<double> MLower(Lower_9x12, 0.);
    Matrix<double> MNotLower(NotLower_9x12, 0.);

    // Merge
    C = Ones;
    kronecker(C, complement(Ones), NoAccumulate(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, Ones);

    C = Ones;
    kronecker(C, complement(MNotLower), NoAccumulate(), Times<double>(), Empty3x3, transpose(B));
    BOOST_CHECK_EQUAL(C, MNotLower);

    C = Ones;
    kronecker(C, complement(MNotLower), NoAccumulate(), Times<double>(), A, transpose(Empty4x3));
    BOOST_CHECK_EQUAL(C, MNotLower);

    // Replace
    C = Ones;
    kronecker(C, complement(Ones), NoAccumulate(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, Empty);

    C = Ones;
    kronecker(C, complement(MNotLower), NoAccumulate(), Times<double>(), Empty3x3, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, Empty);

    C = Ones;
    kronecker(C, complement(MNotLower), NoAccumulate(), Times<double>(), A, transpose(Empty4x3), REPLACE);
    BOOST_CHECK_EQUAL(C, Empty);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_NoAccum_ABTM_dense)
{
    Matrix<double> A(Ones_3x3, 0.);
    Matrix<double> B(Ones_4x3, 0.);

    Matrix<double> MLower(Lower_9x12, 0.);
    Matrix<double> MNotLower(NotLower_9x12, 0.);

    Matrix<double> Ones(Ones_9x12, 0.);
    Matrix<double> C(9, 12);

    C = Ones;
    kronecker(C, complement(MNotLower), NoAccumulate(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, Ones);

    C = Ones;
    kronecker(C, complement(MNotLower), NoAccumulate(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, MLower);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_NoAccum_ABT_empty_rows)
{
    Matrix<double> Ones(Ones_9x12, 0.);
    Matrix<double> A(Ar_sparse_3x3, 0.);
    Matrix<double> B(Br_sparse_4x3, 0.);
    Matrix<double> answer(Answer_rr_sparse_9x12, 0.);
    Matrix<double> C(9, 12);

    Matrix<double> MLower(Lower_9x12, 0.);
    Matrix<double> MNotLower(NotLower_9x12, 0.);

    C = Ones;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MLower, answer);
    kronecker(C, complement(MNotLower), NoAccumulate(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, answer);

    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MNotLower, answer);
    kronecker(C, complement(MNotLower), NoAccumulate(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, answer);

}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_NoAccum_ABT_emptyRowA_emptyColB)
{
    Matrix<double> C(9,12);
    Matrix<double> Ones(Ones_9x12, 0.);
    Matrix<double> A(Ar_sparse_3x3, 0.);
    Matrix<double> B(Bc_sparse_4x3, 0.);
    Matrix<double> answer(Answer_rc_sparse_9x12, 0.);

    Matrix<double> MLower(Lower_9x12, 0.);
    Matrix<double> MNotLower(NotLower_9x12, 0.);


    // REPLACE
    C = Ones;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MLower, answer);
    kronecker(C, complement(MNotLower), NoAccumulate(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MNotLower, answer);
    kronecker(C, complement(MNotLower), NoAccumulate(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_NoAccum_ABT_emptyColA_emptyRowB)
{
    Matrix<double> C(9,12);
    Matrix<double> Ones(Ones_9x12, 0.);
    Matrix<double> A(Ac_sparse_3x3, 0.);
    Matrix<double> B(Br_sparse_4x3, 0.);
    Matrix<double> answer(Answer_cr_sparse_9x12, 0.);

    Matrix<double> MLower(Lower_9x12, 0.);
    Matrix<double> MNotLower(NotLower_9x12, 0.);

    // REPLACE
    C = Ones;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MLower, answer);
    kronecker(C, complement(MNotLower), NoAccumulate(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MNotLower, answer);
    kronecker(C, complement(MNotLower), NoAccumulate(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_NoAccum_ABT_emptyRowM)
{
    Matrix<double> C(9,12);
    Matrix<double> Ones(Ones_9x12, 0.);
    Matrix<double> A(A_sparse_3x3, 0.);
    Matrix<double> B(B_sparse_4x3, 0.);
    Matrix<double> answer(Answer_sparse_9x12, 0.);

    Matrix<double> MLower(Lower_9x12, 0.);
    Matrix<double> MNotLower(NotLower_9x12, 0.);


    // Replace
    C = Ones;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MLower, answer);
    kronecker(C, complement(MNotLower), NoAccumulate(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MNotLower, answer);
    kronecker(C, complement(MNotLower), NoAccumulate(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_NoAccum_ABT_ABdup)
{
    Matrix<double> C(9,9);
    Matrix<double> Ones(Ones_9x9, 0.);
    Matrix<double> A(A_sparse_3x3, 0.);
    Matrix<double> answer(AAT_sparse_9x9, 0.);

    Matrix<double> MLower(Lower_9x9, 0.);
    Matrix<double> MNotLower(NotLower_9x9, 0.);

    // Replace
    C = Ones;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MLower, answer);
    kronecker(C, complement(MNotLower), NoAccumulate(), Times<double>(), A, transpose(A), REPLACE);

    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MNotLower, answer);
    kronecker(C, complement(MNotLower), NoAccumulate(), Times<double>(), A, transpose(A));
    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_NoAccum_ABT_ACdup)
{
    Matrix<double> Ones(Ones_3x4, 0.);
    Matrix<double> mat(B_sparse_3x4, 0.);
    Matrix<double> C(3, 4);
    Matrix<double> B(1, 1);    B.setElement(0, 0, 1.0);
    Matrix<double> answer(mat);

    Matrix<double> MLower(Lower_3x4, 0.);
    Matrix<double> MNotLower(NotLower_3x4, 0.);

    // Build some matrices.

    // Replace
    C = mat;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MLower, answer);
    kronecker(C, complement(MNotLower), NoAccumulate(), Times<double>(), C, transpose(B), REPLACE);

    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    C = mat;
    kronecker(C, complement(MNotLower), NoAccumulate(), Times<double>(), C, transpose(B));

    BOOST_CHECK_EQUAL(C, mat);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_NoAccum_ABT_BCdup)
{
    // Build some matrices.
    Matrix<double> mat(A_sparse_3x3, 0.);
    Matrix<double> C(3,3);
    Matrix<double> A(1, 1);    A.setElement(0, 0, 1.0);
    Matrix<double> answer(AT_sparse_3x3, 0.);

    Matrix<double> MLower(Lower_3x3, 0.);
    Matrix<double> MNotLower(NotLower_3x3, 0.);

    // Replace
    C = mat;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MLower, answer);
    kronecker(C, complement(MNotLower), NoAccumulate(), Times<double>(), A, transpose(C), REPLACE);

    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    C = mat;
    kronecker(C, complement(MNotLower), NoAccumulate(), Times<double>(), A, transpose(C));

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_NoAccum_ABT_MCdup)
{
    // Build some matrices.
    Matrix<double> C(9,12);
    Matrix<double> Ones(Ones_9x12, 0.);
    Matrix<double> A(A_sparse_3x3, 0.);
    Matrix<double> B(B_sparse_4x3, 0.);
    Matrix<double> answer(Answer_sparse_9x12, 0.);

    Matrix<double> MLower(Lower_9x12, 0.);
    Matrix<double> MNotLower(NotLower_9x12, 0.);

    // Replace
    C = MLower;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MNotLower, answer);
    kronecker(C, complement(C), NoAccumulate(), Times<double>(), A, transpose(B), REPLACE);

    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    C = MLower;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MLower, answer);
    kronecker(C, complement(C), NoAccumulate(), Times<double>(), A, transpose(B));

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
// CompMask_Accum
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_Accum_ABT)
{
    Matrix<double> C(9, 12);
    Matrix<double> A(A_sparse_3x3, 0.); // 1 0 0 / 0 2 0 / 3 0 4
    Matrix<double> B(B_sparse_4x3, 0.); // 1 1 0 0 / 0 2 2 0 / 3 0 0 3

    Matrix<double> Identity(Identity_3x3, 0.0);

    Matrix<double> answer(9, 12);
    Matrix<double> Empty(9, 12);
    Matrix<double> Ones(Ones_9x12, 0.);

    Matrix<double> MLower(Lower_9x12, 0.);
    Matrix<double> MNotLower(NotLower_9x12, 0.);

    Matrix<double> Answer_9x12(Answer_sparse_9x12, 0.);
    Matrix<double> Answer_9x12_Lower(Answer_sparse_9x12_Lower, 0.);
    Matrix<double> Answer_9x12_NotLower(Answer_sparse_9x12_NotLower, 0.);
    Matrix<double> Answer_9x12_Lower_Ones(Answer_sparse_9x12_Lower_Ones, 0.);
    Matrix<double> Answer_9x12_NotLower_Ones(Answer_sparse_9x12_NotLower_Ones, 0.);

    // Merge
    // Mempty vs Mfull vs Mlower

    C = Ones;
    kronecker(C, complement(Ones), Plus<double>(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, Ones);

    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(),
             Ones, Answer_9x12);
    kronecker(C, complement(Empty),  Plus<double>(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, answer); //Answer_9x12);

    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(),
             Ones, Answer_9x12_Lower);
    kronecker(C, complement(MNotLower), Plus<double>(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, answer);

    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(),
             Ones, Answer_9x12_NotLower);
    kronecker(C, complement(MLower), Plus<double>(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    // Mempty vs Mfull vs Mlower

    C = Ones;
    kronecker(C, complement(Ones), Plus<double>(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, Empty);

    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(),
             Ones, Answer_9x12);
    kronecker(C, complement(Empty),  Plus<double>(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, answer);

    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(),
             MLower, Answer_9x12_Lower);
    kronecker(C, complement(MNotLower), Plus<double>(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, answer);

    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(),
             MNotLower, Answer_9x12_NotLower);
    kronecker(C, complement(MLower), Plus<double>(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_Accum_ABTM_empty)
{
    Matrix<double> C(9, 12);
    Matrix<double> A(A_sparse_3x3, 0.); // 1 0 0 / 0 2 0 / 3 0 4
    Matrix<double> B(B_sparse_4x3, 0.); // 1 1 0 0 / 0 2 2 0 / 3 0 0 3

    Matrix<double> Ones(Ones_9x12, 0.);
    Matrix<double> Empty(9, 12);
    Matrix<double> Empty3x3(3, 3);
    Matrix<double> Empty4x3(4, 3);

    Matrix<double> MLower(Lower_9x12, 0.);
    Matrix<double> MNotLower(NotLower_9x12, 0.);

    // Merge
    C = Ones;
    kronecker(C, complement(Ones), Plus<double>(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, Ones);

    C = Ones;
    kronecker(C, complement(MNotLower), Plus<double>(), Times<double>(), Empty3x3, transpose(B));
    BOOST_CHECK_EQUAL(C, Ones);

    C = Ones;
    kronecker(C, complement(MNotLower), Plus<double>(), Times<double>(), A, transpose(Empty4x3));
    BOOST_CHECK_EQUAL(C, Ones);

    // Replace
    C = Ones;
    kronecker(C, complement(Ones), Plus<double>(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, Empty);

    C = Ones;
    kronecker(C, complement(MNotLower), Plus<double>(), Times<double>(), Empty3x3, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, MLower);

    C = Ones;
    kronecker(C, complement(MNotLower), Plus<double>(), Times<double>(), A, transpose(Empty4x3), REPLACE);
    BOOST_CHECK_EQUAL(C, MLower);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_Accum_ABTM_dense)
{
    Matrix<double> A(Ones_3x3, 0.);
    Matrix<double> B(Ones_4x3, 0.);

    Matrix<double> MLower(Lower_9x12, 0.);
    Matrix<double> MNotLower(NotLower_9x12, 0.);

    Matrix<double> Ones(Ones_9x12, 0.);
    Matrix<double> C(9, 12);
    Matrix<double> answer(9, 12);

    C = Ones;
    kronecker(C, complement(MNotLower), Plus<double>(), Times<double>(), A, transpose(B));
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(),
             Ones, MLower);
    BOOST_CHECK_EQUAL(C, answer);

    C = Ones;
    kronecker(C, complement(MNotLower), Plus<double>(), Times<double>(), A, transpose(B), REPLACE);
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(),
             MLower, MLower);
    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_Accum_ABT_empty_rows)
{
    Matrix<double> Ones(Ones_9x12, 0.);
    Matrix<double> A(Ar_sparse_3x3, 0.);
    Matrix<double> B(Br_sparse_4x3, 0.);
    Matrix<double> answer(Answer_rr_sparse_9x12, 0.);
    Matrix<double> C(9, 12);

    Matrix<double> MLower(Lower_9x12, 0.);
    Matrix<double> MNotLower(NotLower_9x12, 0.);

    C = Ones;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MLower, answer);
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MLower, answer);
    kronecker(C, complement(MNotLower), Plus<double>(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, answer);

    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MNotLower, answer);
    kronecker(C, complement(MNotLower), Plus<double>(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_Accum_ABT_emptyRowA_emptyColB)
{
    Matrix<double> C(9,12);
    Matrix<double> Ones(Ones_9x12, 0.);
    Matrix<double> A(Ar_sparse_3x3, 0.);
    Matrix<double> B(Bc_sparse_4x3, 0.);
    Matrix<double> answer(Answer_rc_sparse_9x12, 0.);

    Matrix<double> MLower(Lower_9x12, 0.);
    Matrix<double> MNotLower(NotLower_9x12, 0.);

    // REPLACE
    C = Ones;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MLower, answer);
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MLower, answer);
    kronecker(C, complement(MNotLower), Plus<double>(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MNotLower, answer);
    kronecker(C, complement(MNotLower), Plus<double>(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_Accum_ABT_emptyColA_emptyRowB)
{
    Matrix<double> C(9,12);
    Matrix<double> Ones(Ones_9x12, 0.);
    Matrix<double> A(Ac_sparse_3x3, 0.);
    Matrix<double> B(Br_sparse_4x3, 0.);
    Matrix<double> answer(Answer_cr_sparse_9x12, 0.);

    Matrix<double> MLower(Lower_9x12, 0.);
    Matrix<double> MNotLower(NotLower_9x12, 0.);

    // REPLACE
    C = Ones;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MLower, answer);
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MLower, answer);
    kronecker(C, complement(MNotLower), Plus<double>(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MNotLower, answer);
    kronecker(C, complement(MNotLower), Plus<double>(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_Accum_ABTM_emptyRowM)
{
    Matrix<double> C(9,12);
    Matrix<double> Ones(Ones_9x12, 0.);
    Matrix<double> A(A_sparse_3x3, 0.);
    Matrix<double> B(B_sparse_4x3, 0.);
    Matrix<double> answer(Answer_sparse_9x12, 0.);

    Matrix<double> MLower(Lower_9x12, 0.);
    Matrix<double> MNotLower(NotLower_9x12, 0.);

    // Replace
    C = Ones;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MLower, answer);
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MLower, answer);
    kronecker(C, complement(MNotLower), Plus<double>(), Times<double>(), A, transpose(B), REPLACE);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MNotLower, answer);
    kronecker(C, complement(MNotLower), Plus<double>(), Times<double>(), A, transpose(B));
    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_Accum_ABT_ABdup)
{
    Matrix<double> C(9,9);
    Matrix<double> Ones(Ones_9x9, 0.);
    Matrix<double> A(A_sparse_3x3, 0.);
    Matrix<double> answer(AAT_sparse_9x9, 0.);

    Matrix<double> MLower(Lower_9x9, 0.);
    Matrix<double> MNotLower(NotLower_9x9, 0.);

    // Replace
    C = Ones;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MLower, answer);
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MLower, answer);
    kronecker(C, complement(MNotLower), Plus<double>(), Times<double>(), A, transpose(A), REPLACE);

    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    C = Ones;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MNotLower, answer);
    kronecker(C, complement(MNotLower), Plus<double>(), Times<double>(), A, transpose(A));
    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_Accum_ABT_ACdup)
{
    // Build some matrices.
    Matrix<double> Ones(Ones_3x4, 0.);
    Matrix<double> mat(B_sparse_3x4, 0.);
    Matrix<double> C(3, 4);
    Matrix<double> B(1, 1);    B.setElement(0, 0, 1.0);
    Matrix<double> answer(mat);

    Matrix<double> MLower(Lower_3x4, 0.);
    Matrix<double> MNotLower(NotLower_3x4, 0.);

    // Replace
    C = mat;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MLower, answer);
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), answer, answer);
    kronecker(C, complement(MNotLower), Plus<double>(), Times<double>(), C, transpose(B), REPLACE);

    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    C = mat;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MLower, mat);
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), mat, answer);
    kronecker(C, complement(MNotLower), Plus<double>(), Times<double>(), C, transpose(B));
    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_Accum_ABT_BCdup)
{
    // Build some matrices.
    Matrix<double> mat(A_sparse_3x3, 0.);
    Matrix<double> C(3,3);
    Matrix<double> A(1, 1);    A.setElement(0, 0, 1.0);
    Matrix<double> answer(mat);

    Matrix<double> MLower(Lower_3x3, 0.);
    Matrix<double> MNotLower(NotLower_3x3, 0.);

    // Replace
    C = mat;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MLower, transpose(mat));
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), answer, mat);
    kronecker(C, complement(MNotLower), Plus<double>(), Times<double>(), A, transpose(C), REPLACE);

    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    C = mat;
    kronecker(C, complement(MNotLower), Plus<double>(), Times<double>(), A, transpose(C));

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_Accum_ABT_MCdup)
{
    // Build some matrices.
    Matrix<double> C(9,12);
    Matrix<double> Ones(Ones_9x12, 0.);
    Matrix<double> A(A_sparse_3x3, 0.);
    Matrix<double> B(B_sparse_4x3, 0.);
    Matrix<double> answer(Answer_sparse_9x12, 0.);

    Matrix<double> MLower(Lower_9x12, 0.);
    Matrix<double> MNotLower(NotLower_9x12, 0.);

    // Replace
    C = MLower;
    eWiseMult(answer, NoMask(), NoAccumulate(), Times<double>(), MNotLower, answer);
    kronecker(C, complement(C), Plus<double>(), Times<double>(), A, transpose(B), REPLACE);

    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    C = MLower;
    eWiseAdd(answer, NoMask(), NoAccumulate(), Plus<double>(), MLower, answer);
    kronecker(C, complement(C), Plus<double>(), Times<double>(), A, transpose(B));

    BOOST_CHECK_EQUAL(C, answer);
}

BOOST_AUTO_TEST_SUITE_END()
