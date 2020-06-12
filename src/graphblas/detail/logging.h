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

#pragma once

#include <typeinfo>

// Basic debugging
#if GRAPHBLAS_LOGGING_LEVEL > 0

    #define GRB_LOG(x)          do { std::cout << "GRB " << x << std::endl; } while(0)
    #define GRB_LOG_FN_BEGIN(x) do { std::cout << "GRB >>> Begin: " << x << std::endl; } while(0)
    #define GRB_LOG_FN_END(x)   do { std::cout << "GRB <<< End:   " << x << std::endl; } while(0)

    //  Verbose debugging
    #if GRAPHBLAS_LOGGING_LEVEL > 1

        #define GRB_LOG_VERBOSE(x) do { std::cout << "GRB --- " << x << std::endl; } while(0)
        #define GRB_LOG_VERBOSE_ACCUM(x) do { std::cout << "GRB --- accum: " << typeid(x).name() << std::endl; } while(0)
        #define GRB_LOG_VERBOSE_OP(x) do { std::cout << "GRB --- op: " << typeid(x).name() << std::endl; } while(0)
        #define GRB_LOG_VERBOSE_OUTP(x) do { std::cout << "GRB --- outp: " << ((x == grb::MERGE) ? "MERGE" : "REPLACE") << std::endl; } while(0)

    #else

        #define GRB_LOG_VERBOSE(x)
        #define GRB_LOG_VERBOSE_ACCUM(x)
        #define GRB_LOG_VERBOSE_OP(x)
        #define GRB_LOG_VERBOSE_OUTP(x)

    #endif

#else

    #define GRB_LOG(x)
    #define GRB_LOG_FN_BEGIN(x)
    #define GRB_LOG_FN_END(x)

    #define GRB_LOG_VERBOSE(x)
    #define GRB_LOG_VERBOSE_ACCUM(x)
    #define GRB_LOG_VERBOSE_OP(x)
    #define GRB_LOG_VERBOSE_OUTP(x)

#endif
