/*
 * Copyright (c) 2017 Carnegie Mellon University.
 * All Rights Reserved.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS," WITH NO WARRANTIES WHATSOEVER. CARNEGIE
 * MELLON UNIVERSITY EXPRESSLY DISCLAIMS TO THE FULLEST EXTENT PERMITTED BY
 * LAW ALL EXPRESS, IMPLIED, AND STATUTORY WARRANTIES, INCLUDING, WITHOUT
 * LIMITATION, THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE, AND NON-INFRINGEMENT OF PROPRIETARY RIGHTS.
 *
 * This Program is distributed under a BSD license.  Please see LICENSE file or
 * permission@sei.cmu.edu for more information.  DM-0002659
 */

#ifndef GB_LOGGING_H
#define GB_LOGGING_H

// Basic debugging
#if GRAPHBLAS_LOGGING_LEVEL > 0

    #define GRB_LOG(x)          do { std::cout << "GRB " << x << std::endl; } while(0)
    #define GRB_LOG_FN_BEGIN(x) do { std::cout << "GRB >>> Begin: " << x << std::endl; } while(0)
    #define GRB_LOG_FN_END(x)   do { std::cout << "GRB <<< End:   " << x << std::endl; } while(0)

    //  Verbose debugging
    #if GRAPHBLAS_LOGGING_LEVEL > 1

        #define GRB_LOG_VERBOSE(x) do { std::cout << "GRB --- " << x << std::endl; } while(0)
        #define GRB_LOG_VERBOSE_ACCUM(x) do { std::cout << "GRB --- accum: TBD" << std::endl; } while(0)
        #define GRB_LOG_VERBOSE_OP(x) do { std::cout << "GRB --- op: TBD" << std::endl; } while(0)
        #define GRB_LOG_VERBOSE_REPLACE(x) do { std::cout << "GRB --- replace: " << std::boolalpha << x << std::endl; } while(0)

    #else

        #define GRB_LOG_VERBOSE(x)
        #define GRB_LOG_VERBOSE_ACCUM(x)
        #define GRB_LOG_VERBOSE_OP(x)
        #define GRB_LOG_VERBOSE_REPLACE(x)

    #endif

#else

    #define GRB_LOG(x)
    #define GRB_LOG_FN_BEGIN(x)
    #define GRB_LOG_FN_END(x)

    #define GRB_LOG_VERBOSE(x)
    #define GRB_LOG_VERBOSE_ACCUM(x)
    #define GRB_LOG_VERBOSE_OP(x)
    #define GRB_LOG_VERBOSE_REPLACE(x)

#endif

#endif //GB_LOGGING_H
