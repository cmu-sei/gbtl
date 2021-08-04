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

#include <functional>
#include <utility>
#include <vector>
#include <iterator>
#include <iostream>
#include <graphblas/algebra.hpp>
#include <atomic>

#include "test/Timer.hpp"

// #define INST_TIMING_MVX

#define GKC_MXV_V3

// Includes for compatability with exsiting GBTL "sequential" data structures and methods
#include "sparse_mxv_old.hpp"
// Includes for the GKC variants:
#if defined(GKC_MXV_V1)
#include "mxv_v1/sparse_mxv_nomask.hpp"
#include "mxv_v1/sparse_mxv_masked.hpp"
#elif defined(GKC_MXV_V2)
#include "mxv_v2_consolidated/sparse_mxv_all.hpp"
#elif defined(GKC_MXV_V3)
#include "mxv_v3_consolidated/sparse_mxv_all.hpp"
#elif defined(GKC_MXV_V4)
#include "mxv_v4_consolidated/sparse_mxv_all.hpp"
#else
#error "At least one GKC MxV implementation set must be included!"
#endif


// This header is a landing point to include mxv implementations.
// vxm is accomplished by forwarding to template-matching mxv implementations.


//****************************************************************************
