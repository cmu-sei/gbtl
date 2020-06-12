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

// C API Version equivalents
#define GRB_VERSION    1
#define GRB_SUBVERSION 3

namespace grb
{
    auto getVersion()
    {
        return std::make_tuple(static_cast<unsigned int>(GRB_VERSION),
                               static_cast<unsigned int>(GRB_SUBVERSION));
    }
}

#include <graphblas/detail/config.hpp>

#include <graphblas/types.hpp>
#include <graphblas/exceptions.hpp>

#include <graphblas/algebra.hpp>

#include <graphblas/Matrix.hpp>
#include <graphblas/Vector.hpp>
#include <graphblas/StructureView.hpp>
#include <graphblas/ComplementView.hpp>
#include <graphblas/StructuralComplementView.hpp>
#include <graphblas/TransposeView.hpp>

#include <graphblas/operations.hpp>
#include <graphblas/matrix_utils.hpp>

#define GB_INCLUDE_BACKEND_ALL 1
#include <backend_include.hpp>
