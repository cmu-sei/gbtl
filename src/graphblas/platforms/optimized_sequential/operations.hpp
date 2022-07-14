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

/**
 * Implementations of all GraphBLAS functions optimized for the sequential
 * (CPU) backend.
 */

#pragma once

#include <functional>
#include <utility>
#include <vector>
#include <iterator>

#include <graphblas/algebra.hpp>

// Add individual operation files here
#include <graphblas/platforms/optimized_sequential/sparse_mxm.hpp>
#include <graphblas/platforms/optimized_sequential/sparse_mxv.hpp>
#include <graphblas/platforms/optimized_sequential/sparse_vxm.hpp>
#include <graphblas/platforms/optimized_sequential/sparse_ewisemult.hpp>
#include <graphblas/platforms/optimized_sequential/sparse_ewiseadd.hpp>
#include <graphblas/platforms/optimized_sequential/sparse_extract.hpp>
#include <graphblas/platforms/optimized_sequential/sparse_assign.hpp>
#include <graphblas/platforms/optimized_sequential/sparse_apply.hpp>
#include <graphblas/platforms/optimized_sequential/sparse_reduce.hpp>
#include <graphblas/platforms/optimized_sequential/sparse_transpose.hpp>
#include <graphblas/platforms/optimized_sequential/sparse_kronecker.hpp>
