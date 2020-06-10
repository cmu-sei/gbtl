/*
 * GraphBLAS Template Library, Version 2.1
 *
 * Copyright 2020 Carnegie Mellon University, Battelle Memorial Institute, and
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
#include <graphblas/platforms/sequential/sparse_mxm.hpp>
#include <graphblas/platforms/sequential/sparse_mxv.hpp>
#include <graphblas/platforms/sequential/sparse_vxm.hpp>
#include <graphblas/platforms/sequential/sparse_ewisemult.hpp>
#include <graphblas/platforms/sequential/sparse_ewiseadd.hpp>
#include <graphblas/platforms/sequential/sparse_extract.hpp>
#include <graphblas/platforms/sequential/sparse_assign.hpp>
#include <graphblas/platforms/sequential/sparse_apply.hpp>
#include <graphblas/platforms/sequential/sparse_reduce.hpp>
#include <graphblas/platforms/sequential/sparse_transpose.hpp>
#include <graphblas/platforms/sequential/sparse_kronecker.hpp>
