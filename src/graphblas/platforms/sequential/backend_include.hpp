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

// DO NOT ADD HEADER INCLUSION PROTECTION
// This file is a dispatch mechanism to allow us to include different
// sets of files as specified by the user.

#if(GB_INCLUDE_BACKEND_ALL)
#include <graphblas/platforms/sequential/sequential.hpp>
#endif

#if(GB_INCLUDE_BACKEND_MATRIX)
#include <graphblas/platforms/sequential/Matrix.hpp>
#undef GB_INCLUDE_BACKEND_MATRIX
#endif

#if(GB_INCLUDE_BACKEND_VECTOR)
#include <graphblas/platforms/sequential/Vector.hpp>
#undef GB_INCLUDE_BACKEND_VECTOR
#endif

#if(GB_INCLUDE_BACKEND_UTILITY)
#include <graphblas/platforms/sequential/utility.hpp>
#undef GB_INCLUDE_BACKEND_UTILITY
#endif

#if(GB_INCLUDE_BACKEND_TRANSPOSE_VIEW)
#include <graphblas/platforms/sequential/TransposeView.hpp>
#undef GB_INCLUDE_BACKEND_TRANSPOSE_VIEW
#endif

#if(GB_INCLUDE_BACKEND_COMPLEMENT_VIEW)
#include <graphblas/platforms/sequential/ComplementView.hpp>
#undef GB_INCLUDE_BACKEND_COMPLEMENT_VIEW
#endif

#if(GB_INCLUDE_BACKEND_OPERATIONS)
#include <graphblas/platforms/sequential/operations.hpp>
#undef GB_INCLUDE_BACKEND_OPERATIONS
#endif
