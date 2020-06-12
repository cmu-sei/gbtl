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

// !!!! DO NOT ADD HEADER INCLUSION PROTECTION !!!!

// This file is a dispatch mechanism to allow us to include different
// sets of files as specified by the user.

// 1. Global search and replace GB_BACKEND_NAME with the name of the
// platform directory.

// 2. Specify a single include file that contains the include
// directives for all of the platform's header files
#if(GB_INCLUDE_BACKEND_ALL)
#include <graphblas/platforms/GB_BACKEND_NAME/GB_BACKEND_NAME.hpp>
#undef GB_INCLUDE_BACKEND_ALL
#endif

// 3. Specify the include file that defines the platform's Matrix base class
#if(GB_INCLUDE_BACKEND_MATRIX)
#include <graphblas/platforms/GB_BACKEND_NAME/Matrix.hpp>
#undef GB_INCLUDE_BACKEND_MATRIX
#endif

// 4. Specify the include file that defines the platform's Vector base class
#if(GB_INCLUDE_BACKEND_VECTOR)
#include <graphblas/platforms/GB_BACKEND_NAME/Vector.hpp>
#undef GB_INCLUDE_BACKEND_VECTOR
#endif

// 5. Specify the include file(s) that defines the platform's
// operations functions.
#if(GB_INCLUDE_BACKEND_OPERATIONS)
#include <graphblas/platforms/GB_BACKEND_NAME/operations.hpp>
#undef GB_INCLUDE_BACKEND_OPERATIONS
#endif
