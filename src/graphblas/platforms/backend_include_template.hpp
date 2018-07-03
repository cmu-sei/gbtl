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

// 5. Specify the include file that defines the pretty_print and
// pretty_print_matrix functions.
#if(GB_INCLUDE_BACKEND_UTILITY)
#include <graphblas/platforms/GB_BACKEND_NAME/utility.hpp>
#undef GB_INCLUDE_BACKEND_UTILITY
#endif

// 6. Specify the include file that defines the platform's TransposeView class
#if(GB_INCLUDE_BACKEND_TRANSPOSE_VIEW)
#include <graphblas/platforms/GB_BACKEND_NAME/TransposeView.hpp>
#undef GB_INCLUDE_BACKEND_TRANSPOSE_VIEW
#endif

// 7. Specify the include file that defines the platform's ComplementView class
#if(GB_INCLUDE_BACKEND_COMPLEMENT_VIEW)
#include <graphblas/platforms/GB_BACKEND_NAME/ComplementView.hpp>
#undef GB_INCLUDE_BACKEND_COMPLEMENT_VIEW
#endif

// 8. Specify the include file(s) that defines the platform's
// operations functions.
#if(GB_INCLUDE_BACKEND_OPERATIONS)
#include <graphblas/platforms/GB_BACKEND_NAME/operations.hpp>
#undef GB_INCLUDE_BACKEND_OPERATIONS
#endif
