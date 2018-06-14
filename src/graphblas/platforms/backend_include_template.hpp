/*
 * Copyright (c) 2015 Carnegie Mellon University and The Trustees of Indiana
 * University.
 * All Rights Reserved.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS," WITH NO WARRANTIES WHATSOEVER. CARNEGIE
 * MELLON UNIVERSITY AND THE TRUSTEES OF INDIANA UNIVERSITY EXPRESSLY DISCLAIM
 * TO THE FULLEST EXTENT PERMITTED BY LAW ALL EXPRESS, IMPLIED, AND STATUTORY
 * WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT OF PROPRIETARY RIGHTS.
 *
 * This Program is distributed under a BSD license.  Please see LICENSE file or
 * permission@sei.cmu.edu for more information.  DM-0002659
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
