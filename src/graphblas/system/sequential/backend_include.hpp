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

#if(GB_INCLUDE_BACKEND_ALL)
#include <graphblas/system/sequential/sequential.hpp>
#endif

#if(GB_INCLUDE_BACKEND_MATRIX)
#include <graphblas/system/sequential/Matrix.hpp>
#undef GB_INCLUDE_BACKEND_MATRIX
#endif

#if(GB_INCLUDE_BACKEND_VECTOR)
#include <graphblas/system/sequential/Vector.hpp>
#undef GB_INCLUDE_BACKEND_VECTOR
#endif

#if(GB_INCLUDE_BACKEND_UTILITY)
#include <graphblas/system/sequential/utility.hpp>
#undef GB_INCLUDE_BACKEND_UTILITY
#endif

#if(GB_INCLUDE_BACKEND_TRANSPOSE_VIEW)
#include <graphblas/system/sequential/TransposeView.hpp>
#undef GB_INCLUDE_BACKEND_TRANSPOSE_VIEW
#endif

#if(GB_INCLUDE_BACKEND_COMPLEMENT_VIEW)
#include <graphblas/system/sequential/ComplementView.hpp>
#undef GB_INCLUDE_BACKEND_COMPLEMENT_VIEW
#endif

#if(GB_INCLUDE_BACKEND_OPERATIONS)
#include <graphblas/system/sequential/operations.hpp>
#undef GB_INCLUDE_BACKEND_OPERATIONS
#endif
