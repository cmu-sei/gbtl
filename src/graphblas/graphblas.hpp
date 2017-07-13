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

#ifndef GRAPHBLAS_HPP
#define GRAPHBLAS_HPP

#pragma once

#include <graphblas/detail/config.hpp>

#include <graphblas/types.hpp>
#include <graphblas/exceptions.hpp>

//#include <graphblas/accum.hpp>
#include <graphblas/algebra.hpp>

#include <graphblas/Matrix.hpp>
#include <graphblas/Vector.hpp>
#include <graphblas/ComplementView.hpp>
#include <graphblas/TransposeView.hpp>
//#include <graphblas/View.hpp> // deprecated

#include <graphblas/operations.hpp>

//#include <graphblas/matrix_utils.hpp> // Does not belong
//#include <graphblas/linalg_utils.hpp>  // deprecated
//#include <graphblas/utility.hpp>   // deprecated

#define __GB_SYSTEM_HEADER <graphblas/system/__GB_SYSTEM_ROOT/__GB_SYSTEM_ROOT.hpp>
#include __GB_SYSTEM_HEADER
#undef __GB_SYSTEM_HEADER

#endif // GRAPHBLAS_HPP
