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

#include <graphblas/algebra.hpp>

#include <graphblas/Matrix.hpp>
#include <graphblas/Vector.hpp>
#include <graphblas/ComplementView.hpp>
#include <graphblas/TransposeView.hpp>

#include <graphblas/operations.hpp>
#include <graphblas/matrix_utils.hpp>

#define GB_INCLUDE_BACKEND_ALL 1
#include <graphblas/backend_include.hpp>

#endif // GRAPHBLAS_HPP
