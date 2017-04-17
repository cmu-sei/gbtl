/*
 * Copyright (c) 2017 Carnegie Mellon University.
 * All Rights Reserved.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS," WITH NO WARRANTIES WHATSOEVER. CARNEGIE
 * MELLON UNIVERSITY EXPRESSLY DISCLAIMS TO THE FULLEST EXTENT PERMITTED BY
 * LAW ALL EXPRESS, IMPLIED, AND STATUTORY WARRANTIES, INCLUDING, WITHOUT
 * LIMITATION, THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE, AND NON-INFRINGEMENT OF PROPRIETARY RIGHTS.
 *
 * This Program is distributed under a BSD license.  Please see LICENSE file or
 * permission@sei.cmu.edu for more information.  DM-0002659
 */

/**
 * Implementations of all GraphBLAS functions optimized for the sequential
 * (CPU) backend.
 */

#ifndef GB_SEQUENTIAL_OPERATIONS_SPARSE_HPP
#define GB_SEQUENTIAL_OPERATIONS_SPARSE_HPP

#pragma once

#include <functional>
#include <utility>
#include <vector>
#include <iterator>
#include <iostream>
#include <graphblas/accum.hpp>
#include <graphblas/algebra.hpp>
#include <graphblas/system/sequential/TransposeView.hpp>
#include <graphblas/system/sequential/NegateView.hpp>

// Add individual operation files here
//#include <graphblas/system/sequential/.hpp>
#include <graphblas/system/sequential/sparse_matrixBuild.hpp>
#include <graphblas/system/sequential/sparse_mxm.hpp>
#include <graphblas/system/sequential/sparse_mxv.hpp>

#endif // GB_SEQUENTIAL_OPERATIONS_SPARSE_HPP
