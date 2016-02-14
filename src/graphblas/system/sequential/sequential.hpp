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

#ifndef GB_SEQUENTIAL_HPP
#define GB_SEQUENTIAL_HPP

#pragma once

#include <graphblas/system/sequential/utility.hpp>

#include <graphblas/system/sequential/LilMatrix.hpp>
#include <graphblas/system/sequential/DiaMatrix.hpp>
#include <graphblas/system/sequential/CsrMatrix.hpp>
#include <graphblas/system/sequential/CscMatrix.hpp>
#include <graphblas/system/sequential/coo.hpp>
#include <graphblas/system/sequential/ConstantMatrix.hpp>

#include <graphblas/system/sequential/RowView.hpp>
#include <graphblas/system/sequential/ColumnView.hpp>
#include <graphblas/system/sequential/TransposeView.hpp>
#include <graphblas/system/sequential/RowExtendedView.hpp>
#include <graphblas/system/sequential/ColumnExtendedView.hpp>
#include <graphblas/system/sequential/NegateView.hpp>

#include <graphblas/system/sequential/operations.hpp>

#endif // GB_SEQUENTIAL_HPP
