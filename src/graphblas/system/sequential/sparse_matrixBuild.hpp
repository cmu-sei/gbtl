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

#ifndef GB_SEQUENTIAL_SPARSE_MATRIX_BUILD_HPP
#define GB_SEQUENTIAL_SPARSE_MATRIX_BUILD_HPP

#pragma once

#include <functional>
#include <utility>
#include <vector>
#include <iterator>
#include <iostream>
#include <graphblas/accum.hpp>
#include <graphblas/algebra.hpp>

#include "sparse_helpers.hpp"


//****************************************************************************

namespace GraphBLAS
{
    namespace backend
    {
        template<typename CMatrixT,
                 typename RAIteratorIT,
                 typename RAIteratorJT,
                 typename RAIteratorVT,
                 typename BinaryOpT>
        inline void matrixBuild(CMatrixT           &C,
                                RAIteratorIT        row_it,
                                RAIteratorJT        col_it,
                                RAIteratorVT        val_it,
                                IndexType           num_vals,
                                BinaryOpT           dup)
        {
            if (C.get_nvals() != 0)
            {
                /// @todo Do we silently clear or throw an exception
                //throw NotEmptyException("matrixBuild");
            }

            // C.build() currently calls C.clear()
            C.build(row_it, col_it, val_it, num_vals, dup);
        }


    } // backend
} // GraphBLAS

#endif
