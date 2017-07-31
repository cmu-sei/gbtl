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

/**
 * Implementations of all GraphBLAS functions optimized for the sequential
 * (CPU) backend.
 */

#ifndef GB_SEQUENTIAL_OPERATIONS_HPP
#define GB_SEQUENTIAL_OPERATIONS_HPP

#pragma once

#include <functional>
#include <utility>
#include <vector>
#include <iterator>

#include <graphblas/algebra.hpp>
#include <graphblas/system/sequential/TransposeView.hpp>
#include <graphblas/system/sequential/ComplementView.hpp>

// Add individual operation files here
#include <graphblas/system/sequential/sparse_mxm.hpp>
#include <graphblas/system/sequential/sparse_mxv.hpp>
#include <graphblas/system/sequential/sparse_vxm.hpp>
#include <graphblas/system/sequential/sparse_ewisemult.hpp>
#include <graphblas/system/sequential/sparse_ewiseadd.hpp>
#include <graphblas/system/sequential/sparse_extract.hpp>
#include <graphblas/system/sequential/sparse_assign.hpp>
#include <graphblas/system/sequential/sparse_apply.hpp>
#include <graphblas/system/sequential/sparse_reduce.hpp>
#include <graphblas/system/sequential/sparse_transpose.hpp>


namespace GraphBLAS
{
    namespace backend
    {
        /**
         *
         */

        template<typename MatrixT>
        inline MatrixComplementView<MatrixT> matrix_complement(MatrixT const &Mask)
        {
            return MatrixComplementView<MatrixT>(Mask);
        }

        template<typename VectorT>
        inline VectorComplementView<VectorT> vector_complement(VectorT const &mask)
        {
            return VectorComplementView<VectorT>(mask);
        }


        /**
         *
         */
        template<typename MatrixT>
        inline TransposeView<MatrixT> transpose(MatrixT const &A)
        {
            return TransposeView<MatrixT>(A);
        }

    } // backend
} // GraphBLAS

#endif // GB_SEQUENTIAL_OPERATIONS_HPP
