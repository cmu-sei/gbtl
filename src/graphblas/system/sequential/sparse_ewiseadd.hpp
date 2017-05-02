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

#ifndef GB_SEQUENTIAL_SPARSE_EWISEADD_HPP
#define GB_SEQUENTIAL_SPARSE_EWISEADD_HPP

#pragma once

#include <functional>
#include <utility>
#include <vector>
#include <iterator>
#include <iostream>
#include <graphblas/types.hpp>
#include <graphblas/accum.hpp>
#include <graphblas/algebra.hpp>

#include "sparse_helpers.hpp"
#include "LilSparseMatrix.hpp"


//****************************************************************************

namespace GraphBLAS
{
    namespace backend
    {
        //**********************************************************************
        /// Implementation of 4.3.4.2 eWiseMult: Matrix variant
        template<typename CMatrixT,
                 typename MMatrixT,
                 typename AccumT,
                 typename BinaryOpT,
                 typename AMatrixT,
                 typename BMatrixT>
        inline void eWiseAdd(CMatrixT       &C,
                             MMatrixT const &Mask,
                             AccumT const   &accum,
                             BinaryOpT       op,
                             AMatrixT const &A,
                             BMatrixT const &B,
                             bool            replace_flag = false)
        {
            // @todo: Make errors match the spec
            // @todo: support semiring op

            // ??? Do we need to make defensive copies of everything if we don't
            // really support NON-BLOCKING?
            check_matrix_size(C, Mask,
                              "eWiseMult(mat): failed C == Mask dimension checks");
            check_matrix_size(C, A,
                              "eWiseMult(mat): failed C == A dimension checks");
            check_matrix_size(A, B,
                              "eWiseMult(mat): failed A == B dimension checks");

            IndexType num_rows(A.nrows());
            IndexType num_cols(A.ncols());

            typedef typename AMatrixT::ScalarType AScalarType;
            typedef typename BMatrixT::ScalarType BScalarType;
            typedef typename CMatrixT::ScalarType CScalarType;
            typedef std::vector<std::tuple<IndexType,AScalarType> > ARowType;
            typedef std::vector<std::tuple<IndexType,BScalarType> > BRowType;
            typedef std::vector<std::tuple<IndexType,CScalarType> > CRowType;

            // =================================================================
            // Do the basic ewise-and work: T = A .* B
            typedef typename BinaryOpT::result_type D3ScalarType;
            typedef std::vector<std::tuple<IndexType,D3ScalarType> > TRowType;
            LilSparseMatrix<D3ScalarType> T(num_rows, num_cols);

            // Build this completely based on the semiring
            if ((A.nvals() > 0) || (B.nvals() > 0))
            {
                // create a row of result at a time
                TRowType T_row;
                for (IndexType row_idx = 0; row_idx < num_rows; ++row_idx)
                {
                    BRowType B_row(B.getRow(row_idx));

                    if (!B_row.empty())
                    {
                        ARowType A_row(A.getRow(row_idx));
                        if (!A_row.empty())
                        {
                            /// @todo Need to wrap op in helper that can
                            /// extract mult() as operator() if op is Semiring
                            ewise_or(T_row, A_row, B_row, op);

                            if (!T_row.empty())
                            {
                                T.setRow(row_idx, T_row);
                                T_row.clear();
                            }
                        }
                    }
                }
            }

            // =================================================================
            // Accumulate into Z

            LilSparseMatrix<CScalarType> Z(num_rows, num_cols);
            ewise_or_opt_accum(Z, C, T, accum);

            // =================================================================
            // Copy Z into the final output considering mask and replace
            write_with_opt_mask(C, Z, Mask, replace_flag);
        } // ewisemult

    } // backend
} // GraphBLAS

#endif
