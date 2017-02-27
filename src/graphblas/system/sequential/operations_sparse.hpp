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

namespace GraphBLAS { namespace backend {
    /**
     *
     */
    template<typename AMatrixT,
             typename BMatrixT,
             typename CMatrixT,
             typename SemiringT,
             typename AccumT>
    inline void mxm(AMatrixT const &A,
                    BMatrixT const &B,
                    CMatrixT       &C,
                    SemiringT       op,
                    AccumT          accum)
    {
        IndexType nrow_A, ncol_A;
        IndexType nrow_B, ncol_B;
        IndexType nrow_C, ncol_C;
        
        A.nrows(nrow_A);
        A.ncols(ncol_A);
        B.nrows(nrow_B);
        B.ncols(ncol_B);
        C.nrows(nrow_C);
        C.ncols(ncol_C);
        
        if (ncol_A != nrow_B || nrow_A != nrow_C || ncol_B != ncol_C)
        {
            throw DimensionException("mxm: matrix dimensions are mismatched");
        }
        
        /*
         if (!std::is_same(A.ScalarType, B.ScalarType) ||
         !std::is_same(A.ScalarType, C.ScalarType))
         {
         // throw exception
         }
         */
        
        IndexType irow;
        IndexType icol;
        std::vector<IndexArrayType> indA;   // Row-column
        std::vector<IndexArrayType> indB;   // Column-row
        std::vector<IndexType> ind_intersection;
        
        indA.resize(nrow_A);
        indB.resize(ncol_B);
        
        auto tmp_sum = op.zero();
        auto tmp_product = op.zero();
        for (irow = 0; irow < nrow_C; irow++)
        {
            A.getColumnIndices(irow, indA[irow]);
            for (icol = 0; icol < ncol_C; icol++)
            {
                B.getRowIndices(icol, indB[icol]);
                if (!indA[irow].empty() && !indB[icol].empty())
                {
                    ind_intersection.clear();
                    std::set_intersection(indA[irow].begin(), indA[irow].end(),
                                          indB[icol].begin(), indB[icol].end(),
                                          std::back_inserter(ind_intersection));
                    if (!ind_intersection.empty())
                    {
                        tmp_sum = op.zero();
                        for (auto kk : ind_intersection) // Range-based loop, access by value
                        {
                            // Matrix multiply kernel
                            tmp_product = op.mult(A.get_value_at(irow,kk), B.get_value_at(kk,icol));
                            tmp_sum = op.add(tmp_sum, tmp_product);
                        }
                        C.set_value_at(irow, icol, tmp_sum);
                    }
                }
            }
        }
    }
}}

#endif // GB_SEQUENTIAL_OPERATIONS_SPARSE_HPP
