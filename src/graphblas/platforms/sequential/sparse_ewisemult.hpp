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

#ifndef GB_SEQUENTIAL_SPARSE_EWISEMULT_HPP
#define GB_SEQUENTIAL_SPARSE_EWISEMULT_HPP

#pragma once

#include <functional>
#include <utility>
#include <vector>
#include <iterator>
#include <iostream>
#include <graphblas/types.hpp>
#include <graphblas/algebra.hpp>

#include "sparse_helpers.hpp"
#include "LilSparseMatrix.hpp"

#include "graphblas/detail/logging.h"

//****************************************************************************

namespace GraphBLAS
{
    namespace backend
    {
        //**********************************************************************
        /// Implementation of 4.3.4.1 eWiseMult: Vector variant
        template<typename WScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  //can be BinaryOp, Monoid (not Semiring)
                 typename UVectorT,
                 typename VVectorT,
                 typename... WTagsT>
        inline void eWiseMult(
            GraphBLAS::backend::Vector<WScalarT, WTagsT...> &w,
            MaskT                                     const &mask,
            AccumT                                           accum,
            BinaryOpT                                        op,
            UVectorT                                  const &u,
            VVectorT                                  const &v,
            bool                                             replace_flag = false)
        {
            // =================================================================
            // Do the basic ewise-and work: t = u .* v
            typedef typename BinaryOpT::result_type D3ScalarType;
            std::vector<std::tuple<IndexType,D3ScalarType> > t_contents;

            if ((u.nvals() > 0) && (v.nvals() > 0))
            {
                auto u_contents(u.getContents());
                auto v_contents(v.getContents());

                ewise_and(t_contents, u_contents, v_contents, op);
            }

            // =================================================================
            // Accumulate into Z
            std::vector<std::tuple<IndexType,WScalarT> > z_contents;
            ewise_or_opt_accum_1D(z_contents, w, t_contents, accum);

            // =================================================================
            // Copy Z into the final output considering mask and replace
            write_with_opt_mask_1D(w, z_contents, mask, replace_flag);
        }

        //**********************************************************************
        /// Implementation of 4.3.4.2 eWiseMult: Matrix variant
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  //can be BinaryOp, Monoid (not Semiring)
                 typename AMatrixT,
                 typename BMatrixT,
                 typename... CTagsT>
        inline void eWiseMult(
            GraphBLAS::backend::Matrix<CScalarT, CTagsT...> &C,
            MaskT                                     const &Mask,
            AccumT                                           accum,
            BinaryOpT                                        op,
            AMatrixT                                  const &A,
            BMatrixT                                  const &B,
            bool                                             replace_flag = false)
        {
            IndexType num_rows(A.nrows());
            IndexType num_cols(A.ncols());

            typedef typename AMatrixT::ScalarType AScalarType;
            typedef typename BMatrixT::ScalarType BScalarType;

            typedef std::vector<std::tuple<IndexType,AScalarType> > ARowType;
            typedef std::vector<std::tuple<IndexType,BScalarType> > BRowType;
            typedef std::vector<std::tuple<IndexType,CScalarT> > CRowType;

            // =================================================================
            // Do the basic ewise-and work: T = A .* B
            typedef typename BinaryOpT::result_type D3ScalarType;
            typedef std::vector<std::tuple<IndexType,D3ScalarType> > TRowType;
            LilSparseMatrix<D3ScalarType> T(num_rows, num_cols);

            if ((A.nvals() > 0) && (B.nvals() > 0))
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
                            ewise_and(T_row, A_row, B_row, op);

                            if (!T_row.empty())
                            {
                                T.setRow(row_idx, T_row);
                                T_row.clear();
                            }
                        }
                    }
                }
            }

//            GRB_LOG_E(">>> T <<<");
//            GRB_LOG_E(T);

            // =================================================================
            // Accumulate into Z

            LilSparseMatrix<CScalarT> Z(num_rows, num_cols);
            ewise_or_opt_accum(Z, C, T, accum);

//            GRB_LOG_E(">>> Z <<< ");
//            GRB_LOG_E(Z);

            // =================================================================
            // Copy Z into the final output considering mask and replace
            write_with_opt_mask(C, Z, Mask, replace_flag);

//            GRB_LOG_E(">>> C <<< ");
//            GRB_LOG_E(C);

        } // ewisemult

    } // backend
} // GraphBLAS

#endif
