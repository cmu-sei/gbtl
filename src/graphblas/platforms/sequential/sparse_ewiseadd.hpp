/*
 * GraphBLAS Template Library, Version 2.0
 *
 * Copyright 2018 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors. All Rights Reserved.
 *
 * THIS MATERIAL WAS PREPARED AS AN ACCOUNT OF WORK SPONSORED BY AN AGENCY OF
 * THE UNITED STATES GOVERNMENT.  NEITHER THE UNITED STATES GOVERNMENT NOR THE
 * UNITED STATES DEPARTMENT OF ENERGY, NOR THE UNITED STATES DEPARTMENT OF
 * DEFENSE, NOR CARNEGIE MELLON UNIVERSITY, NOR BATTELLE, NOR ANY OF THEIR
 * EMPLOYEES, NOR ANY JURISDICTION OR ORGANIZATION THAT HAS COOPERATED IN THE
 * DEVELOPMENT OF THESE MATERIALS, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS,
 * OR USEFULNESS OR ANY INFORMATION, APPARATUS, PRODUCT, SOFTWARE, OR PROCESS
 * DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED
 * RIGHTS..
 *
 * Released under a BSD (SEI)-style license, please see license.txt or contact
 * permission@sei.cmu.edu for full terms.
 *
 * This release is an update of:
 *
 * 1. GraphBLAS Template Library (GBTL)
 * (https://github.com/cmu-sei/gbtl/blob/1.0.0/LICENSE) Copyright 2015 Carnegie
 * Mellon University and The Trustees of Indiana. DM17-0037, DM-0002659
 *
 * DM18-0559
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
#include <graphblas/algebra.hpp>

#include "sparse_helpers.hpp"
#include "LilSparseMatrix.hpp"


//****************************************************************************

namespace GraphBLAS
{
    namespace backend
    {
        //**********************************************************************
        /// Implementation of 4.3.5.1 eWiseAdd: Vector variant
        template<typename WScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  //can be BinaryOp, Monoid (not Semiring)
                 typename UVectorT,
                 typename VVectorT,
                 typename ...WTagsT>
        inline Info eWiseAdd(
            GraphBLAS::backend::Vector<WScalarT, WTagsT...> &w,
            MaskT                                     const &mask,
            AccumT                                           accum,
            BinaryOpT                                        op,
            UVectorT                                  const &u,
            VVectorT                                  const &v,
            bool                                             replace_flag = false)
        {
            // =================================================================
            // Do the basic ewise-and work: T = A .* B
            typedef typename BinaryOpT::result_type D3ScalarType;
            std::vector<std::tuple<IndexType,D3ScalarType> > t_contents;

            if ((u.nvals() > 0) || (v.nvals() > 0))
            {
                auto u_contents(u.getContents());
                auto v_contents(v.getContents());

                ewise_or(t_contents, u_contents, v_contents, op);
            }

            // =================================================================
            // Accumulate into Z
            std::vector<std::tuple<IndexType,WScalarT> > z_contents;
            ewise_or_opt_accum_1D(z_contents, w, t_contents, accum);

            // =================================================================
            // Copy Z into the final output considering mask and replace
            write_with_opt_mask_1D(w, z_contents, mask, replace_flag);
            return SUCCESS;
        }

        //**********************************************************************
        /// Implementation of 4.3.5.2 eWiseAdd: Matrix variant
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  //can be BinaryOp, Monoid (not Semiring)
                 typename AMatrixT,
                 typename BMatrixT,
                 typename ...CTagsT>
        inline Info eWiseAdd(
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

            if ((A.nvals() > 0) || (B.nvals() > 0))
            {
                // create a row of result at a time
                TRowType T_row;
                for (IndexType row_idx = 0; row_idx < num_rows; ++row_idx)
                {
                    ARowType A_row(A.getRow(row_idx));
                    BRowType B_row(B.getRow(row_idx));

                    if (B_row.empty())
                    {
                        T.setRow(row_idx, A_row);
                    }
                    else if (A_row.empty())
                    {
                        T.setRow(row_idx, B_row);
                    }
                    else
                    {
                        ewise_or(T_row, A_row, B_row, op);

                        if (!T_row.empty())
                        {
                            T.setRow(row_idx, T_row);
                            T_row.clear();
                        }
                    }
                }
            }

            // =================================================================
            // Accumulate into Z
            LilSparseMatrix<CScalarT> Z(num_rows, num_cols);
            ewise_or_opt_accum(Z, C, T, accum);

            // =================================================================
            // Copy Z into the final output considering mask and replace
            write_with_opt_mask(C, Z, Mask, replace_flag);

            return SUCCESS;
        } // ewisemult

    } // backend
} // GraphBLAS

#endif
