/*
 * GraphBLAS Template Library (GBTL), Version 3.0
 *
 * Copyright 2020 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors.
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
 * RIGHTS.
 *
 * Released under a BSD-style license, please see LICENSE file or contact
 * permission@sei.cmu.edu for full terms.
 *
 * [DISTRIBUTION STATEMENT A] This material has been approved for public release
 * and unlimited distribution.  Please see Copyright notice for non-US
 * Government use and distribution.
 *
 * DM20-0442
 */

#pragma once

#include <functional>
#include <utility>
#include <vector>
#include <iterator>
#include <iostream>
#include <graphblas/types.hpp>
#include <graphblas/algebra.hpp>

#include "sparse_helpers.hpp"
#include "sparse_transpose.hpp"
#include "LilSparseMatrix.hpp"

#include "graphblas/detail/logging.h"

//****************************************************************************

namespace grb
{
    namespace backend
    {
        //**********************************************************************
        /// Implementation of 4.3.4.1 eWiseMult: Vector variant
        //**********************************************************************
        template<typename WScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  //can be BinaryOp, Monoid (not Semiring)
                 typename UVectorT,
                 typename VVectorT,
                 typename... WTagsT>
        inline void eWiseMult(
            grb::backend::Vector<WScalarT, WTagsT...>       &w,
            MaskT                                     const &mask,
            AccumT                                    const &accum,
            BinaryOpT                                        op,
            UVectorT                                  const &u,
            VVectorT                                  const &v,
            OutputControlEnum                                outp)
        {
            // =================================================================
            // Do the basic ewise-and work: t = u .* v
            using D3ScalarType =
                decltype(op(std::declval<typename UVectorT::ScalarType>(),
                            std::declval<typename VVectorT::ScalarType>()));
            std::vector<std::tuple<IndexType,D3ScalarType> > t_contents;

            if ((u.nvals() > 0) && (v.nvals() > 0))
            {
                ewise_and(t_contents, u.getContents(), v.getContents(), op);
            }

            // =================================================================
            // Accumulate into Z
            using ZScalarType = typename std::conditional_t<
                std::is_same_v<AccumT, NoAccumulate>,
                D3ScalarType,
                decltype(accum(std::declval<WScalarT>(),
                               std::declval<D3ScalarType>()))>;
            std::vector<std::tuple<IndexType,ZScalarType> > z_contents;
            ewise_or_opt_accum_1D(z_contents, w, t_contents, accum);

            // =================================================================
            // Copy Z into the final output considering mask and replace/merge
            write_with_opt_mask_1D(w, z_contents, mask, outp);
        }

        //**********************************************************************
        /// Implementation of 4.3.4.2 eWiseMult: Matrix variant A .* B
        //**********************************************************************
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  //can be BinaryOp, Monoid (not Semiring)
                 typename AMatrixT,
                 typename BMatrixT,
                 typename... CTagsT>
        inline void eWiseMult(
            grb::backend::Matrix<CScalarT, CTagsT...>       &C,
            MaskT                                     const &Mask,
            AccumT                                    const &accum,
            BinaryOpT                                        op,
            AMatrixT                                  const &A,
            BMatrixT                                  const &B,
            OutputControlEnum                                outp)
        {
            GRB_LOG_VERBOSE("C<M,z> := A .* B");
            IndexType num_rows(A.nrows());
            IndexType num_cols(A.ncols());

            // =================================================================
            // Do the basic ewise-and work: T = A .* B
            using D3ScalarType =
                decltype(op(std::declval<typename AMatrixT::ScalarType>(),
                            std::declval<typename BMatrixT::ScalarType>()));
            using TRowType = std::vector<std::tuple<IndexType,D3ScalarType> >;
            LilSparseMatrix<D3ScalarType> T(num_rows, num_cols);

            if ((A.nvals() > 0) && (B.nvals() > 0))
            {
                // create one row of result at a time
                TRowType T_row;
                for (IndexType row_idx = 0; row_idx < num_rows; ++row_idx)
                {
                    if (!B[row_idx].empty() && !A[row_idx].empty())
                    {
                        ewise_and(T_row, A[row_idx], B[row_idx], op);

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
            using ZScalarType = typename std::conditional_t<
                std::is_same_v<AccumT, NoAccumulate>,
                D3ScalarType,
                decltype(accum(std::declval<CScalarT>(),
                               std::declval<D3ScalarType>()))>;
            LilSparseMatrix<ZScalarType> Z(num_rows, num_cols);
            ewise_or_opt_accum(Z, C, T, accum);

            // =================================================================
            // Copy Z into the final output considering mask and replace/merge
            write_with_opt_mask(C, Z, Mask, outp);

        } // ewisemult

        //**********************************************************************
        /// Implementation of 4.3.4.2 eWiseMult: Matrix variant A' .* B
        //**********************************************************************
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  //can be BinaryOp, Monoid (not Semiring)
                 typename AMatrixT,
                 typename BMatrixT,
                 typename... CTagsT>
        inline void eWiseMult(
            grb::backend::Matrix<CScalarT, CTagsT...>       &C,
            MaskT                                     const &Mask,
            AccumT                                    const &accum,
            BinaryOpT                                        op,
            TransposeView<AMatrixT>                   const &AT,
            BMatrixT                                  const &B,
            OutputControlEnum                                outp)
        {
            GRB_LOG_VERBOSE("C<M,z> := A' .* B");
            auto const &A(AT.m_mat);

            AMatrixT Atran(A.ncols(), A.nrows());
            grb::backend::transpose(Atran, NoMask(), NoAccumulate(), A, REPLACE);
            grb::backend::eWiseMult(C, Mask, accum, op, Atran, B, outp);
        } // ewisemult

        //**********************************************************************
        /// Implementation of 4.3.4.2 eWiseMult: Matrix variant A .* B'
        //**********************************************************************
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  //can be BinaryOp, Monoid (not Semiring)
                 typename AMatrixT,
                 typename BMatrixT,
                 typename... CTagsT>
        inline void eWiseMult(
            grb::backend::Matrix<CScalarT, CTagsT...>       &C,
            MaskT                                     const &Mask,
            AccumT                                    const &accum,
            BinaryOpT                                        op,
            AMatrixT                                  const &A,
            TransposeView<BMatrixT>                   const &BT,
            OutputControlEnum                                outp)
        {
            GRB_LOG_VERBOSE("C<M,z> := A .* B'");
            auto const &B(BT.m_mat);

            AMatrixT Btran(B.ncols(), B.nrows());
            grb::backend::transpose(Btran, NoMask(), NoAccumulate(), B, REPLACE);
            grb::backend::eWiseMult(C, Mask, accum, op, A, Btran, outp);
        } // ewisemult

        //**********************************************************************
        /// Implementation of 4.3.4.2 eWiseMult: Matrix variant A' .* B'
        //**********************************************************************
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  //can be BinaryOp, Monoid (not Semiring)
                 typename AMatrixT,
                 typename BMatrixT,
                 typename... CTagsT>
        inline void eWiseMult(
            grb::backend::Matrix<CScalarT, CTagsT...>       &C,
            MaskT                                     const &Mask,
            AccumT                                    const &accum,
            BinaryOpT                                        op,
            TransposeView<AMatrixT>                   const &AT,
            TransposeView<BMatrixT>                   const &BT,
            OutputControlEnum                                outp)
        {
            GRB_LOG_VERBOSE("C<M,z> := A' .* B'");
            auto const &A(AT.m_mat);
            auto const &B(BT.m_mat);
            IndexType num_rows(A.nrows());
            IndexType num_cols(A.ncols());

            // =================================================================
            // Do the basic ewise-and work: T = A' .* B'
            using D3ScalarType =
                decltype(op(std::declval<typename AMatrixT::ScalarType>(),
                            std::declval<typename BMatrixT::ScalarType>()));
            using TRowType = std::vector<std::tuple<IndexType,D3ScalarType> >;
            LilSparseMatrix<D3ScalarType> T(num_cols, num_rows);

            if ((A.nvals() > 0) && (B.nvals() > 0))
            {
                // create one column of result at a time
                TRowType T_col;
                for (IndexType row_idx = 0; row_idx < num_rows; ++row_idx)
                {
                    T_col.clear();
                    if (!B[row_idx].empty()  && !A[row_idx].empty())
                    {
                        ewise_and(T_col, A[row_idx], B[row_idx], op);

                        for (auto && [col_idx, val] : T_col)
                        {
                            T[col_idx].emplace_back(row_idx, val);
                        }
                    }
                }
                T.recomputeNvals();
            }

            // =================================================================
            // Accumulate into Z
            using ZScalarType = typename std::conditional_t<
                std::is_same_v<AccumT, NoAccumulate>,
                D3ScalarType,
                decltype(accum(std::declval<CScalarT>(),
                               std::declval<D3ScalarType>()))>;
            LilSparseMatrix<ZScalarType> Z(num_cols, num_rows);
            ewise_or_opt_accum(Z, C, T, accum);

            // =================================================================
            // Copy Z into the final output considering mask and replace/merge
            write_with_opt_mask(C, Z, Mask, outp);
        } // ewisemult

    } // backend
} // grb
