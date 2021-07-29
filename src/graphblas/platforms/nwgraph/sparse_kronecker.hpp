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
#include <chrono>

#include <graphblas/detail/logging.h>
#include <graphblas/types.hpp>
#include <graphblas/algebra.hpp>

#include "sparse_helpers.hpp"
#include "LilSparseMatrix.hpp"


//****************************************************************************

namespace grb
{
    namespace backend
    {
        //**********************************************************************
        /// Implementation of 4.3.11 kronecker: Matrix kronecker product
        //**********************************************************************
        template<typename CMatrixT,
                 typename MMatrixT,
                 typename AccumT,
                 typename BinaryOpT,
                 typename AMatrixT,
                 typename BMatrixT>
        inline void kronecker(CMatrixT            &C,
                              MMatrixT    const   &M,
                              AccumT      const   &accum,
                              BinaryOpT            op,
                              AMatrixT    const   &A,
                              BMatrixT    const   &B,
                              OutputControlEnum    outp)
        {
            // Dimension checks happen in front end
            IndexType nrow_A(A.nrows());
            //IndexType ncol_A(A.ncols());
            IndexType nrow_B(B.nrows());
            IndexType ncol_B(B.ncols());
            //Frontend checks the dimensions, but use C explicitly
            IndexType nrow_C(C.nrows());
            IndexType ncol_C(C.ncols());

            using AScalarType = typename AMatrixT::ScalarType;
            using BScalarType = typename BMatrixT::ScalarType;
            using CScalarType = typename CMatrixT::ScalarType;

            // =================================================================
            // Do the basic product work with the binaryop.
            using TScalarType = decltype(op(std::declval<AScalarType>(),
                                            std::declval<BScalarType>()));
            LilSparseMatrix<TScalarType> T(nrow_C, ncol_C);

            if ((A.nvals() > 0) && (B.nvals() > 0))
            {
                // create a row of the result at a time
                for (IndexType row_idxA = 0; row_idxA < nrow_A; ++row_idxA)
                {
                    if (A[row_idxA].empty()) continue;

                    for (IndexType row_idxB = 0; row_idxB < nrow_B; ++row_idxB)
                    {
                        if (B[row_idxB].empty()) continue;

                        IndexType T_row_idx(row_idxA*nrow_B + row_idxB);

                        for (auto&& [col_idxA, val_A] : A[row_idxA])
                        {
                            for (auto&& [col_idxB, val_B] : B[row_idxB])
                            {
                                TScalarType T_val(op(val_A, val_B));
                                T[T_row_idx].emplace_back(
                                    (col_idxA*ncol_B + col_idxB), T_val);
                            }
                        }
                    }
                    T.recomputeNvals();
                }
            }

            GRB_LOG_VERBOSE("T: " << T);

            // =================================================================
            // Accumulate into Z
            using ZScalarType = typename std::conditional_t<
                std::is_same_v<AccumT, NoAccumulate>,
                TScalarType,
                decltype(accum(std::declval<CScalarType>(),
                               std::declval<TScalarType>()))>;

            LilSparseMatrix<ZScalarType> Z(nrow_C, ncol_C);

            ewise_or_opt_accum(Z, C, T, accum);

            GRB_LOG_VERBOSE("Z: " << Z);

            // =================================================================
            // Copy Z into the final output considering mask and replace/merge
            write_with_opt_mask(C, Z, M, outp);

        }

        //**********************************************************************
        /// Implementation of 4.3.11 kronecker: Matrix kronecker product
        //**********************************************************************
        template<typename CMatrixT,
                 typename MMatrixT,
                 typename AccumT,
                 typename BinaryOpT,
                 typename AMatrixT,
                 typename BMatrixT>
        inline void kronecker(CMatrixT                        &C,
                              MMatrixT                const   &M,
                              AccumT                  const   &accum,
                              BinaryOpT                        op,
                              TransposeView<AMatrixT> const   &AT,
                              BMatrixT                const   &B,
                              OutputControlEnum                outp)
        {
            auto const &A(AT.m_mat);

            // Dimension checks happen in front end
            IndexType nrow_A(A.nrows());
            //IndexType ncol_A(A.ncols());
            IndexType nrow_B(B.nrows());
            IndexType ncol_B(B.ncols());
            //Frontend checks the dimensions, but use C explicitly
            IndexType nrow_C(C.nrows());
            IndexType ncol_C(C.ncols());

            using AScalarType = typename AMatrixT::ScalarType;
            using BScalarType = typename BMatrixT::ScalarType;
            using CScalarType = typename CMatrixT::ScalarType;

            // =================================================================
            // Do the basic product work with the binaryop.
            using TScalarType = decltype(op(std::declval<AScalarType>(),
                                            std::declval<BScalarType>()));
            LilSparseMatrix<TScalarType> T(nrow_C, ncol_C);

            if ((A.nvals() > 0) && (B.nvals() > 0))
            {
                // create a row of the result at a time
                for (IndexType row_idxA = 0; row_idxA < nrow_A; ++row_idxA)
                {
                    if (A[row_idxA].empty()) continue;

                    for (IndexType row_idxB = 0; row_idxB < nrow_B; ++row_idxB)
                    {
                        if (B[row_idxB].empty()) continue;

                        for (auto&& [col_idxA, val_A] : A[row_idxA])
                        {
                            IndexType T_row_idx(col_idxA*nrow_B + row_idxB);

                            for (auto&& [col_idxB, val_B] : B[row_idxB])
                            {
                                TScalarType T_val(op(val_A, val_B));
                                T[T_row_idx].emplace_back(
                                    (row_idxA*ncol_B + col_idxB), T_val);
                            }
                        }
                    }
                    T.recomputeNvals();
                }
            }

            // =================================================================
            // Accumulate into Z
            using ZScalarType = typename std::conditional_t<
                std::is_same_v<AccumT, NoAccumulate>,
                TScalarType,
                decltype(accum(std::declval<CScalarType>(),
                               std::declval<TScalarType>()))>;

            LilSparseMatrix<ZScalarType> Z(nrow_C, ncol_C);

            ewise_or_opt_accum(Z, C, T, accum);

            GRB_LOG_VERBOSE("Z: " << Z);

            // =================================================================
            // Copy Z into the final output considering mask and replace/merge
            write_with_opt_mask(C, Z, M, outp);

        }

        //**********************************************************************
        /// Implementation of 4.3.11 kronecker: Matrix kronecker product
        //**********************************************************************
        template<typename CMatrixT,
                 typename MMatrixT,
                 typename AccumT,
                 typename BinaryOpT,
                 typename AMatrixT,
                 typename BMatrixT>
        inline void kronecker(CMatrixT                        &C,
                              MMatrixT                const   &M,
                              AccumT                  const   &accum,
                              BinaryOpT                        op,
                              AMatrixT                const   &A,
                              TransposeView<BMatrixT> const   &BT,
                              OutputControlEnum                outp)
        {
            auto const &B(BT.m_mat);

            // Dimension checks happen in front end
            IndexType nrow_A(A.nrows());
            //IndexType ncol_A(A.ncols());
            IndexType nrow_B(B.nrows());
            IndexType ncol_B(B.ncols());
            //Frontend checks the dimensions, but use C explicitly
            IndexType nrow_C(C.nrows());
            IndexType ncol_C(C.ncols());

            using AScalarType = typename AMatrixT::ScalarType;
            using BScalarType = typename BMatrixT::ScalarType;
            using CScalarType = typename CMatrixT::ScalarType;

            // =================================================================
            // Do the basic product work with the binaryop.
            using TScalarType = decltype(op(std::declval<AScalarType>(),
                                            std::declval<BScalarType>()));
            LilSparseMatrix<TScalarType> T(nrow_C, ncol_C);

            if ((A.nvals() > 0) && (B.nvals() > 0))
            {
                // create a row of the result at a time
                for (IndexType row_idxA = 0; row_idxA < nrow_A; ++row_idxA)
                {
                    if (A[row_idxA].empty()) continue;

                    for (auto&& [col_idxA, val_A] : A[row_idxA])
                    {
                        for (IndexType row_idxB = 0; row_idxB < nrow_B; ++row_idxB)
                        {
                            if (B[row_idxB].empty()) continue;

                            IndexType T_col_idx(col_idxA*nrow_B + row_idxB);

                            for (auto&& [col_idxB, val_B] : B[row_idxB])
                            {
                                TScalarType T_val(op(val_A, val_B));
                                IndexType T_row_idx(row_idxA*ncol_B + col_idxB);
                                T[T_row_idx].emplace_back(T_col_idx, T_val);
                            }
                        }
                    }
                    T.recomputeNvals();
                }
            }

            // =================================================================
            // Accumulate into Z
            using ZScalarType = typename std::conditional_t<
                std::is_same_v<AccumT, NoAccumulate>,
                TScalarType,
                decltype(accum(std::declval<CScalarType>(),
                               std::declval<TScalarType>()))>;

            LilSparseMatrix<ZScalarType> Z(nrow_C, ncol_C);

            ewise_or_opt_accum(Z, C, T, accum);

            GRB_LOG_VERBOSE("Z: " << Z);

            // =================================================================
            // Copy Z into the final output considering mask and replace/merge
            write_with_opt_mask(C, Z, M, outp);

        }

        //**********************************************************************
        /// Implementation of 4.3.11 kronecker: Matrix kronecker product
        //**********************************************************************
        template<typename CMatrixT,
                 typename MMatrixT,
                 typename AccumT,
                 typename BinaryOpT,
                 typename AMatrixT,
                 typename BMatrixT>
        inline void kronecker(CMatrixT                        &C,
                              MMatrixT                const   &M,
                              AccumT                  const   &accum,
                              BinaryOpT                        op,
                              TransposeView<AMatrixT> const   &AT,
                              TransposeView<BMatrixT> const   &BT,
                              OutputControlEnum                outp)
        {
            auto const &A(AT.m_mat);
            auto const &B(BT.m_mat);

            // Dimension checks happen in front end
            IndexType nrow_A(A.nrows());
            //IndexType ncol_A(A.ncols());
            IndexType nrow_B(B.nrows());
            IndexType ncol_B(B.ncols());
            //Frontend checks the dimensions, but use C explicitly
            IndexType nrow_C(C.nrows());
            IndexType ncol_C(C.ncols());

            using AScalarType = typename AMatrixT::ScalarType;
            using BScalarType = typename BMatrixT::ScalarType;
            using CScalarType = typename CMatrixT::ScalarType;

            // =================================================================
            // Do the basic product work with the binaryop.
            using TScalarType = decltype(op(std::declval<AScalarType>(),
                                            std::declval<BScalarType>()));
            LilSparseMatrix<TScalarType> T(nrow_C, ncol_C);

            if ((A.nvals() > 0) && (B.nvals() > 0))
            {
                // create a row of the result at a time
                for (IndexType row_idxA = 0; row_idxA < nrow_A; ++row_idxA)
                {
                    if (A[row_idxA].empty()) continue;

                    for (IndexType row_idxB = 0; row_idxB < nrow_B; ++row_idxB)
                    {
                        if (B[row_idxB].empty()) continue;
                        IndexType T_col_idx(row_idxA*nrow_B + row_idxB);

                        for (auto&& [col_idxA, val_A] : A[row_idxA])
                        {
                            for (auto&& [col_idxB, val_B] : B[row_idxB])
                            {
                                TScalarType T_val(op(val_A, val_B));
                                IndexType T_row_idx(col_idxA*ncol_B + col_idxB);
                                T[T_row_idx].emplace_back(T_col_idx, T_val);
                            }
                        }
                    }
                    T.recomputeNvals();
                }
            }

            // =================================================================
            // Accumulate into Z
            using ZScalarType = typename std::conditional_t<
                std::is_same_v<AccumT, NoAccumulate>,
                TScalarType,
                decltype(accum(std::declval<CScalarType>(),
                               std::declval<TScalarType>()))>;

            LilSparseMatrix<ZScalarType> Z(nrow_C, ncol_C);

            ewise_or_opt_accum(Z, C, T, accum);

            GRB_LOG_VERBOSE("Z: " << Z);

            // =================================================================
            // Copy Z into the final output considering mask and replace/merge
            write_with_opt_mask(C, Z, M, outp);

        }

    } // backend
} // grb
