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
        /// Implementation of 4.3.1 mxm: Matrix-matrix multiply: A +.* B
        //**********************************************************************
        template<typename CMatrixT,
                 typename MMatrixT,
                 typename AccumT,
                 typename SemiringT,
                 typename AMatrixT,
                 typename BMatrixT>
        inline void mxm(CMatrixT            &C,
                        MMatrixT    const   &M,
                        AccumT      const   &accum,
                        SemiringT            op,
                        AMatrixT    const   &A,
                        BMatrixT    const   &B,
                        OutputControlEnum    outp)
        {
            GRB_LOG_VERBOSE("C<M,z> := (A*B)");

            using CScalarType = typename CMatrixT::ScalarType;

            // =================================================================
            // Do the axpy work with the semiring.
            using TScalarType = typename SemiringT::result_type;
            LilSparseMatrix<TScalarType> T(C.nrows(), C.ncols());

            typename LilSparseMatrix<TScalarType>::RowType T_row;

            for (IndexType i = 0; i < A.nrows(); ++i)
            {
                for (auto&& [k, a_ik] : A[i])
                {
                    if (B[k].empty()) continue;

                    // T[i] += (a_ik*B[k])  // must reduce in D3
                    axpy(T[i], op, a_ik, B[k]);
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

            LilSparseMatrix<ZScalarType> Z(C.nrows(), C.ncols());

            ewise_or_opt_accum(Z, C, T, accum);

            GRB_LOG_VERBOSE("Z: " << Z);

            // =================================================================
            // Copy Z into the final output considering mask and replace/merge
            write_with_opt_mask(C, Z, M, outp);
        } // mxm

        //**********************************************************************
        /// Implementation of 4.3.1 mxm: Matrix-matrix multiply: A' +.* B
        //**********************************************************************
        template<typename CMatrixT,
                 typename MMatrixT,
                 typename AccumT,
                 typename SemiringT,
                 typename AMatrixT,
                 typename BMatrixT>
        inline void mxm(CMatrixT                        &C,
                        MMatrixT                const   &M,
                        AccumT                  const   &accum,
                        SemiringT                        op,
                        TransposeView<AMatrixT> const   &AT,
                        BMatrixT                const   &B,
                        OutputControlEnum                outp)
        {
            GRB_LOG_VERBOSE("C<M,z> := (A'*B)");
            auto const &A(AT.m_mat);

            using CScalarType = typename CMatrixT::ScalarType;

            // =================================================================
            // Do the basic axpy work with the semiring.
            using TScalarType = typename SemiringT::result_type;
            LilSparseMatrix<TScalarType> T(C.nrows(), C.ncols());

            for (IndexType k = 0; k < A.nrows(); ++k)
            {
                if (A[k].empty() || B[k].empty()) continue;

                for (auto&& [i, a_ki] : A[k])
                {
                    // T[i] += (a_ki*B[k])  // must reduce in D3, hence T.
                    axpy(T[i], op, a_ki, B[k]);
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

            LilSparseMatrix<ZScalarType> Z(C.nrows(), C.ncols());

            ewise_or_opt_accum(Z, C, T, accum);

            GRB_LOG_VERBOSE("Z: " << Z);

            // =================================================================
            // Copy Z into the final output considering mask and replace/merge
            write_with_opt_mask(C, Z, M, outp);
        } // mxm

        //**********************************************************************
        /// Implementation of 4.3.1 mxm: Matrix-matrix multiply: A +.* B'
        //**********************************************************************
        template<typename CMatrixT,
                 typename MMatrixT,
                 typename AccumT,
                 typename SemiringT,
                 typename AMatrixT,
                 typename BMatrixT>
        inline void mxm(CMatrixT                        &C,
                        MMatrixT                const   &M,
                        AccumT                  const   &accum,
                        SemiringT                        op,
                        AMatrixT                const   &A,
                        TransposeView<BMatrixT> const   &BT,
                        OutputControlEnum                outp)
        {
            GRB_LOG_VERBOSE("C<M,z> := (A*B')");
            auto const &B(BT.m_mat);

            // Dimension checks happen in front end
            IndexType nrow_A(A.nrows());
            IndexType nrow_B(B.nrows());

            using CScalarType = typename CMatrixT::ScalarType;

            // =================================================================
            // Do the basic dot-product work with the semiring.
            using TScalarType = typename SemiringT::result_type;
            LilSparseMatrix<TScalarType> T(C.nrows(), C.ncols());

            // Build this completely based on the semiring
            if ((A.nvals() > 0) && (B.nvals() > 0))
            {
                // create a row of result at a time
                for (IndexType row_idx = 0; row_idx < nrow_A; ++row_idx)
                {
                    if (!A[row_idx].empty())
                    {
                        for (IndexType col_idx = 0; col_idx < nrow_B; ++col_idx)
                        {
                            if (!B[col_idx].empty())
                            {
                                TScalarType T_val;
                                if (dot(T_val, A[row_idx], B[col_idx], op))
                                {
                                    T[row_idx].emplace_back(col_idx, T_val);
                                }
                            }
                        }
                    }
                }
                T.recomputeNvals();
            }

            GRB_LOG_VERBOSE("T: " << T);

            // =================================================================
            // Accumulate into Z
            using ZScalarType = typename std::conditional_t<
                std::is_same_v<AccumT, NoAccumulate>,
                TScalarType,
                decltype(accum(std::declval<CScalarType>(),
                               std::declval<TScalarType>()))>;

            LilSparseMatrix<ZScalarType> Z(C.nrows(), C.ncols());

            ewise_or_opt_accum(Z, C, T, accum);

            GRB_LOG_VERBOSE("Z: " << Z);

            // =================================================================
            // Copy Z into the final output considering mask and replace/merge
            write_with_opt_mask(C, Z, M, outp);
        } // mxm

        //**********************************************************************
        /// Implementation of 4.3.1 mxm: Matrix-matrix multiply: A' +.* B'
        //**********************************************************************
        template<typename CMatrixT,
                 typename MMatrixT,
                 typename AccumT,
                 typename SemiringT,
                 typename AMatrixT,
                 typename BMatrixT>
        inline void mxm(CMatrixT                        &C,
                        MMatrixT                const   &M,
                        AccumT                  const   &accum,
                        SemiringT                        op,
                        TransposeView<AMatrixT> const   &AT,
                        TransposeView<BMatrixT> const   &BT,
                        OutputControlEnum                outp)
        {
            GRB_LOG_VERBOSE("C<M,z> := (A'*B')");
            auto const &A(AT.m_mat);
            auto const &B(BT.m_mat);

            using CScalarType = typename CMatrixT::ScalarType;

            // =================================================================
            // Do the basic dot-product work with the semiring.
            using TScalarType = typename SemiringT::result_type;
            LilSparseMatrix<TScalarType> T(C.nrows(), C.ncols());
            typename LilSparseMatrix<TScalarType>::RowType T_row;

            // compute transpose T = B +.* A (one row at a time and transpose)
            for (IndexType i = 0; i < B.nrows(); ++i)
            {
                // this part is same as sparse_mxm_NoMask_NoAccum_AB
                T_row.clear();
                for (auto&& [k, b_ik] : B[i])
                {
                    if (A[k].empty()) continue;

                    // T[i] += (b_ik*A[k])  // must reduce in D3
                    axpy(T_row, op, b_ik, A[k]);
                }

                //C.setCol(i, T_row); // this is a push_back form of setCol
                for (auto const &t : T_row)
                {
                    IndexType j(std::get<0>(t));

                    T[j].emplace_back(i, static_cast<CScalarType>(std::get<1>(t)));
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

            LilSparseMatrix<ZScalarType> Z(C.nrows(), C.ncols());

            ewise_or_opt_accum(Z, C, T, accum);

            GRB_LOG_VERBOSE("Z: " << Z);

            // =================================================================
            // Copy Z into the final output considering mask and replace/merge
            write_with_opt_mask(C, Z, M, outp);
        } // mxm

    } // backend
} // grb
