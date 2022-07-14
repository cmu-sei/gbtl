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
        // Compute C' = (A'*B')' = B*A, assuming C, B, and A are unique
        template<typename CScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void ATBT_NoMask_NoAccum_kernel(
            LilSparseMatrix<CScalarT>       &C,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B)
        {
            C.clear();
            using TScalarType = typename SemiringT::result_type;
            typename LilSparseMatrix<TScalarType>::RowType T_row;

            // compute transpose T = B +.* A (one row at a time and transpose)
            for (IndexType i = 0; i < B.nrows(); ++i)
            {
                // this part is same as sparse_mxm_NoMask_NoAccum_AB
                T_row.clear();
                for (auto const &Bi_elt : B[i])
                {
                    IndexType    k(std::get<0>(Bi_elt));
                    AScalarT  b_ik(std::get<1>(Bi_elt));

                    if (A[k].empty()) continue;

                    // T[i] += (b_ik*A[k])  // must reduce in D3
                    axpy(T_row, semiring, b_ik, A[k]);
                }

                //C.setCol(i, T_row); // this is a push_back form of setCol
                for (auto const &t : T_row)
                {
                    IndexType j(std::get<0>(t));

                    C[j].emplace_back(i, static_cast<CScalarT>(std::get<1>(t)));
                }
            }
            C.recomputeNvals();
        }

        //**********************************************************************
        //**********************************************************************
        //**********************************************************************

        //**********************************************************************
        template<typename CScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_NoMask_NoAccum_ATBT(
            LilSparseMatrix<CScalarT>       &C,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B)
        {
            // C = (A +.* B')
            // short circuit conditions
            if ((A.nvals() == 0) || (B.nvals() == 0))
            {
                C.clear();
                return;
            }

            // =================================================================
            using TScalarType = typename SemiringT::result_type;
            typename LilSparseMatrix<TScalarType>::RowType T_row;

            if (((void*)&C == (void*)&A) || ((void*)&C == (void*)&B))
            {
                LilSparseMatrix<CScalarT> Ctmp(C.nrows(), C.ncols());
                ATBT_NoMask_NoAccum_kernel(Ctmp, semiring, A, B);

                for (IndexType i = 0; i < C.nrows(); ++i)
                {
                    C.setRow(i, Ctmp[i]);
                }
            }
            else
            {
                ATBT_NoMask_NoAccum_kernel(C, semiring, A, B);
            }
            C.recomputeNvals();

            GRB_LOG_VERBOSE("C: " << C);
        }

        //**********************************************************************
        template<typename CScalarT,
                 typename AccumT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_NoMask_Accum_ATBT(
            LilSparseMatrix<CScalarT>       &C,
            AccumT                    const &accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B)
        {
            // C = C + (A +.* B')
            // short circuit conditions
            if ((A.nvals() == 0) || (B.nvals() == 0))
            {
                return; // do nothing
            }

            // =================================================================
            using TScalarType = typename SemiringT::result_type;

            if (((void*)&C == (void*)&B) || ((void*)&C == (void*)&A))
            {
                // create temporary to prevent overwrite of inputs
                // T = A' +.* B'
                LilSparseMatrix<TScalarType> T(C.nrows(), C.ncols());
                ATBT_NoMask_NoAccum_kernel(T, semiring, A, B);
                for (IndexType i = 0; i < C.nrows(); ++i)
                {
                    // C[i] = C[i] + T[i]
                    C.mergeRow(i, T[i], accum);
                }
            }
            else
            {
                typename LilSparseMatrix<TScalarType>::RowType T_row;
                LilSparseMatrix<TScalarType> T(C.nrows(), C.ncols());
                ATBT_NoMask_NoAccum_kernel(T, semiring, A, B);

                // accumulate
                for (IndexType i = 0; i < C.nrows(); ++i)
                {
                    // C[i] = C[i] + T[i]
                    C.mergeRow(i, T[i], accum);
                }
            }
        }

        //**********************************************************************
        template<typename CScalarT,
                 typename MScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_Mask_NoAccum_ATBT(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            bool                             structure_flag,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            OutputControlEnum                outp)
        {
            // C<M,z> = A +.* B
            //        =               [M .* (A' +.* B')], z = "replace"
            //        = [!M .* C]  U  [M .* (A' +.* B')], z = "merge"
            // short circuit conditions
            if ((outp == REPLACE) &&
                ((A.nvals() == 0) || (B.nvals() == 0) || (M.nvals() == 0)))
            {
                C.clear();
                return;
            }
            else if ((outp == MERGE) && (M.nvals() == 0))
            {
                return; // do nothing
            }

            // =================================================================
            using TScalarType = typename SemiringT::result_type;
            typename LilSparseMatrix<TScalarType>::RowType T_row;
            LilSparseMatrix<TScalarType> T(C.nrows(), C.ncols());

            // compute transpose T = B +.* A (one row at a time and transpose)
            for (IndexType i = 0; i < B.nrows(); ++i)
            {
                // this part is same as sparse_mxm_NoMask_NoAccum_AB swapping
                // A and B and computing the transpose of C.
                T_row.clear();

                for (auto const &Bi_elt : B[i])
                {
                    IndexType    k(std::get<0>(Bi_elt));
                    AScalarT  b_ik(std::get<1>(Bi_elt));

                    if (A[k].empty()) continue;

                    // T[i] += (b_ik*A[k])  // must reduce in D3
                    axpy(T_row, semiring, b_ik, A[k]);
                }

                // Transpose the result
                //T.setCol(i, T_row); // this is a push_back form of setCol
                for (auto const &t : T_row)
                {
                    IndexType j(std::get<0>(t));
                    T[j].push_back(std::make_pair(i, std::get<1>(t)));
                }
            }

            typename LilSparseMatrix<CScalarT>::RowType Z_row, C_row;
            for (IndexType i = 0; i < C.nrows(); ++i)
            {
                // Z = M[i] .* T[i]
                Z_row.clear();
                auto m_it = M[i].begin();
                for (auto t_ij : T[i])
                {
                    IndexType j(std::get<0>(t_ij));
                    if (advance_and_check_mask_iterator(
                            m_it, M[i].end(), structure_flag, j))
                    {
                        Z_row.push_back(
                            std::make_tuple(
                                j, static_cast<CScalarT>(std::get<1>(t_ij))));
                    }
                }

                if (outp == REPLACE)
                {
                    C.setRow(i, Z_row);
                }
                else /* merge */
                {
                    // C[i] = !M[i].*C[i] U Z_row
                    C_row.clear();
                    masked_merge(C_row, M[i], structure_flag, false, C[i], Z_row);
                    C.setRow(i, C_row);
                }
            }
        }

        //**********************************************************************
        template<typename CScalarT,
                 typename MScalarT,
                 typename AccumT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_Mask_Accum_ATBT(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            bool                             structure_flag,
            AccumT                    const &accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            OutputControlEnum                outp)
        {
            // C<M,z> = C + (A' +.* B')
            //        =               [M .* [C + (A' +.* B')]], z = "replace"
            //        = [!M .* C]  U  [M .* [C + (A' +.* B')]], z = "merge"
            // short circuit conditions
            if ((outp == REPLACE) && (M.nvals() == 0))
            {
                C.clear();
                return;
            }
            else if ((outp == MERGE) && (M.nvals() == 0))
            {
                return; // do nothing
            }

            // =================================================================
            using TScalarType = typename SemiringT::result_type;
            using ZScalarType = decltype(accum(std::declval<CScalarT>(),
                                               std::declval<TScalarType>()));
            typename LilSparseMatrix<TScalarType>::RowType T_row;
            typename LilSparseMatrix<ZScalarType>::RowType Z_row;
            typename LilSparseMatrix<CScalarT>::RowType    C_row;
            LilSparseMatrix<TScalarType> T(C.nrows(), C.ncols());

            // compute transpose T' = B +.* A (one row at a time and transpose)
            for (IndexType i = 0; i < B.nrows(); ++i)
            {
                // this part is same as sparse_mxm_NoMask_NoAccum_AB swapping
                // A and B and computing the transpose of C.
                T_row.clear();

                for (auto const &Bi_elt : B[i])
                {
                    IndexType    k(std::get<0>(Bi_elt));
                    AScalarT  b_ik(std::get<1>(Bi_elt));

                    if (A[k].empty()) continue;

                    // T[i] += (b_ik*A[k])  // must reduce in D3
                    axpy(T_row, semiring, b_ik, A[k]);
                }

                // Transpose the result
                //T.setCol(i, T_row); // this is a push_back form of setCol
                for (auto const &t : T_row)
                {
                    IndexType j(std::get<0>(t));
                    T[j].push_back(std::make_pair(i, std::get<1>(t)));
                }
            }

            bool const complement_flag = false;

            for (IndexType i = 0; i < C.nrows(); ++i)
            {
                // T_row = M[i] .* T[i]
                T_row.clear();
                auto m_it = M[i].begin();
                for (auto t_ij : T[i])
                {
                    IndexType j(std::get<0>(t_ij));
                    if (advance_and_check_mask_iterator(
                            m_it, M[i].end(), structure_flag, j))
                    {
                        T_row.push_back(t_ij);
                    }
                }

                // Z[i] = (M[i] .* C[i]) + T[i]
                Z_row.clear();
                masked_accum(Z_row,
                             M[i], structure_flag, complement_flag,
                             accum, C[i], T_row);

                if (outp == MERGE)
                {
                    // C[i]  = [!M .* C]  U  Z[i]
                    C_row.clear();
                    masked_merge(C_row,
                                 M[i], structure_flag, complement_flag,
                                 C[i], Z_row);
                    C.setRow(i, C_row);  // set even if it is empty.
                }
                else // z = replace
                {
                    // C[i] = Z[i]
                    C.setRow(i, Z_row);
                }
            }
        }

        //**********************************************************************
        template<typename CScalarT,
                 typename MScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_CompMask_NoAccum_ATBT(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            bool                             structure_flag,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            OutputControlEnum                outp)
        {
            // C<M,z> = A +.* B
            //        =               [M .* (A' +.* B')], z = "replace"
            //        = [!M .* C]  U  [M .* (A' +.* B')], z = "merge"
            // short circuit conditions
            if ((outp == REPLACE) && ((A.nvals() == 0) || (B.nvals() == 0)))
            {
                C.clear();
                return;
            }

            // =================================================================
            using TScalarType = typename SemiringT::result_type;
            typename LilSparseMatrix<TScalarType>::RowType T_row;
            LilSparseMatrix<TScalarType> T(C.nrows(), C.ncols());

            // compute transpose T = B +.* A (one row at a time and transpose)
            for (IndexType i = 0; i < B.nrows(); ++i)
            {
                // this part is same as sparse_mxm_NoMask_NoAccum_AB swapping
                // A and B and computing the transpose of C.
                T_row.clear();

                for (auto const &Bi_elt : B[i])
                {
                    IndexType    k(std::get<0>(Bi_elt));
                    AScalarT  b_ik(std::get<1>(Bi_elt));

                    if (A[k].empty()) continue;

                    // T[i] += (b_ik*A[k])  // must reduce in D3
                    axpy(T_row, semiring, b_ik, A[k]);
                }

                // Transpose the result
                //T.setCol(i, T_row); // this is a push_back form of setCol
                for (auto const &t : T_row)
                {
                    IndexType j(std::get<0>(t));
                    T[j].push_back(std::make_pair(i, std::get<1>(t)));
                }
            }

            typename LilSparseMatrix<CScalarT>::RowType Z_row, C_row;
            for (IndexType i = 0; i < C.nrows(); ++i)
            {
                // Z = !M[i] .* T[i]
                Z_row.clear();
                auto m_it = M[i].begin();
                for (auto t_ij : T[i])
                {
                    IndexType j(std::get<0>(t_ij));
                    if (!advance_and_check_mask_iterator(
                            m_it, M[i].end(), structure_flag, j))
                    {
                        Z_row.push_back(
                            std::make_tuple(
                                j, static_cast<CScalarT>(std::get<1>(t_ij))));
                    }
                }

                if (outp == REPLACE)
                {
                    C.setRow(i, Z_row);
                }
                else /* merge */
                {
                    // C[i] = M[i].*C[i] U Z_row
                    C_row.clear();
                    masked_merge(C_row,
                                 M[i], structure_flag, true,
                                 C[i], Z_row);
                    C.setRow(i, C_row);
                }
            }
        }

        //**********************************************************************
        template<typename CScalarT,
                 typename MScalarT,
                 typename AccumT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_CompMask_Accum_ATBT(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            bool                             structure_flag,
            AccumT                    const &accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            OutputControlEnum                outp)
        {
            // C<!M,z> = C + (A' +.* B')
            //         =              [!M .* [C + (A' +.* B')]], z = "replace"
            //         = [M .* C]  U  [!M .* [C + (A' +.* B')]], z = "merge"
            // short circuit conditions
            if ((outp == REPLACE) &&
                (M.nvals() == 0) &&
                ((A.nvals() == 0) || (B.nvals() == 0)))
            {
                return; // do nothing
            }

            // =================================================================
            using TScalarType = typename SemiringT::result_type;
            using ZScalarType = decltype(accum(std::declval<CScalarT>(),
                                               std::declval<TScalarType>()));
            typename LilSparseMatrix<TScalarType>::RowType T_row;
            typename LilSparseMatrix<ZScalarType>::RowType  Z_row;
            typename LilSparseMatrix<CScalarT>::RowType C_row;
            LilSparseMatrix<TScalarType> T(C.nrows(), C.ncols());

            // compute transpose T' = B +.* A (one row at a time and transpose)
            for (IndexType i = 0; i < B.nrows(); ++i)
            {
                // this part is same as sparse_mxm_NoMask_NoAccum_AB swapping
                // A and B and computing the transpose of C.
                T_row.clear();

                for (auto const &Bi_elt : B[i])
                {
                    IndexType    k(std::get<0>(Bi_elt));
                    AScalarT  b_ik(std::get<1>(Bi_elt));

                    if (A[k].empty()) continue;

                    // T[i] += (b_ik*A[k])  // must reduce in D3
                    axpy(T_row, semiring, b_ik, A[k]);
                }

                // Transpose the result
                //T.setCol(i, T_row); // this is a push_back form of setCol
                for (auto const &t : T_row)
                {
                    IndexType j(std::get<0>(t));
                    T[j].push_back(std::make_pair(i, std::get<1>(t)));
                }
            }

            bool const complement_flag = true;

            for (IndexType i = 0; i < C.nrows(); ++i)
            {
                // T_row = M[i] .* T[i]
                T_row.clear();
                auto m_it = M[i].begin();
                for (auto t_ij : T[i])
                {
                    IndexType j(std::get<0>(t_ij));
                    if (!advance_and_check_mask_iterator(
                            m_it, M[i].end(), structure_flag, j))
                    {
                        T_row.push_back(t_ij);
                    }
                }

                // Z[i] = (M[i] .* C[i]) + T[i]
                Z_row.clear();
                masked_accum(Z_row,
                             M[i], structure_flag, complement_flag,
                             accum, C[i], T_row);

                if (outp == REPLACE)
                {
                    // C[i] = Z[i]
                    C.setRow(i, Z_row);
                }
                else
                {
                    // C[i]  = [!M .* C]  U  Z[i]
                    C_row.clear();
                    masked_merge(C_row,
                                 M[i], structure_flag, complement_flag,
                                 C[i], Z_row);
                    C.setRow(i, C_row);  // set even if it is empty.
                }
            }
        }

    } // backend
} // grb
