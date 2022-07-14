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
        //**********************************************************************

        //**********************************************************************
        // Perform C = A +.* B' where C, A, and B are all unique
        template<typename CScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void ABT_NoMask_NoAccum_kernel(
            LilSparseMatrix<CScalarT>       &C,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B)
        {
            using TScalarType = typename SemiringT::result_type;
            typename LilSparseMatrix<CScalarT>::RowType C_row;

            for (IndexType i = 0; i < A.nrows(); ++i)
            {
                C_row.clear();
                if (A[i].empty()) continue;

                // fill row i of T
                for (IndexType j = 0; j < B.nrows(); ++j)
                {
                    if (B[j].empty()) continue;

                    TScalarType t_ij;

                    // Perform the dot product
                    // C[i][j] = T_ij = (CScalarT) (A[i] . B[j])
                    if (dot(t_ij, A[i], B[j], semiring))
                    {
                        C_row.emplace_back(j, static_cast<CScalarT>(t_ij));
                    }
                }

                C.setRow(i, C_row);  // set even if it is empty.
            }
        }

        //**********************************************************************
        // Perform C = C + AB', where C, A, and B must all be unique
        template<typename CScalarT,
                 typename AccumT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void ABT_NoMask_Accum_kernel(
            LilSparseMatrix<CScalarT>       &C,
            AccumT                    const &accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B)
        {
            using TScalarType = typename SemiringT::result_type;
            std::vector<std::tuple<IndexType,TScalarType> > T_row;

            for (IndexType i = 0; i < A.nrows(); ++i)
            {
                if (A[i].empty()) continue;

                T_row.clear();

                // Compute row i of T
                // T[i] = (CScalarT) (A[i] *.+ B')
                for (IndexType j = 0; j < B.nrows(); ++j)
                {
                    if (B[j].empty()) continue;

                    TScalarType t_ij;

                    // Perform the dot product
                    // T[i][j] = (CScalarT) (A[i] . B[j])
                    if (dot(t_ij, A[i], B[j], semiring))
                    {
                        T_row.emplace_back(j, t_ij);
                    }
                }

                if (!T_row.empty())
                {
                    // C[i] = C[i] + T[i]
                    C.mergeRow(i, T_row, accum);
                }
            }
        }

        //**********************************************************************
        // Perform C<M,z> = A +.* B' where A, B, M, and C are all unique
        template<typename CScalarT,
                 typename MScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void ABT_Mask_NoAccum_kernel(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            bool                             structure_flag,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            OutputControlEnum                outp)
        {
            using TScalarType = typename SemiringT::result_type;
            typename LilSparseMatrix<TScalarType>::RowType T_row;
            typename LilSparseMatrix<CScalarT>::RowType C_row;

            for (IndexType i = 0; i < A.nrows(); ++i)
            {
                bool const complement_flag = false;
                T_row.clear();

                // T[i] = M[i] .* (A[i] dot B[j])
                if (!A[i].empty() && !M[i].empty())
                {
                    auto M_iter(M[i].begin());
                    for (IndexType j = 0; j < B.nrows(); ++j)
                    {
                        if (B[j].empty() ||
                            !advance_and_check_mask_iterator(
                                M_iter, M[i].end(), structure_flag, j))
                            continue;

                        // Perform the dot product
                        TScalarType t_ij;
                        if (dot(t_ij, A[i], B[j], semiring))
                        {
                            T_row.emplace_back(j, t_ij);
                        }
                    }
                }

                if (outp == REPLACE)
                {
                    // C[i] = T[i], z = "replace"
                    C.setRow(i, T_row);
                }
                else /* merge */
                {
                    // C[i] = [!M .* C]  U  T[i], z = "merge"
                    C_row.clear();
                    masked_merge(C_row,
                                 M[i], structure_flag, complement_flag,
                                 C[i], T_row);
                    C.setRow(i, C_row);
                }
            }
        }

        //**********************************************************************
        // Perform C<M,z> = C + (A +.* B') where A, B, M, and C are all unique
        template<typename CScalarT,
                 typename MScalarT,
                 typename AccumT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void ABT_Mask_Accum_kernel(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            bool                             structure_flag,
            AccumT                    const &accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            OutputControlEnum                outp)
        {
            using TScalarType = typename SemiringT::result_type;
            using ZScalarType = decltype(accum(std::declval<CScalarT>(),
                                               std::declval<TScalarType>()));
            typename LilSparseMatrix<TScalarType>::RowType  T_row;
            typename LilSparseMatrix<ZScalarType>::RowType  Z_row;
            typename LilSparseMatrix<CScalarT>::RowType     C_row;

            for (IndexType i = 0; i < A.nrows(); ++i)
            {
                bool const complement_flag = false;  /// @todo constexpr?
                T_row.clear();

                if (!A[i].empty() && !M[i].empty())
                {
                    auto m_it(M[i].begin());

                    // Compute: T[i] = M[i] .* {C[i] + (A +.* B')[i]}
                    for (IndexType j = 0; j < B.nrows(); ++j)
                    {
                        // See if B[j] has data and M[i] allows write.
                        if (B[j].empty() ||
                            !advance_and_check_mask_iterator(
                                m_it, M[i].end(), structure_flag, j))
                        {
                            continue;
                        }

                        // Perform the dot product and accum if necessary
                        TScalarType t_ij;
                        if (dot(t_ij, A[i], B[j], semiring))
                        {
                            T_row.emplace_back(j, t_ij);
                        }
                    }
                }

                // Z[i] = (M .* C) + T[i]
                Z_row.clear();
                masked_accum(Z_row,
                             M[i], structure_flag, complement_flag,
                             accum, C[i], T_row);

                if (outp == REPLACE)
                {
                    C.setRow(i, Z_row);
                }
                else /* merge */
                {
                    // C[i] := (!M[i] .* C[i])  U  Z[i]
                    C_row.clear();
                    masked_merge(C_row,
                                 M[i], structure_flag, complement_flag,
                                 C[i], Z_row);
                    C.setRow(i, C_row);  // set even if it is empty.
                }
            }
        }

        //**********************************************************************
        // Perform C<!M,z> = (A +.* B') where A, B, M, and C are all unique
        template<typename CScalarT,
                 typename MScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void ABT_CompMask_NoAccum_kernel(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            bool                             structure_flag,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            OutputControlEnum                outp)
        {
            using TScalarType = typename SemiringT::result_type;
            typename LilSparseMatrix<TScalarType>::RowType T_row;
            typename LilSparseMatrix<CScalarT>::RowType    C_row;

            for (IndexType i = 0; i < A.nrows(); ++i)
            {
                bool const complement_flag = true;

                T_row.clear();

                // T[i] = !M[i] .* (A[i] dot B[j])
                if (!A[i].empty()) // && !M[i].empty()) cannot do mask shortcut
                {
                    auto M_iter(M[i].begin());
                    for (IndexType j = 0; j < B.nrows(); ++j)
                    {
                        if (B[j].empty() ||
                            advance_and_check_mask_iterator(
                                M_iter, M[i].end(), structure_flag, j))
                            continue;

                        // Perform the dot product
                        TScalarType t_ij;
                        if (dot(t_ij, A[i], B[j], semiring))
                        {
                            T_row.emplace_back(j, t_ij);
                        }
                    }
                }

                if (outp == REPLACE)
                {
                    // C[i] = T[i], z = "replace"
                    C.setRow(i, T_row);
                }
                else /* merge */
                {
                    // C[i] = [M .* C]  U  T[i], z = "merge"
                    C_row.clear();
                    masked_merge(C_row,
                                 M[i], structure_flag, complement_flag,
                                 C[i], T_row);
                    C.setRow(i, C_row);
                }
            }
        }

        //**********************************************************************
        // Perform C<!M,z> = C + (A +.* B) where A, B, M, and C are all unique
        template<typename CScalarT,
                 typename MScalarT,
                 typename AccumT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void ABT_CompMask_Accum_kernel(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            bool                             structure_flag,
            AccumT                    const &accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            OutputControlEnum                outp)
        {
            using TScalarType = typename SemiringT::result_type;
            using ZScalarType = decltype(accum(std::declval<CScalarT>(),
                                               std::declval<TScalarType>()));
            typename LilSparseMatrix<TScalarType>::RowType T_row;
            typename LilSparseMatrix<ZScalarType>::RowType Z_row;
            typename LilSparseMatrix<CScalarT>::RowType    C_row;

            for (IndexType i = 0; i < A.nrows(); ++i)
            {
                bool const complement_flag = true;

                T_row.clear();

                if (!A[i].empty()) // && !M[i].empty()) cannot do mask shortcut
                {
                    auto m_it(M[i].begin());

                    // Compute: T[i] = M[i] .* {C[i] + (A +.* B')[i]}
                    for (IndexType j = 0; j < B.nrows(); ++j)
                    {
                        // See if B[j] has data and M[i] allows write.
                        if (B[j].empty() ||
                            advance_and_check_mask_iterator(
                                m_it, M[i].end(), structure_flag, j))
                        {
                            continue;
                        }

                        // Perform the dot product and accum if necessary
                        TScalarType t_ij;
                        if (dot(t_ij, A[i], B[j], semiring))
                        {
                            T_row.emplace_back(j, t_ij);
                        }
                    }
                }

                // Z[i] = (M .* C) + T[i]
                Z_row.clear();
                masked_accum(Z_row,
                             M[i], structure_flag, complement_flag,
                             accum, C[i], T_row);


                if (outp == REPLACE)
                {
                    C.setRow(i, Z_row);
                }
                else /* merge */
                {
                    // T[i] := (M[i] .* C[i])  U  Z[i]
                    C_row.clear();
                    masked_merge(C_row,
                                 M[i], structure_flag, complement_flag,
                                 C[i], Z_row);
                    C.setRow(i, C_row);  // set even if it is empty.
                }
            }
        }

        //**********************************************************************
        //**********************************************************************
        //**********************************************************************

        //**********************************************************************
        template<typename CScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_NoMask_NoAccum_ABT(
            LilSparseMatrix<CScalarT>       &C,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B)
        {
            //std::cout << "sparse_mxm_NoMask_NoAccum_ABT COMPLETED.\n";

            // C = (A +.* B')
            // short circuit conditions
            if ((A.nvals() == 0) || (B.nvals() == 0))
            {
                C.clear();
                return;
            }

            // =================================================================

            if ((void*)&C == (void*)&B)
            {
                //std::cout << "ABT USING CTMP\n";
                // create temporary to prevent overwrite of inputs
                LilSparseMatrix<CScalarT> Ctmp(C.nrows(), C.ncols());
                ABT_NoMask_NoAccum_kernel(Ctmp, semiring, A, B);
                C.swap(Ctmp);
            }
            else
            {
                ABT_NoMask_NoAccum_kernel(C, semiring, A, B);
            }

            GRB_LOG_VERBOSE("C: " << C);
        }

        //**********************************************************************
        template<typename CScalarT,
                 typename AccumT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_NoMask_Accum_ABT(
            LilSparseMatrix<CScalarT>       &C,
            AccumT                    const &accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B)
        {
            // C = C + (A +.* B')
            // short circuit conditions?
            if ((A.nvals() == 0) || (B.nvals() == 0))
            {
                return;  // do nothing
            }

            // =================================================================

            if ((void*)&C == (void*)&B)
            {
                // create temporary to prevent overwrite of inputs
                LilSparseMatrix<CScalarT> Ctmp(C.nrows(), C.ncols());
                ABT_NoMask_NoAccum_kernel(Ctmp, semiring, A, B);
                for (IndexType i = 0; i < C.nrows(); ++i)
                {
                    C.mergeRow(i, Ctmp[i], accum);
                }
            }
            else
            {
                ABT_NoMask_Accum_kernel(C, accum, semiring, A, B);
            }

            GRB_LOG_VERBOSE("C: " << C);
        }

        //**********************************************************************
        template<typename CScalarT,
                 typename MScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_Mask_NoAccum_ABT(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            bool                             structure_flag,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            OutputControlEnum                outp)
        {
            //std::cout << "sparse_mxm_Mask_NoAccum_ABT COMPLETED.\n";

            // C<M,z> = (A +.* B')
            //        =             [M .* (A +.* B')], z = replace
            //        = [!M .* C] U [M .* (A +.* B')], z = merge
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

            if ((void*)&C == (void*)&B)
            {
                bool const complement_flag = false;

                // create temporary to prevent overwrite of inputs
                LilSparseMatrix<CScalarT> Ctmp(C.nrows(), C.ncols());
                ABT_Mask_NoAccum_kernel(Ctmp,
                                        M, structure_flag,
                                        semiring, A, B, REPLACE);

                if (outp == REPLACE)
                {
                    C.swap(Ctmp);
                }
                else
                {
                    typename LilSparseMatrix<CScalarT>::RowType C_row;
                    for (IndexType i = 0; i < C.nrows(); ++i)
                    {
                        // C[i] = [!M .* C]  U  T[i], z = "merge"
                        C_row.clear();
                        masked_merge(C_row,
                                     M[i], structure_flag, complement_flag,
                                     C[i], Ctmp[i]);
                        C.setRow(i, C_row);
                    }
                }
            }
            else
            {
                ABT_Mask_NoAccum_kernel(C,
                                        M, structure_flag,
                                        semiring, A, B, outp);
            }

            GRB_LOG_VERBOSE("C: " << C);
        }

        //**********************************************************************
        template<typename CScalarT,
                 typename MScalarT,
                 typename AccumT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_Mask_Accum_ABT(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            bool                             structure_flag,
            AccumT                    const &accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            OutputControlEnum                outp)
        {
            //std::cout << "sparse_mxm_Mask_Accum_ABT COMPLETED.\n";

            // C<M,z> = C + (A +.* B')
            //        =               [M .* [C + (A +.* B')]], z = "replace"
            //        = [!M .* C]  U  [M .* [C + (A +.* B')]], z = "merge"
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

            if ((void*)&C == (void*)&B)
            {
                bool const complement_flag = false;

                using TScalarType = typename SemiringT::result_type;
                using ZScalarType = decltype(accum(std::declval<CScalarT>(),
                                                   std::declval<TScalarType>()));

                // create temporary to prevent overwrite of inputs
                LilSparseMatrix<TScalarType> Ctmp(C.nrows(), C.ncols());
                ABT_Mask_NoAccum_kernel(Ctmp,
                                        M, structure_flag,
                                        semiring, A, B, REPLACE);

                typename LilSparseMatrix<ZScalarType>::RowType  Z_row;

                for (IndexType i = 0; i < C.nrows(); ++i)
                {
                    Z_row.clear();
                    // Z[i] = (M .* C) + Ctmp[i]
                    masked_accum(Z_row,
                                 M[i], structure_flag, complement_flag,
                                 accum, C[i], Ctmp[i]);

                    if (outp == REPLACE)
                    {
                        C.setRow(i, Z_row);
                    }
                    else
                    {
                        typename LilSparseMatrix<CScalarT>::RowType C_row;
                        // C[i] = [!M .* C]  U  Ctmp[i], z = "merge"
                        C_row.clear();
                        masked_merge(C_row,
                                     M[i], structure_flag, complement_flag,
                                     C[i], Z_row);
                        C.setRow(i, C_row);
                    }
                }
            }
            else
            {
                ABT_Mask_Accum_kernel(C,
                                      M, structure_flag,
                                      accum, semiring, A, B, outp);
            }

            GRB_LOG_VERBOSE("C: " << C);
        }

        //**********************************************************************
        template<typename CScalarT,
                 typename MScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_CompMask_NoAccum_ABT(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            bool                             structure_flag,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            OutputControlEnum                outp)
        {
            //std::cout << "sparse_mxm_CompMask_NoAccum_ABT COMPLETED.\n";

            // C<M,z> =            [!M .* (A +.* B')], z = replace
            //        = [M .* C] U [!M .* (A +.* B')], z = merge
            // short circuit conditions
            if ((outp == REPLACE) &&
                ((A.nvals() == 0) || (B.nvals() == 0)))
            {
                C.clear();
                return;
            }
            //else if ((outp == MERGE) && (M.nvals() == 0)) // this is NoMask_NoAccum

            // =================================================================

            if ((void*)&C == (void*)&B)
            {
                bool const complement_flag = true;

                // create temporary to prevent overwrite of inputs
                LilSparseMatrix<CScalarT> Ctmp(C.nrows(), C.ncols());
                ABT_CompMask_NoAccum_kernel(Ctmp,
                                            M, structure_flag,
                                            semiring, A, B, REPLACE);

                if (outp == REPLACE)
                {
                    C.swap(Ctmp);
                }
                else
                {
                    typename LilSparseMatrix<CScalarT>::RowType C_row;
                    for (IndexType i = 0; i < C.nrows(); ++i)
                    {
                        // C[i] = [!M .* C]  U  T[i], z = "merge"
                        C_row.clear();
                        masked_merge(C_row,
                                     M[i], structure_flag, complement_flag,
                                     C[i], Ctmp[i]);
                        C.setRow(i, C_row);
                    }
                }
            }
            else
            {
                ABT_CompMask_NoAccum_kernel(C,
                                            M, structure_flag,
                                            semiring, A, B, outp);
            }

            GRB_LOG_VERBOSE("C: " << C);
        }

        //**********************************************************************
        template<typename CScalarT,
                 typename MScalarT,
                 typename AccumT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_CompMask_Accum_ABT(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            bool                             structure_flag,
            AccumT                    const &accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            OutputControlEnum                outp)
        {
            //std::cout << "sparse_mxm_CompMask_Accum_ABT COMPLETED.\n";

            // C<M,z> = C + (A +.* B')
            //        =              [!M .* [C + (A +.* B')]], z = "replace"
            //        = [M .* C]  U  [!M .* [C + (A +.* B')]], z = "merge"
            // short circuit conditions: NONE?

            // =================================================================

            if ((void*)&C == (void*)&B)
            {
                bool const complement_flag = true;

                using TScalarType = typename SemiringT::result_type;
                using ZScalarType = decltype(accum(std::declval<CScalarT>(),
                                                   std::declval<TScalarType>()));

                // create temporary to prevent overwrite of inputs
                LilSparseMatrix<TScalarType> Ctmp(C.nrows(), C.ncols());
                ABT_CompMask_NoAccum_kernel(Ctmp,
                                            M, structure_flag,
                                            semiring, A, B, REPLACE);

                typename LilSparseMatrix<ZScalarType>::RowType  Z_row;

                for (IndexType i = 0; i < C.nrows(); ++i)
                {
                    Z_row.clear();
                    // Z[i] = (M .* C) + Ctmp[i]
                    masked_accum(Z_row,
                                 M[i], structure_flag, complement_flag,
                                 accum, C[i], Ctmp[i]);

                    if (outp == REPLACE)
                    {
                        C.setRow(i, Z_row);
                    }
                    else
                    {
                        typename LilSparseMatrix<CScalarT>::RowType C_row;
                        // C[i] = [!M .* C]  U  Ctmp[i], z = "merge"
                        C_row.clear();
                        masked_merge(C_row,
                                     M[i], structure_flag, complement_flag,
                                     C[i], Z_row);
                        C.setRow(i, C_row);
                    }
                }
            }
            else
            {
                ABT_CompMask_Accum_kernel(C,
                                          M, structure_flag,
                                          accum, semiring, A, B, outp);
            }

            GRB_LOG_VERBOSE("C: " << C);
        }

    } // backend
} // grb
