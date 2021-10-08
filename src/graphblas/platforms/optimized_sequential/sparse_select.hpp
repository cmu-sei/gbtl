/*
 * GraphBLAS Template Library (GBTL), Version 3.0
 *
 * Copyright 2021 Carnegie Mellon University, Battelle Memorial Institute, and
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
#include <graphblas/exceptions.hpp>
#include <graphblas/algebra.hpp>

#include "sparse_helpers.hpp"
#include "LilSparseMatrix.hpp"

//******************************************************************************

namespace grb
{
    namespace backend
    {
        //**********************************************************************
        // Implementation of 4.3.9.1 Vector variant of Select:
        // w<m,r> := w + u<op(u,ind(u),val)>
        template<typename WScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename IndexUnaryOpT,
                 typename UVectorT,
                 typename ValueT>
        inline void select(
            grb::backend::BitmapSparseVector<WScalarT>      &w,
            MaskT                                     const &mask,
            AccumT                                    const &accum,
            IndexUnaryOpT                                    op,
            UVectorT                                  const &u,
            ValueT                                           val,
            OutputControlEnum                                outp)
        {
            GRB_LOG_VERBOSE("w<m,r> := w + u<op(u,ind(u),val)>");
            // =================================================================
            // Select using the index unary operator from u into t.
            using UScalarType = typename UVectorT::ScalarType;
            using TScalarType = typename UVectorT::ScalarType;
            std::vector<std::tuple<IndexType,TScalarType> > t_contents;

            if (u.nvals() > 0)
            {
                for (grb::IndexType idx = 0; idx < u.size(); ++idx)
                {
                    if (check_mask_1D(mask, idx) &&
                        u.hasElementNoCheck(idx) &&
                        op(u.extractElementNoCheck(idx), {idx}, val))
                    {
                        t_contents.emplace_back(idx,
                                                u.extractElementNoCheck(idx));
                    }
                }
            }

            GRB_LOG_VERBOSE("t: " << t_contents);

            // =================================================================
            // Accumulate into Z
            using ZScalarType = std::conditional_t<
                std::is_same_v<AccumT, NoAccumulate>,
                TScalarType,
                decltype(accum(std::declval<WScalarT>(),
                               std::declval<TScalarType>()))>;

            std::vector<std::tuple<IndexType,ZScalarType> > z_contents;
            ewise_or_opt_accum_1D(z_contents, w, t_contents, accum);

            GRB_LOG_VERBOSE("z: " << z_contents);

            // =================================================================
            // Copy Z into the final output considering mask and replace/merge
            write_with_opt_mask_1D(w, z_contents, mask, outp);
        }

        //**********************************************************************
        // Implementation of 4.3.9.2 Matrix variant of Select:
        // C<M,r> := A<op(A,ind(A),val)>
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename IndexUnaryOpT,
                 typename AMatrixT,
                 typename ValueT>
        inline void select(
            grb::backend::LilSparseMatrix<CScalarT>         &C,
            MaskT                                     const &Mask,
            AccumT                                    const &accum,
            IndexUnaryOpT                                    op,
            AMatrixT                                  const &A,
            ValueT                                           val,
            OutputControlEnum                                outp)
        {
            GRB_LOG_VERBOSE("C<M,r> := A<op(A,ind(A),val)>");
            IndexType nrows(A.nrows());
            IndexType ncols(A.ncols());

            // =================================================================
            // Select using the index unary operator from A into T.
            using AScalarType = typename AMatrixT::ScalarType;
            using TScalarType = typename AMatrixT::ScalarType;
            LilSparseMatrix<TScalarType> T(nrows, ncols);

            for (IndexType row_idx = 0; row_idx < A.nrows(); ++row_idx)
            {
                for (auto&& [col_idx, a_val] : A[row_idx])
                {
                    if (op(a_val, {row_idx, col_idx}, val))
                    {
                        T[row_idx].emplace_back(col_idx, a_val);
                    }
                }
            }
            T.recomputeNvals();

            GRB_LOG_VERBOSE("T: " << T);

            // =================================================================
            // Accumulate T via C into Z
            using ZScalarType = std::conditional_t<
                std::is_same_v<AccumT, NoAccumulate>,
                TScalarType,
                decltype(accum(std::declval<CScalarT>(),
                               std::declval<TScalarType>()))>;

            LilSparseMatrix<ZScalarType> Z(nrows, ncols);
            ewise_or_opt_accum(Z, C, T, accum);

            GRB_LOG_VERBOSE("Z: " << Z);

            // =================================================================
            // Copy Z into the final output considering mask and replace/merge
            write_with_opt_mask(C, Z, Mask, outp);
        }

        //**********************************************************************
        // Implementation of 4.3.9.2 Transpose Matrix variant of Select:
        // C<M,r> := A'<op(A',ind(A'),val)>
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename IndexUnaryOpT,
                 typename AMatrixT,
                 typename ValueT>
        inline void select(
            grb::backend::LilSparseMatrix<CScalarT>         &C,
            MaskT                                     const &Mask,
            AccumT                                    const &accum,
            IndexUnaryOpT                                    op,
            TransposeView<AMatrixT>                   const &AT,
            ValueT                                           val,
            OutputControlEnum                                outp)
        {
            GRB_LOG_VERBOSE("C<M,r> := A'<op(A',ind(A'),val)>");
            auto const &A(AT.m_mat);
            IndexType nrows(A.nrows());
            IndexType ncols(A.ncols());

            // =================================================================
            // Select using the index unary operator from A into T.
            using AScalarType = typename AMatrixT::ScalarType;
            using TScalarType = typename AMatrixT::ScalarType;
            LilSparseMatrix<TScalarType> T(ncols, nrows);

            for (IndexType row_idx = 0; row_idx < A.nrows(); ++row_idx)
            {
                for (auto&& [col_idx, a_val] : A[row_idx])
                {
                    // idx's swapped for transpose of A
                    if (op(a_val, {col_idx, row_idx}, val))
                    {
                        T[col_idx].emplace_back(row_idx, a_val);
                    }
                }
            }
            T.recomputeNvals();

            GRB_LOG_VERBOSE("T: " << T);

            // =================================================================
            // Accumulate T via C into Z
            using ZScalarType = std::conditional_t<
                std::is_same_v<AccumT, NoAccumulate>,
                TScalarType,
                decltype(accum(std::declval<CScalarT>(),
                               std::declval<TScalarType>()))>;

            LilSparseMatrix<ZScalarType> Z(ncols, nrows);
            ewise_or_opt_accum(Z, C, T, accum);

            GRB_LOG_VERBOSE("Z: " << Z);

            // =================================================================
            // Copy Z into the final output considering mask and replace/merge
            write_with_opt_mask(C, Z, Mask, outp);
        }
    }
}
