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
        // Implementation of 4.3.10 Matrix transpose
        template<typename CMatrixT,
                 typename MaskT,
                 typename AccumT,
                 typename AMatrixT>
        inline void transpose(CMatrixT          &C,
                              MaskT       const &mask,
                              AccumT      const &accum,
                              AMatrixT    const &A,
                              OutputControlEnum  outp)
        {
            GRB_LOG_VERBOSE("C<M,z> := A'");
            IndexType nrows(A.nrows());
            IndexType ncols(A.ncols());

            // =================================================================
            // Transpose A into T.
            LilSparseMatrix<typename AMatrixT::ScalarType> T(ncols, nrows);
            if (A.nvals() > 0)
            {
                for (IndexType row_idx = 0; row_idx < A.nrows(); ++row_idx)
                {
                    for (auto && [col_idx, val] : A[row_idx])
                    {
                        T[col_idx].emplace_back(row_idx, val);
                    }
                }
                T.recomputeNvals();
            }
            // =================================================================
            // Accumulate T via C into Z
            using ZScalarType = typename std::conditional_t<
                std::is_same_v<AccumT, NoAccumulate>,
                typename AMatrixT::ScalarType,
                decltype(accum(std::declval<typename CMatrixT::ScalarType>(),
                               std::declval<typename AMatrixT::ScalarType>()))>;

            LilSparseMatrix<ZScalarType> Z(ncols, nrows);
            ewise_or_opt_accum(Z, C, T, accum);

            // =================================================================
            // Copy Z into the final output considering mask and replace/merge
            write_with_opt_mask(C, Z, mask, outp);
        }

        //**********************************************************************
        // Implementation of 4.3.10 Matrix transpose
        //**********************************************************************
        template<typename CMatrixT,
                 typename MaskT,
                 typename AccumT,
                 typename AMatrixT>
        inline void transpose(CMatrixT                      &C,
                              MaskT                   const &mask,
                              AccumT                  const &accum,
                              TransposeView<AMatrixT> const &AT,
                              OutputControlEnum              outp)
        {
            GRB_LOG_VERBOSE("C<M,z> := (A')'");
            auto const &A(AT.m_mat);
            IndexType nrows(A.nrows());
            IndexType ncols(A.ncols());

            // =================================================================
            /// Do nothing for T if A is TransposeView, Use A in next step.

            // =================================================================
            // Accumulate A via C into Z
            using ZScalarType = typename std::conditional_t<
                std::is_same_v<AccumT, NoAccumulate>,
                typename AMatrixT::ScalarType,
                decltype(accum(std::declval<typename CMatrixT::ScalarType>(),
                               std::declval<typename AMatrixT::ScalarType>()))>;

            LilSparseMatrix<ZScalarType> Z(nrows, ncols);
            ewise_or_opt_accum(Z, C, A, accum);

            // =================================================================
            // Copy Z into the final output considering mask and replace/merge
            write_with_opt_mask(C, Z, mask, outp);
        }
    }
}
