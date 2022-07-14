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
        // Implementation of 4.3.8.1 Vector variant of Apply: w<m,z> := op(u)
        template<typename WScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename UnaryOpT,
                 typename UVectorT,
                 typename ...WTagsT>
        inline void apply(
            grb::backend::Vector<WScalarT, WTagsT...>       &w,
            MaskT                                     const &mask,
            AccumT                                    const &accum,
            UnaryOpT                                         op,
            UVectorT                                  const &u,
            OutputControlEnum                                outp)
        {
            GRB_LOG_VERBOSE("w<m,z> := op(u)");
            // =================================================================
            // Apply the unary operator from u into t.
            using UScalarType = typename UVectorT::ScalarType;
            using TScalarType = decltype(op(std::declval<UScalarType>()));
            std::vector<std::tuple<IndexType,TScalarType> > t_contents;

            if (u.nvals() > 0)
            {
                for (auto&& [idx, val] : u.getContents()) {
                    t_contents.emplace_back(idx, op(val));
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
        // Implementation of 4.3.8.2 Matrix variant of Apply: C<M,z> := op(A)
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename UnaryOpT,
                 typename AMatrixT,
                 typename ...CTagsT>
        inline void apply(
            grb::backend::Matrix<CScalarT, CTagsT...>       &C,
            MaskT                                     const &Mask,
            AccumT                                    const &accum,
            UnaryOpT                                         op,
            AMatrixT                                  const &A,
            OutputControlEnum                                outp)
        {
            GRB_LOG_VERBOSE("C<M,z> := op(A)");
            IndexType nrows(A.nrows());
            IndexType ncols(A.ncols());

            // =================================================================
            // Apply the unary operator from A into T.
            using AScalarType = typename AMatrixT::ScalarType;
            using TScalarType = decltype(op(std::declval<AScalarType>()));
            LilSparseMatrix<TScalarType> T(nrows, ncols);

            for (IndexType row_idx = 0; row_idx < A.nrows(); ++row_idx)
            {
                for (auto&& [a_idx, a_val] : A[row_idx])
                {
                    T[row_idx].emplace_back(a_idx, op(a_val));
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
        // Implementation of 4.3.8.2 Matrix variant of Apply: C<M,z> := op(A')
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename UnaryOpT,
                 typename AMatrixT,
                 typename ...CTagsT>
        inline void apply(
            grb::backend::Matrix<CScalarT, CTagsT...>       &C,
            MaskT                                     const &Mask,
            AccumT                                    const &accum,
            UnaryOpT                                         op,
            TransposeView<AMatrixT>                   const &AT,
            OutputControlEnum                                outp)
        {
            GRB_LOG_VERBOSE("C<M,z> := op(A')");
            auto const &A(AT.m_mat);
            IndexType nrows(A.nrows());
            IndexType ncols(A.ncols());

            // =================================================================
            // Apply the unary operator from A into T.
            using AScalarType = typename AMatrixT::ScalarType;
            using TScalarType = decltype(op(std::declval<AScalarType>()));
            LilSparseMatrix<TScalarType> T(ncols, nrows);

            for (IndexType row_idx = 0; row_idx < A.nrows(); ++row_idx)
            {
                for (auto&& [a_idx, a_val] : A[row_idx])
                {
                    T[a_idx].emplace_back(row_idx, op(a_val)); // idx's swapped
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

        //**********************************************************************
        // Implementation of 4.3.8.3 Vector variant of Apply w/ binaryop+bind1st:
        // w<m,z> := op(val, u)
        /// @note this is not necessary in the C++ API, here for demonstration.
        template<typename WScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,
                 typename ValueT,
                 typename UVectorT,
                 typename ...WTagsT>
        inline void apply_binop_1st(
            grb::backend::Vector<WScalarT, WTagsT...>       &w,
            MaskT                                     const &mask,
            AccumT                                    const &accum,
            BinaryOpT                                        op,
            ValueT                                    const &val,
            UVectorT                                  const &u,
            OutputControlEnum                                outp)
        {
            GRB_LOG_VERBOSE("w<m,z> := op(val, u)");
            // =================================================================
            // Apply the binary operator to u and val and store into T.
            using UScalarType = typename UVectorT::ScalarType;
            using TScalarType = decltype(op(std::declval<ValueT>(),
                                            std::declval<UScalarType>()));
            std::vector<std::tuple<IndexType,TScalarType> > t_contents;

            if (u.nvals() > 0)
            {
                for (auto&& [idx, u_val] : u.getContents()) {
                    t_contents.emplace_back(idx, op(val, u_val));
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
        // Implementation of 4.3.8.3 Vector variant of Apply w/ binaryop+bind2nd:
        // w<m,z> := op(u, val)
        /// @note this is not necessary in the C++ API, here for demonstration.
        template<typename WScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,
                 typename UVectorT,
                 typename ValueT,
                 typename ...WTagsT>
        inline void apply_binop_2nd(
            grb::backend::Vector<WScalarT, WTagsT...>       &w,
            MaskT                                     const &mask,
            AccumT                                    const &accum,
            BinaryOpT                                        op,
            UVectorT                                  const &u,
            ValueT                                    const &val,
            OutputControlEnum                                outp)
        {
            GRB_LOG_VERBOSE("w<m,z> := op(u, val)");
            // =================================================================
            // Apply the binary operator to u and val and store into T.
            // This is really the guts of what makes this special.
            using UScalarType = typename UVectorT::ScalarType;
            using TScalarType = decltype(op(std::declval<UScalarType>(),
                                            std::declval<ValueT>()));
            std::vector<std::tuple<IndexType,TScalarType> > t_contents;

            if (u.nvals() > 0)
            {
                for (auto&& [idx, u_val] : u.getContents()) {
                    t_contents.emplace_back(idx, op(u_val, val));
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
        // Implementation of 4.3.8.4 Matrix variant of Apply w/ binaryop+bind1st
        // C<M,z> := op(val, A)
        /// @note this is not necessary in the C++ API, here for demonstration.
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,
                 typename ValueT,
                 typename AMatrixT,
                 typename ...CTagsT>
        inline void apply_binop_1st(
            grb::backend::Matrix<CScalarT, CTagsT...>       &C,
            MaskT                                     const &Mask,
            AccumT                                    const &accum,
            BinaryOpT                                        op,
            ValueT                                    const &val,
            AMatrixT                                  const &A,
            OutputControlEnum                                outp)
        {
            GRB_LOG_VERBOSE("C<M,z> := op(val, A)");
            IndexType nrows(A.nrows());
            IndexType ncols(A.ncols());

            // =================================================================
            // Apply the unary operator from A into T.
            using AScalarType = typename AMatrixT::ScalarType;
            using TScalarType = decltype(op(std::declval<ValueT>(),
                                            std::declval<AScalarType>()));
            LilSparseMatrix<TScalarType> T(nrows, ncols);

            for (IndexType row_idx = 0; row_idx < A.nrows(); ++row_idx)
            {
                for (auto&& [a_idx, a_val] : A[row_idx])
                {
                    T[row_idx].emplace_back(a_idx, op(val, a_val));
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
        // Implementation of 4.3.8.4 Matrix variant of Apply w/ binaryop+bind1st
        // C<M,z> := op(val, A')
        /// @note this is not necessary in the C++ API, here for demonstration.
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,
                 typename ValueT,
                 typename AMatrixT,
                 typename ...CTagsT>
        inline void apply_binop_1st(
            grb::backend::Matrix<CScalarT, CTagsT...>       &C,
            MaskT                                     const &Mask,
            AccumT                                    const &accum,
            BinaryOpT                                        op,
            ValueT                                    const &val,
            TransposeView<AMatrixT>                   const &AT,
            OutputControlEnum                                outp)
        {
            GRB_LOG_VERBOSE("C<M,z> := op(val, A')");
            auto const &A(AT.m_mat);
            IndexType nrows(A.nrows());
            IndexType ncols(A.ncols());

            // =================================================================
            // Apply the unary operator from A into T.
            using AScalarType = typename AMatrixT::ScalarType;
            using TScalarType = decltype(op(std::declval<ValueT>(),
                                            std::declval<AScalarType>()));
            LilSparseMatrix<TScalarType> T(ncols, nrows);

            for (IndexType row_idx = 0; row_idx < A.nrows(); ++row_idx)
            {
                for (auto&& [a_idx, a_val] : A[row_idx])
                {
                    T[a_idx].emplace_back(row_idx, op(val, a_val)); // idx's swapped
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


        //**********************************************************************
        // Implementation of 4.3.8.4 Matrix variant of Apply w/ binaryop+bind2nd
        // C<M,z> := op(A, val)
        /// @note this is not necessary in the C++ API, here for demonstration.
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,
                 typename AMatrixT,
                 typename ValueT,
                 typename ...CTagsT>
        inline void apply_binop_2nd(
            grb::backend::Matrix<CScalarT, CTagsT...>       &C,
            MaskT                                     const &Mask,
            AccumT                                    const &accum,
            BinaryOpT                                        op,
            AMatrixT                                  const &A,
            ValueT                                    const &val,
            OutputControlEnum                                outp)
        {
            GRB_LOG_VERBOSE("C<M,z> := op(A, val)");
            IndexType nrows(A.nrows());
            IndexType ncols(A.ncols());

            // =================================================================
            // Apply the unary operator from A into T.
            // This is really the guts of what makes this special.
            using AScalarType = typename AMatrixT::ScalarType;
            using TScalarType = decltype(op(std::declval<AScalarType>(),
                                            std::declval<ValueT>()));
            LilSparseMatrix<TScalarType> T(nrows, ncols);

            for (IndexType row_idx = 0; row_idx < A.nrows(); ++row_idx)
            {
                for (auto&& [a_idx, a_val] : A[row_idx])
                {
                    T[row_idx].emplace_back(a_idx, op(a_val, val));
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
        // Implementation of 4.3.8.4 Matrix variant of Apply w/ binaryop+bind2nd
        // C<M,z> := op(A', val)
        /// @note this is not necessary in the C++ API, here for demonstration.
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,
                 typename AMatrixT,
                 typename ValueT,
                 typename ...CTagsT>
        inline void apply_binop_2nd(
            grb::backend::Matrix<CScalarT, CTagsT...>       &C,
            MaskT                                     const &Mask,
            AccumT                                    const &accum,
            BinaryOpT                                        op,
            TransposeView<AMatrixT>                   const &AT,
            ValueT                                    const &val,
            OutputControlEnum                                outp)
        {
            GRB_LOG_VERBOSE("C<M,z> := op(A', val)");
            auto const &A(AT.m_mat);
            IndexType nrows(A.nrows());
            IndexType ncols(A.ncols());

            // =================================================================
            // Apply the unary operator from A into T.
            // This is really the guts of what makes this special.
            using AScalarType = typename AMatrixT::ScalarType;
            using TScalarType = decltype(op(std::declval<AScalarType>(),
                                            std::declval<ValueT>()));
            LilSparseMatrix<TScalarType> T(ncols, nrows);

            for (IndexType row_idx = 0; row_idx < A.nrows(); ++row_idx)
            {
                for (auto&& [a_idx, a_val] : A[row_idx])
                {
                    T[a_idx].emplace_back(row_idx, op(a_val, val)); // idx's swapped
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
