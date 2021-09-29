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
#include <graphblas/algebra.hpp>

#include "sparse_helpers.hpp"

//****************************************************************************

namespace grb
{
    namespace backend
    {
        //**********************************************************************
        template <typename ZScalarT,
                  typename WScalarT,
                  typename TScalarT,
                  typename BinaryOpT>
        void opt_accum_scalar(ZScalarT       &z,
                              WScalarT const &w,
                              TScalarT const &t,
                              BinaryOpT       accum)
        {
            z = static_cast<ZScalarT>(accum(w, t));
        }

        //**********************************************************************
        // Specialized version that gets used when we don't have an accumulator
        template <typename ZScalarT,
                  typename WScalarT,
                  typename TScalarT>
        void opt_accum_scalar(ZScalarT                &z,
                              WScalarT          const &w,
                              TScalarT          const &t,
                              grb::NoAccumulate        accum)
        {
            z = static_cast<ZScalarT>(t);
        }

        //************************************************************************
        /// A reduction of a sparse vector (vector<tuple(index,value)>) using a
        /// binary op or a monoid.
        template <typename D1, typename D3, typename BinaryOpT>
        bool reduction(
            D3                                                &ans,
            std::vector<std::tuple<grb::IndexType,D1> > const &vec,
            BinaryOpT                                          op)
        {
            if (vec.empty())
            {
                return false;
            }

            using D3ScalarType =
                decltype(op(std::declval<D1>(), std::declval<D1>()));
            D3ScalarType tmp;

            if (vec.size() == 1)
            {
                tmp = static_cast<D3ScalarType>(std::get<1>(vec[0]));
            }
            else
            {
                /// @note Since op is associative and commutative left to right
                /// ordering is not strictly required.
                tmp = op(std::get<1>(vec[0]), std::get<1>(vec[1]));

                /// @todo replace with call to std::reduce?
                for (size_t idx = 2; idx < vec.size(); ++idx)
                {
                    tmp = op(tmp, std::get<1>(vec[idx]));
                }
            }

            ans = static_cast<D3>(tmp);
            return true;
        }

        //************************************************************************
        /// A reduction of a sparse vector (vector<tuple(index,value)>) using a
        /// binary op or a monoid.
        template <typename D1, typename D3, typename BinaryOpT>
        bool reduction(
            D3                           &ans,
            BitmapSparseVector<D1> const &vec,
            BinaryOpT                     op)
        {
            if (vec.nvals() == 0)
            {
                return false;
            }

            using D3ScalarType =
                decltype(op(std::declval<D1>(), std::declval<D1>()));
            D3ScalarType tmp;

            bool val_set = false;
            for (grb::IndexType ix = 0; ix < vec.size(); ++ix)
            {
                if (vec.hasElementNoCheck(ix))
                {
                    if (val_set)
                    {
                        tmp = op(tmp, vec.extractElementNoCheck(ix));
                    }
                    else
                    {
                        val_set = true;
                        tmp = static_cast<D3ScalarType>(
                            vec.extractElementNoCheck(ix));
                    }
                }
            }

            ans = static_cast<D3>(tmp);
            return true;
        }

        //********************************************************************
        /// Implementation of 4.3.9.1 reduce: Standard Matrix to Vector variant:
        /// w<m,z> := op_j(A[:,j])
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  // monoid or binary op only
                 typename AMatrixT>
        inline void reduce(WVectorT          &w,
                           MaskT       const &mask,
                           AccumT      const &accum,
                           BinaryOpT          op,
                           AMatrixT    const &A,
                           OutputControlEnum  outp)
        {
            GRB_LOG_VERBOSE("w<m,z> := op_j(A[:,j]) (row reduce)");

            // =================================================================
            // Do the basic reduction work with the binary op
            using TScalarType =
                decltype(op(std::declval<typename AMatrixT::ScalarType>(),
                            std::declval<typename AMatrixT::ScalarType>()));
            std::vector<std::tuple<IndexType, TScalarType> > t;

            if (A.nvals() > 0)
            {
                for (IndexType row_idx = 0; row_idx < A.nrows(); ++row_idx)
                {
                    /// @todo There is something hinky with domains here.  How
                    /// does one perform the reduction in A domain but produce
                    /// partial results in D3(op)?
                    TScalarType t_val;
                    if (reduction(t_val, A[row_idx], op))
                    {
                        t.emplace_back(row_idx, t_val);
                    }
                }
            }

            // =================================================================
            // Accumulate into Z
            // Type generator for z: D3(accum), or D(w) if no accum.
            using ZScalarType = typename std::conditional_t<
                std::is_same_v<AccumT, NoAccumulate>,
                TScalarType,
                decltype(accum(std::declval<typename WVectorT::ScalarType>(),
                               std::declval<TScalarType>()))>;
            std::vector<std::tuple<IndexType, ZScalarType> > z;
            ewise_or_opt_accum_1D(z, w, t, accum);

            // =================================================================
            // Copy Z into the final output, w, considering mask and replace/merge
            write_with_opt_mask_1D(w, z, mask, outp);
        }

        //********************************************************************
        /// Implementation of 4.3.9.1 reduce: Standard Matrix to Vector variant:
        /// w<m,z> := op_j(A'[:,j]) = op_j(A[j,:]) column reduce
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  // monoid or binary op only
                 typename AMatrixT>
        inline void reduce(WVectorT                      &w,
                           MaskT                   const &mask,
                           AccumT                  const &accum,
                           BinaryOpT                      op,
                           TransposeView<AMatrixT> const &AT,
                           OutputControlEnum              outp)
        {
            GRB_LOG_VERBOSE("w<m,z> := op_j(A[:,j]) = op_j(A[j,:]) (col reduce)");
            auto const &A(AT.m_mat);

            // =================================================================
            // Do the basic reduction work with the binary op
            using TScalarType =
                decltype(op(std::declval<typename AMatrixT::ScalarType>(),
                            std::declval<typename AMatrixT::ScalarType>()));
            std::vector<std::tuple<IndexType, TScalarType> > t;

            if (A.nvals() > 0)
            {
                for (IndexType row_idx = 0; row_idx < A.nrows(); ++row_idx)
                {
                    /// @todo There is something hinky with domains here.  How
                    /// does one perform the reduction in A domain but produce
                    /// partial results in D3(op)?
                    if (!A[row_idx].empty())
                        xpey(t, A[row_idx], op);
                }
            }

            // =================================================================
            // Accumulate into Z
            // Type generator for z: D3(accum), or D(w) if no accum.
            using ZScalarType = typename std::conditional_t<
                std::is_same_v<AccumT, NoAccumulate>,
                TScalarType,
                decltype(accum(std::declval<typename WVectorT::ScalarType>(),
                               std::declval<TScalarType>()))>;
            std::vector<std::tuple<IndexType, ZScalarType> > z;
            ewise_or_opt_accum_1D(z, w, t, accum);

            // =================================================================
            // Copy Z into the final output, w, considering mask and replace/merge
            write_with_opt_mask_1D(w, z, mask, outp);
        }

        //********************************************************************
        /// Implementation of 4.3.9.2 reduce: Vector to scalar variant
        template<typename ValueT,
                 typename AccumT,
                 typename MonoidT, // monoid only
                 typename UVectorT>
        inline void reduce_vector_to_scalar(ValueT         &val,
                                            AccumT   const &accum,
                                            MonoidT         op,
                                            UVectorT const &u)
        {
            // =================================================================
            // Do the basic reduction work with the monoid
            using UScalarType = typename UVectorT::ScalarType;
            using TScalarType = decltype(op(std::declval<UScalarType>(),
                                            std::declval<UScalarType>()));

            TScalarType t = op.identity();

            if (u.nvals() > 0)
            {
                reduction(t, u, op);
            }

            // =================================================================
            // Accumulate into Z
            // Type generator for z: D3(accum), or D(w) if no accum.
            using ZScalarType = typename std::conditional_t<
                std::is_same_v<AccumT, NoAccumulate>,
                TScalarType,
                decltype(accum(std::declval<ValueT>(),
                               std::declval<TScalarType>()))>;

            ZScalarType z;
            opt_accum_scalar(z, val, t, accum);

            // Copy Z into the final output
            val = z;
        }

        //********************************************************************
        /// Implementation of 4.3.9.3 reduce: Matrix to scalar variant
        template<typename ValueT,
                 typename AccumT,
                 typename MonoidT, // monoid only
                 typename AMatrixT>
        inline void reduce_matrix_to_scalar(ValueT         &val,
                                            AccumT   const &accum,
                                            MonoidT         op,
                                            AMatrixT const &A)
        {
            // =================================================================
            // Do the basic reduction work with the monoid
            using TScalarType =
                decltype(op(std::declval<typename AMatrixT::ScalarType>(),
                            std::declval<typename AMatrixT::ScalarType>()));
            TScalarType t = op.identity();

            if (A.nvals() > 0)
            {
                for (IndexType row_idx = 0; row_idx < A.nrows(); ++row_idx)
                {
                    /// @todo There is something hinky with domains here.  How
                    /// does one perform the reduction in A domain but produce
                    /// partial results in D3(op)?
                    TScalarType tmp;

                    if (!A[row_idx].empty())
                    {
                        if (reduction(tmp, A[row_idx], op)) // reduce each row
                        {
                            t = op(t, tmp); // reduce across rows
                        }
                    }
                }
            }

            // =================================================================
            // Accumulate into Z
            // Type generator for z: D3(accum), or D(w) if no accum.
            using ZScalarType = typename std::conditional_t<
                std::is_same_v<AccumT, NoAccumulate>,
                TScalarType,
                decltype(accum(std::declval<ValueT>(),
                               std::declval<TScalarType>()))>;

            ZScalarType z;
            opt_accum_scalar(z, val, t, accum);

            // Copy Z into the final output
            val = z;
        }

        //********************************************************************
        /// Implementation of 4.3.9.3 reduce: transpose Matrix to scalar variant
        template<typename ValueT,
                 typename AccumT,
                 typename MonoidT, // monoid only
                 typename AMatrixT>
        inline void reduce_matrix_to_scalar(
            ValueT                        &val,
            AccumT                  const &accum,
            MonoidT                        op,
            TransposeView<AMatrixT> const &AT)
        {
            auto const &A(AT.m_mat);

            // =================================================================
            // Do the basic reduction work with the monoid
            using TScalarType =
                decltype(op(std::declval<typename AMatrixT::ScalarType>(),
                            std::declval<typename AMatrixT::ScalarType>()));
            TScalarType t = op.identity();

            if (A.nvals() > 0)
            {
                for (IndexType row_idx = 0; row_idx < A.nrows(); ++row_idx)
                {
                    /// @todo There is something hinky with domains here.  How
                    /// does one perform the reduction in A domain but produce
                    /// partial results in D3(op)?
                    TScalarType tmp;

                    if (!A[row_idx].empty())
                    {
                        if (reduction(tmp, A[row_idx], op)) // reduce each row
                        {
                            t = op(t, tmp); // reduce across rows
                        }
                    }
                }
            }

            // =================================================================
            // Accumulate into Z
            // Type generator for z: D3(accum), or D(w) if no accum.
            using ZScalarType = typename std::conditional_t<
                std::is_same_v<AccumT, NoAccumulate>,
                TScalarType,
                decltype(accum(std::declval<ValueT>(),
                               std::declval<TScalarType>()))>;

            ZScalarType z;
            opt_accum_scalar(z, val, t, accum);

            // Copy Z into the final output
            val = z;
        }

    } // backend
} // grb
