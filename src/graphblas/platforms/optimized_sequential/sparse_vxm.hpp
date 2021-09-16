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
        //********************************************************************
        /// Implementation of 4.3.2 vxm: u * A
        //********************************************************************
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename SemiringT,
                 typename UVectorT,
                 typename AScalarT>
        inline void vxm(WVectorT          &w,
                        MaskT       const &mask,
                        AccumT      const &accum,
                        SemiringT          op,
                        UVectorT    const &u,
                        LilSparseMatrix<AScalarT>    const &A,
                        OutputControlEnum  outp)
        {
            GRB_LOG_VERBOSE("w<M,z> := u +.* A");

            // =================================================================
            // Use axpy approach with the semi-ring.
            using TScalarType = typename SemiringT::result_type;
            BitmapSparseVector<TScalarType> t(w.size());

            if ((A.nvals() > 0) && (u.nvals() > 0))
            {
                for (IndexType row_idx = 0; row_idx < u.size(); ++row_idx)
                {
                    if (u.hasElementNoCheck(row_idx) && !A[row_idx].empty())
                    {
                        axpy(t, op, u.extractElementNoCheck(row_idx), A[row_idx]);
                    }
                }
            }

            // =================================================================
            // Accumulate into final output, w, considering mask and replace/merge
            using ZScalarType = typename std::conditional_t<
                std::is_same_v<AccumT, NoAccumulate>,
                TScalarType,
                decltype(accum(std::declval<typename WVectorT::ScalarType>(),
                               std::declval<TScalarType>()))>;

            opt_accum_with_opt_mask_1D(w, mask, accum, t, outp);
        }

        //**********************************************************************
        //**********************************************************************
        //**********************************************************************

        //********************************************************************
        /// Implementation of 4.3.2 vxm: u * A'
        //********************************************************************
        // template<typename WVectorT,
        //          typename MaskT,
        //          typename AccumT,
        //          typename SemiringT,
        //          typename AMatrixT,
        //          typename UVectorT>
        // inline void vxm(WVectorT                      &w,
        //                 MaskT                   const &mask,
        //                 AccumT                  const &accum,
        //                 SemiringT                      op,
        //                 UVectorT                const &u,
        //                 TransposeView<AMatrixT> const &AT,
        //                 OutputControlEnum              outp)
        // {
        //     GRB_LOG_VERBOSE("w<M,z> := u +.* A'");
        //     backend::mxv(w, mask, accum, op, AT.m_mat, u, outp);
        // }


        //**********************************************************************
        /// Implementation for 4.3.2 vxm: u * A'
        //**********************************************************************
        template<typename WVectorT,
                 typename SemiringT,
                 typename UVectorT,
                 typename AMatrixT>
        inline void vxm(WVectorT           &w,
                        NoMask       const &mask,
                        NoAccumulate const &accum,
                        SemiringT           op,
                        UVectorT     const &u,
                        TransposeView<AMatrixT> const &AT,
                        OutputControlEnum  outp)
        {
            //mxv_dot_nomask_noaccum(w, op, A, u);
            GRB_LOG_VERBOSE("w := u +.* A'");
            auto const &A(AT.m_mat);
            w.clear();

            // =================================================================
            // Do the basic dot-product work with the semi-ring.
            using TScalarType = typename SemiringT::result_type;

            if ((A.nvals() > 0) && (u.nvals() > 0))
            {
                for (IndexType row_idx = 0; row_idx < w.size(); ++row_idx)
                {
                    if ( !A[row_idx].empty() )
                    {
                        TScalarType t_val;
                        if (dot_dense_sparse(t_val, u, A[row_idx], op))
                        {
                            w.setElementNoCheck(row_idx, t_val);
                        }
                    }
                }
            }
        }

        //**********************************************************************
        /// Implementation for 4.3.2 mxv: w + u * A'
        //**********************************************************************
        template<typename WVectorT,
                 typename AccumT,
                 typename SemiringT,
                 typename UVectorT,
                 typename AMatrixT>
        inline void vxm(WVectorT           &w,
                        NoMask       const &mask,
                        AccumT       const &accum,
                        SemiringT           op,
                        UVectorT     const &u,
                        TransposeView<AMatrixT> const &AT,
                        OutputControlEnum  outp)
        {
            //mxv_dot_nomask_accum(w, accum, op, A, u);
            GRB_LOG_VERBOSE("w := w + (u +.* A')");
            auto const &A(AT.m_mat);

            // =================================================================
            // Do the basic dot-product work with the semi-ring.
            using TScalarType = typename SemiringT::result_type;

            if ((A.nvals() > 0) && (u.nvals() > 0))
            {
                for (IndexType row_idx = 0; row_idx < w.size(); ++row_idx)
                {
                    if ( !A[row_idx].empty() )
                    {
                        TScalarType t_val;
                        if (dot_dense_sparse(t_val, u, A[row_idx], op))
                        {
                            if (w.hasElementNoCheck(row_idx))
                            {
                                w.setElement(
                                    row_idx,
                                    accum(w.extractElementNoCheck(row_idx),
                                          t_val));
                            }
                            else
                            {
                                w.setElement(row_idx, t_val);
                            }
                        }
                    }
                }
            }
        }

        //**********************************************************************
        /// Implementation for 4.3.2 mxv: <m,r>(u * A')
        //**********************************************************************
        template<typename WVectorT,
                 typename MaskT,
                 typename SemiringT,
                 typename UVectorT,
                 typename AMatrixT>
        inline void vxm(WVectorT           &w,
                        MaskT        const &mask,
                        NoAccumulate const &accum,
                        SemiringT           op,
                        UVectorT     const &u,
                        TransposeView<AMatrixT> const &AT,
                        OutputControlEnum   outp)
        {
            //mxv_dot_mask_noaccum(w, mask, op, A, u, outp);
            GRB_LOG_VERBOSE("w<M,r> := (u +.* A')");
            auto const &A(AT.m_mat);

            // =================================================================
            // Do the basic dot-product work with the semi-ring.
            using TScalarType = typename SemiringT::result_type;
            std::vector<std::tuple<IndexType, TScalarType> > t;

            if (outp == REPLACE)
                w.clear();

            if ((A.nvals() > 0) && (u.nvals() > 0))
            {
                for (IndexType row_idx = 0; row_idx < w.size(); ++row_idx)
                {
                    bool element_set = false;
                    if (check_mask_1D(mask, row_idx))
                    {
                        //w.removeElementNoCheck(row_idx);
                        if (!A[row_idx].empty())
                        {
                            TScalarType t_val;
                            if (dot_dense_sparse(t_val, u, A[row_idx], op))
                            {
                                w.setElementNoCheck(row_idx, t_val);
                                element_set = true;
                            }
                            else if (outp == MERGE)
                            {
                                w.removeElementNoCheck(row_idx);
                            }
                        }
                        else if (outp == MERGE)
                        {
                            w.removeElementNoCheck(row_idx);
                        }
                    }
                }
            }
        }

        //**********************************************************************
        /// Implementation for 4.3.2 mxv: <m,r>(w + u * A')
        //**********************************************************************
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename SemiringT,
                 typename UVectorT,
                 typename AMatrixT>
        inline void vxm(WVectorT          &w,
                        MaskT       const &mask,
                        AccumT      const &accum,
                        SemiringT          op,
                        UVectorT    const &u,
                        TransposeView<AMatrixT> const &AT,
                        OutputControlEnum  outp)
        {
            //mxv_dot_mask_accum(w, mask, accum, op, A, u, outp);
            GRB_LOG_VERBOSE("w<M,r> := w + (u +.* A')");
            auto const &A(AT.m_mat);

            // =================================================================
            // Do the basic dot-product work with the semi-ring.
            using TScalarType = typename SemiringT::result_type;

            // =================================================================
            // Accumulate into Z
            using ZScalarType =
                decltype(accum(std::declval<typename WVectorT::ScalarType>(),
                               std::declval<TScalarType>()));
            std::vector<std::tuple<IndexType, ZScalarType> > z;

            if ((A.nvals() > 0) && (u.nvals() > 0))
            {
                for (IndexType row_idx = 0; row_idx < w.size(); ++row_idx)
                {
                    if (check_mask_1D(mask, row_idx) && !A[row_idx].empty())
                    {
                        TScalarType t_val;
                        if (dot_dense_sparse(t_val, u, A[row_idx], op))
                        {
                            if (w.get_bitmap()[row_idx])
                            {
                                z.emplace_back(row_idx,
                                               accum(w.get_vals()[row_idx],
                                                     t_val));
                            }
                            else
                            {
                                z.emplace_back(row_idx,
                                               static_cast<ZScalarType>(t_val));
                            }
                        }
                        else if (w.get_bitmap()[row_idx])
                        {
                            z.emplace_back(
                                row_idx,
                                static_cast<ZScalarType>(w.get_vals()[row_idx]));
                        }
                    }
                }
            }

            // =================================================================
            // Copy Z into the final output, w, considering mask and replace/merge
            write_with_opt_mask_1D_sparse_dense(w, z, mask, outp);
        }
    } // backend
} // grb
