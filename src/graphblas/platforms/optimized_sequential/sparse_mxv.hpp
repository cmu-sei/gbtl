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

#define GRAPHBLAS_LOGGING_LEVEL 1

//****************************************************************************

namespace grb
{
    namespace backend
    {
        //**********************************************************************
        /// Implementation for 4.3.3 mxv: A * u
        //**********************************************************************
        template<typename WVectorT,
                 typename SemiringT,
                 typename AScalarT,
                 typename UVectorT>
        inline void mxv(WVectorT           &w,
                        NoMask       const &mask,
                        NoAccumulate const &accum,
                        SemiringT           op,
                        LilSparseMatrix<AScalarT>     const &A,
                        UVectorT     const &u,
                        OutputControlEnum  outp)
        {
            mxv_dot_nomask_noaccum(w, op, A, u);
        }

        template<typename WVectorT,
                 typename AccumT,
                 typename SemiringT,
                 typename AScalarT,
                 typename UVectorT>
        inline void mxv(WVectorT           &w,
                        NoMask       const &mask,
                        AccumT       const &accum,
                        SemiringT           op,
                        LilSparseMatrix<AScalarT>     const &A,
                        UVectorT     const &u,
                        OutputControlEnum  outp)
        {
            mxv_dot_nomask_accum(w, accum, op, A, u);
        }

        template<typename WVectorT,
                 typename MaskT,
                 typename SemiringT,
                 typename AScalarT,
                 typename UVectorT>
        inline void mxv(WVectorT           &w,
                        MaskT        const &mask,
                        NoAccumulate const &accum,
                        SemiringT           op,
                        LilSparseMatrix<AScalarT>     const &A,
                        UVectorT     const &u,
                        OutputControlEnum   outp)
        {
            mxv_dot_mask_noaccum(w, mask, op, A, u, outp);
        }

        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename SemiringT,
                 typename AScalarT,
                 typename UVectorT>
        inline void mxv(WVectorT          &w,
                        MaskT       const &mask,
                        AccumT      const &accum,
                        SemiringT          op,
                        LilSparseMatrix<AScalarT>     const &A,
                        UVectorT    const &u,
                        OutputControlEnum  outp)
        {
            mxv_dot_mask_accum(w, mask, accum, op, A, u, outp);
        }

        //**********************************************************************
        //**********************************************************************
        template<typename WVectorT,
                 typename SemiringT,
                 typename AMatrixT,
                 typename UVectorT>
        inline void mxv_dot_nomask_noaccum(WVectorT          &w,
                                           SemiringT          op,
                                           AMatrixT    const &A,
                                           UVectorT    const &u)
        {
            GRB_LOG_VERBOSE("w := A +.* u");
            w.clear();

            // =================================================================
            // Do the basic dot-product work with the semi-ring.
            using TScalarType = typename SemiringT::result_type;

            if ((A.nvals() > 0) && (u.nvals() > 0))
            {
                //auto u_contents(u.getContents());
                for (IndexType row_idx = 0; row_idx < w.size(); ++row_idx)
                {
                    if ( !A[row_idx].empty() )
                    {
                        TScalarType t_val;
                        if (dot_sparse_dense(t_val, A[row_idx], u, op))
                        {
                            w.setElementNoCheck(row_idx, t_val);
                        }
                    }
                }
            }
        }

        //**********************************************************************
        //**********************************************************************
        template<typename WVectorT,
                 typename AccumT,
                 typename SemiringT,
                 typename AMatrixT,
                 typename UVectorT>
        inline void mxv_dot_nomask_accum(WVectorT          &w,
                                         AccumT      const &accum,
                                         SemiringT          op,
                                         AMatrixT    const &A,
                                         UVectorT    const &u)
        {
            GRB_LOG_VERBOSE("w := w + (A +.* u)");

            // =================================================================
            // Do the basic dot-product work with the semi-ring.
            using TScalarType = typename SemiringT::result_type;

            if ((A.nvals() > 0) && (u.nvals() > 0))
            {
                //auto u_contents(u.getContents());
                for (IndexType row_idx = 0; row_idx < w.size(); ++row_idx)
                {
                    if ( !A[row_idx].empty() )
                    {
                        TScalarType t_val;
                        if (dot_sparse_dense(t_val, A[row_idx], u, op))
                        {
                            if (w.hasElementNoCheck(row_idx))
                            {
                                w.setElement(row_idx,
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
        //**********************************************************************
        template<typename WVectorT,
                 typename MaskT,
                 typename SemiringT,
                 typename AMatrixT,
                 typename UVectorT>
        inline void mxv_dot_mask_noaccum(WVectorT          &w,
                                         MaskT       const &mask,
                                         SemiringT          op,
                                         AMatrixT    const &A,
                                         UVectorT    const &u,
                                         OutputControlEnum  outp)
        {
            GRB_LOG_VERBOSE("w<M,r> := (A +.* u)");

            // =================================================================
            // Do the basic dot-product work with the semi-ring.
            using TScalarType = typename SemiringT::result_type;
            std::vector<std::tuple<IndexType, TScalarType> > t;

            if (outp == REPLACE)
                w.clear();

            if ((A.nvals() > 0) && (u.nvals() > 0))
            {
                //auto u_contents(u.getContents());
                for (IndexType row_idx = 0; row_idx < w.size(); ++row_idx)
                {
                    bool element_set = false;
                    if (check_mask(mask, row_idx))
                    {
                        //w.removeElementNoCheck(row_idx);
                        if (!A[row_idx].empty())
                        {
                            TScalarType t_val;
                            if (dot_sparse_dense(t_val, A[row_idx], u, op))
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
        //**********************************************************************
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename SemiringT,
                 typename AMatrixT,
                 typename UVectorT>
        inline void mxv_dot_mask_accum(WVectorT          &w,
                                       MaskT       const &mask,
                                       AccumT      const &accum,
                                       SemiringT          op,
                                       AMatrixT    const &A,
                                       UVectorT    const &u,
                                       OutputControlEnum  outp)
        {
            GRB_LOG_VERBOSE("w<M,r> := w + (A +.* u)");

            // =================================================================
            // Do the basic dot-product work with the semi-ring.
            using TScalarType = typename SemiringT::result_type;
#if defined(ORIGINAL_CODE)
            std::vector<std::tuple<IndexType, TScalarType> > t;
#else
            // =================================================================
            // Accumulate into Z
            using ZScalarType =
                decltype(accum(std::declval<typename WVectorT::ScalarType>(),
                               std::declval<TScalarType>()));
            std::vector<std::tuple<IndexType, ZScalarType> > z;
#endif

            if ((A.nvals() > 0) && (u.nvals() > 0))
            {
                //auto u_contents(u.getContents());
                for (IndexType row_idx = 0; row_idx < w.size(); ++row_idx)
                {
                    //if ((check_mask(mask, row_idx) && !A[row_idx].empty()))
                    if ( !A[row_idx].empty() )
                    {
                        TScalarType t_val;
                        if (dot_sparse_dense(t_val, A[row_idx], u, op))
                        {
#if defined(ORIGINAL_CODE)
                            t.emplace_back(row_idx, t_val);
#else
                            if (std::is_same_v<AccumT, grb::NoAccumulate>)
                            {
                                z.emplace_back(row_idx, (ZScalarType)t_val);
                                //std::cout << row_idx << ": no accum t\n";
                            }
                            else
                            {
                                if (w.get_bitmap()[row_idx])
                                {
                                    z.emplace_back(row_idx,
                                                   accum(w.get_vals()[row_idx],
                                                         t_val));
                                    //std::cout << row_idx << ": accum(w, t)\n";
                                }
                                else
                                {
                                    z.emplace_back(row_idx, (ZScalarType)t_val);
                                    //std::cout << row_idx << ": accum(no w, t)\n";
                                }
                            }
                        }
                    }
                    else
                    {
                        if ((!std::is_same_v<AccumT, grb::NoAccumulate>) && (w.get_bitmap()[row_idx]))
                        {
                            z.emplace_back(row_idx, w.get_vals()[row_idx]);
                            //std::cout << row_idx << ": accum(w, no t)\n";
#endif
                        }
                    }
                }
            }

#if defined(ORIGINAL_CODE)
            ewise_or_opt_accum_1D(z, w, t, accum);
#endif

            // =================================================================
            // Copy Z into the final output, w, considering mask and replace/merge
            write_with_opt_mask_1D_sparse_dense(w, z, mask, outp);
        }

        //**********************************************************************
        //**********************************************************************
        //**********************************************************************

        //**********************************************************************
        /// Implementation of 4.3.3 mxv: A' * u
        //**********************************************************************
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename SemiringT,
                 typename AMatrixT,
                 typename UVectorT>
        inline void mxv(WVectorT                      &w,
                        MaskT                   const &mask,
                        AccumT                  const &accum,
                        SemiringT                      op,
                        TransposeView<AMatrixT> const &AT,
                        UVectorT                const &u,
                        OutputControlEnum              outp)
        {
            GRB_LOG_VERBOSE("w<M,z> := A' +.* u");
            auto const &A(AT.m_mat);

            // =================================================================
            // Use axpy approach with the semi-ring.
            using TScalarType = typename SemiringT::result_type;
            std::vector<std::tuple<IndexType, TScalarType> > t;

            if ((A.nvals() > 0) && (u.nvals() > 0))
            {
                for (IndexType row_idx = 0; row_idx < u.size(); ++row_idx)
                {
                    if (u.hasElement(row_idx) && !A[row_idx].empty())
                    {
                        axpy(t, op, u.extractElement(row_idx), A[row_idx]);
                    }
                }
            }

            // =================================================================
            // Accumulate into Z
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
    } // backend
} // grb
