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
#include <atomic>

//****************************************************************************

namespace grb
{
    namespace backend
    {
        //**********************************************************************
        /// Implementation for mxv with GKC Matrix and GKC Sparse Vector: A * u
        // Designed for general case of masking and with a null or non-null accumulator.
        // w = [!m.*w]+U {[m.*w]+m.*(A*u)}
        // DOT PRODUCT
        //**********************************************************************
        // w, mask, and u are vectors. A is a matrix.
        template <typename AccumT,
                  typename MaskT,
                  typename SemiringT,
                  typename ScalarT>
        inline void mxv(GKCSparseVector<ScalarT> &w,
                        MaskT const &mask,
                        AccumT const &accum,
                        SemiringT op,
                        GKCMatrix<ScalarT> const &A,
                        GKCSparseVector<ScalarT> const &u,
                        OutputControlEnum outp)
        {
            // Assumption: we have a non-null mask
            GRB_LOG_VERBOSE("w<M,z> := A +.* u");
            // w = [!m.*w]+U {[m.*w]+m.*(A*u)}
            using TScalarType = typename SemiringT::result_type;

            // Accumulate is null, clear on replace due to null mask (from signature):
            // if constexpr (std::is_same_v<AccumT, grb::NoAccumulate>){
            //     if (outp == REPLACE)
            //     {
            //         w.clear();
            //     }
            // }

            // If we have a mask and the output control is REPLACE, delete
            // pre-existing elements not in the mask
            if (outp == REPLACE)
            {
                for (auto idx = 0; idx < w.size(); idx++)
                {
                    if (!mask.hasElement(idx) || !mask.extractElement(idx))
                    {
                        w.boolRemoveElement(idx);
                    }
                }
            } // Otherwise, if Merging, just leave values in place.

            if ((A.nvals() > 0) && (u.nvals() > 0))
            {
                // Iterate over set values in mask, matching with
                // non-empty entries in u and rows in A
                // for (auto && row_idx : mask.getIndices())
                for (auto idx = 0; idx < mask.nvals(); idx++)
                {
                    auto row_idx = *(mask.idxBegin() + idx);
                    auto AIst = A.idxBegin(row_idx);
                    auto AInd = A.idxEnd(row_idx);
                    auto AWst = A.wgtBegin(row_idx);
                    auto AWnd = A.wgtEnd(row_idx);
                    auto UIst = u.getIndices().begin();
                    auto UInd = u.getIndices().end();
                    auto UWst = u.getWeights().begin();
                    auto UWnd = u.getWeights().end();
                    // Do dot product here, into w directly
                    bool value_set(false);
                    TScalarType sum;
                    while (AIst < AInd && UIst < UInd)
                    {
                        if (*AIst == *UIst)
                        {
                            if (value_set)
                            {
                                sum = op.add(sum, op.mult(*AWst, *UWst));
                            }
                            else
                            {
                                sum = op.mult(*AWst, *UWst);
                                value_set = true;
                            }
                            AIst++;
                            AWst++;
                            UIst++;
                            UWst++;
                        }
                        else if (*AIst < *UIst)
                        {
                            AIst++;
                            AWst++;
                        }
                        else
                        {
                            UIst++;
                            UWst++;
                        }
                    }
                    // Handle accumulation:
                    if (value_set)
                    {
                        if constexpr (std::is_same_v<AccumT, NoAccumulate>)
                        {
                            w.setElement(row_idx, sum);
                        }
                        else // Accumulate
                        {
                            w.mergeSetElement(row_idx, sum, accum);
                        }
                    }
                    else
                    {
                        if constexpr (std::is_same_v<AccumT, NoAccumulate>)
                        {
                            w.boolRemoveElement(row_idx);
                        }
                    }
                } // End of fused mxv loop
            }     // End of early exit
            // w.sortSelf();
        }

        //**********************************************************************
        //**********************************************************************
        //**********************************************************************

        //**********************************************************************
        /// Implementation for mxv with GKC Matrix and GKC Sparse Vector: A' * u
        // Designed for general case of masking and with a non-null accumulator.
        // w = [!m.*w]+U {[m.*w]+m.*(A'*u)}
        // AXPY (Ax + y) approach
        //**********************************************************************
        template <
            typename MaskT,
            typename AccumT,
            typename SemiringT,
            typename ScalarT>
        inline void mxv(GKCSparseVector<ScalarT> &w,
                        MaskT const &mask,
                        AccumT const &accum,
                        SemiringT op,
                        TransposeView<GKCMatrix<ScalarT>> const &AT,
                        GKCSparseVector<ScalarT> const &u,
                        OutputControlEnum outp)
        {
            GRB_LOG_VERBOSE("w<M,z> := A' +.* u");
            // w = [!m.*w]+U {[m.*w]+m.*(A'*u)}
            auto const &A(AT.m_mat);

            // =================================================================
            // If we have a mask and the output control is REPLACE, delete
            // pre-existing elements not in the mask
            if (outp == REPLACE)
            {
                if constexpr (std::is_same_v<AccumT, grb::NoAccumulate>){
                    w.clear();
                }
                else {
                    for (auto idx = 0; idx < w.size(); idx++)
                    {
                        if (!mask.hasElement(idx) || !mask.extractElement(idx))
                        {
                            w.boolRemoveElement(idx);
                            // todo: remove at internal index.
                        }
                    }
                }
            } // Otherwise, if Merging, just leave values in place.

            // Use axpy approach with the semi-ring.
            using TScalarType = typename SemiringT::result_type;
            // Create tmp vector to place computed values
            GKCSparseVector<TScalarType> t(w.size());

            // if constexpr (!std::is_same_v<AccumT, grb::NoAccumulate>){
            //     flags.resize(w.nvals());
            //     // Create a bool vector to coordinate accum vs add op
            //     // Could be used to coordinate CAS on the vector w?
            // }

            if ((A.nvals() > 0) && (u.nvals() > 0))
            {
                // Create flags for locking output locations
                std::vector<char> flags(w.size(), 0);
                for (auto &&idx : mask.getIndices())
                {
                    flags[idx] = 1;
                }
                auto UIst = u.idxBegin();
                auto UInd = u.idxEnd();
                auto UWst = u.wgtBegin();
                for (; UIst < UInd; UIst++, UWst++)
                {
                    auto AIst = A.idxBegin(*UIst);
                    auto AWst = A.wgtBegin(*UIst);
                    for (; AIst < A.idxEnd(*UIst); AIst++, AWst++)
                    {
                        if (flags[*AIst] != 0) // If allowed by the mask
                        {
                            auto res = op.mult(*UWst, *AWst);
                            // CAS LOOP on flag
                            // const char one(1);
                            // const char two(2);
                            // while (__sync_bool_compare_and_swap(flags.data() + *AIst, one, two))
                            // {
                            // };
                            // Merge element with additive operation
                            t.mergeSetElement(*AIst, res, grb::AdditiveMonoidFromSemiring(op));
                            // Release CAS LOOP on flag
                            // flags[*AIst] = one;
                        }
                    }
                } // End loop over output vector
                // Merge/accumulate if needed
                if constexpr (!std::is_same_v<AccumT, grb::NoAccumulate>)
                {
                    auto TIst = t.idxBegin();
                    auto TInd = t.idxEnd();
                    auto TWst = t.wgtBegin();
                    while (TIst != TInd)
                    {
                        // std::cout << *TIst << std::endl;
                        // if (w.hasElement(*TIst)){
                        //     std::cout << "Merging " << w.extractElement(*TIst) << " and " << *TWst << std::endl;
                        // }
                        w.mergeSetElement(*TIst, *TWst, accum);
                        TIst++;
                        TWst++;
                    }
                }
                else
                {
                    auto TIst = t.idxBegin();
                    auto TInd = t.idxEnd();
                    auto TWst = t.wgtBegin();
                    while (TIst != TInd)
                    {
                        w.setElement(*TIst, *TWst);
                        TIst++;
                        TWst++;
                    }
                }
            }
            w.setUnsorted();
        }
    } // backend
} // grb
