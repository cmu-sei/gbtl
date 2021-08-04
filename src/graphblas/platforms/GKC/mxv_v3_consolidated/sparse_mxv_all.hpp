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

#include "sparse_helper_proto.hpp"

#include "../test/Timer.hpp"

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
    Timer<std::chrono::steady_clock, std::chrono::microseconds> my_timer;
    Timer<std::chrono::steady_clock, std::chrono::microseconds> my_timer2;
    size_t dots = 0;
    double dots_time = 0;

    my_timer2.start();
            // Assumption: we have a non-null mask
            GRB_LOG_VERBOSE("w<M,z> := A +.* u");
            // w = [!m.*w]+U {[m.*w]+m.*(A*u)}
            using TScalarType = typename SemiringT::result_type;

            // Accumulate is null, clear on replace due to null mask (from signature):
            if constexpr (std::is_same_v<AccumT, grb::NoAccumulate>)
            {
                if constexpr (std::is_same_v<MaskT, grb::NoMask>)
                {
                    w.clear();
                }
                else // Have mask and no accum
                {
                    if (outp == REPLACE)
                    {
                        w.clear();
                    }
                }
            }
            else
            {
                if constexpr (!std::is_same_v<MaskT, grb::NoMask>)
                // Have accumulate op AND a mask
                {
                    if (outp == REPLACE)
                    {
#define V2_REPLACE_DELETE 1
#if V2_REPLACE_DELETE
                        intersect_delete(mask, w);
#else
                        // If we have a mask and the output control is REPLACE, delete
                        // pre-existing elements not in the mask
                        for (auto idx = 0; idx < w.size(); idx++)
                        {
                            using MaskTypeT = typename MaskT::ScalarType;
                            MaskTypeT val;
                            if (!mask.boolExtractElement(idx, val))
                            {
                                if (!val)
                                    w.boolRemoveElement(idx);
                            }
                        }
#endif
                    } // Otherwise, if Merging, just leave values in place.
                }
            }

            if ((A.nvals() > 0) && (u.nvals() > 0))
            {
                // Iterate over set values in mask, matching with
                // non-empty entries in u and rows in A
                // for (auto && row_idx : mask.getIndices())
                for (IndexType row_idx = 0; row_idx < A.nrows(); row_idx++)
                {
                    bool do_compute;
                    if constexpr (std::is_same_v<MaskT, grb::NoMask>)
                    {
                        do_compute = true;
                    }
                    else
                    {
                        do_compute = mask.hasElement(row_idx);
                    }
                    if (do_compute)
                    {
                        // Dot product begin
                        my_timer.start();
                        auto AIst = A.idxBegin(row_idx);
                        auto AInd = A.idxEnd(row_idx);
                        auto AWst = A.wgtBegin(row_idx);
                        auto AWnd = A.wgtEnd(row_idx);
                        auto UIst = u.idxBegin();
                        auto UInd = u.idxEnd();
                        auto UWst = u.wgtBegin();
                        auto UWnd = u.wgtEnd();
                        // Do dot product here, into w directly
                        bool value_set = false;
                        TScalarType sum;
                        while (AIst < AInd && UIst < UInd)
                        {
                            if (*AIst == *UIst)
                            {
                                sum = op.mult(*AWst, *UWst);
                                value_set = true;
                                AIst++;
                                AWst++;
                                UIst++;
                                UWst++;
                                break;
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
                        while (AIst < AInd && UIst < UInd)
                        {
                            if (*AIst == *UIst)
                            {
                                sum = op.add(sum, op.mult(*AWst, *UWst));
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
                        // Dot end
                        my_timer.stop();
                        dots_time += my_timer.elapsed();
                        dots ++;
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
                    }
                    // Not needed, done above:
                    // else // Not in mask
                    // {
                    //     if constexpr (!std::is_same_v<AccumT, NoAccumulate>){
                    //         if (outp == REPLACE)
                    //         { // Remove the element in w
                    //             w.boolRemoveElement(row_idx);
                    //         }
                    //     }
                    // }
                } // End of fused mxv loop
            }     // End of early exit
            // w.sortSelf();
            my_timer2.stop();
            std::cerr << my_timer2.elapsed() << ", " << dots << ", " << dots_time << std::endl;

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
            // Use axpy approach with the semi-ring.
            using TScalarType = typename SemiringT::result_type;
            // Create tmp vector to place computed values
            GKCSparseVector<TScalarType> t(w.size());

            // Decision: densify mask for easy reference
            std::vector<bool> mask_vec;
            if constexpr (!std::is_same_v <MaskT, grb::NoMask>)
            {
                mask_vec = std::vector<bool>(mask.size());
                for (auto itr = mask.idxBegin(); itr < mask.idxEnd(); itr++)
                {
                    mask_vec[*itr] = 1;
                }
            }

            // Accumulate is null, clear on replace due to null mask (from signature):
            if constexpr (std::is_same_v<AccumT, grb::NoAccumulate>)
            {
                if constexpr (std::is_same_v<MaskT, grb::NoMask>)
                {
                    w.clear();
                }
                else // Have mask and no accum
                {
                    if (outp == REPLACE)
                    {
                        w.clear();
                    }
                }
            }
            else if constexpr (!std::is_same_v<MaskT, grb::NoMask>)
            // Have accumulate op AND a mask
            {
                if (outp == REPLACE)
                {
                    // If we have a mask and the output control is REPLACE, delete
                    // pre-existing elements not in the mask
                    for (auto idx = 0; idx < mask_vec.size(); idx++)
                    {
                        if (!mask_vec[idx]) // Can reverse for complement?
                        {
                            w.boolRemoveElement(idx);
                        }
                    }
                } // Otherwise, if Merging, just leave values in place.
            }

            if ((A.nvals() > 0) && (u.nvals() > 0))
            {
                // Create flags for locking output locations
                // std::vector<char> flags(w.size(), 0);
                // for (auto &&idx : mask.getIndices())
                // {
                // flags[idx] = 1;
                // }
                auto UIst = u.idxBegin();
                auto UInd = u.idxEnd();
                auto UWst = u.wgtBegin();
                for (; UIst < UInd; UIst++, UWst++)
                {
                    auto AIst = A.idxBegin(*UIst);
                    auto AWst = A.wgtBegin(*UIst);
                    for (; AIst < A.idxEnd(*UIst); AIst++, AWst++)
                    {
                        if (std::is_same_v<MaskT, grb::NoMask> || mask_vec[*AIst] != 0) // If allowed by the mask
                        {
                            auto res = op.mult(*UWst, *AWst);
                            // CAS LOOP on flag
                            // const char one(1);
                            // const char two(2);
                            // while (__sync_bool_compare_and_swap(flags.data()+*AIst, one, two)){};
                            // Merge element with additive operation
                            t.mergeSetElement(*AIst, res, grb::AdditiveMonoidFromSemiring(op));
                            // Release CAS LOOP on flag
                            // flags[*AIst] = one;
                        }
                    }
                } // End loop over input vector
                // Merge/accumulate if needed
                if constexpr (!std::is_same_v<AccumT, grb::NoAccumulate>)
                {
                    auto TIst = t.idxBegin();
                    auto TInd = t.idxEnd();
                    auto TWst = t.wgtBegin();
                    while (TIst != TInd)
                    {
                        w.mergeSetElement(*TIst, *TWst, accum);
                        TIst++;
                        TWst++;
                    }
                }
                else // No accumulate, just merge or replace
                {
                    if (outp == REPLACE)
                    { // Set and forget
                        if constexpr (std::is_same_v<ScalarT, TScalarType>)
                        {
                            //w.swap(t);
                            w = std::move(t);
                        }
                        else
                        {
                            w = t;
                        }
                    }
                    else // Merge into existing
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
            }
            w.setUnsorted();
        }
    } // backend
} // grb
