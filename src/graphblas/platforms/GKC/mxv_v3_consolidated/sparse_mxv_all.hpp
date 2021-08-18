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
#ifdef INST_TIMING_MVX
            Timer<std::chrono::steady_clock, std::chrono::microseconds> my_timer;
            Timer<std::chrono::steady_clock, std::chrono::microseconds> my_timer2;
            size_t dots = 0;
            double dots_time = 0;
            my_timer2.start();
#endif
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
            } // Other side is handled at bottom of main for loop.

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
                        /// @todo: can a null mask be complemented?
                    }
                    else
                    {
                        // Handle two cases: normal mask, complement
                        if constexpr (is_complement_v<MaskT>)
                        {
                            auto mask_inner = mask.m_vec;
                            using MaskTypeT = typename decltype(mask_inner)::ScalarType;
                            MaskTypeT test_val;
                            // do_compute = !mask_inner.hasElement(row_idx);
                            do_compute = !mask_inner.boolExtractElement(row_idx, test_val);
                            do_compute |= !(bool)test_val;
                        }
                        else // Just a standard vector, no view
                        {
                            using MaskTypeT = typename MaskT::ScalarType;
                            MaskTypeT test_val;
                            // do_compute = mask.hasElement(row_idx);
                            do_compute = mask.boolExtractElement(row_idx, test_val);
                            do_compute &= (bool)test_val;
                        }
                    }
                    if (do_compute)
                    {
                        // Dot product begin
#ifdef INST_TIMING_MVX
                        my_timer.start();
#endif
                        auto AIst = A.idxBegin(row_idx);
                        auto AInd = A.idxEnd(row_idx);
                        auto AWst = A.wgtBegin(row_idx);
                        //                        auto AWnd = A.wgtEnd(row_idx);
                        //                        auto UIst = u.idxBegin();
                        //                        auto UInd = u.idxEnd();

                        //                        auto UWst = u.wgtBegin();
                        //                        auto UWnd = u.wgtEnd();
                        // Do dot product here, into w directly
                        bool value_set = false;
                        TScalarType sum;

                        for (auto AIst = A.idxBegin(row_idx);
                             AIst != AInd;
                             ++AIst, ++AWst)
                        {
                            if (u.hasElement(*AIst))
                            {
                                auto uw = u[*AIst];
                                if (value_set)
                                    sum = op.add(sum, op.mult(*AWst, uw));
                                else
                                {
                                    sum = op.mult(*AWst, uw);
                                    value_set = true;
                                }
                            }
                        }

                        // Dot end
#ifdef INST_TIMING_MVX
                        my_timer.stop();
                        dots_time += my_timer.elapsed();
                        dots++;
#endif
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
                    else // This side is only taken when not in (complemented) mask
                    {
                        // if we have accumulate, outp is REPLACE, and mask is false:
                        if constexpr (!std::is_same_v<AccumT, grb::NoAccumulate>)
                        {
                            if (outp == REPLACE)
                                w.boolRemoveElement(row_idx);
                        }
                    }
                } // End of fused mxv loop
            }     // End of early exit
            else  // Need to cleanup values not deleted...
            {
                if constexpr (!std::is_same_v<MaskT, grb::NoMask> &&
                              !std::is_same_v<AccumT, grb::NoAccumulate>)
                // Have accumulate op AND a mask
                {
                    if (outp == REPLACE)
                    {
                        // If we have a mask and the output control is REPLACE, delete
                        // pre-existing elements not in the mask
                        for (auto idx = 0; idx < w.size(); idx++)
                        {
                            bool remove;
                            if constexpr (is_complement_v<MaskT>)
                            {
                                auto mask_inner = mask.m_vec;
                                using MaskTypeT = typename decltype(mask_inner)::ScalarType;
                                MaskTypeT val;
                                remove = mask_inner.boolExtractElement(idx, val);
                                remove &= val; // Reverse for later logic
                            }
                            else // Just a standard vector, no view
                            {
                                using MaskTypeT = typename MaskT::ScalarType;
                                MaskTypeT val;
                                remove = !mask.boolExtractElement(idx, val);
                                remove |= !val;
                            }
                            if (remove) // Remove if NOT in the mask.
                                w.boolRemoveElement(idx);
                        }
                    } 
                }
            }
#ifdef INST_TIMING_MVX
            my_timer2.stop();
            std::cerr << my_timer2.elapsed() << ", " << dots << ", " << dots_time << std::endl;
#endif
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

            vxm(w, mask, accum, op, u, A, outp);
            /*
	    
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
	    */
        }
    } // backend
} // grb
