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

//****************************************************************************

namespace grb
{
    namespace backend
    {

        //**********************************************************************
        /// Implementation for mxv with GKC Matrix and GKC Sparse Vector: A * u
        // Designed for case of no mask and null or non-null accumulator.
        // w = w + (A*u)
        //**********************************************************************
        // w, mask, and u are vectors. A is a matrix.
        template <typename AccumT,
                  typename SemiringT,
                  typename ScalarT>
        inline void mxv(GKCSparseVector<ScalarT> &w,
                        grb::NoMask const &mask,
                        AccumT const &accum,
                        SemiringT op,
                        GKCMatrix<ScalarT> const &A,
                        GKCSparseVector<ScalarT> const &u,
                        OutputControlEnum outp)
        {
            GRB_LOG_VERBOSE("w<M,z> := A +.* u");
            // std::cout << "Hi! DOT" << std::endl;
            // w = w + (A*u)

            // Accumulate is null, clear on replace due to null mask (from signature):
            if constexpr (std::is_same_v<AccumT, grb::NoAccumulate>) w.clear();

            // =================================================================
            // Do the basic dot-product work with the semi-ring.
            using TScalarType = typename SemiringT::result_type;
            // Generate a cast type for the accumulator, if present:

            using ZType = decltype(accum(
                std::declval<ScalarT>(),
                std::declval<ScalarT>()));
            using ZScalarType = typename std::conditional<
                std::is_same_v<AccumT, grb::NoAccumulate>, 
                ScalarT,
                ZType>;



            // Assume input is sorted
            // u.sortSelf();

            if ((A.nvals() > 0) && (u.nvals() > 0))
            {
                for (IndexType row_idx = 0; row_idx < w.size(); ++row_idx) 
                {
                    auto AIst = A.idxBegin(row_idx);
                    auto AInd = A.idxEnd(row_idx);
                    auto AWst = A.wgtBegin(row_idx);
                    auto AWnd = A.wgtEnd(row_idx);
                    auto UIst = u.idxBegin();
                    auto UInd = u.idxEnd();
                    auto UWst = u.wgtBegin();
                    auto UWnd = u.wgtEnd();
                    // Do dot product here, into t
                    bool value_set(false);
                    TScalarType sum;
                    //if (AIst < AInd && UIst < UInd) std::cout << "row idx: " << row_idx << std::endl;
                    while (AIst < AInd && UIst < UInd)
                    {                            
                        if (*AIst == *UIst)
                        {
                            //std::cout << "AIdx: " << *AIst << " UIdx: " << *UIst << std::endl;
                            //std::cout << "AWgt: " << *AWst<< " UWgt: " << *UWst << std::endl;
                            if (value_set)
                            {
                                sum = op.add(sum, op.mult(*AWst, *UWst));
                            }
                            else 
                            {                                
                                sum = op.mult(*AWst, *UWst);
                                value_set = true;
                            }
                            AIst++; AWst++;
                            UIst++; UWst++;
                        }
                        else if (*AIst < *UIst)
                        {                                
                            AIst++; AWst++;
                        }
                        else 
                        {
                            UIst++; UWst++;
                        }
                        //if (value_set) std::cout << "Sum: " << sum << " ";
                    }
                    /// @todo: outputs control enum, masking, etc...
                    // Handle accumulation:
                    if (value_set) {
                        if constexpr (std::is_same_v<AccumT, NoAccumulate>)
                        {
                            w.setElement(row_idx, sum); // make sure there is an accumulate cast?
                        } 
                        else
                        {
                            w.mergeSetElement(row_idx, sum, accum);
                        }
                    }  
                    //std::cout << std::endl;
                }
            }
            // w.sortSelf();
        }

        //**********************************************************************
        //**********************************************************************
        //**********************************************************************

        //**********************************************************************
        /// Implementation of mxv for GKC Matrix and Sparse Vector: A' * u
        // Designed for case of no mask and null or non-null accumulator.
        // w = w + (A'*u)
        //**********************************************************************
        template <
                  typename AccumT,
                  typename SemiringT,
                  typename ScalarT>
        inline void mxv(GKCSparseVector<ScalarT> &w,
                        grb::NoMask const &mask,
                        AccumT const &accum,
                        SemiringT op,
                        TransposeView<GKCMatrix<ScalarT>> const &AT,
                        GKCSparseVector<ScalarT> const &u,
                        OutputControlEnum outp)
        {
            GRB_LOG_VERBOSE("w<M,z> := A' +.* u");
            // std::cout << "Hi! AXPY" << std::endl;
            // w = w+(A'*u)
            auto const &A(AT.m_mat);

            // =================================================================
            // Accumulate is null, clear on replace due to null mask (from signature):
            if constexpr (std::is_same_v<AccumT, grb::NoAccumulate>) w.clear();
            
            // Use axpy approach with the semi-ring.
            using TScalarType = typename SemiringT::result_type;
            GKCSparseVector<TScalarType> t(w.size());

            if ((A.nvals() > 0) && (u.nvals() > 0))
            {
                // Create flags for outputs
                // std::vector<char> flags(w.size(), 0);

                auto UIst = u.idxBegin();
                auto UInd = u.idxEnd();
                auto UWst = u.wgtBegin();
                for ( ; UIst < UInd; UIst++, UWst++)
                {
                    auto AIst = A.idxBegin(*UIst); 
                    auto AWst = A.wgtBegin(*UIst);
                    auto AInd = A.idxEnd(*UIst);
                    for ( ; AIst < AInd; AIst++, AWst++)
                    {
                        // const char one(1);
                        // const char two(2);
                        auto res = op.mult(*UWst, *AWst);
                        // CAS LOOP on flag
                        // while (__sync_bool_compare_and_swap(flags.data()+*AIst, one, two)){};
                        // Merge element with additive operation
                        // TODO: make inserts for mergeSetElement thread safe!
                        t.mergeSetElement(*AIst, res, grb::AdditiveMonoidFromSemiring(op));
                        // Release CAS LOOP on flag
                        // flags[*AIst] = one;
                    }
                } // End loop over output vector
                // Accumulate if needed
                if constexpr (!std::is_same_v<AccumT, grb::NoAccumulate>)
                { // todo: replace with specialized GKCVector merge
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
                else
                {
                    // Set w to equal t (copy)
                    // We can do this because the mask is empty, and no accumulate means 
                    // that any old values in w should be discarded.
                    // auto TIst = t.idxBegin();
                    // auto TInd = t.idxEnd();
                    // auto TWst = t.wgtBegin();
                    // while (TIst != TInd)
                    // {
                    //     w.setElement(*TIst, *TWst);
                    //     TIst++;
                    //     TWst++;
                    // }
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
            }
            w.setUnsorted();
        }

    } // backend
} // grb