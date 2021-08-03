
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
        // Intersect and delete from second GKC vector. Requires sorted vectors to not suck
        template <typename ScalarT, typename TMaskType>
        inline void intersect_delete(GKCSparseVector<TMaskType> const & mask, GKCSparseVector<ScalarT> & mute)
        {
            if (mask.nvals() == 0) 
            {
                mute.clear();
                return;
            }

            // assert(mute.isWeighted);
            // Ouch....
            if (!mute.isSorted()) mute.sortSelf();
            if (!mask.isSorted()) mask.sortSelf();

            // Get iterators for the vectors
            auto MaskIdxSt = mask.idxBegin();
            auto MaskIdxNd = mask.idxEnd();
            auto MaskWgtSt = mask.wgtBegin();
            auto MuteIdxSt = mute.idxBegin();
            auto MuteIdxNd = mute.idxEnd();
            auto MuteWgtSt = mute.wgtBegin();
            auto ExtraIdxSt = mute.idxBegin();
            auto ExtraWgtSt = mute.wgtBegin();
            // Merge the two vectors from beginning to end, overwriting w as the iterators progress.
            // Rule: if a value is NOT in the mask, leave it in w. 
            // Else: overwrite the element in w.
            size_t removed = 0;
            bool mask_weighted = mask.isWeighted();
            while (MaskIdxSt != MaskIdxNd && MuteIdxSt != MuteIdxNd)
            {
                if (*MuteIdxSt < *MaskIdxSt)
                // Not in the mask, delete it
                // Move mute iterators up to catch up to mask
                // Count one element deleted
                {
                    MuteIdxSt++;
                    MuteWgtSt++;
                    removed++;
                }
                else if (*MuteIdxSt > *MaskIdxSt)
                // Mute has index not in mask, delete it
                // Move mask iterators forward to catch up to mute
                // Count one element deleted
                {
                    MaskIdxSt++;
                    MaskWgtSt++;
                }
                else
                // Is in mask index space...
                // Check if mask item is "true"
                {
                    if (!mask_weighted || (mask_weighted && *MaskWgtSt))
                    {
                        // Keep element, which means copy from Mute*St 
                        // to Extra*St
                        *ExtraIdxSt = *MuteIdxSt;
                        *ExtraWgtSt = *MuteWgtSt;
                        MaskIdxSt++;
                        MaskWgtSt++;
                        MuteIdxSt++;
                        MuteWgtSt++; 
                        ExtraIdxSt++;
                        ExtraWgtSt++;
                    }
                    else
                    // Get rid of the element, since mask val was false 
                    {
                        MaskIdxSt++;
                        MaskWgtSt++;
                        MuteIdxSt++;
                        MuteWgtSt++; 
                        removed++;
                    }
                }
            }
            // Set proper size
            mute.truncateNVals(removed);
        }
    }
}