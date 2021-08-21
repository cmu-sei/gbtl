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
                 typename AMatrixT>
        inline void vxm(WVectorT          &w,
                        MaskT       const &mask,
                        AccumT      const &accum,
                        SemiringT          op,
                        UVectorT    const &u,
                        AMatrixT    const &A,
                        OutputControlEnum  outp)
        {
            GRB_LOG_VERBOSE("w<M,z> := u +.* A");
            backend::mxv(w, mask, accum, op, grb::TransposeView<AMatrixT>(A), u, outp);
        }

        //**********************************************************************
        //**********************************************************************
        //**********************************************************************

        //********************************************************************
        /// Implementation of 4.3.2 vxm: u * A'
        //********************************************************************
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename SemiringT,
                 typename AMatrixT,
                 typename UVectorT>
        inline void vxm(WVectorT                      &w,
                        MaskT                   const &mask,
                        AccumT                  const &accum,
                        SemiringT                      op,
                        UVectorT                const &u,
                        TransposeView<AMatrixT> const &AT,
                        OutputControlEnum              outp)
        {
            GRB_LOG_VERBOSE("w<M,z> := u +.* A'");
            backend::mxv(w, mask, accum, op, AT.m_mat, u, outp);
        }

    } // backend
} // grb
