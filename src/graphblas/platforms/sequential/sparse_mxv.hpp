/*
 * GraphBLAS Template Library, Version 2.0
 *
 * Copyright 2018 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors. All Rights Reserved.
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
 * RIGHTS..
 *
 * Released under a BSD (SEI)-style license, please see license.txt or contact
 * permission@sei.cmu.edu for full terms.
 *
 * This release is an update of:
 *
 * 1. GraphBLAS Template Library (GBTL)
 * (https://github.com/cmu-sei/gbtl/blob/1.0.0/LICENSE) Copyright 2015 Carnegie
 * Mellon University and The Trustees of Indiana. DM17-0037, DM-0002659
 *
 * DM18-0559
 */

/**
 * Implementation of all sparse mxv for the sequential (CPU) backend.
 */

#ifndef GB_SEQUENTIAL_SPARSE_MXV_HPP
#define GB_SEQUENTIAL_SPARSE_MXV_HPP

#pragma once

#include <functional>
#include <utility>
#include <vector>
#include <iterator>
#include <iostream>
#include <graphblas/algebra.hpp>

#include "sparse_helpers.hpp"


//****************************************************************************

namespace GraphBLAS
{
    namespace backend
    {
        //********************************************************************
        /// Implementation of 4.3.3 mxv: Matrix-Vector variant
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename SemiringT,
                 typename AMatrixT,
                 typename UVectorT>
        inline Info mxv(WVectorT        &w,
                        MaskT     const &mask,
                        AccumT           accum,
                        SemiringT        op,
                        AMatrixT  const &A,
                        UVectorT  const &u,
                        bool             replace_flag = false)
        {
            // =================================================================
            // Do the basic dot-product work with the semi-ring.
            typedef typename SemiringT::result_type D3ScalarType;
            typedef typename AMatrixT::ScalarType AScalarType;
            typedef std::vector<std::tuple<IndexType,AScalarType> >  ARowType;

            std::vector<std::tuple<IndexType, D3ScalarType> > t;

            if ((A.nvals() > 0) && (u.nvals() > 0))
            {
                auto u_contents(u.getContents());
                for (IndexType row_idx = 0; row_idx < w.size(); ++row_idx)
                {
                    ARowType const &A_row(A.getRow(row_idx));

                    if (!A_row.empty())
                    {
                        D3ScalarType t_val;
                        if (dot(t_val, A_row, u_contents, op))
                        {
                            t.push_back(std::make_tuple(row_idx, t_val));
                        }
                    }
                }
            }

            // =================================================================
            // Accumulate into Z
            typedef typename std::conditional<std::is_same<AccumT, NoAccumulate>::value,
                                              D3ScalarType,
                                              typename AccumT::result_type>::type ZScalarType;
            std::vector<std::tuple<IndexType, ZScalarType> > z;
            ewise_or_opt_accum_1D(z, w, t, accum);

            // =================================================================
            // Copy Z into the final output, w, considering mask and replace
            write_with_opt_mask_1D(w, z, mask, replace_flag);
            return SUCCESS;
        }

    } // backend
} // GraphBLAS

#endif
