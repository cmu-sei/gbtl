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
 * Implementation of the sparse matrix apply function.
 */

#ifndef GB_SEQUENTIAL_SPARSE_TRANSPOSE_HPP
#define GB_SEQUENTIAL_SPARSE_TRANSPOSE_HPP

#pragma once

#include <functional>
#include <utility>
#include <vector>
#include <iterator>
#include <iostream>
#include <graphblas/types.hpp>
#include <graphblas/exceptions.hpp>
#include <graphblas/algebra.hpp>

#include "sparse_helpers.hpp"
#include "LilSparseMatrix.hpp"

//******************************************************************************

namespace GraphBLAS
{
    namespace backend
    {
        //**********************************************************************
        // Implementation of 4.3.10 Matrix transpose
        template<typename CMatrixT,
                 typename MaskT,
                 typename AccumT,
                 typename AMatrixT>
        inline Info transpose(CMatrixT       &C,
                              MaskT    const &mask,
                              AccumT          accum,
                              AMatrixT const &A,
                              bool            replace_flag = false)
        {
            typedef typename AMatrixT::ScalarType                   AScalarType;
            typedef std::vector<std::tuple<IndexType,AScalarType> > ARowType;

            typedef typename CMatrixT::ScalarType                   CScalarType;
            typedef std::vector<std::tuple<IndexType,CScalarType> > CRowType;

            IndexType nrows(A.nrows());
            IndexType ncols(A.ncols());

            // =================================================================
            // Apply the unary operator from A into T.
            // This is really the guts of what makes this special.

            /// @todo Do something different if A is TransposeView
            LilSparseMatrix<AScalarType> T(ncols, nrows);
            for (IndexType ridx = A.nrows(); ridx > 0; --ridx)
            {
                IndexType row_idx = ridx - 1;
                auto a_row = A.getRow(row_idx);
                for (auto elt = a_row.begin(); elt != a_row.end(); ++elt)
                {
                    T.setElement(std::get<0>(*elt), row_idx, std::get<1>(*elt));

                    /// @todo Would be nice to push_back on each row directly
                    //IndexType idx = std::get<0>(elt);
                    //T.getRow(idx).push_back(
                    //    std::make_tuple(row_idx, std::get<1>(elt)));
                }
            }

            GRB_LOG_VERBOSE("T: " << T);

            // =================================================================
            // Accumulate T via C into Z
            typedef typename std::conditional<
                std::is_same<AccumT, NoAccumulate>::value,
                AScalarType,
                typename AccumT::result_type>::type  ZScalarType;

            LilSparseMatrix<ZScalarType> Z(ncols, nrows);
            ewise_or_opt_accum(Z, C, T, accum);

            GRB_LOG_VERBOSE("Z: " << Z);

            // =================================================================
            // Copy Z into the final output considering mask and replace
            write_with_opt_mask(C, Z, mask, replace_flag);
            return SUCCESS;
        }
    }
}



#endif //GB_SEQUENTIAL_SPARSE_TRANSPOSE_HPP
