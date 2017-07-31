/*
 * Copyright (c) 2017 Carnegie Mellon University.
 * All Rights Reserved.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS," WITH NO WARRANTIES WHATSOEVER. CARNEGIE
 * MELLON UNIVERSITY EXPRESSLY DISCLAIMS TO THE FULLEST EXTENT PERMITTED BY
 * LAW ALL EXPRESS, IMPLIED, AND STATUTORY WARRANTIES, INCLUDING, WITHOUT
 * LIMITATION, THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE, AND NON-INFRINGEMENT OF PROPRIETARY RIGHTS.
 *
 * This Program is distributed under a BSD license.  Please see LICENSE file or
 * permission@sei.cmu.edu for more information.  DM-0002659
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
        inline void transpose(CMatrixT       &C,
                              MaskT    const &mask,
                              AccumT          accum,
                              AMatrixT const &A,
                              bool            replace_flag = false)
        {
            check_dimensions(C, "C", mask, "mask");
            if ((C.nrows() != A.ncols()) || (C.ncols() != A.nrows()))
            {
                throw DimensionException("Transpose input dimension");
            }

            typedef typename AMatrixT::ScalarType                   AScalarType;
            typedef std::vector<std::tuple<IndexType,AScalarType> > ARowType;

            typedef typename CMatrixT::ScalarType                   CScalarType;
            typedef std::vector<std::tuple<IndexType,CScalarType> > CRowType;

            IndexType nrows(A.nrows());
            IndexType ncols(A.ncols());

            //std::cerr << ">>> C in <<< " << std::endl;
            //std::cerr << C << std::endl;

            //std::cerr << ">>> A in <<< " << std::endl;
            //std::cerr << A << std::endl;

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

            //std::cerr << ">>> T <<< " << std::endl;
            //std::cerr << T << std::endl;

            // =================================================================
            // Accumulate T via C into Z

            LilSparseMatrix<CScalarType> Z(ncols, nrows);
            ewise_or_opt_accum(Z, C, T, accum);

            //std::cerr << ">>> Z <<< " << std::endl;
            //std::cerr << Z << std::endl;

            // =================================================================
            // Copy Z into the final output considering mask and replace
            write_with_opt_mask(C, Z, mask, replace_flag);

            //std::cerr << ">>> C <<< " << std::endl;
            //std::cerr << C << std::endl;
        }

    }
}



#endif //GB_SEQUENTIAL_SPARSE_APPLY_HPP
