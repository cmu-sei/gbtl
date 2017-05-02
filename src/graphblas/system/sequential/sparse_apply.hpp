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
 * Implementation of the sparse matrix extract function.
 */

#ifndef GB_SEQUENTIAL_SPARSE_APPLY_HPP
#define GB_SEQUENTIAL_SPARSE_APPLY_HPP

#pragma once

#include <functional>
#include <utility>
#include <vector>
#include <iterator>
#include <iostream>
#include <graphblas/types.hpp>
#include <graphblas/exceptions.hpp>
#include <graphblas/accum.hpp>
#include <graphblas/algebra.hpp>

#include "sparse_helpers.hpp"
#include "LilSparseMatrix.hpp"

//******************************************************************************

namespace GraphBLAS
{
    namespace backend
    {

        //**********************************************************************
        // Implementation of 4.3.8.2 Matrix variant of Apply

        template<typename CMatrixT,
                 typename MaskT,
                 typename AccumT,
                 typename UnaryFunctionT,
                 typename AMatrixT>
        inline void apply(CMatrixT             &C,
                          MaskT          const &mask,
                          AccumT                accum,
                          UnaryFunctionT        op,
                          AMatrixT       const &A,
                          bool                  replace_flag = false)
        {
            check_dimensions(C, "C", A, "A");
            check_dimensions(C, "C", mask, "mask");

            typedef typename CMatrixT::ScalarType                   AScalarType;
            typedef std::vector<std::tuple<IndexType,AScalarType> > ARowType;

            typedef typename CMatrixT::ScalarType                   CScalarType;
            typedef std::vector<std::tuple<IndexType,CScalarType> > CRowType;

            typedef typename UnaryFunctionT::result_type            TScalarType;
            typedef std::vector<std::tuple<IndexType,TScalarType> > TRowType;


            IndexType nrows(A.nrows());
            IndexType ncols(A.ncols());

            //std::cerr << ">>> C in <<< " << std::endl;
            //std::cerr << C << std::endl;

            //std::cerr << ">>> A in <<< " << std::endl;
            //std::cerr << A << std::endl;

            // =================================================================
            // Apply the unary operator from A into T.
            // This is really the guts of what makes this special.

            LilSparseMatrix<TScalarType> T(nrows, ncols);

            ARowType a_row;
            CRowType t_row;

            IndexType a_idx;
            AScalarType a_val;

            for (IndexType row_idx = 0; row_idx < A.nrows(); ++row_idx)
            {
                a_row = A.getRow(row_idx);
                if (!a_row.empty())
                {
                    t_row.clear();
                    auto row_iter = a_row.begin();
                    while (row_iter != a_row.end())
                    {
                        std::tie(a_idx, a_val) = *row_iter;
                        TScalarType t_val = static_cast<TScalarType>(op(a_val));
                        t_row.push_back(std::make_tuple(a_idx,t_val));
                        ++row_iter;
                    }

                    if (!t_row.empty())
                        T.setRow(row_idx, t_row);
                }
            }

            //std::cerr << ">>> T <<< " << std::endl;
            //std::cerr << T << std::endl;

            // =================================================================
            // Accumulate T via C into Z

            LilSparseMatrix<CScalarType> Z(nrows, ncols);
            ewise_or_opt_accum(Z, C, T, accum);

            //std::cerr << ">>> Z <<< " << std::endl;
            //std::cerr << Z << std::endl;

            // =================================================================
            // Copy Z into the final output considering mask and replace
            write_with_opt_mask(C, Z, mask, replace_flag);

            ///std::cerr << ">>> C <<< " << std::endl;
            //std::cerr << C << std::endl;
        };

    }
}



#endif //GB_SEQUENTIAL_SPARSE_APPLY_HPP
