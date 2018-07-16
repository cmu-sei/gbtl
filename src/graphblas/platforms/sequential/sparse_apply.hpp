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
#include <graphblas/algebra.hpp>

#include "sparse_helpers.hpp"
#include "LilSparseMatrix.hpp"

//******************************************************************************

namespace GraphBLAS
{
    namespace backend
    {
        //**********************************************************************
        // Implementation of 4.3.8.1 Vector variant of Apply
        template<typename WScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename UnaryFunctionT,
                 typename UVectorT,
                 typename ...WTagsT>
        inline Info apply(
            GraphBLAS::backend::Vector<WScalarT, WTagsT...> &w,
            MaskT                                     const &mask,
            AccumT                                           accum,
            UnaryFunctionT                                   op,
            UVectorT                                  const &u,
            bool                                             replace_flag = false)
        {
            // =================================================================
            // Apply the unary operator from A into T.
            // This is really the guts of what makes this special.
            typedef typename UVectorT::ScalarType        UScalarType;
            typedef typename UnaryFunctionT::result_type TScalarType;
            std::vector<std::tuple<IndexType,TScalarType> > t_contents;

            if (u.nvals() > 0)
            {
                auto u_contents(u.getContents());
                auto row_iter = u_contents.begin();
                while (row_iter != u_contents.end())
                {
                    GraphBLAS::IndexType u_idx;
                    UScalarType          u_val;
                    std::tie(u_idx, u_val) = *row_iter;
                    TScalarType t_val = static_cast<TScalarType>(op(u_val));
                    t_contents.push_back(std::make_tuple(u_idx,t_val));
                    ++row_iter;
                }
            }

            GRB_LOG_VERBOSE("t: " << t_contents);

            // =================================================================
            // Accumulate into Z
            typedef typename std::conditional<
                std::is_same<AccumT, NoAccumulate>::value,
                TScalarType,
                typename AccumT::result_type>::type  ZScalarType;

            std::vector<std::tuple<IndexType,ZScalarType> > z_contents;
            ewise_or_opt_accum_1D(z_contents, w, t_contents, accum);

            GRB_LOG_VERBOSE("z: " << z_contents);

            // =================================================================
            // Copy Z into the final output considering mask and replace
            write_with_opt_mask_1D(w, z_contents, mask, replace_flag);
            return SUCCESS;
        }

        //**********************************************************************
        // Implementation of 4.3.8.2 Matrix variant of Apply
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename UnaryFunctionT,
                 typename AMatrixT,
                 typename ...CTagsT>
        inline Info apply(
            GraphBLAS::backend::Matrix<CScalarT, CTagsT...> &C,
            MaskT                                     const &mask,
            AccumT                                           accum,
            UnaryFunctionT                                   op,
            AMatrixT                                  const &A,
            bool                                             replace_flag = false)
        {
            typedef typename AMatrixT::ScalarType                   AScalarType;
            typedef std::vector<std::tuple<IndexType,AScalarType> > ARowType;

            typedef CScalarT                                        CScalarType;
            typedef std::vector<std::tuple<IndexType,CScalarType> > CRowType;

            typedef typename UnaryFunctionT::result_type            TScalarType;
            typedef std::vector<std::tuple<IndexType,TScalarType> > TRowType;


            IndexType nrows(A.nrows());
            IndexType ncols(A.ncols());

            // =================================================================
            // Apply the unary operator from A into T.
            // This is really the guts of what makes this special.
            LilSparseMatrix<TScalarType> T(nrows, ncols);

            ARowType a_row;
            TRowType t_row;

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

            GRB_LOG_VERBOSE("T: " << T);

            // =================================================================
            // Accumulate T via C into Z
            typedef typename std::conditional<
                std::is_same<AccumT, NoAccumulate>::value,
                TScalarType,
                typename AccumT::result_type>::type  ZScalarType;

            LilSparseMatrix<ZScalarType> Z(nrows, ncols);
            ewise_or_opt_accum(Z, C, T, accum);

            GRB_LOG_VERBOSE("Z: " << Z);

            // =================================================================
            // Copy Z into the final output considering mask and replace
            write_with_opt_mask(C, Z, mask, replace_flag);
            return SUCCESS;
        }
    }
}



#endif //GB_SEQUENTIAL_SPARSE_APPLY_HPP
