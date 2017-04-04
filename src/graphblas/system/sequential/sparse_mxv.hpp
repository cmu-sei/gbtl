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
 * Implementations of all GraphBLAS functions optimized for the sequential
 * (CPU) backend.
 */

#ifndef GB_SEQUENTIAL_SPARSE_MXV_HPP
#define GB_SEQUENTIAL_SPARSE_MXV_HPP

#pragma once

#include <functional>
#include <utility>
#include <vector>
#include <iterator>
#include <iostream>
#include <graphblas/accum.hpp>
#include <graphblas/algebra.hpp>

#include "sparse_helpers.hpp"


//****************************************************************************

namespace GraphBLAS
{
    namespace backend
    {
        //**********************************************************************
        /// Matrix-vector multiply for LilSparseMatrix and SparseBitmapVector
        /// @todo Need to figure out how to specialize
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename SemiringT,
                 typename AMatrixT,
                 typename UVectorT>
        inline void mxv(WVectorT        &w,
                        MaskT     const &mask,
                        AccumT           accum,
                        SemiringT        op,
                        AMatrixT  const &A,
                        UVectorT  const &u,
                        bool             replace_flag)
        {
            // The following should be checked by the frontend only:
            if ((w.get_size() != A.get_nrows()) ||
                (u.get_size() != A.get_ncols()) ||
                (w.get_size() != mask.get_size()))
            {
                throw DimensionException("mxv: dimensions are not compatible.");
            }

            typedef typename WVectorT::ScalarType WScalarType;
            typedef typename AMatrixT::ScalarType AScalarType;
            typedef typename UVectorT::ScalarType UScalarType;
            typedef std::vector<std::tuple<IndexType,AScalarType> > ARowType;

            IndexType      num_elts(w.get_size());
            IndexArrayType t_indices;
            std::vector<WScalarType> t_values;

            if (u.get_nvals() > 0)
            {
                /// @todo need a heuristic for switching between two modes
                if (u.get_size()/u.get_nvals() >= 4)
                {
                    auto u_contents(u.get_contents());
                    for (IndexType row_idx = 0; row_idx < num_elts; ++row_idx)
                    {
                        //std::cerr << "**1** PROCESSING MATRIX ROW " << row_idx
                        //          << " *****" << std::endl;
                        ARowType const &A_row(A.get_row(row_idx));

                        if (!A_row.empty())
                        {
                            WScalarType t_val;
                            if (dot(t_val, A_row, u_contents, op))
                            {
                                t_indices.push_back(row_idx);
                                t_values.push_back(t_val);
                            }
                        }
                    }
                }
                else
                {
                    std::vector<bool> const &u_bitmap(u.get_bitmap());
                    std::vector<UScalarType> const &u_values(u.get_vals());

                    for (IndexType row_idx = 0; row_idx < num_elts; ++row_idx)
                    {
                        //std::cerr << "**2** PROCESSING MATRIX ROW " << row_idx
                        //          << " *****" << std::endl;
                        ARowType const &A_row(A.get_row(row_idx));

                        if (!A_row.empty())
                        {
                            WScalarType t_val;
                            if (dot2(t_val, A_row, u_bitmap, u_values, u.get_nvals(), op))
                            {
                                t_indices.push_back(row_idx);
                                t_values.push_back(t_val);
                            }
                        }
                    }
                }
            }

            // accum here
            // mask here
            // store in w
            w.clear();
            w.build(t_indices.begin(), t_values.begin(), t_values.size());
        }
    } // backend
} // GraphBLAS

#endif
