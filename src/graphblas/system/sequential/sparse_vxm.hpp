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

#ifndef GB_SEQUENTIAL_SPARSE_VXM_HPP
#define GB_SEQUENTIAL_SPARSE_VXM_HPP

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
                 typename AccumT,
                 typename SemiringT,
                 typename AMatrixT,
                 typename UVectorT>
        inline void vxm(WVectorT        &w,
                        AccumT           accum,
                        SemiringT        op,
                        UVectorT  const &u,
                        AMatrixT  const &A)
        {
            if ((w.size() != A.ncols()) ||
                (u.size() != A.nrows()))
            {
                throw DimensionException("vxm(nomask): dimensions are not compatible.");
            }

            typedef typename AccumT::result_type AccumScalarType;
            typedef typename SemiringT::result_type D3ScalarType;
            typedef typename WVectorT::ScalarType WScalarType;
            typedef typename AMatrixT::ScalarType AScalarType;
            typedef typename UVectorT::ScalarType UScalarType;
            typedef std::vector<std::tuple<IndexType,AScalarType> > AColType;
            typedef std::vector<std::tuple<IndexType,D3ScalarType> > TRowType;
            typedef std::vector<std::tuple<IndexType,AccumScalarType> > ZRowType;

            IndexType  num_elts(w.size());
            TRowType   t;   // t = <D3(op), nrows(A), contents of A.op.u>

            if (u.nvals() > 0)
            {
                auto u_contents(u.getContents());
                for (IndexType col_idx = 0; col_idx < num_elts; ++col_idx)
                {
                    //std::cerr << "***** PROCESSING MATRIX COL " << col_idx
                    //          << " *****" << std::endl;
                    AColType const &A_col(A.getCol(col_idx));

                    if (!A_col.empty())
                    {
                        D3ScalarType t_val;
                        if (dot(t_val, u_contents, A_col, op))
                        {
                            t.push_back(std::make_tuple(col_idx, t_val));
                        }
                    }
                }
            }

            // accum here
            /// @todo  Detect if accum is GrB_NULL, and take a short cut
            // Perform accum
            ZRowType z;
            ewise_or(z, w.getContents(), t, accum);

            // store in w
            w.clear();
            for (auto tupl : z)
            {
                w.setElement(std::get<0>(tupl), std::get<1>(tupl));
            }
        }

        //**********************************************************************
        /// Matrix-vector multiply for LilSparseMatrix and SparseBitmapVector
        /// @todo Need to figure out how to specialize
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename SemiringT,
                 typename AMatrixT,
                 typename UVectorT>
        inline void vxm(WVectorT        &w,
                        MaskT     const &mask,
                        AccumT           accum,
                        SemiringT        op,
                        UVectorT  const &u,
                        AMatrixT  const &A,
                        bool             replace_flag = false)
        {
            /// @todo move the dimension checks to the backend
            if ((w.size() != mask.size()) ||
                (w.size() != A.ncols()) ||
                (u.size() != A.nrows()))
            {
                throw DimensionException("vxm(mask): dimensions are not compatible.");
            }

            typedef typename AccumT::result_type AccumScalarType;
            typedef typename SemiringT::result_type D3ScalarType;
            typedef typename MaskT::ScalarType MaskScalarType;
            typedef typename WVectorT::ScalarType WScalarType;
            typedef typename AMatrixT::ScalarType AScalarType;
            typedef typename UVectorT::ScalarType UScalarType;
            typedef std::vector<std::tuple<IndexType,AScalarType> > AColType;
            typedef std::vector<std::tuple<IndexType,D3ScalarType> > TRowType;
            typedef std::vector<std::tuple<IndexType,AccumScalarType> > ZRowType;

            IndexType  num_elts(w.size());
            // t = <D3(op), nrows(A), contents of A.op.u>
            TRowType   t;
            //IndexArrayType t_indices;
            //std::vector<D3ScalarType> t_values;

            if ((u.nvals() > 0) && (mask.nvals() > 0))
            {
                auto u_contents(u.getContents());
                for (IndexType col_idx = 0; col_idx < num_elts; ++col_idx)
                {
                    //std::cerr << "***** PROCESSING MATRIX COL " << col_idx
                    //          << " *****" << std::endl;
                    AColType const &A_col(A.getCol(col_idx));

                    if (!A_col.empty() &&
                        mask.hasElement(col_idx) &&
                        (mask.extractElement(col_idx) != false))
                    {
                        D3ScalarType t_val;
                        if (dot(t_val, u_contents, A_col, op))
                        {
                            t.push_back(std::make_tuple(col_idx, t_val));
                        }
                    }
                }
            }

            // accum here
            /// @todo  Detect if accum is GrB_NULL, and take a short cut
            // Perform accum
            ZRowType z;
            if (replace_flag)
            {
                ewise_or_mask(z, w.getContents(), t, mask.getContents(),
                              accum, replace_flag);
            }
            else // merge
            {
                ewise_or(z, w.getContents(), t, accum);
            }
            // store in w
            w.clear();
            for (auto tupl : z)
            {
                w.setElement(std::get<0>(tupl), std::get<1>(tupl));
            }

        }

    } // backend
} // GraphBLAS

#endif
