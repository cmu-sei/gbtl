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
                 typename AccumT,
                 typename SemiringT,
                 typename AMatrixT,
                 typename UVectorT>
        inline void mxv(WVectorT        &w,
                        AccumT           accum,
                        SemiringT        op,
                        AMatrixT  const &A,
                        UVectorT  const &u)
        {
            if ((w.size() != A.nrows()) ||
                (u.size() != A.ncols()))
            {
                throw DimensionException("mxv(nomask): dimensions are not compatible.");
            }


            typedef typename AccumT::result_type AccumScalarType;
            typedef typename SemiringT::result_type D3ScalarType;
            typedef typename WVectorT::ScalarType WScalarType;
            typedef typename AMatrixT::ScalarType AScalarType;
            typedef typename UVectorT::ScalarType UScalarType;
            typedef std::vector<std::tuple<IndexType,AScalarType> > ARowType;
            typedef std::vector<std::tuple<IndexType,D3ScalarType> > TRowType;
            typedef std::vector<std::tuple<IndexType,AccumScalarType> > ZRowType;

            IndexType  num_elts(w.size());
            TRowType   t;   // t = <D3(op), nrows(A), contents of A.op.u>

            if (u.nvals() > 0)
            {
                // Two different approaches to performing the A *.+ u portion
                // of the computation is selected by this if statement.
                /// @todo need a heuristic for switching between two modes
                if (u.size()/u.nvals() >= 4)
                {
                    auto u_contents(u.getContents());
                    for (IndexType row_idx = 0; row_idx < num_elts; ++row_idx)
                    {
                        //std::cerr << "**1** PROCESSING MATRIX ROW " << row_idx
                        //          << " *****" << std::endl;
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
                else
                {
                    std::vector<bool> const &u_bitmap(u.get_bitmap());
                    std::vector<UScalarType> const &u_values(u.get_vals());

                    for (IndexType row_idx = 0; row_idx < num_elts; ++row_idx)
                    {
                        //std::cerr << "**2** PROCESSING MATRIX ROW " << row_idx
                        //          << " *****" << std::endl;
                        ARowType const &A_row(A.getRow(row_idx));

                        if (!A_row.empty())
                        {
                            D3ScalarType t_val;
                            if (dot2(t_val, A_row, u_bitmap, u_values, u.nvals(), op))
                            {
                                t.push_back(std::make_tuple(row_idx, t_val));
                            }
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
        inline void mxv(WVectorT        &w,
                        MaskT     const &mask,
                        AccumT           accum,
                        SemiringT        op,
                        AMatrixT  const &A,
                        UVectorT  const &u,
                        bool             replace_flag = false)
        {
            if ((w.size() != mask.size()) ||
                (w.size() != A.nrows()) ||
                (u.size() != A.ncols()))
            {
                throw DimensionException("mxv(mask): dimensions are not compatible.");
            }

            typedef typename AccumT::result_type AccumScalarType;
            typedef typename SemiringT::result_type D3ScalarType;
            typedef typename MaskT::ScalarType MaskScalarType;
            typedef typename WVectorT::ScalarType WScalarType;
            typedef typename AMatrixT::ScalarType AScalarType;
            typedef typename UVectorT::ScalarType UScalarType;
            typedef std::vector<std::tuple<IndexType,AScalarType> > ARowType;
            typedef std::vector<std::tuple<IndexType,D3ScalarType> > TRowType;
            typedef std::vector<std::tuple<IndexType,AccumScalarType> > ZRowType;

            IndexType  num_elts(w.size());
            TRowType   t;

            if ((u.nvals() > 0) && (mask.nvals() > 0))
            {
                // Two different approaches to performing the A *.+ u portion
                // of the computation is selected by this if statement.
                /// @todo need a heuristic for switching between two modes
                if (u.size()/u.nvals() >= 4)
                {
                    auto u_contents(u.getContents());
                    for (IndexType row_idx = 0; row_idx < num_elts; ++row_idx)
                    {
                        //std::cerr << "**1** PROCESSING VECTOR ELEMENT " << row_idx
                        //          << " *****" << std::endl;
                        ARowType const &A_row(A.getRow(row_idx));

                        if (!A_row.empty() &&
                            mask.hasElement(row_idx) &&
                            (mask.extractElement(row_idx) != false))
                        {
                            D3ScalarType t_val;
                            if (dot(t_val, A_row, u_contents, op))
                            {
                                t.push_back(std::make_tuple(row_idx, t_val));
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
                        //std::cerr << "**2** PROCESSING VECTOR ELEMENT " << row_idx
                        //          << " *****" << std::endl;
                        ARowType const &A_row(A.getRow(row_idx));

                        if (!A_row.empty() &&
                            mask.hasElement(row_idx) &&
                            (mask.extractElement(row_idx) != false))
                        {
                            D3ScalarType t_val;
                            if (dot2(t_val, A_row, u_bitmap, u_values, u.nvals(), op))
                            {
                                t.push_back(std::make_tuple(row_idx, t_val));
                            }
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
