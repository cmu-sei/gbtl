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

#ifndef GB_SEQUENTIAL_MXV_SPARSE_HPP
#define GB_SEQUENTIAL_MXV_SPARSE_HPP

#pragma once

#include <functional>
#include <utility>
#include <vector>
#include <iterator>
#include <iostream>
#include <graphblas/accum.hpp>
#include <graphblas/algebra.hpp>
#include <graphblas/system/sequential/TransposeView.hpp>
#include <graphblas/system/sequential/NegateView.hpp>

#if 0
//****************************************************************************
namespace {
    /// Perform the dot product of a row of a matrix with a sparse vector without
    /// pulling the indices out of the vector first.
    template <typename D1, typename D2, typename D3, typename SemiringT>
    bool dot(D3                                                      &ans,
             std::vector<std::tuple<GraphBLAS::IndexType,D1> > const &A_row,
             std::vector<bool>                                 const &u_bitmap,
             std::vector<D2>                                   const &u_vals,
             GraphBLAS::IndexType                                     u_nvals,
             SemiringT                                                op)
    {
        bool value_set(false);
        ans = op.zero();

        if ((u_nvals == 0) || A_row.empty())
        {
            return value_set;
        }

        // find first stored value in u
        GraphBLAS::IndexType u_idx(0);
        while (!u_bitmap[u_idx]) ++u_idx; // skip unstored elements

        // pull first value out of the row
        auto A_iter = A_row.begin();
        D1 a_val;
        GraphBLAS::IndexType a_idx;

        // loop through both ordered sets to compute sparse dot prod
        while ((A_iter != A_row.end()) && (u_idx < u_vals.size()))
        {
            std::tie(a_idx, a_val) = *A_iter;
            //std::cerr << "Examine u index = " << u_idx << "," << u_vals[u_idx]
            //          << ", A col_idx = " << a_idx << "," << a_val << std::endl;

            if (u_idx == a_idx)
            {
                //std::cerr << ans << " + "  << a_val << " * "  << u_vals[u_idx]
                //          << " = " << op.mult(a_val, u_vals[u_idx]) << std::endl;

                ans = op.add(ans, op.mult(a_val, u_vals[u_idx]));
                value_set = true;

                //std::cerr << "Equal, mutliply_accum, ans = " << ans << std::endl;

                do { ++u_idx; } while ((u_idx < u_vals.size()) && !u_bitmap[u_idx]);
                ++A_iter;
            }
            else if (u_idx > a_idx)
            {
                //std::cerr << "Advancing A_iter" << std::endl;
                ++A_iter;
            }
            else
            {
                //std::cerr << "Advancing u_iter" << std::endl;
                do { ++u_idx; } while ((u_idx < u_vals.size()) && !u_bitmap[u_idx]);
            }
        }

        return value_set;
    }

    //************************************************************************
    /// A dot product with elements extracted from vector
    template <typename D1, typename D2, typename D3, typename SemiringT>
    bool dot(D3                                                      &ans,
             std::vector<std::tuple<GraphBLAS::IndexType,D1> > const &A_row,
             std::vector<std::tuple<GraphBLAS::IndexType,D2> > const &u_contents,
             SemiringT                                                op)
    {
        bool value_set(false);
        ans = op.zero();

        if (u_contents.empty() || A_row.empty())
        {
            return value_set;
        }

        auto A_iter = A_row.begin();
        auto u_iter = u_contents.begin();

        // pull first value out of the row
        D1 a_val;
        D2 u_val;
        GraphBLAS::IndexType a_idx, u_idx;

        // loop through both ordered sets to compute sparse dot prod
        while ((A_iter != A_row.end()) &&
               (u_iter != u_contents.end()))
        {
            std::tie(a_idx, a_val) = *A_iter;
            std::tie(u_idx, u_val) = *u_iter;

            //std::cerr << "Examine u idx,val = " << u_idx << "," << u_val
            //          << "; A col_idx,val = " << a_idx << "," << a_val << std::endl;

            if (u_idx == a_idx)
            {
                //std::cerr << ans << " + " << a_val << " * " << u_val << " = ";
                ans = op.add(ans, op.mult(a_val, u_val));
                value_set = true;
                //std::cerr << ans << std::endl;

                ++u_iter;
                ++A_iter;
            }
            else if (u_idx > a_idx)
            {
                //std::cerr << "Advancing A_iter" << std::endl;
                ++A_iter;
            }
            else
            {
                //std::cerr << "Advancing u_iter" << std::endl;
                ++u_iter;
            }
        }

        return value_set;
    }
}

//****************************************************************************
namespace GraphBLAS {
    namespace backend {

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
            // The following should be checked by the frontend:
            if ((w.get_size() != A.nrows()) ||
                (u.get_size() != A.ncols()) ||
                (w.get_size() != mask.get_size()))
            {
                throw DimensionException("mxv: dimensions are not compatible.");
            }

            typedef typename WVectorT::ScalarType WScalarType;
            typedef typename AMatrixT::ScalarType AScalarType;
            typedef typename UVectorT::ScalarType UScalarType;
            typedef std::vector<std::tuple<IndexType,AScalarType> > ARowType;

            IndexType      num_elts(w.get_size());
            IndexArrayType w_indices;
            std::vector<WScalarType> w_values;

            if (u.nvals() > 0)
            {
                /// @todo need a heuristic for switching between two modes
                if (u.get_size()/u.nvals() >= 4)
                {
                    auto u_contents(u.get_contents());
                    for (IndexType row_idx = 0; row_idx < num_elts; ++row_idx)
                    {
                        //std::cerr << "**1** PROCESSING MATRIX ROW " << row_idx
                        //          << " *****" << std::endl;
                        ARowType const &A_row(A.getRow(row_idx));

                        if (!A_row.empty())
                        {
                            WScalarType w_val;
                            if (dot(w_val, A_row, u_contents, op))
                            {
                                w_indices.push_back(row_idx);
                                w_values.push_back(w_val);
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
                            WScalarType w_val;
                            if (dot(w_val, A_row, u_bitmap, u_values, u.nvals(), op))
                            {
                                w_indices.push_back(row_idx);
                                w_values.push_back(w_val);
                            }
                        }
                    }
                }
            }

            // accum here
            // mask here
            // store in w
            w.clear();
            w.build(w_indices.begin(), w_values.begin(), w_values.size());
        }
    }  // backend
}  // GraphBLAS
#endif
#endif // GB_SEQUENTIAL_MXV_SPARSE_HPP
