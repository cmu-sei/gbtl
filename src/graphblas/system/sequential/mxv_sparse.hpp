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
            IndexArrayType w_indices;
            std::vector<WScalarType> w_values;

            if (u.get_nvals() > 0)
            {
                IndexArrayType u_indices(u.get_indices());

                //**************************************************************

                IndexArrayType A_indices;

                for (IndexType row_idx = 0; row_idx < num_elts; ++row_idx)
                {
                    std::cerr << "***** PROCESSING MATRIX ROW " << row_idx << " *****" << std::endl;
                    ARowType const &A_row(A.get_row(row_idx));

                    if (!A_row.empty())
                    {
                        bool value_set = false;
                        WScalarType tmp_sum = op.zero();

                        typename ARowType::const_iterator A_iter = A_row.begin();
                        IndexArrayType::const_iterator    u_iter = u_indices.begin();

                        // pull first value out of the row
                        AScalarType a_val;
                        IndexType a_idx;
                        std::tie(a_idx, a_val) = *A_iter;

                        // loop through both ordered sets to compute sparse dot prod
                        while ((A_iter != A_row.end()) && (u_iter != u_indices.end()))
                        {
                            std::cerr << "Examine u index = " << *u_iter
                                      << "," << u.get_value_at(*u_iter)
                                      << ", A col_idx = "
                                      << a_idx << "," << a_val << std::endl;

                            if (*u_iter == a_idx)
                            {
                                std::cerr << tmp_sum << " + "
                                          << a_val << " * " << u.get_value_at(*u_iter)
                                          << " = "
                                          << op.mult(a_val, u.get_value_at(*u_iter))
                                          << std::endl;
                                tmp_sum = op.add(tmp_sum,
                                                 op.mult(a_val,
                                                         u.get_value_at(*u_iter)));
                                value_set = true;
                                std::cerr << "Equal, mutliply_accum, tmp_sum = "
                                          << tmp_sum << std::endl;
                                ++u_iter;
                                ++A_iter;
                                std::tie(a_idx, a_val) = *A_iter;
                            }
                            else if (*u_iter > a_idx)
                            {
                                std::cerr << "Advancing A_iter" << std::endl;
                                ++A_iter;
                                std::tie(a_idx, a_val) = *A_iter;
                            }
                            else
                            {
                                std::cerr << "Advancing u_iter" << std::endl;
                                ++u_iter;
                            }
                        }

                        if (value_set)
                        {
                            std::cerr << "Pushing idx,val: " << row_idx << "," << tmp_sum
                                      << std::endl;
                            w_values.push_back(tmp_sum);
                            w_indices.push_back(row_idx);
                        }
                    }
                    else
                    {
                        std::cerr << "A row empty." << std::endl;
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

#endif // GB_SEQUENTIAL_MXV_SPARSE_HPP
