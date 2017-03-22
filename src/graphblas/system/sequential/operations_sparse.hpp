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

#ifndef GB_SEQUENTIAL_OPERATIONS_SPARSE_HPP
#define GB_SEQUENTIAL_OPERATIONS_SPARSE_HPP

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


//****************************************************************************
namespace {
    /// Perform the dot product of a row of a matrix with a sparse vector without
    /// pulling the indices out of the vector first.
    template <typename D1, typename D2, typename D3, typename SemiringT>
    bool dot2(D3                                                      &ans,
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
    /// A dot product of two sparse vectors (vectors<tuple(index,value)>)
    template <typename D1, typename D2, typename D3, typename SemiringT>
    bool dot(D3                                                      &ans,
             std::vector<std::tuple<GraphBLAS::IndexType,D1> > const &vec1,
             std::vector<std::tuple<GraphBLAS::IndexType,D2> > const &vec2,
             SemiringT                                                op)
    {
        bool value_set(false);
        ans = op.zero();

        if (vec2.empty() || vec1.empty())
        {
            return value_set;
        }

        auto v1_it = vec1.begin();
        auto v2_it = vec2.begin();

        // pull first value out of the row
        D1 a_val;
        D2 u_val;
        GraphBLAS::IndexType a_idx, u_idx;

        // loop through both ordered sets to compute sparse dot prod
        while ((v1_it != vec1.end()) &&
               (v2_it != vec2.end()))
        {
            std::tie(a_idx, a_val) = *v1_it;
            std::tie(u_idx, u_val) = *v2_it;

            //std::cerr << "Examine u idx,val = " << u_idx << "," << u_val
            //          << "; A col_idx,val = " << a_idx << "," << a_val << std::endl;

            if (u_idx == a_idx)
            {
                //std::cerr << ans << " + " << a_val << " * " << u_val << " = ";
                ans = op.add(ans, op.mult(a_val, u_val));
                value_set = true;
                //std::cerr << ans << std::endl;

                ++v2_it;
                ++v1_it;
            }
            else if (u_idx > a_idx)
            {
                //std::cerr << "Advancing v1_it" << std::endl;
                ++v1_it;
            }
            else
            {
                //std::cerr << "Advancing v2_it" << std::endl;
                ++v2_it;
            }
        }

        return value_set;
    }

    //************************************************************************
    /// Apply element-wise operation to union on sparse vectors.
    template <typename D1, typename D2, typename D3, typename BinaryOpT>
    bool ewise_or(std::vector<std::tuple<GraphBLAS::IndexType,D3> >       &ans,
                  std::vector<std::tuple<GraphBLAS::IndexType,D1> > const &vec1,
                  std::vector<std::tuple<GraphBLAS::IndexType,D2> > const &vec2,
                  BinaryOpT                                                op)
    {
        ans.clear();
        auto v1_it = vec1.begin();
        auto v2_it = vec2.begin();

        D1 v1_val;
        D2 v2_val;
        GraphBLAS::IndexType v1_idx, v2_idx;

        // loop through both ordered sets to compute ewise_or
        while ((v1_it != vec1.end()) || (v2_it != vec2.end()))
        {
            if ((v1_it != vec1.end()) && (v2_it != vec2.end()))
            {
                std::tie(v1_idx, v1_val) = *v1_it;
                std::tie(v2_idx, v2_val) = *v2_it;

                if (v2_idx == v1_idx)
                {
                    //std::cerr << ans << " + " << v1_val << " * " << v2_val << " = ";
                    ans.push_back(std::make_tuple(v1_idx,
                                                  static_cast<D3>(op(v1_val, v2_val))));
                    //std::cerr << ans << std::endl;

                    ++v2_it;
                    ++v1_it;
                }
                else if (v2_idx > v1_idx)
                {
                    //std::cerr << "Copying v1, Advancing v1_it" << std::endl;
                    ans.push_back(std::make_tuple(v1_idx, v1_val));
                    ++v1_it;
                }
                else
                {
                    //std::cerr << "Copying v2, Advancing v2_it" << std::endl;
                    ans.push_back(std::make_tuple(v2_idx, v2_val));
                    ++v2_it;
                }
            }
            else if (v1_it != vec1.end())
            {
                std::tie(v1_idx, v1_val) = *v1_it;
                ans.push_back(std::make_tuple(v1_idx, static_cast<D3>(v1_val)));
                ++v1_it;
            }
            else // v2_it != vec2.end())
            {
                std::tie(v2_idx, v2_val) = *v2_it;
                ans.push_back(std::make_tuple(v2_idx, static_cast<D3>(v2_val)));
                ++v2_it;
            }
        }
    }

    //************************************************************************
    /// Apply element-wise operation to intersection of sparse vectors.
    template <typename D1, typename D2, typename D3, typename BinaryOpT>
    bool ewise_and(std::vector<std::tuple<GraphBLAS::IndexType,D3> >       &ans,
                   std::vector<std::tuple<GraphBLAS::IndexType,D1> > const &vec1,
                   std::vector<std::tuple<GraphBLAS::IndexType,D2> > const &vec2,
                   BinaryOpT                                                op)
    {
        ans.clear();
        auto v1_it = vec1.begin();
        auto v2_it = vec2.begin();

        D1 v1_val;
        D2 v2_val;
        GraphBLAS::IndexType v1_idx, v2_idx;

        // loop through both ordered sets to compute ewise_or
        while ((v1_it != vec1.end()) && (v2_it != vec2.end()))
        {
            std::tie(v1_idx, v1_val) = *v1_it;
            std::tie(v2_idx, v2_val) = *v2_it;

            if (v2_idx == v1_idx)
            {
                //std::cerr << ans << " + " << v1_val << " * " << v2_val << " = ";
                ans.push_back(std::make_tuple(v1_idx,
                                              static_cast<D3>(op(v1_val, v2_val))));
                //std::cerr << ans << std::endl;

                ++v2_it;
                ++v1_it;
            }
            else if (v2_idx > v1_idx)
            {
                //std::cerr << "Advancing v1_it" << std::endl;
                ++v1_it;
            }
            else
            {
                //std::cerr << "Advancing v2_it" << std::endl;
                ++v2_it;
            }
        }
    }
}

//****************************************************************************

namespace GraphBLAS
{
    namespace backend
    {
        //**************************************************************************
        //Matrix-Matrix multiply for LilSparseMatrix
        //**************************************************************************
        template<typename CMatrixT,
                 typename AccumT,
                 typename SemiringT,
                 typename AMatrixT,
                 typename BMatrixT>
        inline void mxm(CMatrixT       &C,
                        AccumT          accum,
                        SemiringT       op,
                        AMatrixT const &A,
                        BMatrixT const &B)
        {
            IndexType nrow_A(A.get_nrows());
            IndexType ncol_A(A.get_ncols());
            IndexType nrow_B(B.get_nrows());
            IndexType ncol_B(B.get_ncols());
            IndexType nrow_C(C.get_nrows());
            IndexType ncol_C(C.get_ncols());

            if (ncol_A != nrow_B || nrow_A != nrow_C || ncol_B != ncol_C)
            {
                throw DimensionException("mxm: matrix dimensions are not compatible");
            }

            IndexType irow;
            IndexType icol;
            std::vector<IndexArrayType> indA;   // Row-column
            std::vector<IndexArrayType> indB;   // Column-row
            std::vector<IndexType> ind_intersection;

            indA.resize(nrow_A);
            indB.resize(ncol_B);

            auto tmp_sum = op.zero();
            auto tmp_product = op.zero();
            for (irow = 0; irow < nrow_C; irow++)
            {
                A.getColumnIndices(irow, indA[irow]);
                for (icol = 0; icol < ncol_C; icol++)
                {
                    if (irow == 0)
                    {
                        B.getRowIndices(icol, indB[icol]);
                    }
                    if (!indA[irow].empty() && !indB[icol].empty())
                    {
                        ind_intersection.clear();
                        std::set_intersection(indA[irow].begin(), indA[irow].end(),
                                              indB[icol].begin(), indB[icol].end(),
                                              std::back_inserter(ind_intersection));
                        if (!ind_intersection.empty())
                        {
                            tmp_sum = op.zero();
                            // Range-based loop, access by value
                            for (auto kk : ind_intersection)
                            {
                                // Matrix multiply kernel
                                tmp_product = op.mult(A.get_value_at(irow,kk),
                                                      B.get_value_at(kk,icol));
                                tmp_sum = op.add(tmp_sum, tmp_product);
                            }
#if 0
                            try {
                                std::cout << "\nTry";
                                C.set_value_at(irow, icol,
                                               accum(C.get_value_at(irow, icol),
                                                     tmp_sum));
                            } catch (int e) {
                                std::cout << "\nCatch";
                                C.set_value_at(irow, icol, tmp_sum);
                            }
                            //C.set_value_at(irow, icol,
                            //               accum(C.get_value_at(irow, icol),
                            //                     tmp_sum));
                            //C.set_value_at(irow, icol, tmp_sum);
#else
                            C.set_value_at(irow, icol, tmp_sum);
#endif
                        }
                    }
                }
            }
        }


        //**********************************************************************
        /// Matrix-matrix multiply for LilSparseMatrix'es
        template<typename CMatrixT,
                 typename AccumT,
                 typename SemiringT,
                 typename AMatrixT,
                 typename BMatrixT>
        inline void mxm_v2(CMatrixT       &C,
                           AccumT          accum,
                           SemiringT       op,
                           AMatrixT const &A,
                           BMatrixT const &B)
        {
            IndexType nrow_A(A.get_nrows());
            IndexType ncol_A(A.get_ncols());
            IndexType nrow_B(B.get_nrows());
            IndexType ncol_B(B.get_ncols());
            IndexType nrow_C(C.get_nrows());
            IndexType ncol_C(C.get_ncols());

            // The following should be checked by the frontend only:
            if (ncol_A != nrow_B || nrow_A != nrow_C || ncol_B != ncol_C)
            {
                throw DimensionException("mxm: matrix dimensions are not compatible");
            }

            typedef typename AMatrixT::ScalarType AScalarType;
            typedef typename BMatrixT::ScalarType BScalarType;
            typedef typename CMatrixT::ScalarType CScalarType;
            typedef std::vector<std::tuple<IndexType,AScalarType> > ARowType;
            typedef std::vector<std::tuple<IndexType,BScalarType> > BColType;
            typedef std::vector<std::tuple<IndexType,CScalarType> > CColType;

            CMatrixT T(nrow_C, ncol_C);
            if ((A.get_nvals() > 0) && (B.get_nvals() > 0))
            {
                // create a column of result at a time
                CColType T_col;
                for (IndexType col_idx = 0; col_idx < ncol_B; ++col_idx)
                {
                    BColType B_col(B.get_col(col_idx));

                    if (!B_col.empty())
                    {
                        for (IndexType row_idx = 0; row_idx < nrow_A; ++row_idx)
                        {
                            ARowType const &A_row(A.get_row(row_idx));
                            if (!A_row.empty())
                            {
                                CScalarType C_val;
                                if (dot(C_val, A_row, B_col, op))
                                {
                                    T_col.push_back(
                                        std::make_tuple(row_idx, C_val));
                                }
                            }
                        }
                        if (!T_col.empty())
                        {
                            T.set_col(col_idx, T_col);
                            T_col.clear();
                        }
                    }
                }
            }

            /// @todo  Detect if accum is GrB_NULL, and take a short cut
            // Perform accum
            CColType tmp_row;
            for (IndexType row_idx = 0; row_idx < nrow_C; ++row_idx)
            {
                ewise_or(tmp_row, C.get_row(row_idx), T.get_row(row_idx), accum);
                C.set_row(row_idx, tmp_row);
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
            IndexArrayType w_indices;
            std::vector<WScalarType> w_values;

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
                        ARowType const &A_row(A.get_row(row_idx));

                        if (!A_row.empty())
                        {
                            WScalarType w_val;
                            if (dot(w_val, A_row, u_bitmap, u_values, u.get_nvals(), op))
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
    } //backend
} //GraphBLAS

#endif // GB_SEQUENTIAL_OPERATIONS_SPARSE_HPP
