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

#ifndef GB_SEQUENTIAL_SPARSE_MXM_HPP
#define GB_SEQUENTIAL_SPARSE_MXM_HPP

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
        //**************************************************************************
        //Matrix-Matrix multiply for LilSparseMatrix
        // @deprecated - see version 2
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

            // T, intermediate matrix holding the product of A and B.
            // T = <D3(op), nrows(A), ncols(B), contents of A.op.B>
            LilSparseMatrix<typename SemiringT::result_type> T(nrow_A, ncol_B);
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
        /// Matrix-matrix multiply for LilSparseMatrix'es with a mask.
        // NOTE: The mask is a actually a boolean matrix not an exists/doesn't exist
        // as described in the background.
        template<typename CMatrixT,
                 typename MMatrixT,                             
                 typename AccumT,
                 typename SemiringT,
                 typename AMatrixT,
                 typename BMatrixT>
        inline void mxm_v2_mask(CMatrixT       &C,
                                MMatrixT const &M,              // Mask
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
            IndexType nrow_M(M.get_nrows());
            IndexType ncol_M(M.get_ncols());

            // @todo: Move these checks into the front end when we get that wired up
            check_transposed_dimensions(A, "A", B, "B");
            check_dimensions(A, "A", C, "C");
            check_dimensions(C, "C", M, "M");

            typedef typename AMatrixT::ScalarType AScalarType;
            typedef typename BMatrixT::ScalarType BScalarType;
            typedef typename CMatrixT::ScalarType CScalarType;
            typedef typename MMatrixT::ScalarType MScalarType;
            typedef std::vector<std::tuple<IndexType,AScalarType> > ARowType;
            typedef std::vector<std::tuple<IndexType,BScalarType> > BColType;
            typedef std::vector<std::tuple<IndexType,CScalarType> > CColType;
            typedef std::vector<std::tuple<IndexType,MScalarType> > MColType;

            typedef typename std::tuple<IndexType,MScalarType> MColEntryType;

            // T, intermediate matrix holding the product of A and B.
            // T = <D3(op), nrows(A), ncols(B), contents of A.op.B>
            LilSparseMatrix<typename SemiringT::result_type> T(nrow_A, ncol_B);
            if ((A.get_nvals() > 0) && (B.get_nvals() > 0))
            {
                // create a column of results at a time in the T matrix
                CColType T_col;

                for (IndexType col_idx = 0; col_idx < ncol_B; ++col_idx)
                {
                    BColType B_col(B.get_col(col_idx));
                    MColType M_col(M.get_col(col_idx));

                    IndexType m_idx;
                    MScalarType m_val;

                    // There should be content both in the column we are multiplying
                    // and the mask.
                    if (!B_col.empty() && !M_col.empty())
                    {
                        // Ideally we'd have a sparse iterator that we could just track
                        // in parallel to the row/col index.
                        auto m_it = M_col.begin();
                        if (m_it != M_col.end())
                            std::tie(m_idx, m_val) = *m_it;

                        for (IndexType row_idx = 0; row_idx < nrow_A; ++row_idx)
                        {
                            // Check the mask to see if we need to do the operation
                            // for this row.  
                            if (m_it != M_col.end() && row_idx == m_idx)
                            {
                                if (m_val )
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

                                // Find the next mask cell
                                ++m_it;
                                if ( m_it != M_col.end())
                                    std::tie(m_idx, m_val) = *m_it;
                            }

                        }

                        // If we didn't do anything at all, then don't add it
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
            // NOTE: We do not consult the mask!  Since no value should have been
            // generated above, and we ASSUME ('cause we looked at the code) that 
            // the ewise_or does a no-op on cells that have not value, then we 
            // shouldn't need to mask.
            CColType tmp_row;
            for (IndexType row_idx = 0; row_idx < nrow_C; ++row_idx)
            {
                ewise_or(tmp_row, C.get_row(row_idx), T.get_row(row_idx), accum);
                C.set_row(row_idx, tmp_row);
            }
        }



    } // backend


} // GraphBLAS

#endif
