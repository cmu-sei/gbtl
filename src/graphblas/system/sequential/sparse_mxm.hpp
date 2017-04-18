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
        //**********************************************************************
        /// Matrix-matrix multiply for LilSparseMatrix'es

        // Okay! This is a total mess.  If we change the name to "mxm" it conflicts
        // with the names in operations and sequentail/operations.  Lots of weird
        // wiring going on here.  I (andrew) can't get this thing to properly resolve
        // all the symbols.  It is either a namespace error, or overload error or 
        // something.  Switching to more strict signatures ends up with weird
        // errors in GraphBLAS::backend::Matrix when operations tries to get m_mat.
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
            IndexType nrow_A(A.nrows());
            IndexType ncol_A(A.ncols());
            IndexType nrow_B(B.nrows());
            IndexType ncol_B(B.ncols());
            IndexType nrow_C(C.nrows());
            IndexType ncol_C(C.ncols());

            // The following should be checked by the frontend only:
            // Inside parts must match
            check_inside_dimensions(A, "A", B, "B");

            // Output must match
            check_outside_dimensions(A, "A", B, "B", C, "C");

            typedef typename AMatrixT::ScalarType AScalarType;
            typedef typename BMatrixT::ScalarType BScalarType;
            typedef typename CMatrixT::ScalarType CScalarType;
            typedef std::vector<std::tuple<IndexType,AScalarType> > ARowType;
            typedef std::vector<std::tuple<IndexType,BScalarType> > BColType;
            typedef std::vector<std::tuple<IndexType,CScalarType> > CColType;

            // T, intermediate matrix holding the product of A and B.
            // T = <D3(op), nrows(A), ncols(B), contents of A.op.B>
            LilSparseMatrix<typename SemiringT::result_type> T(nrow_A, ncol_B);
            if ((A.nvals() > 0) && (B.nvals() > 0))
            {
                // create a column of result at a time
                CColType T_col;
                for (IndexType col_idx = 0; col_idx < ncol_B; ++col_idx)
                {
                    BColType B_col(B.getCol(col_idx));

                    if (!B_col.empty())
                    {
                        for (IndexType row_idx = 0; row_idx < nrow_A; ++row_idx)
                        {
                            ARowType const &A_row(A.getRow(row_idx));
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
                            T.setCol(col_idx, T_col);
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
                ewise_or(tmp_row, C.getRow(row_idx), T.getRow(row_idx), accum);
                C.setRow(row_idx, tmp_row);
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
        inline void mxm(CMatrixT       &C,
                        MMatrixT const &M,              // Mask
                        AccumT          accum,
                        SemiringT       op,
                        AMatrixT const &A,
                        BMatrixT const &B,
                        bool            replace = false)
        {
            IndexType nrow_A(A.nrows());
            IndexType ncol_A(A.ncols());
            IndexType nrow_B(B.nrows());
            IndexType ncol_B(B.ncols());
            IndexType nrow_C(C.nrows());
            IndexType ncol_C(C.ncols());
            IndexType nrow_M(M.nrows());
            IndexType ncol_M(M.ncols());

            // @todo: Move these checks into the front end when we get that wired up
            // @todo: Move these checks into the front end when we get that wired up
            check_inside_dimensions(A, "A", B, "B");
            check_outside_dimensions(A, "A", B, "B", C, "C");
            check_dimensions(A, "A", M, "M");

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
            if ((A.nvals() > 0) && (B.nvals() > 0))
            {
                // create a column of results at a time in the T matrix
                CColType T_col;

                for (IndexType col_idx = 0; col_idx < ncol_B; ++col_idx)
                {
                    BColType B_col(B.getCol(col_idx));
                    MColType M_col(M.getCol(col_idx));

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
                                    ARowType const &A_row(A.getRow(row_idx));
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
                            T.setCol(col_idx, T_col);
                            T_col.clear();
                        }
                    }
                }
            }

            /// @todo  Detect if accum is GrB_NULL, and take a short cut
            // Perform accum

            // NOTE: We do not consult the mask!  Since no value should have been
            // generated above, and we ASSUME ('cause we looked at the code) that 
            // the ewise_or does a no-op (accum-wise) on cells that have no value, then we 
            // shouldn't need to mask here as well.
            CColType tmp_row;
            if (replace)
            {
                for (IndexType row_idx = 0; row_idx < nrow_C; ++row_idx)
                {
                    ewise_or_mask(tmp_row, C.getRow(row_idx), T.getRow(row_idx), M.getRow(row_idx), accum);
                    C.setRow(row_idx, tmp_row);
                }
            }
            else
            {
                for (IndexType row_idx = 0; row_idx < nrow_C; ++row_idx)
                {
                    ewise_or(tmp_row, C.getRow(row_idx), T.getRow(row_idx), accum);
                    C.setRow(row_idx, tmp_row);
                }
            }
        }
    } // backend


} // GraphBLAS

#endif
