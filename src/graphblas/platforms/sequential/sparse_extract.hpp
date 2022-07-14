/*
 * GraphBLAS Template Library (GBTL), Version 3.0
 *
 * Copyright 2020 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors.
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
 * RIGHTS.
 *
 * Released under a BSD-style license, please see LICENSE file or contact
 * permission@sei.cmu.edu for full terms.
 *
 * [DISTRIBUTION STATEMENT A] This material has been approved for public release
 * and unlimited distribution.  Please see Copyright notice for non-US
 * Government use and distribution.
 *
 * DM20-0442
 */

#pragma once

#include <functional>
#include <utility>
#include <vector>
#include <iterator>
#include <type_traits>
#include <iostream>

#include <graphblas/detail/logging.h>
#include <graphblas/types.hpp>
#include <graphblas/exceptions.hpp>
#include <graphblas/algebra.hpp>
#include <graphblas/indices.hpp>

#include "sparse_helpers.hpp"
#include "LilSparseMatrix.hpp"

//******************************************************************************

namespace grb
{
    namespace backend
    {
        //**********************************************************************
        /**
         * Extracts a series of values from the vector based on the passed in
         * indices.
         * @tparam CScalarT  The type of the output scalar.
         * @tparam AScalarT  The type of the input scalar.
         * @tparam SequenceT A random access iterator into a container of indices
         *
         * @param vec_dest The output vector.
         * @param vec_src The input vector.
         * @param begin   Iterator at begining of sequence of indices to extract.
         * @param end     Iterator at end of sequence of indices to extract.
         */
        template<typename CScalarT,
                 typename AScalarT,
                 typename IteratorT>
        void vectorExtract(
                std::vector<std::tuple<IndexType, CScalarT> >       &vec_dest,
                std::vector<std::tuple<IndexType, AScalarT> > const &vec_src,
                IteratorT           begin,
                IteratorT           end)
        {
            // This is expensive but the indices can be duplicates and
            // out of order.

            vec_dest.clear();

            GRB_LOG_VERBOSE("vectorExtract: sizeof(vec_src): " << vec_src.size());

            IndexType out_idx = 0;
            for (auto col_it = begin; col_it != end; ++col_it, ++out_idx)
            {
                GRB_LOG_VERBOSE("out_idx = " << out_idx);
                IndexType wanted_idx = *col_it;

                // Search through the outputs find one that matches.
                auto A_it = vec_src.begin();
                if (increment_while_below(A_it, vec_src.end(), wanted_idx))
                {
                    vec_dest.emplace_back(
                        out_idx, static_cast<CScalarT>(std::get<1>(*A_it)));
                }
            }
        }

        // *******************************************************************
        template<typename CScalarT,
                 typename AScalarT,
                 typename SequenceT>
        void vectorExtract(
                std::vector<std::tuple<IndexType, CScalarT> >       &vec_dest,
                std::vector<std::tuple<IndexType, AScalarT> > const &vec_src,
                SequenceT                                            indices)
        {
            vectorExtract(vec_dest, vec_src, indices.begin(), indices.end());
        }

        // *******************************************************************
        template<typename CScalarT,
                 typename AScalarT,
                 typename RowIteratorT,
                 typename ColIteratorT>
        void matrixExtract(LilSparseMatrix<CScalarT>          &C,
                           LilSparseMatrix<AScalarT>  const   &A,
                           RowIteratorT                        row_begin,
                           RowIteratorT                        row_end,
                           ColIteratorT                        col_begin,
                           ColIteratorT                        col_end)
        {
            std::vector<std::tuple<IndexType,CScalarT> > out_row;
            C.clear();

            // Walk the rows
            IndexType out_row_index = 0;

            for (auto row_it = row_begin;
                 row_it != row_end;
                 ++row_it, ++out_row_index)
            {
                auto row(A[*row_it]);

                // Extract the values from the row
                vectorExtract(out_row, row, col_begin, col_end);

                if (!out_row.empty())
                    C.setRow(out_row_index, out_row);
            }
        }

        // *******************************************************************
        template<typename CScalarT,
                 typename AMatrixT,
                 typename RowIteratorT,
                 typename ColIteratorT>
        void matrixExtract(LilSparseMatrix<CScalarT>     &C,
                           TransposeView<AMatrixT> const &AT,
                           RowIteratorT                   row_begin, // of AT
                           RowIteratorT                   row_end,
                           ColIteratorT                   col_begin, // of AT
                           ColIteratorT                   col_end)
        {
            auto const &A(AT.m_mat);
            C.clear();

            // Walk the rows of A (cols of AT) and put into columns of C.
            IndexType out_row_idx = 0;

            for (auto col_it = col_begin;  col_it != col_end;
                 ++col_it, ++out_row_idx)
            {
                GRB_LOG_VERBOSE("matrixExtract(AT): out_row(C)=" << out_row_idx
                                << ", in_col(AT): " << *col_it);

                // Extract the values from the rows of A (cols of AT) and place
                // them into the *colums* of C
                //
                // Emplace_back version of:
                //    vectorExtract(out_row, A[*col_it], row_begin, row_end);

                IndexType out_col_idx = 0;
                for (auto row_it = row_begin; row_it != row_end;
                     ++row_it, ++out_col_idx)
                {
                    GRB_LOG_VERBOSE("matrixExtract(AT): out_col(C)=" << out_col_idx
                                    << ", in_row(AT): " << *row_it);

                    IndexType wanted_idx = *row_it;

                    // Search through from the beginning of the row each to find
                    // indices that match. This is expensive but the indices can
                    // be duplicates and out of order.
                    auto A_it = A[*col_it].begin();
                    if (increment_while_below(A_it, A[*col_it].end(), wanted_idx))
                    {
                        GRB_LOG_VERBOSE("C["<<out_row_idx << ","
                                        << out_col_idx << "] := "
                                        << std::get<1>(*A_it));

                        C[out_col_idx].emplace_back(
                            out_row_idx,
                            static_cast<CScalarT>(std::get<1>(*A_it)));
                    }
                }
            }
            C.recomputeNvals();
        }

        /**
         * Extract a sub matrix from A to C as specified via the row indices.
         * This is always destructive to C.
         * @tparam CMatrixT The type of matrix for C
         * @tparam AMatrixT The type of matrix for A
         * @param C Where to place the outputs
         * @param A The input matrix.  (Won't be changed)
         * @param row_indices A set of indices indicating which rows to extract.
         * @param col_indices A set of indices indicating which columns to extract.
         */
        template<typename CMatrixT,
                 typename AMatrixT,
                 typename RowSequenceT,
                 typename ColSequenceT>
        void matrixExtract(CMatrixT                           &C,
                           AMatrixT                   const   &A,
                           RowSequenceT               const   &row_indices,
                           ColSequenceT               const   &col_indices)
        {
            // NOTE!! Backend code. We expect that all dimension checks done elsewhere.

            matrixExtract(C, A,
                          row_indices.begin(), row_indices.end(),
                          col_indices.begin(), col_indices.end());


        }

        //********************************************************************
        template <typename WScalarT, typename AScalarT, typename IteratorT>
        void extractColumn(
            std::vector< std::tuple<IndexType, WScalarT> >         &vec_dest,
            LilSparseMatrix<AScalarT>                        const &A,
            IteratorT                                               row_begin,
            IteratorT                                               row_end,
            IndexType                                               col_index)
        {
            vec_dest.clear();

            // Walk the rows, extracting the cell if it exists
            IndexType out_row_index = 0;
            for (IteratorT it = row_begin; it != row_end; ++it, ++out_row_index)
            {
                // Find the column within the row
                for (auto&& [tmp_idx, tmp_value] : A[*it])
                {
                    if (tmp_idx == col_index)
                    {
                        vec_dest.emplace_back(out_row_index,
                                              static_cast<WScalarT>(tmp_value));
                        break;
                    }
                    else if (tmp_idx > col_index)
                    {
                        break;
                    }
                }
            }
        };

        //********************************************************************
        // Extract a row of a TransposeView of a matrix
        template <typename WScalarT, typename AMatrixT, typename IteratorT>
        void extractColumn(
            std::vector< std::tuple<IndexType, WScalarT> >        &vec_dest,
            TransposeView<AMatrixT>                         const &AT,
            IteratorT                                              row_begin,
            IteratorT                                              row_end,
            IndexType                                              col_index)
        {
            auto const &row(AT.m_mat[col_index]);
            vec_dest.clear();

            // Walk the row, extracting the cell if it exists and is in row_indices

            /// @todo Perf. can be improved for 'in order' row_indices
            /// with "continuation"
            IndexType out_row_index = 0;

            for (IteratorT it = row_begin; it != row_end; ++it, ++out_row_index)
            {
                /// @todo Perf: replace this scan with a binary search

                // Costly: keep rewalking same row due to duplicates
                // and/or out of order indices
                for (auto&& [in_row_index, in_val] : row)
                {
                    if (in_row_index == *it)
                    {
                        vec_dest.emplace_back(
                            out_row_index, static_cast<WScalarT>(in_val));
                    }
                    else if (in_row_index > *it)
                    {
                        break;
                    }
                }
            }
        }

        //**********************************************************************
        //**********************************************************************
        //**********************************************************************

        // Vector variant

        /**
         * 4.3.6.1 extract: Standard vector variant
         * Extract a sub-vector from a larger vector as specified by a set of row
         *  indices and a set of column indices. The result is a vector whose
         *  size is equal to size of the sets of indices.
         */
        template<typename WVectorT,
                 typename MVectorT,
                 typename AccumT,
                 typename UVectorT,
                 typename SequenceT>
        void extract(WVectorT                 &w,
                     MVectorT           const &mask,
                     AccumT             const &accum,
                     UVectorT           const &u,
                     SequenceT          const &indices,
                     OutputControlEnum         outp)
        {
            GRB_LOG_VERBOSE("w<m,z> := u(indices)");
            check_index_array_content(indices, u.size(),
                                      "extract(std vec): indices >= u.size");

            GRB_LOG_VERBOSE("u inside: " << u);

            // =================================================================
            // Extract to T
            using UScalarType =typename UVectorT::ScalarType;
            std::vector<std::tuple<IndexType, UScalarType> > t;
            vectorExtract(t, u.getContents(),
                          setupIndices(indices,
                                       std::min(w.size(), u.size())));

            GRB_LOG_VERBOSE("t: " << t);

            // =================================================================
            // Accumulate into Z
            using ZScalarType = typename std::conditional_t<
                std::is_same_v<AccumT, NoAccumulate>,
                UScalarType,
                decltype(accum(std::declval<typename WVectorT::ScalarType>(),
                               std::declval<UScalarType>()))>;

            std::vector<std::tuple<IndexType, ZScalarType> > z;
            ewise_or_opt_accum_1D(z, w, t, accum);

            GRB_LOG_VERBOSE("z: " << z);

            // =================================================================
            // Copy Z into the final output considering mask and replace/merge
            write_with_opt_mask_1D(w, z, mask, outp);

            GRB_LOG_VERBOSE("w (Result): " << w);
        };

        //**********************************************************************
        /**
         * 4.3.6.2 extract: Standard matrix variant: C = A(rows, cols)
         * Extract a sub-matrix from a larger matrix as specfied by a set of row
         *  indices and a set of column indices. The result is a matrix whose
         *  size is equal to size of the sets of indices.
         */
        template<typename CMatrixT,
                 typename MMatrixT,
                 typename AccumT,
                 typename AMatrixT,
                 typename RowSequenceT,
                 typename ColSequenceT>
        void extract(CMatrixT                   &C,
                     MMatrixT           const   &Mask,
                     AccumT             const   &accum,
                     AMatrixT           const   &A,
                     RowSequenceT       const   &row_indices,
                     ColSequenceT       const   &col_indices,
                     OutputControlEnum           outp)
        {
            GRB_LOG_VERBOSE("C<M,z> := A(rows, cols) (supports A')");
            check_index_array_content(row_indices, A.nrows(),
                                      "extract(std mat): row_indices >= A.nrows");
            check_index_array_content(col_indices, A.ncols(),
                                      "extract(std mat): col_indices >= A.ncols");

            // =================================================================
            // Extract to T
            using AScalarType = typename AMatrixT::ScalarType;
            LilSparseMatrix<AScalarType> T(C.nrows(), C.ncols());
            matrixExtract(T, A,
                          setupIndices(row_indices,
                                       std::min(A.nrows(), C.nrows())),
                          setupIndices(col_indices,
                                       std::min(A.ncols(), C.ncols())));

            GRB_LOG_VERBOSE("T: " << T);

            // =================================================================
            // Accumulate into Z
            using ZScalarType = typename std::conditional_t<
                std::is_same_v<AccumT, NoAccumulate>,
                AScalarType,
                decltype(accum(std::declval<typename CMatrixT::ScalarType>(),
                               std::declval<AScalarType>()))>;

            LilSparseMatrix<ZScalarType> Z(C.nrows(), C.ncols());
            ewise_or_opt_accum(Z, C, T, accum);

            GRB_LOG_VERBOSE("Z: " << Z);

            // =================================================================
            // Copy Z into the final output considering mask and replace/merge
            write_with_opt_mask(C, Z, Mask, outp);

            GRB_LOG_VERBOSE("C (Result): " << C);
        };

        //**********************************************************************
        /**
         * 4.3.6.3 extract: Column (and row) variant
         *
         * Extract from one column of a matrix into a vector. Note that with
         * the transpose descriptor for the source matrix, elements of an
         * arbitrary row of the matrix can be extracted with this function as
         * well.
         */
        template<typename WVectorT,
                 typename MaskVectorT,
                 typename AccumT,
                 typename AMatrixT,
                 typename SequenceT>
        void extract(WVectorT                 &w,
                     MaskVectorT        const &mask,
                     AccumT             const &accum,
                     AMatrixT           const &A,
                     SequenceT          const &row_indices,
                     IndexType                 col_index,
                     OutputControlEnum         outp)
        {
            GRB_LOG_VERBOSE("C<M,z> := A(rows, j) (supports A')");
            check_index_array_content(row_indices, A.nrows(),
                                      "extract(col): row_indices >= A.nrows");

            // =================================================================
            // Extract to T
            using AScalarType = typename AMatrixT::ScalarType;
            std::vector<std::tuple<IndexType, AScalarType>> t;

            auto seq = setupIndices(row_indices,
                                    std::min(A.nrows(), w.size()));
            extractColumn(t, A, seq.begin(), seq.end(), col_index);

            GRB_LOG_VERBOSE("t: " << t);

            // =================================================================
            // Accumulate into Z
            using ZScalarType = typename std::conditional_t<
                std::is_same_v<AccumT, NoAccumulate>,
                AScalarType,
                decltype(accum(std::declval<typename WVectorT::ScalarType>(),
                               std::declval<AScalarType>()))>;

            std::vector<std::tuple<IndexType, ZScalarType> > z;
            ewise_or_opt_accum_1D(z, w, t, accum);

            GRB_LOG_VERBOSE("z: " << z);

            // =================================================================
            // Copy Z into the final output considering mask and replace/merge
            write_with_opt_mask_1D(w, z, mask, outp);

            GRB_LOG_VERBOSE("w (Result): " << w);
        }
    }
}
