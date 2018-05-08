/*
 * Copyright (c) 2018 Carnegie Mellon University.
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
 * Implementation of the sparse matrix extract function.
 */
#ifndef GB_SEQUENTIAL_SPARSE_EXTRACT_HPP
#define GB_SEQUENTIAL_SPARSE_EXTRACT_HPP

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

namespace GraphBLAS
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
                IndexType tmp_idx;
                AScalarT tmp_value;

                // Search through the outputs find one that matches.
                auto A_it = vec_src.begin();
                increment_while_below(A_it, vec_src.end(), wanted_idx);
                if (A_it != vec_src.end())
                {
                    std::tie(tmp_idx, tmp_value) = *A_it;
                    if (tmp_idx == wanted_idx)
                        vec_dest.push_back(
                                std::make_tuple(out_idx,
                                                static_cast<CScalarT>(tmp_value)));
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
            //typedef std::vector<std::tuple<IndexType,AScalarT> > ARowType;
            typedef std::vector<std::tuple<IndexType,CScalarT> > CRowType;

            C.clear();

            // Walk the rows
            IndexType out_row_index = 0;

            for (auto row_it = row_begin;
                 row_it != row_end;
                 ++row_it, ++out_row_index)
            {
                auto row(A.getRow(*row_it));

                IndexType tmp_idx;
                AScalarT tmp_value;
                CRowType out_row;

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
        void matrixExtract(LilSparseMatrix<CScalarT>              &C,
                           backend::TransposeView<AMatrixT> const &A,
                           RowIteratorT                            row_begin,
                           RowIteratorT                            row_end,
                           ColIteratorT                            col_begin,
                           ColIteratorT                            col_end)
        {
            typedef typename AMatrixT::ScalarType AScalarT;
            //typedef std::vector<std::tuple<IndexType,AScalarT> > ARowType;
            typedef std::vector<std::tuple<IndexType,CScalarT> > CRowType;

            C.clear();

            // Walk the rows
            IndexType out_row_index = 0;

            for (auto row_it = row_begin;
                 row_it != row_end;
                 ++row_it, ++out_row_index)
            {
                auto row(A.getRow(*row_it));

                IndexType tmp_idx;
                AScalarT tmp_value;
                CRowType out_row;

                // Extract the values from the row
                vectorExtract(out_row, row, col_begin, col_end);

                if (!out_row.empty())
                    C.setRow(out_row_index, out_row);
            }
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
            // NOTE!! - Backend code. We expect that all dimension checks done elsewhere.

            matrixExtract(C, A,
                          row_indices.begin(), row_indices.end(),
                          col_indices.begin(), col_indices.end());


        }

        //********************************************************************
        template <typename WScalarT, typename AScalarT, typename IteratorT>
        void extractColumn(
            std::vector< std::tuple<IndexType, WScalarT> >         &vec_dest,
            LilSparseMatrix<AScalarT>                       const  &A,
            IteratorT                                               row_begin,
            IteratorT                                               row_end,
            IndexType                                               col_index)
        {
            // Walk the rows, extracting the cell if it exists
            typedef std::vector<std::tuple<IndexType,AScalarT> > ARowType;

            vec_dest.clear();

            // Walk the rows.

            IndexType out_row_index = 0;
            for (IteratorT it = row_begin; it != row_end; ++it, ++out_row_index)
            {
                ARowType row(A.getRow(*it));

                IndexType tmp_idx;
                AScalarT tmp_value;

                // Now, find the column
                auto row_it = row.begin();
                while (row_it != row.end())
                {
                    std::tie(tmp_idx, tmp_value) = *row_it;
                    if (tmp_idx == col_index)
                    {
                        vec_dest.push_back(
                                std::make_tuple(out_row_index,
                                                static_cast<WScalarT>(tmp_value)));
                        break;
                    }
                    else if (tmp_idx > col_index)
                    {
                        break;
                    }
                    ++row_it;
                }
            }
        };

        //********************************************************************
        // Extract a row of a matrix using TransposeView
        template <typename WScalarT, typename AMatrixT, typename IteratorT>
        void extractColumn(
            std::vector< std::tuple<IndexType, WScalarT> >  &vec_dest,
            backend::TransposeView<AMatrixT> const          &Atrans,
            IteratorT                                        row_begin,
            IteratorT                                        row_end,
            IndexType                                        col_index)
        {
            // Walk the row, extracting the cell if it exists and is in row_indices
            typedef typename AMatrixT::ScalarType AScalarType;
            typedef std::vector<std::tuple<IndexType,AScalarType> > ARowType;

            vec_dest.clear();

            auto row(Atrans.getCol(col_index));

            // Walk the 'row'
            /// @todo Perf. can be improved for 'in order' row_indices with "continuation"
            IndexType out_row_index = 0;

            //for (IndexType idx = 0; idx < row_indices.size(); ++idx)
            for (IteratorT it = row_begin; it != row_end; ++it, ++out_row_index)
            {
                auto row_it = row.begin();
                while (row_it != row.end())
                {
                    IndexType in_row_index(std::get<0>(*row_it));
                    if (in_row_index == *it) //row_indices[idx])
                    {
                        vec_dest.push_back(
                            std::make_tuple(out_row_index, //idx,
                                            static_cast<WScalarT>(std::get<1>(*row_it))));
                    }
                    ++row_it;
                }
            } // for
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
                     AccumT                    accum,
                     UVectorT           const &u,
                     SequenceT          const &indices,
                     bool                      replace_flag = false)
        {
            check_index_array_content(indices, u.size(),
                                      "extract(std vec): indices >= u.size");

            typedef typename WVectorT::ScalarType WScalarType;
            typedef std::vector<std::tuple<IndexType,WScalarType> > CColType;

            GRB_LOG_VERBOSE("U inside: " << u);

            // =================================================================
            // Extract to T
            typedef typename UVectorT::ScalarType UScalarType;
            std::vector<std::tuple<IndexType, UScalarType> > t;
            auto u_contents(u.getContents());
            vectorExtract(t, u_contents,
                          setupIndices(indices,
                                       std::min(w.size(), u.size())));

            GRB_LOG_VERBOSE("T: " << t);

            // =================================================================
            // Accumulate into Z
            typedef typename std::conditional<
                std::is_same<AccumT, NoAccumulate>::value,
                UScalarType,
                typename AccumT::result_type>::type  ZScalarType;

            std::vector<std::tuple<IndexType, ZScalarType> > z;
            ewise_or_opt_accum_1D(z, w, t, accum);

            GRB_LOG_VERBOSE("Z: " << z);

            // =================================================================
            // Copy Z into the final output considering mask and replace
            write_with_opt_mask_1D(w, z, mask, replace_flag);

            GRB_LOG_VERBOSE("W (Result): " << w);
        };

        //**********************************************************************
        /**
         * 4.3.6.2 extract: Standard matrix variant
         * Extract a sub-matrix from a larger matrix as specied by a set of row
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
                     AccumT                      accum,
                     AMatrixT           const   &A,
                     RowSequenceT       const   &row_indices,
                     ColSequenceT       const   &col_indices,
                     bool                        replace_flag = false)
        {
            check_index_array_content(row_indices, A.nrows(),
                                      "extract(std mat): row_indices >= A.nrows");
            check_index_array_content(col_indices, A.ncols(),
                                      "extract(std mat): col_indices >= A.ncols");

            // =================================================================
            // Extract to T
            typedef typename AMatrixT::ScalarType AScalarType;
            LilSparseMatrix<AScalarType> T(C.nrows(), C.ncols());
            matrixExtract(T, A,
                          setupIndices(row_indices,
                                       std::min(A.nrows(), C.nrows())),
                          setupIndices(col_indices,
                                       std::min(A.ncols(), C.ncols())));

            GRB_LOG_VERBOSE("T: " << T);

            // =================================================================
            // Accumulate into Z
            typedef typename std::conditional<
                std::is_same<AccumT, NoAccumulate>::value,
                AScalarType,
                typename AccumT::result_type>::type  ZScalarType;

            LilSparseMatrix<ZScalarType> Z(C.nrows(), C.ncols());
            ewise_or_opt_accum(Z, C, T, accum);

            GRB_LOG_VERBOSE("Z: " << Z);

            // =================================================================
            // Copy Z into the final output considering mask and replace
            write_with_opt_mask(C, Z, Mask, replace_flag);

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
                     AccumT                    accum,
                     AMatrixT           const &A,
                     SequenceT          const &row_indices,
                     IndexType                 col_index,
                     bool                      replace_flag = false)
        {
            check_index_array_content(row_indices, A.nrows(),
                                      "extract(col): row_indices >= A.nrows");

            // =================================================================
            // Extract to T
            typedef typename AMatrixT::ScalarType AScalarType;
            typedef std::vector<std::tuple<IndexType, AScalarType>> TVectorType;
            TVectorType t;

            auto seq = setupIndices(row_indices,
                                    std::min(A.nrows(), w.size()));
            extractColumn(t, A, seq.begin(), seq.end(), col_index);

            GRB_LOG_VERBOSE("t: " << t);

            // =================================================================
            // Accumulate into Z
            typedef typename std::conditional<
                std::is_same<AccumT, NoAccumulate>::value,
                AScalarType,
                typename AccumT::result_type>::type  ZScalarType;

            std::vector<std::tuple<IndexType, ZScalarType> > z;
            ewise_or_opt_accum_1D(z, w, t, accum);

            GRB_LOG_VERBOSE("z: " << z);

            // =================================================================
            // Copy Z into the final output considering mask and replace
            write_with_opt_mask_1D(w, z, mask, replace_flag);

            GRB_LOG_VERBOSE("w (Result): " << w);
        }
    }
}



#endif //GB_SEQUENTIAL_SPARSE_EXTRACT_HPP
