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
 * Implementation of the sparse matrix extract function.
 */
#ifndef GB_SEQUENTIAL_SPARSE_EXTRACT_HPP
#define GB_SEQUENTIAL_SPARSE_EXTRACT_HPP

#pragma once

#include <functional>
#include <utility>
#include <vector>
#include <iterator>
#include <iostream>
#include <graphblas/types.hpp>
#include <graphblas/exceptions.hpp>
#include <graphblas/algebra.hpp>

#include "sparse_helpers.hpp"
#include "LilSparseMatrix.hpp"


//******************************************************************************

namespace GraphBLAS
{
    namespace backend
    {
        //**********************************************************************

        template < typename CScalarT,
                   typename AScalarT,
                   typename IteratorT>
        void vectorExtract(
                std::vector< std::tuple<IndexType, CScalarT> >  &vec_dest,
                std::vector< std::tuple<IndexType, AScalarT> > const &vec_src,
                IteratorT           begin,
                IteratorT           end)
        {
            // This is expensive but the indices can be duplicates and
            // out of order.

            vec_dest.clear();

            IndexType out_idx = 0;
            for (auto col_it = begin; col_it != end; ++col_it, ++out_idx)
            {
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

        /**
         * Extracts a series of values from the vector based on the passed in
         * indices.
         * @tparam CScalarT The type of the output scalar.
         * @tparam AScalarT The type of the input scalar.
         * @param vec_dest The output vector.
         * @param vec_src The input vector.
         * @param indices The indices to extract.
         */
        template < typename CScalarT, typename AScalarT>
        void vectorExtract(
            std::vector< std::tuple<IndexType, CScalarT> >  &vec_dest,
            std::vector< std::tuple<IndexType, AScalarT> > const &vec_src,
            IndexArrayType const & indices)
        {
            // This is expensive but the indices can be duplicates and
            // out of order.

            vec_dest.clear();
            for (IndexType out_idx = 0; out_idx < indices.size(); ++out_idx)
            {
                IndexType wanted_idx = indices[out_idx];
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
                 typename AMatrixT>
        void matrixExtract(CMatrixT                           &C,
                           AMatrixT                   const   &A,
                           IndexArrayType             const   &row_indices,
                           IndexArrayType             const   &col_indices)
        {
            // NOTE!! - Backend code. We expect that all dimension checks done elsewhere.

            if (&row_indices == &GrB_ALL && &col_indices == &GrB_ALL)
            {
                matrixExtract(C, A,
                              index_iterator(0), index_iterator(A.nrows()),
                              index_iterator(0), index_iterator(A.ncols()));
            }
            else if (&row_indices == &GrB_ALL)
            {
                matrixExtract(C, A,
                              index_iterator(0), index_iterator(A.nrows()),
                              col_indices.begin(), col_indices.end());
            }
            else if (&col_indices == &GrB_ALL)
            {
                matrixExtract(C, A,
                              row_indices.begin(), row_indices.end(),
                              index_iterator(0), index_iterator(A.ncols()));
            }
            else
            {
                matrixExtract(C, A,
                              row_indices.begin(), row_indices.end(),
                              col_indices.begin(), col_indices.end());
            }
        }
#if 0
        /// deprecated
        template<typename CScalarT,
                 typename AScalarT>
        void matrixExtract(LilSparseMatrix<CScalarT>          &C,
                           LilSparseMatrix<AScalarT>  const   &A,
                           IndexArrayType             const   &row_indices,
                           IndexArrayType             const   &col_indices)
        {
            // NOTE!! - Backend code. We expect that all dimension checks done elsewhere.

            if (&row_indices == &GrB_ALL && &col_indices == &GrB_ALL)
            {
                matrixExtract(C, A,
                              index_iterator(0), index_iterator(A.nrows()),
                              index_iterator(0), index_iterator(A.ncols()));
            }
            else if (&row_indices == &GrB_ALL)
            {
                matrixExtract(C, A,
                              index_iterator(0), index_iterator(A.nrows()),
                              col_indices.begin(), col_indices.end());
            }
            else if (&col_indices == &GrB_ALL)
            {
                matrixExtract(C, A,
                              row_indices.begin(), row_indices.end(),
                              index_iterator(0), index_iterator(A.ncols()));
            }
            else
            {
                matrixExtract(C, A,
                              row_indices.begin(), row_indices.end(),
                              col_indices.begin(), col_indices.end());
            }
        }
#endif
        //********************************************************************
        template < typename WScalarT, typename AScalarT, typename IteratorT>
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
        template < typename WScalarT, typename AMatrixT, typename IteratorT>
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
            /// @todo Perf. can be improved for in order row_indices with "continuation"
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

#if 0
        //********************************************************************
        // Extract a row of a matrix using TransposeView
        template < typename WScalarT, typename AMatrixT, typename IteratorT>
        void extractColumn(
            std::vector< std::tuple<IndexType, WScalarT> >        &vec_dest,
            backend::TransposeView<AMatrixT>                const &Atrans,
            IndexArrayType                                  const &row_indices,
            IndexType                                              col_index)
        {
            // Walk the row, extracting the cell if it exists and is in row_indices
            typedef typename AMatrixT::ScalarType AScalarType;
            typedef std::vector<std::tuple<IndexType,AScalarType> > ARowType;

            vec_dest.clear();

            ARowType row(Atrans.getCol(col_index));

            // Walk the 'row'
            /// @todo Perf. can be improved for in order row_indices with "continuation"
            for (IndexType idx = 0; idx < row_indices.size(); ++idx)
            {
                auto row_it = row.begin();
                while (row_it != row.end())
                {
                    IndexType in_row_index(std::get<0>(*row_it));
                    if (in_row_index == row_indices[idx])
                    {
                        vec_dest.push_back(
                            std::make_tuple(idx,
                                            static_cast<WScalarT>(std::get<1>(*row_it))));
                    }
                    ++row_it;
                }
            } // for
        }
#endif
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
                 typename UVectorT >
        void extract(WVectorT                   &w,
                     MVectorT           const   &mask,
                     AccumT                      accum,
                     UVectorT           const   &u,
                     IndexArrayType     const   &indices,
                     bool                        replace = false)
        {
            // Add in dimensional checks
            if (indices.size() != w.size())
            {
                throw DimensionException(
                        "Size of output (" +
                        std::to_string(w.size()) +
                        ") does not equal size of indices (" +
                        std::to_string(indices.size()) + " ).");
            }

            // Validate other inputs
            check_vector_size(w, mask, "size(w) != size(mask)");

            typedef typename WVectorT::ScalarType WScalarType;
            typedef std::vector<std::tuple<IndexType,WScalarType> > CColType;

            // =================================================================
            // Extract to T
            typedef typename UVectorT::ScalarType UScalarType;
            std::vector<std::tuple<IndexType, UScalarType> > t;
            auto u_contents(u.getContents());
            vectorExtract(t, u_contents, indices);

            // =================================================================
            // Accumulate into Z
            std::vector<std::tuple<IndexType, WScalarType> > z;
            ewise_or_opt_accum_1D(z, w, t, accum);

            // =================================================================
            // Copy Z into the final output considering mask and replace
            write_with_opt_mask_1D(w, z, mask, replace);
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
                 typename AMatrixT >
        void extract(CMatrixT                   &C,
                     MMatrixT           const   &Mask,
                     AccumT                      accum,
                     AMatrixT           const   &A,
                     IndexArrayType     const   &row_indices,
                     IndexArrayType     const   &col_indices,
                     bool                        replace = false)
        {
            // Validate inputs
            check_matrix_size(C, Mask, "extract(matrix) Mask dimension check");

            // Add in dimensional checks
            check_index_array_dimension(
                row_indices, C.nrows(),
                "extract row_indices dimension: Number of ROWS in output (" +
                std::to_string(C.nrows()) +
                ") does not indicated number of ROWS indices (" +
                std::to_string(row_indices.size()) + " ).");
            check_index_array_dimension(
                col_indices, C.ncols(),
                "extract col_indices dimension: Number of COLUMNS in output (" +
                std::to_string(C.ncols()) +
                ") does not indicated number of COLUMN indices (" +
                std::to_string(col_indices.size()) + " ).");

            check_index_array_content(row_indices, A.nrows(),
                                      "extract row_indices content check");
            check_index_array_content(col_indices, A.ncols(),
                                      "extract col_indices content check");

            typedef typename CMatrixT::ScalarType CScalarType;
            typedef std::vector<std::tuple<IndexType,CScalarType> > CColType;

            //std::cerr << ">>> C in <<< " << std::endl;
            //std::cerr << C << std::endl;

            //std::cerr << ">>> Mask <<< " << std::endl;
            //std::cerr << Mask << std::endl;

            // =================================================================
            // Extract to T
            LilSparseMatrix<CScalarType> T(C.nrows(), C.ncols());
            matrixExtract(T, A, row_indices, col_indices);

            //std::cerr << ">>> T <<< " << std::endl;
            //std::cerr << T << std::endl;

            // =================================================================
            // Accumulate into Z

            LilSparseMatrix<CScalarType> Z(C.nrows(), C.ncols());
            ewise_or_opt_accum(Z, C, T, accum);

            //std::cerr << ">>> Z <<< " << std::endl;
            //std::cerr << Z << std::endl;

            // =================================================================
            // Copy Z into the final output considering mask and replace
            write_with_opt_mask(C, Z, Mask, replace);

            //std::cerr << ">>> C <<< " << std::endl;
            //std::cerr << C << std::endl;
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
                 typename MaskT,
                 typename AccumT,
                 typename AMatrixT>
        void extract(WVectorT                  &w,
                     MaskT              const  &mask,
                     AccumT                     accum,
                     AMatrixT           const  &A,
                     IndexArrayType     const  &row_indices,
                     IndexType                  col_index,
                     bool                       replace_flag = false)
        {
            // @TODO: Validate inputs

            // Should explicitly define the vector type, or just piggy
            // back on WVectorT?
            //WVectorT wTmp;
            typedef typename WVectorT::ScalarType WScalarType;
            typedef std::vector<std::tuple<IndexType, WScalarType>> WVectorType;

            // =================================================================
            // Extract to T
            WVectorType t;

            if (&row_indices == &GrB_ALL)
            {
                //std::cout << "!!!!==== GrB_ALL - Extract!!" << std::endl;
                extractColumn(t, A, index_iterator(0),
                              index_iterator(A.nrows()), col_index);
            }
            else
            {
                extractColumn(t, A, row_indices.begin(), row_indices.end(), col_index);
            }
            //std::cerr << ">>> t <<< " << std::endl;
            //pretty_print(std::cerr, t);

            // =================================================================
            // Accumulate into Z
            std::vector<std::tuple<IndexType, WScalarType> > z;
            ewise_or_opt_accum_1D(z, w, t, accum);

            //std::cerr << ">>> z <<< " << std::endl;
            //pretty_print(std::cerr, z);

            // =================================================================
            // Copy Z into the final output considering mask and replace
            write_with_opt_mask_1D(w, z, mask, replace_flag);

            //std::cerr << ">>> w <<< " << std::endl;
            //std::cerr << w << std::endl;
        }
    }
}



#endif //GB_SEQUENTIAL_SPARSE_EXTRACT_HPP
