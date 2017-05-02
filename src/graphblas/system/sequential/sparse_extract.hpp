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
#include <graphblas/accum.hpp>
#include <graphblas/algebra.hpp>

#include "sparse_helpers.hpp"
#include "LilSparseMatrix.hpp"


//*************************************************************************************************

namespace GraphBLAS
{
    namespace backend
    {

        //************************************************************************

        // Move to sparse helpers
        /**
         * Extracts a series of values from the vector based on the passed in
         * indicies.
         * @tparam CScalarT The type of the output scalar.
         * @tparam AScalarT The type of the input scalar.
         * @param vec_dest The output vector.
         * @param vec_src The input vector.
         * @param indicies The indicies to extract.
         */
        template < typename CScalarT, typename AScalarT>
        void vectorExtract(std::vector< std::tuple<IndexType, CScalarT> >  &vec_dest,
                           std::vector< std::tuple<IndexType, AScalarT> > const &vec_src,
                           IndexArrayType const & indicies)
        {
            // This is expensive but the indicies can be out of duplicate and
            // out of order.

            vec_dest.clear();
            for (IndexType out_idx = 0; out_idx < indicies.size(); ++out_idx)
            {
                IndexType wanted_idx = indicies[out_idx];
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
         * Extract a sub matrix from A to C as specified via the row indicies.
         * This is always destructive to C.
         * @tparam CScalarT The type of scalar in C.
         * @tparam AScalarT The type of scalar in A.
         * @param C Where to place the outputs
         * @param A The input matrix.  (Won't be changed)
         * @param row_indicies A set of indicies indicating which rows to extract.
         * @param col_indicies A set of indicies indicating which columns to extract.
         */
        template<typename CScalarT,
                 typename AScalarT>
        void matrixExtract(LilSparseMatrix<CScalarT>          &C,
                           LilSparseMatrix<AScalarT>  const   &A,
                           IndexArrayType             const   &row_indicies,
                           IndexArrayType             const   &col_indicies)
        {
            // NOTE!! - Backend code. We expect that all dimension checks done elsewhere.

            typedef std::vector<std::tuple<IndexType,AScalarT> > ARowType;
            typedef std::vector<std::tuple<IndexType,CScalarT> > CRowType;

            C.clear();

            // Walk the rows
            for (IndexType out_row_index = 0;
                 out_row_index < row_indicies.size();
                 ++out_row_index)
            {
                ARowType row(A.getRow(row_indicies[out_row_index]));
                auto row_it = row.begin();

                IndexType tmp_idx;
                AScalarT tmp_value;
                CRowType out_row;

                // Extract the values from the row
                vectorExtract(out_row, row, col_indicies);

                if (!out_row.empty())
                    C.setRow(out_row_index, out_row);
            }
        }

        //**********************************************************************
        //**********************************************************************
        //**********************************************************************

        // Vector variant

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
                     MMatrixT           const   &mask,
                     AccumT                      accum,
                     AMatrixT           const   &A,
                     IndexArrayType     const   &row_indicies,
                     IndexArrayType     const   &col_indicies,
                     bool                        replace = false)
        // Descriptor
        {
            // Add in dimensional checks
            if (row_indicies.size() != C.nrows())
            {
                throw DimensionException(
                        "Number of ROWS in output (" +
                        std::to_string(C.nrows()) +
                        ") does not indicated number of ROWS indicies (" +
                        std::to_string(row_indicies.size()) + " ).");
            }

            if (col_indicies.size() != C.ncols())
            {
                throw DimensionException(
                        "Number of COLUMNS in output (" +
                        std::to_string(C.ncols()) +
                        ") does not indicated number of COLUMN indicies (" +
                        std::to_string(col_indicies.size()) + " ).");
            }

            // Validate other inputs
            check_dimensions(C, "C", mask, "mask");

            typedef typename CMatrixT::ScalarType CScalarType;
            typedef std::vector<std::tuple<IndexType,CScalarType> > CColType;

            //std::cerr << ">>> C in <<< " << std::endl;
            //std::cerr << C << std::endl;

            //std::cerr << ">>> Mask <<< " << std::endl;
            //std::cerr << mask << std::endl;

            // =================================================================
            // Extract to T
            LilSparseMatrix<CScalarType> T(C.nrows(), C.ncols());
            matrixExtract(T, A, row_indicies, col_indicies);

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
            write_with_opt_mask(C, Z, mask, replace);

            //std::cerr << ">>> C <<< " << std::endl;
            //std::cerr << C << std::endl;
        };

    }
}



#endif //GB_SEQUENTIAL_SPARSE_EXTRACT_HPP
