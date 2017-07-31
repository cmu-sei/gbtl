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

#ifndef GB_SEQUENTIAL_SPARSE_ASSIGN_HPP
#define GB_SEQUENTIAL_SPARSE_ASSIGN_HPP

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
        struct IndexCompare
        {
            inline bool operator()(std::pair<IndexType, IndexType> const &i1,
                                   std::pair<IndexType, IndexType> const &i2)
            {
                return i1.first < i2.first;
            }
        };

        // Builds a simple mapping
        void compute_outin_mapping(IndexArrayType const & indicies,
                      std::vector<std::pair<IndexType, IndexType>> &inputOrder)
        {
            // Walk the indicies generating generating pairs showing the existing
            auto index_it = indicies.begin();
            IndexType idx = 0;
            while (index_it != indicies.end())
            {
                inputOrder.push_back(std::make_pair(*index_it, idx));
                ++index_it;
                ++idx;
            }

            // Sort them because we want to deal with them in output order.
            std::sort(inputOrder.begin(), inputOrder.end(), IndexCompare());
        }

        template < typename TScalarT, typename AScalarT>
        void vectorExpand(std::vector< std::tuple<IndexType, TScalarT> >  &vec_dest,
                          std::vector< std::tuple<IndexType, AScalarT> > const &vec_src,
                          std::vector< std::pair<IndexType, IndexType> > const &indicies)
        {
            // The indicies are pairs of ( output_index, input_index)
            // We do it this way, so we get the output in the right
            // order to begin with

            //std::cerr << "Expanding row " << std::endl;

            // Walk the output/input pairs building the output in correct order.
            auto index_it = indicies.begin();

            // We start at the beginning of the source and work our way through
            // it.  We reset to beginning when the input is before us.
            // This way we reduce thrash a little bit.
            auto src_it = vec_src.begin();

            while (index_it != indicies.end())
            {
                IndexType src_idx = 0;
                AScalarT src_val;

                // Walk the source data looking for that value.  If we
                // find it, then we insert into output
                //std::cerr << "Walking source vector " << std::endl;
                while (src_it != vec_src.end())
                {
                    std::tie(src_idx, src_val) = *src_it;
                    if (src_idx == index_it->second)
                    {
                        vec_dest.push_back(std::make_tuple(
                           index_it->first, static_cast<TScalarT>(src_val) ));
                        //std::cerr << "Added dest idx=" << index_it->first << ", val=" << src_val << std::endl;
                        break;
                    }
                    else if (src_idx > index_it->second)
                    {
                        // We passed it.  We might use this later
                        break;
                    }
                    ++src_it;
                }

                // If we didn't find anything in sourece (ran out)
                // then that is okay.  We don't put anything into the
                // output. We don't need to add a sentinel or anything.

                // If we got here we have dealt with the output value.
                // Let's get the next one.
                ++index_it;

                // If the next index is less than where we were, (before)
                // let's start from the beginning again.
                // IMPROVEMENT:  Back up?
                if (index_it != indicies.end() &&
                        (src_it == vec_src.end() || index_it->second < src_idx))
                {
                    //std::cerr << "Resetting src_it " << std::endl;
                    src_it = vec_src.begin();
                }
            }
        }

        template<typename TScalarT,
                 typename AScalarT>
        void matrixExpand(LilSparseMatrix<TScalarT>          &T,
                          LilSparseMatrix<AScalarT>  const   &A,
                          IndexArrayType             const   &row_indicies,
                          IndexArrayType             const   &col_indicies)
        {
            // NOTE!! - Backend code. We expect that all dimension
            // checks done elsewhere.

            typedef std::vector<std::tuple<IndexType,AScalarT> > ARowType;
            typedef std::vector<std::tuple<IndexType,TScalarT> > TRowType;

            T.clear();

            // Build the mapping pairs once up front
            std::vector<std::pair<IndexType, IndexType>> oi_pairs;
            compute_outin_mapping(col_indicies, oi_pairs);

            // Walk the rows
            for (IndexType in_row_index = 0;
                 in_row_index < row_indicies.size();
                 ++in_row_index)
            {
                IndexType out_row_index = row_indicies[in_row_index];
                ARowType row(A.getRow(in_row_index));
                auto row_it = row.begin();

                TRowType out_row;

                // Extract the values from the row
                //std::cerr << "Expanding row " << in_row_index << " to " << out_row_index << std::endl;
                vectorExpand(out_row, row, oi_pairs);

                if (!out_row.empty())
                    T.setRow(out_row_index, out_row);
            }
        }

//        template <typename ValueT >
//        void assignConstant(LilSparseMatrix<ValueT>             &T,
//                            ValueT                     const    value,
//                            IndexArrayType             const   &row_indicies,
//                            IndexArrayType             const   &col_indicies)
//        {
//            typedef std::vector<std::tuple<IndexType,ValueT> > TRowType;
//
//            for (auto row_it = row_indicies.begin();
//                 row_it != row_indicies.end();
//                 ++row_it)
//            {
//                TRowType out_row;
//                for (auto col_it = col_indicies.begin();
//                     col_it != col_indicies.end();
//                     ++col_it)
//                {
//                    // @todo: add bounds check
//                    out_row.push_back(std::make_tuple(*col_it, value));
//                }
//
//                // @todo: add bounds check
//                if (!out_row.empty())
//                    T.setRow(*row_it, out_row);
//            }
//        };

        template <typename ValueT, typename RowIteratorT, typename ColIteratorT >
        void assignConstant(LilSparseMatrix<ValueT>             &T,
                            ValueT                     const    value,
                            RowIteratorT                        row_begin,
                            RowIteratorT                        row_end,
                            ColIteratorT                        col_begin,
                            ColIteratorT                        col_end)
        {
            typedef std::vector<std::tuple<IndexType,ValueT> > TRowType;

            for (auto row_it = row_begin; row_it != row_end; ++row_it)
            {
                TRowType out_row;
                for (auto col_it = col_begin; col_it != col_end; ++col_it)
                {
                    // @todo: add bounds check
                    out_row.push_back(std::make_tuple(*col_it, value));
                }

                // @todo: add bounds check
                if (!out_row.empty())
                    T.setRow(*row_it, out_row);
            }
        };


        template <typename ValueT>
        void assignConstant(LilSparseMatrix<ValueT>            &T,
                            ValueT                     const    val,
                            IndexArrayType             const   &row_indices,
                            IndexArrayType             const   &col_indices)
        {
            // Sort row indicies and col_indicies

            if (&row_indices == &GrB_ALL && &col_indices == &GrB_ALL)
            {
                assignConstant(T, val,
                               index_iterator(0), index_iterator(T.nrows()),
                               index_iterator(0), index_iterator(T.ncols()));
            }
            else if (&row_indices == &GrB_ALL)
            {
                IndexArrayType sorted_cols(col_indices);
                std::sort(sorted_cols.begin(), sorted_cols.end());
                assignConstant(T, val,
                               index_iterator(0), index_iterator(T.nrows()),
                               sorted_cols.begin(), sorted_cols.end());
            }
            else if (&col_indices == &GrB_ALL)
            {
                IndexArrayType sorted_rows(row_indices);
                std::sort(sorted_rows.begin(), sorted_rows.end());
                assignConstant(T, val,
                               sorted_rows.begin(), sorted_rows.end(),
                               index_iterator(0), index_iterator(T.ncols()));
            }
            else
            {
                IndexArrayType sorted_rows(row_indices);
                IndexArrayType sorted_cols(col_indices);
                std::sort(sorted_rows.begin(), sorted_rows.end());
                std::sort(sorted_cols.begin(), sorted_cols.end());
                assignConstant(T, val, sorted_rows.begin(), sorted_rows.end(),
                               sorted_cols.begin(), sorted_cols.end());
            }
        }

        //=====================================================================
        //=====================================================================

        // 4.3.7.2 assign: Standard matrix variant
        template<typename CMatrixT,
                typename MaskT,
                typename AccumT,
                typename AMatrixT>
        inline void assign(CMatrixT                 &C,
                           MaskT            const   &mask,
                           AccumT                    accum,
                           AMatrixT         const   &A,
                           IndexArrayType   const   &row_indices,
                           IndexArrayType   const   &col_indices,
                           bool                     replace = false)
        {
            typedef typename CMatrixT::ScalarType                   CScalarType;

            // This basically "expands" A into C

            // @todo Add bounds and type checks
            check_matrix_size(C, mask, "assign(matrix) mask dimension check");
            check_index_array_dimension(row_indices, A.nrows(),
                                        "assign row_indices dimension check");
            check_index_array_dimension(col_indices, A.ncols(),
                                        "assign col_indices dimension check");
            check_index_array_content(row_indices, C.nrows(),
                                      "assign row_indices content check");
            check_index_array_content(col_indices, C.ncols(),
                                      "assign col_indices content check");

            //std::cerr << ">>> C in <<< " << std::endl;
            //std::cerr << C << std::endl;

            //std::cerr << ">>> Mask <<< " << std::endl;
            //std::cerr << mask << std::endl;

            // =================================================================
            // Expand to T

            LilSparseMatrix<CScalarType> T(C.nrows(), C.ncols());
            matrixExpand(T, A, row_indices, col_indices);

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

        //======================================================================

        // 4.3.7.5: assign: Constant vector variant
        template<typename WVectorT,
                typename MaskT,
                typename AccumT,
                typename ValueT>
        inline void assign_constant(WVectorT             &w,
                                    MaskT          const &mask,
                                    AccumT                accum,
                                    ValueT                val,
                                    IndexArrayType const &indices,
                                    bool                  replace_flag = false)
        {
            // @todo : Added dimension and error checks

            std::vector<std::tuple<IndexType, ValueT> > T;

            // Set all in T
            if (&indices == &GrB_ALL)
            {
                for (auto it = index_iterator(0); it != index_iterator(w.size()); ++it)
                    T.push_back(std::make_tuple(*it, val));
            }
            else
            {
                for (auto it = indices.begin(); it != indices.end(); ++it)
                    T.push_back(std::make_tuple(*it, val));
            }

            // =================================================================
            // Accumulate into Z

            typedef typename WVectorT::ScalarType WScalarType;
            std::vector<std::tuple<IndexType, WScalarType> > z;
            ewise_or_opt_accum_1D(z, w, T, accum);

            // =================================================================
            // Copy Z into the final output, w, considering mask and replace
            write_with_opt_mask_1D(w, z, mask, replace_flag);
        };

        //======================================================================
        // 4.3.7.6: assign: Constant Matrix Variant
        template<typename CMatrixT,
                typename MaskT,
                typename AccumT,
                typename ValueT>
        inline void assign_constant(CMatrixT             &C,
                                    MaskT          const &Mask,
                                    AccumT                accum,
                                    ValueT                val,
                                    IndexArrayType const &row_indices,
                                    IndexArrayType const &col_indices,
                                    bool                  replace_flag = false)
        {
            typedef typename CMatrixT::ScalarType CScalarType;

            // @todo Add bounds and type checks

            //std::cerr << ">>> C in <<< " << std::endl;
            //std::cerr << C << std::endl;

            //std::cerr << ">>> Mask <<< " << std::endl;
            //std::cerr << mask << std::endl;

            // =================================================================
            // Assign spots in T
            LilSparseMatrix<ValueT> T(C.nrows(), C.ncols());
            assignConstant(T, val, row_indices, col_indices);

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
            write_with_opt_mask(C, Z, Mask, replace_flag);

            //std::cerr << ">>> C <<< " << std::endl;
            //std::cerr << C << std::endl;
        };

    }
}

#endif //GB_SEQUENTIAL_SPARSE_ASSIGN_HPP
