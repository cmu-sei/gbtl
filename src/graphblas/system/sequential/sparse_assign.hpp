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
#include <type_traits>
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
        template <typename SequenceT>
        void compute_outin_mapping(SequenceT const & Indices,
                      std::vector<std::pair<IndexType, IndexType>> &inputOrder)
        {
            // Walk the Indices generating pairs of the mapping
            auto index_it = Indices.begin();
            IndexType idx = 0;
            while (index_it != Indices.end())
            {
                inputOrder.push_back(std::make_pair(*index_it, idx));
                ++index_it;
                ++idx;
            }

            // Sort them because we want to deal with them in output order.
            std::sort(inputOrder.begin(), inputOrder.end(), IndexCompare());
        }

        template <typename TScalarT,
                  typename AScalarT>
        void vectorExpand(std::vector<std::tuple<IndexType, TScalarT>>  &vec_dest,
                          std::vector<std::tuple<IndexType, AScalarT>> const &vec_src,
                          std::vector<std::pair<IndexType, IndexType>> const &Indices)
        {
            // The Indices are pairs of ( output_index, input_index)
            // We do it this way, so we get the output in the right
            // order to begin with

            //std::cerr << "Expanding row " << std::endl;

            // Walk the output/input pairs building the output in correct order.
            auto index_it = Indices.begin();

            // We start at the beginning of the source and work our way through
            // it.  We reset to beginning when the input is before us.
            // This way we reduce thrash a little bit.
            auto src_it = vec_src.begin();

            while (index_it != Indices.end())
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
                if (index_it != Indices.end() &&
                        (src_it == vec_src.end() || index_it->second < src_idx))
                {
                    //std::cerr << "Resetting src_it " << std::endl;
                    src_it = vec_src.begin();
                }
            }
        }

        template<typename TScalarT,
                 typename AScalarT,
                 typename RowSequenceT,
                 typename ColSequenceT>
        void matrixExpand(LilSparseMatrix<TScalarT>          &T,
                          LilSparseMatrix<AScalarT>  const   &A,
                          RowSequenceT               const   &row_Indices,
                          ColSequenceT               const   &col_Indices)
        {
            // NOTE!! - Backend code. We expect that all dimension
            // checks done elsewhere.

            typedef std::vector<std::tuple<IndexType,AScalarT> > ARowType;
            typedef std::vector<std::tuple<IndexType,TScalarT> > TRowType;

            T.clear();

            // std::cerr << "row_indices:";
            // for (auto it : row_Indices) std::cerr << " " << it;
            // std::cerr << std::endl;

            // std::cerr << "col_indices:";
            // for (auto it : col_Indices) std::cerr << " " << it;
            // std::cerr << std::endl;

            // Build the mapping pairs once up front
            std::vector<std::pair<IndexType, IndexType>> oi_pairs;
            compute_outin_mapping(col_Indices, oi_pairs);

            // Walk the rows
            for (IndexType in_row_index = 0;
                 in_row_index < row_Indices.size();
                 ++in_row_index)
            {
                IndexType out_row_index = row_Indices[in_row_index];
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
        }


        template <typename ValueT,
                typename RowIndicesT,
                typename ColIndicesT>
        void assignConstant(LilSparseMatrix<ValueT>            &T,
                            ValueT                     const    val,
                            RowIndicesT               const   &row_indices,
                            ColIndicesT               const   &col_indices)
        {
            // Sort row Indices and col_Indices

            // @TODO: Deal with sorting

//            IndexSequence sorted_rows(row_indices);
//            IndexSequence sorted_cols(col_indices);
//            std::sort(sorted_rows.begin(), sorted_rows.end());
//            std::sort(sorted_cols.begin(), sorted_cols.end());
//            assignConstant(T, val,
//                           sorted_rows.begin(), sorted_rows.end(),
//                           sorted_cols.begin(), sorted_cols.end());
//            assignConstant(T, val,
//                           row_indices.begin(), row_indices.end(),
//                           col_indices.begin(), col_indices.end());

            assignConstant(T, val,
                           row_indices.begin(), row_indices.end(),
                           col_indices.begin(), col_indices.end());
        }

        //=====================================================================
        //=====================================================================

        // 4.3.7.1: assign - standard vector variant
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename UVectorT,
                 typename SequenceT>
        inline void assign(WVectorT           &w,
                           MaskT        const &mask,
                           AccumT              accum,
                           UVectorT     const &u,
                           SequenceT    const &indices,
                           bool                replace_flag)
        {
            GRB_LOG_VERBOSE("reference backend - 4.3.7.1");

            check_index_array_content(indices, w.size(),
                                      "assign(std vec): indices content check");

            std::vector<std::pair<IndexType, IndexType>> oi_pairs;
            compute_outin_mapping(setupIndices(indices, u.size()), oi_pairs);

            // =================================================================
            // Expand to t
            typedef typename UVectorT::ScalarType UScalarType;
            std::vector<std::tuple<IndexType, UScalarType> > t;
            auto u_contents(u.getContents());
            vectorExpand(t, u_contents, oi_pairs);

            GRB_LOG_VERBOSE("t: " << t);

            // =================================================================
            // Accumulate into z

            typedef typename std::conditional<std::is_same<AccumT, NoAccumulate>::value,
                    typename WVectorT::ScalarType,
                    typename AccumT::result_type>::type ZScalarType;

            std::vector<std::tuple<IndexType, ZScalarType> > z;
            ewise_or_stencil_opt_accum_1D(z, w, t,
                                          setupIndices(indices, u.size()),
                                          accum);

            GRB_LOG_VERBOSE("z: " << z);

            // =================================================================
            // Copy z into the final output considering mask and replace
            write_with_opt_mask_1D(w, z, mask, replace_flag);
        }

        //=====================================================================
        //=====================================================================

        // 4.3.7.2 assign: Standard matrix variant
        template<typename CMatrixT,
                 typename MaskT,
                 typename AccumT,
                 typename AMatrixT,
                 typename RowSequenceT,
                 typename ColSequenceT>
        inline void assign(CMatrixT               &C,
                           MaskT            const &mask,
                           AccumT                  accum,
                           AMatrixT         const &A,
                           RowSequenceT     const &row_indices,
                           ColSequenceT     const &col_indices,
                           bool                    replace = false)
        {
            typedef typename CMatrixT::ScalarType  CScalarType;
            typedef typename AMatrixT::ScalarType  AScalarType;

            // execution error checks
            check_index_array_content(row_indices, C.nrows(),
                                      "assign(std mat): row_indices content check");
            check_index_array_content(col_indices, C.ncols(),
                                      "assign(std mat): col_indices content check");

            // =================================================================
            // Expand to T
            LilSparseMatrix<AScalarType> T(C.nrows(), C.ncols());
            matrixExpand(T, A,
                         setupIndices(row_indices, A.nrows()),
                         setupIndices(col_indices, A.ncols()));

            GRB_LOG_VERBOSE("T: " << T);

            // =================================================================
            // Accumulate into Z
            typedef typename std::conditional<std::is_same<AccumT, NoAccumulate>::value,
                    typename CMatrixT::ScalarType,
                    typename AccumT::result_type>::type ZScalarType;

            LilSparseMatrix<ZScalarType> Z(C.nrows(), C.ncols());
            ewise_or_stencil_opt_accum(Z, C, T,
                                       setupIndices(row_indices, A.nrows()),
                                       setupIndices(col_indices, A.ncols()),
                                       accum);

            GRB_LOG_VERBOSE("Z:  " << Z);

            // =================================================================
            // Copy Z into the final output considering mask and replace
            write_with_opt_mask(C, Z, mask, replace);
        }

        //=====================================================================
        //=====================================================================

        // 4.3.7.3 assign: Column variant
        template<typename CMatrixT,
                 typename MaskT,
                 typename AccumT,
                 typename UVectorT,
                 typename SequenceT>
        inline void assign(CMatrixT               &C,
                           MaskT            const &mask,
                           AccumT                  accum,
                           UVectorT         const &u,
                           SequenceT        const &row_indices,
                           IndexType               col_index,
                           bool                    replace = false)
        {
            // IMPLEMENTATION NOTE: This function does not directly follow our
            // standard implementation method.  We leverage a different assign
            // variant and wrap it's contents with this.

            // execution error checks
            check_index_array_content(row_indices, C.nrows(),
                                      "assign(col): indices content check");

            // EXTRACT the column of C matrix
            typedef typename CMatrixT::ScalarType CScalarType;
            auto C_col(C.getCol(col_index));
            Vector<CScalarType> c_vec(C.nrows());
            for (auto it : C_col)
            {
                c_vec.setElement(std::get<0>(it), std::get<1>(it));
            }

            // ----------- standard vector variant 4.3.7.1 -----------
            assign(c_vec, mask, accum, u, row_indices, replace);
            // ----------- standard vector variant 4.3.7.1 -----------

            // REPLACE the column of C matrix
            std::vector<IndexType>   ic(c_vec.nvals());
            std::vector<CScalarType> vc(c_vec.nvals());
            c_vec.extractTuples(ic.begin(), vc.begin());

            std::vector<std::tuple<IndexType,CScalarType> > col_data;

            for (IndexType idx = 0; idx < ic.size(); ++idx)
            {
                col_data.push_back(std::make_tuple(ic[idx],vc[idx]));
            }

            C.setCol(col_index, col_data);
        }

        //=====================================================================
        //=====================================================================

        // 4.3.7.4 assign: Row variant
        template<typename CMatrixT,
                 typename MaskT,
                 typename AccumT,
                 typename UVectorT,
                 typename SequenceT>
        inline void assign(CMatrixT               &C,
                           MaskT            const &mask,
                           AccumT                  accum,
                           UVectorT         const &u,
                           IndexType               row_index,
                           SequenceT        const &col_indices,
                           bool                    replace = false)
        {
            // IMPLEMENTATION NOTE: This function does not directly follow our
            // standard implementation method.  We leverage a different assign
            // variant and wrap it's contents with this.

            // execution error checks
            check_index_array_content(col_indices, C.ncols(),
                                      "assign(row): indices content check");

            // EXTRACT the row of C matrix
            typedef typename CMatrixT::ScalarType CScalarType;
            auto C_row(C.getRow(row_index));
            Vector<CScalarType> c_vec(C.ncols());
            for (auto it : C_row)
            {
                c_vec.setElement(std::get<0>(it), std::get<1>(it));
            }

            // ----------- standard vector variant 4.3.7.1 -----------
            assign(c_vec, mask, accum, u, col_indices, replace);
            // ----------- standard vector variant 4.3.7.1 -----------

            // REPLACE the row of C matrix
            std::vector<IndexType>   ic(c_vec.nvals());
            std::vector<CScalarType> vc(c_vec.nvals());
            c_vec.extractTuples(ic.begin(), vc.begin());

            std::vector<std::tuple<IndexType,CScalarType> > row_data;

            for (IndexType idx = 0; idx < ic.size(); ++idx)
            {
                row_data.push_back(std::make_tuple(ic[idx],vc[idx]));
            }

            C.setRow(row_index, row_data);
        }

        //======================================================================
        //======================================================================

        // 4.3.7.5: assign: Constant vector variant
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename ValueT,
                 typename SequenceT>
        inline void assign_constant(WVectorT             &w,
                                    MaskT          const &mask,
                                    AccumT                accum,
                                    ValueT                val,
                                    SequenceT      const &indices,
                                    bool                  replace_flag = false)
        {
            // execution error checks
            check_index_array_content(indices, w.size(),
                                      "assign(const vec): indices content check");

            std::vector<std::tuple<IndexType, ValueT> > t;

            // Set all in T
            auto seq = setupIndices(indices, w.size());
            for (auto it = seq.begin(); it != seq.end(); ++it)
                t.push_back(std::make_tuple(*it, val));

            // =================================================================
            // Accumulate into Z

            typedef typename std::conditional<std::is_same<AccumT, NoAccumulate>::value,
                    typename WVectorT::ScalarType,
                    typename AccumT::result_type>::type ZScalarType;

            std::vector<std::tuple<IndexType, ZScalarType> > z;
            ewise_or_stencil_opt_accum_1D(z, w, t,
                                          setupIndices(indices, w.size()),
                                          accum);

            // =================================================================
            // Copy Z into the final output, w, considering mask and replace
            write_with_opt_mask_1D(w, z, mask, replace_flag);
        }

        //======================================================================
        //======================================================================

        // 4.3.7.6: assign: Constant Matrix Variant
        template<typename CMatrixT,
                 typename MaskT,
                 typename AccumT,
                 typename ValueT,
                 typename RowIndicesT,
                 typename ColIndicesT>
        inline void assign_constant(CMatrixT             &C,
                                    MaskT          const &Mask,
                                    AccumT                accum,
                                    ValueT                val,
                                    RowIndicesT  const &row_indices,
                                    ColIndicesT  const &col_indices,
                                    bool                  replace_flag = false)
        {
            typedef typename CMatrixT::ScalarType CScalarType;

            // execution error checks
            check_index_array_content(row_indices, C.nrows(),
                                      "assign(std mat): row_indices content check");
            check_index_array_content(col_indices, C.ncols(),
                                      "assign(std mat): col_indices content check");

            // =================================================================
            // Assign spots in T
            LilSparseMatrix<ValueT> T(C.nrows(), C.ncols());
            assignConstant(T, val,
                           setupIndices(row_indices, C.nrows()),
                           setupIndices(col_indices, C.ncols()));

            GRB_LOG_VERBOSE("T: " << T);

            // =================================================================
            // Accumulate into Z
            typedef typename std::conditional<std::is_same<AccumT, NoAccumulate>::value,
                    typename CMatrixT::ScalarType,
                    typename AccumT::result_type>::type ZScalarType;

            LilSparseMatrix<CScalarType> Z(C.nrows(), C.ncols());
            ewise_or_stencil_opt_accum(Z, C, T,
                                       setupIndices(row_indices, C.nrows()),
                                       setupIndices(col_indices, C.ncols()),
                                       accum);

            GRB_LOG_VERBOSE("Z: " << Z);

            // =================================================================
            // Copy Z into the final output considering mask and replace
            write_with_opt_mask(C, Z, Mask, replace_flag);
        }
    }
}

#endif //GB_SEQUENTIAL_SPARSE_ASSIGN_HPP
