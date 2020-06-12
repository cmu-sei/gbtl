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
#include <iostream>
#include <type_traits>
#include <graphblas/types.hpp>
#include <graphblas/exceptions.hpp>
#include <graphblas/algebra.hpp>

#include "sparse_helpers.hpp"
#include "LilSparseMatrix.hpp"

//******************************************************************************

namespace grb
{
    namespace backend
    {
        //********************************************************************
        struct IndexCompare
        {
            inline bool operator()(std::tuple<IndexType, IndexType> const &i1,
                                   std::tuple<IndexType, IndexType> const &i2)
            {
                return std::get<0>(i1) < std::get<0>(i2);
            }
        };

        //********************************************************************
        // Builds a simple mapping
        template <typename SequenceT>
        void compute_outin_mapping(
            SequenceT                                 const &Indices,
            std::vector<std::tuple<IndexType, IndexType>>   &inputOrder)
        {
            inputOrder.clear();

            // Walk the Indices generating pairs of the mapping
            auto index_it = Indices.begin();
            IndexType idx = 0;
            while (index_it != Indices.end())
            {
                inputOrder.emplace_back(*index_it, idx);
                ++index_it;
                ++idx;
            }

            // Sort them because we want to deal with them in output order.
            std::sort(inputOrder.begin(), inputOrder.end(), IndexCompare());
        }

        //********************************************************************
        template <typename TScalarT,
                  typename AScalarT>
        void vectorExpand(
            std::vector<std::tuple<IndexType, TScalarT>>        &vec_dest,
            std::vector<std::tuple<IndexType, AScalarT>>  const &vec_src,
            std::vector<std::tuple<IndexType, IndexType>> const &Indices)
        {
            vec_dest.clear();
            // The Indices are pairs of ( output_index, input_index)
            // We do it this way, so we get the output in the right
            // order to begin with

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
                while (src_it != vec_src.end())
                {
                    std::tie(src_idx, src_val) = *src_it;

                    if (src_idx == std::get<1>(*index_it))
                    {
                        vec_dest.emplace_back(
                            std::get<0>(*index_it), static_cast<TScalarT>(src_val));
                        break;
                    }
                    else if (src_idx > std::get<1>(*index_it))
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
                    (src_it == vec_src.end() || std::get<1>(*index_it) < src_idx))
                {
                    src_it = vec_src.begin();
                }
            }
        }

        //********************************************************************
        // non-transposed case.
        template<typename TScalarT,
                 typename AScalarT,
                 typename RowSequenceT,
                 typename ColSequenceT>
        void matrixExpand(LilSparseMatrix<TScalarT>          &T,
                          LilSparseMatrix<AScalarT>  const   &A,
                          RowSequenceT               const   &row_Indices,
                          ColSequenceT               const   &col_Indices)
        {
            T.clear();

            // Build the mapping pairs once up front
            std::vector<std::tuple<IndexType, IndexType>> oi_pairs;
            compute_outin_mapping(col_Indices, oi_pairs);

            // Walk the input rows (in order specified by input)
            for (IndexType in_row_index = 0;
                 in_row_index < row_Indices.size();
                 ++in_row_index)
            {
                if (!A[in_row_index].empty())
                {
                    IndexType out_row_index = row_Indices[in_row_index];
                    std::vector<std::tuple<IndexType,TScalarT> > out_row;

                    // Extract the values from the row
                    vectorExpand(out_row, A[in_row_index], oi_pairs);

                    if (!out_row.empty())
                        T.setRow(out_row_index, out_row);
                }
            }
        }

        //********************************************************************
        // transposed case
        template<typename TScalarT,
                 typename AMatrixT,
                 typename RowSequenceT,
                 typename ColSequenceT>
        void matrixExpand(LilSparseMatrix<TScalarT>        &T,
                          TransposeView<AMatrixT>    const &AT,
                          RowSequenceT               const &row_Indices, // of AT
                          ColSequenceT               const &col_Indices) // of AT
        {
            auto const &A(AT.m_mat);
            T.clear();

            // Build the mapping pairs once up front (rows of AT -> cols of T)
            std::vector<std::tuple<IndexType, IndexType>> oi_col_pairs;
            std::vector<std::tuple<IndexType, IndexType>> oi_row_pairs;
            compute_outin_mapping(col_Indices, oi_col_pairs);
            compute_outin_mapping(row_Indices, oi_row_pairs);

            std::vector<std::tuple<IndexType,TScalarT> > out_col;

            // Walk the input columns (rows of A) in ascending output order
            for (auto&& [out_col_index, in_col_index] : oi_col_pairs)
            {
                // Extract the values from the row and set col (push_back on rows)
                vectorExpand(out_col, A[in_col_index], oi_row_pairs);

                for (auto&& [out_row_index, val] : out_col)
                {
                    T[out_row_index].emplace_back(out_col_index, val);
                }
            }
            T.recomputeNvals();
        }

        //********************************************************************
        template <typename ValueT, typename RowIteratorT, typename ColIteratorT >
        void assignConstant(LilSparseMatrix<ValueT>             &T,
                            ValueT                     const    value,
                            RowIteratorT                        row_begin,
                            RowIteratorT                        row_end,
                            ColIteratorT                        col_begin,
                            ColIteratorT                        col_end)
        {
            std::vector<std::tuple<IndexType,ValueT> > out_row;

            for (auto row_it = row_begin; row_it != row_end; ++row_it)
            {
                out_row.clear();
                for (auto col_it = col_begin; col_it != col_end; ++col_it)
                {
                    // @todo: add bounds check
                    out_row.emplace_back(*col_it, value);
                }

                // @todo: add bounds check
                if (!out_row.empty())
                    T.setRow(*row_it, out_row);
            }
        }

        //********************************************************************
        template <typename ValueT,
                typename RowIndicesT,
                typename ColIndicesT>
        void assignConstant(LilSparseMatrix<ValueT>           &T,
                            ValueT                    const    val,
                            RowIndicesT               const   &row_indices,
                            ColIndicesT               const   &col_indices)
        {
            // @TODO: Deal with sorting
            //
            // Sort row Indices and col_Indices
            // IndexSequence sorted_rows(row_indices);
            // IndexSequence sorted_cols(col_indices);
            // std::sort(sorted_rows.begin(), sorted_rows.end());
            // std::sort(sorted_cols.begin(), sorted_cols.end());
            // assignConstant(T, val,
            //                sorted_rows.begin(), sorted_rows.end(),
            //                sorted_cols.begin(), sorted_cols.end());
            // assignConstant(T, val,
            //                row_indices.begin(), row_indices.end(),
            //                col_indices.begin(), col_indices.end());

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
                           AccumT       const &accum,
                           UVectorT     const &u,
                           SequenceT    const &indices,
                           OutputControlEnum   outp)
        {
            GRB_LOG_VERBOSE("reference backend - 4.3.7.1");

            check_index_array_content(indices, w.size(),
                                      "assign(std vec): indices content check");

            std::vector<std::tuple<IndexType, IndexType>> oi_pairs;
            compute_outin_mapping(setupIndices(indices, u.size()), oi_pairs);

            // =================================================================
            // Expand to t
            using UScalarType = typename UVectorT::ScalarType;
            std::vector<std::tuple<IndexType, UScalarType> > t;
            auto u_contents(u.getContents());
            vectorExpand(t, u_contents, oi_pairs);

            GRB_LOG_VERBOSE("t: " << t);

            // =================================================================
            // Accumulate into z
            using ZScalarType = typename std::conditional_t<
                std::is_same_v<AccumT, NoAccumulate>,
                typename WVectorT::ScalarType, /// @todo UScalarType?
                decltype(accum(std::declval<typename WVectorT::ScalarType>(),
                               std::declval<UScalarType>()))>;

            std::vector<std::tuple<IndexType, ZScalarType> > z;
            ewise_or_stencil_opt_accum_1D(z, w, t,
                                          setupIndices(indices, u.size()),
                                          accum);

            GRB_LOG_VERBOSE("z: " << z);

            // =================================================================
            // Copy z into the final output considering mask and replace/merge
            write_with_opt_mask_1D(w, z, mask, outp);
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
                           AccumT           const &accum,
                           AMatrixT         const &A,
                           RowSequenceT     const &row_indices,
                           ColSequenceT     const &col_indices,
                           OutputControlEnum       outp)
        {
            using AScalarType = typename AMatrixT::ScalarType;

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
            using ZScalarType = typename std::conditional_t<
                std::is_same_v<AccumT, NoAccumulate>,
                typename CMatrixT::ScalarType, /// @todo AScalarType?
                decltype(accum(std::declval<typename CMatrixT::ScalarType>(),
                               std::declval<AScalarType>()))>;

            LilSparseMatrix<ZScalarType> Z(C.nrows(), C.ncols());
            ewise_or_stencil_opt_accum(Z, C, T,
                                       setupIndices(row_indices, A.nrows()),
                                       setupIndices(col_indices, A.ncols()),
                                       accum);

            GRB_LOG_VERBOSE("Z:  " << Z);

            // =================================================================
            // Copy Z into the final output considering mask and replace/merge
            write_with_opt_mask(C, Z, mask, outp);
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
                           AccumT           const &accum,
                           UVectorT         const &u,
                           SequenceT        const &row_indices,
                           IndexType               col_index,
                           OutputControlEnum       outp)
        {
            // IMPLEMENTATION NOTE: This function does not directly follow our
            // standard implementation method.  We leverage a different assign
            // variant and wrap it's contents with this.

            // execution error checks
            check_index_array_content(row_indices, C.nrows(),
                                      "assign(col): indices content check");

            // EXTRACT the column of C matrix
            using CScalarType = typename CMatrixT::ScalarType;
            auto C_col(C.getCol(col_index));
            Vector<CScalarType> c_vec(C.nrows());
            for (auto it : C_col)
            {
                c_vec.setElement(std::get<0>(it), std::get<1>(it));
            }

            // ----------- standard vector variant 4.3.7.1 -----------
            assign(c_vec, mask, accum, u, row_indices, outp);
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
                           AccumT           const &accum,
                           UVectorT         const &u,
                           IndexType               row_index,
                           SequenceT        const &col_indices,
                           OutputControlEnum       outp)
        {
            // IMPLEMENTATION NOTE: This function does not directly follow our
            // standard implementation method.  We leverage a different assign
            // variant and wrap it's contents with this.  Because of this the
            // performance is usually much less than ideal.

            // execution error checks
            check_index_array_content(col_indices, C.ncols(),
                                      "assign(row): indices content check");

            // EXTRACT the row of C matrix
            /// @todo creating a Vector and then extracting later can be COSTLY
            using CScalarType = typename CMatrixT::ScalarType;
            auto C_row(C[row_index]);
            Vector<CScalarType> c_vec(C.ncols());
            for (auto&& [col_idx, val] : C[row_index])
            {
                c_vec.setElement(col_idx, val);
            }

            // ----------- standard vector variant 4.3.7.1 -----------
            assign(c_vec, mask, accum, u, col_indices, outp);
            // ----------- standard vector variant 4.3.7.1 -----------

            // REPLACE the row of C matrix
            std::vector<IndexType>   ic(c_vec.nvals());
            std::vector<CScalarType> vc(c_vec.nvals());
            c_vec.extractTuples(ic.begin(), vc.begin());

            std::vector<std::tuple<IndexType,CScalarType> > row_data;

            for (IndexType idx = 0; idx < ic.size(); ++idx)
            {
                row_data.emplace_back(ic[idx],vc[idx]);
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
                                    AccumT         const &accum,
                                    ValueT                val,
                                    SequenceT      const &indices,
                                    OutputControlEnum     outp)
        {
            // execution error checks
            check_index_array_content(indices, w.size(),
                                      "assign(const vec): indices content check");

            std::vector<std::tuple<IndexType, ValueT> > t;

            // Set all in T
            auto seq = setupIndices(indices, w.size());
            for (auto it = seq.begin(); it != seq.end(); ++it)
                t.emplace_back(*it, val);

            GRB_LOG_VERBOSE("t: " << t);

            // =================================================================
            // Accumulate into Z
            using ZScalarType = typename std::conditional_t<
                std::is_same_v<AccumT, NoAccumulate>,
                typename WVectorT::ScalarType,  /// @todo ValueT?
                decltype(accum(std::declval<typename WVectorT::ScalarType>(),
                               std::declval<ValueT>()))>;

            std::vector<std::tuple<IndexType, ZScalarType> > z;
            ewise_or_stencil_opt_accum_1D(z, w, t,
                                          setupIndices(indices, w.size()),
                                          accum);

            GRB_LOG_VERBOSE("z: " << z);

            // =================================================================
            // Copy Z into the final output, w, considering mask and replace/merge
            write_with_opt_mask_1D(w, z, mask, outp);
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
                                    AccumT         const &accum,
                                    ValueT                val,
                                    RowIndicesT    const &row_indices,
                                    ColIndicesT    const &col_indices,
                                    OutputControlEnum     outp)
        {
            using CScalarType = typename CMatrixT::ScalarType;

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
            using ZScalarType = typename std::conditional_t<
                std::is_same_v<AccumT, NoAccumulate>,
                typename CMatrixT::ScalarType,  /// @todo ValueT?
                decltype(accum(std::declval<CScalarType>(),
                               std::declval<ValueT>()))>;

            LilSparseMatrix<ZScalarType> Z(C.nrows(), C.ncols());
            ewise_or_stencil_opt_accum(Z, C, T,
                                       setupIndices(row_indices, C.nrows()),
                                       setupIndices(col_indices, C.ncols()),
                                       accum);

            GRB_LOG_VERBOSE("Z: " << Z);

            // =================================================================
            // Copy Z into the final output considering mask and replace/merge
            write_with_opt_mask(C, Z, Mask, outp);
        }
    }
}
