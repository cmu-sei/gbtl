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
        // AllIndices SUPPORT
        // This is where we turn AllIndices into the correct range
        template <typename SequenceT>
        bool searchIndices(SequenceT seq, IndexType tgt)
        {
            // This branch assumes sequence is unordered.
            for (auto it : seq)
            {
                if (it == tgt) return true;
            }
            return false;
        }

        //********************************************************************
        bool searchIndices(AllIndices seq, IndexType tgt)
        {
            return true;
        }

        //**********************************************************************
        /// Apply element-wise operation to union on sparse vectors.
        /// Indices in the stencil indicate where elements of vec2 should be
        /// used (whether there is a stored value or not); otherwise the value
        /// in vec1 should be taken.  Note: that it is assumed that if a value
        /// is stored in vec2 then the corresponding location is contained in
        /// stencil indices.
        ///
        /// Truth table (for each element, i, of the answer, where '-' means
        /// no stored value):
        ///
        ///  vec1_i   vec2   s_i   ans_i
        ///    -        -     -      -
        ///    -        -     x      -
        ///    -        x --> x    vec2_i
        ///    x        -     -    vec1_i
        ///    x        -     x      -    (take vec1_i which is no stored value)
        ///    x        x --> x    vec1_i
        ///
        /// \tparam D1
        /// \tparam D2
        /// \tparam D3
        /// \tparam SequenceT  Could be a out of order subset of indices
        ///
        /// \param ans   A row of the answer (Z or z), starts empty
        /// \param vec1  A row of the output container (C or w), indices increasing order
        /// \param vec2  A row of the T (or t) container, indices in increasing order
        /// \param stencil_indices  Assumed to not be in order
        ///
        template <typename D1, typename D2, typename D3, typename SequenceT>
        void ewise_or_stencil(
            std::vector<std::tuple<grb::IndexType,D3> >       &ans,
            std::vector<std::tuple<grb::IndexType,D1> > const &vec1,
            std::vector<std::tuple<grb::IndexType,D2> > const &vec2,
            SequenceT                                          stencil_indices)
        {
            ans.clear();

            //auto stencil_it = stencil_indices.begin();
            //if (v1_it)
            //while ((stencil_it != stencil_indices.end()) &&
            //       (*stencil_it < std::get<0>(*v1_it)))
            //{
            //    ++stencil_it;
            //}

            D1 v1_val;
            D2 v2_val;
            grb::IndexType v1_idx, v2_idx;

            // loop through both ordered sets to compute ewise_or
            auto v1_it = vec1.begin();
            auto v2_it = vec2.begin();
            while ((v1_it != vec1.end()) || (v2_it != vec2.end()))
            {
                if ((v1_it != vec1.end()) && (v2_it != vec2.end()))
                {
                    std::tie(v1_idx, v1_val) = *v1_it;
                    std::tie(v2_idx, v2_val) = *v2_it;

                    // If v1 and v2 both have stored values, it is assumed index
                    // is in stencil_indices so v2 should be stored
                    if (v2_idx == v1_idx)
                    {
                        ans.emplace_back(v2_idx, static_cast<D3>(v2_val));

                        ++v2_it;
                        ++v1_it;
                    }
                    // In this case v1 has a value and not v2.  We need to search
                    // stencil indices to see if index is present
                    else if (v1_idx < v2_idx) // advance v1 and annihilate
                    {
                        if (!searchIndices(stencil_indices, v1_idx))
                        {
                            ans.emplace_back(v1_idx, static_cast<D3>(v1_val));
                        }
                        ++v1_it;
                    }
                    else
                    {
                        //std::cerr << "Copying v2, Advancing v2_it" << std::endl;
                        ans.emplace_back(v2_idx, static_cast<D3>(v2_val));
                        ++v2_it;
                    }
                }
                else if (v1_it != vec1.end())  // vec2 exhausted
                {
                    std::tie(v1_idx, v1_val) = *v1_it;

                    if (!searchIndices(stencil_indices, v1_idx))
                    {
                        ans.emplace_back(v1_idx, static_cast<D3>(v1_val));
                    }
                    ++v1_it;
                }
                else // v2_it != vec2.end()) and vec1 exhausted
                {
                    std::tie(v2_idx, v2_val) = *v2_it;
                    ans.emplace_back(v2_idx, static_cast<D3>(v2_val));
                    ++v2_it;
                }
            }
        }

        //**********************************************************************
        /// Apply element-wise operation to union on sparse vectors.
        /// Indices in the stencil indicate where elements of vec2 should be
        /// used (whether there is a stored value or not); otherwise the value
        /// in vec1 should be taken.  Note: that it is assumed that if a value
        /// is stored in vec2 then the corresponding location is contained in
        /// stencil indices.
        ///
        /// Truth table (for each element, i, of the answer, where '-' means
        /// no stored value):
        ///
        ///  vec1_i   vec2   s_i   ans_i
        ///    -        -     -      -
        ///    -        -     x      -
        ///    -        x --> x    vec2_i
        ///    x        -     -    vec1_i
        ///    x        -     x      -    (take vec1_i which is no stored value)
        ///    x        x --> x    vec1_i
        ///
        /// \tparam D1
        /// \tparam D2
        /// \tparam D3
        /// \tparam SequenceT  Could be a out of order subset of indices
        ///
        /// \param ans   A row of the answer (Z or z), starts empty
        /// \param vec1  A row of the output container (C or w), indices increasing order
        /// \param vec2  A row of the T (or t) container, indices in increasing order
        /// \param stencil_indices  Assumed to not be in order
        ///
        template <typename D1, typename D2, typename D3, typename SequenceT>
        void ewise_or_stencil_dense_sparse(
            std::vector<std::tuple<grb::IndexType,D3> >       &ans,
            BitmapSparseVector<D1>                      const &vec1,
            std::vector<std::tuple<grb::IndexType,D2> > const &vec2,
            SequenceT                                          stencil_indices)
        {
            ans.clear();

            D2 v2_val;
            grb::IndexType v1_idx, v2_idx;

            // loop through both ordered sets to compute ewise_or
            //auto v1_it = vec1.begin();
            auto v2_it = vec2.begin();
            if (v2_it != vec2.end())
            {
                std::tie(v2_idx, v2_val) = *v2_it;
            }

            //while ((v1_it != vec1.end()) || (v2_it != vec2.end()))
            for (v1_idx = 0; v1_idx < vec1.size(); ++v1_idx)
            {
                if (v2_it != vec2.end())
                {
                    // If v1 and v2 both have stored values, it is assumed index
                    // is in stencil_indices so v2 should be stored
                    if (vec1.hasElementNoCheck(v1_idx) && (v2_idx == v1_idx))
                    {
                        ans.emplace_back(v2_idx, static_cast<D3>(v2_val));

                        ++v2_it;  std::tie(v2_idx, v2_val) = *v2_it;
                    }
                    // In this case v1 has a value and not v2.  We need to search
                    // stencil indices to see if index is present
                    else if (vec1.hasElementNoCheck(v1_idx) &&
                             (v1_idx < v2_idx))
                    {
                        // advance v1 and annihilate
                        if (!searchIndices(stencil_indices, v1_idx))
                        {
                            ans.emplace_back(
                                v1_idx,
                                static_cast<D3>(
                                    vec1.extractElementNoCheck(v1_idx)));
                        }
                    }
                    else if (v1_idx == v2_idx)
                    {
                        //std::cerr << "Copying v2, Advancing v2_it" << std::endl;
                        ans.emplace_back(v2_idx, static_cast<D3>(v2_val));
                        ++v2_it;  std::tie(v2_idx, v2_val) = *v2_it;
                    }
                }
                else if (vec1.hasElementNoCheck(v1_idx))  // vec2 exhausted
                {
                    if (!searchIndices(stencil_indices, v1_idx))
                    {
                        ans.emplace_back(
                            v1_idx,
                            static_cast<D3>(
                                vec1.extractElementNoCheck(v1_idx)));
                    }
                }
            }
        }

        //**********************************************************************
        // for sparse_assign
        template <typename ZScalarT,
                  typename WVectorT,
                  typename TScalarT,
                  typename SequenceT,
                  typename BinaryOpT >
        void ewise_or_stencil_opt_accum_1D(
            std::vector<std::tuple<grb::IndexType,ZScalarT>>       &z,
            WVectorT const                                         &w,
            std::vector<std::tuple<grb::IndexType,TScalarT>> const &t,
            SequenceT const                                        &indices,
            BinaryOpT                                               accum)
        {
            // If there is an accumulate operations, do nothing with the stencil
            ewise_or_dense_sparse_v2(z, w, t, accum);
        }

        //**********************************************************************
        // for sparse_assign
        template <typename ZScalarT,
                  typename WVectorT,
                  typename TScalarT,
                  typename SequenceT>
        void ewise_or_stencil_opt_accum_1D(
            std::vector<std::tuple<grb::IndexType,ZScalarT>>       &z,
            WVectorT const                                         &w,
            std::vector<std::tuple<grb::IndexType,TScalarT>> const &t,
            SequenceT const                                        &indices,
            grb::NoAccumulate)
        {
            // If there is no accumulate we need to annihilate stored values
            // in w that fall in the stencil
            ewise_or_stencil_dense_sparse(z, w, t, indices);
        }


        //**********************************************************************
        template < typename ZMatrixT,
                   typename CMatrixT,
                   typename TMatrixT,
                   typename RowSequenceT,
                   typename ColSequenceT,
                   typename BinaryOpT >
        void ewise_or_stencil_opt_accum(ZMatrixT           &Z,
                                        CMatrixT     const &C,
                                        TMatrixT     const &T,
                                        RowSequenceT const &row_indices,
                                        ColSequenceT const &col_indices,
                                        BinaryOpT           accum)
        {
            // If there is an accumulate operation, do nothing with the stencil
            using ZScalarType = typename ZMatrixT::ScalarType;
            using ZRowType = std::vector<std::tuple<IndexType,ZScalarType> >;

            ZRowType tmp_row;
            IndexType nRows(Z.nrows());

            for (IndexType row_idx = 0; row_idx < nRows; ++row_idx)
            {
                ewise_or(tmp_row, C[row_idx], T[row_idx], accum);
                Z.setRow(row_idx, tmp_row);
            }
        }

        //**********************************************************************
        template < typename ZMatrixT,
                   typename CMatrixT,
                   typename TMatrixT,
                   typename RowSequenceT,
                   typename ColSequenceT>
        void ewise_or_stencil_opt_accum(ZMatrixT           &Z,
                                        CMatrixT     const &C,
                                        TMatrixT     const &T,
                                        RowSequenceT const &row_indices,
                                        ColSequenceT const &col_indices,
                                        grb::NoAccumulate)
        {
            // If there is no accumulate, we need to annihilate stored values
            // in C that fall in the stencil
            using ZScalarType = typename ZMatrixT::ScalarType;
            using ZRowType = std::vector<std::tuple<IndexType,ZScalarType> >;

            ZRowType tmp_row;
            IndexType nRows(Z.nrows());

            for (IndexType row_idx = 0; row_idx < nRows; ++row_idx)
            {
                if (searchIndices(row_indices, row_idx))
                {
                    // Row Stenciled. merge C, T, using col stencil\n";
                    ewise_or_stencil(tmp_row, C[row_idx], T[row_idx],
                                     col_indices);
                    Z.setRow(row_idx, tmp_row);
                }
                else
                {
                    // Row not stenciled.  Take row from C only
                    // There should be nothing in T for this row
                    Z.setRow(row_idx, C[row_idx]);
                }
            }
        }

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
        template <typename TScalarT,
                  typename AScalarT>
        void vectorExpand(
            std::vector<std::tuple<IndexType, TScalarT>>        &vec_dest,
            BitmapSparseVector<AScalarT>                  const &vec_src,
            std::vector<std::tuple<IndexType, IndexType>> const &Indices)
        {
            vec_dest.clear();
            // The Indices are pairs of (output_index, input_index)
            // We do it this way, so we get the output in the right
            // order to begin with

            // Walk the output/input pairs building the output in correct order.
            for (auto&& [dest_idx, src_idx] : Indices)
            {
                if (vec_src.hasElementNoCheck(src_idx))
                {
                    vec_dest.emplace_back(
                        dest_idx,
                        static_cast<TScalarT>(
                            vec_src.extractElementNoCheck(src_idx)));
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
            vectorExpand(t, u, oi_pairs);

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
