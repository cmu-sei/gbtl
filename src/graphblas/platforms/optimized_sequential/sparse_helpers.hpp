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

/**
 * Implementations of all GraphBLAS functions optimized for the sequential
 * (CPU) backend.
 */

#pragma once

#include <functional>
#include <utility>
#include <vector>
#include <iterator>
#include <iostream>
#include <string>
#include <graphblas/algebra.hpp>
#include <graphblas/indices.hpp>

//****************************************************************************

namespace grb
{
    namespace backend
    {
        //**********************************************************************

        template <typename DstMatrixT,
                  typename SrcMatrixT>
        void sparse_copy(DstMatrixT &dstMatrix,
                         SrcMatrixT const &srcMatrix)
        {
            using DstScalarType = typename DstMatrixT::ScalarType;
            std::vector<std::tuple<IndexType, DstScalarType> > dstRow;

            // Copying removes the contents of the other matrix so clear it first.
            dstMatrix.clear();

            IndexType nrows(dstMatrix.nrows());
            for (IndexType row_idx = 0; row_idx < nrows; ++row_idx)
            {
                auto&& srcRow = srcMatrix[row_idx];
                dstRow.clear();

                // We need to construct a new row with the appropriate cast!
                for (auto&& [idx, srcVal] : srcRow)
                {
                    dstRow.emplace_back(idx, static_cast<DstScalarType>(srcVal));
                }

                if (!dstRow.empty())
                    dstMatrix.setRow(row_idx, dstRow);
            }
        }

        // @todo: Make a sparse copy where they are the same type for efficiency

        //**********************************************************************
        /// Advance the provided iterator until the value evaluates to true or
        /// the end is reached.
        ///
        /// Iter is iterator to std::vector<std::tuple<grb::IndexType,T>>
        template <typename Iter>
        void increment_until_true(Iter &iter, Iter const &iter_end)
        {
            while ((iter != iter_end) && !(std::get<1>(*iter))) {
                ++iter;
            }
        }

        //**********************************************************************
        /// Increment the provided iterator while the index is less than the
        /// provided index
        ///
        /// @retval true   If targeted index is found (false could mean end is
        ///                reached first or targeted index was not present
        template <typename Iter>
        bool increment_while_below(Iter                 &iter,
                                   Iter           const &iter_end,
                                   grb::IndexType        target_idx)
        {
            while ((iter != iter_end) && (std::get<0>(*iter) < target_idx))
            {
                ++iter;
            }
            return ((iter != iter_end) && (std::get<0>(*iter) == target_idx));
        }

        //**********************************************************************
        /// Increment the provided iterator while the index is less than the
        /// provided index and append any elements to vec along the way
        template <typename Iter, typename V>
        void increment_and_add_while_below(Iter                 &iter,
                                           Iter          const  &iter_end,
                                           grb::IndexType        idx,
                                           V                    &vec)
        {
            while ((iter != iter_end) && (std::get<0>(*iter) < idx)) {
                vec.push_back(*iter);  // anything more efficient?
                ++iter;
            }
        }
#if 0
        //**********************************************************************
        /// Perform the dot product of a row of a matrix with a sparse vector
        /// without pulling the indices out of the vector first.
        template <typename D1, typename D2, typename D3, typename SemiringT>
        bool dot2(D3                                                &ans,
                  std::vector<std::tuple<grb::IndexType,D1> > const &A_row,
                  std::vector<bool>                           const &u_bitmap,
                  std::vector<D2>                             const &u_vals,
                  grb::IndexType                                     u_nvals,
                  SemiringT                                          op)
        {
            bool value_set(false);
            ans = op.zero();

            if ((u_nvals == 0) || A_row.empty())
            {
                return value_set;
            }

            // find first stored value in u
            grb::IndexType u_idx(0);
            while (!u_bitmap[u_idx]) ++u_idx; // skip unstored elements

            // pull first value out of the row
            auto A_iter = A_row.begin();
            D1 a_val;
            grb::IndexType a_idx;

            // loop through both ordered sets to compute sparse dot prod
            while ((A_iter != A_row.end()) && (u_idx < u_vals.size()))
            {
                std::tie(a_idx, a_val) = *A_iter;
                if (u_idx == a_idx)
                {
                    if (value_set)
                    {
                        ans = op.add(ans, op.mult(a_val, u_vals[u_idx]));
                    }
                    else
                    {
                        ans = op.mult(a_val, u_vals[u_idx]);
                        value_set = true;
                    }

                    do { ++u_idx; } while ((u_idx < u_vals.size()) && !u_bitmap[u_idx]);
                    ++A_iter;
                }
                else if (u_idx > a_idx)
                {
                    ++A_iter;
                }
                else
                {
                    do { ++u_idx; } while ((u_idx < u_vals.size()) && !u_bitmap[u_idx]);
                }
            }

            return value_set;
        }
#endif

        //************************************************************************
        /// A dot product of two sparse vectors (vectors<tuple(index,value)>)
        template <typename D1, typename D2, typename D3, typename SemiringT>
        bool dot(D3                                                &ans,
                 std::vector<std::tuple<grb::IndexType,D1> > const &vec1,
                 std::vector<std::tuple<grb::IndexType,D2> > const &vec2,
                 SemiringT                                          op)
        {
            bool value_set(false);

            if (vec2.empty() || vec1.empty())
            {
                return value_set;
            }

            // point to first entries of the vectors
            auto v1_it = vec1.begin();
            auto v2_it = vec2.begin();

            // loop through both ordered sets to compute sparse dot prod
            while ((v1_it != vec1.end()) && (v2_it != vec2.end()))
            {
                if (std::get<0>(*v2_it) == std::get<0>(*v1_it))
                {
                    if (value_set)
                    {
                        ans = op.add(ans, op.mult(std::get<1>(*v1_it),
                                                  std::get<1>(*v2_it)));
                    }
                    else
                    {
                        ans = op.mult(std::get<1>(*v1_it),
                                      std::get<1>(*v2_it));
                        value_set = true;
                    }

                    ++v2_it;
                    ++v1_it;
                }
                else if (std::get<0>(*v2_it) > std::get<0>(*v1_it))
                {
                    ++v1_it;
                }
                else
                {
                    ++v2_it;
                }
            }

            return value_set;
        }

        //************************************************************************
        /// A dot product of two vectors (vec1 sparse, vec2 dense storage))
        template <typename D1, typename D2, typename D3, typename SemiringT>
        bool dot_sparse_dense(
            D3                                                &ans,
            std::vector<std::tuple<grb::IndexType,D1> > const &lhs,
            BitmapSparseVector<D2>                      const &rhs,
            SemiringT                                          op)
        {
            bool value_set(false);

            if (lhs.empty() || (rhs.nvals() == 0))
            {
                return value_set;
            }

            for (auto&& [idx, val] : lhs)
            {
                if (rhs.get_bitmap()[idx])
                {
                    if (value_set)
                    {
                        ans = op.add(ans, op.mult(val,
                                                  rhs.get_vals()[idx]));
                    }
                    else
                    {
                        ans = op.mult(val, rhs.get_vals()[idx]);
                        value_set = true;
                    }
                }
            }

            return value_set;
        }

        //************************************************************************
        /// A dot product of two vectors (vec1 sparse, vec2 dense storage))
        template <typename D1, typename D2, typename D3, typename SemiringT>
        bool dot_dense_sparse(
            D3                                                &ans,
            BitmapSparseVector<D2>                      const &lhs,
            std::vector<std::tuple<grb::IndexType,D1> > const &rhs,
            SemiringT                                          op)
        {
            bool value_set(false);

            if (rhs.empty() || (lhs.nvals() == 0))
            {
                return value_set;
            }

            for (auto&& [idx, val] : rhs)
            {
                if (lhs.get_bitmap()[idx])
                {
                    if (value_set)
                    {
                        ans = op.add(ans, op.mult(lhs.get_vals()[idx], val));
                    }
                    else
                    {
                        ans = op.mult(lhs.get_vals()[idx], val);
                        value_set = true;
                    }
                }
            }

            return value_set;
        }

        //**********************************************************************
        /// Apply element-wise operation to union of 2 sparse vectors, store in 3rd.
        /// ans = op(vec1, vec2)
        ///
        /// @note ans must be a unique vector from either vec1 or vec2
        template <typename D1, typename D2, typename D3, typename BinaryOpT>
        void ewise_or(std::vector<std::tuple<grb::IndexType,D3> >       &ans,
                      std::vector<std::tuple<grb::IndexType,D1> > const &vec1,
                      std::vector<std::tuple<grb::IndexType,D2> > const &vec2,
                      BinaryOpT                                          op)
        {
            if (((void*)&ans == (void*)&vec1) || ((void*)&ans == (void*)&vec2))
            {
                throw PanicException(
                    "backend::ewise_or called with same vector for input and output.");
            }

            ans.clear();

            // point to first entries of the vectors
            auto v1_it = vec1.begin();
            auto v2_it = vec2.begin();

            // loop through both ordered sets to compute ewise_or
            while ((v1_it != vec1.end()) || (v2_it != vec2.end()))
            {
                if ((v1_it != vec1.end()) && (v2_it != vec2.end()))
                {
                    auto&& [v1_idx, v1_val] = *v1_it;
                    auto&& [v2_idx, v2_val] = *v2_it;
                    //std::tie(v1_idx, v1_val) = *v1_it;
                    //std::tie(v2_idx, v2_val) = *v2_it;

                    if (v2_idx == v1_idx)
                    {
                        ans.emplace_back(v1_idx,
                                         static_cast<D3>(op(v1_val, v2_val)));

                        ++v2_it;
                        ++v1_it;
                    }
                    else if (v2_idx > v1_idx)
                    {
                        ans.emplace_back(v1_idx, static_cast<D3>(v1_val));
                        ++v1_it;
                    }
                    else
                    {
                        ans.emplace_back(v2_idx, static_cast<D3>(v2_val));
                        ++v2_it;
                    }
                }
                else if (v1_it != vec1.end())
                {
                    ans.emplace_back(std::get<0>(*v1_it),
                                     static_cast<D3>(std::get<1>(*v1_it)));
                    ++v1_it;
                }
                else // v2_it != vec2.end())
                {
                    ans.emplace_back(std::get<0>(*v2_it),
                                     static_cast<D3>(std::get<1>(*v2_it)));
                    ++v2_it;
                }
            }
        }

        //**********************************************************************
        /// Apply element-wise operation to union of 2 sparse vectors, store in 3rd.
        /// ans = op(vec1, vec2)
        ///
        /// @note ans must be a unique vector from either vec1 or vec2
        template <typename D1, typename D2, typename D3, typename BinaryOpT>
        void ewise_or_dense_sparse_v1(
            std::vector<std::tuple<grb::IndexType,D3> >       &ans,
            BitmapSparseVector<D1>                      const &vec1,
            std::vector<std::tuple<grb::IndexType,D2> > const &vec2,
            BinaryOpT                                          op)
        {
            if (((void*)&ans == (void*)&vec2)) // || ((void*)&ans == (void*)&vec2))
            {
                throw PanicException(
                    "backend::ewise_or called with same vector for input and output.");
            }

            ans.clear();

            // step through all entries of dense (while there are entries in sparse)
            grb::IndexType v1_val_count = vec1.nvals();
            grb::IndexType v1_idx = 0;

            // Step through elements of sparse vector
            for (auto&& [v2_idx, v2_val] : vec2)
            {
                for (; ((v1_idx < vec1.size()) && (v1_idx < v2_idx)); ++v1_idx)
                {
                    if (vec1.get_bitmap()[v1_idx])
                    {
                        ans.emplace_back(v1_idx,
                                         static_cast<D3>(vec1.get_vals()[v1_idx]));
                        --v1_val_count;
                    }
                }
                if (v1_val_count && vec1.get_bitmap()[v1_idx] && (v1_idx == v2_idx))
                {
                    ans.emplace_back(
                        v1_idx,
                        static_cast<D3>(op(vec1.get_vals()[v1_idx], v2_val)));
                    --v1_val_count;
                    ++v1_idx;
                }
                else
                {
                    ans.emplace_back(v1_idx, static_cast<D3>(v2_val));
                }
            }

            for (; v1_val_count && (v1_idx < vec1.size()); ++v1_idx)
            {
                if (vec1.get_bitmap()[v1_idx])
                {
                    ans.emplace_back(v1_idx,
                                     static_cast<D3>(vec1.get_vals()[v1_idx]));
                }
            }
        }

        //**********************************************************************
        /// Apply element-wise operation to union of 2 sparse vectors, store in 3rd.
        /// ans = op(vec1, vec2)
        ///
        /// @note ans must be a unique vector from either vec1 or vec2
        template <typename D1, typename D2, typename D3, typename BinaryOpT>
        void ewise_or_dense_sparse_v2(
            std::vector<std::tuple<grb::IndexType,D3> >       &ans,
            BitmapSparseVector<D1>                      const &vec1,
            std::vector<std::tuple<grb::IndexType,D2> > const &vec2,
            BinaryOpT                                          op)
        {
            if (((void*)&ans == (void*)&vec2)) // || ((void*)&ans == (void*)&vec2))
            {
                throw PanicException(
                    "backend::ewise_or called with same vector for input and output.");
            }

            ans.clear();

            // step through all entries of dense (while there are entries in sparse)
            grb::IndexType v1_val_count = vec1.nvals();
            grb::IndexType v1_idx = 0;
            while (v1_val_count && !vec1.get_bitmap()[v1_idx])
                ++v1_idx;
            auto v2_it(vec2.begin());
            auto v2_end(vec2.end());

            // while there are more values in either input continue to merge
            while (v1_val_count || v2_it != v2_end)
            {
                if (v1_val_count && (v2_it != vec2.end()))
                {
                    //auto&& [v1_idx, v1_val] = *v1_it;
                    auto&& [v2_idx, v2_val] = *v2_it;

                    if (v2_idx == v1_idx)
                    {
                        ans.emplace_back(
                            v1_idx,
                            static_cast<D3>(op(vec1.get_vals()[v1_idx], v2_val)));
                        ++v2_it;

                        //++v1_it;
                        --v1_val_count;
                        ++v1_idx;
                        while (v1_val_count && !vec1.get_bitmap()[v1_idx])
                            ++v1_idx;
                    }
                    else if (v2_idx > v1_idx)
                    {
                        ans.emplace_back(
                            v1_idx, static_cast<D3>(vec1.get_vals()[v1_idx]));
                        //++v1_it;
                        --v1_val_count;
                        ++v1_idx;
                        while (v1_val_count && !vec1.get_bitmap()[v1_idx])
                            ++v1_idx;
                    }
                    else
                    {
                        ans.emplace_back(v2_idx, static_cast<D3>(v2_val));
                        ++v2_it;
                    }
                }
                else if (v1_val_count)
                {
                    ans.emplace_back(
                        v1_idx,
                        static_cast<D3>(vec1.get_vals()[v1_idx]));
                    //++v1_it;
                    --v1_val_count;
                    ++v1_idx;
                    while (v1_val_count && !vec1.get_bitmap()[v1_idx])
                        ++v1_idx;
                }
                else // v2_it != vec2.end())
                {
                    ans.emplace_back(std::get<0>(*v2_it),
                                     static_cast<D3>(std::get<1>(*v2_it)));
                    ++v2_it;
                }
            }
        }

        //**********************************************************************
        /// Apply element-wise operation to union of 2 sparse vectors, store in 3rd.
        /// ans = op(vec1, vec2)
        ///
        /// @note ans must be a unique vector from either vec1 or vec2
        template <typename D1, typename D2, typename D3, typename BinaryOpT>
        void ewise_or_dense_dense_v1(
            std::vector<std::tuple<grb::IndexType,D3> >       &ans,
            BitmapSparseVector<D1>                      const &vec1,
            BitmapSparseVector<D2>                      const &vec2,
            BinaryOpT                                          op)
        {
            ans.clear();

            // step through all entries of both inputs while there are values
            grb::IndexType v1_val_count = vec1.nvals();
            grb::IndexType v2_val_count = vec2.nvals();
            for (grb::IndexType idx = 0;
                 idx < vec1.size() && (v1_val_count || v2_val_count);
                 ++idx)
            {
                if (v1_val_count && vec1.get_bitmap()[idx])
                {
                    --v1_val_count;
                    if (v2_val_count && vec2.get_bitmap()[idx])
                    {
                        --v2_val_count;
                        ans.emplace_back(
                            idx, static_cast<D3>(op(vec1.get_vals()[idx],
                                                    vec2.get_vals()[idx])));
                    }
                    else
                    {
                        ans.emplace_back(idx,
                                         static_cast<D3>(vec1.get_vals()[idx]));
                    }
                }
                else if (v2_val_count && vec2.get_bitmap()[idx])
                {
                    --v2_val_count;
                    ans.emplace_back(idx,
                                     static_cast<D3>(vec2.get_vals()[idx]));
                }
            }
        }

        //**********************************************************************
        /// Apply element-wise operation to union of 2 sparse vectors, store in 3rd.
        /// ans = op(vec1, vec2)
        ///
        /// @note ans must be a unique vector from either vec1 or vec2
        template <typename D1, typename D2, typename D3, typename BinaryOpT>
        void ewise_or_dense_dense_v2(
            std::vector<std::tuple<grb::IndexType,D3> >       &ans,
            BitmapSparseVector<D1>                      const &vec1,
            BitmapSparseVector<D2>                      const &vec2,
            BinaryOpT                                          op)
        {
            ans.clear();

            // step through all entries of both inputs while there are values
            for (grb::IndexType idx = 0; idx < vec1.size(); ++idx)
            {
                if (vec1.get_bitmap()[idx])
                {
                    if (vec2.get_bitmap()[idx])
                    {
                        ans.emplace_back(
                            idx, static_cast<D3>(op(vec1.get_vals()[idx],
                                                    vec2.get_vals()[idx])));
                    }
                    else
                    {
                        ans.emplace_back(idx,
                                         static_cast<D3>(vec1.get_vals()[idx]));
                    }
                }
                else if (vec2.get_bitmap()[idx])
                {
                    ans.emplace_back(idx,
                                     static_cast<D3>(vec2.get_vals()[idx]));
                }
            }
        }

        //**********************************************************************
        //**********************************************************************

        //**********************************************************************
        template < typename ZMatrixT,
                   typename CMatrixT,
                   typename TMatrixT,
                   typename BinaryOpT >
        void ewise_or_opt_accum(ZMatrixT         &Z,
                                CMatrixT const   &C,
                                TMatrixT const   &T,
                                BinaryOpT         accum)
        {
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

        // Specialized version that gets used when we don't have an accumulator
        template < typename ZMatrixT,
                   typename CMatrixT,
                   typename TMatrixT>
        void ewise_or_opt_accum(ZMatrixT                &Z,
                                CMatrixT          const &C,
                                TMatrixT          const &T,
                                grb::NoAccumulate )
        {
            sparse_copy(Z, T);
        }

        //**********************************************************************
        template <typename ZScalarT,
                  typename WVectorT,
                  typename TScalarT,
                  typename BinaryOpT>
        void ewise_or_opt_accum_1D(
            std::vector<std::tuple<grb::IndexType,ZScalarT>>       &z,
            WVectorT const                                         &w,
            std::vector<std::tuple<grb::IndexType,TScalarT>> const &t,
            BinaryOpT                                               accum)
        {
            //z.clear();
            ewise_or_dense_sparse_v2(z, w, t, accum);
        }

        //**********************************************************************
        // Specialized version that gets used when we don't have an accumulator
        template <typename ZScalarT,
                  typename WVectorT,
                  typename TScalarT>
        void ewise_or_opt_accum_1D(
            std::vector<std::tuple<grb::IndexType,ZScalarT>>       &z,
            WVectorT const                                         &w,
            std::vector<std::tuple<grb::IndexType,TScalarT>> const &t,
            grb::NoAccumulate )
        {
            //sparse_copy(z, t);
            for (auto tupl: t)
            {
                z.emplace_back(std::get<0>(tupl),
                               static_cast<ZScalarT>(std::get<1>(tupl)));
            }
        }

        //************************************************************************
        //************************************************************************

        //************************************************************************
        /// Apply element-wise operation to intersection of sparse vectors.
        template <typename D1, typename D2, typename D3, typename BinaryOpT>
        void ewise_and(std::vector<std::tuple<grb::IndexType,D3> >       &ans,
                       std::vector<std::tuple<grb::IndexType,D1> > const &vec1,
                       std::vector<std::tuple<grb::IndexType,D2> > const &vec2,
                       BinaryOpT                                          op)
        {
            ans.clear();

            // todo: early exit if either input vector is empty?

            // point to first entries of the vectors
            auto v1_it = vec1.begin();
            auto v2_it = vec2.begin();

            // loop through both ordered sets to compute ewise_and
            while ((v1_it != vec1.end()) && (v2_it != vec2.end()))
            {
                if (std::get<0>(*v2_it) == std::get<0>(*v1_it))
                {
                    ans.emplace_back(std::get<0>(*v1_it),
                                     static_cast<D3>(op(std::get<1>(*v1_it),
                                                        std::get<1>(*v2_it))));

                    ++v2_it;
                    ++v1_it;
                }
                else if (std::get<0>(*v2_it) > std::get<0>(*v1_it))
                {
                    ++v1_it;
                }
                else
                {
                    ++v2_it;
                }
            }
        }

        //************************************************************************
        /// Apply element-wise operation to intersection of dense vectors.
        template <typename D1, typename D2, typename D3, typename BinaryOpT>
        void ewise_and_dense_dense_v2(
            std::vector<std::tuple<grb::IndexType,D3> >       &ans,
            BitmapSparseVector<D1>                      const &vec1,
            BitmapSparseVector<D2>                      const &vec2,
            BinaryOpT                                          op)
        {
            ans.clear();

            // todo: early exit if either input vector is empty?

            // scan through both bitmaps to compute ewise_and
            for (grb::IndexType idx = 0; idx < vec1.size(); ++idx)
            {
                if (vec1.get_bitmap()[idx] && vec2.get_bitmap()[idx])
                {
                    ans.emplace_back(
                        idx, static_cast<D3>(op(vec1.get_vals()[idx],
                                                vec2.get_vals()[idx])));
                }
            }
        }

        //**********************************************************************
        //**********************************************************************
        /**
         * This merges the values of C and Z into the result vector based on the
         * values of the mask.
         *
         * If outp == REPLACE:
         *
         * \f[ L(C) = {(i,j,Zij):(i,j) \in (ind(Z) \cap ind(M))} \f]
         *
         * If outp == MERGE:
         *
         * \f[ L(C) = {(i,j,Zij):(i,j) \in (ind(C) \cap ind(\neg M))} \cup
         *            {(i,j,Zij):(i,j) \in (ind(Z) \cap ind(M))} \f]
         *
         * @tparam CScalarT The scalar type of the C vector input AND result.
         * @tparam ZScalarT The scalar type of the Z vector input.
         * @tparam MScalarT The scalar type of the mask vector.
         *
         * @param result   Result vector.  We clear this first.
         * @param c_vec    The original c values that may be carried through.
         * @param z_vec    The new values to insert/overlay.
         * @param mask_vec The mask which specifies which values to use.
         * @param outp     If REPLACE, we should always clear the values specified
         *                 by the mask regardless if they are overlayed.
         */
        template < typename CScalarT,
                   typename ZScalarT,
                   typename MScalarT>
        void apply_with_mask(
            std::vector<std::tuple<IndexType, CScalarT> >          &result,
            std::vector<std::tuple<IndexType, CScalarT> > const    &c_vec,
            std::vector<std::tuple<IndexType, ZScalarT> > const    &z_vec,
            std::vector<std::tuple<IndexType, MScalarT> > const    &mask_vec,
            OutputControlEnum                                       outp)
        {
            auto c_it = c_vec.begin();
            auto z_it = z_vec.begin();
            auto mask_it = mask_vec.begin();

            result.clear();

            // Design: This approach is driven by the mask.
            while (mask_it != mask_vec.end())
            {
                // Make sure the mask is on a valid value
                increment_until_true(mask_it, mask_vec.end());

                // If we run out of mask, we are done!
                if (mask_it == mask_vec.end())
                {
                    break;
                }

                // Get the mask location
                auto mask_idx(std::get<0>(*mask_it));

                // If outp==REPLACE, then we don't consider original C values.
                // If outp==MERGE, then we want to keep C values outside the mask
                // Any values in C "outside" the mask should now be applied
                // So, we catch "c" up to the mask.  This is the intersection
                // of C and !M.

                if (outp == MERGE)
                {
                    // This is the part of (ind(C) \cap int(\not M)
                    increment_and_add_while_below(c_it, c_vec.end(), mask_idx,
                                                  result);
                }

                // "Catch up" the input to the mask.
                if (increment_while_below(z_it, z_vec.end(), mask_idx))
                {
                    // Now, at the mask point add the value from Z if we have one.
                    result.emplace_back(
                        mask_idx, static_cast<CScalarT>(std::get<1>(*z_it)));
                }

                // If there is a C here, skip it
                if (c_it != c_vec.end())
                {
                    if (std::get<0>(*c_it) == mask_idx)
                        ++c_it;
                }

                // Now move to the next mask entry.
                ++mask_it;
            } // while mask_it != end

            // Now, we need to add the remaining C values beyond the mask
            if (outp == MERGE)
            {
                // This is the part of (ind(C) \cap int(\not M)
                while (c_it != c_vec.end())
                {
                    result.emplace_back(*c_it);
                    ++c_it;
                }
            }
        } // apply_with_mask

        //**********************************************************************
        // Matrix Mask Churn
        //**********************************************************************

        //**********************************************************************
        // WARNING: costly
        template <typename MatrixT>
        decltype(auto)
        get_structure_row(MatrixT const &mat, IndexType row_idx)
        {
            std::vector<std::tuple<IndexType, bool> > mask_tuples;
            mask_tuples.reserve(mat.ncols() - mat[row_idx].size());

            for (auto&& [ix, val] : mat[row_idx])
            {
                mask_tuples.emplace_back(ix, true);
            }

            return mask_tuples;
        }

        //**********************************************************************
        // WARNING: costly
        template <typename MatrixT>
        decltype(auto)
        get_complement_row(MatrixT const &mat, IndexType row_idx)
        {
            std::vector<std::tuple<IndexType, bool> > mask_tuples;
            auto &row_tuples = mat[row_idx];
            mask_tuples.reserve(mat.ncols() - row_tuples.size());
            auto it = row_tuples.begin();

            for (IndexType ix = 0; ix < mat.ncols(); ++ix)
            {
                if ((it == row_tuples.end()) || (ix < std::get<0>(*it)))
                {
                    mask_tuples.emplace_back(ix, true);
                }
                else // ix == std::get<0>(*it)
                {
                    if (static_cast<bool>(std::get<1>(*it)) == false)
                    {
                        mask_tuples.emplace_back(ix, true);
                    }
                    ++it;
                }
            }

            return mask_tuples;
        }

        //**********************************************************************
        // WARNING: costly
        template <typename MatrixT>
        decltype(auto)
        get_structural_complement_row(MatrixT const &mat, IndexType row_idx)
        {
            std::vector<std::tuple<IndexType, bool> > mask_tuples;
            auto &row_tuples = mat[row_idx];
            mask_tuples.reserve(mat.ncols() - row_tuples.size());
            auto it = row_tuples.begin();

            for (IndexType ix = 0; ix < mat.ncols(); ++ix)
            {
                if ((it == row_tuples.end()) || (ix < std::get<0>(*it)))
                {
                    mask_tuples.emplace_back(ix, true);
                }
                else // ix == std::get<0>(*it), so skip
                {
                    ++it;
                }
            }

            return mask_tuples;
        }

        //**********************************************************************
        //**********************************************************************

        //**********************************************************************
        // Matrix version
        template < typename CMatrixT,
                   typename ZMatrixT,
                   typename MMatrixT>
        void write_with_opt_mask(CMatrixT           &C,
                                 ZMatrixT   const   &Z,
                                 MMatrixT   const   &Mask,
                                 OutputControlEnum   outp)
        {
            using CScalarType = typename CMatrixT::ScalarType;
            using CRowType = std::vector<std::tuple<IndexType, CScalarType> >;

            CRowType tmp_row;
            IndexType nRows(C.nrows());
            for (IndexType row_idx = 0; row_idx < nRows; ++row_idx)
            {
                apply_with_mask(tmp_row, C[row_idx], Z[row_idx],
                                Mask[row_idx], outp);

                // Now, set the new one.  Yes, we can optimize this later
                C.setRow(row_idx, tmp_row);
            }
        }

        //**********************************************************************
        // Matrix version specialized for complement mask
        template < typename CMatrixT,
                   typename ZMatrixT,
                   typename MMatrixT>
        void write_with_opt_mask(
            CMatrixT                                  &C,
            ZMatrixT                            const &Z,
            grb::MatrixComplementView<MMatrixT> const &Mask,
            OutputControlEnum                          outp)
        {
            using CScalarType = typename CMatrixT::ScalarType;
            using CRowType = std::vector<std::tuple<IndexType, CScalarType> >;

            CRowType tmp_row;
            IndexType nRows(C.nrows());
            for (IndexType row_idx = 0; row_idx < nRows; ++row_idx)
            {
                apply_with_mask(tmp_row, C[row_idx], Z[row_idx],
                                get_complement_row(Mask.m_mat, row_idx),
                                outp);

                // Now, set the new one.  Yes, we can optimize this later
                C.setRow(row_idx, tmp_row);
            }
        }

        //**********************************************************************
        // Matrix version specialized for structure mask
        template < typename CMatrixT,
                   typename ZMatrixT,
                   typename MMatrixT>
        void write_with_opt_mask(
            CMatrixT                                 &C,
            ZMatrixT                           const &Z,
            grb::MatrixStructureView<MMatrixT> const &Mask,
            OutputControlEnum                         outp)
        {
            using CScalarType = typename CMatrixT::ScalarType;
            using CRowType = std::vector<std::tuple<IndexType, CScalarType> >;

            CRowType tmp_row;
            IndexType nRows(C.nrows());
            for (IndexType row_idx = 0; row_idx < nRows; ++row_idx)
            {
                apply_with_mask(tmp_row, C[row_idx], Z[row_idx],
                                get_structure_row(Mask.m_mat, row_idx),
                                outp);

                // Now, set the new one.  Yes, we can optimize this later
                C.setRow(row_idx, tmp_row);
            }
        }

        //**********************************************************************
        // Matrix version specialized for structural complement mask
        template < typename CMatrixT,
                   typename ZMatrixT,
                   typename MMatrixT>
        void write_with_opt_mask(
            CMatrixT                                            &C,
            ZMatrixT                                      const &Z,
            grb::MatrixStructuralComplementView<MMatrixT> const &Mask,
            OutputControlEnum                                    outp)
        {
            using CScalarType = typename CMatrixT::ScalarType;
            using CRowType = std::vector<std::tuple<IndexType, CScalarType> >;

            CRowType tmp_row;
            IndexType nRows(C.nrows());
            for (IndexType row_idx = 0; row_idx < nRows; ++row_idx)
            {
                apply_with_mask(tmp_row, C[row_idx], Z[row_idx],
                                get_structural_complement_row(Mask.m_mat, row_idx),
                                outp);

                // Now, set the new one.  Yes, we can optimize this later
                C.setRow(row_idx, tmp_row);
            }
        }

        //**********************************************************************
        // Matrix version specialized for no mask
        template < typename CMatrixT,
                   typename ZMatrixT >
        void write_with_opt_mask(CMatrixT                   &C,
                                 ZMatrixT           const   &Z,
                                 grb::NoMask        const   &foo,
                                 OutputControlEnum           outp)
        {
            sparse_copy(C, Z);
        }

        //**********************************************************************
        // Vector Mask Churn
        //**********************************************************************

        //**********************************************************************
        // WARNING: costly
        template <typename VectorT>
        decltype(auto)
        get_structure_contents(VectorT const &vec)
        {
            std::vector<std::tuple<IndexType, bool> > mask_tuples;

            for (auto [ix, val] : vec.getContents())
            {
                mask_tuples.emplace_back(ix, true);
            }

            return mask_tuples;
        }

        //**********************************************************************
        // WARNING: costly
        template <typename VectorT>
        decltype(auto)
        get_complement_contents(VectorT const &vec)
        {
            std::vector<std::tuple<IndexType, bool> > mask_tuples;
            auto row_tuples(vec.getContents());
            auto it = row_tuples.begin();

            for (IndexType ix = 0; ix < vec.size(); ++ix)
            {
                if ((it == row_tuples.end()) || (ix < std::get<0>(*it)))
                {
                    mask_tuples.emplace_back(ix, true);
                }
                else
                {
                    if (static_cast<bool>(std::get<1>(*it)) == false)
                    {
                        mask_tuples.emplace_back(ix, true);
                    }
                    ++it;
                }
            }

            return mask_tuples;
        }

        //**********************************************************************
        // WARNING: costly
        template <typename VectorT>
        decltype(auto)
        get_structural_complement_contents(VectorT const &vec)
        {
            std::vector<std::tuple<IndexType, bool> > mask_tuples;
            auto row_tuples(vec.getContents());
            auto it = row_tuples.begin();

            for (IndexType ix = 0; ix < vec.size(); ++ix)
            {
                if ((it == row_tuples.end()) || (ix < std::get<0>(*it)))
                {
                    mask_tuples.emplace_back(ix, true);
                }
                else
                {
                    ++it;
                }
            }

            return mask_tuples;
        }

        //**********************************************************************
        //**********************************************************************

        //**********************************************************************
        // Vector version
        template <typename WVectorT,
                  typename ZScalarT,
                  typename MaskT>
        void write_with_opt_mask_1D(
            WVectorT                                           &w,
            std::vector<std::tuple<IndexType, ZScalarT>> const &z,
            MaskT                                        const &mask,
            OutputControlEnum                                   outp)
        {
            using WScalarType = typename WVectorT::ScalarType;
            std::vector<std::tuple<IndexType, WScalarType> > tmp_row;

            apply_with_mask(tmp_row, w.getContents(), z,
                            mask.getContents(),
                            outp);

            // Now, set the new one.  Yes, we can optimize this later
            w.setContents(tmp_row);
        }

        //**********************************************************************
        // Vector version specialized for complement mask
        template <typename WVectorT,
                  typename ZScalarT,
                  typename MaskT>
        void write_with_opt_mask_1D(
            WVectorT                                           &w,
            std::vector<std::tuple<IndexType, ZScalarT>> const &z,
            grb::VectorComplementView<MaskT>             const &mask,
            OutputControlEnum                                   outp)
        {
            using WScalarType = typename WVectorT::ScalarType;
            std::vector<std::tuple<IndexType, WScalarType> > tmp_row;

            apply_with_mask(tmp_row, w.getContents(), z,
                            get_complement_contents(mask.m_vec),
                            outp);

            // Now, set the new one.  Yes, we can optimize this later
            w.setContents(tmp_row);
        }

        //**********************************************************************
        // Vector version specialized for structure mask
        template <typename WVectorT,
                  typename ZScalarT,
                  typename MaskT>
        void write_with_opt_mask_1D(
            WVectorT                                           &w,
            std::vector<std::tuple<IndexType, ZScalarT>> const &z,
            grb::VectorStructureView<MaskT>              const &mask,
            OutputControlEnum                                   outp)
        {
            using WScalarType = typename WVectorT::ScalarType;
            std::vector<std::tuple<IndexType, WScalarType> > tmp_row;

            apply_with_mask(tmp_row, w.getContents(), z,
                            get_structure_contents(mask.m_vec),
                            outp);

            // Now, set the new one.  Yes, we can optimize this later
            w.setContents(tmp_row);
        }


        //**********************************************************************
        // Vector version specialized for structural complement mask
        template <typename WVectorT,
                  typename ZScalarT,
                  typename MaskT>
        void write_with_opt_mask_1D(
            WVectorT                                           &w,
            std::vector<std::tuple<IndexType, ZScalarT>> const &z,
            grb::VectorStructuralComplementView<MaskT>   const &mask,
            OutputControlEnum                                   outp)
        {
            using WScalarType = typename WVectorT::ScalarType;
            std::vector<std::tuple<IndexType, WScalarType> > tmp_row;

            apply_with_mask(tmp_row, w.getContents(), z,
                            get_structural_complement_contents(mask.m_vec),
                            outp);

            // Now, set the new one.  Yes, we can optimize this later
            w.setContents(tmp_row);
        }

        //**********************************************************************
        // Vector version specialized for no mask
        template <typename WVectorT,
                  typename ZScalarT>
        void write_with_opt_mask_1D(
            WVectorT                                           &w,
            std::vector<std::tuple<IndexType, ZScalarT>> const &z,
            grb::NoMask                                  const &foo,
            OutputControlEnum                                   outp)
        {
            //sparse_copy(w, z);
            w.setContents(z);
        }

        //**********************************************************************
        //**********************************************************************

        //**********************************************************************
        // Do one of the following...
        // w          = z,
        // w<    m,r> = z,
        // w<   !m,r> = z,
        // w< s(m),r> = z,
        // w<!s(m),r> = z,
        //
        // w and mask are dense storage, z is sparse
        template <typename WScalarT,
                  typename ZScalarT,
                  typename MaskT>
        void write_with_opt_mask_1D_sparse_dense(
            BitmapSparseVector<WScalarT>                       &w,
            std::vector<std::tuple<IndexType, ZScalarT>> const &z,
            MaskT                                        const &mask,
            OutputControlEnum                                   outp)
        {
            if (outp == REPLACE)
            {
                w.clear();
                for (auto&& [idx, val] : z)
                {
                    if (check_mask_1D(mask, idx))
                    {
                        w.setElementNoCheck(idx, val);
                    }
                }
            }
            else // MERGE
            {
                for (auto&& [idx, val] : z)
                {
                    if (check_mask_1D(mask, idx))
                    {
                        w.setElementNoCheck(idx, val);
                    }
                    else
                    {
                        w.removeElementNoCheck(idx);
                    }
                }
            }
        }

        //********************************************************************
        // mxm helpers (may be of more general use).
        //********************************************************************

        // *******************************************************************
        // Return true if iterator points to location with target_index;
        // otherwise returns the at the insertion point for target_index
        // which could be it_end.
        template <typename TupleIteratorT>
        bool advance_and_check_tuple_iterator(
            TupleIteratorT       &it,
            TupleIteratorT const &it_end,
            IndexType             target_index)
        {
            GRB_LOG_FN_BEGIN("advance_and_check_tuple_iterator: tgt = "
                             << target_index);

            while ((it != it_end) && (std::get<0>(*it) < target_index))
            {
                ++it;
            }

            bool tmp = ((it != it_end) && (std::get<0>(*it) == target_index));
            GRB_LOG_FN_END("advance_and_check_tuple_iterator target_found = "
                           << tmp);
            return tmp;
        }

        // *******************************************************************
        // Only returns true if target index is found AND it evaluates to true
        // or the structure flag is set.
        template <typename TupleIteratorT>
        bool advance_and_check_mask_iterator(
            TupleIteratorT       &it,
            TupleIteratorT const &it_end,
            bool                  structure_flag,
            IndexType             target_index)
        {
            GRB_LOG_FN_BEGIN("advance_and_check_mask_iterator: sflag = "
                             << structure_flag << ", tgt = " << target_index);

            bool tmp =
                (advance_and_check_tuple_iterator(it, it_end, target_index) &&
                 (structure_flag || static_cast<bool>(std::get<1>(*it))));

            GRB_LOG_FN_END("advance_and_check_mask_iterator: result = " << tmp);
            return tmp;
        }

        //**********************************************************************
        //**********************************************************************
        // Check to see if 1D-mask allows writing
        // Operate directly on the BitmapSparseVector.
        //
        // Only returns true if target index is found AND it evaluates to true
        // or the structure flag is set.
        //**********************************************************************
        template <typename MScalarT>
        inline  bool check_mask_1D(
            BitmapSparseVector<MScalarT> const &mask,
            bool                                structure_flag,
            bool                                complement_flag,
            IndexType                           target_index)
        {
            GRB_LOG_FN_BEGIN("check_mask: s/c flags = "
                             << structure_flag << "/" << complement_flag
                             << ", tgt = " << target_index);

            bool tmp =
                (complement_flag !=
                 (mask.hasElementNoCheck(target_index) &&
                  (structure_flag || mask.extractElementNoCheck(target_index))));

            GRB_LOG_FN_END("check_mask: result = " << tmp);
            return tmp;
        }

        template <typename MScalarT>
        inline bool check_mask_1D(
            BitmapSparseVector<MScalarT> const &mask,
            IndexType                           target_index)
        {
            return check_mask_1D(mask, false, false, target_index);
        }

        template <typename MaskT>
        inline bool check_mask_1D(
            grb::VectorComplementView<MaskT> const &mask,
            IndexType                               target_index)
        {
            //std::cout << "C";
            return check_mask_1D(mask.m_vec, false, true, target_index);
        }

        template <typename MaskT>
        inline bool check_mask_1D(
            grb::VectorStructureView<MaskT> const &mask,
            IndexType                              target_index)
        {
            return check_mask_1D(mask.m_vec, true, false, target_index);
        }

        template <typename MaskT>
        inline bool check_mask_1D(
            grb::VectorStructuralComplementView<MaskT> const &mask,
            IndexType                                         target_index)
        {
            return check_mask_1D(mask.m_vec, true, true, target_index);
        }

        inline bool check_mask_1D(NoMask const &mask,
                                  IndexType     target_index)
        {
            return true;
        }

        //**********************************************************************
        //**********************************************************************
        template <typename WScalarT,
                  typename MaskT,
                  typename AccumT,
                  typename TScalarT>
        void opt_accum_with_opt_mask_1D(BitmapSparseVector<WScalarT>       &w,
                                        MaskT                              &mask,
                                        AccumT                              accum,
                                        BitmapSparseVector<TScalarT> const &t,
                                        OutputControlEnum                   outp)
        {
            for (grb::IndexType idx = 0; idx < w.size(); ++idx)
            {
                if (check_mask_1D(mask, idx))
                {
                    if (t.hasElementNoCheck(idx))
                    {
                        if (w.hasElementNoCheck(idx))
                        {
                            w.setElementNoCheck(
                                idx, accum(w.extractElementNoCheck(idx),
                                           t.extractElementNoCheck(idx)));
                        }
                        else
                        {
                            w.setElementNoCheck(idx, t.extractElementNoCheck(idx));
                        }
                    }
                }
                else if (outp == REPLACE)
                {
                    w.removeElementNoCheck(idx);
                }
            }
        }

        //**********************************************************************
        template <typename WScalarT,
                  typename MaskT,
                  typename TScalarT>
        void opt_accum_with_opt_mask_1D(BitmapSparseVector<WScalarT>       &w,
                                        MaskT                              &mask,
                                        grb::NoAccumulate                   accum,
                                        BitmapSparseVector<TScalarT> const &t,
                                        OutputControlEnum                   outp)
        {
            for (grb::IndexType idx = 0; idx < w.size(); ++idx)
            {
                if (check_mask_1D(mask, idx))
                {
                    if (t.hasElementNoCheck(idx))
                    {
                        w.setElementNoCheck(idx, t.extractElementNoCheck(idx));
                    }
                    else
                    {
                        w.removeElementNoCheck(idx);
                    }
                }
                else if (outp == REPLACE)
                {
                    w.removeElementNoCheck(idx);
                }
            }
        }

        //**********************************************************************
        template <typename WScalarT,
                  typename AccumT,
                  typename TScalarT>
        void opt_accum_with_opt_mask_1D(BitmapSparseVector<WScalarT>       &w,
                                        grb::NoMask                  const &m,
                                        AccumT                              accum,
                                        BitmapSparseVector<TScalarT> const &t,
                                        OutputControlEnum                   outp)
        {
            for (grb::IndexType idx = 0; idx < w.size(); ++idx)
            {
                if (t.hasElementNoCheck(idx))
                {
                    if (w.hasElementNoCheck(idx))
                    {
                        w.setElementNoCheck(idx,
                                            accum(w.extractElementNoCheck(idx),
                                                  t.extractElementNoCheck(idx)));
                    }
                    else
                    {
                        w.setElementNoCheck(idx, t.extractElementNoCheck(idx));
                    }
                }
            }
        }

        //**********************************************************************
        template <typename WScalarT,
                  typename TScalarT>
        void opt_accum_with_opt_mask_1D(BitmapSparseVector<WScalarT>       &w,
                                        grb::NoMask const &,
                                        grb::NoAccumulate ,
                                        BitmapSparseVector<TScalarT> const &t,
                                        OutputControlEnum )
        {
            /// @todo implement move and/or swap when t and w same type
            /// @todo move/swap the bitmap regardless of type

            w.clear();
            for (grb::IndexType idx = 0; idx < w.size(); ++idx)
            {
                if (t.hasElementNoCheck(idx))
                {
                    w.setElementNoCheck(idx, t.extractElementNoCheck(idx));
                }
            }
        }

        //**********************************************************************
        /// accumulate one sparse vector with another (applying op in intersection).
        ///  "xpey = x plus equals y"
        /// vec1 += vec2
        ///
        /// @note similarities with axpy()
        template <typename D1, typename D2, typename BinaryOpT>
        void xpey(
            std::vector<std::tuple<grb::IndexType,D1> >       &vec1,
            std::vector<std::tuple<grb::IndexType,D2> > const &vec2,
            BinaryOpT                                          op)
        {
            // point to first entries of the destination vector
            auto v1_it = vec1.begin();

            for (auto&& [v2_idx, v2_val] : vec2)
            {
                // scan forward through vec1 to find insert or merge point
                if (advance_and_check_tuple_iterator(v1_it, vec1.end(), v2_idx))
                {
                    // merge
                    std::get<1>(*v1_it) = op(std::get<1>(*v1_it), v2_val);
                    ++v1_it;
                }
                else
                {
                    // insert
                    v1_it = vec1.insert(
                        v1_it, std::make_tuple(v2_idx, static_cast<D1>(v2_val)));
                    ++v1_it;
                }
            }
        }

        // *******************************************************************
        /// perform the following operation on sparse vectors implemented as
        /// vector<tuple<Index, value>>
        ///
        /// t += a_ik*b[:]
        template<typename CScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        void axpy(
            std::vector<std::tuple<IndexType, CScalarT>>       &t,
            SemiringT                                           semiring,
            AScalarT                                            a,
            std::vector<std::tuple<IndexType, BScalarT>> const &b)
        {
            GRB_LOG_FN_BEGIN("axpy");
            auto t_it = t.begin();

            for (auto&& [j, b_j] : b)
            {
                GRB_LOG_VERBOSE("j = " << j);

                auto t_j(semiring.mult(a, b_j));
                GRB_LOG_VERBOSE("temp = " << t_j);

                // scan through C_row to find insert/merge point
                if (advance_and_check_tuple_iterator(t_it, t.end(), j))
                {
                    GRB_LOG_VERBOSE("Accumulating");
                    std::get<1>(*t_it) = semiring.add(std::get<1>(*t_it), t_j);
                    ++t_it;
                }
                else
                {
                    GRB_LOG_VERBOSE("Inserting");
                    t_it = t.insert(t_it,
                                    std::make_tuple(j, static_cast<CScalarT>(t_j)));
                    ++t_it;
                }
            }
            GRB_LOG_FN_END("axpy");
        }

        // *******************************************************************
        /// perform the following operation on sparse vectors implemented as
        /// vector<tuple<Index, value>>
        ///
        /// t += a[:]*b_kj
        template<typename CScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        void axpy(
            std::vector<std::tuple<IndexType, CScalarT>>       &t,
            SemiringT                                           semiring,
            std::vector<std::tuple<IndexType, BScalarT>> const &a,
            AScalarT                                            b)
        {
            GRB_LOG_FN_BEGIN("axpy");
            auto t_it = t.begin();

            for (auto&& [j, a_j] : a)
            {
                GRB_LOG_VERBOSE("j = " << j);

                auto t_j(semiring.mult(a_j, b));
                GRB_LOG_VERBOSE("temp = " << t_j);

                // scan through C_row to find insert/merge point
                if (advance_and_check_tuple_iterator(t_it, t.end(), j))
                {
                    GRB_LOG_VERBOSE("Accumulating");
                    std::get<1>(*t_it) = semiring.add(std::get<1>(*t_it), t_j);
                    ++t_it;
                }
                else
                {
                    GRB_LOG_VERBOSE("Inserting");
                    t_it = t.insert(t_it,
                                    std::make_tuple(j, static_cast<CScalarT>(t_j)));
                    ++t_it;
                }
            }
            GRB_LOG_FN_END("axpy");
        }

        // *******************************************************************
        /// perform the following operation on sparse vectors implemented as
        /// vector<tuple<Index, value>>
        ///
        /// t += a_ik*b[:]
        template<typename CScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        void axpy(
            BitmapSparseVector<CScalarT>                       &t,
            SemiringT                                           semiring,
            AScalarT                                            a,
            std::vector<std::tuple<IndexType, BScalarT>> const &b)
        {
            GRB_LOG_FN_BEGIN("axpy(dense t)");
            //auto t_it = t.begin();

            for (auto&& [j, b_j] : b)
            {
                GRB_LOG_VERBOSE("j = " << j);

                auto t_j(semiring.mult(a, b_j));
                GRB_LOG_VERBOSE("temp = " << t_j);

                if (t.hasElementNoCheck(j))
                {
                    t.setElementNoCheck(j, semiring.add(t.extractElementNoCheck(j),
                                                        t_j));
                }
                else
                {
                    t.setElementNoCheck(j, t_j);
                }
            }
            GRB_LOG_FN_END("axpy");
        }

        // *******************************************************************
        /// perform the following operation on sparse vectors implemented as
        /// vector<tuple<Index, value>>
        ///
        /// t += b[:]*a_ik
        template<typename CScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        void axpy(
            BitmapSparseVector<CScalarT>                       &t,
            SemiringT                                           semiring,
            std::vector<std::tuple<IndexType, BScalarT>> const &b,
            AScalarT                                            a)
        {
            GRB_LOG_FN_BEGIN("axpy(dense t)");
            //auto t_it = t.begin();

            for (auto&& [j, b_j] : b)
            {
                GRB_LOG_VERBOSE("j = " << j);

                auto t_j(semiring.mult(b_j, a));
                GRB_LOG_VERBOSE("temp = " << t_j);

                if (t.hasElementNoCheck(j))
                {
                    t.setElementNoCheck(j, semiring.add(t.extractElementNoCheck(j),
                                                        t_j));
                }
                else
                {
                    t.setElementNoCheck(j, t_j);
                }
            }
            GRB_LOG_FN_END("axpy");
        }

        // *******************************************************************
        /// perform the following operation on sparse vectors implemented as
        /// vector<tuple<Index, value>>
        ///
        /// t<[m[:]]> += a_ik*b[:]
        template<typename TScalarT,
                 typename MScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        void masked_axpy(
            std::vector<std::tuple<IndexType, TScalarT>>       &t,
            std::vector<std::tuple<IndexType, MScalarT>> const &m,
            bool                                                structure_flag,
            bool                                                complement_flag,
            SemiringT                                           semiring,
            AScalarT                                            a,
            std::vector<std::tuple<IndexType, BScalarT>> const &b)
        {
            GRB_LOG_FN_BEGIN("masked_axpy");

            if (m.empty() && complement_flag)
            {
                axpy(t, semiring, a, b);
                return;
            }

            auto t_it = t.begin();
            auto m_it = m.begin();

            for (auto const &b_elt : b)
            {
                IndexType    j(std::get<0>(b_elt));
                GRB_LOG_VERBOSE("j = " << j);

                // scan through M[i] to see if mask allows write.
                if (advance_and_check_mask_iterator(
                        m_it, m.end(), structure_flag, j) == complement_flag)
                {
                    GRB_LOG_VERBOSE("Skipping j = " << j);
                    continue;
                }

                BScalarT  b_j(std::get<1>(b_elt));

                auto t_j(semiring.mult(a, b_j));
                GRB_LOG_VERBOSE("temp = " << t_j);

                // scan through C_row to find insert/merge point
                if (advance_and_check_tuple_iterator(t_it, t.end(), j))
                {
                    GRB_LOG_VERBOSE("Accumulating");
                    std::get<1>(*t_it) = semiring.add(std::get<1>(*t_it), t_j);
                    ++t_it;
                }
                else
                {
                    GRB_LOG_VERBOSE("Inserting");
                    t_it = t.insert(t_it,
                                    std::make_tuple(j, static_cast<TScalarT>(t_j)));
                    ++t_it;
                }
            }
            GRB_LOG_FN_END("masked_axpy");
        }

        // *******************************************************************
        /// Perform the following operation on sparse vectors implemented as
        /// vector<tuple<Index, value>> (t assumed to be masked already)
        ///
        /// z = m ^ c (accum) t
        template<typename CScalarT,
                 typename AccumT,
                 typename MScalarT,
                 typename AScalarT,
                 typename BScalarT>
        void masked_accum(
            std::vector<std::tuple<IndexType, CScalarT>>       &z,
            std::vector<std::tuple<IndexType, MScalarT>> const &m,
            bool                                                structure_flag,
            bool                                                complement_flag,
            AccumT                                       const &accum,
            std::vector<std::tuple<IndexType, AScalarT>> const &c,
            std::vector<std::tuple<IndexType, BScalarT>> const &t)
        {
            GRB_LOG_FN_BEGIN("masked_accum.v2");
            auto t_it = t.begin();
            auto m_it = m.begin();
            auto c_it = c.begin();

            // for each element of c find out if it is not in mask
            while ((t_it != t.end()) && (c_it != c.end()))
            {
                IndexType t_idx(std::get<0>(*t_it));
                IndexType c_idx(std::get<0>(*c_it));
                if (t_idx < c_idx)
                {
                    // t already masked
                    z.emplace_back(t_idx,
                                   static_cast<CScalarT>(std::get<1>(*t_it)));
                    ++t_it;
                }
                else if (c_idx < t_idx)
                {
                    if (advance_and_check_mask_iterator(
                            m_it, m.end(), structure_flag, c_idx) != complement_flag)
                    {
                        z.emplace_back(c_idx,
                                       static_cast<CScalarT>(std::get<1>(*c_it)));
                    }
                    ++c_it;
                }
                else
                {
                    z.emplace_back(
                        t_idx,
                        static_cast<CScalarT>(accum(std::get<1>(*c_it),
                                                    std::get<1>(*t_it))));
                    ++t_it;
                    ++c_it;
                }
            }

            while (t_it != t.end())
            {
                z.emplace_back(std::get<0>(*t_it),
                               static_cast<CScalarT>(std::get<1>(*t_it)));
                ++t_it;
            }

            while (c_it != c.end())
            {
                IndexType c_idx(std::get<0>(*c_it));
                if (advance_and_check_mask_iterator(
                        m_it, m.end(), structure_flag, c_idx) != complement_flag)
                {
                    z.emplace_back(c_idx, std::get<1>(*c_it));
                }
                ++c_it;
            }
            GRB_LOG_FN_END("masked_accum.v2");
        }


        // *******************************************************************
        /// Perform the following operation on sparse vectors implemented as
        /// vector<tuple<Index, value>>
        ///
        /// if complement_flag == false:
        ///    c = (!m ^ ci) U z, where z = (m ^ t);  i.e., union assumes disjoint sets
        /// else:
        ///   c =   (m ^ ci) U z, where z = (!m ^ t)
        template<typename CScalarT,
                 typename MScalarT,
                 typename ZScalarT>
        void masked_merge(
            std::vector<std::tuple<IndexType, CScalarT>>       &c,
            std::vector<std::tuple<IndexType, MScalarT>> const &m,
            bool                                                structure_flag,
            bool                                                complement_flag,
            std::vector<std::tuple<IndexType, CScalarT>> const &ci,
            std::vector<std::tuple<IndexType, ZScalarT>> const &z)
        {
            GRB_LOG_FN_BEGIN("masked_merge.v2");
            auto m_it = m.begin();
            auto c_it = ci.begin();
            auto z_it = z.begin();

            c.clear();

            IndexType next_z;
            for (auto const &elt : z)
            {
                next_z = std::get<0>(elt);
                while (c_it != ci.end() && (std::get<0>(*c_it) < next_z))
                {
                    IndexType next_c(std::get<0>(*c_it));
                    if (advance_and_check_mask_iterator(
                            m_it, m.end(), structure_flag, next_c) == complement_flag)
                    {
                        c.emplace_back(next_c, std::get<1>(*c_it));
                    }
                    ++c_it;
                }
                c.emplace_back(next_z, static_cast<CScalarT>(std::get<1>(elt)));
            }


            while (c_it != ci.end() && (!z.empty() && (std::get<0>(*c_it) <= next_z)))
            {
                ++c_it;
            }

            while (c_it != ci.end())
            {
                IndexType next_c(std::get<0>(*c_it));
                if (advance_and_check_mask_iterator(
                        m_it, m.end(), structure_flag, next_c) == complement_flag)
                {
                    c.emplace_back(next_c, std::get<1>(*c_it));
                }
                ++c_it;
            }

            GRB_LOG_FN_END("masked_merge.v2");
        }

        //********************************************************************
        // For assign and extract
        //********************************************************************

        //********************************************************************
        // Index-out-of-bounds is an execution error and a responsibility of
        // the backend, but maybe it belongs in graphblas/detail.
        template <typename SequenceT>
        void check_index_array_content(SequenceT   const &array,
                                       IndexType          dim,
                                       std::string const &msg)
        {
            if (!IsAllSequence(array))
            {
                for (auto ix : array)
                {
                    if (ix >= dim)
                    {
                        throw IndexOutOfBoundsException(msg);
                    }
                }
            }
        }

        //********************************************************************
        // AllIndices SUPPORT
        // This is where we turns alls into the correct range

        template <typename SequenceT>
        SequenceT setupIndices(SequenceT seq, IndexType n)
        {
            return seq;
        }

        IndexSequenceRange setupIndices(AllIndices seq, IndexType n)
        {
            return IndexSequenceRange(0, n);
        }

    } // backend
} // grb
