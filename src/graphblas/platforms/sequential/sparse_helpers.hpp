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
        template <typename ScalarT>
        void print_vec(std::ostream &os, std::string label,
                       std::vector<std::tuple<IndexType, ScalarT> > vec)
        {
            os << label << " ";
            bool first = true;

            for (auto&& [idx, val] : vec)
            {
                os << (!first ? "," : " ") << idx << ":" << val;
                first = false;
            }
            os << std::endl;
        }

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
        /// A dot product of two sparse vectors (vectors<tuple(index,value)>)
        template <typename D1, typename D2, typename D3, typename SemiringT>
        bool dot_rev(D3                                                &ans,
                     std::vector<std::tuple<grb::IndexType,D1> > const &vec2,
                     std::vector<std::tuple<grb::IndexType,D2> > const &vec1,
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
                        ans = op.add(ans, op.mult(std::get<1>(*v2_it),
                                                  std::get<1>(*v1_it)));
                    }
                    else
                    {
                        ans = op.mult(std::get<1>(*v2_it),
                                      std::get<1>(*v1_it));
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
        /// A reduction of a sparse vector (vector<tuple(index,value)>) using a
        /// binary op or a monoid.
        template <typename D1, typename D3, typename BinaryOpT>
        bool reduction(
            D3                                                &ans,
            std::vector<std::tuple<grb::IndexType,D1> > const &vec,
            BinaryOpT                                          op)
        {
            if (vec.empty())
            {
                return false;
            }

            using D3ScalarType =
                decltype(op(std::declval<D1>(), std::declval<D1>()));
            D3ScalarType tmp;

            if (vec.size() == 1)
            {
                tmp = static_cast<D3ScalarType>(std::get<1>(vec[0]));
            }
            else
            {
                /// @note Since op is associative and commutative left to right
                /// ordering is not strictly required.
                tmp = op(std::get<1>(vec[0]), std::get<1>(vec[1]));

                /// @todo replace with call to std::reduce?
                for (size_t idx = 2; idx < vec.size(); ++idx)
                {
                    tmp = op(tmp, std::get<1>(vec[idx]));
                }
            }

            ans = static_cast<D3>(tmp);
            return true;
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

        //********************************************************************
        // ALL SUPPORT
        // This is where we turn alls into the correct range

        template <typename SequenceT>
        bool searchIndices(SequenceT seq, IndexType n)
        {
            for (auto it : seq)
            {
                if (it == n) return true;
            }
            return false;
        }

        bool searchIndices(AllIndices seq, IndexType n)
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
            ewise_or(z, w.getContents(), t, accum);
        }

        //**********************************************************************
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
            ewise_or_stencil(z, w.getContents(), t, indices);
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
        void ewise_or_opt_accum(ZMatrixT               &Z,
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
            ewise_or(z, w.getContents(), t, accum);
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

        //**********************************************************************
        template <typename ZScalarT,
                  typename WScalarT,
                  typename TScalarT,
                  typename BinaryOpT>
        void opt_accum_scalar(ZScalarT       &z,
                              WScalarT const &w,
                              TScalarT const &t,
                              BinaryOpT       accum)
        {
            z = static_cast<ZScalarT>(accum(w, t));
        }

        //**********************************************************************
        // Specialized version that gets used when we don't have an accumulator
        template <typename ZScalarT,
                  typename WScalarT,
                  typename TScalarT>
        void opt_accum_scalar(ZScalarT                &z,
                              WScalarT          const &w,
                              TScalarT          const &t,
                              grb::NoAccumulate        accum)
        {
            z = static_cast<ZScalarT>(t);
        }

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

            // loop through both ordered sets to compute ewise_or
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
         *            {(i,j,Zij):(i,j) \in (ind(Z) \cap ind(\neg M))} \f]
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
                else
                {
                    ++it;
                }
            }

            return mask_tuples;
        }

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
        // Vector version
        template <typename WVectorT,
                  typename ZScalarT,
                  typename MaskT>
        void write_with_opt_mask_1D(
            WVectorT                                           &w,
            std::vector<std::tuple<IndexType, ZScalarT>> const &z,
            MaskT const                                        &mask,
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
            WVectorT                                               &w,
            std::vector<std::tuple<IndexType, ZScalarT>>     const &z,
            grb::VectorStructuralComplementView<MaskT>       const &mask,
            OutputControlEnum                                       outp)
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

        //********************************************************************
        // Index-out-of-bounds is an execution error and a responsibility of
        // the backend.
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
        // ALL SUPPORT
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
        /// @todo Need to add support for STRUCTURE_ONLY
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
        /// c += a_ik*b[:]
        template<typename CScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        void axpy(
            std::vector<std::tuple<IndexType, CScalarT>>       &c,
            SemiringT                                           semiring,
            AScalarT                                            a,
            std::vector<std::tuple<IndexType, BScalarT>> const &b)
        {
            GRB_LOG_FN_BEGIN("axpy");
            auto c_it = c.begin();

            for (auto&& [j, b_j] : b)
            {
                GRB_LOG_VERBOSE("j = " << j);

                auto t_j(semiring.mult(a, b_j));
                GRB_LOG_VERBOSE("temp = " << t_j);

                // scan through C_row to find insert/merge point
                if (advance_and_check_tuple_iterator(c_it, c.end(), j))
                {
                    GRB_LOG_VERBOSE("Accumulating");
                    std::get<1>(*c_it) = semiring.add(std::get<1>(*c_it), t_j);
                    ++c_it;
                }
                else
                {
                    GRB_LOG_VERBOSE("Inserting");
                    c_it = c.insert(c_it,
                                    std::make_tuple(j, static_cast<CScalarT>(t_j)));
                    ++c_it;
                }
            }
            GRB_LOG_FN_END("axpy");
        }

        // *******************************************************************
        /// perform the following operation on sparse vectors implemented as
        /// vector<tuple<Index, value>>
        ///
        /// c<[m[:]]> += a_ik*b[:]
        template<typename CScalarT,
                 typename MScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        void masked_axpy(
            std::vector<std::tuple<IndexType, CScalarT>>       &c,
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
                axpy(c, semiring, a, b);
                return;
            }

            auto c_it = c.begin();
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
                if (advance_and_check_tuple_iterator(c_it, c.end(), j))
                {
                    GRB_LOG_VERBOSE("Accumulating");
                    std::get<1>(*c_it) = semiring.add(std::get<1>(*c_it), t_j);
                    ++c_it;
                }
                else
                {
                    GRB_LOG_VERBOSE("Inserting");
                    c_it = c.insert(c_it,
                                    std::make_tuple(j, static_cast<CScalarT>(t_j)));
                    ++c_it;
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

    } // backend
} // grb
