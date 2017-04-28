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
 * Implementations of all GraphBLAS functions optimized for the sequential
 * (CPU) backend.
 */

#ifndef GB_SEQUENTIAL_SPARSE_HELPERS_HPP
#define GB_SEQUENTIAL_SPARSE_HELPERS_HPP

#pragma once

#include <functional>
#include <utility>
#include <vector>
#include <iterator>
#include <iostream>
#include <string>
#include <graphblas/accum.hpp>
#include <graphblas/algebra.hpp>

//****************************************************************************

namespace GraphBLAS
{
    namespace backend
    {
        //**********************************************************************

        template <typename DstMatrixT,
                  typename SrcMatrixT>
        void sparse_copy(DstMatrixT &dstMatrix,
                         SrcMatrixT const &srcMatrix)
        {
            typedef typename SrcMatrixT::ScalarType SrcScalarType;
            typedef typename DstMatrixT::ScalarType DstScalarType;

            typedef std::vector<std::tuple<IndexType, DstScalarType> > RowType;

            IndexType nrows(dstMatrix.nrows());
            RowType tmp_row;
            for (IndexType row_idx = 0; row_idx < nrows; ++row_idx)
            {
                tmp_row = srcMatrix.getRow(row_idx);
                dstMatrix.setRow(row_idx, tmp_row);
            }
        }

        //**********************************************************************

        /// Increments the provided iterate while the value is less than the provided index
        template <typename I>
        void increment_until_true(
                // std::vector<std::tuple<GraphBLAS::IndexType,D> >::const_iterator    &iter,
                // std::vector<std::tuple<GraphBLAS::IndexType,D> >::const_iterator    &iter_end,
                I          &iter,
                I const    &iter_end)
        {
            using value_type = typename std::iterator_traits<I>::value_type;
            using data_type = typename std::tuple_element<1, value_type>::type;

            GraphBLAS::IndexType tmp_idx;
            data_type tmp_val;

            if (iter == iter_end)
                return;

            std::tie(tmp_idx, tmp_val) = *iter;
            while (!tmp_val && iter != iter_end)
            {
                //std::cout << "Iter not true. index: " + std::to_string(tmp_idx) +
                //    " incrementing."  << std::endl;;
                ++iter;
                if (iter == iter_end)
                    return;

                // Reload
                std::tie(tmp_idx, tmp_val) = *iter;
            }
        }

        //**********************************************************************

        /// Increments the provided iterate while the value is less than the provided index
        template <typename I>
        void increment_while_below(
                // std::vector<std::tuple<GraphBLAS::IndexType,D> >::const_iterator    &iter,
                // std::vector<std::tuple<GraphBLAS::IndexType,D> >::const_iterator    &iter_end,
                I         &iter,
                I const   &iter_end,
                GraphBLAS::IndexType                                                 idx)
        {
            using value_type = typename std::iterator_traits<I>::value_type;
            using data_type = typename std::tuple_element<1, value_type>::type;

            GraphBLAS::IndexType tmp_idx;
            data_type tmp_val;

            if (iter == iter_end)
                return;

            std::tie(tmp_idx, tmp_val) = *iter;
            while (tmp_idx < idx && iter != iter_end)
            {
                //std::cerr << "Iter at: " + std::to_string(tmp_idx) +  ", below index: " +
                //    std::to_string(idx) + " incrementing."  << std::endl;;
                ++iter;
                if (iter == iter_end)
                    return;

                // Reload
                std::tie(tmp_idx, tmp_val) = *iter;
            }
        }

        //**********************************************************************
        template <typename I, typename V>
        void increment_and_add_while_below(
                I                              &iter,
                I                       const  &iter_end,
                GraphBLAS::IndexType            idx,
                V                               &vec)
        {
            // @todo: Tighten up the template types above to bind the vector
            // type to the iter type explicitly
            using value_type = typename std::iterator_traits<I>::value_type;
            using data_type = typename std::tuple_element<1, value_type>::type;

            GraphBLAS::IndexType tmp_idx;
            data_type tmp_val;

            if (iter == iter_end)
                return;

            std::tie(tmp_idx, tmp_val) = *iter;
            while (iter != iter_end && tmp_idx < idx )
            {
                vec.push_back(*iter);

                ++iter;
                if (iter == iter_end)
                    return;

                // Reload
                std::tie(tmp_idx, tmp_val) = *iter;
            }
        }

        //**********************************************************************

        /// Perform the dot product of a row of a matrix with a sparse vector without
        /// pulling the indices out of the vector first.
        template <typename D1, typename D2, typename D3, typename SemiringT>
        bool dot2(D3                                                      &ans,
                  std::vector<std::tuple<GraphBLAS::IndexType,D1> > const &A_row,
                  std::vector<bool>                                 const &u_bitmap,
                  std::vector<D2>                                   const &u_vals,
                  GraphBLAS::IndexType                                     u_nvals,
                  SemiringT                                                op)
        {
            bool value_set(false);
            ans = op.zero();

            if ((u_nvals == 0) || A_row.empty())
            {
                return value_set;
            }

            // find first stored value in u
            GraphBLAS::IndexType u_idx(0);
            while (!u_bitmap[u_idx]) ++u_idx; // skip unstored elements

            // pull first value out of the row
            auto A_iter = A_row.begin();
            D1 a_val;
            GraphBLAS::IndexType a_idx;

            // loop through both ordered sets to compute sparse dot prod
            while ((A_iter != A_row.end()) && (u_idx < u_vals.size()))
            {
                std::tie(a_idx, a_val) = *A_iter;
                //std::cerr << "Examine u index = " << u_idx << "," << u_vals[u_idx]
                //          << ", A col_idx = " << a_idx << "," << a_val << std::endl;

                if (u_idx == a_idx)
                {
                    //std::cerr << ans << " + "  << a_val << " * "  << u_vals[u_idx]
                    //          << " = " << op.mult(a_val, u_vals[u_idx]) << std::endl;

                    ans = op.add(ans, op.mult(a_val, u_vals[u_idx]));
                    value_set = true;

                    //std::cerr << "Equal, mutliply_accum, ans = " << ans << std::endl;

                    do { ++u_idx; } while ((u_idx < u_vals.size()) && !u_bitmap[u_idx]);
                    ++A_iter;
                }
                else if (u_idx > a_idx)
                {
                    //std::cerr << "Advancing A_iter" << std::endl;
                    ++A_iter;
                }
                else
                {
                    //std::cerr << "Advancing u_iter" << std::endl;
                    do { ++u_idx; } while ((u_idx < u_vals.size()) && !u_bitmap[u_idx]);
                }
            }

            return value_set;
        }

        //************************************************************************
        /// A dot product of two sparse vectors (vectors<tuple(index,value)>)
        template <typename D1, typename D2, typename D3, typename SemiringT>
        bool dot(D3                                                      &ans,
                 std::vector<std::tuple<GraphBLAS::IndexType,D1> > const &vec1,
                 std::vector<std::tuple<GraphBLAS::IndexType,D2> > const &vec2,
                 SemiringT                                                op)
        {
            bool value_set(false);
            ans = op.zero();

            if (vec2.empty() || vec1.empty())
            {
                return value_set;
            }

            auto v1_it = vec1.begin();
            auto v2_it = vec2.begin();

            // pull first value out of the row
            D1 a_val;
            D2 u_val;
            GraphBLAS::IndexType a_idx, u_idx;

            // loop through both ordered sets to compute sparse dot prod
            while ((v1_it != vec1.end()) &&
                   (v2_it != vec2.end()))
            {
                std::tie(a_idx, a_val) = *v1_it;
                std::tie(u_idx, u_val) = *v2_it;

                //std::cerr << "Examine u idx,val = " << u_idx << "," << u_val
                //          << "; A col_idx,val = " << a_idx << "," << a_val << std::endl;

                if (u_idx == a_idx)
                {
                    //std::cerr << ans << " + " << a_val << " * " << u_val << " = ";
                    ans = op.add(ans, op.mult(a_val, u_val));
                    value_set = true;
                    //std::cerr << ans << std::endl;

                    ++v2_it;
                    ++v1_it;
                }
                else if (u_idx > a_idx)
                {
                    //std::cerr << "Advancing v1_it" << std::endl;
                    ++v1_it;
                }
                else
                {
                    //std::cerr << "Advancing v2_it" << std::endl;
                    ++v2_it;
                }
            }

            return value_set;
        }

        //**********************************************************************
        /// Apply element-wise operation to union on sparse vectors.
        // @todo: What is the bool return type for?
        template <typename D1, typename D2, typename D3, typename BinaryOpT>
        void ewise_or(std::vector<std::tuple<GraphBLAS::IndexType,D3> >       &ans,
                      std::vector<std::tuple<GraphBLAS::IndexType,D1> > const &vec1,
                      std::vector<std::tuple<GraphBLAS::IndexType,D2> > const &vec2,
                      BinaryOpT                                                op)
        {
            ans.clear();
            auto v1_it = vec1.begin();
            auto v2_it = vec2.begin();

            D1 v1_val;
            D2 v2_val;
            GraphBLAS::IndexType v1_idx, v2_idx;

            // loop through both ordered sets to compute ewise_or
            while ((v1_it != vec1.end()) || (v2_it != vec2.end()))
            {
                if ((v1_it != vec1.end()) && (v2_it != vec2.end()))
                {
                    std::tie(v1_idx, v1_val) = *v1_it;
                    std::tie(v2_idx, v2_val) = *v2_it;

                    if (v2_idx == v1_idx)
                    {
                        //std::cerr << ans << " + " << v1_val << " * " << v2_val << " = ";
                        ans.push_back(std::make_tuple(v1_idx,
                                                      static_cast<D3>(op(v1_val, v2_val))));
                        //std::cerr << ans << std::endl;

                        ++v2_it;
                        ++v1_it;
                    }
                    else if (v2_idx > v1_idx)
                    {
                        //std::cerr << "Copying v1, Advancing v1_it" << std::endl;
                        // @todo: Should these static_cast??
                        ans.push_back(std::make_tuple(v1_idx, v1_val));
                        ++v1_it;
                    }
                    else
                    {
                        //std::cerr << "Copying v2, Advancing v2_it" << std::endl;
                        // @todo: Should these static_cast??
                        ans.push_back(std::make_tuple(v2_idx, v2_val));
                        ++v2_it;
                    }
                }
                else if (v1_it != vec1.end())
                {
                    std::tie(v1_idx, v1_val) = *v1_it;
                    ans.push_back(std::make_tuple(v1_idx, static_cast<D3>(v1_val)));
                    ++v1_it;
                }
                else // v2_it != vec2.end())
                {
                    std::tie(v2_idx, v2_val) = *v2_it;
                    ans.push_back(std::make_tuple(v2_idx, static_cast<D3>(v2_val)));
                    ++v2_it;
                }
            }
        }

        //**********************************************************************
        template < typename CMatrixT,
                   typename ZMatrixT,
                   typename TMatrixT,
                   typename BinaryOpT >
        void ewise_or_opt_accum(ZMatrixT         &Z,
                                CMatrixT const   &C,
                                TMatrixT const   &T,
                                BinaryOpT        accum)
        {
            typedef typename ZMatrixT::ScalarType ZScalarType;

            typedef std::vector<std::tuple<IndexType,ZScalarType> > ZRowType;

            ZRowType tmp_row;
            IndexType nRows(Z.nrows());
            for (IndexType row_idx = 0; row_idx < nRows; ++row_idx)
            {
                ewise_or(tmp_row, C.getRow(row_idx), T.getRow(row_idx), accum);
                Z.setRow(row_idx, tmp_row);
            }
        }

        //**********************************************************************

        // Specialized version that gets used when we don't have an accumulator
        template < typename CMatrixT,
                typename ZMatrixT,
                typename TMatrixT>
        void ewise_or_opt_accum(ZMatrixT                    &Z,
                                CMatrixT const              &C,
                                TMatrixT const              &T,
                                GraphBLAS::NoAccumulate )
        {
            sparse_copy(Z, T);
        }

        //************************************************************************

        // @todo:  Remove! We won't use this when we switch to separate op/mask steps
        // @deprecated
        template <typename D1, typename D2, typename D3, typename M, typename BinaryOpT>
        void ewise_or_mask(std::vector<std::tuple<GraphBLAS::IndexType,D3> >       &ans,
                           std::vector<std::tuple<GraphBLAS::IndexType,D1> > const &vec1,
                           std::vector<std::tuple<GraphBLAS::IndexType,D2> > const &vec2,
                           std::vector<std::tuple<GraphBLAS::IndexType,M> > const  &mask,
                           BinaryOpT                                                op,
                           bool                                                     replace)
        {
            // DESIGN:
            // This algo is driven by the mask, so we move to a valid mask entry, then
            // "catch up" the other iterators.  If they match the current valid mask
            // then we process them as with the other ewise option.  Rinse, repeat.
            ans.clear();

            //std::cerr << "" << std::endl;
            //std::cerr << "Starting ewise_or_mask on row" << std::endl;
            if (mask.empty())
            {
                //std::cerr << "Mask exhausted(0)." << std::endl;
                return;
            }

            auto v1_it = vec1.begin();
            auto v2_it = vec2.begin();
            auto mask_it = mask.begin();

            D1 v1_val;
            D2 v2_val;
            M  mask_val;
            GraphBLAS::IndexType v1_idx, v2_idx, mask_idx;

            // Walk the mask.
            while (mask_it != mask.end())
            {
                // Make sure the mask is on a valid value
                increment_until_true(mask_it, mask.end());

                // If we run out of mask, we are done!
                if (mask_it == mask.end())
                {
                    //std::cerr << "Mask exhausted(1)." << std::endl;
                    return;
                }

                std::tie(mask_idx, mask_val) = *mask_it;

                // Increment V1 while less than mask
                increment_while_below(v1_it, vec1.end(), mask_idx);

                // Increment V2 while less than mask
                increment_while_below(v2_it, vec2.end(), mask_idx);

                // If any of the input vectors match, put their values in the output vectors
                // invoking the supplied binary operator if we have both values, otherwise
                // just put in that value.
                if ((v1_it != vec1.end()) && (v2_it != vec2.end()))
                {
                    std::tie(v1_idx, v1_val) = *v1_it;
                    std::tie(v2_idx, v2_val) = *v2_it;

                    if (v1_idx == v2_idx && v1_idx == mask_idx)
                    {
                        //std::cerr << "Accum: " << std::to_string(v1_val) << " op " <<
                        //    std::to_string(v2_val) << std::endl;
                        ans.push_back(std::make_tuple(mask_idx,
                                                      static_cast<D3>(op(v1_val, v2_val))));
                    }
                    else if (v1_idx == mask_idx)
                    {
                        if (!replace)
                        {
                            ans.push_back(std::make_tuple(mask_idx, static_cast<D3>(v1_val)));
                            //std::cerr << "Copying v1. val: " << std::to_string(v1_val) << std::endl;
                        }
                    }
                    else if (v2_idx == mask_idx)
                    {
                        ans.push_back(std::make_tuple(mask_idx, static_cast<D3>(v2_val)));
                        //std::cerr << "Copying v2. val: " <<  std::to_string(v2_val) << std::endl;
                    }
                }
                else if (v1_it != vec1.end())
                {
                    std::tie(v1_idx, v1_val) = *v1_it;
                    if (v1_idx == mask_idx)
                    {
                        if (!replace)
                        {
                            ans.push_back(std::make_tuple(mask_idx, static_cast<D3>(v1_val)));
                            //std::cerr << "Copying v1. val: " << std::to_string(v1_val) << std::endl;
                        }
                    }
                }
                else if (v2_it != vec2.end())
                {
                    std::tie(v2_idx, v2_val) = *v2_it;
                    if (v2_idx == mask_idx)
                    {
                        ans.push_back(std::make_tuple(mask_idx, static_cast<D3>(v2_val)));
                        //std::cerr << "Copying v2. val: " <<  std::to_string(v2_val) << std::endl;
                    }
                }
                else
                {
                    // We have more mask, but no more other vec
                    //std::cerr << "Inputs (not mask) exhausted." << std::endl;
                    return;
                }

                // Now move to the next mask entry.
                ++mask_it;

            } // while mask_it != end

            //std::cerr << "Mask exhausted(2)." << std::endl;
        }

        //************************************************************************
        /// Apply element-wise operation to intersection of sparse vectors.
        template <typename D1, typename D2, typename D3, typename BinaryOpT>
        void ewise_and(std::vector<std::tuple<GraphBLAS::IndexType,D3> >       &ans,
                       std::vector<std::tuple<GraphBLAS::IndexType,D1> > const &vec1,
                       std::vector<std::tuple<GraphBLAS::IndexType,D2> > const &vec2,
                       BinaryOpT                                                op)
        {
            ans.clear();
            auto v1_it = vec1.begin();
            auto v2_it = vec2.begin();

            D1 v1_val;
            D2 v2_val;
            GraphBLAS::IndexType v1_idx, v2_idx;

            // loop through both ordered sets to compute ewise_or
            while ((v1_it != vec1.end()) && (v2_it != vec2.end()))
            {
                std::tie(v1_idx, v1_val) = *v1_it;
                std::tie(v2_idx, v2_val) = *v2_it;

                if (v2_idx == v1_idx)
                {
                    //std::cerr << ans << " + " << v1_val << " * " << v2_val << " = ";
                    ans.push_back(std::make_tuple(v1_idx,
                                                  static_cast<D3>(op(v1_val, v2_val))));
                    //std::cerr << ans << std::endl;

                    ++v2_it;
                    ++v1_it;
                }
                else if (v2_idx > v1_idx)
                {
                    //std::cerr << "Advancing v1_it" << std::endl;
                    ++v1_it;
                }
                else
                {
                    //std::cerr << "Advancing v2_it" << std::endl;
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
         * If replace:
         *
         * L(C) = {(i,j,Zij):(i,j) \in (ind(Z) \cap ind(M))}
         *
         * If NOT replace:
         *
         * L(C) = {(i,j,Zij):(i,j) \in (ind(C) \cap int(\not M))} \cup
         *        {(i,j,Zij):(i,j) \in (ind(Z) \cap int(\not M))}
         *
         * @tparam CScalarT The scalar type of the C vector input AND result.
         * @tparam ZScalarT The scalar type of the Z vector input.
         * @tparam MScalarT The scalar type of the mask vector.
         * @param result Result vector.  We clear this first.
         * @param c_vec The original c values that may be carried through.
         * @param z_vec The new values to insert/overlay.
         * @param mask_vec The mask which specifies which values to use.
         * @param replace If true, we should always clear the values specified
         *                by the mask regardless if they are overlayed.
         */
        template < typename CScalarT,
                   typename ZScalarT,
                   typename MScalarT>
        void apply_with_mask(std::vector<std::tuple<IndexType, CScalarT> >          &result,
                             std::vector<std::tuple<IndexType, CScalarT> > const    &c_vec,
                             std::vector<std::tuple<IndexType, ZScalarT> > const    &z_vec,
                             std::vector<std::tuple<IndexType, MScalarT> > const    &mask_vec,
                             bool                                                    replace)
        {
            auto c_it = c_vec.begin();
            auto z_it = z_vec.begin();
            auto mask_it = mask_vec.begin();

            CScalarT c_val;
            ZScalarT z_val;
            MScalarT mask_val;
            GraphBLAS::IndexType c_idx, z_idx, mask_idx;

            result.clear();

            // Design: This approach is driven by the mask.
            while (mask_it != mask_vec.end())
            {
                // Make sure the mask is on a valid value
                increment_until_true(mask_it, mask_vec.end());

                // If we run out of mask, we are done!
                if (mask_it == mask_vec.end())
                {
                    //std::cerr << "Mask exhausted(1)." << std::endl;
                    return;
                }

                // Get the mask values
                std::tie(mask_idx, mask_val) = *mask_it;

                // If replace then we don't consider original C values.
                // If not replace, then we want to keep C values outide the mask
                // Any values in C "outside" the mask should now be applied
                // So, we catch "c" up to the mask.  This is the intersection
                // of C and !M.

                if (!replace)
                {
                    // This is the part of (ind(C) \cap int(\not M)
                    increment_and_add_while_below(c_it, c_vec.end(), mask_idx,
                                                  result);
                }

                // "Catch up" the input to the mask.
                increment_while_below(z_it, z_vec.end(), mask_idx);

                // Now, at the mask point add the value from Z if we have one.
                if (z_it != z_vec.end())
                {
                    std::tie(z_idx, z_val) = *z_it;
                    if (z_idx == mask_idx)
                    {
                        result.push_back(std::make_tuple(mask_idx, static_cast<CScalarT>(z_val)));
                        //std::cerr << "Copying v1. val: " << std::to_string(v1_val) << std::endl;
                    }
                }

                // If there is a C here, skip it
                if (c_it != c_vec.end())
                {
                    std::tie(c_idx, c_val) = *c_it;
                    if (c_idx == mask_idx)
                        ++c_it;
                }

                // Now move to the next mask entry.
                ++mask_it;
            } // while mask_it != end

            // Now, we need to add the remaining C values beyond the mask
            if (!replace)
            {
                // This is the part of (ind(C) \cap int(\not M)
                while (c_it != c_vec.end())
                {
                    result.push_back(*c_it);
                    ++c_it;
                }
            }
        } // apply_with_mask

        //**********************************************************************
        // Matrix version

        template < typename CMatrixT,
                   typename ZMatrixT,
                   typename MMatrixT>
        void write_with_opt_mask(CMatrixT           &C,
                                 ZMatrixT   const   &Z,
                                 MMatrixT   const   &mask,
                                 bool               replace)
        {
            typedef typename CMatrixT::ScalarType CScalarType;
            typedef typename ZMatrixT::ScalarType ZScalarType;
            typedef typename MMatrixT::ScalarType MScalarType;

            typedef std::vector<std::tuple<IndexType, CScalarType> > CRowType;
            typedef std::vector<std::tuple<IndexType, ZScalarType> > ZRowType;
            typedef std::vector<std::tuple<IndexType, MScalarType> > MRowType;

            CRowType tmp_row;
            IndexType nRows(C.nrows());
            for (IndexType row_idx = 0; row_idx < nRows; ++row_idx)
            {
                apply_with_mask(tmp_row, C.getRow(row_idx), Z.getRow(row_idx),
                                mask.getRow(row_idx), replace);

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
                                 backend::NoMask    const   &foo,
                                 bool                       replace)
        {
            sparse_copy(C, Z);
        }

        //**********************************************************************
        /// Simple check to make sure the matricies are the same size.  Prints
        /// out useful error messages.

        template <typename M1, typename M2>
        void check_dimensions(M1 m1, std::string m1Name, M2 m2, std::string m2Name)
        {
            if (m1.nrows() != m2.nrows())
            {
                throw DimensionException("Matrix ROW counts are not the same. " +
                                         m1Name + " = " + std::to_string(m1.nrows()) + ", " +
                                         m2Name + " = " + std::to_string(m2.nrows()) );
            }

            if (m1.ncols() != m2.ncols())
            {
                throw DimensionException("Matrix COL counts are not the same. " +
                                         m1Name + " = " + std::to_string(m1.ncols()) + ", " +
                                         m2Name + " = " + std::to_string(m2.ncols()) );
            }
        }

        template <typename M1, typename M2>
        void check_inside_dimensions(M1 m1, std::string m1Name, M2 m2, std::string m2Name)
        {
            if (m1.ncols() != m2.nrows())
            {
                throw DimensionException("Matrix COL vs ROW counts (inner) are not the same. " +
                                         m1Name + " col = " + std::to_string(m1.ncols()) + ", " +
                                         m2Name + " row = " + std::to_string(m2.nrows()) );
            }
        }

        template <typename M1, typename M2, typename M3>
        void check_outside_dimensions(M1 m1, std::string m1Name, M2 m2, std::string m2Name, M3 m3, std::string m3Name)
        {
            if (m1.nrows() != m3.nrows())
            {
                throw DimensionException("Matrix ROW vs ROW counts (outer) are not the same. " +
                                         m1Name + " row = " + std::to_string(m1.nrows()) + ", " +
                                         m3Name + " row = " + std::to_string(m3.nrows()) );
            }

            if (m2.ncols() != m3.ncols())
            {
                throw DimensionException("Matrix COL vs COL counts (outer) are not the same. " +
                                         m2Name + " col = " + std::to_string(m2.ncols()) + ", " +
                                         m3Name + " col = " + std::to_string(m3.ncols()) );
            }
        }

    } // backend
} // GraphBLAS

#endif
