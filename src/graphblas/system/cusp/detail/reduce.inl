/*
 * Copyright (c) 2015 Carnegie Mellon University and The Trustees of Indiana
 * University.
 * All Rights Reserved.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS," WITH NO WARRANTIES WHATSOEVER. CARNEGIE
 * MELLON UNIVERSITY AND THE TRUSTEES OF INDIANA UNIVERSITY EXPRESSLY DISCLAIM
 * TO THE FULLEST EXTENT PERMITTED BY LAW ALL EXPRESS, IMPLIED, AND STATUTORY
 * WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT OF PROPRIETARY RIGHTS.
 *
 * This Program is distributed under a BSD license.  Please see LICENSE file or
 * permission@sei.cmu.edu for more information.  DM-0002659
 */

#pragma once

#include "merge.inl"

namespace graphblas
{
namespace backend
{
    namespace detail{

        template <typename AccumT>
        struct tsl_zero
        {
            AccumT accum;
            tsl_zero(AccumT a) : accum(a) {}
            template <typename T1, typename T2>
            __host__ __device__
            T1 operator()(const T1& t1, const T2 t2){
                auto t = thrust::get<1>(t2);
                return t==0 ? t1 :
                    accum(t1, static_cast<T1>(thrust::get<0>(t2)));
            }
        };
    }
    /**
     * @brief Apply a reduction operation to each row of a matrix.
     *
     * @tparam <MatrixT>  Models the Matrix concept
     * @tparam <MonoidT>  Models a binary function
     * @tparam <AccumT>   Models a binary function
     *
     * @param[in]  a  The matrix to perform the row reduction on.
     * @param[out] c  The matrix (column vector) to assign the result to.
     * @param[in]  m  The monoid (binary op) used to reduce the row elements
     * @param[in]  accum   The function for how to assign to destination (e.g.
     *                     Accum to "add" to destination, or Assign to
     *                     "replace" the destimation.
     *
     * @throw DimensionException  If the rows a and c do not match, or the
     *                            the number of columns in c is not one.
     */
    template<typename AMatrixT,
             typename CMatrixT,
             typename MonoidT,
             typename AccumT>
    inline void row_reduce(AMatrixT const &a,
                           CMatrixT       &c, // vector?
                           MonoidT         m,
                           AccumT          accum)
    {

        //assuming cmatrix is of type coo matrix
        CMatrixT temp(a.num_rows, 1, a.num_rows);
        auto reduced = thrust::reduce_by_key(
            a.row_indices.begin(),
            a.row_indices.end(),
            a.values.begin(),
            temp.row_indices.begin(),
            temp.values.begin(),
            thrust::equal_to< IndexType >(),
            m);
        //write col indices:
        thrust::copy_n(
                thrust::make_constant_iterator(0),
                a.num_rows,
                temp.column_indices.begin());

        temp.resize(temp.num_rows, temp.num_cols, thrust::distance(temp.values.begin(), reduced.second));
        detail::merge(temp, c, accum);
    }

    /**
     * @brief Apply a reduction operation to each column of a matrix.
     *
     * @tparam <MatrixT>  Models the Matrix concept
     * @tparam <MonoidT>  Models a binary function
     * @tparam <AccumT>   Models a binary function
     *
     * @param[in]  a  The matrix to perform the column reduction on.
     * @param[out] c  The matrix (row vector) to assign the result to.
     * @param[in]  m  The monoid (binary op) used to reduce the column elements
     * @param[in]  accum   The function for how to assign to destination (e.g.
     *                     Accum to "add" to destination, or Assign to
     *                     "replace" the destimation.
     *
     * @throw DimensionException  If the columns a and c do not match, or the
     *                            the number of rows in c is not one.
     */
    template<typename AMatrixT,
             typename CMatrixT,
             typename MonoidT,
             typename AccumT>
    inline void col_reduce(AMatrixT const &a,
                           CMatrixT       &c, // vector?
                           MonoidT         m,
                           AccumT          accum)
    {
        AMatrixT temp1(a);
        CMatrixT temp2(1, a.num_cols, a.num_entries);
        //sort by column:
        thrust::sort_by_key(
                temp1.column_indices.begin(),
                temp1.column_indices.end(),
                thrust::make_zip_iterator(thrust::make_tuple(temp1.row_indices.begin(), temp1.values.begin())));

        auto reduced = thrust::reduce_by_key(
            temp1.column_indices.begin(),
            temp1.column_indices.end(),
            temp1.values.begin(),
            temp2.column_indices.begin(),
            temp2.values.begin(),
            thrust::equal_to< IndexType >(),
            m);
        //write row indices:
        thrust::copy_n(
                thrust::make_constant_iterator(0),
                a.num_cols,
                temp2.row_indices.begin());


        temp2.resize(temp2.num_rows, temp2.num_cols, thrust::distance(temp2.values.begin(), reduced.second));
        detail::merge(temp2, c, accum);
    } 

    template<typename AMatrixT,
             typename CMatrixT,
             typename MMatrixT,
             typename MonoidT =
                 graphblas::PlusMonoid<typename AMatrixT::ScalarType>,
             typename AccumT =
                 graphblas::math::Assign<typename AMatrixT::ScalarType> >
    inline void rowReduceMasked(AMatrixT const &a,
                           CMatrixT       &c, // vector?
                           MMatrixT       &mask,
                           MonoidT         sum     = MonoidT(),
                           AccumT          accum = AccumT())
    {
        //assuming cmatrix is of type coo matrix
        CMatrixT temp(a.num_rows, 1, a.num_rows);
        auto reduced = thrust::reduce_by_key(
            a.row_indices.begin(),
            a.row_indices.end(),
            a.values.begin(),
            temp.row_indices.begin(),
            temp.values.begin(),
            thrust::equal_to< IndexType >(),
            sum);
        //write col indices:
        thrust::copy_n(
                thrust::make_constant_iterator(0),
                a.num_rows,
                temp.column_indices.begin());

        temp.resize(temp.num_rows, temp.num_cols, thrust::distance(temp.values.begin(), reduced.second));
        //annihilate value with mask:
        //mask is assumed to be binary (1,0) where 1=keep and 0=discard
        //otherwise a switchable zero value in mask will invalidate the following code, and
        //will make the complexity a lot higher
        //c and temp have to be a dense vector by definition, can be optimized.
        //assert this fact:
        if (mask.num_rows != mask.num_entries || temp.num_entries != temp.num_rows || c.num_entries != c.num_rows){
            throw graphblas::DimensionException("vector or mask not dense! aborting::backend::reduce.inl:183");
        }

        detail::tsl_zero<AccumT> tsl(accum);

        thrust::transform(c.values.begin(), c.values.end(),
                thrust::make_zip_iterator(thrust::make_tuple(
                        temp.values.begin(),
                        mask.values.begin())),
                c.values.begin(),
                tsl);
    }

    template<typename AMatrixT,
             typename CMatrixT,
             typename MMatrixT,
             typename MonoidT =
                 graphblas::PlusMonoid<typename AMatrixT::ScalarType>,
             typename AccumT =
                 graphblas::math::Assign<typename AMatrixT::ScalarType> >
    inline void colReduceMasked(AMatrixT const &a,
                           CMatrixT       &c, // vector?
                           MMatrixT       &mask,
                           MonoidT         sum     = MonoidT(),
                           AccumT          accum = AccumT())
    {
        AMatrixT temp1(a);
        CMatrixT temp2(1, a.num_cols, a.num_entries);
        //sort by column:
        thrust::sort_by_key(
                temp1.column_indices.begin(),
                temp1.column_indices.end(),
                thrust::make_zip_iterator(thrust::make_tuple(temp1.row_indices.begin(), temp1.values.begin())));

        auto reduced = thrust::reduce_by_key(
            temp1.column_indices.begin(),
            temp1.column_indices.end(),
            temp1.values.begin(),
            temp2.column_indices.begin(),
            temp2.values.begin(),
            thrust::equal_to< IndexType >(),
            sum);
        //write row indices:
        thrust::copy_n(
                thrust::make_constant_iterator(0),
                a.num_cols,
                temp2.row_indices.begin());


        temp2.resize(temp2.num_rows, temp2.num_cols, thrust::distance(temp2.values.begin(), reduced.second));
        //annihilate value with mask:
        //mask is assumed to be binary (1,0) where 1=keep and 0=discard
        //otherwise a switchable zero value in mask will invalidate the following code, and
        //will make the complexity a lot higher
        //c and temp have to be a dense vector by definition, can be optimized.
        //assert this fact:
        if (mask.num_cols != mask.num_entries || temp2.num_entries != temp2.num_cols || c.num_entries != c.num_cols){
            throw graphblas::DimensionException("vector or mask not dense! aborting::backend::reduce.inl:260");
        }

        detail::tsl_zero<AccumT> tsl(accum);

        thrust::transform(c.values.begin(), c.values.end(),
                thrust::make_zip_iterator(thrust::make_tuple(
                        temp2.values.begin(),
                        mask.values.begin())),
                c.values.begin(),
                tsl);
    }
}
} // graphblas
