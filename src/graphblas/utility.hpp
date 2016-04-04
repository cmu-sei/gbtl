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

#include <graphblas/detail/config.hpp>
#define __GB_SYSTEM_UTILITY_HEADER <graphblas/system/__GB_SYSTEM_ROOT/utility.hpp>
#include __GB_SYSTEM_UTILITY_HEADER
#undef __GB_SYSTEM_UTILITY_HEADER

namespace graphblas
{
    template <typename AMatrixT,
              typename BMatrixT >
    void same_dimension_check(AMatrixT const &a,
                              BMatrixT const &b)
    {
        auto ashape = a.get_shape();
        auto bshape = b.get_shape();
        if (ashape.first!=bshape.first ||
            ashape.second!=bshape.second)
        {
            //throw graphblas::DimensionException();
            //std::cerr<<"warning: dim check failed"<<std::endl;
        }
    }

    template <typename AMatrixT,
              typename BMatrixT >
    void multiply_dimension_check(AMatrixT const &a,
                                  BMatrixT const &b)
    {
        auto ashape = a.get_shape();
        auto bshape = b.get_shape();
        if (ashape.first!=bshape.second)
        {
            //throw graphblas::DimensionException();
            //std::cerr<<"warning: multiply dim check failed"<<std::endl;
        }
    }

    template <typename AVectorT,
              typename SizeT >
    void vector_multiply_dimension_check(AVectorT const &a,
                                         SizeT const &b)
    {
        //a is a vector with a method called "size()"
        if (a.size()!=b)
        {
            throw graphblas::DimensionException();
        }
    }

    /** @note This file contains interfaces that have some use in certain
     *        situations, but their future in GraphBLAS is uncertain.
     */

    /**
     * Constructing an array of indices seems to be impossible using
     * the current GraphBLAS interface.  This function implements
     * this functionality per backend.  In the future, there should be
     * a way to get indices from GraphBLAS operations (e.g., mxm
     * should give indices to plus and times operations of a
     * semiring).
     *
     * @note This function will destroy any values already prenest in
     *       matrix B.
     *
     * @todo Should this be in algorithms?
     *
     * @param[in]     A
     * @param[in,out] B
     * @param[in]     base_index
     */
    template<typename AMatrixT, typename BMatrixT>
    void index_of(AMatrixT const  &A,
                  BMatrixT        &B,
                  IndexType const  base_index = 1)
    {
        backend::index_of(A.m_mat, B.m_mat, base_index);
    }

    /**
     * Replace (in-place) stored values in the elements of the matrix with
     * the value of its column index.
     *
     * @note We cannot currently use apply() to perform this operation because
     *       apply is not passed the location of the element it is operating on.
     *
     * @param[in,out] mat  The matrix to perform the conversion on.
     */
    template<typename MatrixT>
    void col_index_of(MatrixT &mat)
    {
        backend::col_index_of(mat.m_mat);
    }

    /**
     * Replace (in-place) stored values in the elements of the matrix with
     * the value of its row index.
     *
     * @note We cannot currently use apply() to perform this operation because
     *       apply is not passed the location of the element it is operating on.
     *
     * @param[in,out] mat  The matrix to perform the conversion on.
     */
    template<typename MatrixT>
    void row_index_of(MatrixT &mat)
    {
        backend::row_index_of(mat.m_mat);
    }

    /**
     * @brief filters out the elements in one vector from the other.
     *
     * @param[in,out] v1 The vector to be filtered,.
     * @param[in] v2  The vector to filter, .
     *
     * @return size of the filtered vector
     *
     */
    template <typename Vector1,
              typename Vector2,
              typename SizeT >
    SizeT filter(Vector1 &v1,
                SizeT v1size,
                Vector2 const &v2,
                SizeT v2size)
    {
        namespace btl = backend_template_library;

        Vector1 temp(v1size);
        //require c++11
        auto end = btl::set_difference(
                v1.begin(),
                v1.begin()+v1size,
                v2.begin(),
                v2.begin()+v2size,
                temp.begin());

        btl::copy(temp.begin(), end, v1.begin());

        return btl::distance(temp.begin(), end);
    }

    template <typename ConstT, typename BinaryOp>
    struct arithmetic_n{
        ConstT n;
        BinaryOp op;

        arithmetic_n(
                const ConstT & value,
                BinaryOp operation = BinaryOp() ) :
            n(value),
            op(operation)
        {}

        template <typename T>
#ifdef GB_USE_CUSP_GPU
__device__ __host__
#endif
        T operator()(const T& value){
            return op(value, static_cast<T>(n));
        }
    };

}

