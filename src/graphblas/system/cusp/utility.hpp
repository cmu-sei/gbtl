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
#include <thrust/iterator/constant_iterator.h>

#include <graphblas/detail/config.hpp>
#include <graphblas/system/cusp/detail/utility.inl>
#include <cusp/print.h>

namespace graphblas
{
    namespace backend_template_library = thrust;

namespace backend
{
    /**
     * Constructing an array of indices seems to be impossible using
     * the current GraphBLAS interface.  This function implements this
     * functionality per backend.  In the future, there should be a
     * way to get indices from GraphBLAS operations (e.g., mxm should
     * give indices to plus and times operations of a semiring).
     */
    template<typename MatrixA, typename MatrixB>
    void index_of(const MatrixA &A, MatrixB &B, const IndexType base_index);

    template<typename MatrixT>
    void col_index_of(MatrixT &mat);

    template<typename MatrixT>
    void row_index_of(MatrixT &mat);


    template<typename VectorI,
             typename VectorJ,
             typename ValueType,
             typename Accum>
    inline void assign_vector(
        VectorI &v_src,
        VectorI &v_ind,
        VectorI &v_val,
        VectorJ &v_out,
        Accum    accum)
    {
        using namespace detail;
        if (v_val.size() != v_ind.size())
        {
            return;
        }
        assign_vector_helper(v_src, v_ind, v_val.begin(), v_out, accum);
    }

    template<typename VectorI,
             typename VectorJ,
             typename ValueType,
             typename Accum>
    inline void assign_vector(
            VectorI                           &v_src,
            VectorI                           &v_ind,
            const typename VectorI::value_type value,
            VectorJ                           &v_out,
            Accum                              accum)
    {
        using namespace detail;
        thrust::constant_iterator<ValueType> tempValues(value);
        assign_vector_helper(v_src, v_ind, tempValues, v_out, accum);
    }

    //timing:
    //common functions:
    cudaEvent_t start_event, stop_event;

    void start_timer(){
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        cudaEventRecord(start_event);
    }

    void stop_timer(){
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
    }

    float get_elapsed_time(){
        float ms=0;
        cudaEventElapsedTime(&ms, start_event, stop_event);
        return ms;
    }

    /**
     *  @brief Output the matrix in array form.  Mainly for debugging
     *         small matrices.
     *
     *  @param[in] ostr  The output stream to send the contents
     *  @param[in] mat   The matrix to output
     *
     */
    template <typename MatrixT>
    void pretty_print_matrix(std::ostream &ostr, MatrixT const &mat)
    {
        cusp::print(mat, ostr);
    }
}
}
