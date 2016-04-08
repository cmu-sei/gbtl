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

#ifndef MST_HPP
#define MST_HPP


#include <utility>
#include <vector>

#include <iostream>

#include <graphblas/graphblas.hpp>
//****************************************************************************
namespace
{

    //************************************************************************
    /// @todo Consider moving this to algebra.hpp
    template <typename T>
    class LessThan
    {
    public:
        typedef T result_type;

        __device__ __host__ inline result_type operator()(T const &a,
                                                          T const &b)
        {
            return ((a < b) ?
                    static_cast<T>(1) :
                    static_cast<T>(0));
        }
    };
}

namespace algorithms
{
    /**
     * @brief Compute the of the minimal spanning tree for the given graph.
     *
     * This function takes in the adjacency matrix representation of a
     * graph, and computes the weight of the minimal spanning tree of
     * the graph.  This is done using Prim's algorithm.
     *
     * Prim's algorithm works by growing a single set of vertices
     * (call it S) which belong to the spanning tree.  On each iteration,
     * we add the closest vertex to S not already in S, and add the
     * weight of that vertex and its parent to the weight of the minimal
     * spanning tree.
     *
     * For our algorithm, this weight is what is returned when the algorithm
     * terminates.
     *
     * @param[in]  graph  The graph to perform the computation on.
     *
     * @return The weight of the minimum spanning tree of the graph,
     *         followed by a tuple list containing the vertices in the
     *         minimum spanning tree (of the form (row, col, val)).
     */
    template<typename MatrixT>
    std::pair<typename MatrixT::ScalarType,
              std::vector<std::tuple<graphblas::IndexType,
                                     graphblas::IndexType,
                                     typename MatrixT::ScalarType> > >
    mst(MatrixT const &graph)
    {
        using T = typename MatrixT::ScalarType;
        graphblas::IndexType cols, rows;
        graph.get_shape(rows, cols);

        MatrixT A(graph);
        A.set_zero(std::numeric_limits<T>::max());

        MatrixT seen(rows,1);
        T weight = static_cast<T>(0);
        std::vector<graphblas::IndexType> seen_r={1}, seen_c={0};
        std::vector<T> seen_v = {std::numeric_limits<T>::max()};
        graphblas::buildmatrix(seen, seen_r.begin(), seen_c.begin(),
                seen_v.begin(), seen_r.size());

        //seen.set_value_at(1, 0, std::numeric_limits<T>::max());

        MatrixT sources = graphblas::fill<MatrixT>(1, rows, 1);

        MatrixT d(rows, 1);
        MatrixT mask(cols, 1);

        std::vector<T> mask_v = {1};
        graphblas::buildmatrix(seen, seen_r.begin(), seen_c.begin(),
                mask_v.begin(), seen_r.size());
        //mask.set_value_at(1,0,1);

        graphblas::mxv(A, mask, d);

        std::vector<std::tuple<graphblas::IndexType,
                               graphblas::IndexType, T> > edges;

        while (seen.get_nnz() < rows)
        {
            MatrixT temp(rows, 1);
            graphblas::ewiseadd(seen, d, temp);

            //position of a single minimum element in vector
            //this is not really how loops should be used in graphblas, though....
            graphblas::IndexType u = 0;
            for (graphblas::IndexType i = 1; i < rows; ++i)
            {
                if (temp.get_value_at(i, 0) < temp.get_value_at(u, 0))
                {
                    u = i;
                }
            }

            seen.set_value_at(u, 0, std::numeric_limits<T>::max());
            weight += d.get_value_at(u,0);

            edges.push_back(
                std::make_tuple(
                    sources.get_value_at(u, 0),
                    u,
                    A.get_value_at(sources.get_value_at(u,0),u)));

            /*for (graphblas::IndexType i = 1; i < rows; ++i)
            {
                if (A.get_value_at(u,i) < d.get_value_at(i,0))
                {
                    d.set_value_at(i,0, A.get_value_at(u, i));
                    sources.set_value_at(i, 0, u);
                }
            }*/

            // A[u][:]
            MatrixT slice(cols, 1);
            MatrixT slice_mask(cols, 1);
            slice_mask.set_value_at(u,0,1);

            graphblas::mxv(A, slice_mask, slice);

            //if A[u][:] < d
            MatrixT assign_mask(cols, 1);
            ewiseadd(slice, d, assign_mask, LessThan<T>());
            
            //set values of d from d or slice according to mask
            ewisemult(slice, assign_mask, slice);
            ewisemult(d, negate(assign_mask), d);
            ewiseadd(d, slice, d);

            //set values of sources to u according to mask
            MatrixT src_this_step = graphblas::fill<MatrixT>(u, cols, 1);
            ewisemult(src_this_step, assign_mask, src_this_step);
            ewisemult(sources, negate(assign_mask), sources);
            ewiseadd(src_this_step, sources, sources);
           

        }

        return std::make_pair(weight, edges);
    }

} // algorithms

#endif // MST_HPP
