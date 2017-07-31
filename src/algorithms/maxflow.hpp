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

#ifndef MAXFLOW_HPP
#define MAXFLOW_HPP

#include <vector>
#include <random>

#include <graphblas/graphblas.hpp>

//****************************************************************************
template <typename MatrixT, typename VectorT>
static void push(MatrixT const &C,
                 MatrixT       &F,
                 VectorT       &excess,
                 GraphBLAS::IndexType u,
                 GraphBLAS::IndexType v)
{
    using T = typename MatrixT::ScalarType;
    T a = excess.extractElement(u);
    T b = C.extractElement(u, v) - F.extractElement(u, v);
    T send = std::min(a, b);

    F.setElement(u, v, F.extractElement(u, v) + send);
    F.setElement(v, u, F.extractElement(v, u) - send);

    excess.setElement(u, excess.extractElement(u) - send);
    excess.setElement(v, excess.extractElement(v) + send);
}

//****************************************************************************
template <typename MatrixT, typename VectorT>
static void relabel(MatrixT const &C,
                    MatrixT const &F,
                    VectorT       &height,
                    GraphBLAS::IndexType u)
{
    using T = typename MatrixT::ScalarType;
    GraphBLAS::IndexType num_nodes(C.nrows());
    GraphBLAS::IndexType cols(C.ncols());

    T min_height = std::numeric_limits<T>::max();
    for (GraphBLAS::IndexType v = 0; v < num_nodes; ++v)
    {
        if ((C.extractElement(u, v) - F.extractElement(u, v)) > 0)
        {
            T a = height.extractElement(v);
            min_height = std::min(min_height, a);
            height.setElement(u, min_height + 1);
        }
    }
}

//****************************************************************************
/// @todo Does seen have a different scalar type (IndexType)
template <typename MatrixT, typename VectorT>
static void discharge(MatrixT const &C,
                      MatrixT       &F,
                      VectorT       &excess,
                      VectorT       &height,
                      VectorT       &seen,
                      GraphBLAS::IndexType u)
{
    GraphBLAS::IndexType num_nodes(C.nrows());
    GraphBLAS::IndexType cols(C.ncols());

    while (excess.extractElement(u) > 0)
    {
        if (seen.extractElement(u) < num_nodes)
        {
            GraphBLAS::IndexType v = seen.extractElement(u);
            if (((C.extractElement(u, v) - F.extractElement(u, v)) > 0) &&
                (height.extractElement(u) > height.extractElement(v)))
            {
                push(C, F, excess, u, v);
            }
            else
            {
                seen.setElement(u, seen.extractElement(u) + 1);
            }
        }
        else
        {
            relabel(C, F, height, u);
            seen.setElement(u, 0);
        }
    }
}

//****************************************************************************
//****************************************************************************
namespace algorithms
{
    /**
     * @brief Compute the maximum flow through a graph given the capacities of
     *        the edges of that graph using a push-relabel algorithm
     *
     * <ul>
     * <li>Using a push-relabel removes the need to calculate augmented paths
     * (as in Ford Fulkerson)</li>
     * <li>Better lent itself to an implementation using GraphBLAS
     * functions.</li>
     * </ul>
     * <p>In this implementation, a preflow, which is an assignment of flows to
     * edges allowing for more incoming flow than outcoming flow while
     * maintaining capacity constraints, is maintained.</p>
     * <ul>
     * <li>Use the operation <code>push</code> to "push" flow through the flow
     * network.</li>
     * <li><code>relabel</code> maintains the residual network (the <i>residual
     * network</i> is the network at the current iteration in the
     * algorithm).</li>
     * <li>These two operations can only be performed on active vertices at
     * each iteration, where an <i>active</i> vertex is a a vertex such that
     * all edges in the residual graph that contain it only have positive
     * excess flows.</li>
     * </ul>
     *
     * @param[in]  capacity  The capacity matrix of the edges of the graph to
     *                       compute the max flow through.
     * @param[in]  source    The vertex to use as the source.
     * @param[in]  sink      The vertex to use as the sink.
     *
     * @return The value of the maximum flow that can be pushed through the
     *         source.
     */
    template<typename MatrixT>
    typename MatrixT::ScalarType maxflow(MatrixT const        &capacity,
                                         GraphBLAS::IndexType  source,
                                         GraphBLAS::IndexType  sink)
    {
        using T = typename MatrixT::ScalarType;
        GraphBLAS::IndexType rows(capacity.nrows());
        GraphBLAS::IndexType cols(capacity.ncols());

        GraphBLAS::IndexType num_nodes = rows;

        MatrixT flow(rows,cols);

        //MatrixT height(1, num_nodes);
        //MatrixT excess(1, num_nodes);
        //MatrixT seen(1, num_nodes);
        GraphBLAS::Vector<T> height(num_nodes);
        GraphBLAS::Vector<T> excess(num_nodes);
        GraphBLAS::Vector<T> seen(num_nodes);

        std::vector<GraphBLAS::IndexType> list;

        for (GraphBLAS::IndexType i = 0; i < num_nodes; ++i)
        {
            if ((i != source) && (i != sink))
            {
                list.push_back(i);
            }
        }

        height.setElement(source, num_nodes);
        excess.setElement(source,
                          std::numeric_limits<T>::max());
        for (GraphBLAS::IndexType i = 0; i < num_nodes; ++i)
        {
            push(capacity, flow, excess, source, i);
        }

        GraphBLAS::IndexType  p = 0;
        while (p < (num_nodes - 2))
        {
            GraphBLAS::IndexType u = list[p];
            T old_height = height.extractElement(u);
            discharge(capacity, flow, excess, height, seen, u);
            if (height.extractElement(u) > old_height)
            {
                GraphBLAS::IndexType t = list[p];

                list.erase(list.begin() + p);
                list.insert(list.begin() + 0, t);

                p = 0;
            }
            else
            {
                p += 1;
            }
        }

        T maxflow = static_cast<T>(0);
        for (GraphBLAS::IndexType i = 0; i < num_nodes; ++i)
        {
            maxflow += flow.extractElement(source, i);
        }

        return maxflow;
    }
} // algorithms

#endif // MAXFLOW_HPP
